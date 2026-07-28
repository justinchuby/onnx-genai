//! Runtime session, generation, and prefix-connector APIs.

use super::*;

fn generate_uses_scheduler(backend: EngineDecodeBackend) -> bool {
    backend != EngineDecodeBackend::Native
}

fn generation_budget_cap(cap: ScheduledBudgetCap) -> GenerationBudgetCap {
    GenerationBudgetCap {
        requested_max_new_tokens: cap.requested_max_tokens,
        admitted_max_new_tokens: cap.admitted_max_tokens,
        requested_bytes: cap.requested_bytes,
        admitted_bytes: cap.admitted_bytes,
        available_bytes: cap.available_bytes,
    }
}

impl Engine {
    /// Effective context limit for a request, combining model metadata,
    /// per-request override, and decode-path capacity.
    pub fn effective_max_context(&self, options: &GenerateOptions) -> Option<usize> {
        self.max_context_for_request(options)
    }

    #[cfg(feature = "native-backend")]
    fn generate_native_with_callback(
        &mut self,
        mut request: GenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.last_speculative_stats = SpeculativeStats::default();
        if request.options.speculative_mode.is_none() && self.native_shared_kv_proposer.is_some() {
            request.options.speculative_mode = Some(self.speculative_mode.clone());
        }
        reject_native_request_speculation(&request.options)?;
        request.options.validate()?;
        let mut options = request.options;
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        let prompt_tokens = self.tokenize_prompt(&request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        options.max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(&self.tokenizer))?;

        // Speculation ON (implemented greedy prompt-lookup) → the native
        // speculative driver. Every other request stays on the untouched plain
        // M=1 fast path below, preserving the 762 tok/s non-regression guarantee.
        if let Some(plan) = native_speculation_plan(&options, &chain) {
            let mut stats = SpeculativeStats::default();
            let native_session = self
                .native_session
                .as_mut()
                .context("native decoder session is unavailable")?;
            let mut driver = match plan.kind {
                NativeSpeculationKind::PromptLookup { ngram, max_tokens } => {
                    crate::native_speculative::NativeSpeculativeDriver::new_prompt_lookup(
                        native_session,
                        ngram,
                        max_tokens,
                        plan.width,
                    )?
                }
                NativeSpeculationKind::SharedKv => {
                    let proposer = self.native_shared_kv_proposer.as_mut().context(
                        "native shared-KV speculation requested without a loaded proposer session",
                    )?;
                    crate::native_speculative::NativeSpeculativeDriver::new_shared_kv(
                        native_session,
                        &mut proposer.session,
                        &proposer.embedder,
                        &proposer.groups,
                        proposer.hidden_size,
                        plan.width,
                    )?
                }
            };
            let result = augment_backend_error(
                driver.generate(
                    &prompt_tokens,
                    &options,
                    &chain,
                    &self.tokenizer,
                    &mut stats,
                    callback,
                ),
                EngineDecodeBackend::Native,
            );
            self.last_speculative_stats = stats;
            return result;
        }

        let native_session = self
            .native_session
            .as_mut()
            .context("native decoder session is unavailable")?;
        augment_backend_error(
            native_session.generate_with_callback(
                &prompt_tokens,
                &options,
                &chain,
                &self.tokenizer,
                callback,
            ),
            EngineDecodeBackend::Native,
        )
    }

    #[cfg(not(feature = "native-backend"))]
    fn generate_native_with_callback(
        &mut self,
        _request: GenerateRequest,
        _callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        anyhow::bail!(
            "native decoder backend requires building onnx-genai-engine with the 'native-backend' feature"
        )
    }

    fn require_ort_backend(&self, feature: &str) -> anyhow::Result<()> {
        if self.decode_backend == EngineDecodeBackend::Native {
            anyhow::bail!(
                "the native single-session backend does not support {feature}; use independent serialized requests"
            );
        }
        Ok(())
    }

    /// Generate text for a request.
    ///
    pub fn generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult> {
        self.generate_with_callback(request, None)
    }

    /// Generate text using a caller-supplied [`Sampler`] for final token
    /// selection.
    ///
    /// The logit-processor chain (temperature, top-k, top-p, min-p, penalties,
    /// constraints, …) still runs; only the terminal greedy/categorical pick is
    /// replaced by `sampler`. This is the public extension seam that the C ABI
    /// ([`crate::capi`]) exposes to foreign samplers. Not supported on the
    /// native single-session backend.
    pub fn generate_with_sampler(
        &mut self,
        request: GenerateRequest,
        sampler: Box<dyn Sampler>,
    ) -> anyhow::Result<GenerateResult> {
        if self.decode_backend == EngineDecodeBackend::Native {
            anyhow::bail!("custom samplers are not supported on the native single-session backend");
        }
        let session_id = self.create_session()?;
        let result = self.generate_in_session_with_sampler(session_id, request, sampler);
        let close_result = self.close_session(session_id);
        match (result, close_result) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    /// Speculative verification diagnostics from the most recent generation.
    pub fn last_speculative_stats(&self) -> SpeculativeStats {
        self.last_speculative_stats
    }

    /// Native CUDA decode diagnostics from the engine-owned session.
    #[cfg(feature = "native-backend")]
    pub fn native_cuda_debug_stats(&self) -> Option<crate::native_decode::CudaKvDebugStats> {
        self.native_session
            .as_ref()
            .and_then(crate::native_decode::NativeDecodeSession::cuda_kv_debug_stats)
    }

    /// Access the engine-owned Resource Governor handle.
    pub fn governor(&self) -> &EngineResourceGovernor {
        &self.governor
    }

    /// Convenience snapshot of configured and live resource state.
    pub fn resource_snapshot(&self) -> GovernorSnapshot {
        self.governor.snapshot()
    }

    /// Change the live VRAM ceiling when runtime overrides are enabled.
    pub fn set_vram_limit(
        &self,
        limit: ResourceLimit,
    ) -> Result<GovernorReconfigureOutcome, EngineGovernorError> {
        self.governor.set_vram_limit(limit)
    }

    /// Cumulative KV page activity: allocations, frees, and evictions.
    ///
    /// Evictions or allocation failures mean the KV pool is under pressure,
    /// which no per-token latency figure explains on its own.
    /// What the KV page pool is holding right now.
    pub fn page_usage(&self) -> onnx_genai_kv::PageUsage {
        self.kv_cache.page_table.usage()
    }

    pub fn page_stats(&self) -> onnx_genai_kv::PageStats {
        self.kv_cache.page_table.stats()
    }

    /// External KV connector activity from the most recent generation.
    ///
    /// Reflects lookups, would-be prefix extensions, tokens actually fetched and
    /// injected (K4 materialization), and chunk stores. Returns
    /// [`ConnectorStats::default`] when no connector is configured.
    pub fn last_connector_stats(&self) -> ConnectorStats {
        self.connector.stats().clone()
    }

    /// Generate the middle text for a fill-in-the-middle request.
    pub fn generate_fim(
        &mut self,
        prefix: impl AsRef<str>,
        suffix: impl AsRef<str>,
        options: GenerateOptions,
    ) -> anyhow::Result<GenerateResult> {
        let fim_config = self
            .fim_config
            .clone()
            .context("model tokenizer_config.json does not declare recognized FIM tokens")?;
        self.generate_fim_with_config(prefix, suffix, options, &fim_config)
    }

    /// Generate the middle text using an explicit fill-in-the-middle configuration.
    pub fn generate_fim_with_config(
        &mut self,
        prefix: impl AsRef<str>,
        suffix: impl AsRef<str>,
        options: GenerateOptions,
        fim_config: &FimConfig,
    ) -> anyhow::Result<GenerateResult> {
        let prompt = fim_config.format_prompt(prefix.as_ref(), suffix.as_ref());
        let mut request = GenerateRequest::new(prompt);
        request.options = self.fim_options(fim_config, options);
        self.generate(request)
    }

    /// Generate text and optionally stream each generated token to `callback`.
    pub fn generate_with_callback(
        &mut self,
        request: GenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        if !generate_uses_scheduler(self.decode_backend) {
            return self.generate_native_with_callback(request, callback);
        }
        let session_id = self.create_session()?;
        let result = self.generate_in_session_with_callback(session_id, request, callback);
        let close_result = self.close_session(session_id);
        match (result, close_result) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    /// Generate text in a persistent session, reusing the session's accumulated KV state.
    pub fn generate_in_session(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_callback(session_id, request, None)
    }

    /// Generate text in a persistent session with an explicit scheduler priority.
    pub fn generate_in_session_with_priority(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        priority: Priority,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_priority_and_callback(
            session_id, request, priority, None, None,
        )
    }

    /// Generate text in a persistent session and optionally stream generated tokens.
    pub fn generate_in_session_with_callback(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_priority_and_callback(
            session_id,
            request,
            Priority::Normal,
            None,
            callback,
        )
    }

    /// Generate text in a persistent session using a caller-supplied [`Sampler`].
    ///
    /// The custom sampler replaces the built-in greedy/categorical token
    /// selection while the full logit-processor chain (temperature, top-k,
    /// top-p, penalties, constraints, …) still runs first. This is the Rust
    /// extension seam that the C ABI ([`crate::capi`]) plugs foreign samplers
    /// into. The device greedy fast path is bypassed so the sampler always sees
    /// the processed logits.
    pub fn generate_in_session_with_sampler(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        sampler: Box<dyn Sampler>,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_priority_and_callback(
            session_id,
            request,
            Priority::Normal,
            Some(sampler),
            None,
        )
    }

    fn generate_in_session_with_priority_and_callback(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        priority: Priority,
        mut custom_sampler: Option<Box<dyn Sampler>>,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.last_speculative_stats = SpeculativeStats::default();
        request.options.validate()?;
        let mut options = request.options.clone();
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        let prompt_tokens = self.tokenize_prompt(&request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        if !self.sessions.contains_key(&session_id) {
            anyhow::bail!("session {session_id} not found");
        }

        let max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(&self.tokenizer))?;

        let request_id = self.scheduler.enqueue_generate_request(
            session_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            priority,
        );
        let scheduled = match self.scheduler.drive_next_fcfs_result() {
            Ok(Some(scheduled)) => scheduled,
            Ok(None) => {
                self.scheduler.cancel_request(request_id);
                anyhow::bail!(
                    "scheduler had no waiting request after enqueueing request {request_id} for session {session_id}"
                );
            }
            Err(error) => {
                self.scheduler.cancel_request(request_id);
                return Err(error.into());
            }
        };
        if scheduled.request_id != request_id || scheduled.seq_id != session_id {
            self.scheduler.cancel_request(scheduled.request_id);
            if scheduled.request_id != request_id {
                self.scheduler.cancel_request(request_id);
            }
            anyhow::bail!(
                "scheduler admitted request {} for session {}, expected request {} for session {}",
                scheduled.request_id,
                scheduled.seq_id,
                request_id,
                session_id
            );
        }
        let budget_cap = scheduled.budget_cap.map(generation_budget_cap);
        options.max_new_tokens = scheduled.max_tokens;

        let Some(mut state) = self.sessions.remove(&session_id) else {
            self.scheduler.complete(session_id);
            anyhow::bail!("session {session_id} not found");
        };

        let result = (|| -> anyhow::Result<GenerateResult> {
            let prefix_cache_hit_len =
                self.prepare_session_prefix(session_id, &mut state, &prompt_tokens)?;
            let mut loop_state =
                DecodeLoopState::new(prefix_cache_hit_len, options.seed, options.top_logprobs);
            let has_custom_sampler = custom_sampler.is_some();
            loop_state.custom_sampler = custom_sampler.take();

            if self.should_use_speculative(&options) && !has_custom_sampler {
                return self.generate_speculative_loop(crate::speculative::SpeculativeLoopState {
                    session_id,
                    state: &mut state,
                    options: &options,
                    chain: &chain,
                    max_context,
                    prefix_cache_hit_len,
                    generated_tokens: &mut loop_state.generated_tokens,
                    generated_text: &mut loop_state.generated_text,
                    generated_logprobs: &mut loop_state.logprobs,
                    rng: &mut loop_state.rng,
                    callback: callback.as_deref_mut(),
                });
            }

            let mut backend = SessionDecodeLoopBackend {
                session: self
                    .session
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                kv_model: self.kv_model.as_ref(),
                kv_cache: &mut self.kv_cache,
                scheduler: &mut self.scheduler,
                session_id,
                state: &mut state,
            };
            run_decode_loop(
                &mut backend,
                &mut loop_state,
                &options,
                &chain,
                &self.tokenizer,
                max_context,
                callback.as_deref_mut(),
            )
        })()
        .and_then(|mut result| {
            if !exceeded_context_limit(state.tokens.len(), max_context) {
                self.ensure_session_kv_current(session_id, &mut state)?;
                self.insert_cached_prefixes(session_id, &state, prompt_tokens.len())?;
            }
            result.budget_cap = budget_cap;
            Ok(result)
        });
        self.sessions.insert(session_id, state);
        self.scheduler.complete(session_id);
        result
    }

    /// Drive a set of already-arrived prioritized requests to completion.
    ///
    /// This is the Phase 3 engine-facing scheduler drive API. It runs one
    /// sequence at a time for now, but honors priority ordering and scheduler
    /// preemption decisions while preserving session decode state and KV in place.
    pub fn drive_prioritized_requests(
        &mut self,
        requests: Vec<PrioritizedGenerateRequest>,
    ) -> anyhow::Result<Vec<PrioritizedGenerateResult>> {
        let arrivals = requests
            .into_iter()
            .map(|request| ScheduledGenerateArrival {
                arrival_step: 0,
                request,
            })
            .collect();
        self.drive_prioritized_arrivals(arrivals)
    }

    /// Drive prioritized requests that arrive after specific generated-token steps.
    ///
    /// This lets async server code drain newly-arrived requests between scheduler
    /// iterations. Preemption is swap-style: active `EngineSession` state, ORT past
    /// tensors, and mirrored paged KV stay owned by the engine and are resumed
    /// without recomputation.
    pub fn drive_prioritized_arrivals(
        &mut self,
        mut arrivals: Vec<ScheduledGenerateArrival>,
    ) -> anyhow::Result<Vec<PrioritizedGenerateResult>> {
        self.require_ort_backend("prioritized request scheduling")?;
        arrivals.sort_by_key(|arrival| arrival.arrival_step);
        let total_requests = arrivals.len();
        let mut next_arrival = 0;
        let mut generated_steps = 0;
        let mut active: HashMap<SessionId, ActiveGenerate> = HashMap::new();
        let mut results = Vec::with_capacity(total_requests);

        while results.len() < total_requests {
            while next_arrival < arrivals.len()
                && arrivals[next_arrival].arrival_step <= generated_steps
            {
                let arrival = arrivals[next_arrival].clone();
                next_arrival += 1;
                let active_request = self.prepare_active_generate(arrival.request)?;
                if active
                    .insert(active_request.session_id, active_request)
                    .is_some()
                {
                    anyhow::bail!("session already has an active generation request");
                }
            }

            let decision = self.scheduler.schedule();
            self.execute_kv_movement(&decision)?;
            let mut runnable = Vec::new();
            for seq in decision
                .prefill
                .iter()
                .chain(decision.swap_in.iter())
                .chain(decision.decode.iter())
            {
                if !decision.preempt.contains(seq) && !runnable.contains(seq) {
                    runnable.push(*seq);
                }
            }

            if runnable.is_empty() {
                if next_arrival < arrivals.len() {
                    generated_steps = arrivals[next_arrival].arrival_step;
                    continue;
                }
                anyhow::bail!("scheduler made no runnable decision with active requests remaining");
            }

            for session_id in runnable {
                let mut active_request = active.remove(&session_id).with_context(|| {
                    format!("active request for session {session_id} not found")
                })?;
                if let Some(max_tokens) = self.scheduler.running_max_tokens(session_id) {
                    active_request.budget_cap = self
                        .scheduler
                        .running_budget_cap(session_id)
                        .map(generation_budget_cap);
                    active_request.options.max_new_tokens = max_tokens;
                }
                let step_result = self.step_active_generate(&mut active_request)?;
                generated_steps += 1;
                if let Some(result) = step_result {
                    let session_id = active_request.session_id;
                    self.finish_active_generate(active_request)?;
                    results.push(PrioritizedGenerateResult { session_id, result });
                } else {
                    active.insert(session_id, active_request);
                }
            }
        }

        Ok(results)
    }

    /// Execute the KV-cache movement a scheduler decision calls for.
    ///
    /// This is the engine-side counterpart to the scheduler's preemption
    /// bookkeeping: a `ScheduleDecision::preempt` entry evicts that sequence's
    /// paged KV off the hot tier (freeing residency for higher-priority work),
    /// and a `ScheduleDecision::swap_in` entry restores it before the sequence
    /// runs again. Preemption only re-tags the pages' device tier; the KV data
    /// is preserved in place, so a preempted-then-restored sequence resumes with
    /// byte-identical KV and decodes the same tokens as if it had never been
    /// preempted (see [`PagedKvCache::preempt_sequence`]).
    ///
    /// When the scheduler emits neither preemption nor swap-in (the common
    /// single-sequence / no-pressure path) this is a no-op, keeping normal
    /// decoding behavior-identical.
    fn execute_kv_movement(&mut self, decision: &ScheduleDecision) -> anyhow::Result<()> {
        for seq in &decision.preempt {
            self.kv_cache
                .preempt_sequence(*seq)
                .map_err(|e| anyhow::anyhow!("failed to preempt KV for sequence {seq}: {e}"))?;
        }
        for seq in &decision.swap_in {
            self.kv_cache
                .restore_sequence(*seq)
                .map_err(|e| anyhow::anyhow!("failed to restore KV for sequence {seq}: {e}"))?;
        }
        Ok(())
    }

    /// Create a new generation session.
    pub fn create_session(&mut self) -> anyhow::Result<SessionId> {
        self.require_ort_backend("persistent sessions")?;
        let decode_state = self.new_target_decode_state()?;
        let id = self.kv_cache.create_sequence();
        let draft = if let Some(draft_model) = &mut self.draft {
            Some(DraftSession {
                seq: draft_model.kv_cache.create_sequence(),
                tokens: Vec::new(),
                kv_token_count: 0,
                decode_state: DecodeState::new_for_path(
                    &draft_model.session,
                    &draft_model.decode_path,
                )?,
            })
        } else {
            None
        };
        let state = EngineSession {
            tokens: Vec::new(),
            kv_token_count: 0,
            decode_state,
            draft,
            sampled_fastpath_failed: false,
        };
        self.sessions.insert(id, state);
        Ok(id)
    }

    /// Reset a persistent session, freeing its current state while keeping the id usable.
    pub fn reset_session(&mut self, session_id: SessionId) -> anyhow::Result<()> {
        self.require_ort_backend("persistent sessions")?;
        if !self.sessions.contains_key(&session_id) {
            anyhow::bail!("session {session_id} not found");
        }
        self.scheduler.complete(session_id);
        self.kv_cache
            .remove(session_id)
            .map_err(|e| anyhow::anyhow!("Failed to reset KV sequence {session_id}: {e}"))?;
        self.kv_cache.page_table.create_sequence(session_id);
        let decode_state = self.new_target_decode_state()?;
        let state = self
            .sessions
            .get_mut(&session_id)
            .context("session disappeared during reset")?;
        state.tokens.clear();
        state.kv_token_count = 0;
        state.decode_state = decode_state;
        if let (Some(draft_model), Some(draft)) = (&mut self.draft, &mut state.draft) {
            draft_model
                .kv_cache
                .remove(draft.seq)
                .map_err(|e| anyhow::anyhow!("Failed to reset draft KV sequence: {e}"))?;
            draft.seq = draft_model.kv_cache.create_sequence();
            draft.tokens.clear();
            draft.kv_token_count = 0;
            draft.decode_state =
                DecodeState::new_for_path(&draft_model.session, &draft_model.decode_path)?;
        }
        Ok(())
    }

    fn new_target_decode_state(&self) -> anyhow::Result<DecodeState> {
        let session = self
            .session
            .as_deref()
            .context("ORT decoder session is unavailable")?;
        // Bind ports from an explicit `model.io` block when the package declares
        // one; otherwise DecodeState falls back to tensor-name conventions.
        let io = self
            .metadata
            .model
            .as_ref()
            .and_then(|model| model.io.as_ref());
        let fixed_state_budget_bytes = self.governor.snapshot().resolved_limits.host_ram_bytes;
        if matches!(
            &self.speculative_mode,
            SpeculativeMode::Mtp(_) | SpeculativeMode::Eagle3(_) | SpeculativeMode::SharedKv(_)
        ) {
            DecodeState::new_with_io_positions_and_state_budget(
                session,
                io,
                None,
                fixed_state_budget_bytes,
            )
        } else {
            DecodeState::new_for_path_with_io_positions_and_state_budget(
                session,
                &self.decode_path,
                io,
                None,
                fixed_state_budget_bytes,
            )
        }
    }

    /// Close a persistent session and free its associated state.
    pub fn close_session(&mut self, session_id: SessionId) -> anyhow::Result<()> {
        self.require_ort_backend("persistent sessions")?;
        self.scheduler.complete(session_id);
        let state = self
            .sessions
            .remove(&session_id)
            .with_context(|| format!("session {session_id} not found"))?;
        self.kv_cache
            .remove(session_id)
            .map_err(|e| anyhow::anyhow!("Failed to remove KV sequence {session_id}: {e}"))?;
        if let (Some(draft_model), Some(draft)) = (&mut self.draft, state.draft) {
            draft_model
                .kv_cache
                .remove(draft.seq)
                .map_err(|e| anyhow::anyhow!("Failed to remove draft KV sequence: {e}"))?;
        }
        Ok(())
    }

    /// Number of logical tokens retained in a persistent session.
    pub fn session_token_count(&self, session_id: SessionId) -> anyhow::Result<usize> {
        self.require_ort_backend("persistent sessions")?;
        self.sessions
            .get(&session_id)
            .map(|state| state.tokens.len())
            .with_context(|| format!("session {session_id} not found"))
    }

    /// Get the loaded metadata.
    pub fn metadata(&self) -> &InferenceMetadata {
        &self.metadata
    }

    /// Resolved decoder execution backend.
    pub fn decode_backend(&self) -> EngineDecodeBackend {
        self.decode_backend
    }

    /// Auto-detected fill-in-the-middle configuration, if the tokenizer declares one.
    pub fn fim_config(&self) -> Option<&FimConfig> {
        self.fim_config.as_ref()
    }

    fn fim_options(&self, fim_config: &FimConfig, mut options: GenerateOptions) -> GenerateOptions {
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        for eos_token_id in self.tokenizer.eos_token_ids() {
            push_unique_stop_sequence(
                &mut options.stop_sequences,
                StopSequence::Tokens(vec![eos_token_id]),
            );
        }
        for token in [
            fim_config.prefix_token.as_str(),
            fim_config.middle_token.as_str(),
            fim_config.suffix_token.as_str(),
            "<|fim_pad|>",
            "<|endoftext|>",
            "<|file_sep|>",
        ] {
            if let Some(token_id) = self.tokenizer.token_id(token) {
                push_unique_stop_sequence(
                    &mut options.stop_sequences,
                    StopSequence::Tokens(vec![token_id]),
                );
            }
        }
        options
    }

    fn max_context_for_request(&self, options: &GenerateOptions) -> Option<usize> {
        let configured = self
            .metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length)
            .or(options.max_context);
        match self.decode_path_max_len() {
            Some(runtime_max) => {
                Some(configured.map_or(runtime_max, |limit| limit.min(runtime_max)))
            }
            None => configured,
        }
    }

    fn decode_path_max_len(&self) -> Option<usize> {
        match self.decode_path {
            ModelDecodePath::StaticCache { max_len } => Some(max_len),
            ModelDecodePath::PastPresent {
                shared_buffer: true,
                max_len,
                ..
            } => max_len,
            ModelDecodePath::PastPresent { .. } | ModelDecodePath::Legacy => None,
        }
    }

    /// Tokenize `text` with the model's own tokenizer.
    ///
    /// This is the public tokenization seam used by higher-level pipelines to
    /// convert prompt text into token ids (e.g. to compute prompt length or
    /// `max_length`, or to feed [`Engine::embed`] and the generation APIs). It
    /// uses the same tokenizer path as the engine's internal prompt handling.
    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<TokenId>> {
        self.tokenizer.encode(text).map_err(|e| {
            anyhow::anyhow!(
                "failed to tokenize input text with the model's tokenizer: {e}; \
                 verify the model directory contains a valid tokenizer.json"
            )
        })
    }

    fn tokenize_prompt(&self, prompt: &GeneratePrompt) -> anyhow::Result<Vec<TokenId>> {
        match prompt {
            GeneratePrompt::TokenIds(tokens) => Ok(tokens.clone()),
            GeneratePrompt::Text(text) => self
                .tokenizer
                .encode(text)
                .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}")),
        }
    }

    fn prepare_session_prefix(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        prompt_tokens: &[TokenId],
    ) -> anyhow::Result<usize> {
        if self.connector.is_active() {
            self.connector.reset_stats();
        }
        let same_session_hit_len = if state.decode_state.has_runner() {
            state.decode_state.runner_len().min(state.tokens.len())
        } else if state.decode_state.use_kv {
            state.kv_token_count.min(state.tokens.len())
        } else {
            0
        };
        let started_empty = state.tokens.is_empty();
        let mut loaded_prompt_prefix = 0;
        let mut cross_session_hit_len = 0;

        if started_empty && state.decode_state.uses_token_prefix_cache() {
            cross_session_hit_len = self
                .token_prefix_cache
                .iter()
                .map(|cached| common_prefix_len(cached, prompt_tokens).min(cached.len()))
                .filter(|&len| len > 0)
                .max()
                .unwrap_or(0);
        } else if started_empty
            && state.decode_state.use_kv
            && self.kv_model.is_some()
            && self.kv_cache.page_table.tensor_config.is_some()
        {
            let matched = self
                .prefix_cache
                .lookup_shared(prompt_tokens, &mut self.kv_cache.page_table);
            if matched.matched_tokens > 0 {
                cross_session_hit_len = matched.matched_tokens;
                let materialized_len = if matched.matched_tokens == prompt_tokens.len() {
                    matched.matched_tokens.saturating_sub(1)
                } else {
                    matched.matched_tokens
                };
                let page_ids = matched
                    .page_ids
                    .iter()
                    .copied()
                    .take(materialized_len.div_ceil(self.kv_cache.page_table.page_size))
                    .collect::<Vec<_>>();
                for &page_id in &page_ids {
                    self.kv_cache.page_table.retain(page_id);
                }
                self.prefix_cache.release_shared(
                    prompt_tokens,
                    matched.matched_tokens,
                    &mut self.kv_cache.page_table,
                );
                if materialized_len > 0 {
                    attach_pages_to_sequence(
                        &mut self.kv_cache,
                        session_id,
                        &page_ids,
                        materialized_len,
                    )?;
                    let materialized = self
                        .kv_cache
                        .materialize_sequence(session_id)
                        .map_err(|e| anyhow::anyhow!("Failed to materialize prefix KV: {e}"))?;
                    load_materialized_past(
                        self.ort_session()?,
                        self.kv_model.as_ref().expect("checked above"),
                        &mut state.decode_state,
                        &materialized,
                    )?;
                    state.kv_token_count = materialized_len;
                    state
                        .tokens
                        .extend_from_slice(&prompt_tokens[..materialized_len]);
                    loaded_prompt_prefix = materialized_len;
                }
            }
        }

        if started_empty {
            state
                .tokens
                .extend_from_slice(&prompt_tokens[loaded_prompt_prefix..]);
        } else {
            state.tokens.extend_from_slice(prompt_tokens);
        }
        let in_process_hit = same_session_hit_len.max(cross_session_hit_len);

        // K4: consult the external connector for prefix reuse *beyond* the
        // in-process hit. When the active decode path can accept an owned-KV
        // handoff (a ZeroCopyRebind `PastPresent` runner with f32 KV) and the
        // session started empty, fetch the real KV bytes for the contiguous hit
        // chunks and inject them into the runner so prefill genuinely skips
        // those tokens. Because the chunk key is prefix-dependent, an equal key
        // guarantees an identical prefix, so injecting fetched KV at the same
        // absolute positions is byte-exact — proven token-identical by the gold
        // integration test. If injection is not possible we fall back to the
        // reporting-only `lookup_extension`, never claiming a hit we can't serve.
        if self.connector.is_active() {
            let injected = self.try_connector_kv_injection(state, prompt_tokens, in_process_hit)?;
            if let Some(total) = injected {
                return Ok(in_process_hit.max(total));
            }
            let _ = self
                .connector
                .lookup_extension(prompt_tokens, in_process_hit);
        }
        Ok(in_process_hit)
    }

    /// Try to materialize cross-session KV from the connector into the decode
    /// runner, genuinely shortening prefill. Returns `Some(total_len)` (the KV
    /// token count now resident in the runner) when injection happened, else
    /// `None` (caller falls back to reporting-only lookup).
    ///
    /// Only runs for a freshly started session on a ZeroCopyRebind `PastPresent`
    /// runner whose KV is f32. `import_kv` *replaces* the runner KV, so the
    /// boundary must be the current `kv_token_count` (0 for a fresh session).
    /// At least one prompt token is always left un-injected so decode has an
    /// input to feed.
    fn try_connector_kv_injection(
        &mut self,
        state: &mut EngineSession,
        prompt_tokens: &[TokenId],
        in_process_hit: usize,
    ) -> anyhow::Result<Option<usize>> {
        if !state.decode_state.has_runner()
            || !state.decode_state.runner_supports_kv_handoff()
            || state.kv_token_count != 0
            || in_process_hit != 0
        {
            return Ok(None);
        }
        // Scope the immutable `kv_model` borrow so it does not overlap the
        // `&mut self.connector` fetch below.
        let past_is_f32 = match self.kv_model.as_ref() {
            Some(kv_model) => kv_model_past_is_f32(self.ort_session()?, kv_model),
            None => false,
        };
        if !past_is_f32 {
            return Ok(None);
        }

        let boundary = 0usize;
        // Leave at least one prompt token to feed the decoder: cap the fetch to
        // `prompt_len - 1` tokens so `fetched_tokens` equals what we inject.
        let max_tokens = prompt_tokens.len().saturating_sub(1);
        let outcome =
            self.connector
                .fetch_extension(prompt_tokens, boundary, max_tokens, Device::Cpu);
        if outcome.fetched_tokens == 0 {
            return Ok(None);
        }

        let mut chunks = outcome.chunks;
        let mut total: usize = boundary + chunks.iter().map(|c| c.num_tokens).sum::<usize>();
        // Safety net: the `max_tokens` cap already guarantees `total <
        // prompt_len`, but drop trailing chunks if any invariant slipped.
        while total >= prompt_tokens.len() {
            match chunks.pop() {
                Some(dropped) => total -= dropped.num_tokens,
                None => return Ok(None),
            }
        }
        if chunks.is_empty() || total == 0 {
            return Ok(None);
        }

        let placed: Vec<PlacedPayload<'_>> = chunks
            .iter()
            .map(|chunk| PlacedPayload {
                relative_start: chunk.start - boundary,
                payload: &chunk.payload,
            })
            .collect();
        let kv_model = self.kv_model.as_ref().expect("checked present above");
        let kv = past_kv_from_payloads(self.ort_session()?, kv_model, &placed, total)?;
        state.decode_state.import_runner_kv(total, kv)?;
        state.kv_token_count = total;
        Ok(Some(total))
    }

    fn prepare_active_generate(
        &mut self,
        request: PrioritizedGenerateRequest,
    ) -> anyhow::Result<ActiveGenerate> {
        request.request.options.validate()?;
        let mut options = request.request.options.clone();
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        let prompt_tokens = self.tokenize_prompt(&request.request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        if !self.sessions.contains_key(&request.session_id) {
            anyhow::bail!("session {} not found", request.session_id);
        }
        if self.should_use_speculative(&options) {
            anyhow::bail!(
                "prioritized drive API currently supports the single-sequence non-speculative path; batched/speculative drive is future work"
            );
        }

        let max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(&self.tokenizer))?;
        let mut state = self
            .sessions
            .remove(&request.session_id)
            .with_context(|| format!("session {} not found", request.session_id))?;
        let prefix_cache_hit_len =
            self.prepare_session_prefix(request.session_id, &mut state, &prompt_tokens)?;
        let rng = SamplingRng::new(options.seed);
        let logprobs = options.top_logprobs.map(|_| Vec::new());
        self.scheduler.enqueue_generate_request(
            request.session_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            request.priority,
        );
        Ok(ActiveGenerate {
            session_id: request.session_id,
            state,
            options,
            chain,
            max_context,
            prompt_len: prompt_tokens.len(),
            prefix_cache_hit_len,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            logprobs,
            budget_cap: None,
            step: 0,
            rng,
        })
    }

    fn step_active_generate(
        &mut self,
        active: &mut ActiveGenerate,
    ) -> anyhow::Result<Option<GenerateResult>> {
        let mut loop_state = DecodeLoopState {
            generated_tokens: std::mem::take(&mut active.generated_tokens),
            generated_text: std::mem::take(&mut active.generated_text),
            logprobs: active.logprobs.take(),
            step: active.step,
            prefix_cache_hit_len: active.prefix_cache_hit_len,
            rng: std::mem::replace(&mut active.rng, SamplingRng::new(Some(0))),
            custom_sampler: None,
        };
        let step_result = {
            let mut backend = SessionDecodeLoopBackend {
                session: self
                    .session
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                kv_model: self.kv_model.as_ref(),
                kv_cache: &mut self.kv_cache,
                scheduler: &mut self.scheduler,
                session_id: active.session_id,
                state: &mut active.state,
            };
            step_decode_loop(
                &mut backend,
                &mut loop_state,
                &active.options,
                &active.chain,
                &self.tokenizer,
                active.max_context,
                None,
            )?
        };
        active.generated_tokens = loop_state.generated_tokens;
        active.generated_text = loop_state.generated_text;
        active.logprobs = loop_state.logprobs;
        active.step = loop_state.step;
        active.rng = loop_state.rng;
        if let Some(mut result) = step_result {
            result.budget_cap = active.budget_cap;
            return Ok(Some(result));
        }
        if active.generated_tokens.len() >= active.options.max_new_tokens {
            ensure_constrained_finish(
                &active.options,
                &active.generated_text,
                FinishReason::MaxTokens,
            )?;
            return self
                .finish_result(
                    &active.generated_tokens,
                    FinishReason::MaxTokens,
                    active.prefix_cache_hit_len,
                    active.logprobs.as_deref(),
                )
                .map(|mut result| {
                    result.budget_cap = active.budget_cap;
                    Some(result)
                });
        }

        Ok(None)
    }

    fn finish_active_generate(&mut self, mut active: ActiveGenerate) -> anyhow::Result<()> {
        if !exceeded_context_limit(active.state.tokens.len(), active.max_context) {
            self.ensure_session_kv_current(active.session_id, &mut active.state)?;
            self.insert_cached_prefixes(active.session_id, &active.state, active.prompt_len)?;
        }
        self.sessions.insert(active.session_id, active.state);
        self.scheduler.complete(active.session_id);
        Ok(())
    }

    /// Borrow the ORT decoder session, returning an error instead of aborting
    /// the host when the backend is in an invalid state without a session. The
    /// ORT decode paths structurally guarantee a session is present, so this is
    /// a defensive accessor: a future invalid backend state surfaces as a
    /// recoverable error rather than a process abort.
    fn ort_session(&self) -> anyhow::Result<&Session> {
        self.session
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))
    }

    fn ensure_session_kv_current(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
    ) -> anyhow::Result<()> {
        while state.decode_state.use_kv && state.kv_token_count < state.tokens.len() {
            let _ = next_session_token_logits(
                self.session
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!(MISSING_ORT_SESSION))?,
                self.kv_model.as_ref(),
                &mut self.kv_cache,
                session_id,
                state,
            )?;
        }
        Ok(())
    }

    /// Extract the runner's freshly computed KV and store each complete resident
    /// chunk in the connector. Best-effort: any gating failure or extraction
    /// error skips storing (never surfaced to inference). See
    /// [`crate::connector_bridge::ConnectorBridge::store_prefix_with`].
    fn store_connector_prefix(&mut self, state: &EngineSession) {
        if !state.decode_state.runner_supports_kv_handoff() {
            return;
        }
        let config = match (self.kv_model.as_ref(), self.session.as_deref()) {
            (Some(kv_model), Some(session)) if kv_model_past_is_f32(session, kv_model) => {
                kv_model.tensor_config
            }
            _ => return,
        };
        let exported = match state.decode_state.export_runner_kv() {
            Ok(exported) => exported,
            Err(error) => {
                tracing::debug!(%error, "runner KV export failed; not storing to connector");
                return;
            }
        };
        let kv_model = self.kv_model.as_ref().expect("checked present above");
        let layers = match exported_layers_from_runner(kv_model, &exported) {
            Ok(layers) => layers,
            Err(error) => {
                tracing::debug!(%error, "collecting exported runner KV failed; not storing");
                return;
            }
        };
        self.connector.store_prefix_with(
            &state.tokens,
            state.kv_token_count,
            |chunk_start, num_tokens| {
                chunk_payload_from_exported(&layers, config, chunk_start, num_tokens)
            },
        );
    }

    fn insert_cached_prefixes(
        &mut self,
        session_id: SessionId,
        state: &EngineSession,
        prompt_len: usize,
    ) -> anyhow::Result<()> {
        // K4: extract the freshly computed KV for each complete resident chunk
        // and push the real bytes to the external connector for future
        // cross-session / cross-node reuse. Only ZeroCopyRebind `PastPresent`
        // runners with f32 KV can hand off owned tensors; other paths skip
        // (store is a no-op for the default `Null` connector regardless).
        if self.connector.is_active() {
            self.store_connector_prefix(state);
        }
        if state.decode_state.uses_token_prefix_cache() {
            if prompt_len > 0 && prompt_len <= state.kv_token_count {
                self.insert_token_prefix(&state.tokens[..prompt_len]);
            }
            if state.kv_token_count == state.tokens.len() {
                self.insert_token_prefix(&state.tokens);
            }
            return Ok(());
        }
        if self.kv_model.is_none() || state.kv_token_count == 0 {
            return Ok(());
        }
        if prompt_len > 0 && prompt_len <= state.kv_token_count {
            self.insert_cached_prefix(session_id, &state.tokens[..prompt_len])?;
        }
        if state.kv_token_count == state.tokens.len() {
            self.insert_cached_prefix(session_id, &state.tokens)?;
        }
        Ok(())
    }

    fn insert_cached_prefix(
        &mut self,
        session_id: SessionId,
        tokens: &[TokenId],
    ) -> anyhow::Result<()> {
        if tokens.is_empty() || self.prefix_cache.lookup(tokens).0 == tokens.len() {
            return Ok(());
        }
        let page_ids = sequence_pages_for_len(&self.kv_cache, session_id, tokens.len())?;
        self.prefix_cache
            .insert_pages(tokens, &page_ids, &mut self.kv_cache.page_table);
        Ok(())
    }

    fn insert_token_prefix(&mut self, tokens: &[TokenId]) {
        if tokens.is_empty()
            || self
                .token_prefix_cache
                .iter()
                .any(|cached| cached.as_slice() == tokens)
        {
            return;
        }
        self.token_prefix_cache.push(tokens.to_vec());
    }

    pub(crate) fn finish_result(
        &self,
        generated_tokens: &[TokenId],
        finish_reason: FinishReason,
        prefix_cache_hit_len: usize,
        logprobs: Option<&[crate::config::TokenLogprob]>,
    ) -> anyhow::Result<GenerateResult> {
        Ok(GenerateResult {
            text: self
                .tokenizer
                .decode(generated_tokens)
                .map_err(|e| anyhow::anyhow!("Failed to detokenize generated tokens: {e}"))?,
            token_ids: generated_tokens.to_vec(),
            finish_reason,
            prefix_cache_hit_len,
            logprobs: logprobs.map(<[crate::config::TokenLogprob]>::to_vec),
            budget_cap: None,
        })
    }
}

struct SessionDecodeLoopBackend<'a> {
    session: &'a Session,
    kv_model: Option<&'a KvModelInfo>,
    kv_cache: &'a mut PagedKvCache,
    scheduler: &'a mut Scheduler,
    session_id: SessionId,
    state: &'a mut EngineSession,
}

impl DecodeLoopBackend for SessionDecodeLoopBackend<'_> {
    fn context_len(&self) -> usize {
        self.state.tokens.len()
    }

    fn processor_prompt_tokens(&self) -> &[TokenId] {
        &self.state.tokens
    }

    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>> {
        next_session_token_logits(
            self.session,
            self.kv_model,
            self.kv_cache,
            self.session_id,
            self.state,
        )
    }

    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()> {
        self.state.tokens.push(token_id);
        self.scheduler.advance(self.session_id);
        Ok(())
    }

    fn greedy_fastpath_supported(&self) -> bool {
        self.state.decode_state.has_runner() && self.state.decode_state.runner_supports_argmax()
    }

    fn next_token_greedy(&mut self) -> anyhow::Result<TokenId> {
        next_session_token_argmax(
            self.session,
            self.kv_model,
            self.kv_cache,
            self.session_id,
            self.state,
        )?
        .context("greedy fast path unexpectedly returned no token")
    }

    fn sampled_fastpath_supported(&self) -> bool {
        !self.state.sampled_fastpath_failed
            && self.state.decode_state.has_runner()
            && self.state.decode_state.runner_supports_sampled()
    }

    fn next_token_sampled(
        &mut self,
        params: &onnx_genai_ort::DeviceSampleParams,
    ) -> anyhow::Result<Option<TokenId>> {
        next_session_token_sampled(
            self.session,
            self.kv_model,
            self.kv_cache,
            self.session_id,
            self.state,
            params,
        )
    }

    fn sampled_fastpath_failed(&mut self) {
        self.state.sampled_fastpath_failed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ort_generate_uses_scheduler_but_native_generate_bypasses_it() {
        assert!(generate_uses_scheduler(EngineDecodeBackend::Ort));
        assert!(!generate_uses_scheduler(EngineDecodeBackend::Native));
        assert!(generate_uses_scheduler(EngineDecodeBackend::Auto));
    }
}
