//! Runtime session, generation, and prefix-connector APIs.

use super::*;

use super::session_state::{CheckedPosition, SessionLen, SessionStore};

fn generate_uses_scheduler(_backend: EngineDecodeBackend) -> bool {
    true
}

/// Logical token length of an ORT-backed session, the single source of the
/// "session not found" answer for the ORT arm of the shared session policy.
fn ort_logical_len(engine: &Engine, id: SessionId) -> Option<usize> {
    engine.sessions.get(&id).map(|state| state.tokens.len())
}

/// Logical token length of a native-backed session.
#[cfg(feature = "native-backend")]
fn native_logical_len(engine: &Engine, id: SessionId) -> Option<usize> {
    engine
        .native_sessions
        .get(&id)
        .map(|state| state.tokens.len())
}

/// The ORT backend as a [`SessionStore`]: a session in `Engine::sessions`, its
/// paged KV sequence, its scheduler entry, and any aligned draft state.
struct OrtSessions<'a>(&'a mut Engine);

/// Read-only view of the ORT backend for `&self` policy calls (checkpoint,
/// token count), which need only the logical length.
struct OrtSessionsRef<'a>(&'a Engine);

impl SessionLen for OrtSessions<'_> {
    fn logical_len(&self, id: SessionId) -> Option<usize> {
        ort_logical_len(self.0, id)
    }
}

impl SessionLen for OrtSessionsRef<'_> {
    fn logical_len(&self, id: SessionId) -> Option<usize> {
        ort_logical_len(self.0, id)
    }
}

impl SessionStore for OrtSessions<'_> {
    fn validate_rewind(&self, id: SessionId, target: CheckedPosition) -> anyhow::Result<()> {
        let engine = &*self.0;
        let position = target.get();
        // Existence is guaranteed by the shared policy; a vanished session is a
        // no-op here rather than a second not-found path.
        let Some(state) = engine.sessions.get(&id) else {
            return Ok(());
        };
        validate_target_state_rewind_to_len(
            engine.kv_model.as_ref(),
            &engine.kv_cache,
            id,
            state,
            RewindRequest::new(position, RewindRunnerPolicy::RejectRunnerRewind),
        )?;
        if let (Some(draft_model), Some(draft)) = (&engine.draft, &state.draft) {
            let draft_target = position.min(draft.tokens.len());
            validate_draft_state_rewind_to_len(
                draft_model,
                draft,
                RewindRequest::new(draft_target, RewindRunnerPolicy::RejectRunnerRewind),
            )?;
        }
        Ok(())
    }

    fn rewind(&mut self, id: SessionId, target: CheckedPosition) -> anyhow::Result<()> {
        let engine = &mut *self.0;
        let position = target.get();
        engine.scheduler.complete(id);
        let mut state = engine
            .sessions
            .remove(&id)
            .with_context(|| format!("session {id} not found"))?;
        let result = (|| {
            let session = engine.session.as_deref().context(MISSING_ORT_SESSION)?;
            rewind_target_state_to_len(
                session,
                engine.kv_model.as_ref(),
                &mut engine.kv_cache,
                id,
                &mut state,
                RewindRequest::new(position, RewindRunnerPolicy::RejectRunnerRewind),
            )?;
            if let (Some(draft_model), Some(draft)) = (&mut engine.draft, &mut state.draft) {
                let draft_target = position.min(draft.tokens.len());
                rewind_draft_state_to_len(
                    draft_model,
                    draft,
                    RewindRequest::new(draft_target, RewindRunnerPolicy::RejectRunnerRewind),
                )?;
            }
            Ok(())
        })();
        engine.sessions.insert(id, state);
        result
    }

    fn reset(&mut self, id: SessionId) -> anyhow::Result<()> {
        let engine = &mut *self.0;
        engine.scheduler.complete(id);
        engine
            .kv_cache
            .remove(id)
            .map_err(|e| anyhow::anyhow!("Failed to reset KV sequence {id}: {e}"))?;
        engine.kv_cache.page_table.create_sequence(id);
        let decode_state = engine.new_target_decode_state()?;
        let state = engine
            .sessions
            .get_mut(&id)
            .context("session disappeared during reset")?;
        state.tokens.clear();
        state.kv_token_count = 0;
        state.decode_state = decode_state;
        if let (Some(draft_model), Some(draft)) = (&mut engine.draft, &mut state.draft) {
            draft_model
                .kv_cache
                .remove(draft.seq)
                .map_err(|e| anyhow::anyhow!("Failed to reset draft KV sequence: {e}"))?;
            draft.seq = draft_model.kv_cache.create_sequence();
            draft.tokens.clear();
            draft.kv_token_count = 0;
            draft.decode_state = DecodeState::new_for_path_with_io(
                &draft_model.session,
                &draft_model.decode_path,
                draft_model.io.as_ref(),
            )?;
        }
        Ok(())
    }

    fn close(&mut self, id: SessionId) -> anyhow::Result<()> {
        let engine = &mut *self.0;
        engine.scheduler.complete(id);
        let state = engine
            .sessions
            .remove(&id)
            .with_context(|| format!("session {id} not found"))?;
        engine
            .kv_cache
            .remove(id)
            .map_err(|e| anyhow::anyhow!("Failed to remove KV sequence {id}: {e}"))?;
        if let (Some(draft_model), Some(draft)) = (&mut engine.draft, state.draft) {
            draft_model
                .kv_cache
                .remove(draft.seq)
                .map_err(|e| anyhow::anyhow!("Failed to remove draft KV sequence: {e}"))?;
        }
        Ok(())
    }
}

/// The native backend as a [`SessionStore`]: a token history in
/// `Engine::native_sessions` and one in-process decoder that is rewound or reset
/// only while it holds this session's KV (`native_active_session`).
#[cfg(feature = "native-backend")]
struct NativeSessions<'a>(&'a mut Engine);

/// Read-only view of the native backend for `&self` policy calls.
#[cfg(feature = "native-backend")]
struct NativeSessionsRef<'a>(&'a Engine);

#[cfg(feature = "native-backend")]
impl SessionLen for NativeSessions<'_> {
    fn logical_len(&self, id: SessionId) -> Option<usize> {
        native_logical_len(self.0, id)
    }
}

#[cfg(feature = "native-backend")]
impl SessionLen for NativeSessionsRef<'_> {
    fn logical_len(&self, id: SessionId) -> Option<usize> {
        native_logical_len(self.0, id)
    }
}

#[cfg(feature = "native-backend")]
impl SessionStore for NativeSessions<'_> {
    fn validate_rewind(&self, _id: SessionId, _target: CheckedPosition) -> anyhow::Result<()> {
        // The native decoder always admits a rewind and clamps to its own
        // materialized length in `rewind`; there is no runner/draft state to
        // reject it the way ORT's does.
        Ok(())
    }

    fn rewind(&mut self, id: SessionId, target: CheckedPosition) -> anyhow::Result<()> {
        let engine = &mut *self.0;
        let position = target.get();
        if let Some(state) = engine.native_sessions.get_mut(&id) {
            state.tokens.truncate(position);
        }
        if engine.native_active_session == Some(id) {
            let native = engine
                .native_session
                .as_mut()
                .context("native decoder session is unavailable")?;
            native.rewind(position.min(native.current_len()))?;
        }
        let last_access = engine.touch_native_session();
        if let Some(state) = engine.native_sessions.get_mut(&id) {
            state.last_access = last_access;
        }
        Ok(())
    }

    fn reset(&mut self, id: SessionId) -> anyhow::Result<()> {
        let engine = &mut *self.0;
        let last_access = engine.touch_native_session();
        if let Some(state) = engine.native_sessions.get_mut(&id) {
            state.tokens.clear();
            state.last_access = last_access;
        }
        if engine.native_active_session == Some(id) {
            let native = engine
                .native_session
                .as_mut()
                .context("native decoder session is unavailable")?;
            native.reset()?;
            engine.native_active_session = None;
        }
        Ok(())
    }

    fn close(&mut self, id: SessionId) -> anyhow::Result<()> {
        let engine = &mut *self.0;
        engine.native_sessions.remove(&id);
        if engine.native_active_session == Some(id) {
            let native = engine
                .native_session
                .as_mut()
                .context("native decoder session is unavailable")?;
            native.reset()?;
            engine.native_active_session = None;
        }
        if engine.native_default_session == Some(id) {
            engine.native_default_session = None;
        }
        Ok(())
    }
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

#[cfg(feature = "native-backend")]
fn native_workspace_query_rows(
    prompt_rows: usize,
    plan: Option<&NativeSpeculationPlan>,
    effective_max_new_tokens: usize,
    max_context: Option<usize>,
) -> usize {
    let remaining_context = max_context
        .map(|limit| limit.saturating_sub(prompt_rows))
        .unwrap_or(effective_max_new_tokens);
    let verify_rows = plan.map_or(0, |plan| {
        crate::native_speculative::verification_width(
            plan.width,
            effective_max_new_tokens,
            remaining_context,
        )
    });
    prompt_rows.max(verify_rows)
}

impl Engine {
    fn admit_generate_request_with_scheduler(
        &mut self,
        session_id: SessionId,
        prompt_tokens: usize,
        max_new_tokens: usize,
        priority: Priority,
    ) -> anyhow::Result<ScheduledRequest> {
        let request_id = self.scheduler.enqueue_generate_request(
            session_id,
            prompt_tokens,
            max_new_tokens,
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
        Ok(scheduled)
    }

    /// Every token id this model ends generation on.
    ///
    /// A model may end a turn with one token and a message with another; both
    /// stop it. The package's own declaration comes first because it is the
    /// package speaking about itself — a package that ships no tokenizer
    /// side-files still states its EOS — and the tokenizer's ids extend it
    /// rather than replacing it, so neither source can silently drop the
    /// other's.
    pub(crate) fn default_eos_token_ids(&self) -> anyhow::Result<Vec<TokenId>> {
        let mut ids = self.declared_eos_token_ids()?;
        for id in self
            .tokenizer
            .as_ref()
            .map(Tokenizer::eos_token_ids)
            .unwrap_or_default()
        {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    /// EOS ids the package's workflow declares, via the `eos_token_ids` role.
    ///
    /// Read from the workflow's own literal default, so a package states its
    /// stop condition in the one place it states everything else. Without this
    /// the declaration is inert: it would be materialized into the graph's
    /// inputs and never reach the runtime's stop policy, which is what "declared
    /// but dead" meant.
    fn declared_eos_token_ids(&self) -> anyhow::Result<Vec<TokenId>> {
        let workflow = self.workflow.workflow_spec();
        workflow
            .inputs
            .values()
            .filter(|input| {
                matches!(
                    &input.role,
                    onnx_genai_metadata::SemanticInputRole::Runtime { role, .. }
                        if *role == onnx_genai_metadata::RuntimeInputRole::EosTokenIds
                )
            })
            .filter_map(|input| input.default.as_ref())
            .map(literal_token_ids)
            .collect::<anyhow::Result<Vec<_>>>()
            .map(|groups| groups.concat())
    }

    /// Apply the model's stop condition to a request.
    ///
    /// Every declared EOS id becomes a stop sequence, not just the first. A
    /// model with two end tokens is stopped by either, which a single
    /// `eos_token_id` field cannot express — and silently dropping the rest
    /// means generation runs past its end and emits control tokens as text.
    ///
    /// A caller's explicit `eos_token_id` selects which id is *reported* as the
    /// EOS, and does not suppress the others: the model's end tokens are facts
    /// about the model, and a request narrowing them would make the runtime emit
    /// tokens the model meant as terminal.
    fn apply_eos_defaults(&self, options: &mut GenerateOptions) -> anyhow::Result<()> {
        apply_eos_policy(options, &self.default_eos_token_ids()?);
        Ok(())
    }

    /// Effective context limit for a request, combining model metadata,
    /// per-request override, and decode-path capacity.
    /// The package's tokenizer, or an error naming why text decode is
    /// unavailable.
    ///
    /// A workflow package may ship no tokenizer at all (an image-generation
    /// pipeline, for instance). Making that an explicit error here keeps the
    /// absence a diagnosable load-time fact rather than a panic deep in the
    /// decode loop.
    pub(crate) fn require_tokenizer(&self) -> anyhow::Result<&Tokenizer> {
        self.tokenizer
            .as_ref()
            .context("this package declares no tokenizer, so it cannot tokenize or decode text")
    }

    pub fn effective_max_context(&self, options: &GenerateOptions) -> Option<usize> {
        self.max_context_for_request(options)
    }

    #[cfg(feature = "native-backend")]
    fn generate_native_cold_with_callback(
        &mut self,
        mut request: GenerateRequest,
        mut admission_callback: Option<&mut dyn FnMut()>,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.last_speculative_stats = SpeculativeStats::default();
        if request.options.speculative_mode.is_none() && self.mtp.is_some() {
            request.options.speculative_mode = Some(self.speculative_mode.clone());
        }
        reject_native_request_speculation(&request.options)?;
        request.options.validate()?;
        let mut options = request.options;
        self.apply_eos_defaults(&mut options)?;
        let prompt_tokens = self.tokenize_prompt(&request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        options.max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(self.require_tokenizer()?), false)?;
        let speculation_plan = native_speculation_plan(&options, &chain);
        let scheduler_session_id = self.next_native_session_id();
        let scheduled = self.admit_generate_request_with_scheduler(
            scheduler_session_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            Priority::Normal,
        )?;
        let budget_cap = scheduled.budget_cap.map(generation_budget_cap);
        options.max_new_tokens = scheduled.max_tokens;
        let workspace_query_rows = native_workspace_query_rows(
            prompt_tokens.len(),
            speculation_plan.as_ref(),
            options.max_new_tokens,
            options.max_context,
        );
        if let Err(error) = self
            .native_session
            .as_mut()
            .context("native decoder session is unavailable")?
            .prepare_generation_workspace_for_query_rows(&prompt_tokens, workspace_query_rows)
        {
            self.scheduler.complete(scheduler_session_id);
            return Err(error);
        }
        if let Some(callback) = admission_callback.as_mut() {
            callback();
        }

        // Speculation ON (implemented greedy prompt-lookup) → the native
        // speculative driver. Every other request stays on the untouched plain
        // M=1 fast path below, preserving the 762 tok/s non-regression guarantee.
        // Borrowed before the mutable native-session borrows below.
        let tokenizer = self
            .tokenizer
            .as_ref()
            .context("this package declares no tokenizer, so it cannot decode text")?;
        let runtime = &*self.workflow;
        // The authored block body this request iterates, read from the
        // package's own declared loop. Resolved before the mutable native
        // session borrow below, which the driver holds for the whole drive.
        let block_runtime = match speculation_plan.as_ref() {
            Some(_) => Some(self.workflow.iteration_runtime(
                onnx_genai_metadata::decoder_workflow::IterationPolicy::SpeculativeBlock,
            )?),
            None => None,
        };
        let result = if let Some(plan) = speculation_plan {
            let block_runtime = block_runtime.expect("a speculation plan resolves a block body");
            let mut stats = SpeculativeStats::default();
            let result = (|| {
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
                    NativeSpeculationKind::Mtp => {
                        let mtp = self.mtp.as_ref().context(
                            "native MTP speculation requested without a loaded MTP head",
                        )?;
                        // Reuse the generic MtpProposer (guaranteed target token +
                        // K speculative drafts from the ORT MTP head) through the
                        // native driver; the head runs on the ORT CUDA EP while the
                        // hybrid GDN target runs natively. The recurrent-state
                        // commit-by-accepted primitive (#1633) advances the target's
                        // GDN/conv state on accept.
                        let proposer = MtpProposer::new_owned(
                            std::sync::Arc::clone(&mtp.session),
                            onnx_genai_ort::MtpDecodeOptions {
                                kv_mode: mtp.kv_mode,
                                batch_size: 1,
                                hc_mult: mtp.runtime_config.hc_mult,
                                hidden_state_rank4: mtp.runtime_config.target_hidden_layout
                                    == MtpHiddenLayout::Bshc,
                                hidden_output: mtp.runtime_config.mtp_hidden_output.clone(),
                                state_output: mtp.runtime_config.mtp_state_output.clone(),
                            },
                            mtp.embedder.clone(),
                            mtp.lm_head.clone(),
                            mtp.runtime_config.cache_scope,
                        )?;
                        let hidden_size = mtp
                            .runtime_config
                            .hc_mult
                            .saturating_mul(mtp.config.hidden_size);
                        crate::native_speculative::NativeSpeculativeDriver::new_mtp(
                            native_session,
                            proposer,
                            hidden_size,
                            plan.width,
                        )?
                    }
                };
                driver.generate(
                    &prompt_tokens,
                    &options,
                    &chain,
                    tokenizer,
                    &block_runtime,
                    &mut stats,
                    callback,
                )
            })();
            self.last_speculative_stats = stats;
            result
        } else {
            let native_session = self
                .native_session
                .as_mut()
                .context("native decoder session is unavailable")?;
            native_session.generate_with_callback(
                &prompt_tokens,
                &options,
                &chain,
                tokenizer,
                runtime,
                callback,
            )
        };
        self.scheduler.complete(scheduler_session_id);
        let mut result = augment_backend_error(result, EngineDecodeBackend::Native)?;
        result.budget_cap = budget_cap;
        Ok(result)
    }

    #[cfg(not(feature = "native-backend"))]
    fn generate_native_cold_with_callback(
        &mut self,
        _request: GenerateRequest,
        _admission_callback: Option<&mut dyn FnMut()>,
        _callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        anyhow::bail!(
            "native decoder backend requires building onnx-genai-engine with the 'native-backend' feature"
        )
    }

    #[cfg(feature = "native-backend")]
    fn next_native_session_id(&mut self) -> SessionId {
        self.native_session_counter = self.native_session_counter.saturating_add(1);
        SessionId::from(self.native_session_counter)
    }

    /// Stamp a session as most-recently-used and return the new stamp.
    #[cfg(feature = "native-backend")]
    fn touch_native_session(&mut self) -> u64 {
        self.native_access_counter = self.native_access_counter.saturating_add(1);
        self.native_access_counter
    }

    #[cfg(feature = "native-backend")]
    fn create_native_session_state(&mut self) -> anyhow::Result<SessionId> {
        if self.decode_backend != EngineDecodeBackend::Native {
            anyhow::bail!("native session state requires the native decode backend");
        }
        let id = self.next_native_session_id();
        let last_access = self.touch_native_session();
        self.native_sessions.insert(
            id,
            NativeSessionState {
                tokens: Vec::new(),
                last_access,
            },
        );
        self.evict_native_sessions(id);
        Ok(id)
    }

    /// Drop least-recently-used sessions until the retention limit is met.
    ///
    /// `keep` is the session the caller is currently working with and is
    /// structurally excluded: there is a `filter` and deliberately **no**
    /// fallback scan. An earlier version had one, and under a byte budget it
    /// let a session pick itself as the victim and be deleted while the caller
    /// was writing to it -- generation returned `Ok`, and the failure surfaced
    /// later as a bare "session not found".
    ///
    /// With only a count limit that path is unreachable, because every caller
    /// touches `keep` immediately before evicting, making it the *most*
    /// recently used. So this is defensive and no test claims to cover it. It
    /// is written this way because adding any limit that `keep` alone can
    /// exceed would make a fallback reachable again instantly.
    ///
    /// This bounds retained history, not memory. Native sessions hold only a
    /// token list -- one KV cache exists and switching resets it -- so a byte
    /// budget here would bound nothing. That belongs to the resource governor
    /// once native sessions hold leases on the central KV manager.
    #[cfg(feature = "native-backend")]
    fn evict_native_sessions(&mut self, keep: SessionId) {
        while self.native_max_sessions != 0 && self.native_sessions.len() > self.native_max_sessions
        {
            let Some(victim) = self
                .native_sessions
                .iter()
                .filter(|(id, _)| **id != keep)
                .min_by_key(|(_, state)| state.last_access)
                .map(|(&id, _)| id)
            else {
                break;
            };
            self.native_sessions.remove(&victim);
            if self.native_active_session == Some(victim) {
                self.native_active_session = None;
            }
            if self.native_default_session == Some(victim) {
                self.native_default_session = None;
            }
        }
    }

    #[cfg(feature = "native-backend")]
    fn default_native_session(&mut self) -> anyhow::Result<SessionId> {
        if let Some(id) = self.native_default_session
            && self.native_sessions.contains_key(&id)
        {
            return Ok(id);
        }
        let id = self.create_native_session_state()?;
        self.native_default_session = Some(id);
        Ok(id)
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

    /// Mutable access to the engine-owned native decode session, if any.
    ///
    /// Exposed so measurement harnesses can drive the batch-N decode entry
    /// points (`decode_greedy_batch`) against the *fully governed* session —
    /// the one whose weight-offload policy carries the managed no-spill
    /// stable-VA authority. A bare `load_with_resolved_io` session takes the
    /// conservative pointer-unstable default and would decline capture by
    /// construction, so a harness that wants to observe the real
    /// capture-plus-streaming behaviour must go through the engine.
    #[cfg(feature = "native-backend")]
    pub fn native_decode_session_mut(
        &mut self,
    ) -> Option<&mut crate::native_decode::NativeDecodeSession> {
        self.native_session.as_mut()
    }

    /// Teacher-forced native generation over an exact token prefix.
    ///
    /// Divergence and decode-lock audits feed a fixed prefix and read the top
    /// log-probabilities of the single next token, which the ordinary text
    /// entry points cannot express: they tokenize a prompt, and the whole point
    /// is to pin the *exact* ids an oracle agreed on. Exposed here rather than
    /// on the native session so a caller never needs the interpreter type.
    #[cfg(feature = "native-backend")]
    pub fn generate_native_from_token_ids(
        &mut self,
        prompt_tokens: &[TokenId],
        options: &GenerateOptions,
        chain: &crate::logits::ProcessorChain,
        tokenizer: &Tokenizer,
    ) -> anyhow::Result<GenerateResult> {
        let runtime = &*self.workflow;
        let session = self
            .native_session
            .as_mut()
            .context("native decoder session is unavailable")?;
        session.generate(prompt_tokens, options, chain, tokenizer, runtime)
    }

    /// Whether this runtime holds the fused decode session that implements the
    /// declared `onnx-genai.autoregressive-decode` step.
    ///
    /// A question about which executor exists, not about what kind of package
    /// was loaded: a package whose decode step this runtime has no fused
    /// session for still runs the same declared loop, with the interpreter
    /// invoking the component from its own artifact.
    pub(crate) fn holds_decode_core(&self) -> bool {
        #[cfg(feature = "native-backend")]
        if self.native_session.is_some() {
            return true;
        }
        self.session.is_some()
    }

    /// Continue a conversation the interpreter keeps in session-scoped state.
    ///
    /// The session id is bound into the request rather than looked up in a
    /// decode core: a workflow's `scope: session` cells are keyed by it, and
    /// that is where this package's conversation lives.
    pub(crate) fn generate_in_workflow_session(
        &mut self,
        session_id: SessionId,
        request: crate::pipeline::PipelineGenerateRequest,
        on_admitted: Option<&mut dyn FnMut()>,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        anyhow::ensure!(
            self.workflow_sessions.contains_key(&session_id),
            "session {session_id} not found"
        );
        let result = self.generate_with_pipeline_callbacks(
            request.with_session_id(session_id.to_string()),
            on_admitted,
            callback,
        )?;
        // A package that declares its conversation knows how long it is; asking
        // it is what keeps `session_token_count` the same number a decode-core
        // session reports — prompt and generated tokens both — rather than the
        // generated ones alone.
        //
        // The cost of that conversation is stated rather than hidden: a package
        // whose lease is a prompt prefix re-prefills every earlier turn on each
        // new one, because its invocation-scoped cache is released when the
        // invocation ends. Over a conversation of N tokens that is O(N²) prefill
        // work, against O(N) for a decode core whose paged KV survives the turn.
        // It is the cost of continuing a conversation a package can express
        // rather than one it cannot, and a package that wants the linear cost
        // declares its cache session-scoped and is executed by a core that keeps
        // it.
        let declared = self
            .workflow
            .session_conversation_len(&session_id.to_string());
        if let Some(count) = self.workflow_sessions.get_mut(&session_id) {
            match declared {
                Some(length) => *count = length,
                None => *count += result.token_ids.len(),
            }
        }
        Ok(result)
    }

    /// Generate by interpreting every declared component from its artifact.
    ///
    /// The same drive the decode-core path uses, with no core: the interpreter
    /// walks the same declared loop, honours the same bound and predicate, and
    /// publishes the same `tokens` output. What it does not have is a fused
    /// session to route the decode step to, so it invokes the component the
    /// package names.
    fn generate_interpreted(
        &mut self,
        request: GenerateRequest,
        on_admitted: Option<&mut dyn FnMut()>,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_with_pipeline_callbacks(
            crate::pipeline::PipelineGenerateRequest::new(request),
            on_admitted,
            callback,
        )
    }

    /// Admit a prompt-only request through the scheduler for a package this
    /// runtime has no decode core for.
    ///
    /// A no-decode-core request still shares this runtime's scheduler, and —
    /// when the process configured one — its KV byte budget: components own
    /// their own caches, but the byte accounting a shared budget protects is
    /// shared regardless of which executor a step names. Called from
    /// [`crate::engine::workflow_api::Engine::generate_with_pipeline_callbacks`]'s
    /// no-decode-core branch — the one place every prompt-only request for
    /// such a package passes through, whether it arrived cold
    /// ([`Self::generate_interpreted`]) or through a continuing workflow
    /// session ([`Self::generate_in_workflow_session`]) — this gives the path
    /// the same "reject at the door" guarantee
    /// [`Self::generate_native_cold_with_callback`] already has instead of
    /// letting the request fail deep inside node execution once a value the
    /// loop needed never arrives.
    pub(crate) fn admit_interpreted_generate_request(
        &mut self,
        session_id: SessionId,
        prompt: &GeneratePrompt,
        max_new_tokens: usize,
    ) -> anyhow::Result<(Option<GenerationBudgetCap>, usize)> {
        // What this request costs is its prompt plus whatever the runtime will
        // put in front of it, which for a package continuing its conversation by
        // prepending is every earlier turn. Admitting on the request alone
        // under-reserves for exactly the turns that need the most.
        let carried = if self.workflow_sessions.contains_key(&session_id) {
            self.workflow
                .session_prepended_prompt_len(&session_id.to_string())
        } else {
            0
        };
        let prompt_tokens = self.interpreted_prompt_token_count(prompt)? + carried;
        let scheduled = self.admit_generate_request_with_scheduler(
            session_id,
            prompt_tokens,
            max_new_tokens,
            Priority::Normal,
        )?;
        Ok((
            scheduled.budget_cap.map(generation_budget_cap),
            scheduled.max_tokens,
        ))
    }

    /// Prompt length for scheduler admission before a no-decode-core request's
    /// workflow has bound any input.
    ///
    /// The interpreter tokenizes a text prompt itself once the workflow runs
    /// (from the package's own tokenizer, since this engine owns none for a
    /// workflow package); admission needs that count earlier, to gate the run
    /// rather than join it, so it is derived the same way here rather than
    /// waiting for the loop to do it.
    fn interpreted_prompt_token_count(&self, prompt: &GeneratePrompt) -> anyhow::Result<usize> {
        match prompt {
            GeneratePrompt::TokenIds(tokens) => Ok(tokens.len()),
            // Equal-length rows bind into one `[rows, columns]` tensor and run
            // as a single batched step (see
            // `workflow.rs::workflow_request_value`'s `PromptTokens` binding),
            // so the KV byte budget a batch of `rows` sequences needs scales
            // with the whole rectangle, not just its widest row.
            GeneratePrompt::TokenRows(rows) => {
                Ok(rows.iter().map(Vec::len).max().unwrap_or(0) * rows.len())
            }
            GeneratePrompt::Text(text) => {
                let tokenizer = self.workflow.package_tokenizer().context(
                    "this package declares a prompt_tokens input but ships no tokenizer, so a \
                     text prompt cannot be encoded for it; supply token ids instead",
                )?;
                tokenizer
                    .encode(text)
                    .map(|ids| ids.len())
                    .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}"))
            }
        }
    }

    /// Access the engine-owned Resource Governor handle.
    pub fn governor(&self) -> &EngineResourceGovernor {
        &self.governor
    }

    /// Convenience snapshot of configured and live resource state.
    pub fn resource_snapshot(&self) -> GovernorSnapshot {
        self.governor().snapshot()
    }

    /// Bytes by which the device memory ledger exceeds its live ceiling.
    pub fn device_oversubscribed_bytes(&self) -> u64 {
        self.governor().device_oversubscribed_bytes()
    }

    /// Static weight placement computed from `device_policy` at model load.
    pub fn weight_placement_report(&self) -> Option<&WeightPlacementReport> {
        #[cfg(feature = "native-backend")]
        {
            self.weight_placement.as_ref()
        }
        #[cfg(not(feature = "native-backend"))]
        {
            None
        }
    }

    pub fn memory_strategy_plan(&self) -> &MemoryStrategyPlan {
        &self.memory_strategy_plan
    }

    /// Change the live VRAM ceiling when runtime overrides are enabled.
    pub fn set_vram_limit(
        &self,
        limit: ResourceLimit,
    ) -> Result<GovernorReconfigureOutcome, EngineGovernorError> {
        self.governor().set_vram_limit(limit)
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

    /// Attach a lock-free mirror of the KV pool's aggregate state.
    ///
    /// [`page_usage`](Self::page_usage) cannot serve a live monitor: it is
    /// `O(pages)` and allocates, and reaching it requires `&self` on an engine
    /// that is mutably borrowed for the whole of a generation. The mirror is
    /// updated incrementally at the pool's mutation sites and read from any
    /// thread, so a reader can observe the pool *during* generation, which is
    /// the only time paged-KV behaviour is worth watching.
    ///
    /// Returns `false` when this engine's KV cannot page, so a caller can
    /// report the numbers as not-applicable rather than presenting a pool that
    /// will never move as one that is merely idle. The pool is attached either
    /// way: its capacity is a real number, and a client that knows the
    /// mechanism is inactive can still show it truthfully.
    ///
    /// A caller must not infer this from the *absence* of some other
    /// capability. See [`pages_kv`](Self::pages_kv).
    pub fn attach_kv_telemetry(
        &mut self,
        telemetry: std::sync::Arc<onnx_genai_kv::KvTelemetry>,
    ) -> bool {
        self.kv_cache.page_table.attach_telemetry(telemetry);
        self.pages_kv()
    }

    /// Whether this engine's KV cache can actually hold paged tensors.
    ///
    /// Both halves are required and neither implies the other: an engine can
    /// own a KV model while its page table carries no tensor storage, in which
    /// case the page table is bookkeeping the decoder never consults.
    ///
    /// Do not derive this from the absence of another capability. A pool that
    /// reports a non-zero capacity is not thereby in use, which is exactly the
    /// reading that makes an inactive mechanism look active.
    pub fn pages_kv(&self) -> bool {
        self.kv_model.is_some() && self.kv_cache.page_table.tensor_config.is_some()
    }

    /// External KV connector activity from the most recent generation.
    ///
    /// Reflects lookups, would-be prefix extensions, tokens actually fetched and
    /// injected (K4 materialization), and chunk stores. Returns
    /// [`ConnectorStats::default`] when no connector is configured.
    pub fn last_connector_stats(&self) -> ConnectorStats {
        self.connector.stats().clone()
    }

    #[cfg(feature = "native-backend")]
    pub fn recurrent_prefix_cache_stats(&self) -> RecurrentPrefixCacheStats {
        self.native_recurrent_prefix_stats
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
        self.generate_fim_with_config_and_callbacks(prefix, suffix, options, fim_config, None, None)
    }

    /// Generate FIM text and notify the caller immediately after scheduler admission.
    pub fn generate_fim_with_config_and_callbacks(
        &mut self,
        prefix: impl AsRef<str>,
        suffix: impl AsRef<str>,
        options: GenerateOptions,
        fim_config: &FimConfig,
        admission_callback: Option<&mut dyn FnMut()>,
        token_callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        let prompt = fim_config.format_prompt(prefix.as_ref(), suffix.as_ref());
        let mut request = GenerateRequest::new(prompt);
        request.options = self.fim_options(fim_config, options)?;
        self.generate_with_callbacks(request, admission_callback, token_callback)
    }

    /// Generate text and optionally stream each generated token to `callback`.
    pub fn generate_with_callback(
        &mut self,
        request: GenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_with_callbacks(request, None, callback)
    }

    /// Generate text, notifying `admission_callback` after scheduler admission
    /// and before backend execution, then streaming tokens to `token_callback`.
    pub fn generate_with_callbacks(
        &mut self,
        request: GenerateRequest,
        admission_callback: Option<&mut dyn FnMut()>,
        token_callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        // One entry point, one interpreter, one declared loop. What varies is
        // whether this runtime holds the fused decode session that implements
        // the package's declared `autoregressive-decode` step. Without one, the
        // interpreter invokes every declared component from the artifact the
        // package names — the same loop, the same emits, the same stop.
        if !self.holds_decode_core() {
            return self.generate_interpreted(request, admission_callback, token_callback);
        }
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            // Speculation still runs cold: the native speculative paths own
            // the KV cache themselves and cannot resume a shared prefix. Both
            // cold and session-reusing paths still admit through the scheduler
            // before touching the native backend.
            let native_spec_requested = request.options.speculative_mode.is_some()
                || request.options.num_speculative_tokens.is_some()
                || self.mtp.is_some();
            if request.options.cold_start || native_spec_requested {
                let result = self.generate_native_cold_with_callback(
                    request,
                    admission_callback,
                    token_callback,
                );
                self.native_active_session = None;
                return result;
            }
            let session_id = self.default_native_session()?;
            return self.generate_native_in_session_with_callbacks(
                session_id,
                request,
                admission_callback,
                token_callback,
            );
        }
        if !generate_uses_scheduler(self.decode_backend) {
            #[cfg(not(feature = "native-backend"))]
            {
                return self.generate_native_cold_with_callback(
                    request,
                    admission_callback,
                    token_callback,
                );
            }
        }
        let session_id = self.create_session()?;
        let result = self.generate_in_session_with_priority_and_callback(
            session_id,
            request,
            Priority::Normal,
            None,
            admission_callback,
            token_callback,
        );
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
            session_id, request, priority, None, None, None,
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
            None,
            callback,
        )
    }

    /// Generate in a persistent session with an explicit admission notification.
    pub fn generate_in_session_with_callbacks(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        admission_callback: Option<&mut dyn FnMut()>,
        token_callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_priority_and_callback(
            session_id,
            request,
            Priority::Normal,
            None,
            admission_callback,
            token_callback,
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
            None,
        )
    }

    fn generate_in_session_with_priority_and_callback(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        priority: Priority,
        mut custom_sampler: Option<Box<dyn Sampler>>,
        mut admission_callback: Option<&mut dyn FnMut()>,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        // A package with no decode core keeps its conversation in the
        // session-scoped cells its workflow declares. Routing here rather than
        // in one of the wrappers above is what makes every `generate_in_session`
        // variant reach it — a caller that picked the priority form should not
        // get a different answer about whether sessions exist.
        if !self.holds_decode_core() {
            anyhow::ensure!(
                custom_sampler.is_none(),
                "a package whose components the interpreter invokes declares its sampler; a \
                 caller-supplied one would replace a step the package states in its own graph"
            );
            return self.generate_in_workflow_session(
                session_id,
                crate::pipeline::PipelineGenerateRequest::new(request),
                admission_callback,
                callback,
            );
        }
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            if priority != Priority::Normal {
                anyhow::bail!("native backend does not support prioritized session generation");
            }
            if custom_sampler.is_some() {
                anyhow::bail!("custom samplers are not supported on the native backend");
            }
            return self.generate_native_in_session_with_callbacks(
                session_id,
                request,
                admission_callback,
                callback,
            );
        }
        self.last_speculative_stats = SpeculativeStats::default();
        request.options.validate()?;
        let mut options = request.options.clone();
        self.apply_eos_defaults(&mut options)?;
        let prompt_tokens = self.tokenize_prompt(&request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        if !self.sessions.contains_key(&session_id) {
            anyhow::bail!("session {session_id} not found");
        }

        let max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(
            &options,
            Some(self.require_tokenizer()?),
            custom_sampler.is_some(),
        )?;

        let scheduled = self.admit_generate_request_with_scheduler(
            session_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            priority,
        )?;
        let budget_cap = scheduled.budget_cap.map(generation_budget_cap);
        options.max_new_tokens = scheduled.max_tokens;
        if let Some(callback) = admission_callback.as_mut() {
            callback();
        }

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

            // Borrow the tokenizer and the canonical workflow before the
            // disjoint mutable borrows below: the decode backend takes
            // `&mut self.kv_cache` / `&mut self.scheduler`, so a later `&self`
            // accessor call would overlap.
            let tokenizer = self
                .tokenizer
                .as_ref()
                .context("this package declares no tokenizer, so it cannot decode text")?;
            let runtime = &*self.workflow;
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
            // Every generated token comes out of the interpreter walking the
            // package's declared loop. The backend below is the executor the
            // interpreter routes the declared `autoregressive-decode` step to
            // -- one forward pass, KV stays its business -- and the token
            // policy beside it is the single sampling/stopping implementation,
            // shared with every other package.
            crate::pipeline::generation::generate_with_decode_core(
                runtime,
                &mut backend,
                &mut loop_state,
                &prompt_tokens,
                crate::pipeline::generation::GenerationRequest {
                    options: &options,
                    chain: &chain,
                    tokenizer,
                    max_context,
                },
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
        // A package whose components the interpreter invokes has no paged KV
        // sequence to open, but it does have sessions: its workflow may declare
        // `scope: session` state, and this is the id that state is keyed by.
        // Refusing here would have made "sessions" a property of which executor
        // runs the decode step, which is not what a caller is asking about.
        if !self.holds_decode_core() {
            // What a caller opening a session asks for is that the next turn
            // continue this one. A package that publishes a token stream is one
            // whose turns are a conversation, so a session over it that carries
            // nothing would silently restart every turn — a wrong answer that
            // reads like a forgetful model. Say so instead, and name what the
            // package has to declare. A package that publishes no tokens has no
            // conversation to lose, and its session is an ordinary handle.
            let workflow = self.workflow.workflow_spec();
            let publishes_tokens = workflow
                .outputs
                .values()
                .any(|output| output.role == onnx_genai_metadata::WorkflowOutputRole::Tokens);
            if publishes_tokens && !crate::pipeline::workflow_carries_session_state(workflow) {
                // A typed refusal, not a formatted string: what a front end has
                // to decide is whether the caller asked for something this
                // package cannot do, or whether the server failed. Matching on
                // prose would make that a guess, and it was being reported as a
                // 500 for a package that is simply stateless.
                return Err(PackageCapabilityError::NoSessionState.into());
            }
            self.workflow_session_counter += 1;
            let id = self.workflow_session_counter;
            self.workflow_sessions.insert(id, 0);
            return Ok(id);
        }
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            return self.create_native_session_state();
        }
        let decode_state = self.new_target_decode_state()?;
        let id = self.kv_cache.create_sequence();
        let draft = if let Some(draft_model) = &mut self.draft {
            Some(DraftSession {
                seq: draft_model.kv_cache.create_sequence(),
                tokens: Vec::new(),
                kv_token_count: 0,
                decode_state: DecodeState::new_for_path_with_io(
                    &draft_model.session,
                    &draft_model.decode_path,
                    draft_model.io.as_ref(),
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

    /// Checkpoint the current logical token boundary for a persistent session.
    ///
    /// The returned checkpoint is intentionally small: restoring a session uses
    /// the same rewind machinery as speculative decoding and keeps prefix-cache
    /// page ownership intact. Checkpoints are invalid after the session is
    /// closed or reset.
    pub fn checkpoint_session(&self, session_id: SessionId) -> anyhow::Result<SessionCheckpoint> {
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            return session_state::checkpoint(&NativeSessionsRef(self), session_id);
        }
        self.require_ort_backend("session checkpoints")?;
        session_state::checkpoint(&OrtSessionsRef(self), session_id)
    }

    /// Restore a persistent session to a previously checkpointed token boundary.
    pub fn restore_session(&mut self, checkpoint: SessionCheckpoint) -> anyhow::Result<()> {
        self.rewind_session_to(checkpoint.session_id, checkpoint.position)
    }

    /// Rewind a persistent session by `tokens` logical tokens.
    ///
    /// This mutates only the named session. Cached prefixes remain valid because
    /// they own their page-table references independently of session ownership;
    /// pages that are no longer referenced by the session or a prefix cache entry
    /// are released by the underlying paged KV cache.
    pub fn rewind_session_by(
        &mut self,
        session_id: SessionId,
        tokens: RewindTokenCount,
    ) -> anyhow::Result<SessionPosition> {
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            return session_state::rewind_by(&mut NativeSessions(self), session_id, tokens.get());
        }
        self.require_ort_backend("session rewind")?;
        session_state::rewind_by(&mut OrtSessions(self), session_id, tokens.get())
    }

    /// Rewind a persistent session to an absolute logical token position.
    ///
    /// Rewind reuses the same KV truncation path as speculative decoding, after
    /// validating the requested target against the logical length and backend KV
    /// support so rejected rewinds leave the session untouched.
    pub fn rewind_session_to(
        &mut self,
        session_id: SessionId,
        position: SessionPosition,
    ) -> anyhow::Result<()> {
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            return session_state::rewind_to(&mut NativeSessions(self), session_id, position);
        }
        self.require_ort_backend("session rewind")?;
        session_state::rewind_to(&mut OrtSessions(self), session_id, position)
    }

    /// Capability for session fork, if the selected backend supports safe CoW
    /// fork at the engine level.
    ///
    /// Current ORT decode runners do not expose clone/import semantics strong
    /// enough to fork without deep-copying or aliasing mutable KV, so this
    /// returns `None` today. A future supported backend should return `Some` and
    /// route fork through [`Engine::fork_session`].
    pub fn session_fork_capability(&self) -> Option<SessionForkCapability> {
        None
    }

    /// Fork a persistent session at a logical token boundary.
    ///
    /// Callers can obtain `capability` only from
    /// [`Engine::session_fork_capability`], which is `None` for all current
    /// backends. Keeping the capability token in the signature prevents
    /// unsupported engines from being asked to fork through the typed API.
    pub fn fork_session(
        &mut self,
        _capability: &SessionForkCapability,
        source: SessionId,
        position: SessionPosition,
    ) -> anyhow::Result<SessionId> {
        self.require_ort_backend("session fork")?;
        let state = self
            .sessions
            .get(&source)
            .with_context(|| format!("session {source} not found"))?;
        let position = position.get();
        let current = state.tokens.len();
        if position > current {
            anyhow::bail!(
                "cannot fork session {source} at token {position}; current length is {current}"
            );
        }
        anyhow::bail!(
            "session fork is not yet enabled: safe CoW fork requires cloneable/importable decoder state aligned with paged KV; current ORT runner/static-cache paths would require deep-copying or unsafe KV aliasing"
        )
    }

    /// Reset a persistent session, freeing its current state while keeping the id usable.
    pub fn reset_session(&mut self, session_id: SessionId) -> anyhow::Result<()> {
        // Resetting is the same promise for every package: the id stays usable
        // and everything the conversation accumulated is gone. For an
        // interpreted package that is exactly its session-scoped cells, so the
        // lease is dropped and the token count returns to zero.
        if !self.holds_decode_core() {
            anyhow::ensure!(
                self.workflow_sessions.contains_key(&session_id),
                "session {session_id} not found"
            );
            self.workflow.forget_session(&session_id.to_string());
            if let Some(count) = self.workflow_sessions.get_mut(&session_id) {
                *count = 0;
            }
            return Ok(());
        }
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            return session_state::reset(&mut NativeSessions(self), session_id);
        }
        self.require_ort_backend("persistent sessions")?;
        session_state::reset(&mut OrtSessions(self), session_id)
    }

    fn new_target_decode_state(&self) -> anyhow::Result<DecodeState> {
        let session = self
            .session
            .as_deref()
            .context("ORT decoder session is unavailable")?;
        // Bind ports from explicit metadata or unambiguous tensor shapes.
        let io = self.metadata.decoder_io();
        let fixed_state_budget_bytes = self.governor().snapshot().resolved_limits.host_ram_bytes;
        if matches!(
            &self.speculative_mode,
            SpeculativeMode::Mtp(_) | SpeculativeMode::Eagle3(_)
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
        if !self.holds_decode_core() {
            anyhow::ensure!(
                self.workflow_sessions.remove(&session_id).is_some(),
                "session {session_id} not found"
            );
            self.workflow.forget_session(&session_id.to_string());
            return Ok(());
        }
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            return session_state::close(&mut NativeSessions(self), session_id);
        }
        self.require_ort_backend("persistent sessions")?;
        session_state::close(&mut OrtSessions(self), session_id)
    }

    /// Logical tokens this session's conversation holds — prompts and
    /// generations alike, oldest turn first.
    ///
    /// One meaning for every backend. A decode core reports the tokens its KV
    /// sequence covers; an interpreted package reports the length of the
    /// conversation its workflow declares, or, for a package whose session state
    /// is not a token conversation, the tokens its turns have generated. In
    /// every case it is "how much has this session heard", never "how much did
    /// the last turn produce".
    pub fn session_token_count(&self, session_id: SessionId) -> anyhow::Result<usize> {
        if !self.holds_decode_core() {
            return self
                .workflow_sessions
                .get(&session_id)
                .copied()
                .with_context(|| format!("session {session_id} not found"));
        }
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            return session_state::token_count(&NativeSessionsRef(self), session_id);
        }
        self.require_ort_backend("persistent sessions")?;
        session_state::token_count(&OrtSessionsRef(self), session_id)
    }

    /// Whether the loaded package continues a conversation by putting it in
    /// front of the next turn's prompt.
    ///
    /// This answers only whether the authored workflow explicitly prepends a
    /// token conversation. It is **false** for everything else:
    ///
    /// * a **decode core** keeps its conversation in KV, which the request does
    ///   not carry and the prefill does not repeat;
    /// * a **loop-carried** or **group-held** lease is handed back inside the
    ///   graph, so the tokens it stands for live in a cache the package bounds
    ///   itself rather than in front of a prompt;
    /// * a package with no session state has no conversation at all.
    ///
    /// Answered from the shared classifier, never from "does this session hold
    /// state": those are different questions with the same shape.
    pub fn prepends_session_conversation(&self) -> bool {
        if self.holds_decode_core() {
            return false;
        }
        self.workflow.prepends_session_conversation()
    }

    /// What this session contributes ahead of the next prompt.
    ///
    /// ORT decode-core sessions append a turn to their retained sequence, so
    /// that sequence is attended but served from KV rather than re-prefilled.
    /// Native decode-core sessions replace their cached prefix with the
    /// incoming prompt, so their retained tokens contribute neither value.
    /// Prompt continuation contributes both; graph-carried state contributes
    /// neither because the package bounds that cache itself.
    pub fn session_prefill_carry(
        &self,
        session_id: SessionId,
    ) -> anyhow::Result<SessionPrefillCarry> {
        if self.holds_decode_core() {
            let retained = self.session_token_count(session_id)?;
            #[cfg(feature = "native-backend")]
            if self.decode_backend == EngineDecodeBackend::Native {
                return Ok(SessionPrefillCarry::default());
            }
            return Ok(SessionPrefillCarry {
                attended: retained,
                reprefilled: 0,
            });
        }
        anyhow::ensure!(
            self.workflow_sessions.contains_key(&session_id),
            "session {session_id} not found"
        );
        let prepended = self
            .workflow
            .session_prepended_prompt_len(&session_id.to_string());
        Ok(SessionPrefillCarry {
            attended: prepended,
            reprefilled: prepended,
        })
    }

    /// The tokens a session's conversation holds, oldest first.
    ///
    /// `None` when the package carries its conversation somewhere a token list
    /// cannot describe — a decode core's KV sequence, or a workflow whose
    /// session state is not a declared prompt continuation. A caller reads this
    /// to see what a session has heard without keeping a second copy of it,
    /// which is the copy that drifts.
    pub fn session_conversation(
        &self,
        session_id: SessionId,
    ) -> anyhow::Result<Option<Vec<TokenId>>> {
        if self.holds_decode_core() {
            return Ok(None);
        }
        anyhow::ensure!(
            self.workflow_sessions.contains_key(&session_id),
            "session {session_id} not found"
        );
        self.workflow
            .session_conversation(&session_id.to_string())
            .map(|conversation| {
                conversation
                    .into_iter()
                    .map(|token| {
                        TokenId::try_from(token)
                            .context("a conversation token id does not fit the token type")
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .transpose()
    }

    /// Get the loaded metadata.
    pub fn metadata(&self) -> &InferenceMetadata {
        &self.metadata
    }

    /// Validated `onnx_runtime.*` hints consumed while loading the model.
    pub fn metadata_hints(&self) -> &MetadataHints {
        &self.metadata_hints
    }

    /// Resolved decoder execution backend.
    pub fn decode_backend(&self) -> EngineDecodeBackend {
        self.decode_backend
    }

    /// Execution-provider placement reported by the loaded ORT session.
    ///
    /// This is intentionally read from the live session instead of reconstructing
    /// it from requested settings, so explicit CPU fallbacks and skipped
    /// providers are visible to status/profile output.
    pub fn execution_provider_status(&self) -> String {
        // Reported by whoever owns the sessions: the decode core when it holds
        // the package's one graph, the interpreter's components otherwise.
        match self.session.as_deref() {
            Some(session) => session.execution_provider_status().summary(),
            None => self.workflow.execution_provider_status(),
        }
    }
    /// Latest native activation-memory planner measurement, if the current
    /// backend is native and has executed far enough to resolve concrete shapes.
    ///
    /// Returns `None` unless the planner is switched on, which it is not by
    /// default: set `NXRT_ACTIVATION_MEMORY_PLAN=1` (or
    /// `NXRT_EXEC_PHASE_PROFILE=1`) first. Being native and having executed is
    /// necessary but not sufficient.
    pub fn activation_memory_plan_stats(&self) -> Option<crate::ActivationMemoryPlanSummary> {
        #[cfg(feature = "native-backend")]
        {
            self.native_session
                .as_ref()
                .and_then(crate::native_decode::NativeDecodeSession::activation_memory_plan_stats)
        }
        #[cfg(not(feature = "native-backend"))]
        {
            None
        }
    }

    /// Process-global CUDA VMM arena counters, when this build has the native
    /// CUDA execution provider.
    ///
    /// `None` means the build cannot have an arena. All-zero means no arena was
    /// ever built, which is the normal state without `ONNX_GENAI_CUDA_VMM` --
    /// and is distinguishable from an arena that was built and never committed
    /// anything (`reserved_bytes > 0, commits == 0`), which is the bug #659 hid
    /// behind a log line for an entire release.
    pub fn vmm_arena_stats(&self) -> Option<crate::VmmArenaStats> {
        #[cfg(feature = "native-cuda")]
        {
            let stats = onnx_runtime_ep_cuda::vmm_allocator::global_vmm_stats();
            Some(crate::VmmArenaStats {
                commits: stats.commits,
                releases: stats.releases,
                committed_bytes: stats.committed_bytes,
                reserved_bytes: stats.reserved_bytes,
                peak_committed_bytes: stats.peak_committed_bytes,
                allocations: stats.allocations,
                ref_underflows: stats.ref_underflows,
                byte_underflows: stats.byte_underflows,
                unaccounted_committed_bytes: stats.unaccounted_committed_bytes,
            })
        }
        #[cfg(not(feature = "native-cuda"))]
        {
            None
        }
    }

    /// Auto-detected fill-in-the-middle configuration, if the tokenizer declares one.
    pub fn fim_config(&self) -> Option<&FimConfig> {
        self.fim_config.as_ref()
    }

    fn fim_options(
        &self,
        fim_config: &FimConfig,
        mut options: GenerateOptions,
    ) -> anyhow::Result<GenerateOptions> {
        self.apply_eos_defaults(&mut options)?;
        for token in [
            fim_config.prefix_token.as_str(),
            fim_config.middle_token.as_str(),
            fim_config.suffix_token.as_str(),
            "<|fim_pad|>",
            "<|endoftext|>",
            "<|file_sep|>",
        ] {
            if let Some(token_id) = self
                .tokenizer
                .as_ref()
                .and_then(|tokenizer| tokenizer.token_id(token))
            {
                push_unique_stop_sequence(
                    &mut options.stop_sequences,
                    StopSequence::Tokens(vec![token_id]),
                );
            }
        }
        Ok(options)
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
            ModelDecodePath::PastPresent { .. } | ModelDecodePath::Generic => None,
        }
    }

    /// Tokenize `text` with the model's own tokenizer.
    ///
    /// This is the public tokenization seam used by higher-level pipelines to
    /// convert prompt text into token ids (e.g. to compute prompt length or
    /// `max_length`, or to feed [`Engine::embed`] and the generation APIs). It
    /// uses the same tokenizer path as the engine's internal prompt handling.
    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<TokenId>> {
        // Whoever owns the package's tokenizer answers. The decode core opens
        // one when it holds the package's graph; a package whose components the
        // interpreter invokes keeps its tokenizer with those components.
        let Some(tokenizer) = self.tokenizer.as_ref() else {
            return self.workflow.tokenize(text);
        };
        tokenizer.encode(text).map_err(|e| {
            anyhow::anyhow!(
                "failed to tokenize input text with the model's tokenizer: {e}; \
                 verify the model directory contains a valid tokenizer.json"
            )
        })
    }

    fn tokenize_prompt(&self, prompt: &GeneratePrompt) -> anyhow::Result<Vec<TokenId>> {
        match prompt {
            GeneratePrompt::TokenIds(tokens) => Ok(tokens.clone()),
            GeneratePrompt::TokenRows(_) => {
                anyhow::bail!("multi-row prompts are supported only by workflow pipelines")
            }
            GeneratePrompt::Text(text) => self
                .require_tokenizer()?
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
            && !self.ort_session_has_recurrent_state()
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
        self.apply_eos_defaults(&mut options)?;
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
        let chain = build_processor_chain(&options, Some(self.require_tokenizer()?), false)?;
        // The declared loop is bound and its setup run before scheduler
        // admission: a package this drive cannot advance is refused without
        // having admitted the request or touched its session state.
        let cursor = crate::pipeline::WorkflowGenerationCursor::start(
            &self.workflow,
            crate::pipeline::PipelineGenerateRequest::new(GenerateRequest {
                prompt: crate::GeneratePrompt::TokenIds(prompt_tokens.clone()),
                options: options.clone(),
            }),
            crate::pipeline::generation::DECODE_CORE_CONTRACTS,
            &mut None,
        )?;
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
            cursor,
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
            // Borrowed before the disjoint mutable borrows the backend takes.
            let tokenizer = self
                .tokenizer
                .as_ref()
                .context("this package declares no tokenizer, so it cannot decode text")?;
            let runtime = &*self.workflow;
            let cursor = &mut active.cursor;
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
            // One iteration of the *declared* loop, advanced through the same
            // interpreter method the run-to-completion path drives in a `for`.
            // The scheduler owns which request runs next; the workflow owns
            // what one step of that request is.
            let mut host = crate::pipeline::generation::GenerationNodeHost::new(
                &mut backend,
                &mut loop_state,
                &crate::pipeline::generation::GenerationRequest {
                    options: &active.options,
                    chain: &active.chain,
                    tokenizer,
                    max_context: active.max_context,
                },
                None,
            );
            let (ran, finish) = {
                let mut host_ref: Option<&mut dyn crate::pipeline::WorkflowNodeHost> =
                    Some(&mut host);
                let ran = cursor.advance(runtime, &mut host_ref)?;
                (ran, host.reached_finish())
            };
            let finish = match (ran, finish) {
                (_, Some(reason)) => Some(reason),
                // The predicate ended the loop without this iteration running a
                // step, which means the previous one already reported why.
                (false, None) => Some(FinishReason::MaxTokens),
                (true, None) => None,
            };
            match finish {
                Some(finish_reason) => {
                    ensure_constrained_finish(
                        &active.options,
                        &loop_state.generated_text,
                        finish_reason.clone(),
                    )?;
                    Some(crate::decode_loop::finish_result(
                        tokenizer,
                        &loop_state.generated_tokens,
                        finish_reason,
                        loop_state.prefix_cache_hit_len,
                        loop_state.logprobs.as_deref(),
                    )?)
                }
                None => None,
            }
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
        // The workflow's own emit and the tokens this drive committed describe
        // the same generation; if they disagree, one is wrong and nothing
        // outside can tell which.
        crate::pipeline::generation::verify_emitted_tokens(
            &self.workflow,
            &active.cursor,
            &active.generated_tokens,
        )?;
        active.cursor.finish(&self.workflow)?;
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

    /// Whether the loaded ORT decoder declares loop-carried recurrent state, in
    /// which case paged / materialized-past prefix reuse must be declined and a
    /// full recompute forced (#701), mirroring the native `has_recurrent_state`
    /// gate (#700). Returns `false` when no ORT session is loaded — the paged
    /// reuse path requires one, so the guard cannot suppress a legitimate reuse.
    fn ort_session_has_recurrent_state(&self) -> bool {
        let Some(session) = self.session.as_deref() else {
            return false;
        };
        let io = self.metadata.decoder_io();
        ort_session_has_recurrent_state(session, io)
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
                .require_tokenizer()?
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

// ─── Native session shims and implementation ────────────────────────────────

#[cfg(feature = "native-backend")]
impl Engine {
    fn generate_native_in_session_with_callbacks(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        mut admission_callback: Option<&mut dyn FnMut()>,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.last_speculative_stats = SpeculativeStats::default();
        request.options.validate()?;
        let mut options = request.options;
        reject_native_request_speculation(&options)?;
        self.apply_eos_defaults(&mut options)?;
        let prompt_tokens = self.tokenize_prompt(&request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        options.max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(self.require_tokenizer()?), false)?;
        if native_speculation_plan(&options, &chain).is_some() {
            anyhow::bail!(
                "native session generation does not support speculative decoding; use stateless generate() for native prompt-lookup/shared-KV speculation"
            );
        }
        if !self.native_sessions.contains_key(&session_id) {
            anyhow::bail!("session {session_id} not found");
        }
        let scheduled = self.admit_generate_request_with_scheduler(
            session_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            Priority::Normal,
        )?;
        let budget_cap = scheduled.budget_cap.map(generation_budget_cap);
        options.max_new_tokens = scheduled.max_tokens;
        if let Err(error) = self
            .native_session
            .as_mut()
            .context("native decoder session is unavailable")?
            .prepare_generation_workspace_preserving_state(&prompt_tokens)
        {
            self.scheduler.complete(session_id);
            return Err(error);
        }
        if let Some(callback) = admission_callback.as_mut() {
            callback();
        }

        let result = (|| -> anyhow::Result<GenerateResult> {
            let active_matches = self.native_active_session == Some(session_id);
            let semantic_prefix_len = options
                .semantic_prefix_len
                .filter(|&len| len > 0 && len < prompt_tokens.len());
            if !active_matches {
                let native = self
                    .native_session
                    .as_mut()
                    .context("native decoder session is unavailable")?;
                native.reset()?;
                self.native_active_session = Some(session_id);
            }

            let prefix_len = {
                let state = self
                    .native_sessions
                    .get(&session_id)
                    .with_context(|| format!("session {session_id} not found"))?;
                common_prefix_len(&state.tokens, &prompt_tokens)
            };

            let supports_past_snapshots = self
                .native_session
                .as_ref()
                .is_some_and(|native| native.supports_past_snapshots());
            let mut restored_prefix_len = None;
            let mut should_store_semantic_snapshot = false;
            if !active_matches
                && supports_past_snapshots
                && let Some(boundary) = semantic_prefix_len
            {
                self.native_recurrent_prefix_stats.lookups += 1;
                if let Some((matched, snapshot)) = self
                    .prefix_cache
                    .lookup_snapshot::<NativePrefixSnapshot>(&prompt_tokens)
                    .filter(|(matched, _)| *matched >= boundary && *matched < prompt_tokens.len())
                {
                    let native = self
                        .native_session
                        .as_mut()
                        .context("native decoder session is unavailable")?;
                    match native.restore_past_snapshot(&snapshot.snapshot) {
                        Ok(()) => {
                            self.native_recurrent_prefix_stats.hits += 1;
                            self.native_recurrent_prefix_stats.restored_tokens += matched as u64;
                            restored_prefix_len = Some(matched);
                        }
                        Err(error) => {
                            tracing::debug!(
                                "skipping native recurrent prefix snapshot restore after failure: \
                             {error:#}"
                            );
                        }
                    }
                } else {
                    should_store_semantic_snapshot = true;
                }
            }

            if should_store_semantic_snapshot
                && supports_past_snapshots
                && let Some(boundary) = semantic_prefix_len
            {
                let snapshot = {
                    let native = self
                        .native_session
                        .as_mut()
                        .context("native decoder session is unavailable")?;
                    native.prefill_prefix(&prompt_tokens[..boundary])?;
                    native.snapshot_past()
                };
                match snapshot {
                    Ok(snapshot) => {
                        // Borrow the governor field directly, not through
                        // `governor()`: the closure below runs while
                        // `self.prefix_cache` is mutably borrowed, and a
                        // whole-`self` accessor borrow would conflict.
                        let governor_memory = self.governor.memory();
                        let reserve_snapshot = || {
                            onnx_runtime_memory_governor::MemoryGovernor::reserve(
                                governor_memory,
                                onnx_runtime_memory_governor::Tier::Host,
                                snapshot.bytes(),
                                Holder::RecurrentPrefixSnapshot.role(),
                                Holder::RecurrentPrefixSnapshot.id(),
                            )
                        };
                        let lease = match reserve_snapshot() {
                            Ok(lease) => Some(lease),
                            Err(first_error) if self.prefix_cache.evict_lru_snapshot() => {
                                match reserve_snapshot() {
                                    Ok(lease) => Some(lease),
                                    Err(error) => {
                                        tracing::debug!(
                                            "skipping native recurrent prefix snapshot store; cannot \
                                         reserve {} bytes after evicting an older snapshot: \
                                         {first_error}; retry: {error}",
                                            snapshot.bytes()
                                        );
                                        None
                                    }
                                }
                            }
                            Err(error) => {
                                tracing::debug!(
                                    "skipping native recurrent prefix snapshot store; cannot reserve \
                                 {} bytes: {error}",
                                    snapshot.bytes()
                                );
                                None
                            }
                        };
                        if let Some(lease) = lease {
                            self.prefix_cache.insert_snapshot(
                                &prompt_tokens[..boundary],
                                Arc::new(NativePrefixSnapshot {
                                    snapshot,
                                    _lease: lease,
                                }),
                            );
                            self.native_recurrent_prefix_stats.stores += 1;
                        }
                        restored_prefix_len = Some(boundary);
                    }
                    Err(error) => {
                        tracing::debug!(
                            "skipping native recurrent prefix snapshot store after snapshot failure: \
                         {error:#}"
                        );
                    }
                }
            }

            let mut result = {
                // Borrowed before the mutable native-session borrow below.
                let tokenizer = self
                    .tokenizer
                    .as_ref()
                    .context("this package declares no tokenizer, so it cannot decode text")?;
                let runtime = &*self.workflow;
                let native = self
                    .native_session
                    .as_mut()
                    .context("native decoder session is unavailable")?;

                let mut resume_from = if let Some(restored) = restored_prefix_len {
                    restored
                } else if active_matches {
                    prefix_len.min(native.current_len())
                } else {
                    0
                };
                if resume_from >= prompt_tokens.len() {
                    resume_from = prompt_tokens.len().saturating_sub(1);
                }

                native.generate_incremental_with_callback(
                    &prompt_tokens,
                    resume_from,
                    &options,
                    &chain,
                    tokenizer,
                    runtime,
                    callback,
                )?
            };
            result.budget_cap = budget_cap;
            let last_access = self.touch_native_session();

            let state = self
                .native_sessions
                .get_mut(&session_id)
                .with_context(|| format!("session {session_id} not found"))?;
            state.tokens.truncate(prefix_len);
            state.tokens.extend_from_slice(&prompt_tokens[prefix_len..]);
            state.tokens.extend_from_slice(&result.token_ids);
            state.last_access = last_access;
            self.evict_native_sessions(session_id);

            Ok(result)
        })();
        self.scheduler.complete(session_id);
        result
    }
}

/// Token ids carried by a workflow literal, scalar or list.
///
/// A package may declare one end token or several with the same field; reading
/// both shapes here is what keeps "declare your EOS" from having two spellings.
fn literal_token_ids(literal: &onnx_genai_metadata::LiteralValue) -> anyhow::Result<Vec<TokenId>> {
    use onnx_genai_metadata::{LiteralValue, ScalarValue};
    let scalars = match literal {
        LiteralValue::Scalar(scalar) => std::slice::from_ref(scalar),
        LiteralValue::Elements(elements) => elements.as_slice(),
    };
    scalars
        .iter()
        .map(|scalar| match scalar {
            // Dropping a malformed id would be the worst outcome: the package
            // says "these tokens end me", the runtime silently keeps a subset,
            // and generation runs past an end token the author declared. A
            // package that cannot state its stop condition must fail to load,
            // not load with a quietly smaller one.
            ScalarValue::Integer(value) => TokenId::try_from(*value).map_err(|_| {
                anyhow::anyhow!(
                    "declared end-of-generation token id {value} is not a valid token id"
                )
            }),
            other => anyhow::bail!(
                "declared end-of-generation token ids must be integers; found {other:?}"
            ),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "native-backend")]
    use crate::ProcessorChain;
    #[cfg(feature = "native-backend")]
    use std::path::PathBuf;

    #[test]
    fn ort_and_native_generation_use_scheduler_admission() {
        assert!(generate_uses_scheduler(EngineDecodeBackend::Ort));
        assert!(generate_uses_scheduler(EngineDecodeBackend::Native));
        assert!(generate_uses_scheduler(EngineDecodeBackend::Auto));
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn speculative_workspace_rows_follow_the_exact_runtime_plan() {
        let chain = ProcessorChain::new();
        let mut options = GenerateOptions {
            greedy: true,
            temperature: 0.0,
            speculative_mode: Some(SpeculativeMode::PromptLookup {
                ngram: 1,
                max_tokens: 4,
            }),
            ..GenerateOptions::default()
        };
        let plan = native_speculation_plan(&options, &chain).unwrap();
        assert_eq!(native_workspace_query_rows(1, Some(&plan), 8, None), 4);
        assert_eq!(native_workspace_query_rows(8, Some(&plan), 8, None), 8);

        options.num_speculative_tokens = Some(1024);
        let widened = native_speculation_plan(&options, &chain).unwrap();
        assert_eq!(widened.width, 1024);
        assert_eq!(native_workspace_query_rows(1, Some(&widened), 1, None), 1);
        assert_eq!(
            native_workspace_query_rows(1, Some(&widened), 8, Some(3)),
            2
        );
        let runtime_width = crate::native_speculative::verification_width(widened.width, 5, 3);
        assert_eq!(
            native_workspace_query_rows(1, Some(&widened), 5, Some(4)),
            1usize.max(runtime_width)
        );

        options.num_speculative_tokens = None;
        options.speculative_mode = Some(SpeculativeMode::Mtp(crate::config::MtpConfig {
            head_model: PathBuf::from("mtp.onnx"),
            target_hidden_output: "hidden".to_string(),
            embedding_weights: PathBuf::from("embedding.bin"),
            lm_head_weights: PathBuf::from("lm_head.bin"),
            vocab_size: 1,
            hidden_size: 1,
            kv_mode: onnx_genai_ort::MtpDraftKvMode::GrowCache,
            num_speculative_tokens: 3,
        }));
        let mtp = native_speculation_plan(&options, &chain).unwrap();
        assert_eq!(mtp.width, 4);
        assert_eq!(native_workspace_query_rows(1, Some(&mtp), 8, None), 4);

        assert_eq!(native_workspace_query_rows(3, None, 8, None), 3);
    }

    /// An engine whose package the interpreter drives, sharing one scheduler.
    ///
    /// `holds_decode_core()` is false here (no `session`, no `native_session`),
    /// so generation takes the no-decode-core branch #1723 introduced and #1900
    /// wired admission into. `budget_bytes` is the whole KV byte budget, which
    /// is what decides whether a request is admitted at all.
    ///
    /// This literal was written out three times across these tests before it
    /// was a helper; every field but the budget was identical in all three, and
    /// a fixture that is copied is a fixture that drifts.
    #[cfg(feature = "native-backend")]
    fn interpreted_engine_with_byte_budget(budget_bytes: u64) -> anyhow::Result<Engine> {
        interpreted_engine_with_byte_budget_inner(budget_bytes, false)
    }

    /// The same engine over a package that declares a session-scoped
    /// conversation, so `create_session` succeeds and a test can reach what
    /// happens *inside* the session rather than failing in front of it.
    #[cfg(feature = "native-backend")]
    fn session_capable_engine_with_byte_budget(budget_bytes: u64) -> anyhow::Result<Engine> {
        interpreted_engine_with_byte_budget_inner(budget_bytes, true)
    }

    #[cfg(feature = "native-backend")]
    fn interpreted_engine_with_byte_budget_inner(
        budget_bytes: u64,
        session_scoped: bool,
    ) -> anyhow::Result<Engine> {
        let tokenizer_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json")
            .canonicalize()?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        let scheduler_config = onnx_genai_scheduler::SchedulerConfig {
            bytes_per_token: Some(10),
            ..onnx_genai_scheduler::SchedulerConfig::default()
        };
        let governor = EngineResourceGovernor::new(
            ResourceLimits::default(),
            false,
            ModelKvConfig::known(10, 1),
            0,
        )?;
        Ok(Engine {
            workflow: if session_scoped {
                Box::new(crate::pipeline::generation::test_decoder_runtime_with_session_state()?)
            } else {
                Box::new(crate::pipeline::generation::test_decoder_runtime()?)
            },
            workflow_sessions: HashMap::new(),
            workflow_session_counter: 0,
            decode_backend: EngineDecodeBackend::Native,
            metadata: InferenceMetadata::default(),
            metadata_hints: MetadataHints::default(),
            kv_cache: PagedKvCache::new(1, 1),
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model: None,
            decode_path: ModelDecodePath::Generic,
            scheduler: Scheduler::with_byte_budget(
                scheduler_config,
                onnx_genai_scheduler::ByteBudget::new(budget_bytes),
            ),
            governor,
            sessions: HashMap::new(),
            session: None,
            native_session: None,
            weight_placement: None,
            memory_strategy_plan: MemoryStrategyPlan::unknown(0, None, "test engine fixture"),
            native_sessions: HashMap::new(),
            native_active_session: None,
            native_session_counter: 0,
            native_access_counter: 0,
            native_default_session: None,
            native_max_sessions: 8,
            native_recurrent_prefix_stats: RecurrentPrefixCacheStats::default(),
            draft: None,
            mtp: None,
            eagle3: None,
            tokenizer: Some(tokenizer),
            fim_config: None,
            num_speculative_tokens: 1,
            speculative_mode: SpeculativeMode::None,
            last_speculative_stats: SpeculativeStats::default(),
            connector: ConnectorBridge::null(),
            _environment: None,
        })
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_generate_rejects_over_kv_byte_budget_before_backend_run() -> anyhow::Result<()> {
        let mut engine = interpreted_engine_with_byte_budget(10)?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![1]));
        request.options.max_new_tokens = 1;
        request.options.stop_on_eos = false;

        let mut admitted = false;
        let mut on_admitted = || admitted = true;
        let error = engine
            .generate_with_callbacks(request, Some(&mut on_admitted), None)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("scheduler admission failed: KV byte budget"),
            "{error}"
        );
        assert!(
            !error.contains("native decoder session is unavailable"),
            "native backend was touched before scheduler admission rejected: {error}"
        );
        assert!(!admitted, "refused requests must not signal admission");
        Ok(())
    }

    /// The cold-call test above exercises `generate_interpreted`; a
    /// continuing workflow session reaches the exact same no-decode-core
    /// branch through [`Engine::generate_in_workflow_session`] instead (the
    /// path a server session-continuation route uses), and must be admitted
    /// through the identical scheduler mechanism rather than skip it because
    /// the request already names a session.
    #[cfg(feature = "native-backend")]
    #[test]
    fn native_generate_in_workflow_session_rejects_over_kv_byte_budget() -> anyhow::Result<()> {
        let mut engine = session_capable_engine_with_byte_budget(10)?;

        // No decode core: this engine's sessions are workflow sessions,
        // opened through the same public `create_session` a real caller
        // (e.g. a server session-continuation route) would use.
        assert!(!engine.holds_decode_core());
        let session_id = engine.create_session()?;

        let mut request = crate::pipeline::PipelineGenerateRequest::new(GenerateRequest::new(
            GeneratePrompt::TokenIds(vec![1]),
        ));
        request.request.options.max_new_tokens = 1;
        request.request.options.stop_on_eos = false;

        let mut admitted = false;
        let mut on_admitted = || admitted = true;
        let error = engine
            .generate_in_workflow_session(session_id, request, Some(&mut on_admitted), None)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("scheduler admission failed: KV byte budget"),
            "{error}"
        );
        assert!(
            !error.contains("references unavailable value"),
            "a continuing session must be rejected at admission, not deep inside node \
             execution once an unbound loop value goes missing: {error}"
        );
        assert!(!admitted, "refused requests must not signal admission");
        Ok(())
    }

    /// The half of #1900 that a rejection test cannot reach: the reservation an
    /// admitted request takes is handed back when it finishes.
    ///
    /// Both tests above assert a *refusal*, and a refusal is equally consistent
    /// with an engine that admits nothing and with one that admits correctly
    /// but never releases. So `scheduler.complete()` on this path was not under
    /// test: deleting that one line left the whole 618-test lib suite green.
    ///
    /// The observable chosen here is the user-visible consequence rather than
    /// the internal counter. On a budget sized for exactly one request, a leak
    /// makes the *second* request fail admission -- the first never let go.
    /// A test that asserts "refused because over budget" and a test that
    /// asserts "not refused, because the previous request released" are
    /// falsified by opposite mutations, which is the point of having both.
    ///
    /// The fixture cannot complete a generation -- its interpreted decoder
    /// wants a KV value no component produces -- so neither call returns `Ok`.
    /// That is exactly the case the release has to survive: `complete()` runs
    /// on the error path too, and a request that admits and then fails must not
    /// strand its bytes.
    #[cfg(feature = "native-backend")]
    #[test]
    fn an_admitted_interpreted_request_releases_its_reservation_for_the_next_one()
    -> anyhow::Result<()> {
        // One prompt token plus one new token at 10 bytes each, and nothing to
        // spare: enough for a single request at a time, never for two at once.
        let mut engine = interpreted_engine_with_byte_budget(20)?;
        assert!(!engine.holds_decode_core());

        let mut refusals = Vec::new();
        let mut admissions = 0usize;
        for _ in 0..2 {
            let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![1]));
            request.options.max_new_tokens = 1;
            request.options.stop_on_eos = false;
            let mut on_admitted = || admissions += 1;
            if let Err(error) =
                engine.generate_with_callbacks(request, Some(&mut on_admitted), None)
            {
                let error = error.to_string();
                if error.contains("scheduler admission failed") {
                    refusals.push(error);
                }
            }
        }

        assert!(
            refusals.is_empty(),
            "a request must not be refused for bytes an earlier finished request still holds: {}",
            refusals.join(" | ")
        );
        assert_eq!(
            admissions, 2,
            "the admission callback fires once per request the scheduler accepted"
        );
        assert_eq!(
            engine.scheduler.running_count(),
            0,
            "no sequence may still be running once every request has finished"
        );
        Ok(())
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_backend_batching_capability_reports_single_sequence() -> anyhow::Result<()> {
        let tokenizer_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json")
            .canonicalize()?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        let governor = EngineResourceGovernor::new(
            ResourceLimits::default(),
            false,
            ModelKvConfig::known(10, 1),
            0,
        )?;
        // A native engine reports batch=1 even when the model's KV I/O shape
        // (here StaticCache, which batches under ORT) would otherwise support
        // batching: the cap is a property of the native decode path, not the
        // model. This is the honesty guarantee -- capability is not read off the
        // decode_path alone.
        let mut engine = Engine {
            workflow: Box::new(crate::pipeline::generation::test_decoder_runtime()?),
            workflow_sessions: HashMap::new(),
            workflow_session_counter: 0,
            decode_backend: EngineDecodeBackend::Native,
            metadata: InferenceMetadata::default(),
            metadata_hints: MetadataHints::default(),
            kv_cache: PagedKvCache::new(1, 1),
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model: None,
            decode_path: ModelDecodePath::StaticCache { max_len: 16 },
            scheduler: Scheduler::with_byte_budget(
                onnx_genai_scheduler::SchedulerConfig::default(),
                onnx_genai_scheduler::ByteBudget::new(10),
            ),
            governor,
            sessions: HashMap::new(),
            session: None,
            native_session: None,
            weight_placement: None,
            memory_strategy_plan: MemoryStrategyPlan::unknown(0, None, "test engine fixture"),
            native_sessions: HashMap::new(),
            native_active_session: None,
            native_session_counter: 0,
            native_access_counter: 0,
            native_default_session: None,
            native_max_sessions: 8,
            native_recurrent_prefix_stats: RecurrentPrefixCacheStats::default(),
            draft: None,
            mtp: None,
            eagle3: None,
            tokenizer: Some(tokenizer),
            fim_config: None,
            num_speculative_tokens: 1,
            speculative_mode: SpeculativeMode::None,
            last_speculative_stats: SpeculativeStats::default(),
            connector: ConnectorBridge::null(),
            _environment: None,
        };

        let capability = engine.batching_capability();
        assert_eq!(
            capability.max_concurrent_sequences(),
            Some(1),
            "native decode path pins batch to 1"
        );
        assert!(!capability.supports_batching());
        assert!(!capability.allows(2));
        assert_eq!(capability.effective_max_batch(4), 1);
        assert!(
            capability.reason().contains("native"),
            "reason must name the backend: {}",
            capability.reason()
        );
        // Tie the reported capability to reality: the native engine cannot
        // actually build a batched manager for more than one sequence.
        assert!(
            engine.continuous_batch_manager(2).is_err(),
            "native backend must not build a >1 continuous batch manager"
        );
        Ok(())
    }
}

/// Apply a model's end tokens to a request.
///
/// The one implementation, shared by the single-row path, the continuous batch
/// manager and static batching. Three copies of this is how a model stops
/// correctly on one route and runs past its end on another — which is
/// unobservable from the API and shows up as a serving-only bug.
pub(crate) fn apply_eos_policy(options: &mut GenerateOptions, ids: &[TokenId]) {
    if !options.stop_on_eos {
        return;
    }
    if options.eos_token_id.is_none()
        && let Some(&id) = ids.first()
    {
        options.eos_token_id = Some(id);
    }
    for &id in ids {
        if !options.eos_token_ids.contains(&id) {
            options.eos_token_ids.push(id);
        }
    }
}

/// The legacy direct decode path cannot be selected.
///
/// There is no flag, mode, or constructor that reaches generation without a
/// declared workflow. That used to be a runtime guard on every decode entry
/// point; it is now a property of the type — [`Engine`] holds one
/// non-optional interpreter, and the loader refuses a package that declares no
/// workflow before an engine exists at all.
///
/// These cases pin the two halves that remain checkable: a package with no
/// workflow does not load, and every constructor presents the workflow the
/// package shipped.
#[cfg(test)]
mod canonical_refusal_tests {
    use crate::{Engine, EngineConfig};
    use std::path::PathBuf;

    fn decoder_package() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm")
    }

    /// A package declaring no workflow is refused at load, naming the fix.
    ///
    /// This is what makes the missing-workflow state unreachable rather than
    /// merely guarded: nothing downstream has to check, because no engine with
    /// that state is ever constructed.
    #[test]
    fn a_package_without_a_workflow_does_not_load() -> anyhow::Result<()> {
        let staging = std::env::current_dir()?.join("target/no-workflow-package");
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging)?;
        for name in ["model.onnx.textproto", "tokenizer.json"] {
            std::fs::copy(decoder_package().join(name), staging.join(name))?;
        }
        let Err(error) = Engine::from_dir(&staging, EngineConfig::default()) else {
            panic!("a package declaring no pipeline.workflow must not load");
        };
        let message = format!("{error:#}");
        assert!(
            message.contains("declares no pipeline.workflow"),
            "the refusal must name what is missing: {message}"
        );
        assert!(
            message.contains("migrate_model_io"),
            "the refusal must name the offline conversion: {message}"
        );
        std::fs::remove_dir_all(&staging)?;
        Ok(())
    }

    /// No public constructor skips the workflow.
    #[test]
    fn every_constructor_presents_the_declared_workflow() -> anyhow::Result<()> {
        for engine in [
            Engine::from_dir(&decoder_package(), EngineConfig::default())?,
            Engine::from_dir_with_session_options(
                &decoder_package(),
                EngineConfig::default(),
                onnx_genai_ort::SessionOptions::default(),
            )?,
        ] {
            let workflow = engine
                .package_workflow()
                .expect("a decoder package always presents its declared workflow");
            assert_eq!(
                onnx_genai_metadata::sole_decoder_component(workflow),
                Some(onnx_genai_metadata::decoder_workflow::DECODER_COMPONENT)
            );
        }
        Ok(())
    }

    /// A package whose declared end tokens are malformed fails loudly.
    ///
    /// Filtering them out would mean the package says "these tokens end me" and
    /// the runtime silently keeps a subset — generation then runs past an end
    /// token its author declared, which is invisible from the API.
    #[test]
    fn a_malformed_end_token_declaration_is_an_error() {
        use onnx_genai_metadata::{LiteralValue, ScalarValue};
        let error = super::literal_token_ids(&LiteralValue::Elements(vec![
            ScalarValue::Integer(2),
            ScalarValue::Integer(-1),
        ]))
        .expect_err("a negative token id is not a token id");
        assert!(
            format!("{error:#}").contains("not a valid token id"),
            "{error:#}"
        );

        let error = super::literal_token_ids(&LiteralValue::Scalar(ScalarValue::String(
            "eos".to_string(),
        )))
        .expect_err("a string is not a token id");
        assert!(
            format!("{error:#}").contains("must be integers"),
            "{error:#}"
        );
    }
}
