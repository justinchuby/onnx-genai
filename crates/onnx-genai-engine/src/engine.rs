//! Main generation engine.

use crate::FimConfig;
use crate::decode::{
    DecodeState, ModelDecodePath, detect_model_decode_path, next_session_token_logits,
};
use crate::decode_loop::{
    DecodeLoopBackend, DecodeLoopState, exceeded_context_limit, run_decode_loop, step_decode_loop,
};
use crate::kv_bridge::{
    KvModelInfo, attach_pages_to_sequence, common_prefix_len, infer_kv_model_info,
    load_materialized_past, sequence_pages_for_len,
};
use crate::logits::{StopSequence, TokenId};
use crate::processors::{
    build_processor_chain, ensure_constrained_finish, load_fim_config_from_model_dir,
    push_unique_stop_sequence,
};
use crate::sampling::SamplingRng;
use crate::session::{ActiveGenerate, DraftModel, DraftSession, EngineSession};
use anyhow::Context;
use onnx_genai_kv::{KvCacheOps, PagedKvCache, PrefixCache};
use onnx_genai_metadata::InferenceMetadata;
use onnx_genai_ort::{
    DataType, Eagle3DecodeSession, Environment, ModelDirectory, MtpDecodeSession, Session,
    SessionOptions, Tokenizer,
};
use onnx_genai_scheduler::{Priority, Scheduler};
use std::collections::HashMap;
use std::path::Path;

pub use crate::config::{
    Eagle3Config, EngineConfig, FinishReason, GenerateConstraint, GenerateOptions, GeneratePrompt,
    GenerateRequest, GenerateResult, GenerateToken, GenerateTokenCallback, MtpConfig,
    PrioritizedGenerateRequest, PrioritizedGenerateResult, ScheduledGenerateArrival, SessionId,
    SpeculativeMode, TokenLogprob,
};
use crate::speculative::{LinearEmbedder, LinearLmHead, SpeculativeStats};

/// The generation engine.
pub struct Engine {
    /// Model inference metadata.
    pub(crate) metadata: InferenceMetadata,
    /// KV cache manager.
    pub(crate) kv_cache: PagedKvCache,
    /// Shared-prefix cache for reusing paged KV across sessions.
    pub(crate) prefix_cache: PrefixCache,
    /// Token-only prefix index used by ORT-owned decode sessions until page import/export lands.
    pub(crate) token_prefix_cache: Vec<Vec<TokenId>>,
    /// KV tensor layout inferred from model present/past TensorInfo.
    pub(crate) kv_model: Option<KvModelInfo>,
    /// ORT decode path selected by model I/O introspection.
    pub(crate) decode_path: ModelDecodePath,
    /// Batch scheduler.
    pub(crate) scheduler: Scheduler,
    /// Persistent multi-turn session state, keyed by session id.
    pub(crate) sessions: HashMap<SessionId, EngineSession>,
    /// ORT environment kept alive for the session.
    pub(crate) _environment: Environment,
    /// ORT session for decoder execution.
    pub(crate) session: Box<Session>,
    /// Optional draft model used by the speculative decoding path.
    pub(crate) draft: Option<DraftModel>,
    /// Optional MTP head and target-side projections.
    pub(crate) mtp: Option<MtpModel>,
    /// Optional EAGLE-3 head and target-side embedding.
    pub(crate) eagle3: Option<Eagle3Model>,
    /// Tokenizer loaded from the model directory.
    pub(crate) tokenizer: Tokenizer,
    /// Auto-detected fill-in-the-middle token configuration.
    pub(crate) fim_config: Option<FimConfig>,
    /// Default speculative draft width K.
    pub(crate) num_speculative_tokens: usize,
    /// Default speculative candidate source.
    pub(crate) speculative_mode: SpeculativeMode,
    /// Diagnostics from the most recent generation call.
    pub(crate) last_speculative_stats: SpeculativeStats,
}

pub(crate) struct MtpModel {
    pub(crate) config: MtpConfig,
    pub(crate) session: Box<Session>,
    pub(crate) embedder: LinearEmbedder,
    pub(crate) lm_head: LinearLmHead,
    pub(crate) hidden_output: String,
    pub(crate) kv_mode: onnx_genai_ort::MtpDraftKvMode,
    pub(crate) num_speculative_tokens: usize,
}

pub(crate) struct Eagle3Model {
    pub(crate) config: Eagle3Config,
    pub(crate) session: Box<Session>,
    pub(crate) embedder: LinearEmbedder,
    pub(crate) hidden_outputs: Vec<String>,
    pub(crate) kv_mode: onnx_genai_ort::Eagle3DraftKvMode,
    pub(crate) num_speculative_tokens: usize,
}

impl Engine {
    /// Load a model from a directory.
    pub fn from_dir(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        Self::from_dir_with_session_options(model_dir, config, SessionOptions::default())
    }

    /// Load a model from a directory with explicit ORT session options.
    pub fn from_dir_with_session_options(
        model_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
    ) -> anyhow::Result<Self> {
        let model_directory = ModelDirectory::load(model_dir)
            .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {}", e))?;

        // Initialize scheduler
        let scheduler = Scheduler::new(config.scheduler);

        let environment = Environment::new("onnx-genai-engine")
            .map_err(|e| anyhow::anyhow!("Failed to create ORT environment: {}", e))?;
        let session = Session::new(
            &environment,
            &model_directory.model_path,
            session_options.clone(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to load ORT session: {}", e))?;

        // Resolve inference metadata. Our own `inference_metadata.yaml` is the
        // canonical source of truth. When a model ships without it (e.g. the
        // onnxruntime-genai / Foundry Local models, which carry only a
        // `genai_config.json`), fall back to converting that config into native
        // metadata so share-buffer-capable GQA models still get the O(1)/token
        // decode path instead of the growing rebind path.
        let metadata = if let Some(metadata_path) = &model_directory.metadata_path {
            onnx_genai_metadata::load_metadata(metadata_path)
                .map_err(|e| anyhow::anyhow!("Failed to load metadata: {}", e))?
        } else if let Some(compat) = genai_config_compat_metadata(&model_directory.root, &session)?
        {
            tracing::info!(
                "No inference_metadata.yaml found; derived inference metadata from genai_config.json (onnxruntime-genai compatibility)"
            );
            compat
        } else {
            tracing::warn!("No inference metadata found, using defaults");
            InferenceMetadata {
                required_capabilities: vec![],
                model: None,
                kv_cache: None,
                quantization: None,
                pipeline: None,
                strategy: None,
                speculative: None,
                structured_output: None,
                hardware_requirements: None,
            }
        };

        // Validate capabilities
        let runtime_caps = onnx_genai_metadata::RuntimeCapabilities::default();
        if let Err(unsupported) = onnx_genai_metadata::validate(&metadata, &runtime_caps) {
            anyhow::bail!("Unsupported capabilities: {:?}", unsupported);
        }

        let metadata_max_context = metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length);
        // Our own inference metadata (inference_metadata.yaml), not
        // onnxruntime-genai's genai_config.json, drives the runtime-owned
        // share-buffer KV path for GQA models.
        let shared_kv_max_len = crate::decode::shared_kv_buffer_len_from_metadata(&metadata);
        let sliding_window = crate::decode::sliding_window_from_metadata(&metadata)?;
        let decode_path = detect_model_decode_path(
            &session,
            metadata_max_context,
            shared_kv_max_len,
            sliding_window,
        )?;
        let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let fim_config = load_fim_config_from_model_dir(&model_directory.root)?;
        let kv_model = infer_kv_model_info(&session, config.page_size)?;
        let draft = if let Some(draft_model_path) = &config.draft_model {
            let draft_directory = ModelDirectory::load(draft_model_path)
                .map_err(|e| anyhow::anyhow!("Failed to resolve draft model directory: {}", e))?;
            let draft_session = Session::new(
                &environment,
                &draft_directory.model_path,
                session_options.clone(),
            )
            .map_err(|e| anyhow::anyhow!("Failed to load draft ORT session: {}", e))?;
            let draft_decode_path =
                detect_model_decode_path(&draft_session, metadata_max_context, None, None)?;
            let draft_kv_model = infer_kv_model_info(&draft_session, config.page_size)?;
            let draft_kv_cache = if let Some(kv_model) = &draft_kv_model {
                PagedKvCache::new_with_tensor_config(kv_model.tensor_config, config.num_gpu_pages)
            } else {
                PagedKvCache::new(config.page_size, config.num_gpu_pages)
            };
            Some(DraftModel {
                session: Box::new(draft_session),
                decode_path: draft_decode_path,
                kv_model: draft_kv_model,
                kv_cache: draft_kv_cache,
            })
        } else {
            None
        };
        let kv_cache = if let Some(kv_model) = &kv_model {
            // The paged tensor layout is derived from present-KV outputs: each
            // layer has key/value tensors shaped like [batch, kv_heads, seq, head_dim].
            PagedKvCache::new_with_tensor_config(kv_model.tensor_config, config.num_gpu_pages)
        } else {
            PagedKvCache::new(config.page_size, config.num_gpu_pages)
        };

        let speculative_mode = match config.speculative_mode {
            SpeculativeMode::None if draft.is_some() => SpeculativeMode::DraftModel,
            mode => mode,
        };
        if let SpeculativeMode::PromptLookup { ngram, max_tokens } = &speculative_mode
            && (*ngram == 0 || *max_tokens == 0)
        {
            anyhow::bail!("prompt-lookup ngram and max_tokens must be greater than zero");
        }
        let mtp = if let SpeculativeMode::Mtp(mtp_config) = &speculative_mode {
            crate::config::validate_mtp_config(mtp_config)?;
            let hidden_output = session
                .outputs()
                .iter()
                .find(|output| output.name == mtp_config.target_hidden_output)
                .with_context(|| {
                    format!(
                        "MTP target model must expose hidden-state output '{}'",
                        mtp_config.target_hidden_output
                    )
                })?;
            if hidden_output.dtype != DataType::Float32 {
                anyhow::bail!(
                    "MTP target hidden-state output '{}' must be Float32, got {:?}",
                    hidden_output.name,
                    hidden_output.dtype
                );
            }
            if hidden_output.shape.last().copied().filter(|dim| *dim > 0)
                != Some(mtp_config.hidden_size as i64)
            {
                anyhow::bail!(
                    "MTP target hidden-state output '{}' shape {:?} does not end in configured hidden size {}",
                    hidden_output.name,
                    hidden_output.shape,
                    mtp_config.hidden_size
                );
            }
            let head_session = Session::new(
                &environment,
                &mtp_config.head_model,
                session_options.clone(),
            )
            .map_err(|error| anyhow::anyhow!("Failed to load MTP head: {error}"))?;
            let head_signature = MtpDecodeSession::detect(&head_session)
                .map_err(|error| anyhow::anyhow!("Failed to inspect MTP head: {error}"))?
                .context("configured MTP head model does not expose MTP head I/O")?;
            if head_signature.hidden_size != mtp_config.hidden_size {
                anyhow::bail!(
                    "MTP head hidden size {} does not match configured target hidden size {}",
                    head_signature.hidden_size,
                    mtp_config.hidden_size
                );
            }
            let embedding = read_f32_weights(&mtp_config.embedding_weights)?;
            let lm_head = read_f32_weights(&mtp_config.lm_head_weights)?;
            Some(MtpModel {
                config: mtp_config.clone(),
                session: Box::new(head_session),
                embedder: LinearEmbedder::new(
                    embedding,
                    mtp_config.vocab_size,
                    mtp_config.hidden_size,
                )
                .map_err(|error| anyhow::anyhow!("Invalid MTP embedding weights: {error}"))?,
                lm_head: LinearLmHead::new(lm_head, mtp_config.hidden_size, mtp_config.vocab_size)
                    .map_err(|error| anyhow::anyhow!("Invalid MTP LM-head weights: {error}"))?,
                hidden_output: mtp_config.target_hidden_output.clone(),
                kv_mode: mtp_config.kv_mode,
                num_speculative_tokens: mtp_config.num_speculative_tokens,
            })
        } else {
            None
        };
        let eagle3 = if let SpeculativeMode::Eagle3(eagle_config) = &speculative_mode {
            crate::config::validate_eagle3_config(eagle_config)?;
            for output_name in &eagle_config.target_hidden_outputs {
                let hidden_output = session
                    .outputs()
                    .iter()
                    .find(|output| output.name == *output_name)
                    .with_context(|| {
                        format!(
                            "EAGLE-3 target model must expose hidden-state output '{output_name}'"
                        )
                    })?;
                if hidden_output.dtype != DataType::Float32 {
                    anyhow::bail!(
                        "EAGLE-3 target hidden-state output '{}' must be Float32, got {:?}",
                        hidden_output.name,
                        hidden_output.dtype
                    );
                }
                if hidden_output.shape.last().copied().filter(|dim| *dim > 0)
                    != Some(eagle_config.hidden_size as i64)
                {
                    anyhow::bail!(
                        "EAGLE-3 target hidden-state output '{}' shape {:?} does not end in configured hidden size {}",
                        hidden_output.name,
                        hidden_output.shape,
                        eagle_config.hidden_size
                    );
                }
            }
            let head_session = Session::new(
                &environment,
                &eagle_config.head_model,
                session_options.clone(),
            )
            .map_err(|error| anyhow::anyhow!("Failed to load EAGLE-3 head: {error}"))?;
            let head_signature = Eagle3DecodeSession::detect(&head_session)
                .map_err(|error| anyhow::anyhow!("Failed to inspect EAGLE-3 head: {error}"))?
                .context("configured EAGLE-3 head model does not expose EAGLE-3 head I/O")?;
            if head_signature.hidden_size != eagle_config.hidden_size {
                anyhow::bail!(
                    "EAGLE-3 head hidden size {} does not match configured target hidden size {}",
                    head_signature.hidden_size,
                    eagle_config.hidden_size
                );
            }
            let expected_fused =
                eagle_config.hidden_size * eagle_config.target_hidden_outputs.len();
            if head_signature.fused_hidden_size != expected_fused {
                anyhow::bail!(
                    "EAGLE-3 head fused hidden size {} does not match three target layers totaling {}",
                    head_signature.fused_hidden_size,
                    expected_fused
                );
            }
            if head_signature.draft_vocab_size > eagle_config.vocab_size {
                anyhow::bail!(
                    "EAGLE-3 draft vocabulary {} exceeds target embedding vocabulary {}",
                    head_signature.draft_vocab_size,
                    eagle_config.vocab_size
                );
            }
            let embedding = read_f32_weights(&eagle_config.embedding_weights)?;
            Some(Eagle3Model {
                config: eagle_config.clone(),
                session: Box::new(head_session),
                embedder: LinearEmbedder::new(
                    embedding,
                    eagle_config.vocab_size,
                    eagle_config.hidden_size,
                )
                .map_err(|error| anyhow::anyhow!("Invalid EAGLE-3 embedding weights: {error}"))?,
                hidden_outputs: eagle_config.target_hidden_outputs.clone(),
                kv_mode: eagle_config.kv_mode,
                num_speculative_tokens: eagle_config.num_speculative_tokens,
            })
        } else {
            None
        };

        Ok(Self {
            metadata,
            kv_cache,
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model,
            decode_path,
            scheduler,
            sessions: HashMap::new(),
            _environment: environment,
            session: Box::new(session),
            draft,
            mtp,
            eagle3,
            tokenizer,
            fim_config,
            num_speculative_tokens: config.num_speculative_tokens.max(1),
            speculative_mode,
            last_speculative_stats: SpeculativeStats::default(),
        })
    }

    /// Generate text for a request.
    ///
    pub fn generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult> {
        self.generate_with_callback(request, None)
    }

    /// Speculative verification diagnostics from the most recent generation.
    pub fn last_speculative_stats(&self) -> SpeculativeStats {
        self.last_speculative_stats
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
        self.generate_in_session_with_priority_and_callback(session_id, request, priority, None)
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
            callback,
        )
    }

    fn generate_in_session_with_priority_and_callback(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        priority: Priority,
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

        let request_id = self.scheduler.enqueue_generate_request(
            session_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            priority,
        );
        let scheduled = self
            .scheduler
            .drive_next_fcfs()
            .context("scheduler did not admit the session generate request")?;
        if scheduled.request_id != request_id || scheduled.seq_id != session_id {
            anyhow::bail!(
                "scheduler admitted request {} for session {}, expected request {} for session {}",
                scheduled.request_id,
                scheduled.seq_id,
                request_id,
                session_id
            );
        }

        let max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(&self.tokenizer))?;

        let mut state = self
            .sessions
            .remove(&session_id)
            .with_context(|| format!("session {session_id} not found"))?;
        let prefix_cache_hit_len =
            self.prepare_session_prefix(session_id, &mut state, &prompt_tokens)?;
        let mut loop_state =
            DecodeLoopState::new(prefix_cache_hit_len, options.seed, options.top_logprobs);

        let result = (|| -> anyhow::Result<GenerateResult> {
            if self.should_use_speculative(&options) {
                return self.generate_speculative_loop(
                    session_id,
                    &mut state,
                    &options,
                    &chain,
                    max_context,
                    prefix_cache_hit_len,
                    &mut loop_state.generated_tokens,
                    &mut loop_state.generated_text,
                    &mut loop_state.logprobs,
                    &mut loop_state.rng,
                    callback.as_deref_mut(),
                );
            }

            let mut backend = SessionDecodeLoopBackend {
                session: &self.session,
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
        })();
        if result.is_ok() && !exceeded_context_limit(state.tokens.len(), max_context) {
            self.ensure_session_kv_current(session_id, &mut state)?;
            self.insert_cached_prefixes(session_id, &state, prompt_tokens.len())?;
        }
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

    /// Create a new generation session.
    pub fn create_session(&mut self) -> anyhow::Result<SessionId> {
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
        };
        self.sessions.insert(id, state);
        Ok(id)
    }

    /// Reset a persistent session, freeing its current state while keeping the id usable.
    pub fn reset_session(&mut self, session_id: SessionId) -> anyhow::Result<()> {
        if !self.sessions.contains_key(&session_id) {
            anyhow::bail!("session {session_id} not found");
        }
        self.scheduler.complete(session_id);
        self.kv_cache
            .remove(session_id)
            .map_err(|e| anyhow::anyhow!("Failed to reset KV sequence {session_id}: {}", e))?;
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
                .map_err(|e| anyhow::anyhow!("Failed to reset draft KV sequence: {}", e))?;
            draft.seq = draft_model.kv_cache.create_sequence();
            draft.tokens.clear();
            draft.kv_token_count = 0;
            draft.decode_state =
                DecodeState::new_for_path(&draft_model.session, &draft_model.decode_path)?;
        }
        Ok(())
    }

    fn new_target_decode_state(&self) -> anyhow::Result<DecodeState> {
        if matches!(
            &self.speculative_mode,
            SpeculativeMode::Mtp(_) | SpeculativeMode::Eagle3(_)
        ) {
            DecodeState::new(&self.session)
        } else {
            DecodeState::new_for_path(&self.session, &self.decode_path)
        }
    }

    /// Close a persistent session and free its associated state.
    pub fn close_session(&mut self, session_id: SessionId) -> anyhow::Result<()> {
        self.scheduler.complete(session_id);
        let state = self
            .sessions
            .remove(&session_id)
            .with_context(|| format!("session {session_id} not found"))?;
        self.kv_cache
            .remove(session_id)
            .map_err(|e| anyhow::anyhow!("Failed to remove KV sequence {session_id}: {}", e))?;
        if let (Some(draft_model), Some(draft)) = (&mut self.draft, state.draft) {
            draft_model
                .kv_cache
                .remove(draft.seq)
                .map_err(|e| anyhow::anyhow!("Failed to remove draft KV sequence: {}", e))?;
        }
        Ok(())
    }

    /// Number of logical tokens retained in a persistent session.
    pub fn session_token_count(&self, session_id: SessionId) -> anyhow::Result<usize> {
        self.sessions
            .get(&session_id)
            .map(|state| state.tokens.len())
            .with_context(|| format!("session {session_id} not found"))
    }

    /// Get the loaded metadata.
    pub fn metadata(&self) -> &InferenceMetadata {
        &self.metadata
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

    fn tokenize_prompt(&self, prompt: &GeneratePrompt) -> anyhow::Result<Vec<TokenId>> {
        match prompt {
            GeneratePrompt::TokenIds(tokens) => Ok(tokens.clone()),
            GeneratePrompt::Text(text) => self
                .tokenizer
                .encode(text)
                .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e)),
        }
    }

    fn prepare_session_prefix(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        prompt_tokens: &[TokenId],
    ) -> anyhow::Result<usize> {
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
                        .map_err(|e| anyhow::anyhow!("Failed to materialize prefix KV: {}", e))?;
                    load_materialized_past(
                        &self.session,
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
        Ok(same_session_hit_len.max(cross_session_hit_len))
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

        self.scheduler.enqueue_generate_request(
            request.session_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            request.priority,
        );
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
        };
        let step_result = {
            let mut backend = SessionDecodeLoopBackend {
                session: &self.session,
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
        if step_result.is_some() {
            return Ok(step_result);
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
                .map(Some);
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

    fn ensure_session_kv_current(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
    ) -> anyhow::Result<()> {
        while state.decode_state.use_kv && state.kv_token_count < state.tokens.len() {
            let _ = next_session_token_logits(
                &self.session,
                self.kv_model.as_ref(),
                &mut self.kv_cache,
                session_id,
                state,
            )?;
        }
        Ok(())
    }

    fn insert_cached_prefixes(
        &mut self,
        session_id: SessionId,
        state: &EngineSession,
        prompt_len: usize,
    ) -> anyhow::Result<()> {
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
                .map_err(|e| anyhow::anyhow!("Failed to detokenize generated tokens: {}", e))?,
            token_ids: generated_tokens.to_vec(),
            finish_reason,
            prefix_cache_hit_len,
            logprobs: logprobs.map(<[crate::config::TokenLogprob]>::to_vec),
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

    fn processor_prompt_tokens(&self) -> Vec<TokenId> {
        self.state.tokens.clone()
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
}

/// Best-effort native metadata derived from an onnxruntime-genai
/// `genai_config.json` in `model_dir`, used only when no
/// `inference_metadata.yaml` is present. Returns `Ok(None)` when there is no
/// `genai_config.json`. The KV cache native dtype is read from the loaded
/// session's KV inputs, since it is not present in `genai_config.json`.
fn genai_config_compat_metadata(
    model_dir: &Path,
    session: &Session,
) -> anyhow::Result<Option<InferenceMetadata>> {
    let kv_native_dtype = session
        .inputs()
        .iter()
        .find(|info| crate::decode::is_kv_input(&info.name))
        .and_then(|info| match info.dtype {
            DataType::Float16 => Some("float16"),
            DataType::Float32 => Some("float32"),
            _ => None,
        });
    onnx_genai_genai_config::inference_metadata_from_dir(model_dir, kv_native_dtype)
        .map_err(|e| anyhow::anyhow!("Failed to convert genai_config.json: {}", e))
}

fn read_f32_weights(path: &Path) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read f32 weights from '{}'", path.display()))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        anyhow::bail!(
            "f32 weight file '{}' has byte length {}, which is not divisible by 4",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_loop::logprob_for_token;
    use crate::logits::ProcessorContext;
    use crate::processors::{
        finish_reason_after_token, select_next_token, select_next_token_with_sampler,
    };
    use crate::sampling::Sampler;

    #[test]
    fn token_logprobs_use_log_softmax_and_sorted_top_tokens() {
        let logits = [1.0, f32::NEG_INFINITY, 3.0, 2.0];
        let result = logprob_for_token(&logits, 3, 2);
        let logsumexp = 3.0 + ((1.0_f32 - 3.0).exp() + 1.0 + (2.0_f32 - 3.0).exp()).ln();

        assert_eq!(result.token_id, 3);
        assert_eq!(result.logprob, 2.0 - logsumexp);
        assert!(result.logprob <= 0.0);
        assert!(result.top.windows(2).all(|pair| pair[0].1 >= pair[1].1));
        assert!(result.top.iter().any(|(token_id, _)| *token_id == 3));
        assert!(result.top.iter().all(|(token_id, _)| *token_id != 1));
    }

    #[test]
    fn processor_chain_uses_documented_order() {
        let options = GenerateOptions {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            stop_sequences: vec![StopSequence::Tokens(vec![42])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        assert_eq!(
            chain.names(),
            vec![
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "stop_sequence",
                "temperature",
                "top_k",
                "top_p",
                "min_p"
            ]
        );
    }

    #[test]
    fn processor_chain_includes_json_constraint_before_sampling_filters() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json")
            .canonicalize()?;
        let tokenizer = Tokenizer::from_file(&fixture)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let options = GenerateOptions {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            constraint: Some(GenerateConstraint::Json),
            ..Default::default()
        };

        let chain = build_processor_chain(&options, Some(&tokenizer))?;

        assert_eq!(
            chain.names(),
            vec![
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "json_constraint",
                "temperature",
                "top_k",
                "top_p",
                "min_p"
            ]
        );
        Ok(())
    }

    #[test]
    fn greedy_selection_uses_argmax_after_processors() {
        let options = GenerateOptions {
            greedy: true,
            top_k: 2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 2.0, 4.0, 3.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.0),
            2
        );
    }

    #[test]
    fn sampled_selection_can_pick_non_argmax() {
        let options = GenerateOptions {
            greedy: false,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 0.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.75),
            1
        );
    }

    struct LastTokenSampler;

    impl Sampler for LastTokenSampler {
        fn sample(&mut self, logits: &[f32], _context: &ProcessorContext) -> TokenId {
            logits.len().saturating_sub(1) as TokenId
        }

        fn name(&self) -> &str {
            "last_token"
        }
    }

    #[test]
    fn custom_sampler_can_select_after_default_processors() {
        let options = GenerateOptions {
            top_k: 2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 2.0, 4.0, 3.0];
        let mut sampler = LastTokenSampler;

        assert_eq!(
            select_next_token_with_sampler(&mut logits, &context, &chain, &mut sampler),
            3
        );
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn default_processor_chain_is_empty_for_unchanged_defaults() {
        let options = GenerateOptions::default();
        let chain = build_processor_chain(&options, None).unwrap();
        assert!(chain.names().is_empty());
    }

    #[test]
    fn finish_reason_detects_eos_before_stop_sequence() {
        let options = GenerateOptions {
            eos_token_id: Some(7),
            stop_sequences: vec![StopSequence::Tokens(vec![7])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![7],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(7, &options, &chain, &context),
            Some(FinishReason::EosToken)
        );
    }

    #[test]
    fn finish_reason_detects_stop_sequence() {
        let options = GenerateOptions {
            stop_sequences: vec![StopSequence::Tokens(vec![2, 3])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![1, 2, 3],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(3, &options, &chain, &context),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }

    #[test]
    fn json_constraint_defers_stop_until_value_is_complete() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json),
            stop_sequences: vec![StopSequence::Text("}".to_string())],
            ..Default::default()
        };
        let chain_options = GenerateOptions {
            stop_sequences: options.stop_sequences.clone(),
            ..Default::default()
        };
        let chain = build_processor_chain(&chain_options, None).unwrap();
        let incomplete = ProcessorContext {
            generated_text: "{\"value\":".to_string(),
            ..Default::default()
        };
        let complete = ProcessorContext {
            generated_text: "{\"value\":1}".to_string(),
            ..Default::default()
        };

        assert_eq!(
            finish_reason_after_token(1, &options, &chain, &incomplete),
            None
        );
        assert_eq!(
            finish_reason_after_token(1, &options, &chain, &complete),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }

    #[test]
    fn incomplete_json_constraint_rejects_length_finishes() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json),
            ..Default::default()
        };
        for reason in [FinishReason::MaxTokens, FinishReason::Length] {
            let error = ensure_constrained_finish(&options, "{\"value\":", reason).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("stopped before a complete JSON value")
            );
        }
        ensure_constrained_finish(&options, "{\"value\":1}", FinishReason::MaxTokens).unwrap();
        ensure_constrained_finish(&options, "", FinishReason::EosToken).unwrap();
    }

    #[test]
    fn common_prefix_len_stops_before_rejected_draft_token() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(common_prefix_len(&[7], &[8]), 0);
    }

    #[test]
    fn tiny_fixture_generates_requested_tokens_end_to_end() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids.len(), 3);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn tiny_fixture_returns_opt_in_per_token_logprobs() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir_with_session_options(
            &fixture,
            EngineConfig::default(),
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.top_logprobs = Some(3);

        let result = engine.generate(request)?;
        let logprobs = result.logprobs.as_ref().expect("logprobs requested");

        assert_eq!(logprobs.len(), result.token_ids.len());
        for (token_id, token_logprob) in result.token_ids.iter().zip(logprobs) {
            assert_eq!(*token_id, token_logprob.token_id);
            assert!(token_logprob.logprob <= 0.0);
            assert!(
                token_logprob
                    .top
                    .windows(2)
                    .all(|pair| pair[0].1 >= pair[1].1)
            );
            assert!(
                token_logprob
                    .top
                    .iter()
                    .any(|(top_token_id, _)| top_token_id == token_id)
            );
        }

        let mut disabled = GenerateRequest::new("hello");
        disabled.options.max_new_tokens = 1;
        disabled.options.temperature = 0.0;
        disabled.options.stop_on_eos = false;
        assert!(engine.generate(disabled)?.logprobs.is_none());
        Ok(())
    }

    #[test]
    fn tiny_fixture_uses_past_present_decode_session_with_stable_greedy_output()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(matches!(
            engine.decode_path,
            ModelDecodePath::PastPresent {
                shared_buffer: false,
                ..
            }
        ));
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids, vec![22, 22, 20]);
        Ok(())
    }

    #[test]
    fn scatter_fixture_uses_static_cache_decode_session_with_stable_greedy_output()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm-scatter")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(matches!(
            engine.decode_path,
            ModelDecodePath::StaticCache { max_len } if max_len > 0
        ));
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids, vec![23, 15, 28]);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        Ok(())
    }

    #[test]
    fn tiny_fixture_speculative_matches_plain_greedy_with_k_gt_one() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut speculative = Engine::from_dir(
            &fixture,
            EngineConfig {
                draft_model: Some(fixture.clone()),
                num_speculative_tokens: 3,
                ..Default::default()
            },
        )?;

        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 6;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.num_speculative_tokens = Some(3);

        let baseline_result = baseline.generate(request.clone())?;
        let speculative_result = speculative.generate(request)?;

        assert_eq!(speculative_result.token_ids, baseline_result.token_ids);
        assert_eq!(
            speculative_result.finish_reason,
            baseline_result.finish_reason
        );
        assert_eq!(speculative_result.token_ids.len(), 6);
        Ok(())
    }

    #[test]
    fn tiny_fixture_stops_at_explicit_context_length_without_ort_error() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3]));
        request.options.max_new_tokens = 32;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.max_context = Some(16);

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids.len(), 13);
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_stops_at_explicit_context_length_without_ort_error()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let session_id = engine.create_session()?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3]));
        request.options.max_new_tokens = 32;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.max_context = Some(16);

        let result = engine.generate_in_session(session_id, request)?;

        assert_eq!(result.token_ids.len(), 13);
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert_eq!(engine.session_token_count(session_id)?, 16);
        engine.close_session(session_id)?;
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_persists_context_across_turns() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let session_id = engine.create_session()?;

        let mut first = GenerateRequest::new("hello");
        first.options.max_new_tokens = 2;
        first.options.temperature = 0.0;
        first.options.stop_on_eos = false;
        let first_result = engine.generate_in_session(session_id, first)?;
        let first_count = engine.session_token_count(session_id)?;

        let mut second = GenerateRequest::new(" world");
        second.options.max_new_tokens = 2;
        second.options.temperature = 0.0;
        second.options.stop_on_eos = false;
        let second_result = engine.generate_in_session(session_id, second)?;
        let second_count = engine.session_token_count(session_id)?;

        assert_eq!(first_result.token_ids.len(), 2);
        assert_eq!(second_result.token_ids.len(), 2);
        assert!(second_count > first_count);
        assert!(engine.sessions[&session_id].kv_token_count > 0);
        engine.close_session(session_id)?;
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "requires ONNX_GENAI_FIM_MODEL_DIR to point at a FIM-capable coder model"]
    fn fim_generation_runs_with_fim_capable_model() -> anyhow::Result<()> {
        let Ok(model_dir) = std::env::var("ONNX_GENAI_FIM_MODEL_DIR") else {
            eprintln!("set ONNX_GENAI_FIM_MODEL_DIR to a Qwen2.5-Coder/StarCoder-style model");
            return Ok(());
        };
        let mut engine = Engine::from_dir(Path::new(&model_dir), EngineConfig::default())?;
        assert!(
            engine.fim_config().is_some(),
            "model tokenizer_config.json must expose recognized FIM tokens"
        );

        let mut options = GenerateOptions {
            max_new_tokens: 16,
            temperature: 0.0,
            ..Default::default()
        };
        options
            .stop_sequences
            .push(StopSequence::Text("\n\n".into()));

        let result =
            engine.generate_fim("fn add(a: i32, b: i32) -> i32 {\n    ", "\n}", options)?;

        assert!(!result.token_ids.is_empty());
        Ok(())
    }
}
