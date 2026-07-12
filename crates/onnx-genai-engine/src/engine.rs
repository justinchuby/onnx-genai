//! Main generation engine.

use crate::logits::{
    ConstraintProcessor, FrequencyPenaltyProcessor, GrammarConstraintKind, JsonConstraint,
    LlguidanceConstraint, MinPProcessor, PresencePenaltyProcessor, ProcessorChain,
    ProcessorContext, ProcessorSignal, RepetitionPenaltyProcessor, StopSequence,
    StopSequenceProcessor, TemperatureProcessor, TokenId, TopKProcessor, TopPProcessor,
};
use crate::sampling::{sample_categorical, sample_greedy};
use anyhow::Context;
use onnx_genai_kv::{
    KvCacheOps, KvDType, LayerKv, PageId, PageTensorConfig, PagedKvCache, PrefixCache, SequenceId,
};
use onnx_genai_metadata::InferenceMetadata;
use onnx_genai_ort::{
    DataType, DecodeSession, DecodeSessionOptions, Environment, ModelDirectory, Session,
    SessionOptions, StaticCacheDecodeOptions, StaticCacheDecodeSession, TensorInfo, Tokenizer,
    Value,
};
use onnx_genai_scheduler::{Priority, Scheduler, SchedulerConfig};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Identifier for a persistent generation session.
pub type SessionId = SequenceId;

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Number of GPU pages for KV cache.
    pub num_gpu_pages: usize,
    /// Tokens per KV page.
    pub page_size: usize,
    /// Scheduler config.
    pub scheduler: SchedulerConfig,
    /// Optional draft model directory used for greedy speculative decoding.
    pub draft_model: Option<PathBuf>,
    /// Number of draft tokens proposed per speculative step.
    pub num_speculative_tokens: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_gpu_pages: 1024,
            page_size: 16,
            scheduler: SchedulerConfig::default(),
            draft_model: None,
            num_speculative_tokens: 4,
        }
    }
}

/// Prompt input accepted by Phase 1 generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratePrompt {
    /// Raw prompt text.
    Text(String),
    /// Already-tokenized prompt ids.
    TokenIds(Vec<TokenId>),
}

impl From<String> for GeneratePrompt {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for GeneratePrompt {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<TokenId>> for GeneratePrompt {
    fn from(value: Vec<TokenId>) -> Self {
        Self::TokenIds(value)
    }
}

/// User-controllable decoding options for Phase 1 generation.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Maximum tokens to produce after the prompt.
    pub max_new_tokens: usize,
    /// Temperature applied before sampling. Zero forces greedy selection.
    pub temperature: f32,
    /// Nucleus sampling probability. Values >= 1 disable top-p filtering.
    pub top_p: f32,
    /// Keep only the top-k logits before sampling. Zero disables top-k filtering.
    pub top_k: usize,
    /// Min-p sampling threshold. Zero disables min-p filtering.
    pub min_p: f32,
    /// Repetition penalty applied to prompt and generated tokens. Values <= 1 disable it.
    pub repetition_penalty: f32,
    /// OpenAI-style count penalty: logit[t] -= frequency_penalty * count(t).
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty: logit[t] -= presence_penalty once if seen.
    pub presence_penalty: f32,
    /// If true, choose argmax after processors; otherwise sample categorically.
    pub greedy: bool,
    /// Text or token sequences that terminate generation when matched as a suffix.
    pub stop_sequences: Vec<StopSequence>,
    /// Optional EOS token id.
    pub eos_token_id: Option<TokenId>,
    /// Whether matching `eos_token_id` terminates generation.
    pub stop_on_eos: bool,
    /// Optional maximum total context length (prompt + generated tokens).
    /// Used when model metadata does not declare `model.max_sequence_length`.
    pub max_context: Option<usize>,
    /// Optional per-request override for speculative draft width K.
    pub num_speculative_tokens: Option<usize>,
    /// Optional constrained decoding grammar. None preserves unconstrained generation.
    pub constraint: Option<GenerateConstraint>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            greedy: true,
            stop_sequences: Vec::new(),
            eos_token_id: None,
            stop_on_eos: true,
            max_context: None,
            num_speculative_tokens: None,
            constraint: None,
        }
    }
}

/// Built-in constrained decoding grammars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerateConstraint {
    /// Constrain output to one complete, well-formed JSON value.
    Json,
    /// Constrain output to a JSON value accepted by the provided JSON Schema.
    JsonSchema(String),
    /// Constrain output to text matching the provided Rust regular expression.
    Regex(String),
    /// Constrain output to the provided llguidance Lark grammar.
    Lark(String),
}

impl GenerateOptions {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if self.max_new_tokens == 0 {
            anyhow::bail!("max_new_tokens must be greater than zero");
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            anyhow::bail!("temperature must be finite and non-negative");
        }
        if !self.top_p.is_finite() || self.top_p < 0.0 {
            anyhow::bail!("top_p must be finite and non-negative");
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            anyhow::bail!("min_p must be finite and between 0 and 1");
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            anyhow::bail!("repetition_penalty must be finite and greater than zero");
        }
        if !self.frequency_penalty.is_finite() {
            anyhow::bail!("frequency_penalty must be finite");
        }
        if !self.presence_penalty.is_finite() {
            anyhow::bail!("presence_penalty must be finite");
        }
        if self.max_context == Some(0) {
            anyhow::bail!("max_context must be greater than zero when provided");
        }
        if self.num_speculative_tokens == Some(0) {
            anyhow::bail!("num_speculative_tokens must be greater than zero when provided");
        }
        Ok(())
    }
}

/// A single generation request.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Prompt text or token ids.
    pub prompt: GeneratePrompt,
    /// Decoding options.
    pub options: GenerateOptions,
}

impl GenerateRequest {
    pub fn new(prompt: impl Into<GeneratePrompt>) -> Self {
        Self {
            prompt: prompt.into(),
            options: GenerateOptions::default(),
        }
    }
}

/// A generation request with an explicit scheduler priority.
#[derive(Debug, Clone)]
pub struct PrioritizedGenerateRequest {
    pub session_id: SessionId,
    pub request: GenerateRequest,
    pub priority: Priority,
}

/// A prioritized request that becomes visible to the engine after a decode-step count.
#[derive(Debug, Clone)]
pub struct ScheduledGenerateArrival {
    pub arrival_step: usize,
    pub request: PrioritizedGenerateRequest,
}

/// Result for one request driven through the priority scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrioritizedGenerateResult {
    pub session_id: SessionId,
    pub result: GenerateResult,
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The configured maximum number of new tokens was reached.
    MaxTokens,
    /// The configured EOS token was generated.
    EosToken,
    /// A stop sequence matched; index refers to `GenerateOptions::stop_sequences`.
    StopSequence { index: usize },
    /// The model context window was reached before another decode step could run.
    Length,
}

/// Final generation output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateResult {
    /// Detokenized generated text.
    pub text: String,
    /// Generated token ids, excluding prompt tokens.
    pub token_ids: Vec<TokenId>,
    /// Termination reason.
    pub finish_reason: FinishReason,
    /// Number of prompt/context tokens whose KV state was reused from the prefix cache.
    pub prefix_cache_hit_len: usize,
}

/// Per-token streaming event shape for future callback/iterator APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateToken {
    pub token_id: TokenId,
    pub text: String,
    pub finish_reason: Option<FinishReason>,
}

/// Streaming callback shape. Returning an error aborts generation.
pub type GenerateTokenCallback<'a> = dyn FnMut(GenerateToken) -> anyhow::Result<()> + Send + 'a;

/// The generation engine.
pub struct Engine {
    /// Model inference metadata.
    metadata: InferenceMetadata,
    /// KV cache manager.
    kv_cache: PagedKvCache,
    /// Shared-prefix cache for reusing paged KV across sessions.
    prefix_cache: PrefixCache,
    /// Token-only prefix index used by ORT-owned decode sessions until page import/export lands.
    token_prefix_cache: Vec<Vec<TokenId>>,
    /// KV tensor layout inferred from model present/past TensorInfo.
    kv_model: Option<KvModelInfo>,
    /// ORT decode path selected by model I/O introspection.
    decode_path: ModelDecodePath,
    /// Batch scheduler.
    scheduler: Scheduler,
    /// Persistent multi-turn session state, keyed by session id.
    sessions: HashMap<SessionId, EngineSession>,
    /// ORT environment kept alive for the session.
    _environment: Environment,
    /// ORT session for decoder execution.
    session: Box<Session>,
    /// Optional draft model used by the speculative decoding path.
    draft: Option<DraftModel>,
    /// Tokenizer loaded from the model directory.
    tokenizer: Tokenizer,
    /// Default speculative draft width K.
    num_speculative_tokens: usize,
}

impl Engine {
    /// Load a model from a directory.
    pub fn from_dir(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        let model_directory = ModelDirectory::load(model_dir)
            .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {}", e))?;

        // Load metadata
        let metadata = if let Some(metadata_path) = &model_directory.metadata_path {
            onnx_genai_metadata::load_metadata(metadata_path)
                .map_err(|e| anyhow::anyhow!("Failed to load metadata: {}", e))?
        } else {
            tracing::warn!("No inference metadata found, using defaults");
            InferenceMetadata {
                required_capabilities: vec![],
                model: None,
                kv_cache: None,
                quantization: None,
                pipeline: None,
                strategy: None,
                structured_output: None,
                hardware_requirements: None,
            }
        };

        // Validate capabilities
        let runtime_caps = onnx_genai_metadata::RuntimeCapabilities::default();
        if let Err(unsupported) = onnx_genai_metadata::validate(&metadata, &runtime_caps) {
            anyhow::bail!("Unsupported capabilities: {:?}", unsupported);
        }

        // Initialize scheduler
        let scheduler = Scheduler::new(config.scheduler);

        let environment = Environment::new("onnx-genai-engine")
            .map_err(|e| anyhow::anyhow!("Failed to create ORT environment: {}", e))?;
        let session = Session::new(
            &environment,
            &model_directory.model_path,
            SessionOptions::default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to load ORT session: {}", e))?;
        let metadata_max_context = metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length);
        let decode_path = detect_model_decode_path(&session, metadata_max_context)?;
        let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let kv_model = infer_kv_model_info(&session, config.page_size)?;
        let draft = if let Some(draft_model_path) = &config.draft_model {
            let draft_directory = ModelDirectory::load(draft_model_path)
                .map_err(|e| anyhow::anyhow!("Failed to resolve draft model directory: {}", e))?;
            let draft_session = Session::new(
                &environment,
                &draft_directory.model_path,
                SessionOptions::default(),
            )
            .map_err(|e| anyhow::anyhow!("Failed to load draft ORT session: {}", e))?;
            let draft_decode_path = detect_model_decode_path(&draft_session, metadata_max_context)?;
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
            tokenizer,
            num_speculative_tokens: config.num_speculative_tokens.max(1),
        })
    }

    /// Generate text for a request.
    ///
    pub fn generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult> {
        self.generate_with_callback(request, None)
    }

    /// Generate text and optionally stream each generated token to `callback`.
    pub fn generate_with_callback(
        &mut self,
        request: GenerateRequest,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        let session_id = self.create_session()?;
        let result =
            self.generate_in_session_with_callback(session_id, request, callback.as_deref_mut());
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
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_priority_and_callback(
            session_id,
            request,
            Priority::Normal,
            callback.as_deref_mut(),
        )
    }

    fn generate_in_session_with_priority_and_callback(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        priority: Priority,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
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
        let mut generated_tokens = Vec::new();
        let mut generated_text = String::new();

        let mut state = self
            .sessions
            .remove(&session_id)
            .with_context(|| format!("session {session_id} not found"))?;
        let prefix_cache_hit_len =
            self.prepare_session_prefix(session_id, &mut state, &prompt_tokens)?;

        let result = (|| -> anyhow::Result<GenerateResult> {
            if self.should_use_speculative(&options) {
                return self.generate_speculative_loop(
                    session_id,
                    &mut state,
                    &options,
                    &chain,
                    max_context,
                    prefix_cache_hit_len,
                    &mut generated_tokens,
                    &mut generated_text,
                    callback.as_deref_mut(),
                );
            }

            for step in 0..options.max_new_tokens {
                if reached_context_limit(state.tokens.len(), max_context) {
                    ensure_constrained_finish(&options, &generated_text, FinishReason::Length)?;
                    return Ok(GenerateResult {
                        text: self.tokenizer.decode(&generated_tokens).map_err(|e| {
                            anyhow::anyhow!("Failed to detokenize generated tokens: {}", e)
                        })?,
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Length,
                        prefix_cache_hit_len,
                    });
                }

                let mut context = ProcessorContext {
                    prompt_tokens: state.tokens.clone(),
                    generated_tokens: generated_tokens.clone(),
                    generated_text: generated_text.clone(),
                    step,
                };

                let mut logits = next_session_token_logits(
                    &self.session,
                    self.kv_model.as_ref(),
                    &mut self.kv_cache,
                    session_id,
                    &mut state,
                )?;
                let token_id = select_next_token(&mut logits, &context, &options, &chain, 0.0);
                generated_tokens.push(token_id);
                state.tokens.push(token_id);
                self.scheduler.advance(session_id);

                let token_text = self
                    .tokenizer
                    .decode(&[token_id])
                    .map_err(|e| anyhow::anyhow!("Failed to detokenize token {token_id}: {}", e))?;
                generated_text.push_str(&token_text);
                context.generated_tokens = generated_tokens.clone();
                context.generated_text = generated_text.clone();

                let finish_reason = finish_reason_after_token(token_id, &options, &chain, &context);
                if let Some(callback) = callback.as_deref_mut() {
                    callback(GenerateToken {
                        token_id,
                        text: token_text,
                        finish_reason: finish_reason.clone(),
                    })?;
                }

                if let Some(finish_reason) = finish_reason {
                    return Ok(GenerateResult {
                        text: self.tokenizer.decode(&generated_tokens).map_err(|e| {
                            anyhow::anyhow!("Failed to detokenize generated tokens: {}", e)
                        })?,
                        token_ids: generated_tokens,
                        finish_reason,
                        prefix_cache_hit_len,
                    });
                }
            }

            ensure_constrained_finish(&options, &generated_text, FinishReason::MaxTokens)?;
            Ok(GenerateResult {
                text: self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("Failed to detokenize generated tokens: {}", e))?,
                token_ids: generated_tokens,
                finish_reason: FinishReason::MaxTokens,
                prefix_cache_hit_len,
            })
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
        let decode_state = DecodeState::new_for_path(&self.session, &self.decode_path)?;
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
        let state = self
            .sessions
            .get_mut(&session_id)
            .context("session disappeared during reset")?;
        state.tokens.clear();
        state.kv_token_count = 0;
        state.decode_state = DecodeState::new_for_path(&self.session, &self.decode_path)?;
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

        if started_empty && state.decode_state.has_runner() {
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
            step: 0,
        })
    }

    fn step_active_generate(
        &mut self,
        active: &mut ActiveGenerate,
    ) -> anyhow::Result<Option<GenerateResult>> {
        if reached_context_limit(active.state.tokens.len(), active.max_context) {
            ensure_constrained_finish(
                &active.options,
                &active.generated_text,
                FinishReason::Length,
            )?;
            return self
                .finish_result(
                    &active.generated_tokens,
                    FinishReason::Length,
                    active.prefix_cache_hit_len,
                )
                .map(Some);
        }

        let mut context = ProcessorContext {
            prompt_tokens: active.state.tokens.clone(),
            generated_tokens: active.generated_tokens.clone(),
            generated_text: active.generated_text.clone(),
            step: active.step,
        };

        let mut logits = next_session_token_logits(
            &self.session,
            self.kv_model.as_ref(),
            &mut self.kv_cache,
            active.session_id,
            &mut active.state,
        )?;
        let token_id =
            select_next_token(&mut logits, &context, &active.options, &active.chain, 0.0);
        active.generated_tokens.push(token_id);
        active.state.tokens.push(token_id);
        self.scheduler.advance(active.session_id);

        let token_text = self
            .tokenizer
            .decode(&[token_id])
            .map_err(|e| anyhow::anyhow!("Failed to detokenize token {token_id}: {}", e))?;
        active.generated_text.push_str(&token_text);
        context.generated_tokens = active.generated_tokens.clone();
        context.generated_text = active.generated_text.clone();

        active.step += 1;
        if let Some(finish_reason) =
            finish_reason_after_token(token_id, &active.options, &active.chain, &context)
        {
            return self
                .finish_result(
                    &active.generated_tokens,
                    finish_reason,
                    active.prefix_cache_hit_len,
                )
                .map(Some);
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
        if state.decode_state.has_runner() {
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

    fn should_use_speculative(&self, options: &GenerateOptions) -> bool {
        self.draft.is_some()
            // Grammar processors carry per-request parser state; draft/verify
            // would need separate parser branches for speculative candidates.
            && options.constraint.is_none()
            && (options.greedy || options.temperature == 0.0)
            && options
                .num_speculative_tokens
                .unwrap_or(self.num_speculative_tokens)
                > 0
            && self.kv_model.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_speculative_loop(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        options: &GenerateOptions,
        chain: &ProcessorChain,
        max_context: Option<usize>,
        prefix_cache_hit_len: usize,
        generated_tokens: &mut Vec<TokenId>,
        generated_text: &mut String,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        let draft_width = options
            .num_speculative_tokens
            .unwrap_or(self.num_speculative_tokens)
            .max(1);
        let mut step = 0;

        loop {
            if generated_tokens.len() >= options.max_new_tokens {
                ensure_constrained_finish(options, generated_text, FinishReason::MaxTokens)?;
                return self.finish_result(
                    generated_tokens,
                    FinishReason::MaxTokens,
                    prefix_cache_hit_len,
                );
            }
            if reached_context_limit(state.tokens.len(), max_context) {
                ensure_constrained_finish(options, generated_text, FinishReason::Length)?;
                return self.finish_result(
                    generated_tokens,
                    FinishReason::Length,
                    prefix_cache_hit_len,
                );
            }

            let remaining_tokens = options.max_new_tokens - generated_tokens.len();
            let remaining_context = max_context
                .map(|limit| limit.saturating_sub(state.tokens.len()))
                .unwrap_or(remaining_tokens);
            let width = draft_width
                .min(remaining_tokens)
                .min(remaining_context)
                .max(1);

            let base_len = state.tokens.len();
            let base_generated_len = generated_tokens.len();
            let mut base_logits = next_session_token_logits(
                &self.session,
                self.kv_model.as_ref(),
                &mut self.kv_cache,
                session_id,
                state,
            )?;

            let draft_tokens = {
                let draft_model = self
                    .draft
                    .as_mut()
                    .context("speculative decoding requested without a draft model")?;
                let draft_state = state
                    .draft
                    .as_mut()
                    .context("speculative session missing draft state")?;
                draft_state.tokens = state.tokens[..base_len].to_vec();
                if draft_state.kv_token_count > base_len {
                    rewind_draft_state_to_len(draft_model, draft_state, base_len)?;
                }
                propose_draft_tokens(
                    draft_model,
                    draft_state,
                    width,
                    generated_tokens,
                    generated_text,
                    step,
                    options,
                    chain,
                )?
            };

            state.tokens.extend_from_slice(&draft_tokens);
            let verified_logits = if state.decode_state.has_runner() {
                let logits =
                    run_decode_session_logits(&mut state.decode_state, &draft_tokens, base_len)?;
                self.kv_cache
                    .append(session_id, draft_tokens.len())
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to advance KV sequence {session_id}: {}", e)
                    })?;
                state.kv_token_count += draft_tokens.len();
                logits
            } else {
                let outputs = run_decode_step(
                    &self.session,
                    &mut state.decode_state,
                    &draft_tokens,
                    base_len,
                )?;
                if state.decode_state.use_kv {
                    if let Some(kv_model) = &self.kv_model {
                        mirror_present_kv_to_pages(
                            &self.session,
                            kv_model,
                            &mut self.kv_cache,
                            session_id,
                            &outputs,
                            base_len,
                            draft_tokens.len(),
                        )?;
                    } else {
                        self.kv_cache
                            .append(session_id, draft_tokens.len())
                            .map_err(|e| {
                                anyhow::anyhow!("Failed to advance KV sequence {session_id}: {}", e)
                            })?;
                    }
                    state.kv_token_count += draft_tokens.len();
                }
                extract_logits_sequence(&self.session, outputs)?
            };

            let mut target_logits = Vec::with_capacity(draft_tokens.len() + 1);
            target_logits.push(std::mem::take(&mut base_logits));
            target_logits.extend(verified_logits);

            let mut accepted = 0;
            let mut replacement = None;
            for idx in 0..draft_tokens.len() {
                let mut context = ProcessorContext {
                    prompt_tokens: state.tokens[..base_len].to_vec(),
                    generated_tokens: generated_tokens
                        .iter()
                        .copied()
                        .chain(draft_tokens[..idx].iter().copied())
                        .collect(),
                    generated_text: self
                        .tokenizer
                        .decode(
                            &generated_tokens
                                .iter()
                                .copied()
                                .chain(draft_tokens[..idx].iter().copied())
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to detokenize speculative context: {}", e)
                        })?,
                    step: step + idx,
                };
                let target_token =
                    select_next_token(&mut target_logits[idx], &context, options, chain, 0.0);
                if target_token == draft_tokens[idx] {
                    accepted += 1;
                } else {
                    replacement = Some(target_token);
                    context.generated_tokens.push(target_token);
                    break;
                }
            }

            let mut commit_tokens = draft_tokens[..accepted].to_vec();
            let rewind_len = base_len + accepted;
            rewind_target_state_to_len(
                &self.session,
                self.kv_model.as_ref(),
                &mut self.kv_cache,
                session_id,
                state,
                rewind_len,
            )?;

            if let Some(token) = replacement {
                commit_tokens.push(token);
            } else if generated_tokens.len() + commit_tokens.len() < options.max_new_tokens
                && !reached_context_limit(base_len + commit_tokens.len(), max_context)
            {
                let mut context = ProcessorContext {
                    prompt_tokens: state.tokens[..base_len].to_vec(),
                    generated_tokens: generated_tokens
                        .iter()
                        .copied()
                        .chain(draft_tokens.iter().copied())
                        .collect(),
                    generated_text: self
                        .tokenizer
                        .decode(
                            &generated_tokens
                                .iter()
                                .copied()
                                .chain(draft_tokens.iter().copied())
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to detokenize speculative context: {}", e)
                        })?,
                    step: step + draft_tokens.len(),
                };
                let token = select_next_token(
                    target_logits
                        .last_mut()
                        .context("target verification did not produce next-token logits")?,
                    &context,
                    options,
                    chain,
                    0.0,
                );
                context.generated_tokens.push(token);
                commit_tokens.push(token);
            }

            for (commit_idx, token_id) in commit_tokens.into_iter().enumerate() {
                if generated_tokens.len() >= options.max_new_tokens
                    || (commit_idx >= accepted
                        && reached_context_limit(state.tokens.len(), max_context))
                {
                    break;
                }
                generated_tokens.push(token_id);
                if commit_idx >= accepted {
                    state.tokens.push(token_id);
                }
                self.scheduler.advance(session_id);
                let token_text = self
                    .tokenizer
                    .decode(&[token_id])
                    .map_err(|e| anyhow::anyhow!("Failed to detokenize token {token_id}: {}", e))?;
                generated_text.push_str(&token_text);
                let context = ProcessorContext {
                    prompt_tokens: state.tokens[..base_len.min(state.tokens.len())].to_vec(),
                    generated_tokens: generated_tokens.clone(),
                    generated_text: generated_text.clone(),
                    step,
                };
                let finish_reason = finish_reason_after_token(token_id, options, chain, &context);
                if let Some(callback) = callback.as_deref_mut() {
                    callback(GenerateToken {
                        token_id,
                        text: token_text,
                        finish_reason: finish_reason.clone(),
                    })?;
                }
                step += 1;
                if let Some(finish_reason) = finish_reason {
                    trim_overmaterialized_target_kv(
                        &self.session,
                        self.kv_model.as_ref(),
                        &mut self.kv_cache,
                        session_id,
                        state,
                    )?;
                    self.sync_draft_to_target(state)?;
                    return self.finish_result(
                        generated_tokens,
                        finish_reason,
                        prefix_cache_hit_len,
                    );
                }
            }

            self.sync_draft_to_target(state)?;

            if generated_tokens.len() == base_generated_len {
                anyhow::bail!("speculative decoding made no progress");
            }
        }
    }

    fn sync_draft_to_target(&mut self, state: &mut EngineSession) -> anyhow::Result<()> {
        if let (Some(draft_model), Some(draft_state)) = (&mut self.draft, &mut state.draft) {
            let common_len = common_prefix_len(&draft_state.tokens, &state.tokens);
            if draft_state.kv_token_count > common_len {
                rewind_draft_state_to_len(draft_model, draft_state, common_len)?;
            }
            draft_state.tokens = state.tokens.clone();
        }
        Ok(())
    }

    fn finish_result(
        &self,
        generated_tokens: &[TokenId],
        finish_reason: FinishReason,
        prefix_cache_hit_len: usize,
    ) -> anyhow::Result<GenerateResult> {
        Ok(GenerateResult {
            text: self
                .tokenizer
                .decode(generated_tokens)
                .map_err(|e| anyhow::anyhow!("Failed to detokenize generated tokens: {}", e))?,
            token_ids: generated_tokens.to_vec(),
            finish_reason,
            prefix_cache_hit_len,
        })
    }
}

struct EngineSession {
    /// Logical token context retained across turns.
    tokens: Vec<TokenId>,
    /// Prefix length currently materialized in `decode_state.past`.
    kv_token_count: usize,
    /// ORT-managed past tensors retained between calls.
    decode_state: DecodeState,
    /// Optional draft-model state aligned to this target sequence.
    draft: Option<DraftSession>,
}

struct ActiveGenerate {
    session_id: SessionId,
    state: EngineSession,
    options: GenerateOptions,
    chain: ProcessorChain,
    max_context: Option<usize>,
    prompt_len: usize,
    prefix_cache_hit_len: usize,
    generated_tokens: Vec<TokenId>,
    generated_text: String,
    step: usize,
}

struct DraftModel {
    session: Box<Session>,
    decode_path: ModelDecodePath,
    kv_model: Option<KvModelInfo>,
    kv_cache: PagedKvCache,
}

struct DraftSession {
    seq: SessionId,
    tokens: Vec<TokenId>,
    kv_token_count: usize,
    decode_state: DecodeState,
}

#[derive(Debug, Clone)]
pub(crate) struct KvModelInfo {
    tensor_config: PageTensorConfig,
    layers: Vec<KvLayerIo>,
}

#[derive(Debug, Clone)]
struct KvLayerIo {
    key_present: String,
    value_present: String,
    key_past: String,
    value_past: String,
}

#[derive(Debug, Clone)]
enum ModelDecodePath {
    StaticCache {
        max_len: usize,
    },
    PastPresent {
        shared_buffer: bool,
        max_len: Option<usize>,
    },
    Legacy,
}

enum DecodeRunner {
    StaticCache(StaticCacheDecodeSession<'static>),
    PastPresent(DecodeSession<'static>),
}

pub(crate) struct DecodeState {
    pub(crate) use_kv: bool,
    past: HashMap<String, Value>,
    present_to_past: HashMap<String, String>,
    kv_inputs: Vec<String>,
    runner: Option<DecodeRunner>,
}

impl DecodeState {
    pub(crate) fn new(session: &Session) -> anyhow::Result<Self> {
        let kv_inputs = session
            .inputs()
            .iter()
            .filter(|info| is_kv_input(&info.name))
            .map(|info| info.name.clone())
            .collect::<Vec<_>>();
        let present_outputs = session
            .outputs()
            .iter()
            .filter(|info| is_present_output(&info.name))
            .map(|info| info.name.clone())
            .collect::<Vec<_>>();

        if kv_inputs.is_empty() && present_outputs.is_empty() {
            return Ok(Self {
                use_kv: false,
                past: HashMap::new(),
                present_to_past: HashMap::new(),
                kv_inputs,
                runner: None,
            });
        }

        let mut present_to_past = HashMap::new();
        for output in &present_outputs {
            if let Some(input) = matching_past_input(output, &kv_inputs) {
                present_to_past.insert(output.clone(), input.clone());
            }
        }

        if kv_inputs.is_empty()
            || present_outputs.is_empty()
            || present_to_past.len() != present_outputs.len()
        {
            anyhow::bail!(
                "model exposes incomplete KV I/O; past inputs: {:?}, present outputs: {:?}",
                kv_inputs,
                present_outputs
            );
        }

        Ok(Self {
            use_kv: true,
            past: HashMap::new(),
            present_to_past,
            kv_inputs,
            runner: None,
        })
    }

    fn new_for_path(session: &Session, path: &ModelDecodePath) -> anyhow::Result<Self> {
        match path {
            ModelDecodePath::Legacy => Self::new(session),
            ModelDecodePath::StaticCache { .. } => Ok(Self {
                use_kv: true,
                past: HashMap::new(),
                present_to_past: HashMap::new(),
                kv_inputs: Vec::new(),
                runner: Some(DecodeRunner::StaticCache(StaticCacheDecodeSession::new(
                    stable_session_ref(session),
                    StaticCacheDecodeOptions { batch_size: 1 },
                )?)),
            }),
            ModelDecodePath::PastPresent {
                shared_buffer,
                max_len,
            } => {
                let mut state = Self::new(session)?;
                if state.use_kv {
                    state.runner = Some(DecodeRunner::PastPresent(DecodeSession::new(
                        stable_session_ref(session),
                        DecodeSessionOptions {
                            batch_size: 1,
                            max_length: *max_len,
                            past_present_share_buffer: Some(*shared_buffer),
                        },
                    )?));
                }
                Ok(state)
            }
        }
    }

    fn has_runner(&self) -> bool {
        self.runner.is_some()
    }

    fn runner_len(&self) -> usize {
        match &self.runner {
            Some(DecodeRunner::StaticCache(session)) => session.current_len(),
            Some(DecodeRunner::PastPresent(session)) => session.past_len(),
            None => 0,
        }
    }

    fn rewind_runner(&mut self, target_len: usize) -> anyhow::Result<()> {
        match &mut self.runner {
            Some(DecodeRunner::StaticCache(session)) => session.rewind(target_len)?,
            Some(DecodeRunner::PastPresent(session)) => session.rewind(target_len)?,
            None => {
                self.past.clear();
            }
        }
        Ok(())
    }
}

fn next_session_token_logits(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
) -> anyhow::Result<Vec<f32>> {
    let (input_tokens, past_len) = session_decode_input_tokens(state)?;
    let input_len = input_tokens.len();
    if state.decode_state.has_runner() {
        let logits = run_decode_session_logits(&mut state.decode_state, &input_tokens, past_len)?;
        kv_cache
            .append(seq, input_len)
            .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {}", e))?;
        state.kv_token_count += input_len;
        return logits
            .into_iter()
            .last()
            .context("decode session produced no logits");
    }
    let outputs = run_decode_step(session, &mut state.decode_state, &input_tokens, past_len)?;
    if state.decode_state.use_kv {
        if let Some(kv_model) = kv_model {
            mirror_present_kv_to_pages(
                session, kv_model, kv_cache, seq, &outputs, past_len, input_len,
            )?;
        } else {
            kv_cache
                .append(seq, input_len)
                .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {}", e))?;
        }
        state.kv_token_count += input_len;
    }
    extract_next_token_logits(session, outputs)
}

fn next_draft_token_logits(
    draft_model: &mut DraftModel,
    draft_state: &mut DraftSession,
) -> anyhow::Result<Vec<f32>> {
    let (input_tokens, past_len) = draft_decode_input_tokens(draft_state)?;
    let input_len = input_tokens.len();
    if draft_state.decode_state.has_runner() {
        let logits =
            run_decode_session_logits(&mut draft_state.decode_state, &input_tokens, past_len)?;
        draft_model
            .kv_cache
            .append(draft_state.seq, input_len)
            .map_err(|e| anyhow::anyhow!("Failed to advance draft KV sequence: {}", e))?;
        draft_state.kv_token_count += input_len;
        return logits
            .into_iter()
            .last()
            .context("draft decode session produced no logits");
    }
    let outputs = run_decode_step(
        &draft_model.session,
        &mut draft_state.decode_state,
        &input_tokens,
        past_len,
    )?;
    if draft_state.decode_state.use_kv {
        if let Some(kv_model) = &draft_model.kv_model {
            mirror_present_kv_to_pages(
                &draft_model.session,
                kv_model,
                &mut draft_model.kv_cache,
                draft_state.seq,
                &outputs,
                past_len,
                input_len,
            )?;
        } else {
            draft_model
                .kv_cache
                .append(draft_state.seq, input_len)
                .map_err(|e| anyhow::anyhow!("Failed to advance draft KV sequence: {}", e))?;
        }
        draft_state.kv_token_count += input_len;
    }
    extract_next_token_logits(&draft_model.session, outputs)
}

#[allow(clippy::too_many_arguments)]
fn propose_draft_tokens(
    draft_model: &mut DraftModel,
    draft_state: &mut DraftSession,
    width: usize,
    generated_tokens: &[TokenId],
    generated_text: &str,
    first_step: usize,
    options: &GenerateOptions,
    chain: &ProcessorChain,
) -> anyhow::Result<Vec<TokenId>> {
    let prompt_len = draft_state
        .tokens
        .len()
        .saturating_sub(generated_tokens.len());
    let mut proposed = Vec::with_capacity(width);
    let mut draft_generated = generated_tokens.to_vec();
    let mut draft_text = generated_text.to_string();

    for offset in 0..width {
        let mut logits = next_draft_token_logits(draft_model, draft_state)?;
        let context = ProcessorContext {
            prompt_tokens: draft_state.tokens[..prompt_len.min(draft_state.tokens.len())].to_vec(),
            generated_tokens: draft_generated.clone(),
            generated_text: draft_text.clone(),
            step: first_step + offset,
        };
        let token = select_next_token(&mut logits, &context, options, chain, 0.0);
        proposed.push(token);
        draft_generated.push(token);
        draft_state.tokens.push(token);
        draft_text.clear();
    }

    Ok(proposed)
}

fn session_decode_input_tokens(state: &EngineSession) -> anyhow::Result<(Vec<TokenId>, usize)> {
    if state.decode_state.use_kv {
        if state.kv_token_count > state.tokens.len() {
            anyhow::bail!(
                "session KV token count {} exceeds logical context length {}",
                state.kv_token_count,
                state.tokens.len()
            );
        }
        let input_tokens = state.tokens[state.kv_token_count..].to_vec();
        if input_tokens.is_empty() {
            anyhow::bail!("session decode step has no new token to feed");
        }
        Ok((input_tokens, state.kv_token_count))
    } else {
        if state.tokens.is_empty() {
            anyhow::bail!("decode step requires at least one context token");
        }
        Ok((state.tokens.clone(), 0))
    }
}

fn draft_decode_input_tokens(state: &DraftSession) -> anyhow::Result<(Vec<TokenId>, usize)> {
    if state.decode_state.use_kv {
        if state.kv_token_count > state.tokens.len() {
            anyhow::bail!(
                "draft KV token count {} exceeds logical context length {}",
                state.kv_token_count,
                state.tokens.len()
            );
        }
        let input_tokens = state.tokens[state.kv_token_count..].to_vec();
        if input_tokens.is_empty() {
            anyhow::bail!("draft decode step has no new token to feed");
        }
        Ok((input_tokens, state.kv_token_count))
    } else {
        if state.tokens.is_empty() {
            anyhow::bail!("draft decode step requires at least one context token");
        }
        Ok((state.tokens.clone(), 0))
    }
}

fn run_decode_session_logits(
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    if token_ids.is_empty() {
        anyhow::bail!("decode session step requires at least one input token");
    }
    let current_len = decode_state.runner_len();
    if current_len > past_len {
        decode_state.rewind_runner(past_len)?;
    } else if current_len < past_len {
        anyhow::bail!(
            "decode session cursor {} is behind requested past length {}; replay is required",
            current_len,
            past_len
        );
    }

    let input_ids = token_ids
        .iter()
        .map(|&id| i64::from(id))
        .collect::<Vec<_>>();
    match decode_state
        .runner
        .as_mut()
        .context("decode session runner not initialized")?
    {
        DecodeRunner::PastPresent(runner) => {
            let total_len = past_len + input_ids.len();
            let attention_mask = vec![1_i64; total_len];
            let position_ids = (past_len..total_len)
                .map(|pos| i64::try_from(pos).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let logits = runner.step(&input_ids, &attention_mask, &position_ids)?;
            extract_logits_value_sequence(&logits)
        }
        DecodeRunner::StaticCache(runner) => {
            if runner.current_len() == 0 {
                let position_ids = (0..input_ids.len())
                    .map(|pos| i64::try_from(pos).context("position id exceeds i64 range"))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let logits = runner.prefill(&input_ids, &position_ids)?;
                extract_logits_value_sequence(&logits)
            } else {
                let mut logits = Vec::with_capacity(input_ids.len());
                for &token in &input_ids {
                    let pos = i64::try_from(runner.current_len())
                        .context("position id exceeds i64 range")?;
                    let value = runner.step(&[token], &[pos])?;
                    logits.push(extract_logits_value_next(&value)?);
                }
                Ok(logits)
            }
        }
    }
    .map_err(|error| {
        let message = error.to_string();
        if is_gather_out_of_bounds(&message) {
            anyhow::anyhow!(
                "model context length exceeded during ORT decode; configure inference metadata `model.max_sequence_length` or GenerateOptions::max_context to stop cleanly before the context window is exceeded: {}",
                error
            )
        } else {
            error
        }
    })
}

fn run_decode_step(
    session: &Session,
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
) -> anyhow::Result<Vec<Value>> {
    run_decode_step_with_extra(session, decode_state, token_ids, past_len, &[])
}

pub(crate) fn run_decode_step_with_extra(
    session: &Session,
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
    extra_inputs: &[(String, Value)],
) -> anyhow::Result<Vec<Value>> {
    if token_ids.is_empty() {
        anyhow::bail!("decode step requires at least one input token");
    }

    let seq_len = token_ids.len();
    let total_len = past_len + seq_len;
    let input_ids = token_ids
        .iter()
        .map(|&id| i64::from(id))
        .collect::<Vec<_>>();
    let attention_mask = vec![1_i64; total_len];
    let position_ids = (past_len..total_len)
        .map(|pos| i64::try_from(pos).context("position id exceeds i64 range"))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut owned_inputs: Vec<(String, Value)> = Vec::new();
    for info in session.inputs() {
        let lower = info.name.to_ascii_lowercase();
        if lower == "input_ids" || lower.ends_with(".input_ids") {
            ensure_i64(info)?;
            owned_inputs.push((
                info.name.clone(),
                Value::from_slice_i64(&input_ids, &[1, seq_len as i64])?,
            ));
        } else if lower == "attention_mask" || lower.ends_with(".attention_mask") {
            ensure_i64(info)?;
            owned_inputs.push((
                info.name.clone(),
                Value::from_slice_i64(&attention_mask, &[1, total_len as i64])?,
            ));
        } else if lower == "position_ids" || lower.ends_with(".position_ids") {
            ensure_i64(info)?;
            owned_inputs.push((
                info.name.clone(),
                Value::from_slice_i64(&position_ids, &[1, seq_len as i64])?,
            ));
        } else if decode_state.use_kv && decode_state.kv_inputs.contains(&info.name) {
            let value = if past_len == 0 {
                empty_past_value(info)?
            } else {
                clone_value(decode_state.past.get(&info.name).with_context(|| {
                    format!("missing cached KV tensor for input '{}'", info.name)
                })?)?
            };
            owned_inputs.push((info.name.clone(), value));
        } else if let Some((_, value)) = extra_inputs.iter().find(|(name, _)| name == &info.name) {
            owned_inputs.push((info.name.clone(), clone_value(value)?));
        } else {
            anyhow::bail!(
                "unsupported model input '{}' with shape {:?}; supported inputs are input_ids, attention_mask, position_ids, past key-values, and pipeline-routed extra inputs",
                info.name,
                info.shape
            );
        }
    }

    let input_refs = owned_inputs
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<Vec<_>>();
    let outputs = session.run(&input_refs).map_err(|e| {
        let message = e.to_string();
        if is_gather_out_of_bounds(&message) {
            anyhow::anyhow!(
                "model context length exceeded during ORT decode; configure inference metadata `model.max_sequence_length` or GenerateOptions::max_context to stop cleanly before the context window is exceeded: {}",
                e
            )
        } else {
            anyhow::anyhow!("ORT session run failed: {}", e)
        }
    })?;

    if decode_state.use_kv {
        decode_state.past.clear();
        for (name, value) in session.output_names().iter().zip(outputs.iter()) {
            if let Some(past_name) = decode_state.present_to_past.get(name) {
                decode_state
                    .past
                    .insert(past_name.clone(), clone_value(value)?);
            }
        }
    }

    Ok(outputs)
}

pub(crate) fn infer_kv_model_info(
    session: &Session,
    page_size: usize,
) -> anyhow::Result<Option<KvModelInfo>> {
    let mut key_outputs = Vec::new();
    let mut value_outputs = Vec::new();
    for info in session
        .outputs()
        .iter()
        .filter(|info| is_present_output(&info.name))
    {
        let lower = info.name.to_ascii_lowercase();
        if lower.contains("key") {
            key_outputs.push(info.clone());
        } else if lower.contains("value") {
            value_outputs.push(info.clone());
        }
    }

    if key_outputs.is_empty() && value_outputs.is_empty() {
        return Ok(None);
    }
    key_outputs.sort_by_key(|info| kv_layer_index(&info.name).unwrap_or(usize::MAX));
    value_outputs.sort_by_key(|info| kv_layer_index(&info.name).unwrap_or(usize::MAX));
    if key_outputs.len() != value_outputs.len() {
        anyhow::bail!(
            "model exposes mismatched present key/value outputs: {} keys, {} values",
            key_outputs.len(),
            value_outputs.len()
        );
    }

    let (num_kv_heads, head_dim) = infer_kv_heads_and_head_dim(&key_outputs[0])?;
    let config = PageTensorConfig {
        num_layers: key_outputs.len(),
        num_kv_heads,
        head_dim,
        page_size,
        dtype: KvDType::F32,
    };
    let kv_inputs = session
        .inputs()
        .iter()
        .filter(|info| is_kv_input(&info.name))
        .map(|info| info.name.clone())
        .collect::<Vec<_>>();
    let mut layers = Vec::with_capacity(key_outputs.len());
    for (key, value) in key_outputs.iter().zip(value_outputs.iter()) {
        if key.dtype != DataType::Float32 || value.dtype != DataType::Float32 {
            anyhow::bail!("KV present outputs must be Float32");
        }
        let key_past = matching_past_input(&key.name, &kv_inputs)
            .with_context(|| format!("missing past input for present output '{}'", key.name))?
            .clone();
        let value_past = matching_past_input(&value.name, &kv_inputs)
            .with_context(|| format!("missing past input for present output '{}'", value.name))?
            .clone();
        layers.push(KvLayerIo {
            key_present: key.name.clone(),
            value_present: value.name.clone(),
            key_past,
            value_past,
        });
    }

    Ok(Some(KvModelInfo {
        tensor_config: config,
        layers,
    }))
}

fn detect_model_decode_path(
    session: &Session,
    metadata_max_context: Option<usize>,
) -> anyhow::Result<ModelDecodePath> {
    if let Some(signature) = StaticCacheDecodeSession::detect(session)? {
        return Ok(ModelDecodePath::StaticCache {
            max_len: signature.max_len,
        });
    }

    let has_kv_inputs = session.inputs().iter().any(|info| is_kv_input(&info.name));
    let has_present_outputs = session
        .outputs()
        .iter()
        .any(|info| is_present_output(&info.name));
    if has_kv_inputs || has_present_outputs {
        let shared_buffer =
            session.past_present_share_buffer_supported() && metadata_max_context.is_some();
        return Ok(ModelDecodePath::PastPresent {
            shared_buffer,
            max_len: metadata_max_context.filter(|_| shared_buffer),
        });
    }

    Ok(ModelDecodePath::Legacy)
}

fn stable_session_ref(session: &Session) -> &'static Session {
    // Decode sessions live inside EngineSession, while their referenced Session is
    // boxed in Engine/DraftModel and dropped only after all EngineSessions. The
    // boxed allocation is stable across Engine moves; the transmute narrows that
    // invariant to ORT's current reference-based decode-session API.
    unsafe { std::mem::transmute::<&Session, &'static Session>(session) }
}

fn infer_kv_heads_and_head_dim(info: &TensorInfo) -> anyhow::Result<(usize, usize)> {
    if info.dtype != DataType::Float32 || info.shape.len() < 3 {
        anyhow::bail!(
            "present KV output '{}' must be Float32 rank >= 3, got {:?} {:?}",
            info.name,
            info.dtype,
            info.shape
        );
    }
    let head_dim = *info
        .shape
        .last()
        .filter(|dim| **dim > 0)
        .with_context(|| format!("cannot infer KV head_dim from '{}'", info.name))?
        as usize;
    let num_kv_heads = info
        .shape
        .iter()
        .enumerate()
        .find_map(|(idx, &dim)| {
            (idx != 0 && idx + 1 != info.shape.len() && dim > 0).then_some(dim as usize)
        })
        .with_context(|| format!("cannot infer KV heads from '{}'", info.name))?;
    Ok((num_kv_heads, head_dim))
}

fn mirror_present_kv_to_pages(
    session: &Session,
    kv_model: &KvModelInfo,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    outputs: &[Value],
    past_len: usize,
    input_len: usize,
) -> anyhow::Result<()> {
    let output_lookup = session
        .output_names()
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let layer_data = kv_model
        .layers
        .iter()
        .map(|layer| {
            let key = outputs[*output_lookup
                .get(layer.key_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.key_present))?]
            .to_vec_f32()?;
            let key_shape = outputs[*output_lookup
                .get(layer.key_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.key_present))?]
            .shape()
            .to_vec();
            let value = outputs[*output_lookup
                .get(layer.value_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.value_present))?]
            .to_vec_f32()?;
            let value_shape = outputs[*output_lookup
                .get(layer.value_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.value_present))?]
            .shape()
            .to_vec();
            Ok((key, key_shape, value, value_shape))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    for offset in 0..input_len {
        let token_pos = past_len + offset;
        let owned_layers = layer_data
            .iter()
            .map(|(key, key_shape, value, value_shape)| {
                Ok((
                    extract_present_token(key, key_shape, kv_model.tensor_config, token_pos)?,
                    extract_present_token(value, value_shape, kv_model.tensor_config, token_pos)?,
                ))
            })
            .collect::<anyhow::Result<Vec<(Vec<f32>, Vec<f32>)>>>()?;
        let borrowed = owned_layers
            .iter()
            .map(|(key, value)| LayerKv {
                key: key.as_slice(),
                value: value.as_slice(),
            })
            .collect::<Vec<_>>();
        kv_cache
            .append_token_kv(seq, &borrowed)
            .map_err(|e| anyhow::anyhow!("Failed to mirror present KV into pages: {}", e))?;
    }
    Ok(())
}

fn extract_present_token(
    data: &[f32],
    shape: &[i64],
    config: PageTensorConfig,
    token_pos: usize,
) -> anyhow::Result<Vec<f32>> {
    let axes = kv_tensor_axes(shape, config, token_pos)?;
    let strides = row_major_strides(shape);
    let mut token = Vec::with_capacity(config.num_kv_heads * config.head_dim);
    for head in 0..config.num_kv_heads {
        for dim in 0..config.head_dim {
            let mut indices = vec![0_usize; shape.len()];
            indices[axes.head] = head;
            indices[axes.sequence] = token_pos;
            indices[axes.head_dim] = dim;
            let flat = indices
                .iter()
                .zip(strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>();
            token.push(
                *data
                    .get(flat)
                    .context("present KV tensor index out of bounds")?,
            );
        }
    }
    Ok(token)
}

fn load_materialized_past(
    session: &Session,
    kv_model: &KvModelInfo,
    decode_state: &mut DecodeState,
    materialized: &onnx_genai_kv::MaterializedKv,
) -> anyhow::Result<()> {
    let input_shapes = session
        .inputs()
        .iter()
        .map(|info| (info.name.as_str(), info.shape.as_slice()))
        .collect::<HashMap<_, _>>();
    decode_state.past.clear();
    for (idx, layer) in kv_model.layers.iter().enumerate() {
        let key_shape = past_shape(
            input_shapes
                .get(layer.key_past.as_str())
                .copied()
                .context("missing key past input shape")?,
            materialized.sequence_len,
        )?;
        let value_shape = past_shape(
            input_shapes
                .get(layer.value_past.as_str())
                .copied()
                .context("missing value past input shape")?,
            materialized.sequence_len,
        )?;
        decode_state.past.insert(
            layer.key_past.clone(),
            Value::from_vec_f32(materialized.layers[idx].key.clone(), &key_shape)?,
        );
        decode_state.past.insert(
            layer.value_past.clone(),
            Value::from_vec_f32(materialized.layers[idx].value.clone(), &value_shape)?,
        );
    }
    Ok(())
}

fn past_shape(shape: &[i64], sequence_len: usize) -> anyhow::Result<Vec<i64>> {
    if shape.len() < 3 {
        anyhow::bail!("KV past shape rank must be >= 3, got {:?}", shape);
    }
    let seq_axis = shape.len() - 2;
    Ok(shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if axis == 0 {
                1
            } else if axis == seq_axis {
                sequence_len as i64
            } else {
                dim
            }
        })
        .collect())
}

fn attach_pages_to_sequence(
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    page_ids: &[PageId],
    len: usize,
) -> anyhow::Result<()> {
    if !kv_cache
        .page_table
        .get_sequence(seq)
        .context("sequence not found")?
        .is_empty()
    {
        anyhow::bail!("cannot attach prefix pages to a non-empty sequence");
    }
    for &page_id in page_ids {
        kv_cache.page_table.push_page(seq, page_id);
    }
    kv_cache.page_table.set_sequence_len(seq, len);
    Ok(())
}

fn rewind_target_state_to_len(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
    len: usize,
) -> anyhow::Result<()> {
    state.tokens.truncate(len);
    rewind_decode_state_to_len(
        session,
        kv_model,
        kv_cache,
        seq,
        &mut state.decode_state,
        &mut state.kv_token_count,
        len,
    )
}

fn trim_overmaterialized_target_kv(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
) -> anyhow::Result<()> {
    if state.kv_token_count > state.tokens.len() {
        rewind_target_state_to_len(session, kv_model, kv_cache, seq, state, state.tokens.len())?;
    }
    Ok(())
}

fn rewind_draft_state_to_len(
    draft_model: &mut DraftModel,
    state: &mut DraftSession,
    len: usize,
) -> anyhow::Result<()> {
    state.tokens.truncate(len);
    rewind_decode_state_to_len(
        &draft_model.session,
        draft_model.kv_model.as_ref(),
        &mut draft_model.kv_cache,
        state.seq,
        &mut state.decode_state,
        &mut state.kv_token_count,
        len,
    )
}

fn common_prefix_len(left: &[TokenId], right: &[TokenId]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn rewind_decode_state_to_len(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    decode_state: &mut DecodeState,
    kv_token_count: &mut usize,
    len: usize,
) -> anyhow::Result<()> {
    if !decode_state.use_kv {
        *kv_token_count = 0;
        return Ok(());
    }
    if *kv_token_count == len {
        return Ok(());
    }
    if decode_state.has_runner() {
        kv_cache
            .rewind_to(seq, len)
            .map_err(|e| anyhow::anyhow!("Failed to rewind KV sequence {seq} to {len}: {}", e))?;
        decode_state.rewind_runner(len)?;
        *kv_token_count = len;
        return Ok(());
    }
    if kv_model.is_none() && *kv_token_count != len {
        anyhow::bail!("cannot rewind ORT KV tensors without paged KV materialization");
    }
    kv_cache
        .rewind_to(seq, len)
        .map_err(|e| anyhow::anyhow!("Failed to rewind KV sequence {seq} to {len}: {}", e))?;
    *kv_token_count = len;
    if len == 0 {
        decode_state.past.clear();
        return Ok(());
    }
    let kv_model = kv_model.context("missing KV model after rewind check")?;
    let materialized = kv_cache
        .materialize_sequence(seq)
        .map_err(|e| anyhow::anyhow!("Failed to materialize rewound KV sequence {seq}: {}", e))?;
    load_materialized_past(session, kv_model, decode_state, &materialized)
}

fn sequence_pages_for_len(
    kv_cache: &PagedKvCache,
    seq: SessionId,
    len: usize,
) -> anyhow::Result<Vec<PageId>> {
    let pages_needed = len.div_ceil(kv_cache.page_table.page_size);
    Ok(kv_cache
        .page_table
        .get_sequence(seq)
        .with_context(|| format!("sequence {seq} not found"))?
        .iter()
        .copied()
        .take(pages_needed)
        .collect())
}

struct KvTensorAxes {
    head: usize,
    sequence: usize,
    head_dim: usize,
}

fn kv_tensor_axes(
    shape: &[i64],
    config: PageTensorConfig,
    token_pos: usize,
) -> anyhow::Result<KvTensorAxes> {
    let head_dim = shape
        .iter()
        .rposition(|&dim| dim == config.head_dim as i64)
        .context("KV tensor head_dim axis not found")?;
    let head = shape
        .iter()
        .enumerate()
        .find_map(|(idx, &dim)| {
            (idx != head_dim && dim == config.num_kv_heads as i64).then_some(idx)
        })
        .context("KV tensor head axis not found")?;
    let sequence = shape
        .iter()
        .enumerate()
        .find_map(|(idx, &dim)| {
            (idx != head && idx != head_dim && dim as usize > token_pos).then_some(idx)
        })
        .context("KV tensor sequence axis not found")?;
    Ok(KvTensorAxes {
        head,
        sequence,
        head_dim,
    })
}

fn row_major_strides(shape: &[i64]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for idx in (0..shape.len().saturating_sub(1)).rev() {
        strides[idx] = strides[idx + 1] * shape[idx + 1] as usize;
    }
    strides
}

fn kv_layer_index(name: &str) -> Option<usize> {
    name.split(|ch: char| !ch.is_ascii_digit())
        .find(|part| !part.is_empty())
        .and_then(|part| part.parse().ok())
}

pub(crate) fn extract_next_token_logits(
    session: &Session,
    outputs: Vec<Value>,
) -> anyhow::Result<Vec<f32>> {
    let logits_index = session
        .output_names()
        .iter()
        .position(|name| name == "logits")
        .or_else(|| {
            session
                .output_names()
                .iter()
                .position(|name| name.to_ascii_lowercase().contains("logits"))
        })
        .context("model did not expose a logits output")?;
    let logits = outputs
        .get(logits_index)
        .context("logits output index was out of range")?;
    let shape = logits.shape();
    let data = logits
        .to_vec_f32()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {}", e))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(data),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            let start = (*seq as usize - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            let start = (*seq as usize - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {:?}", other),
    }
}

fn extract_logits_sequence(
    session: &Session,
    outputs: Vec<Value>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let logits_index = session
        .output_names()
        .iter()
        .position(|name| name == "logits")
        .or_else(|| {
            session
                .output_names()
                .iter()
                .position(|name| name.to_ascii_lowercase().contains("logits"))
        })
        .context("model did not expose a logits output")?;
    let logits = outputs
        .get(logits_index)
        .context("logits output index was out of range")?;
    let shape = logits.shape();
    let data = logits
        .to_vec_f32()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {}", e))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(vec![data]),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {:?}", other),
    }
}

fn extract_logits_value_next(logits: &Value) -> anyhow::Result<Vec<f32>> {
    let sequence = extract_logits_value_sequence(logits)?;
    sequence
        .into_iter()
        .last()
        .context("logits tensor did not contain any sequence rows")
}

fn extract_logits_value_sequence(logits: &Value) -> anyhow::Result<Vec<Vec<f32>>> {
    let shape = logits.shape();
    let data = logits
        .to_vec_f32()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {}", e))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(vec![data]),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {:?}", other),
    }
}

fn ensure_i64(info: &TensorInfo) -> anyhow::Result<()> {
    if info.dtype != DataType::Int64 {
        anyhow::bail!("input '{}' must be Int64, got {:?}", info.name, info.dtype);
    }
    Ok(())
}

fn empty_past_value(info: &TensorInfo) -> anyhow::Result<Value> {
    if info.dtype != DataType::Float32 {
        anyhow::bail!(
            "KV input '{}' must be Float32 for Phase 1, got {:?}",
            info.name,
            info.dtype
        );
    }
    if info.shape.len() < 3 {
        anyhow::bail!(
            "KV input '{}' has unsupported shape {:?}",
            info.name,
            info.shape
        );
    }
    let seq_axis = info.shape.len() - 2;
    let mut shape = Vec::with_capacity(info.shape.len());
    for (axis, &dim) in info.shape.iter().enumerate() {
        let value = if axis == 0 {
            1
        } else if axis == seq_axis {
            0
        } else if dim > 0 {
            dim
        } else {
            anyhow::bail!(
                "cannot infer static dimension {} for empty KV input '{}' shape {:?}",
                axis,
                info.name,
                info.shape
            );
        };
        shape.push(value);
    }
    Value::from_slice_f32(&[], &shape)
        .map_err(|e| anyhow::anyhow!("Failed to create empty KV input '{}': {}", info.name, e))
}

pub(crate) fn clone_value(value: &Value) -> anyhow::Result<Value> {
    match value.dtype() {
        DataType::Float32 => Value::from_slice_f32(&value.to_vec_f32()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Float32 ORT value: {}", e)),
        DataType::Int64 => Value::from_slice_i64(&value.to_vec_i64()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Int64 ORT value: {}", e)),
        dtype => anyhow::bail!("unsupported cached ORT value dtype: {:?}", dtype),
    }
}

fn is_kv_input(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("past") && (lower.contains("key") || lower.contains("value"))
}

fn is_present_output(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("present") && (lower.contains("key") || lower.contains("value"))
}

fn matching_past_input<'a>(present_name: &str, inputs: &'a [String]) -> Option<&'a String> {
    let present_suffix = kv_suffix(present_name)?;
    inputs
        .iter()
        .find(|input| kv_suffix(input).as_deref() == Some(present_suffix.as_str()))
}

fn kv_suffix(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    for prefix in [
        "past_key_values.",
        "present_key_values.",
        "past.",
        "present.",
    ] {
        if let Some(suffix) = lower.strip_prefix(prefix) {
            return Some(suffix.to_string());
        }
    }
    None
}

fn reached_context_limit(current_context_len: usize, max_context: Option<usize>) -> bool {
    max_context.is_some_and(|limit| current_context_len >= limit)
}

fn exceeded_context_limit(current_context_len: usize, max_context: Option<usize>) -> bool {
    max_context.is_some_and(|limit| current_context_len > limit)
}

fn is_gather_out_of_bounds(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("gather")
        && (lower.contains("indices element out of data bounds")
            || lower.contains("idx=") && lower.contains("out of"))
}

pub(crate) fn build_processor_chain(
    options: &GenerateOptions,
    tokenizer: Option<&Tokenizer>,
) -> anyhow::Result<ProcessorChain> {
    let mut chain = ProcessorChain::new();

    if options.repetition_penalty != 1.0 {
        chain.add(Box::new(RepetitionPenaltyProcessor {
            penalty: options.repetition_penalty,
        }));
    }

    if options.frequency_penalty != 0.0 {
        chain.add(Box::new(FrequencyPenaltyProcessor {
            frequency_penalty: options.frequency_penalty,
        }));
    }

    if options.presence_penalty != 0.0 {
        chain.add(Box::new(PresencePenaltyProcessor {
            presence_penalty: options.presence_penalty,
        }));
    }

    if !options.stop_sequences.is_empty() {
        chain.add(Box::new(StopSequenceProcessor::new(
            options.stop_sequences.clone(),
        )));
    }

    if let Some(constraint) = &options.constraint {
        let tokenizer = tokenizer.context("constrained decoding requires a tokenizer")?;
        let token_texts = tokenizer_token_texts(tokenizer);
        match constraint {
            GenerateConstraint::Json => {
                chain.add(Box::new(ConstraintProcessor::new(
                    Box::new(JsonConstraint),
                    token_texts,
                    options.eos_token_id,
                )));
            }
            GenerateConstraint::JsonSchema(schema) => {
                chain.add(Box::new(ConstraintProcessor::new(
                    build_llguidance_constraint(
                        GrammarConstraintKind::JsonSchema,
                        schema,
                        tokenizer,
                        &token_texts,
                        options.eos_token_id,
                    )?,
                    token_texts,
                    options.eos_token_id,
                )));
            }
            GenerateConstraint::Regex(regex) => {
                chain.add(Box::new(ConstraintProcessor::new(
                    build_llguidance_constraint(
                        GrammarConstraintKind::Regex,
                        regex,
                        tokenizer,
                        &token_texts,
                        options.eos_token_id,
                    )?,
                    token_texts,
                    options.eos_token_id,
                )));
            }
            GenerateConstraint::Lark(grammar) => {
                chain.add(Box::new(ConstraintProcessor::new(
                    build_llguidance_constraint(
                        GrammarConstraintKind::Lark,
                        grammar,
                        tokenizer,
                        &token_texts,
                        options.eos_token_id,
                    )?,
                    token_texts,
                    options.eos_token_id,
                )));
            }
        }
    }

    if options.temperature > 0.0 && options.temperature != 1.0 {
        chain.add(Box::new(TemperatureProcessor {
            temperature: options.temperature,
        }));
    }

    if options.top_k > 0 {
        chain.add(Box::new(TopKProcessor {
            top_k: options.top_k,
        }));
    }

    if options.top_p < 1.0 {
        chain.add(Box::new(TopPProcessor {
            top_p: options.top_p,
        }));
    }

    if options.min_p > 0.0 {
        chain.add(Box::new(MinPProcessor {
            min_p: options.min_p,
        }));
    }

    Ok(chain)
}

fn build_llguidance_constraint(
    kind: GrammarConstraintKind,
    grammar: &str,
    tokenizer: &Tokenizer,
    token_texts: &[Option<String>],
    eos_token_id: Option<TokenId>,
) -> anyhow::Result<Box<dyn crate::logits::Constraint>> {
    match LlguidanceConstraint::from_hf_tokenizer(
        kind,
        grammar,
        tokenizer.inner(),
        token_texts.len(),
        eos_token_id,
    ) {
        Ok(constraint) => Ok(Box::new(constraint)),
        Err(hf_error) => LlguidanceConstraint::from_token_texts(
            kind,
            grammar,
            token_texts,
            eos_token_id,
        )
        .map(|constraint| Box::new(constraint) as Box<dyn crate::logits::Constraint>)
        .with_context(|| {
            format!(
                "failed to initialize llguidance with HuggingFace tokenizer ({hf_error}) or decoded-token fallback"
            )
        }),
    }
}

pub(crate) fn ensure_constrained_finish(
    options: &GenerateOptions,
    generated_text: &str,
    finish_reason: FinishReason,
) -> anyhow::Result<()> {
    if matches!(
        (&options.constraint, finish_reason),
        (
            Some(GenerateConstraint::Json),
            FinishReason::MaxTokens | FinishReason::Length
        )
    ) && !JsonConstraint::is_complete(generated_text)
    {
        anyhow::bail!(
            "JSON constrained decoding stopped before a complete JSON value; increase max_new_tokens or max_context"
        );
    }
    Ok(())
}

fn tokenizer_token_texts(tokenizer: &Tokenizer) -> Vec<Option<String>> {
    let vocab = tokenizer.inner().get_vocab(true);
    let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
    let mut token_texts = vec![None; max_id + 1];
    for id in 0..=max_id {
        token_texts[id] = tokenizer.decode(&[id as TokenId]).ok();
    }
    token_texts
}

pub(crate) fn select_next_token(
    logits: &mut [f32],
    context: &ProcessorContext,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    rng_value: f32,
) -> TokenId {
    chain.process(logits, context);
    if options.greedy || options.temperature == 0.0 {
        sample_greedy(logits)
    } else {
        sample_categorical(logits, rng_value)
    }
}

pub(crate) fn finish_reason_after_token(
    token_id: TokenId,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    context: &ProcessorContext,
) -> Option<FinishReason> {
    if options.stop_on_eos && options.eos_token_id == Some(token_id) {
        return Some(FinishReason::EosToken);
    }

    match chain.signal(context) {
        Some(ProcessorSignal::StopSequence { index })
            if !matches!(&options.constraint, Some(GenerateConstraint::Json))
                || JsonConstraint::is_complete(&context.generated_text) =>
        {
            Some(FinishReason::StopSequence { index })
        }
        Some(ProcessorSignal::StopSequence { .. }) => None,
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
