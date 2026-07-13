//! Public generation API types and configuration.

use crate::logits::{StopSequence, TokenId};
use onnx_genai_kv::{KvDType, SequenceId};
use onnx_genai_ort::{Eagle3DraftKvMode, MtpDraftKvMode};
use onnx_genai_scheduler::{Priority, SchedulerConfig};
use std::path::PathBuf;

/// Files and target-model outputs required for multi-token prediction.
///
/// The target decoder must emit both logits and the configured last-layer
/// hidden-state output on every forward. The embedding and LM-head files must
/// contain the exact target weights as little-endian f32 matrices; mismatched
/// weights remain greedy-correct because every candidate is target-verified,
/// but will reduce acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpConfig {
    /// ONNX model containing the MTP head.
    pub head_model: PathBuf,
    /// Target decoder output containing `[batch, sequence, hidden]` states.
    pub target_hidden_output: String,
    /// Raw little-endian f32 target embedding weights in `[vocab, hidden]` order.
    pub embedding_weights: PathBuf,
    /// Raw little-endian f32 target LM-head weights in `[hidden, vocab]` order.
    pub lm_head_weights: PathBuf,
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Target hidden size.
    pub hidden_size: usize,
    /// MTP-head cache strategy.
    pub kv_mode: MtpDraftKvMode,
    /// Number of speculative tokens produced after the guaranteed target token.
    pub num_speculative_tokens: usize,
}

/// Files and target-model outputs required for EAGLE-3 speculation.
///
/// EAGLE-3 consumes exactly three target hidden-state outputs (low, middle,
/// high), concatenates their last-token rows, and autoregressively recycles the
/// draft head's own hidden output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3Config {
    /// ONNX model containing the EAGLE-3 draft head.
    pub head_model: PathBuf,
    /// Low, middle, and high target hidden-state output names, in that order.
    pub target_hidden_outputs: Vec<String>,
    /// Raw little-endian f32 target embedding weights in `[vocab, hidden]` order.
    pub embedding_weights: PathBuf,
    /// Target vocabulary size used by the shared embedding table.
    pub vocab_size: usize,
    /// Width of each target hidden state and token embedding.
    pub hidden_size: usize,
    /// EAGLE-3 head cache strategy.
    pub kv_mode: Eagle3DraftKvMode,
    /// Number of speculative tokens produced after the guaranteed target token.
    pub num_speculative_tokens: usize,
}

/// Built-in speculative candidate source.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SpeculativeMode {
    /// Disable speculative decoding.
    #[default]
    None,
    /// Propose tokens with the configured draft model.
    DraftModel,
    /// Copy continuations from the most recent matching context n-gram.
    PromptLookup {
        /// Number of trailing context tokens used as the lookup key.
        ngram: usize,
        /// Maximum copied continuation length per verification step.
        max_tokens: usize,
    },
    /// Propose from a target hidden state with an external MTP head.
    Mtp(MtpConfig),
    /// Propose autoregressively from fused low/middle/high target hidden states.
    Eagle3(Eagle3Config),
}

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
    /// Default speculative source. For compatibility, a configured
    /// `draft_model` selects `DraftModel` when this remains `None`.
    pub speculative_mode: SpeculativeMode,
    /// Storage dtype for the host-side paged KV cache mirror.
    ///
    /// Controls how KV tensors are stored in the paged cache after being
    /// written from model outputs. The model's own I/O dtype (Float32 /
    /// Float16) is independent of this setting; the cache quantises/
    /// dequantises internally.  Defaults to `KvDType::F32` (no quantisation).
    pub kv_cache_dtype: KvDType,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_gpu_pages: 1024,
            page_size: 16,
            scheduler: SchedulerConfig::default(),
            draft_model: None,
            num_speculative_tokens: 4,
            speculative_mode: SpeculativeMode::None,
            kv_cache_dtype: KvDType::F32,
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
    /// Optional seed for reproducible categorical sampling.
    pub seed: Option<u64>,
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
    /// Optional per-request speculative mode override.
    pub speculative_mode: Option<SpeculativeMode>,
    /// Optional constrained decoding grammar. None preserves unconstrained generation.
    pub constraint: Option<GenerateConstraint>,
    /// Return per-token log probabilities and this many highest-probability alternatives.
    ///
    /// Values are computed from the final post-processor distribution used for sampling.
    /// The chosen token is always included in `TokenLogprob::top`, in addition to the
    /// requested alternatives when it is not already among them.
    pub top_logprobs: Option<usize>,
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
            seed: None,
            stop_sequences: Vec::new(),
            eos_token_id: None,
            stop_on_eos: true,
            max_context: None,
            num_speculative_tokens: None,
            speculative_mode: None,
            constraint: None,
            top_logprobs: None,
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
        if let Some(SpeculativeMode::PromptLookup { ngram, max_tokens }) = &self.speculative_mode {
            if *ngram == 0 {
                anyhow::bail!("prompt-lookup ngram must be greater than zero");
            }
            if *max_tokens == 0 {
                anyhow::bail!("prompt-lookup max_tokens must be greater than zero");
            }
        }
        if let Some(SpeculativeMode::Mtp(config)) = &self.speculative_mode {
            validate_mtp_config(config)?;
        }
        if let Some(SpeculativeMode::Eagle3(config)) = &self.speculative_mode {
            validate_eagle3_config(config)?;
        }
        Ok(())
    }
}

pub(crate) fn validate_mtp_config(config: &MtpConfig) -> anyhow::Result<()> {
    if config.target_hidden_output.is_empty() {
        anyhow::bail!("MTP target_hidden_output must not be empty");
    }
    if config.vocab_size == 0 || config.hidden_size == 0 {
        anyhow::bail!("MTP vocab_size and hidden_size must be greater than zero");
    }
    if config.num_speculative_tokens == 0 {
        anyhow::bail!("MTP num_speculative_tokens must be greater than zero");
    }
    Ok(())
}

pub(crate) fn validate_eagle3_config(config: &Eagle3Config) -> anyhow::Result<()> {
    if config.target_hidden_outputs.len() != 3
        || config
            .target_hidden_outputs
            .iter()
            .any(|name| name.is_empty())
    {
        anyhow::bail!(
            "EAGLE-3 target_hidden_outputs must contain exactly three non-empty low/middle/high output names"
        );
    }
    if config.vocab_size == 0 || config.hidden_size == 0 {
        anyhow::bail!("EAGLE-3 vocab_size and hidden_size must be greater than zero");
    }
    if config.num_speculative_tokens == 0 {
        anyhow::bail!("EAGLE-3 num_speculative_tokens must be greater than zero");
    }
    Ok(())
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateResult {
    /// Detokenized generated text.
    pub text: String,
    /// Generated token ids, excluding prompt tokens.
    pub token_ids: Vec<TokenId>,
    /// Termination reason.
    pub finish_reason: FinishReason,
    /// Number of prompt/context tokens whose KV state was reused from the prefix cache.
    pub prefix_cache_hit_len: usize,
    /// Per-generated-token log probabilities, or `None` when not requested.
    pub logprobs: Option<Vec<TokenLogprob>>,
}

/// Log-probability metadata for one generated token.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    /// The selected token id.
    pub token_id: TokenId,
    /// Natural-log probability of the selected token.
    pub logprob: f32,
    /// Highest-probability tokens and their natural-log probabilities, sorted descending.
    pub top: Vec<(TokenId, f32)>,
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
