//! Public generation API types and configuration.

use crate::logits::{StopSequence, TokenId};
use onnx_genai_kv::{CachePriority, DEFAULT_CHUNK_SIZE, KvDType, LocalTieredConfig, SequenceId};
use onnx_genai_ort::{Eagle3DraftKvMode, MtpDraftKvMode};
use onnx_genai_scheduler::{Priority, ResourceLimit, ResourceLimits, SchedulerConfig};
use serde::Deserialize;
use std::path::PathBuf;

#[cfg(feature = "native-backend")]
use crate::native_decode::NativeDecodeDevice;

/// Error returned when a user-facing resource limit cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "invalid resource limit {input:?}: {reason}; use a byte count (for example 8589934592), \
     a binary/decimal byte string (for example 8GiB or 8GB), a fraction in [0, 1] \
     (for example 0.9), or \"auto\""
)]
pub struct LimitParseError {
    input: String,
    reason: String,
}

/// Parse a user-facing resource ceiling.
///
/// Integers without a suffix are bytes. Decimal values without a suffix are
/// fractions, while suffixed values may be integral or decimal byte quantities.
pub fn parse_resource_limit(input: &str) -> Result<ResourceLimit, LimitParseError> {
    let input = input.trim();
    if input.eq_ignore_ascii_case("auto") {
        return Ok(ResourceLimit::Auto);
    }
    if let Ok(bytes) = input.parse::<u64>() {
        return Ok(ResourceLimit::Bytes(bytes));
    }

    let unit_start = input
        .find(|character: char| character.is_ascii_alphabetic())
        .unwrap_or(input.len());
    let (number, unit) = input.split_at(unit_start);
    if unit.is_empty() {
        let fraction = number.parse::<f64>().map_err(|_| {
            limit_error(
                input,
                "the value is neither an integer byte count nor a numeric fraction",
            )
        })?;
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(limit_error(
                input,
                "a unitless decimal is a fraction, but this value is outside [0, 1]",
            ));
        }
        return Ok(ResourceLimit::Fraction(fraction as f32));
    }

    let multiplier = match unit.to_ascii_uppercase().as_str() {
        "KIB" => 1_u64 << 10,
        "MIB" => 1_u64 << 20,
        "GIB" => 1_u64 << 30,
        "KB" => 1_000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        _ => {
            return Err(limit_error(
                input,
                format!("the unit {unit:?} is not supported"),
            ));
        }
    };
    if number.chars().all(|character| character.is_ascii_digit()) {
        let quantity = number.parse::<u64>().map_err(|_| {
            limit_error(
                input,
                format!("the integral byte quantity {number:?} does not fit in u64"),
            )
        })?;
        let bytes = quantity.checked_mul(multiplier).ok_or_else(|| {
            limit_error(
                input,
                format!(
                    "multiplying {quantity} by the {unit} unit size ({multiplier} bytes) \
                     overflows u64; use a smaller byte quantity or a smaller unit"
                ),
            )
        })?;
        return Ok(ResourceLimit::Bytes(bytes));
    }
    let quantity = number.parse::<f64>().map_err(|_| {
        limit_error(
            input,
            format!("the numeric part {number:?} is not a valid non-negative number"),
        )
    })?;
    let bytes = quantity * multiplier as f64;
    if !quantity.is_finite() || quantity < 0.0 || !bytes.is_finite() || bytes >= u64::MAX as f64 {
        return Err(limit_error(
            input,
            "the byte quantity is negative, non-finite, or exceeds u64",
        ));
    }
    Ok(ResourceLimit::Bytes(bytes.round() as u64))
}

fn limit_error(input: impl Into<String>, reason: impl Into<String>) -> LimitParseError {
    LimitParseError {
        input: input.into(),
        reason: reason.into(),
    }
}

/// Error returned while decoding the resource-governor YAML surface.
#[derive(Debug, thiserror::Error)]
pub enum EngineConfigError {
    #[error("failed to parse engine YAML resource limits: {0}; check serving.memory.limits syntax")]
    Yaml(#[from] serde_yaml::Error),
    #[error("failed to parse serving.memory.limits.{field}: {source}")]
    Limit {
        field: &'static str,
        #[source]
        source: LimitParseError,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LimitValue {
    String(String),
    Integer(u64),
    Float(f64),
}

impl LimitValue {
    fn parse(self, field: &'static str) -> Result<ResourceLimit, EngineConfigError> {
        let parsed = match self {
            Self::String(value) => parse_resource_limit(&value),
            Self::Integer(value) => Ok(ResourceLimit::Bytes(value)),
            Self::Float(value) if value.is_finite() && (0.0..=1.0).contains(&value) => {
                Ok(ResourceLimit::Fraction(value as f32))
            }
            Self::Float(value) => Err(limit_error(
                value.to_string(),
                "a YAML floating-point limit is a fraction, but this value is outside [0, 1]",
            )),
        };
        parsed.map_err(|source| EngineConfigError::Limit { field, source })
    }
}

#[derive(Debug, Default, Deserialize)]
struct LimitsYaml {
    vram_limit: Option<LimitValue>,
    host_ram_limit: Option<LimitValue>,
    disk_spill_limit: Option<LimitValue>,
    #[serde(default)]
    allow_runtime_override: bool,
}

#[derive(Debug, Default, Deserialize)]
struct MemoryYaml {
    #[serde(default)]
    limits: LimitsYaml,
}

#[derive(Debug, Default, Deserialize)]
struct ServingYaml {
    #[serde(default)]
    memory: MemoryYaml,
}

#[derive(Debug, Default, Deserialize)]
struct EngineConfigYaml {
    #[serde(default)]
    serving: ServingYaml,
}

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

/// Files and target-model outputs required for shared-KV draft speculation
/// (originally introduced for Gemma4 `*-assistant` draft models).
///
/// The proposer is a shared-KV draft: it owns no KV cache and instead reads
/// slices of the target model's paged KV cache. It carries its own internal
/// `lm_head`, but it does *not* own an input embedding table: each step builds
/// `inputs_embeds = concat(target_input_embedding(last_token), hidden)`, so the
/// engine supplies the target's raw input-token embedding via
/// [`SharedKvProposerConfig::input_embedding_weights`]. The first step seeds
/// `hidden` from the target's last hidden state and `last_token` from the last
/// context token; every later step threads the proposer's own `projected_state`
/// and the previously drafted token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedKvProposerConfig {
    /// ONNX model containing the shared-KV proposer graph.
    pub assistant_model: PathBuf,
    /// Target decoder output containing `[batch, sequence, hidden]` states,
    /// used to seed the first assistant step. Must be Float32.
    pub target_hidden_output: String,
    /// Raw little-endian f32 target input-token embedding weights in
    /// `[vocab_size, backbone_hidden_size]` order, used to build the token-
    /// embedding half of each step's `inputs_embeds`.
    pub input_embedding_weights: PathBuf,
    /// Target backbone hidden size `H`.
    pub backbone_hidden_size: usize,
    /// Vocabulary size of the assistant's own `logits` output.
    pub vocab_size: usize,
    /// Number of speculative tokens produced after the guaranteed target token.
    pub num_speculative_tokens: usize,
    /// Shared-KV binding groups mapping assistant `shared_kv.<name>` inputs to
    /// target KV layer indices.
    pub shared_kv: Vec<SharedKvBinding>,
}

/// One shared-KV binding for [`SharedKvProposerConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedKvBinding {
    /// Assistant input group name, e.g. `sliding_attention` / `full_attention`.
    pub name: String,
    /// Target KV layer indices whose cache feeds this shared-KV slice.
    pub target_layers: Vec<usize>,
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
    /// Propose with a shared-KV draft proposer that reads target KV slices.
    SharedKv(SharedKvProposerConfig),
}

/// Identifier for a persistent generation session.
pub type SessionId = SequenceId;

/// Distributed KV connector backend selection (DESIGN §38, K3).
///
/// Model-agnostic by construction: a backend carries only its own generic
/// settings, never per-model branches. `Null` is the default and reproduces the
/// engine's in-process-only prefix reuse exactly.
#[derive(Debug, Clone, Default)]
pub enum KvConnectorBackend {
    /// No external connector: KV lives only in the local paged cache.
    #[default]
    Null,
    /// Single-node tiered (GPU→CPU, optional disk) connector.
    LocalTiered(LocalTieredConfig),
}

/// Generic configuration for wiring a [`KvCacheConnector`](onnx_genai_kv::KvCacheConnector)
/// into the engine (DESIGN §38, K3).
///
/// Every field is a backend-neutral parameter. `model_id` only namespaces cache
/// keys (opaque; never interpreted); when `None` the engine derives a stable id
/// from the model directory.
#[derive(Debug, Clone)]
pub struct KvConnectorConfig {
    /// Which connector backend to use. Defaults to [`KvConnectorBackend::Null`].
    pub backend: KvConnectorBackend,
    /// Opaque model identity used to namespace cache keys. `None` => derived
    /// from the model directory name.
    pub model_id: Option<String>,
    /// Tokens per cached chunk for keying. `0` => [`DEFAULT_CHUNK_SIZE`].
    pub chunk_size: usize,
    /// Priority applied to chunks stored to the connector.
    pub store_priority: CachePriority,
    /// Estimated prefill recompute cost per token (ms), used as the
    /// fetch-vs-recompute baseline against a location's `estimated_load_ms`.
    pub recompute_ms_per_token: f64,
}

impl Default for KvConnectorConfig {
    fn default() -> Self {
        Self {
            backend: KvConnectorBackend::Null,
            model_id: None,
            chunk_size: DEFAULT_CHUNK_SIZE,
            store_priority: CachePriority::Session,
            recompute_ms_per_token: 0.05,
        }
    }
}

/// Model-execution backend selected for decoder generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EngineDecodeBackend {
    /// Use the native runtime for models containing native-only operators;
    /// otherwise use ONNX Runtime.
    #[default]
    Auto,
    /// Always use ONNX Runtime.
    Ort,
    /// Always use the native runtime.
    Native,
}

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Decoder execution backend. [`EngineDecodeBackend::Auto`] preserves ORT
    /// for existing models and selects native execution only when required.
    pub decode_backend: EngineDecodeBackend,
    /// Native decoder device override. `None` follows the execution provider in
    /// [`onnx_genai_ort::SessionOptions`], including `ONNX_GENAI_EP`.
    #[cfg(feature = "native-backend")]
    pub native_device: Option<NativeDecodeDevice>,
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
    /// Optional distributed KV connector (DESIGN §38). Defaults to
    /// [`KvConnectorBackend::Null`], which preserves in-process-only behavior.
    pub kv_connector: KvConnectorConfig,
    /// Vendor-neutral hot, warm, and cold resource ceilings (DESIGN §26.11).
    pub limits: ResourceLimits,
    /// Permit programmatic resource-limit changes after engine initialization.
    pub allow_runtime_override: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            decode_backend: EngineDecodeBackend::Auto,
            #[cfg(feature = "native-backend")]
            native_device: None,
            num_gpu_pages: 1024,
            page_size: 16,
            scheduler: SchedulerConfig::default(),
            draft_model: None,
            num_speculative_tokens: 4,
            speculative_mode: SpeculativeMode::None,
            kv_cache_dtype: KvDType::F32,
            kv_connector: KvConnectorConfig::default(),
            limits: ResourceLimits::default(),
            allow_runtime_override: false,
        }
    }
}

impl EngineConfig {
    /// Decode the `serving.memory.limits` YAML surface documented in §26.11.4.
    ///
    /// Engine settings outside that block retain their programmatic defaults.
    pub fn from_yaml(yaml: &str) -> Result<Self, EngineConfigError> {
        let document: EngineConfigYaml = serde_yaml::from_str(yaml)?;
        let yaml_limits = document.serving.memory.limits;
        let mut config = Self::default();
        if let Some(limit) = yaml_limits.vram_limit {
            config.limits.vram_limit = limit.parse("vram_limit")?;
        }
        if let Some(limit) = yaml_limits.host_ram_limit {
            config.limits.host_ram_limit = limit.parse("host_ram_limit")?;
        }
        if let Some(limit) = yaml_limits.disk_spill_limit {
            config.limits.disk_spill_limit = Some(limit.parse("disk_spill_limit")?);
        }
        config.allow_runtime_override = yaml_limits.allow_runtime_override;
        Ok(config)
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

#[cfg(test)]
mod resource_limit_tests {
    use super::*;

    #[test]
    fn parses_integer_bytes_and_all_supported_units() {
        let cases = [
            ("42", 42),
            ("2KiB", 2 * 1024),
            ("2MiB", 2 * 1024 * 1024),
            ("2GiB", 2 * 1024 * 1024 * 1024),
            ("2KB", 2_000),
            ("2MB", 2_000_000),
            ("2GB", 2_000_000_000),
            ("1.5KiB", 1536),
        ];
        for (input, expected) in cases {
            assert_eq!(
                parse_resource_limit(input).unwrap(),
                ResourceLimit::Bytes(expected),
                "{input}"
            );
        }
    }

    #[test]
    fn parses_fraction_and_case_insensitive_auto() {
        assert_eq!(
            parse_resource_limit("0.5").unwrap(),
            ResourceLimit::Fraction(0.5)
        );
        assert_eq!(
            parse_resource_limit("1.0").unwrap(),
            ResourceLimit::Fraction(1.0)
        );
        assert_eq!(parse_resource_limit("AuTo").unwrap(), ResourceLimit::Auto);
    }

    #[test]
    fn rejects_out_of_range_fractions_unknown_units_and_invalid_numbers() {
        for input in ["1.01", "-0.1", "NaN", "inf"] {
            let error = parse_resource_limit(input).unwrap_err().to_string();
            assert!(error.contains("invalid resource limit"), "{error}");
            assert!(error.contains("use a byte count"), "{error}");
        }
        for input in ["8TiB", "8G", "8Gi", "8XB"] {
            let error = parse_resource_limit(input).unwrap_err().to_string();
            assert!(error.contains("not supported"), "{input}: {error}");
            assert!(error.contains("8GiB"), "{input}: {error}");
        }
        for input in ["GiB", "oneGiB", "-1GiB", "1e100GiB"] {
            let error = parse_resource_limit(input).unwrap_err().to_string();
            assert!(error.contains("invalid resource limit"), "{input}: {error}");
        }
    }

    #[test]
    fn rejects_integral_unit_overflow_at_exact_boundary() {
        let error = parse_resource_limit("17179869184GiB")
            .unwrap_err()
            .to_string();
        assert!(error.contains("overflows u64"), "{error}");
        assert!(error.contains("use a smaller byte quantity"), "{error}");
    }

    #[test]
    fn engine_config_defaults_to_scheduler_resource_defaults() {
        let config = EngineConfig::default();
        assert_eq!(config.decode_backend, EngineDecodeBackend::Auto);
        assert_eq!(config.limits, ResourceLimits::default());
        assert!(!config.allow_runtime_override);
    }

    #[test]
    fn yaml_limits_parse_fraction_bytes_auto_null_and_override() {
        let config = EngineConfig::from_yaml(
            r#"
    serving:
      memory:
        limits:
          vram_limit: "0.5"
          host_ram_limit: "8GiB"
          disk_spill_limit: "auto"
          allow_runtime_override: true
    "#,
        )
        .unwrap();
        assert_eq!(config.limits.vram_limit, ResourceLimit::Fraction(0.5));
        assert_eq!(
            config.limits.host_ram_limit,
            ResourceLimit::Bytes(8_u64 << 30)
        );
        assert_eq!(config.limits.disk_spill_limit, Some(ResourceLimit::Auto));
        assert!(config.allow_runtime_override);

        let disabled = EngineConfig::from_yaml(
            "serving:\n  memory:\n    limits:\n      disk_spill_limit: null\n",
        )
        .unwrap();
        assert_eq!(disabled.limits.disk_spill_limit, None);
    }

    #[test]
    fn yaml_accepts_numeric_fraction_and_reports_field_context() {
        for (value, expected) in [("1.0", 1.0), ("0.5", 0.5)] {
            let config = EngineConfig::from_yaml(&format!(
                "serving:\n  memory:\n    limits:\n      vram_limit: {value}\n"
            ))
            .unwrap();
            assert_eq!(config.limits.vram_limit, ResourceLimit::Fraction(expected));
        }

        let error =
            EngineConfig::from_yaml("serving:\n  memory:\n    limits:\n      vram_limit: 1.5\n")
                .unwrap_err()
                .to_string();
        assert!(error.contains("vram_limit"), "{error}");
        assert!(error.contains("outside [0, 1]"), "{error}");
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
        if let Some(SpeculativeMode::SharedKv(config)) = &self.speculative_mode {
            validate_shared_kv_proposer_config(config)?;
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

pub(crate) fn validate_shared_kv_proposer_config(
    config: &SharedKvProposerConfig,
) -> anyhow::Result<()> {
    if config.target_hidden_output.is_empty() {
        anyhow::bail!("shared-KV proposer target_hidden_output must not be empty");
    }
    if config.backbone_hidden_size == 0 || config.vocab_size == 0 {
        anyhow::bail!(
            "shared-KV proposer backbone_hidden_size and vocab_size must be greater than zero"
        );
    }
    if config.num_speculative_tokens == 0 {
        anyhow::bail!("shared-KV proposer num_speculative_tokens must be greater than zero");
    }
    if config.shared_kv.is_empty() {
        anyhow::bail!("shared-KV proposer requires at least one shared_kv binding group");
    }
    for group in &config.shared_kv {
        if group.name.is_empty() {
            anyhow::bail!("shared-KV proposer shared_kv group name must not be empty");
        }
        if group.target_layers.is_empty() {
            anyhow::bail!(
                "shared-KV proposer shared_kv group '{}' must list at least one target layer",
                group.name
            );
        }
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
