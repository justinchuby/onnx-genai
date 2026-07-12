//! Typed structs for all inference metadata spec sections.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer};

/// Top-level inference metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct InferenceMetadata {
    /// Capabilities that a runtime MUST support to load this model.
    #[serde(default)]
    pub required_capabilities: Vec<String>,

    /// Model build-time properties.
    #[serde(default)]
    pub model: Option<ModelCapabilities>,

    /// KV cache configuration and tolerance.
    #[serde(default)]
    pub kv_cache: Option<KvCacheSpec>,

    /// Weight quantization intent.
    #[serde(default)]
    pub quantization: Option<QuantizationIntent>,

    /// Multi-model pipeline definition.
    #[serde(default)]
    pub pipeline: Option<PipelineSpec>,

    /// Speculative decoding strategy.
    #[serde(default)]
    pub strategy: Option<StrategySpec>,

    /// Speculator model declaration. This is the preferred, native source for
    /// speculator discovery; HuggingFace `config.json` is a compatibility fallback.
    #[serde(default, alias = "speculator_config")]
    pub speculative: Option<SpeculatorConfig>,

    /// Structured output support declaration.
    #[serde(default)]
    pub structured_output: Option<StructuredOutputSpec>,

    /// Hardware requirements for distribution matching.
    #[serde(default)]
    pub hardware_requirements: Option<HardwareRequirements>,
}

/// Configuration published with a standalone speculative proposer model.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SpeculatorConfig {
    /// Proposal architecture used by the speculator.
    #[serde(alias = "method")]
    pub proposal_type: ProposalType,
    /// Maximum number of tokens proposed per verification step.
    #[serde(default = "default_num_speculative_tokens", alias = "tokens_per_step")]
    pub num_speculative_tokens: usize,
    /// Verifier model this speculator was trained against.
    #[serde(default)]
    pub verifier: Option<SpeculatorVerifier>,
}

fn default_num_speculative_tokens() -> usize {
    4
}

/// Verifier identity embedded in a speculator package.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SpeculatorVerifier {
    pub name_or_path: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
}

/// Speculator proposal architecture, preserving unknown future values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposalType {
    Eagle3,
    PEagle,
    Mtp,
    DFlash,
    Unknown(String),
}

impl<'de> Deserialize<'de> for ProposalType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.to_ascii_lowercase().as_str() {
            "eagle" | "eagle3" | "eagle-3" => Self::Eagle3,
            "peagle" | "p-eagle" => Self::PEagle,
            "mtp" => Self::Mtp,
            "dflash" | "d-flash" => Self::DFlash,
            _ => Self::Unknown(value),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCapabilities {
    pub attention: Option<AttentionConfig>,
    pub max_sequence_length: Option<usize>,
    pub speculative: Option<SpeculativeModelInfo>,
    pub runtime_configurable: Option<RuntimeConfigurable>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttentionConfig {
    #[serde(rename = "type")]
    pub attention_type: String,
    pub num_kv_heads: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub sliding_window: Option<usize>,
    /// Fallback behavior for runtimes that don't recognize `attention_type`.
    pub fallback_behavior: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeculativeModelInfo {
    pub has_draft_heads: Option<bool>,
    pub self_speculative_depth: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfigurable {
    pub kv_cache: Option<RuntimeKvConfig>,
    pub prefix_cache: Option<bool>,
    pub continuous_batching: Option<bool>,
    pub chunked_prefill: Option<ChunkedPrefillConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeKvConfig {
    pub dtype: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkedPrefillConfig {
    pub chunk_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KvCacheSpec {
    pub native_dtype: Option<String>,
    pub quantization_tolerance: Option<KvQuantTolerance>,
    pub sensitive_layers: Option<Vec<i32>>,
    pub operations: Option<KvCacheOperations>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KvQuantTolerance {
    pub key: Option<KvComponentTolerance>,
    pub value: Option<KvComponentTolerance>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KvComponentTolerance {
    pub default: Option<String>,
    pub per_layer: Option<Vec<LayerPrecisionOverride>>,
    pub quantization_axis: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerPrecisionOverride {
    pub layers: Vec<i32>,
    pub min_precision: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KvCacheOperations {
    pub rewind_safe: Option<bool>,
    pub fork_precision_policy: Option<String>,
    pub checkpoint_serializable: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationIntent {
    pub default: Option<String>,
    pub overrides: Option<Vec<QuantizationOverride>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationOverride {
    pub layers: Option<Vec<i32>>,
    pub component: Option<String>,
    pub precision: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PipelineSpec {
    /// Named model components that participate in the pipeline DAG.
    pub models: BTreeMap<String, PipelineComponentSpec>,
    /// Directed tensor/data edges between component ports.
    #[serde(default)]
    pub dataflow: Vec<DataflowEdge>,
    /// Loop/execution strategy for the pipeline.
    pub strategy: PipelineStrategy,
    /// Optional per-component phase gating.
    #[serde(default)]
    pub phases: BTreeMap<String, PhaseConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PipelineComponentSpec {
    /// ONNX filename relative to the model directory root.
    pub filename: String,
    /// Component role, for example encoder, decoder, draft, denoiser, or vocoder.
    #[serde(rename = "type")]
    pub role: String,
    /// Optional execution/device preference declared by the model package.
    pub device_preference: Option<String>,
    /// Optional tokenizer filename relative to the model directory root. If absent,
    /// loaders may use a shared top-level tokenizer.json when present.
    pub tokenizer: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataflowEdge {
    /// Source endpoint as `component.output_name`.
    pub from: String,
    /// Destination endpoint as `component.input_name`.
    pub to: String,
    pub dtype: Option<String>,
    pub device_transfer: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PhaseConfig {
    pub run_on: PhaseRunOn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseRunOn {
    PromptOnly,
    EveryStep,
    FinalOnly,
    OnDemand,
    Other(String),
}

impl<'de> Deserialize<'de> for PhaseRunOn {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "prompt_only" => Self::PromptOnly,
            "every_step" | "always" => Self::EveryStep,
            "final_only" => Self::FinalOnly,
            "on_demand" => Self::OnDemand,
            _ => Self::Other(value),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PipelineStrategy {
    pub kind: PipelineStrategyKind,

    /// Autoregressive decoder component name.
    pub decoder: Option<String>,
    pub max_tokens: Option<usize>,
    pub stop_conditions: Option<Vec<serde_json::Value>>,
    pub kv_cache: Option<serde_json::Value>,
    pub speculative: Option<serde_json::Value>,

    /// Single-pass component name.
    pub model: Option<String>,
    pub batching: Option<serde_json::Value>,

    /// Iterative/diffusion component and loop configuration.
    pub denoiser: Option<String>,
    pub scheduler: Option<String>,
    pub num_steps: Option<usize>,
    pub guidance_scale: Option<f32>,
    pub state: Option<serde_json::Value>,

    /// Composite strategy stages.
    #[serde(default)]
    pub stages: Vec<PipelineStrategyStage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PipelineStrategyStage {
    pub name: String,
    pub strategy: Box<PipelineStrategy>,
    pub run_on: Option<PhaseRunOn>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineStrategyKind {
    Autoregressive,
    Iterative,
    SinglePass,
    Composite,
    Other(String),
}

impl<'de> Deserialize<'de> for PipelineStrategyKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "autoregressive" => Self::Autoregressive,
            "iterative" | "diffusion_steps" | "diffusion-steps" => Self::Iterative,
            "single_pass" | "single-pass" => Self::SinglePass,
            "composite" => Self::Composite,
            _ => Self::Other(value),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct StrategySpec {
    pub kind: String,
    pub draft: Option<DraftConfig>,
    pub verify: Option<VerifyConfig>,
    pub acceptance: Option<String>,
    pub tokens_per_step: Option<usize>,
    pub topology: Option<String>,
    pub performance_hints: Option<PerformanceHints>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DraftConfig {
    pub producer: String,
    pub session: Option<String>,
    pub depth: Option<usize>,
    pub heads: Option<String>,
    pub ngram: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VerifyConfig {
    pub method: Option<String>,
    pub session: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PerformanceHints {
    pub expected_acceptance_rate: Option<f32>,
    pub optimal_k: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StructuredOutputSpec {
    pub supported_formats: Option<Vec<String>>,
    pub training_format: Option<String>,
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HardwareRequirements {
    pub min_memory_gb: Option<f32>,
    pub required_dtypes: Option<Vec<String>>,
    pub beneficial_dtypes: Option<Vec<String>>,
    pub kv_cache_memory_per_1k_tokens_mb: Option<f32>,
    pub supports_tensor_parallel: Option<bool>,
    pub min_tp_degree: Option<usize>,
}
