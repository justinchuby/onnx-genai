//! Typed structs for all inference metadata spec sections.

use serde::Deserialize;

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

    /// Structured output support declaration.
    #[serde(default)]
    pub structured_output: Option<StructuredOutputSpec>,

    /// Hardware requirements for distribution matching.
    #[serde(default)]
    pub hardware_requirements: Option<HardwareRequirements>,
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
    pub models: Option<serde_json::Value>,
    pub dataflow: Option<Vec<DataflowEdge>>,
    pub phases: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataflowEdge {
    pub from: String,
    pub to: String,
    pub dtype: Option<String>,
    pub device_transfer: Option<bool>,
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
