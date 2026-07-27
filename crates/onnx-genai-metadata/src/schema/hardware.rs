use super::*;

/// Model-side hardware requirements and distribution-matching hints.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct HardwareRequirements {
    /// Minimum aggregate accelerator or system memory in GiB.
    #[schemars(range(min = 0.0))]
    pub min_memory_gb: Option<f32>,

    /// Dtypes the selected device or execution provider must support.
    #[schemars(with = "Option<Vec<schema_vocabulary::DType>>")]
    pub required_dtypes: Option<Vec<String>>,

    /// Dtypes that improve performance or memory use but are not mandatory.
    #[schemars(with = "Option<Vec<schema_vocabulary::DType>>")]
    pub beneficial_dtypes: Option<Vec<String>>,

    /// Estimated KV-cache memory in MiB per 1,000 cached tokens.
    #[schemars(range(min = 0.0))]
    pub kv_cache_memory_per_1k_tokens_mb: Option<f32>,

    /// Whether the model can be partitioned with tensor parallelism.
    pub supports_tensor_parallel: Option<bool>,

    /// Minimum useful tensor-parallel degree when tensor parallelism is selected.
    #[schemars(range(min = 1))]
    pub min_tp_degree: Option<usize>,
}

/// Explicit sparse mixture-of-experts structure and graph representation.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct MixtureOfExpertsSpec {
    /// Expert graph representation: `dense_fallback`, `moe`, or `qmoe`.
    #[schemars(with = "schema_vocabulary::MoERepresentation")]
    pub representation: String,

    /// Number of independently routed experts.
    #[schemars(range(min = 1))]
    pub routed_expert_count: usize,

    /// Number of dense shared experts evaluated for every token.
    #[schemars(range(min = 0))]
    pub shared_expert_count: usize,

    /// Number of routed experts selected for each token.
    #[schemars(range(min = 1))]
    pub experts_per_token: usize,

    /// Intermediate width of each routed expert FFN.
    #[schemars(range(min = 1))]
    pub expert_intermediate_size: usize,

    /// Total intermediate width of the always-on shared-expert FFN.
    #[schemars(range(min = 0))]
    pub shared_expert_intermediate_size: usize,

    /// Expert FFN activation name, such as `silu`.
    #[schemars(length(min = 1))]
    pub activation: String,

    /// Router scoring, selection, normalization, and scaling semantics.
    pub router: MoERouterSpec,
}

/// Explicit router semantics, kept separate from expert FFN execution.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
#[schemars(transform = schema_helpers::moe_router)]
pub struct MoERouterSpec {
    /// Elementwise score operation applied to router logits.
    #[schemars(with = "schema_vocabulary::MoERouterScoreFunction")]
    pub score_function: String,

    /// Expert selection operation applied to the scores.
    #[schemars(with = "schema_vocabulary::MoERouterSelectionMethod")]
    pub selection_method: String,

    /// Whether selected aggregation weights are normalized to sum to one.
    pub normalize_weights: bool,

    /// Multiplicative scale applied to final aggregation weights.
    #[schemars(range(min = 0.0))]
    pub scaling_factor: f32,

    /// Number of expert groups considered by grouped selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub group_count: Option<usize>,

    /// Number of groups retained per token by grouped selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub groups_per_token: Option<usize>,

    /// Reduction used to score a group before group TopK.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::MoEGroupScore>")]
    pub group_score: Option<String>,
}
