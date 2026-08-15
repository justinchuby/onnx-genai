use super::*;

/// Author-declared text-generation defaults (sampling and beam search).
///
/// Mirrors the `search` section of an onnxruntime-genai `genai_config.json`.
/// Every field is optional so only values the author declared are carried over.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct GenerationDefaults {
    /// Whether to randomize sampling through `top_k`/`top_p` (else greedy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub do_sample: Option<bool>,

    /// Softmax temperature applied before sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Number of highest-probability tokens kept for top-k filtering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Nucleus (top-p) cumulative-probability threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Penalty applied to already-generated tokens (`1.0` = no penalty).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Number of beams for beam search (`1` = no beam search).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_beams: Option<usize>,

    /// Number of sequences returned after search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_return_sequences: Option<usize>,

    /// Minimum final sequence length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_length: Option<usize>,

    /// Maximum final sequence length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,

    /// Exponential length penalty used with beam search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length_penalty: Option<f32>,

    /// Disallow repeating n-grams of this size (`0` = disabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_repeat_ngram_size: Option<usize>,

    /// Diversity penalty for diverse beam groups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diversity_penalty: Option<f32>,

    /// Whether beam search stops once enough beams have finished.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_stopping: Option<bool>,
}

/// Special / control token ids declared by a model author.
///
/// Every field is optional; `eos_token_id` is normalized to a list because
/// onnxruntime-genai accepts either a scalar or an array for it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct SpecialTokens {
    /// Padding token id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_token_id: Option<i64>,

    /// Beginning-of-stream token id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token_id: Option<i64>,

    /// End-of-stream token ids (one or more).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_token_id: Option<Vec<i64>>,

    /// Separator token id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sep_token_id: Option<i64>,

    /// Token an encoder-decoder model starts decoding with, when not `bos`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_start_token_id: Option<i64>,

    /// Image placeholder token id (VLMs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_token_id: Option<i64>,

    /// Video placeholder token id (VLMs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_token_id: Option<i64>,

    /// Vision-segment start token id (VLMs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_start_token_id: Option<i64>,
}

/// Configuration published with a standalone speculative proposer model.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, JsonSchema)]
#[schemars(transform = schema_helpers::speculator_config_aliases)]
pub struct SpeculatorConfig {
    /// Proposal architecture used by the speculator.
    ///
    /// The deprecated `method` alias is accepted on input.
    #[serde(alias = "method")]
    pub proposal_type: ProposalType,

    /// Maximum number of tokens proposed per verifier step; defaults to 4.
    ///
    /// The deprecated `tokens_per_step` alias is accepted on input.
    #[serde(default = "default_num_speculative_tokens", alias = "tokens_per_step")]
    #[schemars(range(min = 1))]
    pub num_speculative_tokens: usize,

    /// Identity of the verifier model against which this proposer was trained.
    #[serde(default)]
    pub verifier: Option<SpeculatorVerifier>,

    /// Relative path (from the model directory) to the proposer ONNX model.
    ///
    /// Used by the `shared_kv` proposer to locate the
    /// proposer graph. Optional for forward compatibility with proposer
    /// families that do not ship a standalone model file.
    #[serde(default)]
    pub model: Option<String>,

    /// Explicit proposer graph execution contract.
    ///
    /// This uses the same architecture-neutral I/O vocabulary as a target
    /// decoder. `sequence_source` selects token ids versus embeddings,
    /// `kv_ownership` selects private past/present state versus references to
    /// target-owned cache, and the output fields assign semantic roles.
    #[serde(default)]
    pub io: Option<ModelIoSpec>,

    /// Target backbone hidden size `H` shared with the proposer.
    ///
    /// For `shared_kv`, `inputs_embeds` is `[B, q, 2*H]` and
    /// `projected_state` is `[B, q, H]`.
    #[serde(default)]
    #[schemars(range(min = 1))]
    pub backbone_hidden_size: Option<usize>,

    /// Vocabulary size of the proposer's own `logits` output.
    #[serde(default)]
    #[schemars(range(min = 1))]
    pub vocab_size: Option<usize>,

    /// Name of the proposer output threaded forward between steps.
    ///
    /// Defaults to `projected_state` for `shared_kv`.
    #[serde(default)]
    pub projected_state_output: Option<String>,

    /// Name of the proposer's draft-distribution output.
    ///
    /// Defaults to `logits` for `shared_kv`.
    #[serde(default)]
    pub logits_output: Option<String>,

    /// Relative path (from the model directory) to the target model's raw
    /// input-token embedding table, as a little-endian f32 matrix in
    /// `[vocab_size, backbone_hidden_size]` order.
    ///
    /// The `shared_kv` proposer builds each step's `inputs_embeds` as
    /// `concat(target_input_embedding(last_token), hidden)`, so it must be able
    /// to look up the target's input embedding of the last drafted/accepted
    /// token. Required for the `shared_kv` proposer.
    #[serde(default)]
    pub input_embedding: Option<String>,

    /// Shared-KV binding groups consumed by the proposer.
    ///
    /// Each group names an assistant input prefix
    /// (`shared_kv.<name>.{key,value}`) and the target KV layer indices whose
    /// cache feeds that slice. Empty for proposers that own their KV cache.
    #[serde(default)]
    pub shared_kv: Vec<SharedKvGroup>,

    /// Target decoder output carrying the recurrent MTP seed.
    ///
    /// Defaults to `hidden_states` for `mtp`.
    #[serde(default)]
    pub target_hidden_output: Option<String>,

    /// Layout of `target_hidden_output`.
    ///
    /// Mobius MTP sidecars use `BSHC`: batch, sequence, Hyper-Connection lane,
    /// hidden.
    #[serde(default)]
    pub target_hidden_layout: Option<MtpHiddenLayout>,

    /// Target hidden width `H`.
    #[serde(default)]
    #[schemars(range(min = 1))]
    pub target_hidden_size: Option<usize>,

    /// Number of Hyper-Connection lanes `C`.
    #[serde(default)]
    #[schemars(range(min = 1))]
    pub hc_mult: Option<usize>,

    /// Sidecar output projected through the shared target LM head.
    ///
    /// Defaults to `mtp_hidden`.
    #[serde(default)]
    pub mtp_hidden_output: Option<String>,

    /// Sidecar recurrent Hyper-Connection state output.
    ///
    /// Defaults to `mtp_state`.
    #[serde(default)]
    pub mtp_state_output: Option<String>,

    /// Lifetime of the sidecar's KV state.
    ///
    /// Defaults to `proposal_local`.
    #[serde(default)]
    pub kv_mode: Option<MtpKvMode>,

    /// Target embedding initializer shared with the MTP sidecar.
    #[serde(default)]
    pub embedding: Option<MtpTargetInitializer>,

    /// Target LM-head initializer shared with the MTP sidecar.
    #[serde(default)]
    pub lm_head: Option<MtpTargetInitializer>,
}

/// Layout of the target state consumed by an MTP sidecar.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, JsonSchema)]
pub enum MtpHiddenLayout {
    /// `[batch, sequence, hidden]` legacy layout.
    #[serde(rename = "BSH")]
    Bsh,
    /// `[batch, sequence, hc_mult, hidden]` Mobius Hyper-Connection layout.
    #[serde(rename = "BSHC")]
    Bshc,
}

/// Lifetime declared for an MTP sidecar's private KV state.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MtpKvMode {
    /// Reset sidecar KV at every target verification iteration.
    ProposalLocal,
    /// Retain only KV corresponding to the accepted draft prefix.
    AcceptedPrefix,
}

/// Exact target-model initializer reference used by an MTP sidecar.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct MtpTargetInitializer {
    /// Initializer ownership source. The Phase-1 contract requires
    /// `target_initializer`.
    pub source: MtpWeightSource,
    /// Exact initializer name in the target ONNX graph.
    pub name: String,
}

/// Ownership source for an MTP shared weight.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MtpWeightSource {
    /// Borrow the named initializer from the target model package.
    TargetInitializer,
}

/// One shared-KV binding group for a shared-KV proposer.
///
/// A `shared_kv` proposer graph exposes `shared_kv.<name>.key` and
/// `shared_kv.<name>.value` inputs bound to slices of the target model's paged
/// KV cache. `target_layers` lists the target KV layer indices feeding this
/// slice.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SharedKvGroup {
    /// Assistant input prefix, e.g. `sliding_attention` or `full_attention`.
    pub name: String,

    /// Target KV layer indices whose cache feeds this shared-KV slice.
    #[serde(default)]
    pub target_layers: Vec<usize>,

    /// Proposer input receiving this group's shared key cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_input: Option<String>,

    /// Proposer input receiving this group's shared value cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_input: Option<String>,

    /// Target decoder past-KV input whose current key cache is referenced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_key_input: Option<String>,

    /// Target decoder past-KV input whose current value cache is referenced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_value_input: Option<String>,
}

fn default_num_speculative_tokens() -> usize {
    4
}

/// Verifier identity embedded in a speculator package.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SpeculatorVerifier {
    /// HuggingFace-style verifier repository name or local model path.
    pub name_or_path: Option<String>,

    /// Verifier architecture class names, in preference order.
    #[serde(default)]
    pub architectures: Vec<String>,
}

/// Speculator proposal architecture.
///
/// Known spellings are enumerated in the generated schema while unknown
/// strings remain valid to preserve forward compatibility.
#[derive(Debug, Clone, PartialEq, Eq, JsonSchema)]
#[schemars(with = "String", transform = schema_helpers::proposal_type)]
pub enum ProposalType {
    /// EAGLE or EAGLE-3 proposer.
    Eagle3,
    /// P-EAGLE proposer.
    PEagle,
    /// Multi-token prediction proposer.
    Mtp,
    /// D-Flash proposer.
    DFlash,
    /// Shared-KV proposer: the draft model shares the target's KV cache.
    SharedKv,
    /// Future proposal architecture not recognized by this runtime version.
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
            "shared_kv" | "shared-kv" => Self::SharedKv,
            _ => Self::Unknown(value),
        })
    }
}

/// Generic inference strategy declaration.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct StrategySpec {
    /// Strategy vocabulary entry, such as `speculative`.
    #[schemars(with = "schema_vocabulary::StrategyKind")]
    pub kind: String,

    /// Draft-token producer configuration for speculative decoding.
    pub draft: Option<DraftConfig>,

    /// Verification configuration for speculative decoding.
    pub verify: Option<VerifyConfig>,

    /// Draft-token acceptance rule.
    #[schemars(with = "Option<schema_vocabulary::AcceptanceMethod>")]
    pub acceptance: Option<String>,

    /// Number of draft tokens attempted per verification step.
    #[schemars(range(min = 1))]
    pub tokens_per_step: Option<usize>,

    /// Proposal topology, such as `linear` or `tree`.
    #[schemars(with = "Option<schema_vocabulary::ProposalTopology>")]
    pub topology: Option<String>,

    /// Model-publisher performance guidance.
    pub performance_hints: Option<PerformanceHints>,
}

/// Draft-token producer configuration.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct DraftConfig {
    /// Producer family: `draft_model`, `self_speculative`, `ngram`, or `extra_heads`.
    #[schemars(with = "schema_vocabulary::DraftProducer")]
    pub producer: String,

    /// Named runtime session or pipeline component used as the producer.
    pub session: Option<String>,

    /// Self-speculative early-exit depth.
    #[schemars(range(min = 1))]
    pub depth: Option<usize>,

    /// Named draft-head layout or selection.
    pub heads: Option<String>,

    /// Runtime-specific n-gram or prompt-lookup configuration.
    pub ngram: Option<serde_json::Value>,
}

/// Draft-token verification configuration.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct VerifyConfig {
    /// Verification method, such as `single_forward`.
    #[schemars(with = "Option<schema_vocabulary::VerificationMethod>")]
    pub method: Option<String>,

    /// Named verifier session or pipeline component.
    pub session: Option<String>,
}

/// Publisher-provided speculative decoding performance guidance.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct PerformanceHints {
    /// Expected fraction of proposed tokens accepted, from 0.0 through 1.0.
    #[schemars(range(min = 0.0, max = 1.0))]
    pub expected_acceptance_rate: Option<f32>,

    /// Recommended number of draft tokens per verification step.
    #[schemars(range(min = 1))]
    pub optimal_k: Option<usize>,
}

/// Structured-output capabilities and model formatting conventions.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct StructuredOutputSpec {
    /// Supported constraint formats, such as JSON Schema, regular expressions, or CFGs.
    #[schemars(with = "Option<Vec<schema_vocabulary::StructuredOutputFormat>>")]
    pub supported_formats: Option<Vec<String>>,

    /// Format in which the model was trained to emit tool calls or structured values.
    pub training_format: Option<String>,

    /// Literal token sequences that terminate a structured response.
    pub stop_sequences: Option<Vec<String>>,
}
