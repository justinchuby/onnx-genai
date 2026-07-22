//! Typed structs for all inference metadata spec sections.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

fn deserialize_non_empty_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.is_empty() {
        return Err(serde::de::Error::custom("presence keys must not be empty"));
    }
    Ok(value)
}

fn deserialize_optional_non_empty_string<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)?.map_or(Ok(None), |value| {
        if value.is_empty() {
            Err(serde::de::Error::custom("presence keys must not be empty"))
        } else {
            Ok(Some(value))
        }
    })
}

/// ONNX inference metadata consumed by runtimes and emitted by model builders.
///
/// Every top-level section is optional for incremental adoption. Unknown fields
/// are allowed and must be ignored by readers for forward compatibility.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[schemars(
    title = "ONNX Inference Metadata",
    description = "Portable, runtime-agnostic inference metadata for ONNX generative models. All top-level sections are optional, and unknown fields are permitted for forward-compatible schema evolution.",
    extend("$id" = "https://github.com/onnx/onnx/issues/8184"),
    transform = schema_helpers::inference_metadata_aliases
)]
pub struct InferenceMetadata {
    /// Schema version of this inference-metadata document, e.g. `"v1"`.
    ///
    /// Absent means the initial `"v1"` contract (readers default to `v1`).
    /// Bump this only for breaking schema changes; additive fields keep the
    /// same major version and rely on the forward-compatible "ignore unknown
    /// fields" rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,

    /// Capability identifiers that a runtime MUST support or refuse to load the model.
    #[serde(default)]
    #[schemars(
        extend("examples" = [["kv_cache", "grouped_query_attention"]]),
        inner(length(min = 1))
    )]
    pub required_capabilities: Vec<String>,

    /// Build-time model properties and runtime-configurable capabilities.
    #[serde(default)]
    pub model: Option<ModelCapabilities>,

    /// KV-cache storage, quantization tolerance, and operational semantics.
    #[serde(default)]
    pub kv_cache: Option<KvCacheSpec>,

    /// Model weight quantization intent, independent of the packed representation.
    #[serde(default)]
    pub quantization: Option<QuantizationIntent>,

    /// Declarative multi-model pipeline and its dataflow graph.
    #[serde(default)]
    pub pipeline: Option<PipelineSpec>,

    /// Generic inference strategy, including speculative decoding.
    #[serde(default)]
    pub strategy: Option<StrategySpec>,

    /// Standalone speculative proposer declaration.
    ///
    /// This is the preferred native source for speculator discovery;
    /// HuggingFace `config.json` is a compatibility fallback. The deprecated
    /// `speculator_config` alias is accepted on input.
    #[serde(default, alias = "speculator_config")]
    pub speculative: Option<SpeculatorConfig>,

    /// Structured-output formats and model training conventions.
    #[serde(default)]
    pub structured_output: Option<StructuredOutputSpec>,

    /// Minimum and beneficial hardware capabilities used for distribution matching.
    #[serde(default)]
    pub hardware_requirements: Option<HardwareRequirements>,

    /// Author-declared text-generation / search defaults.
    ///
    /// Populated from an onnxruntime-genai `genai_config.json` `search` block.
    /// Every field is optional; readers treat an absent value as "use the
    /// runtime default".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationDefaults>,

    /// Special / control token ids declared by the model author.
    ///
    /// Populated from the model-level token id fields of a `genai_config.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<SpecialTokens>,

    /// Declared, architecture-neutral input preprocessing programs.
    ///
    /// Carries the typed multimodal preprocessing contract (currently the image
    /// transform program and its named tensor outputs). Every operation and
    /// output is generic, parameterized data — never a model family, vendor
    /// string, or baked-in shape. Absent means the model declares no native
    /// preprocessing program and a runtime must obtain it elsewhere or fail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preprocessing: Option<PreprocessingSpec>,
}

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

/// Model properties that are baked into the graph or advertised as configurable.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ModelCapabilities {
    /// Attention architecture and dimensions.
    pub attention: Option<AttentionConfig>,

    /// Maximum total sequence length, in tokens.
    #[schemars(range(min = 1))]
    pub max_sequence_length: Option<usize>,

    /// Vocabulary size (rows of the token-embedding / logits table).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub vocab_size: Option<usize>,

    /// Built-in draft-head or self-speculative model properties.
    pub speculative: Option<SpeculativeModelInfo>,

    /// Features that a serving runtime may configure at load time.
    pub runtime_configurable: Option<RuntimeConfigurable>,

    /// Explicit graph I/O port bindings for the single-decoder LLM path.
    ///
    /// When present, the runtime binds decode-step inputs and outputs from the
    /// declared names instead of inferring them from tensor-name conventions.
    /// When absent, the runtime falls back to the historical name conventions
    /// (a temporary, transitional behavior).
    #[serde(default)]
    pub io: Option<ModelIoSpec>,
}

/// Explicit binding of the graph ports the decode step reads and writes.
///
/// Every field is optional so a model package can declare only the ports its
/// graph exposes. Any port left unset falls back to the runtime's historical
/// tensor-name convention (a temporary, transitional behavior removed once all
/// emitters populate this block). Declaring an `io` block lets a graph use
/// arbitrary tensor names — the runtime never infers a port by name or dtype
/// for a declared port.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
pub struct ModelIoSpec {
    /// Which declared sequence port drives autoregressive execution.
    ///
    /// Absent preserves the historical `token_ids` behavior. Declaring
    /// `inputs_embeds` requires `inputs_embeds_input`; declaring `token_ids`
    /// requires `token_input`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_source: Option<SequenceInputKind>,

    /// Whether this graph owns past/present KV state or reads target-owned KV.
    ///
    /// Absent preserves the historical `owned` behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_ownership: Option<KvOwnership>,

    /// Token-id input (e.g. `input_ids`).
    ///
    /// A graph MAY declare this together with `inputs_embeds_input`: some fused
    /// decoders consume a raw token stream AND a routed pre-embedded sequence in
    /// the same forward pass. The two are not mutually exclusive; declaring both
    /// is a valid, explicit contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub token_input: Option<String>,

    /// Pre-embedded / routed sequence input (e.g. `inputs_embeds`).
    ///
    /// May be declared alongside `token_input` (see its documentation): a graph
    /// that consumes both a raw token input and one or more routed sequence
    /// inputs is explicitly permitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub inputs_embeds_input: Option<String>,

    /// Attention-mask input, if the graph takes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub attention_mask_input: Option<String>,

    /// Position-ids input, if the graph takes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub position_ids_input: Option<String>,

    /// Logits output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub logits_output: Option<String>,

    /// Per-token hidden-state output for embedding / VLM hidden extraction, if
    /// the graph exposes a distinct hidden output (e.g. `last_hidden_state`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub hidden_output: Option<String>,

    /// Past-KV cache inputs, in the SAME order as `kv_outputs` (positional
    /// pairing). Length must match `kv_outputs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(length(min = 1)))]
    pub kv_inputs: Option<Vec<String>>,

    /// Present-KV cache outputs, paired positionally with `kv_inputs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(length(min = 1)))]
    pub kv_outputs: Option<Vec<String>>,

    /// Encoder-hidden-states input for an encoder-decoder (cross-attention)
    /// decoder graph (e.g. `encoder_hidden_states`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub encoder_hidden_states_input: Option<String>,

    /// Cross-attention past-KV cache inputs for an encoder-decoder decoder, in
    /// the SAME order as `cross_kv_outputs`. These are the encoder-derived KV
    /// tensors, distinct from the self-attention `kv_inputs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(length(min = 1)))]
    pub cross_kv_inputs: Option<Vec<String>>,

    /// Cross-attention present-KV cache outputs (produced by the encoder for an
    /// encoder-decoder model), paired positionally with `cross_kv_inputs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(length(min = 1)))]
    pub cross_kv_outputs: Option<Vec<String>>,

    /// How the paired `kv_inputs`/`kv_outputs` cache tensors evolve each step.
    ///
    /// This declares GROWING/append versus fixed shared-buffer cache semantics
    /// explicitly, and is deliberately kept separate from `state_pairs` (which
    /// describes fixed recurrent tensors that are wholly REPLACED). The KV pair
    /// lists are the authoritative sparse layer ports: the runtime binds exactly
    /// the ports named in `kv_inputs`/`kv_outputs` and never expands them from a
    /// total layer count. Absent means the historical growing-cache default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::KvUpdateKind>")]
    pub kv_update: Option<String>,

    /// Fixed-shape loop-carried recurrent state ports, distinct from KV cache.
    ///
    /// Each pair binds an input port to its matching output port and declares
    /// how the input is initialized and how the output feeds the next step
    /// (`replace` semantics for fixed recurrent tensors). These are neither KV
    /// cache nor fixed conditioning; the sparse set of state ports comes from
    /// this declared list, never expanded from a layer count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub state_pairs: Option<Vec<LoopStatePair>>,

    /// Optional graph inputs and their explicit absent-value contracts, keyed by
    /// the real ONNX input port name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub optional_inputs: BTreeMap<String, OptionalInputSpec>,
}

/// Presence and absent-value contract for one optional graph input.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
pub struct OptionalInputSpec {
    /// Opaque, non-empty request presence key; not a port or model name.
    #[serde(deserialize_with = "deserialize_non_empty_string")]
    #[schemars(length(min = 1))]
    pub presence: String,

    /// Tensor value supplied when the presence key is absent.
    pub absent: AbsentInputSpec,
}

/// Explicit tensor fallback for an absent optional graph input.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
pub struct AbsentInputSpec {
    /// Fallback materialization kind.
    pub kind: AbsentInputKind,

    /// Runtime-resolved shape of the fallback tensor.
    pub shape: Vec<TensorDimension>,
}

/// Supported absent-input fallback kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AbsentInputKind {
    /// Materialize a zero-initialized tensor.
    Zeros,
}

/// One fixed or runtime-resolved tensor-shape dimension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum TensorDimension {
    /// A fixed, non-negative dimension.
    Fixed(#[schemars(range(min = 0))] i64),
    /// A runtime shape symbol.
    Symbol(#[schemars(length(min = 1))] String),
}

impl<'de> Deserialize<'de> for TensorDimension {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Representation {
            Fixed(i64),
            Symbol(String),
        }

        match Representation::deserialize(deserializer)? {
            Representation::Fixed(value) if value >= 0 => Ok(Self::Fixed(value)),
            Representation::Fixed(_) => Err(serde::de::Error::custom(
                "tensor dimensions must be non-negative",
            )),
            Representation::Symbol(value) if !value.is_empty() => Ok(Self::Symbol(value)),
            Representation::Symbol(_) => {
                Err(serde::de::Error::custom("tensor symbols must not be empty"))
            }
        }
    }
}

/// Primary autoregressive sequence source for a decoder or proposer graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SequenceInputKind {
    /// Integer token ids supplied through `token_input`.
    TokenIds,
    /// Precomputed floating-point embeddings supplied through
    /// `inputs_embeds_input`.
    InputsEmbeds,
}

/// Ownership model for a graph's KV cache inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KvOwnership {
    /// The graph consumes past KV and emits replacement/extended present KV.
    Owned,
    /// The graph reads references to KV owned and advanced by another decoder.
    Shared,
}

/// One fixed-shape loop-carried recurrent-state port pair.
///
/// Generic and architecture-neutral: the runtime zero/other-initializes `input`
/// on the first step, runs the graph, and copies `output` back into `input` for
/// the next step (`replace` update). This models any fixed recurrent tensor
/// (convolution state, linear-attention recurrent state, and so on) without
/// referencing a model family. It is intentionally distinct from growing or
/// shared-buffer KV cache, which is declared through `kv_inputs`/`kv_outputs`
/// and `kv_update`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[schemars(transform = schema_helpers::loop_state_pair)]
pub struct LoopStatePair {
    /// Graph input port that receives the carried state for this step.
    #[schemars(length(min = 1))]
    pub input: String,

    /// Graph output port that produces the next-step state.
    #[schemars(length(min = 1))]
    pub output: String,

    /// How `input` is initialized before the first step (e.g. `zeros`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "schema_vocabulary::StateInitKind")]
    pub init: Option<String>,

    /// How `output` becomes the next step's `input` (fixed state uses `replace`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "schema_vocabulary::StateUpdateKind")]
    pub update: Option<String>,
}

/// Build-time attention architecture and dimensions.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AttentionConfig {
    /// Attention architecture.
    ///
    /// Canonical values include `multi_head`, `grouped_query`, and
    /// `multi_latent`; future values are allowed when paired with a usable
    /// `fallback_behavior`.
    #[serde(rename = "type")]
    #[schemars(with = "schema_vocabulary::AttentionType")]
    pub attention_type: String,

    /// Number of key/value heads; required by runtimes that need explicit GQA dimensions.
    #[schemars(range(min = 1))]
    pub num_kv_heads: Option<usize>,

    /// Number of query/attention heads.
    #[schemars(range(min = 1))]
    pub num_attention_heads: Option<usize>,

    /// Per-head hidden dimension.
    #[schemars(range(min = 1))]
    pub head_dim: Option<usize>,

    /// Sliding-window length in tokens, or null for full-context attention.
    #[schemars(range(min = 1))]
    pub sliding_window: Option<usize>,

    /// Number of leading "attention sink" tokens always retained alongside the
    /// sliding window (StreamingLLM). Only meaningful when `sliding_window` is
    /// set; `null` or `0` disables sink retention. These first tokens stabilize
    /// the attention distribution and are never evicted by the window.
    #[schemars(range(min = 0))]
    pub sink_tokens: Option<usize>,

    /// Representation compatibility for the attention key-sequence lengths.
    ///
    /// Absent means the canonical contiguous `int32 [batch_size]` representation
    /// is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_sequence_lengths: Option<KeySequenceLengthsSpec>,

    /// Compatible attention behavior for runtimes that do not recognize `type`.
    #[schemars(with = "Option<schema_vocabulary::AttentionType>")]
    pub fallback_behavior: Option<String>,
}

/// Explicit compatibility rules for attention key-sequence-length metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct KeySequenceLengthsSpec {
    /// Optional scalar compatibility. `unit_batch` authorizes a contiguous
    /// rank-0 one-element `int32` tensor only when the attention batch is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_broadcast: Option<SequenceLengthScalarBroadcast>,
}

/// Permitted scalar compatibility for attention key-sequence lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SequenceLengthScalarBroadcast {
    /// Interpret one rank-0 value as the canonical one-element vector only for
    /// an attention batch of exactly one.
    UnitBatch,
}

/// Build-time support for self-contained speculative decoding.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SpeculativeModelInfo {
    /// Whether the exported graph contains Medusa/EAGLE/MTP-style draft heads.
    pub has_draft_heads: Option<bool>,

    /// Early-exit layer depth usable for self-speculation.
    #[schemars(range(min = 1))]
    pub self_speculative_depth: Option<usize>,
}

/// Features whose concrete settings may be selected by the runtime.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RuntimeConfigurable {
    /// Supported runtime-selectable KV-cache dtypes.
    pub kv_cache: Option<RuntimeKvConfig>,

    /// Whether prefix caching may be enabled.
    pub prefix_cache: Option<bool>,

    /// Whether continuous batching may be enabled.
    pub continuous_batching: Option<bool>,

    /// Chunked-prefill support and preferred chunk size.
    pub chunked_prefill: Option<ChunkedPrefillConfig>,
}

/// Runtime-selectable KV-cache representations.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RuntimeKvConfig {
    /// Non-empty list of supported KV-cache scalar dtypes, in preference order.
    #[schemars(with = "Vec<schema_vocabulary::DType>", length(min = 1))]
    pub dtype: Vec<String>,
}

/// Runtime chunked-prefill preference.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ChunkedPrefillConfig {
    /// Preferred number of prompt tokens processed in each prefill chunk.
    #[schemars(range(min = 1))]
    pub chunk_size: Option<usize>,
}

/// KV-cache storage, precision tolerance, and operational guarantees.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct KvCacheSpec {
    /// Native KV scalar dtype produced by the model before optional compression.
    #[schemars(with = "Option<schema_vocabulary::DType>")]
    pub native_dtype: Option<String>,

    /// Independent precision tolerance for key and value tensors.
    pub quantization_tolerance: Option<KvQuantTolerance>,

    /// Layer indices that should retain high precision; negative indices count from the end.
    pub sensitive_layers: Option<Vec<i32>>,

    /// Cache mutation and persistence operations known to be safe for this model.
    pub operations: Option<KvCacheOperations>,
}

/// Precision tolerance for key and value cache components.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct KvQuantTolerance {
    /// Key-cache precision tolerance.
    pub key: Option<KvComponentTolerance>,

    /// Value-cache precision tolerance.
    pub value: Option<KvComponentTolerance>,
}

/// Quantization tolerance for one KV-cache component.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct KvComponentTolerance {
    /// Default minimum acceptable scalar dtype for this component.
    #[schemars(with = "Option<schema_vocabulary::DType>")]
    pub default: Option<String>,

    /// Layer-specific minimum-precision overrides.
    pub per_layer: Option<Vec<LayerPrecisionOverride>>,

    /// Quantization scaling axis, such as `per_tensor`, `per_channel`, or `per_token`.
    #[schemars(with = "Option<schema_vocabulary::QuantizationAxis>")]
    pub quantization_axis: Option<String>,
}

/// Minimum precision required by a set of model layers.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct LayerPrecisionOverride {
    /// Non-empty layer-index list; negative indices count from the final layer.
    #[schemars(length(min = 1))]
    pub layers: Vec<i32>,

    /// Minimum acceptable scalar dtype for the listed layers.
    #[schemars(with = "schema_vocabulary::DType")]
    pub min_precision: String,
}

/// Operational guarantees for mutable KV-cache state.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct KvCacheOperations {
    /// Whether truncating cache state to an earlier token position is correctness-preserving.
    pub rewind_safe: Option<bool>,

    /// Precision policy for a copy-on-write fork, such as `inherit` or `highest`.
    #[schemars(with = "Option<schema_vocabulary::ForkPrecisionPolicy>")]
    pub fork_precision_policy: Option<String>,

    /// Whether checkpoints can be serialized for suspend/resume or migration.
    pub checkpoint_serializable: Option<bool>,
}

/// Runtime-independent model-weight quantization intent.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct QuantizationIntent {
    /// Default precision or quantization recipe for model weights.
    #[schemars(with = "Option<schema_vocabulary::Precision>")]
    pub default: Option<String>,

    /// Layer- or component-specific precision overrides.
    pub overrides: Option<Vec<QuantizationOverride>>,
}

/// Precision override for selected layers or a named graph component.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct QuantizationOverride {
    /// Layer indices to which the override applies; negative indices count from the end.
    pub layers: Option<Vec<i32>>,

    /// Logical component path, for example `attention.qk` or `lm_head`.
    #[schemars(length(min = 1))]
    pub component: Option<String>,

    /// Required precision or quantization recipe.
    #[schemars(with = "schema_vocabulary::Precision")]
    pub precision: String,
}

/// Multi-model pipeline represented as a directed acyclic dataflow graph.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct PipelineSpec {
    /// Named model components in the pipeline DAG; at least one component is required.
    #[schemars(extend("minProperties" = 1))]
    pub models: BTreeMap<String, PipelineComponentSpec>,

    /// Directed tensor or data edges between component ports.
    #[serde(default)]
    pub dataflow: Vec<DataflowEdge>,

    /// Loop and execution strategy for the pipeline.
    pub strategy: PipelineStrategy,

    /// Optional per-component phase gating, keyed by component name.
    #[serde(default)]
    pub phases: BTreeMap<String, PhaseConfig>,

    /// Vision-language model token-expansion contract.
    ///
    /// When present, the engine uses these fields to replace each image
    /// placeholder token in the prompt with the declared expanded image-token
    /// sequence before KV-cache allocation.
    #[serde(default)]
    pub vision: Option<PipelineVisionConfig>,

    /// Declared position-id generation and prefill→decode continuation program.
    ///
    /// Generic and architecture-neutral: parameterized by rank, axis labels, and
    /// section sizes so it expresses both ordinary rank-2 linear positions and
    /// rank-N multimodal coordinates as data — never a model-family branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<PositionProgram>,
}

/// Declared position-id program for a decoder graph.
///
/// The runtime constructs the position tensor from these declared parameters
/// instead of assuming a fixed rank-2 layout. `rank` 1 (with a single axis)
/// expresses ordinary linear positions; `rank` N expresses multi-axis
/// multimodal coordinates. Axis labels and section sizes are opaque DATA — the
/// runtime never infers them from a model name.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct PositionProgram {
    /// Graph input port that receives the position ids (arbitrary name, DATA).
    #[schemars(length(min = 1))]
    pub input: String,

    /// Number of coordinate streams carried by the position tensor.
    ///
    /// `1` is an ordinary linear position stream; values `> 1` describe
    /// multi-axis multimodal coordinates. The physical ONNX tensor rank is
    /// declared separately by `tensor_rank`.
    #[schemars(range(min = 1))]
    pub rank: usize,

    /// Physical ONNX tensor rank.
    ///
    /// Rank 2 declares a conventional `[batch, sequence]` linear input. Higher
    /// ranks declare an explicit coordinate axis in addition to batch/sequence
    /// axes. Absent preserves the legacy mapping (`rank == 1` means tensor rank
    /// 2; otherwise tensor rank 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 2))]
    pub tensor_rank: Option<usize>,

    /// How the position values are generated for prefill.
    ///
    /// `linear` generates ordinary sequence positions. `processor_coordinates`
    /// consumes the declared processor summaries to construct multi-axis
    /// coordinates. Future generation programs remain extensible capability
    /// strings rather than model-family branches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::PositionGeneration>")]
    pub generation: Option<String>,

    /// Optional coordinate-stream labels, one per stream (DATA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(length(min = 1)))]
    pub axes: Option<Vec<String>>,

    /// Optional section sizes for sectioned rotary position embeddings.
    ///
    /// Opaque list of per-section widths; their meaning is model DATA, not a
    /// runtime branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sections: Option<Vec<usize>>,

    /// Declared dtype of the position tensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::TensorDType>")]
    pub dtype: Option<String>,

    /// How positions continue from the prompt (prefill) into per-token decode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::PositionContinuation>")]
    pub continuation: Option<String>,

    /// Optional processor-summary endpoints this program reads to compute
    /// multi-axis coordinates (e.g. a declared grid-dimensions output). Each
    /// entry is an arbitrary endpoint name (DATA), never a model-family hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(length(min = 1)))]
    pub processor_summaries: Option<Vec<String>>,
}

/// Image placeholder token-expansion contract for encoder-free VLM pipelines.
///
/// Every field is optional and additive: legacy documents that declare only
/// `image_placeholder_token_id` and `tokens_per_tile` keep working. The richer
/// fields mirror the generic expansion the preprocessor already models
/// (separate emitted image token, per-tile/per-patch count source, per-image
/// correspondence, optional row/column separators, and thumbnail order). All of
/// it is generic data — no field names or values reference a model family.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct PipelineVisionConfig {
    /// Token ID of the image placeholder in the tokenized prompt.
    ///
    /// The engine replaces every occurrence of this token with the expanded
    /// image token sequence before sequence-length and KV-cache sizing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_placeholder_token_id: Option<i64>,

    /// Number of image tokens each tile expands to.
    ///
    /// The total per-tile expansion is `tokens_per_tile * num_tiles`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub tokens_per_tile: Option<usize>,

    /// Token ID emitted for each expanded image position.
    ///
    /// Distinct from `image_placeholder_token_id`: the placeholder marks WHERE
    /// to expand, while this is the token actually written into the expanded
    /// sequence. When absent, the placeholder token itself is repeated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_token_id: Option<i64>,

    /// Where the per-placeholder token count comes from (per tile, per patch, or
    /// a declared grid). Generic selector, never a model-family branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::ImageTokenCountSource>")]
    pub token_count_source: Option<String>,

    /// Named preprocessing value that supplies per-image counts or grid
    /// dimensions when `token_count_source` is data-derived.
    ///
    /// This is an arbitrary processor output name. A runtime resolves the name
    /// from the declared preprocessing program; it never dispatches on familiar
    /// tensor names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub token_count_summary: Option<String>,

    /// Number of image tokens each patch expands to, used when the count source
    /// is per patch. Declared data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub tokens_per_patch: Option<usize>,

    /// Whether each placeholder occurrence corresponds to one input image in
    /// prompt order. Absent means the historical one-placeholder-per-image rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placeholder_per_image: Option<bool>,

    /// How prompt placeholders correspond to input images.
    ///
    /// `prompt_order` pairs each placeholder with the next input image.
    /// `explicit_indices` reads correspondence from `correspondence_summary`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::ImageCorrespondence>")]
    pub image_correspondence: Option<String>,

    /// Named preprocessing value containing explicit image correspondence data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub correspondence_summary: Option<String>,

    /// Optional token ID emitted between rows of a tiled image grid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_separator_token_id: Option<i64>,

    /// Optional token ID emitted between columns within a grid row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column_separator_token_id: Option<i64>,

    /// Order of the optional global thumbnail tile relative to the local grid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::ThumbnailOrder>")]
    pub thumbnail_order: Option<String>,
}

/// Declared, architecture-neutral input preprocessing programs.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, JsonSchema)]
pub struct PreprocessingSpec {
    /// Typed image preprocessing transform program and its named tensor outputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImagePreprocessingProgram>,
}

/// Generic image preprocessing program: an ordered transform pipeline plus the
/// named tensor outputs it emits.
///
/// The program is expressed entirely as parameterized, architecture-neutral
/// data. Transform operations are generic (decode, resize, rescale, normalize,
/// tile, patchify, pad); outputs bind a produced tensor to an ARBITRARY pipeline
/// endpoint name with a DECLARED dtype. A model may name an output
/// `pixel_position_ids`, `image_grid_thw`, or anything else — that string is
/// data carried in the model's metadata, never a branch in the runtime.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct ImagePreprocessingProgram {
    /// Ordered list of generic transform operations applied to decoded pixels.
    #[serde(default)]
    pub transforms: Vec<ImageTransform>,

    /// Named tensor outputs the program emits, each bound to a pipeline endpoint.
    #[schemars(length(min = 1))]
    pub outputs: Vec<ImageOutputBinding>,
}

/// One generic image transform operation.
///
/// `op` selects the operation from a generic vocabulary; the remaining fields
/// are the parameters that operation reads (only the relevant ones are set).
/// Every parameter is model DATA — concrete sizes, patch sizes, means, and so on
/// live in a model's fixture, never as constants baked into this schema.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct ImageTransform {
    /// Generic operation selector (e.g. `resize`, `normalize`, `patchify`).
    #[schemars(with = "schema_vocabulary::ImageTransformOp")]
    pub op: String,

    /// Named values consumed by this transform.
    ///
    /// Absent means the operation consumes the immediately preceding value.
    /// Explicit names allow branching programs without tensor-name heuristics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub inputs: Option<Vec<String>>,

    /// Named values produced by this transform.
    ///
    /// These names are processor-local data. Final graph bindings select them
    /// through `ImageOutputBinding::source`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub outputs: Option<Vec<String>>,

    /// Target size for a `resize` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<ImageSizeSpec>,

    /// Resize/crop mode (e.g. `pad`, `crop`, `stretch`) — generic string data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub mode: Option<String>,

    /// Interpolation filter for a `resize` operation — generic string data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub interpolation: Option<String>,

    /// Minimum pixel area for an aspect-preserving `pixel_area` resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub min_pixels: Option<usize>,

    /// Maximum pixel area for an aspect-preserving `pixel_area` resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_pixels: Option<usize>,

    /// Required divisibility of both resized dimensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub size_multiple: Option<usize>,

    /// Maximum number of spatial patches for a patch-budget resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_patches: Option<usize>,

    /// Spatial pooling edge used when resolving a patch-budget resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub pooling_kernel_size: Option<usize>,

    /// Scalar multiplier for a `rescale` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,

    /// Per-channel mean for a `normalize` operation (length is model data).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean: Option<Vec<f32>>,

    /// Per-channel standard deviation for a `normalize` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub std: Option<Vec<f32>>,

    /// Edge length of a square tile for a `tile` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub tile_size: Option<usize>,

    /// Maximum number of local tiles for a `tile` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_tiles: Option<usize>,

    /// Whether a `tile` operation also emits a global thumbnail tile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_thumbnail: Option<bool>,

    /// Ordering of a global thumbnail relative to local tiles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::ThumbnailOrder>")]
    pub thumbnail_order: Option<String>,

    /// Interpolation filter used specifically for a global thumbnail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub thumbnail_interpolation: Option<String>,

    /// RGB canvas fill value applied before dynamic tiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canvas_pad_value: Option<f64>,

    /// Pixel edge represented by one validity-mask cell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub mask_patch_size: Option<usize>,

    /// Edge length of a square patch for a `patchify` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub patch_size: Option<usize>,

    /// Number of identical temporal frames packed into each spatial patch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub temporal_patch_size: Option<usize>,

    /// Spatial patch-group edge controlling packed patch traversal order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub merge_size: Option<usize>,

    /// Flattened patch feature order (`channels_first` or `channels_last`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub channel_order: Option<String>,

    /// Patch-coordinate component order (`yx` or `xy`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub coordinate_order: Option<String>,

    /// Whether `patchify` flattens each patch into a single feature vector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flatten: Option<bool>,

    /// Fill value for a `pad` operation, or sentinel for padded coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_value: Option<f64>,

    /// Exact first-axis length produced by a `pad` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub target_length: Option<usize>,
}

/// A square size or an explicit width/height for an image transform.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ImageSizeSpec {
    /// A single edge length applied to both dimensions.
    Square(u32),
    /// Explicit width and height.
    Dimensions {
        /// Target width in pixels.
        width: u32,
        /// Target height in pixels.
        height: u32,
    },
}

/// One named tensor output produced by an image preprocessing program.
///
/// The output binds a generic content role to an ARBITRARY endpoint name with a
/// DECLARED dtype. Neither the name nor the content role is inferred from a model
/// identity, and the dtype is always explicit rather than derived from the model.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct ImageOutputBinding {
    /// Named value produced by a transform.
    ///
    /// Absent preserves the legacy content-derived binding behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub source: Option<String>,

    /// Arbitrary pipeline endpoint name this tensor is bound to (model DATA).
    #[schemars(length(min = 1), example = &"vision_encoder.pixel_values")]
    pub name: String,

    /// Generic content role this tensor carries (pixels, coordinates, grid,
    /// original size, or validity mask) — never a model-family label.
    #[schemars(with = "schema_vocabulary::ImageOutputContent")]
    pub content: String,

    /// Declared output dtype. Always explicit; never inferred from the model.
    #[schemars(with = "schema_vocabulary::TensorDType")]
    pub dtype: String,

    /// Optional sentinel/pad value for padded entries (e.g. `-1` coordinates).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_value: Option<f64>,

    /// Whether the runtime may omit this output when a model does not need it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optional: Option<bool>,
}

/// One executable ONNX model in a pipeline.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct PipelineComponentSpec {
    /// Non-empty ONNX filename relative to the model package root.
    #[schemars(length(min = 1), example = &"decoder.onnx")]
    pub filename: String,

    /// Component role, for example `encoder`, `decoder`, `draft`, `denoiser`, or `vocoder`.
    #[serde(rename = "type")]
    #[schemars(with = "schema_vocabulary::PipelineRole")]
    pub role: String,

    /// Optional execution or device preference declared by the model package.
    #[schemars(with = "Option<schema_vocabulary::DevicePreference>")]
    pub device_preference: Option<String>,

    /// Tokenizer filename relative to the package root.
    ///
    /// If absent, loaders may use a shared top-level `tokenizer.json`.
    #[schemars(length(min = 1), example = &"tokenizer.json")]
    pub tokenizer: Option<String>,

    /// Explicit graph I/O port bindings for this pipeline component.
    ///
    /// When present, the runtime binds decode-step ports from the declared
    /// names instead of inferring them from tensor-name conventions. When
    /// absent, the runtime falls back to the historical name conventions (a
    /// temporary, transitional behavior).
    #[serde(default)]
    pub io: Option<ModelIoSpec>,
}

/// Directed connection between two pipeline component ports.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct DataflowEdge {
    /// Source endpoint in `component.output_name` form.
    #[schemars(regex(pattern = r"^[^.]+\.[^.]+$"), example = &"encoder.hidden_states")]
    pub from: String,

    /// Destination endpoint in `component.input_name` form.
    #[schemars(regex(pattern = r"^[^.]+\.[^.]+$"), example = &"decoder.encoder_hidden_states")]
    pub to: String,

    /// Scalar or logical data type at the component boundary.
    #[schemars(with = "Option<schema_vocabulary::TensorDType>")]
    pub dtype: Option<String>,

    /// Whether the runtime must move the value between execution devices.
    pub device_transfer: Option<bool>,
}

/// Phase gate for one pipeline component.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct PhaseConfig {
    /// Pipeline phase in which the component runs.
    pub run_on: PhaseRunOn,

    /// Opaque presence key required for this component to run.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_non_empty_string",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(length(min = 1))]
    pub when_present: Option<String>,
}

/// Pipeline phase gate.
///
/// Known values are enumerated while future strings remain valid.
#[derive(Debug, Clone, PartialEq, Eq, JsonSchema)]
#[schemars(with = "String", transform = schema_helpers::phase_run_on)]
pub enum PhaseRunOn {
    /// Run only while processing the initial prompt.
    PromptOnly,
    /// Run at every pipeline step; `always` is accepted as an alias.
    EveryStep,
    /// Run only when producing the final output.
    FinalOnly,
    /// Run only when explicitly requested by the application.
    OnDemand,
    /// Future phase gate not recognized by this runtime version.
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

impl Serialize for PhaseRunOn {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match self {
            Self::PromptOnly => "prompt_only",
            Self::EveryStep => "every_step",
            Self::FinalOnly => "final_only",
            Self::OnDemand => "on_demand",
            Self::Other(value) => value,
        })
    }
}

/// Parameterized execution strategy for a pipeline or composite stage.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct PipelineStrategy {
    /// Strategy family; determines which strategy-specific fields are meaningful.
    pub kind: PipelineStrategyKind,

    /// Autoregressive decoder component name.
    pub decoder: Option<String>,

    /// Maximum number of tokens generated by an autoregressive stage.
    #[schemars(range(min = 1))]
    pub max_tokens: Option<usize>,

    /// Runtime-specific stop-condition declarations.
    pub stop_conditions: Option<Vec<serde_json::Value>>,

    /// Runtime-specific KV-cache strategy parameters.
    pub kv_cache: Option<serde_json::Value>,

    /// Runtime-specific speculative execution parameters.
    pub speculative: Option<serde_json::Value>,

    /// Single-pass component name.
    pub model: Option<String>,

    /// Runtime-specific batching parameters.
    pub batching: Option<serde_json::Value>,

    /// Iterative or diffusion denoiser component name.
    pub denoiser: Option<String>,

    /// Scheduler identifier for iterative or diffusion execution.
    pub scheduler: Option<String>,

    /// Number of iterative or diffusion steps.
    #[schemars(range(min = 1))]
    pub num_steps: Option<usize>,

    /// Denoiser input port that receives the per-step timestep/sigma scalar.
    ///
    /// When set, the iterative loop feeds this input a rank-1 `float32` value
    /// each step (from `timesteps` when provided, otherwise the 0-based step
    /// index), so a step-aware denoiser can condition on the current step.
    #[serde(default)]
    pub timestep_input: Option<String>,

    /// Explicit per-step timestep/sigma schedule for an iterative strategy.
    ///
    /// When present its length must equal `num_steps`; when absent the loop
    /// uses the 0-based step index. Requires `timestep_input` to have any effect.
    #[serde(default)]
    pub timesteps: Option<Vec<f32>>,

    /// First step index for a partial (img2img) denoise loop.
    ///
    /// When set, the iterative loop runs `start_step..num_steps` instead of the
    /// full `0..num_steps`, and the seed (`denoiser` sample input) is expected to
    /// already be the encoded image noised to `timesteps[start_step]`. Matches
    /// diffusers' img2img `get_timesteps(num_steps, strength)` skip. Default 0.
    #[serde(default)]
    pub start_step: Option<usize>,

    /// Optional diffusion scheduler applied to the denoiser's loop-carried
    /// output (treating it as a noise prediction) each step.
    #[serde(default)]
    pub scheduler_config: Option<SchedulerSpec>,

    /// Denoiser conditioning input port zeroed for the unconditional pass of
    /// classifier-free guidance. Required when `guidance_scale` != 1.0.
    #[serde(default)]
    pub cfg_conditioning_input: Option<String>,

    /// Classifier-free guidance scale or equivalent strategy-specific multiplier.
    #[schemars(range(min = 0.0))]
    pub guidance_scale: Option<f32>,

    /// Runtime-specific iterative state declaration.
    pub state: Option<serde_json::Value>,

    /// Ordered child stages for a composite strategy.
    #[serde(default)]
    pub stages: Vec<PipelineStrategyStage>,

    /// Outer autoregressive decoder for a `nested_autoregressive` stage.
    ///
    /// The multi-decoder TTS shape: one outer step is one
    /// audio frame. The outer decoder (talker) produces a per-frame
    /// `last_hidden_state` that seeds the inner loop (see `inner`).
    #[serde(default)]
    pub outer: Option<String>,

    /// Inner autoregressive decoder for a `nested_autoregressive` stage.
    ///
    /// The code_predictor: for each outer frame it runs a short inner AR loop of
    /// `num_code_groups` steps over the residual codebooks, seeded at inner step
    /// 0 by the outer decoder's `last_hidden_state` (routed via a dataflow edge
    /// `outer.last_hidden_state -> inner.inputs_embeds`) and threading its own
    /// per-step code embedding on later steps.
    #[serde(default)]
    pub inner: Option<String>,

    /// Inner-loop depth (RVQ residual codebook count) for a
    /// `nested_autoregressive` stage: the number of code tokens collected per
    /// outer frame. Must be at least 1.
    #[schemars(range(min = 1))]
    #[serde(default)]
    pub num_code_groups: Option<usize>,

    /// Optional pre-embedder component driving the outer decoder (talker) of a
    /// `nested_autoregressive` stage through `inputs_embeds` instead of
    /// `input_ids`.
    ///
    /// A codec-driven TTS talker is not driven by token ids: each step's
    /// `inputs_embeds` is materialized from the PREVIOUS frame's codes as
    /// `codec_sum(+ text_embed)` (where
    /// `codec_sum = codec_embed(code_0) + Σ_i cp_codec_weights[i][codes[i+1]]`).
    /// When this field names such a component (inputs
    /// `frame_codes [batch, num_code_groups]` int64 `[+ text_embed [batch, 1,
    /// hidden]]` → output `inputs_embeds [batch, 1, hidden]`), the runtime builds
    /// the outer decoder's per-step `inputs_embeds` through it, keeping the engine
    /// generic. Requires a dataflow edge
    /// `{pre_embedder}.inputs_embeds -> {outer}.inputs_embeds`.
    ///
    /// When absent the outer loop is `input_ids`-driven (backward compatible).
    ///
    /// All graph-specific port bindings (the pre-embedder's `frame_codes` /
    /// optional `text_embed` inputs and the output feeding the outer decoder)
    /// are declared explicitly in [`PreEmbedderSpec`]; the runtime never guesses
    /// them by tensor name or dtype.
    #[serde(default)]
    pub pre_embedder: Option<PreEmbedderSpec>,

    /// Optional prefill embedder component that supplies the outer decoder
    /// (talker) with its real frame-0 PREFILL sequence and the per-frame
    /// trailing-text conditioning of a `nested_autoregressive` stage.
    ///
    /// The talker is prefilled with a multi-position embedding
    /// sequence built from the tokenized prompt, and each subsequent frame is
    /// conditioned on one trailing-text embedding. This component materializes
    /// both from `text_ids`: inputs `text_ids [batch, text_len]` int64 → outputs
    /// `prefill_embeds [batch, prefill_len, hidden]` float (fed DIRECTLY to the
    /// talker's `inputs_embeds` on frame 0) and `trailing_text_embeds [batch,
    /// trailing_len, hidden]` float (one vector consumed per outer frame `k >= 1`
    /// as the pre-embedder's `text_embed`). It runs once in the prompt phase
    /// (`run_on: prompt_only`); its `text_ids` input is auto-seeded from the
    /// tokenized prompt.
    ///
    /// Only meaningful together with [`Self::pre_embedder`] (the frame-`k >= 1`
    /// path feeds the trailing-text vectors through it). When absent, frame 0
    /// uses a zero seed and every `text_embed` is zero (backward compatible).
    ///
    /// All graph-specific port bindings (the prompt input plus the prefill and
    /// trailing-text outputs) are declared explicitly in [`PrefillEmbedderSpec`];
    /// the runtime never guesses them by tensor name or dtype.
    #[serde(default)]
    pub prefill_embedder: Option<PrefillEmbedderSpec>,
}

/// Structured binding for the optional pre-embedder that drives the outer
/// decoder (talker) of a `nested_autoregressive` stage via `inputs_embeds`.
///
/// Every graph-specific port the runtime touches is declared here, so the
/// engine never infers a port by tensor name or dtype.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct PreEmbedderSpec {
    /// Declared model name of the pre-embedder component.
    #[schemars(length(min = 1))]
    pub component: String,

    /// Pre-embedder input port receiving the previous frame's codes
    /// (`int64 [batch, num_code_groups]`).
    #[schemars(length(min = 1))]
    pub frame_codes_input: String,

    /// Optional pre-embedder input port receiving the per-frame trailing-text
    /// conditioning vector (`float [batch, 1, hidden]`). When absent, the
    /// pre-embedder exposes no trailing-text input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_embed_input: Option<String>,
}

/// Structured binding for the optional prefill embedder that supplies the outer
/// decoder (talker) of a `nested_autoregressive` stage with its frame-0 PREFILL
/// sequence and per-frame trailing-text conditioning.
///
/// Every graph-specific port the runtime touches is declared here, so the
/// engine never infers a port by tensor name or dtype.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct PrefillEmbedderSpec {
    /// Declared model name of the (prompt-phase) prefill embedder component.
    #[schemars(length(min = 1))]
    pub component: String,

    /// Prefill-embedder input port receiving the tokenized prompt
    /// (`int64 [batch, text_len]`, e.g. `text_ids`).
    #[schemars(length(min = 1))]
    pub prompt_input: String,

    /// Prefill-embedder output port carrying the talker's frame-0 multi-position
    /// PREFILL sequence (`float [batch, prefill_len, hidden]`), fed DIRECTLY to
    /// the outer decoder's `inputs_embeds` on frame 0.
    #[schemars(length(min = 1))]
    pub prefill_output: String,

    /// Prefill-embedder output port carrying the per-frame trailing-text vectors
    /// (`float [batch, trailing_len, hidden]`), one sliced per outer frame
    /// `k >= 1` into the pre-embedder's `text_embed`.
    #[schemars(length(min = 1))]
    pub trailing_output: String,
}

/// Named child stage of a composite pipeline strategy.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct PipelineStrategyStage {
    /// Non-empty stage name unique within its containing composite.
    #[schemars(length(min = 1))]
    pub name: String,

    /// Execution strategy for this stage.
    pub strategy: Box<PipelineStrategy>,

    /// Optional phase gate for the stage.
    pub run_on: Option<PhaseRunOn>,
}

/// Diffusion scheduler configuration for an iterative strategy.
///
/// The runtime treats the denoiser's loop-carried output as a noise prediction
/// (or, for `masked_diffusion`, as token logits) and applies one scheduler step
/// per iteration. Supported `kind`s: `ddim`, `euler`, `dpmpp_2m` (image
/// diffusion, with optional Karras/exponential sigmas) and `masked_diffusion`
/// (discrete language diffusion).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, JsonSchema)]
pub struct SchedulerSpec {
    /// Scheduler algorithm: `"ddim"`, `"euler"`, `"dpmpp_2m"`, or
    /// `"masked_diffusion"`.
    pub kind: String,

    /// Training timesteps the noise schedule was defined over (default 1000).
    #[schemars(range(min = 2))]
    pub num_train_timesteps: Option<usize>,

    /// Linear beta-schedule start (default 0.00085).
    #[schemars(range(min = 0.0))]
    pub beta_start: Option<f32>,

    /// Linear beta-schedule end (default 0.012).
    #[schemars(range(min = 0.0))]
    pub beta_end: Option<f32>,

    /// Beta schedule shape: `"linear"` (default) or `"scaled_linear"` (Stable
    /// Diffusion).
    pub beta_schedule: Option<String>,

    /// Model output parameterization; `"epsilon"` is supported (default).
    pub prediction_type: Option<String>,

    /// Mask token id for a `masked_diffusion` (language-diffusion) scheduler:
    /// each step commits the highest-confidence still-masked positions.
    pub mask_token_id: Option<i64>,

    /// Sampling temperature for a `masked_diffusion` scheduler. `0` (default)
    /// selects each masked position's argmax token deterministically; a positive
    /// value applies Gumbel noise (`logits.exp() / (-log u)^temperature`) before
    /// the argmax, matching LLaDA's `add_gumbel_noise`. Confidence used for
    /// remasking is always the clean-softmax probability of the chosen token.
    pub temperature: Option<f32>,

    /// Semi-autoregressive block length for a `masked_diffusion` scheduler, in
    /// tokens. When set (and smaller than the masked generation region), each
    /// step only commits tokens inside the current left-to-right block, matching
    /// LLaDA's semi-autoregressive remasking. Defaults to a single block
    /// spanning the whole masked region.
    pub block_length: Option<usize>,

    /// Unmasking strategy for a `masked_diffusion` scheduler:
    ///   * `"low_confidence"` (default) — LLaDA: each step commits the
    ///     highest-confidence still-masked positions (confidence-ranked). Best
    ///     for LLaDA checkpoints, but greedy/confidence-ranked decoding of other
    ///     masked-diffusion LMs (e.g. MDLM) collapses into repetitive text.
    ///   * `"random"` — MDLM-style ancestral: each still-masked position unmasks
    ///     independently with the schedule probability `1/(steps_remaining)`,
    ///     sampling its token from the model's categorical distribution (use
    ///     `temperature: 1.0` for a true categorical sample). This per-position
    ///     stochastic unmasking avoids the degenerate loops that confidence
    ///     ranking produces. The mask token is never emitted.
    pub remasking: Option<String>,

    /// Use the Karras (arXiv:2206.00364, rho=7) sigma spacing instead of the
    /// default linspace spacing. Applies to sigma-space schedulers (`euler`,
    /// `dpmpp_2m`); the most popular ComfyUI scheduler for those samplers.
    pub use_karras_sigmas: Option<bool>,

    /// Use the exponential sigma spacing (`exp(linspace(log σ_max, log σ_min))`)
    /// instead of linspace. Applies to `euler`/`dpmpp_2m`. Mutually exclusive
    /// with `use_karras_sigmas` (Karras takes precedence).
    pub use_exponential_sigmas: Option<bool>,
}

/// Pipeline execution strategy family.
///
/// Known values are enumerated while future strings remain valid.
#[derive(Debug, Clone, Default, PartialEq, Eq, JsonSchema)]
#[schemars(
    with = "String",
    transform = schema_helpers::pipeline_strategy_kind
)]
pub enum PipelineStrategyKind {
    /// Token-by-token autoregressive generation.
    #[default]
    Autoregressive,
    /// Repeated denoising or another bounded iterative loop.
    Iterative,
    /// One invocation with no runtime-managed loop.
    SinglePass,
    /// Ordered composition of nested strategies.
    Composite,
    /// Dual, hierarchically-nested autoregressive loops (multi-decoder TTS).
    ///
    /// An outer decoder (talker) AR loop where each outer step drives an inner
    /// decoder (code_predictor) AR loop; see [`PipelineStrategy::outer`],
    /// [`PipelineStrategy::inner`], and [`PipelineStrategy::num_code_groups`].
    NestedAutoregressive,
    /// Future strategy family not recognized by this runtime version.
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
            "nested_autoregressive" | "nested-autoregressive" => Self::NestedAutoregressive,
            _ => Self::Other(value),
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

mod schema_vocabulary {
    use schemars::JsonSchema;

    macro_rules! extensible_string {
        (
            $(#[$meta:meta])*
            $name:ident,
            $transform:ident,
            $values:ident,
            [$($value:literal),+ $(,)?]
        ) => {
            $(#[$meta])*
            #[derive(JsonSchema)]
            #[schemars(with = "String", transform = super::schema_helpers::$transform)]
            pub(super) struct $name;

            pub(super) const $values: &[&str] = &[$($value),+];
        };
    }

    extensible_string!(
        /// Attention architecture vocabulary with an extension branch.
        AttentionType,
        attention_type,
        ATTENTION_TYPE,
        [
            "multi_head",
            "multi_head_attention",
            "grouped_query",
            "group_query_attention",
            "grouped_query_attention",
            "gqa",
            "multi_latent",
            "multi_latent_attention",
            "mla"
        ]
    );

    extensible_string!(
        /// Scalar dtype vocabulary with common ONNX and runtime aliases.
        DType,
        dtype,
        DTYPE,
        [
            "float32",
            "fp32",
            "float16",
            "fp16",
            "half",
            "bfloat16",
            "bf16",
            "float8_e4m3fn",
            "fp8_e4m3fn",
            "float8_e4m3",
            "fp8_e4m3",
            "float8_e5m2",
            "fp8_e5m2",
            "int8",
            "uint8",
            "int4",
            "uint4"
        ]
    );

    extensible_string!(
        /// Tensor-boundary dtype vocabulary, including non-numeric pipeline values.
        TensorDType,
        tensor_dtype,
        TENSOR_DTYPE,
        [
            "float32",
            "fp32",
            "float16",
            "fp16",
            "bfloat16",
            "bf16",
            "float8_e4m3fn",
            "float8_e5m2",
            "int64",
            "int32",
            "int8",
            "uint8",
            "bool",
            "string"
        ]
    );

    extensible_string!(
        /// Quantization scaling-axis vocabulary.
        QuantizationAxis,
        quantization_axis,
        QUANTIZATION_AXIS,
        ["per_tensor", "per_channel", "per_token", "per_head"]
    );

    extensible_string!(
        /// KV fork-precision policy vocabulary.
        ForkPrecisionPolicy,
        fork_precision_policy,
        FORK_PRECISION_POLICY,
        ["inherit", "highest", "independent"]
    );

    extensible_string!(
        /// Weight precision and quantization-recipe vocabulary.
        Precision,
        precision,
        PRECISION,
        [
            "float32",
            "fp32",
            "float16",
            "fp16",
            "bfloat16",
            "bf16",
            "float8_e4m3fn",
            "float8_e5m2",
            "int8",
            "int4",
            "int4_group128"
        ]
    );

    extensible_string!(
        /// Pipeline component-role vocabulary.
        PipelineRole,
        pipeline_role,
        PIPELINE_ROLE,
        [
            "encoder",
            "vision_encoder",
            "audio_encoder",
            "decoder",
            "draft",
            "denoiser",
            "scheduler",
            "vocoder",
            "speech_synthesis"
        ]
    );

    extensible_string!(
        /// Execution-device preference vocabulary.
        DevicePreference,
        device_preference,
        DEVICE_PREFERENCE,
        [
            "auto", "cpu", "cuda", "rocm", "directml", "coreml", "webgpu", "npu"
        ]
    );

    extensible_string!(
        /// Generic inference-strategy vocabulary.
        StrategyKind,
        strategy_kind,
        STRATEGY_KIND,
        ["speculative"]
    );

    extensible_string!(
        /// Speculative draft-producer vocabulary.
        DraftProducer,
        draft_producer,
        DRAFT_PRODUCER,
        ["draft_model", "self_speculative", "ngram", "extra_heads"]
    );

    extensible_string!(
        /// Speculative verification-method vocabulary.
        VerificationMethod,
        verification_method,
        VERIFICATION_METHOD,
        ["single_forward"]
    );

    extensible_string!(
        /// Speculative acceptance-rule vocabulary.
        AcceptanceMethod,
        acceptance_method,
        ACCEPTANCE_METHOD,
        ["rejection_sampling", "greedy", "typical"]
    );

    extensible_string!(
        /// Speculative proposal-topology vocabulary.
        ProposalTopology,
        proposal_topology,
        PROPOSAL_TOPOLOGY,
        ["linear", "tree"]
    );

    extensible_string!(
        /// Structured-output constraint-format vocabulary.
        StructuredOutputFormat,
        structured_output_format,
        STRUCTURED_OUTPUT_FORMAT,
        ["json_schema", "regex", "context_free_grammar", "choice"]
    );

    extensible_string!(
        /// Generic image transform-operation vocabulary.
        ImageTransformOp,
        image_transform_op,
        IMAGE_TRANSFORM_OP,
        [
            "decode",
            "decode_rgb",
            "convert_rgb",
            "resize",
            "rescale",
            "normalize",
            "tile",
            "flatten",
            "patchify",
            "pad",
            "emit_original_size",
            "emit_transformed_size",
            "emit_validity_mask",
            "emit_patch_coordinates",
            "emit_grid_coordinates"
        ]
    );

    extensible_string!(
        /// Generic image-output content-role vocabulary.
        ImageOutputContent,
        image_output_content,
        IMAGE_OUTPUT_CONTENT,
        [
            "pixels",
            "patch_coordinates",
            "grid_dimensions",
            "original_size",
            "transformed_size",
            "validity_mask"
        ]
    );

    extensible_string!(
        /// Image token-count source vocabulary.
        ImageTokenCountSource,
        image_token_count_source,
        IMAGE_TOKEN_COUNT_SOURCE,
        ["per_tile", "per_patch", "from_grid"]
    );

    extensible_string!(
        /// Prompt-placeholder to image correspondence vocabulary.
        ImageCorrespondence,
        image_correspondence,
        IMAGE_CORRESPONDENCE,
        ["prompt_order", "explicit_indices"]
    );

    extensible_string!(
        /// Optional-thumbnail ordering vocabulary.
        ThumbnailOrder,
        thumbnail_order,
        THUMBNAIL_ORDER,
        ["none", "prepend", "append"]
    );

    extensible_string!(
        /// Position-value generation vocabulary.
        PositionGeneration,
        position_generation,
        POSITION_GENERATION,
        ["linear", "processor_coordinates"]
    );

    extensible_string!(
        /// Prefill→decode position-continuation vocabulary.
        PositionContinuation,
        position_continuation,
        POSITION_CONTINUATION,
        ["linear_increment", "carry_max", "from_grid"]
    );

    extensible_string!(
        /// Paired KV-cache update-semantics vocabulary.
        KvUpdateKind,
        kv_update_kind,
        KV_UPDATE_KIND,
        ["append", "shared_buffer"]
    );

    extensible_string!(
        /// Loop-carried state initialization vocabulary.
        StateInitKind,
        state_init_kind,
        STATE_INIT_KIND,
        ["zeros"]
    );

    extensible_string!(
        /// Loop-carried state update-semantics vocabulary.
        StateUpdateKind,
        state_update_kind,
        STATE_UPDATE_KIND,
        ["replace"]
    );
}

mod schema_helpers {
    use schemars::Schema;
    use serde_json::{Value, json};

    pub(super) fn inference_metadata_aliases(schema: &mut Schema) {
        add_alias(
            schema,
            "speculative",
            "speculator_config",
            "Deprecated alias for `speculative`.",
        );
        forbid_both(schema, "speculative", "speculator_config");
    }

    pub(super) fn speculator_config_aliases(schema: &mut Schema) {
        add_alias(
            schema,
            "proposal_type",
            "method",
            "Deprecated alias for `proposal_type`.",
        );
        add_alias(
            schema,
            "num_speculative_tokens",
            "tokens_per_step",
            "Deprecated alias for `num_speculative_tokens`.",
        );

        if let Some(required) = schema
            .ensure_object()
            .get_mut("required")
            .and_then(Value::as_array_mut)
        {
            required.retain(|name| name != "proposal_type");
        }

        schema.ensure_object().insert(
            "oneOf".into(),
            json!([
                {
                    "required": ["proposal_type"],
                    "not": {"required": ["method"]}
                },
                {
                    "required": ["method"],
                    "not": {"required": ["proposal_type"]}
                }
            ]),
        );
        forbid_both(schema, "num_speculative_tokens", "tokens_per_step");
    }

    pub(super) fn loop_state_pair(schema: &mut Schema) {
        let required = schema
            .ensure_object()
            .entry("required")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("required inserted as an array");
        for property in ["init", "update"] {
            if !required.iter().any(|name| name == property) {
                required.push(json!(property));
            }
        }
    }

    pub(super) fn proposal_type(schema: &mut Schema) {
        extensible_string_enum(
            schema,
            &[
                "eagle",
                "eagle3",
                "eagle-3",
                "peagle",
                "p-eagle",
                "mtp",
                "dflash",
                "d-flash",
                "shared_kv",
                "shared-kv",
            ],
        );
    }

    pub(super) fn phase_run_on(schema: &mut Schema) {
        extensible_string_enum(
            schema,
            &[
                "prompt_only",
                "every_step",
                "always",
                "final_only",
                "on_demand",
            ],
        );
    }

    pub(super) fn pipeline_strategy_kind(schema: &mut Schema) {
        extensible_string_enum(
            schema,
            &[
                "autoregressive",
                "iterative",
                "diffusion_steps",
                "diffusion-steps",
                "single_pass",
                "single-pass",
                "composite",
                "nested_autoregressive",
                "nested-autoregressive",
            ],
        );
    }

    pub(super) fn attention_type(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::ATTENTION_TYPE);
    }

    pub(super) fn dtype(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::DTYPE);
    }

    pub(super) fn tensor_dtype(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::TENSOR_DTYPE);
    }

    pub(super) fn quantization_axis(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::QUANTIZATION_AXIS);
    }

    pub(super) fn fork_precision_policy(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::FORK_PRECISION_POLICY);
    }

    pub(super) fn precision(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::PRECISION);
    }

    pub(super) fn pipeline_role(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::PIPELINE_ROLE);
    }

    pub(super) fn device_preference(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::DEVICE_PREFERENCE);
    }

    pub(super) fn strategy_kind(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::STRATEGY_KIND);
    }

    pub(super) fn draft_producer(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::DRAFT_PRODUCER);
    }

    pub(super) fn verification_method(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::VERIFICATION_METHOD);
    }

    pub(super) fn acceptance_method(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::ACCEPTANCE_METHOD);
    }

    pub(super) fn proposal_topology(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::PROPOSAL_TOPOLOGY);
    }

    pub(super) fn structured_output_format(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::STRUCTURED_OUTPUT_FORMAT);
    }

    pub(super) fn image_transform_op(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::IMAGE_TRANSFORM_OP);
    }

    pub(super) fn image_output_content(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::IMAGE_OUTPUT_CONTENT);
    }

    pub(super) fn image_token_count_source(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::IMAGE_TOKEN_COUNT_SOURCE);
    }

    pub(super) fn image_correspondence(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::IMAGE_CORRESPONDENCE);
    }

    pub(super) fn thumbnail_order(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::THUMBNAIL_ORDER);
    }

    pub(super) fn position_continuation(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::POSITION_CONTINUATION);
    }

    pub(super) fn position_generation(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::POSITION_GENERATION);
    }

    pub(super) fn kv_update_kind(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::KV_UPDATE_KIND);
    }

    pub(super) fn state_init_kind(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::STATE_INIT_KIND);
    }

    pub(super) fn state_update_kind(schema: &mut Schema) {
        extensible_string_enum(schema, super::schema_vocabulary::STATE_UPDATE_KIND);
    }

    fn extensible_string_enum(schema: &mut Schema, known_values: &[&str]) {
        let known_values = json!(known_values);
        let object = schema.ensure_object();
        object.insert("type".into(), json!("string"));
        object.insert(
            "oneOf".into(),
            json!([
                {
                    "title": "Known standard value",
                    "enum": known_values.clone()
                },
                {
                    "title": "Forward-compatible extension value",
                    "type": "string",
                    "not": {"enum": known_values}
                }
            ]),
        );
    }

    fn add_alias(schema: &mut Schema, canonical: &str, alias: &str, description: &str) {
        let object = schema.ensure_object();
        let Some(canonical_schema) = object
            .get("properties")
            .and_then(Value::as_object)
            .and_then(|properties| properties.get(canonical))
            .cloned()
        else {
            return;
        };

        if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
            properties.insert(
                alias.to_owned(),
                json!({
                    "allOf": [canonical_schema],
                    "deprecated": true,
                    "description": description
                }),
            );
        }
    }

    fn forbid_both(schema: &mut Schema, first: &str, second: &str) {
        let constraint = json!({
            "not": {
                "required": [first, second]
            }
        });
        let object = schema.ensure_object();
        object
            .entry("allOf")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("allOf inserted as an array")
            .push(constraint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, Serialize)]
    struct OptionalModalityDocument {
        io: ModelIoSpec,
        phase: PhaseConfig,
    }

    #[test]
    fn optional_modality_schema_round_trips() {
        let old_yaml = r#"
io:
  sequence_source: token_ids
phase:
  run_on: prompt_only
"#;
        let old: OptionalModalityDocument =
            serde_yaml::from_str(old_yaml).expect("old metadata deserializes");
        assert!(old.io.optional_inputs.is_empty());
        assert!(old.phase.when_present.is_none());
        assert_eq!(
            serde_yaml::to_value(&old).expect("old metadata serializes"),
            serde_yaml::from_str::<serde_yaml::Value>(old_yaml).expect("old YAML parses")
        );

        let new_yaml = r#"
io:
  optional_inputs:
    audio_features:
      presence: audio
      absent:
        kind: zeros
        shape: [0, sequence_len]
phase:
  run_on: prompt_only
  when_present: audio
"#;
        let new: OptionalModalityDocument =
            serde_yaml::from_str(new_yaml).expect("optional-modality metadata deserializes");
        let optional = new
            .io
            .optional_inputs
            .get("audio_features")
            .expect("optional input is preserved");
        assert_eq!(optional.presence, "audio");
        assert_eq!(optional.absent.kind, AbsentInputKind::Zeros);
        assert_eq!(
            optional.absent.shape,
            [
                TensorDimension::Fixed(0),
                TensorDimension::Symbol("sequence_len".into())
            ]
        );
        assert_eq!(new.phase.when_present.as_deref(), Some("audio"));
        assert_eq!(
            serde_yaml::to_value(&new).expect("optional-modality metadata serializes"),
            serde_yaml::from_str::<serde_yaml::Value>(new_yaml).expect("new YAML parses")
        );
        assert_eq!(
            serde_yaml::to_value(AbsentInputKind::Zeros).expect("kind serializes"),
            serde_yaml::Value::String("zeros".into())
        );

        assert!(
            serde_yaml::from_str::<TensorDimension>("-1").is_err(),
            "negative fixed dimensions must be rejected"
        );
        assert!(
            serde_yaml::from_str::<OptionalInputSpec>(
                "presence: ''\nabsent:\n  kind: zeros\n  shape: [0]\n"
            )
            .is_err(),
            "empty presence keys must be rejected"
        );
    }

    #[test]
    fn attention_config_parses_sliding_window_and_sink_tokens() {
        let yaml = r#"
attention:
  type: grouped_query
  sliding_window: 4096
  sink_tokens: 4
max_sequence_length: 131072
"#;
        let model: ModelCapabilities = serde_yaml::from_str(yaml).expect("parses");
        let attention = model.attention.expect("attention section");
        assert_eq!(attention.sliding_window, Some(4096));
        assert_eq!(attention.sink_tokens, Some(4));
    }

    #[test]
    fn attention_config_defaults_sink_tokens_to_none() {
        let yaml = r#"
attention:
  type: grouped_query
  sliding_window: 4096
"#;
        let model: ModelCapabilities = serde_yaml::from_str(yaml).expect("parses");
        let attention = model.attention.expect("attention section");
        assert_eq!(attention.sliding_window, Some(4096));
        assert_eq!(attention.sink_tokens, None);
    }
}
