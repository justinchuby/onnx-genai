//! Typed structs for all inference metadata spec sections.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer};

/// ONNX inference metadata consumed by runtimes and emitted by model builders.
///
/// Every top-level section is optional for incremental adoption. Unknown fields
/// are allowed and must be ignored by readers for forward compatibility.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(
    title = "ONNX Inference Metadata",
    description = "Portable, runtime-agnostic inference metadata for ONNX generative models. All top-level sections are optional, and unknown fields are permitted for forward-compatible schema evolution.",
    extend("$id" = "https://github.com/onnx/onnx/issues/8184"),
    transform = schema_helpers::inference_metadata_aliases
)]
pub struct InferenceMetadata {
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
    /// Shared-KV proposer (originally introduced for Gemma4 `*-assistant`).
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

    /// Built-in draft-head or self-speculative model properties.
    pub speculative: Option<SpeculativeModelInfo>,

    /// Features that a serving runtime may configure at load time.
    pub runtime_configurable: Option<RuntimeConfigurable>,
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

    /// Compatible attention behavior for runtimes that do not recognize `type`.
    #[schemars(with = "Option<schema_vocabulary::AttentionType>")]
    pub fallback_behavior: Option<String>,
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
#[derive(Debug, Clone, Deserialize, JsonSchema)]
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
    /// placeholder token in the prompt with `tokens_per_tile * num_tiles`
    /// copies of that token before KV-cache allocation.
    #[serde(default)]
    pub vision: Option<PipelineVisionConfig>,
}

/// Image placeholder token-expansion contract for encoder-free VLM pipelines.
///
/// Both fields must be set together; declaring only one is allowed for
/// forward compatibility but will cause an engine error at generation time
/// when `num_image_tiles` is supplied.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct PipelineVisionConfig {
    /// Token ID of the image placeholder in the tokenized prompt.
    ///
    /// The engine replaces every occurrence of this token with the expanded
    /// image token sequence before sequence-length and KV-cache sizing.
    pub image_placeholder_token_id: Option<i64>,

    /// Number of image tokens each tile expands to.
    ///
    /// The total expansion per placeholder is `tokens_per_tile * num_tiles`.
    #[schemars(range(min = 1))]
    pub tokens_per_tile: Option<usize>,
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
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct PhaseConfig {
    /// Pipeline phase in which the component runs.
    pub run_on: PhaseRunOn,
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

/// Parameterized execution strategy for a pipeline or composite stage.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
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
/// and applies one scheduler step per iteration. Currently only `ddim`
/// (η = 0, epsilon-prediction) with a linear beta schedule is supported.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct SchedulerSpec {
    /// Scheduler algorithm; `"ddim"` is supported.
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

    /// Model output parameterization; `"epsilon"` is supported (default).
    pub prediction_type: Option<String>,
}

/// Pipeline execution strategy family.
///
/// Known values are enumerated while future strings remain valid.
#[derive(Debug, Clone, PartialEq, Eq, JsonSchema)]
#[schemars(
    with = "String",
    transform = schema_helpers::pipeline_strategy_kind
)]
pub enum PipelineStrategyKind {
    /// Token-by-token autoregressive generation.
    Autoregressive,
    /// Repeated denoising or another bounded iterative loop.
    Iterative,
    /// One invocation with no runtime-managed loop.
    SinglePass,
    /// Ordered composition of nested strategies.
    Composite,
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
