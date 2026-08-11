use super::*;

/// Model properties that are baked into the graph or advertised as configurable.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
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
    /// The runtime binds decode-step inputs and outputs from the declared names.
    /// A port that is not declared is resolved ONLY from an unambiguous io-shape
    /// signal; when the shape is ambiguous the runtime fails with an actionable
    /// error naming the exact key to declare, and never guesses from a tensor
    /// name.
    #[serde(default)]
    pub io: Option<ModelIoSpec>,

    /// Explicit sparse mixture-of-experts graph and routing contract.
    ///
    /// This describes graph structure, never a model family. Runtimes use the
    /// declared representation and dimensions instead of inferring them from
    /// node names, initializer shapes, or architecture strings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixture_of_experts: Option<MixtureOfExpertsSpec>,
}

/// Explicit binding of the graph ports the decode step reads and writes.
///
/// Every field is optional so a model package can declare only the ports its
/// graph exposes. A port left unset is resolved ONLY from an unambiguous
/// dtype/shape signal; when the shape cannot disambiguate the port, the runtime
/// fails with an actionable error naming the key to declare rather than
/// interpreting a tensor name. A declared port is always authoritative.
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

    /// Physical layout of this backend's KV cache tensors (`head_major_bnsh` or
    /// `seq_major_bsnh`). This is a per-backend capability — each backend owns
    /// its KV buffers and never reads the other's KV bytes — so the ORT backend
    /// stays head-major while the native backend may declare seq-major. Absent
    /// preserves the historical head-major (BNSH) behavior. See [`KvLayout`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_layout: Option<KvLayout>,

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

    /// Raw audio-feature prompt input for an encoder-decoder encoder graph
    /// (e.g. Whisper `audio_features`, a log-mel `[batch, mels, frames]`
    /// tensor). Declared on the encoder component; a text encoder-decoder uses
    /// `token_input` instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub audio_features_input: Option<String>,

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

    /// Explicit port binding for a fixed-buffer TensorScatter static KV cache.
    ///
    /// A static-cache decoder scatters each step's K/V into pre-allocated,
    /// fixed-length buffers via an integer write-index vector and a non-pad
    /// sequence-length vector, rather than growing/appending a cache. These
    /// control ports are integer vectors and are therefore SHAPE-indistinguish-
    /// able from one another, so shape cannot disambiguate them: the ABI must be
    /// declared explicitly. When present, this spec is authoritative and the
    /// runtime binds exactly these ports. When absent, a graph that exposes the
    /// scatter ABI is REJECTED with an actionable error naming this key rather
    /// than having its integer control ports guessed by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_cache: Option<StaticCacheIoSpec>,
}

/// Explicit port ABI for a fixed-buffer TensorScatter static KV cache.
///
/// Describes GRAPH STRUCTURE, never a model family. The four per-layer cache
/// lists pair positionally per layer and must all have the same length: index
/// `i` in each list is layer `i`'s key/value input and updated key/value output.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
pub struct StaticCacheIoSpec {
    /// Input port carrying the per-token scatter write positions
    /// (`int` vector). Shape-indistinguishable from other integer control
    /// inputs, so it must be named explicitly.
    #[schemars(length(min = 1))]
    pub write_indices_input: String,

    /// Input port carrying the non-pad KV sequence length (`int` vector).
    /// Shape-indistinguishable from `write_indices_input`, so it too must be
    /// named explicitly.
    #[schemars(length(min = 1))]
    pub kv_sequence_length_input: String,

    /// Per-layer static key-cache buffer inputs, positional per layer. Length
    /// must equal `value_cache_inputs`, `key_cache_outputs`, and
    /// `value_cache_outputs`.
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub key_cache_inputs: Vec<String>,

    /// Per-layer static value-cache buffer inputs, paired positionally with
    /// `key_cache_inputs`.
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub value_cache_inputs: Vec<String>,

    /// Per-layer updated key-cache outputs, paired positionally with
    /// `key_cache_inputs`.
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub key_cache_outputs: Vec<String>,

    /// Per-layer updated value-cache outputs, paired positionally with
    /// `value_cache_inputs`.
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub value_cache_outputs: Vec<String>,
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

/// Physical memory layout of a backend's KV cache tensors.
///
/// This is a **per-backend capability**, not a cross-backend constant: the two
/// backends own their KV buffers independently and never read each other's KV
/// bytes, so they may store the cache differently. The ONNX Runtime backend
/// requires head-major BNSH (`[batch, kv_heads, seq, head_dim]`) because ORT's
/// GroupQueryAttention past/present is BNSH on every dispatch path (Flash,
/// cuDNN SDPA, memory-efficient, XQA). The native backend additionally supports
/// seq-major BSNH (`[batch, seq, kv_heads, head_dim]`), which makes each token's
/// live prefix contiguous across heads — shrinking the VMM granule floor by the
/// `kv_heads` factor, removing growth-triggered graph re-capture (the append
/// stride is sequence-length independent), and making page-level prefix sharing
/// (#777) practical. Absent preserves the historical head-major behavior.
///
/// On-device, the native backend selects the layout by stamping the `kv_layout`
/// attribute (`0` = BNSH, `1` = BSNH) on its GroupQueryAttention nodes; the
/// CUDA EP honors it on the fused fp16 single-token decode pair. Seq-major is
/// only enabled end-to-end once the prefill (flash) read is also converted, so
/// the two never disagree about how a shared cache is physically laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KvLayout {
    /// Head-major BNSH `[batch, kv_heads, seq, head_dim]`. ORT-compatible; the
    /// default for both backends.
    HeadMajorBnsh,
    /// Seq-major BSNH `[batch, seq, kv_heads, head_dim]`. Native backend only.
    SeqMajorBsnh,
}

impl KvLayout {
    /// The `kv_layout` GroupQueryAttention attribute value the native backend
    /// stamps for this layout (`0` = BNSH, `1` = BSNH).
    pub fn gqa_attribute_value(self) -> i64 {
        match self {
            KvLayout::HeadMajorBnsh => 0,
            KvLayout::SeqMajorBsnh => 1,
        }
    }
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
