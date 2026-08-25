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

    /// Features that a serving runtime may configure at load time.
    pub runtime_configurable: Option<RuntimeConfigurable>,

    /// Explicit sparse mixture-of-experts graph and routing contract.
    ///
    /// This describes graph structure, never a model family. Runtimes use the
    /// declared representation and dimensions instead of inferring them from
    /// node names, initializer shapes, or architecture strings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixture_of_experts: Option<MixtureOfExpertsSpec>,

    /// Legal tensor, pipeline, and expert sharding facts.
    ///
    /// The caller and runtime choose degree, device mapping, and collective
    /// backend. Portable metadata never standardizes a cross-runtime KV or
    /// cache wire format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharding: Option<ShardingContract>,
}

/// Explicit binding of the graph ports the decode step reads and writes.
///
/// Every field is optional so a model package can declare only the ports its
/// graph exposes. A port left unset is resolved ONLY from an unambiguous
/// dtype/shape signal; when the shape cannot disambiguate the port, the runtime
/// fails with an actionable error naming the key to declare rather than
/// interpreting a tensor name. A declared port is always authoritative.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecoderAbi {
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

    /// Physical layout of this backend's KV cache tensors, as a stride
    /// descriptor. Accepts a readable named layout (`head_major_bnsh` or
    /// `seq_major_bsnh`) or a fully explicit [`KvStrideDescriptor`]. This is a
    /// per-backend capability — each backend owns its KV buffers and never reads
    /// the other's KV bytes — so the ORT backend stays head-major while the
    /// native backend may declare seq-major. Absent preserves the historical
    /// head-major (BNSH) behavior. See [`KvCacheLayout`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_layout: Option<KvCacheLayout>,

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

    /// Whether the graph permits, requires, or forbids the runtime aliasing a
    /// `present` output onto its paired `past` input.
    ///
    /// This is the graph ABI fact that replaced the old `shared_buffer` policy
    /// flag: the package states what aliasing its graph is CORRECT under, and
    /// the runtime alone decides whether to exploit it (execution provider
    /// capability, buffer capacity, and batching are runtime concerns). A graph
    /// that reads a `past` region after the paired `present` write would touch
    /// it must declare `forbidden`, which is the default when the package is
    /// silent — silence never grants an optimization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aliasing: Option<StateAliasing>,

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

/// A runtime KV-cache dimension that an axis stride can be a multiple of.
///
/// Absolute element strides are a serving-time property — they depend on the
/// `cache_capacity` a runtime picks — so metadata cannot store them as numbers.
/// A stride is therefore stored **symbolically**, as the (unordered) set of
/// runtime dimensions it multiplies. The concrete element stride of an axis is
/// the product of the sizes of the dimensions in its factor list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KvStrideDim {
    /// Number of KV heads (`kv_heads` / `N`).
    KvHeads,
    /// Sequence capacity of the growing axis (`cache_capacity` / `S`).
    SeqCapacity,
    /// Per-token head width (`head_dim` / `H`).
    HeadDim,
}

/// Symbolic element stride of each of the four logical KV axes.
///
/// The stride of an axis is the product of the runtime dimensions in its factor
/// list; an **empty** list means unit stride (the innermost, contiguous axis).
/// The innermost axis of every layout the converted kernels honor is
/// `head_dim`, whose stride is `1` (empty), because the fp16 read vectorizes
/// `head_dim` as `half2` and the fused write addresses it as `dst + d`.
///
/// The two historical layouts map onto this as:
///
/// | axis     | head-major BNSH        | seq-major BSNH        |
/// |----------|------------------------|-----------------------|
/// | batch    | `kv_heads·seq·head_dim`| `seq·kv_heads·head_dim`|
/// | head     | `seq·head_dim`         | `head_dim`            |
/// | seq      | `head_dim`             | `kv_heads·head_dim`   |
/// | head_dim | `1`                    | `1`                   |
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
pub struct KvAxisStrides {
    /// Factors of the batch-axis stride.
    #[serde(default)]
    pub batch: Vec<KvStrideDim>,
    /// Factors of the KV-head-axis stride.
    #[serde(default)]
    pub head: Vec<KvStrideDim>,
    /// Factors of the sequence-axis (per-token) stride.
    #[serde(default)]
    pub seq: Vec<KvStrideDim>,
    /// Factors of the head-dim-axis stride. Unit (empty) for every honored
    /// layout.
    #[serde(default)]
    pub head_dim: Vec<KvStrideDim>,
}

fn u64_is_zero(value: &u64) -> bool {
    *value == 0
}

/// A fully explicit KV-cache stride descriptor.
///
/// This is the general form the two named layouts expand into, and the shape a
/// future layout (e.g. token-major) is expressed in without adding an enum
/// variant. The `reservation_*` fields describe a binding that is a **view into
/// a larger reservation** rather than the owner of its whole buffer:
/// token-major stores every layer's tokens in one reservation and hands each
/// `(layer, side)` a sub-view, so its per-token (seq) stride is taken over the
/// reservation's total token count and its data starts at a non-zero offset.
/// Both historical layouts are whole-buffer bindings: `offset == 0` and no
/// reservation override.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
pub struct KvStrideDescriptor {
    /// Symbolic stride of each logical axis.
    pub strides: KvAxisStrides,
    /// Element offset of this binding's first element within the reservation it
    /// views. `0` for a binding that owns its whole buffer — the only case the
    /// converted kernels honor today.
    #[serde(default, skip_serializing_if = "u64_is_zero")]
    pub reservation_offset_elements: u64,
    /// Sequence-axis extent, in token slots, of the reservation this binding
    /// views when the reservation is larger than the binding's own
    /// `cache_capacity`. Absent means the binding spans its own capacity (a
    /// whole-buffer binding). Present expresses a token-major view whose seq
    /// stride collapses the per-`(layer, side)` buffer boundary; not honored by
    /// the converted path yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reservation_seq_slots: Option<u64>,
}

impl KvStrideDescriptor {
    /// Head-major BNSH `[batch, kv_heads, seq, head_dim]` explicit strides.
    pub fn head_major_bnsh() -> Self {
        use KvStrideDim::{HeadDim, KvHeads, SeqCapacity};
        KvStrideDescriptor {
            strides: KvAxisStrides {
                batch: vec![KvHeads, SeqCapacity, HeadDim],
                head: vec![SeqCapacity, HeadDim],
                seq: vec![HeadDim],
                head_dim: vec![],
            },
            reservation_offset_elements: 0,
            reservation_seq_slots: None,
        }
    }

    /// Seq-major BSNH `[batch, seq, kv_heads, head_dim]` explicit strides.
    pub fn seq_major_bsnh() -> Self {
        use KvStrideDim::{HeadDim, KvHeads, SeqCapacity};
        KvStrideDescriptor {
            strides: KvAxisStrides {
                batch: vec![SeqCapacity, KvHeads, HeadDim],
                head: vec![HeadDim],
                seq: vec![KvHeads, HeadDim],
                head_dim: vec![],
            },
            reservation_offset_elements: 0,
            reservation_seq_slots: None,
        }
    }
}

/// Physical memory layout of a backend's KV cache tensors, as a stride
/// descriptor.
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
/// Layout preference is per-EP and per-platform rather than a global constant,
/// and a JIT backend compiles a specialized kernel per descriptor, so this is a
/// descriptor rather than a closed enum: a raw stride tuple is unreadable, so
/// the common cases are still nameable (`head_major_bnsh`, `seq_major_bsnh`)
/// while an explicit [`KvStrideDescriptor`] expresses anything the named forms
/// cannot (e.g. a token-major view).
///
/// On-device, the native backend selects the layout by stamping the `kv_layout`
/// attribute (`0` = BNSH, `1` = BSNH) on its GroupQueryAttention nodes; the
/// CUDA EP honors it on the fused fp16 single-token decode pair. Seq-major is
/// only enabled end-to-end once the prefill (flash) read is also converted, so
/// the two never disagree about how a shared cache is physically laid out.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum KvCacheLayout {
    /// A readable shorthand for a standard layout. Deserializes from the strings
    /// `"head_major_bnsh"` and `"seq_major_bsnh"`; expands to explicit strides
    /// via [`KvCacheLayout::resolve_strides`].
    Named(KvNamedLayout),
    /// A fully explicit stride descriptor for layouts the named forms cannot
    /// express.
    Explicit(KvStrideDescriptor),
}

/// The named, human-readable KV cache layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KvNamedLayout {
    /// Head-major BNSH `[batch, kv_heads, seq, head_dim]`. ORT-compatible; the
    /// default for both backends.
    HeadMajorBnsh,
    /// Seq-major BSNH `[batch, seq, kv_heads, head_dim]`. Native backend only.
    SeqMajorBsnh,
}

impl KvNamedLayout {
    /// The explicit stride descriptor this named layout expands into.
    pub fn strides(self) -> KvStrideDescriptor {
        match self {
            KvNamedLayout::HeadMajorBnsh => KvStrideDescriptor::head_major_bnsh(),
            KvNamedLayout::SeqMajorBsnh => KvStrideDescriptor::seq_major_bsnh(),
        }
    }

    /// The `kv_layout` GroupQueryAttention attribute value the native backend
    /// stamps for this layout (`0` = BNSH, `1` = BSNH).
    pub fn gqa_attribute_value(self) -> i64 {
        match self {
            KvNamedLayout::HeadMajorBnsh => 0,
            KvNamedLayout::SeqMajorBsnh => 1,
        }
    }
}

impl KvCacheLayout {
    /// Named head-major BNSH layout (the default).
    pub fn head_major_bnsh() -> Self {
        KvCacheLayout::Named(KvNamedLayout::HeadMajorBnsh)
    }

    /// Named seq-major BSNH layout (native backend only).
    pub fn seq_major_bsnh() -> Self {
        KvCacheLayout::Named(KvNamedLayout::SeqMajorBsnh)
    }

    /// Expand this layout to its explicit stride descriptor. Named layouts
    /// expand to their canonical strides; an explicit descriptor is returned
    /// unchanged.
    pub fn resolve_strides(&self) -> KvStrideDescriptor {
        match self {
            KvCacheLayout::Named(named) => named.strides(),
            KvCacheLayout::Explicit(descriptor) => descriptor.clone(),
        }
    }

    /// The `kv_layout` GroupQueryAttention attribute value (`0` = BNSH,
    /// `1` = BSNH) this layout stamps, or `None` when the descriptor is not one
    /// of the two attribute-expressible named layouts (e.g. a token-major view
    /// the wire format cannot yet carry).
    pub fn gqa_attribute_value(&self) -> Option<i64> {
        match self {
            KvCacheLayout::Named(named) => Some(named.gqa_attribute_value()),
            KvCacheLayout::Explicit(descriptor) => {
                if *descriptor == KvStrideDescriptor::head_major_bnsh() {
                    Some(0)
                } else if *descriptor == KvStrideDescriptor::seq_major_bsnh() {
                    Some(1)
                } else {
                    None
                }
            }
        }
    }
}

/// One fixed-shape loop-carried recurrent-state port pair.
///
/// Generic and architecture-neutral: the runtime zero/other-initializes `input`
/// on the first step, runs the graph, and copies `output` back into `input` for
/// the next step (`replace` update). This models any fixed recurrent tensor
/// (convolution state, linear-attention recurrent state, and so on) without
/// referencing a model family. It is intentionally distinct from growing KV
/// state, whose logical cells and storage service are declared by a workflow.
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

/// Features whose concrete settings may be selected by the runtime.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfigurable {
    /// Whether prefix caching may be enabled.
    pub prefix_cache: Option<bool>,

    /// Chunked-prefill support and preferred chunk size.
    pub chunked_prefill: Option<ChunkedPrefillConfig>,
}

/// Runtime chunked-prefill preference.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ChunkedPrefillConfig {
    /// Preferred number of prompt tokens processed in each prefill chunk.
    #[schemars(range(min = 1))]
    pub chunk_size: Option<usize>,
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

#[cfg(test)]
mod kv_layout_tests {
    use super::*;

    // The named layouts must expand to exactly the strides the retired
    // `KvLayout::{HeadMajorBnsh, SeqMajorBsnh}` enum variants implied, so the
    // descriptor migration cannot silently change a layout.
    #[test]
    fn head_major_bnsh_strides_match_legacy_enum() {
        use KvStrideDim::{HeadDim, KvHeads, SeqCapacity};
        let strides = KvCacheLayout::head_major_bnsh().resolve_strides().strides;
        assert_eq!(strides.batch, vec![KvHeads, SeqCapacity, HeadDim]);
        assert_eq!(strides.head, vec![SeqCapacity, HeadDim]);
        assert_eq!(strides.seq, vec![HeadDim]);
        assert_eq!(strides.head_dim, Vec::<KvStrideDim>::new());
    }

    #[test]
    fn seq_major_bsnh_strides_match_legacy_enum() {
        use KvStrideDim::{HeadDim, KvHeads, SeqCapacity};
        let strides = KvCacheLayout::seq_major_bsnh().resolve_strides().strides;
        assert_eq!(strides.batch, vec![SeqCapacity, KvHeads, HeadDim]);
        assert_eq!(strides.head, vec![HeadDim]);
        assert_eq!(strides.seq, vec![KvHeads, HeadDim]);
        assert_eq!(strides.head_dim, Vec::<KvStrideDim>::new());
    }

    // Existing metadata that names a layout as a bare string keeps working.
    #[test]
    fn named_layout_deserializes_from_string_alias() {
        let bnsh: KvCacheLayout = serde_json::from_str("\"head_major_bnsh\"").unwrap();
        assert_eq!(bnsh, KvCacheLayout::head_major_bnsh());
        assert_eq!(
            bnsh.resolve_strides(),
            KvStrideDescriptor::head_major_bnsh()
        );
        assert_eq!(bnsh.gqa_attribute_value(), Some(0));

        let bsnh: KvCacheLayout = serde_json::from_str("\"seq_major_bsnh\"").unwrap();
        assert_eq!(bsnh, KvCacheLayout::seq_major_bsnh());
        assert_eq!(bsnh.gqa_attribute_value(), Some(1));
    }

    // The general form round-trips and an explicit descriptor equal to a named
    // layout still maps back onto its wire attribute value.
    #[test]
    fn explicit_descriptor_round_trips() {
        let explicit = KvCacheLayout::Explicit(KvStrideDescriptor::seq_major_bsnh());
        let json = serde_json::to_string(&explicit).unwrap();
        let parsed: KvCacheLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, explicit);
        assert_eq!(parsed.gqa_attribute_value(), Some(1));
    }

    // A token-major-style view (non-zero offset, reservation override) is
    // expressible but is not one of the two attribute-expressible named layouts.
    #[test]
    fn reservation_view_has_no_wire_attribute() {
        let mut descriptor = KvStrideDescriptor::seq_major_bsnh();
        descriptor.reservation_offset_elements = 4096;
        descriptor.reservation_seq_slots = Some(1 << 20);
        let view = KvCacheLayout::Explicit(descriptor);
        assert_eq!(view.gqa_attribute_value(), None);
    }
}
