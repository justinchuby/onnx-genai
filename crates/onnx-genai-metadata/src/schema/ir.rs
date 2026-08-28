use super::*;
use std::collections::BTreeSet;

/// Typed tensor contract used at package and component boundaries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TensorContract {
    #[schemars(with = "schema_vocabulary::TensorDType")]
    pub dtype: String,
    /// Complete tensor shape. Its length is the tensor rank.
    pub shape: Vec<TensorDimension>,
    #[serde(default)]
    pub optional: bool,
    /// How this value relates to the runtime's private request/sequence table.
    ///
    /// This is a structural batching fact, never a row identity. It is the only
    /// information a runtime needs to move, split, or drop this value during
    /// compaction; scheduler slots, epochs, block tables, and sequence handles
    /// stay runtime-private.
    #[serde(default, skip_serializing_if = "BatchLayout::is_shared")]
    pub batch_layout: BatchLayout,
    /// Which dimensions of this tensor may be padded, and where the truth
    /// about how much of each entry is real is recorded.
    ///
    /// A dense tensor is the only shape a fixed-arity component can consume, so
    /// a group whose items carry different amounts of data has to be padded up
    /// to a common extent. Nothing in the padded tensor says where the real
    /// entries stop, and guessing from a sentinel is exactly the hidden
    /// heuristic this schema refuses: validity is named, typed, and validated
    /// like any other value. An empty list means the value carries no padding,
    /// and a runtime must not introduce any.
    ///
    /// One entry per padded dimension: a clip tensor padded along frames and
    /// again along patches states both, because a single companion cannot say
    /// where two independent extents end.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub padding: Vec<PaddedDimension>,
}

impl TensorContract {
    /// Tensor rank, derived exclusively from the required shape.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Whether two graph-visible contracts can share one representation
    /// without an explicit conversion component.
    pub(crate) fn representation_compatible_with(&self, other: &Self) -> bool {
        fn normalize_dtype(dtype: &str) -> &str {
            match dtype {
                "fp32" => "float32",
                "fp16" => "float16",
                "bf16" => "bfloat16",
                other => other,
            }
        }

        normalize_dtype(&self.dtype) == normalize_dtype(&other.dtype)
            && self.rank() == other.rank()
            && self.batch_layout == other.batch_layout
            && self.padding == other.padding
            && self.shape.iter().zip(&other.shape).all(|(left, right)| {
                !matches!(
                    (left, right),
                    (TensorDimension::Fixed(left), TensorDimension::Fixed(right))
                        if left != right
                )
            })
    }
}

/// One padded dimension of a value and the companion that bounds it.
///
/// The dimension is named by its shape symbol rather than by an axis index. The
/// same extent sits at different positions in different values — `patches` is
/// axis 1 of the pixel tensor and axis 0 of a pooled one — and an index would
/// name whichever the author happened to be looking at.
///
/// Padding is always appended: the real entries of a padded dimension form a
/// prefix. That is what lets one integer per enclosing entry say everything a
/// full boolean mask would say, and it is a rule rather than a convention
/// because left padding would shift every position-dependent computation and an
/// interior hole cannot be expressed by a length at all.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PaddedDimension {
    /// Shape symbol of the padded dimension of the owning value.
    #[schemars(length(min = 1))]
    pub dimension: String,
    /// Value giving how many leading entries of `dimension` are real, one entry
    /// per position of the axes outer to it.
    ///
    /// It resolves in the namespace of the contract's owner: a sibling port for
    /// a component port, a workflow value for a workflow input, output, or
    /// state cell.
    #[schemars(length(min = 1))]
    pub valid_lengths: String,
}

/// Structural relationship between a typed value and the runtime batch.
///
/// `shared` values are invariant across requests. `request_aligned` values carry
/// exactly one entry per in-flight request along `axis`, so compaction permutes
/// that axis. `token_packed` values are ragged: the items of every request are
/// concatenated along one physical axis, and the ownership companions say which
/// request each item came from, which is what lets a runtime split and regroup
/// the packed value without any serialized request ID. `runtime_sequence_state`
/// marks a value whose per-sequence storage the runtime owns outright.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BatchLayout {
    #[default]
    Shared,
    RequestAligned {
        axis: usize,
    },
    /// Each request owns a fixed-size contiguous group on ``axis`` (for
    /// example conditional/unconditional classifier-free-guidance rows).
    RequestExpanded {
        axis: usize,
        /// Number of contiguous physical rows owned by each logical request.
        /// Validation requires this to be at least one.
        factor: usize,
    },
    /// Items belonging to many requests concatenated along one axis.
    ///
    /// There is exactly one packed axis. Structure above the item — frames
    /// inside clips inside request rows — is ownership, not a second axis: the
    /// tensor stays a flat run of items and `levels` says how to fold that run
    /// back into requests.
    TokenPacked {
        /// Packed axis of this value.
        ///
        /// Validation requires axis zero. A run of items is only splittable
        /// into per-request pieces without copying when the items are the
        /// outermost, contiguous stride of the tensor, and a packed axis
        /// anywhere else would silently turn every split into a gather.
        axis: usize,
        /// Ownership chain of the packed axis, innermost level first.
        ///
        /// Level zero owns the physically packed positions; the last level owns
        /// into request rows. One level is the ordinary flat case — items in
        /// rows — and two states one grouping in between, as frames in clips in
        /// rows. Validation rejects a third: a runtime walks the whole chain on
        /// every split, and each level multiplies the states that have to be
        /// checked and tested.
        #[schemars(length(min = 1, max = 2))]
        levels: Vec<OwnershipLevel>,
    },
    RuntimeSequenceState,
}

/// One level of a packed value's ownership chain.
///
/// A level is a pair, never a tensor axis: `offsets` gives the exclusive prefix
/// offset of each parent's run and `owner` gives the owning position of each
/// entry at this level. The two together are what a runtime walks to answer
/// "which request does this item belong to".
///
/// Both are `shared`, never `request_aligned`. An exclusive prefix sum is not
/// permutation-followable — permuting rows does not permute a prefix-offset
/// vector, it invalidates it — so a runtime that compacts rebuilds the chain
/// rather than gathering it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OwnershipLevel {
    /// Value holding the exclusive prefix offset of each parent's run at this
    /// level, with a final entry holding the total. Its extent is the parent
    /// count plus one.
    #[schemars(length(min = 1))]
    pub offsets: String,
    /// Value mapping each entry at this level to the position of its parent.
    /// Its extent is this level's own unit count, which at level zero is the
    /// packed extent.
    #[schemars(length(min = 1))]
    pub owner: String,
    /// Where this level's unit count comes from, for a value a component
    /// produces.
    ///
    /// Absent is right for a value a component consumes, whose every count the
    /// caller assembled. An output states it per level, because the levels of
    /// one chain do not answer together: a token-merging encoder decides how
    /// many tokens each clip becomes while leaving which clip belongs to which
    /// request exactly as it found it. A single answer for the whole chain
    /// could only be wrong at one end or the other.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extent: Option<PackedExtent>,
}

/// Where the unit count of one ownership level comes from.
///
/// Neither answer is derivable from the contract: an output of the same rank
/// and symbols as its input may be a per-item transform or a token merger, and
/// the two split at completely different boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PackedExtent {
    /// The level's units correspond one-to-one, in order, with an input
    /// level's, so the output reuses that input's companions unchanged. A
    /// component may drop inner levels it has consumed; it may not invent a
    /// correspondence it does not have.
    Preserved,
    /// The graph decides the extent, so the level's companions are outputs of
    /// the same component. An input's offsets describe an extent this value
    /// does not have.
    Produced,
}

impl PackedExtent {
    /// Serialized spelling, for diagnostics that name what a document declared.
    pub fn name(self) -> &'static str {
        match self {
            Self::Preserved => "preserved",
            Self::Produced => "produced",
        }
    }
}

impl BatchLayout {
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared)
    }

    /// Axis permuted when the runtime compacts the batch, if any.
    pub fn request_axis(&self) -> Option<usize> {
        match self {
            Self::RequestAligned { axis } | Self::RequestExpanded { axis, .. } => Some(*axis),
            Self::Shared | Self::TokenPacked { .. } | Self::RuntimeSequenceState => None,
        }
    }

    /// Number of contiguous tensor rows owned by each request.
    pub fn request_expansion_factor(&self) -> usize {
        match self {
            Self::RequestExpanded { factor, .. } => *factor,
            _ => 1,
        }
    }

    /// Axis along which items belonging to many requests are concatenated.
    ///
    /// This counts items, never request rows. The two are different numbers as
    /// soon as one request contributes more than one item, so nothing that
    /// reasons about rows — row scope, compaction selections, per-row state —
    /// may read this as a row axis.
    pub fn packed_axis(&self) -> Option<usize> {
        match self {
            Self::TokenPacked { axis, .. } => Some(*axis),
            _ => None,
        }
    }

    /// Ownership chain of a packed value, innermost level first.
    pub fn levels(&self) -> &[OwnershipLevel] {
        match self {
            Self::TokenPacked { levels, .. } => levels,
            _ => &[],
        }
    }

    /// Every companion value this layout names, innermost level first.
    pub fn companions(&self) -> Vec<(usize, &'static str, &str)> {
        self.levels()
            .iter()
            .enumerate()
            .flat_map(|(index, level)| {
                [
                    (index, "offsets", level.offsets.as_str()),
                    (index, "owner", level.owner.as_str()),
                ]
            })
            .collect()
    }

    /// Values the innermost level needs to map items back to owners.
    pub fn packing(&self) -> Option<(&str, &str)> {
        self.levels()
            .first()
            .map(|level| (level.offsets.as_str(), level.owner.as_str()))
    }

    /// Whether the runtime must move this value when it compacts or releases rows.
    pub fn is_row_scoped(&self) -> bool {
        !matches!(self, Self::Shared)
    }

    /// Serialized `kind` of this layout, for diagnostics that name what a
    /// document declared rather than how the reader spells it.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::RequestAligned { .. } => "request_aligned",
            Self::RequestExpanded { .. } => "request_expanded",
            Self::TokenPacked { .. } => "token_packed",
            Self::RuntimeSequenceState => "runtime_sequence_state",
        }
    }
}

/// Explicit input/output ports of one executable component.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComponentPorts {
    #[serde(default)]
    pub inputs: BTreeMap<String, TensorContract>,
    #[serde(default)]
    pub outputs: BTreeMap<String, TensorContract>,
    /// Semantic role of the ports whose meaning the workflow's own structure
    /// cannot recover, keyed by port name.
    ///
    /// State ports never need an entry: a state group already names its
    /// per-component `(input, output)` pair, and the fixed-capacity control
    /// ports are named by [`StateUpdate::IndexedScatter`]. What is left is the
    /// per-step dataflow a workflow binds by SSA value, where the binding
    /// records WHICH value reaches a port but not WHAT the component does with
    /// it. A runtime that specializes a decode step — packing tokens, reusing a
    /// logits buffer, skipping a mask it can prove is causal — needs that
    /// second fact, and inferring it from a port's spelling is exactly the
    /// name-guessing this schema refuses everywhere else.
    ///
    /// Roles are architecture-neutral and describe the port, never the model
    /// family that happens to expose it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub roles: BTreeMap<String, PortRole>,
}

/// Architecture-neutral semantic role of one component port.
///
/// This vocabulary names what a value MEANS to the component that consumes or
/// produces it. It deliberately excludes anything a state group already
/// declares, so a role and a state binding can never disagree about the same
/// port.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PortRole {
    /// Discrete token identifiers driving autoregressive execution.
    TokenIds,
    /// Pre-embedded sequence, used when another component owns the embedding.
    InputsEmbeds,
    /// Attention mask over the sequence.
    AttentionMask,
    /// Per-position indices used by position embedding.
    PositionIds,
    /// Unnormalized next-token scores.
    Logits,
    /// Per-token hidden states exposed as a distinct output.
    HiddenStates,
    /// Encoder result consumed by a cross-attending decoder.
    EncoderHiddenStates,
    /// Encoded audio features consumed by a speech decoder.
    AudioFeatures,
}

/// How multiple dataflow values are combined at one destination.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReducerKind {
    First,
    Last,
    Sum,
    Product,
    Mean,
    Min,
    Max,
    Concat,
    Stack,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReducerSpec {
    pub kind: ReducerKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum ScalarValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

/// A literal tensor initializer.
///
/// A single scalar broadcasts to every element of the declared contract, which
/// covers flags, counters, and zero-filled buffers. Workflows whose constants
/// are genuinely per-position -- interleaved stream delay patterns, per-stream
/// initial tokens, fixed schedule tables -- declare the elements explicitly in
/// row-major order instead, so the value stays inside the metadata document and
/// does not become an out-of-band artifact.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum LiteralValue {
    Scalar(ScalarValue),
    Elements(Vec<ScalarValue>),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Cpu,
    Cuda,
    DirectMl,
    CoreMl,
    WebGpu,
    Npu,
}

/// Sound, component-centric workflow IR. Tensor math lives in invoked components.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowSpec {
    pub manifest: WorkflowManifest,
    #[serde(default)]
    pub inputs: BTreeMap<String, WorkflowInput>,
    #[serde(default)]
    pub outputs: BTreeMap<String, WorkflowOutput>,
    pub components: BTreeMap<String, WorkflowComponent>,
    #[serde(default)]
    pub state: BTreeMap<String, WorkflowStateCell>,
    /// Retry and speculation semantics of every declared external effect domain.
    #[serde(default)]
    pub effects: BTreeMap<String, EffectContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving: Option<ServingServiceContract>,
    pub steps: Vec<WorkflowStep>,
}

/// Declared semantics of one external effect domain.
///
/// Retry class and speculation safety are independent axes. A transactional
/// effect may still be unsafe to speculate, and an idempotent effect is not
/// automatically rewindable.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EffectContract {
    /// Minimum retry-relevant class for runtime/server recovery orchestration.
    pub retry: EffectRetryClass,
    /// Whether and how far this effect can participate in speculative execution.
    #[serde(default)]
    pub speculation_safety: SpeculationSafety,
}

/// Retry-relevant classification of an external effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EffectRetryClass {
    /// No observable external effect; replay is always safe.
    Pure,
    /// Repeating the effect with the same inputs is observationally equivalent.
    Idempotent,
    /// The effect participates in an external transaction that can be aborted.
    Transactional,
    /// The effect must never be repeated or rolled back.
    NonRetryable,
}

/// Whether an effect may be executed inside a speculative region.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpeculationSafety {
    /// The effect must not run speculatively.
    #[default]
    None,
    /// The effect's observable state can be cloned before a speculative region.
    Clonable,
    /// The effect can be rewound by at most `max_depth` proposed positions.
    Rewindable { max_depth: usize },
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterServiceContract {
    /// Authoritative, architecture-neutral bindings resolved by producer/import tooling.
    pub target_manifest: LoraTargetManifest,
    /// Explicit load-time tooling fallback. Runtime execution never guesses targets.
    #[serde(default)]
    pub discovery_fallback: AdapterDiscoveryFallback,
    /// Request-scoped adapter-set inputs. These are immutable SSA inputs for one request.
    pub selection: AdapterSelectionContract,
    /// Generic application capability required from the runtime or execution provider.
    pub application_capability: String,
    #[serde(default)]
    pub portable_fallback: bool,
    #[serde(default)]
    pub artifacts: BTreeMap<String, AdapterArtifact>,
    #[serde(default)]
    pub cache: AdapterCacheContract,
    #[serde(default)]
    pub planning: AdapterPlanningContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterSelectionContract {
    /// Int64[batch,max_adapters] segment IDs in composition order.
    pub segments: String,
    /// Int64[batch] number of valid adapter IDs in each row.
    pub adapter_counts: String,
    /// Float32[batch,max_adapters] effective request scales.
    pub scales: String,
    /// Optional bool[batch]; inactive rows never load or apply adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<String>,
    /// Fixed second dimension of segments and scales.
    pub max_adapters: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterArtifact {
    /// Stable non-negative wire ID used by selection.segments.
    pub index: usize,
    pub identity: String,
    pub version: String,
    pub rank: usize,
    pub alpha: f64,
    #[schemars(with = "schema_vocabulary::TensorDType")]
    pub dtype: String,
    #[serde(default)]
    pub weights: Vec<AdapterWeightArtifact>,
    #[serde(default)]
    pub bindings: Vec<AdapterTargetBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<AdapterProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterWeightArtifact {
    pub location: String,
    /// Loader capability required to normalize this source into the canonical artifact.
    pub loader_capability: String,
    /// PEFT `adapter_config.json` paired with the safetensors file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_location: Option<String>,
    /// Whether alpha/rank remains to be applied or is already baked into B.
    pub scale_encoding: AdapterScaleEncoding,
    #[serde(default)]
    pub format: AdapterWeightFormat,
}

#[derive(
    Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum AdapterWeightFormat {
    /// Portable JSON tensor bundle used by the reference fallback.
    #[default]
    Json,
    /// Native ONNX Runtime GenAI adapter bundle.
    OrtGenai,
    /// Hugging Face PEFT `adapter_config.json` plus safetensors.
    HfPeft,
    /// Manifest-keyed safetensors parameter bundle.
    Safetensors,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AdapterScaleEncoding {
    /// Runtime applies the binding/artifact `alpha / rank` factor.
    AlphaOverRank,
    /// Source factors already encode the complete static scale (ORT `TORT` convention).
    Baked,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterTargetBinding {
    /// Stable target ID declared by the authoritative target manifest.
    pub target: String,
    /// Key used to find this target's A/B tensors in the adapter bundle.
    pub weight_key: String,
    /// Per-target rank override; absent uses the artifact rank.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    /// Per-target alpha override; absent uses the artifact alpha.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
}

/// Authoritative generic target map migrated from Phase-2 `LoraTargetManifest`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoraTargetManifest {
    pub targets: Vec<LoraTargetDescriptor>,
}

/// One resolved base projection. Fused-QKV knowledge is lowered to an optional slice.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoraTargetDescriptor {
    pub id: String,
    /// Workflow component name, or `model` for a bare decoder package.
    pub component: String,
    /// Exact immutable base initializer name.
    pub initializer: String,
    /// Optional producer/importer layer identity retained from Phase-2 manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_index: Option<usize>,
    /// Exact ONNX projection node name used for load-time manifest validation.
    pub node_name: String,
    /// Exact graph value produced by the projection.
    pub output_name: String,
    /// Projection activation dtype used by graph-native delta application.
    #[schemars(with = "schema_vocabulary::TensorDType")]
    pub activation_dtype: String,
    pub input_features: usize,
    pub output_features: usize,
    /// Optional rank policy for artifacts binding this target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    /// Optional alpha policy for artifacts binding this target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    /// Resolved child range within a fused output; producer/import tooling owns discovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_slice: Option<LoraTargetSlice>,
    /// Phase-1 graph-native optional A/B inputs. Base-only omits both and is bit-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_inputs: Option<LoraGraphInputBinding>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoraTargetSlice {
    /// Producer-defined semantic label; runtime execution uses only the resolved range.
    pub role: String,
    pub offset: usize,
    pub width: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoraGraphInputBinding {
    pub a: String,
    pub b: String,
    /// Optional graph input for request scale; otherwise scale is folded into stable factors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterProvenance {
    /// Producer/importer that resolved the manifest and normalized the artifact.
    pub producer: String,
    /// Source model or adapter URI without credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Immutable source revision, commit, or content identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AdapterDiscoveryFallback {
    #[default]
    Disabled,
    /// Tooling/load-time graph discovery may produce a resolved manifest; execution may not guess.
    ToolingOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterCacheContract {
    #[serde(default = "default_adapter_cache_entries")]
    pub max_entries: usize,
    #[serde(default)]
    pub eviction: AdapterEvictionPolicy,
}

impl Default for AdapterCacheContract {
    fn default() -> Self {
        Self {
            max_entries: default_adapter_cache_entries(),
            eviction: AdapterEvictionPolicy::default(),
        }
    }
}

fn default_adapter_cache_entries() -> usize {
    16
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AdapterEvictionPolicy {
    #[default]
    Lru,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterPlanningContract {
    #[serde(default = "default_true")]
    pub bucket_by_adapter_set: bool,
    #[serde(default = "default_true")]
    pub stable_buffers: bool,
    #[serde(default = "default_true")]
    pub invalidate_capture_on_eviction: bool,
}

impl Default for AdapterPlanningContract {
    fn default() -> Self {
        Self {
            bucket_by_adapter_set: true,
            stable_buffers: true,
            invalidate_capture_on_eviction: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowManifest {
    #[serde(default)]
    pub adapter_abis: BTreeMap<String, String>,
    #[serde(default)]
    pub capabilities: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowInput {
    pub contract: TensorContract,
    pub role: SemanticInputRole,
    pub source: WorkflowInputSource,
    #[serde(default = "default_true")]
    #[schemars(default = "default_true")]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<LiteralValue>,
    /// Initial scalar bool SSA value indicating whether the caller supplied this input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub present_as: Option<String>,
    /// Whether an application may supply a previously computed typed value in
    /// place of recomputing it, such as a cached encoder result.
    ///
    /// Transport, remote caching, and identity of the supplied value remain
    /// runtime-owned; metadata only declares that the substitution is legal.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub externally_suppliable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SemanticInputRole {
    Runtime {
        version: String,
        role: RuntimeInputRole,
    },
    Opaque,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeInputRole {
    PromptText,
    PromptTokens,
    NegativePromptText,
    NegativePromptTokens,
    Media,
    MaxIterations,
    MaxOutputTokens,
    /// Token ids that end generation.
    ///
    /// The request value overrides
    /// `package.tokenizer.special_tokens.eos_token_id`. It never carries an
    /// authored package default.
    EosTokenIds,
    /// Number of valid entries in each row of a padded EOS-id tensor.
    EosTokenLengths,
    Seed,
    GuidanceScale,
    Width,
    Height,
    DenoisingStrength,
    SamplingTemperature,
    SamplingTopK,
    SamplingTopP,
    SamplingMinP,
    Constraint,
    SessionId,
    /// Runtime-minted gather of source batch positions for beam or speculative
    /// row expansion. Values are positions inside the current batch, never
    /// scheduler slots, request IDs, or epochs.
    RowSelection,
    AdapterSegments,
    AdapterCounts,
    AdapterScales,
    AdapterActive,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkflowInputSource {
    Request,
    Application { name: String },
    Literal,
    Artifact { path: String },
}

#[derive(Debug, Clone, PartialEq, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowOutput {
    pub contract: TensorContract,
    pub role: WorkflowOutputRole,
    /// The sole publication family for this output boundary.
    ///
    /// The default preserves documents authored before output protocols were
    /// introduced. New schema-versioned documents must state this field; see
    /// the parser's version gate.
    #[serde(default)]
    pub family: WorkflowOutputFamily,
    /// Whether `family` was explicitly authored rather than supplied by the
    /// pre-v1.5 compatibility default.
    #[doc(hidden)]
    #[schemars(skip)]
    pub family_authored: bool,
    pub value_range: Option<PixelValueRange>,
    pub stage: OutputStage,
    /// Concrete media delivery contract for a post-processing output.
    ///
    /// Tensor shape alone cannot distinguish PCM samples from encoded WAV bytes,
    /// nor can it carry the sample rate and channel count required by an audio
    /// serving API. This remains architecture-neutral and intentionally contains
    /// no model-family identifiers or artifact fingerprints.
    pub media: Option<MediaOutputContract>,
}

/// Deserialization preserves the pre-output-protocol materialized behavior for
/// documents whose schema version predates this field. The generated schema
/// intentionally remains stricter: versioned output-protocol documents must
/// state their family, which the parser enforces before this compatibility
/// default is applied.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkflowOutputWire {
    contract: TensorContract,
    role: WorkflowOutputRole,
    #[serde(default)]
    family: AuthoredWorkflowOutputFamily,
    #[serde(default)]
    value_range: Option<PixelValueRange>,
    stage: OutputStage,
    #[serde(default)]
    media: Option<MediaOutputContract>,
}

#[derive(Default)]
struct AuthoredWorkflowOutputFamily {
    value: WorkflowOutputFamily,
    authored: bool,
}

impl<'de> Deserialize<'de> for AuthoredWorkflowOutputFamily {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            value: WorkflowOutputFamily::deserialize(deserializer)?,
            authored: true,
        })
    }
}

impl<'de> Deserialize<'de> for WorkflowOutput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = WorkflowOutputWire::deserialize(deserializer)?;
        Ok(Self {
            contract: wire.contract,
            role: wire.role,
            family: wire.family.value,
            family_authored: wire.family.authored,
            value_range: wire.value_range,
            stage: wire.stage,
            media: wire.media,
        })
    }
}

impl Serialize for WorkflowOutput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct as _;

        let mut fields = 3;
        fields += usize::from(self.family_authored);
        fields += usize::from(self.value_range.is_some());
        fields += usize::from(self.media.is_some());
        let mut output = serializer.serialize_struct("WorkflowOutput", fields)?;
        output.serialize_field("contract", &self.contract)?;
        output.serialize_field("role", &self.role)?;
        if self.family_authored {
            output.serialize_field("family", &self.family)?;
        }
        if let Some(value_range) = &self.value_range {
            output.serialize_field("value_range", value_range)?;
        }
        output.serialize_field("stage", &self.stage)?;
        if let Some(media) = &self.media {
            output.serialize_field("media", media)?;
        }
        output.end()
    }
}

/// Publication semantics selected once for a workflow output.
///
/// This classification describes workflow semantics only. Adapters render the
/// resulting publications separately and cannot change an allowed operation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkflowOutputFamily {
    /// One value, replaced or grown along a declared axis.
    #[default]
    Materialized,
    /// Ordered, discrete typed occurrences.
    Events,
    /// A stream of typed, transaction-addressable revisions.
    Revisions {
        /// Exact revision-envelope protocol version.
        #[schemars(length(min = 1))]
        version: String,
    },
}

impl WorkflowOutputFamily {
    /// The exact typed-revision protocol version, when this is a revision
    /// output. Materialized values and discrete events have no revision
    /// envelope.
    pub fn revision_version(&self) -> Option<&str> {
        match self {
            Self::Revisions { version } => Some(version),
            Self::Materialized | Self::Events => None,
        }
    }
}

/// Numeric interpretation of pixels emitted by an image or video workflow
/// output.
///
/// A frame carries the same pixels a still image does, so both output roles
/// read their normalization from one contract rather than from two vocabularies
/// that could disagree.
///
/// This is an output contract, not a model-family hint: consumers must never
/// infer normalization from observed pixel values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PixelValueRange {
    ZeroToOne,
    NegativeOneToOne,
    #[serde(rename = "zero_to_255")]
    ZeroTo255,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowOutputRole {
    Tokens,
    Text,
    Image,
    /// A frame sequence. Distinct from `Image` because a consumer has to know
    /// the value carries a temporal axis and may be published incrementally.
    Video,
    Audio,
    Tensor,
    Event,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MediaOutputContract {
    pub container: MediaContainer,
    pub encoding: MediaEncoding,
    /// Sample rate of the encoded response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    /// Sample rate of a pre-adapter waveform. When it differs from
    /// `sample_rate_hz`, the API boundary resamples before encoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_sample_rate_hz: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<u16>,
    #[serde(default)]
    pub delivery: MediaDelivery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MediaContainer {
    Raw,
    Wav,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MediaEncoding {
    PcmS16Le,
    PcmF32Le,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MediaDelivery {
    #[default]
    Buffered,
    Streaming,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OutputStage {
    PreAdapter,
    PostAdapter,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowComponent {
    pub implementation: ComponentImplementation,
    #[serde(default)]
    pub ports: ComponentPorts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract: Option<ComponentContract>,
    /// Allow an application to select another declared component with the same
    /// versioned contract ABI for this invocation.
    #[serde(default)]
    pub application_overridable: bool,
    #[serde(default)]
    pub effects: Vec<String>,
    /// Declared per-request row scope of this component's private state.
    ///
    /// A component with row scope must implement the mandatory row ABI
    /// (`compact(selection)` and `release(row)`). This is an ABI invariant, not
    /// a negotiated capability: a runtime may not load a package whose
    /// row-scoped component cannot be compacted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_scope: Option<ComponentRowScope>,
    /// Non-dataflow facts that change this component's observable state.
    ///
    /// Cache correctness dependencies of ONNX components are derived from the
    /// workflow SSA graph. Native and external components must declare any
    /// additional state they read that is not visible as a typed input.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub cache_affects_state: BTreeSet<String>,
    /// How much one invocation of this component may carry.
    ///
    /// Absence means one request row per invocation: a runtime that wants to
    /// serve two requests calls the component twice. That is the safe reading
    /// for every component whose producer has not thought about batching, so it
    /// is the default. A component that declares a capacity is stating that its
    /// artifact accepts several contributions stacked on one axis, which is the
    /// fact a scheduler needs before it may coalesce work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_capacity: Option<ComponentBatchCapacity>,
}

/// Batching capacity of one invocation of a component.
///
/// This describes the shape of an invocation, never a scheduling policy: which
/// dimensions must already agree before two contributions can share a call, and
/// how much the assembled call may materialize. A runtime decides whether to
/// group at all, and may always group fewer; the package only says what its
/// artifact tolerates.
///
/// Everything here is keyed by shape symbol rather than by axis index. Ports of
/// one component routinely differ in rank — a rank-3 payload, a rank-1
/// companion, a rank-2 pooled output — so an axis index is only meaningful
/// relative to one port, while a symbol names the same quantity on every port
/// that mentions it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComponentBatchCapacity {
    /// Shape symbols whose extents every item in a group must agree on.
    ///
    /// A symbol here names a property of one item — a patch width, a mel bin, a
    /// frame count of a fixed-length clip — that the artifact cannot vary
    /// within a single call. Items that disagree on it cannot share an
    /// invocation, so a scheduler splits them into separate groups.
    ///
    /// It may not name a count: not the extent of a packed axis, and not the
    /// unit or run count of an ownership level. Those are the numbers a packed
    /// layout exists to let vary per request, and pinning one would describe a
    /// fixed-shape batch the package did not declare. A video whose frame count
    /// really is fixed is expressed by pinning the frame dimension of an
    /// ordinary request-aligned tensor and declaring no frame ownership level
    /// at all.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uniform_dimensions: Vec<String>,
    /// Upper bounds on what one assembled invocation materializes, keyed by
    /// shape symbol.
    ///
    /// These are static geometry — a fixed position table, an exported
    /// constant, a kernel bound — never a measured throughput sweet spot, which
    /// would be a cost model and belongs to the runtime. Every entry is an
    /// upper bound and never an obligation: a runtime may group fewer,
    /// including one. An empty list bounds nothing beyond what the runtime's
    /// own row budget already bounds.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub budgets: Vec<CapacityBudget>,
}

/// One materialized-footprint bound of an assembled invocation.
///
/// A budget bounds what the invocation materializes, not what is nominally
/// valid. A packed dimension's footprint is the sum of the items' valid extents,
/// which is exactly the packed extent because packing stores no padding; a
/// padded dimension's footprint is the enclosing count times the padded extent,
/// which is the rectangle the runtime allocates and the kernel reads. Budgeting
/// the valid sum instead would let one long and fifteen short items pass a bound
/// the group then blows through.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CapacityBudget {
    /// Shape symbols whose materialized extents multiply to the bounded
    /// quantity, in order.
    ///
    /// One symbol bounds that dimension directly — items at an ownership level,
    /// positions on a packed axis. Several bound an activation-shaped quantity,
    /// such as total padded patch slots, without the package ever naming bytes.
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub dimensions: Vec<String>,
    /// Largest product of those extents one invocation may carry.
    #[schemars(range(min = 1))]
    pub max_total: usize,
}

/// Row scope of a component's runtime-private state.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComponentRowScope {
    /// Batch axis of the component's row-scoped ports.
    pub axis: usize,
    /// Whether the component retains state between invocations for each row.
    #[serde(default)]
    pub stateful: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComponentContract {
    /// Versioned semantic capability identifier. It never selects execution behavior.
    pub id: String,
    pub version: String,
    /// How closely a substituted implementation must reproduce this contract.
    ///
    /// A runtime may freely choose any equivalent implementation, but the
    /// declared class bounds what "equivalent" means. Only a
    /// `distribution_preserving` (or `bitwise`) contract may be optimized
    /// speculatively without caller opt-in.
    #[serde(default)]
    pub equivalence: EquivalenceClass,
    /// Semantic role to concrete component port name.
    #[serde(default)]
    pub bindings: BTreeMap<String, String>,
    /// Contract parameters that are not tensor ports, such as adapter actions.
    #[serde(default)]
    pub parameters: BTreeMap<String, ScalarValue>,
}

/// Correctness bound on substituting an equivalent component implementation.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum EquivalenceClass {
    /// Every substituted implementation must produce bit-identical outputs.
    Bitwise,
    /// Outputs may differ numerically but must preserve the output distribution.
    DistributionPreserving,
    /// Only the declared semantics are preserved; the distribution may change.
    #[default]
    Semantic,
}

impl EquivalenceClass {
    /// Whether a runtime may auto-enable speculative optimization of this contract.
    ///
    /// Semantic equivalence permits an implementation whose output distribution
    /// differs, so speculation would silently change results and requires an
    /// explicit caller opt-in.
    pub fn permits_automatic_speculation(self) -> bool {
        matches!(self, Self::Bitwise | Self::DistributionPreserving)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComponentImplementation {
    Onnx {
        artifact: String,
    },
    Adapter {
        abi: String,
        version: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        artifact: Option<String>,
    },
    Binding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectTransition {
    pub consumes: String,
    pub produces: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkflowStep {
    Sequence {
        steps: Vec<WorkflowStep>,
    },
    Invoke {
        component: String,
        #[serde(default)]
        inputs: BTreeMap<String, String>,
        #[serde(default)]
        outputs: BTreeMap<String, String>,
    },
    Loop {
        #[serde(default)]
        setup: Vec<WorkflowStep>,
        steps: Vec<WorkflowStep>,
        continue_when: String,
        max_iterations: String,
        #[serde(default)]
        termination: WorkflowLoopTermination,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        iteration: Option<Box<WorkflowLoopIteration>>,
        #[serde(default)]
        carried: Vec<WorkflowCarry>,
    },
    Branch {
        predicate: String,
        cases: BTreeMap<String, WorkflowStep>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<Box<WorkflowStep>>,
        #[serde(default)]
        outputs: BTreeMap<String, WorkflowBranchOutput>,
    },
    Emit {
        /// SSA value published by a value-carrying operation. `retract` and
        /// `finalize` carry no value, so this is absent for those operations.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        when: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        valid_length: Option<String>,
        output: String,
        /// Logical stream within the declared output. Omission selects the
        /// output's default stream; it never selects an adapter.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[schemars(length(min = 1))]
        stream: Option<String>,
        mode: WorkflowEmitMode,
        /// Axis along which the output grows; defaults to the final axis.
        ///
        /// A rank-four or deeper value must name it: the final axis of a media
        /// tensor is a spatial extent, never the one an append concatenates.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        axis: Option<usize>,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowLoopTermination {
    #[default]
    Predicate,
    GenerationEos,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowCarry {
    pub cell: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial: Option<String>,
    pub next: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorkflowNode {
    Sequence {
        nodes: Vec<WorkflowNode>,
    },
    Invoke {
        component: String,
        inputs: BTreeMap<String, String>,
        outputs: BTreeMap<String, String>,
        effects: BTreeMap<String, EffectTransition>,
    },
    Loop {
        setup: Box<WorkflowNode>,
        body: Box<WorkflowNode>,
        continue_when: String,
        max_iterations: String,
        termination: WorkflowLoopTermination,
        /// Optional zero-based induction value, scoped to this loop's body and continue_when.
        iteration: Option<Box<WorkflowLoopIteration>>,
        carried: Vec<WorkflowLoopCarry>,
        effects: BTreeMap<String, WorkflowLoopEffect>,
    },
    Branch {
        predicate: String,
        cases: BTreeMap<String, WorkflowNode>,
        default: Option<Box<WorkflowNode>>,
        outputs: BTreeMap<String, WorkflowBranchOutput>,
        effects: BTreeMap<String, WorkflowBranchEffectMerge>,
    },
    Emit {
        value: String,
        /// Optional scalar or rank-one boolean guard. False rows are suppressed.
        when: Option<String>,
        /// Optional scalar or rank-one integer SSA value limiting the emitted prefix
        /// on the value's growth axis, globally or per batch row.
        valid_length: Option<String>,
        output: String,
        stream: Option<String>,
        mode: WorkflowEmitMode,
        /// Axis along which an appended or length-limited output grows.
        ///
        /// Defaults to the final axis, which is where a token sequence grows. A
        /// value whose sequence axis sits elsewhere - video frames in
        /// `[batch, channels, frames, height, width]`, for instance - names it
        /// here so incremental publication does not concatenate the wrong axis.
        /// A rank-four or deeper value has no defensible default and must say.
        axis: Option<usize>,
        effect_name: String,
        effect: EffectTransition,
    },
    Transfer {
        input: String,
        output: String,
        device: DeviceKind,
    },
    /// Planner-lowered optimizer/capture unit. Never serialized.
    ExecutionIsland {
        id: usize,
    },
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowLoopIteration {
    /// SSA value containing the current zero-based iteration.
    pub value: String,
    /// `int64` scalar or rank-one broadcast contract.
    pub contract: TensorContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowBranchOutput {
    pub cases: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowBranchEffectMerge {
    pub incoming: String,
    pub cases: BTreeMap<String, String>,
    pub default: Option<String>,
    pub produces: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowLoopCarry {
    pub cell: String,
    pub current: String,
    pub current_source: WorkflowLoopCarrySource,
    pub body_input: String,
    pub body_output: String,
    pub next: String,
    pub read_effect: EffectTransition,
    pub write_effect: EffectTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowLoopCarrySource {
    Initializer,
    Explicit,
    PriorState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowLoopEffect {
    pub incoming: String,
    pub body_input: String,
    pub body_output: String,
    pub produces: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowEmitMode {
    Replace,
    Append,
    Event,
    Retract,
    Finalize,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowStateCell {
    pub contract: TensorContract,
    #[serde(default)]
    pub class: WorkflowStateClass,
    pub scope: WorkflowStateScope,
    pub initializer: String,
    pub recurrence: ShapeRecurrence,
    /// Who owns the physical storage of this cell.
    ///
    /// Workflow-managed cells follow ordinary SSA liveness. Runtime-managed and
    /// external cells must also declare a logical release boundary so a runtime
    /// knows when the semantic value stops being reachable.
    #[serde(default)]
    pub management: StateManagement,
    /// Logical point at which the runtime may release this cell's storage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_boundary: Option<StateReleaseBoundary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_group: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionLeaseContract>,
}

/// Storage ownership of one workflow state cell.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StateManagement {
    /// Ordinary SSA-managed value; liveness is derived from the workflow graph.
    #[default]
    Workflow,
    /// The runtime owns the physical storage and its allocation policy.
    Runtime,
    /// An external service owns the storage; only the typed handle is portable.
    External,
}

/// Logical boundary after which a managed or external state cell is releasable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StateReleaseBoundary {
    /// Releasable when the invocation that created it completes.
    Invocation,
    /// Releasable when the session that owns it ends.
    Session,
    /// Releasable when its owning batch row is released.
    Row,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStateClass {
    #[default]
    Semantic,
    Advisory,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStateScope {
    Invocation,
    Session,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ShapeRecurrence {
    Invariant,
    Growing {
        axis: usize,
        increment: String,
        max: String,
    },
    /// The selected axis may grow or shrink between iterations, but never exceed `max`.
    Bounded {
        axis: usize,
        max: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionLeaseContract {
    #[serde(default)]
    pub policy: SessionMutationPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    #[serde(default)]
    pub optimistic_metadata_version: bool,
    /// How this cell continues one invocation's work into the next.
    ///
    /// Absent means the lease is opaque: the runtime keeps the cell's value for
    /// the session and hands it back as the cell's value, which is all a cell
    /// consumed by the workflow's own steps needs. A cell the workflow never
    /// reads — a conversation the *request binding* has to carry — has no such
    /// reader, so it states which binding continues it here rather than leaving
    /// a runtime to guess.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation: Option<SessionContinuation>,
}

/// How a session-scoped cell rejoins the next invocation.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionContinuation {
    /// The cell holds every token this session has seen, and each invocation's
    /// prompt continues it.
    ///
    /// The value bound to `prompt_input` is the cell's value followed by the
    /// caller's tokens; when the invocation completes, the cell becomes that
    /// concatenation followed by the tokens published to `tokens_output`. A
    /// session that holds no value yet contributes nothing, so the first turn
    /// of a conversation and a request with no session are the same execution.
    ///
    /// This is what a package whose prefill starts from empty state declares:
    /// the conversation is carried as tokens the next prefill consumes, not as
    /// a cache the next prefill would have to be re-authored to accept.
    PromptPrefix {
        /// Workflow input carrying the `prompt_tokens` runtime role.
        prompt_input: String,
        /// Workflow output carrying the `tokens` role this cell accumulates.
        tokens_output: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SessionMutationPolicy {
    #[default]
    Exclusive,
    CopyOnWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ServingServiceContract {
    pub active: String,
    pub done: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_len: Option<String>,
    /// Semantic state groups whose graph ABI the runtime must honor.
    pub state_service: StateServiceContract,
}

/// Semantic model-state contract.
///
/// This declares what the state *means* and which graph ABI facts constrain the
/// runtime. It never selects paged, shared-buffer, or separate storage, a slot
/// allocation algorithm, a compaction algorithm, or a device.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateServiceContract {
    #[serde(default)]
    pub groups: BTreeMap<String, StateGroupContract>,
}

/// One semantic group of model state sharing a kind, geometry, and graph ABI.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateGroupContract {
    /// Semantic kind of the state in this group.
    pub kind: StateKind,
    /// Axis whose extent represents logical sequence positions.
    ///
    /// Required for sequence-growing and indexed-scatter state. Fixed-size
    /// recurrent state updated by replacement has no logical sequence extent
    /// and therefore omits this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_axis: Option<usize>,
    /// Graph-visible element layout of the state tensors.
    pub layout: String,
    /// Semantic state cell holding the current logical length of each row.
    ///
    /// Required only when the logical length is graph-visible. Absent means the
    /// runtime derives length from its private sequence table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_lengths: Option<String>,
    /// Whether the component may write `present` into the `past` buffer.
    #[serde(default)]
    pub aliasing: StateAliasing,
    /// How the graph writes each step's positions into this group's buffers.
    ///
    /// Absent means the buffer extends along `sequence_axis` as the sequence
    /// grows, which is the historical behavior. `indexed_scatter` declares a
    /// buffer of fixed capacity whose new positions are written at destinations
    /// the graph reads from a declared value. That distinction is what makes
    /// rewind, row replacement, and inactive-row compaction expressible: in a
    /// growing buffer the valid region is the whole tensor, while in a
    /// fixed-capacity buffer it is a declared prefix that the shape cannot
    /// reveal.
    ///
    /// This describes the GRAPH's update discipline, not an allocator. The
    /// physical buffer and where it lives remain runtime-owned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update: Option<StateUpdate>,
    /// Graph-visible total-length input, when the component reads one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_length: Option<String>,
    /// Prefix reuse and eviction semantics of this group.
    #[serde(default)]
    pub reuse: StateReuse,
    /// Rollback, snapshot, and fork bounds usable by the runtime.
    #[serde(default)]
    pub capabilities: StateGroupCapabilities,
    /// The versioned checkpoint adapter through which this group's state may
    /// leave the process portably.
    ///
    /// Absent means the group's state is private: it may still move between
    /// processes, but only through a private runtime protocol that requires a
    /// matching protocol and build on both ends (prefill/decode disaggregation,
    /// encoder/decoder interchange). Those transfers are fast precisely because
    /// they are not portable, and treating one as a portable export is how a
    /// cluster silently corrupts state across a rolling upgrade.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<StateCheckpointContract>,
    #[serde(default)]
    pub ports: BTreeMap<String, BTreeMap<String, StatePortAlias>>,
}

/// The only portable, cross-build path for a state group's contents.
///
/// This is deliberately not a wire format: metadata names the adapter and its
/// version, and the adapter owns the encoding. A portable checkpoint is slow and
/// survives a version change; a private transfer is fast and does not.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateCheckpointContract {
    /// Versioned adapter identifier, for example `onnx-genai.kv-checkpoint`.
    #[schemars(length(min = 1))]
    pub adapter: String,
    /// Adapter version this group's checkpoints are written against.
    #[schemars(length(min = 1))]
    pub version: String,
}

/// Semantic kind of a model state group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StateKind {
    /// Dense causal attention over the full sequence.
    FullAttention,
    /// Causal attention restricted to a sliding window.
    SlidingAttention,
    /// Compressed latent attention state (MLA).
    MultiLatentAttention,
    /// Fixed-size recurrent or state-space carry.
    Recurrent,
    /// Cross-attention state keyed by an encoder result.
    CrossAttention,
    /// Encoder output retained across decoder steps.
    Encoder,
}

/// How a state group's buffers absorb each step's new positions.
///
/// Both variants describe what the GRAPH does. Neither selects a storage
/// strategy, a slot allocator, or a device: a runtime is free to back an
/// `append` group with a fixed arena or an `indexed_scatter` group with paged
/// storage, so long as the graph sees what it declared.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StateUpdate {
    /// Each step's positions extend the buffer along `sequence_axis`.
    ///
    /// The valid region is the whole tensor, so no write cursor is graph-visible
    /// and the buffer's shape carries the length.
    Append,
    /// Each step replaces the complete fixed-size state tensor.
    ///
    /// This is the common discipline for recurrent accumulators, state-space
    /// carries, and causal-convolution history. The algorithm does not need a
    /// distinct state kind: separate groups already declare each tensor's
    /// shape, ports, lifetime, and rollback behavior.
    Replace,
    /// Each step's positions are scattered into a buffer of FIXED capacity at
    /// destinations the graph reads from `write_indices`.
    ///
    /// The tensor's extent along `sequence_axis` is the capacity, not the
    /// length: the valid region is the prefix named by the group's
    /// `logical_lengths`. Because destinations are data rather than position,
    /// rewinding a row is a cursor move, replacing a row reuses its slots, and
    /// rows of unequal length share one rectangular buffer.
    IndexedScatter {
        /// Semantic state cell carrying this step's per-row destination
        /// positions along `sequence_axis`.
        ///
        /// A cell rather than a step output because the write cursor is part of
        /// the group's state: it must be checkpointed, forked, and rewound with
        /// the buffer it indexes, or a restored row would overwrite live
        /// positions.
        #[schemars(length(min = 1))]
        write_indices: String,
        /// Integer-scalar workflow value giving the fixed extent of
        /// `sequence_axis` that the graph was built against.
        ///
        /// This is a graph fact, not a deployment budget. It bounds legal write
        /// destinations; it does not say where the buffer lives, when it is
        /// allocated, or how many rows a deployment admits.
        #[schemars(length(min = 1))]
        capacity: String,
        /// Per-component input port that receives the destinations, keyed by
        /// component name.
        ///
        /// The same class of fact as [`StateGroupContract::ports`]: which port
        /// of which component carries this group's ABI. A runtime cannot
        /// recover it from the step graph, because destinations are an ordinary
        /// integer vector and are shape-indistinguishable from every other
        /// integer control input.
        #[serde(default)]
        write_indices_ports: BTreeMap<String, String>,
        /// Per-component input port that receives the graph-visible valid
        /// length, keyed by component name.
        ///
        /// Exactly the same problem as `write_indices_ports`, and unsolvable the
        /// same way: the length is a rank-1 integer vector, so it is
        /// shape-indistinguishable from the destinations sitting next to it. A
        /// graph that reads no length port declares none.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        kv_length_ports: BTreeMap<String, String>,
    },
}

/// Legality of aliasing a component's `present` output onto its `past` input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StateAliasing {
    /// The runtime may alias input and output buffers but need not.
    Permitted,
    /// The component only works correctly when input and output alias.
    Required,
    /// Aliasing input and output buffers is incorrect for this component.
    #[default]
    Forbidden,
}

/// Semantic reuse and eviction legality of a state group.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateReuse {
    /// Whether a shared token prefix produces identical state that may be reused.
    #[serde(default)]
    pub prefix_reusable: bool,
    /// Whether dropping the oldest positions preserves declared semantics.
    #[serde(default)]
    pub evictable_prefix: bool,
}

/// Rollback, snapshot, and fork bounds declared for one state group.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateGroupCapabilities {
    /// Maximum number of trailing positions that can be discarded correctly.
    ///
    /// Absent means the group cannot be rolled back at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_positions: Option<usize>,
    /// Whether the runtime can snapshot and restore this group.
    #[serde(default)]
    pub snapshot: bool,
    /// Whether the runtime can fork this group into an independent row.
    #[serde(default)]
    pub fork: bool,
    /// Other groups that must be rolled back, snapshotted, or forked together.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub cascade: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StatePortAlias {
    pub input: String,
    /// Graph output port carrying this pair's next-step value.
    ///
    /// Required for a read-write transition. A `read_only` binding MAY omit it:
    /// a pure borrowed-state reader — e.g. a shared-KV drafter that consumes
    /// another decoder's cache and advances nothing — exposes no present output
    /// at all, so there is nothing to name. A read-only reader whose artifact
    /// still emits a discarded present output for kernel-ABI reasons may name it
    /// here, but that value is never a state transition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Whether this component advances the state or only observes a frozen
    /// value produced by another component in the same service group.
    ///
    /// A read-only binding still names the graph's present output when the
    /// artifact exposes one for kernel ABI reasons, but that output is not a
    /// state transition and must not be aliased back onto the input.
    #[serde(default, skip_serializing_if = "StatePortAccess::is_read_write")]
    pub access: StatePortAccess,
    /// Which half of an attention cache this port pair carries.
    ///
    /// A graph that splits keys and values into separate buffers exposes two
    /// aliases per layer that are shape-identical and therefore
    /// indistinguishable; a graph that packs them exposes one. Only the
    /// producer knows which it built, and recovering it from a port's spelling
    /// would be the name-guessing this schema refuses. Absent means the group
    /// does not distinguish halves, which is correct for recurrent and latent
    /// state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<StatePortRole>,
    /// Zero-based layer index of this port pair within its group.
    ///
    /// Required when a group binds more than one alias of the same
    /// [`StatePortRole`], because the map key is a producer-chosen label and
    /// its lexicographic order is not the layer order (`layer.10` sorts before
    /// `layer.2`). A runtime that pairs per-layer buffers positionally would
    /// otherwise silently transpose two layers' caches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
}

/// State access performed by one component binding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StatePortAccess {
    /// The component consumes the current value and produces its successor.
    #[default]
    ReadWrite,
    /// The component consumes a frozen value; any graph output is discarded.
    ReadOnly,
}

impl StatePortAccess {
    pub fn is_read_write(&self) -> bool {
        matches!(self, Self::ReadWrite)
    }
}

/// Which half of a split attention cache a state port pair carries.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum StatePortRole {
    /// Keys of a split key/value cache.
    Key,
    /// Values of a split key/value cache.
    Value,
    /// A single buffer holding keys and values together.
    Combined,
}
