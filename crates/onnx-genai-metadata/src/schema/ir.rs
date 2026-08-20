use super::*;
use std::collections::BTreeSet;

/// Typed tensor contract used at package and component boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TensorContract {
    #[schemars(with = "schema_vocabulary::TensorDType")]
    pub dtype: String,
    #[schemars(range(min = 0))]
    pub rank: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<TensorDimension>>,
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
}

/// Structural relationship between a typed value and the runtime batch.
///
/// `shared` values are invariant across requests. `request_aligned` values carry
/// exactly one entry per in-flight request along `axis`, so compaction permutes
/// that axis. `token_packed` values are ragged: `offsets` names the
/// request-aligned exclusive-prefix offset value and `owner` names the
/// per-item owner mapping, which together let a runtime split and regroup the
/// packed value without any serialized request ID. `runtime_sequence_state`
/// marks a value whose per-sequence storage the runtime owns outright.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BatchLayout {
    #[default]
    Shared,
    RequestAligned {
        axis: usize,
    },
    TokenPacked {
        /// Request-aligned value holding the exclusive prefix offset of each request's items.
        offsets: String,
        /// Item-aligned value mapping each packed item to its owning request row.
        owner: String,
        /// Packed axis of this value.
        axis: usize,
    },
    RuntimeSequenceState,
}

impl BatchLayout {
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared)
    }

    /// Axis permuted when the runtime compacts the batch, if any.
    pub fn request_axis(&self) -> Option<usize> {
        match self {
            Self::RequestAligned { axis } => Some(*axis),
            Self::Shared | Self::TokenPacked { .. } | Self::RuntimeSequenceState => None,
        }
    }

    /// Whether the runtime must move this value when it compacts or releases rows.
    pub fn is_row_scoped(&self) -> bool {
        !matches!(self, Self::Shared)
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
    /// `onnx-genai-targeted-base-v1:sha256:<lowercase hex>` compatibility fingerprint.
    pub base_model_fingerprint: String,
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
    pub base_model_fingerprint: String,
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
    /// Lowercase SHA-256 of the exact external artifact bytes.
    pub sha256: String,
    /// PEFT `adapter_config.json` paired with the safetensors file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_location: Option<String>,
    /// Lowercase SHA-256 of the exact PEFT config bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_sha256: Option<String>,
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
    pub ir_version: String,
    #[serde(default)]
    pub onnx_opsets: BTreeMap<String, u32>,
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
    pub default: Option<ScalarValue>,
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
    Media,
    MaxIterations,
    MaxOutputTokens,
    Seed,
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

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowOutput {
    pub contract: TensorContract,
    pub role: WorkflowOutputRole,
    pub stage: OutputStage,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowOutputRole {
    Tokens,
    Text,
    Image,
    Audio,
    Tensor,
    Event,
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
        iteration: Option<WorkflowLoopIteration>,
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
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        when: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        valid_length: Option<String>,
        output: String,
        mode: WorkflowEmitMode,
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
        iteration: Option<WorkflowLoopIteration>,
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
        /// on the value's final axis, globally or per batch row.
        valid_length: Option<String>,
        output: String,
        mode: WorkflowEmitMode,
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
    pub body_input: String,
    pub body_output: String,
    pub next: String,
    pub read_effect: EffectTransition,
    pub write_effect: EffectTransition,
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
    pub sequence_axis: usize,
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
    pub output: String,
}
