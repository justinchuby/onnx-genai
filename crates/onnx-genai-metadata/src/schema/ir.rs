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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving: Option<ServingServiceContract>,
    pub steps: Vec<WorkflowStep>,
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
    pub custom_op_versions: BTreeMap<String, String>,
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
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComponentContract {
    /// Versioned semantic capability identifier. It never selects execution behavior.
    pub id: String,
    pub version: String,
    /// Semantic role to concrete component port name.
    #[serde(default)]
    pub bindings: BTreeMap<String, String>,
    /// Contract parameters that are not tensor ports, such as adapter actions.
    #[serde(default)]
    pub parameters: BTreeMap<String, ScalarValue>,
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
        #[serde(default)]
        custom_ops: BTreeMap<String, String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_group: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionLeaseContract>,
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
    pub slot_ids: String,
    pub kv_service: KvServiceContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KvServiceContract {
    pub paging: KvPagingMode,
    pub allocation: SlotAllocationMode,
    #[serde(default)]
    pub compaction: bool,
    #[serde(default)]
    pub groups: BTreeMap<String, KvServiceGroupContract>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KvServiceGroupContract {
    pub sequence_axis: usize,
    pub layout: String,
    /// Semantic state cell containing the current logical sequence length for each row.
    pub logical_lengths: String,
    #[serde(default)]
    pub storage: KvStorageMode,
    #[serde(default)]
    pub ports: BTreeMap<String, BTreeMap<String, KvPortAlias>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KvStorageMode {
    #[default]
    Separate,
    SharedBuffer,
    Paged,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KvPortAlias {
    pub input: String,
    pub output: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KvPagingMode {
    None,
    Paged,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SlotAllocationMode {
    Static,
    Runtime,
}
