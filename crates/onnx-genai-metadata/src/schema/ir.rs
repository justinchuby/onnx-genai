use super::*;
use std::collections::BTreeSet;

/// Typed tensor contract used at package and component boundaries.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
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

/// Generic package control-flow algebra.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlFlow {
    Sequence {
        steps: Vec<ControlFlow>,
    },
    Invoke {
        component: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        when: Option<Predicate>,
    },
    Loop {
        body: Box<ControlFlow>,
        #[serde(default)]
        carried: Vec<LoopCarry>,
        termination: Termination,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step_program: Option<String>,
    },
    Branch {
        predicate: Predicate,
        cases: BTreeMap<String, ControlFlow>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<Box<ControlFlow>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoopCarry {
    pub state: String,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Termination {
    Iterations {
        count: usize,
        #[serde(default)]
        start: usize,
    },
    Predicate {
        condition: Predicate,
        max_iterations: usize,
    },
}

/// Data-only predicates for branch and loop termination.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Predicate {
    Present { input: String },
    Bool { value: bool },
    Not { value: Box<Predicate> },
    All { values: Vec<Predicate> },
    Any { values: Vec<Predicate> },
    Equal { left: ScalarExpr, right: ScalarExpr },
    Less { left: ScalarExpr, right: ScalarExpr },
    LessEqual { left: ScalarExpr, right: ScalarExpr },
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScalarExpr {
    Literal { value: ScalarValue },
    Value { source: String },
    Iteration,
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
pub enum StateScope {
    Invocation,
    Loop,
    Request,
    Session,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StateInit {
    Zeros,
    Ones,
    Input { source: String },
    Value { source: String },
    Scalar { value: ScalarValue },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StateUpdate {
    Replace,
    Append { axis: i64 },
    Scatter { axis: i64, indices: String },
    Accumulate,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateDeclaration {
    #[serde(rename = "type")]
    pub contract: TensorContract,
    pub init: StateInit,
    pub update: StateUpdate,
    pub scope: StateScope,
}

/// Generic tensor/scalar program executed between component invocations.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub operations: Vec<ProgramOperation>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramOperation {
    Copy {
        from: String,
        to: String,
    },
    Cast {
        input: String,
        output: String,
        #[schemars(with = "schema_vocabulary::TensorDType")]
        dtype: String,
    },
    Sample {
        logits: String,
        output: String,
        method: SamplingMethod,
    },
    SolverStep {
        estimate: String,
        state: String,
        output: String,
        solver: SolverSpec,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SamplingMethod {
    Greedy,
    Categorical {
        temperature: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_k: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_p: Option<f32>,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SolverSpec {
    pub algorithm: String,
    #[serde(default)]
    pub parameters: BTreeMap<String, f64>,
    #[serde(default)]
    pub schedule: Vec<f64>,
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

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResourceContract {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<DeviceKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BatchingContract {
    pub batch_axis: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch_size: Option<usize>,
    #[serde(default)]
    pub continuous: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PostprocessingSpec {
    pub program: Program,
    #[serde(default)]
    pub outputs: BTreeMap<String, String>,
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
    #[serde(default)]
    pub initial_effects: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving: Option<ServingServiceContract>,
    pub graph: WorkflowNode,
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
    #[serde(default)]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<ScalarValue>,
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
    Constraint,
    SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkflowInputSource {
    Request { field: RuntimeInputRole },
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
    pub ports: ComponentPorts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<PolicyComponentContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<AdapterComponentContract>,
    #[serde(default)]
    pub effects: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<WorkflowResourceContract>,
}

/// Stable semantic roles for ONNX policy-math components.
///
/// Fields map semantic roles to concrete ONNX port names. The corresponding
/// tensor contracts live in [`WorkflowComponent::ports`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "role", rename_all = "snake_case", deny_unknown_fields)]
pub enum PolicyComponentContract {
    TokenSampler {
        mode: SamplingPolicyMode,
        logits: String,
        token: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temperature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_k: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_p: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rng: Option<RngPortContract>,
        effect: String,
    },
    TerminationPredicate {
        tokens: String,
        eos_ids: String,
        iteration: String,
        max_iterations: String,
        done: String,
        effect: String,
    },
    SolverStep {
        state: String,
        estimate: String,
        step: String,
        schedule: String,
        next_state: String,
        effect: String,
    },
    MaskedUpdate {
        state: String,
        proposal: String,
        mask: String,
        step: String,
        next_state: String,
        next_mask: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rng: Option<RngPortContract>,
        effect: String,
    },
    SpeculativeVerifier {
        target_scores: String,
        proposed_tokens: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proposal_scores: Option<String>,
        accepted_tokens: String,
        accepted_len: String,
        done: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rng: Option<RngPortContract>,
        effect: String,
    },
    StateUpdate {
        current: String,
        update: String,
        next: String,
        effect: String,
    },
    AdaptiveProposalBudget {
        current_k: String,
        accepted: String,
        evaluated: String,
        committed_tokens: String,
        filled_proposal_budget: String,
        draft_ms: String,
        target_ms: String,
        estimates: String,
        next_k: String,
        next_estimates: String,
        effect: String,
    },
}

/// Stable semantic roles for versioned runtime adapter ABIs.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "role", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdapterComponentContract {
    GrammarGuidance {
        action: GrammarGuidanceAction,
        state: String,
        tokens: String,
        valid_length: String,
        transition_table: String,
        next_state: String,
        consumed_length: String,
        logits_mask: String,
        forced_tokens: String,
        forced_length: String,
        effect: String,
    },
    Telemetry {
        action: TelemetryAction,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timestamp: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<String>,
        effect: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GrammarGuidanceAction {
    Clone,
    Lookahead,
    Commit,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryAction {
    Start,
    Elapsed,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SamplingPolicyMode {
    Greedy,
    SeededStochastic,
}

/// Counter-based RNG state. Producers should use Philox or Threefry inside ONNX.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RngPortContract {
    pub seed: String,
    pub offset: String,
    pub next_offset: String,
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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EffectTransition {
    pub consumes: String,
    pub produces: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkflowNode {
    Sequence {
        nodes: Vec<WorkflowNode>,
    },
    Invoke {
        component: String,
        #[serde(default)]
        inputs: BTreeMap<String, String>,
        #[serde(default)]
        outputs: BTreeMap<String, String>,
        #[serde(default)]
        effects: BTreeMap<String, EffectTransition>,
    },
    Loop {
        setup: Box<WorkflowNode>,
        body: Box<WorkflowNode>,
        condition: String,
        max_iterations: String,
        /// Optional zero-based induction value, scoped to this loop's body and condition.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        iteration: Option<WorkflowLoopIteration>,
        #[serde(default)]
        carried: Vec<WorkflowLoopCarry>,
    },
    Branch {
        predicate: String,
        cases: BTreeMap<String, WorkflowNode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<Box<WorkflowNode>>,
        #[serde(default)]
        outputs: BTreeMap<String, WorkflowBranchOutput>,
        #[serde(default)]
        effects: BTreeMap<String, WorkflowBranchEffectMerge>,
    },
    Emit {
        value: String,
        /// Optional scalar or rank-one integer SSA value limiting the emitted prefix
        /// on the value's final axis. It must contain exactly one element at runtime.
        #[serde(default, skip_serializing_if = "Option::is_none")]
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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowBranchEffectMerge {
    pub incoming: String,
    pub cases: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    pub produces: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowLoopCarry {
    pub cell: String,
    pub current: String,
    pub body_input: String,
    pub body_output: String,
    pub next: String,
    pub read_effect: EffectTransition,
    pub write_effect: EffectTransition,
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

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowResourceContract {
    #[serde(default)]
    pub allowed_devices: Vec<DeviceKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_device: Option<DeviceKind>,
    pub memory_class: WorkflowMemoryClass,
    pub batching: WorkflowBatchingContract,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowMemoryClass {
    Default,
    Device,
    Host,
    Pinned,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkflowBatchingContract {
    None,
    Stack { axis: usize },
    Ragged { offsets: String },
}
