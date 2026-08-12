use super::*;

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
