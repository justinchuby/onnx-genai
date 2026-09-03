use std::path::PathBuf;

/// Errors produced while loading an ONNX model.
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error(
        "What: node {node} ({op_type}) carries opset version {node_version}, but the graph \
         imports version {graph_version} for domain {domain:?}, and this model cannot be \
         written out. \
         Why: a per-node opset is an in-memory IR concept with no representation in ONNX's \
         protobuf, so serialising would produce a model claiming the wrong operator version \
         with nothing downstream able to detect it. \
         How: serialise before the pass that introduced the node-local version, or run with \
         that fusion disabled."
    )]
    NodeVersionNotRepresentable {
        node: String,
        op_type: String,
        domain: String,
        node_version: i64,
        graph_version: u64,
    },
    #[error("failed to read model file {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse ONNX protobuf: {0}")]
    ProtobufParse(String),

    #[error("failed to parse ONNX protobuf TextFormat: {0}")]
    TextProtoParse(String),

    #[error("unsupported opset: domain={domain}, version={version}")]
    UnsupportedOpset { domain: String, version: u64 },

    #[error(
        "illegal ONNX model: operator {domain}::{op_type} at node {node} uses domain \
         '{domain}' but no corresponding opset_import is declared. RULES #1: the model must \
         declare an opset_import for domain '{domain}'; if you built this graph \
         programmatically, add it before loading; if this is a file, the model is \
         malformed/invalid per the ONNX spec"
    )]
    MissingOpsetImport {
        op_type: String,
        node: String,
        domain: String,
    },

    #[error(
        "illegal ONNX Einsum at node {node}: {detail}. RULES #1: Einsum syntax, schema version, \
         homogeneous dtype, rank, diagonal, and broadcast constraints must be valid at model \
         load. Expected: fix the equation/input metadata or export with the applicable ai.onnx \
         opset (Einsum-12 for opsets 12..27; Einsum-28 for opset 28+)"
    )]
    InvalidEinsum { node: String, detail: String },

    #[error(
        "unsupported ONNX model: operator {domain}::{op_type} at node {node} carries a \
         subgraph attribute '{attr}' (control-flow / nested-graph op) that this runtime cannot \
         execute. RULES #1: ep-cpu recursively executes the standard control-flow ops \
         If/Loop/Scan (ai.onnx), but not {op_type}, so the model cannot be run as-is. \
         Expected: express control flow with If/Loop/Scan, lower/unroll {op_type} into \
         supported ops, or register a kernel able to execute its subgraph body"
    )]
    UnsupportedControlFlow {
        op_type: String,
        node: String,
        domain: String,
        attr: String,
    },

    #[error(
        "illegal ONNX model: operator {domain}::{op_type} at node {node} consumes tensor \
         '{tensor}', but no producer exists — it is not a graph input, not an initializer, and \
         not produced by any upstream node. RULES #1: every consumed tensor must be sourced; \
         the graph is structurally malformed. Expected: add '{tensor}' as a graph input or \
         initializer, or add a node that produces it; if this is a file, the model is invalid \
         per the ONNX spec"
    )]
    DanglingTensorRef {
        op_type: String,
        node: String,
        domain: String,
        tensor: String,
    },

    #[error(
        "illegal ONNX model: tensor '{tensor}' is declared as an initializer but is also \
         produced as an output of node {node} — an initializer must be a constant source with \
         no producer. RULES #1: initializer names must be unique and must not collide with any \
         node output name; a producer-backed initializer would let a kernel write through \
         read-only weight storage. Expected: rename the node output or the initializer so they \
         no longer share a name; if this is a file, the model is malformed per the ONNX spec"
    )]
    InitializerHasProducer { tensor: String, node: String },

    #[error(
        "illegal ONNX model: value '{tensor}' has multiple producers ({first} and {second}). \
         RULES #1: ONNX graphs are in SSA form, so a value name may be assigned only once. \
         Expected: give each graph input and node output a unique name"
    )]
    DuplicateValueProducer {
        tensor: String,
        first: String,
        second: String,
    },

    #[error(
        "illegal ONNX model: operator {domain}::{op_type} at node {node} has attribute \
         '{attr}' referring to function attribute '{ref_attr_name}' outside a FunctionProto. \
         RULES #1: ref_attr_name is only bound while inlining a FunctionProto; it has no \
         executable value in a main graph or control-flow subgraph. Expected: replace it with \
         a concrete attribute value or move the node into a FunctionProto"
    )]
    RefAttributeOutsideFunction {
        op_type: String,
        node: String,
        domain: String,
        attr: String,
        ref_attr_name: String,
    },

    #[error(
        "illegal ONNX model: ir_version {ir_version} is invalid. RULES #1: ir_version is \
         required and ONNX IR versions start at 1. Expected: emit a model with ir_version >= 1"
    )]
    InvalidIrVersion { ir_version: i64 },

    #[error(
        "illegal ONNX model: ir_version {ir_version} requires at least one opset_import \
         (ONNX IR>=3). Expected: add an opset_import for every operator domain used by the \
         model"
    )]
    MissingModelOpsetImport { ir_version: i64 },

    #[error(
        "illegal ONNX model: initializer '{tensor}' in an outer graph is shadowed by a \
         subgraph input of the same name. RULES #1: this runtime does not permit ambiguous \
         initializer/subgraph binding. Expected: rename the subgraph formal input or the \
         outer initializer"
    )]
    SubgraphInputShadowsInitializer { tensor: String },

    #[error(
        "illegal ONNX model: graph output '{tensor}' has no producer in its graph. RULES #1: \
         every output must be a graph input, initializer, or node output in the same scope. \
         Expected: produce '{tensor}' locally or declare it as an input/initializer"
    )]
    GraphOutputMissingProducer { tensor: String },

    #[error("external data file not found: {path}")]
    ExternalDataNotFound { path: PathBuf },

    #[error("external data path rejected ({reason}): {path}")]
    ExternalDataPath { path: String, reason: &'static str },

    #[error("weight mmap failed: {0}")]
    Mmap(String),

    #[error("EPContext node error: {0}")]
    EpContext(String),

    #[error("EPContext external path rejected ({reason}): {path}")]
    EpContextPath { path: String, reason: &'static str },

    #[error("graph construction failed: {0}")]
    GraphBuild(String),

    #[error(
        "illegal ONNX model: model-local function {function} is recursive (call chain: \
         {chain}). RULES #1: ONNX function bodies may reference other model-local functions \
         but MUST NOT be recursive — inlining cannot terminate. Expected: break the cycle so \
         no function transitively calls itself"
    )]
    RecursiveFunction { function: String, chain: String },

    #[error(
        "illegal ONNX model: call to model-local function {function} at node {node} passes \
         {actual} {kind}(s) but the function declares only {formal}. RULES #1: a function \
         call may omit trailing optional {kind}s but must not supply more than are declared. \
         Expected: remove the extra {kind}(s) or fix the function signature"
    )]
    FunctionArityMismatch {
        function: String,
        node: String,
        kind: &'static str,
        formal: usize,
        actual: usize,
    },

    #[error(
        "illegal ONNX model: call to model-local function {function} at node {node} is missing \
         required attribute '{attribute}', and the function declares no default for it. \
         RULES #1: an attribute listed in FunctionProto.attribute has no default and must be \
         supplied at every call site. Expected: set '{attribute}' on the call node, or give \
         the function a default via attribute_proto"
    )]
    MissingRequiredFunctionAttribute {
        function: String,
        node: String,
        attribute: String,
    },

    #[error("unsupported ONNX data_type {raw} at {context}")]
    UnsupportedDataType { raw: i32, context: String },

    #[error("shape inference failed: {0}")]
    ShapeInference(#[from] onnx_runtime_shape_inference::ShapeInferError),

    #[error(transparent)]
    Ir(#[from] onnx_runtime_ir::IrError),
}
