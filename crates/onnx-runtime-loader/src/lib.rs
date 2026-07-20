//! # `onnx-runtime-loader`
//!
//! Loads ONNX models from disk into the [`onnx_runtime_ir::Graph`] IR
//! (see `docs/ORT2.md` §19).
//!
//! Pipeline ([`load_model`] / [`load_model_with_weights`]):
//! 1. [`proto`] — decode the ONNX protobuf (`prost` types generated from the
//!    vendored `onnx.proto3`) into a `ModelProto`.
//! 2. [`graph_builder`] — build an [`onnx_runtime_ir::Graph`] (nodes, values,
//!    symbolic dim interning, opset imports), upholding the §3.5 invariants.
//! 3. [`weights`] — resolve inline and external initializer data (external
//!    files are memory-mapped into a [`WeightStore`]).
//! 4. Static/symbolic shape inference via
//!    [`onnx-runtime-shape-inference`](onnx_runtime_shape_inference): the loader
//!    owns the "loader = shape-inference" seam, so after the [`Graph`] is built
//!    (with initializers applied) it runs the extensible per-op registry to
//!    populate every value's shape and dtype. Values that cannot be resolved
//!    statically (genuinely data-dependent extents) are left symbolic for the
//!    session to resolve just-in-time.
//!
//! ## Obtaining weight bytes at session time
//!
//! Use [`load_model_with_weights`] (or [`load_model_bytes_with_weights`]) to
//! receive both the [`Graph`] and an [`Arc<WeightStore>`]. Then, given any
//! [`onnx_runtime_ir::WeightRef`] stored in `graph.initializers`, call
//! [`WeightStore::bytes`] to get the raw little-endian byte slice:
//!
//! ```ignore
//! let (graph, store) = load_model_with_weights("model.onnx")?;
//! for (vid, weight_ref) in &graph.initializers {
//!     let bytes: &[u8] = store.bytes(weight_ref).expect("weight bytes live");
//!     // ... hand bytes to a kernel
//! }
//! ```
//!
//! The `Arc` keeps all memory maps alive as long as any clone of it exists, so
//! kernel dispatch can store `Arc<WeightStore>` alongside the `Graph` without
//! lifetime coupling.

use std::path::Path;
use std::sync::Arc;

use onnx_runtime_ir::Graph;
use onnx_runtime_shape_inference::{InferenceRegistry, MergePolicy};

use crate::graph_builder::BuiltGraph;

pub mod encoder;
pub mod epcontext;
pub mod function_inline;
pub(crate) mod graph_builder;
pub mod proto;
pub mod weights;
pub mod writer;

mod pathsafe;

pub use encoder::{
    DEFAULT_IR_VERSION, Model, ModelMetadata, encode_model, encode_model_proto, write_model,
};
pub use epcontext::{
    EmbedMode, EpContextBlob, EpContextNode, ep_context_node_ids, ep_context_nodes,
    is_ep_context_op, resolve_ep_context,
};
pub use error::LoaderError;
pub use weights::{
    ExpertQuantization, ExpertStorageOrder, ExpertTensorLayout, ExpertWeightRegion,
    NonPageableReason, Pageability, WeightRegionCatalog, WeightStore,
};
pub use writer::{EpContextDumpConfig, EpContextPartition, dump_ep_context};

mod error {
    use std::path::PathBuf;

    /// Errors produced while loading an ONNX model.
    #[derive(Debug, thiserror::Error)]
    pub enum LoaderError {
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
}

/// Load a model from a filesystem path, producing a fully-built [`Graph`].
///
/// Runs the full pipeline: parse → build → load weights → shape inference.
/// External initializer data is resolved relative to the model file's
/// directory.
///
/// # Note on external weights
///
/// The returned `Graph` stores [`onnx_runtime_ir::WeightRef::External`]
/// descriptors (path / offset / length) for weights held in external data
/// files, but the memory maps that back those bytes are **dropped** when this
/// function returns. Callers that need to read external weight bytes must
/// either re-map the files themselves or use [`load_model_with_weights`] which
/// keeps the maps alive via the returned [`Arc<WeightStore>`].
pub fn load_model(path: impl AsRef<Path>) -> Result<Graph, LoaderError> {
    Ok(load_model_with_weights(path)?.0)
}

/// Load a model from an in-memory protobuf buffer, producing a [`Graph`].
///
/// External initializer data (if any) is resolved relative to the current
/// working directory.
///
/// # Note on external weights
///
/// Same caveat as [`load_model`]: external weight bytes are not accessible
/// from the returned `Graph` alone. Use [`load_model_bytes_with_weights`] to
/// keep them live.
pub fn load_model_bytes(bytes: &[u8]) -> Result<Graph, LoaderError> {
    Ok(load_model_bytes_with_weights(bytes, Path::new("."))?.0)
}

/// Load a model from a filesystem path, returning both the [`Graph`] and the
/// live [`WeightStore`] that backs all initializer data.
///
/// The [`Arc<WeightStore>`] keeps every external-data memory map alive for as
/// long as any clone of the `Arc` exists. At session time, given a
/// [`onnx_runtime_ir::WeightRef`] from `graph.initializers`, call
/// [`WeightStore::bytes`] to obtain the raw little-endian byte slice — this
/// works for both [`WeightRef::Inline`] and [`WeightRef::External`] weights.
///
/// External initializer data is resolved relative to the model file's
/// directory.
pub fn load_model_with_weights(
    path: impl AsRef<Path>,
) -> Result<(Graph, Arc<WeightStore>), LoaderError> {
    let path = path.as_ref();
    let bytes = read_model_binary(path)?;
    let model_dir = path.parent().unwrap_or_else(|| Path::new("."));
    build_from_bytes_with_weights(&bytes, model_dir)
}

/// Read a model file into the binary protobuf bytes of its `ModelProto`.
///
/// Detection is by filename suffix: a path ending in `.textproto` is parsed as
/// ONNX protobuf **TextFormat** and converted to the binary wire encoding (see
/// [`proto::textproto_to_binary`]); any other path is read verbatim as an
/// already-binary `.onnx` model. This is the single seam that lets every
/// path-based loader entry accept git-friendly textproto fixtures while keeping
/// binary `.onnx` loading unchanged.
///
/// Note: textproto has no model-directory context for external weights, so
/// textproto fixtures must inline all initializer data.
pub fn read_model_binary(path: impl AsRef<Path>) -> Result<Vec<u8>, LoaderError> {
    let path = path.as_ref();
    let raw = std::fs::read(path).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if is_textproto_path(path) {
        let text = String::from_utf8(raw)
            .map_err(|e| LoaderError::TextProtoParse(format!("model is not valid UTF-8: {e}")))?;
        proto::textproto_to_binary(&text)
    } else {
        Ok(raw)
    }
}

/// Whether `path` names an ONNX protobuf TextFormat fixture (`*.textproto`).
pub fn is_textproto_path(path: impl AsRef<Path>) -> bool {
    path.as_ref()
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("textproto"))
}

/// Load a model from an in-memory protobuf buffer, returning both the
/// [`Graph`] and the live [`WeightStore`] that backs all initializer data.
///
/// External initializer data (if any) is resolved relative to `base_dir`.
/// The [`Arc<WeightStore>`] keeps every memory map alive for as long as any
/// clone of the `Arc` exists.
pub fn load_model_bytes_with_weights(
    bytes: &[u8],
    base_dir: impl AsRef<Path>,
) -> Result<(Graph, Arc<WeightStore>), LoaderError> {
    build_from_bytes_with_weights(bytes, base_dir.as_ref())
}

fn build_from_bytes_with_weights(
    bytes: &[u8],
    model_dir: &Path,
) -> Result<(Graph, Arc<WeightStore>), LoaderError> {
    let model = proto::decode_model(bytes)?;
    validate_model_proto(&model)?;
    let BuiltGraph {
        mut graph,
        name_map,
    } = graph_builder::build_graph(&model)?;

    // Fail-fast legality check that needs no weights: reject illegal opset
    // imports before we touch the (potentially large) weight files.
    validate_opset_imports(&graph)?;

    let store = weights::load_weights(&model, model_dir, &name_map)?;
    // Copy descriptors into the graph; the store's mmaps stay alive via Arc.
    for (&vid, weight) in &store.weights {
        graph.set_initializer(vid, weight.clone());
    }

    // Structural IR validation runs here — *after* initializers are attached —
    // rather than inside `graph_builder::build_graph`. A top-level initializer
    // is only recorded in `graph.initializers` by the weight-loading path above,
    // so validating earlier would mis-flag a legal initializer that is also a
    // graph output (constant pass-through) or a pre-IR-4 graph input that is
    // also an initializer as a producer-less `MissingProducer`. Validating the
    // fully-assembled graph recognizes those values as initializer sources.
    graph
        .validate()
        .map_err(|errs| LoaderError::GraphBuild(format!("{errs:?}")))?;

    // Full fail-fast validation once initializers are attached (so
    // initializer-backed values are recognized as sourced). Rejects
    // statically-knowable unsupported/illegal constructs before shape
    // inference or execution — see [`validate_model`].
    validate_model(&graph)?;

    // Static/symbolic shape inference (the loader owns this seam). Run the
    // extensible per-op registry over the fully-built graph — inputs,
    // initializers, and node outputs — to populate every value's shape and
    // dtype. `Permissive`: prefer the more specific dim on a benign
    // disagreement and keep going, and reconcile graph outputs with their
    // declared shapes rather than clobbering them. Values that stay symbolic
    // (genuinely data-dependent extents) are left for the session's JIT
    // fallback to resolve at run time.
    let registry = InferenceRegistry::default_registry();
    let opset_imports = graph.opset_imports.clone();
    registry.infer_graph(&mut graph, &opset_imports, MergePolicy::Permissive)?;

    Ok((graph, Arc::new(store)))
}

/// Fail-fast, load-time validation of everything statically knowable to be
/// illegal or unsupported (RULES #1: fail at *load*, never via a silent
/// sentinel at run time).
///
/// This is the single cohesive entry point wired into **both** load paths — the
/// disk/bytes loader ([`build_from_bytes_with_weights`]) and the session's
/// programmatic entry ([`onnx_runtime_session`]'s `from_parts`/`from_graph`) —
/// so the checks cannot drift between the two. It runs, in order:
///
/// Protobuf-only invariants (`ir_version`, raw SSA names, `ref_attr_name`,
/// subgraph shadows, and output names) run earlier in
/// [`validate_model_proto`], before graph construction coalesces names or drops
/// protobuf-only fields. This IR-level phase then runs:
///
/// 1. [`validate_opset_imports`] — every node's domain must declare an opset.
/// 2. [`validate_no_control_flow`] — allow the implemented subgraph-bearing ops
///    (`If`/`Loop`/`Scan`) and reject any other op carrying a `GraphProto`
///    attribute the executor cannot run.
/// 3. [`validate_no_dangling_refs`] — every consumed tensor must be sourced
///    (graph input, initializer, or an upstream node output).
/// 4. [`validate_no_initializer_producer`] — an initializer must be a constant
///    source; reject any initializer value that is also a node output (shares a
///    `ValueId` with a producer), which the IR structural check does not cover.
///
/// Each rejection names the offending node/op/tensor and explains what is
/// expected. No sentinel defaults, no silent skips.
///
/// Structural invariants that the IR builder already enforces via
/// [`onnx_runtime_ir::Graph::validate`] at build time — duplicate output
/// names, dangling value ids, producer/consumer link consistency, and data
/// dependency cycles — are intentionally *not* re-checked here to avoid drift;
/// this function adds the checks that path does not cover.
pub fn validate_model(graph: &Graph) -> Result<(), LoaderError> {
    validate_opset_imports(graph)?;
    validate_no_control_flow(graph)?;
    validate_no_dangling_refs(graph)?;
    validate_no_initializer_producer(graph)?;
    Ok(())
}

/// Validate ONNX model metadata and protobuf-level graph invariants that are
/// intentionally not preserved by the runtime IR (notably `ir_version` and
/// `AttributeProto::ref_attr_name`).
///
/// This runs before graph construction, ensuring invalid names cannot be
/// coalesced into a single IR value and attribute references cannot be dropped.
pub fn validate_model_proto(model: &proto::onnx::ModelProto) -> Result<(), LoaderError> {
    use std::collections::HashSet;

    use proto::onnx::GraphProto;

    // Lower sanity bound only: `ir_version` is a required ONNX field and IR
    // versions start at 1, so reject an absent (0) or negative version.
    if model.ir_version < 1 {
        return Err(LoaderError::InvalidIrVersion {
            ir_version: model.ir_version,
        });
    }
    // No upper bound. Per the maintainer directive, new ONNX IR versions are
    // effectively always backward-compatible (they add fields/metadata rather
    // than breaking existing model semantics), so gating on a version ceiling
    // only produces false-positive rejections of otherwise-valid newer models.
    // If a genuinely unsupported construct ever ships, gate on that specific
    // FEATURE at load time — never on the IR version number.
    if model.ir_version >= 3 && model.opset_import.is_empty() {
        return Err(LoaderError::MissingModelOpsetImport {
            ir_version: model.ir_version,
        });
    }

    fn node_description(node: &proto::onnx::NodeProto, index: usize) -> String {
        if node.name.is_empty() {
            format!("<unnamed node #{index}>")
        } else {
            format!("{:?}", node.name)
        }
    }

    fn check_graph(graph: &GraphProto) -> Result<(), LoaderError> {
        let mut producers = std::collections::HashMap::new();
        for input in &graph.input {
            if !input.name.is_empty() {
                producers.insert(input.name.clone(), "graph input".to_string());
            }
        }
        for (index, node) in graph.node.iter().enumerate() {
            let node_description = node_description(node, index);
            for output in &node.output {
                if output.is_empty() {
                    continue;
                }
                let producer = format!("output of {node_description}");
                if let Some(first) = producers.insert(output.clone(), producer.clone()) {
                    return Err(LoaderError::DuplicateValueProducer {
                        tensor: output.clone(),
                        first,
                        second: producer,
                    });
                }
            }
            for attribute in &node.attribute {
                if !attribute.ref_attr_name.is_empty() {
                    return Err(LoaderError::RefAttributeOutsideFunction {
                        op_type: node.op_type.clone(),
                        node: node_description.clone(),
                        domain: display_domain(&node.domain),
                        attr: attribute.name.clone(),
                        ref_attr_name: attribute.ref_attr_name.clone(),
                    });
                }
            }
        }

        let sources: HashSet<&str> = graph
            .input
            .iter()
            .map(|input| input.name.as_str())
            .chain(
                graph
                    .initializer
                    .iter()
                    .map(|initializer| initializer.name.as_str()),
            )
            .chain(
                graph
                    .node
                    .iter()
                    .flat_map(|node| node.output.iter().map(String::as_str)),
            )
            .collect();
        for output in &graph.output {
            if !output.name.is_empty() && !sources.contains(output.name.as_str()) {
                return Err(LoaderError::GraphOutputMissingProducer {
                    tensor: output.name.clone(),
                });
            }
        }

        let outer_initializers: HashSet<&str> = graph
            .initializer
            .iter()
            .map(|initializer| initializer.name.as_str())
            .collect();
        for node in &graph.node {
            for attribute in &node.attribute {
                let subgraphs = attribute.g.iter().chain(attribute.graphs.iter());
                for subgraph in subgraphs {
                    if let Some(input) = subgraph
                        .input
                        .iter()
                        .find(|input| outer_initializers.contains(input.name.as_str()))
                    {
                        return Err(LoaderError::SubgraphInputShadowsInitializer {
                            tensor: input.name.clone(),
                        });
                    }
                    check_graph(subgraph)?;
                }
            }
        }
        Ok(())
    }

    if let Some(graph) = &model.graph {
        check_graph(graph)?;
    }
    Ok(())
}

/// Human-readable node label for diagnostics: the quoted ONNX node name, or a
/// synthetic `<unnamed node #id>` when the model left it blank.
fn node_label(node: &onnx_runtime_ir::Node) -> String {
    if node.name.is_empty() {
        format!("<unnamed node #{}>", node.id.0)
    } else {
        format!("{:?}", node.name)
    }
}

/// Canonical display domain for a node (`""` renders as `ai.onnx`).
fn display_domain(domain: &str) -> String {
    if domain.is_empty() {
        "ai.onnx".to_string()
    } else {
        domain.to_string()
    }
}

/// Reject subgraph-bearing (control-flow) ops the runtime cannot execute.
///
/// The CPU executor implements the three standard subgraph-bearing control-flow
/// ops — `If`, `Loop`, and `Scan` (default `ai.onnx` domain) — by recursively
/// executing their nested [`onnx_runtime_ir::Attribute::Graph`]/`Graphs` bodies.
/// Any *other* op that smuggles a subgraph attribute (a control-flow construct
/// this runtime does not implement, or a custom op hiding a nested graph) is
/// still rejected fast: the executor has no path to run it, so a silent skip or
/// a late panic would be worse than a clear load-time error.
///
/// The check descends into every nested subgraph as well, so an unimplemented
/// control-flow op buried inside an `If`/`Loop`/`Scan` body is caught at load
/// rather than surfacing only when that branch/iteration executes.
pub fn validate_no_control_flow(graph: &Graph) -> Result<(), LoaderError> {
    use onnx_runtime_ir::Attribute;

    fn is_default_domain(domain: &str) -> bool {
        domain.is_empty() || domain == "ai.onnx"
    }

    /// The standard subgraph-bearing ops the CPU executor can run recursively.
    fn is_implemented_control_flow(op_type: &str, domain: &str) -> bool {
        is_default_domain(domain) && matches!(op_type, "If" | "Loop" | "Scan")
    }

    fn check_graph(graph: &Graph) -> Result<(), LoaderError> {
        for (_, node) in graph.nodes.iter() {
            // Report attributes in a deterministic order for stable diagnostics.
            let mut subgraph_attrs: Vec<&String> = node
                .attributes
                .iter()
                .filter(|(_, v)| matches!(v, Attribute::Graph(_) | Attribute::Graphs(_)))
                .map(|(k, _)| k)
                .collect();
            subgraph_attrs.sort();
            if let Some(attr) = subgraph_attrs.first() {
                // A subgraph body is fine when its owner is an implemented
                // control-flow op; otherwise fail fast.
                if !is_implemented_control_flow(&node.op_type, &node.domain) {
                    return Err(LoaderError::UnsupportedControlFlow {
                        op_type: node.op_type.clone(),
                        node: node_label(node),
                        domain: display_domain(&node.domain),
                        attr: (*attr).clone(),
                    });
                }
            }
        }
        // Descend into nested bodies so an unimplemented construct inside an
        // implemented op's subgraph is still caught at load time.
        for subgraph in graph.subgraphs.values() {
            check_graph(subgraph)?;
        }
        Ok(())
    }

    check_graph(graph)
}

/// Reject graphs with a node input that has no source.
///
/// The graph builder materializes an unresolved input name as a fresh named
/// value with no producer (see `graph_builder::get_or_create`); such a value is
/// legal only if it is a graph input or an initializer. Any other producer-less
/// consumed value is a dangling reference — a structurally malformed graph that
/// [`onnx_runtime_ir::Graph::validate`] does not catch (it only requires graph
/// *outputs* to be sourced, not node inputs). We reject it at load, naming the
/// offending node and tensor.
///
/// Must run after initializers are attached to `graph.initializers` so
/// initializer-backed inputs are recognized as sourced.
pub fn validate_no_dangling_refs(graph: &Graph) -> Result<(), LoaderError> {
    use std::collections::HashSet;

    let graph_inputs: HashSet<_> = graph.inputs.iter().copied().collect();

    for (_, node) in graph.nodes.iter() {
        for vid in node.input_values() {
            let Some(value) = graph.values.get(vid) else {
                // A dangling value id is caught by IR-level structural
                // validation; nothing to report here.
                continue;
            };
            let is_sourced = value.producer.is_some()
                || graph_inputs.contains(&vid)
                || graph.initializers.contains_key(&vid);
            if !is_sourced {
                let tensor = value
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("<anonymous value #{}>", vid.0));
                return Err(LoaderError::DanglingTensorRef {
                    op_type: node.op_type.clone(),
                    node: node_label(node),
                    domain: display_domain(&node.domain),
                    tensor,
                });
            }
        }
    }
    Ok(())
}

/// Reject graphs where an initializer value is also produced by a node.
///
/// The graph builder maps tensor *names* → [`onnx_runtime_ir::ValueId`] for both
/// node inputs and node outputs (see `graph_builder::get_or_create`). If a node
/// output name collides with an initializer name, the node output reuses the
/// initializer's `ValueId` and `connect_edges` then sets `producer = Some(node)`
/// on that shared value. [`onnx_runtime_ir::Graph::validate`] rejects a *graph
/// input* with a producer but has no equivalent check for an *initializer*, so
/// such a malformed graph passes structural validation.
///
/// This matters for memory-safety: the session's weight-streaming path borrows
/// an initializer's read-only mmap bytes zero-copy. A producer-backed
/// initializer would let a kernel write through that read-only storage
/// (SIGSEGV on external data, aliasing UB inline). The executor already refuses
/// to borrow producer-backed initializers, but rejecting the graph here fails
/// fast and cleanly regardless of the execution path. We name the tensor and
/// the offending producing node.
pub fn validate_no_initializer_producer(graph: &Graph) -> Result<(), LoaderError> {
    for &vid in graph.initializers.keys() {
        let Some(value) = graph.values.get(vid) else {
            continue;
        };
        if let Some(producer) = value.producer {
            let tensor = value
                .name
                .clone()
                .unwrap_or_else(|| format!("<anonymous value #{}>", vid.0));
            let node = if graph.nodes.contains(producer) {
                node_label(graph.node(producer))
            } else {
                format!("<node #{}>", producer.0)
            };
            return Err(LoaderError::InitializerHasProducer { tensor, node });
        }
    }
    Ok(())
}
///
/// ONNX treats `""` and `"ai.onnx"` as equivalent spellings of the default
/// domain. Model-level imports also govern nodes nested in subgraphs.
pub fn validate_opset_imports(graph: &Graph) -> Result<(), LoaderError> {
    fn has_import(imports: &std::collections::HashMap<String, u64>, domain: &str) -> bool {
        imports.contains_key(domain)
            || (domain.is_empty() && imports.contains_key("ai.onnx"))
            || (domain == "ai.onnx" && imports.contains_key(""))
    }

    fn validate_graph(
        graph: &Graph,
        imports: &std::collections::HashMap<String, u64>,
    ) -> Result<(), LoaderError> {
        for (_, node) in graph.nodes.iter() {
            if !has_import(imports, &node.domain) {
                let domain = if node.domain.is_empty() {
                    "ai.onnx".to_string()
                } else {
                    node.domain.clone()
                };
                let node_name = if node.name.is_empty() {
                    format!("<unnamed node #{}>", node.id.0)
                } else {
                    format!("{:?}", node.name)
                };
                return Err(LoaderError::MissingOpsetImport {
                    op_type: node.op_type.clone(),
                    node: node_name,
                    domain,
                });
            }
        }
        for subgraph in graph.subgraphs.values() {
            validate_graph(subgraph, imports)?;
        }
        Ok(())
    }

    validate_graph(graph, &graph.opset_imports)
}
