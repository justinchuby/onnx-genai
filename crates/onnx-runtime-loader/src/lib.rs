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
pub mod graph_builder;
pub mod proto;
pub mod weights;
pub mod writer;

mod pathsafe;

pub use encoder::{
    encode_model, encode_model_proto, write_model, Model, ModelMetadata, DEFAULT_IR_VERSION,
};
pub use epcontext::{
    ep_context_node_ids, ep_context_nodes, is_ep_context_op, resolve_ep_context, EmbedMode,
    EpContextBlob, EpContextNode,
};
pub use error::LoaderError;
pub use weights::WeightStore;
pub use writer::{dump_ep_context, EpContextDumpConfig, EpContextPartition};

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
             subgraph attribute '{attr}' (control-flow / nested-graph op). RULES #1: this runtime \
             (ep-cpu) does not execute subgraph-bearing ops such as If/Loop/Scan yet, so the model \
             cannot be run as-is. Expected: a flat graph with no nested subgraphs; to proceed, \
             lower/unroll the control flow (e.g. export without dynamic loops) or wait for \
             control-flow support"
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
    let bytes = std::fs::read(path).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let model_dir = path.parent().unwrap_or_else(|| Path::new("."));
    build_from_bytes_with_weights(&bytes, model_dir)
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
/// 1. [`validate_opset_imports`] — every node's domain must declare an opset.
/// 2. [`validate_no_control_flow`] — reject subgraph-bearing ops (If/Loop/Scan
///    and any op carrying a `GraphProto` attribute) the CPU EP cannot execute.
/// 3. [`validate_no_dangling_refs`] — every consumed tensor must be sourced
///    (graph input, initializer, or an upstream node output).
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
/// The CPU EP has no kernels for `If`/`Loop`/`Scan` and never descends into a
/// nested [`onnx_runtime_ir::Attribute::Graph`]/`Graphs` body, so a model that
/// carries one would either fail lazily at run time or — worse — silently skip
/// the subgraph. We reject it at load with a message naming the offending node
/// and its subgraph attribute. Detection is by the presence of a `Graph`/
/// `Graphs` attribute, so it also catches custom ops that smuggle subgraphs.
pub fn validate_no_control_flow(graph: &Graph) -> Result<(), LoaderError> {
    use onnx_runtime_ir::Attribute;

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
            return Err(LoaderError::UnsupportedControlFlow {
                op_type: node.op_type.clone(),
                node: node_label(node),
                domain: display_domain(&node.domain),
                attr: (*attr).clone(),
            });
        }
    }
    Ok(())
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

/// Reject graphs whose nodes use an operator domain without importing its opset.
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
