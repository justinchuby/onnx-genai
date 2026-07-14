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

pub mod epcontext;
pub mod graph_builder;
pub mod proto;
pub mod weights;

pub use epcontext::{
    ep_context_node_ids, ep_context_nodes, is_ep_context_op, resolve_ep_context, EmbedMode,
    EpContextBlob, EpContextNode,
};
pub use error::LoaderError;
pub use weights::WeightStore;

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

        #[error("external data file not found: {path}")]
        ExternalDataNotFound { path: PathBuf },

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

    let store = weights::load_weights(&model, model_dir, &name_map)?;
    // Copy descriptors into the graph; the store's mmaps stay alive via Arc.
    for (&vid, weight) in &store.weights {
        graph.set_initializer(vid, weight.clone());
    }

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
