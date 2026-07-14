//! # `onnx-runtime-loader`
//!
//! Loads ONNX models from disk into the [`onnx_runtime_ir::Graph`] IR
//! (see `docs/ORT2.md` §19).
//!
//! Pipeline ([`load_model`]):
//! 1. [`proto`] — decode the ONNX protobuf (`prost` types generated from the
//!    vendored `onnx.proto3`) into a `ModelProto`.
//! 2. [`graph_builder`] — build an [`onnx_runtime_ir::Graph`] (nodes, values,
//!    symbolic dim interning, opset imports), upholding the §3.5 invariants.
//! 3. [`weights`] — resolve inline and external initializer data (external
//!    files are memory-mapped).
//! 4. [`shape_inference`] — best-effort static/symbolic shape propagation over
//!    the BERT op set.

use std::path::Path;

use onnx_runtime_ir::Graph;

use crate::graph_builder::BuiltGraph;

pub mod graph_builder;
pub mod proto;
pub mod shape_inference;
pub mod weights;

pub use error::LoaderError;

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

        #[error("graph construction failed: {0}")]
        GraphBuild(String),

        #[error(transparent)]
        Ir(#[from] onnx_runtime_ir::IrError),
    }
}

/// Load a model from a filesystem path, producing a fully-built [`Graph`].
///
/// Runs the full pipeline: parse → build → load weights → shape inference.
/// External initializer data is resolved relative to the model file's
/// directory.
pub fn load_model(path: impl AsRef<Path>) -> Result<Graph, LoaderError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let model_dir = path.parent().unwrap_or_else(|| Path::new("."));
    build_from_bytes(&bytes, model_dir)
}

/// Load a model from an in-memory protobuf buffer.
///
/// External initializer data (if any) is resolved relative to the current
/// working directory.
pub fn load_model_bytes(bytes: &[u8]) -> Result<Graph, LoaderError> {
    build_from_bytes(bytes, Path::new("."))
}

fn build_from_bytes(bytes: &[u8], model_dir: &Path) -> Result<Graph, LoaderError> {
    let model = proto::decode_model(bytes)?;
    let BuiltGraph {
        mut graph,
        name_map,
    } = graph_builder::build_graph(&model)?;

    let store = weights::load_weights(&model, model_dir, &name_map)?;
    for (vid, weight) in store.weights {
        graph.set_initializer(vid, weight);
    }

    let graph = shape_inference::run_shape_inference(graph)?;
    Ok(graph)
}
