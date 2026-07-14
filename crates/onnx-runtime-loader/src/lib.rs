//! # `onnx-runtime-loader`
//!
//! Loads ONNX models from disk into the [`onnx_runtime_ir::Graph`] IR
//! (see `docs/ORT2.md` §19). This crate is a **Phase 1 skeleton**: the module
//! structure and public entry points are defined, but the deep implementations
//! (`prost` protobuf decoding, per-op shape inference, weight `mmap`) are left
//! as `todo!()` for their respective downstream tasks.
//!
//! Pipeline ([`load_model`]):
//! 1. [`proto`] — decode the ONNX protobuf into intermediate structs.
//! 2. [`graph_builder`] — build an [`onnx_runtime_ir::Graph`] (nodes, values,
//!    symbolic dim interning, opset imports).
//! 3. [`weights`] — resolve inline and external initializer data (mmap).
//! 4. [`shape_inference`] — best-effort static/symbolic shape propagation.

use std::path::Path;

use onnx_runtime_ir::Graph;

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
pub fn load_model(path: impl AsRef<Path>) -> Result<Graph, LoaderError> {
    let _ = path.as_ref();
    todo!("Phase 1 task `ort2-loader`: parse protobuf, build graph, load weights, infer shapes")
}

/// Load a model from an in-memory protobuf buffer.
pub fn load_model_bytes(bytes: &[u8]) -> Result<Graph, LoaderError> {
    let _ = bytes;
    todo!("Phase 1 task `ort2-loader`: decode ModelProto from bytes")
}
