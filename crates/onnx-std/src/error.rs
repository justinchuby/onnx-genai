//! Error types for the `onnx-std` public API.
//!
//! `onnx-std` wraps the runtime's [`onnx_runtime_loader`] pipeline for the actual
//! protobuf parsing/encoding, so most failures surface as a wrapped
//! [`LoaderError`]. The extra variants capture the file-system framing that the
//! ergonomic [`crate::load_model`] / [`crate::save_model`] entry points add on
//! top.

use std::path::PathBuf;

use onnx_runtime_loader::LoaderError;

/// The result type used throughout the `onnx-std` public API.
pub type Result<T> = std::result::Result<T, Error>;

/// An error produced by an `onnx-std` operation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Reading the model file from disk failed.
    #[error("failed to read model file {path}: {source}")]
    Read {
        /// The path that could not be read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Writing the model file to disk failed.
    #[error("failed to write model file {path}: {source}")]
    Write {
        /// The path that could not be written.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The underlying loader (parse / build / encode) failed.
    #[error(transparent)]
    Loader(#[from] LoaderError),

    /// A textual ONNX model could not be parsed.
    #[error("text parse error at line {line}: {message}")]
    TextParse {
        /// One-based source line containing the error.
        line: usize,
        /// Human-readable description of the malformed construct.
        message: String,
    },

    /// An ONNX protobuf-JSON document could not be encoded or decoded.
    #[error("ONNX JSON error: {0}")]
    Json(String),

    /// An ONNX protobuf TextFormat document could not be encoded or decoded.
    #[error("ONNX TextProto error: {0}")]
    TextProto(String),
}
