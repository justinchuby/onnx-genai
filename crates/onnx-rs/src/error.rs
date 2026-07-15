//! Error types for the `onnx-rs` public API.
//!
//! `onnx-rs` wraps the runtime's [`onnx_runtime_loader`] pipeline for the actual
//! protobuf parsing/encoding, so most failures surface as a wrapped
//! [`LoaderError`]. The extra variants capture the file-system framing that the
//! ergonomic [`crate::load_model`] / [`crate::save_model`] entry points add on
//! top.

use std::path::PathBuf;

use onnx_runtime_loader::LoaderError;

/// The result type used throughout the `onnx-rs` public API.
pub type Result<T> = std::result::Result<T, Error>;

/// An error produced by an `onnx-rs` operation.
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

    /// The underlying loader (parse / build / encode) failed.
    #[error(transparent)]
    Loader(#[from] LoaderError),
}
