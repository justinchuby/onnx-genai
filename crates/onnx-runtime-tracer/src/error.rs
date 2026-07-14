//! Errors returned by the tracer crate.
//!
//! There is exactly one fallible operation in this crate — writing the
//! collected trace to a file — so the error surface is deliberately tiny.
//! Every variant carries enough context (the offending path, the underlying
//! cause) to be *actionable* without the caller having to reconstruct what it
//! was doing, matching the runtime's user-friendly-error rule.

use std::fmt;
use std::io;
use std::path::PathBuf;

/// A convenience `Result` alias for fallible tracer operations.
pub type Result<T> = std::result::Result<T, TracerError>;

/// An error produced while exporting a trace.
#[derive(Debug)]
#[non_exhaustive]
pub enum TracerError {
    /// The Chrome trace could not be serialized to JSON.
    ///
    /// This is effectively unreachable for the built-in [`Event`](crate::Event)
    /// model (it always serializes), but is surfaced rather than panicked so
    /// callers that inject custom `args` values stay in control.
    Serialize {
        /// The underlying `serde_json` failure.
        source: serde_json::Error,
    },
    /// The serialized trace could not be written to the target path.
    Write {
        /// The path we attempted to write to.
        path: PathBuf,
        /// The underlying I/O failure (permission denied, missing parent
        /// directory, disk full, …).
        source: io::Error,
    },
}

impl fmt::Display for TracerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TracerError::Serialize { source } => {
                write!(f, "failed to serialize the Chrome trace to JSON: {source}")
            }
            TracerError::Write { path, source } => write!(
                f,
                "failed to write the Chrome trace to '{}': {source}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for TracerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TracerError::Serialize { source } => Some(source),
            TracerError::Write { source, .. } => Some(source),
        }
    }
}
