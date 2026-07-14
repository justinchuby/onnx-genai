//! Errors returned by the tracer crate.
//!
//! Every variant is written to satisfy the project's flagship rule
//! ([`RULES.md` #1](../../../RULES.md) / `docs/ORT2.md` §35): a failure must tell
//! humans **and** AI agents *what* failed, *why*, and *how to fix it*, and it
//! must carry the most useful context (the offending path, the target format,
//! the underlying cause) so the caller never has to reconstruct what it was
//! doing. Messages should read warmly and actionably, not as a bare label.

use crate::format::TraceFormat;
use std::fmt;
use std::io;
use std::path::PathBuf;

/// A convenience `Result` alias for fallible tracer operations.
pub type Result<T> = std::result::Result<T, TracerError>;

/// An error produced while exporting or writing a trace.
#[derive(Debug)]
#[non_exhaustive]
pub enum TracerError {
    /// The requested trace format is unavailable in this crate build.
    UnsupportedFormat {
        /// The format the caller requested.
        format: TraceFormat,
    },
    /// A trace could not be serialized to its target [`TraceFormat`].
    ///
    /// For the built-in [`TraceEvent`](crate::TraceEvent) model this is
    /// effectively unreachable (the model always serializes); it is surfaced
    /// rather than panicked so callers that inject custom `args` values stay in
    /// control.
    Serialize {
        /// The format we were serializing to (Chrome JSON, JSONL, …).
        format: TraceFormat,
        /// The underlying `serde_json` failure.
        source: serde_json::Error,
    },
    /// The serialized trace could not be written to the target path.
    Write {
        /// The path we attempted to write to.
        path: PathBuf,
        /// The format being written, so the message can name it.
        format: TraceFormat,
        /// The underlying I/O failure (permission denied, missing parent
        /// directory, disk full, …).
        source: io::Error,
    },
}

impl fmt::Display for TracerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TracerError::UnsupportedFormat { format } => write!(
                f,
                "cannot create a FileCollector for {format} output: Perfetto protobuf \
                 output requires the `perfetto` cargo feature, which is not enabled \
                 in this build. Enable the `perfetto` feature (it is enabled by \
                 default, so avoid `--no-default-features` or add it explicitly), \
                 or use `TraceFormat::ChromeJson`; Perfetto UI \
                 (https://ui.perfetto.dev) loads Chrome JSON directly too."
            ),
            TracerError::Serialize { format, source } => write!(
                f,
                "failed to serialize the {format} trace: {source}. \
                 This usually means a custom `args` value on one event could not \
                 be represented as JSON — simplify or remove that event's \
                 metadata and export again."
            ),
            TracerError::Write {
                path,
                format,
                source,
            } => write!(
                f,
                "failed to write the {format} trace to '{}': {source}. \
                 Check that the parent directory exists and is writable, that \
                 there is free disk space, and that no other process holds the \
                 file open; then retry the export (or choose a different output \
                 path).",
                path.display()
            ),
        }
    }
}

impl std::error::Error for TracerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TracerError::UnsupportedFormat { .. } => None,
            TracerError::Serialize { source, .. } => Some(source),
            TracerError::Write { source, .. } => Some(source),
        }
    }
}
