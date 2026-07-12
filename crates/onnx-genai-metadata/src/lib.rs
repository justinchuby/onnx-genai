//! ONNX inference metadata parser and types.
//!
//! Implements the spec from <https://github.com/onnx/onnx/issues/8184>

pub mod parser;
pub mod schema;
pub mod validation;

pub use parser::load_metadata;
pub use schema::*;
pub use validation::{RuntimeCapabilities, validate};

/// Error type for metadata operations.
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Unsupported capabilities: {0:?}")]
    Unsupported(Vec<String>),
}
