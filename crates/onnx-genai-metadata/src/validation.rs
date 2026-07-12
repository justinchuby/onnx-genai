//! Validate metadata against runtime capabilities.

use crate::schema::InferenceMetadata;

/// Capabilities this runtime supports.
pub struct RuntimeCapabilities {
    pub supported: Vec<String>,
}

impl Default for RuntimeCapabilities {
    fn default() -> Self {
        Self {
            supported: vec![
                "kv_cache".to_string(),
                "grouped_query_attention".to_string(),
                "multi_head_attention".to_string(),
                "prefix_cache".to_string(),
                "continuous_batching".to_string(),
            ],
        }
    }
}

/// Validate that all required capabilities are supported.
pub fn validate(
    metadata: &InferenceMetadata,
    runtime: &RuntimeCapabilities,
) -> Result<(), Vec<String>> {
    let unsupported: Vec<String> = metadata
        .required_capabilities
        .iter()
        .filter(|cap| !runtime.supported.contains(cap))
        .cloned()
        .collect();

    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(unsupported)
    }
}

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

// Re-export at crate level
pub use MetadataError as Error;
