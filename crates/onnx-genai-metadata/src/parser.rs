//! Load inference metadata from YAML or JSON files.

use crate::schema::InferenceMetadata;
use std::path::Path;

/// Load inference metadata from a file (YAML or JSON based on extension).
pub fn load_metadata(path: &Path) -> Result<InferenceMetadata, crate::MetadataError> {
    let content = std::fs::read_to_string(path).map_err(crate::MetadataError::Io)?;

    let metadata: InferenceMetadata = match path.extension().and_then(|e| e.to_str()) {
        Some("yaml" | "yml") => serde_yaml::from_str(&content)
            .map_err(|e| crate::MetadataError::Parse(e.to_string()))?,
        Some("json") => serde_json::from_str(&content)
            .map_err(|e| crate::MetadataError::Parse(e.to_string()))?,
        _ => {
            // Try YAML first, then JSON
            if let Ok(m) = serde_yaml::from_str::<InferenceMetadata>(&content) {
                m
            } else {
                serde_json::from_str::<InferenceMetadata>(&content)
                    .map_err(|e| crate::MetadataError::Parse(e.to_string()))?
            }
        }
    };

    Ok(metadata)
}
