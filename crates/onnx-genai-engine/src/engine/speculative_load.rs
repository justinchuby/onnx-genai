//! Weight readers and speculative proposer (MTP / shared-KV) loading.

use super::*;

pub(crate) fn read_f32_weights(path: &Path) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read f32 weights from '{}'", path.display()))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        anyhow::bail!(
            "f32 weight file '{}' has byte length {}, which is not divisible by 4",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

/// Stable, opaque model identity derived from the model directory name.
///
/// Used only to namespace connector cache keys when the caller does not supply
/// an explicit `model_id`. It is never interpreted or branched on.
pub(crate) fn default_connector_model_id(model_directory: &ModelDirectory) -> String {
    model_directory
        .root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "onnx-genai-model".to_string())
}

/// Build the engine's KV connector bridge from generic, model-agnostic config.
pub(crate) fn build_connector_bridge(
    config: &KvConnectorConfig,
    model_directory: &ModelDirectory,
    kv_model: Option<&KvModelInfo>,
) -> anyhow::Result<ConnectorBridge> {
    match &config.backend {
        KvConnectorBackend::Null => Ok(ConnectorBridge::null()),
        KvConnectorBackend::LocalTiered(local_config) => {
            let connector = LocalTieredConnector::new(local_config.clone()).map_err(|error| {
                anyhow::anyhow!("failed to build LocalTiered KV connector: {error}")
            })?;
            let model_id = config
                .model_id
                .clone()
                .unwrap_or_else(|| default_connector_model_id(model_directory));
            let chunk_size = if config.chunk_size == 0 {
                onnx_genai_kv::DEFAULT_CHUNK_SIZE
            } else {
                config.chunk_size
            };
            let num_layers = kv_model
                .map(|model| model.tensor_config.num_layers)
                .unwrap_or(1)
                .max(1);
            ConnectorBridge::new(
                Arc::new(connector),
                model_id,
                chunk_size,
                0..num_layers,
                config.store_priority,
                config.recompute_ms_per_token,
            )
        }
    }
}
