use std::path::{Path, PathBuf};

use onnx_genai_metadata::InferenceMetadata;

use crate::{GENAI_CONFIG_FILE, GenAiConfig, GenAiConfigError, ModelGraphInfo};

/// Path to a `genai_config.json` inside `model_dir`, if one exists.
pub fn find_in_dir(model_dir: &Path) -> Option<PathBuf> {
    let path = model_dir.join(GENAI_CONFIG_FILE);
    path.is_file().then_some(path)
}

/// Load and parse a `genai_config.json` from an explicit path.
pub fn load(path: &Path) -> Result<GenAiConfig, GenAiConfigError> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

/// Best-effort compatibility metadata for a model directory.
///
/// Returns `Ok(None)` when the directory has no `genai_config.json`.
pub fn inference_metadata_from_dir(
    model_dir: &Path,
    kv_native_dtype: Option<&str>,
) -> Result<Option<InferenceMetadata>, GenAiConfigError> {
    let Some(path) = find_in_dir(model_dir) else {
        return Ok(None);
    };
    let config = load(&path)?;
    Ok(Some(config.to_inference_metadata(kv_native_dtype)?))
}

/// Compatibility metadata from an explicitly resolved configuration path.
pub fn inference_metadata_from_path(
    path: &Path,
    kv_native_dtype: Option<&str>,
) -> Result<InferenceMetadata, GenAiConfigError> {
    load(path)?.to_inference_metadata(kv_native_dtype)
}

/// Like [`inference_metadata_from_dir`], but derives the single-decoder KV/state
/// topology from the decoder's actual ONNX graph inventory (`decoder_graph`)
/// rather than expanding KV name patterns over a uniform per-layer count.
///
/// This is the entry point runtime loaders use once the decoder session is
/// available: it lets hybrid SSM/attention decoders load by declaring only the
/// graph's real ports (sparse dense KV plus fixed recurrent `state_pairs`).
pub fn inference_metadata_from_dir_with_graph(
    model_dir: &Path,
    kv_native_dtype: Option<&str>,
    decoder_graph: &ModelGraphInfo,
) -> Result<Option<InferenceMetadata>, GenAiConfigError> {
    let Some(path) = find_in_dir(model_dir) else {
        return Ok(None);
    };
    let config = load(&path)?;
    Ok(Some(config.to_inference_metadata_with_graph(
        kv_native_dtype,
        decoder_graph,
    )?))
}

/// Graph-aware compatibility metadata from an explicitly resolved configuration path.
pub fn inference_metadata_from_path_with_graph(
    path: &Path,
    kv_native_dtype: Option<&str>,
    decoder_graph: &ModelGraphInfo,
) -> Result<InferenceMetadata, GenAiConfigError> {
    load(path)?.to_inference_metadata_with_graph(kv_native_dtype, decoder_graph)
}
