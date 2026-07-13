//! Loading of onnxruntime-genai `genai_config.json` runtime hints.
//!
//! WebGPU GroupQueryAttention (GQA) exports ship a `genai_config.json` that
//! declares on-device KV cache behavior. In particular the `search` block sets
//! `past_present_share_buffer` (the model reuses/aliases one physical KV buffer
//! across decode steps) and `max_length` (the buffer is pre-sized to this many
//! tokens). The engine honors these so GQA models bind a device-resident,
//! max-length shared KV buffer instead of round-tripping fp32 KV through a host
//! cache each step.

use anyhow::Context;
use std::path::Path;

/// Runtime KV hints parsed from `genai_config.json`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GenaiRuntimeConfig {
    /// `search.past_present_share_buffer` — the model shares one KV buffer
    /// between `past_key_values.*` inputs and `present.*` outputs.
    pub(crate) past_present_share_buffer: Option<bool>,
    /// `search.max_length` — logical token capacity used to pre-size the shared
    /// KV buffer.
    pub(crate) max_length: Option<usize>,
    /// `model.context_length` — model context window, used as a fallback when
    /// `max_length` is absent.
    pub(crate) context_length: Option<usize>,
}

impl GenaiRuntimeConfig {
    /// Preferred capacity for pre-sizing the shared KV buffer / bounding context.
    pub(crate) fn effective_max_length(&self) -> Option<usize> {
        self.max_length.or(self.context_length)
    }
}

/// Load `genai_config.json` from a model directory, if present.
///
/// Missing or malformed files are tolerated: a missing file returns `Ok(None)`
/// and a malformed file returns an error only when it exists but cannot be
/// parsed as JSON, so non-GQA models without this file keep loading unchanged.
pub(crate) fn load_genai_config_from_model_dir(
    model_dir: &Path,
) -> anyhow::Result<Option<GenaiRuntimeConfig>> {
    let path = model_dir.join("genai_config.json");
    if !path.is_file() {
        return Ok(None);
    }

    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("invalid JSON in {}", path.display()))?;
    Ok(Some(parse_genai_config(&value)))
}

/// Extract runtime KV hints from a parsed `genai_config.json` document.
fn parse_genai_config(value: &serde_json::Value) -> GenaiRuntimeConfig {
    let model = value.get("model");
    let search = value.get("search");
    GenaiRuntimeConfig {
        past_present_share_buffer: search
            .and_then(|search| search.get("past_present_share_buffer"))
            .and_then(serde_json::Value::as_bool),
        max_length: search
            .and_then(|search| search.get("max_length"))
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize),
        context_length: model
            .and_then(|model| model.get("context_length"))
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_none_when_config_absent() -> anyhow::Result<()> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("definitely-missing-model-dir");
        assert_eq!(load_genai_config_from_model_dir(&dir)?, None);
        Ok(())
    }

    #[test]
    fn parses_share_buffer_and_lengths() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{
                "model": { "context_length": 32768 },
                "search": { "past_present_share_buffer": true, "max_length": 4096 }
            }"#,
        )
        .expect("valid json");
        let config = parse_genai_config(&value);
        assert_eq!(config.past_present_share_buffer, Some(true));
        assert_eq!(config.max_length, Some(4096));
        assert_eq!(config.context_length, Some(32768));
        assert_eq!(config.effective_max_length(), Some(4096));
    }

    #[test]
    fn parses_missing_fields_as_none() {
        let value: serde_json::Value = serde_json::from_str(r#"{ "model": {} }"#).expect("json");
        let config = parse_genai_config(&value);
        assert_eq!(config.past_present_share_buffer, None);
        assert_eq!(config.max_length, None);
        assert_eq!(config.context_length, None);
        assert_eq!(config.effective_max_length(), None);
    }

    #[test]
    fn falls_back_to_context_length() {
        let config = GenaiRuntimeConfig {
            past_present_share_buffer: Some(true),
            max_length: None,
            context_length: Some(8192),
        };
        assert_eq!(config.effective_max_length(), Some(8192));
    }
}
