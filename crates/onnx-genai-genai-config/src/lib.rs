//! Compatibility layer that converts an onnxruntime-genai `genai_config.json`
//! into the native onnx-genai [`InferenceMetadata`] spec.
//!
//! onnx-genai's own `inference_metadata.yaml` remains the preferred, canonical
//! source of truth. This crate exists purely as an *auto-detection fallback*:
//! the many ORT-genai / Foundry Local models in the wild ship only a
//! `genai_config.json` and no `inference_metadata.yaml`, yet they carry the same
//! information the runtime needs to pick the fast decode path — most importantly
//! `search.past_present_share_buffer`, which tells the runtime it may own a
//! single max-length KV buffer aliased `present.* -> past_key_values.*` for
//! O(1)/token decode instead of the growing rebind path.
//!
//! The crate is intentionally independent of `onnx-genai-ort`: the KV native
//! dtype (which lives in the ONNX graph, not in `genai_config.json`) is passed
//! in by the caller, so this crate only depends on `serde` and the metadata
//! spec.

use std::path::{Path, PathBuf};

use onnx_genai_metadata::InferenceMetadata;
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Canonical file name onnxruntime-genai uses for its model config.
pub const GENAI_CONFIG_FILE: &str = "genai_config.json";

/// Errors produced while locating, reading, or parsing a `genai_config.json`.
#[derive(Debug, thiserror::Error)]
pub enum GenAiConfigError {
    /// The file could not be read.
    #[error("failed to read genai_config.json: {0}")]
    Io(#[from] std::io::Error),
    /// The file was not valid JSON or did not match the expected shape.
    #[error("failed to parse genai_config.json: {0}")]
    Parse(#[from] serde_json::Error),
}

/// Minimal, forward-compatible view of an onnxruntime-genai `genai_config.json`.
///
/// Only the fields onnx-genai needs are modeled; unknown fields are ignored so
/// future ORT-genai additions do not break loading.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiConfig {
    /// The `model` section.
    pub model: GenAiModel,
    /// The `search` section (generation defaults, incl. share-buffer hint).
    #[serde(default)]
    pub search: GenAiSearch,
}

/// The `model` section of `genai_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiModel {
    /// Maximum total context length in tokens.
    #[serde(default)]
    pub context_length: Option<usize>,
    /// Architecture identifier (e.g. `"qwen2"`).
    #[serde(rename = "type", default)]
    pub model_type: Option<String>,
    /// Decoder graph properties.
    pub decoder: GenAiDecoder,
}

/// The `model.decoder` section of `genai_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiDecoder {
    /// Per-head hidden dimension.
    #[serde(default)]
    pub head_size: Option<usize>,
    /// Number of query/attention heads.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    /// Number of key/value heads (< attention heads implies GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Number of decoder layers.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
}

/// The `search` section of `genai_config.json`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenAiSearch {
    /// Whether the runtime may own a single shared, max-length KV buffer that is
    /// aliased `present.* -> past_key_values.*` across decode steps.
    #[serde(default)]
    pub past_present_share_buffer: bool,
    /// Maximum generated length declared by the model author.
    #[serde(default)]
    pub max_length: Option<usize>,
}

impl GenAiConfig {
    /// Whether the decoder uses grouped/multi-query attention (fewer KV heads
    /// than attention heads).
    pub fn is_group_query_attention(&self) -> bool {
        matches!(
            (
                self.model.decoder.num_key_value_heads,
                self.model.decoder.num_attention_heads,
            ),
            (Some(kv), Some(attn)) if kv < attn
        )
    }

    /// Maximum total sequence length usable to pre-size a shared KV buffer,
    /// preferring the explicit `context_length` then `search.max_length`.
    pub fn max_sequence_length(&self) -> Option<usize> {
        self.model.context_length.or(self.search.max_length)
    }

    /// Whether this model advertises the runtime-owned shared KV buffer path
    /// (share-buffer) for a GQA decoder with a known capacity.
    pub fn shared_kv_buffer_supported(&self) -> bool {
        self.search.past_present_share_buffer
            && self.is_group_query_attention()
            && self.max_sequence_length().is_some()
    }

    /// Convert into native [`InferenceMetadata`].
    ///
    /// `kv_native_dtype` is the KV cache scalar dtype read from the ONNX graph
    /// by the caller (e.g. `"float16"` / `"float32"`); it is not present in
    /// `genai_config.json`. The runtime-owned shared KV buffer path is only
    /// enabled — by emitting `kv_cache.native_dtype` — when the model declares
    /// `search.past_present_share_buffer`, uses GQA, has a known max sequence
    /// length, and a share-buffer-compatible KV dtype is provided. Otherwise
    /// the returned metadata leaves the existing decode paths unchanged.
    pub fn to_inference_metadata(
        &self,
        kv_native_dtype: Option<&str>,
    ) -> Result<InferenceMetadata, GenAiConfigError> {
        let mut attention = Map::new();
        attention.insert(
            "type".into(),
            json!(if self.is_group_query_attention() {
                "group_query_attention"
            } else {
                "multi_head_attention"
            }),
        );
        insert_opt(
            &mut attention,
            "num_kv_heads",
            self.model.decoder.num_key_value_heads,
        );
        insert_opt(
            &mut attention,
            "num_attention_heads",
            self.model.decoder.num_attention_heads,
        );
        insert_opt(&mut attention, "head_dim", self.model.decoder.head_size);

        let mut model = Map::new();
        model.insert("attention".into(), Value::Object(attention));
        insert_opt(
            &mut model,
            "max_sequence_length",
            self.max_sequence_length(),
        );

        let mut root = Map::new();
        root.insert("model".into(), Value::Object(model));

        // Enable the shared KV buffer path only when genai_config declares it and
        // the caller supplied a share-buffer-compatible KV dtype. This mirrors
        // `onnx-genai-engine`'s `shared_kv_buffer_len_from_metadata`, which keys
        // off `kv_cache.native_dtype` + GQA + max_sequence_length.
        if self.shared_kv_buffer_supported()
            && let Some(dtype) = kv_native_dtype
            && is_share_buffer_kv_dtype(dtype)
        {
            root.insert("kv_cache".into(), json!({ "native_dtype": dtype }));
        }

        Ok(serde_json::from_value(Value::Object(root))?)
    }
}

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
/// Returns `Ok(None)` when the directory has no `genai_config.json`. When it
/// does, the config is converted to native [`InferenceMetadata`] using the
/// caller-provided KV dtype (see [`GenAiConfig::to_inference_metadata`]).
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

fn insert_opt(map: &mut Map<String, Value>, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        map.insert(key.into(), json!(value));
    }
}

/// Whether a dtype string denotes a KV dtype the share-buffer GQA path supports
/// (16- or 32-bit floating point). Mirrors the engine's gate.
fn is_share_buffer_kv_dtype(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_lowercase().as_str(),
        "float16" | "fp16" | "half" | "float32" | "fp32" | "float"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_config() -> GenAiConfig {
        serde_json::from_str(
            r#"{
                "model": {
                    "type": "qwen2",
                    "context_length": 32768,
                    "decoder": {
                        "head_size": 64,
                        "hidden_size": 896,
                        "num_attention_heads": 14,
                        "num_hidden_layers": 24,
                        "num_key_value_heads": 2
                    }
                },
                "search": { "past_present_share_buffer": true, "max_length": 32768 }
            }"#,
        )
        .expect("valid genai_config")
    }

    #[test]
    fn detects_gqa_and_capacity() {
        let cfg = qwen_config();
        assert!(cfg.is_group_query_attention());
        assert_eq!(cfg.max_sequence_length(), Some(32768));
        assert!(cfg.shared_kv_buffer_supported());
    }

    #[test]
    fn converts_and_enables_share_buffer_with_fp16() {
        let cfg = qwen_config();
        let md = cfg.to_inference_metadata(Some("float16")).unwrap();
        let attention = md
            .model
            .as_ref()
            .and_then(|m| m.attention.as_ref())
            .expect("attention");
        assert_eq!(attention.attention_type, "group_query_attention");
        assert_eq!(attention.num_kv_heads, Some(2));
        assert_eq!(attention.num_attention_heads, Some(14));
        assert_eq!(attention.head_dim, Some(64));
        assert_eq!(
            md.model.as_ref().and_then(|m| m.max_sequence_length),
            Some(32768)
        );
        assert_eq!(
            md.kv_cache
                .as_ref()
                .and_then(|kv| kv.native_dtype.as_deref()),
            Some("float16")
        );
    }

    #[test]
    fn omits_kv_cache_when_share_buffer_disabled() {
        let mut cfg = qwen_config();
        cfg.search.past_present_share_buffer = false;
        let md = cfg.to_inference_metadata(Some("float16")).unwrap();
        assert!(md.kv_cache.is_none());
    }

    #[test]
    fn omits_kv_cache_for_unsupported_dtype() {
        let cfg = qwen_config();
        let md = cfg.to_inference_metadata(Some("int8")).unwrap();
        assert!(md.kv_cache.is_none());
    }

    #[test]
    fn omits_kv_cache_when_dtype_unknown() {
        let cfg = qwen_config();
        let md = cfg.to_inference_metadata(None).unwrap();
        assert!(md.kv_cache.is_none());
        // Attention info is still populated for non-share-buffer models.
        assert!(md.model.and_then(|m| m.attention).is_some());
    }

    #[test]
    fn non_gqa_model_is_multi_head() {
        let mut cfg = qwen_config();
        cfg.model.decoder.num_key_value_heads = Some(14);
        let md = cfg.to_inference_metadata(Some("float16")).unwrap();
        assert!(!cfg.is_group_query_attention());
        assert!(md.kv_cache.is_none());
        assert_eq!(
            md.model.and_then(|m| m.attention).map(|a| a.attention_type),
            Some("multi_head_attention".to_string())
        );
    }
}
