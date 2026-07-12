//! Minimal Hugging Face tokenizer adapter.

use std::path::Path;

use crate::{OrtError, Result};
use serde_json::Value;

/// Thin wrapper around `tokenizers::Tokenizer` for prompt/token id conversion.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos_token_ids: Vec<u32>,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| OrtError::Tokenizer(err.to_string()))?;
        let eos_token_ids = load_eos_token_ids(path, &inner)?;
        Ok(Self {
            inner,
            eos_token_ids,
        })
    }

    /// Encode a prompt to token ids, including model-defined special tokens.
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(prompt, true)
            .map_err(|err| OrtError::Tokenizer(err.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode a prompt to `i64` ids for ORT `input_ids` tensors.
    pub fn encode_i64(&self, prompt: &str) -> Result<Vec<i64>> {
        self.encode(prompt)
            .map(|ids| ids.into_iter().map(i64::from).collect())
    }

    /// Decode token ids to text, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|err| OrtError::Tokenizer(err.to_string()))
    }

    /// Decode token ids to text, preserving special tokens such as ChatML tags.
    pub fn decode_with_special_tokens(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|err| OrtError::Tokenizer(err.to_string()))
    }

    /// Decode `i64` ORT token ids to text, skipping special tokens.
    pub fn decode_i64(&self, ids: &[i64]) -> Result<String> {
        let ids = ids
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| {
                    OrtError::InvalidArgument(format!("token id out of u32 range: {id}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.decode(&ids)
    }

    /// Decode `i64` ORT token ids to text, preserving special tokens.
    pub fn decode_i64_with_special_tokens(&self, ids: &[i64]) -> Result<String> {
        let ids = ids
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| {
                    OrtError::InvalidArgument(format!("token id out of u32 range: {id}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.decode_with_special_tokens(&ids)
    }

    /// Look up any token string in the tokenizer vocabulary.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Best-effort EOS id lookup for common tokenizer conventions.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_ids.first().copied()
    }

    /// EOS/stop token ids loaded from generation/tokenizer config.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        self.eos_token_ids.clone()
    }

    /// Access the underlying tokenizer for advanced callers.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

fn load_eos_token_ids(path: &Path, tokenizer: &tokenizers::Tokenizer) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    let model_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let generation_config = model_dir.join("generation_config.json");
    if generation_config.is_file() {
        let value = read_json_file(&generation_config)?;
        collect_generation_eos_ids(value.get("eos_token_id"), &mut ids);
    }

    let tokenizer_config = model_dir.join("tokenizer_config.json");
    if tokenizer_config.is_file() {
        let value = read_json_file(&tokenizer_config)?;
        if let Some(token) = value.get("eos_token").and_then(eos_token_string) {
            push_unique_token_id(&mut ids, tokenizer, token);
        }
    }

    if ids.is_empty() {
        for token in ["<|endoftext|>", "</s>", "<eos>", "[EOS]"] {
            push_unique_token_id(&mut ids, tokenizer, token);
        }
    }

    Ok(ids)
}

fn read_json_file(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|err| {
        OrtError::InvalidArgument(format!("invalid JSON in {}: {err}", path.display()))
    })
}

fn collect_generation_eos_ids(value: Option<&Value>, ids: &mut Vec<u32>) {
    match value {
        Some(Value::Number(number)) => {
            if let Some(id) = number.as_u64().and_then(|id| u32::try_from(id).ok()) {
                push_unique_id(ids, id);
            }
        }
        Some(Value::Array(values)) => {
            for value in values {
                collect_generation_eos_ids(Some(value), ids);
            }
        }
        _ => {}
    }
}

fn eos_token_string(value: &Value) -> Option<&str> {
    match value {
        Value::String(token) => Some(token),
        Value::Object(map) => map.get("content").and_then(Value::as_str),
        _ => None,
    }
}

fn push_unique_token_id(ids: &mut Vec<u32>, tokenizer: &tokenizers::Tokenizer, token: &str) {
    if let Some(id) = tokenizer.token_to_id(token) {
        push_unique_id(ids, id);
    }
}

fn push_unique_id(ids: &mut Vec<u32>, id: u32) {
    if !ids.contains(&id) {
        ids.push(id);
    }
}
