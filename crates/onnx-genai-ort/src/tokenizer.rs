//! Minimal Hugging Face tokenizer adapter.

use std::path::Path;

use crate::{OrtError, Result};

/// Thin wrapper around `tokenizers::Tokenizer` for prompt/token id conversion.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|err| OrtError::Tokenizer(err.to_string()))?;
        Ok(Self { inner })
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

    /// Look up any token string in the tokenizer vocabulary.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Best-effort EOS id lookup for common tokenizer conventions.
    pub fn eos_token_id(&self) -> Option<u32> {
        ["<|endoftext|>", "</s>", "<eos>", "[EOS]"]
            .iter()
            .find_map(|token| self.token_id(token))
    }

    /// Access the underlying tokenizer for advanced callers.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
