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
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| OrtError::Tokenizer(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode a prompt to token ids, including model-defined special tokens.
    ///
    /// Chat templates usually emit `bos_token` themselves, while the tokenizer's
    /// post-processor prepends the same token again. Feeding a duplicated BOS to
    /// the model corrupts the very first attention step and derails generation,
    /// so drop the extra copy when the post-processor produced one.
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let ids = self.encode_inner(prompt, true)?;
        if ids.len() < 2 || ids[0] != ids[1] {
            return Ok(ids);
        }
        let plain = self.encode_inner(prompt, false)?;
        if plain.first() == ids.first() && ids.len() == plain.len() + 1 {
            return Ok(plain);
        }
        Ok(ids)
    }

    fn encode_inner(&self, prompt: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(prompt, add_special_tokens)
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

    /// Access the underlying tokenizer for advanced callers.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
