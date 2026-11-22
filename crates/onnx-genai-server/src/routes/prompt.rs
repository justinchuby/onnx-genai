//! Native text/token mixtures share the upstream insertion policy across routes.
use super::{ApiError, Tokenizer};
use onnx_genai_ort::TokenizationOptions;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

#[derive(Debug)]
pub(super) enum PromptPart {
    Text(String),
    Token(u32),
}
#[derive(Debug)]
pub(super) enum MixedPrompt {
    Text(String),
    Parts(Vec<PromptPart>),
}
impl<'de> Deserialize<'de> for MixedPrompt {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::parse(Value::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}
impl MixedPrompt {
    pub(super) fn parse_completion(value: Value) -> Result<Self, String> {
        let Value::Object(mut object) = value else {
            return Self::parse(value);
        };
        let prompt = object.remove("prompt_string").ok_or_else(|| {
            "prompt object requires prompt_string containing text or a mixed text/token row"
                .to_owned()
        })?;
        if let Some(media) = object.remove("multimodal_data")
            && media != serde_json::json!([])
        {
            return Err("multimodal_data requires native media preparation; use the chat content API with declared model media inputs".into());
        }
        if !object.is_empty() {
            return Err(format!(
                "unsupported prompt object fields {:?}; send prompt_string and omit media",
                object.keys().collect::<Vec<_>>()
            ));
        }
        Self::parse(prompt)
    }
    pub(super) fn parse(value: Value) -> Result<Self, String> {
        match value {
            Value::String(text) => Ok(Self::Text(text)),
            Value::Array(parts) => parts.into_iter().enumerate().map(|(index, part)| {
                match part {
                    Value::String(text) => Ok(PromptPart::Text(text)),
                    Value::Number(number) => number.as_u64().and_then(|id| u32::try_from(id).ok())
                        .map(PromptPart::Token)
                        .ok_or_else(||format!("input[{index}] must be an unsigned 32-bit token id; negative and fractional ids are invalid")),
                    _ => Err(format!("input[{index}] must be text or an unsigned token id; send a flat text/token row here")),
                }
            }).collect::<Result<Vec<_>, _>>().map(Self::Parts),
            _ => Err("input must be text or an array mixing text and unsigned token ids".into()),
        }
    }
    pub(super) fn encode(
        &self,
        tokenizer: &Tokenizer,
        options: TokenizationOptions,
    ) -> Result<Vec<u32>, ApiError> {
        let encode = |text: &str, add_special| {
            tokenizer.encode_with_options(text, TokenizationOptions {
            add_special, ..options
        }).map_err(|error|ApiError::invalid_request(format!("prompt/content tokenization failed: {error}; check the package tokenizer.json")))
        };
        match self {
            Self::Text(text) => encode(text, options.add_special),
            Self::Parts(parts) => {
                let mut ids = Vec::new();
                for (index, part) in parts.iter().enumerate() {
                    match part {
                        PromptPart::Text(text) => {
                            ids.extend(encode(text, index == 0 && options.add_special)?)
                        }
                        PromptPart::Token(id) => {
                            if tokenizer.inner().id_to_token(*id).is_none() {
                                return Err(ApiError::invalid_request(format!(
                                    "input[{index}] token id {id} is absent from the selected model vocabulary; tokenize with the same model"
                                )));
                            }
                            ids.push(*id);
                        }
                    }
                }
                Ok(ids)
            }
        }
    }
}
