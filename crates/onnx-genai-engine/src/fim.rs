//! Fill-in-the-middle prompt formatting.

use serde_json::Value;

/// Fill-in-the-middle prompt ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FimFormat {
    /// Prefix → Suffix → Middle.
    PSM,
    /// Suffix → Prefix → Middle.
    SPM,
}

/// Fill-in-the-middle special-token configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FimConfig {
    pub prefix_token: String,
    pub middle_token: String,
    pub suffix_token: String,
    pub format: FimFormat,
}

impl FimConfig {
    /// Auto-detect common FIM special tokens from a tokenizer_config.json value.
    pub fn from_tokenizer_config(config: &Value) -> Option<Self> {
        for (prefix_token, suffix_token, middle_token) in [
            ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
            ("<PRE>", "<SUF>", "<MID>"),
        ] {
            if tokenizer_config_mentions_all(config, [prefix_token, suffix_token, middle_token]) {
                return Some(Self {
                    prefix_token: prefix_token.to_string(),
                    middle_token: middle_token.to_string(),
                    suffix_token: suffix_token.to_string(),
                    format: FimFormat::PSM,
                });
            }
        }
        None
    }

    /// Format a FIM prompt for the configured model.
    pub fn format_prompt(&self, prefix: &str, suffix: &str) -> String {
        match self.format {
            FimFormat::PSM => format!(
                "{}{}{}{}{}",
                self.prefix_token, prefix, self.suffix_token, suffix, self.middle_token
            ),
            FimFormat::SPM => format!(
                "{}{}{}{}{}",
                self.suffix_token, suffix, self.prefix_token, prefix, self.middle_token
            ),
        }
    }
}

fn tokenizer_config_mentions_all(config: &Value, tokens: [&str; 3]) -> bool {
    tokens
        .into_iter()
        .all(|token| tokenizer_config_mentions(config, token))
}

fn tokenizer_config_mentions(value: &Value, token: &str) -> bool {
    match value {
        Value::String(text) => text.contains(token),
        Value::Array(values) => values
            .iter()
            .any(|value| tokenizer_config_mentions(value, token)),
        Value::Object(map) => map
            .values()
            .any(|value| tokenizer_config_mentions(value, token)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_fim_pipe_tokens_from_tokenizer_config() {
        let config = json!({
            "added_tokens_decoder": {
                "151659": { "content": "<|fim_prefix|>", "special": true },
                "151660": { "content": "<|fim_middle|>", "special": true },
                "151661": { "content": "<|fim_suffix|>", "special": true }
            }
        });

        let fim = FimConfig::from_tokenizer_config(&config).expect("FIM config");

        assert_eq!(fim.prefix_token, "<|fim_prefix|>");
        assert_eq!(fim.middle_token, "<|fim_middle|>");
        assert_eq!(fim.suffix_token, "<|fim_suffix|>");
        assert_eq!(fim.format, FimFormat::PSM);
    }

    #[test]
    fn detects_pre_mid_suf_tokens_from_tokenizer_config() {
        let config = json!({
            "additional_special_tokens": ["<PRE>", "<MID>", "<SUF>"]
        });

        let fim = FimConfig::from_tokenizer_config(&config).expect("FIM config");

        assert_eq!(fim.prefix_token, "<PRE>");
        assert_eq!(fim.middle_token, "<MID>");
        assert_eq!(fim.suffix_token, "<SUF>");
        assert_eq!(fim.format, FimFormat::PSM);
    }

    #[test]
    fn formats_psm_prompt() {
        let fim = FimConfig {
            prefix_token: "<|fim_prefix|>".to_string(),
            middle_token: "<|fim_middle|>".to_string(),
            suffix_token: "<|fim_suffix|>".to_string(),
            format: FimFormat::PSM,
        };

        assert_eq!(
            fim.format_prompt("fn main() {", "}"),
            "<|fim_prefix|>fn main() {<|fim_suffix|>}<|fim_middle|>"
        );
    }

    #[test]
    fn formats_spm_prompt() {
        let fim = FimConfig {
            prefix_token: "<PRE>".to_string(),
            middle_token: "<MID>".to_string(),
            suffix_token: "<SUF>".to_string(),
            format: FimFormat::SPM,
        };

        assert_eq!(
            fim.format_prompt("left", "right"),
            "<SUF>right<PRE>left<MID>"
        );
    }
}
