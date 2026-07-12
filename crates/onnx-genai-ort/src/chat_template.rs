//! Hugging Face chat-template rendering.

use std::fmt;
use std::path::Path;

use minijinja::{Environment, context};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::{OrtError, Result};

const DEFAULT_CHAT_TEMPLATE: &str = r#"{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"#;

/// A loaded chat template for a model directory.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    template: String,
}

/// Chat roles understood by common Hugging Face chat templates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Other(String),
}

impl ChatRole {
    pub fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Other(role) => role,
        }
    }
}

impl fmt::Display for ChatRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for ChatRole {
    fn from(value: &str) -> Self {
        match value {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            other => Self::Other(other.to_string()),
        }
    }
}

impl From<String> for ChatRole {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl Serialize for ChatRole {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ChatRole {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Self::from)
    }
}

/// A single chat message passed to a Hugging Face chat template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

impl ChatMessage {
    pub fn new(role: impl Into<ChatRole>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    pub fn with_tool_calls(mut self, tool_calls: Value) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }
}

impl ChatTemplate {
    /// Load `chat_template.jinja` or `tokenizer_config.json` from a model directory.
    ///
    /// A standalone `chat_template.jinja` takes precedence to match ORT-GenAI.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let standalone = model_dir.join("chat_template.jinja");
        if standalone.is_file() {
            return Ok(Self {
                template: std::fs::read_to_string(standalone)?,
            });
        }

        let tokenizer_config = model_dir.join("tokenizer_config.json");
        if tokenizer_config.is_file() {
            let text = std::fs::read_to_string(&tokenizer_config)?;
            let value: Value = serde_json::from_str(&text).map_err(|err| {
                OrtError::InvalidArgument(format!(
                    "invalid JSON in {}: {err}",
                    tokenizer_config.display()
                ))
            })?;
            if let Some(template) = value.get("chat_template").and_then(Value::as_str) {
                return Ok(Self {
                    template: template.to_string(),
                });
            }
        }

        Ok(Self {
            template: DEFAULT_CHAT_TEMPLATE.to_string(),
        })
    }

    /// Render chat messages using this template.
    ///
    /// `tools`, when present, must be a JSON object/array string and is exposed to
    /// templates as the `tools` variable. `add_generation_prompt` is exposed using
    /// the Hugging Face variable name.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: Option<&str>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let tools = match tools {
            Some(tools) => serde_json::from_str::<Value>(tools).map_err(|err| {
                OrtError::InvalidArgument(format!("invalid tools JSON for chat template: {err}"))
            })?,
            None => Value::Null,
        };

        let mut env = Environment::new();
        env.add_filter("tojson", minijinja::filters::tojson);
        env.add_template("chat", &self.template)
            .map_err(|err| OrtError::InvalidArgument(format!("invalid chat template: {err}")))?;
        let template = env
            .get_template("chat")
            .map_err(|err| OrtError::InvalidArgument(format!("invalid chat template: {err}")))?;
        template
            .render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => add_generation_prompt,
            })
            .map_err(|err| OrtError::InvalidArgument(format!("chat template render failed: {err}")))
    }
}
