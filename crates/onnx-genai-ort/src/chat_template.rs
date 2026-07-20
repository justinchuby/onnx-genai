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
    bos_token: Option<String>,
    eos_token: Option<String>,
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
    /// A `ChatTemplate` backed by the built-in default (`DEFAULT_CHAT_TEMPLATE`).
    ///
    /// Model-independent — needs no model directory. Rendering this template is
    /// identical to what [`ChatTemplate::from_model_dir`] yields when a model ships
    /// no `chat_template.jinja` and no `chat_template` in `tokenizer_config.json`:
    /// it emits `role: content` lines plus an optional `assistant:` generation prompt.
    pub fn builtin_default() -> Self {
        Self {
            template: DEFAULT_CHAT_TEMPLATE.to_string(),
            bos_token: None,
            eos_token: None,
        }
    }

    /// Load `chat_template.jinja` or `tokenizer_config.json` from a model directory.
    ///
    /// A standalone `chat_template.jinja` takes precedence to match ORT-GenAI.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let tokenizer_config = model_dir.join("tokenizer_config.json");
        let tokenizer_config_value = if tokenizer_config.is_file() {
            let text = std::fs::read_to_string(&tokenizer_config)?;
            Some(serde_json::from_str::<Value>(&text).map_err(|err| {
                OrtError::InvalidArgument(format!(
                    "invalid JSON in {}: {err}",
                    tokenizer_config.display()
                ))
            })?)
        } else {
            None
        };
        let (bos_token, eos_token) = tokenizer_config_value
            .as_ref()
            .map(special_tokens)
            .unwrap_or_default();

        let standalone = model_dir.join("chat_template.jinja");
        if standalone.is_file() {
            return Ok(Self {
                template: std::fs::read_to_string(standalone)?,
                bos_token,
                eos_token,
            });
        }

        if let Some(value) = tokenizer_config_value {
            if let Some(template) = value.get("chat_template").and_then(Value::as_str) {
                return Ok(Self {
                    template: template.to_string(),
                    bos_token,
                    eos_token,
                });
            }
        }

        Ok(Self {
            template: DEFAULT_CHAT_TEMPLATE.to_string(),
            bos_token,
            eos_token,
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
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // Hugging Face chat templates are authored for Jinja2 on Python, so they
        // freely call Python string methods (`startswith`, `endswith`, `split`,
        // `strip`/`lstrip`/`rstrip`, `title`, ...) that minijinja does not expose
        // natively. `minijinja-contrib`'s pycompat callback resolves those method
        // calls; without it real-world templates (e.g. qwen3) fail to render.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", minijinja::filters::tojson);
        env.add_function("raise_exception", raise_exception);
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
                bos_token => self.bos_token.as_deref().unwrap_or_default(),
                eos_token => self.eos_token.as_deref().unwrap_or_default(),
            })
            .map_err(|err| OrtError::InvalidArgument(format!("chat template render failed: {err}")))
    }
}

fn special_tokens(config: &Value) -> (Option<String>, Option<String>) {
    (
        special_token(config, "bos_token"),
        special_token(config, "eos_token"),
    )
}

fn special_token(config: &Value, key: &str) -> Option<String> {
    let value = config.get(key)?;
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
        .map(ToString::to_string)
}

fn raise_exception(message: String) -> std::result::Result<(), minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        message,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn sample_messages() -> Vec<ChatMessage> {
        vec![ChatMessage::system("be brief"), ChatMessage::user("hello")]
    }

    #[test]
    fn builtin_default_matches_from_model_dir_default_path() {
        // A directory with no template files yields the built-in default from
        // `from_model_dir`; a non-existent path exercises that same fallback
        // without touching the filesystem.
        let from_dir =
            ChatTemplate::from_model_dir(Path::new("nonexistent-model-dir-for-test")).unwrap();
        let builtin = ChatTemplate::builtin_default();

        let messages = sample_messages();
        for add_generation_prompt in [false, true] {
            assert_eq!(
                builtin
                    .render(&messages, None, add_generation_prompt)
                    .unwrap(),
                from_dir
                    .render(&messages, None, add_generation_prompt)
                    .unwrap(),
            );
        }
    }

    #[test]
    fn builtin_default_renders_role_content_lines_and_generation_prompt() {
        let messages = sample_messages();
        let without = ChatTemplate::builtin_default()
            .render(&messages, None, false)
            .unwrap();
        assert_eq!(without, "system: be brief\nuser: hello\n");

        let with = ChatTemplate::builtin_default()
            .render(&messages, None, true)
            .unwrap();
        assert_eq!(with, "system: be brief\nuser: hello\nassistant: ");
    }

    #[test]
    fn standalone_template_loads_string_and_object_special_tokens() {
        let dir = test_dir("special-tokens");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("chat_template.jinja"),
            "{{ bos_token }}|{{ eos_token }}",
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"bos_token":{"content":"<bos>","lstrip":false},"eos_token":"<eos>"}"#,
        )
        .unwrap();

        let rendered = ChatTemplate::from_model_dir(&dir)
            .unwrap()
            .render(&[], None, false)
            .unwrap();
        assert_eq!(rendered, "<bos>|<eos>");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn render_supports_python_string_methods_used_by_hf_templates() {
        // Real HF templates (qwen3, etc.) call Python str methods that minijinja
        // lacks natively; the pycompat callback must resolve them.
        let template = ChatTemplate {
            template: concat!(
                "{{ 'hello world' is string }}",
                "|{{ '<tool_response>x</tool_response>'.startswith('<tool_response>') }}",
                "|{{ '<tool_response>x</tool_response>'.endswith('</tool_response>') }}",
                "|{{ 'a</think>b'.split('</think>')[-1] }}",
                "|{{ '\n keep \n'.strip('\n') }}"
            )
            .to_string(),
            bos_token: None,
            eos_token: None,
        };

        assert_eq!(
            template.render(&[], None, false).unwrap(),
            "true|true|true|b| keep "
        );
    }

    #[test]
    fn raise_exception_returns_render_error() {
        let template = ChatTemplate {
            template: "{{ raise_exception('invalid messages') }}".to_string(),
            bos_token: None,
            eos_token: None,
        };

        let error = template.render(&[], None, false).unwrap_err();
        assert!(error.to_string().contains("invalid messages"));
    }

    #[test]
    fn render_uses_hugging_face_block_whitespace_controls() {
        let template = ChatTemplate {
            template: "before\n    {% if true %}\n    value\n    {% endif %}\nafter".to_string(),
            bos_token: None,
            eos_token: None,
        };

        assert_eq!(
            template.render(&[], None, false).unwrap(),
            "before\n    value\nafter"
        );
    }

    fn test_dir(name: &str) -> PathBuf {
        std::env::current_dir().unwrap().join(format!(
            "chat-template-test-{}-{}",
            std::process::id(),
            name
        ))
    }
}
