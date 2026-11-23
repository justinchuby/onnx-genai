//! Hugging Face chat-template rendering.

use std::borrow::Cow;
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
    tool_template: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    /// OpenAI's optional message `name`, which tool templates read to label a
    /// tool result with the function that produced it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The assistant tool call this message answers, which tool templates use
    /// to recover the function name when the caller sends no `name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn new(role: impl Into<ChatRole>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            name: None,
            tool_call_id: None,
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

    /// Attach the OpenAI tool-result identity a template needs to name the
    /// function a `tool` message answers.
    pub fn with_tool_result(mut self, name: Option<String>, tool_call_id: Option<String>) -> Self {
        self.name = name;
        self.tool_call_id = tool_call_id;
        self
    }
}

impl ChatTemplate {
    /// The resolved base/default template source this model ships.
    ///
    /// Exposed so a front end can read the conventions a package declares —
    /// reasoning delimiters, for instance — from the model's own data rather
    /// than inferring them from its name.
    pub fn source(&self) -> &str {
        &self.template
    }

    /// The explicit tool-use source, if this package ships a separate variant.
    pub fn tool_source(&self) -> Option<&str> {
        self.tool_template.as_deref()
    }

    /// A `ChatTemplate` backed by the built-in default (`DEFAULT_CHAT_TEMPLATE`).
    ///
    /// Model-independent — needs no model directory. Rendering this template is
    /// identical to what [`ChatTemplate::from_model_dir`] yields when a model ships
    /// no `chat_template.jinja` and no `chat_template` in `tokenizer_config.json`:
    /// it emits `role: content` lines plus an optional `assistant:` generation prompt.
    pub fn builtin_default() -> Self {
        Self {
            template: DEFAULT_CHAT_TEMPLATE.to_string(),
            tool_template: None,
            bos_token: None,
            eos_token: None,
        }
    }

    /// A `ChatTemplate` backed by an explicit Jinja source, carrying no
    /// `bos_token`/`eos_token` of its own.
    ///
    /// Lets a caller render against a template it already holds — one embedded
    /// in a package manifest, say — instead of a model directory on disk.
    pub fn from_source(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            tool_template: None,
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
        if let Some(template) = Self::from_model_assets(model_dir, tokenizer_config_value.as_ref())?
        {
            return Ok(template);
        }
        let (bos_token, eos_token) = tokenizer_config_value
            .as_ref()
            .map(special_tokens)
            .unwrap_or_default();
        Ok(Self {
            template: DEFAULT_CHAT_TEMPLATE.into(),
            tool_template: None,
            bos_token,
            eos_token,
        })
    }

    /// Load declared single/named templates from one tokenizer-config snapshot.
    /// Independent Jinja assets take precedence over config entries. Absence is
    /// explicit so a caller may choose its own documented fallback formatter.
    pub fn from_model_assets(model_dir: &Path, config: Option<&Value>) -> Result<Option<Self>> {
        let (bos_token, eos_token) = config.map(special_tokens).unwrap_or_default();
        let mut templates = std::collections::BTreeMap::new();
        let base = model_dir.join("chat_template.jinja");
        if let Some(source) = read_optional_template(&base)? {
            templates.insert("default".to_owned(), source);
        }
        let directory = model_dir.join("chat_templates");
        match std::fs::read_dir(&directory) {
            Ok(entries) => {
                for entry in entries {
                    let entry = entry.map_err(|error| template_io_error(&directory, error))?;
                    let path = entry.path();
                    if path
                        .extension()
                        .is_none_or(|extension| extension != "jinja")
                    {
                        continue;
                    }
                    let name=path.file_stem().and_then(|name|name.to_str()).filter(|name|!name.is_empty())
                    .ok_or_else(||OrtError::InvalidArgument(format!("chat template {} needs a nonempty UTF-8 name; rename the Jinja file",path.display())))?;
                    let source = std::fs::read_to_string(&path)
                        .map_err(|error| template_io_error(&path, error))?;
                    if templates.insert(name.to_owned(), source).is_some() {
                        return Err(OrtError::InvalidArgument(format!(
                            "duplicate chat template name {name:?} in {}; keep one independent Jinja source for this name",
                            model_dir.display()
                        )));
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(template_io_error(&directory, error)),
        }
        if templates.is_empty() {
            let Some(value) = config
                .and_then(|config| config.get("chat_template"))
                .filter(|value| !value.is_null())
            else {
                return Ok(None);
            };
            if let Some(source) = value.as_str() {
                if source.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(Self {
                    template: source.to_owned(),
                    tool_template: None,
                    bos_token,
                    eos_token,
                }));
            }
            let scope = format!(
                "{} chat_template",
                model_dir.join("tokenizer_config.json").display()
            );
            if let Some(map) = value.as_object() {
                for (name, source) in map {
                    insert_named_template(&mut templates, name, source, &scope)?;
                }
            } else if let Some(entries) = value.as_array() {
                for (index, entry) in entries.iter().enumerate() {
                    let entry_scope = format!("{scope}[{index}]");
                    let name=entry.get("name").and_then(Value::as_str).ok_or_else(||OrtError::InvalidArgument(format!("{entry_scope}.name must be a nonempty string; declare the template name")))?;
                    let source=entry.get("template").ok_or_else(||OrtError::InvalidArgument(format!("{entry_scope}.template must be a Jinja string; provide the named source")))?;
                    insert_named_template(&mut templates, name, source, &entry_scope)?;
                }
            } else {
                return Err(OrtError::InvalidArgument(format!(
                    "{scope} must be a Jinja string, name-to-source object or list of name/template objects; repair the declared templates"
                )));
            }
        }
        if templates.values().all(String::is_empty) {
            return Ok(None);
        }
        let template=templates.get("default").filter(|source|!source.is_empty()).or_else(||templates.get("tool_use").filter(|source|!source.is_empty()))
            .ok_or_else(||OrtError::InvalidArgument(format!("chat templates in {} need a default or tool_use source; available names are {:?}; provide chat_template.jinja or name the base template default",model_dir.display(),templates.keys().collect::<Vec<_>>())))?.clone();
        Ok(Some(Self {
            template,
            tool_template: templates
                .remove("tool_use")
                .filter(|source| !source.is_empty()),
            bos_token,
            eos_token,
        }))
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
        self.render_with_reasoning_effort(messages, tools, add_generation_prompt, None)
    }

    /// Render chat messages, selecting how much the model should reason first.
    ///
    /// A reasoning model's template decides how much thinking to request from a
    /// keyword the caller supplies, defaulting to its own (often maximal) value
    /// when none arrives — which is why an agent's turn can otherwise spend its
    /// whole token budget thinking. There are two spellings of that keyword in
    /// the wild, `reasoning_effort` and `reasoning_strength`, for one concept,
    /// so the caller's choice is exposed under both and the template reads
    /// whichever it was authored against.
    pub fn render_with_reasoning_effort(
        &self,
        messages: &[ChatMessage],
        tools: Option<&str>,
        add_generation_prompt: bool,
        reasoning_effort: Option<&str>,
    ) -> Result<String> {
        self.render_with_kwargs(
            messages,
            tools,
            add_generation_prompt,
            reasoning_effort,
            &Default::default(),
        )
    }

    /// Reject structural variable overrides before rendering or request admission.
    pub fn validate_kwargs(kwargs: &std::collections::BTreeMap<String, Value>) -> Result<()> {
        for key in [
            "messages",
            "tools",
            "add_generation_prompt",
            "bos_token",
            "eos_token",
        ] {
            if kwargs.contains_key(key) {
                return Err(OrtError::InvalidArgument(format!(
                    "chat_template_kwargs.{key} is reserved; set the corresponding request field instead"
                )));
            }
        }
        Ok(())
    }

    /// Render additional template variables supplied by a protocol client.
    /// Structural variables remain owned by the renderer.
    pub fn render_with_kwargs(
        &self,
        messages: &[ChatMessage],
        tools: Option<&str>,
        add_generation_prompt: bool,
        reasoning_effort: Option<&str>,
        kwargs: &std::collections::BTreeMap<String, Value>,
    ) -> Result<String> {
        Self::validate_kwargs(kwargs)?;
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
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", raise_exception);
        let has_tools=match &tools {
            Value::Null=>false,
            Value::Array(tools)=>!tools.is_empty(),
            Value::Object(tools)=>!tools.is_empty(),
            _=>return Err(OrtError::InvalidArgument("chat-template tools must be a JSON array, object or null; provide structured tool definitions".into())),
        };
        let source = if has_tools {
            self.tool_template.as_deref().unwrap_or(&self.template)
        } else {
            &self.template
        };
        let template_source = normalize_hf_jinja(source);
        env.add_template("chat", &template_source)
            .map_err(|err| OrtError::InvalidArgument(format!("invalid chat template: {err}")))?;
        let template = env
            .get_template("chat")
            .map_err(|err| OrtError::InvalidArgument(format!("invalid chat template: {err}")))?;
        template
            .render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => add_generation_prompt,
                reasoning_effort => reasoning_effort,
                reasoning_strength => reasoning_effort,
                bos_token => self.bos_token.as_deref().unwrap_or_default(),
                eos_token => self.eos_token.as_deref().unwrap_or_default(),
                ..minijinja::Value::from_serialize(kwargs),
            })
            .map_err(|err| OrtError::InvalidArgument(format!("chat template render failed: {err}")))
    }
}

#[derive(Debug, thiserror::Error)]
#[error(
    "read chat template asset {path}: {source}; check the asset path, UTF-8 encoding and read permissions"
)]
struct TemplateAssetReadError {
    path: std::path::PathBuf,
    #[source]
    source: std::io::Error,
}
fn template_io_error(path: &Path, error: std::io::Error) -> OrtError {
    std::io::Error::new(
        error.kind(),
        TemplateAssetReadError {
            path: path.to_owned(),
            source: error,
        },
    )
    .into()
}
fn read_optional_template(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(source) => Ok(Some(source)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(template_io_error(path, error)),
    }
}
fn insert_named_template(
    templates: &mut std::collections::BTreeMap<String, String>,
    name: &str,
    source: &Value,
    scope: &str,
) -> Result<()> {
    if name.trim().is_empty() {
        return Err(OrtError::InvalidArgument(format!(
            "{scope} template name must not be empty; give each source a unique name"
        )));
    }
    let source = source.as_str().ok_or_else(|| {
        OrtError::InvalidArgument(format!(
            "{scope} template {name:?} must contain a Jinja string; repair its source"
        ))
    })?;
    if templates
        .insert(name.to_owned(), source.to_owned())
        .is_some()
    {
        return Err(OrtError::InvalidArgument(format!(
            "{scope} duplicates template {name:?}; keep one source per name"
        )));
    }
    Ok(())
}

/// Parenthesize the single-keyword conditional form emitted by some Hugging
/// Face templates, such as `namespace(name=value if value else '')`.
///
/// Jinja2 accepts that expression directly. MiniJinja parses the `if` as a new
/// call argument unless the conditional value is parenthesized.
fn normalize_hf_jinja(template: &str) -> Cow<'_, str> {
    let mut normalized = None::<String>;
    let mut copied = 0;
    let mut search_from = 0;

    while let Some(relative_start) = template[search_from..].find("namespace(") {
        let start = search_from + relative_start;
        let body_start = start + "namespace(".len();
        let Some(relative_end) = template[body_start..].find(')') else {
            break;
        };
        let end = body_start + relative_end;
        let body = &template[body_start..end];
        let Some((name, value)) = body.split_once('=') else {
            search_from = end + 1;
            continue;
        };
        let name = name.trim();
        let value = value.trim();
        let is_single_keyword = !name.is_empty()
            && name
                .chars()
                .all(|character| character == '_' || character.is_ascii_alphanumeric())
            && !value.contains(',');
        let is_unparenthesized_conditional = value.contains(" if ")
            && value.contains(" else ")
            && !(value.starts_with('(') && value.ends_with(')'));
        if !is_single_keyword || !is_unparenthesized_conditional {
            search_from = end + 1;
            continue;
        }

        let output = normalized.get_or_insert_with(|| String::with_capacity(template.len() + 2));
        output.push_str(&template[copied..body_start]);
        output.push_str(name);
        output.push_str("=(");
        output.push_str(value);
        output.push_str("))");
        copied = end + 1;
        search_from = end + 1;
    }

    match normalized {
        Some(mut normalized) => {
            normalized.push_str(&template[copied..]);
            Cow::Owned(normalized)
        }
        None => Cow::Borrowed(template),
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

/// `tojson` that emits plain JSON, the way Hugging Face chat templates expect.
///
/// MiniJinja's built-in filter escapes `<`, `>` and `&` as `\u003c`, `\u003e`
/// and `\u0026` so the output is safe to embed in HTML, and writes compact
/// `{"a":1}` separators. A chat prompt is not HTML: Transformers registers a
/// `tojson` backed by `json.dumps`, and llama.cpp mirrors it, so a tool schema
/// containing those characters -- which any shell or glob tool description has
/// -- reaches the model verbatim and spaced there but mangled here. Models
/// trained on the unescaped form can then fail to recognize their own tool
/// definitions and never stop reasoning.
///
/// This matches `json.dumps`: unescaped text, `", "` / `": "` separators when
/// compact, and the document's own key order (serde_json is built with
/// `preserve_order`).
fn tojson(
    value: minijinja::Value,
    kwargs: minijinja::value::Kwargs,
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let to_error = |err: serde_json::Error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "cannot serialize to JSON",
        )
        .with_source(err)
    };
    let mut buffer = Vec::new();
    match indent {
        Some(indent) => {
            let indent = " ".repeat(indent);
            let formatter = serde_json::ser::PrettyFormatter::with_indent(indent.as_bytes());
            let mut serializer = serde_json::Serializer::with_formatter(&mut buffer, formatter);
            serde::Serialize::serialize(&value, &mut serializer).map_err(to_error)?;
        }
        None => {
            let mut serializer =
                serde_json::Serializer::with_formatter(&mut buffer, PythonCompactFormatter);
            serde::Serialize::serialize(&value, &mut serializer).map_err(to_error)?;
        }
    }
    let rendered = String::from_utf8(buffer).map_err(|err| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "JSON serialization produced invalid UTF-8",
        )
        .with_source(err)
    })?;
    Ok(minijinja::Value::from_safe_string(rendered))
}

/// serde_json formatter matching Python's default `json.dumps` separators.
///
/// `json.dumps` writes `", "` between items and `": "` after a key unless an
/// indent is requested; serde_json's default omits both spaces. Chat templates
/// splice tool schemas straight into the prompt, so that whitespace is part of
/// what the model was trained to see.
struct PythonCompactFormatter;

impl serde_json::ser::Formatter for PythonCompactFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
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
    fn named_templates_select_declared_default_and_nonempty_tool_variants() {
        let dir = tempfile::tempdir().unwrap();
        let base = "{{ bos_token }}plain:{{ messages[0].content }}";
        let tool = "{{ bos_token }}tools={{ tools|length }}:{{ messages[0].content }}";
        for named in [
            serde_json::json!({"default":base,"tool_use":tool}),
            serde_json::json!([
                {"name":"default","template":base},{"name":"tool_use","template":tool}
            ]),
        ] {
            let config = serde_json::json!({"chat_template":named,"bos_token":{"content":"<B>"}});
            let template = ChatTemplate::from_model_assets(dir.path(), Some(&config))
                .unwrap()
                .unwrap();
            assert_eq!(template.source(), base);
            assert_eq!(template.tool_source(), Some(tool));
            let messages = [ChatMessage::user("hello")];
            for tools in [None, Some("[]"), Some("{}"), Some("null")] {
                assert_eq!(
                    template.render(&messages, tools, false).unwrap(),
                    "<B>plain:hello"
                );
            }
            assert_eq!(
                template
                    .render(&messages, Some(r#"[{"type":"function"}]"#), false)
                    .unwrap(),
                "<B>tools=1:hello"
            );
            for invalid in ["42", "true", "[invalid"] {
                assert!(
                    template
                        .render(&messages, Some(invalid), false)
                        .unwrap_err()
                        .to_string()
                        .contains("tools")
                );
            }
        }
    }

    #[test]
    fn named_template_assets_override_config_and_tool_only_assets_supply_the_base() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("chat_templates")).unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{{ bos_token }}file-base",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("chat_templates/tool_use.jinja"),
            "file-tools",
        )
        .unwrap();
        let config = serde_json::json!({"chat_template":false,"bos_token":"<B>"});
        let loaded = ChatTemplate::from_model_assets(dir.path(), Some(&config))
            .unwrap()
            .unwrap();
        assert_eq!(loaded.render(&[], None, false).unwrap(), "<B>file-base");
        assert_eq!(
            loaded.render(&[], Some("[{}]"), false).unwrap(),
            "file-tools"
        );
        std::fs::remove_file(dir.path().join("chat_template.jinja")).unwrap();
        let loaded = ChatTemplate::from_model_assets(dir.path(), Some(&config))
            .unwrap()
            .unwrap();
        assert_eq!(loaded.source(), "file-tools");
        assert_eq!(loaded.render(&[], None, false).unwrap(), "file-tools");
    }

    #[test]
    fn named_template_configuration_errors_name_the_source_and_repair() {
        let dir = tempfile::tempdir().unwrap();
        for (named, expected) in [
            (serde_json::json!(42), "must be a Jinja string"),
            (
                serde_json::json!({"default":42}),
                "must contain a Jinja string",
            ),
            (serde_json::json!([{ "template":"base"}]), ".name must be"),
            (
                serde_json::json!([{ "name":"default"}]),
                ".template must be",
            ),
            (
                serde_json::json!([{ "name":"default","template":"a"},{"name":"default","template":"b"}]),
                "duplicates template",
            ),
            (serde_json::json!({"":"base"}), "name must not be empty"),
            (serde_json::json!({"custom":"base"}), "available names"),
        ] {
            let error = ChatTemplate::from_model_assets(
                dir.path(),
                Some(&serde_json::json!({"chat_template":named})),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(expected), "{named}: {error}");
            assert!(error.contains(&dir.path().display().to_string()), "{error}");
        }
        let loaded = ChatTemplate::from_model_assets(
            dir.path(),
            Some(&serde_json::json!({"chat_template":{"default":"base","tool_use":""}})),
        )
        .unwrap()
        .unwrap();
        assert_eq!(loaded.tool_source(), None);
        assert_eq!(loaded.render(&[], Some("[{}]"), false).unwrap(), "base");
    }

    #[test]
    fn empty_named_template_assets_remain_absent() {
        let dir = tempfile::tempdir().unwrap();
        for declaration in [
            serde_json::json!(""),
            serde_json::json!([]),
            serde_json::json!({}),
            serde_json::json!({"default":"","tool_use":""}),
        ] {
            assert!(
                ChatTemplate::from_model_assets(
                    dir.path(),
                    Some(&serde_json::json!({"chat_template":declaration}))
                )
                .unwrap()
                .is_none()
            );
        }
        std::fs::write(dir.path().join("chat_template.jinja"), "").unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"bos_token":"<B>","chat_template":42}"#,
        )
        .unwrap();
        let fallback = ChatTemplate::from_model_dir(dir.path()).unwrap();
        assert_eq!(fallback.source(), DEFAULT_CHAT_TEMPLATE);
        assert_eq!(fallback.bos_token.as_deref(), Some("<B>"));
    }

    #[test]
    fn named_template_asset_errors_preserve_the_io_cause() {
        use std::error::Error as _;
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("chat_template.jinja")).unwrap();
        let error = ChatTemplate::from_model_assets(dir.path(), None).unwrap_err();
        assert!(matches!(error, OrtError::Io(_)));
        assert!(error.to_string().contains("chat_template.jinja"));
        assert!(error.source().is_some());
        std::fs::remove_dir(dir.path().join("chat_template.jinja")).unwrap();
        std::fs::write(dir.path().join("chat_template.jinja"), "base").unwrap();
        std::fs::create_dir(dir.path().join("chat_templates")).unwrap();
        std::fs::write(dir.path().join("chat_templates/default.jinja"), "duplicate").unwrap();
        assert!(
            ChatTemplate::from_model_assets(dir.path(), None)
                .unwrap_err()
                .to_string()
                .contains("duplicate chat template name")
        );
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
            tool_template: None,
            bos_token: None,
            eos_token: None,
        };

        assert_eq!(
            template.render(&[], None, false).unwrap(),
            "true|true|true|b| keep "
        );
    }

    // Tool-calling templates name each result from the message's `name`, or
    // resolve it from `tool_call_id` against the assistant call it answers.
    // Both must survive into the render context.
    #[test]
    fn render_exposes_tool_result_identity() {
        let template = ChatTemplate {
            template: "{% for m in messages %}{{ m.get('name') or m.get('tool_call_id') or '?' }}|{% endfor %}"
                .to_string(),
            tool_template: None,
            bos_token: None,
            eos_token: None,
        };
        let messages = [
            ChatMessage::new(ChatRole::Tool, "named")
                .with_tool_result(Some("get_weather".to_string()), None),
            ChatMessage::new(ChatRole::Tool, "by id")
                .with_tool_result(None, Some("call_0".to_string())),
            ChatMessage::user("plain"),
        ];

        assert_eq!(
            template.render(&messages, None, false).unwrap(),
            "get_weather|call_0|?|"
        );
    }

    // Reasoning templates read one of two spellings of the same keyword, and a
    // template that reads neither is unaffected by the caller's choice.
    #[test]
    fn render_exposes_reasoning_effort_under_both_spellings() {
        let template = ChatTemplate {
            template: concat!(
                "{{ reasoning_effort if reasoning_effort is defined and reasoning_effort else 'unset' }}",
                "|{{ reasoning_strength if reasoning_strength is defined and reasoning_strength else 'unset' }}"
            )
            .to_string(),
            tool_template: None,
            bos_token: None,
            eos_token: None,
        };

        assert_eq!(
            template
                .render_with_reasoning_effort(&[], None, false, Some("low"))
                .unwrap(),
            "low|low"
        );
        // An unset effort leaves the template on its own default.
        assert_eq!(template.render(&[], None, false).unwrap(), "unset|unset");
    }

    #[test]
    fn render_supports_hf_conditional_namespace_keyword() {
        let template = ChatTemplate {
            template: concat!(
                "{% set tcid = 'call-1' %}",
                "{% set state = namespace(name=tcid if tcid else '') %}",
                "{{ state.name }}"
            )
            .to_string(),
            tool_template: None,
            bos_token: None,
            eos_token: None,
        };

        assert_eq!(template.render(&[], None, false).unwrap(), "call-1");
        assert_eq!(
            template.source(),
            concat!(
                "{% set tcid = 'call-1' %}",
                "{% set state = namespace(name=tcid if tcid else '') %}",
                "{{ state.name }}"
            ),
            "compatibility normalization must not mutate the model template"
        );
    }

    #[test]
    fn tojson_does_not_html_escape_tool_schemas() {
        // A tool description carrying shell syntax must reach the model as the
        // author wrote it; escaping `<`, `>` or `&` corrupts the definitions the
        // model was trained to recognize.
        let template = ChatTemplate {
            template: "{{ tools | tojson }}".to_string(),
            tool_template: None,
            bos_token: None,
            eos_token: None,
        };
        let tools = r#"[{"description":"run `cd <dir> && ls`"}]"#;

        let rendered = template.render(&[], Some(tools), false).unwrap();

        assert_eq!(rendered, r#"[{"description": "run `cd <dir> && ls`"}]"#);
    }

    #[test]
    fn tojson_honors_the_indent_argument() {
        let template = ChatTemplate {
            template: "{{ tools | tojson(indent=2) }}".to_string(),
            tool_template: None,
            bos_token: None,
            eos_token: None,
        };

        let rendered = template.render(&[], Some(r#"["a&b"]"#), false).unwrap();

        assert_eq!(rendered, "[\n  \"a&b\"\n]");
    }

    #[test]
    fn raise_exception_returns_render_error() {
        let template = ChatTemplate {
            template: "{{ raise_exception('invalid messages') }}".to_string(),
            tool_template: None,
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
            tool_template: None,
            bos_token: None,
            eos_token: None,
        };

        assert_eq!(
            template.render(&[], None, false).unwrap(),
            "before\n    value\nafter"
        );
    }

    #[test]
    fn template_kwargs_and_assistant_reasoning_reach_the_same_renderer() {
        let template = ChatTemplate::from_source(
            "{{ enable_thinking }}|{{ messages[0].reasoning_content }}|{{ add_generation_prompt }}",
        );
        let mut message = ChatMessage::assistant("answer");
        message.reasoning_content = Some("reason".into());
        let kwargs = std::collections::BTreeMap::from([(
            "enable_thinking".to_string(),
            serde_json::json!(false),
        )]);
        assert_eq!(
            template
                .render_with_kwargs(&[message], None, true, None, &kwargs)
                .unwrap(),
            "false|reason|true"
        );
        let kwargs =
            std::collections::BTreeMap::from([("messages".to_string(), serde_json::json!([]))]);
        let error = template
            .render_with_kwargs(&[], None, false, None, &kwargs)
            .unwrap_err()
            .to_string();
        assert!(error.contains("chat_template_kwargs.messages"));
        assert!(error.contains("reserved"));
    }

    fn test_dir(name: &str) -> PathBuf {
        std::env::current_dir().unwrap().join(format!(
            "chat-template-test-{}-{}",
            std::process::id(),
            name
        ))
    }
}
