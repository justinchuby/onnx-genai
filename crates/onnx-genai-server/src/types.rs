use std::collections::BTreeMap;

use onnx_genai::StopSequence;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioSpeechRequest {
    pub model: String,
    pub input: String,
    /// Bypass declared text assembly when the caller supplies the complete raw prompt.
    #[serde(default)]
    pub raw_prompt: bool,
    #[serde(default)]
    pub instructions: String,
    #[serde(default = "default_wav_response_format")]
    pub response_format: String,
    #[serde(default)]
    pub stream: bool,
    /// Optional bounded workflow-unit budget for short non-streaming renders.
    #[serde(default)]
    pub max_output_units: Option<usize>,
}

fn default_wav_response_format() -> String {
    "wav".to_string()
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: EmbeddingEncodingFormat,
    #[serde(default)]
    pub dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    String(String),
    Strings(Vec<String>),
    Tokens(Vec<u32>),
    TokenArrays(Vec<Vec<u32>>),
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingEncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: EmbeddingVector,
    pub index: usize,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

impl EmbeddingVector {
    pub fn from_floats(values: Vec<f32>, format: EmbeddingEncodingFormat) -> Self {
        match format {
            EmbeddingEncodingFormat::Float => Self::Float(values),
            EmbeddingEncodingFormat::Base64 => {
                use base64::{Engine as _, engine::general_purpose::STANDARD};

                let mut bytes = Vec::with_capacity(values.len() * size_of::<f32>());
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                Self::Base64(STANDARD.encode(bytes))
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Deprecated by OpenAI in favour of `max_completion_tokens`, and absent
    /// from requests aimed at reasoning models. See [`Self::output_budget`].
    #[serde(default, deserialize_with = "deserialize_token_budget")]
    pub max_tokens: Option<usize>,
    /// The budget covering reasoning *and* answer tokens, which is the only
    /// output limit OpenAI accepts for a reasoning model.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// Absent when the caller left sampling to the model. See
    /// [`Self::sampling_overrides`].
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub dynatemp_range: f32,
    #[serde(default = "default_typical_p")]
    pub dynatemp_exponent: f32,
    /// Absent when the caller left sampling to the model. See
    /// [`Self::sampling_overrides`].
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default)]
    pub top_a: f32,
    #[serde(default = "default_typical_p", rename = "typ_p")]
    pub typical_p: f32,
    #[serde(default = "default_repetition_penalty", rename = "repeat_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default, deserialize_with = "deserialize_seed")]
    pub seed: Option<u64>,
    #[serde(default)]
    pub dry_multiplier: f32,
    #[serde(default = "default_dry_base")]
    pub dry_base: f32,
    #[serde(default = "default_dry_allowed_length")]
    pub dry_allowed_length: usize,
    #[serde(default)]
    pub dry_sequence_breakers: Vec<String>,
    /// `0` disables Mirostat; `1` and `2` select the corresponding algorithm.
    #[serde(default)]
    pub mirostat: u8,
    #[serde(default = "default_mirostat_tau")]
    pub mirostat_tau: f32,
    #[serde(default = "default_mirostat_eta")]
    pub mirostat_eta: f32,
    #[serde(default)]
    pub xtc_probability: f32,
    #[serde(default = "default_xtc_threshold")]
    pub xtc_threshold: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopInput>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub logprobs: bool,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    /// How much a reasoning model should think before answering; `None` leaves
    /// the model's chat template on its own default.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub chat_template_kwargs: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub stream_options: StreamOptions,
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    #[serde(default)]
    pub continue_final_message: bool,
    /// Retain extensions so unsupported active controls cannot silently disappear.
    #[serde(flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

fn default_true() -> bool {
    true
}

fn deserialize_token_budget<'de, D: Deserializer<'de>>(d: D) -> Result<Option<usize>, D::Error> {
    let value = Option::<i64>::deserialize(d)?;
    match value {
        None | Some(-1) => Ok(None),
        Some(value) => usize::try_from(value).map(Some).map_err(|_| {
            D::Error::custom("max_tokens must be -1 (server limit) or a non-negative integer")
        }),
    }
}

fn deserialize_seed<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    let value = Option::<serde_json::Number>::deserialize(d)?;
    match value {
        None => Ok(None),
        Some(value) if value.as_i64() == Some(-1) => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            D::Error::custom("seed must be -1 (random) or an unsigned 64-bit integer")
        }),
    }
}

/// One completion input after HTTP batch classification, with shared controls.
#[derive(Debug)]
pub struct CompletionRequest {
    pub prompt: crate::MixedPrompt,
    pub suffix: Option<String>,
    pub logprobs: Option<usize>,
    pub sampling: std::sync::Arc<ChatCompletionRequest>,
}
impl<'de> Deserialize<'de> for CompletionRequest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut body = serde_json::Map::<String, serde_json::Value>::deserialize(deserializer)?;
        let prompt = body
            .remove("prompt")
            .ok_or_else(|| D::Error::custom("prompt is required for completions"))?;
        let prompt = crate::MixedPrompt::parse_completion(prompt).map_err(D::Error::custom)?;
        let suffix = body
            .remove("suffix")
            .map(serde_json::from_value::<Option<String>>)
            .transpose()
            .map_err(D::Error::custom)?
            .flatten();
        let logprobs = body
            .remove("logprobs")
            .map(serde_json::from_value::<Option<usize>>)
            .transpose()
            .map_err(|error| {
                D::Error::custom(format!(
                    "logprobs must be a non-negative integer or null: {error}"
                ))
            })?
            .flatten();
        for field in [
            "messages",
            "top_logprobs",
            "chat_template_kwargs",
            "add_generation_prompt",
        ] {
            if body.contains_key(field) {
                return Err(D::Error::custom(format!(
                    "{field} is not a raw text completion field; use /v1/chat/completions for chat rendering and use integer logprobs for legacy token probabilities"
                )));
            }
        }
        if let Some(echo) = body.remove("echo")
            && echo != false
        {
            return Err(D::Error::custom(
                "echo=true is unsupported; omit echo or send false to return only generated text",
            ));
        }
        body.insert(
            "messages".into(),
            serde_json::json!([{"role":"user","content":""}]),
        );
        body.insert("logprobs".into(), serde_json::json!(logprobs.is_some()));
        if let Some(count) = logprobs {
            body.insert("top_logprobs".into(), serde_json::json!(count));
        }
        let sampling: ChatCompletionRequest =
            serde_path_to_error::deserialize(serde_json::Value::Object(body))
                .map_err(D::Error::custom)?;
        if sampling.has_tool_context() || sampling.reasoning_effort.is_some() {
            return Err(D::Error::custom(
                "tools and reasoning_effort require /v1/chat/completions; raw completions do not render a chat template",
            ));
        }
        Ok(Self {
            prompt,
            suffix,
            logprobs,
            sampling: std::sync::Arc::new(sampling),
        })
    }
}

/// How much a reasoning model should think before answering, in OpenAI's
/// `reasoning_effort` vocabulary.
///
/// A closed set rather than a free string so an unsupported spelling is
/// rejected with the accepted values instead of silently leaving the model on
/// its own default, which for a reasoning model is often maximal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

impl ChatCompletionRequest {
    /// How many tokens this turn may decode, given the server's cap.
    ///
    /// `max_completion_tokens` supersedes `max_tokens`, which OpenAI deprecated
    /// for chat completions and rejects outright for reasoning models. A client
    /// that sends neither has asked for no limit of its own — OpenAI decodes
    /// until the model stops or the context runs out — so the budget falls back
    /// to the server's cap rather than to a small fixed default. That
    /// distinction matters most for a reasoning model, whose private thinking
    /// spends the same budget as its answer: too small a fallback truncates the
    /// turn mid-thought and returns nothing at all.
    pub(crate) fn output_budget(&self, cap: usize) -> usize {
        self.requested_output_budget()
            .map_or(cap, |(_, requested)| requested)
            .min(cap)
    }

    /// The output budget the client asked for and the field it used, or `None`
    /// when it named neither and left the limit to the server.
    pub(crate) fn requested_output_budget(&self) -> Option<(&'static str, usize)> {
        self.max_completion_tokens
            .map(|budget| ("max_completion_tokens", budget))
            .or_else(|| self.max_tokens.map(|budget| ("max_tokens", budget)))
    }

    pub(crate) fn wants_constrained_json(&self) -> bool {
        matches!(
            self.response_format.as_ref(),
            Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
        )
    }

    pub(crate) fn has_tool_context(&self) -> bool {
        self.tools.as_ref().is_some_and(|tools| !tools.is_empty())
            || self.tool_choice.is_some()
            || self.messages.iter().any(|message| {
                message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
                    || message.tool_call_id.is_some()
                    || message.role == "tool"
            })
    }

    pub(crate) fn image_urls(&self) -> Vec<String> {
        self.messages
            .iter()
            .filter_map(|message| message.content.as_ref())
            .flat_map(ChatMessageContent::image_urls)
            .map(ToString::to_string)
            .collect()
    }

    pub(crate) fn input_audio(&self) -> Vec<InputAudio> {
        self.messages
            .iter()
            .filter_map(|message| message.content.as_ref())
            .flat_map(ChatMessageContent::input_audio)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<ChatMessageContent>,
    /// The model's private thinking, carried beside the answer rather than
    /// inside it so a client can show its progress without the two being
    /// mistaken for one another.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatMessageToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// OpenAI's optional message `name`; on a `tool` message it names the
    /// function that produced the result, which tool templates render.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<ChatMessageContentPart>),
}

/// Content part `type` values this server understands, quoted in every
/// rejection so a client learns the accepted set from the error alone.
const SUPPORTED_CONTENT_PART_TYPES: &str = "\"text\", \"image_url\", \"input_audio\"";

/// OpenAI's chat `content` is either a plain string or an array of typed parts.
///
/// This is deserialized by hand rather than with `#[serde(untagged)]` because
/// an untagged enum collapses every inner failure into "data did not match any
/// variant", discarding which part was wrong and why. Multimodal requests carry
/// the most structure and therefore fail in the most ways, so each one reports
/// its own index, its `type`, and the accepted shape.
impl<'de> Deserialize<'de> for ChatMessageContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::String(text) => Ok(Self::Text(text)),
            serde_json::Value::Array(items) => {
                let mut parts = Vec::with_capacity(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    parts.push(content_part(index, item).map_err(D::Error::custom)?);
                }
                Ok(Self::Parts(parts))
            }
            other => Err(D::Error::custom(format!(
                "What: a chat message's `content` was rejected. \
                 Why: it is {}, but content must be a string or an array of typed parts. \
                 How: send \"content\": \"...\" for plain text, or an array of parts with `type` in {SUPPORTED_CONTENT_PART_TYPES}.",
                json_kind(&other)
            ))),
        }
    }
}

/// Human name for a JSON value's kind, for use in rejection messages.
fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Parse one content part, naming the offending index and `type` on failure.
fn content_part(index: usize, value: serde_json::Value) -> Result<ChatMessageContentPart, String> {
    let object = value.as_object().ok_or_else(|| {
        format!(
            "What: chat message content part {index} was rejected. \
             Why: it is {}, but every content part must be an object carrying a `type` field. \
             How: send {{\"type\": \"text\", \"text\": \"...\"}} or another part whose `type` is in {SUPPORTED_CONTENT_PART_TYPES}.",
            json_kind(&value)
        )
    })?;
    let part_type = object
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            format!(
                "What: chat message content part {index} was rejected. \
                 Why: it has no string `type` field, so the server cannot tell text from image or audio. \
                 How: add a `type` of {SUPPORTED_CONTENT_PART_TYPES}."
            )
        })?
        .to_string();
    let shape = match part_type.as_str() {
        "text" => "{\"type\": \"text\", \"text\": \"...\"}",
        "image_url" => {
            "{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,...\" | \"https://...\"}}"
        }
        "input_audio" => {
            "{\"type\": \"input_audio\", \"input_audio\": {\"data\": \"<base64>\", \"format\": \"wav\"}}"
        }
        other => {
            return Err(format!(
                "What: chat message content part {index} was rejected. \
                 Why: its `type` is \"{other}\", which this server does not implement. \
                 How: use a supported part type: {SUPPORTED_CONTENT_PART_TYPES}."
            ));
        }
    };
    serde_json::from_value(value).map_err(|error| {
        format!(
            "What: chat message content part {index} (type \"{part_type}\") was rejected. \
             Why: {error}. \
             How: send it as {shape}."
        )
    })
}

impl ChatMessageContent {
    pub(crate) fn text(&self) -> String {
        self.render(None)
    }

    /// Render the parts as prompt text, writing `image_placeholder` wherever an
    /// image sits.
    ///
    /// OpenAI content parts are ordered, and that order carries meaning: in
    /// "compare [A] with [B]" the images belong where they are written.
    /// Dropping them and re-attaching the placeholders elsewhere would silently
    /// re-associate the text with the wrong picture, so each image is rendered
    /// in place. Without a placeholder (a model with no image contract) the
    /// parts are dropped as before — the request is rejected shortly after.
    pub(crate) fn render(&self, image_placeholder: Option<&str>) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ChatMessageContentPart::Text { text } => Some(text.as_str()),
                    ChatMessageContentPart::ImageUrl { .. } => image_placeholder,
                    ChatMessageContentPart::InputAudio { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    pub(crate) fn image_urls(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::Text(_) => [].as_slice(),
            Self::Parts(parts) => parts.as_slice(),
        }
        .iter()
        .filter_map(|part| match part {
            ChatMessageContentPart::ImageUrl { image_url } => Some(image_url.url.as_str()),
            ChatMessageContentPart::Text { .. } | ChatMessageContentPart::InputAudio { .. } => None,
        })
    }

    pub(crate) fn input_audio(&self) -> impl Iterator<Item = &InputAudio> {
        match self {
            Self::Text(_) => [].as_slice(),
            Self::Parts(parts) => parts.as_slice(),
        }
        .iter()
        .filter_map(|part| match part {
            ChatMessageContentPart::InputAudio { input_audio } => Some(input_audio),
            ChatMessageContentPart::Text { .. } | ChatMessageContentPart::ImageUrl { .. } => None,
        })
    }
}

impl From<String> for ChatMessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChatMessageContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
    /// OpenAI's fidelity hint (`auto`, `low`, `high`). Accepted for client
    /// compatibility but not acted on: how an image is resized and tiled is
    /// declared by the model package's `preprocessing.image` program, never
    /// chosen by the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Serialize)]
pub struct AudioTranscriptionResponse {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChatMessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatMessageToolCallFunction,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChatMessageToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatToolFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Specific(ToolChoiceSpecific),
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceSpecific {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolChoiceFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaSpec },
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonSchemaSpec {
    pub name: String,
    pub schema: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopInput {
    One(String),
    Many(Vec<String>),
}

impl StopInput {
    pub(crate) fn into_texts(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }

    pub(crate) fn into_sequences(self) -> Vec<StopSequence> {
        self.into_texts()
            .into_iter()
            .map(StopSequence::Text)
            .collect()
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_token_count: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub workflow_outputs: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub logprobs: Option<ChatLogprobs>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<ChatTokenLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatTokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<ChatTopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub workflow_outputs: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: &'static str,
    pub logprobs: Option<CompletionLogprobs>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<BTreeMap<String, f32>>,
    pub text_offset: Vec<usize>,
}

fn default_typical_p() -> f32 {
    1.0
}
fn default_repetition_penalty() -> f32 {
    1.0
}
fn default_dry_base() -> f32 {
    1.75
}
fn default_dry_allowed_length() -> usize {
    2
}
fn default_mirostat_tau() -> f32 {
    5.0
}
fn default_mirostat_eta() -> f32 {
    0.1
}
fn default_xtc_threshold() -> f32 {
    0.1
}
