use std::collections::BTreeMap;

use onnx_genai::StopSequence;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

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
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Deprecated by OpenAI in favour of `max_completion_tokens`, and absent
    /// from requests aimed at reasoning models. See [`Self::output_budget`].
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// The budget covering reasoning *and* answer tokens, which is the only
    /// output limit OpenAI accepts for a reasoning model.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// Absent when the caller left sampling to the model. See
    /// [`Self::sampling_overrides`].
    #[serde(default)]
    pub temperature: Option<f32>,
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
    #[serde(default = "default_typical_p")]
    pub typical_p: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub dry_multiplier: f32,
    #[serde(default = "default_dry_base")]
    pub dry_base: f32,
    #[serde(default = "default_dry_allowed_length")]
    pub dry_allowed_length: usize,
    #[serde(default)]
    pub dry_sequence_breakers: Vec<u32>,
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
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Absent when the caller left sampling to the model.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Absent when the caller left sampling to the model.
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopInput>,
    #[serde(default)]
    pub logprobs: Option<usize>,
}

/// How much a reasoning model should think before answering, in OpenAI's
/// `reasoning_effort` vocabulary.
///
/// A closed set rather than a free string so an unsupported spelling is
/// rejected with the accepted values instead of silently leaving the model on
/// its own default, which for a reasoning model is often maximal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
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

/// `POST /v1/images/generations` request.
///
/// Mirrors OpenAI's images API for the fields a local diffusion package can
/// honor. Sampling knobs OpenAI does not expose (`negative_prompt`, `steps`,
/// `guidance_scale`, `seed`) are documented extensions; when omitted, the
/// package's own declared `num_steps` / `guidance_scale` are used.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    /// Number of images to render. Defaults to 1.
    #[serde(default)]
    pub n: Option<usize>,
    /// `"<width>x<height>"` in pixels; both must be multiples of 8.
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub response_format: Option<ImageResponseFormat>,
    /// End-user identifier. Accepted and ignored; this server has no per-user state.
    #[serde(default)]
    pub user: Option<String>,

    // ── onnx-genai extensions ──
    /// Classifier-free-guidance unconditional prompt.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub guidance_scale: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageResponseFormat {
    #[default]
    B64Json,
    Url,
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    /// Base64-encoded PNG bytes.
    pub b64_json: String,
}

/// `POST /v1/audio/speech` request.
///
/// Mirrors OpenAI's speech API for the fields a local TTS package can honor.
/// `voice` is accepted for client compatibility: which voice a package speaks
/// with is a property of the exported model, not a request parameter.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub response_format: Option<SpeechResponseFormat>,
    /// Playback speed. Accepted only as 1.0; this server does not resample.
    #[serde(default)]
    pub speed: Option<f32>,

    // ── onnx-genai extensions ──
    /// Maximum audio tokens to decode. Defaults to the package's `max_tokens`.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Override for a package whose metadata omits `pipeline.audio.sample_rate`.
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SpeechResponseFormat {
    #[default]
    Wav,
    /// Raw little-endian 16-bit PCM, without a container.
    Pcm,
    Mp3,
    Opus,
    Aac,
    Flac,
}

impl SpeechResponseFormat {
    /// The `Content-Type` for a supported format, or `None` when this server
    /// cannot encode it.
    pub fn content_type(self) -> Option<&'static str> {
        match self {
            Self::Wav => Some("audio/wav"),
            Self::Pcm => Some("audio/L16"),
            Self::Mp3 | Self::Opus | Self::Aac | Self::Flac => None,
        }
    }

    /// Lowercase name, for error messages.
    pub fn label(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Pcm => "pcm",
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AudioTranscriptionResponse {
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatMessageToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: &'static str,
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

fn default_max_tokens() -> usize {
    256
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
