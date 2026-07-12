//! OpenAI-compatible HTTP server wiring for onnx-genai.

use std::{
    convert::Infallible,
    net::SocketAddr,
    path::Path,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use onnx_genai::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
    GenerateToken, StopSequence,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
pub struct AppState {
    model_id: String,
    engine: SharedEngine,
}

#[derive(Clone)]
struct SharedEngine(Arc<Mutex<Engine>>);

// Phase 2 serializes all generation through a mutex. ORT sessions already declare Send/Sync;
// this wrapper keeps the HTTP layer server-only until the engine exposes its own Send bound.
unsafe impl Send for SharedEngine {}
unsafe impl Sync for SharedEngine {}

impl SharedEngine {
    fn new(engine: Engine) -> Self {
        Self(Arc::new(Mutex::new(engine)))
    }

    fn generate(&self, request: GenerateRequest) -> anyhow::Result<onnx_genai::GenerateResult> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?
            .generate(request)
    }

    fn generate_with_callback(
        &self,
        request: GenerateRequest,
        callback: &mut onnx_genai::GenerateTokenCallback<'_>,
    ) -> anyhow::Result<onnx_genai::GenerateResult> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?
            .generate_with_callback(request, Some(callback))
    }
}

impl AppState {
    pub fn load(model_dir: &Path, model_id: Option<String>) -> anyhow::Result<Self> {
        let engine = Engine::from_dir(model_dir, EngineConfig::default())?;
        let model_id = model_id.unwrap_or_else(|| infer_model_id(model_dir));
        Ok(Self::new(model_id, engine))
    }

    pub fn new(model_id: String, engine: Engine) -> Self {
        Self {
            model_id,
            engine: SharedEngine::new(engine),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

pub async fn serve(addr: SocketAddr, state: AppState) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app(state)).await?;
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopInput>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopInput {
    One(String),
    Many(Vec<String>),
}

impl StopInput {
    fn into_sequences(self) -> Vec<StopSequence> {
        match self {
            Self::One(value) => vec![StopSequence::Text(value)],
            Self::Many(values) => values.into_iter().map(StopSequence::Text).collect(),
        }
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
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    model: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorResponse {
            error: ErrorBody {
                message: self.message,
                kind: "server_error",
            },
        });
        (self.status, body).into_response()
    }
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model: state.model_id,
    })
}

async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: state.model_id,
            object: "model",
            created: now_unix(),
            owned_by: "onnx-genai",
        }],
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    validate_request(&request)?;
    if request.stream {
        Ok(stream_chat_completion(state, request).into_response())
    } else {
        let response = run_chat_completion(state, request).await?;
        Ok(Json(response).into_response())
    }
}

async fn run_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, ApiError> {
    let id = completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let generation_request = build_generate_request(&request);
    let engine = state.engine.clone();

    let result = tokio::task::spawn_blocking(move || engine.generate(generation_request))
        .await
        .map_err(|err| ApiError::internal(format!("generation task failed: {err}")))?
        .map_err(|err| ApiError::internal(format!("generation failed: {err}")))?;

    let completion_tokens = result.token_ids.len();
    Ok(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: result.text,
            },
            finish_reason: finish_reason_label(&result.finish_reason),
        }],
        usage: Some(Usage {
            prompt_tokens: 0,
            completion_tokens,
            total_tokens: completion_tokens,
        }),
    })
}

fn stream_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let id = completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let generation_request = build_generate_request(&request);
    let engine = state.engine.clone();
    let (tx, rx) = mpsc::channel(16);

    tokio::task::spawn_blocking(move || {
        let send_chunk = |chunk: ChatCompletionChunk| -> anyhow::Result<()> {
            tx.blocking_send(Ok(Event::default().data(serde_json::to_string(&chunk)?)))
                .context("stream receiver closed")
        };

        send_chunk(role_chunk(&id, created, &model))?;

        let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
            send_chunk(content_chunk(&id, created, &model, token.text))
        };

        let result = engine.generate_with_callback(generation_request, &mut callback);

        match result {
            Ok(result) => send_chunk(done_chunk(
                &id,
                created,
                &model,
                finish_reason_label(&result.finish_reason),
            ))?,
            Err(err) => tx
                .blocking_send(Ok(Event::default().event("error").data(
                    serde_json::to_string(&ErrorResponse {
                        error: ErrorBody {
                            message: format!("generation failed: {err}"),
                            kind: "server_error",
                        },
                    })?,
                )))
                .context("stream receiver closed")?,
        }

        tx.blocking_send(Ok(Event::default().data("[DONE]")))
            .context("stream receiver closed")?;
        Ok::<(), anyhow::Error>(())
    });

    Sse::new(ReceiverStream::new(rx))
}

fn role_chunk(id: &str, created: u64, model: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                content: None,
            },
            finish_reason: None,
        }],
    }
}

fn content_chunk(id: &str, created: u64, model: &str, content: String) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: Some(content),
            },
            finish_reason: None,
        }],
    }
}

fn done_chunk(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &'static str,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta::default(),
            finish_reason: Some(finish_reason),
        }],
    }
}

fn validate_request(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    if request.max_tokens == 0 {
        return Err(ApiError::bad_request(
            "max_tokens must be greater than zero",
        ));
    }
    if !request.temperature.is_finite() || request.temperature < 0.0 {
        return Err(ApiError::bad_request(
            "temperature must be finite and non-negative",
        ));
    }
    if !request.top_p.is_finite() || request.top_p < 0.0 {
        return Err(ApiError::bad_request(
            "top_p must be finite and non-negative",
        ));
    }
    Ok(())
}

pub fn build_generate_request(request: &ChatCompletionRequest) -> GenerateRequest {
    let mut options = GenerateOptions {
        max_new_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        ..GenerateOptions::default()
    };
    if let Some(stop) = request.stop.clone() {
        options.stop_sequences = stop.into_sequences();
    }

    GenerateRequest {
        prompt: GeneratePrompt::Text(build_prompt(&request.messages)),
        options,
    }
}

/// Build the Phase 2 chat prompt with a simple role-tagged template:
/// `<|role|>\n{content}\n` for every message, followed by `<|assistant|>\n`.
/// Model-specific templates will replace this once tokenizer chat templates are wired.
pub fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|");
        prompt.push_str(message.role.trim());
        prompt.push_str("|>\n");
        prompt.push_str(&message.content);
        prompt.push('\n');
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn finish_reason_label(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::MaxTokens => "length",
        FinishReason::EosToken | FinishReason::StopSequence { .. } => "stop",
    }
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn completion_id() -> String {
    format!("chatcmpl-{}", now_unix())
}

fn infer_model_id(model_dir: &Path) -> String {
    model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("onnx-genai-model")
        .to_string()
}
