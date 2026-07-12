//! OpenAI-compatible HTTP server wiring for onnx-genai.

use std::{
    collections::HashMap,
    convert::Infallible,
    net::SocketAddr,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{delete, get, post},
};
use onnx_genai::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
    GenerateResult, GenerateToken, SessionId, StopSequence,
};
use onnx_genai_engine::GenerateConstraint;
use onnx_genai_ort::{ModelDirectory, Tokenizer};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

const SESSION_ID_HEADER: &str = "x-session-id";

#[derive(Clone)]
pub struct AppState {
    model_id: String,
    engine: SharedEngine,
    tokenizer: Arc<Tokenizer>,
    sessions: SessionRegistry,
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

    fn create_session(&self) -> anyhow::Result<SessionId> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?
            .create_session()
    }

    fn close_session(&self, session_id: SessionId) -> anyhow::Result<()> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?
            .close_session(session_id)
    }

    fn generate_in_session(
        &self,
        session_id: SessionId,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?
            .generate_in_session(session_id, request)
    }

    fn generate_in_session_with_callback(
        &self,
        session_id: SessionId,
        request: GenerateRequest,
        callback: &mut onnx_genai::GenerateTokenCallback<'_>,
    ) -> anyhow::Result<GenerateResult> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?
            .generate_in_session_with_callback(session_id, request, Some(callback))
    }

    fn session_token_count(&self, session_id: SessionId) -> anyhow::Result<usize> {
        self.0
            .lock()
            .map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?
            .session_token_count(session_id)
    }
}

#[derive(Clone)]
struct SessionRegistry {
    sessions: Arc<Mutex<HashMap<String, SessionId>>>,
    next_id: Arc<AtomicU64>,
}

impl SessionRegistry {
    fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    fn insert(&self, client_id: String, engine_session_id: SessionId) -> anyhow::Result<()> {
        self.sessions
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?
            .insert(client_id, engine_session_id);
        Ok(())
    }

    fn get(&self, client_id: &str) -> anyhow::Result<Option<SessionId>> {
        Ok(self
            .sessions
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?
            .get(client_id)
            .copied())
    }

    fn remove(&self, client_id: &str) -> anyhow::Result<Option<SessionId>> {
        Ok(self
            .sessions
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?
            .remove(client_id))
    }

    fn next_client_id(&self) -> String {
        let sequence = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("sess-{}-{sequence}", now_unix())
    }
}

impl AppState {
    pub fn load(model_dir: &Path, model_id: Option<String>) -> anyhow::Result<Self> {
        let model_directory = ModelDirectory::load(model_dir)
            .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {}", e))?;
        let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let engine = Engine::from_dir(model_dir, EngineConfig::default())?;
        let model_id = model_id.unwrap_or_else(|| infer_model_id(model_dir));
        Ok(Self::new(model_id, engine, tokenizer))
    }

    pub fn new(model_id: String, engine: Engine, tokenizer: Tokenizer) -> Self {
        Self {
            model_id,
            engine: SharedEngine::new(engine),
            tokenizer: Arc::new(tokenizer),
            sessions: SessionRegistry::new(),
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
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/{id}", delete(delete_session))
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
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

impl ChatCompletionRequest {
    fn wants_json_object(&self) -> bool {
        matches!(
            self.response_format.as_ref().map(|format| &format.kind),
            Some(ResponseFormatType::JsonObject)
        )
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: ResponseFormatType,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatType {
    Text,
    JsonObject,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopInput {
    One(String),
    Many(Vec<String>),
}

impl StopInput {
    fn into_texts(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }

    fn into_sequences(self) -> Vec<StopSequence> {
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
struct SessionResponse {
    id: String,
    object: &'static str,
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

struct PreparedGenerateRequest {
    request: GenerateRequest,
    prompt_tokens: usize,
}

#[derive(Debug)]
struct StopBoundaryBuffer {
    stop_sequences: Vec<String>,
    pending: String,
}

impl StopBoundaryBuffer {
    fn new(stop_sequences: Vec<String>) -> Self {
        Self {
            stop_sequences: stop_sequences
                .into_iter()
                .filter(|sequence| !sequence.is_empty())
                .collect(),
            pending: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> String {
        if self.stop_sequences.is_empty() {
            return text.to_string();
        }

        self.pending.push_str(text);
        if let Some(stop_start) = self.earliest_stop_start() {
            let safe = self.pending[..stop_start].to_string();
            self.pending.clear();
            return safe;
        }

        let keep = self.longest_stop_prefix_suffix_len();
        let emit_len = self.pending.len().saturating_sub(keep);
        if emit_len == 0 {
            return String::new();
        }

        let safe = self.pending[..emit_len].to_string();
        self.pending = self.pending[emit_len..].to_string();
        safe
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn earliest_stop_start(&self) -> Option<usize> {
        self.stop_sequences
            .iter()
            .filter_map(|sequence| self.pending.find(sequence))
            .min()
    }

    fn longest_stop_prefix_suffix_len(&self) -> usize {
        let mut keep = 0;
        for sequence in &self.stop_sequences {
            for (prefix_len, _) in sequence.char_indices().skip(1) {
                if self.pending.ends_with(&sequence[..prefix_len]) {
                    keep = keep.max(prefix_len);
                }
            }
        }
        keep
    }
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

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
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

async fn create_session(State(state): State<AppState>) -> Result<Json<SessionResponse>, ApiError> {
    let client_id = state.sessions.next_client_id();
    let engine = state.engine.clone();
    let engine_session_id = tokio::task::spawn_blocking(move || engine.create_session())
        .await
        .map_err(|err| ApiError::internal(format!("session create task failed: {err}")))?
        .map_err(|err| ApiError::internal(format!("session create failed: {err}")))?;

    state
        .sessions
        .insert(client_id.clone(), engine_session_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;

    Ok(Json(SessionResponse {
        id: client_id,
        object: "session",
    }))
}

async fn delete_session(
    State(state): State<AppState>,
    AxumPath(client_id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let engine_session_id = state
        .sessions
        .remove(&client_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?
        .ok_or_else(|| ApiError::not_found(format!("session {client_id} not found")))?;

    let engine = state.engine.clone();
    tokio::task::spawn_blocking(move || engine.close_session(engine_session_id))
        .await
        .map_err(|err| ApiError::internal(format!("session close task failed: {err}")))?
        .map_err(|err| ApiError::internal(format!("session close failed: {err}")))?;

    Ok(StatusCode::NO_CONTENT)
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    validate_request(&request)?;
    let session_id = session_id_from_headers(&headers)?;
    if request.stream {
        Ok(stream_chat_completion(state, request, session_id)?.into_response())
    } else {
        let response = run_chat_completion(state, request, session_id).await?;
        Ok(Json(response).into_response())
    }
}

async fn run_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
    client_session_id: Option<String>,
) -> Result<ChatCompletionResponse, ApiError> {
    let id = completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let prepared =
        prepare_generate_request(&request, &state.tokenizer, client_session_id.is_some())
            .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
    let prompt_tokens = prepared.prompt_tokens;
    let generation_request = prepared.request;
    let engine = state.engine.clone();
    let session_lookup = client_session_id
        .as_deref()
        .map(|id| get_or_create_session(&state, id))
        .transpose()?;

    let session_for_count = session_lookup;
    let wants_json_object = request.wants_json_object();
    let result = tokio::task::spawn_blocking(move || match session_lookup {
        Some(engine_session_id) => {
            engine.generate_in_session(engine_session_id, generation_request)
        }
        None => engine.generate(generation_request),
    })
    .await
    .map_err(|err| ApiError::internal(format!("generation task failed: {err}")))?;

    let session_token_count = session_for_count
        .map(|engine_session_id| {
            state
                .engine
                .session_token_count(engine_session_id)
                .map_err(|err| ApiError::internal(format!("session token count failed: {err}")))
        })
        .transpose()?;

    let (content, completion_tokens, finish_reason) = match result {
        Ok(result) => (
            result.text,
            result.token_ids.len(),
            finish_reason_label(&result.finish_reason),
        ),
        Err(err) if wants_json_object && json_constraint_stopped_incomplete(&err) => {
            ("{}".to_string(), 0, "stop")
        }
        Err(err) => return Err(ApiError::internal(format!("generation failed: {err}"))),
    };
    let total_tokens = prompt_tokens + completion_tokens;
    Ok(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason,
        }],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        }),
        session_id: client_session_id,
        session_token_count,
    })
}

fn stream_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
    client_session_id: Option<String>,
) -> Result<Sse<ReceiverStream<Result<Event, Infallible>>>, ApiError> {
    let id = completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let stop_sequences = request
        .stop
        .clone()
        .map(StopInput::into_texts)
        .unwrap_or_default();
    let prepared =
        prepare_generate_request(&request, &state.tokenizer, client_session_id.is_some())
            .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
    let wants_json_object = request.wants_json_object();
    let generation_request = prepared.request;
    let engine = state.engine.clone();
    let (tx, rx) = mpsc::channel(16);
    let session_lookup = client_session_id
        .as_deref()
        .map(|id| get_or_create_session(&state, id));

    tokio::task::spawn_blocking(move || {
        let session_lookup = match session_lookup.transpose() {
            Ok(value) => value,
            Err(err) => {
                tx.blocking_send(Ok(Event::default().event("error").data(
                    serde_json::to_string(&ErrorResponse {
                        error: ErrorBody {
                            message: format!("session setup failed: {}", err.message),
                            kind: "server_error",
                        },
                    })?,
                )))
                .context("stream receiver closed")?;
                tx.blocking_send(Ok(Event::default().data("[DONE]")))
                    .context("stream receiver closed")?;
                return Ok::<(), anyhow::Error>(());
            }
        };

        let send_chunk = |chunk: ChatCompletionChunk| -> anyhow::Result<()> {
            tx.blocking_send(Ok(Event::default().data(serde_json::to_string(&chunk)?)))
                .context("stream receiver closed")
        };

        send_chunk(role_chunk(&id, created, &model))?;

        let mut stop_buffer = StopBoundaryBuffer::new(stop_sequences);
        let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
            let finish_reason = token.finish_reason.clone();
            let content = stop_buffer.push(&token.text);
            if !wants_json_object && !content.is_empty() {
                send_chunk(content_chunk(&id, created, &model, content))?;
            }
            if matches!(finish_reason, Some(FinishReason::StopSequence { .. })) {
                stop_buffer.pending.clear();
            }
            Ok(())
        };

        let result = match session_lookup {
            Some(engine_session_id) => engine.generate_in_session_with_callback(
                engine_session_id,
                generation_request,
                &mut callback,
            ),
            None => engine.generate_with_callback(generation_request, &mut callback),
        };

        match result {
            Ok(result) => {
                if wants_json_object {
                    if !result.text.is_empty() {
                        send_chunk(content_chunk(&id, created, &model, result.text))?;
                    }
                } else if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                    let content = stop_buffer.flush();
                    if !content.is_empty() {
                        send_chunk(content_chunk(&id, created, &model, content))?;
                    }
                }
                send_chunk(done_chunk(
                    &id,
                    created,
                    &model,
                    finish_reason_label(&result.finish_reason),
                ))?;
            }
            Err(err) if wants_json_object && json_constraint_stopped_incomplete(&err) => {
                send_chunk(content_chunk(&id, created, &model, "{}".to_string()))?;
                send_chunk(done_chunk(&id, created, &model, "stop"))?;
            }
            Err(err) => {
                tx.blocking_send(Ok(Event::default().event("error").data(
                    serde_json::to_string(&ErrorResponse {
                        error: ErrorBody {
                            message: format!("generation failed: {err}"),
                            kind: "server_error",
                        },
                    })?,
                )))
                .context("stream receiver closed")?;
            }
        }

        tx.blocking_send(Ok(Event::default().data("[DONE]")))
            .context("stream receiver closed")?;
        Ok::<(), anyhow::Error>(())
    });

    Ok(Sse::new(ReceiverStream::new(rx)))
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

fn session_id_from_headers(headers: &HeaderMap) -> Result<Option<String>, ApiError> {
    let Some(value) = headers.get(SESSION_ID_HEADER) else {
        return Ok(None);
    };
    let session_id = value
        .to_str()
        .map_err(|_| ApiError::bad_request("X-Session-Id must be valid UTF-8"))?
        .trim();
    if session_id.is_empty() {
        return Err(ApiError::bad_request("X-Session-Id must not be empty"));
    }
    Ok(Some(session_id.to_string()))
}

fn get_or_create_session(state: &AppState, client_id: &str) -> Result<SessionId, ApiError> {
    if let Some(engine_session_id) = state
        .sessions
        .get(client_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?
    {
        return Ok(engine_session_id);
    }

    let engine_session_id = state
        .engine
        .create_session()
        .map_err(|err| ApiError::internal(format!("session create failed: {err}")))?;
    state
        .sessions
        .insert(client_id.to_string(), engine_session_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    Ok(engine_session_id)
}

pub fn build_generate_request(request: &ChatCompletionRequest) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::Text(build_prompt(&request.messages)),
        options: build_generate_options(request),
    }
}

fn prepare_generate_request(
    request: &ChatCompletionRequest,
    tokenizer: &Tokenizer,
    session: bool,
) -> anyhow::Result<PreparedGenerateRequest> {
    let prompt = if session {
        build_session_prompt(&request.messages)
    } else {
        build_prompt(&request.messages)
    };
    let token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?;
    let prompt_tokens = token_ids.len();
    Ok(PreparedGenerateRequest {
        request: GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options: build_generate_options(request),
        },
        prompt_tokens,
    })
}

fn build_generate_options(request: &ChatCompletionRequest) -> GenerateOptions {
    let mut options = GenerateOptions {
        max_new_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        ..GenerateOptions::default()
    };
    if let Some(stop) = request.stop.clone() {
        options.stop_sequences = stop.into_sequences();
    }
    if request.wants_json_object() {
        options.constraint = Some(GenerateConstraint::Json);
    }
    options
}

fn json_constraint_stopped_incomplete(err: &anyhow::Error) -> bool {
    err.to_string()
        .contains("JSON constrained decoding stopped before a complete JSON value")
}

fn build_session_prompt(messages: &[ChatMessage]) -> String {
    messages
        .last()
        .map(|message| message.content.clone())
        .unwrap_or_default()
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
        FinishReason::MaxTokens | FinishReason::Length => "length",
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

#[cfg(test)]
mod tests {
    use super::StopBoundaryBuffer;

    #[test]
    fn stop_boundary_buffer_holds_partial_stop_sequence() {
        let mut buffer = StopBoundaryBuffer::new(vec!["tok20".to_string()]);
        assert_eq!(buffer.push("to"), "");
        assert_eq!(buffer.push("k"), "");
        assert_eq!(buffer.push("2"), "");
        assert_eq!(buffer.push("1"), "tok21");
        assert_eq!(buffer.flush(), "");
    }

    #[test]
    fn stop_boundary_buffer_suppresses_matched_stop_sequence() {
        let mut buffer = StopBoundaryBuffer::new(vec!["tok20".to_string()]);
        assert_eq!(buffer.push("hello tok"), "hello ");
        assert_eq!(buffer.push("20"), "");
        assert_eq!(buffer.flush(), "");
    }
}
