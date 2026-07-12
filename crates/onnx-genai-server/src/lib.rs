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
use onnx_genai_ort::{ChatMessage as TemplateChatMessage, ChatTemplate, ModelDirectory, Tokenizer};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

const SESSION_ID_HEADER: &str = "x-session-id";

#[derive(Clone)]
pub struct AppState {
    model_id: String,
    engine: SharedEngine,
    tokenizer: Arc<Tokenizer>,
    chat_template: Option<Arc<ChatTemplate>>,
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
        let chat_template = load_chat_template(model_dir)?;
        let engine = Engine::from_dir(model_dir, EngineConfig::default())?;
        let model_id = model_id.unwrap_or_else(|| infer_model_id(model_dir));
        Ok(Self::new_with_template(
            model_id,
            engine,
            tokenizer,
            chat_template,
        ))
    }

    pub fn new(model_id: String, engine: Engine, tokenizer: Tokenizer) -> Self {
        Self::new_with_template(model_id, engine, tokenizer, None)
    }

    pub fn new_with_template(
        model_id: String,
        engine: Engine,
        tokenizer: Tokenizer,
        chat_template: Option<ChatTemplate>,
    ) -> Self {
        Self {
            model_id,
            engine: SharedEngine::new(engine),
            tokenizer: Arc::new(tokenizer),
            chat_template: chat_template.map(Arc::new),
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
    #[serde(default)]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
}

impl ChatCompletionRequest {
    fn wants_json_object(&self) -> bool {
        matches!(
            self.response_format.as_ref().map(|format| &format.kind),
            Some(ResponseFormatType::JsonObject)
        )
    }

    fn has_tool_context(&self) -> bool {
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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatMessageToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
struct ChunkToolCall {
    index: usize,
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: ChatMessageToolCallFunction,
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
    let prepared = prepare_generate_request(
        &request,
        &state.tokenizer,
        state.chat_template.as_deref(),
        client_session_id.is_some(),
    )
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

    let (content, tool_calls, completion_tokens, finish_reason) = match result {
        Ok(result) => {
            let parsed =
                parse_assistant_output(result.text, finish_reason_label(&result.finish_reason));
            (
                parsed.content,
                parsed.tool_calls,
                result.token_ids.len(),
                parsed.finish_reason,
            )
        }
        Err(err) if wants_json_object && json_constraint_stopped_incomplete(&err) => {
            (Some("{}".to_string()), None, 0, "stop")
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
                tool_calls,
                tool_call_id: None,
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
    let user_stop_sequences = request
        .stop
        .clone()
        .map(StopInput::into_texts)
        .unwrap_or_default();
    let prepared = prepare_generate_request(
        &request,
        &state.tokenizer,
        state.chat_template.as_deref(),
        client_session_id.is_some(),
    )
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

        let mut stop_buffer = StopBoundaryBuffer::new(user_stop_sequences);
        let mut buffered_text = String::new();
        let buffer_for_tool_detection = request.has_tool_context()
            && !matches!(
                request.tool_choice,
                Some(ToolChoice::Mode(ToolChoiceMode::None))
            );
        let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
            let finish_reason = token.finish_reason.clone();
            let content = stop_buffer.push(&token.text);
            if buffer_for_tool_detection {
                buffered_text.push_str(&content);
            } else if !wants_json_object && !content.is_empty() {
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
                if buffer_for_tool_detection {
                    if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                        buffered_text.push_str(&stop_buffer.flush());
                    }
                    let tool_calls = parse_tool_calls(&buffered_text);
                    if tool_calls.is_empty() {
                        if !buffered_text.is_empty() {
                            send_chunk(content_chunk(&id, created, &model, buffered_text))?;
                        }
                        send_chunk(done_chunk(
                            &id,
                            created,
                            &model,
                            finish_reason_label(&result.finish_reason),
                        ))?;
                    } else {
                        send_chunk(tool_calls_chunk(&id, created, &model, tool_calls))?;
                        send_chunk(done_chunk(&id, created, &model, "tool_calls"))?;
                    }
                } else if wants_json_object {
                    if !result.text.is_empty() {
                        send_chunk(content_chunk(&id, created, &model, result.text))?;
                    }
                    send_chunk(done_chunk(
                        &id,
                        created,
                        &model,
                        finish_reason_label(&result.finish_reason),
                    ))?;
                } else {
                    if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
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
                tool_calls: None,
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
                tool_calls: None,
            },
            finish_reason: None,
        }],
    }
}

fn tool_calls_chunk(
    id: &str,
    created: u64,
    model: &str,
    tool_calls: Vec<ChatMessageToolCall>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
                tool_calls: Some(
                    tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(index, call)| ChunkToolCall {
                            index,
                            id: call.id,
                            kind: call.kind,
                            function: call.function,
                        })
                        .collect(),
                ),
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
        prompt: GeneratePrompt::Text(build_prompt(request)),
        options: build_generate_options(request),
    }
}

fn prepare_generate_request(
    request: &ChatCompletionRequest,
    tokenizer: &Tokenizer,
    chat_template: Option<&ChatTemplate>,
    session: bool,
) -> anyhow::Result<PreparedGenerateRequest> {
    let prompt = if session && !request.has_tool_context() {
        build_session_prompt(&request.messages)
    } else {
        render_prompt(request, chat_template)?
    };
    let token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?;
    let prompt_tokens = token_ids.len();
    Ok(PreparedGenerateRequest {
        request: GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options: build_generate_options_with_tokenizer(request, tokenizer),
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

fn build_generate_options_with_tokenizer(
    request: &ChatCompletionRequest,
    tokenizer: &Tokenizer,
) -> GenerateOptions {
    let mut options = build_generate_options(request);
    let eos_token_ids = tokenizer.eos_token_ids();
    if let Some(first) = eos_token_ids.first().copied() {
        options.eos_token_id = Some(first);
    }
    for eos_token_id in eos_token_ids {
        let eos_sequence = StopSequence::Tokens(vec![eos_token_id]);
        if !options.stop_sequences.contains(&eos_sequence) {
            options.stop_sequences.push(eos_sequence);
        }
    }
    if let Some(im_end_id) = tokenizer.token_id("<|im_end|>") {
        let im_end_sequence = StopSequence::Tokens(vec![im_end_id]);
        if !options.stop_sequences.contains(&im_end_sequence) {
            options.stop_sequences.push(im_end_sequence);
        }
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
        .and_then(|message| message.content.clone())
        .unwrap_or_default()
}

fn render_prompt(
    request: &ChatCompletionRequest,
    chat_template: Option<&ChatTemplate>,
) -> anyhow::Result<String> {
    if let Some(chat_template) = chat_template {
        let messages = request
            .messages
            .iter()
            .map(|message| {
                let mut template_message = TemplateChatMessage::new(
                    message.role.as_str(),
                    message.content.clone().unwrap_or_default(),
                );
                if let Some(tool_calls) = &message.tool_calls {
                    template_message =
                        template_message.with_tool_calls(serde_json::to_value(tool_calls)?);
                }
                Ok(template_message)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tools_json = request
            .tools
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        return chat_template
            .render(&messages, tools_json.as_deref(), true)
            .map_err(|err| anyhow::anyhow!("chat template render failed: {err}"));
    }
    Ok(build_prompt(request))
}

/// Build the Phase 2 chat prompt with a simple role-tagged template:
/// `<|role|>\n{content}\n` for every message, followed by `<|assistant|>\n`.
/// Model-specific templates will replace this once tokenizer chat templates are wired.
pub fn build_prompt(request: &ChatCompletionRequest) -> String {
    let mut prompt = String::new();
    if let Some(tools) = request.tools.as_ref().filter(|tools| !tools.is_empty()) {
        prompt.push_str("<|tools|>\n");
        prompt.push_str(&serde_json::to_string(tools).unwrap_or_else(|_| "[]".to_string()));
        prompt.push('\n');
    }
    if let Some(tool_choice) = &request.tool_choice {
        prompt.push_str("<|tool_choice|>\n");
        prompt.push_str(&tool_choice_prompt(tool_choice));
        prompt.push('\n');
    }
    for message in &request.messages {
        prompt.push_str("<|");
        prompt.push_str(message.role.trim());
        prompt.push_str("|>\n");
        if let Some(tool_call_id) = &message.tool_call_id {
            prompt.push_str("tool_call_id: ");
            prompt.push_str(tool_call_id);
            prompt.push('\n');
        }
        if let Some(content) = &message.content {
            prompt.push_str(content);
        }
        if let Some(tool_calls) = &message.tool_calls {
            if message.content.is_some() {
                prompt.push('\n');
            }
            prompt
                .push_str(&serde_json::to_string(tool_calls).unwrap_or_else(|_| "[]".to_string()));
        }
        prompt.push('\n');
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn tool_choice_prompt(tool_choice: &ToolChoice) -> String {
    match tool_choice {
        ToolChoice::Mode(mode) => match mode {
            ToolChoiceMode::Auto => "auto".to_string(),
            ToolChoiceMode::None => "none".to_string(),
            ToolChoiceMode::Required => "required".to_string(),
        },
        ToolChoice::Specific(choice) => format!("function: {}", choice.function.name),
    }
}

pub fn parse_tool_calls(output: &str) -> Vec<ChatMessageToolCall> {
    let mut calls = Vec::new();
    let mut rest = output;
    while let Some(start) = rest.find("<tool_call>") {
        rest = &rest[start + "<tool_call>".len()..];
        let Some(end) = rest.find("</tool_call>") else {
            break;
        };
        let body = rest[..end].trim();
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
            if let Some(call) = parsed_tool_call_to_openai(calls.len(), value) {
                calls.push(call);
            }
        }
        rest = &rest[end + "</tool_call>".len()..];
    }
    calls
}

#[derive(Debug, Clone)]
pub struct ParsedAssistantOutput {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ChatMessageToolCall>>,
    pub finish_reason: &'static str,
}

pub fn parse_assistant_output(
    output: String,
    default_finish_reason: &'static str,
) -> ParsedAssistantOutput {
    let tool_calls = parse_tool_calls(&output);
    if tool_calls.is_empty() {
        ParsedAssistantOutput {
            content: Some(output),
            tool_calls: None,
            finish_reason: default_finish_reason,
        }
    } else {
        ParsedAssistantOutput {
            content: None,
            tool_calls: Some(tool_calls),
            finish_reason: "tool_calls",
        }
    }
}

fn parsed_tool_call_to_openai(
    index: usize,
    value: serde_json::Value,
) -> Option<ChatMessageToolCall> {
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    Some(ChatMessageToolCall {
        id: format!("call_{index}"),
        kind: "function".to_string(),
        function: ChatMessageToolCallFunction {
            name,
            arguments: serde_json::to_string(&arguments).ok()?,
        },
    })
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

fn load_chat_template(model_dir: &Path) -> anyhow::Result<Option<ChatTemplate>> {
    let standalone = model_dir.join("chat_template.jinja");
    let tokenizer_config = model_dir.join("tokenizer_config.json");
    let has_template = standalone.is_file()
        || tokenizer_config.is_file()
            && std::fs::read_to_string(&tokenizer_config)
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| value.get("chat_template").cloned())
                .and_then(|value| value.as_str().map(ToString::to_string))
                .is_some();
    if has_template {
        Ok(Some(ChatTemplate::from_model_dir(model_dir)?))
    } else {
        Ok(None)
    }
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
