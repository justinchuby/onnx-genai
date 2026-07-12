//! OpenAI-compatible HTTP server wiring for onnx-genai.

use std::{
    collections::HashMap,
    convert::Infallible,
    net::SocketAddr,
    path::Path,
    sync::{Arc, Mutex},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{delete, get, post},
};
use onnx_genai::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
    GenerateResult, GenerateToken, SessionId, StopSequence,
};
use onnx_genai_engine::{ContinuousBatchEvent, ContinuousBatchManager, GenerateConstraint};
use onnx_genai_ort::{ChatMessage as TemplateChatMessage, ChatTemplate, ModelDirectory, Tokenizer};
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

const SESSION_ID_HEADER: &str = "x-session-id";
const DEFAULT_MAX_OUTPUT_TOKENS: usize = 4096;
const DEFAULT_MAX_SESSIONS: usize = 256;
const DEFAULT_MAX_PENDING: usize = 256;
const DEFAULT_MAX_BATCH: usize = 4;
const MAX_SESSION_ID_LEN: usize = 128;
const DRIVER_OUTPUT_BUFFER: usize = 16;
const OVERLOAD_RETRY_AFTER_SECS: u64 = 1;

#[derive(Clone)]
pub struct AppState {
    model_id: String,
    engine: EngineDriver,
    tokenizer: Arc<Tokenizer>,
    chat_template: Option<Arc<ChatTemplate>>,
    sessions: SessionRegistry,
    config: ServerConfig,
    model_max_context: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct ServerConfig {
    pub max_output_tokens: usize,
    pub max_sessions: usize,
    /// Maximum generation requests admitted to the driver, including active and queued work.
    pub max_pending: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            max_sessions: DEFAULT_MAX_SESSIONS,
            max_pending: DEFAULT_MAX_PENDING,
        }
    }
}

impl ServerConfig {
    fn validate(self) -> anyhow::Result<Self> {
        if self.max_output_tokens == 0 {
            anyhow::bail!("max_output_tokens must be greater than zero");
        }
        if self.max_sessions == 0 {
            anyhow::bail!("max_sessions must be greater than zero");
        }
        if self.max_pending == 0 {
            anyhow::bail!("max_pending must be greater than zero");
        }
        Ok(self)
    }
}

#[derive(Clone)]
struct EngineDriver {
    commands: mpsc::Sender<DriverCommand>,
    generation_capacity: Arc<Semaphore>,
}

enum DriverCommand {
    CreateSession(tokio::sync::oneshot::Sender<anyhow::Result<SessionId>>),
    CloseSession {
        session_id: SessionId,
        response: tokio::sync::oneshot::Sender<anyhow::Result<()>>,
    },
    SessionTokenCount {
        session_id: SessionId,
        response: tokio::sync::oneshot::Sender<anyhow::Result<usize>>,
    },
    Generate {
        session_id: Option<SessionId>,
        request: GenerateRequest,
        events: mpsc::Sender<DriverEvent>,
        permit: OwnedSemaphorePermit,
    },
}

#[derive(Debug)]
enum DriverEvent {
    Token(GenerateToken),
    Finished(GenerateResult),
    Error(String),
}

struct EngineOwner(Engine);

#[derive(Debug)]
enum GenerateSubmitError {
    Overloaded,
    DriverStopped,
}

struct DriverRoute {
    events: mpsc::Sender<DriverEvent>,
    _permit: OwnedSemaphorePermit,
}

// SAFETY: The engine is moved exactly once into the dedicated driver thread.
// All ORT runners, sessions, KV state, and the continuous batch manager stay
// owned by that thread and are accessed only by processing channel commands.
unsafe impl Send for EngineOwner {}

impl EngineDriver {
    fn start(engine: Engine, max_batch: usize, max_pending: usize) -> Self {
        let (commands, rx) = mpsc::channel(max_pending);
        let generation_capacity = Arc::new(Semaphore::new(max_pending));
        let owner = EngineOwner(engine);
        thread::Builder::new()
            .name("onnx-genai-batch-driver".to_string())
            .spawn(move || run_engine_driver(owner, rx, max_batch))
            .expect("failed to spawn onnx-genai engine driver");
        Self {
            commands,
            generation_capacity,
        }
    }

    async fn create_session(&self) -> anyhow::Result<SessionId> {
        let (response, rx) = tokio::sync::oneshot::channel();
        self.commands
            .send(DriverCommand::CreateSession(response))
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    async fn close_session(&self, session_id: SessionId) -> anyhow::Result<()> {
        let (response, rx) = tokio::sync::oneshot::channel();
        self.commands
            .send(DriverCommand::CloseSession {
                session_id,
                response,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    async fn session_token_count(&self, session_id: SessionId) -> anyhow::Result<usize> {
        let (response, rx) = tokio::sync::oneshot::channel();
        self.commands
            .send(DriverCommand::SessionTokenCount {
                session_id,
                response,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    async fn generate(
        &self,
        session_id: Option<SessionId>,
        request: GenerateRequest,
    ) -> Result<mpsc::Receiver<DriverEvent>, GenerateSubmitError> {
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (events, rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        self.commands
            .send(DriverCommand::Generate {
                session_id,
                request,
                events,
                permit,
            })
            .await
            .map_err(|_| GenerateSubmitError::DriverStopped)?;
        Ok(rx)
    }
}

fn run_engine_driver(owner: EngineOwner, rx: mpsc::Receiver<DriverCommand>, max_batch: usize) {
    let mut engine = owner.0;
    let static_batch_supported = engine.continuous_batch_manager(max_batch).is_ok();
    if static_batch_supported {
        tracing::info!(max_batch, "static-cache continuous batch driver enabled");
        run_static_engine_driver(&mut engine, rx, max_batch);
    } else {
        tracing::info!("continuous batch driver disabled; using per-request engine path");
        run_fallback_engine_driver(&mut engine, rx);
    }
}

fn run_fallback_engine_driver(engine: &mut Engine, mut rx: mpsc::Receiver<DriverCommand>) {
    while let Some(command) = rx.blocking_recv() {
        handle_driver_command(engine, command);
    }
}

fn run_static_engine_driver(
    engine: &mut Engine,
    mut rx: mpsc::Receiver<DriverCommand>,
    max_batch: usize,
) {
    // The current ContinuousBatchManager API accepts GenerateRequest only.
    // X-Session-Id requests keep using the driver's per-request engine path so
    // persistent engine KV/session semantics are preserved until the manager
    // grows a SessionId-aware submit API.
    let mut deferred = std::collections::VecDeque::new();
    loop {
        while let Ok(command) = rx.try_recv() {
            match command {
                command @ DriverCommand::Generate {
                    session_id: None, ..
                } => deferred.push_back(command),
                command => deferred.push_back(command),
            }
        }

        let Some(first) = deferred.pop_front().or_else(|| rx.blocking_recv()) else {
            break;
        };

        match first {
            DriverCommand::Generate {
                session_id: None,
                request,
                events,
                permit,
            } => {
                run_static_batch_until_idle(
                    engine,
                    &mut rx,
                    &mut deferred,
                    max_batch,
                    request,
                    events,
                    permit,
                );
            }
            command => handle_driver_command(engine, command),
        }
    }
}

fn run_static_batch_until_idle(
    engine: &Engine,
    rx: &mut mpsc::Receiver<DriverCommand>,
    deferred: &mut std::collections::VecDeque<DriverCommand>,
    max_batch: usize,
    first_request: GenerateRequest,
    first_events: mpsc::Sender<DriverEvent>,
    first_permit: OwnedSemaphorePermit,
) {
    let mut manager = match engine.continuous_batch_manager(max_batch) {
        Ok(manager) => manager,
        Err(err) => {
            let _ = first_events.try_send(DriverEvent::Error(format!(
                "continuous batch setup failed: {err}"
            )));
            return;
        }
    };
    let mut routes: HashMap<usize, DriverRoute> = HashMap::new();
    let mut abandoned = HashMap::new();
    submit_to_continuous_manager(
        &mut manager,
        &mut routes,
        &mut abandoned,
        first_request,
        first_events,
        first_permit,
    );

    loop {
        while let Ok(command) = rx.try_recv() {
            match command {
                DriverCommand::Generate {
                    session_id: None,
                    request,
                    events,
                    permit,
                } => submit_to_continuous_manager(
                    &mut manager,
                    &mut routes,
                    &mut abandoned,
                    request,
                    events,
                    permit,
                ),
                command => deferred.push_back(command),
            }
        }

        if let Err(err) = manager.step() {
            let message = format!("continuous batch generation failed: {err}");
            for (_, route) in routes.drain() {
                let _ = route.events.try_send(DriverEvent::Error(message.clone()));
            }
            break;
        }
        route_continuous_events(manager.poll(), &mut routes, &mut abandoned);
        if manager.is_idle() {
            break;
        }
    }
}

fn submit_to_continuous_manager(
    manager: &mut ContinuousBatchManager<'_>,
    routes: &mut HashMap<usize, DriverRoute>,
    abandoned: &mut HashMap<usize, OwnedSemaphorePermit>,
    request: GenerateRequest,
    events: mpsc::Sender<DriverEvent>,
    permit: OwnedSemaphorePermit,
) {
    match manager.submit(request) {
        Ok(handle) => {
            routes.insert(
                handle.id,
                DriverRoute {
                    events,
                    _permit: permit,
                },
            );
            route_continuous_events(manager.poll(), routes, abandoned);
        }
        Err(err) => {
            let _ = events.try_send(DriverEvent::Error(err.to_string()));
        }
    }
}

fn route_continuous_events(
    events: Vec<ContinuousBatchEvent>,
    routes: &mut HashMap<usize, DriverRoute>,
    abandoned: &mut HashMap<usize, OwnedSemaphorePermit>,
) {
    for event in events {
        match event {
            ContinuousBatchEvent::Token { handle, token } => {
                // A slow or disconnected consumer loses its route immediately. The
                // driver never waits for output capacity; it keeps stepping every
                // other row while the manager retires the abandoned row.
                let delivery_failed = routes
                    .get(&handle.id)
                    .is_some_and(|route| route.events.try_send(DriverEvent::Token(token)).is_err());
                if delivery_failed {
                    if let Some(route) = routes.remove(&handle.id) {
                        abandoned.insert(handle.id, route._permit);
                    }
                }
            }
            ContinuousBatchEvent::Finished { handle, result } => {
                if let Some(route) = routes.remove(&handle.id) {
                    let _ = route.events.try_send(DriverEvent::Finished(result));
                } else {
                    abandoned.remove(&handle.id);
                }
            }
        }
    }
}

fn handle_driver_command(engine: &mut Engine, command: DriverCommand) {
    match command {
        DriverCommand::CreateSession(response) => {
            let _ = response.send(engine.create_session());
        }
        DriverCommand::CloseSession {
            session_id,
            response,
        } => {
            let _ = response.send(engine.close_session(session_id));
        }
        DriverCommand::SessionTokenCount {
            session_id,
            response,
        } => {
            let _ = response.send(engine.session_token_count(session_id));
        }
        DriverCommand::Generate {
            session_id,
            request,
            events,
            permit,
        } => run_fallback_generation(engine, session_id, request, events, permit),
    }
}

fn run_fallback_generation(
    engine: &mut Engine,
    session_id: Option<SessionId>,
    request: GenerateRequest,
    events: mpsc::Sender<DriverEvent>,
    _permit: OwnedSemaphorePermit,
) {
    let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
        events
            .try_send(DriverEvent::Token(token))
            .context("stream receiver closed")
    };
    let result = match session_id {
        Some(session_id) => {
            engine.generate_in_session_with_callback(session_id, request, Some(&mut callback))
        }
        None => engine.generate_with_callback(request, Some(&mut callback)),
    };
    match result {
        Ok(result) => {
            let _ = events.try_send(DriverEvent::Finished(result));
        }
        Err(err) => {
            let _ = events.try_send(DriverEvent::Error(err.to_string()));
        }
    }
}

#[derive(Clone)]
struct SessionRegistry {
    inner: Arc<Mutex<SessionRegistryInner>>,
    max_sessions: usize,
}

#[derive(Debug)]
struct SessionRegistryInner {
    sessions: HashMap<String, SessionEntry>,
    access_clock: u64,
}

#[derive(Debug)]
struct SessionEntry {
    engine_session_id: SessionId,
    last_access: u64,
}

impl SessionRegistry {
    fn new(max_sessions: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SessionRegistryInner {
                sessions: HashMap::new(),
                access_clock: 0,
            })),
            max_sessions,
        }
    }

    fn insert(
        &self,
        client_id: String,
        engine_session_id: SessionId,
    ) -> anyhow::Result<Option<SessionId>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?;
        let evicted = if inner.sessions.len() >= self.max_sessions {
            inner
                .sessions
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(id, _)| id.clone())
                .and_then(|id| {
                    inner
                        .sessions
                        .remove(&id)
                        .map(|entry| entry.engine_session_id)
                })
        } else {
            None
        };
        inner.access_clock = inner.access_clock.saturating_add(1);
        let last_access = inner.access_clock;
        inner.sessions.insert(
            client_id,
            SessionEntry {
                engine_session_id,
                last_access,
            },
        );
        Ok(evicted)
    }

    fn get(&self, client_id: &str) -> anyhow::Result<Option<SessionId>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?;
        if !inner.sessions.contains_key(client_id) {
            return Ok(None);
        }
        inner.access_clock = inner.access_clock.saturating_add(1);
        let last_access = inner.access_clock;
        let entry = inner
            .sessions
            .get_mut(client_id)
            .expect("entry checked above");
        entry.last_access = last_access;
        Ok(Some(entry.engine_session_id))
    }

    fn remove(&self, client_id: &str) -> anyhow::Result<Option<SessionId>> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?
            .sessions
            .remove(client_id)
            .map(|entry| entry.engine_session_id))
    }

    fn next_client_id(&self) -> anyhow::Result<String> {
        let mut bytes = [0_u8; 16];
        getrandom::fill(&mut bytes).context("OS CSPRNG failed")?;
        Ok(format!("sess-{}", hex_token(&bytes)))
    }
}

impl AppState {
    pub fn load(model_dir: &Path, model_id: Option<String>) -> anyhow::Result<Self> {
        Self::load_with_config(model_dir, model_id, ServerConfig::default())
    }

    pub fn load_with_config(
        model_dir: &Path,
        model_id: Option<String>,
        config: ServerConfig,
    ) -> anyhow::Result<Self> {
        let config = config.validate()?;
        let model_directory = ModelDirectory::load(model_dir)
            .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {}", e))?;
        let model_max_context = load_model_max_context(model_directory.metadata_path.as_deref())?;
        let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let chat_template = load_chat_template(model_dir)?;
        let engine = Engine::from_dir(model_dir, EngineConfig::default())?;
        let model_id = model_id.unwrap_or_else(|| infer_model_id(model_dir));
        Ok(Self::new_with_template_and_config(
            model_id,
            engine,
            tokenizer,
            chat_template,
            config,
            model_max_context,
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
        Self::new_with_template_and_config(
            model_id,
            engine,
            tokenizer,
            chat_template,
            ServerConfig::default(),
            None,
        )
    }

    fn new_with_template_and_config(
        model_id: String,
        engine: Engine,
        tokenizer: Tokenizer,
        chat_template: Option<ChatTemplate>,
        config: ServerConfig,
        model_max_context: Option<usize>,
    ) -> Self {
        let config = config.validate().expect("validated server config");
        Self {
            model_id,
            engine: EngineDriver::start(engine, DEFAULT_MAX_BATCH, config.max_pending),
            tokenizer: Arc::new(tokenizer),
            chat_template: chat_template.map(Arc::new),
            sessions: SessionRegistry::new(config.max_sessions),
            config,
            model_max_context,
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
    // Security posture: the server has no built-in authentication. The CLI defaults
    // to 127.0.0.1, enforces max_tokens/max_sessions caps, and issues CSPRNG session
    // ids; binding a non-loopback --addr should be done only behind an auth proxy.
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
    retry_after_secs: Option<u64>,
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
            retry_after_secs: None,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            retry_after_secs: None,
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
            retry_after_secs: None,
        }
    }

    fn too_many_requests(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: message.into(),
            retry_after_secs: Some(OVERLOAD_RETRY_AFTER_SECS),
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
        let mut response = (self.status, body).into_response();
        if let Some(seconds) = self.retry_after_secs {
            response.headers_mut().insert(
                header::RETRY_AFTER,
                HeaderValue::from_str(&seconds.to_string()).expect("valid retry-after"),
            );
        }
        response
    }
}

fn map_generate_submit_error(err: GenerateSubmitError) -> ApiError {
    match err {
        GenerateSubmitError::Overloaded => ApiError::too_many_requests(
            "generation capacity exceeded; retry after the server finishes queued work",
        ),
        GenerateSubmitError::DriverStopped => ApiError::internal("engine driver stopped"),
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
    let client_id = state
        .sessions
        .next_client_id()
        .map_err(|err| ApiError::internal(format!("session id generation failed: {err}")))?;
    let engine_session_id = state
        .engine
        .create_session()
        .await
        .map_err(|err| ApiError::internal(format!("session create failed: {err}")))?;

    let evicted = state
        .sessions
        .insert(client_id.clone(), engine_session_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    close_evicted_session(&state, evicted).await?;

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

    state
        .engine
        .close_session(engine_session_id)
        .await
        .map_err(|err| ApiError::internal(format!("session close failed: {err}")))?;

    Ok(StatusCode::NO_CONTENT)
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    validate_request(&request, &state.config)?;
    let session_id = session_id_from_headers(&headers)?;
    if request.stream {
        Ok(stream_chat_completion(state, request, session_id)
            .await?
            .into_response())
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
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        state.model_max_context,
    )?;
    let prompt_tokens = prepared.prompt_tokens;
    let mut generation_request = prepared.request;
    generation_request.options.max_context = state.model_max_context;
    let session_lookup = if let Some(id) = client_session_id.as_deref() {
        Some(get_or_create_session(&state, id).await?)
    } else {
        None
    };

    let session_for_count = session_lookup;
    let wants_json_object = request.wants_json_object();
    let result = collect_generation_result(
        state
            .engine
            .generate(session_lookup, generation_request)
            .await
            .map_err(map_generate_submit_error)?,
    )
    .await
    .map_err(|err| ApiError::internal(format!("generation failed: {err}")));

    let session_token_count = if let Some(engine_session_id) = session_for_count {
        Some(
            state
                .engine
                .session_token_count(engine_session_id)
                .await
                .map_err(|err| ApiError::internal(format!("session token count failed: {err}")))?,
        )
    } else {
        None
    };

    let (content, tool_calls, completion_tokens, finish_reason) = match result {
        Ok(result) => {
            let default_finish_reason = finish_reason_label(&result.finish_reason);
            let parsed = if tools_parseable_from_output(&request) {
                parse_assistant_output(result.text, default_finish_reason)
            } else {
                ParsedAssistantOutput {
                    content: Some(result.text),
                    tool_calls: None,
                    finish_reason: default_finish_reason,
                }
            };
            (
                parsed.content,
                parsed.tool_calls,
                result.token_ids.len(),
                parsed.finish_reason,
            )
        }
        Err(err)
            if wants_json_object && json_constraint_stopped_incomplete_message(&err.message) =>
        {
            (Some("{}".to_string()), None, 0, "stop")
        }
        Err(err) => return Err(err),
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

async fn stream_chat_completion(
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
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        state.model_max_context,
    )?;
    let wants_json_object = request.wants_json_object();
    let mut generation_request = prepared.request;
    generation_request.options.max_context = state.model_max_context;
    let (tx, rx) = mpsc::channel(16);
    let session_lookup = if let Some(id) = client_session_id.as_deref() {
        Some(get_or_create_session(&state, id).await?)
    } else {
        None
    };
    let mut driver_rx = state
        .engine
        .generate(session_lookup, generation_request)
        .await
        .map_err(map_generate_submit_error)?;

    tokio::spawn(async move {
        send_stream_chunk(&tx, role_chunk(&id, created, &model)).await?;

        let mut stop_buffer = StopBoundaryBuffer::new(user_stop_sequences);
        let mut buffered_text = String::new();
        let buffer_for_tool_detection =
            request.has_tool_context() && tools_parseable_from_output(&request);
        let result = loop {
            match driver_rx.recv().await {
                Some(DriverEvent::Token(token)) => {
                    let finish_reason = token.finish_reason.clone();
                    let content = stop_buffer.push(&token.text);
                    if buffer_for_tool_detection {
                        buffered_text.push_str(&content);
                    } else if !wants_json_object && !content.is_empty() {
                        send_stream_chunk(&tx, content_chunk(&id, created, &model, content))
                            .await?;
                    }
                    if matches!(finish_reason, Some(FinishReason::StopSequence { .. })) {
                        stop_buffer.pending.clear();
                    }
                }
                Some(DriverEvent::Finished(result)) => break Ok(result),
                Some(DriverEvent::Error(message)) => break Err(message),
                None => break Err("generation stream ended before result".to_string()),
            }
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
                            send_stream_chunk(
                                &tx,
                                content_chunk(&id, created, &model, buffered_text),
                            )
                            .await?;
                        }
                        send_stream_chunk(
                            &tx,
                            done_chunk(
                                &id,
                                created,
                                &model,
                                finish_reason_label(&result.finish_reason),
                            ),
                        )
                        .await?;
                    } else {
                        send_stream_chunk(&tx, tool_calls_chunk(&id, created, &model, tool_calls))
                            .await?;
                        send_stream_chunk(&tx, done_chunk(&id, created, &model, "tool_calls"))
                            .await?;
                    }
                } else if wants_json_object {
                    if !result.text.is_empty() {
                        send_stream_chunk(&tx, content_chunk(&id, created, &model, result.text))
                            .await?;
                    }
                    send_stream_chunk(
                        &tx,
                        done_chunk(
                            &id,
                            created,
                            &model,
                            finish_reason_label(&result.finish_reason),
                        ),
                    )
                    .await?;
                } else {
                    if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                        let content = stop_buffer.flush();
                        if !content.is_empty() {
                            send_stream_chunk(&tx, content_chunk(&id, created, &model, content))
                                .await?;
                        }
                    }
                    send_stream_chunk(
                        &tx,
                        done_chunk(
                            &id,
                            created,
                            &model,
                            finish_reason_label(&result.finish_reason),
                        ),
                    )
                    .await?;
                }
            }
            Err(err) if wants_json_object && json_constraint_stopped_incomplete_message(&err) => {
                send_stream_chunk(&tx, content_chunk(&id, created, &model, "{}".to_string()))
                    .await?;
                send_stream_chunk(&tx, done_chunk(&id, created, &model, "stop")).await?;
            }
            Err(err) => {
                tx.send(Ok(Event::default().event("error").data(
                    serde_json::to_string(&ErrorResponse {
                        error: ErrorBody {
                            message: format!("generation failed: {err}"),
                            kind: "server_error",
                        },
                    })?,
                )))
                .await
                .context("stream receiver closed")?;
            }
        }

        tx.send(Ok(Event::default().data("[DONE]")))
            .await
            .context("stream receiver closed")?;
        Ok::<(), anyhow::Error>(())
    });

    Ok(Sse::new(ReceiverStream::new(rx)))
}

async fn collect_generation_result(
    mut rx: mpsc::Receiver<DriverEvent>,
) -> Result<GenerateResult, String> {
    while let Some(event) = rx.recv().await {
        match event {
            DriverEvent::Token(_) => {}
            DriverEvent::Finished(result) => return Ok(result),
            DriverEvent::Error(message) => return Err(message),
        }
    }
    Err("generation stream ended before result".to_string())
}

async fn send_stream_chunk(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    chunk: ChatCompletionChunk,
) -> anyhow::Result<()> {
    tx.send(Ok(Event::default().data(serde_json::to_string(&chunk)?)))
        .await
        .context("stream receiver closed")
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

fn validate_request(
    request: &ChatCompletionRequest,
    config: &ServerConfig,
) -> Result<(), ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    if request.max_tokens == 0 {
        return Err(ApiError::bad_request(
            "max_tokens must be greater than zero",
        ));
    }
    if request.max_tokens > config.max_output_tokens {
        return Err(ApiError::bad_request(format!(
            "max_tokens must be less than or equal to the server cap of {}",
            config.max_output_tokens
        )));
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
    validate_tool_choice(request)?;
    Ok(())
}

fn enforce_context_cap(
    prompt_tokens: usize,
    max_tokens: usize,
    model_max_context: Option<usize>,
) -> Result<(), ApiError> {
    let Some(model_max_context) = model_max_context else {
        return Ok(());
    };
    let total = prompt_tokens
        .checked_add(max_tokens)
        .ok_or_else(|| ApiError::bad_request("prompt_tokens + max_tokens overflowed"))?;
    if total > model_max_context {
        return Err(ApiError::bad_request(format!(
            "prompt token count ({prompt_tokens}) plus max_tokens ({max_tokens}) exceeds model context limit ({model_max_context})"
        )));
    }
    Ok(())
}

fn validate_tool_choice(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    let Some(tool_choice) = &request.tool_choice else {
        return Ok(());
    };
    match tool_choice {
        ToolChoice::Mode(ToolChoiceMode::Required) => {
            if !request
                .tools
                .as_ref()
                .is_some_and(|tools| tools.iter().any(|tool| tool.kind == "function"))
            {
                return Err(ApiError::bad_request(
                    "tool_choice required requires at least one function tool",
                ));
            }
        }
        ToolChoice::Specific(choice) => {
            if choice.kind != "function" {
                return Err(ApiError::bad_request(
                    "specific tool_choice type must be function",
                ));
            }
            if !request.tools.as_ref().is_some_and(|tools| {
                tools.iter().any(|tool| {
                    tool.kind == "function" && tool.function.name == choice.function.name
                })
            }) {
                return Err(ApiError::bad_request(format!(
                    "tool_choice function '{}' was not provided in tools",
                    choice.function.name
                )));
            }
        }
        ToolChoice::Mode(ToolChoiceMode::Auto | ToolChoiceMode::None) => {}
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
    if session_id.len() > MAX_SESSION_ID_LEN {
        return Err(ApiError::bad_request(format!(
            "X-Session-Id must be at most {MAX_SESSION_ID_LEN} bytes"
        )));
    }
    Ok(Some(session_id.to_string()))
}

async fn get_or_create_session(state: &AppState, client_id: &str) -> Result<SessionId, ApiError> {
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
        .await
        .map_err(|err| ApiError::internal(format!("session create failed: {err}")))?;
    let evicted = state
        .sessions
        .insert(client_id.to_string(), engine_session_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    close_evicted_session(state, evicted).await?;
    Ok(engine_session_id)
}

async fn close_evicted_session(
    state: &AppState,
    evicted: Option<SessionId>,
) -> Result<(), ApiError> {
    if let Some(evicted) = evicted {
        state
            .engine
            .close_session(evicted)
            .await
            .map_err(|err| ApiError::internal(format!("evicted session close failed: {err}")))?;
    }
    Ok(())
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
    if let Some(constraint) = forced_tool_choice_constraint(request) {
        options.constraint = Some(constraint);
    }
    options
}

fn forced_tool_choice_constraint(request: &ChatCompletionRequest) -> Option<GenerateConstraint> {
    let schemas = forced_tool_choice_schemas(request)?;
    let schema = if schemas.len() == 1 {
        schemas.into_iter().next()?
    } else {
        serde_json::json!({ "anyOf": schemas })
    };
    let schema = serde_json::to_string(&schema).ok()?;
    Some(GenerateConstraint::Lark(format!(
        "start: \"<tool_call>\\n\" tool \"\\n</tool_call>\"\ntool: %json {schema}\n"
    )))
}

fn forced_tool_choice_schemas(request: &ChatCompletionRequest) -> Option<Vec<serde_json::Value>> {
    let tools = request
        .tools
        .as_ref()?
        .iter()
        .filter(|tool| tool.kind == "function");
    let selected = match request.tool_choice.as_ref()? {
        ToolChoice::Mode(ToolChoiceMode::Required) => tools.collect::<Vec<_>>(),
        ToolChoice::Specific(choice) if choice.kind == "function" => tools
            .filter(|tool| tool.function.name == choice.function.name)
            .collect::<Vec<_>>(),
        ToolChoice::Mode(ToolChoiceMode::Auto | ToolChoiceMode::None) | ToolChoice::Specific(_) => {
            Vec::new()
        }
    };

    let schemas = selected
        .into_iter()
        .map(tool_call_schema_for_tool)
        .collect::<Vec<_>>();
    (!schemas.is_empty()).then_some(schemas)
}

fn tool_call_schema_for_tool(tool: &ChatTool) -> serde_json::Value {
    let arguments_schema = tool
        .function
        .parameters
        .clone()
        .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "enum": [tool.function.name.clone()] },
            "arguments": arguments_schema
        },
        "required": ["name", "arguments"],
        "additionalProperties": false
    })
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

fn json_constraint_stopped_incomplete_message(message: &str) -> bool {
    message.contains("JSON constrained decoding stopped before a complete JSON value")
}

fn tools_parseable_from_output(request: &ChatCompletionRequest) -> bool {
    !matches!(
        request.tool_choice,
        Some(ToolChoice::Mode(ToolChoiceMode::None))
    )
}

fn tools_offered_to_model(request: &ChatCompletionRequest) -> Option<&Vec<ChatTool>> {
    if matches!(
        request.tool_choice,
        Some(ToolChoice::Mode(ToolChoiceMode::None))
    ) {
        return None;
    }
    request.tools.as_ref().filter(|tools| !tools.is_empty())
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
        let tools_json = tools_offered_to_model(request)
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
    if let Some(tools) = tools_offered_to_model(request) {
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
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(body)
            && let Some(call) = parsed_tool_call_to_openai(calls.len(), value)
        {
            calls.push(call);
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
    // OpenAI tool calls end the assistant turn. The batch row finishes normally
    // with finish_reason=tool_calls; role=tool follow-up messages are submitted
    // as a new turn rather than pausing and resuming mid-token.
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

fn hex_token(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
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

fn load_model_max_context(metadata_path: Option<&Path>) -> anyhow::Result<Option<usize>> {
    let Some(metadata_path) = metadata_path else {
        return Ok(None);
    };
    let metadata = onnx_genai_metadata::load_metadata(metadata_path)
        .with_context(|| format!("failed to load {}", metadata_path.display()))?;
    Ok(metadata.model.and_then(|model| model.max_sequence_length))
}

#[cfg(test)]
mod tests {
    use super::{
        AppState, ChatCompletionRequest, DriverCommand, Engine, EngineConfig, EngineDriver,
        ServerConfig, StopBoundaryBuffer, app, build_generate_request, collect_generation_result,
    };
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header},
    };
    use serde_json::{Value, json};
    use std::{path::PathBuf, time::Duration};
    use tokio::{sync::mpsc, time::timeout};
    use tower::ServiceExt;

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

    #[tokio::test]
    async fn generation_over_capacity_returns_429_with_retry_after() {
        let model_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
        let state = AppState::load_with_config(
            &model_dir,
            Some("tiny-llm".to_string()),
            ServerConfig {
                max_output_tokens: 16,
                max_sessions: 8,
                max_pending: 1,
            },
        )
        .unwrap();
        let _occupied = state
            .engine
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .unwrap();

        let response = app(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model": "tiny-llm",
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 1
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "1");
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("generation capacity exceeded")
        );
    }

    #[tokio::test]
    async fn stalled_output_route_does_not_block_another_completion() {
        let model_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter");
        let engine = Engine::from_dir(&model_dir, EngineConfig::default()).unwrap();
        let driver = EngineDriver::start(engine, 2, 2);
        let slow_request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "tiny-llm-scatter",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8
        }))
        .unwrap();
        let fast_request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "tiny-llm-scatter",
            "messages": [{"role": "user", "content": "world"}],
            "max_tokens": 2
        }))
        .unwrap();
        let (slow_tx, _slow_rx) = mpsc::channel(1);
        let slow_permit = driver
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .unwrap();
        driver
            .commands
            .send(DriverCommand::Generate {
                session_id: None,
                request: build_generate_request(&slow_request),
                events: slow_tx,
                permit: slow_permit,
            })
            .await
            .unwrap();
        let fast_rx = driver
            .generate(None, build_generate_request(&fast_request))
            .await
            .unwrap();

        let fast_result = timeout(Duration::from_secs(5), collect_generation_result(fast_rx))
            .await
            .expect("fast request timed out behind stalled consumer")
            .expect("fast request failed");
        assert_eq!(fast_result.token_ids.len(), 2);
    }
}
