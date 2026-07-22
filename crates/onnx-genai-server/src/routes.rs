use std::{
    convert::Infallible,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use axum::{
    Json,
    extract::{Multipart, Path as AxumPath, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response, Sse, sse::Event},
};
use onnx_genai::{
    FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult, SessionId,
    StopSequence,
};
use onnx_genai_engine::{
    EmbeddingOptions, EngineGovernorError, GenerateConstraint, GovernorSnapshot, ResourceLimit,
    TokenLogprob, parse_resource_limit,
};
use onnx_genai_ort::{ChatMessage as TemplateChatMessage, ChatTemplate, Tokenizer};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    driver::{
        DriverEvent, EngineDriver, GenerateSubmitError, PipelineInputBundle, PipelineInputTensor,
        PipelineTensor,
    },
    registry::ModelHandle,
    session::SessionRegistry,
    sse::{
        StopBoundaryBuffer, completion_chunk, completion_done_chunk, content_chunk, done_chunk,
        role_chunk, send_completion_stream_chunk, send_stream_chunk, tool_calls_chunk,
    },
    state::{AppState, ServerConfig},
    types::{
        AudioTranscriptionResponse, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
        ChatLogprobs, ChatMessage, ChatMessageContent, ChatMessageToolCall,
        ChatMessageToolCallFunction, ChatTokenLogprob, ChatTool, ChatTopLogprob, CompletionChoice,
        CompletionLogprobs, CompletionRequest, CompletionResponse, EmbeddingData, EmbeddingInput,
        EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, EmbeddingVector, InputAudio,
        StopInput, ToolChoice, ToolChoiceMode, Usage,
    },
};

const SESSION_ID_HEADER: &str = "x-session-id";
const MAX_SESSION_ID_LEN: usize = 128;
const OVERLOAD_RETRY_AFTER_SECS: u64 = 1;
const MAX_CHAT_TOP_LOGPROBS: usize = 20;
const MAX_COMPLETION_LOGPROBS: usize = 5;
/// Path of the downloadable Perfetto trace endpoint, reported by the trace
/// status endpoint so clients can discover the export without guessing.
const PERFETTO_EXPORT_PATH: &str = "/v1/debug/trace/perfetto";
/// OTLP span export is intentionally deferred (see issue #13); the status
/// endpoint reports this honestly rather than pretending it works.
const OTLP_EXPORT_STATUS: &str = "deferred: OTLP span export is not implemented (Perfetto export is available at /v1/debug/trace/perfetto)";

#[derive(Debug, Serialize)]
pub(crate) struct ModelsResponse {
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
pub(crate) struct HealthResponse {
    status: &'static str,
    model: String,
}

/// Node-status contract polled by the cluster router (§34.8) every 1-2s.
///
/// Field honesty: values are populated from generic runtime state where a getter
/// exists (`queue_depth`, `active_sessions`, `healthy`, `node_id`). Metrics the
/// server cannot yet measure are reported as documented zeros/empties rather than
/// fabricated — see the per-field comments in [`status`]. All values are
/// model-agnostic; `node_id` names this node, never a model.
#[derive(Debug, Serialize)]
pub(crate) struct NodeStatus {
    node_id: String,
    healthy: bool,
    kv_usage: f32,
    kv_pages_used: u32,
    kv_pages_total: u32,
    kv_pages_shared: u32,
    queue_depth: u32,
    active_sessions: u32,
    paused_sessions: u32,
    tokens_per_second: f64,
    batch_utilization: f32,
    sessions: Vec<SessionStatus>,
    prefix_hashes: Vec<String>,
}

/// Per-session detail entry in [`NodeStatus::sessions`] (§34.8).
#[derive(Debug, Serialize)]
pub(crate) struct SessionStatus {
    id: String,
    priority: String,
    kv_pages: u32,
    state: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct DebugConfigResponse {
    model_id: String,
    pipeline: bool,
    max_output_tokens: usize,
    max_sessions: usize,
    max_queue_depth: usize,
    model_max_context: Option<usize>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DebugSessionsResponse {
    active_sessions: u64,
    max_sessions: usize,
    sessions: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DebugKvResponse {
    prefix_cache_hits: u64,
    prefix_cache_lookups: u64,
    prefix_cache_hit_rate: f64,
    active_batch_size: u64,
    pending_queue_depth: u64,
    available_admission_slots: usize,
    rejected_requests: u64,
    engine_kv_introspection: &'static str,
}

#[derive(Debug, Serialize)]
pub(crate) struct ResourcesResponse {
    configured_limits: ConfiguredResourceLimits,
    resolved_limits: ResolvedResourceLimits,
    derived_kv_budget: DerivedKvBudget,
    vram: ResourceTier,
    host_ram: ResourceTier,
    disk_spill: Option<ResourceTier>,
}

#[derive(Debug, Serialize)]
struct ConfiguredResourceLimits {
    vram: String,
    host_ram: String,
    disk_spill: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResolvedResourceLimits {
    vram_bytes: u64,
    host_ram_bytes: u64,
    disk_spill_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct DerivedKvBudget {
    bytes: u64,
    total_pages: u64,
    max_total_tokens: u64,
    reserved_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ResourceTier {
    used: u64,
    limit: u64,
    headroom: u64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SetVramLimitRequest {
    limit: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct DebugTraceResponse {
    tracing_span: &'static str,
    latest_trace_id: String,
    /// Discovery info for the Perfetto (Chrome Trace Event Format) export.
    perfetto_export: PerfettoExportInfo,
    otlp_export: &'static str,
}

/// Discovery payload describing the downloadable Perfetto trace export.
#[derive(Debug, Serialize)]
pub(crate) struct PerfettoExportInfo {
    /// Endpoint that serves the Perfetto/Chrome-trace JSON document.
    endpoint: &'static str,
    /// Number of timeline events currently retained in the in-memory sink.
    recorded_events: usize,
    /// Whether the profiler is actively collecting spans into the sink. Spans
    /// are only recorded while `ONNX_GENAI_TRACE` is set; when unset the export
    /// is a well-formed but empty trace.
    collecting: bool,
    /// Human-readable note describing how to populate the trace.
    note: &'static str,
}

#[derive(Debug, Serialize)]
pub(crate) struct AdminModelObject {
    id: String,
    loaded: bool,
    is_default: bool,
    /// Epoch-millisecond timestamp of the last request routed to this model,
    /// present only while the model is loaded.
    last_request_at: Option<u64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct AdminModelsResponse {
    object: &'static str,
    data: Vec<AdminModelObject>,
}

#[derive(Debug, Serialize)]
pub(crate) struct AdminLoadResponse {
    id: String,
    loaded: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct SessionResponse {
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
pub(crate) struct ApiError {
    status: StatusCode,
    message: String,
    retry_after_secs: Option<u64>,
}

struct PreparedGenerateRequest {
    request: GenerateRequest,
    prompt_tokens: usize,
}

pub(crate) struct PreparedCompletion {
    pub(crate) generation: CompletionGeneration,
    prompt_tokens: usize,
}

pub(crate) enum CompletionGeneration {
    Plain(GenerateRequest),
    Fim {
        prefix: String,
        suffix: String,
        options: GenerateOptions,
    },
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

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: message.into(),
            retry_after_secs: None,
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
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

/// Route a request to the correct loaded model.
///
/// - **Non-empty `requested`** — resolves the exact id.  If the model is
///   configured but not currently loaded, it is lazily loaded (blocking the
///   request until ready).  Returns a 404 only if the id is not configured at
///   all; never falls back to the default model for a named request.
/// - **Empty `requested`** — falls back to the default model, lazily loading it
///   if necessary, preserving the single-model UX where clients omit `model`.
async fn resolve_model(
    registry: &crate::registry::ModelRegistry,
    requested: &str,
) -> Result<Arc<ModelHandle>, ApiError> {
    // Fast path: already loaded (handles empty -> default).
    if let Some(handle) = registry.resolve(requested) {
        return Ok(handle);
    }
    // Determine the concrete id to lazily load.
    let id = if requested.trim().is_empty() {
        registry
            .default_id()
            .ok_or_else(|| ApiError::internal("no model loaded"))?
    } else {
        requested.to_string()
    };
    if !registry.contains_available(&id) {
        return Err(ApiError::not_found(format!(
            "model '{requested}' not found"
        )));
    }
    registry
        .load(&id)
        .await
        .map_err(|err| ApiError::internal(format!("failed to load model '{id}': {err}")))
}

pub(crate) async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model: state.registry.default_id().unwrap_or_default().to_string(),
    })
}

pub(crate) async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: state
            .registry
            .ids()
            .into_iter()
            .map(|id| ModelObject {
                id,
                object: "model",
                created: now_unix(),
                owned_by: "onnx-genai",
            })
            .collect(),
    })
}

/// `GET /v1/status` — node-status contract for the cluster router (§34.8).
///
/// Real values: `queue_depth` (admission queue), `active_sessions` (session
/// registry), `healthy`, `node_id`. Everything else is a documented placeholder
/// because the underlying getter does not exist yet — see per-field comments.
pub(crate) async fn status(State(state): State<AppState>) -> Json<NodeStatus> {
    let snapshot = crate::metrics::snapshot();
    Json(NodeStatus {
        // Node-level id from server config; independent of any loaded model.
        node_id: state.config.node_id.clone(),
        // Healthy while the node has a default model registered to serve.
        healthy: state.registry.default_id().is_some(),
        // KV page statistics: the engine does not yet expose paged-KV
        // introspection (see /v1/debug/kv), so these stay 0 until a getter exists.
        kv_usage: 0.0,      // not yet tracked
        kv_pages_used: 0,   // not yet tracked
        kv_pages_total: 0,  // not yet tracked
        kv_pages_shared: 0, // not yet tracked
        // Real: admission/backpressure queue depth (§36).
        queue_depth: u32::try_from(snapshot.pending_requests).unwrap_or(u32::MAX),
        // Real: aggregate active sessions across the node.
        active_sessions: u32::try_from(snapshot.active_sessions).unwrap_or(u32::MAX),
        paused_sessions: 0, // not yet tracked (no preemption/pause state exposed)
        tokens_per_second: 0.0, // not yet tracked (only cumulative token totals recorded)
        batch_utilization: 0.0, // not yet tracked (max batch size not surfaced to the server)
        // Per-session detail: session ids are real (redacted, since full ids are
        // bearer tokens — see session.rs). priority/kv_pages/state are not yet
        // tracked, so they carry documented placeholders rather than invented values.
        sessions: state
            .sessions
            .client_ids_redacted()
            .unwrap_or_default()
            .into_iter()
            .map(|id| SessionStatus {
                id,
                priority: "unknown".to_string(), // not yet tracked
                kv_pages: 0,                     // not yet tracked
                state: "unknown".to_string(),    // not yet tracked
            })
            .collect(),
        // System-prompt prefix hashes are not yet surfaced by the engine.
        prefix_hashes: Vec::new(),
    })
}

pub(crate) async fn debug_config(State(state): State<AppState>) -> Json<DebugConfigResponse> {
    let handle = state
        .registry
        .resolve("")
        .expect("at least one model is loaded");
    Json(DebugConfigResponse {
        model_id: handle.id.clone(),
        pipeline: handle.pipeline,
        max_output_tokens: state.config.max_output_tokens,
        max_sessions: state.config.max_sessions,
        max_queue_depth: state.config.max_queue_depth,
        model_max_context: handle.model_max_context,
    })
}

pub(crate) async fn debug_sessions(
    State(state): State<AppState>,
) -> Result<Json<DebugSessionsResponse>, ApiError> {
    let snapshot = crate::metrics::snapshot();
    let sessions = state
        .sessions
        .client_ids_redacted()
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    Ok(Json(DebugSessionsResponse {
        active_sessions: snapshot.active_sessions,
        max_sessions: state.sessions.max_sessions(),
        sessions,
    }))
}

pub(crate) async fn debug_kv(State(state): State<AppState>) -> Json<DebugKvResponse> {
    let handle = state
        .registry
        .resolve("")
        .expect("at least one model is loaded");
    let snapshot = crate::metrics::snapshot();
    let prefix_cache_hit_rate = if snapshot.prefix_cache_lookups == 0 {
        0.0
    } else {
        snapshot.prefix_cache_hits as f64 / snapshot.prefix_cache_lookups as f64
    };
    Json(DebugKvResponse {
        prefix_cache_hits: snapshot.prefix_cache_hits,
        prefix_cache_lookups: snapshot.prefix_cache_lookups,
        prefix_cache_hit_rate,
        active_batch_size: snapshot.current_batch_size,
        pending_queue_depth: snapshot.pending_requests,
        available_admission_slots: handle.engine.generation_capacity.available_permits(),
        rejected_requests: snapshot.rejections,
        engine_kv_introspection: "unavailable: engine does not yet expose KV page statistics",
    })
}

pub(crate) async fn resources(
    State(state): State<AppState>,
) -> Result<Json<ResourcesResponse>, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let snapshot = handle
        .engine
        .resource_snapshot()
        .await
        .map_err(|err| ApiError::internal(format!("resource snapshot failed: {err}")))?;
    Ok(Json(snapshot.into()))
}

pub(crate) async fn admin_set_vram_limit(
    State(state): State<AppState>,
    Json(request): Json<SetVramLimitRequest>,
) -> Result<Json<ResourcesResponse>, ApiError> {
    let limit = parse_resource_limit(&request.limit)
        .map_err(|err| ApiError::bad_request(err.to_string()))?;
    let handle = state
        .registry
        .resolve("")
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let snapshot = handle
        .engine
        .set_vram_limit(limit)
        .await
        .map_err(|err| ApiError::internal(format!("resource override failed: {err}")))?
        .map_err(|err| match err {
            EngineGovernorError::RuntimeOverrideDisabled => ApiError::forbidden(err.to_string()),
            EngineGovernorError::Resource(_) => ApiError::conflict(err.to_string()),
        })?;
    Ok(Json(snapshot.into()))
}

pub(crate) async fn debug_trace() -> Json<DebugTraceResponse> {
    let latest_trace_id = crate::metrics::latest_trace_id();
    let recorded_events = onnx_genai_ort::profile::trace_event_count();
    let collecting = onnx_genai_ort::profile::tracing_enabled();
    Json(DebugTraceResponse {
        tracing_span: "http.request",
        latest_trace_id: format!("{latest_trace_id:016x}"),
        perfetto_export: PerfettoExportInfo {
            endpoint: PERFETTO_EXPORT_PATH,
            recorded_events,
            collecting,
            note: "GET the endpoint for a Chrome Trace Event Format document (open in https://ui.perfetto.dev). Run with ONNX_GENAI_TRACE set to collect decode spans.",
        },
        otlp_export: OTLP_EXPORT_STATUS,
    })
}

/// `GET /v1/debug/trace/perfetto` — download the accumulated decode-timeline as
/// a Chrome Trace Event Format (Perfetto) JSON document.
///
/// The document is built from the process-global profiler sink in
/// `onnx-genai-ort`, which records real ORT `session.run` timings and engine
/// step spans while `ONNX_GENAI_TRACE` is set. When no spans have been
/// recorded the response is a well-formed but empty trace (`traceEvents: []`) —
/// never fabricated events. The recorded events carry only stage names and
/// timings (no session IDs or user data), so no redaction is required.
pub(crate) async fn debug_trace_perfetto() -> Response {
    let document = onnx_genai_ort::profile::trace_document();
    let body = match serde_json::to_vec(&document) {
        Ok(body) => body,
        Err(err) => {
            return ApiError::internal(format!("failed to serialize Perfetto trace: {err}"))
                .into_response();
        }
    };
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
            (
                header::CONTENT_DISPOSITION,
                HeaderValue::from_static("attachment; filename=\"onnx-genai-trace.json\""),
            ),
        ],
        body,
    )
        .into_response()
}

/// `GET /v1/admin/models` — list every configured model with loaded/available
/// status and, for loaded models, the last-request timestamp.
pub(crate) async fn admin_list_models(State(state): State<AppState>) -> Json<AdminModelsResponse> {
    let data = state
        .registry
        .statuses()
        .into_iter()
        .map(|status| AdminModelObject {
            id: status.id,
            loaded: status.loaded,
            is_default: status.is_default,
            last_request_at: status.last_request_at,
        })
        .collect();
    Json(AdminModelsResponse {
        object: "list",
        data,
    })
}

/// `POST /v1/admin/models/{id}/load` — load a configured model.  404 if the id is
/// unknown, 500 if the model fails to build.
pub(crate) async fn admin_load_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<AdminLoadResponse>, ApiError> {
    if !state.registry.contains_available(&id) {
        return Err(ApiError::not_found(format!("model '{id}' not found")));
    }
    state
        .registry
        .load(&id)
        .await
        .map_err(|err| ApiError::internal(format!("failed to load model '{id}': {err}")))?;
    Ok(Json(AdminLoadResponse { id, loaded: true }))
}

/// `DELETE /v1/admin/models/{id}` — unload a loaded model.  The spec is kept
/// available so the model can be lazily reloaded on a later request.  404 if the
/// model is not currently loaded.
pub(crate) async fn admin_unload_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    state
        .registry
        .unload(&id)
        .map_err(|_| ApiError::not_found(format!("model '{id}' is not loaded")))?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(feature = "metrics")]
pub(crate) async fn prometheus_metrics(State(state): State<AppState>) -> Response {
    let mut output = crate::metrics::encode_prometheus();
    if let Some(handle) = state.registry.resolve("")
        && let Ok(snapshot) = handle.engine.resource_snapshot().await
    {
        output.push_str(&crate::metrics::encode_resource_governor(&snapshot));
    }
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        output,
    )
        .into_response()
}

impl From<GovernorSnapshot> for ResourcesResponse {
    fn from(snapshot: GovernorSnapshot) -> Self {
        Self {
            configured_limits: ConfiguredResourceLimits {
                vram: format_resource_limit(snapshot.configured_limits.vram_limit),
                host_ram: format_resource_limit(snapshot.configured_limits.host_ram_limit),
                disk_spill: snapshot
                    .configured_limits
                    .disk_spill_limit
                    .map(format_resource_limit),
            },
            resolved_limits: ResolvedResourceLimits {
                vram_bytes: snapshot.resolved_limits.vram_bytes,
                host_ram_bytes: snapshot.resolved_limits.host_ram_bytes,
                disk_spill_bytes: snapshot.resolved_limits.disk_spill_bytes,
            },
            derived_kv_budget: DerivedKvBudget {
                bytes: snapshot.derived_budget.kv_bytes,
                total_pages: snapshot.derived_budget.total_pages,
                max_total_tokens: snapshot.derived_budget.max_total_tokens,
                reserved_bytes: snapshot.derived_budget.reserved_bytes,
            },
            vram: ResourceTier::from(snapshot.vram),
            host_ram: ResourceTier::from(snapshot.host_ram),
            disk_spill: snapshot.disk_spill.map(ResourceTier::from),
        }
    }
}

impl From<onnx_genai::scheduler::TierSnapshot> for ResourceTier {
    fn from(snapshot: onnx_genai::scheduler::TierSnapshot) -> Self {
        Self {
            used: snapshot.used,
            limit: snapshot.limit,
            headroom: snapshot.headroom,
        }
    }
}

fn format_resource_limit(limit: ResourceLimit) -> String {
    match limit {
        ResourceLimit::Bytes(bytes) => bytes.to_string(),
        ResourceLimit::Fraction(fraction) => fraction.to_string(),
        ResourceLimit::Auto => "auto".to_string(),
    }
}

pub(crate) async fn create_session(
    State(state): State<AppState>,
) -> Result<Json<SessionResponse>, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    if handle.pipeline {
        return Err(ApiError::bad_request(
            "sessions are not supported by pipeline models",
        ));
    }
    let client_id = state
        .sessions
        .next_client_id()
        .map_err(|err| ApiError::internal(format!("session id generation failed: {err}")))?;
    let engine_session_id = handle
        .engine
        .create_session()
        .await
        .map_err(|err| ApiError::internal(format!("session create failed: {err}")))?;

    let evicted = state
        .sessions
        .insert(client_id.clone(), engine_session_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    close_evicted_session(&handle.engine, evicted).await?;

    Ok(Json(SessionResponse {
        id: client_id,
        object: "session",
    }))
}

pub(crate) async fn delete_session(
    State(state): State<AppState>,
    AxumPath(client_id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let engine_session_id = state
        .sessions
        .remove(&client_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?
        .ok_or_else(|| ApiError::not_found(format!("session {client_id} not found")))?;

    handle
        .engine
        .close_session(engine_session_id)
        .await
        .map_err(|err| ApiError::internal(format!("session close failed: {err}")))?;

    Ok(StatusCode::NO_CONTENT)
}

pub(crate) async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    if handle.pipeline {
        return Err(ApiError::bad_request(
            "/v1/completions is not supported by pipeline models",
        ));
    }
    validate_completion_request(&request, &state.config)?;
    let session_id = session_id_from_headers(&headers)?;
    if request.suffix.is_some() && handle.fim_config.is_none() {
        return Err(ApiError::bad_request(
            "FIM is not supported by this model because its tokenizer configuration does not declare recognized FIM tokens",
        ));
    }
    if request.suffix.is_some() && session_id.is_some() {
        return Err(ApiError::bad_request(
            "X-Session-Id is not supported for FIM completions",
        ));
    }

    if request.stream {
        Ok(stream_completion(state, handle, request, session_id)
            .await?
            .into_response())
    } else {
        Ok(Json(run_completion(state, handle, request, session_id).await?).into_response())
    }
}

pub(crate) async fn embeddings(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    validate_embedding_request(&request, &handle.tokenizer)?;

    let encoding_format = request.encoding_format;
    let model = request.model.clone();

    let inputs: Vec<Vec<u32>> = match request.input {
        EmbeddingInput::String(text) => {
            let tokens = handle
                .tokenizer
                .encode(&text)
                .map_err(|err| ApiError::internal(format!("input tokenization failed: {err}")))?;
            vec![tokens]
        }
        EmbeddingInput::Strings(texts) => {
            let mut all = Vec::with_capacity(texts.len());
            for text in &texts {
                let tokens = handle.tokenizer.encode(text).map_err(|err| {
                    ApiError::internal(format!("input tokenization failed: {err}"))
                })?;
                all.push(tokens);
            }
            all
        }
        EmbeddingInput::TokenArrays(arrays) => arrays,
    };

    let total_tokens: usize = inputs.iter().map(|ids| ids.len()).sum();

    let mut data = Vec::with_capacity(inputs.len());
    for (index, input_ids) in inputs.into_iter().enumerate() {
        let vector = handle
            .engine
            .embed(input_ids, EmbeddingOptions::default())
            .await
            .map_err(|err| ApiError::internal(format!("embedding failed: {err}")))?;
        data.push(EmbeddingData {
            object: "embedding",
            embedding: EmbeddingVector::from_floats(vector, encoding_format),
            index,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

pub(crate) async fn audio_transcriptions(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut file = None;
    let mut filename = None;
    let mut language = None;
    let mut response_format = "json".to_string();
    let mut model_name = String::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::bad_request(format!("invalid multipart form: {err}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                filename = field.file_name().map(ToString::to_string);
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|err| {
                            ApiError::bad_request(format!("failed to read audio file: {err}"))
                        })?
                        .to_vec(),
                );
            }
            "language" => {
                language = Some(field.text().await.map_err(|err| {
                    ApiError::bad_request(format!("invalid language field: {err}"))
                })?);
            }
            "response_format" => {
                response_format = field.text().await.map_err(|err| {
                    ApiError::bad_request(format!("invalid response_format field: {err}"))
                })?;
            }
            "model" => {
                model_name = field
                    .text()
                    .await
                    .map_err(|err| ApiError::bad_request(format!("invalid model field: {err}")))?;
            }
            _ => {}
        }
    }

    let handle = resolve_model(&state.registry, &model_name).await?;

    let bytes = file.ok_or_else(|| ApiError::bad_request("multipart field 'file' is required"))?;
    if !matches!(response_format.as_str(), "json" | "text") {
        return Err(ApiError::bad_request(format!(
            "unsupported response_format '{response_format}'; expected 'json' or 'text'"
        )));
    }
    if filename
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(".mp3"))
    {
        return Err(ApiError::bad_request(
            "MP3 audio is not supported yet; provide a PCM16 WAV file",
        ));
    }
    let spec = handle
        .audio_input
        .as_ref()
        .ok_or_else(|| ApiError::bad_request("this model does not support audio transcription"))?;
    let input = crate::audio_input::preprocess_wav(&bytes, spec)
        .map_err(|err| ApiError::bad_request(format!("invalid audio input: {err}")))?;
    let max_tokens = spec
        .max_tokens
        .unwrap_or(state.config.max_output_tokens)
        .min(state.config.max_output_tokens);
    let token_ids = audio_decoder_prompt(&handle.tokenizer, language.as_deref())?;
    let prompt_tokens = token_ids.len();
    let request = GenerateRequest {
        prompt: GeneratePrompt::TokenIds(token_ids),
        options: GenerateOptions {
            max_new_tokens: max_tokens,
            temperature: 0.0,
            max_context: handle.model_max_context,
            ..GenerateOptions::default()
        },
    };
    let result = collect_generation_result(
        handle
            .engine
            .generate_pipeline(
                request,
                Some(PipelineInputBundle {
                    tensors: vec![PipelineTensor::Fp32(PipelineInputTensor {
                        endpoint: input.endpoint,
                        data: input.data,
                        shape: input.shape,
                        num_tiles: None,
                    })],
                    image_summaries: Vec::new(),
                }),
            )
            .await
            .map_err(map_generate_submit_error)?,
    )
    .await
    .map_err(|err| ApiError::internal(format!("transcription failed: {err}")))?;
    crate::metrics::add_prompt_tokens(prompt_tokens);

    match response_format.as_str() {
        "json" => Ok(Json(AudioTranscriptionResponse { text: result.text }).into_response()),
        "text" => Ok((
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            result.text,
        )
            .into_response()),
        _ => unreachable!("response format validated before generation"),
    }
}

fn validate_embedding_request(
    request: &EmbeddingRequest,
    tokenizer: &Tokenizer,
) -> Result<(), ApiError> {
    if request.dimensions == Some(0) {
        return Err(ApiError::bad_request(
            "dimensions must be greater than zero",
        ));
    }

    let validate_tokens = |tokens: &[u32]| {
        if tokens.is_empty() {
            Err(ApiError::bad_request(
                "embedding input must contain at least one token",
            ))
        } else {
            Ok(())
        }
    };
    match &request.input {
        EmbeddingInput::String(input) => {
            let tokens = tokenizer.encode(input).map_err(|err| {
                ApiError::bad_request(format!("input tokenization failed: {err}"))
            })?;
            validate_tokens(&tokens)
        }
        EmbeddingInput::Strings(inputs) => {
            if inputs.is_empty() {
                return Err(ApiError::bad_request(
                    "embedding input array must not be empty",
                ));
            }
            for input in inputs {
                let tokens = tokenizer.encode(input).map_err(|err| {
                    ApiError::bad_request(format!("input tokenization failed: {err}"))
                })?;
                validate_tokens(&tokens)?;
            }
            Ok(())
        }
        EmbeddingInput::TokenArrays(inputs) => {
            if inputs.is_empty() {
                return Err(ApiError::bad_request(
                    "embedding input array must not be empty",
                ));
            }
            for tokens in inputs {
                validate_tokens(tokens)?;
            }
            Ok(())
        }
    }
}

async fn run_completion(
    state: AppState,
    handle: Arc<ModelHandle>,
    request: CompletionRequest,
    client_session_id: Option<String>,
) -> Result<CompletionResponse, ApiError> {
    let id = text_completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let requested_logprobs = request.logprobs;
    let tokenizer = handle.tokenizer.clone();
    let prepared = prepare_completion(&request, &handle)?;
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        handle.model_max_context,
    )?;
    let result = collect_generation_result(
        submit_completion(
            &handle,
            &state.sessions,
            prepared.generation,
            client_session_id.as_deref(),
        )
        .await?,
    )
    .await
    .map_err(|err| ApiError::internal(format!("generation failed: {err}")))?;
    crate::metrics::add_prompt_tokens(prepared.prompt_tokens);
    let completion_tokens = result.token_ids.len();
    let logprobs = completion_logprobs(&result, &tokenizer, requested_logprobs)
        .map_err(|err| ApiError::internal(format!("logprobs conversion failed: {err}")))?;

    Ok(CompletionResponse {
        id,
        object: "text_completion",
        created,
        model,
        choices: vec![CompletionChoice {
            text: result.text,
            index: 0,
            finish_reason: finish_reason_label(&result.finish_reason),
            logprobs,
        }],
        usage: Usage {
            prompt_tokens: prepared.prompt_tokens,
            completion_tokens,
            total_tokens: prepared.prompt_tokens + completion_tokens,
        },
    })
}

async fn stream_completion(
    state: AppState,
    handle: Arc<ModelHandle>,
    request: CompletionRequest,
    client_session_id: Option<String>,
) -> Result<Sse<ReceiverStream<Result<Event, Infallible>>>, ApiError> {
    let id = text_completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let requested_logprobs = request.logprobs;
    let tokenizer = handle.tokenizer.clone();
    let user_stop_sequences = request
        .stop
        .clone()
        .map(StopInput::into_texts)
        .unwrap_or_default();
    let prepared = prepare_completion(&request, &handle)?;
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        handle.model_max_context,
    )?;
    let mut driver_rx = submit_completion(
        &handle,
        &state.sessions,
        prepared.generation,
        client_session_id.as_deref(),
    )
    .await?;
    crate::metrics::add_prompt_tokens(prepared.prompt_tokens);
    let (tx, rx) = mpsc::channel(16);

    tokio::spawn(async move {
        let mut stop_buffer = StopBoundaryBuffer::new(user_stop_sequences.clone());
        let mut emitted_text = false;
        let result = loop {
            match driver_rx.recv().await {
                Some(DriverEvent::Token(token)) => {
                    if requested_logprobs.is_some() {
                        continue;
                    }
                    let finish_reason = token.finish_reason.clone();
                    let text = stop_buffer.push(&token.text);
                    if !text.is_empty() {
                        emitted_text = true;
                        send_completion_stream_chunk(
                            &tx,
                            completion_chunk(&id, created, &model, text, None),
                        )
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
                if let Some(requested_logprobs) = requested_logprobs {
                    send_completion_logprob_chunks(
                        &tx,
                        (&id, created, &model),
                        &result,
                        &tokenizer,
                        requested_logprobs,
                        &user_stop_sequences,
                    )
                    .await?;
                } else if !emitted_text && !result.text.is_empty() {
                    send_completion_stream_chunk(
                        &tx,
                        completion_chunk(&id, created, &model, result.text, None),
                    )
                    .await?;
                } else if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                    let text = stop_buffer.flush();
                    if !text.is_empty() {
                        send_completion_stream_chunk(
                            &tx,
                            completion_chunk(&id, created, &model, text, None),
                        )
                        .await?;
                    }
                }
                send_completion_stream_chunk(
                    &tx,
                    completion_done_chunk(
                        &id,
                        created,
                        &model,
                        finish_reason_label(&result.finish_reason),
                    ),
                )
                .await?;
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

pub(crate) async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    validate_request(&request, &state.config)?;
    let session_id = session_id_from_headers(&headers)?;
    if handle.pipeline && session_id.is_some() {
        return Err(ApiError::bad_request(
            "X-Session-Id is not supported by pipeline models",
        ));
    }
    let image_urls = request.image_urls();
    let input_audio = request.input_audio();
    if !image_urls.is_empty() && !input_audio.is_empty() {
        return Err(ApiError::bad_request(
            "image and audio inputs cannot be combined in one request",
        ));
    }
    if input_audio.len() > 1 {
        return Err(ApiError::bad_request(
            "only one input_audio content part is supported per request",
        ));
    }
    if !image_urls.is_empty() && !handle.pipeline {
        return Err(ApiError::bad_request(
            "this model does not support image input",
        ));
    }
    if !image_urls.is_empty() && handle.vision_input.is_none() {
        return Err(ApiError::bad_request(
            "this pipeline model does not support image input",
        ));
    }
    if !input_audio.is_empty() && !handle.pipeline {
        return Err(ApiError::bad_request(
            "this model does not support audio input",
        ));
    }
    if !input_audio.is_empty() && handle.audio_input.is_none() {
        return Err(ApiError::bad_request(
            "this pipeline model does not support audio input",
        ));
    }
    if request.stream {
        Ok(
            stream_chat_completion(state, handle, request, session_id, image_urls, input_audio)
                .await?
                .into_response(),
        )
    } else {
        let response =
            run_chat_completion(state, handle, request, session_id, image_urls, input_audio)
                .await?;
        Ok(Json(response).into_response())
    }
}

async fn run_chat_completion(
    state: AppState,
    handle: Arc<ModelHandle>,
    request: ChatCompletionRequest,
    client_session_id: Option<String>,
    image_urls: Vec<String>,
    input_audio: Vec<InputAudio>,
) -> Result<ChatCompletionResponse, ApiError> {
    let id = completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let requested_top_logprobs = request
        .logprobs
        .then_some(request.top_logprobs.unwrap_or(0));
    let tokenizer = handle.tokenizer.clone();
    let mut prepared = prepare_generate_request(
        &request,
        &handle.tokenizer,
        handle.chat_template.as_deref(),
        client_session_id.is_some(),
    )
    .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
    if !input_audio.is_empty() {
        prepared = prepare_audio_generate_request(&request, &handle.tokenizer)?;
    }
    let pipeline_input = if !image_urls.is_empty() {
        Some(
            preprocess_chat_images(
                &image_urls,
                handle
                    .vision_input
                    .as_ref()
                    .expect("vision input checked before generation"),
                &mut prepared,
            )
            .await?,
        )
    } else if let Some(audio) = input_audio.first() {
        Some(preprocess_chat_audio(audio, &handle)?)
    } else {
        None
    };
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        handle.model_max_context,
    )?;
    let prompt_tokens = prepared.prompt_tokens;
    let mut generation_request = prepared.request;
    generation_request.options.max_context = handle.model_max_context;
    let session_lookup = if let Some(id) = client_session_id.as_deref() {
        Some(get_or_create_session(&handle.engine, &state.sessions, id).await?)
    } else {
        None
    };

    let session_for_count = session_lookup;
    let wants_json_object = request.wants_json_object();
    let result = collect_generation_result(if handle.pipeline {
        handle
            .engine
            .generate_pipeline(generation_request, pipeline_input)
            .await
            .map_err(map_generate_submit_error)?
    } else {
        handle
            .engine
            .generate(session_lookup, generation_request)
            .await
            .map_err(map_generate_submit_error)?
    })
    .await
    .map_err(|err| ApiError::internal(format!("generation failed: {err}")));
    crate::metrics::add_prompt_tokens(prompt_tokens);

    let session_token_count = if let Some(engine_session_id) = session_for_count {
        Some(
            handle
                .engine
                .session_token_count(engine_session_id)
                .await
                .map_err(|err| ApiError::internal(format!("session token count failed: {err}")))?,
        )
    } else {
        None
    };

    let (content, tool_calls, completion_tokens, finish_reason, logprobs) = match result {
        Ok(result) => {
            let default_finish_reason = finish_reason_label(&result.finish_reason);
            let logprobs = chat_logprobs(&result, &tokenizer, requested_top_logprobs)
                .map_err(|err| ApiError::internal(format!("logprobs conversion failed: {err}")))?;
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
                logprobs,
            )
        }
        Err(err)
            if wants_json_object && json_constraint_stopped_incomplete_message(&err.message) =>
        {
            (Some("{}".to_string()), None, 0, "stop", None)
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
                content: content.map(ChatMessageContent::Text),
                tool_calls,
                tool_call_id: None,
            },
            finish_reason,
            logprobs,
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
    handle: Arc<ModelHandle>,
    request: ChatCompletionRequest,
    client_session_id: Option<String>,
    image_urls: Vec<String>,
    input_audio: Vec<InputAudio>,
) -> Result<Sse<ReceiverStream<Result<Event, Infallible>>>, ApiError> {
    let id = completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let requested_top_logprobs = request
        .logprobs
        .then_some(request.top_logprobs.unwrap_or(0));
    let tokenizer = handle.tokenizer.clone();
    let user_stop_sequences = request
        .stop
        .clone()
        .map(StopInput::into_texts)
        .unwrap_or_default();
    let mut prepared = prepare_generate_request(
        &request,
        &handle.tokenizer,
        handle.chat_template.as_deref(),
        client_session_id.is_some(),
    )
    .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
    if !input_audio.is_empty() {
        prepared = prepare_audio_generate_request(&request, &handle.tokenizer)?;
    }
    let pipeline_input = if !image_urls.is_empty() {
        Some(
            preprocess_chat_images(
                &image_urls,
                handle
                    .vision_input
                    .as_ref()
                    .expect("vision input checked before generation"),
                &mut prepared,
            )
            .await?,
        )
    } else if let Some(audio) = input_audio.first() {
        Some(preprocess_chat_audio(audio, &handle)?)
    } else {
        None
    };
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        handle.model_max_context,
    )?;
    let wants_json_object = request.wants_json_object();
    let mut generation_request = prepared.request;
    generation_request.options.max_context = handle.model_max_context;
    let (tx, rx) = mpsc::channel(16);
    let session_lookup = if let Some(id) = client_session_id.as_deref() {
        Some(get_or_create_session(&handle.engine, &state.sessions, id).await?)
    } else {
        None
    };
    let mut driver_rx = if handle.pipeline {
        handle
            .engine
            .generate_pipeline(generation_request, pipeline_input)
            .await
            .map_err(map_generate_submit_error)?
    } else {
        handle
            .engine
            .generate(session_lookup, generation_request)
            .await
            .map_err(map_generate_submit_error)?
    };
    crate::metrics::add_prompt_tokens(prepared.prompt_tokens);

    tokio::spawn(async move {
        send_stream_chunk(&tx, role_chunk(&id, created, &model)).await?;

        let mut stop_buffer = StopBoundaryBuffer::new(user_stop_sequences.clone());
        let mut buffered_text = String::new();
        let buffer_for_tool_detection =
            request.has_tool_context() && tools_parseable_from_output(&request);
        let result = loop {
            match driver_rx.recv().await {
                Some(DriverEvent::Token(token)) => {
                    if requested_top_logprobs.is_some() {
                        continue;
                    }
                    let finish_reason = token.finish_reason.clone();
                    let content = stop_buffer.push(&token.text);
                    if buffer_for_tool_detection {
                        buffered_text.push_str(&content);
                    } else if !wants_json_object && !content.is_empty() {
                        send_stream_chunk(&tx, content_chunk(&id, created, &model, content, None))
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
                if let Some(requested_top_logprobs) = requested_top_logprobs {
                    let tool_calls = if buffer_for_tool_detection {
                        parse_tool_calls(&result.text)
                    } else {
                        Vec::new()
                    };
                    if tool_calls.is_empty() {
                        send_chat_logprob_chunks(
                            &tx,
                            (&id, created, &model),
                            &result,
                            &tokenizer,
                            requested_top_logprobs,
                            &user_stop_sequences,
                        )
                        .await?;
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
                } else if buffer_for_tool_detection {
                    if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                        buffered_text.push_str(&stop_buffer.flush());
                    }
                    let tool_calls = parse_tool_calls(&buffered_text);
                    if tool_calls.is_empty() {
                        if !buffered_text.is_empty() {
                            send_stream_chunk(
                                &tx,
                                content_chunk(&id, created, &model, buffered_text, None),
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
                        send_stream_chunk(
                            &tx,
                            content_chunk(&id, created, &model, result.text, None),
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
                    if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                        let content = stop_buffer.flush();
                        if !content.is_empty() {
                            send_stream_chunk(
                                &tx,
                                content_chunk(&id, created, &model, content, None),
                            )
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
                send_stream_chunk(
                    &tx,
                    content_chunk(&id, created, &model, "{}".to_string(), None),
                )
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

pub(crate) async fn collect_generation_result(
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

fn preprocess_chat_audio(
    input: &InputAudio,
    handle: &ModelHandle,
) -> Result<PipelineInputBundle, ApiError> {
    let bytes = crate::audio_input::decode_chat_audio(input)
        .map_err(|err| ApiError::bad_request(format!("invalid audio input: {err}")))?;
    let spec = handle
        .audio_input
        .as_ref()
        .expect("audio input checked before generation");
    let input = crate::audio_input::preprocess_wav(&bytes, spec)
        .map_err(|err| ApiError::bad_request(format!("invalid audio input: {err}")))?;
    Ok(PipelineInputBundle {
        tensors: vec![PipelineTensor::Fp32(PipelineInputTensor {
            endpoint: input.endpoint,
            shape: input.shape,
            data: input.data,
            num_tiles: None,
        })],
        image_summaries: Vec::new(),
    })
}

async fn preprocess_chat_images(
    image_urls: &[String],
    spec: &crate::image_input::VisionInputSpec,
    prepared: &mut PreparedGenerateRequest,
) -> Result<PipelineInputBundle, ApiError> {
    let bundle = crate::image_input::load_and_preprocess(image_urls, spec)
        .await
        .map_err(|err| ApiError::bad_request(format!("invalid image input: {err:#}")))?;
    let token_ids = match &prepared.request.prompt {
        GeneratePrompt::TokenIds(token_ids) => token_ids,
        GeneratePrompt::Text(_) => {
            return Err(ApiError::internal(
                "What: image placeholder expansion received an untokenized prompt. Why: the server preprocessing order was violated. How: tokenize before preprocessing and expansion.",
            ));
        }
    };
    let expanded = spec
        .expand_prompt(token_ids, &bundle)
        .map_err(|err| ApiError::bad_request(format!("invalid image input: {err:#}")))?;
    prepared.prompt_tokens = expanded.len();
    prepared.request.prompt = GeneratePrompt::TokenIds(expanded);
    Ok(PipelineInputBundle {
        tensors: bundle
            .tensors
            .into_iter()
            .map(|tensor| PipelineTensor::Typed {
                endpoint: tensor.endpoint,
                expected_dtype: tensor.expected_dtype,
                expected_shape: tensor.expected_shape,
                shape: tensor.shape,
                data: tensor.data,
            })
            .collect(),
        image_summaries: bundle.images,
    })
}

fn prepare_audio_generate_request(
    request: &ChatCompletionRequest,
    tokenizer: &Tokenizer,
) -> Result<PreparedGenerateRequest, ApiError> {
    let token_ids = audio_decoder_prompt(tokenizer, None)?;
    let prompt_tokens = token_ids.len();
    Ok(PreparedGenerateRequest {
        request: GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options: build_generate_options_with_tokenizer(request, tokenizer),
        },
        prompt_tokens,
    })
}

fn audio_decoder_prompt(
    tokenizer: &Tokenizer,
    language: Option<&str>,
) -> Result<Vec<u32>, ApiError> {
    let mut token_ids = vec![
        tokenizer
            .token_id("<|startoftranscript|>")
            .or_else(|| tokenizer.eos_token_id())
            .unwrap_or(0),
    ];
    if let Some(language) = language.filter(|value| !value.is_empty()) {
        let token = format!("<|{}|>", language.to_ascii_lowercase());
        token_ids.push(tokenizer.token_id(&token).ok_or_else(|| {
            ApiError::bad_request(format!(
                "language '{language}' is not supported by this model tokenizer"
            ))
        })?);
    }
    for token in ["<|transcribe|>", "<|notimestamps|>"] {
        if let Some(token_id) = tokenizer.token_id(token) {
            token_ids.push(token_id);
        }
    }
    Ok(token_ids)
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
    if request
        .top_logprobs
        .is_some_and(|count| count > MAX_CHAT_TOP_LOGPROBS)
    {
        return Err(ApiError::bad_request(format!(
            "top_logprobs must be between 0 and {MAX_CHAT_TOP_LOGPROBS}"
        )));
    }
    if request.top_logprobs.is_some() && !request.logprobs {
        return Err(ApiError::bad_request(
            "top_logprobs requires logprobs to be true",
        ));
    }
    validate_tool_choice(request)?;
    Ok(())
}

fn validate_completion_request(
    request: &CompletionRequest,
    config: &ServerConfig,
) -> Result<(), ApiError> {
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
    if !request.min_p.is_finite() || !(0.0..=1.0).contains(&request.min_p) {
        return Err(ApiError::bad_request(
            "min_p must be finite and between 0 and 1",
        ));
    }
    if !request.frequency_penalty.is_finite() {
        return Err(ApiError::bad_request("frequency_penalty must be finite"));
    }
    if !request.presence_penalty.is_finite() {
        return Err(ApiError::bad_request("presence_penalty must be finite"));
    }
    if request
        .logprobs
        .is_some_and(|count| count > MAX_COMPLETION_LOGPROBS)
    {
        return Err(ApiError::bad_request(format!(
            "logprobs must be between 0 and {MAX_COMPLETION_LOGPROBS}"
        )));
    }
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
        .ok_or_else(|| {
            ApiError::bad_request(
                "What: request admission length overflowed. Why: final prefill length plus max_tokens does not fit usize. How: reduce the prompt, image expansion size, or max_tokens.",
            )
        })?;
    if total > model_max_context {
        return Err(ApiError::bad_request(format!(
            "What: request admission exceeded the model context limit. \
             Why: final prefill length ({prompt_tokens}) after placeholder expansion plus max_tokens ({max_tokens}) is {total}, above {model_max_context}. \
             How: reduce the prompt, image count/expansion size, or max_tokens."
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

async fn get_or_create_session(
    engine: &EngineDriver,
    sessions: &SessionRegistry,
    client_id: &str,
) -> Result<SessionId, ApiError> {
    if let Some(engine_session_id) = sessions
        .get(client_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?
    {
        return Ok(engine_session_id);
    }

    let engine_session_id = engine
        .create_session()
        .await
        .map_err(|err| ApiError::internal(format!("session create failed: {err}")))?;
    let evicted = sessions
        .insert(client_id.to_string(), engine_session_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    close_evicted_session(engine, evicted).await?;
    Ok(engine_session_id)
}

async fn close_evicted_session(
    engine: &EngineDriver,
    evicted: Option<SessionId>,
) -> Result<(), ApiError> {
    if let Some(evicted) = evicted {
        engine
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

pub(crate) fn prepare_completion(
    request: &CompletionRequest,
    handle: &ModelHandle,
) -> Result<PreparedCompletion, ApiError> {
    let mut options = build_completion_options(request, &handle.tokenizer);
    options.max_context = handle.model_max_context;
    if let Some(suffix) = request.suffix.as_ref() {
        let fim_config = handle
            .fim_config
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("FIM is not supported by this model"))?;
        let prompt = fim_config.format_prompt(&request.prompt, suffix);
        let prompt_tokens = tokenize_prompt(&handle.tokenizer, &prompt)?;
        Ok(PreparedCompletion {
            generation: CompletionGeneration::Fim {
                prefix: request.prompt.clone(),
                suffix: suffix.clone(),
                options,
            },
            prompt_tokens,
        })
    } else {
        let token_ids = handle
            .tokenizer
            .encode(&request.prompt)
            .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
        let prompt_tokens = token_ids.len();
        Ok(PreparedCompletion {
            generation: CompletionGeneration::Plain(GenerateRequest {
                prompt: GeneratePrompt::TokenIds(token_ids),
                options,
            }),
            prompt_tokens,
        })
    }
}

async fn submit_completion(
    handle: &ModelHandle,
    sessions: &SessionRegistry,
    generation: CompletionGeneration,
    client_session_id: Option<&str>,
) -> Result<mpsc::Receiver<DriverEvent>, ApiError> {
    match generation {
        CompletionGeneration::Plain(request) => {
            let session_id = if let Some(id) = client_session_id {
                Some(get_or_create_session(&handle.engine, sessions, id).await?)
            } else {
                None
            };
            handle
                .engine
                .generate(session_id, request)
                .await
                .map_err(map_generate_submit_error)
        }
        CompletionGeneration::Fim {
            prefix,
            suffix,
            options,
        } => {
            let fim_config = handle
                .fim_config
                .clone()
                .ok_or_else(|| ApiError::bad_request("FIM is not supported by this model"))?;
            handle
                .engine
                .generate_fim(prefix, suffix, fim_config, options)
                .await
                .map_err(map_generate_submit_error)
        }
    }
}

fn tokenize_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<usize, ApiError> {
    tokenizer
        .encode(prompt)
        .map(|tokens| tokens.len())
        .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))
}

fn build_completion_options(request: &CompletionRequest, tokenizer: &Tokenizer) -> GenerateOptions {
    let mut options = GenerateOptions {
        max_new_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        min_p: request.min_p,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        top_logprobs: request.logprobs,
        ..GenerateOptions::default()
    };
    if let Some(stop) = request.stop.clone() {
        options.stop_sequences = stop.into_sequences();
    }
    add_tokenizer_stop_sequences(&mut options, tokenizer);
    options
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
        top_logprobs: request
            .logprobs
            .then_some(request.top_logprobs.unwrap_or(0)),
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
    add_tokenizer_stop_sequences(&mut options, tokenizer);
    options
}

fn add_tokenizer_stop_sequences(options: &mut GenerateOptions, tokenizer: &Tokenizer) {
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
        .and_then(|message| message.content.as_ref())
        .map(ChatMessageContent::text)
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
                    message
                        .content
                        .as_ref()
                        .map(ChatMessageContent::text)
                        .unwrap_or_default(),
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
            prompt.push_str(&content.text());
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

fn chat_logprobs(
    result: &GenerateResult,
    tokenizer: &Tokenizer,
    requested_top_logprobs: Option<usize>,
) -> anyhow::Result<Option<ChatLogprobs>> {
    let Some(requested_top_logprobs) = requested_top_logprobs else {
        return Ok(None);
    };
    let logprobs = result
        .logprobs
        .as_deref()
        .context("engine did not return requested logprobs")?;
    if logprobs.len() != result.token_ids.len() {
        anyhow::bail!(
            "engine returned {} logprob records for {} generated tokens",
            logprobs.len(),
            result.token_ids.len()
        );
    }
    let content = logprobs
        .iter()
        .map(|entry| chat_token_logprob(tokenizer, entry, requested_top_logprobs))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Some(ChatLogprobs { content }))
}

fn chat_token_logprob(
    tokenizer: &Tokenizer,
    entry: &TokenLogprob,
    requested_top_logprobs: usize,
) -> anyhow::Result<ChatTokenLogprob> {
    let token = decode_logprob_token(tokenizer, entry.token_id)?;
    let top_logprobs = entry
        .top
        .iter()
        .take(requested_top_logprobs)
        .map(|&(token_id, logprob)| {
            let token = decode_logprob_token(tokenizer, token_id)?;
            Ok(ChatTopLogprob {
                bytes: token.as_bytes().to_vec(),
                token,
                logprob,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(ChatTokenLogprob {
        bytes: token.as_bytes().to_vec(),
        token,
        logprob: entry.logprob,
        top_logprobs,
    })
}

fn completion_logprobs(
    result: &GenerateResult,
    tokenizer: &Tokenizer,
    requested_top_logprobs: Option<usize>,
) -> anyhow::Result<Option<CompletionLogprobs>> {
    let Some(requested_top_logprobs) = requested_top_logprobs else {
        return Ok(None);
    };
    let logprobs = result
        .logprobs
        .as_deref()
        .context("engine did not return requested logprobs")?;
    if logprobs.len() != result.token_ids.len() {
        anyhow::bail!(
            "engine returned {} logprob records for {} generated tokens",
            logprobs.len(),
            result.token_ids.len()
        );
    }

    let mut tokens = Vec::with_capacity(logprobs.len());
    let mut token_logprobs = Vec::with_capacity(logprobs.len());
    let mut top_logprobs = Vec::with_capacity(logprobs.len());
    let mut text_offset = Vec::with_capacity(logprobs.len());
    let mut offset = 0;
    for entry in logprobs {
        let token = decode_logprob_token(tokenizer, entry.token_id)?;
        text_offset.push(offset);
        offset += token.len();
        tokens.push(token);
        token_logprobs.push(entry.logprob);
        top_logprobs.push(
            entry
                .top
                .iter()
                .take(requested_top_logprobs)
                .map(|&(token_id, logprob)| {
                    Ok((decode_logprob_token(tokenizer, token_id)?, logprob))
                })
                .collect::<anyhow::Result<_>>()?,
        );
    }
    Ok(Some(CompletionLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }))
}

fn decode_logprob_token(tokenizer: &Tokenizer, token_id: u32) -> anyhow::Result<String> {
    let decoded = tokenizer
        .decode(&[token_id])
        .with_context(|| format!("failed to decode token id {token_id}"))?;
    if !decoded.is_empty() {
        return Ok(decoded);
    }
    tokenizer
        .inner()
        .id_to_token(token_id)
        .with_context(|| format!("token id {token_id} is not in the tokenizer vocabulary"))
}

async fn send_completion_logprob_chunks(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    response: (&str, u64, &str),
    result: &GenerateResult,
    tokenizer: &Tokenizer,
    requested_top_logprobs: usize,
    stop_sequences: &[String],
) -> anyhow::Result<()> {
    let (id, created, model) = response;
    let logprobs = completion_logprobs(result, tokenizer, Some(requested_top_logprobs))?
        .context("requested completion logprobs were not built")?;
    let stream_text = result
        .token_ids
        .iter()
        .map(|&token_id| tokenizer.decode(&[token_id]).map_err(anyhow::Error::from))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let visible_text = truncate_tokens_at_stop(&stream_text, stop_sequences);
    for (index, text) in visible_text.into_iter().enumerate() {
        if text.is_empty() {
            continue;
        }
        send_completion_stream_chunk(
            tx,
            completion_chunk(
                id,
                created,
                model,
                text,
                Some(CompletionLogprobs {
                    tokens: vec![logprobs.tokens[index].clone()],
                    token_logprobs: vec![logprobs.token_logprobs[index]],
                    top_logprobs: vec![logprobs.top_logprobs[index].clone()],
                    text_offset: vec![logprobs.text_offset[index]],
                }),
            ),
        )
        .await?;
    }
    Ok(())
}

async fn send_chat_logprob_chunks(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    response: (&str, u64, &str),
    result: &GenerateResult,
    tokenizer: &Tokenizer,
    requested_top_logprobs: usize,
    stop_sequences: &[String],
) -> anyhow::Result<()> {
    let (id, created, model) = response;
    let logprobs = chat_logprobs(result, tokenizer, Some(requested_top_logprobs))?
        .context("requested chat logprobs were not built")?;
    let stream_text = result
        .token_ids
        .iter()
        .map(|&token_id| tokenizer.decode(&[token_id]).map_err(anyhow::Error::from))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let visible_text = truncate_tokens_at_stop(&stream_text, stop_sequences);
    for (index, content) in visible_text.into_iter().enumerate() {
        if content.is_empty() {
            continue;
        }
        send_stream_chunk(
            tx,
            content_chunk(
                id,
                created,
                model,
                content,
                Some(ChatLogprobs {
                    content: vec![logprobs.content[index].clone()],
                }),
            ),
        )
        .await?;
    }
    Ok(())
}

fn truncate_tokens_at_stop(tokens: &[String], stop_sequences: &[String]) -> Vec<String> {
    let text = tokens.concat();
    let cutoff = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
        .unwrap_or(text.len());
    let mut cursor = 0;
    let mut visible = Vec::new();
    for token in tokens {
        if cursor >= cutoff {
            break;
        }
        let mut end = (cutoff - cursor).min(token.len());
        while !token.is_char_boundary(end) {
            end -= 1;
        }
        visible.push(token[..end].to_string());
        cursor += token.len();
    }
    visible
}

fn finish_reason_label(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::MaxTokens | FinishReason::Length => "length",
        FinishReason::EosToken | FinishReason::StopSequence { .. } => "stop",
    }
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

fn text_completion_id() -> String {
    format!("cmpl-{}", now_unix())
}
