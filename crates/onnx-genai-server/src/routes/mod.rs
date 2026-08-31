use std::{
    convert::Infallible,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use axum::{
    Json,
    extract::{FromRequest, Multipart, Path as AxumPath, Request, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response, Sse, sse::Event},
};
use onnx_genai::{FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult};
use onnx_genai_engine::{
    DryConfig, EmbeddingOptions, EngineGovernorError, GenerateConstraint, GovernorSnapshot,
    MirostatConfig, MirostatVersion, PackageExecutionError, ResourceLimit, SamplingOverrides,
    TokenLogprob, XtcConfig, parse_resource_limit,
};
use onnx_genai_metadata::GenerationDefaults;
use onnx_genai_ort::{ChatMessage as TemplateChatMessage, ChatTemplate, Tokenizer};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    driver::{
        DriverEvent, DriverFailure, DriverFailureKind, DriverGeneration, GenerateSubmitError,
    },
    lease::{ModelSessionPlacement, SessionLeaseGuard},
    multimodal::MultimodalInput,
    registry::{ModelHandle, ModelRegistry},
    session::{SessionCloseError, SessionRegistry, SessionRegistryError},
    sse::{
        StopBoundaryBuffer, completion_chunk, completion_done_chunk, content_chunk, done_chunk,
        reasoning_chunk, role_chunk, send_completion_stream_chunk, send_stream_chunk,
        send_workflow_publications, tool_call_delta_chunks,
    },
    state::{AppState, DEFAULT_MAX_OUTPUT_TOKENS, ServerConfig},
    types::{
        AudioTranscriptionResponse, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
        ChatLogprobs, ChatMessage, ChatMessageContent, ChatMessageToolCall, ChatTokenLogprob,
        ChatTopLogprob, CompletionChoice, CompletionLogprobs, CompletionRequest,
        CompletionResponse, EmbeddingData, EmbeddingInput, EmbeddingRequest, EmbeddingResponse,
        EmbeddingUsage, EmbeddingVector, InputAudio, ReasoningEffort, ResponseFormat, StopInput,
        ToolChoice, ToolChoiceMode, Usage,
    },
};

mod admin;
mod completions;
mod images;
mod multimodal;
mod sessions;
mod speech;

#[cfg(feature = "metrics")]
pub(crate) use admin::prometheus_metrics;
pub(crate) use admin::{
    admin_list_models, admin_load_model, admin_set_vram_limit, admin_unload_model,
    admin_warmup_model, debug_config, debug_kv, debug_profile, debug_sessions, debug_trace,
    debug_trace_perfetto, health, models, resources, status,
};
#[cfg(test)]
pub(crate) use completions::prepare_completion;
pub use completions::{build_generate_request, build_generate_request_with_protocol, build_prompt};
pub(crate) use completions::{
    chat_completions, collect_generation_result, completions, embeddings,
};
pub(crate) use images::{
    a1111_img2img, a1111_models, a1111_options, a1111_samplers, a1111_txt2img, openai_images,
};
pub(crate) use multimodal::audio_transcriptions;
pub(crate) use sessions::{create_session, delete_session, fork_session};
pub(crate) use speech::audio_speech;

const SESSION_ID_HEADER: &str = "x-session-id";
const MAX_SESSION_ID_LEN: usize = 128;
const OVERLOAD_RETRY_AFTER_SECS: u64 = 1;
const MEMORY_OVERLOAD_MESSAGE: &str =
    "request exceeds the configured KV memory limit; retry later or reduce context/output length";
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
    workers: Vec<WorkerStatusInfo>,
}

/// Per-session detail entry in [`NodeStatus::sessions`] (§34.8).
#[derive(Debug, Serialize)]
pub(crate) struct SessionStatus {
    id: String,
    priority: String,
    kv_pages: u32,
    state: String,
}

/// Per-model worker occupancy and health. Worker ids are local to a model, so
/// the model id is part of every observable identity.
#[derive(Debug, Serialize)]
pub(crate) struct WorkerStatusInfo {
    model_id: String,
    worker_id: usize,
    active_turns: usize,
    live_sessions: usize,
    health: &'static str,
    kv_pages_in_use: usize,
    kv_pages_total: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct DebugConfigResponse {
    model_id: String,
    /// How many components the package's serialized `pipeline.workflow`
    /// declares.
    ///
    /// Every loaded package serializes one, so this is a fact about the file on
    /// disk rather than a report of which executor the runtime chose. An
    /// operator debugging a package wants to know what it *says*; what runs it
    /// is visible in the execution-provider and island diagnostics beside it.
    workflow_components: usize,
    /// How many of those components name an ONNX graph, as opposed to a step
    /// the runtime implements.
    workflow_graph_components: usize,
    /// Whether that workflow declares a generation loop.
    workflow_declares_generation_loop: bool,
    max_output_tokens: usize,
    max_sessions: usize,
    max_queue_depth: usize,
    ort_session_workers: usize,
    model_max_context: Option<usize>,
    workers: Vec<WorkerStatusInfo>,
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
    /// Whether the decode path can advance more than one sequence per step.
    /// `active_batch_size` counts admitted generations, so it can be > 1 even
    /// when `batch_supported` is false and nothing is actually co-decoded — this
    /// pair is what makes the difference observable (issue #750).
    batch_supported: bool,
    /// The batch width that actually takes effect after clamping the requested
    /// `--max-batch` to what the decode path can honor.
    effective_max_batch: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct ResourcesResponse {
    configured_limits: ConfiguredResourceLimits,
    resolved_limits: ResolvedResourceLimits,
    derived_kv_budget: DerivedKvBudget,
    vram: ResourceTier,
    host_ram: ResourceTier,
    disk_spill: Option<ResourceTier>,
    /// Honest, decode-path-sourced batching capability for this model. Lets an
    /// operator read `supported=false` / `effective_max_batch=1` directly rather
    /// than inferring it from a debug-level log line (issue #750). `None` only
    /// when the response is built without a resolved engine handle.
    #[serde(skip_serializing_if = "Option::is_none")]
    batching: Option<BatchingInfo>,
    /// Resolved memory strategy for this model: the chosen strategy, whether
    /// weight streaming/offload is active, whether the managed no-spill VMM path
    /// is the allocator, and the resolved device budget. Makes the #755 managed
    /// VMM default observable rather than implicit. `None` when built without a
    /// resolved engine handle.
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_strategy: Option<MemoryStrategyInfo>,
}

/// Operator-facing memory strategy report for `/v1/resources` (issue #755).
#[derive(Debug, Serialize)]
pub(crate) struct MemoryStrategyInfo {
    /// Effective strategy the runtime applied (e.g. `FullResident`,
    /// `DynamicWeightResidency`, `MoeRoutingAware`, `Compatibility`).
    pub(crate) strategy: String,
    /// Whether weight streaming/offload is active for this load.
    pub(crate) weight_offload_enabled: bool,
    /// Whether the managed no-spill VMM path (authority-scoped physical-handle
    /// pool, committed-granule admission, no WDDM shared-memory spill) is the
    /// allocator. `true` by default on native CUDA since #755.
    pub(crate) managed_no_spill: bool,
    /// Whether offload was auto-enabled by inference (model exceeds the resolved
    /// device budget) rather than requested by an explicit override.
    pub(crate) auto_enabled: bool,
    /// The resolved device budget in bytes (committed physical bytes cap).
    pub(crate) resolved_device_budget_bytes: Option<u64>,
    /// The managed VMM committed-byte ceiling, when the managed path is active.
    pub(crate) managed_limit_bytes: Option<u64>,
    /// Whether the model weights fit the resolved device budget.
    pub(crate) fits_resolved_device_budget: Option<bool>,
}

/// Operator-facing batching capability report for `/v1/resources`.
#[derive(Debug, Serialize)]
pub(crate) struct BatchingInfo {
    /// Whether the decode path can advance more than one sequence per step.
    pub(crate) supported: bool,
    /// The `--max-batch` width the operator requested (or the server default).
    pub(crate) requested_max_batch: u64,
    /// The width that actually takes effect after clamping to the decode path's
    /// structural limit.
    pub(crate) effective_max_batch: u64,
    /// Human-readable explanation naming the backend / decode path.
    pub(crate) reason: String,
}

#[derive(Debug, Serialize)]
struct ConfiguredResourceLimits {
    vram: String,
    host_ram: String,
    disk_spill: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResolvedResourceLimits {
    vram_bytes: Option<u64>,
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
    /// Where to get stage totals instead of a full timeline.
    aggregate_profile: &'static str,
}

/// Aggregate decode-stage costs for this process.
#[derive(Debug, Serialize)]
pub(crate) struct DebugProfileResponse {
    /// Whether stages are being accumulated at all.
    collecting: bool,
    note: &'static str,
    stages: Vec<ProfileStage>,
    memory_strategy_plans: Vec<ModelMemoryStrategyPlan>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ProfileStage {
    stage: &'static str,
    total_ms: f64,
    calls: u64,
    us_per_call: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelMemoryStrategyPlan {
    model_id: String,
    plan: onnx_genai_engine::MemoryStrategyPlan,
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
pub(crate) struct AdminWarmupResponse {
    id: String,
    warmed: bool,
    duration_ms: u128,
}

#[derive(Debug, Serialize)]
pub(crate) struct SessionResponse {
    id: String,
    object: &'static str,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SessionForkRequest {
    position: usize,
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
    pub(crate) status: StatusCode,
    pub(crate) message: String,
    pub(crate) kind: &'static str,
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
/// `error.type` for a request that conflicts with the state or the capability of
/// what is loaded.
///
/// A 409 is never a fault: nothing broke, and the caller can act on it. Reporting
/// one as `server_error` told a client to retry a crash.
pub(crate) const CONFLICT_ERROR_KIND: &str = "conflict_error";

/// `error.type` for a request the loaded package cannot serve as asked.
pub(crate) const INVALID_REQUEST_ERROR_KIND: &str = "invalid_request_error";

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "server_error",
            retry_after_secs: None,
        }
    }

    /// A request the loaded package cannot serve, and no retry will change.
    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: INVALID_REQUEST_ERROR_KIND,
            retry_after_secs: None,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
            retry_after_secs: None,
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            kind: "unavailable_error",
            retry_after_secs: None,
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
            kind: "server_error",
            retry_after_secs: None,
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: message.into(),
            kind: "server_error",
            retry_after_secs: None,
        }
    }

    /// The request conflicts with the state or the capability of what is
    /// loaded.
    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
            kind: CONFLICT_ERROR_KIND,
            retry_after_secs: None,
        }
    }

    fn too_many_requests(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: message.into(),
            kind: "resource_limit_error",
            retry_after_secs: Some(OVERLOAD_RETRY_AFTER_SECS),
        }
    }

    fn payload_too_large(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: message.into(),
            kind: "invalid_request_error",
            retry_after_secs: None,
        }
    }
}

/// Turn a session-creation failure into the status it actually is.
///
/// A package that declares no conversation is not a server fault and not a
/// malformed request: the caller asked this package for something it does not
/// support, which is what 409 says. Reporting it as a 500 told a client to retry
/// something that will never succeed, and hid a package defect behind an
/// operational one.
pub(crate) fn session_create_failure(error: anyhow::Error) -> ApiError {
    if let Some(unavailable) = error
        .chain()
        .find_map(|source| source.downcast_ref::<crate::worker::WorkerUnavailable>())
    {
        return ApiError::unavailable(unavailable.to_string());
    }
    match onnx_genai_engine::package_execution_error(&error) {
        Some(capability) => package_execution_failure(capability),
        None => ApiError::internal(format!("session create failed: {error}")),
    }
}

pub(crate) fn session_fork_failure(error: anyhow::Error) -> ApiError {
    if let Some(unavailable) = error
        .chain()
        .find_map(|source| source.downcast_ref::<crate::worker::WorkerUnavailable>())
    {
        return ApiError::unavailable(unavailable.to_string());
    }
    let Some(fork) = error
        .chain()
        .find_map(|source| source.downcast_ref::<onnx_genai_engine::SessionForkError>())
    else {
        return ApiError::internal(format!("session fork failed: {error}"));
    };
    match fork {
        onnx_genai_engine::SessionForkError::SessionNotFound { .. } => {
            ApiError::not_found(fork.to_string())
        }
        onnx_genai_engine::SessionForkError::PositionOutOfBounds { .. }
        | onnx_genai_engine::SessionForkError::PositionNotCommitted { .. } => {
            ApiError::invalid_request(fork.to_string())
        }
        onnx_genai_engine::SessionForkError::UnsupportedParticipant { .. }
        | onnx_genai_engine::SessionForkError::SnapshotFailed { .. }
        | onnx_genai_engine::SessionForkError::StalePlan { .. }
        | onnx_genai_engine::SessionForkError::ForeignOrigin { .. }
        | onnx_genai_engine::SessionForkError::StaleOrigin { .. } => {
            ApiError::conflict(fork.to_string())
        }
        onnx_genai_engine::SessionForkError::Publication(_) => ApiError::internal(fork.to_string()),
    }
}

/// The status a package refusal actually is, read off the variant.
///
/// Shared by every layer that can raise one — session creation, generation, and
/// the routing lease — so an `ExclusiveLeaseConflict` answers 409 whether it was
/// decided before the turn was enqueued or inside the pass that ran it. Matching
/// the type rather than the wording is what keeps those answers identical.
pub(crate) fn package_execution_failure(capability: PackageExecutionError) -> ApiError {
    match capability {
        PackageExecutionError::NoSessionState => ApiError::conflict(capability.to_string()),
        PackageExecutionError::DFlashExecutionUnavailable { .. } => {
            ApiError::conflict(capability.to_string())
        }
        PackageExecutionError::CandidateTreeExecutionUnavailable { .. } => {
            ApiError::conflict(capability.to_string())
        }
        PackageExecutionError::CandidateTreeRawWorkflowApi { .. }
        | PackageExecutionError::DFlashRawWorkflowApi { .. } => {
            ApiError::invalid_request(capability.to_string())
        }
        PackageExecutionError::ConversationOverBound { .. } => {
            ApiError::invalid_request(capability.to_string())
        }
        PackageExecutionError::ExclusiveLeaseConflict { .. } => {
            ApiError::conflict(capability.to_string())
        }
    }
}

pub(crate) fn generation_failure(error: DriverFailure) -> ApiError {
    match error.kind {
        DriverFailureKind::MemoryOverload => {
            tracing::warn!(
                error = %error,
                "generation rejected by the KV memory governor"
            );
            ApiError::too_many_requests(MEMORY_OVERLOAD_MESSAGE)
        }
        // The caller asked this package for something it cannot serve as asked.
        // A conversation past its declared bound is a request that is too large
        // and the caller can shorten; a busy session is a conflict that the same
        // request succeeds at once the turn in flight finishes. Both are read
        // off the engine's own type, so neither status depends on wording.
        DriverFailureKind::PackageExecution(capability) => package_execution_failure(capability),
        DriverFailureKind::WorkerUnavailable => ApiError::unavailable(error.message),
        DriverFailureKind::Internal => {
            ApiError::internal(format!("generation failed: {}", error.message))
        }
    }
}

/// JSON body extractor that reports rejections as OpenAI-shaped [`ApiError`]s.
///
/// Axum's stock `Json` rejection returns a 422 whose body is a bare
/// "Failed to deserialize the JSON body into the target type: ..." string. That
/// loses the what/why/how contract every user-facing failure owes the caller,
/// so this wrapper re-frames the rejection — and passes through the detailed
/// messages the multimodal content-part parser already produces.
pub(crate) struct ApiJson<T>(pub(crate) T);

impl<S, T> FromRequest<S> for ApiJson<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(request, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
                Err(ApiError::payload_too_large(rejection.body_text()))
            }
            Err(rejection) => Err(ApiError::bad_request(describe_json_rejection(&rejection))),
        }
    }
}

/// Drop serde's trailing " at line N column M", which points into a body the
/// caller cannot see and only dilutes the actionable message.
pub(crate) fn strip_serde_position(message: &str) -> &str {
    match message.rfind(" at line ") {
        Some(index) if message[index..].ends_with(char::is_numeric) => message[..index].trim_end(),
        _ => message,
    }
}

/// Turn an axum JSON rejection into an actionable message.
fn describe_json_rejection(rejection: &JsonRejection) -> String {
    let text = rejection.body_text();
    // Content-part rejections are already what/why/how; surface them verbatim
    // rather than burying them behind axum's generic prefix.
    if let Some(start) = text.find("What: ") {
        return strip_serde_position(&text[start..]).to_string();
    }
    match rejection {
        JsonRejection::MissingJsonContentType(_) => {
            "What: the request was rejected before parsing. \
             Why: it did not declare `Content-Type: application/json`. \
             How: send the header `Content-Type: application/json` with a JSON body."
                .to_string()
        }
        JsonRejection::JsonSyntaxError(_) => format!(
            "What: the request body could not be parsed as JSON. \
             Why: {text}. \
             How: send a well-formed JSON object."
        ),
        _ => format!(
            "What: the request body did not match this endpoint's schema. \
             Why: {text}. \
             How: check the field names and types against the OpenAI API reference for this endpoint."
        ),
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorResponse {
            error: ErrorBody {
                message: self.message,
                kind: self.kind,
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
        GenerateSubmitError::Failed(error) => generation_failure(error),
    }
}

fn map_registry_error(err: crate::registry::RegistryError) -> ApiError {
    tracing::error!(error = %err, "model registry operation failed");
    ApiError::internal("model registry failed")
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
    if let Some(handle) = registry.resolve(requested).map_err(map_registry_error)? {
        return Ok(handle);
    }
    // Determine the concrete id to lazily load.
    let id = if requested.trim().is_empty() {
        registry
            .default_id()
            .map_err(map_registry_error)?
            .ok_or_else(|| ApiError::internal("no model loaded"))?
    } else {
        requested.to_string()
    };
    if !registry
        .contains_available(&id)
        .map_err(map_registry_error)?
    {
        return Err(ApiError::not_found(format!(
            "model '{requested}' not found"
        )));
    }
    registry
        .load(&id)
        .await
        .map_err(|err| ApiError::internal(format!("failed to load model '{id}': {err}")))
}

fn audio_decoder_prompt(
    tokenizer: &Tokenizer,
    language: Option<&str>,
) -> Result<Vec<u32>, ApiError> {
    crate::multimodal::audio_decoder_prompt(tokenizer, language)
        .map_err(|error| ApiError::bad_request(format!("{error:#}")))
}

/// Close a session the registry let go of, on the model that owns it, under the
/// lease it was handed back holding.
///
/// The engine is resolved from the guard rather than passed in. That is the
/// whole point of the guard carrying a model: the registry spans every loaded
/// model, so the conversation an eviction hands back is routinely *not* the one
/// the evicting request is working on. Closing it on the caller's engine would
/// have destroyed whatever conversation happened to share its placement there —
/// and two engines sharing a placement is the normal case, since every engine
/// numbers its sessions from its own counter.
///
/// The guard is consumed here, so the lease is released exactly when the engine
/// has answered — never earlier, which would let a turn start on a session that
/// is being closed, and never later, which would strand the id.
async fn close_leased_session(
    registry: &ModelRegistry,
    lease: SessionLeaseGuard,
) -> Result<(), ApiError> {
    let model = lease.model().as_str().to_string();
    let Some(handle) = registry.resolve(&model).map_err(map_registry_error)? else {
        // The model was unloaded while this conversation was bound to it. Its
        // engine — and therefore the session — is already gone, so there is
        // nothing left to close; dropping the guard frees the lease.
        tracing::warn!(
            model = %model,
            "session's model is no longer loaded; its engine session went with it",
        );
        return Ok(());
    };
    handle.engine.close_session(lease).await.map_err(|err| {
        err.chain()
            .find_map(|source| source.downcast_ref::<crate::worker::WorkerUnavailable>())
            .map_or_else(
                || ApiError::internal(format!("session close failed: {err}")),
                |unavailable| ApiError::unavailable(unavailable.to_string()),
            )
    })
}

/// Close an evicted binding, if the registry handed one back.
async fn close_evicted_session(
    registry: &ModelRegistry,
    evicted: Option<SessionLeaseGuard>,
) -> Result<(), ApiError> {
    match evicted {
        Some(evicted) => close_leased_session(registry, evicted).await,
        None => Ok(()),
    }
}

/// The status a session-registry refusal actually is.
///
/// A full registry whose every conversation has a turn in flight is a resource
/// limit, not a fault and not a bad request: the same call succeeds as soon as
/// any turn ends, which is exactly what the overload 429 already means here,
/// `Retry-After` included. Reusing that mapping keeps "the server is at
/// capacity" one answer rather than two.
pub(crate) fn session_registry_failure(error: SessionRegistryError) -> ApiError {
    match error {
        SessionRegistryError::AtCapacity { .. } => ApiError::too_many_requests(error.to_string()),
        SessionRegistryError::Poisoned | SessionRegistryError::AlreadyBound => {
            ApiError::internal(format!("session registry failed: {error}"))
        }
    }
}

/// The status a refused close actually is.
///
/// A close that races a live turn is the same conflict an overlapping turn is,
/// and is reported through the same typed mapping so both answer 409 with the
/// same shape.
pub(crate) fn session_close_failure(client_id: &str, error: SessionCloseError) -> ApiError {
    match error {
        SessionCloseError::NotFound => {
            ApiError::not_found(format!("session {client_id} not found"))
        }
        SessionCloseError::Busy(capability) => package_execution_failure(capability),
        SessionCloseError::Registry(error) => session_registry_failure(error),
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod overload_tests {
    use super::*;
    use axum::body::to_bytes;

    #[tokio::test]
    async fn memory_admission_failure_maps_to_stable_overload_response() {
        let response = generation_failure(DriverFailure {
            message: "scheduler admission failed: KV byte budget exhausted".to_string(),
            kind: DriverFailureKind::MemoryOverload,
        })
        .into_response();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "1");
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["type"], "resource_limit_error");
        assert_eq!(body["error"]["message"], MEMORY_OVERLOAD_MESSAGE);
        assert!(!body.to_string().contains("scheduler admission failed"));
    }

    #[tokio::test]
    async fn unreclaimable_mapped_capacity_is_a_pre_header_overload() {
        let error: anyhow::Error = onnx_runtime_memory_governor::MemoryError::CapacityUnavailable {
            tier: "device",
            requested: 4096,
            available: 0,
            role: onnx_runtime_memory_governor::MemoryRole::KvCache,
            detail: "mapped holder could not reach its tentative reclaim target".into(),
            source: None,
        }
        .into();
        let response = generation_failure(DriverFailure::from_engine_error(&error)).into_response();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "1");
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["type"], "resource_limit_error");
    }

    #[test]
    fn unrelated_generation_failure_remains_internal() {
        let error = generation_failure(DriverFailure {
            message: "backend execution failed".to_string(),
            kind: DriverFailureKind::Internal,
        });

        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.kind, "server_error");
    }

    #[test]
    fn failed_worker_maps_to_typed_service_unavailable() {
        let error = generation_failure(DriverFailure {
            message: "engine worker 1 failed; sessions placed on this worker are unavailable"
                .to_string(),
            kind: DriverFailureKind::WorkerUnavailable,
        });

        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.kind, "unavailable_error");
    }
}
