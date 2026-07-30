use std::{
    collections::BTreeMap,
    convert::Infallible,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use axum::{
    Json,
    extract::{
        FromRequest, Multipart, Path as AxumPath, Query, Request, State, rejection::JsonRejection,
    },
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response, Sse, sse::Event},
};
use base64::Engine as _;
use onnx_genai::text_to_audio::TextToAudioRequest;
use onnx_genai::text_to_image::TextToImageRequest;
use onnx_genai::{
    FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult, SessionId,
    StopSequence,
};
use onnx_genai_engine::{
    DryConfig, EmbeddingOptions, EngineGovernorError, GenerateConstraint, GovernorSnapshot,
    MirostatConfig, MirostatVersion, ResourceLimit, SamplingOverrides, TokenLogprob, XtcConfig,
    parse_resource_limit,
};
use onnx_genai_metadata::GenerationDefaults;
use onnx_genai_ort::{ChatMessage as TemplateChatMessage, ChatTemplate, Tokenizer};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    driver::{DriverEvent, EngineDriver, GenerateSubmitError},
    multimodal::MultimodalInput,
    registry::ModelHandle,
    session::SessionRegistry,
    sse::{
        StopBoundaryBuffer, completion_chunk, completion_done_chunk, content_chunk, done_chunk,
        role_chunk, send_completion_stream_chunk, send_stream_chunk, tool_call_delta_chunks,
    },
    state::{AppState, ServerConfig},
    types::{
        AudioTranscriptionResponse, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
        ChatLogprobs, ChatMessage, ChatMessageContent, ChatMessageToolCall,
        ChatMessageToolCallFunction, ChatTokenLogprob, ChatTool, ChatTopLogprob, CompletionChoice,
        CompletionLogprobs, CompletionRequest, CompletionResponse, EmbeddingData, EmbeddingInput,
        EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, EmbeddingVector, ImageData,
        ImageGenerationRequest, ImageGenerationResponse, ImageResponseFormat, InputAudio,
        ResponseFormat, SpeechRequest, SpeechResponseFormat, StopInput, ToolChoice, ToolChoiceMode,
        Usage,
    },
};

pub(crate) mod admin;
mod completions;
mod multimodal;
mod sessions;

#[cfg(feature = "metrics")]
pub(crate) use admin::prometheus_metrics;
pub(crate) use admin::{
    admin_list_models, admin_load_model, admin_set_vram_limit, admin_unload_model,
    admin_warmup_model, debug_config, debug_kv, debug_kv_blocks, debug_profile, debug_sessions,
    debug_trace, debug_trace_perfetto, health, models, resources, status,
};
pub use completions::{
    ParsedAssistantOutput, build_generate_request, build_prompt, parse_assistant_output,
    parse_tool_calls,
};
pub(crate) use completions::{
    chat_completions, collect_generation_result, completions, embeddings,
};
#[cfg(test)]
pub(crate) use completions::{image_placeholder_text, prepare_completion};
pub(crate) use multimodal::{audio_speech, audio_transcriptions, image_generations};
pub(crate) use sessions::{create_session, delete_session};

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
    /// Directory mtime, in epoch seconds.
    ///
    /// Omitted when the directory cannot be stat'd. This previously returned
    /// `now_unix()`, so polling twice reported two different creation times for
    /// the same model -- a fabrication that survived because it is conventional
    /// OpenAI-compat boilerplate that nobody reads.
    #[serde(skip_serializing_if = "Option::is_none")]
    created: Option<u64>,
    owned_by: &'static str,
    /// Whether the model is currently resident, as opposed to configured and
    /// loadable on demand.
    loaded: bool,
    /// Whether this is the model that an empty/omitted `model` field resolves to.
    is_default: bool,
    /// Configured directory. Absolute on loopback; the basename otherwise, so a
    /// non-loopback deployment does not leak the operator's username and
    /// filesystem layout on an endpoint with no authentication in front of it.
    path: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct HealthResponse {
    status: &'static str,
    model: String,
}

/// Why a field is absent from [`NodeStatus`].
///
/// A field is omitted rather than sent as `null` or `0`, and this map says why.
/// The distinction matters: `not-applicable` means the mechanism is not in play
/// on this node's model at all, while `unavailable` means it is in play but not
/// yet measurable. Rendering either as a zero would claim a measurement.
#[derive(Debug, Serialize)]
pub(crate) struct FieldUnavailable {
    /// One of `unavailable` or `not-applicable`.
    pub(crate) code: &'static str,
    /// Human-readable explanation, safe to show in a UI.
    pub(crate) detail: &'static str,
}

impl FieldUnavailable {
    pub(crate) const fn unavailable(detail: &'static str) -> Self {
        Self {
            code: "unavailable",
            detail,
        }
    }

    pub(crate) const fn not_applicable(detail: &'static str) -> Self {
        Self {
            code: "not-applicable",
            detail,
        }
    }
}

/// Node-status contract polled by the cluster router (§34.8) every 1-2s.
///
/// **Unmeasurable fields are omitted, never zeroed.** A zero is a measurement
/// claim: `kv_pages_used: 0` asserts an empty pool, which is a different and
/// stronger statement than "this node cannot tell you". Every omission is
/// explained in [`NodeStatus::unavailable`], so a client can render "not
/// applicable" with a reason instead of drawing a flat line at zero.
///
/// Omission specifically, rather than `null`: the router's deserialization
/// mirror (`onnx-genai-router/src/node.rs`) fills missing keys via
/// `#[serde(default)]` but **rejects an explicit `null` into a bare `f32`**,
/// which would fail the parse and mark this node unhealthy.
///
/// **Caveat for the router, which is why the omission is not cost-free:**
/// `kv_usage` feeds load-balancing, and a missing value defaults to `0.0`,
/// i.e. "this node's KV is empty" — biasing traffic *toward* a node that
/// simply cannot report. That was equally true of the hardcoded `0.0` this
/// replaces, so it is not a regression, but the router needs to distinguish
/// unknown from empty before it can route on this field honestly.
///
/// All values are model-agnostic except `model_id`, which echoes the model this
/// node serves so a captured payload is self-describing.
#[derive(Debug, Serialize)]
pub(crate) struct NodeStatus {
    node_id: String,
    /// The model this node serves, echoed so a saved response identifies itself
    /// without depending on which origin it came from.
    #[serde(skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    healthy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_usage: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_pages_used: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_pages_total: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_pages_shared: Option<u32>,
    queue_depth: u32,
    active_sessions: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    paused_sessions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_per_second: Option<f64>,
    batch_utilization: f32,
    /// The numerator of [`Self::batch_utilization`], **unclamped**.
    ///
    /// Published because the ratio alone cannot be inverted. `batch_utilization`
    /// saturates at 1.0 (see `batch_utilization`), so a client re-deriving
    /// `round(ratio * capacity)` reads "4 of 4" whether four generations are in
    /// flight or nine. Both inputs to that derivation are honest and the result
    /// is not: the clamp discards precisely the overload case, which is the
    /// most interesting state a continuous-batching demo has.
    ///
    /// With this field the client reads both terms directly and can detect
    /// `batch_in_flight > batch_capacity` itself rather than being told "full".
    batch_in_flight: u32,
    /// The denominator of [`Self::batch_utilization`], published so a client
    /// can render "3 of 4" rather than a bare percentage it must trust.
    ///
    /// Deliberately NOT named `max_batch`. It is
    /// `min(max_batch, max_queue_depth)` — see
    /// `AppConfig::effective_batch_capacity` — because admission is often the
    /// tighter constraint, and `max_batch` alone overstates the ceiling. A
    /// client that received this as `max_batch` would re-derive the wrong
    /// quantity from an honest one.
    batch_capacity: u32,
    sessions: Vec<SessionStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefix_hashes: Option<Vec<String>>,
    /// Explanations for every field omitted above, keyed by field name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    unavailable: BTreeMap<&'static str, FieldUnavailable>,
}

/// Per-session detail entry in [`NodeStatus::sessions`] (§34.8).
#[derive(Debug, Serialize)]
pub(crate) struct SessionStatus {
    id: String,
    /// Omitted rather than `"unknown"`: a literal "unknown" string still
    /// occupies the field as if it were a value, and UIs render it as one.
    #[serde(skip_serializing_if = "Option::is_none")]
    priority: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_pages: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<String>,
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
    /// Completed generations that reused a cached prefix.
    ///
    /// Named for generations, not tokens or lookups, because that is what the
    /// counter actually increments on: once per completed generation, when the
    /// reused-prefix length was greater than zero. See ARCHITECTURE.md §5.13.
    generations_with_prefix_reuse: u64,
    /// Completed generations, the denominator of the reuse rate.
    ///
    /// This was `prefix_cache_lookups`, which was a real measurement of the
    /// wrong quantity: it increments unconditionally on every completed
    /// generation, including on models that never consult the prefix cache at
    /// all, so it would read non-zero on a build with the cache deleted.
    generations_completed: u64,
    /// Fraction of completed generations that reused a prefix. Omitted entirely
    /// when no generation has completed -- 0/0 is not a 0% reuse rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_prefix_reuse_rate: Option<f64>,
    /// Prompt tokens served from a cached prefix instead of being recomputed.
    ///
    /// The reuse *rate* says how often caching fired; this says what it saved.
    /// A cache reusing 8 tokens of a 900-token prompt and one reusing 890
    /// produce the identical rate, so the rate alone cannot show whether prefix
    /// caching is doing anything worth having.
    prefix_tokens_reused: u64,
    active_batch_size: u64,
    pending_queue_depth: u64,
    available_admission_slots: usize,
    rejected_requests: u64,
    /// Where to find the paged-KV page statistics, which live on their own
    /// endpoint because they are per-page and windowed rather than scalar.
    ///
    /// This field used to read "engine does not yet expose KV page statistics".
    /// It does now, so the pointer replaces the apology.
    block_table_endpoint: &'static str,
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
}

#[derive(Debug, Serialize)]
pub(crate) struct ProfileStage {
    stage: &'static str,
    total_ms: f64,
    calls: u64,
    us_per_call: f64,
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

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Query for [`debug_kv_blocks`]. Both bounds are optional; the window is
/// always reported back so a client never has to assume what it received.
#[derive(Debug, Deserialize)]
pub(crate) struct BlockWindowQuery {
    #[serde(default)]
    pub(crate) start: usize,
    pub(crate) count: Option<usize>,
}

/// Per-page KV block table, as DENSE parallel arrays.
///
/// Parallel arrays rather than an array of objects because this is polled at
/// 1 Hz and every key would otherwise be repeated once per page. The four
/// arrays are always the same length, indexed together, and that length is
/// always `window.scanned`.
///
/// **Position is the contract.** Index `i` describes page `window.start + i`,
/// unconditionally. `page_ids[i]` is therefore always `window.start + i` and is
/// echoed deliberately redundantly: it lets a client assert the invariant in
/// one line, so any future re-introduction of compaction fails loudly at the
/// seam instead of animating a migration that never happened.
///
/// A `null` in `ref_counts` / `filled_slots` / `tiers` means that page has
/// NEVER BEEN WRITTEN -- an absence of observation. It does NOT mean free: a
/// released page reports `ref_count: 0`, which is a measurement.
#[derive(Debug, Serialize)]
pub(crate) struct BlockTable {
    page_ids: Vec<u32>,
    ref_counts: Vec<Option<u32>>,
    filled_slots: Vec<Option<usize>>,
    /// `0` = hot tier, `1` = cold. A page demoted to cold keeps its references,
    /// so it is still in use -- tier is not a proxy for free. The meaning of
    /// each value is served in `tier_names` rather than assumed, because a
    /// bare integer whose vocabulary lives in another repository is not data,
    /// it is a citation: adding a tier in Rust would silently MISLABEL the
    /// panel rather than break it.
    tiers: Vec<Option<u8>>,
}

/// The tier vocabulary, in one place, served on the wire.
///
/// Kept beside the `BlockState::tier` producer rather than duplicated in the
/// client: a hand-maintained copy in the dashboard would drift silently the
/// first time a tier is added here, and a mislabelled page is worse than an
/// unlabelled one.
const TIER_NAMES: [(u8, &str); 2] = [(0, "hot"), (1, "cold")];

/// The window a [`BlockTableResponse`] actually covers.
#[derive(Debug, Serialize)]
pub(crate) struct BlockWindow {
    /// First page id examined.
    start: usize,
    /// How many page ids were EXAMINED, after clamping to `MAX_WINDOW`.
    ///
    /// This is also the length of every array in `blocks`, because the block
    /// table is dense: asking for 256 page ids yields 256 entries, some of
    /// which may be `null` for pages never written.
    scanned: usize,
    /// How many of those pages have actually been observed, i.e. are non-null
    /// in `blocks`. `scanned - observed` is the number of pages the pool has
    /// never touched -- which is the whole pool at startup and shrinks as the
    /// demo runs. It is a fact about our KNOWLEDGE, not about occupancy.
    observed: usize,
    /// Pages the mirror can describe, so a client knows when it has them all.
    total: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct BlockTableResponse {
    /// Echoed so a captured response identifies the model it describes.
    model_id: Option<String>,
    /// **Check this before rendering.** False means paged KV is not the
    /// mechanism this model uses, and every number below would describe a pool
    /// the decoder never consults.
    applicable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    unavailable: Option<FieldUnavailable>,
    #[serde(skip_serializing_if = "Option::is_none")]
    page_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pages_in_use: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pages_shared: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hot_capacity: Option<usize>,
    /// Cumulative pressure signals. `hot_evictions` is the real "pool is full"
    /// indicator; `allocation_failures` stays zero because the pool grows by
    /// demoting to the cold tier rather than failing.
    #[serde(skip_serializing_if = "Option::is_none")]
    hot_evictions: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allocation_failures: Option<u64>,
    /// What each value in `blocks.tiers` MEANS, served rather than assumed.
    ///
    /// A bare tier integer whose vocabulary lives only in the Rust source is a
    /// citation, not data: adding a tier here would leave the panel rendering
    /// confidently with the wrong label instead of failing. Serving the map
    /// makes an unknown tier detectable by the client that has to draw it.
    /// Serialised as `tiers` to match the ratified shape (D258). The Rust name
    /// stays `tier_names` because at this level `tiers` would read as the
    /// per-page column of the same name inside `blocks` -- which is a
    /// different thing: that one is the values, this one is their vocabulary.
    #[serde(rename = "tiers", skip_serializing_if = "Option::is_none")]
    tier_names: Option<BTreeMap<u8, &'static str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    window: Option<BlockWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocks: Option<BlockTable>,
}

impl BlockTableResponse {
    /// Default page count, chosen so a response stays legible as a block table
    /// rather than becoming a texture.
    pub(crate) const DEFAULT_WINDOW: usize = 256;
    pub(crate) const MAX_WINDOW: usize = 1024;

    pub(crate) fn not_applicable(model_id: Option<String>, detail: &'static str) -> Self {
        Self {
            model_id,
            applicable: false,
            unavailable: Some(FieldUnavailable::not_applicable(detail)),
            page_size: None,
            pages_in_use: None,
            pages_shared: None,
            hot_capacity: None,
            hot_evictions: None,
            allocation_failures: None,
            // Omitted with the rest: a tier vocabulary beside no tiers would
            // imply a block table that this response explicitly does not carry.
            tier_names: None,
            window: None,
            blocks: None,
        }
    }

    /// The decode path is not chosen yet. Distinct from `not_applicable`:
    /// this one will change on its own, and a client should keep polling.
    pub(crate) fn pending(model_id: Option<String>, detail: &'static str) -> Self {
        let mut response = Self::not_applicable(model_id, detail);
        response.unavailable = response.unavailable.map(|mut u| {
            u.code = "pending";
            u
        });
        response
    }

    pub(crate) fn live(
        model_id: Option<String>,
        snapshot: onnx_genai_engine::KvTelemetrySnapshot,
        start: usize,
        total: usize,
        states: Vec<Option<onnx_genai_engine::BlockState>>,
    ) -> Self {
        // `scanned` is derived from what was actually examined, never from the
        // requested count. The mirror clamps at its own end, so a request for
        // 256 pages against a 40-page pool examines 40 -- and publishing the
        // request instead of the result would describe 216 pages that were
        // never looked at.
        let scanned = states.len();
        let mut blocks = BlockTable {
            page_ids: Vec::with_capacity(scanned),
            ref_counts: Vec::with_capacity(scanned),
            filled_slots: Vec::with_capacity(scanned),
            tiers: Vec::with_capacity(scanned),
        };
        let mut observed = 0;
        for (offset, state) in states.iter().enumerate() {
            // Derived from the position, not read from the state, so the
            // invariant holds even for pages we have never observed.
            blocks.page_ids.push((start + offset) as u32);
            match state {
                Some(state) => {
                    observed += 1;
                    blocks.ref_counts.push(Some(state.ref_count));
                    blocks.filled_slots.push(Some(state.filled_slots));
                    blocks.tiers.push(Some(state.tier));
                }
                None => {
                    blocks.ref_counts.push(None);
                    blocks.filled_slots.push(None);
                    blocks.tiers.push(None);
                }
            }
        }
        Self {
            model_id,
            applicable: true,
            unavailable: None,
            page_size: Some(snapshot.page_size),
            pages_in_use: Some(snapshot.pages_in_use),
            pages_shared: Some(snapshot.pages_shared),
            hot_capacity: Some(snapshot.hot_capacity),
            hot_evictions: Some(snapshot.hot_evictions),
            allocation_failures: Some(snapshot.allocation_failures),
            tier_names: Some(TIER_NAMES.iter().map(|(k, v)| (*k, *v)).collect()),
            window: Some(BlockWindow {
                start,
                scanned,
                observed,
                total,
            }),
            blocks: Some(blocks),
        }
    }
}
