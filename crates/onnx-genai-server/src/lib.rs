//! OpenAI-compatible HTTP server wiring for onnx-genai.
//!
//! The default-on `metrics` feature exposes the atomic registry at `GET /metrics`;
//! disable it with `--no-default-features` when Prometheus exposition is not needed.
//! `GET /v1/debug/trace` reports tracing integration status and links to the
//! Perfetto export at `GET /v1/debug/trace/perfetto`, which serves the recorded
//! decode timeline as a Chrome Trace Event Format document. It merges engine,
//! native-runtime, and execution-provider spans when available. OTLP span export
//! is intentionally deferred (see issue #13).

#![forbid(unsafe_code)]

use std::{net::SocketAddr, time::Instant};

use axum::{
    Router,
    extract::{DefaultBodyLimit, Request},
    middleware,
    middleware::Next,
    response::Response,
    routing::{delete, get, post},
};
use tracing::Instrument;

const MEDIA_UPLOAD_BODY_LIMIT: usize = 25 * 1024 * 1024;

mod audio_input;
mod cli;
mod driver;
mod image_generation;
mod image_input;
mod lease;
mod metrics;
mod models_config;
pub mod multimodal;
mod registry;
mod routes;
pub mod runtime_args;
mod session;
mod speech;
mod sse;
mod state;
mod tool_protocol;
mod types;
mod worker;

pub use cli::{ServeArgs, run_serve};
pub use models_config::{ModelSpec, ModelsConfig, from_models_dir};
pub use multimodal::MultimodalSpecs;
pub use registry::EvictionPolicy;
pub use routes::{
    ParsedAssistantOutput, build_generate_request, build_prompt, parse_assistant_output,
};
pub use runtime_args::{
    CpuArgs, DeviceChoice, EngineArgs, decode_backend_name, parse_decode_backend, parse_device,
};
#[cfg(feature = "native-backend")]
pub use state::parse_native_device;
pub use state::{
    AppState, OrtSessionWorkerCount, ServerConfig, default_node_id, parse_kv_cache_dtype,
};
pub use tool_protocol::{ToolCallStream, ToolParseOutcome, ToolProtocol, ToolProtocolError};
pub use types::{
    AudioSpeechRequest, AudioTranscriptionResponse, ChatChoice, ChatCompletionRequest,
    ChatCompletionResponse, ChatLogprobs, ChatMessage, ChatMessageContent, ChatMessageContentPart,
    ChatMessageToolCall, ChatMessageToolCallFunction, ChatTokenLogprob, ChatTool, ChatToolFunction,
    ChatTopLogprob, CompletionChoice, CompletionLogprobs, CompletionRequest, CompletionResponse,
    EmbeddingData, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingRequest, EmbeddingResponse,
    EmbeddingUsage, EmbeddingVector, ImageUrl, InputAudio, JsonSchemaSpec, ResponseFormat,
    StopInput, ToolChoice, ToolChoiceFunction, ToolChoiceMode, ToolChoiceSpecific, Usage,
};

pub fn app(state: AppState) -> Router {
    let mut router = Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::models))
        .route("/v1/status", get(routes::status))
        .route("/v1/resources", get(routes::resources))
        .route("/v1/sessions", post(routes::create_session))
        .route("/v1/sessions/{id}", delete(routes::delete_session))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/embeddings", post(routes::embeddings))
        .route(
            "/v1/images/generations",
            post(routes::openai_images).layer(DefaultBodyLimit::max(MEDIA_UPLOAD_BODY_LIMIT)),
        )
        .route(
            "/sdapi/v1/txt2img",
            post(routes::a1111_txt2img).layer(DefaultBodyLimit::max(MEDIA_UPLOAD_BODY_LIMIT)),
        )
        .route(
            "/sdapi/v1/img2img",
            post(routes::a1111_img2img).layer(DefaultBodyLimit::max(MEDIA_UPLOAD_BODY_LIMIT)),
        )
        .route("/sdapi/v1/sd-models", get(routes::a1111_models))
        .route("/sdapi/v1/samplers", get(routes::a1111_samplers))
        .route("/sdapi/v1/options", get(routes::a1111_options))
        .route(
            "/v1/audio/transcriptions",
            post(routes::audio_transcriptions)
                .layer(DefaultBodyLimit::max(MEDIA_UPLOAD_BODY_LIMIT)),
        )
        .route("/v1/audio/speech", post(routes::audio_speech))
        .route("/v1/chat/completions", post(routes::chat_completions));
    if state.config.enable_debug_endpoints {
        router = router
            .route("/v1/debug/config", get(routes::debug_config))
            .route("/v1/debug/sessions", get(routes::debug_sessions))
            .route("/v1/debug/kv", get(routes::debug_kv))
            .route("/v1/debug/trace", get(routes::debug_trace))
            .route("/v1/debug/profile", get(routes::debug_profile))
            .route(
                "/v1/debug/trace/perfetto",
                get(routes::debug_trace_perfetto),
            );
    }
    if state.config.enable_admin_endpoints {
        router = router
            .route("/v1/admin/models", get(routes::admin_list_models))
            .route("/v1/admin/models/{id}/load", post(routes::admin_load_model))
            .route(
                "/v1/admin/models/{id}/warm",
                post(routes::admin_warmup_model),
            )
            .route("/v1/admin/models/{id}", delete(routes::admin_unload_model))
            .route(
                "/v1/admin/resources/vram-limit",
                post(routes::admin_set_vram_limit),
            );
    }
    #[cfg(feature = "metrics")]
    let router = router.route("/metrics", get(routes::prometheus_metrics));
    router
        .with_state(state)
        .layer(middleware::from_fn(trace_request))
}

async fn trace_request(request: Request, next: Next) -> Response {
    let trace_id = metrics::request_started();
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let started = Instant::now();
    let span = tracing::info_span!(
        "http.request",
        trace_id = format_args!("{trace_id:016x}"),
        method = %method,
        path = %path,
        status = tracing::field::Empty,
        latency_ms = tracing::field::Empty,
    );
    async move {
        let response = next.run(request).await;
        let status = response.status();
        metrics::request_finished(&path, status);
        tracing::Span::current().record("status", status.as_u16());
        tracing::Span::current().record("latency_ms", started.elapsed().as_millis() as u64);
        response
    }
    .instrument(span)
    .await
}

pub async fn serve(addr: SocketAddr, state: AppState) -> anyhow::Result<()> {
    // Security posture: the server has no built-in authentication. The CLI defaults
    // to 127.0.0.1, enforces max_tokens/max_sessions caps, and issues CSPRNG session
    // ids; binding a non-loopback --addr should be done only behind an auth proxy.
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app(state)).await?;
    Ok(())
}

#[cfg(test)]
mod tests;
