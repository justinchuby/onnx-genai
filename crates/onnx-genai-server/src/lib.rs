//! OpenAI-compatible HTTP server wiring for onnx-genai.
//!
//! The default-on `metrics` feature exposes the atomic registry at `GET /metrics`;
//! disable it with `--no-default-features` when Prometheus exposition is not needed.
//! `GET /v1/debug/trace` reports tracing integration status and links to the
//! Perfetto export at `GET /v1/debug/trace/perfetto`, which serves the recorded
//! decode timeline as a Chrome Trace Event Format document. OTLP span export is
//! intentionally deferred (see issue #13).

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

mod audio_input;
mod driver;
mod image_input;
mod metrics;
mod models_config;
mod registry;
mod routes;
mod session;
mod sse;
mod state;
mod types;

pub use models_config::{ModelsConfig, ModelSpec, from_models_dir};
pub use registry::EvictionPolicy;
pub use routes::{
    ParsedAssistantOutput, build_generate_request, build_prompt, parse_assistant_output,
    parse_tool_calls,
};
pub use state::{AppState, ServerConfig, default_node_id, parse_kv_cache_dtype};
pub use types::{
    AudioTranscriptionResponse, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatLogprobs, ChatMessage, ChatMessageContent, ChatMessageContentPart, ChatMessageToolCall,
    ChatMessageToolCallFunction, ChatTokenLogprob, ChatTool, ChatToolFunction, ChatTopLogprob,
    CompletionChoice, CompletionLogprobs, CompletionRequest, CompletionResponse, EmbeddingData,
    EmbeddingEncodingFormat, EmbeddingInput, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    EmbeddingVector, ImageUrl, InputAudio, ResponseFormat, ResponseFormatType, StopInput,
    ToolChoice, ToolChoiceFunction, ToolChoiceMode, ToolChoiceSpecific, Usage,
};

pub fn app(state: AppState) -> Router {
    let mut router = Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::models))
        .route("/v1/status", get(routes::status))
        .route("/v1/sessions", post(routes::create_session))
        .route("/v1/sessions/{id}", delete(routes::delete_session))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/embeddings", post(routes::embeddings))
        .route(
            "/v1/audio/transcriptions",
            post(routes::audio_transcriptions).layer(DefaultBodyLimit::max(25 * 1024 * 1024)),
        )
        .route("/v1/chat/completions", post(routes::chat_completions));
    if state.config.enable_debug_endpoints {
        router = router
            .route("/v1/debug/config", get(routes::debug_config))
            .route("/v1/debug/sessions", get(routes::debug_sessions))
            .route("/v1/debug/kv", get(routes::debug_kv))
            .route("/v1/debug/trace", get(routes::debug_trace))
            .route(
                "/v1/debug/trace/perfetto",
                get(routes::debug_trace_perfetto),
            );
    }
    if state.config.enable_admin_endpoints {
        router = router
            .route("/v1/admin/models", get(routes::admin_list_models))
            .route(
                "/v1/admin/models/{id}/load",
                post(routes::admin_load_model),
            )
            .route("/v1/admin/models/{id}", delete(routes::admin_unload_model));
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
