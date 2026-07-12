//! OpenAI-compatible HTTP server wiring for onnx-genai.

use std::net::SocketAddr;

use axum::{
    Router,
    routing::{delete, get, post},
};

mod driver;
mod routes;
mod session;
mod sse;
mod state;
mod types;

pub use routes::{
    ParsedAssistantOutput, build_generate_request, build_prompt, parse_assistant_output,
    parse_tool_calls,
};
pub use state::{AppState, ServerConfig};
pub use types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatMessageToolCall,
    ChatMessageToolCallFunction, ChatTool, ChatToolFunction, CompletionChoice, CompletionRequest,
    CompletionResponse, ResponseFormat, ResponseFormatType, StopInput, ToolChoice,
    ToolChoiceFunction, ToolChoiceMode, ToolChoiceSpecific, Usage,
};

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::models))
        .route("/v1/sessions", post(routes::create_session))
        .route("/v1/sessions/{id}", delete(routes::delete_session))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/chat/completions", post(routes::chat_completions))
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

#[cfg(test)]
mod tests;
