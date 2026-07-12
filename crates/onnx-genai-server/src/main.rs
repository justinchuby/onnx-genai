//! OpenAI-compatible HTTP server.

use axum::{Json, Router, routing::post};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: usize,
    message: Message,
    finish_reason: String,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    1.0
}

async fn chat_completions(
    Json(request): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    tracing::info!("Received request for model: {}", request.model);

    // TODO: Route to engine for actual generation
    Json(ChatCompletionResponse {
        id: "chatcmpl-placeholder".to_string(),
        object: "chat.completion".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: "Hello! This is a placeholder response. Engine integration coming soon."
                    .to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let app = Router::new().route("/v1/chat/completions", post(chat_completions));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    tracing::info!("Starting onnx-genai server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
