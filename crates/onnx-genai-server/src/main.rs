use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;
use onnx_genai_server::{AppState, ServerConfig, serve};

#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai-server",
    about = "OpenAI-compatible HTTP server for onnx-genai"
)]
struct Cli {
    /// Model directory containing the ONNX model and tokenizer. Falls back to ONNX_GENAI_MODEL.
    #[arg(long, env = "ONNX_GENAI_MODEL")]
    model: PathBuf,

    /// Model id reported by /v1/models. Defaults to the model directory name.
    #[arg(long)]
    model_id: Option<String>,

    /// Socket address to bind.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: SocketAddr,

    /// Maximum requested output tokens per chat completion. Falls back to ONNX_GENAI_MAX_OUTPUT_TOKENS.
    #[arg(long, env = "ONNX_GENAI_MAX_OUTPUT_TOKENS", default_value_t = 4096)]
    max_output_tokens: usize,

    /// Maximum concurrent server sessions before least-recently-used eviction. Falls back to ONNX_GENAI_MAX_SESSIONS.
    #[arg(long, env = "ONNX_GENAI_MAX_SESSIONS", default_value_t = 256)]
    max_sessions: usize,

    /// Maximum active plus queued generation requests. Falls back to ONNX_GENAI_MAX_PENDING.
    #[arg(long, env = "ONNX_GENAI_MAX_PENDING", default_value_t = 256)]
    max_pending: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let state = AppState::load_with_config(
        &cli.model,
        cli.model_id,
        ServerConfig {
            max_output_tokens: cli.max_output_tokens,
            max_sessions: cli.max_sessions,
            max_pending: cli.max_pending,
        },
    )?;
    tracing::info!(addr = %cli.addr, model = state.model_id(), "starting onnx-genai server");
    serve(cli.addr, state).await
}
