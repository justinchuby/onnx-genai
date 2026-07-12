use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;
use onnx_genai_server::{AppState, serve};

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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let state = AppState::load(&cli.model, cli.model_id)?;
    tracing::info!(addr = %cli.addr, model = state.model_id(), "starting onnx-genai server");
    serve(cli.addr, state).await
}
