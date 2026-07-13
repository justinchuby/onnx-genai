use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;
use onnx_genai_engine::KvDType;
use onnx_genai_server::{AppState, ServerConfig, parse_kv_cache_dtype, serve};

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

    /// Maximum active plus queued generation requests. Falls back to ONNX_GENAI_MAX_QUEUE_DEPTH.
    #[arg(long, env = "ONNX_GENAI_MAX_QUEUE_DEPTH", default_value_t = 256)]
    max_queue_depth: usize,

    /// Enable /v1/debug/* introspection endpoints. Off by default. Use only on loopback-bound
    /// servers or behind an authenticated proxy. Falls back to ONNX_GENAI_DEBUG_ENDPOINTS=1.
    #[arg(long, env = "ONNX_GENAI_DEBUG_ENDPOINTS")]
    enable_debug_endpoints: bool,

    /// Storage dtype for the host-side paged KV cache mirror.
    /// Accepted values: f32, int8, fp8_e4m3fn, fp8_e5m2.
    /// Falls back to ONNX_GENAI_KV_CACHE_DTYPE. Defaults to f32 (no quantisation).
    #[arg(
        long,
        env = "ONNX_GENAI_KV_CACHE_DTYPE",
        value_parser = parse_kv_cache_dtype,
        default_value = "f32"
    )]
    kv_cache_dtype: KvDType,
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
            max_queue_depth: cli.max_queue_depth,
            enable_debug_endpoints: cli.enable_debug_endpoints,
            engine_config: onnx_genai_engine::EngineConfig {
                kv_cache_dtype: cli.kv_cache_dtype,
                ..Default::default()
            },
        },
    )?;
    tracing::info!(addr = %cli.addr, model = state.model_id(), "starting onnx-genai server");
    serve(cli.addr, state).await
}
