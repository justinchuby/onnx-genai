use std::{net::SocketAddr, path::PathBuf};

use clap::{ArgGroup, Parser};
use onnx_genai_engine::KvDType;
#[cfg(feature = "native-backend")]
use onnx_genai_server::parse_native_device;
use onnx_genai_server::{
    AppState, ModelSpec, ModelsConfig, ServerConfig, default_node_id, from_models_dir,
    parse_kv_cache_dtype, serve,
};

#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai-server",
    about = "OpenAI-compatible HTTP server for onnx-genai",
    group(
        ArgGroup::new("model_source")
            .required(true)
            .args(["model", "models_dir", "models_config"])
    )
)]
struct Cli {
    /// Single-model mode: path to a model directory containing the ONNX model and tokenizer.
    /// Mutually exclusive with --models-dir and --models-config.
    /// Falls back to ONNX_GENAI_MODEL.
    #[arg(long, env = "ONNX_GENAI_MODEL", group = "model_source")]
    model: Option<PathBuf>,

    /// Model id reported by /v1/models (single-model mode only).
    /// Defaults to the model directory name.
    /// Ignored when --models-dir or --models-config is used.
    #[arg(long, requires = "model")]
    model_id: Option<String>,

    /// Multi-model mode: parent directory whose immediate subdirectories are each
    /// treated as one model (id = directory name, eager = true).
    /// Mutually exclusive with --model and --models-config.
    /// Falls back to ONNX_GENAI_MODELS_DIR.
    #[arg(long, env = "ONNX_GENAI_MODELS_DIR", group = "model_source")]
    models_dir: Option<PathBuf>,

    /// Multi-model mode: path to a TOML or JSON config file declaring the model list.
    /// Mutually exclusive with --model and --models-dir.
    /// Falls back to ONNX_GENAI_MODELS_CONFIG.
    #[arg(long, env = "ONNX_GENAI_MODELS_CONFIG", group = "model_source")]
    models_config: Option<PathBuf>,

    /// Node-level identifier reported by GET /v1/status for the cluster router (§34.8).
    /// Model-agnostic: it names this server process, not any model.
    /// Falls back to ONNX_GENAI_NODE_ID, then to the hostname or a generated id.
    #[arg(long, env = "ONNX_GENAI_NODE_ID")]
    node_id: Option<String>,

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

    /// Enable /v1/admin/models/* runtime model-management endpoints (load, unload, list).
    /// Off by default. Use only on loopback-bound servers or behind an authenticated proxy.
    /// Falls back to ONNX_GENAI_ADMIN_ENDPOINTS=1.
    #[arg(long, env = "ONNX_GENAI_ADMIN_ENDPOINTS")]
    enable_admin_endpoints: bool,

    /// Maximum number of models kept loaded in memory at once. When exceeded, loading
    /// another model evicts the least-recently-used one (never below one model).
    /// Omit for unlimited. Falls back to ONNX_GENAI_MAX_LOADED_MODELS.
    #[arg(long, env = "ONNX_GENAI_MAX_LOADED_MODELS")]
    max_loaded_models: Option<usize>,

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

    /// Device for native decoder execution: cpu, cuda, or cuda:<index>.
    /// Falls back to ONNX_GENAI_EP when omitted.
    #[cfg(feature = "native-backend")]
    #[arg(long, env = "ONNX_GENAI_NATIVE_DEVICE", value_parser = parse_native_device)]
    native_device: Option<onnx_genai_engine::NativeDecodeDevice>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let server_config = ServerConfig {
        node_id: cli.node_id.unwrap_or_else(default_node_id),
        max_output_tokens: cli.max_output_tokens,
        max_sessions: cli.max_sessions,
        max_queue_depth: cli.max_queue_depth,
        enable_debug_endpoints: cli.enable_debug_endpoints,
        enable_admin_endpoints: cli.enable_admin_endpoints,
        max_loaded_models: cli.max_loaded_models,
        eviction_policy: Default::default(),
        engine_config: onnx_genai_engine::EngineConfig {
            kv_cache_dtype: cli.kv_cache_dtype,
            #[cfg(feature = "native-backend")]
            native_device: cli.native_device,
            ..Default::default()
        },
    };

    // Build the model spec list from whichever source flag was provided.
    // Exactly one of --model / --models-dir / --models-config is required (ArgGroup).
    let specs: Vec<ModelSpec> = if let Some(model_path) = cli.model {
        let model_id = cli.model_id.unwrap_or_else(|| {
            model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("onnx-genai-model")
                .to_string()
        });
        vec![ModelSpec { id: model_id, path: model_path, eager: true }]
    } else if let Some(models_dir) = cli.models_dir {
        from_models_dir(&models_dir)?
    } else if let Some(config_path) = cli.models_config {
        ModelsConfig::from_file(&config_path)?.models
    } else {
        unreachable!("ArgGroup enforces that exactly one model_source arg is provided")
    };

    let state = AppState::load_from_specs(specs, server_config)?;
    tracing::info!(addr = %cli.addr, model = state.model_id(), "starting onnx-genai server");
    serve(cli.addr, state).await
}
