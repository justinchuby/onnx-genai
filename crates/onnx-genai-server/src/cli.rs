//! Reusable command-line surface for the OpenAI-compatible server.
//!
//! Both the standalone `onnx-genai-server` binary and the unified `onnx-genai`
//! CLI (`onnx-genai serve`) parse [`ServeArgs`] and hand it to [`run_serve`], so
//! the server's flags live in exactly one place.

use std::{net::SocketAddr, path::PathBuf};

use clap::{ArgGroup, Args};
use onnx_genai_engine::KvDType;

#[cfg(feature = "native-backend")]
use crate::parse_native_device;
use crate::{
    AppState, ModelSpec, ModelsConfig, ServerConfig, default_node_id, from_models_dir,
    parse_kv_cache_dtype, serve,
};

/// Flags for the OpenAI-compatible HTTP server.
#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("model_source")
        .required(true)
        .args(["model", "models_dir", "models_config"])
))]
pub struct ServeArgs {
    /// Single-model mode: path to a model directory containing the ONNX model and tokenizer.
    /// Mutually exclusive with --models-dir and --models-config.
    /// Falls back to ONNX_GENAI_MODEL.
    #[arg(long, env = "ONNX_GENAI_MODEL", group = "model_source")]
    pub model: Option<PathBuf>,

    /// Model id reported by /v1/models (single-model mode only).
    /// Defaults to the model directory name.
    /// Ignored when --models-dir or --models-config is used.
    #[arg(long, requires = "model")]
    pub model_id: Option<String>,

    /// Multi-model mode: parent directory whose immediate subdirectories are each
    /// treated as one model (id = directory name, eager = true).
    /// Mutually exclusive with --model and --models-config.
    /// Falls back to ONNX_GENAI_MODELS_DIR.
    #[arg(long, env = "ONNX_GENAI_MODELS_DIR", group = "model_source")]
    pub models_dir: Option<PathBuf>,

    /// Multi-model mode: path to a TOML or JSON config file declaring the model list.
    /// Mutually exclusive with --model and --models-dir.
    /// Falls back to ONNX_GENAI_MODELS_CONFIG.
    #[arg(long, env = "ONNX_GENAI_MODELS_CONFIG", group = "model_source")]
    pub models_config: Option<PathBuf>,

    /// Node-level identifier reported by GET /v1/status for the cluster router (§34.8).
    /// Model-agnostic: it names this server process, not any model.
    /// Falls back to ONNX_GENAI_NODE_ID, then to the hostname or a generated id.
    #[arg(long, env = "ONNX_GENAI_NODE_ID")]
    pub node_id: Option<String>,

    /// Socket address to bind.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub addr: SocketAddr,

    /// Maximum requested output tokens per chat completion. Falls back to ONNX_GENAI_MAX_OUTPUT_TOKENS.
    #[arg(long, env = "ONNX_GENAI_MAX_OUTPUT_TOKENS", default_value_t = 4096)]
    pub max_output_tokens: usize,

    /// Maximum concurrent server sessions before least-recently-used eviction. Falls back to ONNX_GENAI_MAX_SESSIONS.
    #[arg(long, env = "ONNX_GENAI_MAX_SESSIONS", default_value_t = 256)]
    pub max_sessions: usize,

    /// Maximum active plus queued generation requests. Falls back to ONNX_GENAI_MAX_QUEUE_DEPTH.
    #[arg(long, env = "ONNX_GENAI_MAX_QUEUE_DEPTH", default_value_t = 256)]
    pub max_queue_depth: usize,

    /// Maximum concurrent generations per decode batch. Falls back to ONNX_GENAI_MAX_BATCH.
    ///
    /// NOT the denominator of the batch occupancy reported by /v1/status. That
    /// denominator is `min(max_batch, max_queue_depth)`, because admission is
    /// often the tighter constraint: with `--max-batch 4 --max-queue-depth 1`
    /// the batch can never hold more than one generation, and dividing by 4
    /// would draw a fully saturated server as three-quarters idle.
    #[arg(long, env = "ONNX_GENAI_MAX_BATCH", default_value_t = 4)]
    pub max_batch: usize,

    /// Enable /v1/debug/* introspection endpoints. Off by default. Use only on loopback-bound
    /// servers or behind an authenticated proxy. Falls back to ONNX_GENAI_DEBUG_ENDPOINTS=1.
    #[arg(long, env = "ONNX_GENAI_DEBUG_ENDPOINTS")]
    pub enable_debug_endpoints: bool,

    /// Enable /v1/admin/models/* runtime model-management endpoints (load, unload, list).
    /// Off by default. Use only on loopback-bound servers or behind an authenticated proxy.
    /// Falls back to ONNX_GENAI_ADMIN_ENDPOINTS=1.
    #[arg(long, env = "ONNX_GENAI_ADMIN_ENDPOINTS")]
    pub enable_admin_endpoints: bool,

    /// Maximum number of models kept loaded in memory at once. When exceeded, loading
    /// another model evicts the least-recently-used one (never below one model).
    /// Omit for unlimited. Falls back to ONNX_GENAI_MAX_LOADED_MODELS.
    #[arg(long, env = "ONNX_GENAI_MAX_LOADED_MODELS")]
    pub max_loaded_models: Option<usize>,

    /// Storage dtype for the host-side paged KV cache mirror.
    /// Accepted values: f32, int8, fp8_e4m3fn, fp8_e5m2.
    /// Falls back to ONNX_GENAI_KV_CACHE_DTYPE. Defaults to f32 (no quantisation).
    #[arg(
        long,
        env = "ONNX_GENAI_KV_CACHE_DTYPE",
        value_parser = parse_kv_cache_dtype,
        default_value = "f32"
    )]
    pub kv_cache_dtype: KvDType,

    /// Directory of static demo-dashboard assets served at GET /demo.
    /// Defaults to ./examples/serving-dashboard when it exists.
    /// Falls back to ONNX_GENAI_DEMO_ASSETS_DIR.
    #[arg(long, env = "ONNX_GENAI_DEMO_ASSETS_DIR")]
    pub demo_assets_dir: Option<PathBuf>,

    /// Device for native decoder execution: cpu, cuda, or cuda:<index>.
    /// Falls back to ONNX_GENAI_EP when omitted.
    #[cfg(feature = "native-backend")]
    #[arg(long, env = "ONNX_GENAI_NATIVE_DEVICE", value_parser = parse_native_device)]
    pub native_device: Option<onnx_genai_engine::NativeDecodeDevice>,
}

/// Build the server state from [`ServeArgs`] and serve until shutdown.
pub async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let server_config = ServerConfig {
        node_id: args.node_id.unwrap_or_else(default_node_id),
        max_output_tokens: args.max_output_tokens,
        max_sessions: args.max_sessions,
        max_queue_depth: args.max_queue_depth,
        max_batch: args.max_batch,
        enable_debug_endpoints: args.enable_debug_endpoints,
        enable_admin_endpoints: args.enable_admin_endpoints,
        max_loaded_models: args.max_loaded_models,
        eviction_policy: Default::default(),
        demo_assets_dir: crate::demo_assets::resolve_demo_assets_dir(args.demo_assets_dir),
        engine_config: onnx_genai_engine::EngineConfig {
            kv_cache_dtype: args.kv_cache_dtype,
            #[cfg(feature = "native-backend")]
            native_device: args.native_device,
            ..Default::default()
        },
    };

    // Build the model spec list from whichever source flag was provided.
    // Exactly one of --model / --models-dir / --models-config is required (ArgGroup).
    let specs: Vec<ModelSpec> = if let Some(model_path) = args.model {
        // The `onnx-genai generate` CLI coerces a config-file path to its parent
        // directory; the server does not, so the same argument that works there
        // otherwise fails deep inside the loader with a message that does not
        // mention the actual mistake. Fail here, naming the fix.
        if model_path.is_file() {
            let parent = model_path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| std::path::Path::new("."));
            anyhow::bail!(
                "--model expects a model DIRECTORY, but '{}' is a file.\n\
                 Try: --model {}\n\
                 (The `onnx-genai generate` CLI accepts a config-file path and uses \
                 its parent directory; the server takes the directory itself.)",
                model_path.display(),
                parent.display(),
            );
        }
        let model_id = args.model_id.unwrap_or_else(|| {
            model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("onnx-genai-model")
                .to_string()
        });
        vec![ModelSpec {
            id: model_id,
            path: model_path,
            eager: true,
            warmup: false,
        }]
    } else if let Some(models_dir) = args.models_dir {
        from_models_dir(&models_dir)?
    } else if let Some(config_path) = args.models_config {
        ModelsConfig::from_file(&config_path)?.models
    } else {
        unreachable!("ArgGroup enforces that exactly one model_source arg is provided")
    };

    let state = AppState::load_from_specs(specs, server_config)?;
    tracing::info!(addr = %args.addr, model = state.model_id(), "starting onnx-genai server");
    serve(args.addr, state).await
}
