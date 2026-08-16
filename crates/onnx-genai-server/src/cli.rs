//! Reusable command-line surface for the OpenAI-compatible server.
//!
//! Both the standalone `onnx-genai-server` binary and the unified `onnx-genai`
//! CLI (`onnx-genai serve`) parse [`ServeArgs`] and hand it to [`run_serve`], so
//! the server's flags live in exactly one place.

use std::{net::SocketAddr, path::PathBuf};

use clap::{ArgGroup, Args};
use onnx_genai_engine::{KvDType, ResourceLimit, parse_resource_limit};

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

    /// Maximum number of sequences decoded concurrently in one continuous batch.
    /// Omit to let the server pick a default and clamp it to what the model's
    /// decode path can honor. Setting a value greater than 1 on a backend that
    /// cannot batch (the native runtime, or a legacy / non-shared-buffer ONNX
    /// Runtime model) is refused at startup rather than silently ignored.
    /// Falls back to ONNX_GENAI_MAX_BATCH.
    #[arg(long, env = "ONNX_GENAI_MAX_BATCH")]
    pub max_batch: Option<usize>,

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

    /// VRAM ceiling for the engine resource governor. Resolved before model load so
    /// native CUDA can choose governed weight offload instead of relying on WDDM paging.
    /// Accepted values: bytes, fractions such as 0.9, byte strings such as 8GiB, or auto.
    #[arg(long, env = "ONNX_GENAI_VRAM_LIMIT", value_parser = parse_limit)]
    pub vram_limit: Option<ResourceLimit>,

    /// Device for native decoder execution: cpu, cuda, or cuda:<index>.
    /// Falls back to ONNX_GENAI_EP when omitted.
    #[cfg(feature = "native-backend")]
    #[arg(long, env = "ONNX_GENAI_NATIVE_DEVICE", value_parser = parse_native_device)]
    pub native_device: Option<onnx_genai_engine::NativeDecodeDevice>,
}

fn parse_limit(input: &str) -> Result<ResourceLimit, String> {
    parse_resource_limit(input).map_err(|error| error.to_string())
}

fn server_config_from_args(args: &ServeArgs) -> ServerConfig {
    let mut engine_config = onnx_genai_engine::EngineConfig {
        kv_cache_dtype: args.kv_cache_dtype,
        #[cfg(feature = "native-backend")]
        native_device: args.native_device.clone(),
        #[cfg(feature = "native-backend")]
        decode_backend: if args.native_device.is_some() {
            onnx_genai_engine::EngineDecodeBackend::Native
        } else {
            onnx_genai_engine::EngineDecodeBackend::default()
        },
        ..Default::default()
    };
    if let Some(vram_limit) = args.vram_limit {
        engine_config.limits.vram_limit = vram_limit;
    }
    // #1064: `--max-batch N` is what *requests* the native persistent decode
    // batch extent. Previously the extent could only be turned on by
    // `ONNX_GENAI_NATIVE_DECODE_BATCH`, so `--max-batch N` was refused at
    // startup: the batching capability was derived from a decode session nobody
    // had asked to build in batch shape, which always reported 1. Only values
    // above 1 are passed through, so the single-sequence default stays exactly as
    // it was. The ORT backend ignores this field and keeps its own batching.
    #[cfg(feature = "native-backend")]
    if let Some(max_batch) = args.max_batch.filter(|&batch| batch > 1) {
        engine_config.native_decode_batch = Some(max_batch);
    }

    ServerConfig {
        node_id: args.node_id.clone().unwrap_or_else(default_node_id),
        max_output_tokens: args.max_output_tokens,
        max_sessions: args.max_sessions,
        max_queue_depth: args.max_queue_depth,
        max_batch: args.max_batch,
        enable_debug_endpoints: args.enable_debug_endpoints,
        enable_admin_endpoints: args.enable_admin_endpoints,
        max_loaded_models: args.max_loaded_models,
        eviction_policy: Default::default(),
        engine_config,
    }
}

/// Build the server state from [`ServeArgs`] and serve until shutdown.
pub async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let server_config = server_config_from_args(&args);

    // Build the model spec list from whichever source flag was provided.
    // Exactly one of --model / --models-dir / --models-config is required (ArgGroup).
    let specs: Vec<ModelSpec> = if let Some(model_path) = args.model {
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Debug, Parser)]
    struct TestCli {
        #[command(flatten)]
        serve: ServeArgs,
    }

    /// #1064: `--max-batch N` must *shape the decode session*, not merely be
    /// checked against a capability derived from a session built for one
    /// sequence. Before this, batch-N could only be enabled by
    /// `ONNX_GENAI_NATIVE_DECODE_BATCH`, so `--max-batch 4` was refused at
    /// startup with "this backend decodes at most 1 sequence(s) concurrently".
    #[cfg(feature = "native-backend")]
    #[test]
    fn serve_max_batch_requests_the_native_decode_batch_extent() {
        let cli = TestCli::parse_from(["test", "--model", "model-dir", "--max-batch", "4"]);

        assert_eq!(
            server_config_from_args(&cli.serve)
                .engine_config
                .native_decode_batch,
            Some(4)
        );
    }

    /// The single-sequence default must stay exactly as it was: neither an
    /// omitted flag nor an explicit `1` may shape a batch grid, since batch 1 is
    /// the #750 byte-identity reference.
    #[cfg(feature = "native-backend")]
    #[test]
    fn serve_leaves_single_sequence_decode_untouched() {
        let omitted = TestCli::parse_from(["test", "--model", "model-dir"]);
        let explicit_one =
            TestCli::parse_from(["test", "--model", "model-dir", "--max-batch", "1"]);

        assert_eq!(
            server_config_from_args(&omitted.serve)
                .engine_config
                .native_decode_batch,
            None
        );
        assert_eq!(
            server_config_from_args(&explicit_one.serve)
                .engine_config
                .native_decode_batch,
            None
        );
    }

    #[test]
    fn serve_vram_limit_flows_into_engine_config_before_load() {
        let cli =
            TestCli::parse_from(["test", "--model", "model-dir", "--vram-limit", "6000000000"]);

        let config = server_config_from_args(&cli.serve);

        assert_eq!(
            config.engine_config.limits.vram_limit,
            ResourceLimit::Bytes(6_000_000_000)
        );
    }

    #[test]
    fn serve_vram_limit_accepts_fraction_and_auto() {
        let fraction = TestCli::parse_from(["test", "--model", "model-dir", "--vram-limit", "0.5"]);
        let auto = TestCli::parse_from(["test", "--model", "model-dir", "--vram-limit", "auto"]);

        assert_eq!(
            server_config_from_args(&fraction.serve)
                .engine_config
                .limits
                .vram_limit,
            ResourceLimit::Fraction(0.5)
        );
        assert_eq!(
            server_config_from_args(&auto.serve)
                .engine_config
                .limits
                .vram_limit,
            ResourceLimit::Auto
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn explicit_native_device_selects_native_decode() {
        let cli = TestCli::parse_from(["test", "--model", "model-dir", "--native-device", "cpu"]);

        assert_eq!(
            server_config_from_args(&cli.serve)
                .engine_config
                .decode_backend,
            onnx_genai_engine::EngineDecodeBackend::Native
        );
    }
}
