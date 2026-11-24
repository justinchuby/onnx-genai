//! Reusable command-line surface for the OpenAI-compatible server.
//!
//! Both the standalone `onnx-genai-server` binary and the unified `onnx-genai`
//! CLI (`onnx-genai serve`) parse [`ServeArgs`] and hand it to [`run_serve`], so
//! the server's flags live in exactly one place.

use std::{net::SocketAddr, path::PathBuf};

use clap::{ArgGroup, Args};
use onnx_genai_engine::KvDType;

use crate::types::ReasoningEffort;
use crate::{
    AppState, ModelSpec, ModelsConfig, OrtSessionWorkerCount, ServerConfig, default_node_id,
    from_models_dir, parse_kv_cache_dtype,
    runtime_args::{CpuArgs, EngineArgs},
    serve,
};

/// Flags for the OpenAI-compatible HTTP server.
#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("model_source")
        .required(true)
        .args(["model", "models_dir", "models_config"])
))]
pub struct ServeArgs {
    /// Single-model mode: path to a model directory containing the ONNX model
    /// and tokenizer, given positionally as every other subcommand takes it.
    /// Mutually exclusive with --models-dir and --models-config.
    /// Falls back to ONNX_GENAI_MODEL.
    #[arg(value_name = "MODEL", env = "ONNX_GENAI_MODEL", group = "model_source")]
    pub model: Option<PathBuf>,

    /// Model id reported by /v1/models (single-model mode only).
    /// Defaults to the model directory name.
    /// Rejected when --models-dir or --models-config is used.
    #[arg(long)]
    pub model_id: Option<String>,

    /// Multi-model mode: parent directory whose immediate subdirectories are each
    /// treated as one model (id = directory name, eager = true).
    /// Mutually exclusive with --model and --models-config.
    /// Falls back to ONNX_GENAI_MODELS_DIR.
    #[arg(long, env = "ONNX_GENAI_MODELS_DIR", group = "model_source")]
    pub models_dir: Option<PathBuf>,

    /// Multi-model mode: path to a TOML or JSON config file declaring the model list.
    /// Mutually exclusive with the positional MODEL and --models-dir.
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
    /// Maximum completion prompts in one native or OpenAI request.
    #[arg(long, default_value = "128")]
    pub max_completion_prompts: std::num::NonZeroUsize,
    /// Maximum tokens in one reranking query/document pair.
    #[arg(long, default_value = "8192")]
    pub max_rerank_input_tokens: std::num::NonZeroUsize,
    /// Maximum number of documents in one reranking request.
    #[arg(long, default_value = "128")]
    pub max_rerank_documents: std::num::NonZeroUsize,

    /// Maximum active/retained resumable generations (X-Conversation-Id).
    #[arg(long, default_value_t = 16)]
    pub max_replay_streams: usize,

    /// Maximum retained SSE bytes per conversation; older prefixes are evicted.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    pub replay_buffer_bytes: usize,

    /// Enable the upstream POST /props acknowledgement (no properties are changed).
    #[arg(long, env = "NXRT_PROPS")]
    pub props: bool,

    /// Enable host filesystem/command tools for trusted clients: all or a comma-separated list.
    #[arg(long, value_delimiter = ',', env = "NXRT_TOOLS")]
    pub tools: Vec<crate::BuiltinTool>,
    /// Maximum in-flight host-tool calls, including responses still being consumed.
    #[arg(long, default_value = "8")]
    pub tools_max_concurrent: std::num::NonZeroUsize,
    /// Host memory budget for tool request/response and file I/O staging, in bytes.
    #[arg(long, default_value = "268435456")]
    pub tools_max_host_bytes: std::num::NonZeroU64,
    /// Additional trusted browser origin permitted to invoke host tools.
    #[arg(long)]
    pub tools_allow_origin: Vec<String>,

    /// Cursor-compatible stdio MCP configuration file (executes configured commands).
    #[arg(long, env = "NXRT_MCP_SERVERS_CONFIG")]
    pub mcp_servers_config: Option<PathBuf>,
    /// Inline Cursor-compatible stdio MCP configuration JSON.
    #[arg(long, env = "NXRT_MCP_SERVERS_JSON")]
    pub mcp_servers_json: Option<String>,

    /// Maximum bytes per incoming MCP stdio message (up to upstream's 8 MiB limit).
    #[arg(long, default_value = "262144")]
    pub mcp_max_message_bytes: std::num::NonZeroUsize,

    /// Enable the official chat frontend's HTTP(S) MCP proxy (off by default).
    #[arg(long)]
    pub ui_mcp_proxy: bool,
    /// Permit private/local targets through the explicitly enabled MCP proxy.
    #[arg(long, requires = "ui_mcp_proxy")]
    pub ui_mcp_proxy_allow_private_urls: bool,
    /// Maximum bytes in one MCP proxy request or response.
    #[arg(long, default_value = "67108864")]
    pub ui_mcp_proxy_max_bytes: std::num::NonZeroUsize,
    /// Maximum simultaneous outbound MCP proxy transfers.
    #[arg(long, default_value = "16")]
    pub ui_mcp_proxy_max_connections: std::num::NonZeroUsize,
    /// Maximum wait without progress during an MCP proxy transfer, in seconds.
    #[arg(long, default_value = "600")]
    pub ui_mcp_proxy_timeout_secs: std::num::NonZeroU64,

    /// Reasoning effort applied when a request omits `reasoning_effort`.
    ///
    /// Reasoning models pick their own default when the template receives no
    /// effort, and that default is often the maximum. Agent clients frequently
    /// never send the field at all, so the model can spend an entire token
    /// budget thinking and emit no answer. Setting this gives the operator a
    /// floor without overriding clients that do ask. Unset leaves the model's
    /// own default in place. Falls back to ONNX_GENAI_DEFAULT_REASONING_EFFORT.
    #[arg(long, env = "ONNX_GENAI_DEFAULT_REASONING_EFFORT", value_enum)]
    pub default_reasoning_effort: Option<ReasoningEffort>,

    /// Maximum concurrent server sessions before least-recently-used eviction. Falls back to ONNX_GENAI_MAX_SESSIONS.
    #[arg(long, env = "ONNX_GENAI_MAX_SESSIONS", default_value_t = 256)]
    pub max_sessions: usize,

    /// Maximum active plus queued generation requests. Falls back to ONNX_GENAI_MAX_QUEUE_DEPTH.
    #[arg(long, env = "ONNX_GENAI_MAX_QUEUE_DEPTH", default_value_t = 256)]
    pub max_queue_depth: usize,

    /// Number of owner-thread ORT engines used to run distinct sessions in
    /// parallel. This is not an ORT intra-op/inter-op thread count. Values above
    /// 1 are accepted only for supported ORT sessions and fail closed otherwise.
    /// Falls back to ONNX_GENAI_ORT_SESSION_WORKERS.
    #[arg(long, env = "ONNX_GENAI_ORT_SESSION_WORKERS", default_value = "1")]
    pub ort_session_workers: OrtSessionWorkerCount,

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

    /// Load declared adapters without applying them until selected by API.
    #[arg(long, env = "NXRT_LORA_INIT_WITHOUT_APPLY")]
    pub lora_init_without_apply: bool,
    /// Directory for llama.cpp slot save/restore/erase actions.
    #[arg(long, env = "NXRT_SLOT_SAVE_PATH")]
    pub slot_save_path: Option<std::path::PathBuf>,
    #[arg(
        long,
        env = "NXRT_SLOT_SAVE_MAX_FILE_BYTES",
        default_value = "4294967296"
    )]
    pub slot_save_max_file_bytes: std::num::NonZeroU64,
    #[arg(
        long,
        env = "NXRT_SLOT_SAVE_MAX_TOTAL_BYTES",
        default_value = "17179869184"
    )]
    pub slot_save_max_total_bytes: std::num::NonZeroU64,
    #[arg(long, env = "NXRT_SLOT_SAVE_MAX_FILES", default_value = "128")]
    pub slot_save_max_files: std::num::NonZeroUsize,

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

    #[command(flatten)]
    pub engine: EngineArgs,

    #[command(flatten)]
    pub cpu: CpuArgs,
}

fn server_config_from_args(args: &ServeArgs) -> ServerConfig {
    let mut engine_config = args.engine.to_config();
    engine_config.kv_cache_dtype = args.kv_cache_dtype;
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
        endpoint_props: args.props,
        tools: crate::ToolsConfig {
            enabled: args.tools.clone(),
            mcp: crate::McpSources {
                file: args.mcp_servers_config.clone(),
                json: args.mcp_servers_json.as_deref().map(std::sync::Arc::from),
                max_message_bytes: args.mcp_max_message_bytes,
            },
            max_concurrent: args.tools_max_concurrent,
            max_host_bytes: args.tools_max_host_bytes,
            allow_origins: args.tools_allow_origin.clone(),
        },
        max_completion_prompts: args.max_completion_prompts,
        max_rerank_input_tokens: args.max_rerank_input_tokens,
        max_rerank_documents: args.max_rerank_documents,
        max_replay_streams: args.max_replay_streams,
        replay_buffer_bytes: args.replay_buffer_bytes,
        mcp_proxy_policy: if !args.ui_mcp_proxy {
            crate::McpProxyPolicy::Disabled
        } else if args.ui_mcp_proxy_allow_private_urls {
            crate::McpProxyPolicy::AllowPrivate
        } else {
            crate::McpProxyPolicy::Public
        },
        mcp_proxy_max_bytes: args.ui_mcp_proxy_max_bytes,
        mcp_proxy_max_connections: args.ui_mcp_proxy_max_connections,
        mcp_proxy_timeout_secs: args.ui_mcp_proxy_timeout_secs,
        default_reasoning_effort: args.default_reasoning_effort,
        max_sessions: args.max_sessions,
        max_queue_depth: args.max_queue_depth,
        ort_session_workers: args.ort_session_workers,
        max_batch: args.max_batch,
        enable_debug_endpoints: args.enable_debug_endpoints,
        enable_admin_endpoints: args.enable_admin_endpoints,
        slot_save_path: args.slot_save_path.clone(),
        lora_init_without_apply: args.lora_init_without_apply,
        slot_save_max_file_bytes: args.slot_save_max_file_bytes,
        slot_save_max_total_bytes: args.slot_save_max_total_bytes,
        slot_save_max_files: args.slot_save_max_files,
        max_loaded_models: args.max_loaded_models,
        eviction_policy: Default::default(),
        engine_config,
    }
}

/// Build the server state from [`ServeArgs`] and serve until shutdown.
pub async fn run_serve(mut args: ServeArgs) -> anyhow::Result<()> {
    args.cpu.apply().map_err(anyhow::Error::msg)?;
    let server_config = server_config_from_args(&args);
    let model_path = args.model.take();

    // Build the model spec list from whichever source flag was provided.
    // Exactly one model source is required (ArgGroup).
    let specs: Vec<ModelSpec> = if let Some(model_path) = model_path {
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
    } else {
        if args.model_id.is_some() {
            anyhow::bail!(
                "What: --model-id was given alongside a multi-model source. \
                 Why: --models-dir and --models-config name every model themselves, \
                 so there is no single id to override. \
                 How: drop --model-id, or serve one model with --model."
            );
        }
        if let Some(models_dir) = args.models_dir {
            from_models_dir(&models_dir)?
        } else if let Some(config_path) = args.models_config {
            ModelsConfig::from_file(&config_path)?.models
        } else {
            unreachable!("ArgGroup enforces that exactly one model_source arg is provided")
        }
    };

    let state = AppState::load_from_specs(specs, server_config)?;
    tracing::info!(addr = %args.addr, model = state.model_id(), "starting onnx-genai server");
    serve(args.addr, state).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use onnx_genai_engine::ResourceLimit;

    #[derive(Debug, Parser)]
    struct TestCli {
        #[command(flatten)]
        serve: ServeArgs,
    }

    #[test]
    fn lora_init_without_apply_flag_preserves_explicit_operator_choice() {
        let disabled = TestCli::parse_from(["test", "model-dir", "--lora-init-without-apply"]);
        assert!(server_config_from_args(&disabled.serve).lora_init_without_apply);
        let defaults = TestCli::parse_from(["test", "model-dir"]);
        assert!(!server_config_from_args(&defaults.serve).lora_init_without_apply);
    }

    #[test]
    fn slot_storage_flags_parse_positive_bounds_and_path() {
        let args = TestCli::try_parse_from([
            "test",
            "model-dir",
            "--slot-save-path",
            "states",
            "--slot-save-max-file-bytes",
            "1024",
            "--slot-save-max-total-bytes",
            "4096",
            "--slot-save-max-files",
            "3",
        ])
        .unwrap();
        let config = server_config_from_args(&args.serve);
        assert_eq!(
            config.slot_save_path.as_deref(),
            Some(std::path::Path::new("states"))
        );
        assert_eq!(config.slot_save_max_file_bytes.get(), 1024);
        assert_eq!(config.slot_save_max_total_bytes.get(), 4096);
        assert_eq!(config.slot_save_max_files.get(), 3);
        for flag in [
            "--slot-save-max-file-bytes",
            "--slot-save-max-total-bytes",
            "--slot-save-max-files",
        ] {
            assert!(TestCli::try_parse_from(["test", "model-dir", flag, "0"]).is_err());
        }
    }

    #[test]
    fn mcp_configuration_is_explicit_and_message_bounds_are_positive() {
        let disabled = TestCli::parse_from(["test", "model-dir"]);
        assert!(
            !server_config_from_args(&disabled.serve)
                .tools
                .mcp
                .configured()
        );
        let enabled = TestCli::parse_from([
            "test",
            "model-dir",
            "--mcp-servers-config",
            "mcp.json",
            "--mcp-servers-json",
            r#"{"mcpServers":{}}"#,
            "--mcp-max-message-bytes",
            "524288",
        ]);
        let config = server_config_from_args(&enabled.serve);
        assert!(config.tools.enabled.is_empty());
        assert_eq!(config.tools.mcp.file, Some(PathBuf::from("mcp.json")));
        assert_eq!(
            config.tools.mcp.json.as_deref(),
            Some(r#"{"mcpServers":{}}"#)
        );
        assert_eq!(config.tools.mcp.max_message_bytes.get(), 524288);
        assert!(
            TestCli::try_parse_from(["test", "model-dir", "--mcp-max-message-bytes", "0"]).is_err()
        );
    }

    #[test]
    fn props_acknowledgement_requires_explicit_enable() {
        let disabled = TestCli::parse_from(["test", "model-dir"]);
        assert!(!server_config_from_args(&disabled.serve).endpoint_props);
        let enabled = TestCli::parse_from(["test", "model-dir", "--props"]);
        assert!(server_config_from_args(&enabled.serve).endpoint_props);
    }

    #[test]
    fn builtin_tools_require_explicit_selection_and_bounded_resources() {
        let disabled = TestCli::parse_from(["test", "model-dir"]);
        assert!(
            server_config_from_args(&disabled.serve)
                .tools
                .enabled
                .is_empty()
        );
        let enabled = TestCli::parse_from([
            "test",
            "model-dir",
            "--tools",
            "read_file,exec_shell_command",
            "--tools-max-concurrent",
            "2",
            "--tools-max-host-bytes",
            "1048576",
            "--tools-allow-origin",
            "https://trusted.invalid",
        ]);
        let config = server_config_from_args(&enabled.serve);
        assert_eq!(
            config.tools.enabled,
            vec![
                crate::BuiltinTool::ReadFile,
                crate::BuiltinTool::ExecShellCommand
            ]
        );
        assert_eq!(config.tools.max_concurrent.get(), 2);
        assert_eq!(config.tools.max_host_bytes.get(), 1048576);
        assert_eq!(config.tools.allow_origins, vec!["https://trusted.invalid"]);
        assert!(TestCli::try_parse_from(["test", "model-dir", "--tools", "unknown"]).is_err());
        for flag in ["--tools-max-concurrent", "--tools-max-host-bytes"] {
            assert!(TestCli::try_parse_from(["test", "model-dir", flag, "0"]).is_err());
        }
    }

    #[test]
    fn mcp_proxy_flags_require_explicit_enable_and_positive_limits() {
        let disabled = TestCli::parse_from(["test", "model-dir"]);
        assert_eq!(
            server_config_from_args(&disabled.serve).mcp_proxy_policy,
            crate::McpProxyPolicy::Disabled
        );
        assert!(
            TestCli::try_parse_from(["test", "model-dir", "--ui-mcp-proxy-allow-private-urls"])
                .is_err()
        );
        let public = TestCli::parse_from(["test", "model-dir", "--ui-mcp-proxy"]);
        assert_eq!(
            server_config_from_args(&public.serve).mcp_proxy_policy,
            crate::McpProxyPolicy::Public
        );
        let private = TestCli::parse_from([
            "test",
            "model-dir",
            "--ui-mcp-proxy",
            "--ui-mcp-proxy-allow-private-urls",
        ]);
        assert_eq!(
            server_config_from_args(&private.serve).mcp_proxy_policy,
            crate::McpProxyPolicy::AllowPrivate
        );
        for flag in [
            "--ui-mcp-proxy-max-bytes",
            "--ui-mcp-proxy-max-connections",
            "--ui-mcp-proxy-timeout-secs",
        ] {
            assert!(TestCli::try_parse_from(["test", "model-dir", flag, "0"]).is_err());
        }
    }

    /// Every subcommand takes the model as the first positional argument, and
    /// this one is no exception: a command line learned from `run` works here.
    #[test]
    fn the_model_is_given_positionally_like_every_other_subcommand() {
        let mut args = TestCli::parse_from(["test", "model-dir"]).serve;

        assert_eq!(args.model.take(), Some(PathBuf::from("model-dir")));
    }

    /// A single-model server serves one model. A second path is a typo or a
    /// misremembered multi-model flag, and starting anyway serves a model
    /// nobody asked for.
    #[test]
    fn a_second_model_path_is_refused_rather_than_ignored() {
        TestCli::try_parse_from(["test", "one", "two"])
            .expect_err("two different models were named");
    }

    #[test]
    fn ort_session_worker_count_is_explicit_bounded_and_defaults_to_one() {
        let default = TestCli::parse_from(["test", "model-dir"]);
        assert_eq!(default.serve.ort_session_workers.get(), 1);

        let parallel = TestCli::parse_from(["test", "model-dir", "--ort-session-workers", "2"]);
        assert_eq!(parallel.serve.ort_session_workers.get(), 2);

        for invalid in ["0", "65"] {
            TestCli::try_parse_from(["test", "model-dir", "--ort-session-workers", invalid])
                .expect_err("worker count outside 1..=64 must be refused");
        }
    }

    /// `--model-id` names *the* model, so a multi-model source has nothing for
    /// it to name. Silently ignoring it hid a typo'd invocation.
    #[tokio::test]
    async fn a_model_id_is_refused_alongside_a_multi_model_source() {
        let args =
            TestCli::parse_from(["test", "--models-dir", "does-not-exist", "--model-id", "x"])
                .serve;

        let error = run_serve(args).await.expect_err("no single model to name");
        assert!(error.to_string().contains("--model-id"), "{error}");
    }

    /// `--device` is the spelling every other subcommand uses, and naming a
    /// device has always implied the native decoder on this command.
    #[cfg(feature = "native-backend")]
    #[test]
    fn the_shared_device_flag_selects_the_native_backend() {
        let cli = TestCli::parse_from(["test", "model-dir", "--device", "cpu"]);

        let config = server_config_from_args(&cli.serve).engine_config;
        assert_eq!(
            config.decode_backend,
            onnx_genai_engine::EngineDecodeBackend::Native
        );
        assert_eq!(
            config.native_device,
            Some(onnx_genai_engine::NativeDecodeDevice::Cpu)
        );
    }

    /// #1064: `--max-batch N` must *shape the decode session*, not merely be    /// checked against a capability derived from a session built for one
    /// sequence. Before this, batch-N could only be enabled by
    /// `ONNX_GENAI_NATIVE_DECODE_BATCH`, so `--max-batch 4` was refused at
    /// startup with "this backend decodes at most 1 sequence(s) concurrently".
    #[cfg(feature = "native-backend")]
    #[test]
    fn serve_max_batch_requests_the_native_decode_batch_extent() {
        let cli = TestCli::parse_from(["test", "model-dir", "--max-batch", "4"]);

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
        let omitted = TestCli::parse_from(["test", "model-dir"]);
        let explicit_one = TestCli::parse_from(["test", "model-dir", "--max-batch", "1"]);

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
        let cli = TestCli::parse_from(["test", "model-dir", "--vram-limit", "6000000000"]);

        let config = server_config_from_args(&cli.serve);

        assert_eq!(
            config.engine_config.limits.vram_limit,
            ResourceLimit::Bytes(6_000_000_000)
        );
    }

    #[test]
    fn serve_vram_limit_accepts_fraction_and_auto() {
        let fraction = TestCli::parse_from(["test", "model-dir", "--vram-limit", "0.5"]);
        let auto = TestCli::parse_from(["test", "model-dir", "--vram-limit", "auto"]);

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
    #[test]
    fn native_prompt_batch_limit_is_positive_and_flows_to_server_config() {
        let cli = TestCli::parse_from(["test", "model-dir", "--max-completion-prompts", "3"]);
        assert_eq!(
            server_config_from_args(&cli.serve)
                .max_completion_prompts
                .get(),
            3
        );
        let error = TestCli::try_parse_from(["test", "model-dir", "--max-completion-prompts", "0"])
            .unwrap_err();
        assert!(error.to_string().contains("--max-completion-prompts"));
    }
    #[test]
    fn reranking_resource_limits_are_positive_and_flow_to_server_config() {
        let cli = TestCli::parse_from([
            "test",
            "model-dir",
            "--max-rerank-input-tokens",
            "256",
            "--max-rerank-documents",
            "8",
        ]);
        let config = server_config_from_args(&cli.serve);
        assert_eq!(config.max_rerank_input_tokens.get(), 256);
        assert_eq!(config.max_rerank_documents.get(), 8);
        for flag in ["--max-rerank-input-tokens", "--max-rerank-documents"] {
            let error = TestCli::try_parse_from(["test", "model-dir", flag, "0"]).unwrap_err();
            assert!(error.to_string().contains(flag));
        }
    }
}
