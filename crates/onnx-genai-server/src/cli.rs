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
        .args(["model", "models_dir", "models_config"])
))]
#[command(group(
    ArgGroup::new("startup_source")
        .required(true)
        .multiple(true)
        .args(["model", "models_dir", "models_config", "model_cache_dir"])
))]
pub struct ServeArgs {
    /// API keys as comma-separated CSV; repeat the option to add keys.
    #[arg(long, env = "LLAMA_API_KEY", hide_env_values = true)]
    pub api_key: Vec<String>,
    /// UTF-8 file with one key per line; empty lines and # comments are ignored.
    #[arg(long, env = "LLAMA_ARG_API_KEY_FILE")]
    pub api_key_file: Option<PathBuf>,

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

    /// Operator-owned directory enabling background Hub package downloads and deletion.
    /// May be used alone to start an empty model catalog, or alongside configured models.
    #[arg(long, env = "NXRT_MODEL_CACHE_DIR")]
    pub model_cache_dir: Option<PathBuf>,
    /// Total managed-cache byte bound, including incomplete downloads and manifests.
    #[arg(long, default_value = "68719476736")]
    pub model_cache_max_bytes: std::num::NonZeroU64,
    /// Total managed-cache regular file bound, including manifests.
    #[arg(long, default_value = "4096")]
    pub model_cache_max_files: std::num::NonZeroUsize,
    /// Maximum published and incomplete models in the managed cache.
    #[arg(long, default_value = "128")]
    pub model_cache_max_models: std::num::NonZeroUsize,
    /// Maximum concurrent model metadata resolutions and file downloads.
    #[arg(long, default_value = "2")]
    pub model_downloads: std::num::NonZeroUsize,

    /// Node-level identifier reported by GET /v1/status for the cluster router (§34.8).
    /// Model-agnostic: it names this server process, not any model.
    /// Falls back to ONNX_GENAI_NODE_ID, then to the hostname or a generated id.
    #[arg(long, env = "ONNX_GENAI_NODE_ID")]
    pub node_id: Option<String>,

    /// Socket address to bind.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub addr: SocketAddr,

    /// Maximum decoded output tokens per completion or file transcription. Falls back to ONNX_GENAI_MAX_OUTPUT_TOKENS.
    #[arg(long, env = "ONNX_GENAI_MAX_OUTPUT_TOKENS", default_value_t = 4096)]
    pub max_output_tokens: usize,
    /// Maximum tokens in a generative transcription decoder prompt.
    #[arg(long, default_value_t = onnx_genai_engine::transcription::generative::PromptLimits::default().max_input_tokens)]
    pub max_transcription_input_tokens: std::num::NonZeroUsize,
    /// Maximum rendered transcription prompt bytes.
    #[arg(long, default_value_t = onnx_genai_engine::transcription::generative::PromptLimits::default().template.max_output_bytes)]
    pub max_transcription_prompt_bytes: std::num::NonZeroUsize,
    /// Maximum interpreter operations for one transcription prompt.
    #[arg(long, default_value_t = onnx_genai_engine::transcription::generative::PromptLimits::default().template.max_operations)]
    pub max_transcription_template_operations: std::num::NonZeroU64,
    /// Maximum acoustic frames decoded per file transcription.
    #[arg(long, default_value_t = onnx_genai_engine::transcription::TranscriptionLimits::default().max_frames)]
    pub max_transcription_frames: std::num::NonZeroUsize,
    /// Maximum rendered UTF-8 bytes per file transcription.
    #[arg(long, default_value_t = onnx_genai_engine::transcription::TranscriptionLimits::default().max_text_bytes)]
    pub max_transcription_text_bytes: std::num::NonZeroUsize,
    /// Maximum completion prompts in one native or OpenAI request.
    #[arg(long, default_value = "128")]
    pub max_completion_prompts: std::num::NonZeroUsize,
    /// Maximum tokens in one reranking query/document pair.
    #[arg(long, default_value = "8192")]
    pub max_rerank_input_tokens: std::num::NonZeroUsize,
    /// Maximum number of documents in one reranking request.
    #[arg(long, default_value = "128")]
    pub max_rerank_documents: std::num::NonZeroUsize,
    /// Unique encoded image bytes retained by one reranking request.
    #[arg(long, default_value = "67108864")]
    pub max_rerank_media_bytes: std::num::NonZeroUsize,

    /// Unique encoded image bytes retained by one SystemOne request.
    #[arg(long, default_value = "67108864")]
    pub max_decision_media_bytes: std::num::NonZeroUsize,

    /// Maximum tokens per non-generative decision prompt variant.
    #[arg(long, default_value_t = onnx_genai_engine::decision::DecisionLimits::default().max_input_tokens)]
    pub max_decision_input_tokens: std::num::NonZeroUsize,
    /// Maximum rendered bytes per decision prompt variant, before tokenization.
    #[arg(long, default_value_t = onnx_genai_engine::decision::DecisionLimits::default().template.max_output_bytes)]
    pub max_decision_prompt_bytes: std::num::NonZeroUsize,
    /// Maximum Jinja interpreter operations per decision prompt variant.
    #[arg(long, default_value_t = onnx_genai_engine::decision::DecisionLimits::default().template.max_operations)]
    pub max_decision_template_operations: std::num::NonZeroU64,
    /// Maximum questions in one SystemOne decision request.
    #[arg(long, default_value = "64")]
    pub max_decision_questions: std::num::NonZeroUsize,
    /// Maximum options per decision question (also bounded by model metadata).
    #[arg(long, default_value = "128")]
    pub max_decision_options: std::num::NonZeroUsize,

    /// Maximum active/retained resumable generations (X-Conversation-Id).
    #[arg(long, default_value_t = 16)]
    pub max_replay_streams: usize,

    /// Maximum retained SSE bytes per conversation; older prefixes are evicted.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    pub replay_buffer_bytes: usize,

    /// Enable the upstream POST /props acknowledgement (no properties are changed).
    #[arg(long, env = "NXRT_PROPS")]
    pub props: bool,

    /// Idle SSE comment interval in seconds; -1 or 0 disables pings.
    #[arg(long, default_value = "30", allow_hyphen_values = true)]
    pub sse_ping_interval: crate::SsePingInterval,

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
    /// servers or with API keys/an authenticated proxy. Falls back to ONNX_GENAI_DEBUG_ENDPOINTS=1.
    #[arg(long, env = "ONNX_GENAI_DEBUG_ENDPOINTS")]
    pub enable_debug_endpoints: bool,

    /// Enable /v1/admin/models/* runtime model-management endpoints (load, unload, list).
    /// Off by default. Use only on loopback-bound servers or with API keys/an authenticated proxy.
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

fn server_config_from_args(args: &ServeArgs) -> anyhow::Result<ServerConfig> {
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

    Ok(ServerConfig {
        api_keys: crate::ApiKeys::from_sources(&args.api_key, args.api_key_file.as_deref())?,
        node_id: args.node_id.clone().unwrap_or_else(default_node_id),
        max_output_tokens: args.max_output_tokens,
        max_transcription_input_tokens: args.max_transcription_input_tokens,
        max_transcription_prompt_bytes: args.max_transcription_prompt_bytes,
        max_transcription_template_operations: args.max_transcription_template_operations,
        max_transcription_frames: args.max_transcription_frames,
        max_transcription_text_bytes: args.max_transcription_text_bytes,
        realtime: crate::RealtimeLimits::default(),
        endpoint_props: args.props,
        model_cache_dir: args.model_cache_dir.clone(),
        model_cache_max_bytes: args.model_cache_max_bytes,
        model_cache_max_files: args.model_cache_max_files,
        model_cache_max_models: args.model_cache_max_models,
        model_downloads: args.model_downloads,
        max_completion_prompts: args.max_completion_prompts,
        max_rerank_input_tokens: args.max_rerank_input_tokens,
        max_rerank_documents: args.max_rerank_documents,
        max_rerank_media_bytes: args.max_rerank_media_bytes,
        max_decision_media_bytes: args.max_decision_media_bytes,
        max_decision_input_tokens: args.max_decision_input_tokens,
        max_decision_prompt_bytes: args.max_decision_prompt_bytes,
        max_decision_template_operations: args.max_decision_template_operations,
        max_decision_questions: args.max_decision_questions,
        max_decision_options: args.max_decision_options,
        max_replay_streams: args.max_replay_streams,
        replay_buffer_bytes: args.replay_buffer_bytes,
        sse_ping_interval: args.sse_ping_interval,
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
    })
}

/// Build the server state from [`ServeArgs`] and serve until shutdown.
pub async fn run_serve(mut args: ServeArgs) -> anyhow::Result<()> {
    args.cpu.apply().map_err(anyhow::Error::msg)?;
    let server_config = server_config_from_args(&args)?;
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
            package_variant: None,
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
            Vec::new() // Explicit cache-only startup; the required startup_source group admits it.
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
    fn sse_ping_seconds_flow_from_cli_to_server_policy() {
        for value in ["-1", "0", "1", "30"] {
            let args =
                TestCli::try_parse_from(["test", "model", "--sse-ping-interval", value]).unwrap();
            assert_eq!(
                server_config_from_args(&args.serve)
                    .unwrap()
                    .sse_ping_interval,
                value.parse().unwrap()
            );
        }
        for value in ["-2", "0.5", "2147483648"] {
            assert!(
                TestCli::try_parse_from(["test", "model", "--sse-ping-interval", value]).is_err()
            );
        }
    }

    #[test]
    fn api_key_options_combine_csv_and_file_and_fail_on_unreadable_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "# comment\nfile-key\n").unwrap();
        let args = TestCli::try_parse_from([
            "test",
            "model",
            "--api-key",
            "one,two",
            "--api-key",
            "three",
            "--api-key-file",
            file.path().to_str().unwrap(),
        ])
        .unwrap();
        let config = server_config_from_args(&args.serve).unwrap();
        assert!(format!("{:?}", config.api_keys).contains("4"));
        assert!(!format!("{config:?}").contains("file-key"));
        let missing = TestCli::try_parse_from([
            "test",
            "model",
            "--api-key-file",
            "/this-api-key-file-does-not-exist",
        ])
        .unwrap();
        assert!(
            format!("{:#}", server_config_from_args(&missing.serve).unwrap_err())
                .contains("read API key file")
        );
    }
    #[test]
    fn managed_cache_startup_combines_with_one_source_and_keeps_positive_bounds() {
        let args = TestCli::try_parse_from([
            "test",
            "--model-cache-dir",
            "cache",
            "--model-cache-max-bytes",
            "8192",
            "--model-cache-max-files",
            "16",
            "--model-cache-max-models",
            "3",
            "--model-downloads",
            "2",
        ])
        .unwrap();
        let config = server_config_from_args(&args.serve).unwrap();
        assert_eq!(
            config.model_cache_dir.as_deref(),
            Some(std::path::Path::new("cache"))
        );
        assert_eq!(
            (
                config.model_cache_max_bytes.get(),
                config.model_cache_max_files.get(),
                config.model_cache_max_models.get(),
                config.model_downloads.get()
            ),
            (8192, 16, 3, 2)
        );
        assert!(TestCli::try_parse_from(["test", "model", "--model-cache-dir", "cache"]).is_ok());
        assert!(TestCli::try_parse_from(["test"]).is_err());
        assert!(
            TestCli::try_parse_from([
                "test",
                "model",
                "--models-dir",
                "models",
                "--model-cache-dir",
                "cache"
            ])
            .is_err()
        );
        for flag in [
            "--model-cache-max-bytes",
            "--model-cache-max-files",
            "--model-cache-max-models",
            "--model-downloads",
        ] {
            assert!(
                TestCli::try_parse_from(["test", "--model-cache-dir", "cache", flag, "0"]).is_err(),
                "{flag}"
            );
        }
    }

    #[test]
    fn lora_init_without_apply_flag_preserves_explicit_operator_choice() {
        let disabled = TestCli::parse_from(["test", "model-dir", "--lora-init-without-apply"]);
        assert!(
            server_config_from_args(&disabled.serve)
                .unwrap()
                .lora_init_without_apply
        );
        let defaults = TestCli::parse_from(["test", "model-dir"]);
        assert!(
            !server_config_from_args(&defaults.serve)
                .unwrap()
                .lora_init_without_apply
        );
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
        let config = server_config_from_args(&args.serve).unwrap();
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
    fn props_acknowledgement_requires_explicit_enable() {
        let disabled = TestCli::parse_from(["test", "model-dir"]);
        assert!(
            !server_config_from_args(&disabled.serve)
                .unwrap()
                .endpoint_props
        );
        let enabled = TestCli::parse_from(["test", "model-dir", "--props"]);
        assert!(
            server_config_from_args(&enabled.serve)
                .unwrap()
                .endpoint_props
        );
    }

    #[test]
    fn tool_execution_and_proxy_options_are_not_server_capabilities() {
        for flag in [
            "--tools",
            "--tools-max-concurrent",
            "--tools-max-host-bytes",
            "--tools-allow-origin",
            "--mcp-servers-config",
            "--mcp-servers-json",
            "--mcp-max-message-bytes",
            "--tools-runtime",
            "--ui-mcp-proxy",
            "--ui-mcp-proxy-allow-private-urls",
            "--ui-mcp-proxy-max-bytes",
            "--ui-mcp-proxy-max-connections",
            "--ui-mcp-proxy-timeout-secs",
        ] {
            let error = TestCli::try_parse_from(["test", "model-dir", flag]).unwrap_err();
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::UnknownArgument,
                "{flag}: {error}"
            );
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

        let config = server_config_from_args(&cli.serve).unwrap().engine_config;
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
                .unwrap()
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
                .unwrap()
                .engine_config
                .native_decode_batch,
            None
        );
        assert_eq!(
            server_config_from_args(&explicit_one.serve)
                .unwrap()
                .engine_config
                .native_decode_batch,
            None
        );
    }

    #[test]
    fn serve_vram_limit_flows_into_engine_config_before_load() {
        let cli = TestCli::parse_from(["test", "model-dir", "--vram-limit", "6000000000"]);

        let config = server_config_from_args(&cli.serve).unwrap();

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
                .unwrap()
                .engine_config
                .limits
                .vram_limit,
            ResourceLimit::Fraction(0.5)
        );
        assert_eq!(
            server_config_from_args(&auto.serve)
                .unwrap()
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
                .unwrap()
                .max_completion_prompts
                .get(),
            3
        );
        let error = TestCli::try_parse_from(["test", "model-dir", "--max-completion-prompts", "0"])
            .unwrap_err();
        assert!(error.to_string().contains("--max-completion-prompts"));
    }
    #[test]
    fn transcription_resource_limits_are_positive_and_forwarded() {
        let cli = TestCli::parse_from([
            "test",
            "model-dir",
            "--max-transcription-frames",
            "17",
            "--max-transcription-text-bytes",
            "512",
            "--max-transcription-input-tokens",
            "13",
            "--max-transcription-prompt-bytes",
            "1024",
            "--max-transcription-template-operations",
            "99",
        ]);
        let config = server_config_from_args(&cli.serve).unwrap();
        assert_eq!(config.max_transcription_frames.get(), 17);
        assert_eq!(config.max_transcription_text_bytes.get(), 512);
        assert_eq!(config.max_transcription_input_tokens.get(), 13);
        assert_eq!(config.max_transcription_prompt_bytes.get(), 1024);
        assert_eq!(config.max_transcription_template_operations.get(), 99);
        for flag in [
            "--max-transcription-input-tokens",
            "--max-transcription-prompt-bytes",
            "--max-transcription-template-operations",
            "--max-transcription-frames",
            "--max-transcription-text-bytes",
        ] {
            assert!(TestCli::try_parse_from(["test", "model-dir", flag, "0"]).is_err());
        }
    }
    #[test]
    fn decision_resource_limits_are_positive_and_forwarded() {
        let args = TestCli::parse_from([
            "test",
            "model-dir",
            "--max-decision-media-bytes",
            "4096",
            "--max-decision-input-tokens",
            "256",
            "--max-decision-questions",
            "3",
            "--max-decision-options",
            "8",
        ]);
        let config = server_config_from_args(&args.serve).unwrap();
        assert_eq!(config.max_decision_media_bytes.get(), 4096);
        assert_eq!(config.max_decision_input_tokens.get(), 256);
        assert_eq!(config.max_decision_questions.get(), 3);
        assert_eq!(config.max_decision_options.get(), 8);
        for flag in [
            "--max-decision-media-bytes",
            "--max-decision-input-tokens",
            "--max-decision-questions",
            "--max-decision-prompt-bytes",
            "--max-decision-template-operations",
            "--max-decision-options",
        ] {
            assert!(
                TestCli::try_parse_from(["test", "model-dir", flag, "0"]).is_err(),
                "{flag}"
            );
        }
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
            "--max-rerank-media-bytes",
            "2048",
        ]);
        let config = server_config_from_args(&cli.serve).unwrap();
        assert_eq!(config.max_rerank_input_tokens.get(), 256);
        assert_eq!(config.max_rerank_documents.get(), 8);
        assert_eq!(config.max_rerank_media_bytes.get(), 2048);
        for flag in [
            "--max-rerank-input-tokens",
            "--max-rerank-documents",
            "--max-rerank-media-bytes",
        ] {
            let error = TestCli::try_parse_from(["test", "model-dir", flag, "0"]).unwrap_err();
            assert!(error.to_string().contains(flag));
        }
    }
}
