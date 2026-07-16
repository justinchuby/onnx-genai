use std::{path::Path, sync::Arc};

use anyhow::Context;
use onnx_genai::{Engine, EngineConfig};
use onnx_genai_engine::KvDType;
#[cfg(feature = "native-backend")]
use onnx_genai_engine::NativeDecodeDevice;
use onnx_genai_metadata::PipelineStrategy;
use onnx_genai_ort::{
    ChatTemplate, DataType, ModelDirectory, PipelineModelDirectory, PipelineModels, Tokenizer,
};

#[cfg(test)]
use onnx_genai_engine::FimConfig;

use crate::{
    audio_input::AudioInputSpec,
    driver::EngineDriver,
    image_input::VisionInputSpec,
    models_config::ModelSpec,
    registry::{EvictionPolicy, ModelHandle, ModelRegistry},
    session::SessionRegistry,
};

const DEFAULT_MAX_OUTPUT_TOKENS: usize = 4096;
const DEFAULT_MAX_SESSIONS: usize = 256;
const DEFAULT_MAX_QUEUE_DEPTH: usize = 256;
const DEFAULT_MAX_BATCH: usize = 4;

/// Parse a user-supplied KV cache dtype string.
///
/// Extends `KvDType::from_metadata_name` with the terse `"f32"` alias that is
/// the canonical default for the `--kv-cache-dtype` flag.
pub fn parse_kv_cache_dtype(s: &str) -> Result<KvDType, String> {
    let lower = s.trim().to_ascii_lowercase();
    let normalised = match lower.as_str() {
        "f32" => "float32",
        other => other,
    };
    KvDType::from_metadata_name(normalised).map_err(|_| {
        format!("invalid KV cache dtype '{s}'; accepted values: f32, int8, fp8_e4m3fn, fp8_e5m2")
    })
}

/// Parse a native decoder device as `cpu`, `cuda`, or `cuda:<index>`.
#[cfg(feature = "native-backend")]
pub fn parse_native_device(s: &str) -> Result<NativeDecodeDevice, String> {
    let value = s.trim().to_ascii_lowercase();
    if value == "cpu" {
        return Ok(NativeDecodeDevice::Cpu);
    }
    if value == "cuda" {
        return parse_native_cuda_device(None);
    }
    if let Some(index) = value.strip_prefix("cuda:") {
        let index = index
            .parse::<u32>()
            .map_err(|_| format!("invalid native device '{s}'; CUDA index must be a u32"))?;
        return parse_native_cuda_device(Some(index));
    }
    Err(format!(
        "invalid native device '{s}'; accepted values: cpu, cuda, cuda:<index>"
    ))
}

#[cfg(all(feature = "native-backend", feature = "cuda"))]
fn parse_native_cuda_device(index: Option<u32>) -> Result<NativeDecodeDevice, String> {
    Ok(NativeDecodeDevice::Cuda { index })
}

#[cfg(all(feature = "native-backend", not(feature = "cuda")))]
fn parse_native_cuda_device(_index: Option<u32>) -> Result<NativeDecodeDevice, String> {
    Err("native CUDA requires building onnx-genai-server with the 'cuda' feature".to_string())
}

#[derive(Clone)]
pub struct AppState {
    pub(crate) registry: ModelRegistry,
    pub(crate) sessions: SessionRegistry,
    pub(crate) config: ServerConfig,
}

/// Resolve a default node identifier for the §34 cluster router's node-status
/// contract. This is a NODE-level id, independent of any loaded model.
///
/// Resolution order: the host's name (`HOSTNAME`/`COMPUTERNAME`), else a stable
/// random `node-<hex>` id generated from the OS CSPRNG. Never derived from a model.
pub fn default_node_id() -> String {
    if let Some(host) = std::env::var_os("HOSTNAME")
        .or_else(|| std::env::var_os("COMPUTERNAME"))
        .and_then(|value| value.into_string().ok())
    {
        let host = host.trim();
        if !host.is_empty() {
            return host.to_string();
        }
    }
    let mut bytes = [0_u8; 8];
    if getrandom::fill(&mut bytes).is_ok() {
        return format!("node-{}", hex(&bytes));
    }
    "node".to_string()
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Node-level identifier reported by `GET /v1/status` for the cluster router
    /// (§34.8). Independent of any model; defaults to the hostname or a generated id.
    pub node_id: String,
    pub max_output_tokens: usize,
    pub max_sessions: usize,
    /// Maximum generation requests admitted to the driver, including active and queued work.
    pub max_queue_depth: usize,
    /// Enable the /v1/debug/* introspection endpoints. Off by default; enable with
    /// `--enable-debug-endpoints` or `ONNX_GENAI_DEBUG_ENDPOINTS=1`. These endpoints
    /// expose server internals and should only be used on loopback-bound instances or
    /// behind an authenticated reverse proxy.
    pub enable_debug_endpoints: bool,
    /// Enable the /v1/admin/models/* runtime model-management endpoints. Off by
    /// default; enable with `--enable-admin-endpoints` or `ONNX_GENAI_ADMIN_ENDPOINTS=1`.
    /// These endpoints load and unload models at runtime and should only be exposed
    /// on loopback-bound instances or behind an authenticated reverse proxy.
    pub enable_admin_endpoints: bool,
    /// Maximum number of models kept loaded in memory at once. `None` (the default)
    /// means unlimited. When set, loading an additional model beyond the cap evicts
    /// the least-recently-used loaded model (never dropping below one model).
    pub max_loaded_models: Option<usize>,
    /// Policy used to pick an eviction victim when `max_loaded_models` is exceeded.
    pub eviction_policy: EvictionPolicy,
    /// Engine configuration, including KV cache storage dtype.
    pub engine_config: EngineConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            node_id: default_node_id(),
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            max_sessions: DEFAULT_MAX_SESSIONS,
            max_queue_depth: DEFAULT_MAX_QUEUE_DEPTH,
            enable_debug_endpoints: false,
            enable_admin_endpoints: false,
            max_loaded_models: None,
            eviction_policy: EvictionPolicy::Lru,
            engine_config: EngineConfig::default(),
        }
    }
}

impl ServerConfig {
    fn validate(self) -> anyhow::Result<Self> {
        if self.node_id.trim().is_empty() {
            anyhow::bail!("node_id must not be empty");
        }
        if self.max_output_tokens == 0 {
            anyhow::bail!("max_output_tokens must be greater than zero");
        }
        if self.max_sessions == 0 {
            anyhow::bail!("max_sessions must be greater than zero");
        }
        if self.max_queue_depth == 0 {
            anyhow::bail!("max_queue_depth must be greater than zero");
        }
        if self.max_loaded_models == Some(0) {
            anyhow::bail!("max_loaded_models must be greater than zero when set");
        }
        Ok(self)
    }
}
impl AppState {
    pub fn load(model_dir: &Path, model_id: Option<String>) -> anyhow::Result<Self> {
        Self::load_with_config(model_dir, model_id, ServerConfig::default())
    }

    /// Load a single model from `model_dir`, wrapping it in a one-entry registry.
    ///
    /// This is the single-`--model` startup path.  The model is recorded as an
    /// eager spec so it is both loaded at startup and reloadable after an unload.
    pub fn load_with_config(
        model_dir: &Path,
        model_id: Option<String>,
        config: ServerConfig,
    ) -> anyhow::Result<Self> {
        let config = config.validate()?;
        let model_id = model_id.unwrap_or_else(|| infer_model_id(model_dir));
        let spec = ModelSpec {
            id: model_id,
            path: model_dir.to_path_buf(),
            eager: true,
        };
        let registry = ModelRegistry::from_specs(std::slice::from_ref(&spec), config.clone())?;
        Ok(Self {
            registry,
            sessions: SessionRegistry::new(config.max_sessions),
            config,
        })
    }

    /// Load multiple models from a list of `ModelSpec`s and build a multi-entry registry.
    ///
    /// **M3 loading strategy:** every spec is recorded as available.  Specs with
    /// `eager = true` are loaded at startup; `eager = false` specs are lazily loaded
    /// on first request.  The first spec becomes the default model.
    ///
    /// Fails fast if any eager spec fails to load.
    pub fn load_from_specs(specs: Vec<ModelSpec>, config: ServerConfig) -> anyhow::Result<Self> {
        if specs.is_empty() {
            anyhow::bail!("at least one model spec is required");
        }
        let config = config.validate()?;
        let registry = ModelRegistry::from_specs(&specs, config.clone())?;
        Ok(Self {
            registry,
            sessions: SessionRegistry::new(config.max_sessions),
            config,
        })
    }

    pub fn new(model_id: String, engine: Engine, tokenizer: Tokenizer) -> Self {
        Self::new_with_template(model_id, engine, tokenizer, None)
    }

    pub fn new_with_template(
        model_id: String,
        engine: Engine,
        tokenizer: Tokenizer,
        chat_template: Option<ChatTemplate>,
    ) -> Self {
        Self::new_with_template_and_config(
            model_id,
            engine,
            tokenizer,
            chat_template,
            ServerConfig::default(),
            None,
        )
    }

    fn new_with_template_and_config(
        model_id: String,
        engine: Engine,
        tokenizer: Tokenizer,
        chat_template: Option<ChatTemplate>,
        config: ServerConfig,
        model_max_context: Option<usize>,
    ) -> Self {
        let config = config.validate().expect("validated server config");
        let fim_config = engine.fim_config().cloned();
        let engine_driver = EngineDriver::start(engine, DEFAULT_MAX_BATCH, config.max_queue_depth);
        let handle = ModelHandle::new(
            model_id,
            engine_driver,
            Arc::new(tokenizer),
            chat_template.map(Arc::new),
            model_max_context,
            fim_config,
            false,
            None,
            None,
        );
        let registry = ModelRegistry::from_handle(Arc::new(handle), config.clone());
        Self {
            registry,
            sessions: SessionRegistry::new(config.max_sessions),
            config,
        }
    }

    /// Returns the id of the first loaded model, for use in log messages and the CLI.
    pub fn model_id(&self) -> String {
        self.registry
            .default_id()
            .unwrap_or_else(|| "onnx-genai-model".to_string())
    }
}

#[cfg(test)]
impl AppState {
    /// Replace the fim_config of the default (sole) loaded model.
    ///
    /// Used in tests that need FIM without a real model that declares FIM tokens.
    pub(crate) fn with_default_fim_config(self, fim_config: Option<FimConfig>) -> Self {
        self.registry.set_default_fim_config(fim_config);
        self
    }
}

fn strategy_max_tokens(strategy: &PipelineStrategy) -> Option<usize> {
    strategy.max_tokens.or_else(|| {
        strategy
            .stages
            .iter()
            .find_map(|stage| strategy_max_tokens(&stage.strategy))
    })
}

fn infer_model_id(model_dir: &Path) -> String {
    model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("onnx-genai-model")
        .to_string()
}

fn load_chat_template(model_dir: &Path) -> anyhow::Result<Option<ChatTemplate>> {
    let standalone = model_dir.join("chat_template.jinja");
    let tokenizer_config = model_dir.join("tokenizer_config.json");
    let has_template = standalone.is_file()
        || tokenizer_config.is_file()
            && std::fs::read_to_string(&tokenizer_config)
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| value.get("chat_template").cloned())
                .and_then(|value| value.as_str().map(ToString::to_string))
                .is_some();
    if has_template {
        Ok(Some(ChatTemplate::from_model_dir(model_dir)?))
    } else {
        Ok(None)
    }
}

fn load_model_max_context(metadata_path: Option<&Path>) -> anyhow::Result<Option<usize>> {
    let Some(metadata_path) = metadata_path else {
        return Ok(None);
    };
    let metadata = onnx_genai_metadata::load_metadata(metadata_path)
        .with_context(|| format!("failed to load {}", metadata_path.display()))?;
    Ok(metadata.model.and_then(|model| model.max_sequence_length))
}

/// Build one model handle (plain or pipeline) from a `ModelSpec`.
///
/// `config` must already be validated.  This is the single shared construction
/// path used by both startup (`ModelRegistry::from_specs`) and runtime lazy
/// loading (`ModelRegistry::load`).  It is a **blocking** function (it calls
/// `Engine::from_dir`, which takes seconds) and must therefore be invoked from a
/// blocking context (e.g. at startup or via `tokio::task::spawn_blocking`).
pub(crate) fn build_handle(spec: &ModelSpec, config: &ServerConfig) -> anyhow::Result<ModelHandle> {
    let model_dir = spec.path.as_path();
    let model_id = spec.id.clone();
    let model_directory = ModelDirectory::load(model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {}", e))?;
    let model_max_context = load_model_max_context(model_directory.metadata_path.as_deref())?;
    let chat_template = load_chat_template(model_dir)?;
    let metadata = model_directory
        .metadata_path
        .as_deref()
        .map(onnx_genai_metadata::load_metadata)
        .transpose()
        .with_context(|| format!("failed to load metadata from {}", model_dir.display()))?;
    if metadata
        .as_ref()
        .is_some_and(|metadata| metadata.pipeline.is_some())
    {
        return build_pipeline_handle(model_dir, model_id, config, model_max_context, chat_template);
    }
    let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    let engine = Engine::from_dir(model_dir, config.engine_config.clone())?;
    let fim_config = engine.fim_config().cloned();
    let engine_driver = EngineDriver::start(engine, DEFAULT_MAX_BATCH, config.max_queue_depth);
    Ok(ModelHandle::new(
        model_id,
        engine_driver,
        Arc::new(tokenizer),
        chat_template.map(Arc::new),
        model_max_context,
        fim_config,
        false,
        None,
        None,
    ))
}

fn build_pipeline_handle(
    model_dir: &Path,
    model_id: String,
    config: &ServerConfig,
    model_max_context: Option<usize>,
    chat_template: Option<ChatTemplate>,
) -> anyhow::Result<ModelHandle> {
    let directory = PipelineModelDirectory::load(model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to resolve pipeline directory: {e}"))?;
    let tokenizer_path = directory
        .spec
        .models
        .values()
        .find(|component| component.role == "decoder")
        .and_then(|component| component.tokenizer.as_ref())
        .map(|path| model_dir.join(path))
        .or(directory.tokenizer_paths.shared.clone())
        .context("pipeline model has no decoder or shared tokenizer")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load pipeline tokenizer: {e}"))?;

    let models = PipelineModels::load(model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to inspect pipeline models: {e}"))?;
    let vision_inputs = models
        .sessions
        .iter()
        .flat_map(|(component, session)| {
            session.inputs().iter().filter_map(move |input| {
                (input.name == "pixel_values")
                    .then_some((format!("{component}.{}", input.name), input))
            })
        })
        .collect::<Vec<_>>();
    let vision_input = match vision_inputs.as_slice() {
        [] => None,
        [(endpoint, input)] => {
            if input.dtype != DataType::Float32 {
                anyhow::bail!(
                    "vision input '{endpoint}' must be Float32, but the model declares {:?}",
                    input.dtype
                );
            }
            Some(VisionInputSpec::from_input_and_metadata(
                endpoint.clone(),
                &input.shape,
                Some(&directory.metadata_path),
            )?)
        }
        _ => anyhow::bail!("pipeline declares multiple pixel_values inputs"),
    };
    let audio_inputs = models
        .sessions
        .iter()
        .flat_map(|(component, session)| {
            session.inputs().iter().filter_map(move |input| {
                (input.name == "input_features")
                    .then_some((format!("{component}.{}", input.name), input))
            })
        })
        .collect::<Vec<_>>();
    let pipeline_max_tokens = strategy_max_tokens(&directory.spec.strategy);
    let audio_input = match audio_inputs.as_slice() {
        [] => None,
        [(endpoint, input)] => {
            if input.dtype != DataType::Float32 {
                anyhow::bail!(
                    "audio input '{endpoint}' must be Float32, but the model declares {:?}",
                    input.dtype
                );
            }
            Some(AudioInputSpec::from_input(
                endpoint.clone(),
                &input.shape,
                pipeline_max_tokens,
            )?)
        }
        _ => anyhow::bail!("pipeline declares multiple input_features inputs"),
    };
    drop(models);

    let engine = Engine::from_pipeline_dir(model_dir, config.engine_config.clone())?;
    Ok(ModelHandle::new(
        model_id,
        EngineDriver::start_pipeline(engine, config.max_queue_depth),
        Arc::new(tokenizer),
        chat_template.map(Arc::new),
        model_max_context,
        None,
        true,
        vision_input,
        audio_input,
    ))
}
