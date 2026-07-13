use std::{path::Path, sync::Arc, time::Instant};

use anyhow::Context;
use onnx_genai::{Engine, EngineConfig};
use onnx_genai_engine::KvDType;
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
    registry::{ModelHandle, ModelRegistry},
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

#[derive(Clone)]
pub struct AppState {
    pub(crate) registry: ModelRegistry,
    pub(crate) sessions: SessionRegistry,
    pub(crate) config: ServerConfig,
    pub(crate) started_at: Instant,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub max_output_tokens: usize,
    pub max_sessions: usize,
    /// Maximum generation requests admitted to the driver, including active and queued work.
    pub max_queue_depth: usize,
    /// Enable the /v1/debug/* introspection endpoints. Off by default; enable with
    /// `--enable-debug-endpoints` or `ONNX_GENAI_DEBUG_ENDPOINTS=1`. These endpoints
    /// expose server internals and should only be used on loopback-bound instances or
    /// behind an authenticated reverse proxy.
    pub enable_debug_endpoints: bool,
    /// Engine configuration, including KV cache storage dtype.
    pub engine_config: EngineConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            max_sessions: DEFAULT_MAX_SESSIONS,
            max_queue_depth: DEFAULT_MAX_QUEUE_DEPTH,
            enable_debug_endpoints: false,
            engine_config: EngineConfig::default(),
        }
    }
}

impl ServerConfig {
    fn validate(self) -> anyhow::Result<Self> {
        if self.max_output_tokens == 0 {
            anyhow::bail!("max_output_tokens must be greater than zero");
        }
        if self.max_sessions == 0 {
            anyhow::bail!("max_sessions must be greater than zero");
        }
        if self.max_queue_depth == 0 {
            anyhow::bail!("max_queue_depth must be greater than zero");
        }
        Ok(self)
    }
}
impl AppState {
    pub fn load(model_dir: &Path, model_id: Option<String>) -> anyhow::Result<Self> {
        Self::load_with_config(model_dir, model_id, ServerConfig::default())
    }

    pub fn load_with_config(
        model_dir: &Path,
        model_id: Option<String>,
        config: ServerConfig,
    ) -> anyhow::Result<Self> {
        let config = config.validate()?;
        let model_directory = ModelDirectory::load(model_dir)
            .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {}", e))?;
        let model_max_context = load_model_max_context(model_directory.metadata_path.as_deref())?;
        let chat_template = load_chat_template(model_dir)?;
        let model_id = model_id.unwrap_or_else(|| infer_model_id(model_dir));
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
            return Self::load_pipeline(
                model_dir,
                model_id,
                config,
                model_max_context,
                chat_template,
            );
        }

        let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let engine = Engine::from_dir(model_dir, config.engine_config.clone())?;
        Ok(Self::new_with_template_and_config(
            model_id,
            engine,
            tokenizer,
            chat_template,
            config,
            model_max_context,
        ))
    }

    fn load_pipeline(
        model_dir: &Path,
        model_id: String,
        config: ServerConfig,
        model_max_context: Option<usize>,
        chat_template: Option<ChatTemplate>,
    ) -> anyhow::Result<Self> {
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
        let handle = ModelHandle::new(
            model_id,
            EngineDriver::start_pipeline(engine, config.max_queue_depth),
            Arc::new(tokenizer),
            chat_template.map(Arc::new),
            model_max_context,
            None,
            true,
            vision_input,
            audio_input,
        );
        let mut registry = ModelRegistry::new();
        registry.insert(Arc::new(handle));
        Ok(Self {
            registry,
            sessions: SessionRegistry::new(config.max_sessions),
            config,
            started_at: Instant::now(),
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
        let mut registry = ModelRegistry::new();
        registry.insert(Arc::new(handle));
        Self {
            registry,
            sessions: SessionRegistry::new(config.max_sessions),
            config,
            started_at: Instant::now(),
        }
    }

    /// Returns the id of the first loaded model, for use in log messages and the CLI.
    pub fn model_id(&self) -> &str {
        self.registry.default_id().unwrap_or("onnx-genai-model")
    }
}

#[cfg(test)]
impl AppState {
    /// Replace the fim_config of the default (sole) loaded model.
    ///
    /// Used in tests that need FIM without a real model that declares FIM tokens.
    pub(crate) fn with_default_fim_config(mut self, fim_config: Option<FimConfig>) -> Self {
        let id = self
            .registry
            .default_id()
            .expect("registry must have a model")
            .to_string();
        let old_arc = self
            .registry
            .models
            .remove(&id)
            .expect("default model must exist");
        let old = Arc::try_unwrap(old_arc)
            .unwrap_or_else(|_| panic!("unique handle ownership during test setup"));
        let new_handle = Arc::new(ModelHandle { fim_config, ..old });
        self.registry.insert(new_handle);
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
