use std::{path::Path, sync::Arc, time::Instant};

use anyhow::Context;
use onnx_genai::{Engine, EngineConfig};
use onnx_genai_engine::FimConfig;
use onnx_genai_ort::{
    ChatTemplate, DataType, ModelDirectory, PipelineModelDirectory, PipelineModels, Tokenizer,
};

use crate::{driver::EngineDriver, image_input::VisionInputSpec, session::SessionRegistry};

const DEFAULT_MAX_OUTPUT_TOKENS: usize = 4096;
const DEFAULT_MAX_SESSIONS: usize = 256;
const DEFAULT_MAX_PENDING: usize = 256;
const DEFAULT_MAX_BATCH: usize = 4;

#[derive(Clone)]
pub struct AppState {
    pub(crate) model_id: String,
    pub(crate) engine: EngineDriver,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) chat_template: Option<Arc<ChatTemplate>>,
    pub(crate) sessions: SessionRegistry,
    pub(crate) config: ServerConfig,
    pub(crate) model_max_context: Option<usize>,
    pub(crate) fim_config: Option<FimConfig>,
    pub(crate) pipeline: bool,
    pub(crate) vision_input: Option<VisionInputSpec>,
    pub(crate) started_at: Instant,
}

#[derive(Debug, Clone, Copy)]
pub struct ServerConfig {
    pub max_output_tokens: usize,
    pub max_sessions: usize,
    /// Maximum generation requests admitted to the driver, including active and queued work.
    pub max_pending: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            max_sessions: DEFAULT_MAX_SESSIONS,
            max_pending: DEFAULT_MAX_PENDING,
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
        if self.max_pending == 0 {
            anyhow::bail!("max_pending must be greater than zero");
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
        let engine = Engine::from_dir(model_dir, EngineConfig::default())?;
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
        drop(models);

        let engine = Engine::from_pipeline_dir(model_dir, EngineConfig::default())?;
        Ok(Self {
            model_id,
            engine: EngineDriver::start_pipeline(engine, config.max_pending),
            tokenizer: Arc::new(tokenizer),
            chat_template: chat_template.map(Arc::new),
            sessions: SessionRegistry::new(config.max_sessions),
            config,
            model_max_context,
            fim_config: None,
            pipeline: true,
            vision_input,
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
        Self {
            model_id,
            engine: EngineDriver::start(engine, DEFAULT_MAX_BATCH, config.max_pending),
            tokenizer: Arc::new(tokenizer),
            chat_template: chat_template.map(Arc::new),
            sessions: SessionRegistry::new(config.max_sessions),
            config,
            model_max_context,
            fim_config,
            pipeline: false,
            vision_input: None,
            started_at: Instant::now(),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
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
