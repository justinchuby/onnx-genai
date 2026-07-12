use std::{path::Path, sync::Arc};

use anyhow::Context;
use onnx_genai::{Engine, EngineConfig};
use onnx_genai_engine::FimConfig;
use onnx_genai_ort::{ChatTemplate, ModelDirectory, Tokenizer};

use crate::{driver::EngineDriver, session::SessionRegistry};

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
        let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let chat_template = load_chat_template(model_dir)?;
        let engine = Engine::from_dir(model_dir, EngineConfig::default())?;
        let model_id = model_id.unwrap_or_else(|| infer_model_id(model_dir));
        Ok(Self::new_with_template_and_config(
            model_id,
            engine,
            tokenizer,
            chat_template,
            config,
            model_max_context,
        ))
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
