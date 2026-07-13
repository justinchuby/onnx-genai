use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use onnx_genai_engine::FimConfig;
use onnx_genai_ort::{ChatTemplate, Tokenizer};

use crate::{
    audio_input::AudioInputSpec,
    driver::EngineDriver,
    image_input::VisionInputSpec,
};

/// All per-model state bundled together.
///
/// Wrapped in `Arc` inside `ModelRegistry` so that route handlers can hold a
/// cheap clone of the pointer while the registry itself is also cheaply cloned
/// by Axum's `State` extractor.
pub(crate) struct ModelHandle {    pub(crate) id: String,
    pub(crate) engine: EngineDriver,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) chat_template: Option<Arc<ChatTemplate>>,
    pub(crate) model_max_context: Option<usize>,
    pub(crate) fim_config: Option<FimConfig>,
    pub(crate) pipeline: bool,
    pub(crate) vision_input: Option<VisionInputSpec>,
    pub(crate) audio_input: Option<AudioInputSpec>,
    /// Epoch-millisecond timestamp of the last call to `ModelRegistry::resolve`.
    /// Initialised to construction time; updated on every resolve for future LRU eviction.
    pub(crate) last_request_at: AtomicU64,
}

impl ModelHandle {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: String,
        engine: EngineDriver,
        tokenizer: Arc<Tokenizer>,
        chat_template: Option<Arc<ChatTemplate>>,
        model_max_context: Option<usize>,
        fim_config: Option<FimConfig>,
        pipeline: bool,
        vision_input: Option<VisionInputSpec>,
        audio_input: Option<AudioInputSpec>,
    ) -> Self {
        Self {
            id,
            engine,
            tokenizer,
            chat_template,
            model_max_context,
            fim_config,
            pipeline,
            vision_input,
            audio_input,
            last_request_at: AtomicU64::new(now_millis()),
        }
    }
}

/// Registry of loaded models, keyed by model id.
///
/// For Milestone 1 the registry always contains exactly one entry (the eagerly
/// loaded startup model).  The API is designed to accommodate multiple entries
/// in later milestones without breaking callers.
#[derive(Clone)]
pub(crate) struct ModelRegistry {
    // pub(crate) so the test helper in state.rs can swap handles directly.
    pub(crate) models: HashMap<String, Arc<ModelHandle>>,
}

impl ModelRegistry {
    pub(crate) fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Insert (or replace) a model handle keyed by `handle.id`.
    pub(crate) fn insert(&mut self, handle: Arc<ModelHandle>) {
        self.models.insert(handle.id.clone(), handle);
    }

    /// Resolve a handle by name, updating `last_request_at`.
    ///
    /// If `requested` is empty or no entry with that id exists, falls back to
    /// the sole/first loaded model.  This preserves today's single-model
    /// behaviour where `request.model` is not yet used for routing.
    pub(crate) fn resolve(&self, requested: &str) -> Option<Arc<ModelHandle>> {
        let handle = if !requested.is_empty() {
            self.models
                .get(requested)
                .or_else(|| self.models.values().next())
        } else {
            self.models.values().next()
        }?;
        handle.last_request_at.store(now_millis(), Ordering::Relaxed);
        Some(Arc::clone(handle))
    }

    /// Returns the ids of all loaded models.
    pub(crate) fn ids(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Returns the id of the first loaded model, or `None` if the registry is empty.
    pub(crate) fn default_id(&self) -> Option<&str> {
        self.models.keys().next().map(String::as_str)
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
