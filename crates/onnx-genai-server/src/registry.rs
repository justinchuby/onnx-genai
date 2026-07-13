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
/// Milestone 2 supports multiple entries loaded at startup from a `--models-dir`
/// or `--models-config` file.  The single-`--model` startup path remains
/// unchanged and always produces exactly one entry.
///
/// Insertion order is tracked explicitly so that `resolve("")`, `default_id()`,
/// and `ids()` are all deterministic regardless of HashMap iteration order.
#[derive(Clone)]
pub(crate) struct ModelRegistry {
    pub(crate) models: HashMap<String, Arc<ModelHandle>>,
    /// Ids in insertion order (first insert wins for each id).
    order: Vec<String>,
    /// Id of the first-inserted model; set once and never overwritten.
    default_id: Option<String>,
}

impl ModelRegistry {
    pub(crate) fn new() -> Self {
        Self {
            models: HashMap::new(),
            order: Vec::new(),
            default_id: None,
        }
    }

    /// Insert a model handle keyed by `handle.id`.
    ///
    /// If the id is new, it is appended to the insertion-order list and, if
    /// this is the very first model, recorded as the default.  Reinserting an
    /// existing id updates the stored handle without changing order or default.
    pub(crate) fn insert(&mut self, handle: Arc<ModelHandle>) {
        let id = handle.id.clone();
        if !self.models.contains_key(&id) {
            if self.default_id.is_none() {
                self.default_id = Some(id.clone());
            }
            self.order.push(id.clone());
        }
        self.models.insert(id, handle);
    }

    /// Replace an existing handle in-place, preserving insertion order and
    /// `default_id`.  Panics if `handle.id` is not already registered.
    ///
    /// Use this instead of `remove` + `insert` when the id must not change
    /// position (e.g. test helpers that swap in a patched handle).
    #[cfg(test)]
    pub(crate) fn replace(&mut self, handle: Arc<ModelHandle>) {
        assert!(
            self.models.contains_key(&handle.id),
            "replace: model id '{}' is not registered",
            handle.id
        );
        self.models.insert(handle.id.clone(), handle);
    }

    /// Resolve a handle by name, updating `last_request_at`.
    ///
    /// Behaviour differs by whether `requested` is empty/whitespace:
    ///
    /// - **Empty / whitespace** — falls back to the first-inserted model
    ///   (deterministic; preserves single-model UX and the lenient "omit model"
    ///   case from the OpenAI spec).
    /// - **Non-empty** — looks up the exact id.  Returns `None` if not found;
    ///   the caller maps this to a 404 (see `routes::resolve_model`).
    ///
    /// This is the M2 change: named-but-unknown model ids no longer silently
    /// fall back to the default model.
    pub(crate) fn resolve(&self, requested: &str) -> Option<Arc<ModelHandle>> {
        let handle = if !requested.trim().is_empty() {
            self.models.get(requested)?
        } else {
            let default = self.default_id.as_deref()?;
            self.models.get(default)?
        };
        handle.last_request_at.store(now_millis(), Ordering::Relaxed);
        Some(Arc::clone(handle))
    }

    /// Returns the ids of all loaded models in **insertion order**.
    pub(crate) fn ids(&self) -> Vec<String> {
        self.order.clone()
    }

    /// Returns the id of the first-inserted model, or `None` if the registry
    /// is empty.
    pub(crate) fn default_id(&self) -> Option<&str> {
        self.default_id.as_deref()
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use tokio::sync::{Semaphore, mpsc};

    use super::*;
    use crate::driver::EngineDriver;

    /// Build a minimal `ModelHandle` stub backed by the tiny-llm tokenizer fixture.
    /// The stub has a dead command channel (no engine thread); it is only used to
    /// exercise registry ordering — never for actual generation.
    fn stub_handle(id: &str, tokenizer: Arc<Tokenizer>) -> Arc<ModelHandle> {
        let (tx, _rx) = mpsc::channel(1);
        Arc::new(ModelHandle {
            id: id.to_string(),
            engine: EngineDriver {
                commands: tx,
                generation_capacity: Arc::new(Semaphore::new(0)),
            },
            tokenizer,
            chat_template: None,
            model_max_context: None,
            fim_config: None,
            pipeline: false,
            vision_input: None,
            audio_input: None,
            last_request_at: AtomicU64::new(0),
        })
    }

    fn load_tokenizer() -> Arc<Tokenizer> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json");
        Arc::new(Tokenizer::from_file(&path).expect("load test tokenizer"))
    }

    /// Rust's default SipHash randomises per-process, so with 5 ids that span
    /// the full alphabet a single run will expose any reliance on HashMap order.
    /// The ids are chosen so their lexicographic order differs from insertion
    /// order, making any accidental alphabetical sort immediately visible.
    #[test]
    fn registry_insertion_order_is_deterministic() {
        let tokenizer = load_tokenizer();
        let ids = ["gamma", "alpha", "delta", "beta", "epsilon"];

        let mut registry = ModelRegistry::new();
        for id in &ids {
            registry.insert(stub_handle(id, Arc::clone(&tokenizer)));
        }

        // resolve("") must always return the first-inserted model.
        let resolved = registry.resolve("").expect("resolve empty should succeed");
        assert_eq!(
            resolved.id, "gamma",
            "resolve(\"\") must return the first-inserted model (\"gamma\"), got \"{}\"",
            resolved.id,
        );

        // default_id() must be the first-inserted id.
        assert_eq!(registry.default_id(), Some("gamma"));

        // ids() must be in insertion order, not HashMap order.
        assert_eq!(
            registry.ids(),
            ids,
            "ids() must return ids in insertion order",
        );
    }

    /// Replacing a handle must not disturb insertion order or default_id.
    #[test]
    fn registry_replace_preserves_order_and_default() {
        let tokenizer = load_tokenizer();
        let ids = ["a", "b", "c"];

        let mut registry = ModelRegistry::new();
        for id in &ids {
            registry.insert(stub_handle(id, Arc::clone(&tokenizer)));
        }

        // Replace "b" — order and default must be unchanged.
        registry.replace(stub_handle("b", Arc::clone(&tokenizer)));

        assert_eq!(registry.ids(), vec!["a", "b", "c"]);
        assert_eq!(registry.default_id(), Some("a"));
        assert!(registry.resolve("b").is_some());
    }

    /// Re-inserting the same id must not append it to the order list.
    #[test]
    fn registry_reinsert_does_not_duplicate_order() {
        let tokenizer = load_tokenizer();
        let mut registry = ModelRegistry::new();
        registry.insert(stub_handle("x", Arc::clone(&tokenizer)));
        registry.insert(stub_handle("x", Arc::clone(&tokenizer))); // same id again
        registry.insert(stub_handle("y", Arc::clone(&tokenizer)));

        assert_eq!(registry.ids(), vec!["x", "y"]);
        assert_eq!(registry.default_id(), Some("x"));
    }
}
