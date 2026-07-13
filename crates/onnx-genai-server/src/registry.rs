use std::{
    collections::HashMap,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use onnx_genai_engine::FimConfig;
use onnx_genai_ort::{ChatTemplate, Tokenizer};

use crate::{
    audio_input::AudioInputSpec,
    driver::EngineDriver,
    image_input::VisionInputSpec,
    models_config::ModelSpec,
    state::{ServerConfig, build_handle},
};

/// Policy used to choose which loaded model to evict when the loaded-model cap is
/// exceeded.  Only least-recently-used is implemented today; the enum exists so
/// future policies can be added without changing call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Evict the loaded model with the smallest `last_request_at`.
    #[default]
    Lru,
}

/// All per-model state bundled together.
///
/// Wrapped in `Arc` inside `ModelRegistry` so that route handlers can hold a
/// cheap clone of the pointer while the registry itself is also cheaply cloned
/// by Axum's `State` extractor.
pub(crate) struct ModelHandle {
    pub(crate) id: String,
    pub(crate) engine: EngineDriver,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) chat_template: Option<Arc<ChatTemplate>>,
    pub(crate) model_max_context: Option<usize>,
    pub(crate) fim_config: Option<FimConfig>,
    pub(crate) pipeline: bool,
    pub(crate) vision_input: Option<VisionInputSpec>,
    pub(crate) audio_input: Option<AudioInputSpec>,
    /// Epoch-millisecond timestamp of the last call to `ModelRegistry::resolve`.
    /// Initialised to construction time; updated on every resolve for LRU eviction.
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

/// Loaded/available status for a single model, returned by the admin listing.
#[derive(Debug, Clone)]
pub(crate) struct ModelStatus {
    pub(crate) id: String,
    pub(crate) loaded: bool,
    pub(crate) is_default: bool,
    /// `last_request_at` (epoch millis) if the model is currently loaded.
    pub(crate) last_request_at: Option<u64>,
}

/// Mutable interior of the registry, guarded by a single `RwLock`.
struct RegistryInner {
    /// Currently loaded models, keyed by id.
    models: HashMap<String, Arc<ModelHandle>>,
    /// Loaded ids in insertion order (first load wins for each id).
    order: Vec<String>,
    /// Id of the default model.  Set once at construction and never overwritten,
    /// even if the model is later unloaded (it is lazily reloaded on demand).
    default_id: Option<String>,
    /// Every configured spec, whether currently loaded or not.  Populated fully
    /// at startup so that lazy / unloaded models can be (re)loaded on demand.
    available: HashMap<String, ModelSpec>,
}

/// Registry of models, providing runtime load / unload / lazy-load with LRU
/// eviction.
///
/// The registry is a cheaply-cloneable shared handle: cloning it clones a few
/// `Arc`s, so `AppState` (and therefore every Axum request) shares one registry.
/// All mutable state lives behind `Arc<RwLock<RegistryInner>>`.
///
/// **Locking discipline:** the heavy model build (`Engine::from_dir`, tokenizer
/// and chat template) is always performed *outside* the lock via
/// `tokio::task::spawn_blocking`; the `RwLock` is only ever held for the short,
/// synchronous critical sections that mutate the maps. No lock is ever held
/// across an `.await`, so the synchronous `std::sync::RwLock` is deadlock-free here.
#[derive(Clone)]
pub(crate) struct ModelRegistry {
    inner: Arc<RwLock<RegistryInner>>,
    /// Server configuration needed to (re)build handles at runtime: engine
    /// config, queue depth, the loaded-model cap and the eviction policy.
    config: Arc<ServerConfig>,
    /// Per-id load guards, ensuring two concurrent requests for the same lazy id
    /// build the model only once; the second waiter observes the first result.
    load_locks: Arc<tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl ModelRegistry {
    /// Build a registry from a list of specs, loading the eager ones immediately.
    ///
    /// All specs (eager or not) are recorded in `available`.  Eager specs are also
    /// built and inserted into `models`; non-eager specs are left for lazy loading.
    /// The first spec in the list becomes the default model.
    ///
    /// This is a **blocking** constructor (it builds eager models synchronously)
    /// and is only called at startup.
    pub(crate) fn from_specs(specs: &[ModelSpec], config: ServerConfig) -> anyhow::Result<Self> {
        if specs.is_empty() {
            anyhow::bail!("at least one model spec is required");
        }
        let mut available = HashMap::new();
        for spec in specs {
            available.insert(spec.id.clone(), spec.clone());
        }
        let default_id = Some(specs[0].id.clone());
        let mut inner = RegistryInner {
            models: HashMap::new(),
            order: Vec::new(),
            default_id,
            available,
        };
        for spec in specs.iter().filter(|s| s.eager) {
            tracing::info!(id = %spec.id, path = %spec.path.display(), "loading model");
            let handle = build_handle(spec, &config).with_context(|| {
                format!(
                    "failed to load model '{}' from '{}'",
                    spec.id,
                    spec.path.display()
                )
            })?;
            inner.insert_loaded(Arc::new(handle));
        }
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
            config: Arc::new(config),
            load_locks: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        })
    }

    /// Build a registry around a single, already-constructed handle.
    ///
    /// Used by the `AppState::new*` constructors that start from a live `Engine`
    /// rather than a spec path.  Because there is no backing spec, the model is
    /// not recorded in `available` and therefore cannot be lazily reloaded after
    /// an unload.
    pub(crate) fn from_handle(handle: Arc<ModelHandle>, config: ServerConfig) -> Self {
        let default_id = Some(handle.id.clone());
        let mut inner = RegistryInner {
            models: HashMap::new(),
            order: Vec::new(),
            default_id,
            available: HashMap::new(),
        };
        inner.insert_loaded(handle);
        Self {
            inner: Arc::new(RwLock::new(inner)),
            config: Arc::new(config),
            load_locks: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Resolve an already-loaded handle by name, updating `last_request_at`.
    ///
    /// - **Empty / whitespace** — falls back to the default model.
    /// - **Non-empty** — looks up the exact id.
    ///
    /// Returns `None` if the target is not currently loaded (either unknown or a
    /// lazy/unloaded model).  Callers wanting lazy loading use
    /// `routes::resolve_model`, which falls through to [`ModelRegistry::load`].
    pub(crate) fn resolve(&self, requested: &str) -> Option<Arc<ModelHandle>> {
        let inner = self.inner.read().expect("registry lock poisoned");
        let handle = if !requested.trim().is_empty() {
            inner.models.get(requested)?
        } else {
            let default = inner.default_id.as_deref()?;
            inner.models.get(default)?
        };
        handle.last_request_at.store(now_millis(), Ordering::Relaxed);
        Some(Arc::clone(handle))
    }

    /// Returns `true` if `id` is a configured model (loaded or not).
    pub(crate) fn contains_available(&self, id: &str) -> bool {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .available
            .contains_key(id)
    }

    /// Returns the ids of all currently loaded models in insertion order.
    pub(crate) fn ids(&self) -> Vec<String> {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .order
            .clone()
    }

    /// Returns the id of the default model, or `None` if none is configured.
    pub(crate) fn default_id(&self) -> Option<String> {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .default_id
            .clone()
    }

    /// Snapshot of every configured model with its loaded/available status,
    /// ordered by configured id for determinism.
    pub(crate) fn statuses(&self) -> Vec<ModelStatus> {
        let inner = self.inner.read().expect("registry lock poisoned");
        let default = inner.default_id.as_deref();
        let mut statuses: Vec<ModelStatus> = inner
            .available
            .keys()
            .map(|id| {
                let loaded = inner.models.get(id);
                ModelStatus {
                    id: id.clone(),
                    loaded: loaded.is_some(),
                    is_default: default == Some(id.as_str()),
                    last_request_at: loaded.map(|h| h.last_request_at.load(Ordering::Relaxed)),
                }
            })
            .collect();
        statuses.sort_by(|a, b| a.id.cmp(&b.id));
        statuses
    }

    /// Load (or return the already-loaded) model for `id`.
    ///
    /// The heavy construction runs on a blocking thread pool via
    /// `spawn_blocking`; the registry lock is only taken for the brief insert +
    /// eviction critical section afterwards.  A per-id async guard serialises
    /// concurrent loads of the same id so the model is built only once.
    pub(crate) async fn load(&self, id: &str) -> anyhow::Result<Arc<ModelHandle>> {
        // Fast path: already loaded.
        if let Some(handle) = self.get_loaded(id) {
            return Ok(handle);
        }
        // Validate the id is configured before doing any work.
        let spec = self
            .spec_for(id)
            .ok_or_else(|| anyhow::anyhow!("unknown model id '{id}'"))?;

        // Serialise concurrent loads of the same id.
        let guard = self.load_guard(id).await;
        let _held = guard.lock().await;

        // Re-check after acquiring the guard: another waiter may have loaded it.
        if let Some(handle) = self.get_loaded(id) {
            return Ok(handle);
        }

        let config = Arc::clone(&self.config);
        let spec_for_build = spec.clone();
        tracing::info!(id = %spec.id, path = %spec.path.display(), "lazy-loading model");
        let handle = tokio::task::spawn_blocking(move || build_handle(&spec_for_build, &config))
            .await
            .context("model load task panicked")?
            .with_context(|| {
                format!(
                    "failed to load model '{}' from '{}'",
                    spec.id,
                    spec.path.display()
                )
            })?;
        let handle = Arc::new(handle);

        // Insert + evict under the write lock (no await held).
        {
            let mut inner = self.inner.write().expect("registry lock poisoned");
            inner.insert_loaded(Arc::clone(&handle));
            inner.enforce_eviction(self.config.max_loaded_models, id);
        }
        Ok(handle)
    }

    /// Unload a model: drop its handle from `models`/`order` but keep the spec in
    /// `available` so it can be lazily reloaded.  In-flight requests that already
    /// hold an `Arc<ModelHandle>` keep the engine alive until they finish.
    ///
    /// Returns an error if the id is not currently loaded (mapped to 404).
    pub(crate) fn unload(&self, id: &str) -> anyhow::Result<()> {
        let mut inner = self.inner.write().expect("registry lock poisoned");
        if inner.remove_loaded(id) {
            tracing::info!(id = %id, "unloaded model");
            Ok(())
        } else {
            anyhow::bail!("model '{id}' is not loaded")
        }
    }

    fn get_loaded(&self, id: &str) -> Option<Arc<ModelHandle>> {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .models
            .get(id)
            .map(Arc::clone)
    }

    fn spec_for(&self, id: &str) -> Option<ModelSpec> {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .available
            .get(id)
            .cloned()
    }

    /// Get-or-create the per-id async load guard.
    async fn load_guard(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.load_locks.lock().await;
        Arc::clone(
            locks
                .entry(id.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }
}

impl RegistryInner {
    /// Insert a loaded handle, appending to `order` if the id is new.  Never
    /// changes `default_id` (that is fixed at construction).
    fn insert_loaded(&mut self, handle: Arc<ModelHandle>) {
        let id = handle.id.clone();
        if !self.models.contains_key(&id) {
            self.order.push(id.clone());
        }
        self.models.insert(id, handle);
    }

    /// Remove a loaded handle from `models` and `order`.  Returns `true` if it
    /// was present.  The spec stays in `available` for later reloads.
    fn remove_loaded(&mut self, id: &str) -> bool {
        if self.models.remove(id).is_some() {
            self.order.retain(|existing| existing != id);
            true
        } else {
            false
        }
    }

    /// Evict least-recently-used models until `models.len() <= max`.
    ///
    /// The model currently being loaded (`loading_id`) is never evicted, and the
    /// default model is only evicted as a last resort.  Because eviction only
    /// runs when `len > max` and `max >= 1`, the registry never drops below one
    /// loaded model.
    fn enforce_eviction(&mut self, max_loaded: Option<usize>, loading_id: &str) {
        let Some(max) = max_loaded else {
            return;
        };
        while self.models.len() > max {
            let Some(victim) = self.pick_lru_victim(loading_id) else {
                break;
            };
            tracing::info!(id = %victim, "evicting model (LRU)");
            self.remove_loaded(&victim);
        }
    }

    /// Choose the LRU victim, excluding `loading_id` and preferring non-default
    /// models.  Returns `None` if nothing else is evictable.
    fn pick_lru_victim(&self, loading_id: &str) -> Option<String> {
        let default = self.default_id.as_deref();
        // Prefer evicting a non-default model.
        let non_default = self
            .models
            .iter()
            .filter(|(id, _)| id.as_str() != loading_id && Some(id.as_str()) != default)
            .min_by_key(|(_, h)| h.last_request_at.load(Ordering::Relaxed))
            .map(|(id, _)| id.clone());
        if non_default.is_some() {
            return non_default;
        }
        // Fall back to evicting the default only if it is the sole candidate.
        self.models
            .iter()
            .filter(|(id, _)| id.as_str() != loading_id)
            .min_by_key(|(_, h)| h.last_request_at.load(Ordering::Relaxed))
            .map(|(id, _)| id.clone())
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
impl ModelRegistry {
    /// Build an empty registry for unit tests (no models, no available specs).
    pub(crate) fn new_for_test() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RegistryInner {
                models: HashMap::new(),
                order: Vec::new(),
                default_id: None,
                available: HashMap::new(),
            })),
            config: Arc::new(ServerConfig::default()),
            load_locks: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Insert a pre-built handle directly, setting it as default if it is the
    /// first inserted.  Mirrors the old `insert` used by ordering tests.
    pub(crate) fn insert_for_test(&self, handle: Arc<ModelHandle>) {
        let mut inner = self.inner.write().unwrap();
        if inner.default_id.is_none() {
            inner.default_id = Some(handle.id.clone());
        }
        inner.insert_loaded(handle);
    }

    /// Enforce eviction directly (used by eviction unit tests).
    pub(crate) fn enforce_eviction_for_test(&self, max_loaded: Option<usize>, loading_id: &str) {
        let mut inner = self.inner.write().unwrap();
        inner.enforce_eviction(max_loaded, loading_id);
    }

    /// Replace the default model's `fim_config` in place (test-only helper).
    pub(crate) fn set_default_fim_config(&self, fim_config: Option<FimConfig>) {
        let mut inner = self.inner.write().unwrap();
        let id = inner
            .default_id
            .clone()
            .expect("registry must have a default model");
        let old_arc = inner.models.remove(&id).expect("default model must exist");
        let old = Arc::try_unwrap(old_arc)
            .unwrap_or_else(|_| panic!("unique handle ownership during test setup"));
        let new_handle = Arc::new(ModelHandle { fim_config, ..old });
        inner.models.insert(id, new_handle);
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use tokio::sync::{Semaphore, mpsc};

    use super::*;
    use crate::driver::EngineDriver;

    /// Build a minimal `ModelHandle` stub backed by the tiny-llm tokenizer fixture.
    /// The stub has a dead command channel (no engine thread); it is only used to
    /// exercise registry ordering / eviction — never for actual generation.
    fn stub_handle(id: &str, tokenizer: Arc<Tokenizer>) -> Arc<ModelHandle> {
        stub_handle_at(id, tokenizer, now_millis())
    }

    fn stub_handle_at(id: &str, tokenizer: Arc<Tokenizer>, last_request_at: u64) -> Arc<ModelHandle> {
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
            last_request_at: AtomicU64::new(last_request_at),
        })
    }

    fn load_tokenizer() -> Arc<Tokenizer> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json");
        Arc::new(Tokenizer::from_file(&path).expect("load test tokenizer"))
    }

    #[test]
    fn registry_insertion_order_is_deterministic() {
        let tokenizer = load_tokenizer();
        let ids = ["gamma", "alpha", "delta", "beta", "epsilon"];

        let registry = ModelRegistry::new_for_test();
        for id in &ids {
            registry.insert_for_test(stub_handle(id, Arc::clone(&tokenizer)));
        }

        let resolved = registry.resolve("").expect("resolve empty should succeed");
        assert_eq!(resolved.id, "gamma");
        assert_eq!(registry.default_id().as_deref(), Some("gamma"));
        assert_eq!(registry.ids(), ids);
    }

    #[test]
    fn registry_reinsert_does_not_duplicate_order() {
        let tokenizer = load_tokenizer();
        let registry = ModelRegistry::new_for_test();
        registry.insert_for_test(stub_handle("x", Arc::clone(&tokenizer)));
        registry.insert_for_test(stub_handle("x", Arc::clone(&tokenizer)));
        registry.insert_for_test(stub_handle("y", Arc::clone(&tokenizer)));

        assert_eq!(registry.ids(), vec!["x", "y"]);
        assert_eq!(registry.default_id().as_deref(), Some("x"));
    }

    /// Eviction must remove the least-recently-used **non-default** model first.
    #[test]
    fn eviction_picks_least_recently_used_non_default() {
        let tokenizer = load_tokenizer();
        let registry = ModelRegistry::new_for_test();
        // default = "a" (oldest timestamp), "b" newest, "c" middle.
        registry.insert_for_test(stub_handle_at("a", Arc::clone(&tokenizer), 100));
        registry.insert_for_test(stub_handle_at("b", Arc::clone(&tokenizer), 300));
        registry.insert_for_test(stub_handle_at("c", Arc::clone(&tokenizer), 200));

        // Cap at 2 while "loading" b; the LRU non-default ("c") must be evicted,
        // even though the default "a" has an older timestamp.
        registry.enforce_eviction_for_test(Some(2), "b");

        let mut ids = registry.ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"], "LRU non-default 'c' should be evicted");
    }

    /// Eviction must never drop below one model and only evicts the default as a
    /// last resort.
    #[test]
    fn eviction_never_evicts_below_one_and_spares_default() {
        let tokenizer = load_tokenizer();
        let registry = ModelRegistry::new_for_test();
        registry.insert_for_test(stub_handle_at("a", Arc::clone(&tokenizer), 100)); // default
        registry.insert_for_test(stub_handle_at("b", Arc::clone(&tokenizer), 300));

        // Cap at 1 while loading "b": the only evictable candidate is default "a".
        registry.enforce_eviction_for_test(Some(1), "b");
        assert_eq!(registry.ids(), vec!["b"], "default evicted as last resort");
    }

    #[test]
    fn unload_removes_from_models_but_reports_missing_when_absent() {
        let tokenizer = load_tokenizer();
        let registry = ModelRegistry::new_for_test();
        registry.insert_for_test(stub_handle("a", Arc::clone(&tokenizer)));
        registry.insert_for_test(stub_handle("b", Arc::clone(&tokenizer)));

        registry.unload("b").expect("unload loaded model");
        assert_eq!(registry.ids(), vec!["a"]);
        // Unloading an id that is not loaded is an error (mapped to 404).
        assert!(registry.unload("b").is_err());
    }
}
