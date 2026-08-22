use std::{
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
    sync::{
        Arc, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use onnx_genai::{GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_engine::FimConfig;
use onnx_genai_metadata::GenerationDefaults;
use onnx_genai_ort::{ChatTemplate, Tokenizer};

use crate::{
    driver::EngineDriver,
    image_generation::ImagePipelineSpec,
    models_config::ModelSpec,
    multimodal::MultimodalSpecs,
    state::{ServerConfig, ServerMemoryAuthorities, build_handle_with_authorities},
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
    /// The model author's declared generation defaults (`do_sample`,
    /// `temperature`, `top_p`, `top_k`), or `None` when the package declares
    /// none or is a pipeline (whose sampling is governed by its plan, not a
    /// single decoder's `search` block). Resolved into every request's
    /// `GenerateOptions` so a model that ships `do_sample: true` samples by
    /// default instead of being forced greedy, matching the CLI.
    pub(crate) generation_defaults: Option<GenerationDefaults>,
    pub(crate) fim_config: Option<FimConfig>,
    /// Declared image/audio input contracts, or `None` for a single decoder
    /// graph. Shared with the CLI so both front ends admit the same inputs.
    pub(crate) multimodal: Option<MultimodalSpecs>,
    pub(crate) speech: Option<crate::speech::SpeechCapability>,
    pub(crate) image_pipeline: Option<ImagePipelineSpec>,
    /// Whether the package declares a channel whose content the caller must not
    /// be shown, i.e. whether a generated turn carries private reasoning that
    /// has to be filtered out of everything this server returns.
    pub(crate) private_channels: bool,
    /// Epoch-millisecond timestamp of the last call to `ModelRegistry::resolve`.
    /// Initialised to construction time; updated on every resolve for LRU eviction.
    pub(crate) last_request_at: AtomicU64,
    /// Set only after a warmup generation completes successfully.
    warmed: AtomicBool,
    warmup_lock: std::sync::Mutex<()>,
}

/// Everything needed to construct a [`ModelHandle`].
///
/// A struct rather than a long positional argument list: the fields are mostly
/// optional and same-typed, so positional construction was easy to get wrong.
pub(crate) struct ModelHandleParts {
    pub(crate) id: String,
    /// Directory the model was loaded from. Read once at construction to decide
    /// whether the package declares a private reasoning channel; not retained
    /// on the handle, which has no other package-relative file to resolve.
    pub(crate) model_dir: PathBuf,
    pub(crate) engine: EngineDriver,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) chat_template: Option<Arc<ChatTemplate>>,
    pub(crate) model_max_context: Option<usize>,
    pub(crate) generation_defaults: Option<GenerationDefaults>,
    pub(crate) fim_config: Option<FimConfig>,
    pub(crate) multimodal: Option<MultimodalSpecs>,
    pub(crate) speech: Option<crate::speech::SpeechCapability>,
    pub(crate) image_pipeline: Option<ImagePipelineSpec>,
}

impl ModelHandle {
    pub(crate) fn new(parts: ModelHandleParts) -> anyhow::Result<Self> {
        let ModelHandleParts {
            id,
            model_dir,
            engine,
            tokenizer,
            chat_template,
            model_max_context,
            generation_defaults,
            fim_config,
            multimodal,
            speech,
            image_pipeline,
        } = parts;
        let private_channels = declares_private_channels(&model_dir);
        Ok(Self {
            id,
            engine,
            tokenizer,
            chat_template,
            model_max_context,
            generation_defaults,
            fim_config,
            multimodal,
            speech,
            image_pipeline,
            private_channels,
            last_request_at: AtomicU64::new(now_millis()),
            warmed: AtomicBool::new(false),
            warmup_lock: std::sync::Mutex::new(()),
        })
    }

    /// Run one deterministic generation matching the loaded model's output
    /// contract to initialize lazy runtime allocations.
    fn warmup(&self) -> anyhow::Result<Duration> {
        if self.warmed.load(Ordering::Acquire) {
            return Ok(Duration::ZERO);
        }
        let _guard = self
            .warmup_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("model warmup lock poisoned"))?;
        if self.warmed.load(Ordering::Acquire) {
            return Ok(Duration::ZERO);
        }
        let started = Instant::now();
        if let Some(image_pipeline) = &self.image_pipeline {
            let request = image_pipeline.warmup_request(&self.tokenizer, self.model_max_context)?;
            self.engine.warmup_image(request)?;
        } else {
            let prompt = self
                .tokenizer
                .encode("warmup")
                .context("failed to tokenize warmup prompt")?;
            self.engine.warmup(GenerateRequest {
                prompt: GeneratePrompt::TokenIds(prompt),
                options: GenerateOptions {
                    max_new_tokens: 1,
                    max_context: self.model_max_context,
                    ..GenerateOptions::default()
                },
            })?;
        }
        self.warmed.store(true, Ordering::Release);
        Ok(started.elapsed())
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

#[derive(Debug)]
pub(crate) struct RegistryError;

impl fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("registry lock poisoned")
    }
}

impl std::error::Error for RegistryError {}

#[derive(Debug)]
pub(crate) enum WarmupError {
    Registry(RegistryError),
    NotLoaded,
    Failed(anyhow::Error),
}

impl fmt::Display for WarmupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registry(error) => error.fmt(formatter),
            Self::NotLoaded => formatter.write_str("model is not loaded"),
            Self::Failed(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for WarmupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Registry(error) => Some(error),
            Self::NotLoaded => None,
            Self::Failed(error) => error.source(),
        }
    }
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
    policy_lock: Arc<tokio::sync::RwLock<()>>,
    authority_provider: Arc<ServerMemoryAuthorities>,
}

impl ModelRegistry {
    pub(crate) fn aggregate_growth_metrics(
        &self,
    ) -> anyhow::Result<Option<onnx_genai_engine::MappedGrowthMetrics>> {
        self.authority_provider.aggregate_growth_metrics()
    }

    fn config_for_load(&self) -> anyhow::Result<ServerConfig> {
        let mut config = (*self.config).clone();
        config.engine_config.limits.vram_limit = self.authority_provider.configured_limit()?;
        Ok(config)
    }

    /// Reconfigure the process-wide device policy and every loaded engine.
    ///
    /// The server exposes one unqualified VRAM flag, so this applies to all
    /// device compatibility domains. A failed shrink is rejected by the
    /// provider before either current ledgers or future-load policy changes.
    pub(crate) async fn set_vram_limit(
        &self,
        limit: onnx_genai_engine::ResourceLimit,
    ) -> anyhow::Result<Option<onnx_genai_engine::GovernorSnapshot>> {
        let _policy = self.policy_lock.write().await;
        if !self.config.engine_config.allow_runtime_override {
            return Err(onnx_genai_engine::EngineGovernorError::RuntimeOverrideDisabled.into());
        }
        let old_limit = self.authority_provider.configured_limit()?;
        let handles = self.read()?.models.values().cloned().collect::<Vec<_>>();
        if handles.is_empty() {
            self.authority_provider.reconfigure_limit(limit)?;
            return Ok(None);
        }
        // Drain and hold every admission permit so no request can consume
        // newly exposed headroom between provider commit and per-engine commit.
        let mut admission_guards = Vec::with_capacity(handles.len());
        for handle in &handles {
            admission_guards.push(
                Arc::clone(&handle.engine.generation_capacity)
                    .acquire_many_owned(handle.engine.generation_capacity_size)
                    .await
                    .map_err(|_| anyhow::anyhow!("engine admission semaphore closed"))?,
            );
        }
        self.authority_provider.reconfigure_limit(limit)?;
        let mut latest = None;
        let mut updated = Vec::new();
        for handle in handles {
            match handle.engine.set_vram_limit(limit).await {
                Ok(Ok(snapshot)) => {
                    latest = Some(snapshot);
                    updated.push(handle);
                }
                Ok(Err(error)) => {
                    let _ = self.authority_provider.reconfigure_limit(old_limit);
                    for updated_handle in updated {
                        let _ = updated_handle.engine.set_vram_limit(old_limit).await;
                    }
                    return Err(error.into());
                }
                Err(error) => {
                    let _ = self.authority_provider.reconfigure_limit(old_limit);
                    for updated_handle in updated {
                        let _ = updated_handle.engine.set_vram_limit(old_limit).await;
                    }
                    return Err(error);
                }
            }
        }
        Ok(latest)
    }

    pub(crate) async fn aggregate_resource_snapshot(
        &self,
    ) -> anyhow::Result<Option<onnx_genai_engine::GovernorSnapshot>> {
        let handle = self.read()?.models.values().next().cloned();
        let Some(handle) = handle else {
            return Ok(None);
        };
        let mut snapshot = handle.engine.resource_snapshot().await?;
        if let Some((used, limit, headroom)) = self.authority_provider.aggregate_vram()? {
            snapshot.vram.used = used;
            snapshot.vram.limit = limit;
            snapshot.vram.headroom = headroom;
            // Keep `resolved_limits.vram_bytes` honest: it is the resolved device
            // (VRAM) capacity limit, which stays `None` when the device capacity
            // could not be measured (#947). The aggregate authority ceiling on a
            // device-less box is a host-RAM-derived advisory bound (surfaced via
            // `vram.limit`), not a measured VRAM capacity, so do not relabel it
            // as one. Only refresh it when a real device capacity was resolved.
            if snapshot.resolved_limits.vram_bytes.is_some() {
                snapshot.resolved_limits.vram_bytes = Some(limit);
            }
        }
        Ok(Some(snapshot))
    }

    pub(crate) fn any_loaded(&self) -> Result<Option<Arc<ModelHandle>>, RegistryError> {
        Ok(self.read()?.models.values().next().cloned())
    }

    pub(crate) fn memory_strategy_plans(
        &self,
    ) -> Result<Vec<(String, Arc<onnx_genai_engine::MemoryStrategyPlan>)>, RegistryError> {
        Ok(self
            .read()?
            .models
            .iter()
            .map(|(id, handle)| (id.clone(), handle.engine.memory_strategy_plan()))
            .collect())
    }

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
        let inner = RegistryInner {
            models: HashMap::new(),
            order: Vec::new(),
            default_id,
            available,
        };
        let authority_provider = Arc::new(ServerMemoryAuthorities::new(
            config.engine_config.limits.vram_limit,
        ));
        let registry = Self {
            inner: Arc::new(RwLock::new(inner)),
            config: Arc::new(config.clone()),
            load_locks: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            policy_lock: Arc::new(tokio::sync::RwLock::new(())),
            authority_provider,
        };
        for spec in specs.iter().filter(|s| s.eager) {
            tracing::info!(id = %spec.id, path = %spec.path.display(), "loading model");
            let handle = build_handle_with_authorities(
                spec,
                &config,
                Arc::clone(&registry.authority_provider),
            )
            .with_context(|| {
                format!(
                    "failed to load model '{}' from '{}'",
                    spec.id,
                    spec.path.display()
                )
            })?;
            registry.write()?.insert_loaded(Arc::new(handle));
            if spec.warmup {
                let warmup_registry = registry.clone();
                let warmup_id = spec.id.clone();
                std::thread::spawn(move || warmup_registry.warmup(&warmup_id))
                    .join()
                    .map_err(|_| anyhow::anyhow!("model warmup thread panicked"))?
                    .with_context(|| format!("failed to warm model '{}'", spec.id))?;
            }
        }
        Ok(registry)
    }

    /// Build a registry around a single, already-constructed handle.
    ///
    /// Used by the `AppState::new*` constructors that start from a live `Engine`
    /// rather than a spec path.  Because there is no backing spec, the model is
    /// not recorded in `available` and therefore cannot be lazily reloaded after
    /// an unload.
    pub(crate) fn from_handle(handle: Arc<ModelHandle>, config: ServerConfig) -> Self {
        let authority_provider = Arc::new(ServerMemoryAuthorities::new(
            config.engine_config.limits.vram_limit,
        ));
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
            policy_lock: Arc::new(tokio::sync::RwLock::new(())),
            authority_provider,
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
    pub(crate) fn resolve(
        &self,
        requested: &str,
    ) -> Result<Option<Arc<ModelHandle>>, RegistryError> {
        let inner = self.read()?;
        let handle = if !requested.trim().is_empty() {
            inner.models.get(requested)
        } else {
            inner
                .default_id
                .as_deref()
                .and_then(|default| inner.models.get(default))
        };
        let Some(handle) = handle else {
            return Ok(None);
        };
        handle
            .last_request_at
            .store(now_millis(), Ordering::Relaxed);
        Ok(Some(Arc::clone(handle)))
    }

    /// Returns `true` if `id` is a configured model (loaded or not).
    pub(crate) fn contains_available(&self, id: &str) -> Result<bool, RegistryError> {
        Ok(self.read()?.available.contains_key(id))
    }

    /// Returns the ids of all currently loaded models in insertion order.
    pub(crate) fn ids(&self) -> Result<Vec<String>, RegistryError> {
        Ok(self.read()?.order.clone())
    }

    /// Returns the id of the default model, or `None` if none is configured.
    pub(crate) fn default_id(&self) -> Result<Option<String>, RegistryError> {
        Ok(self.read()?.default_id.clone())
    }

    /// Snapshot of every configured model with its loaded/available status,
    /// ordered by configured id for determinism.
    pub(crate) fn statuses(&self) -> Result<Vec<ModelStatus>, RegistryError> {
        let inner = self.read()?;
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
        Ok(statuses)
    }

    /// Load (or return the already-loaded) model for `id`.
    ///
    /// The heavy construction runs on a blocking thread pool via
    /// `spawn_blocking`; the registry lock is only taken for the brief insert +
    /// eviction critical section afterwards.  A per-id async guard serialises
    /// concurrent loads of the same id so the model is built only once.
    pub(crate) async fn load(&self, id: &str) -> anyhow::Result<Arc<ModelHandle>> {
        // Fast path: already loaded.
        if let Some(handle) = self.get_loaded(id)? {
            return Ok(handle);
        }
        // Validate the id is configured before doing any work.
        let spec = self
            .spec_for(id)?
            .ok_or_else(|| anyhow::anyhow!("unknown model id '{id}'"))?;

        // Serialise concurrent loads of the same id.
        let guard = self.load_guard(id).await;
        let _held = guard.lock().await;

        // Re-check after acquiring the guard: another waiter may have loaded it.
        if let Some(handle) = self.get_loaded(id)? {
            return Ok(handle);
        }

        let _policy = self.policy_lock.read().await;
        let config = self.config_for_load()?;
        let authority_provider = Arc::clone(&self.authority_provider);
        let spec_for_build = spec.clone();
        tracing::info!(id = %spec.id, path = %spec.path.display(), "lazy-loading model");
        let handle = tokio::task::spawn_blocking(move || {
            build_handle_with_authorities(&spec_for_build, &config, authority_provider)
        })
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
            let mut inner = self.write()?;
            inner.insert_loaded(Arc::clone(&handle));
            inner.enforce_eviction(self.config.max_loaded_models, id);
        }
        if spec.warmup {
            let registry = self.clone();
            let warmup_id = id.to_owned();
            tokio::task::spawn_blocking(move || registry.warmup(&warmup_id))
                .await
                .context("model warmup task panicked")?
                .map_err(anyhow::Error::new)
                .with_context(|| format!("failed to warm model '{id}'"))?;
        }
        Ok(handle)
    }

    /// Warm a currently loaded model. Unknown and unloaded models return an
    /// error; a successfully warmed model returns a zero duration on repeats.
    pub(crate) fn warmup(&self, id: &str) -> Result<Duration, WarmupError> {
        self.get_loaded(id)
            .map_err(WarmupError::Registry)?
            .ok_or(WarmupError::NotLoaded)?
            .warmup()
            .map_err(WarmupError::Failed)
    }

    /// Unload a model: drop its handle from `models`/`order` but keep the spec in
    /// `available` so it can be lazily reloaded.  In-flight requests that already
    /// hold an `Arc<ModelHandle>` keep the engine alive until they finish.
    ///
    /// Returns an error if the id is not currently loaded (mapped to 404).
    pub(crate) fn unload(&self, id: &str) -> anyhow::Result<()> {
        let mut inner = self.write()?;
        if inner.remove_loaded(id) {
            tracing::info!(id = %id, "unloaded model");
            Ok(())
        } else {
            anyhow::bail!("model '{id}' is not loaded")
        }
    }

    fn get_loaded(&self, id: &str) -> Result<Option<Arc<ModelHandle>>, RegistryError> {
        Ok(self.read()?.models.get(id).map(Arc::clone))
    }

    fn spec_for(&self, id: &str) -> Result<Option<ModelSpec>, RegistryError> {
        Ok(self.read()?.available.get(id).cloned())
    }

    fn read(&self) -> Result<RwLockReadGuard<'_, RegistryInner>, RegistryError> {
        self.inner.read().map_err(|_| RegistryError)
    }

    fn write(&self) -> Result<RwLockWriteGuard<'_, RegistryInner>, RegistryError> {
        self.inner.write().map_err(|_| RegistryError)
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
            policy_lock: Arc::new(tokio::sync::RwLock::new(())),
            authority_provider: Arc::new(ServerMemoryAuthorities::new(
                ServerConfig::default().engine_config.limits.vram_limit,
            )),
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

    pub(crate) fn poison_for_test(&self) {
        let inner = Arc::clone(&self.inner);
        let _ = std::thread::spawn(move || {
            let _guard = inner.write().expect("test registry lock must be available");
            panic!("poison registry lock for HTTP error test");
        })
        .join();
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

    pub(crate) fn is_warmed_for_test(&self, id: &str) -> bool {
        self.inner
            .read()
            .unwrap()
            .models
            .get(id)
            .is_some_and(|handle| handle.warmed.load(Ordering::Acquire))
    }
}

/// Whether a package declares a response channel the caller must not be shown.
///
/// A reasoning model spells its turn as channels addressed to different
/// recipients, and declares in `response_template` which one carries the answer
/// and which carries its private thinking. Only a package that declares the
/// private one needs its output filtered, so every other model streams and
/// returns exactly what it generated.
fn declares_private_channels(model_dir: &Path) -> bool {
    let Ok(bytes) = std::fs::read(model_dir.join("tokenizer_config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    config
        .pointer("/response_template/fields/reasoning_content/open_pattern")
        .and_then(serde_json::Value::as_str)
        .is_some()
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use tokio::sync::{Semaphore, mpsc};

    use super::*;
    use crate::driver::EngineDriver;

    // Only a package that declares a private reasoning channel needs its output
    // filtered, so the flag is read from the package rather than guessed at.
    #[test]
    fn a_declared_reasoning_channel_arms_filtering() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({
                "response_template": {
                    "fields": {
                        "content": {"open_pattern": "to=user<\\|message\\|>"},
                        "reasoning_content": {"open_pattern": "to=self<\\|message\\|>"}
                    }
                }
            })
            .to_string(),
        )
        .expect("write config");

        assert!(declares_private_channels(dir.path()));
    }

    // A model that declares no private channel, or ships no template at all,
    // must stream exactly what it generated.
    #[test]
    fn an_undeclared_reasoning_channel_leaves_output_untouched() {
        let dir = tempfile::tempdir().expect("tempdir");

        assert!(!declares_private_channels(dir.path()));

        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({"response_template": {"fields": {"content": {}}}}).to_string(),
        )
        .expect("write config");

        assert!(!declares_private_channels(dir.path()));
    }

    /// Build a minimal `ModelHandle` stub backed by the tiny-llm tokenizer fixture.
    /// The stub has a dead command channel (no engine thread); it is only used to
    /// exercise registry ordering / eviction — never for actual generation.
    fn stub_handle(id: &str, tokenizer: Arc<Tokenizer>) -> Arc<ModelHandle> {
        stub_handle_at(id, tokenizer, now_millis())
    }

    fn stub_handle_at(
        id: &str,
        tokenizer: Arc<Tokenizer>,
        last_request_at: u64,
    ) -> Arc<ModelHandle> {
        Arc::new(ModelHandle {
            id: id.to_string(),
            engine: stub_engine_driver(),
            tokenizer,
            chat_template: None,
            model_max_context: None,
            generation_defaults: None,
            fim_config: None,
            multimodal: None,
            speech: None,
            image_pipeline: None,
            private_channels: false,
            last_request_at: AtomicU64::new(last_request_at),
            warmed: AtomicBool::new(false),
            warmup_lock: std::sync::Mutex::new(()),
        })
    }

    fn stub_engine_driver() -> EngineDriver {
        let (tx, _rx) = mpsc::channel(1);
        EngineDriver {
            is_workflow: false,
            workflow_provenance: "none",
            commands: tx,
            generation_capacity: Arc::new(Semaphore::new(0)),
            generation_capacity_size: 0,
            // A test double drives no engine, so there is no pool to
            // mirror. Left in the default `Unknown` state rather than
            // asserted not-applicable: nothing here has determined a
            // decode path, and "pending" is the only claim that holds.
            kv_telemetry: Default::default(),
            resource_snapshot: Default::default(),
            memory_strategy_plan: Arc::new(onnx_genai_engine::MemoryStrategyPlan::unknown(
                0,
                None,
                "registry test stub",
            )),
            device_authority: None,
            // A stub drives no engine, so it advertises the honest "no
            // batching" report used by every non-batching backend.
            batching: Arc::new(crate::driver::BatchingReport::single_sequence_stub()),
        }
    }

    #[test]
    fn registry_does_not_infer_speech_capability_from_adapter_files_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("inference_metadata.yaml"),
            "pipeline:\n  workflow:\n    components:\n      prompt:\n        contract:\n          id: onnx-genai.text-assembly\n        implementation:\n          artifact: speech_processor.json\n",
        )
        .expect("metadata");
        std::fs::write(
            dir.path().join("speech_processor.json"),
            r#"{"max_input_tokens":1,"max_output_units":1,"segments":[{"literal":"x"}]}"#,
        )
        .expect("processor");

        let handle = ModelHandle::new(ModelHandleParts {
            id: "adapter-only".to_string(),
            model_dir: dir.path().to_path_buf(),
            engine: stub_engine_driver(),
            tokenizer: load_tokenizer(),
            chat_template: None,
            model_max_context: None,
            generation_defaults: None,
            fim_config: None,
            multimodal: None,
            speech: None,
            image_pipeline: None,
        })
        .expect("handle");

        assert!(
            handle.speech.is_none(),
            "registry must require the loader's compatible audio-output decision"
        );
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

        let resolved = registry
            .resolve("")
            .expect("registry lock must be available")
            .expect("resolve empty should succeed");
        assert_eq!(resolved.id, "gamma");
        assert_eq!(registry.default_id().unwrap().as_deref(), Some("gamma"));
        assert_eq!(registry.ids().unwrap(), ids);
    }

    #[tokio::test]
    async fn production_load_path_shares_device_authority_and_metrics_ledger() {
        let model_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter");
        let specs = vec![
            ModelSpec {
                id: "first".to_string(),
                path: model_dir.clone(),
                eager: true,
                warmup: false,
            },
            ModelSpec {
                id: "second".to_string(),
                path: model_dir,
                eager: true,
                warmup: false,
            },
        ];
        let registry = ModelRegistry::from_specs(&specs, ServerConfig::default()).unwrap();
        let first = registry.resolve("first").unwrap().unwrap();
        let second = registry.resolve("second").unwrap().unwrap();
        let first_authority = first.engine.device_authority.as_ref().unwrap().clone();
        let second_authority = second.engine.device_authority.as_ref().unwrap().clone();

        assert_eq!(
            first_authority.authority_id(),
            second_authority.authority_id()
        );
        let aggregate_used = first_authority.used_bytes();
        assert!(aggregate_used > 0);
        assert_eq!(
            first.engine.resource_snapshot().await.unwrap().vram.used,
            aggregate_used
        );
        assert_eq!(
            second.engine.resource_snapshot().await.unwrap().vram.used,
            aggregate_used
        );

        drop(first);
        registry.unload("first").unwrap();
        for _ in 0..100 {
            if first_authority.used_bytes() < aggregate_used {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(first_authority.used_bytes() < aggregate_used);
        assert!(first_authority.used_bytes() > 0);
        assert_eq!(
            second.engine.resource_snapshot().await.unwrap().vram.used,
            first_authority.used_bytes()
        );
        let metrics_snapshot = registry
            .aggregate_resource_snapshot()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(metrics_snapshot.vram.used, first_authority.used_bytes());
        assert_eq!(metrics_snapshot.vram.limit, first_authority.limit_bytes());
        assert_eq!(
            metrics_snapshot.vram.headroom,
            first_authority.headroom_bytes()
        );
    }

    #[tokio::test]
    async fn runtime_limit_update_applies_to_future_lazy_loads() {
        let model_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter");
        let specs = vec![
            ModelSpec {
                id: "eager".to_string(),
                path: model_dir.clone(),
                eager: true,
                warmup: false,
            },
            ModelSpec {
                id: "lazy".to_string(),
                path: model_dir,
                eager: false,
                warmup: false,
            },
        ];
        let mut config = ServerConfig::default();
        config.engine_config.allow_runtime_override = true;
        let registry = ModelRegistry::from_specs(&specs, config).unwrap();
        let new_limit = onnx_genai_engine::ResourceLimit::Bytes(7 << 30);
        let snapshot = registry.set_vram_limit(new_limit).await.unwrap().unwrap();
        let effective_limit = snapshot.vram.limit;

        let lazy = registry.load("lazy").await.unwrap();
        assert_eq!(
            lazy.engine.resource_snapshot().await.unwrap().vram.limit,
            effective_limit
        );
        assert_eq!(
            registry.authority_provider.configured_limit().unwrap(),
            new_limit
        );
    }

    #[tokio::test]
    async fn failed_runtime_shrink_preserves_policy_and_ledger_limit() {
        let model_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter");
        let specs = vec![ModelSpec {
            id: "model".to_string(),
            path: model_dir,
            eager: true,
            warmup: false,
        }];
        let mut config = ServerConfig::default();
        config.engine_config.allow_runtime_override = true;
        let registry = ModelRegistry::from_specs(&specs, config).unwrap();
        let old_policy = registry.authority_provider.configured_limit().unwrap();
        let handle = registry.resolve("model").unwrap().unwrap();
        let before = handle.engine.resource_snapshot().await.unwrap();

        let error = registry
            .set_vram_limit(onnx_genai_engine::ResourceLimit::Bytes(1))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("cannot satisfy lowered resource limit"),
            "expected a limit-rejection error, got: {error}"
        );
        let after = handle.engine.resource_snapshot().await.unwrap();
        assert_eq!(after.vram.limit, before.vram.limit);
        assert_eq!(after.vram.used, before.vram.used);
        assert_eq!(
            registry.authority_provider.configured_limit().unwrap(),
            old_policy
        );
    }

    #[tokio::test]
    async fn runtime_limit_update_with_no_loaded_models_applies_to_future_load() {
        let model_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter");
        let specs = vec![
            ModelSpec {
                id: "eager".to_string(),
                path: model_dir.clone(),
                eager: true,
                warmup: false,
            },
            ModelSpec {
                id: "lazy".to_string(),
                path: model_dir,
                eager: false,
                warmup: false,
            },
        ];
        let mut config = ServerConfig::default();
        config.engine_config.allow_runtime_override = true;
        let registry = ModelRegistry::from_specs(&specs, config).unwrap();
        registry.unload("eager").unwrap();
        let new_limit = onnx_genai_engine::ResourceLimit::Bytes(7 << 30);

        assert!(registry.set_vram_limit(new_limit).await.unwrap().is_none());
        assert_eq!(
            registry.authority_provider.configured_limit().unwrap(),
            new_limit
        );
        let lazy = registry.load("lazy").await.unwrap();
        assert_eq!(
            lazy.engine.resource_snapshot().await.unwrap().vram.limit,
            7 << 30
        );
    }

    #[test]
    fn registry_reinsert_does_not_duplicate_order() {
        let tokenizer = load_tokenizer();
        let registry = ModelRegistry::new_for_test();
        registry.insert_for_test(stub_handle("x", Arc::clone(&tokenizer)));
        registry.insert_for_test(stub_handle("x", Arc::clone(&tokenizer)));
        registry.insert_for_test(stub_handle("y", Arc::clone(&tokenizer)));

        assert_eq!(registry.ids().unwrap(), vec!["x", "y"]);
        assert_eq!(registry.default_id().unwrap().as_deref(), Some("x"));
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

        let mut ids = registry.ids().unwrap();
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
        assert_eq!(
            registry.ids().unwrap(),
            vec!["b"],
            "default evicted as last resort"
        );
    }

    #[test]
    fn unload_removes_from_models_but_reports_missing_when_absent() {
        let tokenizer = load_tokenizer();
        let registry = ModelRegistry::new_for_test();
        registry.insert_for_test(stub_handle("a", Arc::clone(&tokenizer)));
        registry.insert_for_test(stub_handle("b", Arc::clone(&tokenizer)));

        registry.unload("b").expect("unload loaded model");
        assert_eq!(registry.ids().unwrap(), vec!["a"]);
        // Unloading an id that is not loaded is an error (mapped to 404).
        assert!(registry.unload("b").is_err());
    }

    #[test]
    fn poisoned_lock_returns_an_error() {
        let registry = ModelRegistry::new_for_test();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = registry.inner.write().expect("new lock must be available");
            panic!("poison registry lock");
        }));

        assert_eq!(
            registry
                .resolve("")
                .err()
                .expect("poisoned lock must fail")
                .to_string(),
            "registry lock poisoned"
        );
    }
}
