use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::Context;
use onnx_genai::{Engine, EngineConfig};
#[cfg(feature = "native-backend")]
use onnx_genai_engine::NativeDecodeDevice;
use onnx_genai_engine::{
    DeviceCompatibilityDomain, DeviceMemoryAuthority, KvDType, MappedGrowthMetrics,
    MemoryAuthorityProvider, ResourceLimit,
};
use onnx_genai_ort::{ChatTemplate, ModelDirectory, PipelineModelDirectory, Tokenizer};

#[cfg(test)]
use onnx_genai_engine::FimConfig;

use crate::{
    driver::EngineDriver,
    models_config::ModelSpec,
    registry::{EvictionPolicy, ModelHandle, ModelHandleParts, ModelRegistry},
    session::SessionRegistry,
};

pub(crate) const DEFAULT_MAX_OUTPUT_TOKENS: usize = 4096;
const DEFAULT_MAX_SESSIONS: usize = 256;
const DEFAULT_MAX_QUEUE_DEPTH: usize = 256;
const DEFAULT_MAX_BATCH: usize = 4;

#[derive(Debug)]
pub(crate) struct ServerMemoryAuthorities {
    effective_limit: Mutex<ResourceLimit>,
    authorities: Mutex<HashMap<DeviceCompatibilityDomain, DeviceMemoryAuthority>>,
}

impl ServerMemoryAuthorities {
    pub(crate) fn new(configured_limit: ResourceLimit) -> Self {
        Self {
            effective_limit: Mutex::new(configured_limit),
            authorities: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn configured_limit(&self) -> anyhow::Result<ResourceLimit> {
        self.effective_limit
            .lock()
            .map(|limit| *limit)
            .map_err(|_| anyhow::anyhow!("server device-limit lock poisoned"))
    }

    pub(crate) fn aggregate_vram(&self) -> anyhow::Result<Option<(u64, u64, u64)>> {
        let authorities = self
            .authorities
            .lock()
            .map_err(|_| anyhow::anyhow!("server device-authority registry lock poisoned"))?;
        if authorities.is_empty() {
            return Ok(None);
        }
        let used = authorities.values().fold(0_u64, |total, authority| {
            total.saturating_add(authority.used_bytes())
        });
        let limit = authorities.values().fold(0_u64, |total, authority| {
            total.saturating_add(authority.limit_bytes())
        });
        Ok(Some((used, limit, limit.saturating_sub(used))))
    }

    pub(crate) fn aggregate_growth_metrics(&self) -> anyhow::Result<Option<MappedGrowthMetrics>> {
        let authorities = self
            .authorities
            .lock()
            .map_err(|_| anyhow::anyhow!("server device-authority registry lock poisoned"))?;
        if authorities.is_empty() {
            return Ok(None);
        }
        let mut total = MappedGrowthMetrics::default();
        for authority in authorities.values() {
            let metrics = authority.growth_metrics();
            total.attempts = total.attempts.saturating_add(metrics.attempts);
            total.bytes_transferred = total
                .bytes_transferred
                .saturating_add(metrics.bytes_transferred);
            total.failures = total.failures.saturating_add(metrics.failures);
            total.rollbacks = total.rollbacks.saturating_add(metrics.rollbacks);
            total.weight_mapped = total.weight_mapped.saturating_add(metrics.weight_mapped);
            total.kv_mapped = total.kv_mapped.saturating_add(metrics.kv_mapped);
            total.workspace_mapped = total
                .workspace_mapped
                .saturating_add(metrics.workspace_mapped);
            total.total_owned = total.total_owned.saturating_add(metrics.total_owned);
            total.live_holders = total.live_holders.saturating_add(metrics.live_holders);
            total.mapped_bytes = total.mapped_bytes.saturating_add(metrics.mapped_bytes);
            total.total_allowance_bytes = total
                .total_allowance_bytes
                .saturating_add(metrics.total_allowance_bytes);
        }
        Ok(Some(total))
    }

    /// Atomically apply one operator policy to every device authority.
    ///
    /// The server flag is process-wide, so the scope is deliberately all
    /// devices. Validation completes for every ledger before any mutation.
    pub(crate) fn reconfigure_limit(&self, limit: ResourceLimit) -> anyhow::Result<()> {
        let mut effective = self
            .effective_limit
            .lock()
            .map_err(|_| anyhow::anyhow!("server device-limit lock poisoned"))?;
        let authorities = self
            .authorities
            .lock()
            .map_err(|_| anyhow::anyhow!("server device-authority registry lock poisoned"))?;
        let mut ordered = authorities.values().cloned().collect::<Vec<_>>();
        ordered.sort_by_key(|authority| authority.domain().to_string());
        // Resolve the policy against each device's REAL capacity (or an honest
        // "unknown") through the same engine machinery the decode governor uses
        // — never a fabricated 8 GiB constant (#947). A fraction of an unmeasured
        // device is unknown and cannot be resolved; an explicit byte limit is
        // honoured on every device regardless.
        let resolved_limits = ordered
            .iter()
            .map(|authority| {
                let cuda_device_index = match authority.domain() {
                    DeviceCompatibilityDomain::Cuda(index) => Some(*index),
                    DeviceCompatibilityDomain::Host
                    | DeviceCompatibilityDomain::Accelerator { .. } => None,
                };
                // `None` means the limit was a fraction/auto of a device capacity
                // that could not be measured: it is unknown, not a number (#947).
                // We do not fabricate a ceiling. Matching the engine governor's
                // own `set_vram_limit` — which leaves an unmeasured device
                // authority untouched rather than pushing a fabricated number —
                // this authority's advisory ceiling is left unchanged (`None`
                // below) instead of refusing the whole reconfigure. An explicit
                // byte limit still resolves to `Some` and is enforced on every
                // device, measurable or not.
                onnx_genai_engine::resolve_device_vram_limit_bytes(limit, cuda_device_index)
            })
            .collect::<anyhow::Result<Vec<Option<u64>>>>()?;
        let _mapped_growth = ordered
            .iter()
            .map(DeviceMemoryAuthority::pause_mapped_growth)
            .collect::<Result<Vec<_>, _>>()?;
        let guards = ordered
            .iter()
            .map(DeviceMemoryAuthority::pause_reconfiguration)
            .collect::<Vec<_>>();
        #[cfg(feature = "cuda")]
        let pool_gates = ordered
            .iter()
            .map(DeviceMemoryAuthority::physical_pool_operation_gate)
            .collect::<Vec<_>>();
        #[cfg(feature = "cuda")]
        let _pool_operations = pool_gates
            .iter()
            .map(|gate| {
                gate.write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            })
            .collect::<Vec<_>>();
        let mut trim_plan = Vec::with_capacity(ordered.len());
        for ((authority, guard), resolved) in ordered
            .iter()
            .zip(&guards)
            .zip(resolved_limits.iter().copied())
        {
            // Unknown fractional limit: no new ceiling to fit under, nothing to
            // trim — the advisory ceiling stays as it is.
            let Some(resolved) = resolved else {
                trim_plan.push(None);
                continue;
            };
            let used = guard.used();
            let required = used.saturating_sub(resolved);
            let releasable = authority.releasable_unmapped_bytes();
            if required > releasable {
                anyhow::bail!(
                    "cannot satisfy lowered resource limit of {resolved} bytes: {} has {used} \
                     leased bytes but only {releasable} pooled-unmapped bytes are releasable",
                    authority.domain()
                );
            }
            trim_plan.push(Some(required));
        }
        for (((authority, guard), required), resolved) in ordered
            .iter()
            .zip(&guards)
            .zip(trim_plan.iter().copied())
            .zip(resolved_limits.iter().copied())
        {
            let (Some(required), Some(resolved)) = (required, resolved) else {
                continue;
            };
            let released = authority.trim_unmapped_bytes(required)?;
            if released < required || guard.used() > resolved {
                anyhow::bail!(
                    "could not complete lowered resource limit of {resolved} bytes for {}: \
                     planned to release {required} pooled bytes, released {released}, and {} \
                     bytes remain leased; limits and server policy remain unchanged",
                    authority.domain(),
                    guard.used()
                );
            }
        }
        for (guard, resolved) in guards.iter().zip(resolved_limits.iter().copied()) {
            let Some(resolved) = resolved else {
                continue;
            };
            let result = guard.try_set_limit(resolved);
            debug_assert!(
                result.is_ok(),
                "claims are paused and usage was prevalidated"
            );
        }
        *effective = limit;
        Ok(())
    }
}

impl MemoryAuthorityProvider for ServerMemoryAuthorities {
    fn validate_limit(
        &self,
        domain: &DeviceCompatibilityDomain,
        requested: ResourceLimit,
    ) -> anyhow::Result<()> {
        let effective = self
            .effective_limit
            .lock()
            .map_err(|_| anyhow::anyhow!("server device-limit lock poisoned"))?;
        if requested != *effective {
            anyhow::bail!(
                "model device limit {} conflicts with server device-authority limit {} for {}; \
                 configure --vram-limit once at server launch instead of per model",
                describe_limit(requested),
                describe_limit(*effective),
                domain
            );
        }
        Ok(())
    }

    fn authority(
        &self,
        domain: &DeviceCompatibilityDomain,
        resolved_limit_bytes: u64,
    ) -> anyhow::Result<DeviceMemoryAuthority> {
        let mut authorities = self
            .authorities
            .lock()
            .map_err(|_| anyhow::anyhow!("server device-authority registry lock poisoned"))?;
        if let Some(authority) = authorities.get(domain) {
            if authority.limit_bytes() != resolved_limit_bytes {
                anyhow::bail!(
                    "resolved model device limit {resolved_limit_bytes} bytes conflicts with \
                     existing server device-authority limit {} bytes for {}",
                    authority.limit_bytes(),
                    domain
                );
            }
            return Ok(authority.clone());
        }
        let authority = DeviceMemoryAuthority::new(domain.clone(), resolved_limit_bytes);
        authorities.insert(domain.clone(), authority.clone());
        Ok(authority)
    }
}

fn describe_limit(limit: ResourceLimit) -> String {
    match limit {
        ResourceLimit::Auto => "auto".to_string(),
        ResourceLimit::Bytes(bytes) => format!("{bytes} bytes"),
        ResourceLimit::Fraction(fraction) => format!("{fraction} of detected capacity"),
    }
}

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
    /// Reasoning effort applied when a request omits `reasoning_effort`
    /// (`--default-reasoning-effort`). `None` leaves the model's own default in
    /// place, which for a reasoning model is often maximal — enough for an
    /// agent client that never sends the field to spend a whole token budget
    /// thinking and return no answer. An explicit request value always wins.
    pub default_reasoning_effort: Option<crate::types::ReasoningEffort>,
    pub max_sessions: usize,
    /// Maximum generation requests admitted to the driver, including active and queued work.
    pub max_queue_depth: usize,
    /// Operator-requested maximum continuous-batch width (`--max-batch`). `None`
    /// leaves the server default in place and clamps silently to what the decode
    /// path can honor. `Some(n)` with `n > 1` on a backend that cannot batch is a
    /// hard startup refusal rather than a silent no-op — see
    /// [`enforce_requested_max_batch`].
    pub max_batch: Option<usize>,
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
            default_reasoning_effort: None,
            max_sessions: DEFAULT_MAX_SESSIONS,
            max_queue_depth: DEFAULT_MAX_QUEUE_DEPTH,
            max_batch: None,
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
        if self.max_batch == Some(0) {
            anyhow::bail!("max_batch must be greater than zero when set");
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
            warmup: false,
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
        let engine_driver = EngineDriver::start(
            engine,
            config.max_batch.unwrap_or(DEFAULT_MAX_BATCH),
            config.max_queue_depth,
        );
        let handle = ModelHandle::new(ModelHandleParts {
            id: model_id,
            // Test-only constructor: the model was handed in already loaded, so
            // there is no package directory to resolve files against.
            model_dir: std::path::PathBuf::new(),
            engine: engine_driver,
            tokenizer: Arc::new(tokenizer),
            chat_template: chat_template.map(Arc::new),
            model_max_context,
            generation_defaults: None,
            fim_config,
            pipeline: false,
            multimodal: None,
            text_to_image: false,
            text_to_audio: false,
        });
        let registry = ModelRegistry::from_handle(Arc::new(handle), config.clone());
        Self {
            registry,
            sessions: SessionRegistry::new(config.max_sessions),
            config,
        }
    }

    /// Returns the id of the first loaded model, for use in log messages and the CLI.
    pub fn model_id(&self) -> String {
        match self.registry.default_id() {
            Ok(Some(model_id)) => model_id,
            Ok(None) => "onnx-genai-model".to_string(),
            Err(error) => {
                tracing::error!(error = %error, "model registry operation failed");
                "onnx-genai-model".to_string()
            }
        }
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

/// Refuse an explicit `--max-batch > 1` that the engine's decode path cannot
/// honor. This is the startup-time enforcement for issue #750: the native
/// backend (and legacy / non-shared-buffer past/present ORT models) can only
/// decode one sequence at a time, so accepting `--max-batch 8` and silently
/// serving one-at-a-time is dishonest. A `None` request (operator did not set
/// the flag) is never refused — the default width is clamped silently instead.
pub(crate) fn enforce_requested_max_batch(
    engine: &Engine,
    requested: Option<usize>,
) -> anyhow::Result<()> {
    let Some(asked) = requested else {
        return Ok(());
    };
    if asked <= 1 {
        return Ok(());
    }
    let capability = engine.batching_capability();
    if !capability.allows(asked) {
        anyhow::bail!(
            "--max-batch {asked} cannot be honored: {reason}. This backend decodes \
             at most {cap} sequence(s) concurrently; re-launch with --max-batch 1 \
             (or omit the flag) to run single-sequence decoding.",
            reason = capability.reason(),
            cap = capability.effective_max_batch(asked),
        );
    }
    Ok(())
}

/// Build one model handle (plain or pipeline) from a `ModelSpec`.
///
/// `config` must already be validated.  This is the single shared construction
/// path used by both startup (`ModelRegistry::from_specs`) and runtime lazy
/// loading (`ModelRegistry::load`).  It is a **blocking** function (it calls
/// `Engine::from_dir`, which takes seconds) and must therefore be invoked from a
/// blocking context (e.g. at startup or via `tokio::task::spawn_blocking`).
#[cfg(test)]
pub(crate) fn build_handle(spec: &ModelSpec, config: &ServerConfig) -> anyhow::Result<ModelHandle> {
    let authorities = Arc::new(ServerMemoryAuthorities::new(
        config.engine_config.limits.vram_limit,
    ));
    build_handle_with_authorities(spec, config, authorities)
}

pub(crate) fn build_handle_with_authorities(
    spec: &ModelSpec,
    config: &ServerConfig,
    authorities: Arc<ServerMemoryAuthorities>,
) -> anyhow::Result<ModelHandle> {
    onnx_genai_engine::validate_pipeline_backend_request(config.engine_config.decode_backend)?;
    let model_dir = spec.path.as_path();
    let model_id = spec.id.clone();
    let chat_template = load_chat_template(model_dir)?;
    if let Some(directory) = PipelineModelDirectory::load_if_declared(model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to discover pipeline directory: {e}"))?
    {
        // The directory already parsed this package's metadata; re-reading it
        // here could disagree with the pipeline built from it.
        let model_max_context = directory
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.model.as_ref())
            .and_then(|model| model.max_sequence_length);
        return build_pipeline_handle(
            model_dir,
            model_id,
            config,
            model_max_context,
            chat_template,
            directory,
            authorities,
        );
    }
    let model_directory = ModelDirectory::load(model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {e}"))?;
    let model_max_context = load_model_max_context(model_directory.metadata_path.as_deref())?;
    let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
    let engine = Engine::from_dir_with_memory_authority_provider(
        model_dir,
        config.engine_config.clone(),
        authorities,
    )?;
    let fim_config = engine.fim_config().cloned();
    // Capture the model author's declared generation defaults before the engine
    // is moved into the driver, so every request built for this handle can honor
    // a model that ships `do_sample: true` instead of forcing greedy.
    let generation_defaults = engine.metadata().generation.clone();
    // Refuse an explicit `--max-batch > 1` the decode path cannot honor, rather
    // than accepting it and silently falling back to per-request decoding. The
    // check is sourced from the engine's decode path, not from whether an ORT
    // session exists (issue #750).
    enforce_requested_max_batch(&engine, config.max_batch)?;
    let requested_max_batch = config.max_batch.unwrap_or(DEFAULT_MAX_BATCH);
    let engine_driver = EngineDriver::start(engine, requested_max_batch, config.max_queue_depth);
    Ok(ModelHandle::new(ModelHandleParts {
        id: model_id,
        model_dir: model_dir.to_path_buf(),
        engine: engine_driver,
        tokenizer: Arc::new(tokenizer),
        chat_template: chat_template.map(Arc::new),
        model_max_context,
        generation_defaults,
        fim_config,
        pipeline: false,
        multimodal: None,
        text_to_image: false,
        text_to_audio: false,
    }))
}

fn build_pipeline_handle(
    model_dir: &Path,
    model_id: String,
    config: &ServerConfig,
    model_max_context: Option<usize>,
    chat_template: Option<ChatTemplate>,
    directory: PipelineModelDirectory,
    authorities: Arc<ServerMemoryAuthorities>,
) -> anyhow::Result<ModelHandle> {
    let tokenizer_path = crate::multimodal::tokenizer_path(model_dir, &directory)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load pipeline tokenizer: {e}"))?;

    // A package that declares a denoise loop can serve image generation; one
    // whose pipeline ends in a waveform stage can serve speech.
    let text_to_image = directory.spec.strategy.denoiser.is_some();
    let text_to_audio = onnx_genai::text_to_audio::is_text_to_audio(&directory.spec);
    let engine = Engine::from_pipeline_dir_with_memory_authority_provider(
        model_dir,
        config.engine_config.clone(),
        authorities,
    )?;
    let multimodal = crate::multimodal::build(&directory, engine.models())?;
    // The package's own `generation` block is the pipeline's declared sampling
    // regime, so honor it exactly as the CLI does. Without it every request
    // decodes greedily at the OpenAI schema defaults, and a model that ships
    // `do_sample: true` degenerates into repetition.
    let generation_defaults = engine.generation_defaults().cloned();
    Ok(ModelHandle::new(ModelHandleParts {
        id: model_id,
        model_dir: model_dir.to_path_buf(),
        engine: EngineDriver::start_pipeline(engine, config.max_queue_depth),
        tokenizer: Arc::new(tokenizer),
        chat_template: chat_template.map(Arc::new),
        model_max_context,
        generation_defaults,
        fim_config: None,
        pipeline: true,
        multimodal: Some(multimodal),
        text_to_image,
        text_to_audio,
    }))
}

#[cfg(test)]
mod authority_tests {
    #[cfg(not(feature = "native-backend"))]
    use std::path::PathBuf;

    use super::*;

    #[cfg(not(feature = "native-backend"))]
    #[test]
    fn shared_server_load_validates_native_backend_before_model_io() {
        let spec = ModelSpec {
            id: "missing-native".to_string(),
            path: PathBuf::from("this-model-directory-does-not-exist"),
            eager: true,
            warmup: false,
        };
        let config = ServerConfig {
            engine_config: EngineConfig {
                decode_backend: onnx_genai_engine::EngineDecodeBackend::Native,
                ..EngineConfig::default()
            },
            ..ServerConfig::default()
        };

        let error = build_handle(&spec, &config)
            .err()
            .expect("native backend validation must fail")
            .to_string();
        assert!(error.contains("compiled without the 'native-backend' feature"));
        assert!(!error.contains("does not exist"), "{error}");
    }

    #[test]
    fn concurrent_requests_create_one_authority() {
        let provider = Arc::new(ServerMemoryAuthorities::new(ResourceLimit::Bytes(100)));
        let threads = (0..8)
            .map(|_| {
                let provider = Arc::clone(&provider);
                std::thread::spawn(move || {
                    provider
                        .authority(&DeviceCompatibilityDomain::Cuda(0), 100)
                        .unwrap()
                        .authority_id()
                })
            })
            .collect::<Vec<_>>();
        let ids = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert!(ids.iter().all(|id| *id == ids[0]));
    }

    #[test]
    fn conflicting_model_limit_names_both_values() {
        let provider = ServerMemoryAuthorities::new(ResourceLimit::Bytes(100));
        let error = provider
            .validate_limit(
                &DeviceCompatibilityDomain::Cuda(0),
                ResourceLimit::Bytes(90),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("90 bytes"));
        assert!(error.contains("100 bytes"));
    }

    #[test]
    fn auto_policy_cannot_be_replaced_by_model_metadata() {
        let provider = ServerMemoryAuthorities::new(ResourceLimit::Auto);
        provider
            .validate_limit(&DeviceCompatibilityDomain::Cuda(0), ResourceLimit::Auto)
            .unwrap();
        provider
            .validate_limit(
                &DeviceCompatibilityDomain::Cuda(0),
                ResourceLimit::Bytes(90),
            )
            .unwrap_err();
        assert_eq!(provider.configured_limit().unwrap(), ResourceLimit::Auto);
    }

    #[test]
    fn different_device_keys_create_different_authorities() {
        let provider = ServerMemoryAuthorities::new(ResourceLimit::Bytes(100));
        let first = provider
            .authority(&DeviceCompatibilityDomain::Cuda(0), 100)
            .unwrap();
        let second = provider
            .authority(&DeviceCompatibilityDomain::Cuda(1), 100)
            .unwrap();
        assert_ne!(first.authority_id(), second.authority_id());
    }

    #[test]
    fn multi_device_limit_failure_leaves_every_limit_and_policy_unchanged() {
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor, MemoryRole, Tier};

        let provider = ServerMemoryAuthorities::new(ResourceLimit::Bytes(100));
        let first = provider
            .authority(&DeviceCompatibilityDomain::Cuda(0), 100)
            .unwrap();
        let second = provider
            .authority(&DeviceCompatibilityDomain::Cuda(1), 100)
            .unwrap();
        let _first_lease = first
            .reserve(Tier::Device, 20, MemoryRole::Weights, HolderId::new(1))
            .unwrap();
        let _second_lease = second
            .reserve(Tier::Device, 80, MemoryRole::Weights, HolderId::new(2))
            .unwrap();

        provider
            .reconfigure_limit(ResourceLimit::Bytes(50))
            .unwrap_err();
        assert_eq!(first.limit_bytes(), 100);
        assert_eq!(second.limit_bytes(), 100);
        assert_eq!(
            provider.configured_limit().unwrap(),
            ResourceLimit::Bytes(100)
        );
        let future = provider
            .authority(&DeviceCompatibilityDomain::Cuda(2), 100)
            .unwrap();
        assert_eq!(future.limit_bytes(), 100);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn failed_multi_device_preflight_does_not_trim_an_earlier_pool() {
        use cudarc::driver::CudaContext;
        use onnx_runtime_ep_cuda::virtual_memory::{CudaVirtualBacking, PhysicalHandlePool};
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor, MemoryRole, Tier};
        use onnx_runtime_virtual_memory::VirtualBacking;

        let context = CudaContext::new(0).expect("CUDA driver");
        let probe = CudaVirtualBacking::new(Arc::clone(&context), 0);
        let granule = probe.granularity() as u64;
        let initial_limit = granule * 4;
        let provider = ServerMemoryAuthorities::new(ResourceLimit::Bytes(initial_limit));
        let first = provider
            .authority(&DeviceCompatibilityDomain::Cuda(0), initial_limit)
            .unwrap();
        let second = provider
            .authority(&DeviceCompatibilityDomain::Cuda(1), initial_limit)
            .unwrap();
        let pool = PhysicalHandlePool::get_or_create(
            context,
            0,
            (granule * 2) as usize,
            &first,
            HolderId::new(10),
            MemoryRole::Weights,
        )
        .expect("first-device pool");
        let stats = pool.stats();
        let backing = CudaVirtualBacking::with_physical_pool(pool);
        let mut reservation = backing
            .reserve((granule * 2) as usize)
            .expect("reservation");
        backing
            .commit(&mut reservation, 0, (granule * 2) as usize)
            .expect("commit pooled bytes");
        backing
            .release(&mut reservation, 0, (granule * 2) as usize)
            .expect("return pooled bytes");
        let _mapped_second = second
            .reserve(
                Tier::Device,
                granule * 2,
                MemoryRole::Weights,
                HolderId::new(11),
            )
            .expect("non-releasable second-device bytes");
        let before = stats.snapshot();

        provider
            .reconfigure_limit(ResourceLimit::Bytes(granule))
            .expect_err("second device prevents shrink");
        let after = stats.snapshot();
        assert_eq!(after.releases, before.releases);
        assert_eq!(after.creates, before.creates);
        assert_eq!(after.pooled_unmapped_bytes, before.pooled_unmapped_bytes);
        assert_eq!(first.used_bytes(), granule * 2);
        assert_eq!(first.limit_bytes(), initial_limit);
        assert_eq!(second.limit_bytes(), initial_limit);
        assert_eq!(
            provider.configured_limit().unwrap(),
            ResourceLimit::Bytes(initial_limit)
        );
    }

    #[test]
    fn multi_device_limit_success_updates_every_authority_and_policy() {
        let provider = ServerMemoryAuthorities::new(ResourceLimit::Bytes(100));
        let first = provider
            .authority(&DeviceCompatibilityDomain::Cuda(0), 100)
            .unwrap();
        let second = provider
            .authority(&DeviceCompatibilityDomain::Cuda(1), 100)
            .unwrap();

        provider
            .reconfigure_limit(ResourceLimit::Bytes(50))
            .unwrap();
        assert_eq!(first.limit_bytes(), 50);
        assert_eq!(second.limit_bytes(), 50);
        assert_eq!(
            provider.configured_limit().unwrap(),
            ResourceLimit::Bytes(50)
        );
        let future = provider
            .authority(&DeviceCompatibilityDomain::Cuda(2), 50)
            .unwrap();
        assert_eq!(future.limit_bytes(), 50);
    }
}
