//! Main generation engine.

use crate::FimConfig;
use crate::config::{ResolvedMtpConfig, validate_resolved_mtp_config};
use crate::connector_bridge::ConnectorBridge;
use crate::decode::{
    DecodeState, ModelDecodePath, detect_model_decode_path, next_session_token_argmax,
    next_session_token_logits, next_session_token_sampled,
};
use crate::decode_loop::{
    DecodeLoopBackend, DecodeLoopState, exceeded_context_limit, run_decode_loop, step_decode_loop,
};
use crate::kv_bridge::{
    KvModelInfo, PlacedPayload, attach_pages_to_sequence, chunk_payload_from_exported,
    common_prefix_len, exported_layers_from_runner, infer_kv_model_info, kv_model_past_is_f32,
    load_materialized_past, past_kv_from_payloads, sequence_pages_for_len,
};
use crate::logits::{StopSequence, TokenId};
use crate::processors::{
    build_processor_chain, ensure_constrained_finish, load_fim_config_from_model_dir,
    push_unique_stop_sequence,
};
use crate::sampling::{Sampler, SamplingRng};
use crate::session::{ActiveGenerate, DraftModel, DraftSession, EngineSession};
use anyhow::Context;
use onnx_genai_kv::{Device, KvCacheOps, KvDType, LocalTieredConnector, PagedKvCache, PrefixCache};
use onnx_genai_metadata::{InferenceMetadata, ProposalType, SpeculatorProposerStatus};
use onnx_genai_ort::{
    DataType, Eagle3DecodeSession, Environment, ModelDirectory, MtpDecodeSession, Session,
    SessionOptions, SharedKvProposerSession, Tokenizer,
};
use onnx_genai_scheduler::{
    CapacityProvider, CapacityProviders, FixedCapacity, GovernorReconfigureOutcome,
    GovernorSnapshot, ModelKvConfig, Priority, ResourceError, ResourceGovernor, ResourceLimit,
    ResourceLimits, Scheduler, VramBreakdown,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub use crate::config::{
    Eagle3Config, EngineConfig, EngineConfigError, EngineDecodeBackend, FinishReason,
    GenerateConstraint, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult,
    GenerateToken, GenerateTokenCallback, KvConnectorBackend, KvConnectorConfig, LimitParseError,
    MtpCacheScope, MtpConfig, MtpHiddenLayout, MtpWeightSource, PrioritizedGenerateRequest,
    PrioritizedGenerateResult, ScheduledGenerateArrival, SessionId, SharedKvBinding,
    SharedKvProposerConfig, SpeculativeMode, TokenLogprob, parse_resource_limit,
};
pub use crate::connector_bridge::{ConnectorLookupOutcome, ConnectorStats};
use crate::speculative::{
    LinearEmbedder, LinearLmHead, MtpEmbedder, MtpLmHead, SpeculativeStats,
    load_target_initializer_adapters,
};

#[cfg(feature = "native-backend")]
pub(crate) fn resolve_native_decode_device(
    configured: Option<crate::native_decode::NativeDecodeDevice>,
    session_options: &SessionOptions,
) -> anyhow::Result<crate::native_decode::NativeDecodeDevice> {
    use crate::native_decode::NativeDecodeDevice;

    if let Some(device) = configured {
        return validate_native_decode_device(device);
    }

    for provider in &session_options.execution_providers {
        if !provider.caps.is_host() && !(provider.caps.is_gpu() && provider.caps.is_nvidia()) {
            anyhow::bail!(
                "native decoder backend does not support execution provider {provider:?}; supported devices are CPU and CUDA"
            );
        }
    }

    match session_options
        .execution_providers
        .iter()
        .find(|provider| !provider.caps.is_host())
    {
        None => Ok(NativeDecodeDevice::Cpu),
        Some(provider) if provider.caps.is_gpu() && provider.caps.is_nvidia() => {
            let device_id = provider.caps.device_id().unwrap_or(0);
            let index = u32::try_from(device_id).map_err(|_| {
                anyhow::anyhow!(
                    "native decoder backend CUDA device id must be non-negative, got {device_id}"
                )
            })?;
            validate_native_decode_device(NativeDecodeDevice::Cuda { index: Some(index) })
        }
        Some(provider) => {
            unreachable!("unsupported native provider already rejected: {provider:?}")
        }
    }
}

#[cfg(feature = "native-backend")]
fn validate_native_decode_device(
    device: crate::native_decode::NativeDecodeDevice,
) -> anyhow::Result<crate::native_decode::NativeDecodeDevice> {
    match device {
        crate::native_decode::NativeDecodeDevice::Cpu => Ok(device),
        crate::native_decode::NativeDecodeDevice::Cuda { .. } => {
            #[cfg(feature = "cuda")]
            {
                Ok(device)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!(
                    "native decoder backend CUDA device requires building onnx-genai-engine with both the 'native-backend' and 'cuda' features"
                )
            }
        }
    }
}

// Provisional vendor-neutral capacities used until the active EP, OS, and
// filesystem supply real providers. Configured limits are resolved against
// these conservative constants; they never manufacture additional capacity.
const PROVISIONAL_VRAM_CAPACITY_BYTES: u64 = 8 << 30;
const PROVISIONAL_HOST_RAM_CAPACITY_BYTES: u64 = 16 << 30;
const PROVISIONAL_DISK_CAPACITY_BYTES: u64 = 16 << 30;

/// Engine-owned Resource Governor handle.
pub struct EngineResourceGovernor {
    inner: ResourceGovernor,
    allow_runtime_override: bool,
    #[cfg(feature = "native-backend")]
    weight_offload_host_cache: onnx_runtime_ep_cpu::WeightOffloadHostCache,
}

impl EngineResourceGovernor {
    fn new(
        limits: ResourceLimits,
        allow_runtime_override: bool,
        kv_config: ModelKvConfig,
    ) -> Result<Self, ResourceError> {
        let capacities = fallback_capacity_providers(&limits);
        Self::new_with_capacities(limits, allow_runtime_override, capacities, kv_config)
    }

    fn new_with_capacities(
        limits: ResourceLimits,
        allow_runtime_override: bool,
        capacities: CapacityProviders,
        kv_config: ModelKvConfig,
    ) -> Result<Self, ResourceError> {
        // TODO(RULES.md #2, §26.11.4): replace provisional capacities and zero
        // fixed reservations with vendor-neutral EP-backed device capacity/usage
        // queries plus OS/filesystem providers for the warm and cold tiers.
        let inner = ResourceGovernor::new(
            limits,
            capacities,
            VramBreakdown {
                model_weights_bytes: 0,
                activations_bytes: 0,
                ort_overhead_bytes: 0,
            },
            kv_config,
        )?;
        #[cfg(feature = "native-backend")]
        let weight_offload_host_cache = onnx_runtime_ep_cpu::WeightOffloadHostCache::new(
            inner.snapshot().resolved_limits.host_ram_bytes,
        )
        .map_err(|reason| ResourceError::BudgetArithmeticOverflow {
            operation: "configuring the native weight-offload host-cache sub-budget",
            reason: reason.into(),
        })?;
        Ok(Self {
            inner,
            allow_runtime_override,
            #[cfg(feature = "native-backend")]
            weight_offload_host_cache,
        })
    }

    /// Point-in-time configured, resolved, derived, and live per-tier state.
    pub fn snapshot(&self) -> GovernorSnapshot {
        self.inner.snapshot()
    }

    /// Change the live VRAM ceiling when runtime overrides are enabled.
    pub fn set_vram_limit(
        &self,
        limit: ResourceLimit,
    ) -> Result<GovernorReconfigureOutcome, EngineGovernorError> {
        if !self.allow_runtime_override {
            return Err(EngineGovernorError::RuntimeOverrideDisabled);
        }
        // TODO(§26.11.2): execute the returned priority/offload/eviction order
        // across live engine sessions when the outcome reports an overage.
        Ok(self.inner.set_vram_limit(limit)?)
    }

    fn byte_budget(&self) -> onnx_genai_scheduler::ByteBudget {
        self.inner.byte_budget()
    }

    #[cfg(feature = "native-backend")]
    fn weight_offload_host_cache(&self) -> onnx_runtime_ep_cpu::WeightOffloadHostCache {
        self.weight_offload_host_cache.clone()
    }
}

/// Failure from an engine-level live governor operation.
#[derive(Debug, thiserror::Error)]
pub enum EngineGovernorError {
    #[error(
        "runtime resource-limit override is disabled; set \
         serving.memory.limits.allow_runtime_override: true or construct EngineConfig with \
         allow_runtime_override = true before calling set_vram_limit"
    )]
    RuntimeOverrideDisabled,
    #[error(transparent)]
    Resource(#[from] ResourceError),
}

fn fallback_capacity_providers(limits: &ResourceLimits) -> CapacityProviders {
    let disk_spill = limits.disk_spill_limit.map(|_| {
        Arc::new(FixedCapacity::new(
            PROVISIONAL_DISK_CAPACITY_BYTES,
            PROVISIONAL_DISK_CAPACITY_BYTES,
        )) as Arc<dyn CapacityProvider>
    });
    CapacityProviders {
        vram: Arc::new(FixedCapacity::new(
            PROVISIONAL_VRAM_CAPACITY_BYTES,
            PROVISIONAL_VRAM_CAPACITY_BYTES,
        )),
        host_ram: Arc::new(FixedCapacity::new(
            PROVISIONAL_HOST_RAM_CAPACITY_BYTES,
            PROVISIONAL_HOST_RAM_CAPACITY_BYTES,
        )),
        disk_spill,
    }
}

fn governor_kv_config(
    kv_model: Option<&KvModelInfo>,
    config: &EngineConfig,
) -> anyhow::Result<ModelKvConfig> {
    let tokens_per_page = u64::try_from(config.page_size)
        .context("KV page size does not fit the Resource Governor's u64 accounting")?
        .max(1);
    let Some(kv_model) = kv_model else {
        return Ok(ModelKvConfig {
            page_size_bytes: tokens_per_page,
            tokens_per_page,
        });
    };

    let page_size = u64::try_from(config.page_size)
        .context("KV page size does not fit the Resource Governor's u64 accounting")?;
    let mut page_size_bytes = 0_u64;
    for layer in &kv_model.layer_configs {
        let heads = u64::try_from(layer.num_kv_heads)
            .context("KV head count does not fit Resource Governor accounting")?;
        let head_dim = u64::try_from(layer.head_dim)
            .context("KV head dimension does not fit Resource Governor accounting")?;
        let values = 2_u64
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(head_dim))
            .context("KV page value count overflowed Resource Governor accounting")?;
        let layer_bytes = match config.kv_cache_dtype {
            KvDType::F32 => values.checked_mul(4),
            KvDType::Int8 | KvDType::Fp8E4M3Fn | KvDType::Fp8E5M2 => {
                let scales = 2_u64
                    .checked_mul(heads)
                    .and_then(|value| value.checked_mul(page_size))
                    .and_then(|value| value.checked_mul(4))
                    .context(
                        "KV quantization scale size overflowed Resource Governor accounting",
                    )?;
                values.checked_add(scales)
            }
        }
        .context("KV page byte size overflowed Resource Governor accounting")?;
        page_size_bytes = page_size_bytes
            .checked_add(layer_bytes)
            .context("total KV page byte size overflowed Resource Governor accounting")?;
    }
    Ok(ModelKvConfig {
        page_size_bytes: page_size_bytes.max(1),
        tokens_per_page,
    })
}

pub(crate) fn resolved_host_ram_budget(
    config: &EngineConfig,
    kv_model: Option<&KvModelInfo>,
) -> anyhow::Result<u64> {
    let governor = EngineResourceGovernor::new(
        config.limits.clone(),
        config.allow_runtime_override,
        governor_kv_config(kv_model, config)?,
    )
    .context("failed to resolve the engine memory budget for decoder fixed state")?;
    Ok(governor.snapshot().resolved_limits.host_ram_bytes)
}

/// The generation engine.
pub struct Engine {
    /// Resolved decoder execution backend.
    decode_backend: EngineDecodeBackend,
    /// Model inference metadata.
    pub(crate) metadata: InferenceMetadata,
    /// KV cache manager.
    pub(crate) kv_cache: PagedKvCache,
    /// Shared-prefix cache for reusing paged KV across sessions.
    pub(crate) prefix_cache: PrefixCache,
    /// Token-only prefix index used by ORT-owned decode sessions until page import/export lands.
    pub(crate) token_prefix_cache: Vec<Vec<TokenId>>,
    /// KV tensor layout inferred from model present/past TensorInfo.
    pub(crate) kv_model: Option<KvModelInfo>,
    /// ORT decode path selected by model I/O introspection.
    pub(crate) decode_path: ModelDecodePath,
    /// Batch scheduler.
    pub(crate) scheduler: Scheduler,
    /// Per-device resource ceilings and shared scheduler byte budget.
    governor: EngineResourceGovernor,
    /// Persistent multi-turn session state, keyed by session id.
    pub(crate) sessions: HashMap<SessionId, EngineSession>,
    /// ORT session for decoder execution.
    pub(crate) session: Option<Box<Session>>,
    /// Native decoder session. Native execution is single-request and serialized
    /// by the server's fallback driver in this first milestone.
    #[cfg(feature = "native-backend")]
    native_session: Option<crate::native_decode::NativeDecodeSession>,
    /// Native shared-KV proposer loaded from the same metadata contract.
    #[cfg(feature = "native-backend")]
    native_shared_kv_proposer: Option<NativeSharedKvProposerModel>,
    /// Optional draft model used by the speculative decoding path.
    pub(crate) draft: Option<DraftModel>,
    /// Optional MTP head and target-side projections.
    pub(crate) mtp: Option<MtpModel>,
    /// Optional EAGLE-3 head and target-side embedding.
    pub(crate) eagle3: Option<Eagle3Model>,
    /// Optional shared-KV draft proposer.
    pub(crate) shared_kv_proposer: Option<SharedKvProposerModel>,
    /// Tokenizer loaded from the model directory.
    pub(crate) tokenizer: Tokenizer,
    /// Auto-detected fill-in-the-middle token configuration.
    pub(crate) fim_config: Option<FimConfig>,
    /// Default speculative draft width K.
    pub(crate) num_speculative_tokens: usize,
    /// Default speculative candidate source.
    pub(crate) speculative_mode: SpeculativeMode,
    /// Diagnostics from the most recent generation call.
    pub(crate) last_speculative_stats: SpeculativeStats,
    /// Optional distributed KV connector bridge (DESIGN §38, K3). Inert when
    /// configured as `Null` (the default), preserving in-process-only behavior.
    pub(crate) connector: ConnectorBridge,
    /// ORT environment — MUST be the LAST field so it (and the plugin EP factory it owns via
    /// RegisterExecutionProviderLibrary) drops AFTER every Session/draft/mtp/eagle3 field above.
    /// Rust drops struct fields in declaration order; if the env dropped first, ORT would tear down
    /// the plugin EP factory before the sessions, causing a teardown use-after-free (segfault) in
    /// the Metal/MLX plugin EP's allocator/data-transfer/context release path.
    pub(crate) _environment: Environment,
}

// SAFETY: `Engine` owns every ORT or native-runtime handle reachable through
// its sessions and decode state. Neither runtime's sessions, values, bindings,
// allocators, or CPU tensors have thread affinity. Moving the engine transfers
// exclusive ownership; mutation still requires `&mut Engine`. Self-references
// in ORT decode runners point into boxed `Session` allocations, whose addresses
// remain stable when the owning `Engine` moves. This would stop being sound if
// an execution provider introduced thread-affine handles or a field gained
// unsynchronized shared mutation.
unsafe impl Send for Engine {}

pub(crate) struct MtpModel {
    pub(crate) config: MtpConfig,
    pub(crate) runtime_config: ResolvedMtpConfig,
    pub(crate) session: Arc<Session>,
    pub(crate) embedder: MtpEmbedder,
    pub(crate) lm_head: MtpLmHead,
    pub(crate) hidden_output: String,
    pub(crate) kv_mode: onnx_genai_ort::MtpDraftKvMode,
    pub(crate) num_speculative_tokens: usize,
}

pub(crate) struct Eagle3Model {
    pub(crate) config: Eagle3Config,
    pub(crate) session: Box<Session>,
    pub(crate) embedder: LinearEmbedder,
    pub(crate) hidden_outputs: Vec<String>,
    pub(crate) kv_mode: onnx_genai_ort::Eagle3DraftKvMode,
    pub(crate) num_speculative_tokens: usize,
}

pub(crate) struct SharedKvProposerModel {
    pub(crate) config: SharedKvProposerConfig,
    pub(crate) session: Box<Session>,
    /// Target input-token embedding table, used to build the token-embedding
    /// half of each draft step's `inputs_embeds`.
    pub(crate) embedder: LinearEmbedder,
    pub(crate) num_speculative_tokens: usize,
}

#[cfg(feature = "native-backend")]
pub(crate) struct NativeSharedKvProposerModel {
    pub(crate) session: crate::native_decode::NativeProposerSession,
    pub(crate) embedder: LinearEmbedder,
    pub(crate) groups: Vec<onnx_genai_metadata::SharedKvGroup>,
    pub(crate) hidden_size: usize,
}

impl Engine {
    /// Load a model from a directory.
    pub fn from_dir(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        Self::from_dir_with_session_options(model_dir, config, SessionOptions::default())
    }

    /// Load a model from a directory with explicit ORT session options.
    pub fn from_dir_with_session_options(
        model_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
    ) -> anyhow::Result<Self> {
        let model_directory = ModelDirectory::load(model_dir)
            .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {}", e))?;
        let decode_backend =
            resolve_decode_backend(&model_directory.model_path, config.decode_backend)?;
        if decode_backend == EngineDecodeBackend::Native {
            return augment_backend_error(
                Self::from_native_model_directory(model_directory, config, &session_options),
                EngineDecodeBackend::Native,
            );
        }

        // ORT CUDA graph capture is opt-in: it fails with unconstructed OrtValue
        // outputs on some Foundry exports. SessionOptions still honors an explicit
        // ONNX_GENAI_CUDA_GRAPH=1 request; native whole-step capture is separate.
        let mut session_options = session_options;
        configure_ort_cuda_graph(&mut session_options, &model_directory.model_path);

        let environment = Environment::new("onnx-genai-engine")
            .map_err(|e| anyhow::anyhow!("Failed to create ORT environment: {}", e))?;
        let session = augment_backend_error(
            Session::new(
                &environment,
                &model_directory.model_path,
                session_options.clone(),
            )
            .map_err(|e| anyhow::anyhow!("Failed to load ORT session: {}", e)),
            EngineDecodeBackend::Ort,
        )?;

        // Resolve inference metadata. Our own `inference_metadata.yaml` is the
        // canonical source of truth. When a model ships without it (e.g. the
        // onnxruntime-genai / Foundry Local models, which carry only a
        // `genai_config.json`), fall back to converting that config into native
        // metadata so share-buffer-capable GQA models still get the O(1)/token
        // decode path instead of the growing rebind path.
        let metadata = if let Some(metadata_path) = &model_directory.metadata_path {
            onnx_genai_metadata::load_metadata(metadata_path)
                .map_err(|e| anyhow::anyhow!("Failed to load metadata: {}", e))?
        } else if let Some(compat) = genai_config_compat_metadata(&model_directory.root, &session)?
        {
            tracing::info!(
                "No inference_metadata.yaml found; derived inference metadata from genai_config.json (onnxruntime-genai compatibility)"
            );
            compat
        } else {
            tracing::warn!("No inference metadata found, using defaults");
            default_inference_metadata()
        };

        // Validate capabilities
        let runtime_caps = onnx_genai_metadata::RuntimeCapabilities::default();
        if let Err(unsupported) = onnx_genai_metadata::validate(&metadata, &runtime_caps) {
            anyhow::bail!("Unsupported capabilities: {:?}", unsupported);
        }

        // Optional cap on the runtime-owned fixed-capacity KV buffer. Foundry /
        // onnxruntime-genai `genai_config.json` models advertise the model's full
        // `context_length` (e.g. 32k-131k) as their max sequence length, and the
        // shared-buffer decode path pre-allocates a KV buffer of exactly that many
        // tokens up front — regardless of how many tokens a request will actually
        // generate. On memory-constrained devices that over-allocation exhausts
        // VRAM (spilling to shared system memory over PCIe) even for short runs.
        // `ONNX_GENAI_KV_MAX_LEN` caps that capacity to the caller's real
        // generation budget (prompt + max_new_tokens), mirroring the native
        // path's `ONNX_GENAI_CUDA_KV_MAX_LEN`. Unset = unchanged (full context).
        let kv_shared_buffer_cap = shared_buffer_cap_from_env();
        let metadata_max_context = metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length)
            .map(|max_len| cap_kv_len(max_len, kv_shared_buffer_cap));
        // Our own inference metadata (inference_metadata.yaml), not
        // onnxruntime-genai's genai_config.json, drives the runtime-owned
        // share-buffer KV path for GQA models.
        let shared_kv_max_len = crate::decode::shared_kv_buffer_len_from_metadata(&metadata)
            .map(|max_len| cap_kv_len(max_len, kv_shared_buffer_cap));
        let sliding_window = crate::decode::sliding_window_from_metadata(&metadata)?;
        let sink_tokens = crate::decode::sink_tokens_from_metadata(&metadata);
        let decode_path = detect_model_decode_path(
            &session,
            metadata_max_context,
            shared_kv_max_len,
            sliding_window,
            sink_tokens,
        )?;
        let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let fim_config = load_fim_config_from_model_dir(&model_directory.root)?;
        let kv_model = infer_kv_model_info(&session, config.page_size, config.kv_cache_dtype)?;
        let governor_kv_config = governor_kv_config(kv_model.as_ref(), &config)?;
        let governor = EngineResourceGovernor::new(
            config.limits.clone(),
            config.allow_runtime_override,
            governor_kv_config,
        )
        .map_err(|error| anyhow::anyhow!("Failed to initialize Resource Governor: {error}"))?;
        let mut scheduler_config = config.scheduler.clone();
        if scheduler_config.bytes_per_token.is_none() {
            scheduler_config.bytes_per_token = Some(
                governor_kv_config
                    .page_size_bytes
                    .div_ceil(governor_kv_config.tokens_per_page),
            );
        }
        let scheduler = Scheduler::with_byte_budget(scheduler_config, governor.byte_budget());
        let draft = if let Some(draft_model_path) = &config.draft_model {
            let draft_directory = ModelDirectory::load(draft_model_path)
                .map_err(|e| anyhow::anyhow!("Failed to resolve draft model directory: {}", e))?;
            let draft_session = Session::new(
                &environment,
                &draft_directory.model_path,
                session_options.clone(),
            )
            .map_err(|e| anyhow::anyhow!("Failed to load draft ORT session: {}", e))?;
            let draft_decode_path =
                // Draft models are loaded with sliding_window=None and sink_tokens=0:
                // draft architectures are typically distinct from the target (e.g. a
                // smaller model with a full KV cache) and declare their own attention
                // constraints through their own inference metadata. Even if the target
                // uses SWA + attention sinks (sink_tokens > 0), propagating the
                // target's sink_tokens to the draft would be a silent no-op — all
                // sink/window management is gated on `sliding_window.is_some()`, which
                // is None here — and it would mask a future bug if a windowed draft
                // path were introduced without explicitly loading draft metadata.
                // If a draft model needs its own SWA + sinks, load its
                // inference_metadata.yaml and pass the values from there.
                detect_model_decode_path(&draft_session, metadata_max_context, None, None, 0)?;
            let draft_kv_model = infer_kv_model_info(
                &draft_session,
                config.page_size,
                onnx_genai_kv::KvDType::F32,
            )?;
            let draft_kv_cache = if let Some(kv_model) = &draft_kv_model {
                PagedKvCache::new_with_layer_tensor_configs(
                    kv_model.tensor_config.page_size,
                    kv_model.tensor_config.dtype,
                    kv_model.layer_configs.clone(),
                    config.num_gpu_pages,
                )
            } else {
                PagedKvCache::new(config.page_size, config.num_gpu_pages)
            };
            Some(DraftModel {
                session: Box::new(draft_session),
                decode_path: draft_decode_path,
                kv_model: draft_kv_model,
                kv_cache: draft_kv_cache,
            })
        } else {
            None
        };
        let kv_cache = if let Some(kv_model) = &kv_model {
            // The paged tensor layout is derived from present-KV outputs: each
            // layer has key/value tensors shaped like [batch, kv_heads, seq, head_dim].
            // Per-layer geometry (heterogeneous head_dim across layers, e.g. the
            // Gemma-4 sliding/full split) is fed from the model's own KV output
            // shapes so mixed-geometry models page correctly.
            PagedKvCache::new_with_layer_tensor_configs(
                kv_model.tensor_config.page_size,
                kv_model.tensor_config.dtype,
                kv_model.layer_configs.clone(),
                config.num_gpu_pages,
            )
        } else {
            PagedKvCache::new(config.page_size, config.num_gpu_pages)
        };

        let (speculative_mode, resolved_mtp_config) = match config.speculative_mode {
            SpeculativeMode::None if draft.is_some() => (SpeculativeMode::DraftModel, None),
            // No explicit mode: adopt a shared-KV draft proposer advertised by
            // the model's own inference metadata, if the target exposes an f32
            // hidden output the assistant can be seeded from.
            SpeculativeMode::None => {
                if let Some(config) =
                    mtp_config_from_metadata(&metadata, &model_directory.root, &session)?
                {
                    (
                        SpeculativeMode::Mtp(config.public_config.clone()),
                        Some(config),
                    )
                } else {
                    (
                        shared_kv_mode_from_metadata(&model_directory.root, &session)
                            .unwrap_or(SpeculativeMode::None),
                        None,
                    )
                }
            }
            SpeculativeMode::Mtp(config) => (
                SpeculativeMode::Mtp(config.clone()),
                Some(ResolvedMtpConfig::from_manual(config)),
            ),
            mode => (mode, None),
        };
        if let SpeculativeMode::PromptLookup { ngram, max_tokens } = &speculative_mode
            && (*ngram == 0 || *max_tokens == 0)
        {
            anyhow::bail!("prompt-lookup ngram and max_tokens must be greater than zero");
        }
        let mtp = if let Some(mtp_config) = resolved_mtp_config {
            validate_resolved_mtp_config(&mtp_config)?;
            if mtp_config.cache_scope == MtpCacheScope::AcceptedPrefix {
                anyhow::bail!(
                    "MTP kv_mode accepted_prefix is declared but not executable: the frozen Mobius contract does not define correction-token/cache alignment"
                );
            }
            let hidden_output = session
                .outputs()
                .iter()
                .find(|output| output.name == mtp_config.public_config.target_hidden_output)
                .with_context(|| {
                    format!(
                        "MTP target model must expose hidden-state output '{}'",
                        mtp_config.public_config.target_hidden_output
                    )
                })?;
            if !matches!(
                hidden_output.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            ) {
                anyhow::bail!(
                    "MTP target hidden-state output '{}' must be Float32, Float16, or BFloat16, got {:?}",
                    hidden_output.name,
                    hidden_output.dtype
                );
            }
            match mtp_config.target_hidden_layout {
                MtpHiddenLayout::Bsh
                    if hidden_output.shape.len() == 3
                        && hidden_output.shape.last().copied().filter(|dim| *dim > 0)
                            == Some(mtp_config.public_config.hidden_size as i64) => {}
                MtpHiddenLayout::Bshc
                    if hidden_output.shape.len() == 4
                        && hidden_output.shape[2] == mtp_config.hc_mult as i64
                        && hidden_output.shape[3]
                            == mtp_config.public_config.hidden_size as i64 => {}
                _ => anyhow::bail!(
                    "MTP target hidden-state output '{}' shape {:?} does not match configured {:?} with hc_mult {} and hidden size {}",
                    hidden_output.name,
                    hidden_output.shape,
                    mtp_config.target_hidden_layout,
                    mtp_config.hc_mult,
                    mtp_config.public_config.hidden_size
                ),
            }
            let head_session = Session::new(
                &environment,
                &mtp_config.public_config.head_model,
                session_options.clone(),
            )
            .map_err(|error| anyhow::anyhow!("Failed to load MTP head: {error}"))?;
            let decode_options = onnx_genai_ort::MtpDecodeOptions {
                kv_mode: mtp_config.public_config.kv_mode,
                batch_size: 1,
                hc_mult: mtp_config.hc_mult,
                hidden_state_rank4: mtp_config.target_hidden_layout == MtpHiddenLayout::Bshc,
                hidden_output: mtp_config.mtp_hidden_output.clone(),
                state_output: mtp_config.mtp_state_output.clone(),
            };
            let head_signature = MtpDecodeSession::new(&head_session, decode_options)
                .map_err(|error| anyhow::anyhow!("Failed to inspect MTP head: {error}"))?
                .signature()
                .clone();
            if head_signature.hidden_size != mtp_config.public_config.hidden_size {
                anyhow::bail!(
                    "MTP head hidden size {} does not match configured target hidden size {}",
                    head_signature.hidden_size,
                    mtp_config.public_config.hidden_size
                );
            }
            let (embedder, lm_head) = match (
                &mtp_config.embedding_weights,
                &mtp_config.lm_head_weights,
            ) {
                (MtpWeightSource::File(embedding), MtpWeightSource::File(lm_head)) => (
                    MtpEmbedder::Linear(
                        LinearEmbedder::new(
                            read_f32_weights(embedding)?,
                            mtp_config.public_config.vocab_size,
                            mtp_config.public_config.hidden_size,
                        )
                        .map_err(|error| {
                            anyhow::anyhow!("Invalid MTP embedding weights: {error}")
                        })?,
                    ),
                    MtpLmHead::Linear(
                        LinearLmHead::new(
                            read_f32_weights(lm_head)?,
                            mtp_config.public_config.hidden_size,
                            mtp_config.public_config.vocab_size,
                        )
                        .map_err(|error| anyhow::anyhow!("Invalid MTP LM-head weights: {error}"))?,
                    ),
                ),
                (
                    MtpWeightSource::TargetInitializer(embedding),
                    MtpWeightSource::TargetInitializer(lm_head),
                ) => {
                    let (embedder, lm_head, vocab_size) = load_target_initializer_adapters(
                        &model_directory.model_path,
                        embedding,
                        lm_head,
                        mtp_config.public_config.hidden_size,
                    )?;
                    if vocab_size != mtp_config.public_config.vocab_size {
                        anyhow::bail!(
                            "MTP target initializer vocabulary {vocab_size} does not match configured vocabulary {}",
                            mtp_config.public_config.vocab_size
                        );
                    }
                    (embedder, lm_head)
                }
                _ => anyhow::bail!(
                    "MTP embedding_weights and lm_head_weights must both use files or both use target initializers"
                ),
            };
            Some(MtpModel {
                config: mtp_config.public_config.clone(),
                runtime_config: mtp_config.clone(),
                session: Arc::new(head_session),
                embedder,
                lm_head,
                hidden_output: mtp_config.public_config.target_hidden_output.clone(),
                kv_mode: mtp_config.public_config.kv_mode,
                num_speculative_tokens: mtp_config.public_config.num_speculative_tokens,
            })
        } else {
            None
        };
        let eagle3 = if let SpeculativeMode::Eagle3(eagle_config) = &speculative_mode {
            crate::config::validate_eagle3_config(eagle_config)?;
            for output_name in &eagle_config.target_hidden_outputs {
                let hidden_output = session
                    .outputs()
                    .iter()
                    .find(|output| output.name == *output_name)
                    .with_context(|| {
                        format!(
                            "EAGLE-3 target model must expose hidden-state output '{output_name}'"
                        )
                    })?;
                if hidden_output.dtype != DataType::Float32 {
                    anyhow::bail!(
                        "EAGLE-3 target hidden-state output '{}' must be Float32, got {:?}",
                        hidden_output.name,
                        hidden_output.dtype
                    );
                }
                if hidden_output.shape.last().copied().filter(|dim| *dim > 0)
                    != Some(eagle_config.hidden_size as i64)
                {
                    anyhow::bail!(
                        "EAGLE-3 target hidden-state output '{}' shape {:?} does not end in configured hidden size {}",
                        hidden_output.name,
                        hidden_output.shape,
                        eagle_config.hidden_size
                    );
                }
            }
            let head_session = Session::new(
                &environment,
                &eagle_config.head_model,
                session_options.clone(),
            )
            .map_err(|error| anyhow::anyhow!("Failed to load EAGLE-3 head: {error}"))?;
            let head_signature = Eagle3DecodeSession::detect(&head_session)
                .map_err(|error| anyhow::anyhow!("Failed to inspect EAGLE-3 head: {error}"))?
                .context("configured EAGLE-3 head model does not expose EAGLE-3 head I/O")?;
            if head_signature.hidden_size != eagle_config.hidden_size {
                anyhow::bail!(
                    "EAGLE-3 head hidden size {} does not match configured target hidden size {}",
                    head_signature.hidden_size,
                    eagle_config.hidden_size
                );
            }
            let expected_fused =
                eagle_config.hidden_size * eagle_config.target_hidden_outputs.len();
            if head_signature.fused_hidden_size != expected_fused {
                anyhow::bail!(
                    "EAGLE-3 head fused hidden size {} does not match three target layers totaling {}",
                    head_signature.fused_hidden_size,
                    expected_fused
                );
            }
            if head_signature.draft_vocab_size > eagle_config.vocab_size {
                anyhow::bail!(
                    "EAGLE-3 draft vocabulary {} exceeds target embedding vocabulary {}",
                    head_signature.draft_vocab_size,
                    eagle_config.vocab_size
                );
            }
            let embedding = read_f32_weights(&eagle_config.embedding_weights)?;
            Some(Eagle3Model {
                config: eagle_config.clone(),
                session: Box::new(head_session),
                embedder: LinearEmbedder::new(
                    embedding,
                    eagle_config.vocab_size,
                    eagle_config.hidden_size,
                )
                .map_err(|error| anyhow::anyhow!("Invalid EAGLE-3 embedding weights: {error}"))?,
                hidden_outputs: eagle_config.target_hidden_outputs.clone(),
                kv_mode: eagle_config.kv_mode,
                num_speculative_tokens: eagle_config.num_speculative_tokens,
            })
        } else {
            None
        };

        let shared_kv_proposer = if let SpeculativeMode::SharedKv(assistant_config) =
            &speculative_mode
        {
            crate::config::validate_shared_kv_proposer_config(assistant_config)?;
            let hidden_output = session
                .outputs()
                .iter()
                .find(|output| output.name == assistant_config.target_hidden_output)
                .with_context(|| {
                    format!(
                        "shared-KV proposer target model must expose hidden-state output '{}'",
                        assistant_config.target_hidden_output
                    )
                })?;
            if hidden_output.dtype != DataType::Float32 {
                anyhow::bail!(
                    "shared-KV proposer target hidden-state output '{}' must be Float32, got {:?}",
                    hidden_output.name,
                    hidden_output.dtype
                );
            }
            if hidden_output.shape.last().copied().filter(|dim| *dim > 0)
                != Some(assistant_config.backbone_hidden_size as i64)
            {
                anyhow::bail!(
                    "shared-KV proposer target hidden-state output '{}' shape {:?} does not end in configured backbone hidden size {}",
                    hidden_output.name,
                    hidden_output.shape,
                    assistant_config.backbone_hidden_size
                );
            }
            let assistant_session = Session::new(
                &environment,
                &assistant_config.assistant_model,
                session_options.clone(),
            )
            .map_err(|error| anyhow::anyhow!("Failed to load shared-KV proposer model: {error}"))?;
            let signature = SharedKvProposerSession::detect(&assistant_session)
                .map_err(|error| {
                    anyhow::anyhow!("Failed to inspect shared-KV proposer model: {error}")
                })?
                .context("configured shared-KV proposer model does not expose proposer I/O")?;
            if signature.backbone_hidden_size != assistant_config.backbone_hidden_size {
                anyhow::bail!(
                    "shared-KV proposer hidden size {} does not match configured backbone hidden size {}",
                    signature.backbone_hidden_size,
                    assistant_config.backbone_hidden_size
                );
            }
            if signature.vocab_size != assistant_config.vocab_size {
                anyhow::bail!(
                    "shared-KV proposer vocabulary {} does not match configured vocab size {}",
                    signature.vocab_size,
                    assistant_config.vocab_size
                );
            }
            for group in &assistant_config.shared_kv {
                if !signature
                    .shared_kv
                    .iter()
                    .any(|spec| spec.name == group.name)
                {
                    anyhow::bail!(
                        "shared-KV proposer model does not expose shared_kv group '{}'",
                        group.name
                    );
                }
            }
            let embedding = read_f32_weights(&assistant_config.input_embedding_weights)?;
            let embedder = LinearEmbedder::new(
                embedding,
                assistant_config.vocab_size,
                assistant_config.backbone_hidden_size,
            )
            .map_err(|error| {
                anyhow::anyhow!("Invalid shared-KV proposer input embedding weights: {error}")
            })?;
            Some(SharedKvProposerModel {
                config: assistant_config.clone(),
                session: Box::new(assistant_session),
                embedder,
                num_speculative_tokens: assistant_config.num_speculative_tokens,
            })
        } else {
            None
        };

        let connector =
            build_connector_bridge(&config.kv_connector, &model_directory, kv_model.as_ref())?;

        Ok(Self {
            decode_backend,
            metadata,
            kv_cache,
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model,
            decode_path,
            scheduler,
            governor,
            sessions: HashMap::new(),
            _environment: environment,
            session: Some(Box::new(session)),
            #[cfg(feature = "native-backend")]
            native_session: None,
            #[cfg(feature = "native-backend")]
            native_shared_kv_proposer: None,
            draft,
            mtp,
            eagle3,
            shared_kv_proposer,
            tokenizer,
            fim_config,
            num_speculative_tokens: config.num_speculative_tokens.max(1),
            speculative_mode,
            last_speculative_stats: SpeculativeStats::default(),
            connector,
        })
    }

    #[cfg(feature = "native-backend")]
    fn from_native_model_directory(
        model_directory: ModelDirectory,
        config: EngineConfig,
        session_options: &SessionOptions,
    ) -> anyhow::Result<Self> {
        if config.draft_model.is_some() || !matches!(config.speculative_mode, SpeculativeMode::None)
        {
            anyhow::bail!(
                "native decoder backend does not yet support speculative, MTP, EAGLE-3, or shared-KV generation"
            );
        }
        if !matches!(&config.kv_connector.backend, KvConnectorBackend::Null) {
            anyhow::bail!("native decoder backend does not yet support external KV connectors");
        }
        let native_device = resolve_native_decode_device(config.native_device, session_options)?;

        let metadata = if let Some(metadata_path) = &model_directory.metadata_path {
            onnx_genai_metadata::load_metadata(metadata_path)
                .map_err(|e| anyhow::anyhow!("Failed to load metadata: {}", e))?
        } else if let Some(compat) = genai_config_compat_metadata_from_model_path(
            &model_directory.root,
            &model_directory.model_path,
        )? {
            compat
        } else {
            tracing::warn!("No inference metadata found, using defaults");
            default_inference_metadata()
        };
        let runtime_caps = onnx_genai_metadata::RuntimeCapabilities::default();
        if let Err(unsupported) = onnx_genai_metadata::validate(&metadata, &runtime_caps) {
            anyhow::bail!("Unsupported capabilities: {:?}", unsupported);
        }

        let tokenizer = Tokenizer::from_file(&model_directory.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let fim_config = load_fim_config_from_model_dir(&model_directory.root)?;
        let governor_kv_config = governor_kv_config(None, &config)?;
        let governor = EngineResourceGovernor::new(
            config.limits.clone(),
            config.allow_runtime_override,
            governor_kv_config,
        )
        .map_err(|error| anyhow::anyhow!("Failed to initialize Resource Governor: {error}"))?;
        let mut scheduler_config = config.scheduler.clone();
        if scheduler_config.bytes_per_token.is_none() {
            scheduler_config.bytes_per_token = Some(
                governor_kv_config
                    .page_size_bytes
                    .div_ceil(governor_kv_config.tokens_per_page),
            );
        }
        let scheduler = Scheduler::with_byte_budget(scheduler_config, governor.byte_budget());
        let connector = build_connector_bridge(&config.kv_connector, &model_directory, None)?;
        let native_session =
            crate::native_decode::NativeDecodeSession::load_with_weight_offload_host_cache(
                &model_directory.model_path,
                native_device,
                governor.weight_offload_host_cache(),
                metadata.model.as_ref().and_then(|model| model.io.as_ref()),
            )
            .map_err(|error| anyhow::anyhow!("Failed to load native decoder session: {error:#}"))?;
        let (native_shared_kv_proposer, speculative_mode) =
            load_native_shared_kv_proposer(&metadata, &model_directory.root, native_device)?;
        let environment = Environment::new("onnx-genai-engine")
            .map_err(|e| anyhow::anyhow!("Failed to create ORT environment: {}", e))?;

        Ok(Self {
            decode_backend: EngineDecodeBackend::Native,
            metadata,
            kv_cache: PagedKvCache::new(config.page_size, config.num_gpu_pages),
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model: None,
            decode_path: ModelDecodePath::Legacy,
            scheduler,
            governor,
            sessions: HashMap::new(),
            session: None,
            native_session: Some(native_session),
            native_shared_kv_proposer,
            draft: None,
            mtp: None,
            eagle3: None,
            shared_kv_proposer: None,
            tokenizer,
            fim_config,
            num_speculative_tokens: config.num_speculative_tokens.max(1),
            speculative_mode,
            last_speculative_stats: SpeculativeStats::default(),
            connector,
            _environment: environment,
        })
    }

    #[cfg(not(feature = "native-backend"))]
    fn from_native_model_directory(
        _model_directory: ModelDirectory,
        _config: EngineConfig,
        _session_options: &SessionOptions,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "native decoder backend requires building onnx-genai-engine with the 'native-backend' feature"
        )
    }

    #[cfg(feature = "native-backend")]
    fn generate_native_with_callback(
        &mut self,
        mut request: GenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.last_speculative_stats = SpeculativeStats::default();
        if request.options.speculative_mode.is_none() && self.native_shared_kv_proposer.is_some() {
            request.options.speculative_mode = Some(self.speculative_mode.clone());
        }
        reject_native_request_speculation(&request.options)?;
        request.options.validate()?;
        let mut options = request.options;
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        let prompt_tokens = self.tokenize_prompt(&request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        options.max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(&self.tokenizer))?;

        // Speculation ON (implemented greedy prompt-lookup) → the native
        // speculative driver. Every other request stays on the untouched plain
        // M=1 fast path below, preserving the 762 tok/s non-regression guarantee.
        if let Some(plan) = native_speculation_plan(&options, &chain) {
            let mut stats = SpeculativeStats::default();
            let native_session = self
                .native_session
                .as_mut()
                .context("native decoder session is unavailable")?;
            let mut driver = match plan.kind {
                NativeSpeculationKind::PromptLookup { ngram, max_tokens } => {
                    crate::native_speculative::NativeSpeculativeDriver::new_prompt_lookup(
                        native_session,
                        ngram,
                        max_tokens,
                        plan.width,
                    )?
                }
                NativeSpeculationKind::SharedKv => {
                    let proposer = self.native_shared_kv_proposer.as_mut().context(
                        "native shared-KV speculation requested without a loaded proposer session",
                    )?;
                    crate::native_speculative::NativeSpeculativeDriver::new_shared_kv(
                        native_session,
                        &mut proposer.session,
                        &proposer.embedder,
                        &proposer.groups,
                        proposer.hidden_size,
                        plan.width,
                    )?
                }
            };
            let result = augment_backend_error(
                driver.generate(
                    &prompt_tokens,
                    &options,
                    &chain,
                    &self.tokenizer,
                    &mut stats,
                    callback,
                ),
                EngineDecodeBackend::Native,
            );
            self.last_speculative_stats = stats;
            return result;
        }

        let native_session = self
            .native_session
            .as_mut()
            .context("native decoder session is unavailable")?;
        augment_backend_error(
            native_session.generate_with_callback(
                &prompt_tokens,
                &options,
                &chain,
                &self.tokenizer,
                callback,
            ),
            EngineDecodeBackend::Native,
        )
    }

    #[cfg(not(feature = "native-backend"))]
    fn generate_native_with_callback(
        &mut self,
        _request: GenerateRequest,
        _callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        anyhow::bail!(
            "native decoder backend requires building onnx-genai-engine with the 'native-backend' feature"
        )
    }

    fn require_ort_backend(&self, feature: &str) -> anyhow::Result<()> {
        if self.decode_backend == EngineDecodeBackend::Native {
            anyhow::bail!(
                "the native single-session backend does not support {feature}; use independent serialized requests"
            );
        }
        Ok(())
    }

    /// Generate text for a request.
    ///
    pub fn generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult> {
        self.generate_with_callback(request, None)
    }

    /// Generate text using a caller-supplied [`Sampler`] for final token
    /// selection.
    ///
    /// The logit-processor chain (temperature, top-k, top-p, min-p, penalties,
    /// constraints, …) still runs; only the terminal greedy/categorical pick is
    /// replaced by `sampler`. This is the public extension seam that the C ABI
    /// ([`crate::capi`]) exposes to foreign samplers. Not supported on the
    /// native single-session backend.
    pub fn generate_with_sampler(
        &mut self,
        request: GenerateRequest,
        sampler: Box<dyn Sampler>,
    ) -> anyhow::Result<GenerateResult> {
        if self.decode_backend == EngineDecodeBackend::Native {
            anyhow::bail!("custom samplers are not supported on the native single-session backend");
        }
        let session_id = self.create_session()?;
        let result = self.generate_in_session_with_sampler(session_id, request, sampler);
        let close_result = self.close_session(session_id);
        match (result, close_result) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    /// Speculative verification diagnostics from the most recent generation.
    pub fn last_speculative_stats(&self) -> SpeculativeStats {
        self.last_speculative_stats
    }

    /// Native CUDA decode diagnostics from the engine-owned session.
    #[cfg(feature = "native-backend")]
    pub fn native_cuda_debug_stats(&self) -> Option<crate::native_decode::CudaKvDebugStats> {
        self.native_session
            .as_ref()
            .and_then(crate::native_decode::NativeDecodeSession::cuda_kv_debug_stats)
    }

    /// Access the engine-owned Resource Governor handle.
    pub fn governor(&self) -> &EngineResourceGovernor {
        &self.governor
    }

    /// Convenience snapshot of configured and live resource state.
    pub fn resource_snapshot(&self) -> GovernorSnapshot {
        self.governor.snapshot()
    }

    /// Change the live VRAM ceiling when runtime overrides are enabled.
    pub fn set_vram_limit(
        &self,
        limit: ResourceLimit,
    ) -> Result<GovernorReconfigureOutcome, EngineGovernorError> {
        self.governor.set_vram_limit(limit)
    }

    /// External KV connector activity from the most recent generation.
    ///
    /// Reflects lookups, would-be prefix extensions, tokens actually fetched and
    /// injected (K4 materialization), and chunk stores. Returns
    /// [`ConnectorStats::default`] when no connector is configured.
    pub fn last_connector_stats(&self) -> ConnectorStats {
        self.connector.stats().clone()
    }

    /// Generate the middle text for a fill-in-the-middle request.
    pub fn generate_fim(
        &mut self,
        prefix: impl AsRef<str>,
        suffix: impl AsRef<str>,
        options: GenerateOptions,
    ) -> anyhow::Result<GenerateResult> {
        let fim_config = self
            .fim_config
            .clone()
            .context("model tokenizer_config.json does not declare recognized FIM tokens")?;
        self.generate_fim_with_config(prefix, suffix, options, &fim_config)
    }

    /// Generate the middle text using an explicit fill-in-the-middle configuration.
    pub fn generate_fim_with_config(
        &mut self,
        prefix: impl AsRef<str>,
        suffix: impl AsRef<str>,
        options: GenerateOptions,
        fim_config: &FimConfig,
    ) -> anyhow::Result<GenerateResult> {
        let prompt = fim_config.format_prompt(prefix.as_ref(), suffix.as_ref());
        let mut request = GenerateRequest::new(prompt);
        request.options = self.fim_options(fim_config, options);
        self.generate(request)
    }

    /// Generate text and optionally stream each generated token to `callback`.
    pub fn generate_with_callback(
        &mut self,
        request: GenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        if self.decode_backend == EngineDecodeBackend::Native {
            return self.generate_native_with_callback(request, callback);
        }
        let session_id = self.create_session()?;
        let result = self.generate_in_session_with_callback(session_id, request, callback);
        let close_result = self.close_session(session_id);
        match (result, close_result) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    /// Generate text in a persistent session, reusing the session's accumulated KV state.
    pub fn generate_in_session(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_callback(session_id, request, None)
    }

    /// Generate text in a persistent session with an explicit scheduler priority.
    pub fn generate_in_session_with_priority(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        priority: Priority,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_priority_and_callback(
            session_id, request, priority, None, None,
        )
    }

    /// Generate text in a persistent session and optionally stream generated tokens.
    pub fn generate_in_session_with_callback(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_priority_and_callback(
            session_id,
            request,
            Priority::Normal,
            None,
            callback,
        )
    }

    /// Generate text in a persistent session using a caller-supplied [`Sampler`].
    ///
    /// The custom sampler replaces the built-in greedy/categorical token
    /// selection while the full logit-processor chain (temperature, top-k,
    /// top-p, penalties, constraints, …) still runs first. This is the Rust
    /// extension seam that the C ABI ([`crate::capi`]) plugs foreign samplers
    /// into. The device greedy fast path is bypassed so the sampler always sees
    /// the processed logits.
    pub fn generate_in_session_with_sampler(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        sampler: Box<dyn Sampler>,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_in_session_with_priority_and_callback(
            session_id,
            request,
            Priority::Normal,
            Some(sampler),
            None,
        )
    }

    fn generate_in_session_with_priority_and_callback(
        &mut self,
        session_id: SessionId,
        request: GenerateRequest,
        priority: Priority,
        custom_sampler: Option<Box<dyn Sampler>>,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.last_speculative_stats = SpeculativeStats::default();
        request.options.validate()?;
        let mut options = request.options.clone();
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        let prompt_tokens = self.tokenize_prompt(&request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        if !self.sessions.contains_key(&session_id) {
            anyhow::bail!("session {session_id} not found");
        }

        let request_id = self.scheduler.enqueue_generate_request(
            session_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            priority,
        );
        let scheduled = self
            .scheduler
            .drive_next_fcfs()
            .context("scheduler did not admit the session generate request")?;
        if scheduled.request_id != request_id || scheduled.seq_id != session_id {
            anyhow::bail!(
                "scheduler admitted request {} for session {}, expected request {} for session {}",
                scheduled.request_id,
                scheduled.seq_id,
                request_id,
                session_id
            );
        }

        let max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(&self.tokenizer))?;

        let mut state = self
            .sessions
            .remove(&session_id)
            .with_context(|| format!("session {session_id} not found"))?;
        let prefix_cache_hit_len =
            self.prepare_session_prefix(session_id, &mut state, &prompt_tokens)?;
        let mut loop_state =
            DecodeLoopState::new(prefix_cache_hit_len, options.seed, options.top_logprobs);
        let has_custom_sampler = custom_sampler.is_some();
        loop_state.custom_sampler = custom_sampler;

        let result = (|| -> anyhow::Result<GenerateResult> {
            if self.should_use_speculative(&options) && !has_custom_sampler {
                return self.generate_speculative_loop(
                    session_id,
                    &mut state,
                    &options,
                    &chain,
                    max_context,
                    prefix_cache_hit_len,
                    &mut loop_state.generated_tokens,
                    &mut loop_state.generated_text,
                    &mut loop_state.logprobs,
                    &mut loop_state.rng,
                    callback.as_deref_mut(),
                );
            }

            let mut backend = SessionDecodeLoopBackend {
                session: self
                    .session
                    .as_deref()
                    .expect("ORT backend must own a decoder session"),
                kv_model: self.kv_model.as_ref(),
                kv_cache: &mut self.kv_cache,
                scheduler: &mut self.scheduler,
                session_id,
                state: &mut state,
            };
            run_decode_loop(
                &mut backend,
                &mut loop_state,
                &options,
                &chain,
                &self.tokenizer,
                max_context,
                callback.as_deref_mut(),
            )
        })();
        if result.is_ok() && !exceeded_context_limit(state.tokens.len(), max_context) {
            self.ensure_session_kv_current(session_id, &mut state)?;
            self.insert_cached_prefixes(session_id, &state, prompt_tokens.len())?;
        }
        self.sessions.insert(session_id, state);
        self.scheduler.complete(session_id);
        result
    }

    /// Drive a set of already-arrived prioritized requests to completion.
    ///
    /// This is the Phase 3 engine-facing scheduler drive API. It runs one
    /// sequence at a time for now, but honors priority ordering and scheduler
    /// preemption decisions while preserving session decode state and KV in place.
    pub fn drive_prioritized_requests(
        &mut self,
        requests: Vec<PrioritizedGenerateRequest>,
    ) -> anyhow::Result<Vec<PrioritizedGenerateResult>> {
        let arrivals = requests
            .into_iter()
            .map(|request| ScheduledGenerateArrival {
                arrival_step: 0,
                request,
            })
            .collect();
        self.drive_prioritized_arrivals(arrivals)
    }

    /// Drive prioritized requests that arrive after specific generated-token steps.
    ///
    /// This lets async server code drain newly-arrived requests between scheduler
    /// iterations. Preemption is swap-style: active `EngineSession` state, ORT past
    /// tensors, and mirrored paged KV stay owned by the engine and are resumed
    /// without recomputation.
    pub fn drive_prioritized_arrivals(
        &mut self,
        mut arrivals: Vec<ScheduledGenerateArrival>,
    ) -> anyhow::Result<Vec<PrioritizedGenerateResult>> {
        self.require_ort_backend("prioritized request scheduling")?;
        arrivals.sort_by_key(|arrival| arrival.arrival_step);
        let total_requests = arrivals.len();
        let mut next_arrival = 0;
        let mut generated_steps = 0;
        let mut active: HashMap<SessionId, ActiveGenerate> = HashMap::new();
        let mut results = Vec::with_capacity(total_requests);

        while results.len() < total_requests {
            while next_arrival < arrivals.len()
                && arrivals[next_arrival].arrival_step <= generated_steps
            {
                let arrival = arrivals[next_arrival].clone();
                next_arrival += 1;
                let active_request = self.prepare_active_generate(arrival.request)?;
                if active
                    .insert(active_request.session_id, active_request)
                    .is_some()
                {
                    anyhow::bail!("session already has an active generation request");
                }
            }

            let decision = self.scheduler.schedule();
            let mut runnable = Vec::new();
            for seq in decision
                .prefill
                .iter()
                .chain(decision.swap_in.iter())
                .chain(decision.decode.iter())
            {
                if !decision.preempt.contains(seq) && !runnable.contains(seq) {
                    runnable.push(*seq);
                }
            }

            if runnable.is_empty() {
                if next_arrival < arrivals.len() {
                    generated_steps = arrivals[next_arrival].arrival_step;
                    continue;
                }
                anyhow::bail!("scheduler made no runnable decision with active requests remaining");
            }

            for session_id in runnable {
                let mut active_request = active.remove(&session_id).with_context(|| {
                    format!("active request for session {session_id} not found")
                })?;
                let step_result = self.step_active_generate(&mut active_request)?;
                generated_steps += 1;
                if let Some(result) = step_result {
                    let session_id = active_request.session_id;
                    self.finish_active_generate(active_request)?;
                    results.push(PrioritizedGenerateResult { session_id, result });
                } else {
                    active.insert(session_id, active_request);
                }
            }
        }

        Ok(results)
    }

    /// Create a new generation session.
    pub fn create_session(&mut self) -> anyhow::Result<SessionId> {
        self.require_ort_backend("persistent sessions")?;
        let decode_state = self.new_target_decode_state()?;
        let id = self.kv_cache.create_sequence();
        let draft = if let Some(draft_model) = &mut self.draft {
            Some(DraftSession {
                seq: draft_model.kv_cache.create_sequence(),
                tokens: Vec::new(),
                kv_token_count: 0,
                decode_state: DecodeState::new_for_path(
                    &draft_model.session,
                    &draft_model.decode_path,
                )?,
            })
        } else {
            None
        };
        let state = EngineSession {
            tokens: Vec::new(),
            kv_token_count: 0,
            decode_state,
            draft,
            sampled_fastpath_failed: false,
        };
        self.sessions.insert(id, state);
        Ok(id)
    }

    /// Reset a persistent session, freeing its current state while keeping the id usable.
    pub fn reset_session(&mut self, session_id: SessionId) -> anyhow::Result<()> {
        self.require_ort_backend("persistent sessions")?;
        if !self.sessions.contains_key(&session_id) {
            anyhow::bail!("session {session_id} not found");
        }
        self.scheduler.complete(session_id);
        self.kv_cache
            .remove(session_id)
            .map_err(|e| anyhow::anyhow!("Failed to reset KV sequence {session_id}: {}", e))?;
        self.kv_cache.page_table.create_sequence(session_id);
        let decode_state = self.new_target_decode_state()?;
        let state = self
            .sessions
            .get_mut(&session_id)
            .context("session disappeared during reset")?;
        state.tokens.clear();
        state.kv_token_count = 0;
        state.decode_state = decode_state;
        if let (Some(draft_model), Some(draft)) = (&mut self.draft, &mut state.draft) {
            draft_model
                .kv_cache
                .remove(draft.seq)
                .map_err(|e| anyhow::anyhow!("Failed to reset draft KV sequence: {}", e))?;
            draft.seq = draft_model.kv_cache.create_sequence();
            draft.tokens.clear();
            draft.kv_token_count = 0;
            draft.decode_state =
                DecodeState::new_for_path(&draft_model.session, &draft_model.decode_path)?;
        }
        Ok(())
    }

    fn new_target_decode_state(&self) -> anyhow::Result<DecodeState> {
        let session = self
            .session
            .as_deref()
            .context("ORT decoder session is unavailable")?;
        // Bind ports from an explicit `model.io` block when the package declares
        // one; otherwise DecodeState falls back to tensor-name conventions.
        let io = self
            .metadata
            .model
            .as_ref()
            .and_then(|model| model.io.as_ref());
        let fixed_state_budget_bytes = self.governor.snapshot().resolved_limits.host_ram_bytes;
        if matches!(
            &self.speculative_mode,
            SpeculativeMode::Mtp(_) | SpeculativeMode::Eagle3(_) | SpeculativeMode::SharedKv(_)
        ) {
            DecodeState::new_with_io_positions_and_state_budget(
                session,
                io,
                None,
                fixed_state_budget_bytes,
            )
        } else {
            DecodeState::new_for_path_with_io_positions_and_state_budget(
                session,
                &self.decode_path,
                io,
                None,
                fixed_state_budget_bytes,
            )
        }
    }

    /// Close a persistent session and free its associated state.
    pub fn close_session(&mut self, session_id: SessionId) -> anyhow::Result<()> {
        self.require_ort_backend("persistent sessions")?;
        self.scheduler.complete(session_id);
        let state = self
            .sessions
            .remove(&session_id)
            .with_context(|| format!("session {session_id} not found"))?;
        self.kv_cache
            .remove(session_id)
            .map_err(|e| anyhow::anyhow!("Failed to remove KV sequence {session_id}: {}", e))?;
        if let (Some(draft_model), Some(draft)) = (&mut self.draft, state.draft) {
            draft_model
                .kv_cache
                .remove(draft.seq)
                .map_err(|e| anyhow::anyhow!("Failed to remove draft KV sequence: {}", e))?;
        }
        Ok(())
    }

    /// Number of logical tokens retained in a persistent session.
    pub fn session_token_count(&self, session_id: SessionId) -> anyhow::Result<usize> {
        self.require_ort_backend("persistent sessions")?;
        self.sessions
            .get(&session_id)
            .map(|state| state.tokens.len())
            .with_context(|| format!("session {session_id} not found"))
    }

    /// Get the loaded metadata.
    pub fn metadata(&self) -> &InferenceMetadata {
        &self.metadata
    }

    /// Resolved decoder execution backend.
    pub fn decode_backend(&self) -> EngineDecodeBackend {
        self.decode_backend
    }

    /// Auto-detected fill-in-the-middle configuration, if the tokenizer declares one.
    pub fn fim_config(&self) -> Option<&FimConfig> {
        self.fim_config.as_ref()
    }

    fn fim_options(&self, fim_config: &FimConfig, mut options: GenerateOptions) -> GenerateOptions {
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        for eos_token_id in self.tokenizer.eos_token_ids() {
            push_unique_stop_sequence(
                &mut options.stop_sequences,
                StopSequence::Tokens(vec![eos_token_id]),
            );
        }
        for token in [
            fim_config.prefix_token.as_str(),
            fim_config.middle_token.as_str(),
            fim_config.suffix_token.as_str(),
            "<|fim_pad|>",
            "<|endoftext|>",
            "<|file_sep|>",
        ] {
            if let Some(token_id) = self.tokenizer.token_id(token) {
                push_unique_stop_sequence(
                    &mut options.stop_sequences,
                    StopSequence::Tokens(vec![token_id]),
                );
            }
        }
        options
    }

    fn max_context_for_request(&self, options: &GenerateOptions) -> Option<usize> {
        let configured = self
            .metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length)
            .or(options.max_context);
        match self.decode_path_max_len() {
            Some(runtime_max) => {
                Some(configured.map_or(runtime_max, |limit| limit.min(runtime_max)))
            }
            None => configured,
        }
    }

    fn decode_path_max_len(&self) -> Option<usize> {
        match self.decode_path {
            ModelDecodePath::StaticCache { max_len } => Some(max_len),
            ModelDecodePath::PastPresent {
                shared_buffer: true,
                max_len,
                ..
            } => max_len,
            ModelDecodePath::PastPresent { .. } | ModelDecodePath::Legacy => None,
        }
    }

    /// Tokenize `text` with the model's own tokenizer.
    ///
    /// This is the public tokenization seam used by higher-level pipelines to
    /// convert prompt text into token ids (e.g. to compute prompt length or
    /// `max_length`, or to feed [`Engine::embed`] and the generation APIs). It
    /// uses the same tokenizer path as the engine's internal prompt handling.
    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<TokenId>> {
        self.tokenizer.encode(text).map_err(|e| {
            anyhow::anyhow!(
                "failed to tokenize input text with the model's tokenizer: {e}; \
                 verify the model directory contains a valid tokenizer.json"
            )
        })
    }

    fn tokenize_prompt(&self, prompt: &GeneratePrompt) -> anyhow::Result<Vec<TokenId>> {
        match prompt {
            GeneratePrompt::TokenIds(tokens) => Ok(tokens.clone()),
            GeneratePrompt::Text(text) => self
                .tokenizer
                .encode(text)
                .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e)),
        }
    }

    fn prepare_session_prefix(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        prompt_tokens: &[TokenId],
    ) -> anyhow::Result<usize> {
        if self.connector.is_active() {
            self.connector.reset_stats();
        }
        let same_session_hit_len = if state.decode_state.has_runner() {
            state.decode_state.runner_len().min(state.tokens.len())
        } else if state.decode_state.use_kv {
            state.kv_token_count.min(state.tokens.len())
        } else {
            0
        };
        let started_empty = state.tokens.is_empty();
        let mut loaded_prompt_prefix = 0;
        let mut cross_session_hit_len = 0;

        if started_empty && state.decode_state.uses_token_prefix_cache() {
            cross_session_hit_len = self
                .token_prefix_cache
                .iter()
                .map(|cached| common_prefix_len(cached, prompt_tokens).min(cached.len()))
                .filter(|&len| len > 0)
                .max()
                .unwrap_or(0);
        } else if started_empty
            && state.decode_state.use_kv
            && self.kv_model.is_some()
            && self.kv_cache.page_table.tensor_config.is_some()
        {
            let matched = self
                .prefix_cache
                .lookup_shared(prompt_tokens, &mut self.kv_cache.page_table);
            if matched.matched_tokens > 0 {
                cross_session_hit_len = matched.matched_tokens;
                let materialized_len = if matched.matched_tokens == prompt_tokens.len() {
                    matched.matched_tokens.saturating_sub(1)
                } else {
                    matched.matched_tokens
                };
                let page_ids = matched
                    .page_ids
                    .iter()
                    .copied()
                    .take(materialized_len.div_ceil(self.kv_cache.page_table.page_size))
                    .collect::<Vec<_>>();
                for &page_id in &page_ids {
                    self.kv_cache.page_table.retain(page_id);
                }
                self.prefix_cache.release_shared(
                    prompt_tokens,
                    matched.matched_tokens,
                    &mut self.kv_cache.page_table,
                );
                if materialized_len > 0 {
                    attach_pages_to_sequence(
                        &mut self.kv_cache,
                        session_id,
                        &page_ids,
                        materialized_len,
                    )?;
                    let materialized = self
                        .kv_cache
                        .materialize_sequence(session_id)
                        .map_err(|e| anyhow::anyhow!("Failed to materialize prefix KV: {}", e))?;
                    load_materialized_past(
                        self.session
                            .as_deref()
                            .expect("ORT backend must own a decoder session"),
                        self.kv_model.as_ref().expect("checked above"),
                        &mut state.decode_state,
                        &materialized,
                    )?;
                    state.kv_token_count = materialized_len;
                    state
                        .tokens
                        .extend_from_slice(&prompt_tokens[..materialized_len]);
                    loaded_prompt_prefix = materialized_len;
                }
            }
        }

        if started_empty {
            state
                .tokens
                .extend_from_slice(&prompt_tokens[loaded_prompt_prefix..]);
        } else {
            state.tokens.extend_from_slice(prompt_tokens);
        }
        let in_process_hit = same_session_hit_len.max(cross_session_hit_len);

        // K4: consult the external connector for prefix reuse *beyond* the
        // in-process hit. When the active decode path can accept an owned-KV
        // handoff (a ZeroCopyRebind `PastPresent` runner with f32 KV) and the
        // session started empty, fetch the real KV bytes for the contiguous hit
        // chunks and inject them into the runner so prefill genuinely skips
        // those tokens. Because the chunk key is prefix-dependent, an equal key
        // guarantees an identical prefix, so injecting fetched KV at the same
        // absolute positions is byte-exact — proven token-identical by the gold
        // integration test. If injection is not possible we fall back to the
        // reporting-only `lookup_extension`, never claiming a hit we can't serve.
        if self.connector.is_active() {
            let injected = self.try_connector_kv_injection(state, prompt_tokens, in_process_hit)?;
            if let Some(total) = injected {
                return Ok(in_process_hit.max(total));
            }
            let _ = self
                .connector
                .lookup_extension(prompt_tokens, in_process_hit);
        }
        Ok(in_process_hit)
    }

    /// Try to materialize cross-session KV from the connector into the decode
    /// runner, genuinely shortening prefill. Returns `Some(total_len)` (the KV
    /// token count now resident in the runner) when injection happened, else
    /// `None` (caller falls back to reporting-only lookup).
    ///
    /// Only runs for a freshly started session on a ZeroCopyRebind `PastPresent`
    /// runner whose KV is f32. `import_kv` *replaces* the runner KV, so the
    /// boundary must be the current `kv_token_count` (0 for a fresh session).
    /// At least one prompt token is always left un-injected so decode has an
    /// input to feed.
    fn try_connector_kv_injection(
        &mut self,
        state: &mut EngineSession,
        prompt_tokens: &[TokenId],
        in_process_hit: usize,
    ) -> anyhow::Result<Option<usize>> {
        if !state.decode_state.has_runner()
            || !state.decode_state.runner_supports_kv_handoff()
            || state.kv_token_count != 0
            || in_process_hit != 0
        {
            return Ok(None);
        }
        // Scope the immutable `kv_model` borrow so it does not overlap the
        // `&mut self.connector` fetch below.
        match self.kv_model.as_ref() {
            Some(kv_model)
                if kv_model_past_is_f32(
                    self.session
                        .as_deref()
                        .expect("ORT backend must own a decoder session"),
                    kv_model,
                ) => {}
            _ => return Ok(None),
        }

        let boundary = 0usize;
        // Leave at least one prompt token to feed the decoder: cap the fetch to
        // `prompt_len - 1` tokens so `fetched_tokens` equals what we inject.
        let max_tokens = prompt_tokens.len().saturating_sub(1);
        let outcome =
            self.connector
                .fetch_extension(prompt_tokens, boundary, max_tokens, Device::Cpu);
        if outcome.fetched_tokens == 0 {
            return Ok(None);
        }

        let mut chunks = outcome.chunks;
        let mut total: usize = boundary + chunks.iter().map(|c| c.num_tokens).sum::<usize>();
        // Safety net: the `max_tokens` cap already guarantees `total <
        // prompt_len`, but drop trailing chunks if any invariant slipped.
        while total >= prompt_tokens.len() {
            match chunks.pop() {
                Some(dropped) => total -= dropped.num_tokens,
                None => return Ok(None),
            }
        }
        if chunks.is_empty() || total == 0 {
            return Ok(None);
        }

        let placed: Vec<PlacedPayload<'_>> = chunks
            .iter()
            .map(|chunk| PlacedPayload {
                relative_start: chunk.start - boundary,
                payload: &chunk.payload,
            })
            .collect();
        let kv_model = self.kv_model.as_ref().expect("checked present above");
        let kv = past_kv_from_payloads(
            self.session
                .as_deref()
                .expect("ORT backend must own a decoder session"),
            kv_model,
            &placed,
            total,
        )?;
        state.decode_state.import_runner_kv(total, kv)?;
        state.kv_token_count = total;
        Ok(Some(total))
    }

    fn prepare_active_generate(
        &mut self,
        request: PrioritizedGenerateRequest,
    ) -> anyhow::Result<ActiveGenerate> {
        request.request.options.validate()?;
        let mut options = request.request.options.clone();
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer.eos_token_id();
        }
        let prompt_tokens = self.tokenize_prompt(&request.request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        if !self.sessions.contains_key(&request.session_id) {
            anyhow::bail!("session {} not found", request.session_id);
        }
        if self.should_use_speculative(&options) {
            anyhow::bail!(
                "prioritized drive API currently supports the single-sequence non-speculative path; batched/speculative drive is future work"
            );
        }

        self.scheduler.enqueue_generate_request(
            request.session_id,
            prompt_tokens.len(),
            options.max_new_tokens,
            request.priority,
        );
        let max_context = self.max_context_for_request(&options);
        let chain = build_processor_chain(&options, Some(&self.tokenizer))?;
        let mut state = self
            .sessions
            .remove(&request.session_id)
            .with_context(|| format!("session {} not found", request.session_id))?;
        let prefix_cache_hit_len =
            self.prepare_session_prefix(request.session_id, &mut state, &prompt_tokens)?;
        let rng = SamplingRng::new(options.seed);
        let logprobs = options.top_logprobs.map(|_| Vec::new());
        Ok(ActiveGenerate {
            session_id: request.session_id,
            state,
            options,
            chain,
            max_context,
            prompt_len: prompt_tokens.len(),
            prefix_cache_hit_len,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            logprobs,
            step: 0,
            rng,
        })
    }

    fn step_active_generate(
        &mut self,
        active: &mut ActiveGenerate,
    ) -> anyhow::Result<Option<GenerateResult>> {
        let mut loop_state = DecodeLoopState {
            generated_tokens: std::mem::take(&mut active.generated_tokens),
            generated_text: std::mem::take(&mut active.generated_text),
            logprobs: active.logprobs.take(),
            step: active.step,
            prefix_cache_hit_len: active.prefix_cache_hit_len,
            rng: std::mem::replace(&mut active.rng, SamplingRng::new(Some(0))),
            custom_sampler: None,
        };
        let step_result = {
            let mut backend = SessionDecodeLoopBackend {
                session: self
                    .session
                    .as_deref()
                    .expect("ORT backend must own a decoder session"),
                kv_model: self.kv_model.as_ref(),
                kv_cache: &mut self.kv_cache,
                scheduler: &mut self.scheduler,
                session_id: active.session_id,
                state: &mut active.state,
            };
            step_decode_loop(
                &mut backend,
                &mut loop_state,
                &active.options,
                &active.chain,
                &self.tokenizer,
                active.max_context,
                None,
            )?
        };
        active.generated_tokens = loop_state.generated_tokens;
        active.generated_text = loop_state.generated_text;
        active.logprobs = loop_state.logprobs;
        active.step = loop_state.step;
        active.rng = loop_state.rng;
        if step_result.is_some() {
            return Ok(step_result);
        }
        if active.generated_tokens.len() >= active.options.max_new_tokens {
            ensure_constrained_finish(
                &active.options,
                &active.generated_text,
                FinishReason::MaxTokens,
            )?;
            return self
                .finish_result(
                    &active.generated_tokens,
                    FinishReason::MaxTokens,
                    active.prefix_cache_hit_len,
                    active.logprobs.as_deref(),
                )
                .map(Some);
        }

        Ok(None)
    }

    fn finish_active_generate(&mut self, mut active: ActiveGenerate) -> anyhow::Result<()> {
        if !exceeded_context_limit(active.state.tokens.len(), active.max_context) {
            self.ensure_session_kv_current(active.session_id, &mut active.state)?;
            self.insert_cached_prefixes(active.session_id, &active.state, active.prompt_len)?;
        }
        self.sessions.insert(active.session_id, active.state);
        self.scheduler.complete(active.session_id);
        Ok(())
    }

    fn ensure_session_kv_current(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
    ) -> anyhow::Result<()> {
        while state.decode_state.use_kv && state.kv_token_count < state.tokens.len() {
            let _ = next_session_token_logits(
                self.session
                    .as_deref()
                    .expect("ORT backend must own a decoder session"),
                self.kv_model.as_ref(),
                &mut self.kv_cache,
                session_id,
                state,
            )?;
        }
        Ok(())
    }

    /// Extract the runner's freshly computed KV and store each complete resident
    /// chunk in the connector. Best-effort: any gating failure or extraction
    /// error skips storing (never surfaced to inference). See
    /// [`crate::connector_bridge::ConnectorBridge::store_prefix_with`].
    fn store_connector_prefix(&mut self, state: &EngineSession) {
        if !state.decode_state.runner_supports_kv_handoff() {
            return;
        }
        let config = match self.kv_model.as_ref() {
            Some(kv_model)
                if kv_model_past_is_f32(
                    self.session
                        .as_deref()
                        .expect("ORT backend must own a decoder session"),
                    kv_model,
                ) =>
            {
                kv_model.tensor_config
            }
            _ => return,
        };
        let exported = match state.decode_state.export_runner_kv() {
            Ok(exported) => exported,
            Err(error) => {
                tracing::debug!(%error, "runner KV export failed; not storing to connector");
                return;
            }
        };
        let kv_model = self.kv_model.as_ref().expect("checked present above");
        let layers = match exported_layers_from_runner(kv_model, &exported) {
            Ok(layers) => layers,
            Err(error) => {
                tracing::debug!(%error, "collecting exported runner KV failed; not storing");
                return;
            }
        };
        self.connector.store_prefix_with(
            &state.tokens,
            state.kv_token_count,
            |chunk_start, num_tokens| {
                chunk_payload_from_exported(&layers, config, chunk_start, num_tokens)
            },
        );
    }

    fn insert_cached_prefixes(
        &mut self,
        session_id: SessionId,
        state: &EngineSession,
        prompt_len: usize,
    ) -> anyhow::Result<()> {
        // K4: extract the freshly computed KV for each complete resident chunk
        // and push the real bytes to the external connector for future
        // cross-session / cross-node reuse. Only ZeroCopyRebind `PastPresent`
        // runners with f32 KV can hand off owned tensors; other paths skip
        // (store is a no-op for the default `Null` connector regardless).
        if self.connector.is_active() {
            self.store_connector_prefix(state);
        }
        if state.decode_state.uses_token_prefix_cache() {
            if prompt_len > 0 && prompt_len <= state.kv_token_count {
                self.insert_token_prefix(&state.tokens[..prompt_len]);
            }
            if state.kv_token_count == state.tokens.len() {
                self.insert_token_prefix(&state.tokens);
            }
            return Ok(());
        }
        if self.kv_model.is_none() || state.kv_token_count == 0 {
            return Ok(());
        }
        if prompt_len > 0 && prompt_len <= state.kv_token_count {
            self.insert_cached_prefix(session_id, &state.tokens[..prompt_len])?;
        }
        if state.kv_token_count == state.tokens.len() {
            self.insert_cached_prefix(session_id, &state.tokens)?;
        }
        Ok(())
    }

    fn insert_cached_prefix(
        &mut self,
        session_id: SessionId,
        tokens: &[TokenId],
    ) -> anyhow::Result<()> {
        if tokens.is_empty() || self.prefix_cache.lookup(tokens).0 == tokens.len() {
            return Ok(());
        }
        let page_ids = sequence_pages_for_len(&self.kv_cache, session_id, tokens.len())?;
        self.prefix_cache
            .insert_pages(tokens, &page_ids, &mut self.kv_cache.page_table);
        Ok(())
    }

    fn insert_token_prefix(&mut self, tokens: &[TokenId]) {
        if tokens.is_empty()
            || self
                .token_prefix_cache
                .iter()
                .any(|cached| cached.as_slice() == tokens)
        {
            return;
        }
        self.token_prefix_cache.push(tokens.to_vec());
    }

    pub(crate) fn finish_result(
        &self,
        generated_tokens: &[TokenId],
        finish_reason: FinishReason,
        prefix_cache_hit_len: usize,
        logprobs: Option<&[crate::config::TokenLogprob]>,
    ) -> anyhow::Result<GenerateResult> {
        Ok(GenerateResult {
            text: self
                .tokenizer
                .decode(generated_tokens)
                .map_err(|e| anyhow::anyhow!("Failed to detokenize generated tokens: {}", e))?,
            token_ids: generated_tokens.to_vec(),
            finish_reason,
            prefix_cache_hit_len,
            logprobs: logprobs.map(<[crate::config::TokenLogprob]>::to_vec),
        })
    }
}

struct SessionDecodeLoopBackend<'a> {
    session: &'a Session,
    kv_model: Option<&'a KvModelInfo>,
    kv_cache: &'a mut PagedKvCache,
    scheduler: &'a mut Scheduler,
    session_id: SessionId,
    state: &'a mut EngineSession,
}

impl DecodeLoopBackend for SessionDecodeLoopBackend<'_> {
    fn context_len(&self) -> usize {
        self.state.tokens.len()
    }

    fn processor_prompt_tokens(&self) -> &[TokenId] {
        &self.state.tokens
    }

    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>> {
        next_session_token_logits(
            self.session,
            self.kv_model,
            self.kv_cache,
            self.session_id,
            self.state,
        )
    }

    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()> {
        self.state.tokens.push(token_id);
        self.scheduler.advance(self.session_id);
        Ok(())
    }

    fn greedy_fastpath_supported(&self) -> bool {
        self.state.decode_state.has_runner() && self.state.decode_state.runner_supports_argmax()
    }

    fn next_token_greedy(&mut self) -> anyhow::Result<TokenId> {
        next_session_token_argmax(
            self.session,
            self.kv_model,
            self.kv_cache,
            self.session_id,
            self.state,
        )?
        .context("greedy fast path unexpectedly returned no token")
    }

    fn sampled_fastpath_supported(&self) -> bool {
        !self.state.sampled_fastpath_failed
            && self.state.decode_state.has_runner()
            && self.state.decode_state.runner_supports_sampled()
    }

    fn next_token_sampled(
        &mut self,
        params: &onnx_genai_ort::DeviceSampleParams,
    ) -> anyhow::Result<Option<TokenId>> {
        next_session_token_sampled(
            self.session,
            self.kv_model,
            self.kv_cache,
            self.session_id,
            self.state,
            params,
        )
    }

    fn sampled_fastpath_failed(&mut self) {
        self.state.sampled_fastpath_failed = true;
    }
}

fn resolve_decode_backend(
    model_path: &Path,
    requested: EngineDecodeBackend,
) -> anyhow::Result<EngineDecodeBackend> {
    let requested = requested_decode_backend(requested)?;
    match requested {
        EngineDecodeBackend::Ort => Ok(EngineDecodeBackend::Ort),
        EngineDecodeBackend::Native => {
            #[cfg(feature = "native-backend")]
            {
                Ok(EngineDecodeBackend::Native)
            }
            #[cfg(not(feature = "native-backend"))]
            {
                let _ = model_path;
                anyhow::bail!(
                    "native decoder backend requires building onnx-genai-engine with the \
                     'native-backend' feature; set decode_backend = EngineDecodeBackend::Ort \
                     (or ONNX_GENAI_BACKEND=ort) to run this model on ONNX Runtime"
                )
            }
        }
        EngineDecodeBackend::Auto => {
            if model_requires_native_backend(model_path)? {
                #[cfg(feature = "native-backend")]
                {
                    return Ok(EngineDecodeBackend::Native);
                }
                #[cfg(not(feature = "native-backend"))]
                {
                    anyhow::bail!(
                        "model contains native-only operators (pkg.nxrt::BlockQuantizedMatMul); \
                         rebuild with the 'native-backend' feature and select \
                         decode_backend = EngineDecodeBackend::Native \
                         (or ONNX_GENAI_BACKEND=native)"
                    );
                }
            }
            Ok(EngineDecodeBackend::Ort)
        }
    }
}

fn parse_backend_env(value: &str) -> anyhow::Result<EngineDecodeBackend> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(EngineDecodeBackend::Auto),
        "ort" => Ok(EngineDecodeBackend::Ort),
        "native" => Ok(EngineDecodeBackend::Native),
        _ => anyhow::bail!(
            "invalid ONNX_GENAI_BACKEND={value:?}; expected one of: auto, ort, native"
        ),
    }
}

fn requested_decode_backend_with_env(
    requested: EngineDecodeBackend,
    env_lookup: impl FnOnce() -> anyhow::Result<Option<String>>,
) -> anyhow::Result<EngineDecodeBackend> {
    if requested != EngineDecodeBackend::Auto {
        return Ok(requested);
    }
    env_lookup()?.map_or(Ok(EngineDecodeBackend::Auto), |value| {
        parse_backend_env(&value)
    })
}

pub(crate) fn requested_decode_backend(
    requested: EngineDecodeBackend,
) -> anyhow::Result<EngineDecodeBackend> {
    requested_decode_backend_with_env(requested, || match std::env::var("ONNX_GENAI_BACKEND") {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(anyhow::anyhow!(
            "failed to read ONNX_GENAI_BACKEND: {error}"
        )),
    })
}

fn ort_to_native_hint() -> &'static str {
    "ONNX Runtime could not load this model; if it requires native execution, \
     set decode_backend = EngineDecodeBackend::Native (or ONNX_GENAI_BACKEND=native)"
}

fn native_to_ort_hint() -> &'static str {
    "if this model uses operators unsupported by the native backend, \
     set decode_backend = EngineDecodeBackend::Ort (or ONNX_GENAI_BACKEND=ort) \
     to run this model on ONNX Runtime"
}

fn augment_backend_error<T>(
    result: anyhow::Result<T>,
    backend: EngineDecodeBackend,
) -> anyhow::Result<T> {
    let hint = match backend {
        EngineDecodeBackend::Ort => ort_to_native_hint(),
        EngineDecodeBackend::Native => native_to_ort_hint(),
        EngineDecodeBackend::Auto => unreachable!("the selected backend cannot be Auto"),
    };
    result.with_context(|| hint)
}

pub(crate) fn model_requires_native_backend(model_path: &Path) -> anyhow::Result<bool> {
    #[cfg(feature = "native-backend")]
    {
        use prost::Message;

        let bytes = onnx_runtime_loader::read_model_binary(model_path).with_context(|| {
            format!(
                "Failed to inspect model '{}' for native operators",
                model_path.display()
            )
        })?;
        let model = onnx_runtime_loader::proto::ModelProto::decode(bytes.as_slice())
            .context("Failed to parse ONNX model while selecting decoder backend")?;
        Ok(model_proto_requires_native_backend(&model))
    }
    #[cfg(not(feature = "native-backend"))]
    {
        let _ = model_path;
        Ok(false)
    }
}

#[cfg(feature = "native-backend")]
fn model_proto_requires_native_backend(model: &onnx_runtime_loader::proto::ModelProto) -> bool {
    const DOMAIN: &str = "pkg.nxrt";
    const OP_TYPE: &str = "BlockQuantizedMatMul";
    const OPSET_VERSION: i64 = 1;

    let supports_native_opset = model
        .opset_import
        .iter()
        .any(|opset| opset.domain == DOMAIN && opset.version == OPSET_VERSION);
    supports_native_opset
        && model.graph.as_ref().is_some_and(|graph| {
            graph
                .node
                .iter()
                .any(|node| node.domain == DOMAIN && node.op_type == OP_TYPE)
        })
}

#[cfg(feature = "native-backend")]
fn reject_native_request_speculation(options: &GenerateOptions) -> anyhow::Result<()> {
    // Prompt-lookup is now implemented on the native path (WP2); only the
    // not-yet-ported proposer families are rejected.
    let unsupported = match options.speculative_mode.as_ref() {
        None | Some(SpeculativeMode::None) | Some(SpeculativeMode::PromptLookup { .. }) => None,
        Some(SpeculativeMode::DraftModel) => Some("draft-model"),
        Some(SpeculativeMode::Mtp(_)) => Some("MTP"),
        Some(SpeculativeMode::Eagle3(_)) => Some("EAGLE-3"),
        Some(SpeculativeMode::SharedKv(_)) => None,
    };
    if let Some(mode) = unsupported {
        anyhow::bail!(
            "native decoder backend does not yet support per-request {mode} speculative decoding (only prompt-lookup is implemented)"
        );
    }
    // `num_speculative_tokens` only has meaning alongside an implemented native
    // speculative mode; reject it when no such mode selects native speculation.
    if options.num_speculative_tokens.is_some()
        && !matches!(
            options.speculative_mode.as_ref(),
            Some(SpeculativeMode::PromptLookup { .. } | SpeculativeMode::SharedKv(_))
        )
    {
        anyhow::bail!(
            "native decoder backend does not support the per-request num_speculative_tokens option without a prompt-lookup speculative_mode"
        );
    }
    Ok(())
}

/// Prompt-lookup speculation parameters resolved for a native request.
#[cfg(feature = "native-backend")]
struct NativeSpeculationPlan {
    kind: NativeSpeculationKind,
    width: usize,
}

#[cfg(feature = "native-backend")]
#[derive(Clone, Copy)]
enum NativeSpeculationKind {
    PromptLookup { ngram: usize, max_tokens: usize },
    SharedKv,
}

/// Decide whether a native request should run through the speculative driver.
///
/// Returns `Some` only for an implemented, greedy prompt-lookup request with no
/// processor chain and no logprobs — the exact regime in which host-argmax
/// acceptance reproduces plain greedy selection. Every other request (including
/// non-greedy, processor-chain, logprobs, or the default `None` mode) returns
/// `None` and stays on the untouched plain fast path.
#[cfg(feature = "native-backend")]
fn native_speculation_plan(
    options: &GenerateOptions,
    chain: &crate::logits::ProcessorChain,
) -> Option<NativeSpeculationPlan> {
    let (kind, default_width) = match options.speculative_mode.as_ref()? {
        SpeculativeMode::PromptLookup { ngram, max_tokens } => (
            NativeSpeculationKind::PromptLookup {
                ngram: *ngram,
                max_tokens: *max_tokens,
            },
            *max_tokens,
        ),
        SpeculativeMode::SharedKv(config) => (
            NativeSpeculationKind::SharedKv,
            config.num_speculative_tokens.saturating_add(1),
        ),
        _ => return None,
    };
    let greedy = options.greedy || options.temperature == 0.0;
    if !greedy || !chain.is_empty() || options.top_logprobs.is_some() {
        return None;
    }
    let width = options
        .num_speculative_tokens
        .map(|value| {
            value.saturating_add(usize::from(matches!(kind, NativeSpeculationKind::SharedKv)))
        })
        .unwrap_or(default_width)
        .max(1);
    Some(NativeSpeculationPlan { kind, width })
}

fn default_inference_metadata() -> InferenceMetadata {
    InferenceMetadata::default()
}

/// Optional cap (in tokens) on the runtime-owned fixed-capacity KV buffer,
/// read from `ONNX_GENAI_KV_MAX_LEN`. Returns `None` when the variable is
/// unset, empty, or unparseable (in which case the model's full advertised
/// context length is used, preserving prior behavior).
fn shared_buffer_cap_from_env() -> Option<usize> {
    std::env::var("ONNX_GENAI_KV_MAX_LEN")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&cap| cap > 0)
}

/// Apply an optional KV-buffer capacity cap: the effective length is the
/// smaller of the model's advertised max length and the cap, if any.
fn cap_kv_len(model_max_len: usize, cap: Option<usize>) -> usize {
    cap.map_or(model_max_len, |cap| model_max_len.min(cap))
}

/// Like [`genai_config_compat_metadata`], but derives the decoder graph
/// inventory by inspecting the ONNX model file directly (used by the native
/// decoder constructor, which builds metadata before a `Session` exists).
///
/// The native decode loop binds exactly the ports the metadata names, so — like
/// the ORT path — a hybrid SSM/attention decoder must have its KV/state topology
/// read from the graph, not expanded from a uniform layer count. If the graph
/// cannot be inspected, this falls back to pattern-expanded metadata so no
/// currently loading model regresses.
#[cfg(feature = "native-backend")]
fn genai_config_compat_metadata_from_model_path(
    model_dir: &Path,
    model_path: &Path,
) -> anyhow::Result<Option<InferenceMetadata>> {
    let decoder_graph = decoder_graph_info_from_model_path(model_path);
    let result = match &decoder_graph {
        Some(graph) => {
            let kv_native_dtype = graph
                .inputs
                .iter()
                .find(|info| crate::decode::is_kv_input(&info.name))
                .map(|info| info.dtype.as_str());
            onnx_genai_genai_config::inference_metadata_from_dir_with_graph(
                model_dir,
                kv_native_dtype,
                graph,
            )
        }
        None => onnx_genai_genai_config::inference_metadata_from_dir(model_dir, None),
    };
    result.map_err(|e| anyhow::anyhow!("Failed to convert genai_config.json: {}", e))
}

/// Best-effort decoder graph inventory read straight from an ONNX model file,
/// mirroring the ORT loader's graph inspection. Returns `None` on any failure so
/// callers fall back to pattern-expanded metadata. Only the graph interface
/// (port names, dtypes, shapes) is needed — external weight data is never read.
#[cfg(feature = "native-backend")]
fn decoder_graph_info_from_model_path(
    model_path: &Path,
) -> Option<onnx_genai_genai_config::ModelGraphInfo> {
    use onnx_runtime_ir::Dim;
    let graph = onnx_runtime_loader::load_model(model_path).ok()?;
    let tensor_info =
        |id: &onnx_runtime_ir::ValueId| -> Option<onnx_genai_genai_config::GraphTensorInfo> {
            let value = graph.value(*id);
            let name = value.name.clone()?;
            Some(onnx_genai_genai_config::GraphTensorInfo {
                name,
                dtype: ir_dtype_name(value.dtype).to_owned(),
                dimensions: value
                    .shape
                    .iter()
                    .map(|dim| match dim {
                        Dim::Static(value) => Some(*value),
                        Dim::Symbolic(_) => None,
                    })
                    .collect(),
            })
        };
    let inputs = graph
        .inputs
        .iter()
        .map(tensor_info)
        .collect::<Option<Vec<_>>>()?;
    let outputs = graph
        .outputs
        .iter()
        .map(tensor_info)
        .collect::<Option<Vec<_>>>()?;
    Some(onnx_genai_genai_config::ModelGraphInfo { inputs, outputs })
}

/// Canonical lowercase dtype spelling for an `onnx_runtime_ir` graph dtype.
#[cfg(feature = "native-backend")]
fn ir_dtype_name(dtype: onnx_runtime_ir::DataType) -> &'static str {
    use onnx_runtime_ir::DataType;
    match dtype {
        DataType::Float32 => "float32",
        DataType::Float16 => "float16",
        DataType::BFloat16 => "bfloat16",
        DataType::Float64 => "float64",
        DataType::Uint8 => "uint8",
        DataType::Int8 => "int8",
        DataType::Uint16 => "uint16",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::Bool => "bool",
        DataType::String => "string",
        DataType::Complex64 => "complex64",
        DataType::Complex128 => "complex128",
        DataType::Float8E4M3FN => "float8_e4m3fn",
        DataType::Float8E4M3FNUZ => "float8_e4m3fnuz",
        DataType::Float8E5M2 => "float8_e5m2",
        DataType::Float8E5M2FNUZ => "float8_e5m2fnuz",
        DataType::Uint4 => "uint4",
        DataType::Int4 => "int4",
        _ => "undefined",
    }
}

/// Best-effort native metadata derived from an onnxruntime-genai
/// `genai_config.json` in `model_dir`, used only when no
/// `inference_metadata.yaml` is present. Returns `Ok(None)` when there is no
/// `genai_config.json`. The KV cache native dtype is read from the loaded
/// session's KV inputs, since it is not present in `genai_config.json`.
fn genai_config_compat_metadata(
    model_dir: &Path,
    session: &Session,
) -> anyhow::Result<Option<InferenceMetadata>> {
    let kv_native_dtype = session
        .inputs()
        .iter()
        .find(|info| crate::decode::is_kv_input(&info.name))
        .and_then(|info| match info.dtype {
            DataType::Float16 => Some("float16"),
            DataType::BFloat16 => Some("bfloat16"),
            DataType::Float32 => Some("float32"),
            _ => None,
        });
    // Hand the decoder's actual ONNX graph inventory to the compatibility
    // converter so it declares exactly the KV/state ports the graph exposes.
    // onnxruntime-genai `genai_config.json` only carries a uniform per-layer KV
    // name pattern and a total layer count; for hybrid SSM/attention decoders
    // (qwen3.5: most layers are linear-attention with `conv_state`/
    // `recurrent_state`, only the periodic full-attention layers expose dense
    // `key`/`value`) that pattern names ports the graph never exposes and warmup
    // aborts. Deriving the topology from the graph yields sparse `kv_inputs`/
    // `kv_outputs` plus recurrent `state_pairs`; uniform dense-KV decoders are
    // unchanged.
    let decoder_graph = session_model_graph_info(session);
    onnx_genai_genai_config::inference_metadata_from_dir_with_graph(
        model_dir,
        kv_native_dtype,
        &decoder_graph,
    )
    .map_err(|e| anyhow::anyhow!("Failed to convert genai_config.json: {}", e))
}

/// Build a [`ModelGraphInfo`] inventory from a loaded session's input/output
/// port metadata, mirroring the ONNX graph interface the strict compatibility
/// converter consumes (names, dtype spelling, and per-axis static/symbolic
/// dimensions). ORT reports dynamic axes as negative dimensions, which map to
/// symbolic (`None`) entries.
fn session_model_graph_info(session: &Session) -> onnx_genai_genai_config::ModelGraphInfo {
    fn tensor_info(meta: &onnx_genai_ort::TensorInfo) -> onnx_genai_genai_config::GraphTensorInfo {
        onnx_genai_genai_config::GraphTensorInfo {
            name: meta.name.clone(),
            dtype: graph_dtype_name(meta.dtype).to_owned(),
            dimensions: meta
                .shape
                .iter()
                .map(|&dim| usize::try_from(dim).ok())
                .collect(),
        }
    }
    onnx_genai_genai_config::ModelGraphInfo {
        inputs: session.inputs().iter().map(tensor_info).collect(),
        outputs: session.outputs().iter().map(tensor_info).collect(),
    }
}

/// Canonical lowercase dtype spelling used by the compatibility metadata
/// converter's graph inventory (`float32`, `float16`, `bfloat16`, ...).
fn graph_dtype_name(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Float32 => "float32",
        DataType::Float16 => "float16",
        DataType::BFloat16 => "bfloat16",
        DataType::Float8E4M3 => "float8_e4m3fn",
        DataType::Float8E5M2 => "float8_e5m2",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Uint8 => "uint8",
        DataType::Uint16 => "uint16",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::Bool => "bool",
    }
}

/// Preserve explicit ORT graph settings and warn about known-unsafe opt-ins.
fn configure_ort_cuda_graph(options: &mut SessionOptions, model_path: &Path) {
    if options.selects_cuda() && options.graph_capture && model_has_control_flow_nodes(model_path) {
        tracing::warn!(
            "ORT CUDA graph capture was explicitly enabled for model '{}', but it contains control-flow nodes (If/Loop/Scan); capture may fail or run substantially slower",
            model_path.display()
        );
    }
}

/// Whether the ONNX model at `model_path` contains top-level control-flow nodes
/// (`If`/`Loop`/`Scan`). ORT cannot capture a CUDA graph for such models, and
/// requesting capture anyway (via the `enable_cuda_graph` provider option)
/// forces a pathological ~6× slower per-Run path, so the caller must leave graph
/// capture disabled when this returns `true`.
///
/// Returns `true` whenever the model cannot be inspected. CUDA graph capture is
/// an optional optimization, so uncertain models conservatively skip it rather
/// than risking ORT's pathological uncaptured per-Run path.
fn model_has_control_flow_nodes(model_path: &Path) -> bool {
    scan_top_level_control_flow(model_path).unwrap_or(true)
}

/// Control-flow op names that block CUDA graph capture, in the default ONNX
/// domain (`""`/`ai.onnx`).
const CONTROL_FLOW_OPS: [&str; 3] = ["If", "Loop", "Scan"];

/// A deliberately minimal view of an ONNX `ModelProto` carrying only the fields
/// needed to reach each top-level node's `op_type`/`domain`.
///
/// Every other field is *absent* from these structs — crucially
/// `GraphProto.initializer` and its `TensorProto.raw_data`, which hold the
/// multi-gigabyte inline weights of models like the qwen3 exports (whose
/// `model.onnx` is over 1 GB). prost's decoder skips any field not declared here
/// with `Buf::advance` (pointer arithmetic), so those weight bytes are never
/// copied — and, when decoding from a memory map, never even faulted in. This
/// keeps the scan cheap regardless of weight size while reusing prost's
/// well-tested wire parser instead of a bespoke byte walker.
#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanModelProto {
    /// `ModelProto.graph`. Repeated occurrences merge per protobuf semantics, so
    /// nodes from every graph field accumulate here.
    #[prost(message, optional, tag = "7")]
    graph: Option<ScanGraphProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanGraphProto {
    /// `GraphProto.node`. `initializer` (tag 5) and every other field is skipped.
    #[prost(message, repeated, tag = "1")]
    node: Vec<ScanNodeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ScanNodeProto {
    /// `NodeProto.op_type`.
    #[prost(string, tag = "4")]
    op_type: String,
    /// `NodeProto.domain`. `attribute` (tag 5), which carries subgraph bodies, is
    /// skipped, so only top-level nodes are inspected (matching ORT's capture
    /// eligibility, which only cares about the top-level graph).
    #[prost(string, tag = "7")]
    domain: String,
}

/// Scan for top-level control-flow ops in the ONNX model at `model_path`.
///
/// Returns `Some(true)`/`Some(false)` when the model's graph could be parsed, or
/// `None` when the file cannot be opened, memory-mapped, or decoded, or when it
/// carries no graph at all. The caller treats every `None` conservatively as
/// "has control flow".
///
/// Binary models are memory-mapped and decoded into [`ScanModelProto`], whose
/// minimal field set makes prost skip the inline weight tensors without reading
/// them. Textproto fixtures are first converted through the loader's canonical
/// path. An earlier revision gave up on any file over 512 MB and conservatively
/// reported "has control flow", which wrongly disabled CUDA graph capture for
/// large inline-weight models (a ~20% decode-throughput loss).
fn scan_top_level_control_flow(model_path: &Path) -> Option<bool> {
    use prost::Message;

    let model = if onnx_runtime_loader::is_textproto_path(model_path) {
        let bytes = onnx_runtime_loader::read_model_binary(model_path).ok()?;
        ScanModelProto::decode(bytes.as_slice()).ok()?
    } else {
        let file = std::fs::File::open(model_path).ok()?;
        // SAFETY: the model file is treated as immutable for the brief lifetime of
        // this scan. Model files are not rewritten in place while their directory is
        // in use, so no concurrent truncation (which could raise SIGBUS) is expected.
        let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
        ScanModelProto::decode(&mmap[..]).ok()?
    };
    let graph = model.graph?;
    Some(graph.node.iter().any(|node| {
        matches!(node.domain.as_str(), "" | "ai.onnx")
            && CONTROL_FLOW_OPS.contains(&node.op_type.as_str())
    }))
}

fn read_f32_weights(path: &Path) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read f32 weights from '{}'", path.display()))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        anyhow::bail!(
            "f32 weight file '{}' has byte length {}, which is not divisible by 4",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

#[cfg(feature = "native-backend")]
fn load_native_shared_kv_proposer(
    metadata: &InferenceMetadata,
    model_dir: &Path,
    device: crate::native_decode::NativeDecodeDevice,
) -> anyhow::Result<(Option<NativeSharedKvProposerModel>, SpeculativeMode)> {
    let Some(config) = metadata.speculative.as_ref() else {
        return Ok((None, SpeculativeMode::None));
    };
    if config.proposal_type != ProposalType::SharedKv {
        return Ok((None, SpeculativeMode::None));
    }
    if config.io.is_none() {
        tracing::warn!(
            "shared-KV proposer metadata has no explicit speculative.io execution contract; native target decode remains available, but the proposer stays disabled until sequence_source, kv_ownership, and output roles are declared"
        );
        return Ok((None, SpeculativeMode::None));
    }
    let descriptor = onnx_genai_metadata::resolve_speculator_config(model_dir, config.clone());
    let spec = match descriptor.proposer {
        SpeculatorProposerStatus::SharedKv(spec) => spec,
        SpeculatorProposerStatus::Unknown(reason) => {
            anyhow::bail!("invalid native shared-KV proposer metadata: {reason}")
        }
        other => {
            anyhow::bail!("shared-KV metadata resolved to unexpected proposer status {other:?}")
        }
    };
    let target_hidden_output = metadata
            .model
            .as_ref()
            .and_then(|model| model.io.as_ref())
            .and_then(|io| io.hidden_output.clone())
            .context(
                "native shared-KV speculation requires model.io.hidden_output to name the target decoder hidden-state output; add the exact graph output name to inference metadata",
            )?;
    for group in &spec.shared_kv {
        for (field, value) in [
            ("key_input", group.key_input.as_deref()),
            ("value_input", group.value_input.as_deref()),
            ("target_key_input", group.target_key_input.as_deref()),
            ("target_value_input", group.target_value_input.as_deref()),
        ] {
            if value.is_none_or(str::is_empty) {
                anyhow::bail!(
                    "native shared-KV group '{}' is missing `{field}`; declare exact proposer and target KV port names so the runtime never infers cache roles from model or tensor names",
                    group.name
                );
            }
        }
    }
    let weights = read_f32_weights(&spec.input_embedding)?;
    let embedder = LinearEmbedder::new(weights, spec.vocab_size, spec.backbone_hidden_size)
        .context("build native shared-KV target embedding lookup")?;
    let session =
        crate::native_decode::NativeProposerSession::load(&spec.model, device, Some(&spec.io))
            .with_context(|| {
                format!(
                    "load native shared-KV proposer graph '{}'",
                    spec.model.display()
                )
            })?;
    let mode = SpeculativeMode::SharedKv(SharedKvProposerConfig {
        assistant_model: spec.model,
        target_hidden_output,
        input_embedding_weights: spec.input_embedding,
        backbone_hidden_size: spec.backbone_hidden_size,
        vocab_size: spec.vocab_size,
        num_speculative_tokens: spec.num_speculative_tokens,
        shared_kv: spec
            .shared_kv
            .iter()
            .map(|group| SharedKvBinding {
                name: group.name.clone(),
                target_layers: group.target_layers.clone(),
            })
            .collect(),
    });
    Ok((
        Some(NativeSharedKvProposerModel {
            session,
            embedder,
            groups: spec.shared_kv,
            hidden_size: spec.backbone_hidden_size,
        }),
        mode,
    ))
}

/// Resolve a native MTP runtime configuration from the already-loaded metadata.
///
/// The target vocabulary is read from the target `logits` signature; exact
/// embedding and LM-head initializer names remain package references until the
/// MTP model is initialized.
fn mtp_config_from_metadata(
    metadata: &InferenceMetadata,
    model_dir: &Path,
    session: &Session,
) -> anyhow::Result<Option<ResolvedMtpConfig>> {
    let Some(config) = metadata.speculative.as_ref() else {
        return Ok(None);
    };
    if config.proposal_type != ProposalType::Mtp {
        return Ok(None);
    }
    let descriptor = onnx_genai_metadata::resolve_speculator_config(model_dir, config.clone());
    let spec = match descriptor.proposer {
        SpeculatorProposerStatus::Mtp(spec) => spec,
        SpeculatorProposerStatus::Unknown(reason) => {
            anyhow::bail!("Invalid MTP sidecar metadata: {reason}")
        }
        other => anyhow::bail!("MTP metadata resolved to unexpected proposer status {other:?}"),
    };
    let vocab_size = session
        .outputs()
        .iter()
        .find(|output| output.name == "logits")
        .and_then(|output| output.shape.last().copied())
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .context("MTP metadata requires a target logits output with static vocabulary size")?;
    let config = ResolvedMtpConfig::from_sidecar_descriptor(&spec, vocab_size);
    validate_resolved_mtp_config(&config)?;
    Ok(Some(config))
}

/// Build a [`SpeculativeMode::SharedKv`] from a model directory's native
/// inference metadata, or `None` when no supported assistant is advertised.
///
/// The target hidden output name is not part of the shared metadata contract,
/// so it is auto-detected: the first Float32 output whose last dimension equals
/// the advertised backbone hidden size (excluding `logits`).
fn shared_kv_mode_from_metadata(model_dir: &Path, session: &Session) -> Option<SpeculativeMode> {
    let descriptor = onnx_genai_metadata::detect_speculator(model_dir)?;
    let onnx_genai_metadata::SpeculatorProposerStatus::SharedKv(spec) = descriptor.proposer else {
        return None;
    };
    let target_hidden_output = detect_target_hidden_output(session, spec.backbone_hidden_size)?;
    let shared_kv = spec
        .shared_kv
        .into_iter()
        .map(|group| SharedKvBinding {
            name: group.name,
            target_layers: group.target_layers,
        })
        .collect();
    Some(SpeculativeMode::SharedKv(SharedKvProposerConfig {
        assistant_model: spec.model,
        target_hidden_output,
        input_embedding_weights: spec.input_embedding,
        backbone_hidden_size: spec.backbone_hidden_size,
        vocab_size: spec.vocab_size,
        num_speculative_tokens: spec.num_speculative_tokens,
        shared_kv,
    }))
}

/// Find a Float32 hidden-state output ending in `hidden_size` (not `logits`).
fn detect_target_hidden_output(session: &Session, hidden_size: usize) -> Option<String> {
    session
        .outputs()
        .iter()
        .find(|output| {
            output.dtype == DataType::Float32
                && !output.name.to_ascii_lowercase().contains("logits")
                && output.shape.last().copied().filter(|dim| *dim > 0) == Some(hidden_size as i64)
        })
        .map(|output| output.name.clone())
}

/// Stable, opaque model identity derived from the model directory name.
///
/// Used only to namespace connector cache keys when the caller does not supply
/// an explicit `model_id`. It is never interpreted or branched on.
fn default_connector_model_id(model_directory: &ModelDirectory) -> String {
    model_directory
        .root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "onnx-genai-model".to_string())
}

/// Build the engine's KV connector bridge from generic, model-agnostic config.
fn build_connector_bridge(
    config: &KvConnectorConfig,
    model_directory: &ModelDirectory,
    kv_model: Option<&KvModelInfo>,
) -> anyhow::Result<ConnectorBridge> {
    match &config.backend {
        KvConnectorBackend::Null => Ok(ConnectorBridge::null()),
        KvConnectorBackend::LocalTiered(local_config) => {
            let connector = LocalTieredConnector::new(local_config.clone()).map_err(|error| {
                anyhow::anyhow!("failed to build LocalTiered KV connector: {error}")
            })?;
            let model_id = config
                .model_id
                .clone()
                .unwrap_or_else(|| default_connector_model_id(model_directory));
            let chunk_size = if config.chunk_size == 0 {
                onnx_genai_kv::DEFAULT_CHUNK_SIZE
            } else {
                config.chunk_size
            };
            let num_layers = kv_model
                .map(|model| model.tensor_config.num_layers)
                .unwrap_or(1)
                .max(1);
            ConnectorBridge::new(
                Arc::new(connector),
                model_id,
                chunk_size,
                0..num_layers,
                config.store_priority,
                config.recompute_ms_per_token,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_loop::logprob_for_token;
    use crate::logits::ProcessorContext;
    use crate::processors::{
        finish_reason_after_token, select_next_token, select_next_token_with_sampler,
    };
    use crate::sampling::Sampler;

    #[test]
    fn cap_kv_len_uncapped_returns_model_max() {
        assert_eq!(cap_kv_len(32_768, None), 32_768);
    }

    #[test]
    fn cap_kv_len_caps_when_smaller() {
        assert_eq!(cap_kv_len(40_960, Some(512)), 512);
    }

    #[test]
    fn cap_kv_len_ignores_cap_larger_than_model_max() {
        assert_eq!(cap_kv_len(512, Some(40_960)), 512);
    }

    fn test_model_path(label: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NEXT_PATH_ID: AtomicUsize = AtomicUsize::new(0);
        std::env::current_dir()
            .expect("current directory")
            .join(format!(
                ".onnx-genai-{label}-{}-{}.onnx",
                std::process::id(),
                NEXT_PATH_ID.fetch_add(1, Ordering::Relaxed)
            ))
    }

    fn write_scan_model(nodes: &[(&str, &str)]) -> std::path::PathBuf {
        write_scan_model_with_weights(nodes, 0)
    }

    /// Build a valid ONNX model whose graph has the given `(domain, op_type)`
    /// nodes, plus `weight_floats` f32 elements of inline initializer data to
    /// simulate an inline-weight export (the qwen3 case). The prost scan must
    /// skip past this initializer (via `Buf::advance`) rather than reading it.
    fn write_scan_model_with_weights(
        nodes: &[(&str, &str)],
        weight_floats: usize,
    ) -> std::path::PathBuf {
        use onnx::ir::{DataType, Dim, Graph, Node, NodeId, TensorData, WeightRef, static_shape};
        use onnx_std as onnx;
        use prost::Message;

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        for (index, &(domain, op_type)) in nodes.iter().enumerate() {
            let output = graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                static_shape([]),
            );
            let mut node = Node::new(NodeId(index as u32), op_type, vec![], vec![output]);
            node.domain = domain.to_string();
            graph.insert_node(node);
            graph.add_output(output);
        }

        if weight_floats > 0 {
            let weight = graph.create_named_value(
                "inline_weight",
                DataType::Float32,
                vec![Dim::from(weight_floats)],
            );
            let bytes = vec![0u8; weight_floats * std::mem::size_of::<f32>()];
            graph.set_initializer(
                weight,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float32,
                    vec![weight_floats],
                    bytes,
                )),
            );
        }

        let path = test_model_path("control-flow");
        std::fs::write(
            &path,
            onnx::Model::new(graph)
                .to_proto()
                .expect("serialize test model")
                .encode_to_vec(),
        )
        .expect("write test model");
        path
    }

    #[test]
    fn control_flow_scan_ignores_regular_ops() {
        let plain = write_scan_model(&[("", "MatMul"), ("", "Add"), ("", "GroupQueryAttention")]);
        assert!(!model_has_control_flow_nodes(&plain));
        std::fs::remove_file(&plain).ok();
    }

    #[test]
    fn ort_cuda_graph_configuration_is_opt_in() {
        let model = write_scan_model(&[("", "MatMul")]);
        let mut options =
            SessionOptions::with_execution_provider(onnx_genai_ort::ep_selection("cuda"));

        options.graph_capture = false;
        configure_ort_cuda_graph(&mut options, &model);
        assert!(
            !options.graph_capture,
            "ORT capture must stay off by default"
        );

        options.graph_capture = true;
        configure_ort_cuda_graph(&mut options, &model);
        assert!(
            options.graph_capture,
            "an explicit ORT capture opt-in must be preserved"
        );
        std::fs::remove_file(&model).ok();
    }

    #[test]
    fn control_flow_scan_detects_standard_onnx_control_flow_ops() {
        for domain in ["", "ai.onnx"] {
            for op_type in ["If", "Loop", "Scan"] {
                let path = write_scan_model(&[(domain, op_type)]);
                assert!(
                    model_has_control_flow_nodes(&path),
                    "expected standard-domain control-flow op '{domain}:{op_type}' to be detected"
                );
                std::fs::remove_file(&path).ok();
            }
        }
    }

    #[test]
    fn control_flow_scan_ignores_custom_domain_control_flow_names() {
        for op_type in ["If", "Loop", "Scan"] {
            let path = write_scan_model(&[("com.example", op_type)]);
            assert!(
                !model_has_control_flow_nodes(&path),
                "custom-domain op 'com.example:{op_type}' must not disable capture"
            );
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn control_flow_scan_conservatively_skips_uninspectable_models() {
        let missing = std::path::Path::new("does-not-exist-onnx-genai.onnx");
        assert!(model_has_control_flow_nodes(missing));

        let garbage = test_model_path("garbage");
        std::fs::write(&garbage, b"not a protobuf").expect("write garbage model");
        assert!(model_has_control_flow_nodes(&garbage));
        std::fs::remove_file(&garbage).ok();
    }

    #[test]
    fn control_flow_scan_reads_nodes_past_large_inline_weights() {
        // Simulate an inline-weight export (like the qwen3 models, whose
        // `model.onnx` embeds >1 GB of weights): the graph carries a large
        // initializer alongside its nodes. The prost scan must still find
        // the control-flow op — and, for a plain graph, must NOT be fooled into
        // conservatively reporting control flow just because the file is large.
        // 4 Mi f32 elements = 16 MiB of inline initializer data.
        let weight_floats = 4 * 1024 * 1024;

        let with_control_flow =
            write_scan_model_with_weights(&[("", "MatMul"), ("", "If")], weight_floats);
        assert!(
            model_has_control_flow_nodes(&with_control_flow),
            "control-flow op must be detected even behind a large inline initializer"
        );
        std::fs::remove_file(&with_control_flow).ok();

        let plain = write_scan_model_with_weights(
            &[("", "MatMul"), ("", "GroupQueryAttention")],
            weight_floats,
        );
        assert!(
            !model_has_control_flow_nodes(&plain),
            "a large inline-weight model without control flow must remain capture-eligible"
        );
        std::fs::remove_file(&plain).ok();
    }

    #[test]
    fn control_flow_scan_conservatively_handles_truncated_control_flow_model() {
        // A control-flow model whose bytes are truncated anywhere must never
        // parse cleanly into a "no control flow" verdict (which would wrongly
        // enable CUDA graph capture and trigger ORT's ~6x slower per-Run path).
        // Truncation either cuts the graph payload (its length header points past
        // EOF -> None) or stops before the graph is ever seen (no-graph -> None),
        // so every prefix (including the empty file) must fall back to
        // conservative `true`.
        let full = write_scan_model(&[("", "MatMul"), ("", "If")]);
        let bytes = std::fs::read(&full).expect("read full model");
        std::fs::remove_file(&full).ok();

        for truncated_len in 0..bytes.len() {
            let truncated = test_model_path(&format!("truncated-{truncated_len}"));
            std::fs::write(&truncated, &bytes[..truncated_len]).expect("write truncated model");
            assert!(
                model_has_control_flow_nodes(&truncated),
                "a truncated control-flow model (len {truncated_len}) must stay conservative"
            );
            std::fs::remove_file(&truncated).ok();
        }
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn backend_and_control_flow_scans_parse_textproto_fixture() -> anyhow::Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/model.onnx.textproto");

        assert!(!model_requires_native_backend(&path)?);
        assert!(!model_has_control_flow_nodes(&path));
        Ok(())
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn auto_backend_detection_reads_onnx_node_types_not_incidental_strings() {
        use onnx_runtime_loader::proto::{
            ModelProto,
            onnx::{GraphProto, NodeProto, OperatorSetIdProto},
        };

        let mut model = ModelProto {
            producer_name: "BlockQuantizedMatMul appears only in metadata".to_string(),
            graph: Some(GraphProto::default()),
            ..ModelProto::default()
        };
        assert!(!model_proto_requires_native_backend(&model));

        model.graph.as_mut().unwrap().node.push(NodeProto {
            domain: "pkg.nxrt".to_string(),
            op_type: "BlockQuantizedMatMul".to_string(),
            ..NodeProto::default()
        });
        model.opset_import.push(OperatorSetIdProto {
            domain: "pkg.nxrt".to_string(),
            version: 1,
        });
        assert!(model_proto_requires_native_backend(&model));

        model.graph.as_mut().unwrap().node[0].domain = "example.wrong.domain".to_string();
        assert!(!model_proto_requires_native_backend(&model));

        model.graph.as_mut().unwrap().node[0].domain = "pkg.nxrt".to_string();
        model.opset_import[0].version = 2;
        assert!(!model_proto_requires_native_backend(&model));
    }

    #[test]
    fn backend_env_values_are_case_insensitive_and_reject_unknown_values() {
        assert_eq!(
            parse_backend_env("AuTo").unwrap(),
            EngineDecodeBackend::Auto
        );
        assert_eq!(parse_backend_env("ORT").unwrap(), EngineDecodeBackend::Ort);
        assert_eq!(
            parse_backend_env("native").unwrap(),
            EngineDecodeBackend::Native
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Auto, || {
                Ok(Some("nAtIvE".to_owned()))
            })
            .unwrap(),
            EngineDecodeBackend::Native
        );

        let error = parse_backend_env("cuda").unwrap_err().to_string();
        assert!(error.contains("ONNX_GENAI_BACKEND"), "{error}");
        assert!(error.contains("auto, ort, native"), "{error}");
    }

    #[test]
    fn explicit_backend_ignores_env_and_auto_honors_it() {
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Ort, || {
                Err(anyhow::anyhow!("unreadable environment value"))
            })
            .unwrap(),
            EngineDecodeBackend::Ort
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Native, || {
                panic!("explicit backend must not read ONNX_GENAI_BACKEND")
            })
            .unwrap(),
            EngineDecodeBackend::Native
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Auto, || {
                Ok(Some("ort".to_owned()))
            })
            .unwrap(),
            EngineDecodeBackend::Ort
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Auto, || Ok(None)).unwrap(),
            EngineDecodeBackend::Auto
        );
    }

    #[test]
    fn forced_ort_load_failure_includes_native_switch_hint() {
        let error = augment_backend_error::<()>(
            Err(anyhow::anyhow!("simulated native-only model load failure")),
            EngineDecodeBackend::Ort,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("EngineDecodeBackend::Native"), "{error}");
        assert!(error.contains("ONNX_GENAI_BACKEND=native"), "{error}");
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn forced_native_load_or_run_failure_includes_ort_switch_hint() {
        let error = augment_backend_error::<()>(
            Err(anyhow::anyhow!("simulated native decoder load/run failure")),
            EngineDecodeBackend::Native,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("EngineDecodeBackend::Ort"), "{error}");
        assert!(error.contains("ONNX_GENAI_BACKEND=ort"), "{error}");
    }

    #[cfg(not(feature = "native-backend"))]
    #[test]
    fn forced_native_without_feature_reports_how_to_switch() {
        let error = resolve_decode_backend(Path::new("unused.onnx"), EngineDecodeBackend::Native)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ONNX_GENAI_BACKEND=ort"), "{error}");
        assert!(error.contains("EngineDecodeBackend::Ort"), "{error}");
    }

    fn test_capacities() -> CapacityProviders {
        CapacityProviders {
            vram: Arc::new(FixedCapacity::new(1_000, 1_000)),
            host_ram: Arc::new(FixedCapacity::new(2_000, 2_000)),
            disk_spill: None,
        }
    }

    #[test]
    fn governor_handle_reflects_configured_limits_without_a_model() {
        let limits = ResourceLimits {
            vram_limit: ResourceLimit::Fraction(0.5),
            host_ram_limit: ResourceLimit::Auto,
            disk_spill_limit: None,
        };
        let governor = EngineResourceGovernor::new_with_capacities(
            limits.clone(),
            true,
            test_capacities(),
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
        )
        .unwrap();
        let snapshot = governor.snapshot();
        assert_eq!(snapshot.configured_limits, limits);
        assert_eq!(snapshot.resolved_limits.vram_bytes, 500);
        assert_eq!(snapshot.derived_budget.total_pages, 5);
        assert_eq!(snapshot.vram.headroom, 500);
        assert_eq!(snapshot.host_ram.used, 0);
        assert_eq!(snapshot.host_ram.limit, 500);
        assert_eq!(snapshot.host_ram.headroom, 500);
        assert_eq!(snapshot.disk_spill, None);

        let outcome = governor.set_vram_limit(ResourceLimit::Bytes(800)).unwrap();
        assert_eq!(outcome.new_limits.vram_bytes, 800);
        assert_eq!(
            governor.snapshot().configured_limits.vram_limit,
            ResourceLimit::Bytes(800)
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn two_engine_governors_keep_independent_host_cache_budgets() {
        let first = EngineResourceGovernor::new_with_capacities(
            ResourceLimits {
                host_ram_limit: ResourceLimit::Bytes(400),
                ..ResourceLimits::default()
            },
            false,
            test_capacities(),
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
        )
        .unwrap();
        let second = EngineResourceGovernor::new_with_capacities(
            ResourceLimits {
                host_ram_limit: ResourceLimit::Bytes(900),
                ..ResourceLimits::default()
            },
            false,
            test_capacities(),
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
        )
        .unwrap();

        assert_eq!(
            first.weight_offload_host_cache().configured_budget_bytes(),
            400
        );
        assert_eq!(
            second.weight_offload_host_cache().configured_budget_bytes(),
            900
        );
    }

    #[test]
    fn provisional_capacity_clamps_absolute_limits_without_fabricating_capacity() {
        let limits = ResourceLimits {
            vram_limit: ResourceLimit::Bytes(PROVISIONAL_VRAM_CAPACITY_BYTES + 1),
            host_ram_limit: ResourceLimit::Fraction(0.5),
            disk_spill_limit: Some(ResourceLimit::Auto),
        };
        let governor = EngineResourceGovernor::new(
            limits,
            false,
            ModelKvConfig {
                page_size_bytes: 1,
                tokens_per_page: 1,
            },
        )
        .unwrap();
        let snapshot = governor.snapshot();
        assert_eq!(
            snapshot.resolved_limits.vram_bytes,
            PROVISIONAL_VRAM_CAPACITY_BYTES
        );
        assert_eq!(
            snapshot.resolved_limits.host_ram_bytes,
            PROVISIONAL_HOST_RAM_CAPACITY_BYTES / 2
        );
        assert_eq!(
            snapshot.resolved_limits.disk_spill_bytes,
            Some(PROVISIONAL_DISK_CAPACITY_BYTES)
        );
    }

    #[test]
    fn governor_snapshot_reports_usage_limit_and_headroom_for_each_enabled_tier() {
        let capacities = CapacityProviders {
            vram: Arc::new(FixedCapacity::new(1_000, 900)),
            host_ram: Arc::new(FixedCapacity::new(2_000, 1_500)),
            disk_spill: Some(Arc::new(FixedCapacity::new(4_000, 3_000))),
        };
        let governor = EngineResourceGovernor::new_with_capacities(
            ResourceLimits {
                vram_limit: ResourceLimit::Bytes(800),
                host_ram_limit: ResourceLimit::Bytes(1_200),
                disk_spill_limit: Some(ResourceLimit::Bytes(3_000)),
            },
            false,
            capacities,
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
        )
        .unwrap();
        governor.byte_budget().try_reserve(300).unwrap();

        let snapshot = governor.snapshot();
        assert_eq!(
            snapshot.vram,
            onnx_genai_scheduler::TierSnapshot {
                used: 300,
                limit: 800,
                headroom: 500,
            }
        );
        assert_eq!(
            snapshot.host_ram,
            onnx_genai_scheduler::TierSnapshot {
                used: 500,
                limit: 1_200,
                headroom: 700,
            }
        );
        assert_eq!(
            snapshot.disk_spill,
            Some(onnx_genai_scheduler::TierSnapshot {
                used: 1_000,
                limit: 3_000,
                headroom: 2_000,
            })
        );
    }

    #[test]
    fn governor_handle_rejects_disabled_runtime_override() {
        let governor = EngineResourceGovernor::new_with_capacities(
            ResourceLimits::default(),
            false,
            test_capacities(),
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
        )
        .unwrap();
        assert!(matches!(
            governor.set_vram_limit(ResourceLimit::Bytes(800)),
            Err(EngineGovernorError::RuntimeOverrideDisabled)
        ));
    }

    #[test]
    fn token_logprobs_use_log_softmax_and_sorted_top_tokens() {
        let logits = [1.0, f32::NEG_INFINITY, 3.0, 2.0];
        let result = logprob_for_token(&logits, 3, 2);
        let logsumexp = 3.0 + ((1.0_f32 - 3.0).exp() + 1.0 + (2.0_f32 - 3.0).exp()).ln();

        assert_eq!(result.token_id, 3);
        assert_eq!(result.logprob, 2.0 - logsumexp);
        assert!(result.logprob <= 0.0);
        assert!(result.top.windows(2).all(|pair| pair[0].1 >= pair[1].1));
        assert!(result.top.iter().any(|(token_id, _)| *token_id == 3));
        assert!(result.top.iter().all(|(token_id, _)| *token_id != 1));
    }

    #[test]
    fn processor_chain_uses_documented_order() {
        let options = GenerateOptions {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            stop_sequences: vec![StopSequence::Tokens(vec![42])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        assert_eq!(
            chain.names(),
            vec![
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "stop_sequence",
                "temperature",
                "top_k",
                "top_p",
                "min_p"
            ]
        );
    }

    #[test]
    fn processor_chain_includes_json_constraint_before_sampling_filters() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json")
            .canonicalize()?;
        let tokenizer = Tokenizer::from_file(&fixture)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let options = GenerateOptions {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            constraint: Some(GenerateConstraint::Json),
            ..Default::default()
        };

        let chain = build_processor_chain(&options, Some(&tokenizer))?;

        assert_eq!(
            chain.names(),
            vec![
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "json_constraint",
                "temperature",
                "top_k",
                "top_p",
                "min_p"
            ]
        );
        Ok(())
    }

    #[test]
    fn greedy_selection_uses_argmax_after_processors() {
        let options = GenerateOptions {
            greedy: true,
            top_k: 2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 2.0, 4.0, 3.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.0),
            2
        );
    }

    #[test]
    fn sampled_selection_can_pick_non_argmax() {
        let options = GenerateOptions {
            greedy: false,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 0.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.75),
            1
        );
    }

    struct LastTokenSampler;

    impl Sampler for LastTokenSampler {
        fn sample(&mut self, logits: &[f32], _context: &ProcessorContext) -> TokenId {
            logits.len().saturating_sub(1) as TokenId
        }

        fn name(&self) -> &str {
            "last_token"
        }
    }

    #[test]
    fn custom_sampler_can_select_after_default_processors() {
        let options = GenerateOptions {
            top_k: 2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 2.0, 4.0, 3.0];
        let mut sampler = LastTokenSampler;

        assert_eq!(
            select_next_token_with_sampler(&mut logits, &context, &chain, &mut sampler),
            3
        );
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn default_processor_chain_is_empty_for_unchanged_defaults() {
        let options = GenerateOptions::default();
        let chain = build_processor_chain(&options, None).unwrap();
        assert!(chain.names().is_empty());
    }

    #[test]
    fn finish_reason_detects_eos_before_stop_sequence() {
        let options = GenerateOptions {
            eos_token_id: Some(7),
            stop_sequences: vec![StopSequence::Tokens(vec![7])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![7],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(7, &options, &chain, &context),
            Some(FinishReason::EosToken)
        );
    }

    #[test]
    fn finish_reason_detects_stop_sequence() {
        let options = GenerateOptions {
            stop_sequences: vec![StopSequence::Tokens(vec![2, 3])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![1, 2, 3],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(3, &options, &chain, &context),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }

    #[test]
    fn json_constraint_defers_stop_until_value_is_complete() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json),
            stop_sequences: vec![StopSequence::Text("}".to_string())],
            ..Default::default()
        };
        let chain_options = GenerateOptions {
            stop_sequences: options.stop_sequences.clone(),
            ..Default::default()
        };
        let chain = build_processor_chain(&chain_options, None).unwrap();
        let incomplete = ProcessorContext {
            generated_text: "{\"value\":".to_string(),
            ..Default::default()
        };
        let complete = ProcessorContext {
            generated_text: "{\"value\":1}".to_string(),
            ..Default::default()
        };

        assert_eq!(
            finish_reason_after_token(1, &options, &chain, &incomplete),
            None
        );
        assert_eq!(
            finish_reason_after_token(1, &options, &chain, &complete),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }

    #[test]
    fn incomplete_json_constraint_rejects_length_finishes() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json),
            ..Default::default()
        };
        for reason in [FinishReason::MaxTokens, FinishReason::Length] {
            let error = ensure_constrained_finish(&options, "{\"value\":", reason).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("stopped before a complete JSON value")
            );
        }
        ensure_constrained_finish(&options, "{\"value\":1}", FinishReason::MaxTokens).unwrap();
        ensure_constrained_finish(&options, "", FinishReason::EosToken).unwrap();
    }

    #[test]
    fn common_prefix_len_stops_before_rejected_draft_token() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(common_prefix_len(&[7], &[8]), 0);
    }

    #[test]
    fn tiny_fixture_generates_requested_tokens_end_to_end() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids.len(), 3);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn tiny_fixture_returns_opt_in_per_token_logprobs() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir_with_session_options(
            &fixture,
            EngineConfig::default(),
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.top_logprobs = Some(3);

        let result = engine.generate(request)?;
        let logprobs = result.logprobs.as_ref().expect("logprobs requested");

        assert_eq!(logprobs.len(), result.token_ids.len());
        for (token_id, token_logprob) in result.token_ids.iter().zip(logprobs) {
            assert_eq!(*token_id, token_logprob.token_id);
            assert!(token_logprob.logprob <= 0.0);
            assert!(
                token_logprob
                    .top
                    .windows(2)
                    .all(|pair| pair[0].1 >= pair[1].1)
            );
            assert!(
                token_logprob
                    .top
                    .iter()
                    .any(|(top_token_id, _)| top_token_id == token_id)
            );
        }

        let mut disabled = GenerateRequest::new("hello");
        disabled.options.max_new_tokens = 1;
        disabled.options.temperature = 0.0;
        disabled.options.stop_on_eos = false;
        assert!(engine.generate(disabled)?.logprobs.is_none());
        Ok(())
    }

    #[test]
    fn tiny_fixture_uses_past_present_decode_session_with_stable_greedy_output()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(matches!(
            engine.decode_path,
            ModelDecodePath::PastPresent {
                shared_buffer: false,
                ..
            }
        ));
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids, vec![22, 22, 20]);
        Ok(())
    }

    #[test]
    fn scatter_fixture_uses_static_cache_decode_session_with_stable_greedy_output()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm-scatter")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(matches!(
            engine.decode_path,
            ModelDecodePath::StaticCache { max_len } if max_len > 0
        ));
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids, vec![23, 15, 28]);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        Ok(())
    }

    #[test]
    fn tiny_fixture_speculative_matches_plain_greedy_with_k_gt_one() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut speculative = Engine::from_dir(
            &fixture,
            EngineConfig {
                draft_model: Some(fixture.clone()),
                num_speculative_tokens: 3,
                ..Default::default()
            },
        )?;

        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 6;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.num_speculative_tokens = Some(3);

        let baseline_result = baseline.generate(request.clone())?;
        let speculative_result = speculative.generate(request)?;

        assert_eq!(speculative_result.token_ids, baseline_result.token_ids);
        assert_eq!(
            speculative_result.finish_reason,
            baseline_result.finish_reason
        );
        assert_eq!(speculative_result.token_ids.len(), 6);
        Ok(())
    }

    #[test]
    fn tiny_fixture_stops_at_explicit_context_length_without_ort_error() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3]));
        request.options.max_new_tokens = 32;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.max_context = Some(16);

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids.len(), 13);
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_stops_at_explicit_context_length_without_ort_error()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let session_id = engine.create_session()?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3]));
        request.options.max_new_tokens = 32;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.max_context = Some(16);

        let result = engine.generate_in_session(session_id, request)?;

        assert_eq!(result.token_ids.len(), 13);
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert_eq!(engine.session_token_count(session_id)?, 16);
        engine.close_session(session_id)?;
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_persists_context_across_turns() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let session_id = engine.create_session()?;

        let mut first = GenerateRequest::new("hello");
        first.options.max_new_tokens = 2;
        first.options.temperature = 0.0;
        first.options.stop_on_eos = false;
        let first_result = engine.generate_in_session(session_id, first)?;
        let first_count = engine.session_token_count(session_id)?;

        let mut second = GenerateRequest::new(" world");
        second.options.max_new_tokens = 2;
        second.options.temperature = 0.0;
        second.options.stop_on_eos = false;
        let second_result = engine.generate_in_session(session_id, second)?;
        let second_count = engine.session_token_count(session_id)?;

        assert_eq!(first_result.token_ids.len(), 2);
        assert_eq!(second_result.token_ids.len(), 2);
        assert!(second_count > first_count);
        assert!(engine.sessions[&session_id].kv_token_count > 0);
        engine.close_session(session_id)?;
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "requires ONNX_GENAI_FIM_MODEL_DIR to point at a FIM-capable coder model"]
    fn fim_generation_runs_with_fim_capable_model() -> anyhow::Result<()> {
        let Some(model_dir) = onnx_genai_runtime_config::runtime_config()
            .fim_model_dir
            .as_deref()
        else {
            eprintln!("set ONNX_GENAI_FIM_MODEL_DIR to a Qwen2.5-Coder/StarCoder-style model");
            return Ok(());
        };
        let mut engine = Engine::from_dir(model_dir, EngineConfig::default())?;
        assert!(
            engine.fim_config().is_some(),
            "model tokenizer_config.json must expose recognized FIM tokens"
        );

        let mut options = GenerateOptions {
            max_new_tokens: 16,
            temperature: 0.0,
            ..Default::default()
        };
        options
            .stop_sequences
            .push(StopSequence::Text("\n\n".into()));

        let result =
            engine.generate_fim("fn add(a: i32, b: i32) -> i32 {\n    ", "\n}", options)?;

        assert!(!result.token_ids.is_empty());
        Ok(())
    }

    fn local_tiered_engine_config(chunk_size: usize) -> EngineConfig {
        EngineConfig {
            kv_connector: KvConnectorConfig {
                backend: KvConnectorBackend::LocalTiered(onnx_genai_kv::LocalTieredConfig {
                    chunk_size,
                    page_size: chunk_size,
                    ..onnx_genai_kv::LocalTieredConfig::default()
                }),
                chunk_size,
                ..KvConnectorConfig::default()
            },
            ..EngineConfig::default()
        }
    }

    #[test]
    fn null_connector_default_leaves_behavior_unchanged() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(!baseline.connector.is_active());

        let mut request =
            GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3, 5, 6, 7, 8, 9]));
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = baseline.generate(request)?;
        // With the default Null connector, no external activity happens at all.
        assert_eq!(baseline.last_connector_stats(), ConnectorStats::default());
        assert_eq!(result.token_ids.len(), 3);
        Ok(())
    }

    #[test]
    fn local_tiered_connector_stores_prefill_chunks() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, local_tiered_engine_config(2))?;
        assert!(engine.connector.is_active());

        let mut request =
            GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3, 5, 6, 7, 8, 9]));
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let baseline_ids = {
            let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
            baseline.generate(request.clone())?.token_ids
        };

        let result = engine.generate(request)?;

        // Store-after-prefill ran: complete chunks were pushed to the connector.
        assert!(
            engine.last_connector_stats().stores > 0,
            "expected connector store path to push chunks, got {:?}",
            engine.last_connector_stats()
        );
        // The store path is a pure side effect for a first, unseen request:
        // nothing is resident to fetch, so output matches full recompute.
        assert_eq!(result.token_ids, baseline_ids);
        Ok(())
    }

    #[test]
    fn local_tiered_connector_fetch_reuse_is_token_identical() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, local_tiered_engine_config(2))?;

        // Request 1 populates the connector with the prompt's KV chunks.
        let prompt = vec![10, 11, 12, 13, 14, 15];
        let mut warm = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
        warm.options.max_new_tokens = 1;
        warm.options.temperature = 0.0;
        warm.options.stop_on_eos = false;
        engine.generate(warm)?;
        assert!(engine.last_connector_stats().stores > 0);

        // Drop the in-process caches so the connector is the ONLY source of
        // cross-session reuse — simulating a fresh process / different node that
        // shares nothing but the connector.
        engine.token_prefix_cache.clear();
        engine.prefix_cache = PrefixCache::new();

        // Request 2 shares the whole prefix (≥ 1 chunk) with request 1.
        let mut reuse = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
        reuse.options.max_new_tokens = 4;
        reuse.options.temperature = 0.0;
        reuse.options.stop_on_eos = false;
        let reuse_result = engine.generate(reuse)?;
        let stats = engine.last_connector_stats();

        // (a) Prefill was genuinely shortened: real KV bytes were fetched and
        // injected into the runner.
        assert!(
            stats.fetched_tokens > 0 && stats.chunk_hits > 0,
            "expected connector fetch to materialize KV, got {stats:?}"
        );
        // At least one prompt token is always left to feed the decoder.
        assert!(stats.fetched_tokens < prompt.len());

        // (b) Output is byte-for-byte identical to full recompute with a Null
        // connector — proving the materialized KV is correct, not just present.
        let baseline_ids = {
            let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
            let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
            request.options.max_new_tokens = 4;
            request.options.temperature = 0.0;
            request.options.stop_on_eos = false;
            baseline.generate(request)?.token_ids
        };
        assert_eq!(
            reuse_result.token_ids, baseline_ids,
            "connector-reuse output must match full recompute exactly"
        );
        Ok(())
    }
}
