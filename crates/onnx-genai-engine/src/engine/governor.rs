//! Device resolution and the engine-owned resource governor.

use super::*;
use crate::engine::memory_plan::{Holder, ModelMemoryPlan};

#[cfg(feature = "native-backend")]
pub(crate) fn resolve_native_decode_device(
    configured: Option<crate::native_decode::NativeDecodeDevice>,
    session_options: &SessionOptions,
) -> anyhow::Result<crate::native_decode::NativeDecodeDevice> {
    use crate::native_decode::NativeDecodeDevice;

    if let Some(device) = configured {
        return validate_native_decode_device(device);
    }

    match session_options
        .execution_providers
        .iter()
        .find(|provider| !provider.caps.is_host())
    {
        None => Ok(NativeDecodeDevice::Cpu),
        Some(provider) if provider.native_plugin_bridge().is_some() => {
            let bridge = provider.native_plugin_bridge().expect("checked above");
            if bridge.lib.as_os_str().is_empty() || !bridge.lib.is_file() {
                anyhow::bail!(
                    "native decoder backend could not load execution provider {:?}: plugin library '{}' is missing; fix by setting the provider's plugin library environment variable or selecting CPU/CUDA",
                    provider.caps.name,
                    bridge.lib.display()
                );
            }
            validate_native_decode_device(NativeDecodeDevice::Plugin {
                library: bridge.lib,
                registration_name: Some(bridge.registration_name),
                provider_name: bridge.provider_name,
            })
        }
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
            anyhow::bail!(
                "native decoder backend does not support execution provider {:?}: it is neither host, CUDA, nor an ORT plugin with a loadable native bridge; fix by selecting CPU/CUDA or configuring a plugin provider library",
                provider.caps.name
            )
        }
    }
}

#[cfg(feature = "native-backend")]
pub(crate) fn validate_native_decode_device(
    device: crate::native_decode::NativeDecodeDevice,
) -> anyhow::Result<crate::native_decode::NativeDecodeDevice> {
    match device {
        crate::native_decode::NativeDecodeDevice::Cpu => Ok(device),
        crate::native_decode::NativeDecodeDevice::Plugin { ref library, .. } => {
            if library.as_os_str().is_empty() || !library.is_file() {
                anyhow::bail!(
                    "native decoder backend plugin device cannot start because library '{}' is missing; fix by configuring the execution provider plugin library path",
                    library.display()
                );
            }
            Ok(device)
        }
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
pub(crate) const PROVISIONAL_VRAM_CAPACITY_BYTES: u64 = 8 << 30;
pub(crate) const PROVISIONAL_HOST_RAM_CAPACITY_BYTES: u64 = 16 << 30;
pub(crate) const PROVISIONAL_DISK_CAPACITY_BYTES: u64 = 16 << 30;

/// Engine-owned Resource Governor handle.
pub struct EngineResourceGovernor {
    inner: ResourceGovernor,
    allow_runtime_override: bool,
    /// Grants the tier budgets that `inner` merely reports.
    ///
    /// Shared, so every lease this engine hands out is counted against the same
    /// ledger; a per-caller governor would let each holder believe it had the
    /// whole tier to itself.
    memory: onnx_runtime_memory_governor::LedgerGovernor,
    /// The fixed device reservation -- weights and runtime overhead -- held as a
    /// lease rather than subtracted before the ledger sees it.
    ///
    /// Kept for its Drop: unloading the model returns these bytes to the tier.
    plan: std::sync::Mutex<ModelMemoryPlan>,
    #[cfg(feature = "native-backend")]
    weight_offload_host_cache: onnx_runtime_ep_cpu::WeightOffloadHostCache,
}

impl EngineResourceGovernor {
    pub(crate) fn new(
        limits: ResourceLimits,
        allow_runtime_override: bool,
        kv_config: ModelKvConfig,
        model_weights_bytes: u64,
    ) -> Result<Self, ResourceError> {
        let capacities = fallback_capacity_providers(&limits);
        Self::new_with_capacities(
            limits,
            allow_runtime_override,
            capacities,
            kv_config,
            model_weights_bytes,
        )
    }

    pub(crate) fn new_with_capacities(
        limits: ResourceLimits,
        allow_runtime_override: bool,
        capacities: CapacityProviders,
        kv_config: ModelKvConfig,
        model_weights_bytes: u64,
    ) -> Result<Self, ResourceError> {
        // Model weights are measured from the package on disk (graph plus its
        // ONNX external-data blob), so the KV budget is derived from what is
        // actually left rather than from the whole ceiling.
        //
        // TODO(RULES.md #2, §26.11.4): the remaining reservations still read
        // zero — activations and runtime overhead need runtime instrumentation,
        // and device capacity needs a vendor-neutral EP-backed query to replace
        // the provisional constants below.
        let inner = ResourceGovernor::new(
            limits,
            capacities,
            VramBreakdown {
                model_weights_bytes,
                // Not measured. `None` rather than `0`, so nothing downstream
                // can mistake the absence of a number for a number.
                activations_bytes: None,
                ort_overhead_bytes: None,
            },
            kv_config,
        )?;
        // Say it once, out loud. These two are not measured, and a zero in a
        // breakdown is indistinguishable from a measurement of zero -- which is
        // how `activations_bytes: 0` came to be published as fact in the profile
        // JSON until #629.
        //
        // Not a refusal: neither is required to build anything, so being wrong
        // about them makes admission optimistic rather than impossible, and the
        // house rule for that case is to run and say so (#649). A quantity whose
        // absence makes a buffer unbuildable is a different case and does refuse.
        tracing::warn!(
            model_weights_bytes,
            "device memory breakdown: activations and runtime overhead are not measured and are \
             reported as unknown, not as zero; admission is correspondingly optimistic (#514)"
        );
        // The ledger's device tier is the *device*, not a sub-budget of it.
        //
        // It used to be seeded with `derived_budget.kv_bytes`, which made the
        // tier mean "bytes KV may have". That was safe for the one holder there
        // was, and it is exactly why nothing else could join: a weight-residency
        // pool or an activation reservation leased from it would have been
        // charged a second time and taken the room out of KV's allowance.
        //
        // So the tier is the resolved ceiling, and the fixed reservation the
        // ceiling already accounts for -- weights and runtime overhead, per
        // `VramBreakdown` -- is taken as a lease instead of a subtraction. The
        // bytes left for KV are identical; the difference is that they are now
        // what remains after a claim the ledger can see, rather than after
        // arithmetic it cannot.
        let snapshot = inner.snapshot();
        let memory = onnx_runtime_memory_governor::LedgerGovernor::new(
            onnx_runtime_memory_governor::LeaseLedger::new(
                snapshot.resolved_limits.vram_bytes,
                snapshot.resolved_limits.host_ram_bytes,
                snapshot.disk_spill.as_ref().map_or(0, |tier| tier.limit),
            ),
        );
        // `reservation_applied` is false when honouring the reservation would
        // have left no room for even one KV page. It is an estimate over a
        // ceiling that may itself be provisional, so it must never be the reason
        // a model refuses to start -- and when it was not applied there is
        // nothing to charge.
        let mut plan = ModelMemoryPlan::new(memory.clone());
        if snapshot.derived_budget.reservation_applied {
            plan.reserve(
                Holder::FixedDeviceReservation,
                snapshot.derived_budget.reserved_bytes,
            )
            .map_err(|error| ResourceError::BudgetArithmeticOverflow {
                operation: "charging the fixed device reservation to the memory ledger",
                reason: error.to_string(),
            })?;
        }
        #[cfg(feature = "native-backend")]
        // A host-cache lease is a standing claim on RAM. Take it only when the
        // offload path can actually admit experts; otherwise a disabled cache
        // would consume the host tier before recurrent state or host KV get a
        // chance to fit.
        let weight_offload_host_budget =
            if std::env::var_os(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV)
                .is_some_and(|value| value == "1")
            {
                snapshot.resolved_limits.host_ram_bytes
            } else {
                0
            };
        #[cfg(feature = "native-backend")]
        let weight_offload_host_cache = onnx_runtime_ep_cpu::WeightOffloadHostCache::new_leased(
            weight_offload_host_budget,
            &memory,
            Holder::WeightOffloadHostCache.id(),
        )
        .map_err(|reason| ResourceError::BudgetArithmeticOverflow {
            operation: "leasing the native weight-offload host-cache budget",
            reason: reason.to_string(),
        })?;
        Ok(Self {
            inner,
            allow_runtime_override,
            memory,
            plan: std::sync::Mutex::new(plan),
            #[cfg(feature = "native-backend")]
            weight_offload_host_cache,
        })
    }

    /// The authority that grants memory leases against this engine's budgets.
    ///
    /// Separate from [`Self::snapshot`], which only *reports*. Anything that
    /// holds bytes takes a lease here so the tier totals reflect what is
    /// actually occupied rather than what was planned.
    /// Every claim this model makes, in one place.
    pub(crate) fn plan(&self) -> std::sync::MutexGuard<'_, ModelMemoryPlan> {
        self.plan.lock().expect("model memory plan lock poisoned")
    }

    /// What this model actually holds, per holder, as leases rather than as the
    /// estimate `GovernorSnapshot::breakdown` reports.
    ///
    /// The two answer different questions and it is worth being able to compare
    /// them: `breakdown` is what the ceiling was divided up *expecting*, this is
    /// what was *granted*. A gap between them is the interesting case -- either
    /// something holds memory the plan did not predict, or the plan predicted
    /// memory nothing took.
    /// What this model holds *through the plan*, per holder.
    ///
    /// **Not every lease.** The KV pool's lease lives inside `PagedKvCache` and
    /// the weight-residency pool's inside the execution provider, because each
    /// is held by the thing that must outlive it. Those bytes went through the
    /// same ledger and are in [`Self::leased_bytes_on`], but they are not
    /// itemised here: the ledger tracks per-tier usage, not per-holder
    /// attribution, so there is nothing to itemise them from.
    ///
    /// Said plainly because a breakdown that quietly omitted the two largest
    /// holders would be read as the whole picture -- which is how
    /// `activations_bytes: 0` came to be published as fact.
    pub fn leased_breakdown(&self) -> Vec<(&'static str, onnx_runtime_memory_governor::Tier, u64)> {
        self.plan().breakdown()
    }

    /// Bytes leased on `tier`, across every holder this model has.
    ///
    /// Read from the ledger rather than summed from the plan, so it includes the
    /// leases the plan does not itself hold: the KV pool's lives inside
    /// `PagedKvCache` and the weight-residency pool's inside the execution
    /// provider, because each is held by the thing that must outlive it.
    pub fn leased_bytes_on(&self, tier: onnx_runtime_memory_governor::Tier) -> u64 {
        onnx_runtime_memory_governor::MemoryGovernor::used(&self.memory, tier)
    }

    pub fn memory(&self) -> &onnx_runtime_memory_governor::LedgerGovernor {
        &self.memory
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

    pub(crate) fn byte_budget(&self) -> onnx_genai_scheduler::ByteBudget {
        self.inner
            .byte_budget()
            .with_ceiling(std::sync::Arc::new(LedgerAdmissionCeiling {
                memory: self.memory.clone(),
                kv_pool_bytes: self.plan().kv_pool_bytes_handle(),
            }))
    }

    #[cfg(feature = "native-backend")]
    pub(crate) fn weight_offload_host_cache(&self) -> onnx_runtime_ep_cpu::WeightOffloadHostCache {
        self.weight_offload_host_cache.clone()
    }
}

/// Bounds admission by what the memory ledger says is actually free.
///
/// The scheduler's [`ByteBudget`] is seeded at load with the KV budget derived
/// from the device limit less an *estimate* of everything that is not KV. The
/// things that estimate subtracted are not constants: weight residency grows
/// when a model touches more experts than its budget assumed, recurrent state
/// is charged when a hybrid model loads, and a third-party provider may lease
/// for reasons the engine cannot enumerate. A ceiling computed once at load
/// does not see any of it, so admission would keep saying yes against room that
/// had already been spent, and the failure would surface at an allocation far
/// from the decision that caused it.
///
/// [`ByteBudget`]: onnx_genai_scheduler::ByteBudget
///
/// # Why pool grants are added back
///
/// `available(Device)` is what is free after *every* device lease. A KV pool
/// charged to the device tier would be among them, and charging admission for
/// it would count the same bytes twice -- that pool is precisely the memory
/// admitted sequences run in -- so it is added back.
///
/// **Today that add-back is always zero**, because every KV pool holder is
/// `Tier::Host`: the pools are host-allocated despite their `num_gpu_pages`
/// lineage. The arithmetic is here so that moving a pool to the device tier
/// cannot silently halve what admission will accept, which is the kind of
/// change that looks local and is not. `a_host_tier_pool_is_not_added_back`
/// and `a_device_tier_pool_is_added_back` in `memory_plan` pin both halves.
#[derive(Debug)]
struct LedgerAdmissionCeiling {
    memory: onnx_runtime_memory_governor::LedgerGovernor,
    kv_pool_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl onnx_genai_scheduler::AdmissionCeiling for LedgerAdmissionCeiling {
    fn ceiling_bytes(&self) -> u64 {
        use onnx_runtime_memory_governor::MemoryGovernor as _;
        self.memory
            .available(onnx_runtime_memory_governor::Tier::Device)
            .saturating_add(
                self.kv_pool_bytes
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
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

pub(crate) fn fallback_capacity_providers(limits: &ResourceLimits) -> CapacityProviders {
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

pub(crate) fn governor_kv_config(
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

/// Build a governor for a component that owns memory but is not the engine.
///
/// The pipeline needs one for the same reasons the engine does: to size its KV
/// pool from a budget and to lease what it allocates. Returning the governor
/// rather than a single number keeps both uses on **one** ledger — building a
/// second governor per question would let each holder believe it had the whole
/// tier to itself, which is how two pools end up each sure they own the device.
pub(crate) fn component_governor(
    config: &EngineConfig,
    kv_model: Option<&KvModelInfo>,
) -> anyhow::Result<EngineResourceGovernor> {
    EngineResourceGovernor::new(
        config.limits.clone(),
        config.allow_runtime_override,
        governor_kv_config(kv_model, config)?,
        // This resolves ceilings only; the model path is not in scope here, so
        // the weight reservation is left at zero.
        0,
    )
    .context("failed to resolve the engine memory budget for decoder fixed state")
}
