//! Device resolution and the engine-owned resource governor.

use super::*;

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
    _fixed_device_reservation: Option<onnx_runtime_memory_governor::MemoryLease>,
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
                activations_bytes: 0,
                ort_overhead_bytes: 0,
            },
            kv_config,
        )?;
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
        let fixed_reservation = if snapshot.derived_budget.reservation_applied {
            snapshot.derived_budget.reserved_bytes
        } else {
            0
        };
        let fixed_device_reservation = if fixed_reservation > 0 {
            Some(
                onnx_runtime_memory_governor::MemoryGovernor::reserve(
                    &memory,
                    onnx_runtime_memory_governor::Tier::Device,
                    fixed_reservation,
                    onnx_runtime_memory_governor::MemoryRole::Weights,
                    crate::engine::memory_plan::Holder::FixedDeviceReservation.id(),
                )
                .map_err(|error| ResourceError::BudgetArithmeticOverflow {
                    operation: "charging the fixed device reservation to the memory ledger",
                    reason: error.to_string(),
                })?,
            )
        } else {
            None
        };
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
            memory,
            _fixed_device_reservation: fixed_device_reservation,
            #[cfg(feature = "native-backend")]
            weight_offload_host_cache,
        })
    }

    /// The authority that grants memory leases against this engine's budgets.
    ///
    /// Separate from [`Self::snapshot`], which only *reports*. Anything that
    /// holds bytes takes a lease here so the tier totals reflect what is
    /// actually occupied rather than what was planned.
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
        self.inner.byte_budget()
    }

    #[cfg(feature = "native-backend")]
    pub(crate) fn weight_offload_host_cache(&self) -> onnx_runtime_ep_cpu::WeightOffloadHostCache {
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
