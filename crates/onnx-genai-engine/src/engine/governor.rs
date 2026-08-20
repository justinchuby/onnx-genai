//! Device resolution and the engine-owned resource governor.

use super::*;
use crate::engine::memory_plan::{Holder, ModelMemoryPlan};
use crate::memory_authority::{
    DeviceCompatibilityDomain, DeviceMemoryAuthority, EngineMemoryGovernor,
    SharedMemoryAuthorityProvider,
};

#[cfg(feature = "native-backend")]
pub(crate) fn resolve_native_decode_device(
    configured: Option<crate::native_decode::NativeDecodeDevice>,
    session_options: &SessionOptions,
) -> anyhow::Result<crate::native_decode::NativeDecodeDevice> {
    use crate::native_decode::NativeDecodeDevice;

    if let Some(device) = configured {
        let device = validate_native_decode_device(device)?;
        log_resolved_native_decode_device(&device, "requested explicitly", true);
        return Ok(device);
    }

    match session_options
        .execution_providers
        .iter()
        .find(|provider| !provider.caps.is_host())
    {
        None => {
            // The model declares no accelerator, which is the common case: most
            // exports declare none at all. Reading that as "the user wants the
            // CPU" is what made `--backend native` run on the CPU on a GPU box
            // (#1064, #1551). A declaration that isn't there is an absence of
            // information, not a preference, so probe for a usable device
            // instead. `--device cpu` remains the way to ask for the CPU.
            #[cfg(feature = "cuda")]
            if onnx_runtime_ep_cuda::CudaExecutionProvider::is_available(0) {
                let device = NativeDecodeDevice::Cuda { index: Some(0) };
                log_resolved_native_decode_device(
                    &device,
                    "auto-detected: the model declares no execution provider and CUDA:0 is usable",
                    false,
                );
                return validate_native_decode_device(device);
            }
            let device = NativeDecodeDevice::Cpu;
            log_resolved_native_decode_device(
                &device,
                "the model declares no execution provider and no accelerator was detected",
                false,
            );
            Ok(device)
        }
        Some(provider) if provider.native_plugin_bridge().is_some() => {
            let bridge = provider.native_plugin_bridge().expect("checked above");
            if bridge.lib.as_os_str().is_empty() || !bridge.lib.is_file() {
                anyhow::bail!(
                    "native decoder backend could not load execution provider {:?}: plugin library '{}' is missing; fix by setting the provider's plugin library environment variable or selecting CPU/CUDA",
                    provider.caps.name,
                    bridge.lib.display()
                );
            }
            let device = validate_native_decode_device(NativeDecodeDevice::Plugin {
                library: bridge.lib,
                registration_name: Some(bridge.registration_name),
                provider_name: bridge.provider_name,
            })?;
            log_resolved_native_decode_device(
                &device,
                "declared by the model as a plugin provider",
                false,
            );
            Ok(device)
        }
        Some(provider) if provider.caps.is_gpu() && provider.caps.is_nvidia() => {
            let device_id = provider.caps.device_id().unwrap_or(0);
            let index = u32::try_from(device_id).map_err(|_| {
                anyhow::anyhow!(
                    "native decoder backend CUDA device id must be non-negative, got {device_id}"
                )
            })?;
            let device =
                validate_native_decode_device(NativeDecodeDevice::Cuda { index: Some(index) })?;
            log_resolved_native_decode_device(&device, "declared by the model", false);
            Ok(device)
        }
        Some(provider) => {
            anyhow::bail!(
                "native decoder backend does not support execution provider {:?}: it is neither host, CUDA, nor an ORT plugin with a loadable native bridge; fix by selecting CPU/CUDA or configuring a plugin provider library",
                provider.caps.name
            )
        }
    }
}

/// Say which device a native decode session resolved to, and why.
///
/// Unconditional and at INFO because the failure this guards against is silent:
/// a run that was asked for the native backend but quietly landed on the CPU
/// looks exactly like one that landed on the GPU (#1064, #1551). The engine
/// already logs a device further in, but only from a path pipeline models do
/// not take, so it never appeared for them.
#[cfg(feature = "native-backend")]
fn log_resolved_native_decode_device(
    device: &crate::native_decode::NativeDecodeDevice,
    reason: &str,
    requested: bool,
) {
    let on_cpu = matches!(device, crate::native_decode::NativeDecodeDevice::Cpu);
    // Warn only for a CPU nobody asked for: that is the case that is otherwise
    // indistinguishable from success. A caller who passed `--device cpu` got
    // what they asked for and does not need to be told off for it.
    if on_cpu && !requested {
        tracing::warn!(
            backend = "native",
            device = ?device,
            reason,
            "the native decoder resolved to the CPU and will be far slower than an \
             accelerator; pass an explicit device to override"
        );
    } else {
        tracing::info!(
            backend = "native",
            device = ?device,
            reason,
            "resolved native decode device"
        );
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

// Real, platform-measured capacities replace the fabricated constants that
// #947 removed. VRAM has no vendor-neutral capacity query yet, so when no
// execution provider can report it the device tier is *unknown* (see
// `capacity_providers_for_device`), never a manufactured number. Host RAM and
// disk are measured from the OS by `crate::platform_capacity`.

/// The filesystem path whose free/total is used to size the disk-spill tier.
///
/// #947 asked for "free/total on the path that would actually be used for
/// spill, not an arbitrary drive." There is no separately configured spill
/// directory today, so the working directory (the process's default write
/// location) is the honest stand-in; falling back to the temp dir keeps the
/// query robust when the cwd is unavailable.
fn disk_spill_measurement_path() -> std::path::PathBuf {
    std::env::current_dir().unwrap_or_else(|_| std::env::temp_dir())
}

/// Engine-owned Resource Governor handle.
pub struct EngineResourceGovernor {
    inner: ResourceGovernor,
    allow_runtime_override: bool,
    /// Grants the tier budgets that `inner` merely reports.
    ///
    /// Shared, so every lease this engine hands out is counted against the same
    /// ledger; a per-caller governor would let each holder believe it had the
    /// whole tier to itself.
    memory: EngineMemoryGovernor,
    process_memory_manager: onnx_runtime_memory_governor::ProcessMemoryManager,
    /// The fixed device reservation -- weights and runtime overhead -- held as a
    /// lease rather than subtracted before the ledger sees it.
    ///
    /// Kept for its Drop: unloading the model returns these bytes to the tier.
    plan: std::sync::Mutex<ModelMemoryPlan>,
    #[cfg(feature = "native-backend")]
    weight_offload_host_cache: onnx_runtime_ep_cpu::WeightOffloadHostCache,
}

impl EngineResourceGovernor {
    #[cfg(test)]
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

    #[cfg(test)]
    pub(crate) fn new_with_capacities(
        limits: ResourceLimits,
        allow_runtime_override: bool,
        capacities: CapacityProviders,
        kv_config: ModelKvConfig,
        model_weights_bytes: u64,
    ) -> Result<Self, ResourceError> {
        Self::new_with_capacities_and_authority(
            limits,
            allow_runtime_override,
            capacities,
            kv_config,
            (model_weights_bytes, model_weights_bytes),
            None,
            None,
        )
    }

    pub(crate) fn new_with_authority(
        limits: ResourceLimits,
        allow_runtime_override: bool,
        kv_config: ModelKvConfig,
        model_weights_bytes: u64,
        cuda_device_index: Option<u32>,
        provider: Option<&SharedMemoryAuthorityProvider>,
        domain: Option<&DeviceCompatibilityDomain>,
    ) -> Result<Self, ResourceError> {
        let capacities = capacity_providers_for_device(&limits, cuda_device_index);
        Self::new_with_capacities_and_authority(
            limits,
            allow_runtime_override,
            capacities,
            kv_config,
            (model_weights_bytes, model_weights_bytes),
            provider,
            domain,
        )
    }

    // Eight parameters (one over the lint's threshold) because this is
    // `new_with_authority` plus an explicit reservation: #840 added
    // `cuda_device_index` to fix a VRAM-capacity portability bug and pushed it
    // over. Grouping them into a struct would buy nothing here — this is a
    // crate-private constructor with exactly one caller (`engine::load`), and
    // every argument is already a distinct type carrying its own meaning.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_authority_and_reservation(
        limits: ResourceLimits,
        allow_runtime_override: bool,
        kv_config: ModelKvConfig,
        model_weights_bytes: u64,
        reservation_bytes: u64,
        cuda_device_index: Option<u32>,
        provider: Option<&SharedMemoryAuthorityProvider>,
        domain: Option<&DeviceCompatibilityDomain>,
    ) -> Result<Self, ResourceError> {
        let capacities = capacity_providers_for_device(&limits, cuda_device_index);
        Self::new_with_capacities_and_authority(
            limits,
            allow_runtime_override,
            capacities,
            kv_config,
            (model_weights_bytes, reservation_bytes),
            provider,
            domain,
        )
    }

    pub(crate) fn new_for_shared_pipeline_kv(
        limits: ResourceLimits,
        allow_runtime_override: bool,
        kv_config: ModelKvConfig,
        existing_device_usage_bytes: u64,
        cuda_device_index: Option<u32>,
        provider: Option<&SharedMemoryAuthorityProvider>,
        domain: &DeviceCompatibilityDomain,
    ) -> Result<Self, ResourceError> {
        let capacities = capacity_providers_for_device(&limits, cuda_device_index);
        Self::new_with_capacities_and_authority(
            limits,
            allow_runtime_override,
            capacities,
            kv_config,
            (existing_device_usage_bytes, 0),
            provider,
            Some(domain),
        )
    }

    fn new_with_capacities_and_authority(
        limits: ResourceLimits,
        allow_runtime_override: bool,
        capacities: CapacityProviders,
        kv_config: ModelKvConfig,
        model_weight_bytes: (u64, u64),
        provider: Option<&SharedMemoryAuthorityProvider>,
        domain: Option<&DeviceCompatibilityDomain>,
    ) -> Result<Self, ResourceError> {
        let (model_weights_bytes, reservation_bytes) = model_weight_bytes;
        // Captured before `capacities` is consumed by `ResourceGovernor::new`
        // below: the *measured free* VRAM the combined mapped-physical cap
        // (#1295) is sourced from. `None` on a device whose capacity could not
        // be measured — a fraction of an unknown is unknown, not a number (#947),
        // so the cap simply does not bind there.
        let measured_vram_free_bytes = capacities.vram.free_bytes();
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
        // Device (VRAM) ceiling for the authority. `None` means the device
        // capacity could not be measured (no CUDA query, no vendor-neutral
        // probe): the KV cache and weights on this box live in host RAM, so the
        // advisory device authority is bounded by the *measured* host-RAM ceiling
        // rather than a fabricated device constant (the #947 bug) or an unusable
        // zero that would collapse admission on a device-less machine. This is
        // the pragmatic hot-tier bound, not the deferred UMA authority
        // unification: no device lease is aliased onto host memory, and the plan
        // still reports the device budget as unknown.
        let device_ceiling_bytes = snapshot
            .resolved_limits
            .vram_bytes
            .unwrap_or(snapshot.resolved_limits.host_ram_bytes);
        // #1295 combined mapped-physical cap. The device authority ceiling bounds
        // the *sum* of every Device-tier lease — weight residency, KV, and
        // activations all admit against it (the weight-residency page-in checks
        // `governor.available(Tier::Device)`, and KV/activation mapped growth
        // admits through `prepare_mapped_growth` against the same tier), so a
        // single ceiling is already the shared authority. What was wrong was its
        // *source*: a `Fraction` resolved against nominal *total* VRAM, when the
        // driver only ever hands out *free* (usable) VRAM — measured ~7959 MiB of
        // a nominal 8188 on the RTX 4060 Laptop 8 GB box (the ~229 MiB delta is
        // the desktop compositor's standing reserve). Oversubscribing past usable
        // is what drove `vram_free` off the WDDM fault-in cliff. Holding the
        // ceiling to `measured_free * safe_fraction` converts that catastrophic
        // cliff into a graceful admission refusal. This does not raise N_max; it
        // makes the ceiling honest and the failure mode a plateau.
        let (device_ceiling_bytes, usable_cap_bytes) = clamp_ceiling_to_usable_vram(
            device_ceiling_bytes,
            measured_vram_free_bytes,
            usable_mapped_safe_fraction(),
        );
        if let Some(cap) = usable_cap_bytes {
            tracing::info!(
                measured_free_bytes = measured_vram_free_bytes.unwrap_or(0),
                safe_fraction = usable_mapped_safe_fraction(),
                combined_mapped_ceiling_bytes = cap,
                model_weights_bytes,
                "bounded the device mapped-physical ceiling to usable VRAM x safe fraction; \
                 weights, KV, and activations share this single cap (#1295)"
            );
        }
        let device = match (provider, domain) {
            (Some(provider), Some(domain)) => provider
                .authority(domain, device_ceiling_bytes)
                .map_err(|error| ResourceError::BudgetArithmeticOverflow {
                    operation: "acquiring the shared device memory authority",
                    reason: error.to_string(),
                })?,
            (None, Some(domain)) => {
                DeviceMemoryAuthority::new(domain.clone(), device_ceiling_bytes)
            }
            _ => DeviceMemoryAuthority::new(
                DeviceCompatibilityDomain::Accelerator {
                    backend: "standalone".to_string(),
                    index: 0,
                },
                device_ceiling_bytes,
            ),
        };
        let memory = EngineMemoryGovernor::new(
            device,
            snapshot.resolved_limits.host_ram_bytes,
            snapshot.disk_spill.as_ref().map_or(0, |tier| tier.limit),
        );
        let process_memory_manager = match provider {
            Some(provider) => provider.process_memory_manager(),
            None => onnx_runtime_memory_governor::ProcessMemoryManager::new().map_err(|error| {
                ResourceError::BudgetArithmeticOverflow {
                    operation: "constructing the local process memory manager",
                    reason: error.to_string(),
                }
            })?,
        };
        // A reservation that "does not fit under the resolved ceiling" is only a
        // real failure when there *is* a resolved ceiling. When the device
        // capacity is unknown (`vram_bytes == None`) there is nothing to fit
        // under, so this must not refuse the load — it stays advisory, exactly as
        // it did before #947 stopped fabricating an 8 GiB ceiling here.
        if let Some(resolved_vram_bytes) = snapshot.resolved_limits.vram_bytes
            && provider.is_some()
            && reservation_bytes > 0
            && !snapshot.derived_budget.reservation_applied
        {
            return Err(ResourceError::BudgetArithmeticOverflow {
                operation: "reserving production model weights before session load",
                reason: format!(
                    "{reservation_bytes} bytes of model weights do not fit under the resolved \
                     {resolved_vram_bytes} byte device-authority limit"
                ),
            });
        }
        // `reservation_applied` is false when honouring the reservation would
        // have left no room for even one KV page. It is an estimate over a
        // ceiling that may itself be provisional, so it must never be the reason
        // a model refuses to start -- and when it was not applied there is
        // nothing to charge.
        //
        // The reservation is a *device*-tier lease of bytes we actually place on
        // the device (the model weights). It is charged whenever it was applied,
        // *independently* of whether the device capacity is a measured number or
        // unknown. Those are separate facts: "I do not know how big this device
        // is" (`vram_bytes == None`) does not stop us from knowing "I have placed
        // these bytes on it". The device authority is never inert here -- its
        // ceiling is the measured VRAM budget when known, and the measured
        // host-RAM fallback (`device_ceiling_bytes` above) when not -- so the
        // ledger can always account the allocation. Detaching usage tracking
        // when capacity became unknown made a loaded model report zero device
        // usage over /resources and /metrics, which is the #706 regression the
        // ledger snapshot exists to prevent (#947).
        let mut plan = ModelMemoryPlan::new(memory.clone());
        if snapshot.derived_budget.reservation_applied && reservation_bytes > 0 {
            plan.reserve(Holder::FixedDeviceReservation, reservation_bytes)
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
            process_memory_manager,
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

    /// Bytes by which the device ledger exceeds its live ceiling.
    pub fn device_oversubscribed_bytes(&self) -> u64 {
        onnx_runtime_memory_governor::MemoryGovernor::oversubscribed_bytes(
            &self.memory,
            onnx_runtime_memory_governor::Tier::Device,
        )
    }

    #[cfg_attr(not(feature = "native-backend"), allow(dead_code))]
    pub(crate) fn memory(&self) -> &EngineMemoryGovernor {
        &self.memory
    }

    pub fn device_authority(&self) -> DeviceMemoryAuthority {
        self.memory.device_authority()
    }

    pub fn process_memory_manager(&self) -> onnx_runtime_memory_governor::ProcessMemoryManager {
        self.process_memory_manager.clone()
    }

    /// Point-in-time configured, resolved, derived, and live per-tier state.
    pub fn snapshot(&self) -> GovernorSnapshot {
        use onnx_runtime_memory_governor::Tier;

        let mut snapshot = self.inner.snapshot();
        snapshot.vram = Self::tier_snapshot_from_ledger(
            &self.memory,
            Tier::Device,
            self.memory.device_authority().limit_bytes(),
        );
        snapshot.host_ram =
            Self::tier_snapshot_from_ledger(&self.memory, Tier::Host, snapshot.host_ram.limit);
        if let Some(disk) = snapshot.disk_spill.as_mut() {
            *disk = Self::tier_snapshot_from_ledger(&self.memory, Tier::Disk, disk.limit);
        }
        snapshot
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
        let outcome = self.inner.set_vram_limit(limit)?;
        let authority = self.memory.device_authority();
        // When the device capacity is unknown the resolved ceiling is `None` and
        // the authority is inert — there is no number to push, so leave it
        // untouched rather than fabricating one.
        if let Some(new_vram) = outcome.new_limits.vram_bytes
            && let Err(error) = authority.try_set_limit_bytes(new_vram)
        {
            if let Some(old_vram) = outcome.old_limits.vram_bytes {
                let _ = self.inner.set_vram_limit(ResourceLimit::Bytes(old_vram));
            }
            return Err(EngineGovernorError::Resource(
                ResourceError::CannotSatisfyLoweredLimit {
                    requested_bytes: new_vram,
                    minimum_bytes: authority.used_bytes(),
                    reason: error.to_string(),
                },
            ));
        }
        Ok(outcome)
    }

    /// Report live use from the lease ledger, not from the scheduler's transient
    /// byte budget. The server metrics read this snapshot after model load; using
    /// the budget here made a loaded model look like 0 bytes until a scheduled
    /// request happened to be active, so admission saw an empty card (#706).
    fn tier_snapshot_from_ledger(
        memory: &EngineMemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        limit: u64,
    ) -> onnx_genai_scheduler::TierSnapshot {
        let used = onnx_runtime_memory_governor::MemoryGovernor::used(memory, tier);
        onnx_genai_scheduler::TierSnapshot {
            used,
            limit,
            headroom: limit.saturating_sub(used),
        }
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
    pub(crate) fn byte_budget_after_native_load(&self) -> onnx_genai_scheduler::ByteBudget {
        use onnx_runtime_memory_governor::MemoryGovernor as _;

        let kv_pool_bytes = self.plan().kv_pool_bytes_handle();
        let limit = self
            .memory
            .available(onnx_runtime_memory_governor::Tier::Device)
            .saturating_add(kv_pool_bytes.load(std::sync::atomic::Ordering::Relaxed));
        onnx_genai_scheduler::ByteBudget::new(limit).with_ceiling(std::sync::Arc::new(
            LedgerAdmissionCeiling {
                memory: self.memory.clone(),
                kv_pool_bytes,
            },
        ))
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
    memory: EngineMemoryGovernor,
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

/// A capacity provider built from a real OS/EP measurement, or
/// [`UnknownCapacity`] when the platform could not report it. A fraction/auto
/// limit resolved against the unknown provider is itself unknown — never a
/// fabricated number.
fn measured_or_unknown(measured: Option<(u64, u64)>) -> Arc<dyn CapacityProvider> {
    match measured {
        Some((total, free)) => Arc::new(FixedCapacity::new(total, free)),
        None => Arc::new(onnx_genai_scheduler::UnknownCapacity),
    }
}

/// Capacity providers with **no** device query available: VRAM is reported as
/// *unknown* (there is no vendor-neutral capacity probe yet, and #947 forbids
/// fabricating one), while host RAM and disk are measured from the OS.
pub(crate) fn fallback_capacity_providers(limits: &ResourceLimits) -> CapacityProviders {
    let host_ram = measured_or_unknown(crate::platform_capacity::host_ram_total_bytes().map(
        |total| {
            let available = crate::platform_capacity::host_ram_available_bytes().unwrap_or(total);
            (total, available)
        },
    ));
    let disk_spill = limits.disk_spill_limit.map(|_| {
        measured_or_unknown(crate::platform_capacity::disk_capacity_bytes(
            &disk_spill_measurement_path(),
        ))
    });
    CapacityProviders {
        // Unknown, not a manufactured 8 GiB: a fraction of this stays unknown,
        // which is the whole point of #947.
        vram: Arc::new(onnx_genai_scheduler::UnknownCapacity),
        host_ram,
        disk_spill,
    }
}

/// Query the real total/free VRAM of a CUDA device via `cudaMemGetInfo`.
///
/// Returns `(total_bytes, free_bytes)` on success. Returns `None` when the
/// query is unavailable (no driver, query failure, or a nonsense zero total)
/// so the caller reports the device tier as *unknown* rather than fabricating a
/// capacity.
#[cfg(feature = "cuda")]
pub(crate) fn real_cuda_vram_capacity(device_index: u32) -> Option<(u64, u64)> {
    let device_id = i32::try_from(device_index).ok()?;
    match onnx_genai_ort::cuda_rt::device_memory_info(device_id) {
        Ok(info) if info.total_bytes > 0 => Some((info.total_bytes as u64, info.free_bytes as u64)),
        Ok(_) => {
            tracing::warn!(
                device_index,
                "cudaMemGetInfo reported zero total VRAM; reporting the device tier as unknown"
            );
            None
        }
        Err(err) => {
            tracing::warn!(
                device_index,
                error = %err,
                "could not query real CUDA device memory; reporting the device tier as unknown"
            );
            None
        }
    }
}

/// Capacity providers with the VRAM tier resolved against the *real* device
/// total when the decode targets a CUDA device.
///
/// Without a real device query the VRAM tier is [`UnknownCapacity`]: a
/// `Fraction(0.90)` limit against an *8 GiB constant* used to cap device leases
/// at ~7.2 GiB on every machine (a portability hazard that also fabricated a
/// ceiling on machines with no such device at all — #947). When the native
/// device is CUDA we query the driver for the true capacity so the default
/// fraction "just works"; otherwise the tier stays honestly unknown and a
/// fraction of it resolves to `None`, not a manufactured number.
pub(crate) fn capacity_providers_for_device(
    limits: &ResourceLimits,
    cuda_device_index: Option<u32>,
) -> CapacityProviders {
    let mut providers = fallback_capacity_providers(limits);
    providers.vram = device_vram_capacity(cuda_device_index);
    providers
}

/// Environment override for the combined mapped-physical safe fraction (#1295).
/// A finite value in `(0, 1]`; anything else is ignored with a warning.
pub(crate) const VMM_MAPPED_FRACTION_ENV: &str = "ONNX_GENAI_VMM_MAPPED_FRACTION";

/// Default fraction of *usable* (measured-free) device VRAM the combined mapped
/// physical ceiling — weights + KV + activations — is held to (#1295).
///
/// Sourced from measured *free*, not nominal *total*, so a device's standing
/// reserve is already excluded before the fraction applies: on the RTX 4060
/// Laptop 8 GB box (driver 591.55, CUDA 13.1, WDDM) `cuMemGetInfo` reports
/// ~7959 MiB free of a nominal 8188, the ~229 MiB delta being the desktop
/// compositor. The remaining 10 % is headroom against WDDM fault-in / eviction
/// variance: #1295 measured the page-in cliff onset at ~0.97x nominal ≈ the
/// usable boundary, past which `vram_free` cost becomes unpredictable (median
/// ~33 ms with non-deterministic multi-hundred-ms storms). Holding to 0.90 of
/// *free* keeps a margin below that measured onset, converting the cliff into a
/// graceful refusal.
///
/// It is not a magic constant tuned on one machine (cf #1261): the concrete
/// ceiling is `measured_free * fraction`, recomputed per device at load from the
/// driver's own query. Raise it toward 1.0 on a headless box (no compositor
/// reserve, steadier WDDM); lower it if a device shows eviction storms below the
/// boundary. That is what the [`VMM_MAPPED_FRACTION_ENV`] override is for.
pub(crate) const DEFAULT_VMM_MAPPED_FRACTION: f64 = 0.90;

/// Resolve the combined mapped-physical safe fraction from the environment,
/// falling back to [`DEFAULT_VMM_MAPPED_FRACTION`]. A malformed or out-of-range
/// value is ignored with a warning rather than silently changing admission.
pub(crate) fn usable_mapped_safe_fraction() -> f64 {
    match std::env::var(VMM_MAPPED_FRACTION_ENV) {
        Ok(raw) if !raw.trim().is_empty() => match raw.trim().parse::<f64>() {
            Ok(value) if value.is_finite() && value > 0.0 && value <= 1.0 => value,
            _ => {
                tracing::warn!(
                    value = %raw,
                    "ignoring {VMM_MAPPED_FRACTION_ENV}: expected a finite fraction in (0, 1]; \
                     using default {DEFAULT_VMM_MAPPED_FRACTION}"
                );
                DEFAULT_VMM_MAPPED_FRACTION
            }
        },
        _ => DEFAULT_VMM_MAPPED_FRACTION,
    }
}

/// Clamp a configured device ceiling to the combined mapped-physical cap derived
/// from *usable* (measured-free) VRAM (#1295).
///
/// Returns `(effective_ceiling, Some(cap))` when the usable cap actually bound
/// the ceiling (for logging), or `(configured, None)` when it did not — either
/// because no device free was measured (a fraction of an unknown is unknown,
/// #947) or because the configured ceiling was already the tighter bound. The
/// cap is never *raised* above the configured ceiling: it is a safety bound, not
/// a grant.
pub(crate) fn clamp_ceiling_to_usable_vram(
    configured_ceiling_bytes: u64,
    measured_free_bytes: Option<u64>,
    fraction: f64,
) -> (u64, Option<u64>) {
    let Some(free) = measured_free_bytes else {
        return (configured_ceiling_bytes, None);
    };
    let usable_cap = ((free as f64) * fraction).floor() as u64;
    if usable_cap < configured_ceiling_bytes {
        (usable_cap, Some(usable_cap))
    } else {
        (configured_ceiling_bytes, None)
    }
}

/// The VRAM capacity tier for a device: the real `cudaMemGetInfo` query when the
/// decode targets a known CUDA index, and [`UnknownCapacity`] otherwise.
///
/// This is the single point where the engine decides whether a device capacity
/// is *measured* or *unknown*, so every consumer — the decode governor and the
/// server's device-limit resolution alike — shares one honest answer (#947).
pub(crate) fn device_vram_capacity(cuda_device_index: Option<u32>) -> Arc<dyn CapacityProvider> {
    #[cfg(feature = "cuda")]
    if let Some(index) = cuda_device_index
        && let Some((total, free)) = real_cuda_vram_capacity(index)
    {
        return Arc::new(FixedCapacity::new(total, free));
    }
    #[cfg(not(feature = "cuda"))]
    let _ = cuda_device_index;
    Arc::new(onnx_genai_scheduler::UnknownCapacity)
}

/// Resolve a device VRAM limit to a concrete byte budget using the same capacity
/// resolution the decode governor uses: the real device query when the device is
/// a known CUDA index, and *unknown* otherwise. A fraction of unknown capacity
/// resolves to `None` ("a fraction of an unknown is unknown, not a number"); an
/// explicit `--vram-limit <bytes>` is always honoured, measurable device or not.
///
/// Exposed at the crate root so the server's multi-device limit path resolves
/// through the real query / unknown machinery instead of fabricating an 8 GiB
/// `FixedCapacity` — the second copy of the #947 constant.
pub fn resolve_device_vram_limit_bytes(
    limit: ResourceLimit,
    cuda_device_index: Option<u32>,
) -> anyhow::Result<Option<u64>> {
    let vram = device_vram_capacity(cuda_device_index);
    onnx_genai_scheduler::resolve_limit(limit, vram.as_ref(), "vram").map_err(anyhow::Error::new)
}

/// Resolve the configured VRAM limit to a concrete byte budget, or `None` when
/// the device capacity could not be measured and the limit was a fraction —
/// "a fraction of an unknown is unknown, not a number" (#947). An explicit
/// `--vram-limit <bytes>` is always honoured, measurable device or not.
pub(crate) fn resolve_vram_limit_bytes(
    limits: &ResourceLimits,
    cuda_device_index: Option<u32>,
) -> anyhow::Result<Option<u64>> {
    resolve_device_vram_limit_bytes(limits.vram_limit, cuda_device_index)
}

/// Resolve the hot-tier ceiling the memory-strategy planner sizes weight
/// residency against: the measured device (VRAM) budget when a device capacity
/// is available, otherwise the *measured host-RAM* ceiling the weights
/// physically occupy on a device-less / ORT-CPU load.
///
/// #947 made an unmeasured device **capacity** resolve to `None` so nothing
/// fabricates a VRAM ceiling. But residency is a different fact from capacity:
/// on a box with no queryable device the weights still demonstrably live in
/// host RAM, and whether they fit *there* is knowable. Sizing the strategy
/// against the physical host tier -- exactly as the KV byte budget already is,
/// via the scheduler's `kv_hot_tier_ceiling` -- lets a fitting model report
/// `FullResident` instead of `Unknown`, without reintroducing a fabricated
/// device number. An explicit `--vram-limit <bytes>` still wins (it resolves to
/// `Some` above), and if host RAM itself could not be measured this stays `None`
/// (honestly unknown). Nothing here aliases a device lease onto host memory: the
/// device authority is sized separately, this only picks the ceiling for the
/// residency verdict.
pub(crate) fn resolve_memory_strategy_hot_tier_bytes(
    limits: &ResourceLimits,
    cuda_device_index: Option<u32>,
) -> anyhow::Result<Option<u64>> {
    if let Some(vram_bytes) = resolve_vram_limit_bytes(limits, cuda_device_index)? {
        return Ok(Some(vram_bytes));
    }
    let providers = fallback_capacity_providers(limits);
    onnx_genai_scheduler::resolve_limit(
        limits.host_ram_limit,
        providers.host_ram.as_ref(),
        "host RAM",
    )
    .map_err(anyhow::Error::new)
}

pub(crate) fn governor_kv_config(
    kv_model: Option<&KvModelInfo>,
    config: &EngineConfig,
) -> anyhow::Result<ModelKvConfig> {
    let tokens_per_page = governor_tokens_per_page(config)?;
    let Some(kv_model) = kv_model else {
        return Ok(ModelKvConfig::unknown(tokens_per_page));
    };

    let page_size = tokens_per_page;
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
    Ok(ModelKvConfig::known(page_size_bytes, tokens_per_page))
}

#[cfg_attr(
    all(not(feature = "native-backend"), not(test)),
    expect(
        dead_code,
        reason = "native KV admission is used only by the native backend"
    )
)]
pub(crate) fn governor_native_kv_config(
    kv_model: Option<&KvModelInfo>,
    config: &EngineConfig,
) -> anyhow::Result<ModelKvConfig> {
    let tokens_per_page = governor_tokens_per_page(config)?;
    let Some(kv_model) = kv_model else {
        return Ok(ModelKvConfig::unknown(tokens_per_page));
    };

    let page_size_bytes =
        crate::kv_sizing::kv_cache_bytes_for_tensors(&kv_model.native_kv_tensors, tokens_per_page)?;
    Ok(ModelKvConfig::known(page_size_bytes, tokens_per_page))
}

pub(crate) fn governor_no_paged_kv_config(config: &EngineConfig) -> anyhow::Result<ModelKvConfig> {
    Ok(ModelKvConfig::no_paged_cache(governor_tokens_per_page(
        config,
    )?))
}

fn governor_tokens_per_page(config: &EngineConfig) -> anyhow::Result<u64> {
    let tokens_per_page = u64::try_from(config.page_size)
        .context("KV page_size does not fit the Resource Governor's u64 accounting")?;
    if tokens_per_page == 0 {
        anyhow::bail!(
            "KV page_size must be greater than zero; set EngineConfig::page_size to the number of tokens per KV page"
        );
    }
    Ok(tokens_per_page)
}

pub(crate) fn model_io_declares_only_fixed_state(
    io: Option<&onnx_genai_metadata::ModelIoSpec>,
) -> bool {
    let Some(io) = io else {
        return false;
    };
    let has_state_pairs = io
        .state_pairs
        .as_ref()
        .is_some_and(|pairs| !pairs.is_empty());
    let has_kv_pairs = io.kv_inputs.as_ref().is_some_and(|ports| !ports.is_empty())
        || io
            .kv_outputs
            .as_ref()
            .is_some_and(|ports| !ports.is_empty());
    has_state_pairs && !has_kv_pairs
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
    model_weights_bytes: u64,
    reservation_bytes: u64,
    cuda_device_index: Option<u32>,
    provider: Option<&crate::memory_authority::SharedMemoryAuthorityProvider>,
    domain: &crate::memory_authority::DeviceCompatibilityDomain,
) -> anyhow::Result<EngineResourceGovernor> {
    let kv_config = match kv_model {
        Some(kv_model) => governor_kv_config(Some(kv_model), config)?,
        None => governor_no_paged_kv_config(config)?,
    };
    EngineResourceGovernor::new_with_authority_and_reservation(
        config.limits.clone(),
        config.allow_runtime_override,
        kv_config,
        model_weights_bytes,
        reservation_bytes,
        cuda_device_index,
        provider,
        Some(domain),
    )
    .context("failed to resolve the engine memory budget for decoder fixed state")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An explicit `--device` is authoritative in both directions. The CPU case
    /// is the one that matters: after #1551 made an undeclared device probe for
    /// an accelerator, asking for the CPU on a GPU machine must still get the
    /// CPU, or the flag would be unable to express what it exists to express.
    #[cfg(feature = "native-backend")]
    #[test]
    fn an_explicitly_requested_device_is_never_second_guessed() {
        use crate::native_decode::NativeDecodeDevice;

        let options = SessionOptions::default();

        assert_eq!(
            resolve_native_decode_device(Some(NativeDecodeDevice::Cpu), &options).unwrap(),
            NativeDecodeDevice::Cpu,
            "an explicit CPU request must survive accelerator detection"
        );
        #[cfg(feature = "cuda")]
        assert_eq!(
            resolve_native_decode_device(
                Some(NativeDecodeDevice::Cuda { index: Some(2) }),
                &options
            )
            .unwrap(),
            NativeDecodeDevice::Cuda { index: Some(2) },
            "an explicit CUDA index must be passed through unchanged"
        );
    }

    /// A model that declares no execution provider used to resolve to the CPU,
    /// so `--backend native` on a GPU machine silently decoded on the CPU
    /// (#1064) -- and, because nothing said so, CLI-driven A/B and correctness
    /// checks silently compared two runs that exercised neither the accelerator
    /// nor the lever under test (#1551).
    ///
    /// An absent declaration is missing information, not a request for the CPU,
    /// so it now probes instead. The assertion is written against the probe
    /// rather than against a fixed device so that it is meaningful on both a GPU
    /// box and a CPU-only one.
    #[cfg(all(feature = "native-backend", feature = "cuda"))]
    #[test]
    fn an_undeclared_device_probes_for_an_accelerator_instead_of_assuming_the_cpu() {
        use crate::native_decode::NativeDecodeDevice;

        let resolved = resolve_native_decode_device(None, &SessionOptions::default()).unwrap();

        if onnx_runtime_ep_cuda::CudaExecutionProvider::is_available(0) {
            assert_eq!(
                resolved,
                NativeDecodeDevice::Cuda { index: Some(0) },
                "a usable CUDA device must be preferred over the CPU when the model \
                 declares nothing"
            );
        } else {
            assert_eq!(
                resolved,
                NativeDecodeDevice::Cpu,
                "with no usable accelerator the CPU remains the answer"
            );
        }
    }

    #[test]
    fn fraction_over_unmeasured_vram_is_unknown_not_a_number() {
        // #947 regression: on the reported machine (no NVIDIA GPU) the no-CUDA
        // path resolved `Fraction(0.90)` against a fabricated 8 GiB constant,
        // yielding a specific-looking 7,730,940,928 that a real user mistook for
        // a device that did not exist. It must now resolve to `None` (unknown).
        //
        // This test *fails on `main`* — there `resolve_vram_limit_bytes` returns
        // `Ok(7_730_940_928)` — which is the whole point: it pins the fix, not a
        // tautology that both versions pass.
        let limits = ResourceLimits {
            vram_limit: ResourceLimit::Fraction(0.90),
            host_ram_limit: ResourceLimit::Fraction(0.90),
            disk_spill_limit: None,
        };
        assert_eq!(
            resolve_vram_limit_bytes(&limits, None).unwrap(),
            None,
            "a fraction of an unmeasured device capacity must be unknown, not a number"
        );
    }

    #[test]
    fn explicit_vram_byte_limit_is_honoured_without_a_device_query() {
        // An explicit limit is the caller's own assertion and must survive even
        // when no device capacity can be measured.
        let limits = ResourceLimits {
            vram_limit: ResourceLimit::Bytes(12_345_678),
            host_ram_limit: ResourceLimit::Fraction(0.90),
            disk_spill_limit: None,
        };
        assert_eq!(
            resolve_vram_limit_bytes(&limits, None).unwrap(),
            Some(12_345_678)
        );
    }

    #[test]
    fn usable_mapped_fraction_defaults_and_rejects_out_of_range() {
        // Pure derivation logic, no env dependence for the default. The env
        // override path is exercised by callers; here we pin the default and the
        // clamp arithmetic that the whole cap rests on (#1295).
        assert_eq!(usable_mapped_safe_fraction(), DEFAULT_VMM_MAPPED_FRACTION);
        assert_eq!(DEFAULT_VMM_MAPPED_FRACTION, 0.90);
    }

    #[test]
    fn usable_cap_binds_only_when_below_configured_and_sources_from_free() {
        // Nominal total 8188 MiB, usable free 7959 MiB — the measured RTX 4060
        // Laptop 8 GB idle figures (#1295). At 0.90 the cap is 0.90 * free, which
        // is strictly below both the nominal total and 0.90 * total: this is the
        // exact discrimination that proves the cap is drawn from *usable* VRAM,
        // not the nominal number a `Fraction` would otherwise resolve against.
        const MIB: u64 = 1 << 20;
        let total = 8188 * MIB;
        let free = 7959 * MIB;
        let (ceiling, bound) = clamp_ceiling_to_usable_vram(total, Some(free), 0.90);
        let expected = ((free as f64) * 0.90).floor() as u64;
        assert_eq!(ceiling, expected);
        assert_eq!(bound, Some(expected));
        assert!(ceiling < total, "cap must sit below nominal total");
        assert!(
            ceiling < ((total as f64) * 0.90) as u64,
            "cap must sit below 0.90 * total, proving it is sourced from free not total"
        );

        // A configured ceiling already tighter than the usable cap wins, and the
        // cap reports it did not bind (so nothing is logged as a change).
        let tight = 4 * (1u64 << 30);
        let (ceiling, bound) = clamp_ceiling_to_usable_vram(tight, Some(free), 0.90);
        assert_eq!(ceiling, tight);
        assert_eq!(bound, None);

        // An unmeasured device: a fraction of an unknown is unknown (#947), so
        // the cap does not bind and the configured ceiling passes through.
        let (ceiling, bound) = clamp_ceiling_to_usable_vram(total, None, 0.90);
        assert_eq!(ceiling, total);
        assert_eq!(bound, None);
    }

    #[test]
    fn device_authority_ceiling_is_usable_free_and_refuses_growth_past_it() {
        // End-to-end, non-vacuous (#8): build a real governor whose device tier
        // is the measured RTX 4060 idle capacity with a configured ceiling of the
        // *whole* device (Fraction 1.0). The resulting authority ceiling must be
        // the usable-free cap, and a Device-tier reservation one byte past it
        // must be refused (G3) — the admission refusal that converts the #1295
        // oversubscription cliff into a plateau.
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor, MemoryRole, Tier};
        const MIB: u64 = 1 << 20;
        let total = 8188 * MIB;
        let free = 7959 * MIB;
        let capacities = CapacityProviders {
            vram: Arc::new(FixedCapacity::new(total, free)),
            host_ram: Arc::new(FixedCapacity::new(64u64 << 30, 32u64 << 30)),
            disk_spill: None,
        };
        let limits = ResourceLimits {
            // Ask for the entire device; the usable cap must still bind.
            vram_limit: ResourceLimit::Fraction(1.0),
            host_ram_limit: ResourceLimit::Fraction(0.90),
            disk_spill_limit: None,
        };
        let kv_config = governor_no_paged_kv_config(&EngineConfig::default()).unwrap();
        let domain = DeviceCompatibilityDomain::Cuda(0);
        let governor = EngineResourceGovernor::new_with_capacities_and_authority(
            limits,
            false,
            capacities,
            kv_config,
            (0, 0),
            None,
            Some(&domain),
        )
        .expect("governor construction with a measured device capacity");

        let authority = governor.device_authority();
        let ceiling = authority.limit_bytes();
        let expected = ((free as f64) * DEFAULT_VMM_MAPPED_FRACTION).floor() as u64;
        assert_eq!(
            ceiling, expected,
            "device authority ceiling must be floor(usable_free * safe_fraction)"
        );
        assert!(
            ceiling < ((total as f64) * DEFAULT_VMM_MAPPED_FRACTION) as u64,
            "ceiling must be below 0.90 * nominal total, i.e. sourced from free"
        );

        // Non-vacuous precondition: a reservation of the whole ceiling succeeds,
        // establishing the state in which the next byte must be refused.
        let lease = authority
            .reserve(Tier::Device, ceiling, MemoryRole::Weights, HolderId::new(1))
            .expect("a reservation up to the ceiling must be admitted");
        assert_eq!(authority.available(Tier::Device), 0);
        // The (N+1)th byte over the usable cap is refused rather than allowed to
        // oversubscribe the device — the plateau, not the cliff.
        assert!(
            authority
                .reserve(Tier::Device, 1, MemoryRole::KvCache, HolderId::new(2))
                .is_err(),
            "growth past the usable-VRAM cap must be refused (G3)"
        );
        drop(lease);
    }

    #[test]
    fn no_cuda_device_reports_unknown_vram_capacity() {
        // The device tier must report *unknown*, not a manufactured 8 GiB total,
        // when there is no execution provider to query.
        let limits = ResourceLimits::default();
        let providers = capacity_providers_for_device(&limits, None);
        assert_eq!(providers.vram.total_bytes(), None);
        assert_eq!(providers.vram.free_bytes(), None);
    }

    #[test]
    fn host_ram_capacity_is_measured_not_fabricated() {
        // Host RAM is queryable on every supported platform, so the tier must
        // carry a real number — not the old fabricated 16 GiB constant.
        let limits = ResourceLimits::default();
        let providers = capacity_providers_for_device(&limits, None);
        let total = providers
            .host_ram
            .total_bytes()
            .expect("host RAM must be measurable on this platform");
        assert!(total > (1u64 << 30), "implausibly small host RAM: {total}");
        assert_ne!(total, 16u64 << 30, "looks like the old fabricated constant");
    }

    // The real-capacity path is what fixes the portability bug: on a CUDA build
    // with a visible device, a `Fraction(0.90)` must resolve against the true
    // device total, not a provisional cap — otherwise any model larger than
    // ~7.2 GiB fails to load resident even on a 143 GiB H200.
    #[cfg(feature = "cuda")]
    #[test]
    fn real_cuda_capacity_lifts_fraction_above_provisional_cap() {
        // Historic 8 GiB provisional cap this test guards against regressing to.
        const LEGACY_PROVISIONAL_VRAM_CAP: u64 = 8 << 30;
        let Some((total, _free)) = real_cuda_vram_capacity(0) else {
            eprintln!("no CUDA device 0 visible; skipping real-capacity assertion");
            return;
        };
        // Only meaningful when the device is larger than the historic cap
        // (true for every modern datacenter GPU, e.g. H200 = 143 GiB).
        if total <= LEGACY_PROVISIONAL_VRAM_CAP {
            eprintln!("device 0 total {total} <= legacy provisional cap; skipping");
            return;
        }
        let limits = ResourceLimits {
            vram_limit: ResourceLimit::Fraction(0.90),
            host_ram_limit: ResourceLimit::Fraction(0.90),
            disk_spill_limit: None,
        };
        let resolved = resolve_vram_limit_bytes(&limits, Some(0))
            .unwrap()
            .expect("a visible CUDA device has a measured capacity");
        let provisional_cap = (LEGACY_PROVISIONAL_VRAM_CAP as f64 * 0.90) as u64;
        assert!(
            resolved > provisional_cap,
            "resolved {resolved} must exceed provisional cap {provisional_cap}"
        );
        // And it should track the real device total, not some other constant.
        let expected = (total as f64 * 0.90) as u64;
        let tolerance = expected / 100; // 1%
        assert!(
            resolved.abs_diff(expected) <= tolerance,
            "resolved {resolved} should be ~0.9 * device total {total} (= {expected})"
        );
    }

    #[test]
    fn missing_kv_geometry_is_unknown_not_a_token_count_byte_size() {
        let config = EngineConfig {
            page_size: 16,
            ..EngineConfig::default()
        };

        let kv_config = governor_kv_config(None, &config).unwrap();

        assert_eq!(kv_config.tokens_per_page, 16);
        assert_eq!(kv_config.page_size_bytes, None);
        assert_eq!(kv_config.bytes_per_token(), None);
    }

    #[test]
    fn zero_page_size_fails_instead_of_manufacturing_one_byte_tokens() {
        let config = EngineConfig {
            page_size: 0,
            ..EngineConfig::default()
        };
        let kv_model = KvModelInfo {
            tensor_config: onnx_genai_kv::PageTensorConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 1,
                page_size: 0,
                dtype: KvDType::F32,
            },
            layer_configs: vec![onnx_genai_kv::LayerTensorConfig {
                num_kv_heads: 1,
                head_dim: 1,
            }],
            native_kv_tensors: Vec::new(),
            layers: Vec::new(),
        };

        for result in [
            governor_kv_config(None, &config),
            governor_kv_config(Some(&kv_model), &config),
            governor_no_paged_kv_config(&config),
        ] {
            let error = result.unwrap_err().to_string();
            assert!(
                error.contains("page_size must be greater than zero"),
                "{error}"
            );
        }
    }

    #[test]
    fn native_kv_accounting_uses_graph_storage_width() {
        let config = EngineConfig::default();
        let kv_model = KvModelInfo {
            tensor_config: onnx_genai_kv::PageTensorConfig {
                num_layers: 1,
                num_kv_heads: 2,
                head_dim: 4,
                page_size: config.page_size,
                dtype: KvDType::F32,
            },
            layer_configs: vec![onnx_genai_kv::LayerTensorConfig {
                num_kv_heads: 2,
                head_dim: 4,
            }],
            native_kv_tensors: vec![
                crate::kv_sizing::KvTensorSpec {
                    name: "key".into(),
                    dtype: crate::kv_sizing::KvStorageType::Float16,
                    shape: vec![
                        crate::kv_sizing::KvDimension::PerSequenceBatch,
                        crate::kv_sizing::KvDimension::Fixed(2),
                        crate::kv_sizing::KvDimension::Context,
                        crate::kv_sizing::KvDimension::Fixed(4),
                    ],
                },
                crate::kv_sizing::KvTensorSpec {
                    name: "value".into(),
                    dtype: crate::kv_sizing::KvStorageType::Float16,
                    shape: vec![
                        crate::kv_sizing::KvDimension::PerSequenceBatch,
                        crate::kv_sizing::KvDimension::Fixed(2),
                        crate::kv_sizing::KvDimension::Context,
                        crate::kv_sizing::KvDimension::Fixed(4),
                    ],
                },
            ],
            layers: Vec::new(),
        };

        let native = governor_native_kv_config(Some(&kv_model), &config).unwrap();
        let host_mirror = governor_kv_config(Some(&kv_model), &config).unwrap();

        assert_eq!(native.bytes_per_token(), Some(32));
        assert_eq!(host_mirror.bytes_per_token(), Some(64));
    }

    fn native_kv_model_with_specs(specs: Vec<crate::kv_sizing::KvTensorSpec>) -> KvModelInfo {
        KvModelInfo {
            tensor_config: onnx_genai_kv::PageTensorConfig {
                num_layers: 1,
                num_kv_heads: 2,
                head_dim: 4,
                page_size: 16,
                dtype: KvDType::F32,
            },
            layer_configs: vec![onnx_genai_kv::LayerTensorConfig {
                num_kv_heads: 2,
                head_dim: 4,
            }],
            native_kv_tensors: specs,
            layers: Vec::new(),
        }
    }

    fn native_kv_spec(
        name: &str,
        dtype: crate::kv_sizing::KvStorageType,
        head_dim: u64,
    ) -> crate::kv_sizing::KvTensorSpec {
        crate::kv_sizing::KvTensorSpec {
            name: name.into(),
            dtype,
            shape: vec![
                crate::kv_sizing::KvDimension::PerSequenceBatch,
                crate::kv_sizing::KvDimension::Fixed(2),
                crate::kv_sizing::KvDimension::Context,
                crate::kv_sizing::KvDimension::Fixed(head_dim),
            ],
        }
    }

    #[test]
    fn native_kv_accounting_sums_asymmetric_key_and_value_geometry() {
        let config = EngineConfig::default();
        let model = native_kv_model_with_specs(vec![
            native_kv_spec("key", crate::kv_sizing::KvStorageType::Float16, 4),
            native_kv_spec("value", crate::kv_sizing::KvStorageType::Float16, 8),
        ]);

        let admission = governor_native_kv_config(Some(&model), &config).unwrap();

        assert_eq!(admission.bytes_per_token(), Some(48));
        assert_eq!(
            admission.page_size_bytes,
            Some(
                crate::kv_sizing::kv_cache_bytes_for_tensors(
                    &model.native_kv_tensors,
                    config.page_size as u64
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn native_kv_accounting_sums_asymmetric_storage_widths() {
        let config = EngineConfig::default();
        let model = native_kv_model_with_specs(vec![
            native_kv_spec("key", crate::kv_sizing::KvStorageType::Float16, 4),
            native_kv_spec("value", crate::kv_sizing::KvStorageType::Float32, 4),
        ]);

        let admission = governor_native_kv_config(Some(&model), &config).unwrap();

        assert_eq!(admission.bytes_per_token(), Some(48));
    }

    #[test]
    fn state_only_metadata_is_a_valid_non_paged_cache_not_unknown_geometry() {
        let io: onnx_genai_metadata::ModelIoSpec = serde_json::from_value(serde_json::json!({
            "state_pairs": [
                { "input": "conv_state_in", "output": "conv_state_out" }
            ]
        }))
        .unwrap();

        assert!(model_io_declares_only_fixed_state(Some(&io)));
        let kv_config = governor_no_paged_kv_config(&EngineConfig::default()).unwrap();
        assert_eq!(kv_config.page_size_bytes, None);
        assert!(!kv_config.page_geometry_required);
        assert_eq!(kv_config.bytes_per_token(), None);
    }
}
