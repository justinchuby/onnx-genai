//! The [`CudaExecutionProvider`]: a GPU execution provider backed by cudarc +
//! cuBLASLt (`docs/architecture/ORT2.md` §15). Phase 2a wires standard GEMM (`MatMul`) only;
//! everything else returns an actionable "not implemented in CUDA EP Phase 2a"
//! error rather than silently falling back or panicking.
//!
//! # Memory & safety model
//!
//! Mirrors the ep-api safety contract used by the CPU EP, but the buffers live
//! in **device** memory:
//!
//! 1. **Owner-frees** — every [`allocate`](CudaExecutionProvider::allocate)
//!    (`cuMemAlloc`) pairs with exactly one
//!    [`deallocate`](CudaExecutionProvider::deallocate) (`cuMemFree`).
//!    [`onnx_runtime_ep_api::DeviceBuffer`] has no `Drop`, so a dropped handle
//!    leaks but never double-frees.
//! 2. **No cross-EP free** — `deallocate`/`copy` assert the buffer's device
//!    matches this EP's `CUDA:ordinal`.
//! 3. **Bounds** — `copy` rejects a `size` larger than either endpoint.
//! 4. **Opaque device pointers** — a CUDA device pointer is *not* host-
//!    dereferenceable; it only travels between `allocate`, `copy`, and kernels,
//!    exactly as [`onnx_runtime_ep_api::DeviceBuffer`] documents for CUDA.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::{
    Cost, DeviceBuffer, EpConfig, EpError, ExecutionProvider, ExecutionProviderCapabilities, Fence,
    HostToDeviceCopier, Kernel, KernelMatch, LazyWeight, OpRegistry, PagedWeight, Result, deny,
    structural_input_bytes,
};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};
use onnx_runtime_memory_governor::DeviceAllocator;

use crate::kernels::build_cuda_registry_with_metrics;
use crate::kernels::csa_checkpoint::CsaMetrics;
use crate::optimizer::cuda_optimization_passes;
use crate::runtime::{CudaRuntime, cuptr};
use crate::weight_paging::{CudaWeightResidency, DeviceOffloadPolicy};

/// A minimal synchronous host→device uploader backed by the shared CUDA
/// runtime. Handed to the plugin's fused executor via
/// [`ExecutionProvider::host_to_device_copier`] so it can stage host-resident
/// boundary inputs into device scratch on an interspersed CPU→GPU partition
/// (#982). Holding an `Arc<CudaRuntime>` — not the EP — keeps EP teardown
/// semantics unchanged: the runtime is already kept alive by every live kernel.
struct CudaHostToDeviceCopier {
    runtime: Arc<CudaRuntime>,
}

impl HostToDeviceCopier for CudaHostToDeviceCopier {
    unsafe fn copy_host_to_device(&self, src: &[u8], dst: *mut std::ffi::c_void) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: `dst` is a live device allocation of at least `src.len()`
        // bytes (ORT device scratch on this runtime's device), per the trait
        // contract. `htod` is synchronous, so the bytes are resident on return.
        unsafe { self.runtime.htod(src, cuptr(dst)) }
    }
}

/// Default VRAM budget for the device weight-offload residency cache when
/// `ONNX_GENAI_WEIGHT_OFFLOAD` is enabled without an explicit
/// `ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES` override (4 GiB).
pub const DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES: u64 = 4 << 30;

/// Default retained-byte bound for the standalone (plugin, no-governor) VMM
/// arena's physical-handle pool, used when
/// `ONNX_GENAI_CUDA_PHYSICAL_HANDLE_POOL_BYTES` is not set (#956).
///
/// The plugin path has no memory governor to supply a pool bound, so without a
/// default the standalone arena would unmap-and-release every scratch
/// allocation's physical granules on free and re-create them on the next
/// same-size request — trading `cuMemAlloc` per dispatch for
/// `cuMemCreate`/`cuMemRelease` per dispatch. Retaining 256 MiB of unmapped
/// granules (matching the governor path's default) lets repeated same-size
/// scratch requests reuse committed memory. It bounds retained-but-unmapped
/// physical memory, so it cannot leak without bound.
pub const DEFAULT_STANDALONE_PHYSICAL_POOL_BYTES: usize = 256 << 20;

fn dynamic_lending_enabled() -> bool {
    dynamic_lending_enabled_for(
        std::env::var("ONNX_GENAI_DYNAMIC_KV_WEIGHT_LENDING")
            .ok()
            .as_deref(),
    )
}

/// Whether dynamic KV/weight mapped-allowance lending is active for this
/// process (default on; opt out with `ONNX_GENAI_DYNAMIC_KV_WEIGHT_LENDING=0`).
///
/// This is the exact predicate that, together with managed no-spill and an
/// on-demand-committing arena, decides in [`CudaExecutionProvider::adopt_memory_governor`]
/// whether the weight-residency cache is registered as a reclaimable mapped
/// holder. The engine loader queries it to gate elastic weight-budget sizing:
/// lending the full-context KV reservation to weights is only safe when that
/// reclaim path exists to give the space back as KV grows (issue #857).
pub fn dynamic_kv_weight_lending_enabled() -> bool {
    dynamic_lending_enabled()
}

fn mapped_attribution_role(
    _role: onnx_runtime_memory_governor::MemoryRole,
) -> onnx_runtime_memory_governor::MemoryRole {
    // This provider has one suballocating VMM arena. KV and both workspace
    // lifetimes can touch the same physical granule, so mapped attribution is
    // one arena zone even though their content leases/metrics remain distinct.
    onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false }
}

fn dynamic_lending_enabled_for(value: Option<&str>) -> bool {
    !value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

fn auto_dynamic_lending_for(
    governor_present: bool,
    policy: &DeviceOffloadPolicy,
    lending_enabled: bool,
) -> bool {
    governor_present && policy.managed_no_spill && lending_enabled
}

#[derive(Debug)]
enum VmmInitialization<T> {
    Installed(T),
    CompatibilityFallback(String),
}

fn resolve_vmm_initialization<T, E: std::fmt::Display>(
    managed_no_spill: bool,
    requested_limit: Option<u64>,
    result: std::result::Result<T, E>,
) -> Result<VmmInitialization<T>> {
    match result {
        Ok(arena) => Ok(VmmInitialization::Installed(arena)),
        Err(error) if managed_no_spill => Err(EpError::KernelFailed(format!(
            "managed no-spill CUDA initialization failed before model allocation for requested \
             VRAM limit {}: could not build the required VMM arena and physical-handle pool: \
             {error}",
            requested_limit
                .map(|bytes| format!("{bytes} bytes"))
                .unwrap_or_else(|| "unknown".to_string())
        ))),
        Err(error) => Ok(VmmInitialization::CompatibilityFallback(error.to_string())),
    }
}

/// CUDA execution provider (Phase 2a: cudarc + cuBLASLt GEMM).
///
/// Unlike the always-available CPU EP, this provider needs a real device, so
/// [`CudaExecutionProvider::new`] is **fallible** — it returns an error when no
/// CUDA device is present or the driver / cuBLASLt cannot be loaded. Callers on
/// a machine without a GPU should treat that error as "CUDA EP unavailable".
pub struct CudaExecutionProvider {
    device: DeviceId,
    runtime: Arc<CudaRuntime>,
    /// Where this EP's device buffers come from.
    ///
    /// The same DeviceAllocator contract the CPU EP and the ONNX Runtime
    /// allocator use, so an allocator a caller writes serves every backend.
    /// Defaults to CudaDeviceAllocator, which is the cuMemAlloc call this
    /// EP used to make directly.
    memory: Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>,
    governor: Option<Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>>,
    /// Installed inside the constructor (`new_with_policy_and_governor`) when
    /// VMM is enabled, and read in place of `memory` from then on.
    ///
    /// `adopt_memory_governor` does **not** install this cell — by the time it
    /// runs, a governor has become available, but the arena (if any) already
    /// exists from construction; adoption only moves the arena's own ledger
    /// onto the real governor (see [`CudaExecutionProvider::adopt_memory_governor`]).
    /// So this is set at most once, and always before the constructed provider
    /// is returned to a caller — never after allocations may have happened.
    /// That is what makes `with_memory`'s rejection of "inject an allocator
    /// once VMM is already selected" complete rather than racy: nothing after
    /// construction can populate this cell out from under it.
    ///
    /// A `OnceLock` rather than a lock because this is set once, before any
    /// allocation, and read on every one: `get` is a relaxed atomic load, so
    /// the hot path pays nothing for the option to swap.
    vmm: std::sync::OnceLock<Arc<crate::vmm_allocator::CudaVmmAllocator>>,
    /// Allocations and frees this EP made through `memory`.
    ///
    /// Kept here rather than asked of the allocator, because the allocator is
    /// the part a caller replaces and counting is not something the shared
    /// contract should require of them. The EP knows every call it makes.
    ///
    /// These exist because roughly twenty-five tests assert that a warmed,
    /// capture-safe path performs no further allocations, and they assert it by
    /// reading a counter. Before the allocator seam those went through
    /// `CudaRuntime::alloc_raw`, which counted them; afterwards they did not,
    /// and every one of those assertions silently became "0 == 0".
    ep_allocations: Arc<AtomicU64>,
    ep_frees: Arc<AtomicU64>,
    initialized: bool,
    registry: OpRegistry,
    csa_metrics: Arc<CsaMetrics>,
    /// Device weight-offload policy resolved from the environment. When enabled,
    /// the EP advertises the `nxrt` weight-paging capability and pages lazy
    /// weights host↔device on demand during dispatch.
    offload_policy: DeviceOffloadPolicy,
    /// LRU device residency cache. `Some` iff `offload_policy.enabled`.
    residency: Option<Arc<CudaWeightResidency>>,
    mapped_reclaim_registration:
        std::sync::OnceLock<onnx_runtime_memory_governor::MappedHolderRegistration>,
    mapped_requesters: Mutex<
        HashMap<
            onnx_runtime_memory_governor::MemoryRole,
            onnx_runtime_memory_governor::MappedAllowance,
        >,
    >,
}

impl std::fmt::Debug for CudaExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaExecutionProvider")
            .field("device", &self.device)
            .field("initialized", &self.initialized)
            .field("registered_ops", &self.registry.len())
            .finish()
    }
}

impl CudaExecutionProvider {
    /// Construct a CUDA EP bound to `CUDA:ordinal` with the Phase-2a kernels
    /// registered. Fails if the device or CUDA libraries are unavailable.
    pub fn new(ordinal: u32) -> Result<Self> {
        Self::new_with_offload_policy(ordinal, DeviceOffloadPolicy::from_env())
    }

    /// Construct a CUDA EP with an already-resolved weight-offload policy.
    ///
    /// The engine uses this when `--vram-limit` is the authority that enables
    /// offload. Reading only the process environment here would recreate #712:
    /// the limit would be parsed by the CLI while weights still loaded under an
    /// unrelated residency policy.
    pub fn new_with_offload_policy(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
    ) -> Result<Self> {
        Self::new_with_policy_and_governor(ordinal, offload_policy, None)
    }

    /// Construct a CUDA EP with the device authority available before the
    /// allocator reserves or commits memory.
    pub fn new_with_offload_policy_and_governor(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    ) -> Result<Self> {
        Self::new_with_policy_and_governor(ordinal, offload_policy, Some(governor))
    }

    fn new_with_policy_and_governor(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Option<Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>>,
    ) -> Result<Self> {
        let runtime = Arc::new(CudaRuntime::new(ordinal)?);
        let csa_metrics = Arc::new(CsaMetrics::default());
        let registry = build_cuda_registry_with_metrics(runtime.clone(), csa_metrics.clone());
        let auto_dynamic_lending = auto_dynamic_lending_for(
            governor.is_some(),
            &offload_policy,
            dynamic_lending_enabled(),
        );
        let residency = offload_policy.enabled.then(|| {
            let budget = offload_policy
                .device_budget_bytes
                .unwrap_or(DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES);
            Arc::new(
                CudaWeightResidency::new(runtime.clone(), budget)
                    .with_async_pagein(offload_policy.async_pagein)
                    .with_scan_resistant_dense(offload_policy.scan_resistant_dense)
                    .with_byte_aware_residency(offload_policy.byte_aware_residency)
                    .with_evict_order_probe(offload_policy.evict_order_probe)
                    .with_zero_copy_hybrid(offload_policy.zero_copy_hybrid),
            )
        });
        let provider = Self {
            device: DeviceId::cuda(ordinal),
            memory: Arc::new(crate::device_allocator::CudaDeviceAllocator::new(
                runtime.cuda_context(),
            )),
            governor: governor.clone(),
            // Built here rather than at governor adoption, because on the
            // native path the session allocates every tensor it will use while
            // loading -- which is before any governor reaches this provider.
            // An arena installed at adoption is installed at the one moment
            // after which nothing will ask it for memory (#659).
            //
            // Address space is free, so the reservation is generous rather than
            // fitted: 64 GiB comfortably exceeds any single accelerator we
            // target, and running out of *reservation* is a hard failure while
            // leaving it unmapped costs nothing.
            vmm: {
                let cell = std::sync::OnceLock::new();
                if crate::vmm_allocator::vmm_enabled() || auto_dynamic_lending {
                    const RESERVATION_BYTES: usize = 64 << 30;
                    let synchronization_runtime = Arc::clone(&runtime);
                    let teardown_synchronizer: crate::virtual_memory::TeardownSynchronizer =
                        Arc::new(move || {
                            synchronization_runtime
                                .synchronize()
                                .map_err(|error| error.to_string())?;
                            synchronization_runtime
                                .copy_stream()
                                .synchronize()
                                .map_err(|error| error.to_string())
                        });
                    let arena = match governor.as_deref() {
                        Some(governor) => {
                            crate::vmm_allocator::CudaVmmAllocator::new_with_teardown_synchronizer(
                            runtime.cuda_context(),
                            onnx_runtime_memory_governor::DeviceKey::device(ordinal),
                            ordinal as i32,
                            RESERVATION_BYTES,
                            governor,
                            onnx_runtime_memory_governor::HolderId::new(64),
                            onnx_runtime_memory_governor::MemoryRole::Workspace {
                                step_scoped: false,
                            },
                            Arc::clone(&teardown_synchronizer),
                            auto_dynamic_lending.then_some(256usize << 20),
                        )
                        }
                        None => crate::vmm_allocator::CudaVmmAllocator::standalone_with_teardown_synchronizer(
                            runtime.cuda_context(),
                            onnx_runtime_memory_governor::DeviceKey::device(ordinal),
                            ordinal as i32,
                            RESERVATION_BYTES,
                            onnx_runtime_memory_governor::HolderId::new(64),
                            onnx_runtime_memory_governor::MemoryRole::Workspace {
                                step_scoped: false,
                            },
                            teardown_synchronizer,
                            // Standalone (plugin, no-governor) VMM path: retain a
                            // pool of physical granules by default so repeated
                            // same-size ORT scratch requests reuse committed
                            // memory instead of a per-dispatch
                            // cuMemCreate/cuMemRelease churn (#956). An explicit
                            // ONNX_GENAI_CUDA_PHYSICAL_HANDLE_POOL_BYTES still
                            // overrides this. This mirrors the governor path,
                            // which already passes a default pool bound.
                            Some(DEFAULT_STANDALONE_PHYSICAL_POOL_BYTES),
                        ),
                    };
                    match resolve_vmm_initialization(
                        auto_dynamic_lending,
                        offload_policy.managed_limit_bytes,
                        arena,
                    )? {
                        VmmInitialization::Installed(arena) => {
                            eprintln!(
                                "cuda_ep: device allocations go through a VMM arena over \
                                 {RESERVATION_BYTES} bytes of reserved address space; physical \
                                 granules are mapped on demand; strategy={}{}",
                                if auto_dynamic_lending {
                                    "vram-limit dynamic KV/weight lending"
                                } else {
                                    "explicit CUDA VMM"
                                },
                                if auto_dynamic_lending {
                                    " with a retained physical-handle pool"
                                } else {
                                    ""
                                }
                            );
                            let _ = cell.set(Arc::new(arena));
                        }
                        VmmInitialization::CompatibilityFallback(error) => eprintln!(
                            "cuda_ep: WARNING: could not build the VMM arena, falling back to \
                             cuMemAlloc; device allocations will not be charged to the ledger: \
                             {error}"
                        ),
                    }
                }
                cell
            },
            ep_allocations: Arc::new(AtomicU64::new(0)),
            ep_frees: Arc::new(AtomicU64::new(0)),
            runtime,
            initialized: false,
            registry,
            csa_metrics,
            offload_policy,
            residency,
            mapped_reclaim_registration: std::sync::OnceLock::new(),
            mapped_requesters: Mutex::new(HashMap::new()),
        };
        if let (Some(residency), Some(arena), Some(governor)) =
            (provider.residency.as_ref(), provider.vmm.get(), governor)
            && arena.physical_pool_authority().is_some()
        {
            residency
                .install_vmm_admission(Arc::clone(arena), governor)
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: cannot install committed-byte weight admission: {error}"
                    ))
                })?;
        }
        Ok(provider)
    }

    /// The allocator in force: the VMM arena once installed, otherwise the one
    /// this provider was built with.
    ///
    /// `OnceLock::get` is a relaxed atomic load, so the allocation path pays
    /// nothing for the option to swap.
    fn memory(&self) -> &dyn onnx_runtime_memory_governor::DeviceAllocator {
        match self.vmm.get() {
            Some(arena) => arena.as_ref(),
            None => self.memory.as_ref(),
        }
    }

    fn synchronize_before_pooled_unmap(&self) -> Result<()> {
        if self
            .vmm
            .get()
            .is_some_and(|arena| arena.physical_pool_stats().is_some())
        {
            self.runtime.synchronize().map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not synchronize before returning pooled physical memory: \
                     {error}"
                ))
            })?;
            self.runtime.copy_stream().synchronize().map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not synchronize the copy stream before returning pooled \
                     physical memory: {error}"
                ))
            })?;
        }
        Ok(())
    }

    /// Take device buffers from `memory` instead of calling `cuMemAlloc`
    /// directly.
    /// The same `DeviceAllocator` a caller installs on the CPU EP or the ONNX
    /// Runtime side. This is where a CUDA arena belongs: `cudaMalloc` is a
    /// synchronising driver call in the microseconds, so unlike host memory,
    /// device memory genuinely needs one — and behind this seam it is written
    /// once rather than once per backend.
    ///
    /// # Errors
    ///
    /// - If `memory` does not serve this EP's device. Pointers from it are
    ///   handed to kernels as this device's addresses, so a host allocator or
    ///   another device's allocator would produce an address that is invalid
    ///   where it is used. That fails inside a kernel launch, far from the
    ///   substitution that caused it, so it is rejected here instead.
    /// - If a VMM arena is already installed (constructed when
    ///   `ONNX_GENAI_CUDA_VMM` or dynamic lending select it, before this
    ///   method can run). `memory()` always prefers `self.vmm` once it is
    ///   set, so accepting the substitution here would return `Ok(Self)`
    ///   while `memory` sat unused forever — a silently ignored injection.
    ///   Rejecting it instead keeps "which mechanism serves this provider"
    ///   unambiguous: exactly one of VMM or the (possibly substituted) base
    ///   allocator is ever in force, decided at construction, and this method
    ///   cannot un-select VMM once chosen (its address-space reservation and
    ///   any already-mapped granules are not something this method can undo).
    ///   A caller who needs a specific base allocator must build without VMM
    ///   (`ONNX_GENAI_CUDA_VMM` unset and dynamic lending disabled).
    /// - If `memory` advertises a `VirtualBacking`/`SharedMapping` capability
    ///   whose `device()` disagrees with `memory.device()`. A capability that
    ///   disagrees with its own allocator about which device it serves
    ///   cannot be trusted to belong to it. This device check is a sanity
    ///   check on `memory`'s own self-reported answers, not a proof that the
    ///   capability object is genuinely `memory` itself as opposed to some
    ///   other object `memory` merely composes: no self-reported identity
    ///   token can distinguish those two cases from outside the type (see
    ///   the `capability` module's docs in `onnx_runtime_memory_api` for why
    ///   #1186 Phase 2 review's round 6 removed the earlier attempt at that
    ///   proof). A composite allocator whose ordinary allocate/deallocate and
    ///   whose advertised capability are two different objects is expected
    ///   to implement `as_virtual_backing`/`as_shared_mapping` honestly
    ///   enough that this does not arise; proving it structurally is #1186
    ///   Phase 3's scope.
    pub fn with_memory(
        mut self,
        memory: Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>,
    ) -> Result<Self> {
        if let Some(arena) = self.vmm.get() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: this execution provider already selected a VMM arena over CUDA \
                 device {} at construction time; it cannot also be substituted with the \
                 injected allocator, and mixing the two would make the ordinary allocation \
                 mechanism ambiguous. Construct without VMM (unset {} and disable dynamic \
                 KV/weight lending) to use `with_memory` instead.",
                DeviceAllocator::device(arena.as_ref()).index,
                crate::vmm_allocator::CUDA_VMM_ENV
            )));
        }
        let key = memory.device();
        let expected = onnx_runtime_memory_governor::DeviceKey::device(self.device.index);
        if key != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: this execution provider serves CUDA device {}, but the allocator \
                 offered serves {:?} {}; its pointers would not be valid where this EP uses \
                 them. Supply an allocator for CUDA device {}.",
                expected.index, key.tier, key.index, expected.index
            )));
        }
        // A capability's own `device()` must at least agree with the
        // allocator's `device()`. This is a sanity check on self-reported
        // values, not proof the capability object is genuinely `memory`
        // itself rather than a distinct object `memory` composes — see the
        // doc comment above and `onnx_runtime_memory_api::capability`'s
        // module docs for why that stronger proof was removed rather than
        // patched further (#1186 Phase 2 review, round 6).
        if let Some(virtual_backing) = memory.as_virtual_backing()
            && virtual_backing.device() != key
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: the injected allocator's VirtualBacking capability reports \
                 device {:?} {}, but the allocator itself reports {:?} {}; a capability \
                 that disagrees with its own allocator about device identity cannot be \
                 trusted to belong to it",
                virtual_backing.device().tier,
                virtual_backing.device().index,
                key.tier,
                key.index
            )));
        }
        if let Some(shared_mapping) = memory.as_shared_mapping()
            && shared_mapping.device() != key
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: the injected allocator's SharedMapping capability reports \
                 device {:?} {}, but the allocator itself reports {:?} {}; a capability \
                 that disagrees with its own allocator about device identity cannot be \
                 trusted to belong to it",
                shared_mapping.device().tier,
                shared_mapping.device().index,
                key.tier,
                key.index
            )));
        }
        self.memory = memory;
        Ok(self)
    }

    /// Construct and initialize a CUDA execution provider with default settings.
    pub fn initialized(ordinal: u32) -> Result<Self> {
        let mut provider = Self::new(ordinal)?;
        <Self as ExecutionProvider>::initialize(&mut provider, &EpConfig::default())?;
        Ok(provider)
    }

    /// Construct and initialize a CUDA execution provider with an already
    /// resolved weight-offload policy.
    pub fn initialized_with_offload_policy(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
    ) -> Result<Self> {
        let mut provider = Self::new_with_offload_policy(ordinal, offload_policy)?;
        <Self as ExecutionProvider>::initialize(&mut provider, &EpConfig::default())?;
        Ok(provider)
    }

    /// Construct and initialize with a device authority available to allocator
    /// construction.
    pub fn initialized_with_offload_policy_and_governor(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    ) -> Result<Self> {
        let mut provider =
            Self::new_with_offload_policy_and_governor(ordinal, offload_policy, governor)?;
        <Self as ExecutionProvider>::initialize(&mut provider, &EpConfig::default())?;
        Ok(provider)
    }

    /// Construct a CUDA EP on the default device (`CUDA:0`).
    pub fn new_default() -> Result<Self> {
        Self::new(0)
    }

    /// Return whether a fully initialized CUDA EP can be constructed for this
    /// device right now. This checks the driver, wheel/system libraries, device,
    /// and thread binding rather than reporting a compile-time feature.
    pub fn is_available(ordinal: u32) -> bool {
        Self::initialized(ordinal).is_ok()
    }

    /// Borrow the CUDA op registry (shared with the session layer).
    pub fn registry(&self) -> &OpRegistry {
        &self.registry
    }

    /// Borrow the shared CUDA runtime (context + stream + cuBLASLt handle).
    pub fn runtime(&self) -> &Arc<CudaRuntime> {
        &self.runtime
    }

    /// Borrow the shared CSA observability surface (§8). Every CSA kernel this
    /// EP builds records per-layer attention mode, bytes avoided, cursor
    /// lengths, sink mass, and host/device byte counts here; speculative
    /// rollbacks accumulate via the checkpoint journal.
    pub fn csa_metrics(&self) -> &Arc<CsaMetrics> {
        &self.csa_metrics
    }

    /// Build a live GPU weight pager (WEIGHT_OFFLOAD Phase 3b) that binds an
    /// offloaded `pkg.nxrt::BlockQuantizedMoE` weight into a VRAM page, copying
    /// its canonical bytes from `source` host→device. The returned binding is
    /// byte-identical to a resident upload of the same weight.
    pub fn weight_pager<'a, S: onnx_runtime_ep_api::MmapRegionSource>(
        &self,
        source: &'a S,
    ) -> crate::weight_paging::CudaWeightPager<'a, S> {
        crate::weight_paging::CudaWeightPager::new(Arc::clone(&self.runtime), source)
    }

    /// Build a bounded-VRAM [`CudaWeightResidency`] (WEIGHT_OFFLOAD Phase 3b
    /// page-in + eviction) sized by `budget_bytes`, sharing this EP's runtime.
    pub fn weight_residency(&self, budget_bytes: u64) -> crate::weight_paging::CudaWeightResidency {
        crate::weight_paging::CudaWeightResidency::new(Arc::clone(&self.runtime), budget_bytes)
    }

    /// Borrow the live device residency cache used to page lazy weights during
    /// dispatch, or `None` when weight offload is disabled. Tests use this to
    /// assert page-in / eviction counters after a decode.
    pub fn residency(&self) -> Option<&Arc<CudaWeightResidency>> {
        self.residency.as_ref()
    }

    /// The resolved device weight-offload policy for this EP.
    pub fn offload_policy(&self) -> &DeviceOffloadPolicy {
        &self.offload_policy
    }

    fn refund_canonical_mapped_zone(&self, unmapped: u64) {
        if unmapped == 0 {
            return;
        }
        if let Some(requester) = self
            .mapped_requesters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&mapped_attribution_role(
                onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
            ))
        {
            requester.unmap(unmapped);
        }
    }
}

impl ExecutionProvider for CudaExecutionProvider {
    fn name(&self) -> &str {
        "cuda_ep"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda
    }

    fn device_id(&self) -> DeviceId {
        self.device
    }

    fn memory_vendor_id(&self) -> u32 {
        // NVIDIA PCI vendor id — must match the value the plugin factory used
        // to register the CUDA device allocator's `OrtMemoryInfo` (#982).
        0x10DE
    }

    fn host_to_device_copier(&self) -> Option<std::sync::Arc<dyn HostToDeviceCopier>> {
        Some(std::sync::Arc::new(CudaHostToDeviceCopier {
            runtime: Arc::clone(&self.runtime),
        }))
    }

    /// Advertise the `nxrt` weight-paging capability only when device weight
    /// offload is enabled. This is what makes the session build real lazy weight
    /// handles for boundary-matched quantized weights; the default (offload off)
    /// path keeps stock capabilities so the fast resident path is unchanged.
    fn capabilities(&self) -> ExecutionProviderCapabilities {
        if self.residency.is_some() {
            ExecutionProviderCapabilities::nxrt_weight_paging()
        } else {
            ExecutionProviderCapabilities::stock()
        }
    }

    /// Page a lazy weight into VRAM (or reuse a resident page) through the LRU
    /// residency cache, returning a [`PagedWeight`] whose keep-alive pins the
    /// device allocation for the kernel's lifetime. Returns `Ok(None)` when
    /// offload is disabled so dispatch falls back to the resident path.
    fn page_lazy_weight(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn onnx_runtime_ep_api::MmapRegionSource,
    ) -> Result<Option<PagedWeight>> {
        let Some(residency) = self.residency.as_ref() else {
            return Ok(None);
        };
        let page = residency
            .resident_mapped(key, weight, source)
            .map_err(|error| EpError::KernelFailed(format!("weight offload page-in: {error}")))?;
        let device_ptr = page.device_ptr();
        let len = page.len();
        Ok(Some(PagedWeight::new(
            device_ptr,
            self.device,
            len,
            page as Arc<dyn std::any::Any + Send + Sync>,
        )))
    }

    fn prefetch_lazy_weight(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn onnx_runtime_ep_api::MmapRegionSource,
    ) -> Result<bool> {
        let _ = (self, key, weight, source);
        Ok(false)
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        // The context, stream, and cuBLASLt handle are created eagerly in
        // `new`; binding here confirms the device is reachable on this thread.
        self.runtime.bind()?;
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        // Keyed on (op_type, domain, opset) via the registry, the same single
        // source of truth the CPU EP uses.
        if !self.registry.supports(&op.op_type, &op.domain, opset) {
            let domain = if op.domain.is_empty() {
                "ai.onnx"
            } else {
                &op.domain
            };
            if let Some(since) = self
                .registry
                .earliest_since_version(&op.op_type, &op.domain)
            {
                deny!(
                    "no handler for {}::{} at opset {} — this EP registers {} since opset {} (or: add a claim+handler)",
                    domain,
                    op.op_type,
                    opset,
                    op.op_type,
                    since
                );
            }
            deny!(
                "no handler for {}::{} at opset {} — add a claim+handler",
                domain,
                op.op_type,
                opset
            );
        }
        if matches!(op.op_type.as_str(), "FusedMatMulBias" | "FusedGemm")
            && op.domain == "com.microsoft"
            && let Some(reason) = crate::kernels::fused_gemm::unsupported_reason(op, shapes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "BlockQuantizedMatMul"
            && op.domain == "pkg.nxrt"
            && let Some(reason) = crate::kernels::block_quantized_matmul::unsupported_reason(op)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "BlockQuantizedMoE"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::block_quantized_moe::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "CompressedSparseAttention"
            && op.domain == "pkg.nxrt"
            && let Some(reason) = crate::kernels::compressed_sparse_attention::unsupported_reason(
                op,
                shapes,
                input_dtypes,
            )
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "IndexShare"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::index_share::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "PackedVarlenAttention"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::packed_varlen_attention::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "VarlenAttention"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::varlen_attention::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "QMoE"
            && op.domain == "com.microsoft"
            && let Some(reason) = crate::kernels::qmoe::unsupported_reason(op)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "GroupQueryAttention"
            && op.domain == "com.microsoft"
            && let Some(reason) = crate::kernels::group_query_attention::unsupported_reason(op)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "GatherBlockQuantized"
            && op.domain == "com.microsoft"
            && let Some(reason) =
                crate::kernels::gather_block_quantized::unsupported_reason(op, shapes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "CausalConvWithState"
            && op.domain == "com.microsoft"
            && let Some(reason) =
                crate::kernels::causal_conv_with_state::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "LinearAttention"
            && matches!(op.domain.as_str(), "" | "ai.onnx" | "com.microsoft")
            && let Some(reason) =
                crate::kernels::linear_attention::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "Attention"
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::standard_attention::unsupported_reason(opset, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "RotaryEmbedding"
            && matches!(op.domain.as_str(), "" | "ai.onnx" | "com.microsoft")
            && let Some(reason) = crate::kernels::rotary_embedding::unsupported_reason(
                op.domain == "com.microsoft",
                input_dtypes,
            )
        {
            return KernelMatch::unsupported(reason);
        }
        if (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::standard_claims::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if matches!(op.op_type.as_str(), "QuantizeLinear" | "DequantizeLinear")
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) = crate::kernels::quantization::unsupported_reason(op, shapes)
        {
            return KernelMatch::unsupported(reason);
        }
        if matches!(
            op.op_type.as_str(),
            "Equal" | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual"
        ) && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::pointwise::comparison_unsupported_reason(&op.op_type, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if matches!(op.op_type.as_str(), "IsInf" | "IsNaN")
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::unary_predicate::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "PRelu"
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) = crate::kernels::prelu::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if matches!(
            op.op_type.as_str(),
            "BitwiseAnd" | "BitwiseOr" | "BitwiseXor" | "BitwiseNot" | "BitShift"
        ) && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::bitwise::unsupported_reason(&op.op_type, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        let output_layouts = vec![TensorLayout::contiguous(); op.outputs.len()];
        // Report *structure only*, never a machine rate (issue #995). The old
        // `Cost::new(elems*0.01, elems*0.01, 0.0).with_launch_us(10.0)
        // .with_bytes_moved(elems*4)` fabricated three machine constants — a
        // GPU-is-100×-faster-per-element ratio, a 10 µs launch latency, and an
        // f32 byte count (wrong by 8× for the int4 weights that dominate
        // decode) — none of which an EP can know portably. The EP knows only
        // the honest byte traffic from the real dtypes and shapes; the host's
        // bandwidth, FLOP/s, and launch latency are supplied by the placement
        // cost model (`onnx-runtime-cost-model`) from measured rates. Time
        // components are therefore left zero here.
        let bytes_moved = structural_input_bytes(shapes, input_dtypes);
        let cost = Cost::ZERO.with_bytes_moved(bytes_moved);
        KernelMatch::Supported {
            cost,
            required_input_layouts: None,
            output_layouts,
        }
    }

    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>], opset: u64) -> Result<Box<dyn Kernel>> {
        let factory = self
            .registry
            .lookup(&op.op_type, &op.domain, opset)
            .ok_or_else(|| EpError::NoEpForOp {
                domain: if op.domain.is_empty() {
                    "ai.onnx".to_string()
                } else {
                    op.domain.clone()
                },
                op_type: op.op_type.clone(),
                opset,
            })?;
        factory.create(op, shapes)
    }

    fn custom_passes(&self) -> Vec<Box<dyn onnx_runtime_optimizer::OptimizationPass>> {
        cuda_optimization_passes()
    }

    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer> {
        if dynamic_lending_enabled()
            && let Some(virtual_backing) = self.memory().as_virtual_backing()
            && let Some(governor) = self.governor.as_deref()
            && let Some(requester) = self
                .mapped_requesters
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&mapped_attribution_role(
                    onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
                ))
                .cloned()
        {
            let bytes = virtual_backing
                .mapped_bytes_for_allocation(size, alignment)
                .map_err(EpError::Memory)?;
            let grant = governor
                .prepare_mapped_growth(&requester, bytes)
                .map_err(EpError::Memory)?;
            return self.allocate_with_mapped_growth(size, alignment, grant);
        }
        let full = 0..size;
        self.allocate_committed(size, alignment, std::slice::from_ref(&full))
    }

    fn allocate_with_mapped_growth(
        &self,
        size: usize,
        alignment: usize,
        mut grant: onnx_runtime_memory_governor::MappedGrowthGrant,
    ) -> Result<DeviceBuffer> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(EpError::AlignmentError);
        }
        // The atomic capacity-token commit these callers need is inherent to
        // `CudaVmmAllocator` (governor-coupling reasons in
        // `onnx-runtime-cuda-memory::vmm_allocator`), not part of the
        // `VirtualBacking` capability trait. Dynamic mapped growth is only
        // ever entered for the arena this EP itself installed, so the
        // concrete field is used directly rather than the generic
        // `self.memory()` seam.
        let Some(arena) = self.vmm.get() else {
            return Err(EpError::KernelFailed(
                "cuda_ep: mapped growth requires the built-in CUDA VMM arena; the installed \
                 allocator reports a virtual-backing capability but is not the CUDA VMM arena, \
                 so there is no atomic capacity-token commit to use here"
                    .into(),
            ));
        };
        let full = 0..size;
        let allocation = arena
            .allocate_committed_with_capacity(
                size,
                alignment,
                std::slice::from_ref(&full),
                grant.physical_capacity(),
            )
            .map_err(EpError::Memory)?;
        self.ep_allocations.fetch_add(1, Ordering::Relaxed);
        let buffer = unsafe {
            DeviceBuffer::from_raw_parts(
                allocation.allocation.as_ptr().cast(),
                self.device,
                size,
                alignment,
            )
        };
        if let Err(error) = grant.commit_bytes(allocation.newly_mapped_bytes) {
            let ptr = buffer.into_raw();
            if let Some(ptr) = std::ptr::NonNull::new(ptr.cast::<u8>()) {
                // Attribution never committed, so roll back the physical map
                // without running the provider's canonical refund.
                unsafe { arena.deallocate(ptr, size, alignment) };
            }
            return Err(EpError::Memory(error));
        }
        Ok(buffer)
    }

    fn allocate_committed(
        &self,
        size: usize,
        alignment: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<DeviceBuffer> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(EpError::AlignmentError);
        }
        // One allocator for the whole project: the same `DeviceAllocator` a
        // caller can install on the CPU EP or the ONNX Runtime side backs this
        // one. The default is `CudaDeviceAllocator`, which is the `cuMemAlloc`
        // this used to call directly.
        //
        // `size` is passed through unchanged so that `deallocate` can pass the
        // same value; normalising a zero-byte request is the allocator's job,
        // because the contract lets an implementation rely on the two sizes
        // agreeing.
        //
        // An allocator with no `VirtualBacking` capability has no separate
        // commit step: it maps everything at `allocate` time, so the
        // documented fallback is the ordinary eager `allocate` call, ignoring
        // `committed_ranges` exactly as the eager contract promises (the
        // whole footprint is committed regardless of which ranges are named
        // live).
        let ptr = match self.memory().as_virtual_backing() {
            Some(virtual_backing) => virtual_backing
                .allocate_committed(size, alignment, committed_ranges)
                .map_err(EpError::Memory)?,
            None => self.memory().allocate(size, alignment).map_err(EpError::Memory)?,
        };
        self.ep_allocations.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` is a fresh, unique, non-null device allocation of
        // >= `size` bytes owned by this EP and freed exactly once in
        // `deallocate`. It is a device address, never dereferenced on the host.
        Ok(unsafe {
            DeviceBuffer::from_raw_parts(ptr.as_ptr().cast(), self.device, size, alignment)
        })
    }

    fn commit_allocation_range(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        assert_eq!(
            buffer.device(),
            self.device,
            "cuda_ep: refusing to commit a buffer from device {:?}",
            buffer.device()
        );
        // No separate commit step for an eager allocator: everything was
        // already committed at `allocate` time, matching
        // `ExecutionProvider::commit_allocation_range`'s documented no-op
        // default.
        let Some(virtual_backing) = self.memory().as_virtual_backing() else {
            return Ok(());
        };
        let Some(ptr) = std::ptr::NonNull::new(buffer.as_ptr().cast::<u8>() as *mut u8) else {
            return Ok(());
        };
        virtual_backing
            .commit_allocation_range(ptr, buffer.len(), buffer.alignment(), offset, bytes)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not commit range {offset}..{} of {} byte allocation on CUDA device {}: {error}",
                    offset.saturating_add(bytes),
                    buffer.len(),
                    self.device.index
                ))
            })
    }

    fn commit_allocation_ranges(&self, ranges: &[(&DeviceBuffer, usize, usize)]) -> Result<()> {
        // Same documented eager no-op as `commit_allocation_range`.
        let Some(virtual_backing) = self.memory().as_virtual_backing() else {
            return Ok(());
        };
        let raw = ranges
            .iter()
            .map(|&(buffer, offset, bytes)| {
                assert_eq!(
                    buffer.device(),
                    self.device,
                    "cuda_ep: refusing to commit a buffer from device {:?}",
                    buffer.device()
                );
                let ptr = std::ptr::NonNull::new(buffer.as_ptr().cast::<u8>() as *mut u8)
                    .ok_or_else(|| EpError::KernelFailed("cuda_ep: null commit buffer".into()))?;
                Ok(onnx_runtime_memory_governor::AllocationCommitRange {
                    ptr,
                    allocation_bytes: buffer.len(),
                    align: buffer.alignment(),
                    offset,
                    bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        virtual_backing.commit_allocation_ranges(&raw).map_err(|error| {
            EpError::KernelFailed(format!(
                "cuda_ep: could not commit {} binding range(s) atomically on CUDA device {}: {error}",
                raw.len(),
                self.device.index
            ))
        })
    }

    fn commit_allocation_ranges_with_mapped_growth(
        &self,
        ranges: &[(&DeviceBuffer, usize, usize)],
        grant: &mut onnx_runtime_memory_governor::MappedGrowthGrant,
    ) -> Result<u64> {
        // Same capacity-token coupling as `allocate_with_mapped_growth`: the
        // atomic commit is inherent to `CudaVmmAllocator`, so this goes
        // through the concrete field rather than the generic capability.
        let Some(arena) = self.vmm.get() else {
            return Err(EpError::KernelFailed(
                "cuda_ep: mapped growth requires the built-in CUDA VMM arena; the installed \
                 allocator reports a virtual-backing capability but is not the CUDA VMM arena, \
                 so there is no atomic capacity-token commit to use here"
                    .into(),
            ));
        };
        let raw = ranges
            .iter()
            .map(|&(buffer, offset, bytes)| {
                let ptr = std::ptr::NonNull::new(buffer.as_ptr().cast::<u8>() as *mut u8)
                    .ok_or_else(|| {
                        EpError::KernelFailed("cuda_ep: null mapped range buffer".into())
                    })?;
                Ok(onnx_runtime_memory_governor::AllocationCommitRange {
                    ptr,
                    allocation_bytes: buffer.len(),
                    align: buffer.alignment(),
                    offset,
                    bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        arena
            .commit_allocation_ranges_with_capacity(&raw, grant.physical_capacity())
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not commit {} binding range(s) atomically on CUDA device {}: {error}",
                    raw.len(),
                    self.device.index
                ))
            })
    }

    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[(&DeviceBuffer, usize, usize)],
    ) -> Result<u64> {
        // Eager default: every named byte is mapped-attribution bytes,
        // matching `ExecutionProvider::mapped_bytes_for_allocation_ranges`'s
        // documented default.
        let Some(virtual_backing) = self.memory().as_virtual_backing() else {
            return Ok(ranges.iter().fold(0_u64, |total, &(_, _, bytes)| {
                total.saturating_add(bytes as u64)
            }));
        };
        let raw = ranges
            .iter()
            .map(|&(buffer, offset, bytes)| {
                let ptr = std::ptr::NonNull::new(buffer.as_ptr().cast::<u8>() as *mut u8)
                    .ok_or_else(|| {
                        EpError::KernelFailed("cuda_ep: null mapped range buffer".into())
                    })?;
                Ok(onnx_runtime_memory_governor::AllocationCommitRange {
                    ptr,
                    allocation_bytes: buffer.len(),
                    align: buffer.alignment(),
                    offset,
                    bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        virtual_backing
            .mapped_bytes_for_allocation_ranges(&raw)
            .map_err(EpError::Memory)
    }

    fn mapped_bytes_for_allocation(&self, bytes: usize, alignment: usize) -> Result<u64> {
        match self.memory().as_virtual_backing() {
            Some(virtual_backing) => virtual_backing
                .mapped_bytes_for_allocation(bytes, alignment)
                .map_err(EpError::Memory),
            // Eager default: the whole request is mapped bytes, matching
            // `ExecutionProvider::mapped_bytes_for_allocation`'s documented
            // default.
            None => Ok(bytes as u64),
        }
    }

    fn decommit_allocation_range(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<u64> {
        assert_eq!(
            buffer.device(),
            self.device,
            "cuda_ep: refusing to decommit a buffer from device {:?}",
            buffer.device()
        );
        // Eager default: nothing to release early, matching
        // `ExecutionProvider::decommit_allocation_range`'s documented no-op
        // default.
        let Some(virtual_backing) = self.memory().as_virtual_backing() else {
            return Ok(0);
        };
        let Some(ptr) = std::ptr::NonNull::new(buffer.as_ptr().cast::<u8>() as *mut u8) else {
            return Ok(0);
        };
        self.synchronize_before_pooled_unmap()?;
        let unmapped = virtual_backing
            .decommit_allocation_range(ptr, buffer.len(), buffer.alignment(), offset, bytes)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not decommit range {offset}..{} of {} byte allocation on CUDA device {}: {error}",
                    offset.saturating_add(bytes),
                    buffer.len(),
                    self.device.index
                ))
            })?;
        self.refund_canonical_mapped_zone(unmapped);
        Ok(unmapped)
    }

    fn allocation_committed_bytes(&self, buffer: &DeviceBuffer) -> usize {
        let Some(ptr) = std::ptr::NonNull::new(buffer.as_ptr().cast::<u8>() as *mut u8) else {
            return 0;
        };
        match self.memory().as_virtual_backing() {
            Some(virtual_backing) => {
                virtual_backing.allocation_committed_bytes(ptr, buffer.len(), buffer.alignment())
            }
            // Eager default: the whole allocation is committed, matching
            // `ExecutionProvider::allocation_committed_bytes`'s documented
            // default.
            None => buffer.len(),
        }
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()> {
        self.deallocate_with_unmapped(buffer).map(|_| ())
    }

    fn deallocate_with_unmapped(&self, buffer: DeviceBuffer) -> Result<u64> {
        assert_eq!(
            buffer.device(),
            self.device,
            "cuda_ep: refusing to deallocate a buffer from device {:?}",
            buffer.device()
        );
        // Borrowed buffers alias memory owned elsewhere and must never be
        // cuMemFree'd. CUDA does not yet produce borrowed buffers, but keep the
        // invariant sound so one can never be freed here.
        if buffer.is_borrowed() {
            return Ok(0);
        }
        self.synchronize_before_pooled_unmap()?;
        let size = buffer.len();
        let align = buffer.alignment();
        let ptr = buffer.into_raw();
        let Some(ptr) = std::ptr::NonNull::new(ptr.cast::<u8>()) else {
            return Ok(0);
        };
        // Release through the *same* capability-discovery call
        // (`self.memory().as_virtual_backing()`) used in `allocate_committed`,
        // never through the base `DeviceAllocator::deallocate_with_unmapped`,
        // whenever this mechanism advertises `VirtualBacking`. Discovery is a
        // documented per-allocator invariant (stable `Some`/`None`, same
        // object, for the allocator's whole lifetime), so calling it again
        // here is guaranteed to reach the identical concrete mechanism that
        // produced `ptr` — closing the capability-identity gap structurally
        // rather than relying on `DeviceKey` (coarse: two unrelated arenas on
        // the same device compare equal by it) or on every implementation
        // happening to agree by convention (#1186 Phase 2 review, findings
        // 1 and 5).
        //
        // SAFETY: `ptr`, `size` and `align` are the triple this EP obtained
        // from `self.memory` in `allocate`/`allocate_committed`; `into_raw`
        // consumed the owning handle so no alias remains, and this is its
        // single free.
        let unmapped = match self.memory().as_virtual_backing() {
            Some(virtual_backing) => unsafe {
                virtual_backing.deallocate_committed(ptr, size, align)
            },
            None => unsafe { self.memory().deallocate_with_unmapped(ptr, size, align) },
        };
        self.refund_canonical_mapped_zone(unmapped);
        self.ep_frees.fetch_add(1, Ordering::Relaxed);
        Ok(unmapped)
    }

    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()> {
        assert_eq!(
            src.device(),
            self.device,
            "cuda_ep::copy: foreign src buffer"
        );
        assert_eq!(
            dst.device(),
            self.device,
            "cuda_ep::copy: foreign dst buffer"
        );
        if size > src.len() || size > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy: size {size} exceeds src {} or dst {}",
                src.len(),
                dst.len()
            )));
        }
        if size == 0 {
            return Ok(());
        }
        let src_p = cuptr(src.as_ptr());
        let dst_p = cuptr(dst.as_mut_ptr());
        // SAFETY: both endpoints are live device allocations of >= `size` bytes
        // (checked) on this EP's device; `dst` is `&mut` so it cannot alias `src`.
        unsafe { self.runtime.dtod(src_p, dst_p, size) }
    }

    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence> {
        assert_eq!(
            dst.device(),
            self.device,
            "cuda_ep::copy_async: foreign dst buffer"
        );
        if size > dst.len() || size > src.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_async: size {size} exceeds src {} or dst {}",
                src.len(),
                dst.len()
            )));
        }
        if size == 0 {
            return Ok(Fence::signalled());
        }
        let dst_p = cuptr(dst.as_mut_ptr());
        if src.device().is_host_accessible() {
            // Host → device weight prefetch: the real Phase-4 overlap path. The
            // copy is enqueued on the dedicated transfer stream and the returned
            // fence names its completion event.
            // SAFETY: a host-accessible src buffer exposes a dereferenceable host
            // pointer to at least `size` bytes (checked); the async copy keeps
            // reading `src` until the transfer stream completes, which the caller
            // orders via `wait_fence` before mutating or freeing `src`.
            let host = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), size) };
            // SAFETY: `dst` is a live device allocation of >= `size` bytes.
            unsafe { self.runtime.htod_async(host, dst_p) }?;
        } else {
            assert_eq!(
                src.device(),
                self.device,
                "cuda_ep::copy_async: foreign device src buffer"
            );
            let src_p = cuptr(src.as_ptr());
            // SAFETY: both endpoints are live device allocations of >= `size`
            // bytes (checked) on this EP's device; `dst` is `&mut` so it cannot
            // alias `src`. The transfer-stream copy is ordered via the fence.
            unsafe { self.runtime.dtod_async_on_copy_stream(src_p, dst_p, size) }?;
        }
        let fence_id = self.runtime.record_copy_fence()?;
        Ok(Fence::new(fence_id))
    }

    fn wait_fence(&self, fence: &Fence) -> Result<()> {
        // Order the compute stream after the prefetch transfer: a stream-ordered,
        // non host-blocking cross-stream wait so the next kernel reads the fully
        // transferred bytes. An already-signalled fence is a no-op.
        self.runtime.compute_wait_fence(fence.id)
    }

    fn record_compute_fence(&self) -> Result<Fence> {
        // Record a completion event over the compute stream so a later reuse
        // prefetch (via `copy_wait_fence`) waits for this consumer to finish
        // reading a double-buffer slot before overwriting it (WAR ordering).
        let fence_id = self.runtime.record_compute_fence()?;
        Ok(Fence::new(fence_id))
    }

    fn copy_wait_fence(&self, fence: &Fence) -> Result<()> {
        // Order the transfer stream after the prior consumer's compute: a
        // stream-ordered, non host-blocking cross-stream wait so a reuse prefetch
        // never clobbers a staging buffer mid-read. Already-signalled is a no-op.
        self.runtime.copy_wait_fence(fence.id)
    }

    fn device_argmax_supported(&self) -> bool {
        true
    }

    fn device_argmax(
        &self,
        logits: &DeviceBuffer,
        elements: usize,
        batch: usize,
        dtype: DataType,
        result: &mut DeviceBuffer,
    ) -> Result<()> {
        crate::kernels::device_argmax::launch(&self.runtime, logits, elements, batch, dtype, result)
    }

    #[allow(clippy::too_many_arguments)]
    fn device_token_writer(
        &self,
        result: &DeviceBuffer,
        input_ids: &DeviceBuffer,
        position_ids: &DeviceBuffer,
        attention_mask: &DeviceBuffer,
        scratch: &DeviceBuffer,
        capacity: usize,
        next_position: i64,
        mask_len: usize,
        write_position: bool,
        step: u32,
    ) -> Result<()> {
        crate::kernels::device_token_writer::launch(
            &self.runtime,
            result,
            input_ids,
            position_ids,
            attention_mask,
            scratch,
            capacity,
            next_position,
            mask_len,
            write_position,
            step,
        )
    }

    fn copy_from_host(&self, src: &[u8], dst: &mut DeviceBuffer) -> Result<()> {
        assert_eq!(
            dst.device(),
            self.device,
            "cuda_ep::copy_from_host: foreign dst buffer"
        );
        if src.len() > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_from_host: source {} bytes exceeds dst {}",
                src.len(),
                dst.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: `dst` is a live allocation on this CUDA device with enough
        // capacity (checked above), and the synchronous copy completes here.
        unsafe { self.runtime.htod(src, cuptr(dst.as_mut_ptr())) }
    }

    fn copy_from_host_at(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer,
        byte_offset: usize,
    ) -> Result<()> {
        assert_eq!(
            dst.device(),
            self.device,
            "cuda_ep::copy_from_host_at: foreign dst buffer"
        );
        let end = byte_offset.checked_add(src.len()).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep::copy_from_host_at: upload range overflows".into())
        })?;
        if end > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_from_host_at: range {byte_offset}..{end} exceeds dst {}",
                dst.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        let ptr = cuptr(dst.as_mut_ptr())
            .checked_add(byte_offset as u64)
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep::copy_from_host_at: device pointer offset overflows".into(),
                )
            })?;
        // SAFETY: `ptr` names the checked byte range within `dst`, and the
        // synchronous copy completes before this method returns.
        unsafe { self.runtime.htod(src, ptr) }
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        assert_eq!(
            src.device(),
            self.device,
            "cuda_ep::copy_to_host: foreign src buffer"
        );
        if dst.len() > src.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_to_host: destination {} bytes exceeds src {}",
                dst.len(),
                src.len()
            )));
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: `src` is a live allocation on this CUDA device with enough
        // readable bytes (checked above); `dtoh` synchronizes before returning.
        unsafe { self.runtime.dtoh(dst, cuptr(src.as_ptr())) }
    }

    fn begin_device_graph_capture(&self, kernels: &[&dyn Kernel]) -> Result<()> {
        self.runtime.begin_graph_capture(kernels)
    }

    fn end_device_graph_capture(&self) -> Result<()> {
        self.runtime.end_graph_capture()
    }

    fn abort_device_graph_capture(&self) -> Result<()> {
        self.runtime.abort_graph_capture()
    }

    fn replay_device_graph(&self) -> Result<()> {
        self.runtime.replay_graph()
    }

    fn replay_device_graph_segment(&self, index: usize) -> Result<()> {
        self.runtime.replay_graph_segment(index)
    }

    fn reset_device_graph(&self) -> Result<bool> {
        // Graph invalidation (reset / rewind / KV-capacity or shape change /
        // re-capture) is the explicit host reset point for the capture-error
        // latch, so a fresh generation always starts un-poisoned.
        let invalidated = self.runtime.reset_graph()?;
        self.runtime.reset_capture_error()?;
        Ok(invalidated)
    }

    fn check_device_capture_error(&self) -> Result<u32> {
        self.runtime.check_capture_error()
    }

    fn device_allocation_counts(&self) -> Option<(u64, u64)> {
        // The sum of both paths. Kernels still reach the driver through
        // `CudaRuntime::alloc_raw` for their own workspaces, while buffers this
        // EP hands out go through the replaceable allocator. Reporting only one
        // of them is how the capture-safety assertions stopped observing
        // anything without ever going red.
        let counts = self.runtime.allocation_counts();
        Some((
            counts.allocations + self.ep_allocations.load(Ordering::Relaxed),
            counts.frees + self.ep_frees.load(Ordering::Relaxed),
        ))
    }

    fn reserve_workspace(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Result<Option<onnx_runtime_memory_governor::MemoryLease>> {
        if self.memory().as_virtual_backing().is_some() {
            return Ok(None);
        }
        self.governor
            .as_deref()
            .map(|governor| {
                governor.reserve(
                    onnx_runtime_memory_governor::Tier::Device,
                    bytes,
                    role,
                    onnx_runtime_memory_governor::HolderId::new(64),
                )
            })
            .transpose()
            .map_err(Into::into)
    }

    fn prepare_mapped_growth(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Result<Option<onnx_runtime_memory_governor::MappedGrowthGrant>> {
        if bytes == 0
            || !dynamic_lending_enabled()
            || self.memory().as_virtual_backing().is_none()
        {
            return Ok(None);
        }
        let Some(governor) = self.governor.as_deref() else {
            eprintln!(
                "cuda_ep: WARNING: dynamic mapped growth requested without an authority \
                 participant; continuing with ordinary allocator admission"
            );
            return Ok(None);
        };
        let mut requesters = self
            .mapped_requesters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Content roles remain distinct in their leases and metrics, but every
        // allocation in this suballocating arena shares one mapped allowance.
        let role = mapped_attribution_role(role);
        let requester = match requesters.entry(role) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let holder = match role {
                    onnx_runtime_memory_governor::MemoryRole::KvCache => {
                        onnx_runtime_memory_governor::HolderId::new(65)
                    }
                    _ => onnx_runtime_memory_governor::HolderId::new(66),
                };
                entry.insert(
                    governor
                        .reserve_mapped_allowance(
                            onnx_runtime_memory_governor::Tier::Device,
                            0,
                            role,
                            holder,
                        )
                        .map_err(EpError::Memory)?,
                )
            }
        };
        governor
            .prepare_mapped_growth(requester, bytes)
            .map(Some)
            .map_err(EpError::Memory)
    }

    fn release_mapped_growth(&self, bytes: u64, role: onnx_runtime_memory_governor::MemoryRole) {
        // VMM deallocation performs the canonical arena-zone refund. Keeping
        // this hook as a no-op preserves compatibility for callers/providers
        // that do not use the CUDA arena without permitting double release.
        let _ = (bytes, role);
    }

    /// True when the VMM arena is in use: it maps 2 MiB granules as spans are
    /// handed out and leases each one before mapping it, so committed memory
    /// tracks real use rather than the largest request anyone might make.
    ///
    /// False on the `cuMemAlloc` path, which takes physical memory at the
    /// moment it is asked for.
    fn commits_on_demand(&self) -> bool {
        self.memory().as_virtual_backing().is_some()
    }

    fn set_weight_residency_budget(&self, budget_bytes: u64) -> Result<Option<u64>> {
        let Some(residency) = self.residency.as_ref() else {
            return Ok(None);
        };
        residency
            .set_ungoverned_budget(budget_bytes)
            .map(Some)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: cannot set the device weight-residency budget to \
                     {budget_bytes} bytes before governor adoption: {error}"
                ))
            })
    }

    fn adopt_memory_governor(
        &self,
        governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> Result<u64> {
        if let Some(arena) = self.vmm.get() {
            if let Some(authority) = arena.physical_pool_authority()
                && authority != governor.authority_id()
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep: physical-handle pool uses {authority}, but adoption supplied {}; \
                     both must use the same memory authority",
                    governor.authority_id()
                )));
            }
            // The arena has been serving allocations against its own ledger
            // since construction. Move the claim to the real one now that it
            // exists.
            let adoption = arena.adopt_governor(governor, holder);
            if adoption.recorded_bytes > 0 {
                let bytes = adoption.recorded_bytes;
                eprintln!(
                    "cuda_ep: VMM arena joined the memory ledger holding {bytes} bytes already \
                     committed"
                );
            }
            if adoption.unaccounted_bytes > 0 {
                let bytes = adoption.unaccounted_bytes;
                eprintln!(
                    "cuda_ep: WARNING: {bytes} committed VMM arena byte(s) were not recorded in \
                     the memory ledger; profile output will report the accounting fault"
                );
            }
        }

        // The weight-residency cache is the standing pool this EP keeps. With
        // offload disabled there is none, and zero is the honest answer rather
        // than a failure.
        let Some(residency) = self.residency.as_ref() else {
            return Ok(0);
        };
        let governed = residency
            .adopt_governed_budget(governor, tier, holder)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: the device weight-residency cache holds a budget the governor \
                     cannot grant on {tier:?}: {error}"
                ))
            })?;
        if self.offload_policy.managed_no_spill
            && dynamic_lending_enabled()
            && self.mapped_reclaim_registration.get().is_none()
        {
            let reclaimable: Arc<dyn onnx_runtime_memory_governor::ReclaimableMappedHolder> =
                Arc::clone(residency)
                    as Arc<dyn onnx_runtime_memory_governor::ReclaimableMappedHolder>;
            match governor.register_reclaimable_mapped_holder(&reclaimable) {
                Ok(registration) => {
                    let _ = self.mapped_reclaim_registration.set(registration);
                    eprintln!(
                        "cuda_ep: registered CUDA weight residency holder {holder:?} with \
                         {governed} allowance byte(s) for transactional mapped growth"
                    );
                }
                Err(error) => eprintln!(
                    "cuda_ep: WARNING: dynamic KV/weight lending is unavailable because the \
                     memory authority does not provide mapped-growth registration: {error}"
                ),
            }
        }
        Ok(governed)
    }

    fn sync(&self) -> Result<()> {
        self.runtime.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_lending_is_on_by_default_with_behavior_safe_opt_outs() {
        assert!(dynamic_lending_enabled_for(None));
        assert!(dynamic_lending_enabled_for(Some("1")));
        assert!(dynamic_lending_enabled_for(Some("true")));
        for disabled in ["0", " false ", "NO", "Off"] {
            assert!(!dynamic_lending_enabled_for(Some(disabled)));
        }
    }

    #[test]
    fn workspace_lifetimes_share_one_physical_mapping_zone() {
        let step_content =
            onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: true };
        let persistent_content =
            onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false };
        assert_ne!(
            step_content, persistent_content,
            "content accounting keeps lifetime categories distinct"
        );
        let step = mapped_attribution_role(step_content);
        let persistent = mapped_attribution_role(persistent_content);
        assert_eq!(step, persistent);
        assert_eq!(
            step,
            mapped_attribution_role(onnx_runtime_memory_governor::MemoryRole::KvCache),
            "the current provider's KV and workspace suballocate one arena"
        );
    }

    #[test]
    fn only_explicit_managed_policy_auto_enables_vmm_and_opt_out_restores_compatibility() {
        let compatibility = DeviceOffloadPolicy {
            enabled: true,
            ..DeviceOffloadPolicy::default()
        };
        assert!(!auto_dynamic_lending_for(true, &compatibility, true));
        let managed = DeviceOffloadPolicy {
            managed_no_spill: true,
            ..compatibility
        };
        assert!(auto_dynamic_lending_for(true, &managed, true));
        assert!(!auto_dynamic_lending_for(true, &managed, false));
        assert!(!auto_dynamic_lending_for(false, &managed, true));
    }

    #[test]
    fn managed_vmm_failure_is_fatal_before_allocator_fallback() {
        let allocation_attempted = std::sync::atomic::AtomicBool::new(false);
        let result = resolve_vmm_initialization::<(), _>(
            true,
            Some(6 << 30),
            Err("injected VMM initialization failure"),
        );
        if matches!(result, Ok(VmmInitialization::CompatibilityFallback(_))) {
            allocation_attempted.store(true, Ordering::Relaxed);
        }
        let error = result.expect_err("managed mode must not fall back");
        assert!(!allocation_attempted.load(Ordering::Relaxed));
        let message = error.to_string();
        assert!(message.contains("6442450944 bytes"), "{message}");
        assert!(
            message.contains("injected VMM initialization failure"),
            "{message}"
        );
        assert!(message.contains("before model allocation"), "{message}");
    }

    #[test]
    fn compatibility_vmm_failure_keeps_fallback_available() {
        let result = resolve_vmm_initialization::<(), _>(
            false,
            None,
            Err("injected VMM initialization failure"),
        )
        .expect("compatibility mode permits fallback");
        assert!(matches!(
            result,
            VmmInitialization::CompatibilityFallback(reason)
                if reason == "injected VMM initialization failure"
        ));
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn public_constructor_installs_configured_physical_pool() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        unsafe {
            std::env::set_var(crate::vmm_allocator::CUDA_VMM_ENV, "1");
            std::env::set_var(
                crate::vmm_allocator::CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
                (64usize << 20).to_string(),
            );
        }
        let provider = CudaExecutionProvider::new(0).expect("public CUDA provider");
        assert!(
            provider
                .vmm
                .get()
                .is_some_and(|arena| arena.physical_pool_stats().is_some()),
            "public constructor must use the configured physical pool"
        );
        let stats = provider
            .vmm
            .get()
            .and_then(|arena| arena.physical_pool_stats())
            .expect("pool stats");

        let runtime = provider.runtime().clone();
        let write_after_delay = runtime
            .nvrtc_function(
                "cuda_ep_pool_reuse_sync_test",
                r#"
extern "C" __global__ void write_after_delay(unsigned int* out, long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) { }
    *out = 0x736u;
}
"#,
                "write_after_delay",
            )
            .expect("compile delayed writer");
        let first = provider.allocate(4, 256).expect("first allocation");
        let first_ptr = cuptr(first.as_ptr());
        let spin = 8_000_000_i64;
        let mut launch = runtime.stream().launch_builder(&write_after_delay);
        launch.arg(&first_ptr).arg(&spin);
        unsafe {
            launch
                .launch(LaunchConfig::for_num_elems(1))
                .expect("enqueue delayed write")
        };

        provider
            .deallocate(first)
            .expect("deallocation synchronizes before pooled return");
        let second = provider.allocate(4, 256).expect("reused allocation");
        assert_eq!(stats.snapshot().pool_hits, 1);
        let mut value = [0_u8; 4];
        unsafe { runtime.dtoh(&mut value, cuptr(second.as_ptr())) }.expect("read reused mapping");
        assert_eq!(u32::from_ne_bytes(value), 0x736);
        provider.deallocate(second).expect("final deallocation");
    }

    /// #956: the standalone (plugin, no-governor) VMM path serves repeated
    /// same-size scratch requests from a retained physical-handle pool, so the
    /// arena's physical allocation call (`cuMemCreate`, the analog of the
    /// `cuMemAlloc` the default path makes per dispatch) does **not** scale with
    /// the number of allocate/free cycles.
    ///
    /// This constructs the *exact* arena the plugin path builds — `new_default`
    /// takes the `None` governor branch of the constructor, which calls
    /// `standalone_with_teardown_synchronizer` with
    /// `DEFAULT_STANDALONE_PHYSICAL_POOL_BYTES` — directly, so the measurement
    /// establishes its condition instead of depending on a process-global env
    /// var (measurement-discipline #906).
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn standalone_vmm_scratch_reuse_pools_committed_memory_and_does_not_scale_cumemcreate() {
        use onnx_runtime_memory_governor::DeviceAllocator;

        let Ok(provider) = CudaExecutionProvider::new(0) else {
            eprintln!(
                "SKIPPED (no CUDA runtime): the #956 scratch-reuse proof did NOT run. A skip that \
                 reads like a pass is exactly how a knob that never engaged produces a headline."
            );
            panic!("CUDA test path did not run; report as a failed GPU test, not a pass");
        };
        let runtime = provider.runtime().clone();
        let sync_runtime = Arc::clone(&runtime);
        let teardown: crate::virtual_memory::TeardownSynchronizer = Arc::new(move || {
            sync_runtime.synchronize().map_err(|e| e.to_string())?;
            sync_runtime
                .copy_stream()
                .synchronize()
                .map_err(|e| e.to_string())
        });
        let arena = crate::vmm_allocator::CudaVmmAllocator::standalone_with_teardown_synchronizer(
            runtime.cuda_context(),
            onnx_runtime_memory_governor::DeviceKey::device(0),
            0,
            64 << 30,
            onnx_runtime_memory_governor::HolderId::new(64),
            onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
            teardown,
            Some(DEFAULT_STANDALONE_PHYSICAL_POOL_BYTES),
        )
        .expect("standalone pooled arena");

        let stats = arena
            .physical_pool_stats()
            .expect("#956: the default installs a retained physical-handle pool");

        // A representative decode-scratch size. One 2 MiB granule backs it.
        const SCRATCH_BYTES: usize = 512 * 1024;
        // Written-and-verified header proving reused committed memory returns
        // exactly the bytes written this cycle — the numerics property a pooled
        // reuse could break (stale contents surviving a free/reuse). `n` tags
        // the cycle so a stale read from a previous cycle would mismatch.
        let cycle = |arena: &crate::vmm_allocator::CudaVmmAllocator, n: usize| {
            let ptr = arena.allocate(SCRATCH_BYTES, 256).expect("scratch alloc");
            let tag = ((n % 251) + 1) as u8;
            let header = vec![tag; 256];
            // SAFETY: `ptr` is this arena's live SCRATCH_BYTES allocation; the
            // 256-byte header is within it, and the copies are ordered by the
            // synchronous htod/dtoh.
            unsafe {
                runtime
                    .htod(&header, cuptr(ptr.as_ptr().cast::<std::ffi::c_void>()))
                    .expect("write scratch header");
                let mut read_back = vec![0u8; 256];
                runtime
                    .dtoh(
                        &mut read_back,
                        cuptr(ptr.as_ptr().cast::<std::ffi::c_void>()),
                    )
                    .expect("read scratch header");
                assert_eq!(
                    read_back, header,
                    "reused committed memory must return exactly what was written this cycle"
                );
            }
            // SAFETY: `ptr` is this arena's live SCRATCH_BYTES/256 allocation,
            // freed exactly once here.
            unsafe { arena.deallocate(ptr, SCRATCH_BYTES, 256) };
        };

        // Warm up: the first cycle creates and maps the granule, then retains
        // it in the pool on free.
        cycle(&arena, 0);
        let warm = stats.snapshot();

        for n in 0..16 {
            cycle(&arena, n + 1);
        }
        let after16 = stats.snapshot();
        for n in 0..64 {
            cycle(&arena, n + 100);
        }
        let after64 = stats.snapshot();

        eprintln!(
            "#956 standalone scratch reuse: warm(creates={} hits={} owned={}B) \
             +16cyc(creates={} hits={} owned={}B) +64cyc(creates={} hits={} owned={}B)",
            warm.creates,
            warm.pool_hits,
            warm.total_owned_bytes,
            after16.creates,
            after16.pool_hits,
            after16.total_owned_bytes,
            after64.creates,
            after64.pool_hits,
            after64.total_owned_bytes,
        );

        // The arena's physical allocation call does not scale with steps: after
        // warmup no further `cuMemCreate` happens, at 16 or at 64 cycles.
        assert_eq!(
            after16.creates, warm.creates,
            "no new cuMemCreate across 16 reuse cycles"
        );
        assert_eq!(
            after64.creates, warm.creates,
            "no new cuMemCreate across 64 reuse cycles"
        );
        // Measured, not inferred: the pool actually served each request, so its
        // hit count grew one-for-one with the cycle count.
        assert!(
            after16.pool_hits >= warm.pool_hits + 16,
            "16 reuse cycles must be served from the retained pool (measured hits, not an absent \
             symptom): {} -> {}",
            warm.pool_hits,
            after16.pool_hits
        );
        assert!(
            after64.pool_hits >= after16.pool_hits + 64,
            "64 further reuse cycles must be served from the retained pool: {} -> {}",
            after16.pool_hits,
            after64.pool_hits
        );
        // No leak: retained physical bytes are identical at 16 and 64 cycles.
        assert_eq!(
            after64.total_owned_bytes, after16.total_owned_bytes,
            "committed physical bytes must be bounded across steps"
        );
        assert_eq!(
            after64.releases, warm.releases,
            "retained handles are reused, not released per cycle"
        );
    }

    /// `with_memory` must reject a VMM-already-selected provider explicitly
    /// rather than accepting the injected allocator and silently continuing to
    /// prefer the arena: `memory()` always prefers `self.vmm` once it is set,
    /// so a builder that only overwrote `self.memory` would return `Ok(Self)`
    /// while the injected allocator sat unused forever — a successful ignored
    /// injection, exactly the ambiguity Phase 2 of #1186 removes. This proves
    /// both halves: the call itself is an `Err`, and the injected allocator
    /// never actually served a single allocation (its own counter, not an
    /// inference from the error).
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn with_memory_rejects_injection_once_vmm_is_already_selected() {
        // SAFETY: single-process test; forces the VMM arena on for this
        // constructor call so the rejection path being proven is reachable
        // regardless of the ambient environment.
        unsafe { std::env::set_var(crate::vmm_allocator::CUDA_VMM_ENV, "1") };

        let Ok(provider) = CudaExecutionProvider::new(0) else {
            eprintln!(
                "SKIPPED (no CUDA runtime): the with_memory VMM-rejection proof did NOT run."
            );
            panic!("CUDA test path did not run; report as a failed GPU test, not a pass");
        };
        assert!(
            provider.vmm.get().is_some(),
            "the forced-on VMM arena must actually be installed for this proof to test what it \
             claims"
        );

        let context = provider.runtime().cuda_context();
        // A same-device counting allocator: `CudaDeviceAllocator` already
        // tracks every `cuMemAlloc` it makes, so it doubles as a ready-made
        // "was this ever actually asked for memory" witness without a new
        // test-only wrapper type.
        let counting_allocator =
            Arc::new(crate::device_allocator::CudaDeviceAllocator::new(context));

        let error = provider
            .with_memory(counting_allocator.clone())
            .expect_err(
                "with_memory must return Err once a VMM arena is already selected, not Ok(Self) \
                 with the injection silently ignored",
            );
        let message = error.to_string();
        assert!(
            message.contains("already selected a VMM arena"),
            "the rejection must name what happened, not just fail: {message}"
        );
        assert_eq!(
            counting_allocator.cumemalloc_calls(),
            0,
            "the rejected allocator must never have been asked for memory — Err means it was \
             never adopted, not adopted-then-abandoned"
        );
    }

    /// `with_memory`'s capability checks only validate that a
    /// `VirtualBacking`/`SharedMapping` capability's own `device()` agrees
    /// with the allocator's `device()` — a sanity check on self-reported
    /// values, not a structural proof that the capability object is
    /// genuinely the allocator itself rather than some other object the
    /// allocator composes. An earlier design (#1186 Phase 2 review, rounds
    /// 3-4) attempted that stronger proof through a self-reported
    /// `MechanismId`/`TypeId`+pointer identity token; coordinator review
    /// (round 6) found a type can honestly return `Some(self)` from both its
    /// capability accessor and its identity accessor while its capability's
    /// method *bodies* still operate on unrelated state, which no
    /// self-reported token can detect from outside the type. That apparatus
    /// was removed rather than patched further; see
    /// `onnx_runtime_memory_api::capability`'s module docs for the full
    /// rationale, and #1186 Phase 3 for where a real proof belongs. This
    /// test proves the surviving, real (if narrower) device-mismatch check.
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn with_memory_rejects_a_capability_whose_device_disagrees_with_its_allocator() {
        // SAFETY: single-process test; ensure VMM is not force-enabled by a
        // previous test in this binary, so the capability-validation path
        // below (not the "VMM already selected" path) is what actually runs.
        unsafe { std::env::remove_var(crate::vmm_allocator::CUDA_VMM_ENV) };

        let Ok(provider) = CudaExecutionProvider::new(0) else {
            eprintln!(
                "SKIPPED (no CUDA runtime): the with_memory device-mismatch proof did NOT run."
            );
            panic!("CUDA test path did not run; report as a failed GPU test, not a pass");
        };
        assert!(
            provider.vmm.get().is_none(),
            "this proof requires the plain (non-VMM) construction path"
        );

        let context = provider.runtime().cuda_context();
        let base = Arc::new(crate::device_allocator::CudaDeviceAllocator::new(
            context.clone(),
        ));

        /// A `DeviceAllocator` that honestly returns `Some(self)` as its own
        /// `VirtualBacking` capability, but whose `VirtualBacking::device()`
        /// disagrees with its own `DeviceAllocator::device()` — the one
        /// mismatch `with_memory` can still detect without a structural
        /// identity proof, since it only compares self-reported values.
        #[derive(Debug)]
        struct DeviceMismatchedAllocator {
            base: Arc<crate::device_allocator::CudaDeviceAllocator>,
        }

        impl onnx_runtime_memory_governor::DeviceAllocator for DeviceMismatchedAllocator {
            fn allocate(
                &self,
                bytes: usize,
                align: usize,
            ) -> std::result::Result<
                std::ptr::NonNull<u8>,
                onnx_runtime_memory_governor::MemoryError,
            > {
                self.base.allocate(bytes, align)
            }

            unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, bytes: usize, align: usize) {
                // SAFETY: forwarded verbatim; our own caller upholds the same
                // contract `DeviceAllocator::deallocate` documents.
                unsafe { self.base.deallocate(ptr, bytes, align) }
            }

            fn device(&self) -> onnx_runtime_memory_governor::DeviceKey {
                onnx_runtime_memory_governor::DeviceAllocator::device(self.base.as_ref())
            }

            fn as_virtual_backing(
                &self,
            ) -> Option<&dyn onnx_runtime_memory_governor::VirtualBacking> {
                Some(self)
            }
        }

        impl onnx_runtime_memory_governor::VirtualBacking for DeviceMismatchedAllocator {
            fn device(&self) -> onnx_runtime_memory_governor::DeviceKey {
                // Deliberately a different device than `DeviceAllocator::device`
                // above reports.
                onnx_runtime_memory_governor::DeviceKey::device(
                    onnx_runtime_memory_governor::DeviceAllocator::device(self.base.as_ref())
                        .index
                        + 1,
                )
            }

            fn allocate_committed(
                &self,
                _bytes: usize,
                _align: usize,
                _committed_ranges: &[std::ops::Range<usize>],
            ) -> std::result::Result<
                std::ptr::NonNull<u8>,
                onnx_runtime_memory_governor::MemoryError,
            > {
                unimplemented!("test double: never called; with_memory must reject before use")
            }

            unsafe fn deallocate_committed(
                &self,
                _ptr: std::ptr::NonNull<u8>,
                _allocation_bytes: usize,
                _align: usize,
            ) -> u64 {
                unimplemented!("test double: never called")
            }

            fn commit_allocation_range(
                &self,
                _ptr: std::ptr::NonNull<u8>,
                _allocation_bytes: usize,
                _align: usize,
                _offset: usize,
                _bytes: usize,
            ) -> std::result::Result<(), onnx_runtime_memory_governor::MemoryError> {
                unimplemented!("test double: never called")
            }

            fn mapped_bytes_for_allocation_ranges(
                &self,
                _ranges: &[onnx_runtime_memory_governor::AllocationCommitRange],
            ) -> std::result::Result<u64, onnx_runtime_memory_governor::MemoryError> {
                unimplemented!("test double: never called")
            }

            fn mapped_bytes_for_allocation(
                &self,
                _bytes: usize,
                _align: usize,
            ) -> std::result::Result<u64, onnx_runtime_memory_governor::MemoryError> {
                unimplemented!("test double: never called")
            }

            fn decommit_allocation_range(
                &self,
                _ptr: std::ptr::NonNull<u8>,
                _allocation_bytes: usize,
                _align: usize,
                _offset: usize,
                _bytes: usize,
            ) -> std::result::Result<u64, onnx_runtime_memory_governor::MemoryError> {
                unimplemented!("test double: never called")
            }

            fn allocation_committed_bytes(
                &self,
                _ptr: std::ptr::NonNull<u8>,
                _allocation_bytes: usize,
                _align: usize,
            ) -> usize {
                unimplemented!("test double: never called")
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let mismatched: Arc<dyn onnx_runtime_memory_governor::DeviceAllocator> =
            Arc::new(DeviceMismatchedAllocator { base });

        let error = provider.with_memory(mismatched).expect_err(
            "with_memory must reject a VirtualBacking capability whose device() disagrees              with its own allocator's device()",
        );
        let message = error.to_string();
        assert!(
            message.contains("VirtualBacking capability reports"),
            "the rejection must name what happened, not just fail: {message}"
        );
    }


    /// #956 contrast: the default `cuMemAlloc` path makes exactly one driver
    /// allocation per request, so it scales one-for-one with decode steps —
    /// which is the residual the VMM arena removes. Fully isolated (per-instance
    /// counter, no env, direct construction).
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn default_allocator_cumemalloc_scales_one_for_one_with_requests() {
        use onnx_runtime_memory_governor::DeviceAllocator;

        let Ok(provider) = CudaExecutionProvider::new(0) else {
            eprintln!(
                "SKIPPED (no CUDA runtime): the #956 cuMemAlloc-scaling contrast did NOT run."
            );
            panic!("CUDA test path did not run; report as a failed GPU test, not a pass");
        };
        let context = provider.runtime().cuda_context();
        const SCRATCH_BYTES: usize = 512 * 1024;

        let run_cycles = |n: usize| -> u64 {
            let allocator = crate::device_allocator::CudaDeviceAllocator::new(context.clone());
            for _ in 0..n {
                let ptr = allocator.allocate(SCRATCH_BYTES, 256).expect("cuMemAlloc");
                // SAFETY: freed exactly once, same size/align it was allocated.
                unsafe { allocator.deallocate(ptr, SCRATCH_BYTES, 256) };
            }
            allocator.cumemalloc_calls()
        };

        let calls16 = run_cycles(16);
        let calls64 = run_cycles(64);
        eprintln!(
            "#956 default path cuMemAlloc calls: 16 requests -> {calls16}, 64 requests -> {calls64}"
        );
        assert_eq!(calls16, 16, "cuMemAlloc fires once per request");
        assert_eq!(
            calls64, 64,
            "cuMemAlloc scales one-for-one with request count on the default path"
        );
    }

    /// #956 integration: the provider built exactly as the CUDA plugin builds it
    /// (`CudaExecutionProvider::new_default` == `new(0)`) routes every device
    /// allocation — including the ORT scratch the plugin projects through
    /// `allocate`/`deallocate` — through the pooled VMM arena when the arena is
    /// enabled, rather than through `cuMemAlloc`.
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn plugin_construction_path_routes_device_memory_through_pooled_vmm_arena() {
        // SAFETY: single-process test; the plugin path reads this at
        // construction. The `.or(default)` in the standalone constructor means
        // the assertion below holds whether or not a pool-bytes override is also
        // set, so a concurrent test setting it cannot make this vacuous.
        unsafe { std::env::set_var(crate::vmm_allocator::CUDA_VMM_ENV, "1") };

        let provider =
            CudaExecutionProvider::new(0).expect("plugin-path CUDA provider under VMM arena");
        assert!(
            provider.commits_on_demand(),
            "the VMM arena, not the cuMemAlloc path, must serve allocations on the plugin path"
        );
        let stats = provider
            .vmm
            .get()
            .and_then(|arena| arena.physical_pool_stats())
            .expect(
                "#956: the standalone plugin path installs a retained physical-handle pool by \
                 default",
            );

        // The arena actually serves a real EP allocation (creates or reuses a
        // pooled granule) — not merely installed.
        let before = stats.snapshot();
        let buffer = provider
            .allocate(512 * 1024, 256)
            .expect("device allocation via the arena");
        provider.deallocate(buffer).expect("free via the arena");
        let after = stats.snapshot();
        assert!(
            (after.creates + after.pool_hits) > (before.creates + before.pool_hits),
            "the arena must have served the EP allocation (creates {}->{}, hits {}->{})",
            before.creates,
            after.creates,
            before.pool_hits,
            after.pool_hits
        );
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn general_deallocation_refunds_the_canonical_arena_zone() {
        use onnx_runtime_memory_governor::{
            HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, Tier,
        };

        let governor_impl = Arc::new(LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0)));
        let zone_role =
            mapped_attribution_role(onnx_runtime_memory_governor::MemoryRole::Workspace {
                step_scoped: true,
            });
        let zone_allowance = governor_impl
            .reserve_mapped_allowance(Tier::Device, 4 << 20, zone_role, HolderId::new(736))
            .expect("canonical arena allowance");
        let governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync> =
            governor_impl;
        let provider = CudaExecutionProvider::new_with_offload_policy_and_governor(
            0,
            DeviceOffloadPolicy {
                managed_no_spill: true,
                managed_limit_bytes: Some(8 << 30),
                ..DeviceOffloadPolicy::default()
            },
            governor,
        )
        .expect("governed VMM provider");
        provider
            .mapped_requesters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(zone_role, zone_allowance);

        let allocate_pair = || {
            let bytes = provider
                .mapped_bytes_for_allocation(4096, 256)
                .expect("workspace mapped size");
            let grant = provider
                .prepare_mapped_growth(
                    bytes,
                    onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: true },
                )
                .expect("prepare workspace growth")
                .expect("governed grant");
            let governed = provider
                .allocate_with_mapped_growth(4096, 256, grant)
                .expect("governed workspace");
            let ordinary = provider.allocate(4096, 256).expect("ordinary neighbor");
            let requester = provider
                .mapped_requesters
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&zone_role)
                .expect("canonical arena allowance")
                .clone();
            assert!(requester.mapped_bytes() > 0);
            (governed, ordinary, requester)
        };

        let (governed, ordinary, requester) = allocate_pair();
        let mapped = requester.mapped_bytes();
        assert_eq!(provider.deallocate_with_unmapped(governed).unwrap(), 0);
        assert_eq!(requester.mapped_bytes(), mapped);
        assert_eq!(provider.deallocate_with_unmapped(ordinary).unwrap(), mapped);
        assert_eq!(requester.mapped_bytes(), 0);
        provider.release_mapped_growth(
            mapped,
            onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: true },
        );
        assert_eq!(
            requester.mapped_bytes(),
            0,
            "specialized cleanup cannot double-refund the provider-owned zone"
        );

        let (governed, ordinary, requester) = allocate_pair();
        let mapped = requester.mapped_bytes();
        assert_eq!(provider.deallocate_with_unmapped(ordinary).unwrap(), 0);
        assert_eq!(requester.mapped_bytes(), mapped);
        assert_eq!(provider.deallocate_with_unmapped(governed).unwrap(), mapped);
        assert_eq!(requester.mapped_bytes(), 0);

        // Once the arena zone exists, ordinary-only allocation is admitted,
        // charged, and refunded by the same provider-owned path.
        for _ in 0..3 {
            let ordinary = provider.allocate(4096, 256).expect("ordinary allocation");
            assert!(requester.mapped_bytes() > 0);
            provider.deallocate(ordinary).expect("ordinary cleanup");
            assert_eq!(requester.mapped_bytes(), 0);
        }

        let granule = provider
            .mapped_bytes_for_allocation(4096, 256)
            .expect("allocation granule");
        for _ in 0..3 {
            let buffer = provider
                .allocate((granule * 2) as usize, 256)
                .expect("two-granule ordinary allocation");
            assert_eq!(requester.mapped_bytes(), granule * 2);
            assert_eq!(
                provider
                    .decommit_allocation_range(&buffer, granule as usize, granule as usize,)
                    .expect("partial decommit"),
                granule
            );
            assert_eq!(requester.mapped_bytes(), granule);
            assert_eq!(provider.deallocate_with_unmapped(buffer).unwrap(), granule);
            assert_eq!(requester.mapped_bytes(), 0);
        }

        let (governed, ordinary, requester) = allocate_pair();
        let mapped = requester.mapped_bytes();
        assert_eq!(
            provider
                .decommit_allocation_range(&governed, 0, 4096)
                .expect("shared-range decommit"),
            0,
            "the ordinary neighbor retains the shared granule"
        );
        assert_eq!(requester.mapped_bytes(), mapped);
        assert_eq!(provider.deallocate_with_unmapped(ordinary).unwrap(), mapped);
        assert_eq!(requester.mapped_bytes(), 0);
        assert_eq!(provider.deallocate_with_unmapped(governed).unwrap(), 0);
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn provider_drop_synchronizes_before_handle_reuse() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor};

        unsafe {
            std::env::set_var(crate::vmm_allocator::CUDA_VMM_ENV, "1");
            std::env::set_var(
                crate::vmm_allocator::CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
                (64usize << 20).to_string(),
            );
        }
        let governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync> =
            Arc::new(LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0)));
        let first = CudaExecutionProvider::new_with_offload_policy_and_governor(
            0,
            DeviceOffloadPolicy::default(),
            Arc::clone(&governor),
        )
        .expect("first provider");
        let runtime = first.runtime().clone();
        let stats = first
            .vmm
            .get()
            .and_then(|arena| arena.physical_pool_stats())
            .expect("pool stats");
        let write_after_delay = runtime
            .nvrtc_function(
                "cuda_ep_pool_drop_sync_test",
                r#"
extern "C" __global__ void write_after_delay(unsigned int* out, long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) { }
    *out = 0x736u;
}
"#,
                "write_after_delay",
            )
            .expect("compile delayed writer");
        let allocation = first.allocate(4, 256).expect("first allocation");
        let pointer = cuptr(allocation.as_ptr());
        let spin = 8_000_000_i64;
        let mut launch = runtime.stream().launch_builder(&write_after_delay);
        launch.arg(&pointer).arg(&spin);
        unsafe {
            launch
                .launch(LaunchConfig::for_num_elems(1))
                .expect("enqueue delayed write")
        };
        #[allow(clippy::drop_non_drop)]
        drop(allocation);
        drop(first);
        let after_teardown = stats.snapshot();
        assert_eq!(after_teardown.releases, 1);
        assert_eq!(after_teardown.pool_hits, 0);
        assert_eq!(after_teardown.total_owned_bytes, 0);

        let second = CudaExecutionProvider::new_with_offload_policy_and_governor(
            0,
            DeviceOffloadPolicy::default(),
            governor,
        )
        .expect("second provider");
        let later = second.allocate(4, 256).expect("later allocation");
        assert_eq!(
            stats.snapshot().pool_hits,
            0,
            "the old handle was synchronized and released, never reused early"
        );
        second.deallocate(later).expect("final deallocation");
    }

    #[test]
    fn runtime_availability_matches_constructability() {
        let available = CudaExecutionProvider::is_available(0);
        let constructible = CudaExecutionProvider::initialized(0).is_ok();
        assert_eq!(available, constructible);
    }

    // Phase-4 overlap through the public `ExecutionProvider` surface: a host→
    // device `copy_async` returns an awaitable `Fence`, and `wait_fence` orders
    // the compute stream after the transfer. The async copy is delayed behind a
    // spin kernel on the transfer stream, so a consumer launched on the compute
    // stream reads the correct payload only because `wait_fence` established the
    // cross-stream dependency — an already-signalled placeholder fence would let
    // it race ahead and read the pre-transfer poison.
    #[test]
    fn copy_async_fence_orders_h2d_prefetch_through_ep_api() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use std::ffi::c_void;

        const MODULE: &str = "cuda_ep_copy_async_api_test";
        const SOURCE: &str = r#"
extern "C" __global__ void spin_delay(long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) { }
}
extern "C" __global__ void copy_out(const float* in, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i];
}
"#;
        let Ok(ep) = CudaExecutionProvider::initialized(0) else {
            eprintln!("skipping copy_async API test: CUDA EP unavailable");
            return;
        };
        let runtime = ep.runtime().clone();
        let spin_delay = runtime
            .nvrtc_function(MODULE, SOURCE, "spin_delay")
            .unwrap();
        let copy_out = runtime.nvrtc_function(MODULE, SOURCE, "copy_out").unwrap();

        let n = 4096usize;
        let bytes = n * std::mem::size_of::<f32>();
        let n_u64 = n as u64;

        // Pinned host staging holds the payload; wrap it as a borrowed,
        // host-accessible source buffer for `copy_async`.
        let mut staging = runtime.alloc_pinned(bytes).unwrap();
        let payload: Vec<f32> = (0..n).map(|i| 2.0 + (i % 11) as f32).collect();
        staging.as_mut_slice().copy_from_slice(unsafe {
            std::slice::from_raw_parts(payload.as_ptr().cast::<u8>(), bytes)
        });
        // SAFETY: the pinned staging outlives `src` and every use of it, and it
        // is only read (never written) through the borrowed handle.
        let src = unsafe {
            DeviceBuffer::from_borrowed_parts(
                staging.as_slice().as_ptr() as *mut c_void,
                DeviceId::cpu(),
                bytes,
                1,
            )
        };

        let mut dst = ep.allocate(bytes, 256).unwrap();
        let out = ep.allocate(bytes, 256).unwrap();
        let out_p = cuptr(out.as_ptr());

        for _ in 0..8 {
            // Poison the device destination so a premature read is detectable.
            let poison = vec![-321.0f32; n];
            let poison_bytes =
                unsafe { std::slice::from_raw_parts(poison.as_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.htod(poison_bytes, cuptr(dst.as_ptr())) }.unwrap();
            runtime.synchronize().unwrap();

            // Occupy the transfer stream so the async copy cannot finish at once.
            let spin: i64 = 8_000_000;
            let mut delay = runtime.copy_stream().launch_builder(&spin_delay);
            delay.arg(&spin);
            unsafe { delay.launch(LaunchConfig::for_num_elems(1)).unwrap() };

            // Public EP surface: async prefetch, then await its fence.
            let fence = ep.copy_async(&src, &mut dst, bytes).unwrap();
            assert!(
                !fence.is_signalled(),
                "a real transfer must return an unsignalled fence"
            );
            ep.wait_fence(&fence).unwrap();

            // Consume the prefetched buffer on the compute stream.
            let dst_p = cuptr(dst.as_ptr());
            let mut consume = runtime.stream().launch_builder(&copy_out);
            consume.arg(&dst_p).arg(&out_p).arg(&n_u64);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };

            let mut host = vec![0.0f32; n];
            let host_bytes =
                unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(host_bytes, out_p) }.unwrap();
            assert_eq!(
                host, payload,
                "copy_async consumer read poison — the fence did not order the \
                 transfer before the compute-stream read"
            );
        }

        ep.deallocate(dst).unwrap();
        ep.deallocate(out).unwrap();
    }

    // Anti-regression lock for the async, fence-ordered weight page-in (#87 first
    // increment). Both arms drive the transfer and compute streams through the
    // *same* primitive chain `CudaWeightPage::upload_async` composes internally
    // (`htod_async` + `record_copy_fence`), differing only in how the
    // compute-stream consumer is ordered relative to the transfer:
    //
    //   * Positive (real page-in ordering): a spin-delay holds the H2D copy
    //     pending on the transfer stream, then `compute_wait_fence` orders the
    //     compute-stream consumer after it, so the consumer reads the fully
    //     paged-in bytes. Deleting `compute_wait_fence` leaves the consumer to
    //     read the pre-copy POISON, so the lock is non-vacuous.
    //   * Negative (deterministic poison control): the transfer is event-ordered
    //     strictly *after* the consumer (`record_compute_fence` + `copy_wait_fence`),
    //     so with no `compute_wait_fence` the consumer provably reads pre-transfer
    //     POISON. This proves the compute-side wait is load-bearing without a
    //     wall-clock race — an earlier revision raced the consumer against a
    //     spin-delayed copy, which the parallel, captured `cargo test` invocation
    //     flaked whenever GPU contention delayed the consumer kernel past the copy.
    //
    // Every device/pinned allocation is hoisted out of the timing window, so no
    // synchronizing `cuMemAlloc`/`cuMemHostAlloc` can drain the copy-stream
    // spin-delay: the delay→async-copy→fence→consume window is the only thing the
    // positive arm's ordering depends on. A trailing `upload_async` byte-parity
    // check keeps the real allocate+stage+copy+fence entry point under test.
    #[test]
    fn async_pagein_fence_orders_weight_page_in_consumer() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        const MODULE: &str = "cuda_ep_async_pagein_test";
        const SOURCE: &str = r#"
extern "C" __global__ void spin_delay(long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) { }
}
extern "C" __global__ void copy_out(const float* in, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i];
}
"#;
        let Ok(ep) = CudaExecutionProvider::initialized(0) else {
            eprintln!("skipping async page-in fence test: CUDA EP unavailable");
            return;
        };
        let runtime = ep.runtime().clone();
        let spin_delay = runtime
            .nvrtc_function(MODULE, SOURCE, "spin_delay")
            .unwrap();
        let copy_out = runtime.nvrtc_function(MODULE, SOURCE, "copy_out").unwrap();

        let n = 4096usize;
        let bytes = n * std::mem::size_of::<f32>();
        let n_u64 = n as u64;
        let payload: Vec<f32> = (0..n).map(|i| 5.0 + (i % 13) as f32).collect();
        let payload_bytes =
            unsafe { std::slice::from_raw_parts(payload.as_ptr().cast::<u8>(), bytes) };

        // Hoist every device/pinned allocation OUT of the per-iteration timing
        // window: a synchronizing `cuMemAlloc`/`cuMemHostAlloc` between the
        // spin-delay and the consumer would drain the delay and let ordering lean
        // on the alloc instead of the fence. All buffers are reused each iteration.
        let poison = vec![-777.0f32; n];
        let poison_bytes =
            unsafe { std::slice::from_raw_parts(poison.as_ptr().cast::<u8>(), bytes) };
        let pos_dst = ep.allocate(bytes, 256).unwrap();
        let neg_dst = ep.allocate(bytes, 256).unwrap();
        let out = ep.allocate(bytes, 256).unwrap();
        let pos_dst_p = cuptr(pos_dst.as_ptr());
        let neg_dst_p = cuptr(neg_dst.as_ptr());
        let out_p = cuptr(out.as_ptr());
        let mut staging = runtime.alloc_pinned(bytes).unwrap();
        staging.as_mut_slice().copy_from_slice(payload_bytes);
        let spin: i64 = 8_000_000;

        for _ in 0..8 {
            // ── Positive: the real page-in ordering. Poison the destination, hold
            // the H2D copy pending behind a spin-delay, then order the
            // compute-stream consumer after the transfer with `compute_wait_fence`
            // (the exact `htod_async` + `record_copy_fence` chain `upload_async`
            // composes). With the fence the consumer reads the paged-in payload;
            // delete the fence and it reads the poison below.
            unsafe { runtime.htod(poison_bytes, pos_dst_p) }.unwrap();
            runtime.synchronize().unwrap();

            let mut delay = runtime.copy_stream().launch_builder(&spin_delay);
            delay.arg(&spin);
            unsafe { delay.launch(LaunchConfig::for_num_elems(1)).unwrap() };

            unsafe { runtime.htod_async(staging.as_slice(), pos_dst_p) }.unwrap();
            let fence = runtime.record_copy_fence().unwrap();
            runtime.compute_wait_fence(fence).unwrap();

            let mut consume = runtime.stream().launch_builder(&copy_out);
            consume.arg(&pos_dst_p).arg(&out_p).arg(&n_u64);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };
            let mut got = vec![0.0f32; n];
            let got_bytes =
                unsafe { std::slice::from_raw_parts_mut(got.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(got_bytes, out_p) }.unwrap();
            runtime.sync_copy_stream().unwrap();
            assert_eq!(
                got, payload,
                "async page-in consumer read stale bytes — compute_wait_fence did \
                 not order the transfer before the compute-stream read"
            );

            // ── Negative (deterministic poison control): event-order the transfer
            // strictly AFTER the consumer, so with NO `compute_wait_fence` the
            // consumer provably reads pre-transfer poison. The `copy_wait_fence`
            // on a compute-stream fence removes all wall-clock racing — the
            // outcome never depends on the consumer winning against a delayed copy.
            unsafe { runtime.htod(poison_bytes, neg_dst_p) }.unwrap();
            runtime.synchronize().unwrap();

            let mut consume = runtime.stream().launch_builder(&copy_out);
            consume.arg(&neg_dst_p).arg(&out_p).arg(&n_u64);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };
            // Hold the transfer until the consumer above has read `neg_dst`.
            let consumer_fence = runtime.record_compute_fence().unwrap();
            runtime.copy_wait_fence(consumer_fence).unwrap();
            unsafe { runtime.htod_async(staging.as_slice(), neg_dst_p) }.unwrap();
            let _unused_fence = runtime.record_copy_fence().unwrap();

            let mut raced = vec![0.0f32; n];
            let raced_bytes =
                unsafe { std::slice::from_raw_parts_mut(raced.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(raced_bytes, out_p) }.unwrap();
            // Drain the transfer (which lands after the consumer) before the next
            // iteration reuses `neg_dst` / `staging`.
            runtime.sync_copy_stream().unwrap();
            assert_eq!(
                raced, poison,
                "un-ordered async page-in consumer did NOT read poison — the \
                 compute-stream wait is not load-bearing, so this lock proves nothing"
            );

            // ── Real `upload_async` entry point: allocate + stage + async-copy +
            // fence, then a fenced consumer must observe the byte-identical
            // payload. Keeps the production API (not just its primitive chain)
            // under regression cover.
            let staging = runtime.alloc_pinned(payload_bytes.len()).unwrap();
            let (page, page_fence, staging) = crate::weight_paging::CudaWeightPage::upload_async(
                &runtime,
                DataType::Float32,
                vec![n],
                payload_bytes,
                staging,
            )
            .unwrap();
            runtime.compute_wait_fence(page_fence).unwrap();
            drop(staging);
            let page_p = cuptr(page.device_ptr());
            let mut consume = runtime.stream().launch_builder(&copy_out);
            consume.arg(&page_p).arg(&out_p).arg(&n_u64);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };
            let mut paged = vec![0.0f32; n];
            let paged_bytes =
                unsafe { std::slice::from_raw_parts_mut(paged.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(paged_bytes, out_p) }.unwrap();
            assert_eq!(
                paged, payload,
                "upload_async page-in read stale bytes — the returned copy fence \
                 did not order the transfer before the compute-stream read"
            );
            drop(page);
        }

        ep.deallocate(pos_dst).unwrap();
        ep.deallocate(neg_dst).unwrap();
        ep.deallocate(out).unwrap();
    }
}
