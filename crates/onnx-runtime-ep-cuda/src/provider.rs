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
//! 5. **Bound ownership** — every buffer this EP hands out carries the
//!    binding-issued [`OwningAllocation`] that minted it, and final release
//!    consumes that owner, so a stale handle over a reused device address is
//!    refused instead of freeing a live allocation. A buffer that carries no
//!    binding-issued ownership therefore **fails closed** in `deallocate`: it
//!    is refused rather than freed on the strength of its address alone.
//!
//!    This matters at one ABI boundary. ORT's plugin allocator `Free` callback
//!    receives only a pointer, so
//!    `onnx-runtime-ep-plugin`'s `device_free` rebuilds a raw `DeviceBuffer`
//!    from its own pointer→size table. That reconstructed buffer has no owner,
//!    so this EP refuses it. Wiring that path back up means keeping the
//!    `DeviceBuffer` itself in that table instead of its size — the table is
//!    already keyed by the pointer — which is a change in the plugin crate, not
//!    here. The CUDA plugin cdylib is feature-gated and documented as
//!    unvalidated on hardware (issue #768).

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::{
    BoundBufferOwnership, Cost, DeviceBuffer, DeviceGraphSlot, EpConfig, EpError,
    ExecutionProvider, ExecutionProviderCapabilities, Fence, HostToDeviceCopier, Kernel,
    KernelMatch, LazyWeight, OpRegistry, PagedWeight, Result, WorkspaceAllocation, deny,
    structural_input_bytes,
};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};
use onnx_runtime_memory_governor::{
    AllocationChargeMode, AllocationPublication, AllocationReleaseOutcome, AllocationRequest,
    AllocationSettlementStatus, AllocationSettlementToken, AllocationSettlementWait,
    AllocationStepError, AllocationTransactionError, BindingError, DeviceAllocator, MemoryRole,
    OwningAllocation, ProcessMemoryManager, RegisteredMemoryAuthority, RegisteredMemoryContext,
    RegisteredMemoryHolder, RegisteredMemoryMechanism, ScopedMemoryBinding, ScopedVirtualBacking,
};

use crate::deferred_release::{
    CudaDeferredReleaseQueue, CudaStreamFences, DEFAULT_DEFERRED_RELEASE_CAPACITY, ReleaseObserver,
};
use crate::kernels::build_cuda_registry_with_metrics;
use crate::kernels::csa_checkpoint::CsaMetrics;
use crate::optimizer::cuda_optimization_passes;
use crate::runtime::{CudaRuntime, cuptr};
use crate::weight_paging::{CudaWeightResidency, DeviceOffloadPolicy};

/// The provider-owned mapped-attribution zone.
///
/// This is the governor allowance every allocation in the one suballocating
/// arena is charged against. It lives behind an `Arc` because a *deferred*
/// release refunds it after the provider call that started the release has
/// already returned — possibly after the provider itself is gone.
#[derive(Debug, Default)]
pub(crate) struct CudaMappedAttribution {
    requesters: Mutex<
        HashMap<
            onnx_runtime_memory_governor::MemoryRole,
            onnx_runtime_memory_governor::MappedAllowance,
        >,
    >,
}

impl CudaMappedAttribution {
    fn allowance(
        &self,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Option<onnx_runtime_memory_governor::MappedAllowance> {
        self.requesters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&role)
            .cloned()
    }

    /// Refund the canonical arena zone by an **observed** unmapped byte count.
    ///
    /// Never called with an assumed number: every caller passes the value a
    /// structured release or decommit outcome actually reported.
    fn refund(&self, unmapped: u64) {
        if unmapped == 0 {
            return;
        }
        // The allowance is cloned out under the lock and unmapped outside it, so
        // the governor is never called while this lock is held.
        if let Some(requester) = self.allowance(mapped_attribution_role(
            onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
        )) {
            requester.unmap(unmapped);
        }
    }
}

/// The provider-context resource a binding pins.
///
/// Holding the runtime **and** the deferred queue is what makes a queued
/// release safe: while any allocation of this mechanism is live or queued, the
/// CUDA context, both streams, and the queue that will perform the release all
/// outlive it.
#[derive(Debug)]
struct CudaProviderContextPin {
    #[allow(dead_code)]
    runtime: Arc<CudaRuntime>,
    #[allow(dead_code)]
    queue: Arc<CudaDeferredReleaseQueue>,
}

struct CudaConstructionQueueGuard {
    queue: Arc<CudaDeferredReleaseQueue>,
    armed: bool,
}

impl CudaConstructionQueueGuard {
    fn new(queue: Arc<CudaDeferredReleaseQueue>) -> Self {
        Self { queue, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CudaConstructionQueueGuard {
    fn drop(&mut self) {
        if self.armed {
            // Declared before the allocator, so allocator/reservation teardown
            // has already enqueued everything it owns before this guard runs.
            self.queue.close_after_drain();
            self.queue.poll();
        }
    }
}

/// The accounting-authority resource a binding pins.
#[derive(Debug)]
struct CudaAuthorityPin {
    #[allow(dead_code)]
    device: u32,
}

/// Provider accounting applied from the **actual** structured release outcome.
///
/// This runs on the queue worker after the release completed, so the mapped
/// refund and the free counter follow what happened rather than what the
/// caller hoped would happen. Quarantined ownership refunds only what was
/// really unmapped and is never counted as a physical free.
#[derive(Debug)]
struct CudaReleaseAccounting {
    attribution: Arc<CudaMappedAttribution>,
    frees: Arc<AtomicU64>,
}

impl ReleaseObserver for CudaReleaseAccounting {
    fn released(&self, outcome: &AllocationReleaseOutcome) {
        match outcome {
            AllocationReleaseOutcome::Complete { accounting } => {
                self.attribution.refund(accounting.unmapped_bytes);
                self.frees.fetch_add(1, Ordering::Relaxed);
            }
            AllocationReleaseOutcome::Quarantined { accounting, .. } => {
                // Exactly what the allocator reported as unmapped, and no free
                // count: the bytes it kept stay charged.
                self.attribution.refund(accounting.unmapped_bytes);
            }
            AllocationReleaseOutcome::Failed { .. } => {}
        }
    }
}

/// Compose provider mapped-attribution settlement with the process manager's
/// authority/process charge settlement. Both observe the same structured
/// Phase-4 outcome after the allocator action.
#[derive(Debug)]
struct ManagedCudaReleaseAccounting {
    provider: Arc<dyn ReleaseObserver>,
    settlement: AllocationSettlementToken,
}

impl ReleaseObserver for ManagedCudaReleaseAccounting {
    fn released(&self, outcome: &AllocationReleaseOutcome) {
        self.provider.released(outcome);
        // SAFETY: this observer is stored on the exact queue action carrying the
        // prepared release paired with `settlement`.
        unsafe { self.settlement.settle(outcome) };
    }
}

/// One binding registration for this provider's selected allocator.
struct CudaMemoryBinding {
    /// Dropped before registration handles so allocator/reservation teardown
    /// keeps authority delegation and provider context alive.
    binding: ScopedMemoryBinding,
    mechanism: RegisteredMemoryMechanism,
    holder: RegisteredMemoryHolder,
    context: RegisteredMemoryContext,
    authority: RegisteredMemoryAuthority,
    manager: ProcessMemoryManager,
    cuda_context_identity: usize,
    allocator_teardown_complete: Arc<AtomicBool>,
}

impl std::fmt::Debug for CudaMemoryBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaMemoryBinding")
            .field("binding", &self.binding.identity())
            .finish_non_exhaustive()
    }
}

fn binding_failure(operation: &str, error: BindingError) -> EpError {
    EpError::KernelFailed(format!("cuda_ep: {operation}: {error}"))
}

fn manager_failure(operation: &str, error: AllocationTransactionError) -> EpError {
    EpError::KernelFailed(format!("cuda_ep: {operation}: {error}"))
}

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

/// How many times the device's own VRAM the VMM arena reserves in address
/// space.
///
/// The arena is the single address range that weights, the KV carve and decode
/// scratch all sub-allocate from, so it has to cover their sum *plus* whatever
/// fragmentation they leave behind. Every one of those scales with the card:
/// the metadata-less KV path alone asks for ~1.2x *device free* (#1288). A
/// multiple of VRAM therefore stays correct across accelerators in a way that
/// any constant does not.
const RESERVATION_VRAM_MULTIPLE: usize = 16;

/// Floor for the arena reservation, used as-is when the device's VRAM cannot be
/// queried.
const RESERVATION_FLOOR_BYTES: usize = 1 << 40;

/// Smallest reservation the descending ladder will still accept.
const RESERVATION_MIN_BYTES: usize = 64 << 30;

/// Total VRAM of CUDA device `ordinal`, or `None` if the driver will not say.
///
/// Uses `cuDeviceTotalMem`, which needs only a device handle — no current
/// context — so it is safe to call while the provider is still being built.
fn device_total_memory_bytes(ordinal: u32) -> Option<usize> {
    use cudarc::driver::sys as cu;
    let mut device = 0;
    // SAFETY: the driver is initialized (a `CudaRuntime` for this ordinal
    // already exists); both calls only write through the out-pointers below.
    unsafe {
        if cu::cuDeviceGet(&mut device, ordinal as i32) != cu::CUresult::CUDA_SUCCESS {
            return None;
        }
        let mut bytes = 0usize;
        if cu::cuDeviceTotalMem_v2(&mut bytes, device) != cu::CUresult::CUDA_SUCCESS {
            return None;
        }
        (bytes > 0).then_some(bytes)
    }
}

/// Arena reservation sizes to try, largest first.
///
/// Reserving address space is close to free — an unmapped range claims no
/// physical granules — and the driver is generous with it: on an A100 a single
/// `cuMemAddressReserve` of 64 TiB succeeds, as do eight simultaneous 128 GiB
/// reservations. So the first entry is sized for headroom, not fitted, and the
/// ladder exists only so a platform with a tighter address space still gets a
/// ledgered arena instead of silently dropping to the unaccounted `cuMemAlloc`
/// fallback.
fn reservation_ladder(ordinal: u32) -> Vec<usize> {
    reservation_ladder_from_total(device_total_memory_bytes(ordinal))
}

/// The pure half of [`reservation_ladder`], separated so it can be tested
/// without a device present.
fn reservation_ladder_from_total(device_total: Option<usize>) -> Vec<usize> {
    let desired = device_total
        .and_then(|total| total.checked_mul(RESERVATION_VRAM_MULTIPLE))
        .unwrap_or(RESERVATION_FLOOR_BYTES)
        .max(RESERVATION_FLOOR_BYTES);
    let mut ladder = Vec::new();
    let mut size = desired;
    while size > RESERVATION_MIN_BYTES {
        ladder.push(size);
        size /= 2;
    }
    ladder.push(RESERVATION_MIN_BYTES);
    ladder
}

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

/// Whether this provider runs the governed dynamic KV/weight lending strategy.
///
/// Since Phase 7 this no longer selects *whether* the VMM arena is used -- the
/// arena is the only built-in mechanism and is always built. It selects only
/// how the arena is configured: lending providers get a bounded retained
/// physical-handle pool charged to the governor's authority, and everything
/// else gets the plain arena.
fn auto_dynamic_lending_for(
    governor_present: bool,
    policy: &DeviceOffloadPolicy,
    lending_enabled: bool,
) -> bool {
    governor_present && policy.managed_no_spill && lending_enabled
}

fn validate_offload_policy(policy: &DeviceOffloadPolicy) -> Result<()> {
    if policy.byte_aware_residency {
        return Err(EpError::KernelFailed(
            "cuda_ep: byte-aware weight residency is disabled because real-GPU validation found \
             token-identity corruption; the byte-aware policy must remain disabled"
                .into(),
        ));
    }
    Ok(())
}

/// Refuse an allocator that does not serve `expected_index`.
///
/// Split out of [`CudaExecutionProvider::with_memory`] so the decision can be
/// asserted on a machine with no CUDA device: it is decided entirely from
/// [`DeviceAllocator::device`], and the consequence of getting it wrong — a
/// host or foreign-device pointer handed to a kernel — only shows up inside a
/// launch, far from the substitution that caused it.
///
/// [`DeviceAllocator::device`]: onnx_runtime_memory_governor::DeviceAllocator::device
fn reject_foreign_device(
    expected_index: u32,
    offered: onnx_runtime_memory_governor::DeviceKey,
) -> Option<EpError> {
    let expected = onnx_runtime_memory_governor::DeviceKey::device(expected_index);
    (offered != expected).then(|| {
        EpError::KernelFailed(format!(
            "cuda_ep: this execution provider serves CUDA device {}, but the allocator offered \
             serves {:?} {}; its pointers would not be valid where this EP uses them. Supply an \
             allocator for CUDA device {}.",
            expected.index, offered.tier, offered.index, expected.index
        ))
    })
}

/// Refuse to replace a mechanism that has already handed memory out.
///
/// Injection is authoritative, which means the mechanism it replaces stops
/// existing. Pointers already served belong to that mechanism and have to be
/// released through it, so the replacement is only safe while nothing is
/// outstanding. Checked on **two** axes because they can disagree: `served` is
/// what this provider handed out, `committed` is what the built-in arena has
/// mapped, and something that reached the arena without going through the
/// provider's counters would be invisible on the first axis alone.
///
/// Split out of [`CudaExecutionProvider::with_memory`] so both axes can be
/// asserted without a CUDA device.
fn reject_live_mechanism_replacement(
    expected_index: u32,
    served: u64,
    committed: usize,
) -> Option<EpError> {
    (served > 0 || committed > 0).then(|| {
        EpError::KernelFailed(format!(
            "cuda_ep: the mechanism this execution provider is already using on CUDA device \
             {expected_index} has served {served} allocation(s) and has {committed} bytes \
             committed; `with_memory` replaces the mechanism outright and cannot do so \
             underneath pointers that must still be released through it. Inject the allocator \
             before the provider allocates."
        ))
    })
}

/// The CUDA EP could not build its one built-in memory mechanism.
///
/// # Why this is fatal rather than a fallback
///
/// It used to fall back to an eager `cuMemAlloc` allocator, which meant a
/// machine whose driver cannot do virtual memory management ran anyway — with
/// device allocations that were not charged to the ledger, could not be made
/// during CUDA-graph capture, and behaved unlike every machine the change had
/// been measured on. That is a silent capability downgrade, and the way it
/// surfaced was as an unexplained accounting or capture failure much later.
///
/// Failing here trades that for one loud error at the point the capability is
/// actually missing, naming the boundary and the supported way out.
fn vmm_unavailable(
    ordinal: u32,
    requested_limit: Option<u64>,
    error: impl std::fmt::Display,
) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep: CUDA device {ordinal} cannot provide the virtual memory management (VMM) \
         arena, which is this execution provider's only built-in device memory mechanism: \
         {error}.\nSupport boundary: the arena needs a CUDA driver whose virtual memory \
         management entry points (cuMemAddressReserve, cuMemCreate, cuMemMap, cuMemSetAccess) \
         work on this device; devices and driver builds without them are not supported by the \
         built-in mechanism. The unsupported case is detected at construction by \
         cuMemAddressReserve, whose failure is propagated rather than absorbed; the \
         allocation-granularity query is not a capability check, because a driver refusal or a \
         reported zero is replaced with a 2 MiB default.\nOptions: run on a device \
         and driver that support CUDA VMM, or supply your own allocator — including an eager \
         cuMemAlloc one — through `CudaExecutionProvider::with_memory`, which is unchanged and \
         still honoured.{}",
        requested_limit
            .map(|bytes| format!(
                "\nThis provider was additionally asked for a managed no-spill VRAM limit of \
                 {bytes} bytes, which the arena is required for."
            ))
            .unwrap_or_default()
    ))
}

/// The one allocator mechanism selected for this provider.
///
/// There is exactly one, and which one it is is decided once, at construction.
/// [`Vmm`](Self::Vmm) is the built-in mechanism and is what every provider gets
/// unless a caller explicitly replaces it; [`Injected`](Self::Injected) is a
/// caller-supplied allocator installed through
/// [`CudaExecutionProvider::with_memory`].
///
/// The built-in variant stays concrete so its governor-specific capacity-token
/// path keeps working without caching a second allocator or capability handle.
/// Every generic capability lookup still starts from [`CudaMemory::allocator`].
///
/// There is deliberately no third variant for a *built-in* eager allocator.
/// Storing one beside the arena is what made "which mechanism is live" a
/// question with two answers, and neither the accounting nor the capture-safety
/// invariants held for both.
enum CudaMemory {
    Injected(Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>),
    Vmm(Arc<crate::vmm_allocator::CudaVmmAllocator>),
}

#[derive(Debug)]
struct AllocatorTeardownCompletion {
    done: Arc<AtomicBool>,
}

impl Drop for AllocatorTeardownCompletion {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
    }
}

/// Transparent allocator wrapper whose completion signal fires only after the
/// inner allocator destructor and every queue-producing field have finished.
struct TeardownTrackedAllocator {
    /// Dropped first. Its destructor may enqueue reservation teardown.
    inner: Arc<dyn DeviceAllocator>,
    /// Dropped second, after `inner` destruction returns.
    _completion: AllocatorTeardownCompletion,
}

#[derive(Debug, Default)]
struct WorkspaceReleaseBarrier {
    pending:
        Mutex<HashMap<onnx_runtime_memory_governor::AllocationIdentity, AllocationSettlementWait>>,
}

impl WorkspaceReleaseBarrier {
    fn capture(&self, workspace: &WorkspaceAllocation) -> bool {
        let Some(wait) = workspace.buffer().managed_settlement_wait() else {
            return false;
        };
        self.pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(wait.identity(), wait);
        true
    }

    fn wait(&self, timeout: std::time::Duration) -> Option<AllocationSettlementStatus> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let waits = self
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .values()
                .cloned()
                .collect::<Vec<_>>();
            if waits.is_empty() {
                return None;
            }
            let mut released = Vec::new();
            let mut retained = None;
            let mut pending_status = false;
            for wait in waits {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                match wait.wait(remaining) {
                    AllocationSettlementStatus::Released => released.push(wait.identity()),
                    AllocationSettlementStatus::Retained(state) => {
                        retained.get_or_insert(state);
                    }
                    AllocationSettlementStatus::Pending => pending_status = true,
                }
            }
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for identity in released {
                pending.remove(&identity);
            }
            if let Some(state) = retained {
                return Some(AllocationSettlementStatus::Retained(state));
            }
            if pending_status || std::time::Instant::now() >= deadline {
                return Some(AllocationSettlementStatus::Pending);
            }
            if pending.is_empty() {
                return Some(AllocationSettlementStatus::Released);
            }
            drop(pending);
        }
    }
}

impl std::fmt::Debug for TeardownTrackedAllocator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TeardownTrackedAllocator")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl DeviceAllocator for TeardownTrackedAllocator {
    fn allocate(
        &self,
        bytes: usize,
        align: usize,
    ) -> std::result::Result<NonNull<u8>, onnx_runtime_memory_governor::MemoryError> {
        self.inner.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // SAFETY: delegated under the identical allocator contract.
        unsafe { self.inner.deallocate(ptr, bytes, align) };
    }

    unsafe fn deallocate_with_unmapped(&self, ptr: NonNull<u8>, bytes: usize, align: usize) -> u64 {
        // SAFETY: delegated under the identical allocator contract.
        unsafe { self.inner.deallocate_with_unmapped(ptr, bytes, align) }
    }

    unsafe fn release(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        // SAFETY: delegated under the identical allocator contract.
        unsafe { self.inner.release(ptr, bytes, align) }
    }

    fn device(&self) -> onnx_runtime_memory_governor::DeviceKey {
        self.inner.device()
    }

    fn commits_on_demand(&self) -> bool {
        self.inner.commits_on_demand()
    }

    fn as_virtual_backing(&self) -> Option<&dyn onnx_runtime_memory_governor::VirtualBacking> {
        self.inner.as_virtual_backing()
    }

    fn as_shared_mapping(&self) -> Option<&dyn onnx_runtime_memory_governor::SharedMapping> {
        self.inner.as_shared_mapping()
    }
}

impl CudaMemory {
    fn allocator(&self) -> &dyn onnx_runtime_memory_governor::DeviceAllocator {
        match self {
            Self::Injected(allocator) => allocator.as_ref(),
            Self::Vmm(arena) => arena.as_ref(),
        }
    }

    /// The selected allocator as one shared handle, for binding registration.
    ///
    /// The VMM variant stays concrete in this enum so its governor-specific
    /// capacity path keeps working, but the registry is handed the *same*
    /// object: one mechanism, one coherent allocator, one release path.
    fn allocator_arc(&self) -> Arc<dyn onnx_runtime_memory_governor::DeviceAllocator> {
        match self {
            Self::Injected(allocator) => Arc::clone(allocator),
            Self::Vmm(arena) => {
                Arc::clone(arena) as Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>
            }
        }
    }

    fn vmm(&self) -> Option<&Arc<crate::vmm_allocator::CudaVmmAllocator>> {
        match self {
            Self::Injected(_) => None,
            Self::Vmm(arena) => Some(arena),
        }
    }
}

fn assert_commit_buffer_devices(expected: DeviceId, ranges: &[(&DeviceBuffer, usize, usize)]) {
    for &(buffer, _, _) in ranges {
        assert_eq!(
            buffer.device(),
            expected,
            "cuda_ep: refusing to commit a buffer from device {:?}",
            buffer.device()
        );
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
    /// The built-in mechanism is the VMM arena and nothing else; a caller who
    /// wants something different installs it with
    /// [`with_memory`](Self::with_memory), which replaces the arena outright
    /// rather than sitting beside it.
    memory: CudaMemory,
    governor: Option<Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>>,
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
    /// Set by `shutdown`/`Drop`: no new provider-owned work is accepted, but
    /// already-accepted releases keep running on the queue.
    closed: AtomicBool,
    memory_cleanup_armed: AtomicBool,
    workspace_release_barrier: WorkspaceReleaseBarrier,
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
    /// The mapped-attribution zone every arena allocation is charged against.
    attribution: Arc<CudaMappedAttribution>,
    /// Registry-issued identity for the selected allocator: this is what makes a
    /// release generation-checked instead of pointer-keyed.
    memory_binding: CudaMemoryBinding,
    /// Retired allocator registrations that still have old binding-issued
    /// allocations or queued releases. They share the current context/authority
    /// and are removed by the same drain callback.
    retired_memory_mechanisms: Vec<RegisteredMemoryMechanism>,
    retired_allocator_teardown: Vec<Arc<AtomicBool>>,
    /// The context-owned queue that performs every final device release after
    /// both stream tails.
    release_queue: Arc<CudaDeferredReleaseQueue>,
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
        Self::new_with_policy_governor_and_manager(ordinal, offload_policy, None, None)
    }

    /// Construct a CUDA EP with the device authority available before the
    /// allocator reserves or commits memory.
    pub fn new_with_offload_policy_and_governor(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    ) -> Result<Self> {
        Self::new_with_policy_governor_and_manager(ordinal, offload_policy, Some(governor), None)
    }

    /// Construct with a caller-owned process manager shared by sessions/devices.
    pub fn new_with_offload_policy_governor_and_manager(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
        manager: ProcessMemoryManager,
    ) -> Result<Self> {
        Self::new_with_policy_governor_and_manager(
            ordinal,
            offload_policy,
            Some(governor),
            Some(manager),
        )
    }

    fn new_with_policy_governor_and_manager(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Option<Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>>,
        manager: Option<ProcessMemoryManager>,
    ) -> Result<Self> {
        validate_offload_policy(&offload_policy)?;
        let runtime = Arc::new(CudaRuntime::new(ordinal)?);
        let csa_metrics = Arc::new(CsaMetrics::default());
        let registry = build_cuda_registry_with_metrics(runtime.clone(), csa_metrics.clone());
        let auto_dynamic_lending = auto_dynamic_lending_for(
            governor.is_some(),
            &offload_policy,
            dynamic_lending_enabled(),
        );
        // The queue exists before the allocator does: the VMM arena's
        // reservations hand their teardown to it, so no reservation `Drop`
        // anywhere below can reach a stream synchronizer.
        let release_queue = CudaDeferredReleaseQueue::new(
            Box::new(CudaStreamFences::new(Arc::clone(&runtime))),
            DEFAULT_DEFERRED_RELEASE_CAPACITY,
        );
        // Declared before every allocator/reservation owner below so failure
        // unwinding drops those owners first, then closes the queue they used.
        let mut construction_queue_guard =
            CudaConstructionQueueGuard::new(Arc::clone(&release_queue));
        let attribution = Arc::new(CudaMappedAttribution::default());
        if offload_policy.enabled {
            // Before the pager exists, not after: weights on this runtime may be
            // paged from here on, and a page is retired by `weight_paging`
            // rather than by `deallocate`, so the interleave cache is never told
            // the address died and must refuse to key on one. See
            // [`crate::interleave_cache`].
            runtime.set_weights_may_be_paged();
        }
        let residency = offload_policy.enabled.then(|| {
            let budget = offload_policy
                .device_budget_bytes
                .unwrap_or(DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES);
            Arc::new(
                CudaWeightResidency::new(runtime.clone(), budget)
                    .with_deferred_release_queue(Arc::clone(&release_queue))
                    .with_async_pagein(offload_policy.async_pagein)
                    .with_scan_resistant_dense(offload_policy.scan_resistant_dense)
                    .with_byte_aware_residency(offload_policy.byte_aware_residency)
                    .with_evict_order_probe(offload_policy.evict_order_probe)
                    .with_zero_copy_hybrid(offload_policy.zero_copy_hybrid),
            )
        });
        // Built here rather than at governor adoption, because on the native
        // path the session allocates every tensor it will use while loading --
        // which is before any governor reaches this provider. An arena
        // installed at adoption is installed at the one moment after which
        // nothing will ask it for memory (#659).
        //
        // Address space is free, so the reservation is generous rather than
        // fitted, and running out of *reservation* is a hard failure while
        // leaving it unmapped costs nothing. The size is not fixed, though:
        // a device whose address space will not take the generous ask is a
        // device we should still run on, so the ladder retries smaller.
        //
        // There is no branch here any more. The arena is the only built-in
        // mechanism, so this is a construction that either succeeds or fails
        // the provider -- not a selection between two implementations whose
        // invariants differ.
        let memory = {
            let reservation_queue: Arc<dyn crate::virtual_memory::DeferredReservationQueue> =
                Arc::clone(&release_queue)
                    as Arc<dyn crate::virtual_memory::DeferredReservationQueue>;
            let build_arena = |reservation_bytes: usize| match governor.as_deref() {
                Some(governor) => {
                    crate::vmm_allocator::CudaVmmAllocator::new_with_reservation_queue(
                        runtime.cuda_context(),
                        onnx_runtime_memory_governor::DeviceKey::device(ordinal),
                        ordinal as i32,
                        reservation_bytes,
                        governor,
                        onnx_runtime_memory_governor::HolderId::new(64),
                        onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
                        Arc::clone(&reservation_queue),
                        // Only the dynamic-lending path takes a retained pool by
                        // default. Left deliberately unchanged by the removal of
                        // the eager allocator: the pool is authority-owned, and
                        // handing one to every governed provider would newly
                        // subject governor adoption to the authority-match check
                        // in `adopt_memory_governor`. An explicit
                        // ONNX_GENAI_CUDA_PHYSICAL_HANDLE_POOL_BYTES still opts
                        // in.
                        auto_dynamic_lending.then_some(256usize << 20),
                    )
                }
                None => {
                    crate::vmm_allocator::CudaVmmAllocator::standalone_with_reservation_queue(
                        runtime.cuda_context(),
                        onnx_runtime_memory_governor::DeviceKey::device(ordinal),
                        ordinal as i32,
                        reservation_bytes,
                        onnx_runtime_memory_governor::HolderId::new(64),
                        onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
                        Arc::clone(&reservation_queue),
                        // Standalone (plugin, no-governor) VMM path: retain a
                        // pool of physical granules by default so repeated
                        // same-size ORT scratch requests reuse committed
                        // memory instead of a per-dispatch
                        // cuMemCreate/cuMemRelease churn (#956). An explicit
                        // ONNX_GENAI_CUDA_PHYSICAL_HANDLE_POOL_BYTES still
                        // overrides this. This mirrors the governor path,
                        // which already passes a default pool bound.
                        Some(DEFAULT_STANDALONE_PHYSICAL_POOL_BYTES),
                    )
                }
            };
            // Walk the ladder largest-first and keep the first reservation the
            // driver actually hands us, so a tighter address space costs
            // reservation headroom rather than the provider. Only the last rung
            // failing is fatal: there is no second mechanism to fall back to.
            let mut arena = None;
            let mut last_error = None;
            for reservation_bytes in reservation_ladder(ordinal) {
                match build_arena(reservation_bytes) {
                    Ok(built) => {
                        arena = Some((built, reservation_bytes));
                        break;
                    }
                    Err(error) => last_error = Some(error),
                }
            }
            let (arena, reservation_bytes) = arena.ok_or_else(|| {
                vmm_unavailable(
                    ordinal,
                    offload_policy.managed_limit_bytes,
                    last_error.map_or_else(
                        || String::from("no reservation size was attempted"),
                        |error| error.to_string(),
                    ),
                )
            })?;
            eprintln!(
                "cuda_ep: device allocations go through a VMM arena over {reservation_bytes} \
                 bytes of reserved address space; physical granules are mapped on demand; \
                 strategy={}",
                if auto_dynamic_lending {
                    "vram-limit dynamic KV/weight lending with a retained physical-handle pool"
                } else {
                    "built-in CUDA VMM"
                }
            );
            CudaMemory::Vmm(Arc::new(arena))
        };
        let memory_manager = match manager {
            Some(manager) => manager,
            None => ProcessMemoryManager::new()
                .map_err(|error| binding_failure("cannot create process memory manager", error))?,
        };
        let memory_binding = Self::register_memory_binding(
            memory_manager,
            ordinal,
            &runtime,
            &release_queue,
            governor.clone(),
            memory.allocator_arc(),
        )?;
        let provider = Self {
            device: DeviceId::cuda(ordinal),
            governor: governor.clone(),
            memory,
            ep_allocations: Arc::new(AtomicU64::new(0)),
            ep_frees: Arc::new(AtomicU64::new(0)),
            runtime,
            initialized: false,
            closed: AtomicBool::new(false),
            memory_cleanup_armed: AtomicBool::new(false),
            workspace_release_barrier: WorkspaceReleaseBarrier::default(),
            registry,
            csa_metrics,
            offload_policy,
            residency,
            mapped_reclaim_registration: std::sync::OnceLock::new(),
            attribution,
            memory_binding,
            retired_memory_mechanisms: Vec::new(),
            retired_allocator_teardown: Vec::new(),
            release_queue,
        };
        if let Some(residency) = provider.residency.as_ref() {
            residency
                .install_context_scope(provider.memory_binding.binding.context_scope())
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: cannot install weight-residency context gate: {error}"
                    ))
                })?;
        }
        if let (Some(residency), Some(arena), Some(governor)) =
            (provider.residency.as_ref(), provider.memory.vmm(), governor)
        {
            residency
                .install_vmm_admission(Arc::clone(arena), governor)
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: cannot install committed-byte weight admission: {error}"
                    ))
                })?;
        }
        construction_queue_guard.disarm();
        Ok(provider)
    }

    /// Register the construction-selected allocator with a binding registry.
    ///
    /// Every allocation this provider hands out is issued (or adopted) through
    /// the returned binding, so its identity, generation, authority, and
    /// provider context travel with it to the deferred release. The provider
    /// context resource pins the runtime *and* the deferred queue: a queued
    /// release therefore cannot outlive the CUDA context it needs.
    fn register_memory_binding(
        manager: ProcessMemoryManager,
        ordinal: u32,
        runtime: &Arc<CudaRuntime>,
        release_queue: &Arc<CudaDeferredReleaseQueue>,
        governor: Option<Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>>,
        allocator: Arc<dyn DeviceAllocator>,
    ) -> Result<CudaMemoryBinding> {
        let device = onnx_runtime_memory_governor::DeviceKey::device(ordinal);
        let cuda_context_identity =
            crate::virtual_memory::physical_pool_context_identity(runtime.cuda_context().as_ref())
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: cannot identify CUDA context for memory registration: {error}"
                    ))
                })?;
        let loss_listener: Arc<dyn onnx_runtime_memory_governor::DeviceLossListener> =
            Arc::clone(release_queue) as Arc<dyn onnx_runtime_memory_governor::DeviceLossListener>;
        let registration_generation = manager
            .register_device_loss_listener(device, &loss_listener)
            .map_err(|error| manager_failure("cannot register CUDA device-loss listener", error))?;
        let governed_capacity = governor
            .as_ref()
            .map(|governor| {
                governor
                    .used(onnx_runtime_memory_governor::Tier::Device)
                    .checked_add(governor.available(onnx_runtime_memory_governor::Tier::Device))
                    .ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep: device authority capacity overflows u64".into(),
                        )
                    })
            })
            .transpose()?;
        let context = manager
            .register_provider_context(
                device,
                format!("cuda:{ordinal} provider context"),
                Arc::new(CudaProviderContextPin {
                    runtime: Arc::clone(runtime),
                    queue: Arc::clone(release_queue),
                }),
            )
            .map_err(|error| binding_failure("cannot register the CUDA provider context", error))?;
        let authority_resource = Arc::new(CudaAuthorityPin { device: ordinal });
        let authority = match governor {
            Some(governor) => manager.register_authority(
                device,
                format!("cuda:{ordinal} governed authority"),
                authority_resource,
                governor,
            ),
            None => manager.register_compatibility_authority(
                device,
                format!("cuda:{ordinal} compatibility authority"),
                authority_resource,
            ),
        };
        let authority = match authority {
            Ok(authority) => authority,
            Err(error) => {
                let _ = manager.remove_provider_context(&context);
                return Err(manager_failure(
                    "cannot register the CUDA accounting authority",
                    error,
                ));
            }
        };
        if let Some(capacity) = governed_capacity
            && manager.process_limit(onnx_runtime_memory_governor::Tier::Device) != u64::MAX
            && !authority.has_process_delegation(onnx_runtime_memory_governor::Tier::Device)
            && let Err(error) = manager.delegate_authority_capacity(
                &authority,
                onnx_runtime_memory_governor::Tier::Device,
                capacity,
            )
        {
            let _ = manager.remove_authority(&authority);
            let _ = manager.remove_provider_context(&context);
            return Err(manager_failure(
                "cannot delegate process device capacity to CUDA authority",
                error,
            ));
        }
        let binding = match Self::bind_allocator(
            manager.clone(),
            context.clone(),
            authority.clone(),
            None,
            cuda_context_identity,
            allocator,
        ) {
            Ok(binding) => binding,
            Err(error) => {
                let _ = manager.remove_authority(&authority);
                let _ = manager.remove_provider_context(&context);
                return Err(error);
            }
        };
        if let Err(error) = manager.finish_device_registration(device, registration_generation) {
            let _ = manager.retire(&binding.mechanism);
            let _ = manager.remove_mechanism(&binding.mechanism);
            let _ = manager.unregister_holder(&binding.holder);
            let _ = manager.remove_authority(&authority);
            let _ = manager.remove_provider_context(&context);
            return Err(manager_failure(
                "CUDA device was lost during memory registration",
                error,
            ));
        }
        Ok(binding)
    }

    /// Register `allocator` under an existing context/authority and bind it.
    fn bind_allocator(
        manager: ProcessMemoryManager,
        context: RegisteredMemoryContext,
        authority: RegisteredMemoryAuthority,
        holder: Option<RegisteredMemoryHolder>,
        cuda_context_identity: usize,
        allocator: Arc<dyn DeviceAllocator>,
    ) -> Result<CudaMemoryBinding> {
        let device = context.device();
        let allocator_teardown_complete = Arc::new(AtomicBool::new(false));
        let tracked_allocator: Arc<dyn DeviceAllocator> = Arc::new(TeardownTrackedAllocator {
            inner: allocator,
            _completion: AllocatorTeardownCompletion {
                done: Arc::clone(&allocator_teardown_complete),
            },
        });
        let mechanism = manager
            .register_allocator(
                &context,
                &authority,
                format!("cuda:{} allocator mechanism", device.index),
                tracked_allocator,
            )
            .map_err(|error| manager_failure("cannot register the CUDA allocator", error))?;
        if let Err(error) = manager.select(&mechanism) {
            let _ = manager.retire(&mechanism);
            let _ = manager.remove_mechanism(&mechanism);
            return Err(manager_failure("cannot select the CUDA allocator", error));
        }
        let binding = match manager.bind_registered(&mechanism) {
            Ok(binding) => binding,
            Err(error) => {
                let _ = manager.retire(&mechanism);
                let _ = manager.remove_mechanism(&mechanism);
                return Err(manager_failure("cannot bind the CUDA allocator", error));
            }
        };
        let holder = match holder {
            Some(holder) => holder,
            None => match manager.register_holder(
                &authority,
                format!("cuda:{} execution-provider allocations", device.index),
                None,
            ) {
                Ok(holder) => holder,
                Err(error) => {
                    drop(binding);
                    let _ = manager.retire(&mechanism);
                    let _ = manager.remove_mechanism(&mechanism);
                    return Err(manager_failure(
                        "cannot register the CUDA allocation holder",
                        error,
                    ));
                }
            },
        };
        Ok(CudaMemoryBinding {
            binding,
            mechanism,
            holder,
            context,
            authority,
            manager,
            cuda_context_identity,
            allocator_teardown_complete,
        })
    }

    /// The single allocator selected for this provider.
    fn memory(&self) -> &dyn onnx_runtime_memory_governor::DeviceAllocator {
        self.memory.allocator()
    }

    /// The provider-managed mapped-growth path is intentionally narrower than
    /// generic `VirtualBacking`: it consumes governor capacity through inherent
    /// `CudaVmmAllocator` methods, so only the construction-selected in-tree VMM
    /// is eligible. Capability discovery and the explicit accounting promise
    /// are both checked from that selected allocator reference.
    fn managed_vmm(&self) -> Option<&Arc<crate::vmm_allocator::CudaVmmAllocator>> {
        let arena = self.memory.vmm()?;
        let selected: &dyn onnx_runtime_memory_governor::DeviceAllocator = arena.as_ref();
        (selected.commits_on_demand() && selected.as_virtual_backing().is_some()).then_some(arena)
    }

    /// Wait only for the work that could still be reading device memory *now*.
    ///
    /// A fresh completion event is recorded at the tail of the compute stream
    /// and one at the tail of the copy stream, and the host waits on those two
    /// events. This is deliberately **not** `cuCtxSynchronize`: it does not
    /// touch other streams, other contexts, or work enqueued after this call.
    ///
    /// Only the *explicit* partial-decommit API uses it, because a caller that
    /// asks for a range to be unmapped before it proceeds has asked for a wait.
    /// No `Drop`, and no final release path, ever calls this.
    fn wait_for_recorded_stream_tails(&self, operation: &str) -> Result<()> {
        self.runtime.bind()?;
        let context = self.runtime.cuda_context();
        let mut events = Vec::with_capacity(2);
        for (stream, name) in [
            (self.runtime.stream(), "compute"),
            (self.runtime.copy_stream(), "copy"),
        ] {
            let event = context.new_event(None).map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not create a {name}-stream completion event before \
                     {operation}: {error}"
                ))
            })?;
            event.record(stream).map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not record a {name}-stream completion event before \
                     {operation}: {error}"
                ))
            })?;
            events.push((name, event));
        }
        for (name, event) in events {
            event.synchronize().map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not await the {name}-stream completion event before \
                     {operation}: {error}"
                ))
            })?;
        }
        Ok(())
    }

    /// The context-owned deferred release queue.
    ///
    /// Exposed so tests and diagnostics can observe pending/completed/retained
    /// release state rather than inferring it from a byte count.
    pub fn release_queue(&self) -> &Arc<CudaDeferredReleaseQueue> {
        &self.release_queue
    }

    /// Process manager coordinating this provider's memory registrations and
    /// allocation transactions.
    pub fn process_memory_manager(&self) -> ProcessMemoryManager {
        self.memory_binding.manager.clone()
    }

    /// Confirm externally observed CUDA context termination after device loss.
    ///
    /// This is the only device-loss boundary that discharges retained manager
    /// charges. [`mark_device_lost`](Self::mark_device_lost) alone never refunds.
    pub fn confirm_memory_context_terminated(&self) -> Result<()> {
        self.memory_binding
            .manager
            .confirm_context_terminated(&self.memory_binding.context)
            .map_err(|error| manager_failure("cannot confirm CUDA context termination", error))?;
        if let Some(residency) = self.residency.as_ref() {
            residency.confirm_context_terminated();
        }
        if let Some(authority) = self.memory_binding.authority.memory_authority_id() {
            crate::virtual_memory::confirm_physical_handle_pool_context_terminated(
                self.memory_binding.cuda_context_identity,
                authority,
            );
        }
        self.attribution
            .requesters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.memory_cleanup_armed.store(true, Ordering::Release);
        for mechanism in self
            .retired_memory_mechanisms
            .iter()
            .chain(std::iter::once(&self.memory_binding.mechanism))
        {
            self.memory_binding
                .manager
                .remove_mechanism(mechanism)
                .map_err(|error| {
                    manager_failure("cannot remove a terminated CUDA memory mechanism", error)
                })?;
        }
        self.memory_binding
            .manager
            .unregister_holder(&self.memory_binding.holder)
            .map_err(|error| {
                manager_failure("cannot unregister a terminated CUDA memory holder", error)
            })?;
        self.memory_binding
            .manager
            .remove_provider_context(&self.memory_binding.context)
            .map_err(|error| {
                manager_failure("cannot remove a terminated CUDA provider context", error)
            })?;
        if let Err(error) = self
            .memory_binding
            .manager
            .remove_authority(&self.memory_binding.authority)
            && !matches!(
                error,
                AllocationTransactionError::Binding(BindingError::AuthorityInUse(_))
            )
        {
            return Err(manager_failure(
                "cannot remove a terminated CUDA memory authority",
                error,
            ));
        }
        Ok(())
    }

    /// Structured deferred-release observability for this provider.
    pub fn deferred_release_stats(&self) -> crate::deferred_release::DeferredReleaseStats {
        self.release_queue.stats()
    }

    /// Mark this provider's device lost.
    ///
    /// The binding registry stops issuing work, the queue stops querying the
    /// device, and every pending and future release keeps its ownership and its
    /// accounting instead of being freed or refunded.
    pub fn mark_device_lost(&self, reason: &str) {
        // The manager broadcasts to every sibling provider queue before
        // invalidating any mechanism, so no same-device context is left calling
        // CUDA through a queue that missed the loss.
        let _ = self
            .memory_binding
            .manager
            .invalidate_device(self.memory_binding.mechanism.device(), reason);
    }

    /// The binding-issued owner behind a buffer this provider allocated.
    ///
    /// Foreign and raw-owning buffers fail closed here: without the binding
    /// identity there is no generation to validate, and a capability call on an
    /// address this provider did not issue is exactly the pointer-keyed retry
    /// this phase removes.
    fn bound_owner<'a>(
        &self,
        buffer: &'a DeviceBuffer,
        operation: &str,
    ) -> Result<&'a OwningAllocation> {
        let owner = buffer.bound_owner().ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep: {operation} requires a buffer allocated by this provider's bound \
                 allocator on CUDA device {}; the buffer supplied carries no binding-issued \
                 ownership, so its generation cannot be validated",
                self.device.index
            ))
        })?;
        if owner.identity().binding() != self.memory_binding.binding.identity() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: {operation} refused a buffer issued by a different memory binding \
                 ({:?}) than this provider's ({:?})",
                owner.identity().binding(),
                self.memory_binding.binding.identity()
            )));
        }
        Ok(owner)
    }

    /// This binding's virtual-backing capability, or an actionable error.
    fn bound_virtual_backing(&self, operation: &str) -> Result<Option<ScopedVirtualBacking>> {
        self.memory_binding
            .binding
            .virtual_backing()
            .map_err(|error| manager_failure(&format!("{operation}: capability lookup"), error))
    }

    /// Take device buffers from `memory` instead of from the built-in VMM
    /// arena.
    ///
    /// The same `DeviceAllocator` a caller installs on the CPU EP or the ONNX
    /// Runtime side. Removing the built-in eager allocator did not remove this:
    /// an external mechanism — including an eager `cuMemAlloc` one — is still a
    /// supported thing to run this provider on, it is just not something the
    /// provider ships and silently falls back to.
    ///
    /// # What a successful call means
    ///
    /// It is **authoritative**. The built-in arena selected at construction is
    /// retired and `memory` becomes the one mechanism; nothing is left holding
    /// a second allocator, and no allocation is served by anything other than
    /// what the last successful call installed. A call that cannot be honoured
    /// that way returns an error rather than being quietly ignored.
    ///
    /// # Errors
    ///
    /// If `memory` does not serve this EP's device. Pointers from it are handed
    /// to kernels as this device's addresses, so a host allocator or another
    /// device's allocator would produce an address that is invalid where it is
    /// used. That fails inside a kernel launch, far from the substitution that
    /// caused it, so it is rejected here instead.
    ///
    /// If the mechanism being replaced has already served an allocation. The
    /// replacement is a builder step, not a live swap: pointers already handed
    /// out belong to the old mechanism and must be released through it, so
    /// substituting underneath them would strand them. This is checked *before*
    /// anything is allocated through `memory`, so a rejected call leaves the
    /// provider exactly as it was.
    pub fn with_memory(
        mut self,
        memory: Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>,
    ) -> Result<Self> {
        let key = memory.device();
        let expected = onnx_runtime_memory_governor::DeviceKey::device(self.device.index);
        if let Some(error) = reject_foreign_device(expected.index, key) {
            return Err(error);
        }
        let served = self.ep_allocations.load(Ordering::Relaxed);
        let committed = self
            .memory
            .vmm()
            .map(|arena| arena.committed_and_reserved().0)
            .unwrap_or(0);
        if let Some(error) = reject_live_mechanism_replacement(expected.index, served, committed) {
            return Err(error);
        }
        // The binding is rebuilt *before* the allocator is swapped in, so no
        // allocation can escape through the new allocator while the old binding
        // is still the one that would validate its release. The previous
        // mechanism is retired rather than removed: nothing has been allocated
        // through it yet on this path, and retiring keeps any pins it holds
        // intact instead of asserting that it is quiescent.
        let rebound = Self::bind_allocator(
            self.memory_binding.manager.clone(),
            self.memory_binding.context.clone(),
            self.memory_binding.authority.clone(),
            Some(self.memory_binding.holder.clone()),
            self.memory_binding.cuda_context_identity,
            Arc::clone(&memory),
        )?;
        let previous = std::mem::replace(&mut self.memory_binding, rebound);
        let previous_mechanism = previous.mechanism.clone();
        let mut removed = false;
        if let Err(error) = previous.manager.retire(&previous.mechanism) {
            eprintln!(
                "cuda_ep: WARNING: could not retire the construction-selected allocator binding \
                 after `with_memory`: {error}"
            );
        } else if let Err(error) = previous.manager.remove_mechanism(&previous.mechanism) {
            eprintln!(
                "cuda_ep: WARNING: could not remove the unused construction-selected allocator \
                 binding after `with_memory`: {error}"
            );
        } else {
            removed = true;
        }
        if !removed {
            self.retired_memory_mechanisms.push(previous_mechanism);
            self.retired_allocator_teardown
                .push(Arc::clone(&previous.allocator_teardown_complete));
        }
        self.memory = CudaMemory::Injected(memory);
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

    pub fn initialized_with_offload_policy_governor_and_manager(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
        manager: ProcessMemoryManager,
    ) -> Result<Self> {
        let mut provider = Self::new_with_offload_policy_governor_and_manager(
            ordinal,
            offload_policy,
            governor,
            manager,
        )?;
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
        // Weights on this runtime may now be paged, and a page is retired
        // without passing through `deallocate`. See [`crate::interleave_cache`].
        self.runtime.set_weights_may_be_paged();
        crate::weight_paging::CudaWeightPager::new(Arc::clone(&self.runtime), source)
            .with_deferred_release_queue(Arc::clone(&self.release_queue))
            .with_context_scope(self.memory_binding.binding.context_scope())
    }

    /// Build a bounded-VRAM [`CudaWeightResidency`] (WEIGHT_OFFLOAD Phase 3b
    /// page-in + eviction) sized by `budget_bytes`, sharing this EP's runtime.
    pub fn weight_residency(&self, budget_bytes: u64) -> crate::weight_paging::CudaWeightResidency {
        // Weights on this runtime may now be paged, and a page is retired
        // without passing through `deallocate`. See [`crate::interleave_cache`].
        self.runtime.set_weights_may_be_paged();
        let residency =
            crate::weight_paging::CudaWeightResidency::new(Arc::clone(&self.runtime), budget_bytes)
                .with_deferred_release_queue(Arc::clone(&self.release_queue));
        residency
            .install_context_scope(self.memory_binding.binding.context_scope())
            .expect("a new CUDA weight residency has no context scope");
        residency
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
        self.attribution.refund(unmapped);
    }

    /// The accounting observer a deferred release reports its outcome to.
    fn release_accounting(&self) -> Arc<dyn ReleaseObserver> {
        Arc::new(CudaReleaseAccounting {
            attribution: Arc::clone(&self.attribution),
            frees: Arc::clone(&self.ep_frees),
        })
    }

    /// Allocate through the process manager while leaving stream/capability
    /// execution in this provider.
    ///
    /// `manage_eager_charge` is true for the migrated session workspace path.
    /// Remaining generic callers keep compatibility accounting until they move
    /// to a role/holder-aware transaction; their bytes are reported as
    /// unattributed rather than silently appearing governed.
    fn allocate_transaction(
        &self,
        size: usize,
        alignment: usize,
        committed_ranges: &[std::ops::Range<usize>],
        role: MemoryRole,
        manage_eager_charge: bool,
    ) -> Result<DeviceBuffer> {
        self.ensure_accepting_work("device allocations")?;
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(EpError::AlignmentError);
        }
        let virtual_backing = self.bound_virtual_backing("allocating device memory")?;
        let reserve_bytes = match virtual_backing.as_ref() {
            Some(capability) => capability
                .mapped_bytes_for_allocation(size, alignment)
                .map_err(|error| manager_failure("cannot size mapped allocation", error))?,
            None => size as u64,
        };
        let charge_mode = if self.memory().commits_on_demand() && self.governor.is_some() {
            AllocationChargeMode::AuthorityManaged
        } else if manage_eager_charge && self.governor.is_some() {
            AllocationChargeMode::Managed
        } else {
            AllocationChargeMode::Compatibility
        };
        let delegated = self
            .memory_binding
            .authority
            .has_process_delegation(onnx_runtime_memory_governor::Tier::Device);
        let request = AllocationRequest {
            allocation_bytes: size,
            alignment,
            tier: onnx_runtime_memory_governor::Tier::Device,
            role,
            holder: self.memory_binding.holder.clone(),
            charge_mode,
            authority_reserve_bytes: if charge_mode == AllocationChargeMode::Compatibility {
                0
            } else {
                reserve_bytes
            },
            process_reserve_bytes: if charge_mode == AllocationChargeMode::Managed && !delegated {
                reserve_bytes
            } else {
                0
            },
        };
        let managed = self
            .memory_binding
            .binding
            .allocate_with(
                request,
                |context| match virtual_backing.as_ref() {
                    Some(_) => context.allocate_committed(committed_ranges),
                    None => context.allocate_owning(),
                },
                |owner| {
                    let physical = match virtual_backing.as_ref() {
                        Some(capability) => capability
                            .allocation_committed_bytes(owner)
                            .map_err(|error| AllocationStepError::new(error.to_string()))?
                            as u64,
                        None => size as u64,
                    };
                    Ok(match charge_mode {
                        AllocationChargeMode::Managed => AllocationPublication {
                            charged_bytes: physical,
                            process_reserved_bytes: if delegated { 0 } else { physical },
                            physical_bytes: Some(physical),
                            mapped_bytes: Some(physical),
                            unattributed_bytes: 0,
                            shared_physical: None,
                        },
                        AllocationChargeMode::AuthorityManaged => AllocationPublication {
                            // Generic virtual backing cannot expose how much of
                            // this mapping reused an authority-owned pool.
                            // Authority snapshots remain the charge source.
                            charged_bytes: 0,
                            process_reserved_bytes: 0,
                            physical_bytes: None,
                            mapped_bytes: Some(physical),
                            unattributed_bytes: 0,
                            shared_physical: None,
                        },
                        AllocationChargeMode::Compatibility => {
                            AllocationPublication::compatibility(physical, physical)
                        }
                    })
                },
            )
            .map_err(|error| manager_failure("allocation transaction failed", error))?;
        self.ep_allocations.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceBuffer::from_managed_allocation(managed, self.device))
    }

    fn allocate_with_mapped_growth_for_role(
        &self,
        size: usize,
        alignment: usize,
        grant: onnx_runtime_memory_governor::MappedGrowthGrant,
        role: MemoryRole,
    ) -> Result<DeviceBuffer> {
        self.ensure_accepting_work("device allocations")?;
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(EpError::AlignmentError);
        }
        let full = 0..size;
        let Some(arena) = self.managed_vmm() else {
            return Err(EpError::KernelFailed(
                "cuda_ep: mapped growth requires the construction-selected CUDA VMM allocator; \
                 no second allocator handle or capability downcast is used"
                    .into(),
            ));
        };
        let requested = grant.requested_bytes();
        let grant = std::cell::RefCell::new(Some(grant));
        let additional_owned = std::cell::Cell::new(0_u64);
        let newly_mapped = std::cell::Cell::new(0_u64);
        let managed = self
            .memory_binding
            .binding
            .allocate_with(
                AllocationRequest::authority_managed(
                    size,
                    alignment,
                    onnx_runtime_memory_governor::Tier::Device,
                    role,
                    self.memory_binding.holder.clone(),
                    requested,
                ),
                |context| {
                    let allocation = {
                        let mut grant = grant.borrow_mut();
                        let grant = grant.as_mut().expect("growth grant is live until commit");
                        arena
                            .allocate_committed_with_capacity(
                                size,
                                alignment,
                                std::slice::from_ref(&full),
                                grant.physical_capacity(),
                            )
                            .map_err(AllocationStepError::from)?
                    };
                    additional_owned.set(allocation.additional_owned_bytes);
                    newly_mapped.set(allocation.newly_mapped_bytes);
                    // SAFETY: the arena registered in this binding just returned
                    // this unique live allocation and no other owner exists.
                    match unsafe { context.adopt_allocation(allocation.allocation) } {
                        Ok(owner) => Ok(owner),
                        Err(error) => {
                            // No identity escaped. Try the Phase-4 structured
                            // release immediately; a partial failure must retain
                            // the residual mapped attribution while always
                            // releasing the growth-operation guard.
                            // SAFETY: this is the exact unique allocation just
                            // returned by `arena`.
                            let outcome =
                                unsafe { arena.release(allocation.allocation, size, alignment) };
                            if outcome.is_complete() {
                                Err(error)
                            } else {
                                let grant = grant
                                    .borrow_mut()
                                    .take()
                                    .expect("growth grant remains provisional");
                                let retained_mapped = allocation
                                    .newly_mapped_bytes
                                    .saturating_sub(outcome.unmapped_bytes());
                                let settlement = grant.settle_retained_bytes(retained_mapped);
                                Err(AllocationStepError::retained(format!(
                                    "could not publish mapped-capacity ownership ({error}); \
                                     structured rollback left {} byte(s) retained and {} byte(s) \
                                     mapped{}",
                                    outcome
                                        .residual()
                                        .map_or(size as u64, |residual| residual.retained_bytes),
                                    retained_mapped,
                                    settlement.err().map_or(String::new(), |error| format!(
                                        "; conservative attribution settlement reported: \
                                             {error}"
                                    ))
                                )))
                            }
                        }
                    }
                },
                |_| {
                    let actual = newly_mapped.get();
                    grant
                        .borrow_mut()
                        .take()
                        .expect("growth grant commits once")
                        .commit_bytes(actual)
                        .map_err(AllocationStepError::from)?;
                    Ok(AllocationPublication {
                        charged_bytes: additional_owned.get(),
                        process_reserved_bytes: 0,
                        // Physical handles belong to the authority pool and may
                        // outlive this allocation. Per-allocation residency is
                        // therefore unknown rather than equated to mapping.
                        physical_bytes: None,
                        mapped_bytes: Some(actual),
                        unattributed_bytes: 0,
                        shared_physical: None,
                    })
                },
            )
            .map_err(|error| {
                manager_failure("mapped-growth allocation transaction failed", error)
            })?;
        self.ep_allocations.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceBuffer::from_managed_allocation(managed, self.device))
    }

    /// Refuse new provider-owned work after shutdown or teardown.
    ///
    /// Releases of memory this provider already issued keep working: they are
    /// ownership this provider is *obliged* to settle, not new work.
    fn ensure_accepting_work(&self, operation: &str) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: the execution provider for CUDA device {} has been shut down and no \
                 longer accepts {operation}; already-issued allocations can still be released",
                self.device.index
            )));
        }
        Ok(())
    }

    fn wait_for_workspace_release_barrier(
        barrier: &WorkspaceReleaseBarrier,
        timeout: std::time::Duration,
    ) -> Result<()> {
        match barrier.wait(timeout) {
            None | Some(AllocationSettlementStatus::Released) => Ok(()),
            Some(AllocationSettlementStatus::Pending) => Err(EpError::KernelFailed(format!(
                "cuda_ep: workspace replacement timed out after {timeout:?} waiting for its prior \
                 allocation-specific settlement; this error is retryable and unrelated deferred \
                 releases do not participate in the barrier"
            ))),
            Some(AllocationSettlementStatus::Retained(state)) => {
                Err(EpError::KernelFailed(format!(
                    "cuda_ep: prior workspace release settled as {state:?} with ownership retained; \
                     replacement admission remains closed for those still-charged bytes"
                )))
            }
        }
    }

    /// Drop the device residency cache so every page it holds enqueues its
    /// release. Never waits.
    fn retire_residency(&mut self) {
        let Some(residency) = self.residency.take() else {
            return;
        };
        let outstanding = Arc::strong_count(&residency);
        drop(residency);
        if outstanding > 1 {
            eprintln!(
                "cuda_ep: note: {} other holder(s) of the weight residency cache remain; their \
                 pages release when the last one is dropped",
                outstanding - 1
            );
        }
    }

    /// Retire this provider's mechanism and remove its registry pins after every
    /// normally ordered release has settled.
    fn arm_memory_cleanup(&self) {
        if self.memory_cleanup_armed.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Err(error) = self
            .memory_binding
            .manager
            .retire_context(&self.memory_binding.context)
        {
            eprintln!(
                "cuda_ep: WARNING: could not retire the CUDA memory context during teardown: \
                 {error}"
            );
        }
        if let Err(error) = self
            .memory_binding
            .manager
            .retire(&self.memory_binding.mechanism)
        {
            eprintln!(
                "cuda_ep: WARNING: could not retire the CUDA memory mechanism during teardown: \
                 {error}"
            );
        }
        let manager = self.memory_binding.manager.downgrade();
        let mut mechanisms = self.retired_memory_mechanisms.clone();
        mechanisms.push(self.memory_binding.mechanism.clone());
        let mut allocator_teardown = self.retired_allocator_teardown.clone();
        allocator_teardown.push(Arc::clone(&self.memory_binding.allocator_teardown_complete));
        let holder = self.memory_binding.holder.clone();
        let context = self.memory_binding.context.clone();
        let authority = self.memory_binding.authority.clone();
        self.release_queue.set_drain_callback(move || {
            let Some(manager) = manager.upgrade() else {
                return true;
            };
            let mut retry = false;
            let mut retained = false;
            mechanisms.retain(|mechanism| match manager.remove_mechanism(mechanism) {
                Ok(()) => false,
                Err(AllocationTransactionError::Binding(BindingError::UnregisteredMechanism(
                    _,
                ))) => false,
                Err(AllocationTransactionError::Binding(BindingError::InactiveMechanism {
                    ..
                })) => {
                    retry = true;
                    true
                }
                Err(
                    error @ AllocationTransactionError::Binding(
                        BindingError::QuarantinedOwnership { .. },
                    ),
                ) => {
                    eprintln!(
                        "cuda_ep: WARNING: CUDA memory mechanism teardown remains pinned by \
                         quarantined ownership: {error}"
                    );
                    retained = true;
                    true
                }
                Err(error) => {
                    eprintln!(
                        "cuda_ep: WARNING: could not remove a CUDA memory mechanism after queue \
                         drain: {error}"
                    );
                    retained = true;
                    true
                }
            });
            if retry {
                return false;
            }
            if retained {
                return true;
            }
            if allocator_teardown
                .iter()
                .any(|complete| !complete.load(Ordering::Acquire))
            {
                // The provider/binding still owns an allocator. Its destructor
                // may enqueue reservation teardown, so keep the authority and
                // process delegation pinned until a later post-drop idle pass.
                return false;
            }
            if let Err(error) = manager.unregister_holder(&holder) {
                eprintln!(
                    "cuda_ep: WARNING: could not unregister the CUDA memory holder after queue \
                         drain: {error}"
                );
            }
            if let Err(error) = manager.remove_provider_context(&context) {
                eprintln!(
                    "cuda_ep: WARNING: could not remove the CUDA provider-context pin after queue \
                     drain: {error}"
                );
                return true;
            }
            if let Err(error) = manager.remove_authority(&authority)
                && !matches!(
                    error,
                    AllocationTransactionError::Binding(BindingError::AuthorityInUse(_))
                )
            {
                eprintln!(
                    "cuda_ep: WARNING: could not remove the CUDA authority pin after queue drain: \
                     {error}"
                );
            }
            true
        });
    }

    /// Report deferred-release state that a caller should know about.
    fn report_release_state(&self, phase: &str) {
        let stats = self.release_queue.stats();
        for retained in self.release_queue.retained() {
            eprintln!(
                "cuda_ep: WARNING: at {phase}, {} byte(s) of {} ownership remain retained ({}): \
                 {}",
                retained.bytes, retained.label, retained.state, retained.detail
            );
        }
        if stats.pending > 0 {
            eprintln!(
                "cuda_ep: note: {} deferred device release(s) are still ordered behind in-flight \
                 CUDA work at {phase}; the release queue and CUDA context stay alive until they \
                 complete",
                stats.pending
            );
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

    /// Mint a routed-residency guard for a QMoE-family dispatch. Returns
    /// `Ok(None)` when offload is disabled (nothing pages lazily, so nothing
    /// needs a guard against a resize the executor's default path never
    /// initiates); when offload is enabled this always proves whole-bank
    /// today, because no QMoE or BlockQuantizedMoE kernel in this codebase
    /// surfaces `selected_experts` to the host before or during dispatch —
    /// see [`onnx_runtime_ep_api::prove_routed_residency`] for why that makes
    /// [`onnx_runtime_ep_api::RoutedResidencyRequirement::HostKnownExperts`]
    /// structurally unreachable from here as shipped.
    fn acquire_routed_residency(
        &self,
        _key: u64,
        requirement: onnx_runtime_ep_api::RoutedResidencyRequirement,
        catalog: &onnx_runtime_loader::WeightRegionCatalog,
    ) -> Result<Option<Box<dyn onnx_runtime_ep_api::RoutedResidencyGuardHandle>>> {
        let Some(residency) = self.residency.as_ref() else {
            return Ok(None);
        };
        let guard = residency.acquire_routed_residency(requirement, catalog);
        Ok(Some(Box::new(guard)
            as Box<
                dyn onnx_runtime_ep_api::RoutedResidencyGuardHandle,
            >))
    }

    /// Start an ahead-of-need transfer for a lazy weight the executor's
    /// look-ahead has named (issue #82 cycle 7). `Ok(true)` only when a
    /// `BlockQuantizedMoE` transfer was genuinely started asynchronously and
    /// still needs a later [`Self::page_lazy_weight`] call for the same `key`
    /// to promote it; every other boundary (dense `MatMul`/`MatMulNBits`,
    /// `QMoE`) and every case this increment cannot prove safe returns
    /// `Ok(false)` unchanged, identical to this method not existing.
    fn prefetch_lazy_weight(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn onnx_runtime_ep_api::MmapRegionSource,
    ) -> Result<bool> {
        let Some(residency) = self.residency.as_ref() else {
            return Ok(false);
        };
        residency
            .prefetch_block_quantized_moe(key, weight, source)
            .map_err(|error| EpError::KernelFailed(format!("weight prefetch: {error}")))
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        // The context, stream, and cuBLASLt handle are created eagerly in
        // `new`; binding here confirms the device is reachable on this thread.
        self.runtime.bind()?;
        self.initialized = true;
        Ok(())
    }

    /// Stop accepting provider-owned work and retire what this provider holds.
    ///
    /// This deliberately does **not** wait and does not synchronize the device:
    ///
    /// * new provider-owned work is refused (`closed`), and the queue stops
    ///   accepting new requests;
    /// * device residency is dropped, so every weight page it held enqueues its
    ///   release behind the stream fences recorded at that moment;
    /// * releases already accepted keep running afterwards, because the queue,
    ///   the CUDA context, and both streams are pinned by the queue's own worker
    ///   and by the binding's provider-context resource.
    ///
    /// Ownership that was already observed as retained is reported here rather
    /// than being silently dropped: a caller that shuts down with quarantined
    /// device memory should see it.
    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        // Retire residency so every page it holds enqueues its release behind
        // the stream fences recorded at this moment. Releases already accepted
        // finish afterwards, because the queue's worker, the binding's
        // provider-context pin, and each request's own pins keep the CUDA
        // context and streams alive.
        self.retire_residency();
        self.arm_memory_cleanup();
        // The queue is told to close *once drained*, not now: releases of memory
        // this provider already issued are ownership it is obliged to settle,
        // and settling them is what tears down the allocator and its
        // reservations — which produces the last piece of work the queue must
        // accept.
        self.release_queue.close_after_drain();
        // One non-blocking sweep, so a release whose fences are already complete
        // is reflected in the report below instead of looking pending.
        self.release_queue.poll();
        self.report_release_state("shutdown");
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
        cuda_optimization_passes(Some(self.runtime.capabilities()))
    }

    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer> {
        if dynamic_lending_enabled()
            && let Some(arena) = self.managed_vmm()
            && let Some(governor) = self.governor.as_deref()
            && let Some(requester) = self.attribution.allowance(mapped_attribution_role(
                onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
            ))
        {
            let virtual_backing =
                onnx_runtime_memory_governor::DeviceAllocator::as_virtual_backing(arena.as_ref())
                    .expect("managed_vmm requires VirtualBacking");
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
        grant: onnx_runtime_memory_governor::MappedGrowthGrant,
    ) -> Result<DeviceBuffer> {
        self.allocate_with_mapped_growth_for_role(
            size,
            alignment,
            grant,
            MemoryRole::Workspace { step_scoped: false },
        )
    }

    fn allocate_workspace(
        &self,
        size: usize,
        alignment: usize,
        role: MemoryRole,
    ) -> Result<WorkspaceAllocation> {
        Self::wait_for_workspace_release_barrier(
            &self.workspace_release_barrier,
            std::time::Duration::from_secs(30),
        )?;
        let target_mapped = self.mapped_bytes_for_allocation(size, alignment)?;
        if let Some(grant) = self.prepare_mapped_growth(target_mapped, role)? {
            return self
                .allocate_with_mapped_growth_for_role(size, alignment, grant, role)
                .map(|buffer| WorkspaceAllocation::new(buffer, None));
        }
        let full = 0..size;
        self.allocate_transaction(size, alignment, std::slice::from_ref(&full), role, true)
            .map(|buffer| WorkspaceAllocation::new(buffer, None))
    }

    fn replace_workspace(
        &self,
        old: Option<WorkspaceAllocation>,
        size: usize,
        alignment: usize,
        role: MemoryRole,
    ) -> Result<WorkspaceAllocation> {
        if let Some(old) = old {
            self.deallocate_workspace(old)?;
            Self::wait_for_workspace_release_barrier(
                &self.workspace_release_barrier,
                std::time::Duration::from_secs(30),
            )?;
        }
        self.allocate_workspace(size, alignment, role)
    }

    fn deallocate_workspace(&self, workspace: WorkspaceAllocation) -> Result<()> {
        let captured = self.workspace_release_barrier.capture(&workspace);
        let (buffer, lease) = workspace.into_parts();
        assert!(
            lease.is_none(),
            "CUDA manager-backed workspace must keep accounting in ManagedAllocation, not an outer \
             compatibility lease"
        );
        assert!(
            captured,
            "CUDA workspace must carry allocation-specific manager settlement"
        );
        self.deallocate(buffer)
    }

    fn allocate_committed(
        &self,
        size: usize,
        alignment: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<DeviceBuffer> {
        self.allocate_transaction(
            size,
            alignment,
            committed_ranges,
            MemoryRole::Workspace { step_scoped: false },
            false,
        )
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
        let Some(virtual_backing) = self.bound_virtual_backing("committing an allocation range")?
        else {
            // Eager fallback: the whole allocation was already physically
            // backed, so there is no accounting-changing operation to perform.
            return Ok(());
        };
        // Bound capability call: the binding identity and the allocation
        // generation are validated before the concrete VMM commit runs, so a
        // foreign or stale buffer fails closed instead of committing over
        // whatever now occupies the address.
        let owner = self.bound_owner(buffer, "committing an allocation range")?;
        virtual_backing
            .commit_allocation_range(owner, offset, bytes)
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
        assert_commit_buffer_devices(self.device, ranges);
        let Some(virtual_backing) = self.bound_virtual_backing("committing allocation ranges")?
        else {
            // Same explicit eager fallback as the single-range operation.
            return Ok(());
        };
        let owners = ranges
            .iter()
            .map(|&(buffer, offset, bytes)| {
                Ok((
                    self.bound_owner(buffer, "committing allocation ranges")?,
                    offset,
                    bytes,
                ))
            })
            .collect::<Result<Vec<(&OwningAllocation, usize, usize)>>>()?;
        virtual_backing
            .commit_allocation_ranges(&owners)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not commit {} binding range(s) atomically on CUDA device {}: {error}",
                    owners.len(),
                    self.device.index
                ))
            })
    }

    fn commit_allocation_ranges_with_mapped_growth(
        &self,
        ranges: &[(&DeviceBuffer, usize, usize)],
        grant: &mut onnx_runtime_memory_governor::MappedGrowthGrant,
    ) -> Result<u64> {
        let Some(arena) = self.managed_vmm() else {
            return Err(EpError::KernelFailed(
                "cuda_ep: mapped growth requires the construction-selected CUDA VMM allocator; \
                 injected allocators use their ordinary capability path"
                    .into(),
            ));
        };
        // The capacity-token commit is a specialized arena call, so the bound
        // owner is validated explicitly first: identity, generation, and live
        // record are all checked before any concrete VMM range is built.
        let virtual_backing =
            self.bound_virtual_backing("committing allocation ranges with mapped growth")?;
        let raw = ranges
            .iter()
            .map(|&(buffer, offset, bytes)| {
                let owner =
                    self.bound_owner(buffer, "committing allocation ranges with mapped growth")?;
                if let Some(virtual_backing) = virtual_backing.as_ref() {
                    virtual_backing
                        .allocation_committed_bytes(owner)
                        .map_err(|error| {
                            manager_failure("cannot validate a mapped-growth commit range", error)
                        })?;
                }
                Ok(onnx_runtime_memory_governor::AllocationCommitRange {
                    ptr: owner.as_ptr(),
                    allocation_bytes: owner.len(),
                    align: owner.alignment(),
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
        let Some(virtual_backing) = self.memory().as_virtual_backing() else {
            return Ok(ranges.iter().fold(0_u64, |total, &(_, _, bytes)| {
                total.saturating_add(bytes as u64)
            }));
        };
        let raw = ranges
            .iter()
            .map(|&(buffer, offset, bytes)| {
                let owner = self.bound_owner(buffer, "querying mapped bytes for ranges")?;
                Ok(onnx_runtime_memory_governor::AllocationCommitRange {
                    ptr: owner.as_ptr(),
                    allocation_bytes: owner.len(),
                    align: owner.alignment(),
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
            None => Ok(bytes as u64),
        }
    }

    /// Explicitly unmap part of a live allocation, before the caller proceeds.
    ///
    /// This is the one path that still waits, and it waits on **events**, not
    /// on the device: a completion event is recorded at the tail of the compute
    /// stream and one at the tail of the copy stream and only those two are
    /// awaited. Work submitted afterwards, other streams, and other contexts
    /// are unaffected. The wait is here because the caller asked for the range
    /// to be unmapped *now*; nothing on the deferred final-release path waits at
    /// all.
    ///
    /// The structured decommit outcome is then honoured exactly:
    ///
    /// * `Complete` — refund exactly the bytes the allocator unmapped;
    /// * `RolledBack` — refund nothing, because the mapping was restored;
    /// * `Quarantined` — refund only what the outcome reports as actually
    ///   unmapped, keep the residual charged, and fail hard.
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
        let Some(virtual_backing) = self.bound_virtual_backing("decommitting a range")? else {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: allocator for CUDA device {} has no VirtualBacking capability; \
                 partial decommit is unsupported for eager allocations",
                self.device.index
            )));
        };
        let owner = self.bound_owner(buffer, "decommitting a range")?;
        self.wait_for_recorded_stream_tails("an explicit partial decommit")?;
        let Some(arena) = self.managed_vmm() else {
            // An injected VirtualBacking has no structured decommit outcome;
            // its adapter already refuses on anything but a complete unmap.
            let unmapped = virtual_backing
                .decommit_allocation_range(owner, offset, bytes)
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: could not decommit range {offset}..{} of {} byte allocation on CUDA device {}: {error}",
                        offset.saturating_add(bytes),
                        buffer.len(),
                        self.device.index
                    ))
                })?;
            self.refund_canonical_mapped_zone(unmapped);
            return Ok(unmapped);
        };
        // Validate the generation through the bound capability before the
        // concrete arena call, so a stale or foreign buffer cannot reach the
        // VMM at all.
        virtual_backing
            .allocation_committed_bytes(owner)
            .map_err(|error| manager_failure("cannot validate a decommit range", error))?;
        let outcome = arena
            .decommit_allocation_range_outcome(owner.as_ptr(), owner.len(), offset, bytes)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not decommit range {offset}..{} of {} byte allocation on CUDA device {}: {error}",
                    offset.saturating_add(bytes),
                    buffer.len(),
                    self.device.index
                ))
            })?;
        match outcome {
            crate::vmm_allocator::DecommitOutcome::Complete { accounting } => {
                self.refund_canonical_mapped_zone(accounting.unmapped_bytes);
                Ok(accounting.unmapped_bytes)
            }
            crate::vmm_allocator::DecommitOutcome::RolledBack { reason } => {
                // The mapping was restored, so nothing is refunded and the
                // allocation is exactly as usable as it was.
                Err(EpError::KernelFailed(format!(
                    "cuda_ep: decommit of {offset}..{} was refused and the original mapping was \
                     restored, so the allocation is unchanged and still usable: {reason}",
                    offset.saturating_add(bytes)
                )))
            }
            crate::vmm_allocator::DecommitOutcome::Quarantined {
                accounting,
                residual,
                reason,
            } => {
                // Refund only what the outcome reports as actually unmapped;
                // the residual stays charged because it is still owned.
                self.refund_canonical_mapped_zone(accounting.unmapped_bytes);
                Err(EpError::KernelFailed(format!(
                    "cuda_ep: decommit of {offset}..{} could not be rolled back; the whole \
                     allocation at {:#x} is quarantined with {} byte(s) of physical ownership \
                     retained after {} byte(s) were actually unmapped, and can no longer be used \
                     or released: {reason}",
                    offset.saturating_add(bytes),
                    residual.address,
                    residual.retained_bytes,
                    accounting.unmapped_bytes,
                )))
            }
        }
    }

    fn allocation_committed_bytes(&self, buffer: &DeviceBuffer) -> usize {
        let Some(owner) = buffer.bound_owner() else {
            return buffer.len();
        };
        match self.bound_virtual_backing("querying committed bytes") {
            Ok(Some(virtual_backing)) => virtual_backing
                .allocation_committed_bytes(owner)
                .unwrap_or(buffer.len()),
            _ => buffer.len(),
        }
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()> {
        self.deallocate_with_unmapped(buffer).map(|_| ())
    }

    /// Hand final ownership to the deferred queue, ordered after both streams.
    ///
    /// The returned byte count is `0`, and that is the truthful answer: at the
    /// moment this returns, **nothing has been unmapped**. The physical release
    /// happens later, once the compute and copy completion events recorded here
    /// have both been observed, and the mapped-attribution refund is applied
    /// from that release's actual structured outcome by the queue's accounting
    /// observer. Callers that need the refund must observe it (see
    /// [`CudaExecutionProvider::deferred_release_stats`]) rather than reading it
    /// from this legacy return value.
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
        // This address is about to stop naming this weight. Anything derived
        // from it and keyed by it has to go now, before the allocator can hand
        // the address to the next weight of the same size and let its key match
        // (#1726). Doing it here, rather than at an agreed point in teardown,
        // is what makes it precise -- entries for *other* live executors on
        // this shared provider are untouched, and one of those may have the
        // interleaved pointer baked into a captured graph.
        self.runtime
            .invalidate_interleaved_for(crate::runtime::cuptr(buffer.as_ptr()), buffer.len());
        let ownership = match buffer.into_bound_ownership() {
            Ok(owner) => owner,
            Err(foreign) => {
                // Fail closed: without binding-issued ownership there is no
                // generation to validate, so freeing would be a pointer-only
                // retry over an address this provider cannot prove it owns. The
                // buffer is dropped without being freed, which leaks rather than
                // risking a double free.
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep: refusing to free a {} byte buffer on CUDA device {} that carries no \
                     binding-issued ownership; this provider only releases allocations its own \
                     bound allocator issued",
                    foreign.len(),
                    self.device.index
                )));
            }
        };
        let binding_identity = ownership.owner().identity().binding();
        let known_mechanism = binding_identity.mechanism()
            == self.memory_binding.mechanism.identity()
            || self
                .retired_memory_mechanisms
                .iter()
                .any(|mechanism| mechanism.identity() == binding_identity.mechanism());
        if binding_identity.provider_context() != self.memory_binding.context.identity()
            || !known_mechanism
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: refusing to enqueue release for allocation {:?} on CUDA device {}: its \
                 provider context/mechanism is not owned by this execution provider, so this \
                 provider's stream fences and accounting observer cannot order or settle it",
                ownership.owner().identity(),
                self.device.index
            )));
        }
        let (identity, prepared, settlement, observer) = match ownership {
            BoundBufferOwnership::Binding(owner) => {
                let identity = owner.identity();
                let prepared = owner.prepare_release().map_err(|error| {
                    let (error, _owner) = error.into_parts();
                    // Nothing was mutated and the owner is handed back to its
                    // `Drop`, which quarantines rather than frees.
                    binding_failure("cannot prepare a CUDA allocation release", error)
                })?;
                (identity, prepared, None, Some(self.release_accounting()))
            }
            BoundBufferOwnership::Managed(owner) => {
                let identity = owner.identity();
                let prepared = owner.prepare_release().map_err(|error| {
                    let (error, _owner) = error.into_parts();
                    manager_failure(
                        "cannot prepare a managed CUDA allocation release",
                        AllocationTransactionError::Binding(error),
                    )
                })?;
                // SAFETY: the request and settlement remain paired in the queue
                // observer and in the enqueue-refusal branch below.
                let (prepared, settlement) = unsafe { prepared.into_parts() };
                let observer: Arc<dyn ReleaseObserver> = Arc::new(ManagedCudaReleaseAccounting {
                    provider: self.release_accounting(),
                    settlement: settlement.clone(),
                });
                (identity, prepared, Some(settlement), Some(observer))
            }
        };
        match self.release_queue.enqueue_prepared(prepared, observer) {
            Ok(()) => Ok(0),
            Err(error) => {
                let rejection = error.rejection();
                // The queue refused, so the exact request is quarantined at its
                // mechanism: the bytes stay owned, stay charged, and are never
                // handed out again.
                let outcome = error.quarantine();
                if let Some(settlement) = settlement {
                    // SAFETY: this outcome came from the exact refused request
                    // paired with the settlement token.
                    unsafe { settlement.settle(&outcome) };
                }
                Err(EpError::KernelFailed(format!(
                    "cuda_ep: the deferred release queue refused allocation {identity:?} ({}); \
                     its ownership is quarantined ({}) and {} byte(s) remain charged",
                    rejection.name(),
                    outcome.state(),
                    outcome
                        .residual()
                        .map_or(0, |residual| residual.retained_bytes)
                )))
            }
        }
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
        tie_break: onnx_runtime_ep_api::ArgmaxTieBreak,
    ) -> Result<()> {
        crate::kernels::device_argmax::launch(
            &self.runtime,
            logits,
            elements,
            batch,
            dtype,
            result,
            tie_break.select_last_index(),
        )
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

    fn copy_device_to_device(
        &self,
        src: &DeviceBuffer,
        src_offset: usize,
        dst: &mut DeviceBuffer,
        dst_offset: usize,
        bytes: usize,
    ) -> Result<()> {
        assert_eq!(
            src.device(),
            self.device,
            "cuda_ep::copy_device_to_device: foreign src buffer"
        );
        assert_eq!(
            dst.device(),
            self.device,
            "cuda_ep::copy_device_to_device: foreign dst buffer"
        );
        if bytes == 0 {
            return Ok(());
        }
        let src_end = src_offset.checked_add(bytes).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep::copy_device_to_device: src range overflows".into())
        })?;
        if src_end > src.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_device_to_device: src range {src_offset}..{src_end} exceeds {}",
                src.len()
            )));
        }
        let dst_end = dst_offset.checked_add(bytes).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep::copy_device_to_device: dst range overflows".into())
        })?;
        if dst_end > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_device_to_device: dst range {dst_offset}..{dst_end} exceeds {}",
                dst.len()
            )));
        }
        let src_ptr = cuptr(src.as_ptr())
            .checked_add(src_offset as u64)
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep::copy_device_to_device: src pointer offset overflows".into(),
                )
            })?;
        let dst_ptr = cuptr(dst.as_mut_ptr())
            .checked_add(dst_offset as u64)
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep::copy_device_to_device: dst pointer offset overflows".into(),
                )
            })?;
        // SAFETY: both pointers name checked byte ranges within live device
        // allocations that outlive the enqueued copy; the copy is ordered on the
        // EP stream against the surrounding forward passes that read/write them.
        unsafe { self.runtime.dtod_async(src_ptr, dst_ptr, bytes) }
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

    fn begin_device_graph_capture_in(
        &self,
        slot: DeviceGraphSlot,
        kernels: &[&dyn Kernel],
    ) -> Result<()> {
        self.runtime.begin_graph_capture_in(slot, kernels)
    }

    fn end_device_graph_capture_in(&self, slot: DeviceGraphSlot) -> Result<()> {
        self.runtime.end_graph_capture_in(slot)
    }

    fn abort_device_graph_capture_in(&self, slot: DeviceGraphSlot) -> Result<()> {
        self.runtime.abort_graph_capture_in(slot)
    }

    fn replay_device_graph_in(&self, slot: DeviceGraphSlot) -> Result<()> {
        self.runtime.replay_graph_in(slot)
    }

    fn replay_device_graph_segment_in(&self, slot: DeviceGraphSlot, index: usize) -> Result<()> {
        self.runtime.replay_graph_segment_in(slot, index)
    }

    fn reset_device_graph_in(&self, slot: DeviceGraphSlot) -> Result<bool> {
        // Mirror `reset_device_graph`'s capture-error latch reset for whichever
        // slot is torn down, so re-capture into that slot starts un-poisoned.
        let invalidated = self.runtime.reset_graph_in(slot)?;
        self.runtime.reset_capture_error()?;
        Ok(invalidated)
    }

    fn reset_device_validation_error(&self) -> Result<()> {
        self.runtime.reset_capture_error()
    }

    fn has_device_graph_in(&self, slot: DeviceGraphSlot) -> Result<bool> {
        // The CUDA EP owns one `CudaGraphLifecycle` per slot whose segments can be
        // reset out-of-band (kernel-variant eviction retires kernels baked into a
        // captured graph and resets both slots). Report the real per-slot liveness
        // so the executor re-warms instead of replaying an emptied slot.
        self.runtime.has_graph_executable_in(slot)
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

    fn raw_device_allocation_site_stats(
        &self,
    ) -> Vec<onnx_runtime_ep_api::RawDeviceAllocationSiteStats> {
        self.runtime.raw_allocation_site_stats()
    }

    fn reserve_workspace(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Result<Option<onnx_runtime_memory_governor::MemoryLease>> {
        if self.memory().commits_on_demand() {
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
        if bytes == 0 || !dynamic_lending_enabled() || self.managed_vmm().is_none() {
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
            .attribution
            .requesters
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
    /// False only when a caller has injected an allocator that takes physical
    /// memory at the moment it is asked for, the way `cuMemAlloc` does.
    fn commits_on_demand(&self) -> bool {
        self.memory().commits_on_demand()
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
        if let Some(arena) = self.memory.vmm() {
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
        // `ExecutionProvider::sync` is an explicit host/cross-stream completion
        // boundary, not a trailing per-op eager sync that may be deferred.
        self.runtime.drain_for_unmap()?;
        self.runtime.sync_copy_stream()
    }
}

impl Drop for CudaExecutionProvider {
    /// Initiate the same safe close as `shutdown`, without waiting.
    ///
    /// A provider that was dropped without an explicit shutdown must still stop
    /// accepting work and let its residency enqueue its page releases. It must
    /// not wait for them: the queue's worker holds the queue, the CUDA context,
    /// and every request's ownership, so pending releases complete after this
    /// returns. Nothing here panics, synchronizes, or joins a thread.
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        // Residency is retired first so its pages can still enqueue, then the
        // queue is closed: nothing else can reach this provider afterwards.
        self.retire_residency();
        self.arm_memory_cleanup();
        self.release_queue.close_after_drain();
        self.release_queue.poll();
        self.report_release_state("provider teardown");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use onnx_runtime_ep_api::{
        ExecutionProvider, ExternalMmapRegion, LazyWeight, MmapRegionSource, ResidentWeight,
        WeightHandleError,
    };
    use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, Tier};

    use crate::test_support::EnvVarGuard;

    #[test]
    fn known_unsafe_byte_aware_residency_is_rejected_before_cuda_initialization() {
        let error = validate_offload_policy(&DeviceOffloadPolicy {
            byte_aware_residency: true,
            ..DeviceOffloadPolicy::default()
        })
        .expect_err("known-corrupting residency policy must fail closed");

        assert!(error.to_string().contains("token-identity corruption"));
    }

    struct HostMmap {
        mapping_id: usize,
        bytes: Vec<u8>,
    }

    impl MmapRegionSource for HostMmap {
        fn region_bytes(
            &self,
            region: &ExternalMmapRegion,
        ) -> std::result::Result<&[u8], WeightHandleError> {
            if region.mapping_id != self.mapping_id {
                return Err(WeightHandleError::DeviceBinding(format!(
                    "unknown mapping {}",
                    region.mapping_id
                )));
            }
            let end = region
                .offset
                .checked_add(region.len)
                .ok_or_else(|| WeightHandleError::DeviceBinding("region overflow".into()))?;
            self.bytes
                .get(region.offset..end)
                .ok_or_else(|| WeightHandleError::DeviceBinding("region out of bounds".into()))
        }

        fn full_mapping_bytes(&self, mapping_id: usize) -> Option<&[u8]> {
            (mapping_id == self.mapping_id).then_some(self.bytes.as_slice())
        }
    }

    fn lazy_weight_bytes(bytes: &[u8], offset: usize) -> (LazyWeight, HostMmap) {
        let mapping_id = 71;
        let len = bytes.len();
        let mut backing = vec![0xAB; offset];
        backing.extend_from_slice(bytes);
        let host = HostMmap {
            mapping_id,
            bytes: backing,
        };
        let region = ExternalMmapRegion {
            mapping_id,
            offset,
            len,
        };
        let shape = vec![len];
        let resident = bytes.to_vec();
        let lazy = LazyWeight::block_quantized_moe(DataType::Uint8, shape.clone(), vec![region], {
            let shape = shape.clone();
            move || ResidentWeight::new(DataType::Uint8, shape.clone(), resident.clone())
        })
        .expect("lazy weight");
        (lazy, host)
    }

    /// The bug behind #1288/#1514 was a fixed 64 GiB reservation on a card
    /// whose VRAM is 80 GiB, so the arena could not even span the device it
    /// served. Whatever else the ladder does, its first rung must leave room
    /// for the metadata-less KV carve's ~1.2x *device free*.
    #[test]
    fn reservation_ladder_leads_with_a_large_multiple_of_device_vram() {
        let a100_80gb = 85_094_825_984usize;
        let ladder = reservation_ladder_from_total(Some(a100_80gb));
        assert_eq!(ladder[0], a100_80gb * RESERVATION_VRAM_MULTIPLE);
        assert!(
            ladder[0] > a100_80gb * 2,
            "an arena must span far more than the card it serves, got {} for {a100_80gb} bytes of \
             VRAM",
            ladder[0]
        );
    }

    /// A card small enough that a multiple of its VRAM would be *less* headroom
    /// than the floor must still get the floor: address space is close to free,
    /// so there is no reason to hand a small card a small arena.
    #[test]
    fn reservation_ladder_floors_small_cards_and_unknown_vram() {
        let rtx_4060_8gb = 8usize << 30;
        assert_eq!(
            reservation_ladder_from_total(Some(rtx_4060_8gb))[0],
            RESERVATION_FLOOR_BYTES
        );
        assert_eq!(
            reservation_ladder_from_total(None)[0],
            RESERVATION_FLOOR_BYTES,
            "a driver that will not report VRAM must not collapse the arena"
        );
    }

    /// The ladder exists so a platform with a tighter address space still lands
    /// on a *ledgered* arena rather than the unaccounted `cuMemAlloc` fallback,
    /// which means it has to descend and it has to terminate.
    #[test]
    fn reservation_ladder_descends_by_halves_to_the_minimum() {
        let ladder = reservation_ladder_from_total(Some(85_094_825_984));
        assert!(
            ladder.windows(2).all(|pair| pair[0] > pair[1]),
            "ladder must be strictly descending: {ladder:?}"
        );
        assert_eq!(*ladder.last().unwrap(), RESERVATION_MIN_BYTES);
        assert!(
            ladder.iter().all(|&size| size >= RESERVATION_MIN_BYTES),
            "no rung may drop below the minimum: {ladder:?}"
        );
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn explicit_ep_sync_drains_compute_and_copy_streams_when_eager_sync_is_deferred() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let Ok(ep) = CudaExecutionProvider::initialized(0) else {
            eprintln!("skipping explicit sync test: CUDA EP unavailable");
            return;
        };
        let runtime = ep.runtime();
        runtime.set_defer_eager_sync(true);
        let spin_delay = runtime
            .nvrtc_function(
                "cuda_ep_explicit_sync_test",
                r#"
extern "C" __global__ void spin_delay(long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) { }
}
"#,
                "spin_delay",
            )
            .unwrap();

        let spin: i64 = 100_000_000;
        let mut compute = runtime.stream().launch_builder(&spin_delay);
        compute.arg(&spin);
        unsafe { compute.launch(LaunchConfig::for_num_elems(1)).unwrap() };
        let mut copy = runtime.copy_stream().launch_builder(&spin_delay);
        copy.arg(&spin);
        unsafe { copy.launch(LaunchConfig::for_num_elems(1)).unwrap() };

        let context = runtime.cuda_context();
        let compute_done = context.new_event(None).unwrap();
        compute_done.record(runtime.stream()).unwrap();
        let copy_done = context.new_event(None).unwrap();
        copy_done.record(runtime.copy_stream()).unwrap();
        assert!(
            !compute_done.is_complete() || !copy_done.is_complete(),
            "the delayed work must still be pending before the explicit boundary"
        );

        ExecutionProvider::sync(&ep).unwrap();
        assert!(
            compute_done.is_complete() && copy_done.is_complete(),
            "ExecutionProvider::sync must block until both CUDA streams complete"
        );
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn governed_provider_lazy_weight_page_in_refuses_silent_alloc_raw_fallback_without_a_mapped_allowance()
     {
        let mut env = EnvVarGuard::acquire();
        env.set(
            crate::vmm_allocator::CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            "0",
        );

        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0)));
        let provider = match CudaExecutionProvider::initialized_with_offload_policy_and_governor(
            0,
            DeviceOffloadPolicy {
                enabled: true,
                device_budget_bytes: Some((2usize << 20) as u64),
                ..DeviceOffloadPolicy::default()
            },
            governor,
        ) {
            Ok(provider) => provider,
            Err(error) => {
                eprintln!(
                    "skipping governed lazy-weight fallback test: CUDA EP unavailable ({error})"
                );
                return;
            }
        };
        let residency = provider.residency().expect("weight offload residency");
        assert!(
            residency.stable_va_paging_active(),
            "the governed no-pool VMM path must still install stable-VA weight paging"
        );
        let arena = provider.memory.vmm().expect("built-in VMM allocator");
        assert!(
            arena.physical_pool_stats().is_none(),
            "premise: this covers the no-pool governed VMM path"
        );

        let payload = vec![0x5Au8; 4096];
        let (lazy, host) = lazy_weight_bytes(&payload, 128);
        let before = provider.runtime().allocation_counts();
        let error = ExecutionProvider::page_lazy_weight(&provider, 1, &lazy, &host)
            .expect_err("page-in must fail closed until the mapped allowance is adopted");
        assert!(
            error.to_string().contains("mapped-byte allowance"),
            "the refusal must explain the missing governed allowance: {error}"
        );
        assert_eq!(
            provider.runtime().allocation_counts(),
            before,
            "a refused governed page-in must not silently fall back to alloc_raw"
        );
        assert_eq!(
            residency.stats().page_ins,
            0,
            "a pre-admission refusal must not mutate residency state"
        );
        assert_eq!(
            arena.committed_and_reserved().0,
            0,
            "a refused page-in must not commit any VMM bytes"
        );
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn governed_provider_lazy_weight_paging_uses_vmm_without_raw_alloc_even_without_a_physical_pool()
     {
        let mut env = EnvVarGuard::acquire();
        env.set(
            crate::vmm_allocator::CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            "0",
        );

        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0)));
        let provider_governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync> =
            governor.clone();
        let provider = match CudaExecutionProvider::initialized_with_offload_policy_and_governor(
            0,
            DeviceOffloadPolicy {
                enabled: true,
                device_budget_bytes: Some((2usize << 20) as u64),
                ..DeviceOffloadPolicy::default()
            },
            provider_governor,
        ) {
            Ok(provider) => provider,
            Err(error) => {
                eprintln!("skipping governed lazy-weight VMM test: CUDA EP unavailable ({error})");
                return;
            }
        };
        let residency = provider.residency().expect("weight offload residency");
        assert!(
            residency.stable_va_paging_active(),
            "the governed no-pool VMM path must still install stable-VA weight paging"
        );
        let arena = Arc::clone(provider.memory.vmm().expect("built-in VMM allocator"));
        assert!(
            arena.physical_pool_stats().is_none(),
            "premise: this covers the no-pool governed VMM path"
        );
        provider
            .adopt_memory_governor(governor.as_ref(), Tier::Device, HolderId::new(915))
            .expect("adopt mapped weight allowance");

        let runtime = Arc::clone(provider.runtime());
        let before = runtime.allocation_counts();
        let payload = vec![0x41u8; 4096];
        let (lazy, host) = lazy_weight_bytes(&payload, 256);
        let paged = ExecutionProvider::page_lazy_weight(&provider, 7, &lazy, &host)
            .expect("page-in succeeds")
            .expect("offload enabled");
        assert_eq!(paged.len(), payload.len());
        assert_eq!(
            runtime.allocation_counts(),
            before,
            "governed VMM weight pages must not allocate through alloc_raw"
        );
        assert!(
            arena.committed_and_reserved().0 > 0,
            "the VMM allocator must own committed bytes for the paged weight"
        );
        let stats = residency.stats();
        assert_eq!(stats.page_ins, 1);
        assert_eq!(stats.evictions, 0);
        assert!(
            stats.mapped_physical_bytes > 0,
            "the governed path must account mapped bytes through the weight allowance"
        );

        let queue = Arc::clone(provider.release_queue());
        drop(paged);
        drop(provider);
        assert!(
            queue.wait_until_idle(Duration::from_secs(30)),
            "provider teardown must flush deferred VMM weight releases: {:?}",
            queue.stats()
        );
        assert_eq!(
            runtime.allocation_counts(),
            before,
            "teardown of a governed VMM weight page must not free through free_raw"
        );
        assert_eq!(
            arena.committed_and_reserved().0,
            0,
            "teardown must release the committed VMM bytes after the deferred queue drains"
        );
        assert_eq!(
            queue.stats().quarantined,
            0,
            "teardown must not retain ownership on the success path"
        );
    }

    #[derive(Debug)]
    struct WorkspaceTestPin;

    fn managed_host_workspace(bytes: usize) -> WorkspaceAllocation {
        use onnx_runtime_memory_governor::{
            AllocationPublication, AllocationRequest, DeviceKey, HostAllocator, LeaseLedger,
            LedgerGovernor, MemoryGovernor, MemoryRole, ProcessMemoryManager, Tier,
        };

        let manager = ProcessMemoryManager::new().unwrap();
        let context = manager
            .register_provider_context(DeviceKey::HOST, "test context", Arc::new(WorkspaceTestPin))
            .unwrap();
        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            4096,
            0,
        )));
        let authority = manager
            .register_authority(
                DeviceKey::HOST,
                "test authority",
                Arc::new(WorkspaceTestPin),
                governor as Arc<dyn MemoryGovernor + Send + Sync>,
            )
            .unwrap();
        let holder = manager
            .register_holder(&authority, "test workspace", None)
            .unwrap();
        let mechanism = manager
            .register_allocator(
                &context,
                &authority,
                "host allocator",
                Arc::new(HostAllocator),
            )
            .unwrap();
        let owner = manager
            .bind_registered(&mechanism)
            .unwrap()
            .allocate(
                AllocationRequest::managed(
                    bytes,
                    16,
                    Tier::Host,
                    MemoryRole::Workspace { step_scoped: true },
                    holder,
                    bytes as u64,
                ),
                AllocationPublication::exclusive(bytes as u64, bytes as u64, bytes as u64),
            )
            .unwrap();
        WorkspaceAllocation::new(
            DeviceBuffer::from_managed_allocation(owner, DeviceId::cpu()),
            None,
        )
    }

    fn release_managed_workspace(workspace: WorkspaceAllocation) {
        let (buffer, lease) = workspace.into_parts();
        assert!(lease.is_none());
        let BoundBufferOwnership::Managed(owner) = buffer.into_bound_owner().unwrap() else {
            panic!("test workspace lost manager ownership");
        };
        assert!(owner.release_now().unwrap().is_complete());
    }

    #[derive(Debug)]
    struct TestReleaseFence(bool);

    impl crate::deferred_release::ReleaseFence for TestReleaseFence {
        fn is_complete(&self) -> bool {
            self.0
        }
    }

    #[derive(Debug)]
    struct SequencedFenceSource {
        recorded: std::sync::atomic::AtomicUsize,
    }

    impl crate::deferred_release::ReleaseFenceSource for SequencedFenceSource {
        fn record(
            &self,
        ) -> std::result::Result<Vec<Box<dyn crate::deferred_release::ReleaseFence>>, String>
        {
            let index = self
                .recorded
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(vec![Box::new(TestReleaseFence(index != 0))])
        }
    }

    #[derive(Debug)]
    struct UnrelatedRelease;

    impl crate::deferred_release::DeferredReleaseAction for UnrelatedRelease {
        fn execute(self: Box<Self>) -> crate::deferred_release::DeferredActionOutcome {
            crate::deferred_release::DeferredActionOutcome::released(0)
        }

        fn label(&self) -> &'static str {
            "unrelated"
        }
    }

    #[derive(Debug)]
    struct TestSettlementObserver(AllocationSettlementToken);

    impl ReleaseObserver for TestSettlementObserver {
        fn released(&self, outcome: &AllocationReleaseOutcome) {
            // SAFETY: this observer is enqueued with the exact prepared release
            // paired with this token below.
            unsafe { self.0.settle(outcome) };
        }
    }

    fn enqueue_managed_workspace_release(
        queue: &CudaDeferredReleaseQueue,
        workspace: WorkspaceAllocation,
    ) {
        let (buffer, lease) = workspace.into_parts();
        assert!(lease.is_none());
        let BoundBufferOwnership::Managed(owner) = buffer.into_bound_owner().unwrap() else {
            panic!("test workspace lost manager ownership");
        };
        let prepared = owner.prepare_release().unwrap();
        // SAFETY: the request and token remain paired in the observer.
        let (request, settlement) = unsafe { prepared.into_parts() };
        queue
            .enqueue_prepared(request, Some(Arc::new(TestSettlementObserver(settlement))))
            .unwrap();
    }

    #[test]
    fn workspace_barrier_ignores_unrelated_pending_queue_entries() {
        let queue = CudaDeferredReleaseQueue::manual(
            Box::new(SequencedFenceSource {
                recorded: std::sync::atomic::AtomicUsize::new(0),
            }),
            4,
        );
        queue.enqueue(UnrelatedRelease).unwrap();

        let barrier = WorkspaceReleaseBarrier::default();
        let workspace = managed_host_workspace(128);
        assert!(barrier.capture(&workspace));
        enqueue_managed_workspace_release(&queue, workspace);
        assert_eq!(queue.pending(), 2);
        assert_eq!(queue.poll(), 1, "only the workspace release is ready");

        CudaExecutionProvider::wait_for_workspace_release_barrier(
            &barrier,
            std::time::Duration::from_millis(10),
        )
        .unwrap();
        assert_eq!(
            queue.pending(),
            1,
            "the same queue still holds the unrelated deferred release"
        );
    }

    #[test]
    fn workspace_barrier_timeout_is_retryable_after_specific_settlement() {
        let barrier = WorkspaceReleaseBarrier::default();
        let workspace = managed_host_workspace(128);
        assert!(barrier.capture(&workspace));

        assert!(
            CudaExecutionProvider::wait_for_workspace_release_barrier(
                &barrier,
                std::time::Duration::ZERO,
            )
            .is_err()
        );
        release_managed_workspace(workspace);
        CudaExecutionProvider::wait_for_workspace_release_barrier(
            &barrier,
            std::time::Duration::from_millis(10),
        )
        .expect("a later admission retries the same allocation-specific settlement");

        let later = managed_host_workspace(128);
        assert!(barrier.capture(&later));
        release_managed_workspace(later);
        CudaExecutionProvider::wait_for_workspace_release_barrier(
            &barrier,
            std::time::Duration::from_millis(10),
        )
        .expect("a transient timeout must not permanently disable later workspaces");
    }

    #[test]
    fn workspace_barrier_keeps_every_concurrent_release_identity() {
        let barrier = WorkspaceReleaseBarrier::default();
        let first = managed_host_workspace(128);
        let second = managed_host_workspace(128);
        assert!(barrier.capture(&first));
        assert!(barrier.capture(&second));

        release_managed_workspace(second);
        assert_eq!(
            barrier.wait(std::time::Duration::ZERO),
            Some(AllocationSettlementStatus::Pending),
            "one released workspace must not erase another pending identity"
        );
        release_managed_workspace(first);
        assert_eq!(
            barrier.wait(std::time::Duration::from_millis(10)),
            Some(AllocationSettlementStatus::Released)
        );
    }

    #[test]
    #[should_panic(expected = "refusing to commit a buffer from device")]
    fn eager_batched_commit_still_rejects_a_foreign_device_buffer() {
        let foreign = unsafe {
            DeviceBuffer::from_raw_parts(
                std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
                DeviceId::cuda(1),
                64,
                16,
            )
        };
        assert_commit_buffer_devices(DeviceId::cuda(0), &[(&foreign, 0, 16)]);
    }

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

    /// Only an explicit managed no-spill policy selects the governed lending
    /// strategy; everything else gets the plain arena.
    ///
    /// This predicate used to decide whether VMM was used at all. It no longer
    /// does -- the arena is unconditional since Phase 7 -- so what is asserted
    /// here is narrower than the name it used to carry: it picks the retained
    /// pool configuration, not the mechanism.
    #[test]
    fn only_explicit_managed_policy_selects_the_governed_lending_strategy() {
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

    /// The built-in mechanism failing is fatal, and the diagnostic has to be
    /// usable by the person holding the machine it failed on.
    ///
    /// Before Phase 7 this was only fatal under a managed no-spill limit; every
    /// other path logged a warning and silently ran on an eager `cuMemAlloc`
    /// allocator whose allocations were never charged to the ledger. There is
    /// no longer a second mechanism to degrade to, so "unsupported" has to be
    /// said out loud, at the point it is discovered.
    #[test]
    fn built_in_vmm_failure_is_fatal_and_names_the_support_boundary() {
        let error = vmm_unavailable(3, None, "cuMemAddressReserve: CUDA_ERROR_NOT_SUPPORTED");
        let message = error.to_string();
        assert!(
            message.contains("CUDA device 3"),
            "the diagnostic must name the device it failed on: {message}"
        );
        assert!(
            message.contains("cuMemAddressReserve: CUDA_ERROR_NOT_SUPPORTED"),
            "the diagnostic must carry what the driver actually said: {message}"
        );
        assert!(
            message.contains("only built-in device memory mechanism"),
            "the diagnostic must say there is nothing to fall back to: {message}"
        );
        assert!(
            message.contains("Support boundary:")
                && message.contains("cuMemCreate")
                && message.contains("cuMemMap")
                && message.contains("cuMemSetAccess")
                && message.contains("granularity"),
            "the diagnostic must state the documented capability boundary: {message}"
        );
        assert!(
            message.contains("with_memory"),
            "the diagnostic must point at the supported way to run without the built-in \
             mechanism: {message}"
        );
    }

    /// A requested managed no-spill limit is still named, because that caller
    /// asked for a guarantee the arena is the only way to keep.
    #[test]
    fn a_requested_managed_limit_is_named_in_the_unavailability_diagnostic() {
        let without = vmm_unavailable(0, None, "driver refused").to_string();
        assert!(
            !without.contains("6442450944"),
            "no limit was requested, so none may be invented: {without}"
        );
        let with = vmm_unavailable(0, Some(6 << 30), "driver refused").to_string();
        assert!(
            with.contains("6442450944 bytes"),
            "the requested VRAM limit must appear in the diagnostic: {with}"
        );
        assert!(
            with.contains("managed no-spill"),
            "the diagnostic must say which promise the limit belongs to: {with}"
        );
    }

    /// Criterion 4, on the axes that do not need a GPU. An injected mechanism
    /// is either selected authoritatively or refused *before* anything is
    /// allocated through it — never accepted and then ignored.
    #[test]
    fn injection_is_refused_for_a_device_this_provider_does_not_serve() {
        let host = onnx_runtime_memory_governor::HostAllocator.device();
        assert_ne!(
            host,
            onnx_runtime_memory_governor::DeviceKey::device(0),
            "premise: the host allocator must not claim to serve CUDA device 0"
        );
        let refused = reject_foreign_device(0, host).expect("host memory is not CUDA device 0");
        let message = refused.to_string();
        assert!(
            message.contains("CUDA device 0"),
            "the refusal must name the device that was expected: {message}"
        );
        assert!(
            reject_foreign_device(0, onnx_runtime_memory_governor::DeviceKey::device(1)).is_some(),
            "another CUDA device's allocator is refused too, not just host memory"
        );
        assert!(
            reject_foreign_device(0, onnx_runtime_memory_governor::DeviceKey::device(0)).is_none(),
            "the matching device must be accepted, or injection is refused for everyone and \
             criterion 4 is met vacuously"
        );
    }

    /// Both axes are load-bearing. `served` alone would miss an arena that
    /// committed memory outside the provider's own counters, and `committed`
    /// alone would miss an injected mechanism, which has no arena to ask.
    #[test]
    fn replacing_a_mechanism_that_already_served_memory_is_refused_on_both_axes() {
        assert!(
            reject_live_mechanism_replacement(0, 0, 0).is_none(),
            "a fresh provider must accept injection, or the guard refuses everything"
        );
        let by_allocations = reject_live_mechanism_replacement(0, 1, 0)
            .expect("an allocation was served, so the mechanism cannot be swapped")
            .to_string();
        assert!(
            by_allocations.contains("served 1 allocation(s)"),
            "the refusal must report what is outstanding: {by_allocations}"
        );
        let by_commitment = reject_live_mechanism_replacement(0, 0, 2 << 20)
            .expect("the arena has memory mapped, so the mechanism cannot be swapped")
            .to_string();
        assert!(
            by_commitment.contains("2097152 bytes committed"),
            "the refusal must report committed bytes even when no allocation was counted: \
             {by_commitment}"
        );
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn public_constructor_installs_configured_physical_pool() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let mut env = EnvVarGuard::acquire();
        env.set(
            crate::vmm_allocator::CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            &(64usize << 20).to_string(),
        );
        let provider = CudaExecutionProvider::new(0).expect("public CUDA provider");
        assert!(
            provider
                .memory
                .vmm()
                .is_some_and(|arena| arena.physical_pool_stats().is_some()),
            "public constructor must use the configured physical pool"
        );
        let stats = provider
            .memory
            .vmm()
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
            .expect("the pooled return is ordered behind the delayed writer");
        // `deallocate` enqueues; it does not free. The physical handle only
        // returns to the pool once the queue's completion events on both
        // streams have fired, which is also what makes the delayed kernel's
        // write below observable through the reused mapping. Reading
        // `pool_hits` -- or allocating again and expecting a hit -- before the
        // queue settles reads the pool one step too early.
        assert!(
            provider
                .release_queue()
                .wait_until_idle(std::time::Duration::from_secs(30)),
            "the deferred release queue must drain before the pool can hand the handle back: {:?}",
            provider.deferred_release_stats()
        );
        let second = provider.allocate(4, 256).expect("reused allocation");
        assert_eq!(
            stats.snapshot().pool_hits,
            1,
            "the drained release must have returned its handle to the pool"
        );
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
    /// `standalone_with_reservation_queue` with the provider's deferred release
    /// queue and `DEFAULT_STANDALONE_PHYSICAL_POOL_BYTES` — directly, so the
    /// measurement establishes its condition instead of depending on a
    /// process-global env var (measurement-discipline #906).
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
        // The production standalone path: reservation teardown is owned by the
        // provider's deferred queue, so no reservation `Drop` under this arena
        // synchronizes a stream.
        let reservation_queue: Arc<dyn crate::virtual_memory::DeferredReservationQueue> =
            Arc::clone(provider.release_queue())
                as Arc<dyn crate::virtual_memory::DeferredReservationQueue>;
        let arena = crate::vmm_allocator::CudaVmmAllocator::standalone_with_reservation_queue(
            runtime.cuda_context(),
            onnx_runtime_memory_governor::DeviceKey::device(0),
            0,
            64 << 30,
            onnx_runtime_memory_governor::HolderId::new(64),
            onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
            reservation_queue,
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

    /// Criterion 1 and 8, on the plugin construction path: the provider built
    /// exactly as the CUDA plugin builds it (`CudaExecutionProvider::new_default`
    /// == `new(0)`) routes every device allocation — including the ORT scratch
    /// the plugin projects through `allocate`/`deallocate` — through the pooled
    /// VMM arena.
    ///
    /// No environment variable is set here, and that is the point: before
    /// Phase 7 this test had to opt in with `ONNX_GENAI_CUDA_VMM=1` or it would
    /// have measured the eager `cuMemAlloc` path instead (#956).
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn plugin_construction_path_routes_device_memory_through_pooled_vmm_arena() {
        let provider =
            CudaExecutionProvider::new(0).expect("plugin-path CUDA provider under VMM arena");
        assert!(
            provider.commits_on_demand(),
            "the VMM arena, not the cuMemAlloc path, must serve allocations on the plugin path"
        );
        let stats = provider
            .memory
            .vmm()
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
            .attribution
            .requesters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(zone_role, zone_allowance);

        // Deallocation is now deferred behind both stream tails, so the refund
        // arrives when the queue executes the release rather than when
        // `deallocate` returns. Every assertion below therefore *observes* the
        // queue instead of reading the legacy byte count, which is truthfully
        // zero: nothing is unmapped at the moment `deallocate` returns.
        let drain = || {
            assert!(
                provider
                    .release_queue()
                    .wait_until_idle(std::time::Duration::from_secs(30)),
                "the deferred release queue must drain: {:?}",
                provider.deferred_release_stats()
            );
        };

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
                .attribution
                .allowance(zone_role)
                .expect("canonical arena allowance");
            assert!(requester.mapped_bytes() > 0);
            (governed, ordinary, requester)
        };

        let (governed, ordinary, requester) = allocate_pair();
        let mapped = requester.mapped_bytes();
        assert_eq!(
            provider.deallocate_with_unmapped(governed).unwrap(),
            0,
            "nothing is unmapped before the deferred release runs"
        );
        drain();
        assert_eq!(
            requester.mapped_bytes(),
            mapped,
            "the ordinary neighbor still holds the shared granule"
        );
        assert_eq!(provider.deallocate_with_unmapped(ordinary).unwrap(), 0);
        drain();
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
        provider.deallocate(ordinary).expect("ordinary cleanup");
        drain();
        assert_eq!(requester.mapped_bytes(), mapped);
        provider.deallocate(governed).expect("governed cleanup");
        drain();
        assert_eq!(requester.mapped_bytes(), 0);

        // Once the arena zone exists, ordinary-only allocation is admitted,
        // charged, and refunded by the same provider-owned path.
        for _ in 0..3 {
            let ordinary = provider.allocate(4096, 256).expect("ordinary allocation");
            assert!(requester.mapped_bytes() > 0);
            provider.deallocate(ordinary).expect("ordinary cleanup");
            drain();
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
            // Explicit partial decommit still completes synchronously: it waits
            // on freshly recorded compute/copy completion events, not on the
            // device, and refunds exactly what it unmapped.
            assert_eq!(
                provider
                    .decommit_allocation_range(&buffer, granule as usize, granule as usize,)
                    .expect("partial decommit"),
                granule
            );
            assert_eq!(requester.mapped_bytes(), granule);
            assert_eq!(provider.deallocate_with_unmapped(buffer).unwrap(), 0);
            drain();
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
        provider.deallocate(ordinary).expect("ordinary cleanup");
        drain();
        assert_eq!(requester.mapped_bytes(), 0);
        provider.deallocate(governed).expect("governed cleanup");
        drain();
        let stats = provider.deferred_release_stats();
        assert_eq!(stats.pending, 0);
        assert_eq!(
            stats.quarantined, 0,
            "no release may end in retained ownership here: {stats:?}"
        );
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn provider_drop_defers_handle_release_until_in_flight_work_completes() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor};

        let mut env = EnvVarGuard::acquire();
        env.set(
            crate::vmm_allocator::CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            &(64usize << 20).to_string(),
        );
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
            .memory
            .vmm()
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
        // The queue outlives the provider, which is the whole point: it holds
        // the request, the allocator, and the CUDA context until the release is
        // ordered after the kernel below.
        let queue = Arc::clone(first.release_queue());
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
        first
            .deallocate(allocation)
            .expect("the free is accepted while the kernel runs");
        // Teardown no longer drains the streams. The handle must not be reusable
        // yet, because the kernel writing through it has not finished.
        drop(first);
        assert_eq!(
            stats.snapshot().pool_hits,
            0,
            "no handle may be reused before the release is ordered after the kernel"
        );
        assert!(
            queue.wait_until_idle(std::time::Duration::from_secs(60)),
            "the deferred release completes once the kernel does: {:?}",
            queue.stats()
        );
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
            "the old handle was released after the kernel, never reused early"
        );
        second.deallocate(later).expect("final deallocation");
        assert!(
            second
                .release_queue()
                .wait_until_idle(std::time::Duration::from_secs(60)),
            "the second provider's release completes: {:?}",
            second.deferred_release_stats()
        );
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
            let (page, page_fence, staging) =
                crate::weight_paging::CudaWeightPage::upload_async_queued(
                    &runtime,
                    DataType::Float32,
                    vec![n],
                    payload_bytes,
                    staging,
                    Arc::clone(ep.release_queue()),
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

    /// Regression for the MTP dual-slot graph-replay crash: a kernel-variant
    /// eviction resets BOTH the `Primary` (M=1 decode) and `Verify` (M=K
    /// speculative) EP graph slots out-of-band while the executor's host-side
    /// capture signature/schedule stay live. The next replay would then hit an
    /// emptied slot and hard-error ("no executable is installed"). The executor's
    /// pre-replay guard relies on [`ExecutionProvider::has_device_graph_in`]
    /// reporting the real per-slot liveness so it can re-warm instead of crashing.
    /// This locks that contract: after an out-of-band reset of both slots (exactly
    /// what `evict_surplus_variants` does), the trait method reports no executable.
    #[test]
    fn has_device_graph_in_tracks_out_of_band_slot_eviction() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use onnx_runtime_ep_api::{CaptureSupport, Kernel, TensorMut, TensorView};

        const MODULE: &str = "cuda_ep_slot_eviction_test";
        const SOURCE: &str = r#"
extern "C" __global__ void add_one(const float* x, float* y, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + 1.0f;
}
"#;

        struct CapturableKernel;
        impl Kernel for CapturableKernel {
            fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
                Ok(())
            }
            fn capture_support(&self) -> CaptureSupport {
                CaptureSupport::Supported
            }
        }

        let Ok(ep) = CudaExecutionProvider::initialized(0) else {
            eprintln!("skipping slot-eviction liveness test: CUDA EP unavailable");
            return;
        };
        let runtime = ep.runtime().clone();
        let add_one = runtime.nvrtc_function(MODULE, SOURCE, "add_one").unwrap();

        let n = 32usize;
        let size = n * std::mem::size_of::<f32>();
        let p_in = runtime.alloc_raw(size).unwrap();
        let p_out = runtime.alloc_raw(size).unwrap();
        let v_in = runtime.alloc_raw(size).unwrap();
        let v_out = runtime.alloc_raw(size).unwrap();

        let launch = |src, dst| {
            let n_u64 = n as u64;
            let mut builder = runtime.stream().launch_builder(&add_one);
            builder.arg(&src).arg(&dst).arg(&n_u64);
            // SAFETY: signature `(const float*, float*, u64)`; both pointers
            // cover `n` f32 elements and the launch bounds-checks `n`.
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap();
            }
        };

        // Install a captured graph into each slot (as the M=1 base decode and the
        // M=K verify forward do).
        let kernels: [&dyn Kernel; 1] = [&CapturableKernel];
        ep.begin_device_graph_capture_in(DeviceGraphSlot::Primary, &kernels)
            .unwrap();
        launch(p_in, p_out);
        ep.end_device_graph_capture_in(DeviceGraphSlot::Primary)
            .unwrap();
        ep.begin_device_graph_capture_in(DeviceGraphSlot::Verify, &kernels)
            .unwrap();
        launch(v_in, v_out);
        ep.end_device_graph_capture_in(DeviceGraphSlot::Verify)
            .unwrap();

        // Both slots hold a replayable executable.
        assert!(
            ep.has_device_graph_in(DeviceGraphSlot::Primary).unwrap(),
            "Primary must report an installed graph after capture"
        );
        assert!(
            ep.has_device_graph_in(DeviceGraphSlot::Verify).unwrap(),
            "Verify must report an installed graph after capture"
        );

        // Kernel-variant eviction resets BOTH slots out-of-band (mirrors
        // `evict_surplus_variants`), without touching the executor's host state.
        ep.reset_device_graph_in(DeviceGraphSlot::Primary).unwrap();
        ep.reset_device_graph_in(DeviceGraphSlot::Verify).unwrap();

        // The liveness signal the executor's pre-replay guard reads must now show
        // both slots emptied, so it re-warms instead of replaying nothing.
        assert!(
            !ep.has_device_graph_in(DeviceGraphSlot::Primary).unwrap(),
            "Primary must report no executable after out-of-band eviction"
        );
        assert!(
            !ep.has_device_graph_in(DeviceGraphSlot::Verify).unwrap(),
            "Verify must report no executable after out-of-band eviction"
        );

        // SAFETY: both slots reset, dropping all graph ownership before frees.
        unsafe {
            runtime.free_raw(v_out).unwrap();
            runtime.free_raw(v_in).unwrap();
            runtime.free_raw(p_out).unwrap();
            runtime.free_raw(p_in).unwrap();
        }
    }
}
