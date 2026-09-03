//! Shared CUDA runtime state: the driver context, its dedicated stream, and vendor
//! library backends. One [`CudaRuntime`] is created per
//! [`CudaExecutionProvider`] and shared (via `Arc`) into every kernel the
//! provider hands out, so the whole EP drives a single device + stream.

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use arc_swap::{ArcSwap, ArcSwapOption};
use cudarc::driver::sys::{
    CUdevice_attribute, CUdeviceptr, CUfunction, CUfunction_attribute_enum, CUmodule,
};
use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream, LaunchConfig};

use onnx_runtime_ep_api::EpError;
use onnx_runtime_ep_api::Kernel;
use onnx_runtime_ep_api::{
    DeviceGraphOwner, DeviceGraphResource, DeviceGraphSlot, DeviceGraphToken,
    DeviceValidationOwner, DeviceValidationRegistration, DeviceValidationToken,
};
use onnx_runtime_ep_api::{RawDeviceAllocationSiteStats, Result};

use crate::blas::CublasLt;
use crate::byte_telemetry::{
    EventSpec, ObservedBoundary, ObservedCategory, ObservedStatus, PendingObservedBatch,
    ProductionByteRecorder, current_recorder,
};
use crate::cudnn::CudnnBackend;
use crate::dynamic_library::{CudaLibrary, require, wheel_cuda_include_paths};
use crate::error::{driver_err, nvrtc_err};
use crate::graph::CudaGraphLifecycle;
use crate::kernel_cache;
use onnx_runtime_cuda_memory::capture_gate;

/// Counts explicit device allocation/free calls made through a runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaAllocationCounts {
    pub allocations: u64,
    pub frees: u64,
}

/// Counts explicit host/device transfers made through a runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaTransferCounts {
    pub host_to_device: u64,
    pub device_to_host: u64,
    /// Stream-ordered asynchronous host→device copies issued on the dedicated
    /// transfer stream by [`CudaRuntime::htod_async`] (Phase-4 weight prefetch).
    pub async_host_to_device: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaTransferByteCounts {
    pub h2d_attempted: u64,
    pub h2d_completed: u64,
    pub d2h_attempted: u64,
    pub d2h_completed: u64,
    pub d2d_attempted: u64,
    pub d2d_completed: u64,
    pub memset_attempted: u64,
    pub memset_completed: u64,
}

const VALIDATION_PHASE_BITS: u32 = 3;
const VALIDATION_PHASE_MASK: u64 = (1 << VALIDATION_PHASE_BITS) - 1;
const VALIDATION_MAX_GENERATION: u64 = u64::MAX >> VALIDATION_PHASE_BITS;

// Invariant: one coordinator word and each owner's one slot word are the sole
// authority for generation phase, result visibility, and cleanup ownership.
fn next_validation_runtime_id() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .unwrap_or_else(|_| {
            panic!("CUDA validation runtime identity space exhausted; refusing ABA reuse")
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
enum ValidationPhase {
    Idle = 0,
    Resetting = 1,
    Preparing = 2,
    Attaching = 3,
    Active = 4,
    Consuming = 5,
}

fn validation_word(phase: ValidationPhase, generation: u64) -> u64 {
    (generation << VALIDATION_PHASE_BITS) | phase as u64
}

fn validation_phase(word: u64) -> ValidationPhase {
    match word & VALIDATION_PHASE_MASK {
        0 => ValidationPhase::Idle,
        1 => ValidationPhase::Resetting,
        2 => ValidationPhase::Preparing,
        3 => ValidationPhase::Attaching,
        4 => ValidationPhase::Active,
        5 => ValidationPhase::Consuming,
        _ => unreachable!("all validation phase values are encoded by this module"),
    }
}

fn validation_generation(word: u64) -> u64 {
    word >> VALIDATION_PHASE_BITS
}

fn take_validation_generation(next: &AtomicU64) -> Result<u64> {
    next.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
        (generation < VALIDATION_MAX_GENERATION).then_some(generation + 1)
    })
    .map_err(|_| {
        EpError::KernelFailed(
            "cuda_ep: device validation generation space exhausted; rebuild the provider".into(),
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
enum ValidationSlotPhase {
    Idle = 0,
    Pending = 1,
    Complete = 2,
    Retired = 3,
}

fn validation_slot_word(phase: ValidationSlotPhase, generation: u64) -> u64 {
    (generation << VALIDATION_PHASE_BITS) | phase as u64
}

fn validation_slot_phase(word: u64) -> ValidationSlotPhase {
    match word & VALIDATION_PHASE_MASK {
        0 => ValidationSlotPhase::Idle,
        1 => ValidationSlotPhase::Pending,
        2 => ValidationSlotPhase::Complete,
        3 => ValidationSlotPhase::Retired,
        _ => unreachable!("all validation slot phase values are encoded by this module"),
    }
}

#[derive(Debug)]
struct CudaValidationRegistration {
    runtime_id: u64,
    slot: usize,
    retired: bool,
}

#[derive(Debug)]
struct DeviceValidationSlot {
    owner: DeviceValidationOwner,
    state: AtomicU64,
    flags: AtomicU32,
    next: AtomicPtr<DeviceValidationSlot>,
}

impl DeviceValidationSlot {
    fn new(owner: DeviceValidationOwner) -> Self {
        Self {
            owner,
            state: AtomicU64::new(validation_slot_word(ValidationSlotPhase::Idle, 0)),
            flags: AtomicU32::new(0),
            next: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

/// Whether a failed measured H2D operation can still be using its source and
/// destination.
#[derive(Debug)]
pub enum FailedHtodCompletion {
    /// No copy was successfully submitted.
    NotSubmitted,
    /// A fallback copy-stream synchronization proved the submitted copy ended.
    Completed(CopyCompleted),
    /// A copy was submitted and the fallback synchronization also failed.
    MayBeInFlight,
}

/// Failure from [`CudaRuntime::htod_async_elapsed_ms`] with ordering evidence.
#[derive(Debug)]
pub struct HtodAsyncElapsedError {
    detail: String,
    completion: FailedHtodCompletion,
}

impl HtodAsyncElapsedError {
    pub fn into_parts(self) -> (String, FailedHtodCompletion) {
        (self.detail, self.completion)
    }
}

impl std::fmt::Display for HtodAsyncElapsedError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for HtodAsyncElapsedError {}

fn classify_submitted_htod_failure(
    error: String,
    synchronization: std::result::Result<(), String>,
) -> HtodAsyncElapsedError {
    match synchronization {
        Ok(()) => HtodAsyncElapsedError {
            detail: format!("{error}; copy-stream completion was established before rollback"),
            completion: FailedHtodCompletion::Completed(CopyCompleted::new()),
        },
        Err(sync_error) => HtodAsyncElapsedError {
            detail: format!(
                "{error}; cuStreamSynchronize(copy) could not establish completion: {sync_error}"
            ),
            completion: FailedHtodCompletion::MayBeInFlight,
        },
    }
}

fn nvrtc_include_paths() -> Vec<String> {
    let mut candidates = Vec::<PathBuf>::new();
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            candidates.push(PathBuf::from(root).join("include"));
        }
    }
    candidates.push(PathBuf::from("/usr/local/cuda/include"));
    candidates.extend(wheel_cuda_include_paths());

    if let Some(paths) = std::env::var_os("LD_LIBRARY_PATH") {
        for path in std::env::split_paths(&paths) {
            // Recover the wheel root through the shared helper rather than
            // matching one layout's spelling: this previously only recognised
            // `nvidia/cuda_nvrtc/lib`, so a consolidated wheel's loader path
            // never yielded its sibling headers.
            if let Some(root) = onnx_genai_cuda_version_guard::wheel_root_of(&path) {
                candidates.extend(
                    onnx_genai_cuda_version_guard::WHEEL_CUDA_MAJORS
                        .iter()
                        .map(|major| root.join("nvidia").join(major).join("include")),
                );
                candidates.push(root.join("nvidia/cuda_runtime/include"));
            }
        }
    }

    candidates.sort();
    candidates.dedup();
    candidates
        .into_iter()
        // A directory earns its place by carrying headers NVRTC actually needs.
        // Two different ones qualify: `cuda_fp16.h` for the half-precision
        // kernels, and `crt/mma.h` for the tensor-core ones. They live in one
        // directory in a toolkit install but in two separate wheels
        // (`nvidia-cuda-runtime` and `nvidia-cuda-nvcc`), so testing only for
        // `cuda_fp16.h` would silently drop the half that `mma.h` needs.
        .filter(|path| path.join("cuda_fp16.h").is_file() || path.join("crt/mma.h").is_file())
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

fn ptx_arch_for(major: u32, minor: u32) -> String {
    format!("compute_{major}{minor}")
}

fn cubin_arch_for(major: u32, minor: u32) -> String {
    format!("sm_{major}{minor}")
}

const SAFE_MAX_THREADS_PER_BLOCK_FALLBACK: u32 = 256;
const SAFE_SHARED_MEMORY_PER_BLOCK_FALLBACK: u32 = 48 * 1024;
const SAFE_MAX_GRID_DIM_X_FALLBACK: u32 = 65_535;

/// Hardware limits used to select portable CUDA launch configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaDeviceCapabilities {
    compute_capability: (u32, u32),
    max_threads_per_block: u32,
    max_grid_dim_x: u32,
    max_shared_memory_per_block: u32,
    max_shared_memory_per_block_optin: u32,
    multiprocessor_count: u32,
    l2_cache_size: u32,
}

impl CudaDeviceCapabilities {
    fn from_reported_limits(
        compute_capability: (u32, u32),
        max_threads_per_block: Option<u32>,
        max_grid_dim_x: Option<u32>,
        max_shared_memory_per_block: Option<u32>,
        max_shared_memory_per_block_optin: Option<u32>,
        multiprocessor_count: Option<u32>,
        l2_cache_size: Option<u32>,
    ) -> Self {
        let max_threads_per_block = max_threads_per_block
            .filter(|&value| value > 0)
            .unwrap_or(SAFE_MAX_THREADS_PER_BLOCK_FALLBACK);
        let max_grid_dim_x = max_grid_dim_x
            .filter(|&value| value > 0)
            .unwrap_or(SAFE_MAX_GRID_DIM_X_FALLBACK);
        let max_shared_memory_per_block = max_shared_memory_per_block
            .filter(|&value| value > 0)
            .unwrap_or(SAFE_SHARED_MEMORY_PER_BLOCK_FALLBACK);
        let max_shared_memory_per_block_optin = max_shared_memory_per_block_optin
            .filter(|&value| value > 0)
            .unwrap_or(max_shared_memory_per_block)
            .max(max_shared_memory_per_block);
        let multiprocessor_count = multiprocessor_count.filter(|&value| value > 0).unwrap_or(1);
        // L2 size is a hint only (Ada L2-residency tiling); 0 means "unknown",
        // which every consumer must treat as "no L2-residency assumptions".
        let l2_cache_size = l2_cache_size.filter(|&value| value > 0).unwrap_or(0);
        Self {
            compute_capability,
            max_threads_per_block,
            max_grid_dim_x,
            max_shared_memory_per_block,
            max_shared_memory_per_block_optin,
            multiprocessor_count,
            l2_cache_size,
        }
    }

    pub fn compute_capability(self) -> (u32, u32) {
        self.compute_capability
    }

    pub fn max_shared_memory_per_block_optin(self) -> u32 {
        self.max_shared_memory_per_block_optin
    }

    pub fn max_threads_per_block(self) -> u32 {
        self.max_threads_per_block
    }

    pub fn max_grid_dim_x(self) -> u32 {
        self.max_grid_dim_x
    }

    pub fn multiprocessor_count(self) -> u32 {
        self.multiprocessor_count
    }

    /// Device L2 cache size in bytes, or `0` when the driver did not report it.
    /// Used as a *hint* for the pending Ada L2-residency tiling lever; a `0`
    /// here must be treated as "make no L2-residency assumptions".
    pub fn l2_cache_size(self) -> u32 {
        self.l2_cache_size
    }

    /// Arch tier this device maps to (see [`crate::arch::ArchTier`]). The
    /// mapping is total: every compute capability resolves to a tier without
    /// panicking, and `sm_90` resolves to [`crate::arch::ArchTier::Hopper`].
    pub fn arch_tier(self) -> crate::arch::ArchTier {
        crate::arch::ArchTier::from_compute_capability(self.compute_capability)
    }

    /// Default, tier-derived kernel configuration hints for this device. This is
    /// pure scaffolding for the pending RTX/arch kernels: nothing in the live
    /// kernel-selection path consumes it yet, so it cannot change today's
    /// behavior on any device (see [`crate::arch::ArchConfig`]).
    pub fn arch_config(self) -> crate::arch::ArchConfig {
        crate::arch::ArchConfig::for_capabilities(self)
    }

    /// Per-SM resident-warp estimate for the int4/accuracy_level=4 decode
    /// GEMV's one-wave occupancy math, delegated to the arch layer (see
    /// [`crate::arch::decode_resident_warps_per_sm`]). Byte-identical to the
    /// ladder the decode selectors used inline before, so consuming it here does
    /// not change selection on any device (`sm_90` → 64).
    pub fn decode_resident_warps_per_sm(self) -> u32 {
        crate::arch::decode_resident_warps_per_sm(self.compute_capability)
    }

    /// Test-only constructor: synthesize capabilities for an arbitrary arch tier
    /// so the SM-dispatch scaffolding can be exercised without that hardware.
    #[cfg(test)]
    pub(crate) fn for_test(
        compute_capability: (u32, u32),
        multiprocessor_count: u32,
        l2_cache_size: u32,
    ) -> Self {
        Self::from_reported_limits(
            compute_capability,
            None,
            None,
            None,
            None,
            Some(multiprocessor_count),
            Some(l2_cache_size),
        )
    }
}

/// The process-wide CUDA context for `ordinal`, created at most once.
///
/// `CudaContext::new` retains the device's **primary** context, so every
/// `CudaRuntime` on a device already shares one `CUcontext` — caching the
/// `Arc` changes no semantics. What it removes is the retain/release churn:
/// when the last `Arc` for a device dropped, cudarc released the primary
/// context, and a primary-context teardown synchronizes the device. That
/// invalidates any CUDA graph capture in progress on another thread, which is
/// how a test doing nothing but constructing and dropping a runtime could break
/// an unrelated test's capture. Holding one reference for the life of the
/// process keeps the refcount off zero and takes the teardown off the table.
///
/// The entries are intentionally never evicted. A released primary context is
/// exactly the hazard being avoided, and a handful of per-device contexts is
/// bounded by the machine's device count.
fn shared_context(ordinal: u32) -> Result<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<u32, Arc<CudaContext>>>> = OnceLock::new();
    let contexts = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut contexts = contexts.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(context) = contexts.get(&ordinal) {
        return Ok(Arc::clone(context));
    }
    // Creating a context synchronizes the device; see `CudaRuntime::alloc_raw`.
    let _section = capture_gate::synchronizing_section();
    let context =
        CudaContext::new(ordinal as usize).map_err(|e| driver_err("CudaContext::new", e))?;
    contexts.insert(ordinal, Arc::clone(&context));
    Ok(context)
}

fn positive_attribute(context: &CudaContext, attribute: CUdevice_attribute) -> Option<u32> {
    context
        .attribute(attribute)
        .ok()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|&value| value > 0)
}

fn reduction_launch_params(
    preferred_threads: u32,
    max_threads: u32,
    bytes_per_thread: u32,
    max_dynamic_shared_memory: u32,
) -> Option<(u32, u32)> {
    if preferred_threads == 0 || max_threads == 0 || bytes_per_thread == 0 {
        return None;
    }
    let threads_by_shared_memory = max_dynamic_shared_memory / bytes_per_thread;
    let thread_limit = preferred_threads
        .min(max_threads)
        .min(threads_by_shared_memory);
    if thread_limit == 0 {
        return None;
    }
    let threads = 1 << (31 - thread_limit.leading_zeros());
    Some((threads, threads * bytes_per_thread))
}

/// Decide how a dynamic shared-memory request maps onto a device's per-block
/// budgets. `default_budget` is the non-opt-in ceiling (~48&nbsp;KB on every
/// architecture) and `optin_budget` the device-specific opt-in ceiling, both
/// already net of the kernel's static shared memory. Returns:
/// * `Err(())` — the request exceeds even the opt-in ceiling, so no launch on
///   this GPU can satisfy it and the caller must route to a portable fallback.
/// * `Ok(None)` — the request fits the default budget; launch as-is.
/// * `Ok(Some(bytes))` — the request needs the function opted into `bytes` of
///   dynamic shared memory before it can launch.
fn dynamic_shared_memory_optin(
    requested_bytes: u32,
    default_budget: u32,
    optin_budget: u32,
) -> std::result::Result<Option<u32>, ()> {
    if requested_bytes > optin_budget {
        return Err(());
    }
    if requested_bytes > default_budget {
        Ok(Some(requested_bytes))
    } else {
        Ok(None)
    }
}

/// Device context, stream, and vendor-library backends shared across the EP.
/// Environment override for the `alloc_raw` pool bound, in bytes.
///
/// `0` disables pooling, restoring one `cuMemAlloc`/`cuMemFree` pair per
/// request — which is what to set when bisecting a suspected reuse bug.
pub const CUDA_RAW_POOL_BYTES_ENV: &str = "ONNX_GENAI_CUDA_RAW_POOL_BYTES";

/// Enable source-attributed profiling for [`CudaRuntime::alloc_raw`].
///
/// The disabled path performs no map lookup or locking. This is intentionally
/// opt-in because pool hits may occur on kernel dispatch paths.
pub const CUDA_RAW_ALLOCATION_PROFILE_ENV: &str = "ONNX_GENAI_PROFILE_CUDA_RAW_ALLOCATIONS";

/// Default bound on device bytes held in the `alloc_raw` pool.
///
/// Sized for the transient scratch one prefill chunk holds rather than for the
/// model, so pooling never competes with weights for a meaningful share of the
/// device.
const DEFAULT_RAW_POOL_BYTES: u64 = 2 << 30;

fn raw_pool_limit_bytes() -> u64 {
    std::env::var(CUDA_RAW_POOL_BYTES_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_RAW_POOL_BYTES)
}

fn raw_allocation_profile_enabled() -> bool {
    std::env::var_os(CUDA_RAW_ALLOCATION_PROFILE_ENV).is_some_and(|value| {
        matches!(
            value.to_string_lossy().trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RawAllocationSite {
    file: &'static str,
    line: u32,
}

#[derive(Debug, Default)]
struct RawAllocationProfile {
    enabled: bool,
    sites: Mutex<BTreeMap<RawAllocationSite, RawDeviceAllocationSiteStats>>,
}

impl RawAllocationProfile {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            sites: Mutex::new(BTreeMap::new()),
        }
    }

    fn record(
        &self,
        location: &'static std::panic::Location<'static>,
        requested: usize,
        class: usize,
        pool_hit: bool,
    ) {
        if !self.enabled {
            return;
        }
        let key = RawAllocationSite {
            file: location.file(),
            line: location.line(),
        };
        let mut sites = self
            .sites
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let site = sites
            .entry(key)
            .or_insert_with(|| RawDeviceAllocationSiteStats {
                file: key.file,
                line: key.line,
                ..RawDeviceAllocationSiteStats::default()
            });
        site.requests = site.requests.saturating_add(1);
        site.requested_bytes = site.requested_bytes.saturating_add(requested as u64);
        if pool_hit {
            site.pool_hits = site.pool_hits.saturating_add(1);
            site.pool_hit_bytes = site.pool_hit_bytes.saturating_add(class as u64);
        } else {
            site.driver_allocations = site.driver_allocations.saturating_add(1);
            site.driver_bytes = site.driver_bytes.saturating_add(class as u64);
        }
    }

    fn snapshot(&self) -> Vec<RawDeviceAllocationSiteStats> {
        self.sites
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .cloned()
            .collect()
    }
}

/// Round a raw allocation to the size class the pool keys on.
///
/// Rounding up is what lets a recycled block satisfy a slightly different
/// request without ever being too small, and keeps the number of distinct
/// classes bounded when shapes vary a little between chunks.
fn raw_pool_size_class(bytes: usize) -> usize {
    const SMALL: usize = 1 << 20;
    if bytes <= SMALL {
        bytes.next_power_of_two().max(512)
    } else {
        bytes.div_ceil(SMALL) * SMALL
    }
}

/// Source of [`CudaRuntime::runtime_id`]. Monotonic and never reset, so an id
/// is never handed to a second runtime.
static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

struct RawCudaModule {
    module: CUmodule,
    context: Arc<CudaContext>,
}

// Mirrors cudarc's `CudaModule`: the driver module is context-owned and its
// handle may be used from threads that bind that context.
unsafe impl Send for RawCudaModule {}
unsafe impl Sync for RawCudaModule {}

impl Drop for RawCudaModule {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        // SAFETY: this module was loaded exactly once by `load_raw_module` and
        // the runtime cache is its sole owner.
        let _ = unsafe { cudarc::driver::result::module::unload(self.module) };
    }
}

/// Allocation-free launch handle for prepared decode kernels.
///
/// cudarc's safe launch builder allocates argument/event vectors on every
/// launch. This handle retains the loaded module but accepts a fixed stack
/// parameter array, so warmed eager execution and graph recording perform no
/// host allocation.
#[derive(Clone)]
pub(crate) struct RawCudaFunction {
    function: CUfunction,
    _module: Arc<RawCudaModule>,
}

// CUDA function handles are immutable and cudarc gives its equivalent wrapper
// the same cross-thread guarantees.
unsafe impl Send for RawCudaFunction {}
unsafe impl Sync for RawCudaFunction {}

impl RawCudaFunction {
    /// Launch with caller-owned scalar storage and a stack-backed parameter
    /// pointer array.
    ///
    /// # Safety
    /// Every entry in `kernel_params` must point to a live value matching the
    /// loaded kernel ABI, and all referenced device storage must remain live
    /// until the stream has consumed the launch.
    pub(crate) unsafe fn launch(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        kernel_params: &mut [*mut c_void],
    ) -> std::result::Result<(), cudarc::driver::result::DriverError> {
        // SAFETY: upheld by this method's caller contract.
        unsafe {
            cudarc::driver::result::launch_kernel(
                self.function,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream.cu_stream(),
                kernel_params,
            )
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    doc = r#"
The isolated capture-error reset is intentionally absent from ordinary builds.
The runtime type remains publicly usable while the test-only mutation does not:

```
use onnx_runtime_ep_cuda::CudaRuntime;

fn accepts_runtime(_: &CudaRuntime) {}
```

```compile_fail,E0599
use onnx_runtime_ep_cuda::CudaRuntime;

fn production_cannot_reset_capture_error(runtime: &CudaRuntime) {
    unsafe {
        runtime.reset_capture_error_for_isolated_test().unwrap();
    }
}
```
"#
)]
pub struct CudaRuntime {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Dedicated non-blocking transfer stream, distinct from the compute
    /// `stream`, used by [`CudaRuntime::htod_async`] so weight-paging host→device
    /// copies overlap kernels already queued on the compute stream (Phase-4
    /// compute/transfer overlap). Cross-stream ordering between the two streams
    /// is established explicitly through completion events, never an implicit
    /// default-stream barrier.
    copy_stream: Arc<CudaStream>,
    graph: CudaGraphLifecycle,
    verify_graph: CudaGraphLifecycle,
    /// Executor-owned graph lifecycles. The atomically published map makes the
    /// warmed replay lookup allocation- and mutex-free; capture/reset clone the
    /// small map under `owned_graphs_write`.
    owned_graphs: ArcSwap<HashMap<(DeviceGraphOwner, DeviceGraphSlot), Arc<CudaGraphLifecycle>>>,
    owned_graphs_write: Mutex<()>,
    registered_capture_active: AtomicBool,
    unregistered_capture_active: AtomicBool,
    active_capture_resource_ids: ArcSwapOption<Vec<usize>>,
    blas: CublasLt,
    cudnn: CudnnBackend,
    ordinal: u32,
    capabilities: CudaDeviceCapabilities,
    ptx_arch: String,
    cubin_arch: String,
    /// Cache of NVRTC-compiled modules, keyed by a stable module name, so each
    /// runtime compiles a given kernel (e.g. the fused attention softmax) at
    /// most once and reuses the loaded module for every kernel invocation.
    modules: Mutex<HashMap<&'static str, Arc<CudaModule>>>,
    /// Raw launch modules used by kernels whose warmed host path must not
    /// allocate. Kept separate from cudarc's safe-module cache because cudarc
    /// does not expose a raw function handle.
    raw_modules: Mutex<HashMap<&'static str, Arc<RawCudaModule>>>,
    /// Set after a driver rejects the toolkit's PTX ISA. Subsequent modules are
    /// compiled directly to the device's native SM CUBIN instead of repeating
    /// the failed load.
    nvrtc_cubin_fallback: AtomicBool,
    allocations: AtomicU64,
    frees: AtomicU64,
    /// Blocks freed through `free_raw` and held for reuse, keyed by size class.
    ///
    /// `free_raw` is told only a pointer, so the class each live block was
    /// carved from is recorded in `raw_pool_classes` at allocation time. Device
    /// addresses rather than pointers, so this stays `Send`/`Sync`; nothing
    /// here is dereferenced on the host.
    raw_pool: Mutex<HashMap<usize, Vec<CUdeviceptr>>>,
    raw_pool_classes: Mutex<HashMap<CUdeviceptr, usize>>,
    raw_pool_retained: AtomicU64,
    raw_pool_hits: AtomicU64,
    /// Identity for this runtime, unique in this process and never reused.
    ///
    /// Not the ordinal: several runtimes may share a device, and the point of
    /// the id is to tell them apart. Not an address either -- a dropped
    /// runtime's address can be handed to the next one, which is the very reuse
    /// this guards against. See [`crate::interleave_cache`].
    runtime_id: u64,
    /// Interleaved copies of int4 packed weights, keyed by the source weight's
    /// device address.
    ///
    /// A field rather than a process global because the key *is* an address
    /// this runtime's allocator minted: once the runtime goes away the address
    /// means nothing, and serving a later session's weight from an entry keyed
    /// on it is the #1726 defect. See [`crate::interleave_cache`].
    interleave: crate::interleave_cache::InterleaveCache,
    /// Whether device weight offload is on, meaning weight buffers may be paged.
    ///
    /// Set once by the provider when it configures residency. A paged weight's
    /// pages are retired by `weight_paging` without passing through
    /// `deallocate`, so `interleave` is never told the address died and cannot
    /// safely key on one. It refuses to cache at all when this is set; see
    /// [`crate::interleave_cache::InterleaveDevice::interleave_frees_are_observed`].
    weights_may_be_paged: std::sync::atomic::AtomicBool,
    /// The one cuBLASLt workspace shared by every GEMM on this runtime's
    /// compute stream. Allocated on first use and never freed until the
    /// runtime drops.
    ///
    /// cuBLASLt scratch does not carry state between calls, and work on a
    /// single stream is serial, so one buffer per stream is exactly as correct
    /// as one per call — while a per-call allocation costs
    /// [`WORKSPACE_BYTES`] for every attention op in flight. Measured on a
    /// 32-layer 1.8B model, the per-call path put hundreds of MiB of raw
    /// device memory behind a prefill that needed 32 MiB.
    shared_blas_workspace: Mutex<Option<CUdeviceptr>>,
    raw_allocation_profile: RawAllocationProfile,
    host_to_device_copies: AtomicU64,
    device_to_host_copies: AtomicU64,
    async_host_to_device_copies: AtomicU64,
    raw_h2d_attempted: AtomicU64,
    raw_h2d_completed: AtomicU64,
    raw_d2h_attempted: AtomicU64,
    raw_d2h_completed: AtomicU64,
    raw_d2d_attempted: AtomicU64,
    raw_d2d_completed: AtomicU64,
    raw_memset_attempted: AtomicU64,
    raw_memset_completed: AtomicU64,
    /// One gate before the authenticated fixed TLS context. False is the
    /// shipped default, so uninstrumented operations never touch TLS.
    observed_bytes_enabled: AtomicBool,
    forced_synchronizations: AtomicU64,
    /// Completion events recorded on a transfer/compute stream, keyed by an
    /// opaque fence id handed out to the executor inside an
    /// [`onnx_runtime_ep_api::Fence`]. `wait_*_fence` removes and waits on the
    /// event, establishing a stream-ordered (non host-blocking) cross-stream
    /// dependency. See [`CudaRuntime::record_copy_fence`].
    fences: Mutex<HashMap<u64, CudaEvent>>,
    next_fence_id: AtomicU64,
    /// Persistent four-byte device word into which kernels latch an out-of-range
    /// bounds violation during deferred eager execution or CUDA-graph replay. It
    /// is set (via `atomicOr`) and never auto-cleared on the device. Only the
    /// owner-scoped validation lifecycle clears it while opening a generation or
    /// after publishing that generation's result. Graph reset never clears it.
    /// The host reads it after a request synchronization so eager or captured
    /// validation failures become hard errors before outputs are consumed.
    capture_error: CUdeviceptr,
    /// Sole authority for the provider-wide validation phase and cleanup owner.
    /// Recipient slots are setup-allocated and linked while this word is in
    /// `Preparing`/`Attaching`; the `Active -> Consuming` CAS is the cleanup
    /// linearization point.
    validation_state: AtomicU64,
    validation_head: AtomicPtr<DeviceValidationSlot>,
    validation_submitter: AtomicPtr<DeviceValidationSlot>,
    next_validation_generation: AtomicU64,
    validation_runtime_id: u64,
    // Boxes keep intrusive-list addresses stable when setup registration grows.
    #[allow(clippy::vec_box)]
    validation_owners: Mutex<Vec<Box<DeviceValidationSlot>>>,
    registered_validation_owners: AtomicUsize,
    validation_registry_lock_acquisitions: AtomicU64,
    validation_submissions: AtomicU64,
    #[cfg(feature = "gpu-tests")]
    validation_cleanups: AtomicU64,
    #[cfg(feature = "gpu-tests")]
    validation_consumer_pause: AtomicBool,
    #[cfg(feature = "gpu-tests")]
    validation_consumer_claimed: AtomicBool,
    #[cfg(feature = "gpu-tests")]
    validation_reset_pause: AtomicBool,
    #[cfg(feature = "gpu-tests")]
    validation_reset_claimed: AtomicBool,
    /// When set, the public [`CudaRuntime::synchronize`] becomes a no-op so the
    /// redundant trailing per-op eager device syncs (issued by kernels on the
    /// `!capturing` branch) are elided and launches pipeline on the in-order EP
    /// stream. Host-visible reads (`dtoh`/`dtod`) call the private
    /// [`CudaRuntime::force_synchronize`] and are therefore unaffected. On by
    /// default (eager decode is made consistent with the captured path, which
    /// already elides these); disable via `ONNX_GENAI_DEFER_EAGER_SYNC=0`.
    defer_eager_sync: AtomicBool,
    /// Capture-gate section covering this runtime's teardown, acquired in
    /// [`Drop::drop`] rather than at construction.
    ///
    /// **This field must stay last.** Fields drop in declaration order *after*
    /// the `drop` body returns, so a guard bound as a local inside that body is
    /// released too early -- before the modules and streams whose destruction is
    /// the hazard. Declared last, it is released last, and every earlier field's
    /// teardown happens while it is still held.
    teardown_section: Option<capture_gate::SynchronizingSection>,
}

/// Remove one registered event and transfer its ownership to exactly one
/// backend wait. Fence zero and absent/already-consumed ids are no-ops.
///
/// Keeping registry removal and backend dispatch in this small CUDA-agnostic
/// core makes the at-most-once ownership rule directly testable without a
/// second implementation of the production bookkeeping.
fn dispatch_registered_fence_wait<Event>(
    registry: &Mutex<HashMap<u64, Event>>,
    fence_id: u64,
    wait: impl FnOnce(&Event) -> Result<()>,
) -> Result<bool> {
    if fence_id == 0 {
        return Ok(false);
    }
    let event = registry
        .lock()
        .expect("cuda fence registry poisoned")
        .remove(&fence_id);
    let Some(event) = event else {
        return Ok(false);
    };
    wait(&event).map_err(|error| {
        EpError::KernelFailed(format!(
            "cuda_ep: fence {fence_id} was consumed, but its backend wait failed: {error}"
        ))
    })?;
    Ok(true)
}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("ordinal", &self.ordinal)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

impl CudaRuntime {
    /// Initialise the primary context on CUDA device `ordinal`, its dedicated
    /// stream, and a cuBLASLt handle. Returns an error (never panics) when no
    /// such device exists or the CUDA driver / cuBLASLt cannot be loaded.
    pub fn new(ordinal: u32) -> Result<Self> {
        // Preload the wheel-provided dependency chain by its absolute discovered
        // paths. CUDA component wheels live in sibling directories, so relying on
        // cuBLASLt's ambient dependency lookup would make `nxrt[cuda]` depend on
        // a system CUDA installation.
        //
        // Dependants last. `cublas64_*.dll` imports `cublasLt64_*.dll`, and
        // Windows resolves that import through the default search order, which
        // does not include the directory the importing DLL was loaded from. So
        // loading cuBLAS first fails outright unless the wheel directory also
        // happens to be on `PATH` -- which is exactly the case wheel discovery
        // exists to stop depending on. Loading cuBLASLt first puts it in the
        // process, and cuBLAS's import then resolves against the already-loaded
        // module.
        for library in [
            CudaLibrary::Driver,
            CudaLibrary::CublasLt,
            CudaLibrary::Cublas,
        ] {
            require(library).map_err(|message| {
                EpError::KernelFailed(format!(
                    "cuda_ep: {message}; CPU execution remains available"
                ))
            })?;
        }
        // cudart is preloaded when present but is not required, because nothing
        // here calls it. Measured with `dumpbin /dependents` on the NVIDIA cu12
        // wheels: `cublasLt64_12.dll` imports only `KERNEL32.dll`, and
        // `cublas64_12.dll` only cuBLASLt and `KERNEL32.dll` -- NVIDIA links the
        // runtime statically into its redistributables. No `cudaXxx` symbol is
        // resolved anywhere in this crate either.
        //
        // Requiring it therefore only turned "works" into "fails" on machines
        // that could run us. Nothing becomes silent: if cuBLAS genuinely needs
        // cudart on some platform, `require(Cublas)` above already fails, and it
        // names the library that could not load rather than a proxy for it.
        //
        // The wheel is still a real dependency -- NVRTC compiles our f16/bf16
        // kernels against `cuda_fp16.h` and `cuda_bf16.h`, which ship in
        // `nvidia/cuda_runtime/include`. That is a *header* dependency, checked
        // where those kernels are built, not a reason to demand the DLL here.
        let _ = require(CudaLibrary::Runtime);
        let context = shared_context(ordinal)?;
        let major = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| driver_err("querying CUDA compute capability major", e))?;
        let minor = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| driver_err("querying CUDA compute capability minor", e))?;
        let major = u32::try_from(major).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep: CUDA device {ordinal} reported invalid compute capability major {major}"
            ))
        })?;
        let minor = u32::try_from(minor).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep: CUDA device {ordinal} reported invalid compute capability minor {minor}"
            ))
        })?;
        if major == 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: CUDA device {ordinal} reported invalid compute capability {major}.{minor}"
            )));
        }
        let compute_capability = (major, minor);
        let capabilities = CudaDeviceCapabilities::from_reported_limits(
            compute_capability,
            positive_attribute(
                &context,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            ),
            positive_attribute(
                &context,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
            ),
            positive_attribute(
                &context,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            ),
            positive_attribute(
                &context,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            ),
            positive_attribute(
                &context,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            ),
            positive_attribute(
                &context,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
            ),
        );
        let ptx_arch = ptx_arch_for(major, minor);
        let cubin_arch = cubin_arch_for(major, minor);
        // A dedicated non-blocking stream (not the legacy NULL stream, which the
        // driver refuses to capture) so device-resident kernels are eligible for
        // CUDA-graph capture. The whole EP drives this single stream. Synchronous
        // H2D copies explicitly drain it before using and synchronizing the
        // legacy default stream; CUDA does not implicitly order non-blocking
        // streams against that stream.
        let stream = context
            .new_stream()
            .map_err(|e| driver_err("create compute stream", e))?;
        // A second dedicated non-blocking stream for host→device weight
        // prefetch. Keeping transfers off the compute stream is what lets a
        // prefetch of expert N+1's weights overlap the wave-N kernel; the two
        // streams are ordered against each other only through explicit
        // completion events (see `record_copy_fence` / `compute_wait_fence`).
        let copy_stream = context
            .new_stream()
            .map_err(|e| driver_err("create transfer stream", e))?;
        let blas = CublasLt::new()?;
        let cudnn = CudnnBackend::new(stream.clone());
        let graph = CudaGraphLifecycle::new(
            stream.clone(),
            DeviceGraphOwner::new(),
            DeviceGraphSlot::Primary,
        );
        // Second captured-graph slot for the MTP fixed-width verify forward,
        // held independently of `graph` (the M=1 decode step) on the same
        // compute stream so both shapes can be replayed by shape key without
        // per-step recapture (see `DeviceGraphSlot`).
        let verify_graph = CudaGraphLifecycle::new(
            stream.clone(),
            DeviceGraphOwner::new(),
            DeviceGraphSlot::Verify,
        );
        Self {
            context,
            stream,
            copy_stream,
            graph,
            verify_graph,
            owned_graphs: ArcSwap::from_pointee(HashMap::new()),
            owned_graphs_write: Mutex::new(()),
            registered_capture_active: AtomicBool::new(false),
            unregistered_capture_active: AtomicBool::new(false),
            active_capture_resource_ids: ArcSwapOption::empty(),
            blas,
            cudnn,
            ordinal,
            capabilities,
            ptx_arch,
            cubin_arch,
            modules: Mutex::new(HashMap::new()),
            raw_modules: Mutex::new(HashMap::new()),
            nvrtc_cubin_fallback: AtomicBool::new(false),
            allocations: AtomicU64::new(0),
            frees: AtomicU64::new(0),
            raw_pool: Mutex::new(HashMap::new()),
            raw_pool_classes: Mutex::new(HashMap::new()),
            raw_pool_retained: AtomicU64::new(0),
            raw_pool_hits: AtomicU64::new(0),
            runtime_id: NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            interleave: crate::interleave_cache::InterleaveCache::default(),
            weights_may_be_paged: std::sync::atomic::AtomicBool::new(false),
            shared_blas_workspace: Mutex::new(None),
            raw_allocation_profile: RawAllocationProfile::new(raw_allocation_profile_enabled()),
            host_to_device_copies: AtomicU64::new(0),
            device_to_host_copies: AtomicU64::new(0),
            async_host_to_device_copies: AtomicU64::new(0),
            raw_h2d_attempted: AtomicU64::new(0),
            raw_h2d_completed: AtomicU64::new(0),
            raw_d2h_attempted: AtomicU64::new(0),
            raw_d2h_completed: AtomicU64::new(0),
            raw_d2d_attempted: AtomicU64::new(0),
            raw_d2d_completed: AtomicU64::new(0),
            raw_memset_attempted: AtomicU64::new(0),
            raw_memset_completed: AtomicU64::new(0),
            observed_bytes_enabled: AtomicBool::new(false),
            forced_synchronizations: AtomicU64::new(0),
            fences: Mutex::new(HashMap::new()),
            next_fence_id: AtomicU64::new(1),
            capture_error: 0,
            validation_state: AtomicU64::new(validation_word(ValidationPhase::Idle, 0)),
            validation_head: AtomicPtr::new(std::ptr::null_mut()),
            validation_submitter: AtomicPtr::new(std::ptr::null_mut()),
            next_validation_generation: AtomicU64::new(1),
            validation_runtime_id: next_validation_runtime_id(),
            validation_owners: Mutex::new(Vec::new()),
            registered_validation_owners: AtomicUsize::new(0),
            validation_registry_lock_acquisitions: AtomicU64::new(0),
            validation_submissions: AtomicU64::new(0),
            #[cfg(feature = "gpu-tests")]
            validation_cleanups: AtomicU64::new(0),
            #[cfg(feature = "gpu-tests")]
            validation_consumer_pause: AtomicBool::new(false),
            #[cfg(feature = "gpu-tests")]
            validation_consumer_claimed: AtomicBool::new(false),
            #[cfg(feature = "gpu-tests")]
            validation_reset_pause: AtomicBool::new(false),
            #[cfg(feature = "gpu-tests")]
            validation_reset_claimed: AtomicBool::new(false),
            defer_eager_sync: AtomicBool::new(
                // On by default; only an explicit falsey value restores the old
                // always-sync eager path (escape hatch for debugging).
                std::env::var("ONNX_GENAI_DEFER_EAGER_SYNC")
                    .ok()
                    .as_deref()
                    .map(str::trim)
                    .map(str::to_ascii_lowercase)
                    .is_none_or(|v| !matches!(v.as_str(), "0" | "false" | "off" | "no")),
            ),
            teardown_section: None,
        }
        .with_capture_error_word()
    }

    pub(crate) fn enable_observed_bytes(&self) -> Result<()> {
        self.observed_bytes_enabled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep: observed-byte recording is already enabled on this runtime".into(),
                )
            })
    }

    #[inline]
    pub(crate) fn observe_bytes(&self, spec: EventSpec) -> Result<()> {
        if let Some(recorder) = self.observed_byte_recorder() {
            recorder.record(spec).map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: record production observed-byte event: {error}"
                ))
            })?;
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn observe_byte_pair(&self, first: EventSpec, second: EventSpec) -> Result<()> {
        if let Some(recorder) = self.observed_byte_recorder() {
            recorder.record_pair(first, second).map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: record production observed-byte event pair: {error}"
                ))
            })?;
        }
        Ok(())
    }

    pub(crate) fn observed_byte_recorder(&self) -> Option<&ProductionByteRecorder> {
        if !self.observed_bytes_enabled.load(Ordering::Relaxed) {
            return None;
        }
        current_recorder(self.runtime_id)
    }

    pub(crate) fn retained_observed_byte_recorder(&self) -> Result<Option<ProductionByteRecorder>> {
        self.observed_byte_recorder()
            .map(ProductionByteRecorder::retain_for_deferred)
            .transpose()
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: retain session-authorized recorder for deferred settlement: {error}"
                ))
            })
    }

    pub(crate) fn prepare_observation(
        &self,
        specs: &[EventSpec],
    ) -> Result<Option<PendingObservedBatch>> {
        self.observed_byte_recorder()
            .map(|recorder| {
                recorder.prepare(specs).map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: reserve production observed-byte receipt before operation: \
                         {error}"
                    ))
                })
            })
            .transpose()
    }

    pub(crate) fn commit_observation(observation: &mut Option<PendingObservedBatch>) -> Result<()> {
        if let Some(observation) = observation {
            observation.commit().map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: commit production operation receipt: {error}"
                ))
            })?;
        }
        Ok(())
    }

    fn mark_observation_submitted(observation: &mut Option<PendingObservedBatch>) {
        if let Some(observation) = observation {
            observation.mark_submitted();
        }
    }

    fn fail_observation(
        observation: &mut Option<PendingObservedBatch>,
        status: ObservedStatus,
        operation: EpError,
    ) -> EpError {
        match observation
            .as_mut()
            .map(|observation| observation.abort(status))
            .transpose()
        {
            Ok(_) => operation,
            Err(telemetry) => EpError::KernelFailed(format!(
                "{operation}; production operation also failed to publish its {status:?} \
                 observed-byte receipt: {telemetry}"
            )),
        }
    }

    /// Allocate and zero the persistent capture-error latch word. Split out of
    /// [`CudaRuntime::new`] so it can use the runtime's own bound-context
    /// alloc/copy helpers.
    fn with_capture_error_word(mut self) -> Result<Self> {
        let ptr = self.alloc_raw(std::mem::size_of::<u32>())?;
        // SAFETY: `ptr` is a fresh four-byte device allocation owned by this
        // runtime; zeroing it establishes the un-latched initial state.
        unsafe { self.htod(&0_u32.to_ne_bytes(), ptr) }?;
        self.capture_error = ptr;
        Ok(self)
    }

    /// The CUDA device ordinal this runtime drives.
    /// The CUDA context this runtime is bound to.
    ///
    /// Exposed so driver-only components -- the device allocator and the
    /// virtual-memory backing -- can share the context without depending on the
    /// runtime's cudart and cuBLAS preconditions.
    pub fn cuda_context(&self) -> std::sync::Arc<cudarc::driver::CudaContext> {
        std::sync::Arc::clone(&self.context)
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Hardware capabilities reported by the selected CUDA device.
    pub fn capabilities(&self) -> CudaDeviceCapabilities {
        self.capabilities
    }

    /// The cuBLASLt handle.
    pub fn blas(&self) -> &CublasLt {
        &self.blas
    }

    /// The lazily initialized cuDNN backend bound to this runtime's stream.
    pub fn cudnn(&self) -> &CudnnBackend {
        &self.cudnn
    }

    /// The raw CUDA stream the EP submits work on.
    pub fn stream_ptr(&self) -> cudarc::driver::sys::CUstream {
        self.stream.cu_stream()
    }

    /// The EP's compute stream (for `launch_builder`-based kernel launches).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The EP's dedicated host→device transfer stream, used for asynchronous
    /// weight prefetch that overlaps compute (Phase-4). Kept distinct from the
    /// compute [`stream`](Self::stream) so a prefetch of the next expert's
    /// weights runs concurrently with the current wave's kernels.
    pub fn copy_stream(&self) -> &Arc<CudaStream> {
        &self.copy_stream
    }

    /// Begin capture on the EP stream after auditing the complete kernel sequence.
    pub fn begin_graph_capture(&self, kernels: &[&dyn Kernel]) -> Result<()> {
        self.begin_graph_capture_with_resources(kernels, Vec::new())
    }

    /// Begin capture with additional immutable address owners.
    pub fn begin_graph_capture_with_resources(
        &self,
        kernels: &[&dyn Kernel],
        mut resources: Vec<DeviceGraphResource>,
    ) -> Result<()> {
        crate::capture::require_subgraph_graph_capturable(kernels)?;
        resources.extend(
            kernels
                .iter()
                .flat_map(|kernel| kernel.device_graph_resources()),
        );
        self.begin_graph_capture_resources_in(DeviceGraphSlot::Primary, resources)
    }

    /// End stream capture and install the instantiated graph executable.
    pub fn end_graph_capture(&self) -> Result<()> {
        self.end_graph_capture_in(DeviceGraphSlot::Primary)
    }

    /// Abort an in-progress stream capture, discarding any half-recorded graph
    /// and returning the lifecycle to idle so a subsequent [`reset_graph`]
    /// succeeds. Used on the error path of segmented capture.
    pub fn abort_graph_capture(&self) -> Result<()> {
        self.abort_graph_capture_in(DeviceGraphSlot::Primary)
    }

    /// Launch the installed graph executable on the same EP stream.
    ///
    /// Replays every installed segment in capture order (one graph for a
    /// whole-subgraph capture).
    pub fn replay_graph(&self) -> Result<()> {
        self.graph.replay_current()
    }

    /// Launch one installed segment by its zero-based capture-order index.
    pub fn replay_graph_segment(&self, index: usize) -> Result<()> {
        self.graph.replay_current_segment(index)
    }

    /// Number of installed captured segments (1 for a whole-subgraph capture).
    pub fn graph_segment_count(&self) -> Result<usize> {
        self.graph.current_segment_count()
    }

    /// Destroy the installed graph and graph-exec handles.
    ///
    /// Returns whether an executable was invalidated. Reset is rejected while a
    /// capture is active; callers must end the capture first.
    pub fn reset_graph(&self) -> Result<bool> {
        self.graph.reset_current()
    }

    /// Whether this runtime currently owns an instantiated graph executable.
    pub fn has_graph_executable(&self) -> Result<bool> {
        self.graph.has_current_executable()
    }

    /// Raw device pointer to the persistent capture-error latch word, passed to
    /// capture-safe kernels so they can `atomicOr` a bounds-violation code into
    /// it (and read it back to propagate the poison to later kernels/replays).
    pub fn capture_error_ptr(&self) -> CUdeviceptr {
        self.capture_error
    }

    /// Read the latching capture-error word device → host, returning the raw
    /// violation bitmask (zero when no capture-safe kernel has tripped).
    ///
    /// This does not clear the latch: once set, subsequent work stays poisoned
    /// until [`CudaRuntime::reset_capture_error`]. Callers invoke this only after
    /// a request-level host synchronization boundary.
    pub fn check_capture_error(&self) -> Result<u32> {
        let mut bytes = [0_u8; std::mem::size_of::<u32>()];
        // SAFETY: `capture_error` is a live four-byte device allocation owned by
        // this runtime for its whole lifetime.
        unsafe { self.dtoh(&mut bytes, self.capture_error) }?;
        Ok(u32::from_ne_bytes(bytes))
    }

    fn validation_slot<'a>(
        &self,
        registration: &'a DeviceValidationRegistration,
    ) -> Result<&'a DeviceValidationSlot> {
        let key = registration
            .state::<CudaValidationRegistration>()
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep: validation owner {} was registered by a different provider",
                    registration.owner().get()
                ))
            })?;
        if key.runtime_id != self.validation_runtime_id {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: validation owner {} belongs to another CUDA runtime",
                registration.owner().get()
            )));
        }
        if key.retired {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: validation owner {} is already retired",
                registration.owner().get()
            )));
        }
        // SAFETY: the runtime owns every registered slot until consuming
        // `unregister_device_validation_owner` removes it. A live registration
        // cannot be used after that consuming call.
        let slot = unsafe { &*(key.slot as *const DeviceValidationSlot) };
        if slot.owner != registration.owner() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: validation registration owner {} does not match slot owner {}",
                registration.owner().get(),
                slot.owner.get()
            )));
        }
        Ok(slot)
    }

    fn take_validation_generation(&self) -> Result<u64> {
        take_validation_generation(&self.next_validation_generation)
    }

    /// Allocate one stable owner slot during setup. The registry lock is only a
    /// storage/lifetime mechanism; warmed validation never touches it.
    pub(crate) fn register_device_validation_owner(&self) -> Result<DeviceValidationRegistration> {
        let owner = DeviceValidationOwner::new();
        let mut slot = Box::new(DeviceValidationSlot::new(owner));
        let slot_ptr = (&mut *slot) as *mut DeviceValidationSlot;
        self.validation_registry_lock_acquisitions
            .fetch_add(1, Ordering::Relaxed);
        self.validation_owners
            .lock()
            .map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep: device validation setup registry lock is poisoned".into(),
                )
            })?
            .push(slot);
        self.registered_validation_owners
            .fetch_add(1, Ordering::Relaxed);
        Ok(DeviceValidationRegistration::new(
            owner,
            CudaValidationRegistration {
                runtime_id: self.validation_runtime_id,
                slot: slot_ptr as usize,
                retired: false,
            },
        ))
    }

    /// Retire one setup slot. A pending generation is first consumed through the
    /// same state machine, so concurrent legitimate consumption converges on the
    /// sticky result instead of leaking the slot.
    pub(crate) fn unregister_device_validation_owner(
        &self,
        registration: &mut DeviceValidationRegistration,
    ) -> Result<()> {
        let slot = self.validation_slot(registration)?;
        loop {
            let current = slot.state.load(Ordering::Acquire);
            match validation_slot_phase(current) {
                ValidationSlotPhase::Pending => {
                    let token = DeviceValidationToken::new(
                        registration.owner(),
                        validation_generation(current),
                    );
                    self.consume_device_validation(registration, token)?;
                }
                ValidationSlotPhase::Idle | ValidationSlotPhase::Complete => {
                    let retired = validation_slot_word(
                        ValidationSlotPhase::Retired,
                        validation_generation(current),
                    );
                    if slot
                        .state
                        .compare_exchange(current, retired, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        break;
                    }
                }
                ValidationSlotPhase::Retired => {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep: validation owner {} is already retired",
                        registration.owner().get()
                    )));
                }
            }
        }

        let slot_ptr = slot as *const DeviceValidationSlot as usize;
        self.validation_registry_lock_acquisitions
            .fetch_add(1, Ordering::Relaxed);
        let mut owners = self.validation_owners.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep: device validation setup registry lock is poisoned".into(),
            )
        })?;
        let index = owners
            .iter()
            .position(|candidate| {
                (&**candidate as *const DeviceValidationSlot as usize) == slot_ptr
            })
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep: validation owner {} slot is absent from the setup registry",
                    registration.owner().get()
                ))
            })?;
        registration
            .state_mut::<CudaValidationRegistration>()
            .expect("CUDA registration type was checked before retirement")
            .retired = true;
        owners.swap_remove(index);
        self.registered_validation_owners
            .fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    /// Reserve a generation and clear its device latch before recipient
    /// attachment. `Idle -> Resetting` excludes both another begin and the
    /// isolated-test reset in one CAS protocol.
    pub(crate) fn begin_device_validation(
        &self,
        registration: &DeviceValidationRegistration,
    ) -> Result<DeviceValidationToken> {
        let slot = self.validation_slot(registration)?;
        let generation = self.take_validation_generation()?;
        let resetting = validation_word(ValidationPhase::Resetting, generation);
        if let Err(current) = self.validation_state.compare_exchange(
            validation_word(ValidationPhase::Idle, 0),
            resetting,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: previous deferred device validation is still {:?} at generation {}; \
                 consume its bound output or finish its request boundary before submitting \
                 owner={}",
                validation_phase(current),
                validation_generation(current),
                registration.owner().get()
            )));
        }
        if let Err(error) = self.reset_capture_error() {
            self.validation_state
                .store(validation_word(ValidationPhase::Idle, 0), Ordering::Release);
            return Err(error);
        }

        let pending = validation_slot_word(ValidationSlotPhase::Pending, generation);
        loop {
            let current = slot.state.load(Ordering::Acquire);
            match validation_slot_phase(current) {
                ValidationSlotPhase::Idle | ValidationSlotPhase::Complete => {
                    if slot
                        .state
                        .compare_exchange(current, pending, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        break;
                    }
                }
                ValidationSlotPhase::Pending | ValidationSlotPhase::Retired => {
                    self.validation_state
                        .store(validation_word(ValidationPhase::Idle, 0), Ordering::Release);
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep: validation owner {} cannot begin generation {generation} from \
                         slot state {:?} generation {}",
                        registration.owner().get(),
                        validation_slot_phase(current),
                        validation_generation(current)
                    )));
                }
            }
        }
        let slot_ptr = slot as *const DeviceValidationSlot as *mut DeviceValidationSlot;
        slot.next.store(std::ptr::null_mut(), Ordering::Relaxed);
        self.validation_head.store(slot_ptr, Ordering::Relaxed);
        self.validation_submitter.store(slot_ptr, Ordering::Relaxed);
        self.validation_state.store(
            validation_word(ValidationPhase::Preparing, generation),
            Ordering::Release,
        );
        self.validation_submissions.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceValidationToken::new(registration.owner(), generation))
    }

    /// Attach one setup-allocated output slot. `Preparing -> Attaching`
    /// serializes list mutation without a lock; activation succeeds only after
    /// the attachment publishes `Preparing` again.
    pub(crate) fn add_device_validation_recipient(
        &self,
        submission: DeviceValidationToken,
        recipient: &DeviceValidationRegistration,
    ) -> Result<DeviceValidationToken> {
        let generation = submission.generation();
        let preparing = validation_word(ValidationPhase::Preparing, generation);
        let attaching = validation_word(ValidationPhase::Attaching, generation);
        self.validation_state
            .compare_exchange(preparing, attaching, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|current| {
                EpError::KernelFailed(format!(
                    "cuda_ep: validation submission owner={} generation={} cannot attach owner={}; \
                     coordinator is {:?} at generation {}",
                    submission.owner().get(),
                    generation,
                    recipient.owner().get(),
                    validation_phase(current),
                    validation_generation(current)
                ))
            })?;

        let result = (|| {
            let submitter_ptr = self.validation_submitter.load(Ordering::Acquire);
            if submitter_ptr.is_null() {
                return Err(EpError::KernelFailed(
                    "cuda_ep: validation submission has no registered submitter slot".into(),
                ));
            }
            // SAFETY: the submitter slot remains registered while its submission
            // is preparing or active.
            let submitter = unsafe { &*submitter_ptr };
            if submitter.owner != submission.owner() {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep: validation submission owner {} is foreign to active submitter {}",
                    submission.owner().get(),
                    submitter.owner.get()
                )));
            }

            let slot = self.validation_slot(recipient)?;
            let pending = validation_slot_word(ValidationSlotPhase::Pending, generation);
            loop {
                let current = slot.state.load(Ordering::Acquire);
                match validation_slot_phase(current) {
                    ValidationSlotPhase::Idle | ValidationSlotPhase::Complete => {
                        if slot
                            .state
                            .compare_exchange(current, pending, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                        {
                            break;
                        }
                    }
                    ValidationSlotPhase::Pending | ValidationSlotPhase::Retired => {
                        return Err(EpError::KernelFailed(format!(
                            "cuda_ep: validation recipient owner {} cannot attach to generation \
                             {generation} from slot state {:?} generation {}",
                            recipient.owner().get(),
                            validation_slot_phase(current),
                            validation_generation(current)
                        )));
                    }
                }
            }
            let slot_ptr = slot as *const DeviceValidationSlot as *mut DeviceValidationSlot;
            slot.next.store(
                self.validation_head.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
            self.validation_head.store(slot_ptr, Ordering::Relaxed);
            Ok(DeviceValidationToken::new(recipient.owner(), generation))
        })();

        self.validation_state.store(preparing, Ordering::Release);
        result
    }

    /// Seal attachment. This release publishes the reset and complete recipient
    /// list before any thread may acquire cleanup authority.
    pub(crate) fn activate_device_validation(
        &self,
        submission: DeviceValidationToken,
    ) -> Result<()> {
        let generation = submission.generation();
        let submitter_ptr = self.validation_submitter.load(Ordering::Acquire);
        if submitter_ptr.is_null() {
            return Err(EpError::KernelFailed(
                "cuda_ep: validation submission has no registered submitter slot".into(),
            ));
        }
        // SAFETY: the submitter cannot unregister while its generation is
        // preparing.
        let submitter = unsafe { &*submitter_ptr };
        if submitter.owner != submission.owner() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: validation submission owner {} is foreign to active submitter {}",
                submission.owner().get(),
                submitter.owner.get()
            )));
        }
        self.validation_state
            .compare_exchange(
                validation_word(ValidationPhase::Preparing, generation),
                validation_word(ValidationPhase::Active, generation),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|current| {
                EpError::KernelFailed(format!(
                    "cuda_ep: validation submission owner={} generation={} cannot activate; \
                     coordinator is {:?} at generation {}",
                    submission.owner().get(),
                    generation,
                    validation_phase(current),
                    validation_generation(current)
                ))
            })
    }

    fn publish_device_validation(&self, generation: u64, flags: u32) {
        let mut slot_ptr = self.validation_head.load(Ordering::Acquire);
        while !slot_ptr.is_null() {
            // SAFETY: pending slots cannot unregister; this traversal owns the
            // generation's sole `Consuming` authority.
            let slot = unsafe { &*slot_ptr };
            let next = slot.next.load(Ordering::Relaxed);
            slot.flags.store(flags, Ordering::Relaxed);
            let pending = validation_slot_word(ValidationSlotPhase::Pending, generation);
            let complete = validation_slot_word(ValidationSlotPhase::Complete, generation);
            // Release makes the flags payload visible to a sticky reader's
            // Acquire load of `Complete`.
            if let Err(current) =
                slot.state
                    .compare_exchange(pending, complete, Ordering::Release, Ordering::Acquire)
            {
                eprintln!(
                    "[onnx-runtime-ep-cuda] validation owner {} changed unexpectedly while \
                     publishing generation {generation}: phase={:?} generation={}",
                    slot.owner.get(),
                    validation_slot_phase(current),
                    validation_generation(current)
                );
            }
            slot_ptr = next;
        }
        self.validation_head
            .store(std::ptr::null_mut(), Ordering::Relaxed);
        self.validation_submitter
            .store(std::ptr::null_mut(), Ordering::Relaxed);
        self.validation_state
            .store(validation_word(ValidationPhase::Idle, 0), Ordering::Release);
        #[cfg(feature = "gpu-tests")]
        self.validation_cleanups.fetch_add(1, Ordering::Relaxed);
    }

    /// Consume exactly `token`. `Active -> Consuming` is the linearization point
    /// that assigns latch cleanup to one caller; competitors wait for that
    /// caller's release-published sticky result.
    pub(crate) fn consume_device_validation(
        &self,
        registration: &DeviceValidationRegistration,
        token: DeviceValidationToken,
    ) -> Result<u32> {
        let slot = self.validation_slot(registration)?;
        if registration.owner() != token.owner() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: validation token owner={} is foreign to registration owner={}",
                token.owner().get(),
                registration.owner().get()
            )));
        }
        self.consume_device_validation_slot(slot, token)
    }

    pub(crate) fn abort_device_validation_submission(
        &self,
        token: DeviceValidationToken,
    ) -> Result<u32> {
        let slot_ptr = self.validation_submitter.load(Ordering::Acquire);
        if slot_ptr.is_null() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: validation submission owner={} generation={} has no active submitter",
                token.owner().get(),
                token.generation()
            )));
        }
        // SAFETY: a pending submitter cannot unregister before this submission
        // reaches a terminal state.
        let slot = unsafe { &*slot_ptr };
        if slot.owner != token.owner() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: validation submission owner={} is foreign to active submitter={}",
                token.owner().get(),
                slot.owner.get()
            )));
        }
        self.consume_device_validation_slot(slot, token)
    }

    fn consume_device_validation_slot(
        &self,
        slot: &DeviceValidationSlot,
        token: DeviceValidationToken,
    ) -> Result<u32> {
        let pending = validation_slot_word(ValidationSlotPhase::Pending, token.generation());
        let complete = validation_slot_word(ValidationSlotPhase::Complete, token.generation());
        let mut spins = 0_u32;
        loop {
            let owner_state = slot.state.load(Ordering::Acquire);
            if owner_state == complete {
                let flags = slot.flags.load(Ordering::Relaxed);
                // An owner may begin a later generation immediately after the
                // first Acquire; the second rejects that reuse instead of
                // returning the later generation's flags for this token.
                if slot.state.load(Ordering::Acquire) == complete {
                    return Ok(flags);
                }
                continue;
            }
            if owner_state != pending {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep: validation token owner={} generation={} is stale for slot phase \
                     {:?} generation {}",
                    token.owner().get(),
                    token.generation(),
                    validation_slot_phase(owner_state),
                    validation_generation(owner_state)
                )));
            }

            let coordinator = self.validation_state.load(Ordering::Acquire);
            if validation_generation(coordinator) != token.generation() {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep: validation token owner={} generation={} does not belong to \
                     coordinator phase {:?} generation {}",
                    token.owner().get(),
                    token.generation(),
                    validation_phase(coordinator),
                    validation_generation(coordinator)
                )));
            }
            let phase = validation_phase(coordinator);
            match phase {
                ValidationPhase::Preparing | ValidationPhase::Active => {
                    let consuming = validation_word(ValidationPhase::Consuming, token.generation());
                    if self
                        .validation_state
                        .compare_exchange(
                            coordinator,
                            consuming,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_err()
                    {
                        continue;
                    }
                    #[cfg(feature = "gpu-tests")]
                    if phase == ValidationPhase::Active
                        && self.validation_consumer_pause.load(Ordering::Acquire)
                    {
                        self.validation_consumer_claimed
                            .store(true, Ordering::Release);
                        while self.validation_consumer_pause.load(Ordering::Acquire) {
                            std::thread::yield_now();
                        }
                    }
                    if phase == ValidationPhase::Preparing {
                        self.publish_device_validation(token.generation(), 0);
                        return Ok(0);
                    }
                    let result = self
                        .check_capture_error()
                        .and_then(|flags| self.reset_capture_error().map(|()| flags));
                    match result {
                        Ok(flags) => {
                            self.publish_device_validation(token.generation(), flags);
                            return Ok(flags);
                        }
                        Err(error) => {
                            self.validation_state.store(
                                validation_word(ValidationPhase::Active, token.generation()),
                                Ordering::Release,
                            );
                            return Err(error);
                        }
                    }
                }
                ValidationPhase::Resetting
                | ValidationPhase::Attaching
                | ValidationPhase::Consuming => {
                    spins = spins.saturating_add(1);
                    if spins < 64 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
                ValidationPhase::Idle => {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep: validation token owner={} generation={} is pending while the \
                         coordinator is idle",
                        token.owner().get(),
                        token.generation()
                    )));
                }
            }
        }
    }

    /// Clear the latching capture-error word back to the un-poisoned state.
    /// Invoked only when opening or consuming a validation generation. Graph
    /// reset must never clear an unconsumed result.
    fn reset_capture_error(&self) -> Result<()> {
        self.bind()?;
        let bytes = std::mem::size_of::<u32>() as u64;
        let mut observation = self.prepare_observation(&[
            EventSpec::new(
                ObservedCategory::CudaMemset,
                ObservedBoundary::RuntimeCudaMemset,
                ObservedStatus::Submitted,
                bytes,
            ),
            EventSpec::new(
                ObservedCategory::CudaMemset,
                ObservedBoundary::AsyncCompletionUnsupported,
                ObservedStatus::Unsupported,
                0,
            ),
        ])?;
        Self::reserve_raw_bytes(
            &self.raw_memset_attempted,
            bytes,
            "validation-latch memset attempted",
        )?;
        // SAFETY: `capture_error` is a live four-byte device allocation owned by
        // this runtime for its whole lifetime. The stream-ordered clear executes
        // before subsequently submitted eager or captured kernels.
        unsafe {
            cudarc::driver::result::memset_d8_async(
                self.capture_error,
                0,
                std::mem::size_of::<u32>(),
                self.stream.cu_stream(),
            )
        }
        .map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err("clear CUDA validation latch", error),
            )
        })?;
        Self::mark_observation_submitted(&mut observation);
        Self::commit_observation(&mut observation)
    }

    #[cfg(feature = "gpu-tests")]
    pub fn validation_submission_count(&self) -> u64 {
        self.validation_submissions.load(Ordering::Relaxed)
    }

    #[cfg(feature = "gpu-tests")]
    pub fn registered_validation_owner_count(&self) -> usize {
        self.registered_validation_owners.load(Ordering::Relaxed)
    }

    #[cfg(feature = "gpu-tests")]
    pub fn validation_registry_lock_acquisition_count(&self) -> u64 {
        self.validation_registry_lock_acquisitions
            .load(Ordering::Relaxed)
    }

    #[cfg(feature = "gpu-tests")]
    pub fn validation_cleanup_count(&self) -> u64 {
        self.validation_cleanups.load(Ordering::Relaxed)
    }

    #[cfg(feature = "gpu-tests")]
    pub fn pause_validation_consumer_for_test(&self, pause: bool) {
        if pause {
            self.validation_consumer_claimed
                .store(false, Ordering::Release);
        }
        self.validation_consumer_pause
            .store(pause, Ordering::Release);
    }

    #[cfg(feature = "gpu-tests")]
    pub fn validation_consumer_claimed_for_test(&self) -> bool {
        self.validation_consumer_claimed.load(Ordering::Acquire)
    }

    #[cfg(feature = "gpu-tests")]
    pub fn pause_validation_reset_for_test(&self, pause: bool) {
        if pause {
            self.validation_reset_claimed
                .store(false, Ordering::Release);
        }
        self.validation_reset_pause.store(pause, Ordering::Release);
    }

    #[cfg(feature = "gpu-tests")]
    pub fn validation_reset_claimed_for_test(&self) -> bool {
        self.validation_reset_claimed.load(Ordering::Acquire)
    }

    /// Test-only raw reset for isolated kernel probes that do not use the
    /// session validation lifecycle.
    ///
    /// # Safety
    /// The caller must prove no validation generation is active.
    #[cfg(any(test, feature = "gpu-tests"))]
    #[doc(hidden)]
    pub unsafe fn reset_capture_error_for_isolated_test(&self) -> Result<()> {
        let resetting = validation_word(ValidationPhase::Resetting, 0);
        self.validation_state
            .compare_exchange(
                validation_word(ValidationPhase::Idle, 0),
                resetting,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|current| {
                EpError::KernelFailed(format!(
                    "cuda_ep: isolated-test reset refused while validation phase {:?} generation \
                     {} is active",
                    validation_phase(current),
                    validation_generation(current)
                ))
            })?;
        #[cfg(feature = "gpu-tests")]
        {
            if self.validation_reset_pause.load(Ordering::Acquire) {
                self.validation_reset_claimed.store(true, Ordering::Release);
                while self.validation_reset_pause.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }
        }
        let result = self.reset_capture_error();
        self.validation_state
            .store(validation_word(ValidationPhase::Idle, 0), Ordering::Release);
        result
    }

    /// Driver-reported capture status for the EP stream.
    pub fn graph_capture_status(&self) -> Result<cudarc::driver::sys::CUstreamCaptureStatus> {
        self.graph.capture_status()
    }

    /// The [`CudaGraphLifecycle`] backing a given [`DeviceGraphSlot`]. Both slots
    /// share this runtime's single compute stream; only one may be *capturing* at
    /// a time (enforced per-lifecycle), but each may independently hold an
    /// instantiated executable ready to replay.
    fn graph_slot(&self, slot: DeviceGraphSlot) -> &CudaGraphLifecycle {
        match slot {
            DeviceGraphSlot::Primary => &self.graph,
            DeviceGraphSlot::Verify => &self.verify_graph,
        }
    }

    /// Slot-aware [`begin_graph_capture`](Self::begin_graph_capture).
    pub fn begin_graph_capture_in(
        &self,
        slot: DeviceGraphSlot,
        kernels: &[&dyn Kernel],
    ) -> Result<()> {
        crate::capture::require_subgraph_graph_capturable(kernels)?;
        let resources = kernels
            .iter()
            .flat_map(|kernel| kernel.device_graph_resources())
            .collect();
        self.begin_graph_capture_resources_in(slot, resources)
    }

    /// Slot-aware capture with additional immutable address owners.
    pub fn begin_graph_capture_with_resources_in(
        &self,
        slot: DeviceGraphSlot,
        kernels: &[&dyn Kernel],
        mut resources: Vec<DeviceGraphResource>,
    ) -> Result<()> {
        crate::capture::require_subgraph_graph_capturable(kernels)?;
        resources.extend(
            kernels
                .iter()
                .flat_map(|kernel| kernel.device_graph_resources()),
        );
        self.begin_graph_capture_resources_in(slot, resources)
    }

    fn begin_graph_capture_resources_in(
        &self,
        slot: DeviceGraphSlot,
        mut resources: Vec<DeviceGraphResource>,
    ) -> Result<()> {
        resources.sort_unstable_by_key(DeviceGraphResource::identity);
        resources.dedup_by_key(|resource| resource.identity());
        let resource_ids = Arc::new(
            resources
                .iter()
                .map(DeviceGraphResource::identity)
                .collect(),
        );
        self.registered_capture_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep: cannot begin CUDA graph capture while another registered capture \
                     is active on the runtime stream"
                        .into(),
                )
            })?;
        if let Err(error) = self.graph_slot(slot).begin_current(resources) {
            self.registered_capture_active
                .store(false, Ordering::Release);
            return Err(error);
        }
        self.active_capture_resource_ids.store(Some(resource_ids));
        Ok(())
    }

    /// Slot-aware [`end_graph_capture`](Self::end_graph_capture).
    pub fn end_graph_capture_in(&self, slot: DeviceGraphSlot) -> Result<()> {
        let result = self.graph_slot(slot).end_current();
        self.registered_capture_active
            .store(false, Ordering::Release);
        self.active_capture_resource_ids.store(None);
        result
    }

    /// Slot-aware [`abort_graph_capture`](Self::abort_graph_capture).
    pub fn abort_graph_capture_in(&self, slot: DeviceGraphSlot) -> Result<()> {
        let result = self.graph_slot(slot).abort_current();
        self.registered_capture_active
            .store(false, Ordering::Release);
        self.active_capture_resource_ids.store(None);
        result
    }

    /// Slot-aware [`replay_graph`](Self::replay_graph).
    pub fn replay_graph_in(&self, slot: DeviceGraphSlot) -> Result<()> {
        self.graph_slot(slot).replay_current()
    }

    /// Slot-aware [`replay_graph_segment`](Self::replay_graph_segment).
    pub fn replay_graph_segment_in(&self, slot: DeviceGraphSlot, index: usize) -> Result<()> {
        self.graph_slot(slot).replay_current_segment(index)
    }

    /// Slot-aware [`graph_segment_count`](Self::graph_segment_count).
    pub fn graph_segment_count_in(&self, slot: DeviceGraphSlot) -> Result<usize> {
        self.graph_slot(slot).current_segment_count()
    }

    /// Slot-aware [`reset_graph`](Self::reset_graph).
    pub fn reset_graph_in(&self, slot: DeviceGraphSlot) -> Result<bool> {
        self.graph_slot(slot).reset_current()
    }

    /// Slot-aware [`has_graph_executable`](Self::has_graph_executable).
    pub fn has_graph_executable_in(&self, slot: DeviceGraphSlot) -> Result<bool> {
        self.graph_slot(slot).has_current_executable()
    }

    fn owned_graph(
        &self,
        owner: DeviceGraphOwner,
        slot: DeviceGraphSlot,
    ) -> Option<Arc<CudaGraphLifecycle>> {
        self.owned_graphs.load().get(&(owner, slot)).cloned()
    }

    fn owned_graph_for_begin(
        &self,
        owner: DeviceGraphOwner,
        slot: DeviceGraphSlot,
        continuation: Option<DeviceGraphToken>,
    ) -> Result<Arc<CudaGraphLifecycle>> {
        let _writer = self.owned_graphs_write.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep: owned CUDA graph registry was poisoned".into())
        })?;
        if let Some(graph) = self.owned_graph(owner, slot) {
            return Ok(graph);
        }
        if continuation.is_some() {
            return Err(EpError::KernelFailed(
                "cuda_ep: CUDA graph continuation token names a retired executor graph".into(),
            ));
        }
        let graph = Arc::new(CudaGraphLifecycle::new(self.stream.clone(), owner, slot));
        let current = self.owned_graphs.load_full();
        let mut next = (*current).clone();
        next.insert((owner, slot), Arc::clone(&graph));
        self.owned_graphs.store(Arc::new(next));
        Ok(graph)
    }

    /// Begin capture in one executor-owned graph namespace.
    pub(crate) fn begin_owned_graph_capture_in(
        &self,
        owner: DeviceGraphOwner,
        slot: DeviceGraphSlot,
        continuation: Option<DeviceGraphToken>,
        kernels: &[&dyn Kernel],
    ) -> Result<DeviceGraphToken> {
        crate::capture::require_subgraph_graph_capturable(kernels)?;
        let mut resources: Vec<DeviceGraphResource> = kernels
            .iter()
            .flat_map(|kernel| kernel.device_graph_resources())
            .collect();
        resources.sort_unstable_by_key(DeviceGraphResource::identity);
        resources.dedup_by_key(|resource| resource.identity());
        let resource_ids = Arc::new(
            resources
                .iter()
                .map(DeviceGraphResource::identity)
                .collect(),
        );
        self.registered_capture_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep: cannot begin CUDA graph capture while another registered capture \
                     is active on the runtime stream"
                        .into(),
                )
            })?;
        let graph = match self.owned_graph_for_begin(owner, slot, continuation) {
            Ok(graph) => graph,
            Err(error) => {
                self.registered_capture_active
                    .store(false, Ordering::Release);
                return Err(error);
            }
        };
        let token = match graph.begin(continuation, resources) {
            Ok(token) => token,
            Err(error) => {
                self.registered_capture_active
                    .store(false, Ordering::Release);
                return Err(error);
            }
        };
        self.active_capture_resource_ids.store(Some(resource_ids));
        Ok(token)
    }

    pub(crate) fn end_owned_graph_capture(&self, token: DeviceGraphToken) -> Result<()> {
        let result = self
            .owned_graph(token.owner(), token.slot())
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep: cannot end a retired executor-owned CUDA graph".into(),
                )
            })?
            .end(token);
        self.registered_capture_active
            .store(false, Ordering::Release);
        self.active_capture_resource_ids.store(None);
        result
    }

    pub(crate) fn abort_owned_graph_capture(&self, token: DeviceGraphToken) -> Result<()> {
        let result = match self.owned_graph(token.owner(), token.slot()) {
            Some(graph) => graph.abort(token),
            None => Ok(()),
        };
        self.registered_capture_active
            .store(false, Ordering::Release);
        self.active_capture_resource_ids.store(None);
        result
    }

    pub(crate) fn replay_owned_graph(&self, token: DeviceGraphToken) -> Result<()> {
        self.owned_graph(token.owner(), token.slot())
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep: cannot replay a retired executor-owned CUDA graph".into(),
                )
            })?
            .replay(token)
    }

    pub(crate) fn replay_owned_graph_segment(
        &self,
        token: DeviceGraphToken,
        index: usize,
    ) -> Result<()> {
        self.owned_graph(token.owner(), token.slot())
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep: cannot replay a retired executor-owned CUDA graph".into(),
                )
            })?
            .replay_segment(token, index)
    }

    pub(crate) fn reset_owned_graph(&self, token: DeviceGraphToken) -> Result<bool> {
        let _writer = self.owned_graphs_write.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep: owned CUDA graph registry was poisoned".into())
        })?;
        let Some(graph) = self.owned_graph(token.owner(), token.slot()) else {
            return Ok(false);
        };
        let (_, had_graph) = graph.reset(token)?;
        Ok(had_graph)
    }

    pub(crate) fn retire_owned_graphs(&self, owner: DeviceGraphOwner) -> Result<()> {
        let _writer = self.owned_graphs_write.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep: owned CUDA graph registry was poisoned".into())
        })?;
        let current = self.owned_graphs.load_full();
        for ((entry_owner, _), graph) in current.iter() {
            if *entry_owner == owner && graph.current_token()?.is_some() {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep: cannot retire graph owner {} while an installation is live",
                    owner.get()
                )));
            }
        }
        let mut next = (*current).clone();
        next.retain(|(entry_owner, _), _| *entry_owner != owner);
        self.owned_graphs.store(Arc::new(next));
        Ok(())
    }

    pub(crate) fn has_owned_graph(&self, token: DeviceGraphToken) -> Result<bool> {
        match self.owned_graph(token.owner(), token.slot()) {
            Some(graph) => graph.has_executable(token),
            None => Ok(false),
        }
    }

    /// Start capture directly on the stream without installing a lifecycle
    /// ownership sink. Test-only proof that sealed launches fail closed rather
    /// than embedding addresses in an externally owned graph.
    #[doc(hidden)]
    pub fn test_begin_unregistered_graph_capture(&self) -> Result<()> {
        self.stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(|error| driver_err("begin unregistered CUDA graph capture", error))?;
        self.unregistered_capture_active
            .store(true, Ordering::Release);
        Ok(())
    }

    /// End and destroy the raw test capture started above.
    #[doc(hidden)]
    pub fn test_end_unregistered_graph_capture(&self) -> Result<()> {
        self.context
            .bind_to_thread()
            .map_err(|error| driver_err("bind unregistered CUDA graph capture context", error))?;
        // SAFETY: the test helper owns the active capture on this stream.
        let graph = unsafe {
            cudarc::driver::result::stream::end_capture(self.stream.cu_stream())
                .map_err(|error| driver_err("end unregistered CUDA graph capture", error))?
        };
        let result = if !graph.is_null() {
            // SAFETY: no executable was instantiated; this helper exclusively
            // owns the fresh raw graph handle.
            unsafe { cudarc::driver::result::graph::destroy(graph) }
                .map_err(|error| driver_err("destroy unregistered CUDA graph", error))
        } else {
            Ok(())
        };
        self.unregistered_capture_active
            .store(false, Ordering::Release);
        result
    }

    /// Snapshot explicit device allocation/free calls made through this runtime.
    pub fn allocation_counts(&self) -> CudaAllocationCounts {
        CudaAllocationCounts {
            allocations: self.allocations.load(Ordering::Relaxed),
            frees: self.frees.load(Ordering::Relaxed),
        }
    }

    /// Snapshot source-attributed [`Self::alloc_raw`] activity.
    pub fn raw_allocation_site_stats(&self) -> Vec<RawDeviceAllocationSiteStats> {
        self.raw_allocation_profile.snapshot()
    }

    /// Snapshot explicit host/device transfer calls made through this runtime.
    pub fn transfer_counts(&self) -> CudaTransferCounts {
        CudaTransferCounts {
            host_to_device: self.host_to_device_copies.load(Ordering::Relaxed),
            device_to_host: self.device_to_host_copies.load(Ordering::Relaxed),
            async_host_to_device: self.async_host_to_device_copies.load(Ordering::Relaxed),
        }
    }

    pub fn transfer_byte_counts(&self) -> CudaTransferByteCounts {
        CudaTransferByteCounts {
            h2d_attempted: self.raw_h2d_attempted.load(Ordering::Relaxed),
            h2d_completed: self.raw_h2d_completed.load(Ordering::Relaxed),
            d2h_attempted: self.raw_d2h_attempted.load(Ordering::Relaxed),
            d2h_completed: self.raw_d2h_completed.load(Ordering::Relaxed),
            d2d_attempted: self.raw_d2d_attempted.load(Ordering::Relaxed),
            d2d_completed: self.raw_d2d_completed.load(Ordering::Relaxed),
            memset_attempted: self.raw_memset_attempted.load(Ordering::Relaxed),
            memset_completed: self.raw_memset_completed.load(Ordering::Relaxed),
        }
    }

    fn reserve_raw_bytes(counter: &AtomicU64, bytes: u64, operation: &str) -> Result<()> {
        counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(bytes)
            })
            .map(|_| ())
            .map_err(|current| {
                EpError::KernelFailed(format!(
                    "cuda_ep: authoritative {operation} byte counter overflows at {current} + \
                     {bytes}; refusing the uncounted CUDA operation"
                ))
            })
    }

    /// Number of unconditional compute-stream synchronization calls.
    pub fn forced_synchronization_count(&self) -> u64 {
        self.forced_synchronizations.load(Ordering::Relaxed)
    }

    /// Validate a requested dynamic shared-memory allocation against the device
    /// limits and, when it exceeds the default (non-opt-in) per-block budget,
    /// opt the function into the larger dynamic size the hardware supports.
    ///
    /// Every architecture caps *non-opt-in* dynamic shared memory at roughly
    /// 48&nbsp;KB, while the opt-in ceiling is device specific (for example
    /// ~100&nbsp;KB on sm_86/sm_89 consumer cards, ~163&nbsp;KB on sm_80, and
    /// ~227&nbsp;KB on sm_90). A kernel that requests more than 48&nbsp;KB without
    /// setting `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` fails to launch
    /// on *any* GPU, and one that requests more than the device's opt-in ceiling
    /// fails on that specific GPU — a request sized for an H200 can therefore
    /// crash a consumer card outright. This helper returns a loud error (never
    /// launching) when even the opt-in maximum cannot satisfy the request, so the
    /// caller can route to a portable fallback instead of hitting a hard launch
    /// failure. The static shared memory the function already reserves is
    /// subtracted from both budgets.
    pub fn configure_dynamic_shared_memory(
        &self,
        function: &CudaFunction,
        requested_bytes: u32,
    ) -> Result<()> {
        let static_shared_memory = function
            .shared_size_bytes()
            .map_err(|error| driver_err("querying CUDA function static shared memory", error))?;
        let static_shared_memory = u32::try_from(static_shared_memory).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep: CUDA function reported invalid static shared memory {static_shared_memory}"
            ))
        })?;
        let optin_budget = self
            .capabilities
            .max_shared_memory_per_block_optin
            .saturating_sub(static_shared_memory);
        let default_budget = self
            .capabilities
            .max_shared_memory_per_block
            .saturating_sub(static_shared_memory);
        match dynamic_shared_memory_optin(requested_bytes, default_budget, optin_budget) {
            Err(()) => Err(EpError::KernelFailed(format!(
                "cuda_ep: kernel requests {requested_bytes} dynamic shared-memory bytes, but \
                 device SM {}.{} allows at most {optin_budget} opt-in bytes \
                 ({static_shared_memory} already reserved statically); route this shape to a \
                 portable kernel instead of launching",
                self.capabilities.compute_capability.0, self.capabilities.compute_capability.1,
            ))),
            Ok(None) => Ok(()),
            Ok(Some(bytes)) => {
                let bytes_i32 = i32::try_from(bytes).map_err(|_| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: dynamic shared-memory request {bytes} exceeds i32"
                    ))
                })?;
                function
                    .set_attribute(
                        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        bytes_i32,
                    )
                    .map_err(|error| {
                        driver_err("opting CUDA function into dynamic shared memory", error)
                    })
            }
        }
    }

    /// Build a power-of-two reduction launch that fits both the function and
    /// device thread/shared-memory limits. If the launch exceeds the legacy
    /// shared-memory limit, opt the function into the required dynamic size.
    pub fn reduction_launch_config(
        &self,
        function: &CudaFunction,
        grid_x: u32,
        preferred_threads: u32,
        bytes_per_thread: u32,
    ) -> Result<LaunchConfig> {
        let function_max_threads = function
            .max_threads_per_block()
            .map_err(|error| driver_err("querying CUDA function max threads", error))?;
        let function_max_threads = u32::try_from(function_max_threads).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep: CUDA function reported invalid max threads {function_max_threads}"
            ))
        })?;
        let static_shared_memory = function
            .shared_size_bytes()
            .map_err(|error| driver_err("querying CUDA function static shared memory", error))?;
        let static_shared_memory = u32::try_from(static_shared_memory).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep: CUDA function reported invalid static shared memory {static_shared_memory}"
            ))
        })?;
        let max_dynamic_shared_memory = self
            .capabilities
            .max_shared_memory_per_block_optin
            .saturating_sub(static_shared_memory);
        let max_threads = self
            .capabilities
            .max_threads_per_block
            .min(function_max_threads);
        let (threads, shared_mem_bytes) = reduction_launch_params(
            preferred_threads,
            max_threads,
            bytes_per_thread,
            max_dynamic_shared_memory,
        )
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep: reduction launch needs {bytes_per_thread} shared-memory bytes per \
                 thread, but device SM {}.{} allows {max_dynamic_shared_memory} dynamic bytes",
                self.capabilities.compute_capability.0, self.capabilities.compute_capability.1,
            ))
        })?;

        let default_dynamic_shared_memory = self
            .capabilities
            .max_shared_memory_per_block
            .saturating_sub(static_shared_memory);
        if shared_mem_bytes > default_dynamic_shared_memory {
            let shared_mem_bytes_i32 = i32::try_from(shared_mem_bytes).map_err(|_| {
                EpError::KernelFailed(format!(
                    "cuda_ep: dynamic shared-memory request {shared_mem_bytes} exceeds i32"
                ))
            })?;
            function
                .set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes_i32,
                )
                .map_err(|error| {
                    driver_err("opting CUDA function into dynamic shared memory", error)
                })?;
        }

        Ok(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes,
        })
    }

    /// Get a [`CudaFunction`] for entry point `entry` in the NVRTC module named
    /// `module_key`, compiling `src` to PTX and loading it on first use and
    /// reusing the cached module thereafter.
    ///
    /// The compile targets the device's detected virtual compute architecture.
    /// If the installed NVRTC emits a PTX ISA newer than the driver accepts,
    /// compilation is retried for the matching real SM architecture and the
    /// resulting CUBIN is loaded instead. An NVRTC failure surfaces the compiler
    /// log via [`nvrtc_err`] (RULES.md #1).
    ///
    /// Compiler output is also persisted by [`crate::kernel_cache`], so only the
    /// first process to see a given kernel pays NVRTC for it. That matters most
    /// for decode-only kernels: they are first launched inside the first decode
    /// step, where the compile lands as a single multi-hundred-millisecond
    /// inter-token stall.
    pub fn nvrtc_function(
        &self,
        module_key: &'static str,
        src: &str,
        entry: &str,
    ) -> Result<CudaFunction> {
        require(CudaLibrary::Nvrtc).map_err(|message| {
            EpError::KernelFailed(format!(
                "cuda_ep: {message}; CPU execution remains available"
            ))
        })?;
        self.bind()?;
        let module = {
            let mut cache = self.modules.lock().expect("cuda_ep module cache poisoned");
            if let Some(m) = cache.get(module_key) {
                m.clone()
            } else {
                let include_paths = nvrtc_include_paths();
                // Loading a module into the context synchronizes the device,
                // invalidating a CUDA graph capture in progress on another
                // thread. Held across the whole compile-and-load block: the
                // NVRTC half is host work, but splitting the section would only
                // add a second gate acquisition for no benefit. See
                // `alloc_raw`.
                let _section = capture_gate::synchronizing_section();
                let m = if self.nvrtc_cubin_fallback.load(Ordering::Relaxed) {
                    self.load_nvrtc_cubin(module_key, src, &include_paths)?
                } else {
                    let ptx = self.nvrtc_ptx(module_key, src, &include_paths)?;
                    match self.context.load_module(ptx) {
                        Ok(module) => module,
                        Err(error)
                            if error.0
                                == cudarc::driver::sys::CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION =>
                        {
                            self.nvrtc_cubin_fallback.store(true, Ordering::Relaxed);
                            self.load_nvrtc_cubin(module_key, src, &include_paths)?
                        }
                        Err(error) => {
                            return Err(driver_err(
                                &format!("loading NVRTC module '{module_key}'"),
                                error,
                            ));
                        }
                    }
                };
                cache.insert(module_key, m.clone());
                m
            }
        };
        module
            .load_function(entry)
            .map_err(|e| driver_err(&format!("loading NVRTC function '{entry}'"), e))
    }

    /// Resolve a prepared raw function handle for allocation-free launches.
    pub(crate) fn nvrtc_raw_function(
        &self,
        module_key: &'static str,
        src: &str,
        entry: &str,
    ) -> Result<RawCudaFunction> {
        require(CudaLibrary::Nvrtc).map_err(|message| {
            EpError::KernelFailed(format!(
                "cuda_ep: {message}; CPU execution remains available"
            ))
        })?;
        self.bind()?;
        let module = {
            let mut cache = self
                .raw_modules
                .lock()
                .expect("cuda_ep raw module cache poisoned");
            if let Some(module) = cache.get(module_key) {
                module.clone()
            } else {
                let include_paths = nvrtc_include_paths();
                let _section = capture_gate::synchronizing_section();
                let module = if self.nvrtc_cubin_fallback.load(Ordering::Relaxed) {
                    let image = self.nvrtc_cubin_image(module_key, src, &include_paths)?;
                    self.load_raw_module(module_key, image.as_ptr().cast())?
                } else {
                    let ptx = self.nvrtc_ptx(module_key, src, &include_paths)?;
                    let image = CString::new(ptx.to_src()).map_err(|_| {
                        EpError::KernelFailed(format!(
                            "cuda_ep: loading NVRTC module '{module_key}': PTX contains a NUL byte"
                        ))
                    })?;
                    match self.load_raw_module(module_key, image.as_ptr().cast()) {
                        Ok(module) => module,
                        Err(EpError::KernelFailed(message))
                            if message.contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION") =>
                        {
                            self.nvrtc_cubin_fallback.store(true, Ordering::Relaxed);
                            let image = self.nvrtc_cubin_image(module_key, src, &include_paths)?;
                            self.load_raw_module(module_key, image.as_ptr().cast())?
                        }
                        Err(error) => return Err(error),
                    }
                };
                cache.insert(module_key, module.clone());
                module
            }
        };
        let name = CString::new(entry).expect("static kernel entry cannot contain a NUL byte");
        // SAFETY: `module` remains retained by the returned function handle.
        let function = unsafe { cudarc::driver::result::module::get_function(module.module, name) }
            .map_err(|error| driver_err(&format!("loading raw NVRTC function '{entry}'"), error))?;
        Ok(RawCudaFunction {
            function,
            _module: module,
        })
    }

    fn load_raw_module(
        &self,
        module_key: &'static str,
        image: *const c_void,
    ) -> Result<Arc<RawCudaModule>> {
        // SAFETY: callers provide either a live NUL-terminated PTX string or a
        // live CUBIN image for the duration of this synchronous load call.
        let module =
            unsafe { cudarc::driver::result::module::load_data(image) }.map_err(|error| {
                driver_err(&format!("loading raw NVRTC module '{module_key}'"), error)
            })?;
        Ok(Arc::new(RawCudaModule {
            module,
            context: self.context.clone(),
        }))
    }

    /// PTX for `module_key`, from the on-disk cache when possible.
    ///
    /// A cached hit is returned as PTX source rather than a compiled image;
    /// both load through the same driver entry point, and the driver keeps its
    /// own PTX→SASS cache, so the only work skipped here is the NVRTC frontend.
    /// That frontend is the expensive half for this crate's templated kernels.
    fn nvrtc_ptx(
        &self,
        module_key: &'static str,
        src: &str,
        include_paths: &[String],
    ) -> Result<cudarc::nvrtc::Ptx> {
        let key = kernel_cache::CacheKey {
            module_key,
            source: src,
            arch: &self.ptx_arch,
            include_paths,
            kind: kernel_cache::ArtifactKind::Ptx,
        };
        if let Some(bytes) = kernel_cache::load(&key)
            && let Ok(text) = String::from_utf8(bytes)
        {
            return Ok(cudarc::nvrtc::Ptx::from_src(text));
        }
        let opts = cudarc::nvrtc::CompileOptions {
            include_paths: include_paths.to_vec(),
            options: vec![format!("--gpu-architecture={}", self.ptx_arch)],
            ..Default::default()
        };
        let started = Instant::now();
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts)
            .map_err(|e| nvrtc_err(&format!("compiling NVRTC module '{module_key}'"), e))?;
        kernel_cache::record_compile(started.elapsed());
        kernel_cache::store(&key, ptx.to_src().as_bytes());
        Ok(ptx)
    }

    fn load_nvrtc_cubin(
        &self,
        module_key: &'static str,
        src: &str,
        include_paths: &[String],
    ) -> Result<Arc<CudaModule>> {
        let image = self.nvrtc_cubin_image(module_key, src, include_paths)?;
        self.context
            .load_module(cudarc::nvrtc::Ptx::from_binary(image))
            .map_err(|error| {
                driver_err(
                    &format!("loading NVRTC CUBIN fallback module '{module_key}'"),
                    error,
                )
            })
    }

    fn nvrtc_cubin_image(
        &self,
        module_key: &'static str,
        src: &str,
        include_paths: &[String],
    ) -> Result<Vec<u8>> {
        let key = kernel_cache::CacheKey {
            module_key,
            source: src,
            arch: &self.cubin_arch,
            include_paths,
            kind: kernel_cache::ArtifactKind::Cubin,
        };
        let image = match kernel_cache::load(&key) {
            Some(image) => image,
            None => {
                let started = Instant::now();
                let image = self.compile_nvrtc_cubin(module_key, src, include_paths)?;
                kernel_cache::record_compile(started.elapsed());
                kernel_cache::store(&key, &image);
                image
            }
        };
        Ok(image)
    }

    fn compile_nvrtc_cubin(
        &self,
        module_key: &'static str,
        src: &str,
        include_paths: &[String],
    ) -> Result<Vec<u8>> {
        let source = CString::new(src).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep: compiling NVRTC module '{module_key}': source contains a NUL byte"
            ))
        })?;
        let name = CString::new(module_key).expect("static module key cannot contain a NUL byte");
        let program =
            cudarc::nvrtc::result::create_program(source.as_c_str(), Some(name.as_c_str()))
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: creating NVRTC CUBIN module '{module_key}': {error:?}"
                    ))
                })?;
        let mut options = include_paths
            .iter()
            .map(|path| format!("--include-path={path}"))
            .collect::<Vec<_>>();
        options.push(format!("--gpu-architecture={}", self.cubin_arch));

        // SAFETY: `program` is live until the matching destroy call below.
        let compile_result = unsafe { cudarc::nvrtc::result::compile_program(program, &options) };
        if let Err(error) = compile_result {
            // SAFETY: compilation may fail, but the live program still owns its log.
            let log = unsafe { cudarc::nvrtc::result::get_program_log(program) }
                .ok()
                .map(|bytes| {
                    // SAFETY: NVRTC returns a NUL-terminated compiler log.
                    unsafe { CStr::from_ptr(bytes.as_ptr()) }
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_else(|| "<compiler log unavailable>".into());
            // SAFETY: this is the single destroy for the live program.
            let _ = unsafe { cudarc::nvrtc::result::destroy_program(program) };
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: compiling NVRTC CUBIN module '{module_key}' failed ({error:?}); compiler log:\n{log}"
            )));
        }

        let cubin: Result<Vec<u8>> = (|| {
            let mut size = 0usize;
            // SAFETY: `program` compiled successfully and `size` is writable.
            unsafe { cudarc::nvrtc::sys::nvrtcGetCUBINSize(program, &mut size) }
                .result()
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: getting NVRTC CUBIN size for '{module_key}': {error:?}"
                    ))
                })?;
            let mut image = vec![0u8; size];
            // SAFETY: `image` has the exact size reported by NVRTC.
            unsafe { cudarc::nvrtc::sys::nvrtcGetCUBIN(program, image.as_mut_ptr().cast()) }
                .result()
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: getting NVRTC CUBIN for '{module_key}': {error:?}"
                    ))
                })?;
            Ok(image)
        })();
        // SAFETY: this is the single destroy for the live program.
        let destroy_result = unsafe { cudarc::nvrtc::result::destroy_program(program) };
        let image = cubin?;
        destroy_result.map_err(|error| {
            EpError::KernelFailed(format!(
                "cuda_ep: destroying NVRTC CUBIN program '{module_key}': {error:?}"
            ))
        })?;
        Ok(image)
    }

    pub fn require_nvrtc_half_headers(&self, op: &str) -> Result<()> {
        // Ask for the header this actually needs rather than for a non-empty
        // include list. The list may be non-empty because the `crt/` tree from
        // `nvidia-cuda-nvcc` was found while `cuda_fp16.h` is still missing, and
        // then this would wave the kernel through to fail inside NVRTC instead.
        if !nvrtc_include_paths()
            .iter()
            .any(|path| Path::new(path).join("cuda_fp16.h").is_file())
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: f16/bf16 NVRTC kernels require cuda_fp16.h and cuda_bf16.h. \
                 Install the CUDA runtime headers (for pip CUDA 13: `pip install \
                 nvidia-cuda-runtime`; alternatively set CUDA_HOME/CUDA_PATH)."
            )));
        }
        Ok(())
    }

    /// Headers the tensor-core kernels need on top of the half-precision ones.
    ///
    /// `mma.h` ships in `nvidia-cuda-runtime`, but it includes `crt/mma.h`,
    /// which ships in `nvidia-cuda-nvcc`. Installing only the first gets as far
    /// as NVRTC and then fails with
    /// `catastrophic error: cannot open source file "crt/mma.h"` — an error
    /// that names a file rather than the wheel that carries it, which is a long
    /// detour from the fix.
    pub fn require_nvrtc_tensor_core_headers(&self, op: &str) -> Result<()> {
        self.require_nvrtc_half_headers(op)?;
        if !nvrtc_include_paths()
            .iter()
            .any(|path| Path::new(path).join("crt/mma.h").is_file())
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: tensor-core NVRTC kernels include <mma.h>, which needs the crt/ \
                 headers. These are packaged separately from the CUDA runtime headers: `pip \
                 install nvidia-cuda-nvcc` (alternatively set CUDA_HOME/CUDA_PATH to a full \
                 toolkit, which carries both)."
            )));
        }
        Ok(())
    }
    pub fn bind(&self) -> Result<()> {
        self.context
            .bind_to_thread()
            .map_err(|e| driver_err("bind_to_thread", e))
    }

    /// Block until all submitted work on the EP's dedicated stream completes.
    ///
    /// Deferred by default (eager decode is made consistent with the captured
    /// path); this becomes a no-op unless `ONNX_GENAI_DEFER_EAGER_SYNC=0`
    /// restores the old always-sync behavior. The trailing per-op eager syncs
    /// are redundant because (a) kernel→kernel ordering is guaranteed by the
    /// single in-order EP stream and (b) every host-visible read (`dtoh`/`dtod`)
    /// issues its own [`force_synchronize`] before the synchronous copy. Eliding
    /// these lets eager decode pipeline launches the way a captured graph does.
    pub fn synchronize(&self) -> Result<()> {
        if self.defer_eager_sync.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.force_synchronize()
    }

    /// Unconditional stream drain. Used by host-visible reads (`dtoh`/`dtod`)
    /// that must observe fully-produced bytes regardless of the eager-sync
    /// deferral flag.
    fn force_synchronize(&self) -> Result<()> {
        self.forced_synchronizations.fetch_add(1, Ordering::Relaxed);
        self.stream
            .synchronize()
            .map_err(|e| driver_err("stream synchronize", e))
    }

    /// Unconditional compute-stream drain used as a **correctness barrier**
    /// before releasing or remapping device memory a kernel may still be
    /// reading — e.g. immediately before `cuMemUnmap` of an evicted weight
    /// granule in `weight_paging.rs`.
    ///
    /// Unlike [`synchronize`], this ignores the `defer_eager_sync` deferral
    /// (#1383). Eliding a *trailing per-op* eager sync is safe — the single
    /// in-order EP stream preserves kernel→kernel ordering — but eliding a
    /// *pre-unmap* drain is a use-after-unmap: the granule is unmapped while an
    /// in-flight decode kernel still references its VA. This was observed as
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` on the weight-offload repro (11/11 runs).
    ///
    /// A silent variant — a late read returning a *successor weight's* bytes
    /// under stable-slot reuse (#716) — was hypothesised and then **falsified**:
    /// stable slots are keyed by weight `key`, a slot's VA is reused only for
    /// the same key, and on collision admission refuses rather than remaps, so a
    /// stale read hits either decommitted physical memory (faults) or the same
    /// weight's byte-identical bytes — never a different weight. The 11/11 runs
    /// all crashed; 0 diverged. The hazard here is therefore a loud fault, not
    /// silent corruption. See #1439 for the full chain.
    ///
    /// Any caller that must know all prior compute has retired before
    /// freeing/remapping memory MUST use this, never `synchronize()`.
    pub fn drain_for_unmap(&self) -> Result<()> {
        self.force_synchronize()
    }

    /// Toggle the eager-sync deferral at runtime (see [`synchronize`]).
    pub fn set_defer_eager_sync(&self, enabled: bool) {
        self.defer_eager_sync.store(enabled, Ordering::Relaxed);
    }

    /// Whether the eager-fast path is active. When true, kernels skip their
    /// eager-only host-side validation readbacks (index/position bounds checks)
    /// — these are numerically inert (a correct model never trips them) and the
    /// captured path already relies on the device error latch instead of a
    /// per-op D2H. Eliding them removes the blocking scalar D2H that otherwise
    /// serializes each eager op against the GPU.
    pub fn eager_sync_deferred(&self) -> bool {
        self.defer_eager_sync.load(Ordering::Relaxed)
    }

    /// Whether the EP's compute stream is currently capturing into a CUDA graph.
    /// A stream synchronize is illegal during capture, so device-resident kernels
    /// use this to skip the trailing sync while a graph is being recorded.
    pub fn is_capturing(&self) -> Result<bool> {
        Ok(self.graph_capture_status()?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
    }

    pub(crate) fn require_registered_address_capture(
        &self,
        identity: usize,
        label: &str,
    ) -> Result<()> {
        if self.unregistered_capture_active.load(Ordering::Acquire)
            && !self.registered_capture_active.load(Ordering::Acquire)
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: active CUDA graph capture has no registered ownership token for \
                 sealed {label}; refusing to embed its device addresses"
            )));
        }
        if self.registered_capture_active.load(Ordering::Acquire)
            && !self
                .active_capture_resource_ids
                .load()
                .as_ref()
                .is_some_and(|ids| ids.contains(&identity))
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: active CUDA graph capture did not retain the ownership token for \
                  sealed {label}; refusing to embed its device addresses"
            )));
        }
        Ok(())
    }

    /// Number of CUDA graph lifecycle mutex acquisitions by this runtime.
    pub fn graph_lifecycle_lock_acquisition_count(&self) -> u64 {
        self.graph.lock_acquisition_count() + self.verify_graph.lock_acquisition_count()
    }

    /// Positive control for graph-lifecycle lock instrumentation.
    #[doc(hidden)]
    pub fn test_acquire_graph_lifecycle_lock(&self) -> Result<()> {
        self.graph.test_acquire_lock()
    }

    /// This runtime's process-unique identity. See [`Self::runtime_id`].
    pub(crate) fn runtime_id(&self) -> u64 {
        self.runtime_id
    }

    /// The interleaved copy of the `bytes`-byte int4 packed weight at `packed`,
    /// built once per weight and released when this runtime drops. Returns
    /// `(pointer, warm)`; see [`crate::interleave_cache::InterleaveCache::ensure`].
    pub(crate) fn ensure_interleaved_int4(
        &self,
        packed: CUdeviceptr,
        bytes: usize,
    ) -> Result<(CUdeviceptr, bool)> {
        self.interleave.ensure(self, packed, bytes)
    }

    /// Release every interleaved weight copy this runtime holds.
    ///
    /// The backstop, for anything still cached when the runtime itself goes
    /// away. The load-bearing eviction is [`Self::invalidate_interleaved_for`],
    /// which runs per buffer as the provider frees it.
    #[cfg(test)]
    pub(crate) fn release_interleaved_weights(&self) {
        self.interleave.release_all(self);
    }

    /// Drop any interleaved copy derived from the buffer at `base`.
    ///
    /// The provider calls this as it frees a device buffer, passing the whole
    /// allocation — an entry is keyed on a weight's data pointer, which may sit
    /// at an offset inside the buffer being freed. That address names that
    /// weight only until this moment; past it the allocator may hand it to the
    /// next weight of the same size, whose interleave would otherwise be served
    /// out of this entry (#1726). A provider outlives its executors — sibling
    /// plans share one, and the control-flow child-executor cache evicts plans
    /// and frees their initializers back into the same arena — so runtime
    /// scoping alone leaves that window open and this is what closes it.
    pub(crate) fn invalidate_interleaved_for(&self, base: CUdeviceptr, len: usize) {
        self.interleave.invalidate(self, base, len);
    }

    /// Record that device weight offload is on, so weight buffers may be paged.
    ///
    /// One-way: a runtime that has ever paged weights can have had an address
    /// recycled behind the interleave cache's back, so there is no sound way
    /// back to caching for the rest of its life.
    pub(crate) fn set_weights_may_be_paged(&self) {
        self.weights_may_be_paged
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Whether weight buffers on this runtime may be paged by the offload path.
    pub(crate) fn weights_may_be_paged(&self) -> bool {
        self.weights_may_be_paged
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Interleaved weight copies this runtime currently holds.
    #[cfg(test)]
    pub(crate) fn interleaved_weight_count(&self) -> usize {
        self.interleave.len()
    }

    /// Allocate `bytes` (>= 1) of device memory, returning the raw device
    /// pointer. Binds the context first.
    #[track_caller]
    pub fn alloc_raw(&self, bytes: usize) -> Result<CUdeviceptr> {
        let location = std::panic::Location::caller();
        self.bind()?;
        let requested = bytes.max(1);
        let class = raw_pool_size_class(requested);
        if let Some(ptr) = self.take_pooled(class) {
            let mut observation = match self.prepare_observation(&[EventSpec::new(
                ObservedCategory::DeviceAllocation,
                ObservedBoundary::RuntimeDevicePoolReuse,
                ObservedStatus::Reclaimed,
                class as u64,
            )]) {
                Ok(observation) => observation,
                Err(error) => {
                    self.return_pooled(class, ptr);
                    return Err(error);
                }
            };
            self.raw_pool_hits.fetch_add(1, Ordering::Relaxed);
            self.raw_allocation_profile
                .record(location, requested, class, true);
            Self::commit_observation(&mut observation)?;
            return Ok(ptr);
        }
        let mut observation = self.prepare_observation(&[EventSpec::new(
            ObservedCategory::DeviceAllocation,
            ObservedBoundary::RuntimeDeviceAllocate,
            ObservedStatus::Committed,
            class as u64,
        )])?;
        // Past the pool: this path makes a real `cuMemAlloc`, which
        // synchronizes the device and would invalidate a CUDA graph capture
        // running on another thread. Pool hits returned above take no lock, so
        // decode's steady state stays entirely off the gate.
        let _section = capture_gate::synchronizing_section();
        // SAFETY: `malloc_sync` returns a fresh device allocation on the current
        // (bound) context; we own it and free it exactly once via `free_raw`.
        let mut allocated = unsafe { cudarc::driver::result::malloc_sync(class) };
        if allocated.is_err() {
            // Pooled blocks are device memory this runtime is holding back from
            // everyone else. Releasing them before reporting out-of-memory is
            // what keeps a pool from behaving like a leak under pressure.
            if let Err(error) = self.drain_raw_pool() {
                return Err(Self::fail_observation(
                    &mut observation,
                    ObservedStatus::Failed,
                    error,
                ));
            }
            // SAFETY: as above.
            allocated = unsafe { cudarc::driver::result::malloc_sync(class) };
        }
        let ptr = allocated.map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err("cuMemAlloc", error),
            )
        })?;
        self.raw_pool_classes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(ptr, class);
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.raw_allocation_profile
            .record(location, requested, class, false);
        Self::commit_observation(&mut observation)?;
        Ok(ptr)
    }

    /// Blocks `alloc_raw` served from the pool instead of the driver.
    pub fn raw_pool_hits(&self) -> u64 {
        self.raw_pool_hits.load(Ordering::Relaxed)
    }

    /// Device bytes currently held in the `alloc_raw` pool.
    pub fn raw_pool_retained_bytes(&self) -> u64 {
        self.raw_pool_retained.load(Ordering::Relaxed)
    }

    /// The cuBLASLt workspace shared by every GEMM on this runtime's stream.
    ///
    /// Allocated on first use and retained for the runtime's lifetime, so a
    /// warm call makes no allocation and stays CUDA-graph capture-safe. Callers
    /// must not free the returned pointer.
    ///
    /// Sharing is sound because cuBLASLt scratch carries nothing between calls
    /// and work on one stream is serial: two GEMMs on this runtime cannot be
    /// mid-flight at once, so they cannot observe each other's scratch.
    pub fn shared_blas_workspace(&self, bytes: usize) -> Result<CUdeviceptr> {
        let mut slot = self
            .shared_blas_workspace
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(ptr) = *slot {
            return Ok(ptr);
        }
        let ptr = self.alloc_raw(bytes)?;
        *slot = Some(ptr);
        Ok(ptr)
    }

    fn take_pooled(&self, class: usize) -> Option<CUdeviceptr> {
        let mut pool = self.raw_pool.lock().unwrap_or_else(|e| e.into_inner());
        let ptr = pool.get_mut(&class)?.pop()?;
        self.raw_pool_retained
            .fetch_sub(class as u64, Ordering::Relaxed);
        Some(ptr)
    }

    fn return_pooled(&self, class: usize, ptr: CUdeviceptr) {
        self.raw_pool
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry(class)
            .or_default()
            .push(ptr);
        self.raw_pool_retained
            .fetch_add(class as u64, Ordering::Relaxed);
    }

    /// Hold a freed block for reuse, or report that the cap leaves no room.
    fn retain_pooled(&self, ptr: CUdeviceptr) -> bool {
        let limit = raw_pool_limit_bytes();
        if limit == 0 {
            return false;
        }
        let Some(class) = self
            .raw_pool_classes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&ptr)
            .copied()
        else {
            // Not ours to pool: an allocation made before this runtime tracked
            // classes, or a double free the caller is responsible for.
            return false;
        };
        let class_bytes = class as u64;
        // Reserve before inserting, so two threads cannot both claim the last
        // of the budget.
        if self
            .raw_pool_retained
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                (held + class_bytes <= limit).then_some(held + class_bytes)
            })
            .is_err()
        {
            return false;
        }
        self.raw_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(class)
            .or_default()
            .push(ptr);
        true
    }

    /// Return every pooled block to the driver.
    fn drain_raw_pool(&self) -> Result<()> {
        let drained: Vec<(CUdeviceptr, usize)> = {
            let mut pool = self.raw_pool.lock().unwrap_or_else(|e| e.into_inner());
            let mut drained = Vec::new();
            for (class, blocks) in pool.drain() {
                self.raw_pool_retained
                    .fetch_sub((class as u64) * blocks.len() as u64, Ordering::Relaxed);
                drained.extend(blocks.into_iter().map(|ptr| (ptr, class)));
            }
            drained
        };
        // Draining the pool issues one real `cuMemFree` per block; see
        // `alloc_raw`.
        let _section = capture_gate::synchronizing_section();
        let mut classes = self
            .raw_pool_classes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut drained = drained.into_iter();
        while let Some((ptr, class)) = drained.next() {
            let mut observation = match self.prepare_observation(&[EventSpec::new(
                ObservedCategory::DeviceRelease,
                ObservedBoundary::RuntimeDeviceRelease,
                ObservedStatus::Reclaimed,
                class as u64,
            )]) {
                Ok(observation) => observation,
                Err(error) => {
                    self.return_pooled(class, ptr);
                    for (remaining_ptr, remaining_class) in drained {
                        self.return_pooled(remaining_class, remaining_ptr);
                    }
                    return Err(error);
                }
            };
            // SAFETY: every pooled block came from `malloc_sync` on this
            // runtime's context and is freed exactly once — it was removed from
            // the pool here, so `free_raw` cannot also free it.
            if let Err(error) = unsafe { cudarc::driver::result::free_sync(ptr) } {
                self.return_pooled(class, ptr);
                for (remaining_ptr, remaining_class) in drained {
                    self.return_pooled(remaining_class, remaining_ptr);
                }
                return Err(Self::fail_observation(
                    &mut observation,
                    ObservedStatus::Failed,
                    driver_err("cuMemFree while draining raw allocation pool", error),
                ));
            }
            classes.remove(&ptr);
            self.frees.fetch_add(1, Ordering::Relaxed);
            Self::commit_observation(&mut observation)?;
        }
        Ok(())
    }

    #[cfg(feature = "gpu-tests")]
    pub fn test_drain_raw_pool(&self) -> Result<()> {
        self.drain_raw_pool()
    }

    /// Free a device pointer previously returned by [`CudaRuntime::alloc_raw`].
    ///
    /// # Safety
    /// `ptr` must have come from this runtime's `alloc_raw` and not been freed.
    pub unsafe fn free_raw(&self, ptr: CUdeviceptr) -> Result<()> {
        self.bind()?;
        let class = self
            .raw_pool_classes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&ptr)
            .copied()
            .unwrap_or(0);
        let mut observation = self.prepare_observation(&[EventSpec::new(
            ObservedCategory::DeviceRelease,
            ObservedBoundary::RuntimeDeviceRelease,
            ObservedStatus::Reclaimed,
            class as u64,
        )])?;
        // A pooled free deliberately does not count here. `frees` exists to
        // report driver calls — a `cuMemFree` during graph capture invalidates
        // the capture — and retaining a block makes no driver call at all, so
        // counting it would report a capture hazard that did not happen.
        if self.retain_pooled(ptr) {
            if let Some(observation) = observation.as_mut() {
                observation.set_boundary(0, ObservedBoundary::RuntimeDevicePoolReclaim);
            }
            Self::commit_observation(&mut observation)?;
            return Ok(());
        }
        // A real `cuMemFree` follows; see `alloc_raw`.
        let _section = capture_gate::synchronizing_section();
        self.raw_pool_classes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&ptr);
        // SAFETY: caller upholds the single-free contract.
        unsafe { cudarc::driver::result::free_sync(ptr) }.map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err("cuMemFree", error),
            )
        })?;
        self.frees.fetch_add(1, Ordering::Relaxed);
        Self::commit_observation(&mut observation)?;
        Ok(())
    }

    /// Copy `src` host → device (H2D), completing before return.
    ///
    /// The EP compute stream is non-blocking, so CUDA's legacy-default-stream
    /// semantics do not order `cuMemcpyHtoD` against kernels using `dst`.
    /// Retiring prior compute before the copy prevents an overwrite racing a
    /// previous user. Synchronizing the default stream afterwards proves the
    /// final DMA reached `dst`; the host wait is required for pageable memory
    /// because `cuMemcpyHtoD` may return after staging while DMA is still in
    /// flight. A later consumer is submitted only after that completion.
    /// Async prefetch uses [`CudaRuntime::htod_async`] on the dedicated transfer
    /// stream with explicit copy/compute fences instead.
    ///
    /// # Safety
    /// `dst` is a live device allocation of at least `src.len()` bytes.
    pub unsafe fn htod(&self, src: &[u8], dst: CUdeviceptr) -> Result<()> {
        self.bind()?;
        let _section = capture_gate::synchronizing_section();
        if self.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep: synchronous H2D upload is illegal during CUDA graph capture; \
                 warm the host data before capture or use htod_async with an explicit fence"
                    .into(),
            ));
        }
        let bytes = src.len() as u64;
        let mut observation = self.prepare_observation(&[
            EventSpec::new(
                ObservedCategory::H2d,
                ObservedBoundary::RuntimeH2d,
                ObservedStatus::Submitted,
                bytes,
            ),
            EventSpec::new(
                ObservedCategory::H2d,
                ObservedBoundary::RuntimeH2d,
                ObservedStatus::Completed,
                bytes,
            ),
        ])?;
        Self::reserve_raw_bytes(&self.raw_h2d_attempted, bytes, "H2D attempted")?;
        self.stream.synchronize().map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err(
                    "cuStreamSynchronize(compute) before synchronous cuMemcpyHtoD",
                    error,
                ),
            )
        })?;
        // SAFETY: bound context; `dst` covers `src.len()` bytes per the contract.
        unsafe { cudarc::driver::result::memcpy_htod_sync(dst, src) }.map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err("cuMemcpyHtoD", error),
            )
        })?;
        Self::mark_observation_submitted(&mut observation);
        // SAFETY: a null CUstream selects the current context's legacy default
        // stream. Calling the raw API avoids constructing an allocating Arc
        // wrapper on every request-boundary validation reset.
        unsafe { cudarc::driver::result::stream::synchronize(std::ptr::null_mut()) }.map_err(
            |error| {
                Self::fail_observation(
                    &mut observation,
                    ObservedStatus::Failed,
                    driver_err(
                        "cuStreamSynchronize(default) after synchronous cuMemcpyHtoD",
                        error,
                    ),
                )
            },
        )?;
        self.raw_h2d_completed.fetch_add(bytes, Ordering::Relaxed);
        self.host_to_device_copies.fetch_add(1, Ordering::Relaxed);
        Self::commit_observation(&mut observation)?;
        Ok(())
    }

    /// Copy `dst.len()` bytes device → host (D2H). `src` must be large enough.
    ///
    /// # Safety
    /// `src` is a live device allocation of at least `dst.len()` bytes.
    pub unsafe fn dtoh(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<()> {
        self.bind()?;
        let bytes = dst.len() as u64;
        let mut observation = self.prepare_observation(&[
            EventSpec::new(
                ObservedCategory::D2h,
                ObservedBoundary::RuntimeD2h,
                ObservedStatus::Submitted,
                bytes,
            ),
            EventSpec::new(
                ObservedCategory::D2h,
                ObservedBoundary::RuntimeD2h,
                ObservedStatus::Completed,
                bytes,
            ),
        ])?;
        Self::reserve_raw_bytes(&self.raw_d2h_attempted, bytes, "D2H attempted")?;
        // A stream drain plus a synchronous copy on the null stream; see
        // `alloc_raw`.
        let _section = capture_gate::synchronizing_section();
        // Kernels enqueue work on the EP's dedicated non-default stream. Wait
        // before issuing the synchronous driver copy so the host never observes
        // bytes that were still being produced on that stream.
        self.force_synchronize().map_err(|error| {
            Self::fail_observation(&mut observation, ObservedStatus::Failed, error)
        })?;
        // SAFETY: bound context; `src` covers `dst.len()` bytes per the contract.
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(dst, src) }.map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err("cuMemcpyDtoH", error),
            )
        })?;
        Self::mark_observation_submitted(&mut observation);
        self.raw_d2h_completed.fetch_add(bytes, Ordering::Relaxed);
        self.device_to_host_copies.fetch_add(1, Ordering::Relaxed);
        Self::commit_observation(&mut observation)?;
        Ok(())
    }

    /// Copy `bytes` device → device (D2D).
    ///
    /// # Safety
    /// Both pointers are live allocations of at least `bytes` bytes.
    pub unsafe fn dtod(&self, src: CUdeviceptr, dst: CUdeviceptr, bytes: usize) -> Result<()> {
        self.bind()?;
        let mut observation = self.prepare_observation(&[
            EventSpec::new(
                ObservedCategory::D2d,
                ObservedBoundary::RuntimeD2d,
                ObservedStatus::Submitted,
                bytes as u64,
            ),
            EventSpec::new(
                ObservedCategory::D2d,
                ObservedBoundary::RuntimeD2d,
                ObservedStatus::Completed,
                bytes as u64,
            ),
        ])?;
        Self::reserve_raw_bytes(&self.raw_d2d_attempted, bytes as u64, "D2D attempted")?;
        // Kernels enqueue their writes on the EP's dedicated non-default stream,
        // but `cuMemcpyDtoD` issues on the legacy default stream. On a
        // non-blocking compute stream the two are not implicitly ordered, so the
        // copy can race a producer kernel that is still writing `src` (or a
        // consumer that already queued a read of `dst`). Drain the EP stream
        // first so the synchronous copy always sees fully-produced bytes. This
        // mirrors `dtoh`, which synchronizes for the same reason.
        let _section = capture_gate::synchronizing_section();
        self.force_synchronize().map_err(|error| {
            Self::fail_observation(&mut observation, ObservedStatus::Failed, error)
        })?;
        // SAFETY: bound context; both endpoints cover `bytes` per the contract.
        unsafe { cudarc::driver::result::memcpy_dtod_sync(dst, src, bytes) }.map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err("cuMemcpyDtoD", error),
            )
        })?;
        Self::mark_observation_submitted(&mut observation);
        self.raw_d2d_completed
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Self::commit_observation(&mut observation)?;
        Ok(())
    }

    /// Enqueue a device → device copy on the EP stream.
    ///
    /// # Safety
    /// Both pointers are live allocations of at least `bytes` bytes and remain
    /// live until the stream has completed the copy.
    pub unsafe fn dtod_async(
        &self,
        src: CUdeviceptr,
        dst: CUdeviceptr,
        bytes: usize,
    ) -> Result<()> {
        self.bind()?;
        let mut observation = self.prepare_observation(&[
            EventSpec::new(
                ObservedCategory::D2d,
                ObservedBoundary::RuntimeD2d,
                ObservedStatus::Submitted,
                bytes as u64,
            ),
            EventSpec::new(
                ObservedCategory::D2d,
                ObservedBoundary::AsyncCompletionUnsupported,
                ObservedStatus::Unsupported,
                0,
            ),
        ])?;
        Self::reserve_raw_bytes(&self.raw_d2d_attempted, bytes as u64, "async D2D attempted")?;
        // SAFETY: bound context; both endpoints cover `bytes` and the runtime
        // owns the stream on which the copy is ordered.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, self.stream.cu_stream())
        }
        .map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err("cuMemcpyDtoDAsync", error),
            )
        })?;
        Self::mark_observation_submitted(&mut observation);
        Self::commit_observation(&mut observation)?;
        Ok(())
    }

    // ── Phase-4 compute/transfer overlap: async H2D prefetch primitives ──────
    //
    // These build the EP-side *mechanism* the executor's double-buffering
    // *strategy* drives: a stream-ordered host→device copy on the dedicated
    // `copy_stream`, plus completion events that order the compute stream after a
    // transfer (`compute_wait_fence`) and — for double-buffer reuse — the copy
    // stream after a prior consumer (`copy_wait_fence`). No primitive blocks the
    // host; ordering is entirely through CUDA events so a prefetch of the next
    // expert's weights overlaps the current wave's kernels.

    /// Enqueue an asynchronous host → device copy of `src` on the dedicated
    /// transfer stream (not the compute stream), so it overlaps compute.
    /// For genuine overlap `src` should be page-locked (pinned) host memory —
    /// see [`CudaRuntime::alloc_pinned`]; a pageable `src` still copies correctly
    /// but the driver may stage it synchronously.
    ///
    /// # Safety
    /// `dst` is a live device allocation of at least `src.len()` bytes and `src`
    /// must remain valid and unmoved until the transfer completes (order a
    /// consumer after it with [`CudaRuntime::record_copy_fence`] +
    /// [`CudaRuntime::compute_wait_fence`], or drain with [`CudaRuntime::sync_copy_stream`]).
    pub unsafe fn htod_async(&self, src: &[u8], dst: CUdeviceptr) -> Result<()> {
        self.bind()?;
        if src.is_empty() {
            return Ok(());
        }
        let mut observation = self.prepare_observation(&[
            EventSpec::new(
                ObservedCategory::H2d,
                ObservedBoundary::RuntimeH2d,
                ObservedStatus::Submitted,
                src.len() as u64,
            ),
            EventSpec::new(
                ObservedCategory::H2d,
                ObservedBoundary::AsyncCompletionUnsupported,
                ObservedStatus::Unsupported,
                0,
            ),
        ])?;
        Self::reserve_raw_bytes(
            &self.raw_h2d_attempted,
            src.len() as u64,
            "async H2D attempted",
        )?;
        // SAFETY: bound context; `dst` covers `src.len()` bytes per the contract,
        // and the copy is ordered on the runtime-owned transfer stream.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(dst, src, self.copy_stream.cu_stream())
        }
        .map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err("cuMemcpyHtoDAsync", error),
            )
        })?;
        Self::mark_observation_submitted(&mut observation);
        Self::commit_observation(&mut observation)?;
        Ok(())
    }

    /// Measure one host-to-device copy with CUDA events on the transfer stream.
    ///
    /// The span is: record start event, enqueue `cuMemcpyHtoDAsync`, record end
    /// event, then synchronize the end event before calling `cuEventElapsedTime`.
    /// This **does block the host** until the DMA completes; use only on paths
    /// where honest attribution is more important than preserving overlap.
    ///
    /// Returns the elapsed milliseconds alongside a [`CopyCompleted`] witness.
    /// Because the end event is host-synchronized before this returns, the
    /// witness is proof — usable by a caller — that the copy's DMA read of `src`
    /// has *finished* (not merely been enqueued). The pinned-staging pool
    /// requires this witness before a source buffer may be reused, so a future
    /// switch to a non-blocking copy will fail to compile at the reuse site
    /// rather than silently corrupt weights.
    ///
    /// # Safety
    /// `dst` must cover at least `src.len()` bytes in this runtime's current
    /// CUDA context and must remain valid until this function returns.
    pub unsafe fn htod_async_elapsed_ms(
        &self,
        src: &[u8],
        dst: CUdeviceptr,
    ) -> std::result::Result<(f32, CopyCompleted), HtodAsyncElapsedError> {
        let bytes = src.len() as u64;
        self.bind().map_err(|error| HtodAsyncElapsedError {
            detail: error.to_string(),
            completion: FailedHtodCompletion::NotSubmitted,
        })?;
        if src.is_empty() {
            return Ok((0.0, CopyCompleted::new()));
        }
        let mut observation = self
            .prepare_observation(&[
                EventSpec::new(
                    ObservedCategory::H2d,
                    ObservedBoundary::RuntimeH2d,
                    ObservedStatus::Submitted,
                    bytes,
                ),
                EventSpec::new(
                    ObservedCategory::H2d,
                    ObservedBoundary::RuntimeH2d,
                    ObservedStatus::Completed,
                    bytes,
                ),
            ])
            .map_err(|error| HtodAsyncElapsedError {
                detail: error.to_string(),
                completion: FailedHtodCompletion::NotSubmitted,
            })?;
        Self::reserve_raw_bytes(&self.raw_h2d_attempted, bytes, "timed async H2D attempted")
            .map_err(|error| HtodAsyncElapsedError {
                detail: error.to_string(),
                completion: FailedHtodCompletion::NotSubmitted,
            })?;
        let fail_before_submit = |observation: &mut Option<PendingObservedBatch>,
                                  error: EpError| {
            let error = Self::fail_observation(observation, ObservedStatus::Failed, error);
            HtodAsyncElapsedError {
                detail: error.to_string(),
                completion: FailedHtodCompletion::NotSubmitted,
            }
        };
        let start = self
            .context
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| {
                fail_before_submit(&mut observation, driver_err("cuEventCreate(start)", error))
            })?;
        let end = self
            .context
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| {
                fail_before_submit(&mut observation, driver_err("cuEventCreate(end)", error))
            })?;
        start.record(&self.copy_stream).map_err(|error| {
            fail_before_submit(&mut observation, driver_err("cuEventRecord(start)", error))
        })?;
        // SAFETY: caller guarantees `dst` covers `src.len()` bytes. The source
        // must remain live until `end.elapsed_ms` returns, which this method
        // enforces by synchronizing the end event before returning.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(dst, src, self.copy_stream.cu_stream())
        }
        .map_err(|error| {
            fail_before_submit(&mut observation, driver_err("cuMemcpyHtoDAsync", error))
        })?;
        Self::mark_observation_submitted(&mut observation);
        if let Err(error) = end.record(&self.copy_stream) {
            let classified =
                self.settle_submitted_copy_failure(driver_err("cuEventRecord(end)", error));
            let status = if matches!(&classified.completion, FailedHtodCompletion::MayBeInFlight) {
                ObservedStatus::Quarantined
            } else {
                ObservedStatus::Failed
            };
            let operation = Self::fail_observation(
                &mut observation,
                status,
                EpError::KernelFailed(classified.detail.clone()),
            );
            return Err(HtodAsyncElapsedError {
                detail: operation.to_string(),
                completion: classified.completion,
            });
        }
        // `elapsed_ms` host-synchronizes the end event (cudarc `Event::elapsed_ms`
        // calls `end.synchronize()`), so on return the copy is complete on the
        // host timeline — which is exactly what `CopyCompleted` attests.
        let elapsed_ms = match start.elapsed_ms(&end) {
            Ok(elapsed_ms) => elapsed_ms,
            Err(error) => {
                let classified =
                    self.settle_submitted_copy_failure(driver_err("cuEventElapsedTime", error));
                let status =
                    if matches!(&classified.completion, FailedHtodCompletion::MayBeInFlight) {
                        ObservedStatus::Quarantined
                    } else {
                        ObservedStatus::Failed
                    };
                let operation = Self::fail_observation(
                    &mut observation,
                    status,
                    EpError::KernelFailed(classified.detail.clone()),
                );
                return Err(HtodAsyncElapsedError {
                    detail: operation.to_string(),
                    completion: classified.completion,
                });
            }
        };
        self.async_host_to_device_copies
            .fetch_add(1, Ordering::Relaxed);
        self.raw_h2d_completed.fetch_add(bytes, Ordering::Relaxed);
        Self::commit_observation(&mut observation).map_err(|error| HtodAsyncElapsedError {
            detail: format!(
                "H2D DMA completed, but its reserved observed-byte receipt could not commit: \
                 {error}"
            ),
            completion: FailedHtodCompletion::Completed(CopyCompleted::new()),
        })?;
        Ok((elapsed_ms, CopyCompleted::new()))
    }

    /// Set `bytes` device bytes to zero and wait for completion.
    ///
    /// This is the production zero-fill boundary used by synthetic governed
    /// loader controls: it performs no source read or host materialization, and
    /// records useful bytes only after the stream has completed the memset.
    ///
    /// # Safety
    /// `dst` must name at least `bytes` writable bytes in this runtime.
    pub unsafe fn memset_zero(&self, dst: CUdeviceptr, bytes: usize) -> Result<()> {
        self.bind()?;
        if bytes == 0 {
            return self.observe_bytes(EventSpec::new(
                ObservedCategory::CudaMemset,
                ObservedBoundary::RuntimeCudaMemset,
                ObservedStatus::Elided,
                0,
            ));
        }
        let mut observation = self.prepare_observation(&[
            EventSpec::new(
                ObservedCategory::CudaMemset,
                ObservedBoundary::RuntimeCudaMemset,
                ObservedStatus::Submitted,
                bytes as u64,
            ),
            EventSpec::new(
                ObservedCategory::CudaMemset,
                ObservedBoundary::RuntimeCudaMemset,
                ObservedStatus::Completed,
                bytes as u64,
            ),
        ])?;
        Self::reserve_raw_bytes(&self.raw_memset_attempted, bytes as u64, "memset attempted")?;
        unsafe { cudarc::driver::result::memset_d8_async(dst, 0, bytes, self.stream.cu_stream()) }
            .map_err(|error| {
                Self::fail_observation(
                    &mut observation,
                    ObservedStatus::Failed,
                    driver_err("cuMemsetD8Async", error),
                )
            })?;
        Self::mark_observation_submitted(&mut observation);
        self.force_synchronize().map_err(|error| {
            Self::fail_observation(&mut observation, ObservedStatus::Failed, error)
        })?;
        self.raw_memset_completed
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Self::commit_observation(&mut observation)?;
        Ok(())
    }

    fn settle_submitted_copy_failure(&self, error: EpError) -> HtodAsyncElapsedError {
        if let Some(_section) = capture_gate::synchronizing_section() {
            classify_submitted_htod_failure(
                error.to_string(),
                self.copy_stream
                    .synchronize()
                    .map_err(|sync_error| sync_error.to_string()),
            )
        } else {
            classify_submitted_htod_failure(
                error.to_string(),
                Err(
                    "the copy failed on the active capture thread; synchronization was deferred \
                     to preserve capture"
                        .into(),
                ),
            )
        }
    }

    /// Enqueue an asynchronous device → device copy on the transfer stream, so
    /// it overlaps compute the same way [`CudaRuntime::htod_async`] does. Used by
    /// [`copy_async`](onnx_runtime_ep_api::ExecutionProvider::copy_async) when the
    /// source already resides on-device.
    ///
    /// # Safety
    /// Both pointers are live allocations of at least `bytes` bytes and remain
    /// live until the transfer stream completes the copy (order with a fence).
    pub unsafe fn dtod_async_on_copy_stream(
        &self,
        src: CUdeviceptr,
        dst: CUdeviceptr,
        bytes: usize,
    ) -> Result<()> {
        self.bind()?;
        if bytes == 0 {
            return Ok(());
        }
        let mut observation = self.prepare_observation(&[
            EventSpec::new(
                ObservedCategory::D2d,
                ObservedBoundary::RuntimeD2d,
                ObservedStatus::Submitted,
                bytes as u64,
            ),
            EventSpec::new(
                ObservedCategory::D2d,
                ObservedBoundary::AsyncCompletionUnsupported,
                ObservedStatus::Unsupported,
                0,
            ),
        ])?;
        Self::reserve_raw_bytes(
            &self.raw_d2d_attempted,
            bytes as u64,
            "copy-stream D2D attempted",
        )?;

        // SAFETY: bound context; both endpoints cover `bytes` and the copy is
        // ordered on the runtime-owned transfer stream.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, self.copy_stream.cu_stream())
        }
        .map_err(|error| {
            Self::fail_observation(
                &mut observation,
                ObservedStatus::Failed,
                driver_err("cuMemcpyDtoDAsync", error),
            )
        })?;
        Self::mark_observation_submitted(&mut observation);
        Self::commit_observation(&mut observation)?;
        Ok(())
    }

    /// Record a completion event for all work so far enqueued on the transfer
    /// stream and register it under a fresh, opaque fence id (always non-zero).
    /// Await it later on the compute stream with [`CudaRuntime::compute_wait_fence`].
    pub fn record_copy_fence(&self) -> Result<u64> {
        self.record_fence_on(&self.copy_stream)
    }

    /// Record a completion event for all work so far enqueued on the compute
    /// stream and register it under a fresh fence id. Used for double-buffer
    /// reuse: the transfer stream must wait on the previous consumer (via
    /// [`CudaRuntime::copy_wait_fence`]) before overwriting a staging buffer.
    pub fn record_compute_fence(&self) -> Result<u64> {
        self.record_fence_on(&self.stream)
    }

    fn record_fence_on(&self, stream: &CudaStream) -> Result<u64> {
        self.bind()?;
        let event = self
            .context
            .new_event(None)
            .map_err(|e| driver_err("cuEventCreate", e))?;
        event
            .record(stream)
            .map_err(|e| driver_err("cuEventRecord", e))?;
        let id = self.next_fence_id.fetch_add(1, Ordering::Relaxed);
        self.fences
            .lock()
            .expect("cuda fence registry poisoned")
            .insert(id, event);
        Ok(id)
    }

    /// Make the compute stream wait on the transfer completion event named by
    /// `fence_id` (a stream-ordered, non host-blocking cross-stream wait), then
    /// release the event. A later kernel launched on the compute stream is
    /// therefore ordered after the prefetch and observes the full transfer. Id
    /// `0` (an already-signalled fence) and an unknown id are no-ops.
    pub fn compute_wait_fence(&self, fence_id: u64) -> Result<()> {
        self.wait_fence_on(&self.stream, fence_id).map(|_| ())
    }

    /// Make the transfer stream wait on the compute completion event named by
    /// `fence_id`, then release the event. Used before reusing a double-buffer
    /// staging region so the incoming prefetch never overwrites bytes a prior
    /// wave's kernel is still reading (write-after-read hazard).
    pub fn copy_wait_fence(&self, fence_id: u64) -> Result<()> {
        self.wait_fence_on(&self.copy_stream, fence_id).map(|_| ())
    }

    /// Resolve a transfer-stream fence for a genuinely ahead-of-need prefetch
    /// (issue #82 BlockQuantizedMoE prefill prefetch): order the compute stream
    /// after it exactly like [`CudaRuntime::compute_wait_fence`] (cheap,
    /// non-host-blocking), and additionally host-synchronize the same event
    /// before returning.
    ///
    /// The host wait is what proves — to the caller and to the type system —
    /// that the transfer stream's DMA read of the prefetch's pinned source
    /// buffer has *completed*, not merely been enqueued: the returned
    /// [`CopyCompleted`] witness is exactly the token
    /// [`crate::pinned_pool::PinnedStagingPool::release`] /
    /// [`crate::pinned_pool::PooledStaging::retire`] require before a pooled
    /// staging buffer may be reused, so a prefetch's staging buffer returns to
    /// the same shared pool an on-demand page-in uses instead of paying a
    /// fresh `cuMemHostAlloc`/`cuMemFreeHost` pair every time (the exact
    /// steady-state cost issue #837 already fixed for the page-in path).
    /// Because the prefetch was issued strictly before the consumer that
    /// resolves it — while unrelated compute was free to run in between —
    /// this wait blocks only for whatever transfer time genuinely remains,
    /// zero once that intervening compute has already hidden it. Fence id `0`
    /// (already-signalled) and an unknown id (already resolved by an earlier
    /// call) are no-ops that still mint a witness: by definition nothing is
    /// left in flight in either case, so the completion guarantee already
    /// holds.
    ///
    /// A DMA was already submitted by the caller before this fence id was
    /// handed out (it is the whole reason a fence exists to resolve), so
    /// every failure path below reuses [`CudaRuntime::settle_submitted_copy_failure`]
    /// — the same fallback-synchronize-then-classify machinery
    /// [`CudaRuntime::htod_async_elapsed_ms`] uses — instead of returning a
    /// bare driver error: it either proves completion through a coarser
    /// `copy_stream` sync ([`FailedHtodCompletion::Completed`]) or reports
    /// that the source/destination may still be touched by an in-flight read
    /// ([`FailedHtodCompletion::MayBeInFlight`]), so a caller can quarantine
    /// the staging buffer and destination page instead of freeing or reusing
    /// either while a DMA might still be reading/writing them.
    pub fn resolve_prefetch_fence(
        &self,
        fence_id: u64,
    ) -> std::result::Result<CopyCompleted, HtodAsyncElapsedError> {
        if fence_id == 0 {
            return Ok(CopyCompleted::new());
        }
        let event = self
            .fences
            .lock()
            .expect("cuda fence registry poisoned")
            .remove(&fence_id);
        let Some(event) = event else {
            return Ok(CopyCompleted::new());
        };
        if let Err(error) = self.bind() {
            return Err(self.settle_submitted_copy_failure(error));
        }
        // Cross-stream order first (cheap, enqueue-only): any kernel the
        // compute stream launches after this point observes the transfer, even
        // before the host wait below completes.
        if let Err(error) = self.stream.wait(&event) {
            return Err(self.settle_submitted_copy_failure(driver_err("cuStreamWaitEvent", error)));
        }
        // Host-blocking: proves the transfer stream's read of the prefetch's
        // source buffer has finished, so it is safe to free or, preferably,
        // return to the pinned staging pool for reuse.
        if let Err(error) = event.synchronize() {
            return Err(self.settle_submitted_copy_failure(driver_err("cuEventSynchronize", error)));
        }
        Ok(CopyCompleted::new())
    }

    fn wait_fence_on(&self, waiter: &CudaStream, fence_id: u64) -> Result<bool> {
        dispatch_registered_fence_wait(&self.fences, fence_id, |event| {
            self.bind()?;
            waiter
                .wait(event)
                .map_err(|error| driver_err("cuStreamWaitEvent", error))
        })
    }

    #[cfg(test)]
    pub(crate) fn fence_is_registered(&self, fence_id: u64) -> bool {
        self.fences
            .lock()
            .expect("cuda fence registry poisoned")
            .contains_key(&fence_id)
    }

    /// Block the host until every transfer queued on the copy stream completes.
    /// Used on teardown / test paths that read a prefetched buffer without an
    /// intervening event wait.
    pub fn sync_copy_stream(&self) -> Result<()> {
        self.copy_stream
            .synchronize()
            .map_err(|e| driver_err("transfer stream synchronize", e))
    }

    /// Allocate `bytes` of page-locked (pinned) host staging memory suitable as
    /// the source of [`CudaRuntime::htod_async`]. Pinned memory lets the driver
    /// DMA host→device without an internal pageable-staging copy, which is what
    /// makes the transfer genuinely asynchronous and overlappable.
    pub fn alloc_pinned(&self, bytes: usize) -> Result<PinnedStaging> {
        self.bind()?;
        let allocated_bytes = bytes.max(1) as u64;
        let mut observation = self.prepare_observation(&[EventSpec::new(
            ObservedCategory::HostAllocation,
            ObservedBoundary::PinnedHostAllocate,
            ObservedStatus::Committed,
            allocated_bytes,
        )])?;
        // Page-locking host memory synchronizes the device; see `alloc_raw`.
        let _section = capture_gate::synchronizing_section();
        // SAFETY: `malloc_host` returns a fresh page-locked host allocation on
        // the bound context; `PinnedStaging` owns it and frees it once on drop.
        let ptr =
            unsafe { cudarc::driver::result::malloc_host(bytes.max(1), 0) }.map_err(|error| {
                Self::fail_observation(
                    &mut observation,
                    ObservedStatus::Failed,
                    driver_err("cuMemHostAlloc", error),
                )
            })?;
        Self::commit_observation(&mut observation)?;
        Ok(PinnedStaging {
            ptr: ptr.cast::<u8>(),
            len: bytes,
            context: self.context.clone(),
        })
    }
}

/// Witness that a host→device copy issued on the transfer stream has **completed
/// on the host timeline** — not merely been enqueued.
///
/// It is a zero-sized token whose sole field is private to this module, so it
/// can only be minted by a copy primitive here that host-synchronizes the copy
/// before returning (today, [`CudaRuntime::htod_async_elapsed_ms`]). Nothing
/// outside `runtime` can fabricate one.
///
/// Its purpose is a compile-time proof obligation: reusing (or freeing) the
/// pinned buffer that was the *source* of a copy is only sound once that copy's
/// DMA read has finished. The pinned-staging pool's reuse path
/// (`PinnedStagingPool::release` / `PooledStaging::retire`) consumes a
/// `CopyCompleted`. If the page-in path is ever switched to a non-blocking
/// `htod_async` + deferred fence, no `CopyCompleted` is available at the reuse
/// site, so the code **fails to compile** until the author threads the witness
/// through after awaiting the fence — the hazard cannot be reached by accident.
#[derive(Debug)]
#[must_use = "a CopyCompleted witness exists to gate pinned-buffer reuse; dropping it is pointless"]
pub struct CopyCompleted(());

impl CopyCompleted {
    /// Mint a witness. Private to `runtime` so only a host-synchronizing copy
    /// primitive in this module can produce one.
    fn new() -> Self {
        CopyCompleted(())
    }

    /// Test-only constructor. Unit tests in this crate that exercise the pool
    /// without issuing a real copy need a witness; this keeps them honest about
    /// requiring one without granting non-test code the ability to forge it.
    #[cfg(test)]
    pub(crate) fn new_for_test() -> Self {
        CopyCompleted(())
    }
}

/// Owned page-locked (pinned) host staging buffer used as the source of an
/// asynchronous host→device weight prefetch. Freed exactly once on drop through
/// the owning CUDA context.
pub struct PinnedStaging {
    ptr: *mut u8,
    len: usize,
    context: Arc<CudaContext>,
}

// SAFETY: `PinnedStaging` owns a single page-locked host allocation; the raw
// pointer is a plain address that is safe to move between threads. Concurrent
// access to the *contents* is governed by `&`/`&mut self` like any `Vec<u8>`.
unsafe impl Send for PinnedStaging {}
unsafe impl Sync for PinnedStaging {}

impl PinnedStaging {
    /// Number of bytes in the staging buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the staging buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Host-writable view of the staging bytes (fill this before prefetching).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` names a live `len`-byte host allocation uniquely borrowed
        // through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Host-readable view of the staging bytes.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` names a live `len`-byte host allocation shared through
        // `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::fmt::Debug for PinnedStaging {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedStaging")
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for PinnedStaging {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        // Unpinning synchronizes the device; see `CudaRuntime::alloc_raw`.
        let _section = capture_gate::synchronizing_section();
        // SAFETY: `ptr` came from `malloc_host` on this context and is freed
        // exactly once here.
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast::<c_void>()) };
    }
}

impl Drop for CudaRuntime {
    fn drop(&mut self) {
        // Tearing a runtime down unloads its modules and destroys its streams.
        // `cuModuleUnload` and `cuStreamDestroy` synchronize the device, so a
        // runtime going out of scope on one thread can invalidate a CUDA graph
        // capture on another. Those calls are made by `cudarc` as the fields
        // drop, not by this crate, so the section has to cover the whole
        // teardown rather than any individual call.
        //
        // Stored in the last-declared field instead of a local: locals are
        // released when this body returns, which is *before* the fields drop.
        // See `teardown_section`.
        self.teardown_section = capture_gate::synchronizing_section();
        // Before the context goes: an interleaved weight copy is keyed by the
        // source weight's device address, and that address stops meaning
        // anything once this runtime's allocator is gone. Freeing here is what
        // bounds an entry's life by the life of the address that names it.
        self.interleave.release_all(&*self);
        if self.capture_error != 0 {
            // SAFETY: `capture_error` was allocated by this runtime's `alloc_raw`
            // in `with_capture_error_word` and is freed exactly once here.
            let _ = unsafe { self.free_raw(self.capture_error) };
            self.capture_error = 0;
        }
    }
}

/// Reinterpret an EP [`onnx_runtime_ep_api::DeviceBuffer`] raw pointer (or a
/// [`onnx_runtime_ep_api::TensorView`] data pointer) as a CUDA device pointer.
/// CUDA device pointers are integer addresses; the EP stores them in the opaque
/// pointer slot, so this is a value reinterpretation, never a host deref.
#[inline]
pub fn cuptr(raw: *const c_void) -> CUdeviceptr {
    raw as usize as CUdeviceptr
}

/// Inverse of [`cuptr`]: pack a CUDA device pointer into the opaque pointer slot
/// used by [`onnx_runtime_ep_api::DeviceBuffer`].
#[inline]
pub fn raw_ptr(dptr: CUdeviceptr) -> *mut c_void {
    dptr as usize as *mut c_void
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::PushKernelArg;

    #[test]
    fn validation_generation_exhaustion_fails_without_wrap_or_aba() {
        let next = AtomicU64::new(VALIDATION_MAX_GENERATION - 1);
        assert_eq!(
            take_validation_generation(&next).unwrap(),
            VALIDATION_MAX_GENERATION - 1
        );
        let error = take_validation_generation(&next)
            .expect_err("generation allocation must fail instead of wrapping");
        assert!(error.to_string().contains("generation space exhausted"));
        assert_eq!(
            next.load(Ordering::Relaxed),
            VALIDATION_MAX_GENERATION,
            "failed allocation must not wrap or reuse an earlier generation"
        );
    }

    #[test]
    fn fence_dispatch_consumes_fresh_event_once_and_propagates_wait_errors() {
        let registry = Mutex::new(HashMap::from([(7, "fresh event")]));
        let mut backend_waits = 0;
        assert!(
            dispatch_registered_fence_wait(&registry, 7, |event| {
                backend_waits += 1;
                assert_eq!(*event, "fresh event");
                Ok(())
            })
            .expect("fresh fence dispatch")
        );
        assert_eq!(backend_waits, 1, "fresh fence must invoke one backend wait");
        assert!(
            registry.lock().unwrap().is_empty(),
            "fresh event ownership must leave the registry before dispatch"
        );

        for fence_id in [7, 99, 0] {
            assert!(
                !dispatch_registered_fence_wait(&registry, fence_id, |_| {
                    backend_waits += 1;
                    Ok(())
                })
                .expect("resolved, unknown, and zero fences are no-ops")
            );
        }
        assert_eq!(
            backend_waits, 1,
            "resolved, unknown, and zero fences must not redispatch"
        );

        registry.lock().unwrap().insert(11, "failing event");
        let error = dispatch_registered_fence_wait(&registry, 11, |_| {
            backend_waits += 1;
            Err(EpError::KernelFailed(
                "synthetic cuStreamWaitEvent failure".into(),
            ))
        })
        .expect_err("backend wait failure must propagate");
        let detail = error.to_string();
        assert!(
            detail.contains("fence 11")
                && detail.contains("consumed")
                && detail.contains("synthetic cuStreamWaitEvent failure"),
            "wait failure must retain fence ownership and backend context: {detail}"
        );
        assert_eq!(backend_waits, 2);
        assert!(
            registry.lock().unwrap().is_empty(),
            "a failed wait still consumes the uniquely owned event exactly once"
        );
    }

    #[test]
    fn submitted_htod_failure_reports_completion_after_fallback_sync() {
        let error =
            classify_submitted_htod_failure("injected end-event record failure".into(), Ok(()));
        let (detail, completion) = error.into_parts();
        assert!(detail.contains("completion was established"));
        assert!(matches!(completion, FailedHtodCompletion::Completed(_)));
    }

    #[test]
    fn submitted_htod_failure_reports_possible_in_flight_copy_when_sync_fails() {
        let error = classify_submitted_htod_failure(
            "injected end-event record failure".into(),
            Err("injected copy-stream synchronization failure".into()),
        );
        let (detail, completion) = error.into_parts();
        assert!(detail.contains("could not establish completion"));
        assert!(matches!(completion, FailedHtodCompletion::MayBeInFlight));
    }

    #[test]
    fn raw_allocation_profile_attributes_driver_and_pool_paths() {
        let profile = RawAllocationProfile::new(true);
        let location = std::panic::Location::caller();
        profile.record(location, 513, 1024, false);
        profile.record(location, 700, 1024, true);

        let sites = profile.snapshot();
        assert_eq!(sites.len(), 1);
        assert_eq!(
            sites[0],
            RawDeviceAllocationSiteStats {
                file: location.file(),
                line: location.line(),
                requests: 2,
                requested_bytes: 1213,
                driver_allocations: 1,
                driver_bytes: 1024,
                pool_hits: 1,
                pool_hit_bytes: 1024,
            }
        );
    }

    #[test]
    fn disabled_raw_allocation_profile_has_no_hot_path_state() {
        let profile = RawAllocationProfile::new(false);
        profile.record(std::panic::Location::caller(), 4096, 4096, false);
        assert!(profile.snapshot().is_empty());
    }

    /// The pool hands a recycled block to a *different* request than the one it
    /// was carved for, so the only thing standing between reuse and a buffer
    /// overrun is that the class is never smaller than the request. Check that
    /// directly, including the boundary where the rounding rule changes.
    #[test]
    fn raw_pool_size_class_never_undersizes_a_request() {
        let sizes = [
            1usize,
            2,
            511,
            512,
            513,
            4096,
            (1 << 20) - 1,
            1 << 20,
            (1 << 20) + 1,
            3_000_000,
            1 << 24,
        ];
        for bytes in sizes {
            let class = raw_pool_size_class(bytes);
            assert!(
                class >= bytes,
                "class {class} is smaller than the {bytes}-byte request it must satisfy"
            );
            assert!(
                class >= 512,
                "class {class} is below the minimum block size"
            );
        }
        // Monotonic, so a larger request can never land in a smaller class and
        // pick up a block that does not fit it.
        let mut previous = 0;
        for bytes in sizes {
            let class = raw_pool_size_class(bytes);
            assert!(class >= previous, "class fell from {previous} to {class}");
            previous = class;
        }
    }

    /// Pooling only pays if the second request for a shape skips the driver
    /// entirely, and it is only safe if the pointer it hands back is the one it
    /// retained. Both are asserted here rather than inferred from a wall-clock
    /// improvement.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn raw_pool_recycles_freed_blocks_without_reentering_the_driver() {
        let Ok(runtime) = CudaRuntime::new(0) else {
            eprintln!("skipping raw pool reuse: CUDA runtime unavailable");
            return;
        };
        let bytes = 4 << 20;
        let first = runtime.alloc_raw(bytes).unwrap();
        let allocations_after_first = runtime.allocation_counts().allocations;
        // SAFETY: `first` came from this runtime's `alloc_raw` and is freed once.
        unsafe { runtime.free_raw(first).unwrap() };

        let second = runtime.alloc_raw(bytes).unwrap();
        assert_eq!(
            second, first,
            "the pool must hand back the block it just retained"
        );
        assert_eq!(
            runtime.raw_pool_hits(),
            1,
            "the second request must be served from the pool"
        );
        assert_eq!(
            runtime.allocation_counts().allocations,
            allocations_after_first,
            "a pooled request must not reach cuMemAlloc"
        );
        // SAFETY: `second` is the same live block, freed once.
        unsafe { runtime.free_raw(second).unwrap() };
    }

    /// With pooling disabled the runtime must behave exactly as it did before:
    /// every request reaches the driver. This is the bisect switch, so it has
    /// to actually switch something.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn raw_pool_disabled_returns_every_block_to_the_driver() {
        let Ok(runtime) = CudaRuntime::new(0) else {
            eprintln!("skipping raw pool disable: CUDA runtime unavailable");
            return;
        };
        // SAFETY: this ignored GPU test runs serially; no other thread reads the
        // variable concurrently.
        unsafe { std::env::set_var(CUDA_RAW_POOL_BYTES_ENV, "0") };
        let bytes = 4 << 20;
        let first = runtime.alloc_raw(bytes).unwrap();
        // SAFETY: `first` came from `alloc_raw` and is freed once.
        unsafe { runtime.free_raw(first).unwrap() };
        let before = runtime.allocation_counts().allocations;
        let second = runtime.alloc_raw(bytes).unwrap();
        assert_eq!(
            runtime.raw_pool_hits(),
            0,
            "pooling is disabled, so nothing may be served from the pool"
        );
        assert_eq!(
            runtime.allocation_counts().allocations,
            before + 1,
            "with pooling disabled every request must reach cuMemAlloc"
        );
        assert_eq!(runtime.raw_pool_retained_bytes(), 0);
        // SAFETY: `second` is live and freed once.
        unsafe { runtime.free_raw(second).unwrap() };
        // SAFETY: restore the process-global default for other tests.
        unsafe { std::env::remove_var(CUDA_RAW_POOL_BYTES_ENV) };
    }

    #[test]
    fn derives_ptx_arch_from_compute_capability() {
        for (major, minor, expected) in [
            (6, 0, "compute_60"),
            (7, 5, "compute_75"),
            (8, 0, "compute_80"),
            (8, 6, "compute_86"),
            (8, 9, "compute_89"),
            (9, 0, "compute_90"),
            (10, 0, "compute_100"),
            (12, 0, "compute_120"),
        ] {
            assert_eq!(ptx_arch_for(major, minor), expected);
        }
    }

    #[test]
    fn derives_cubin_arch_from_compute_capability() {
        for (major, minor, expected) in [
            (6, 0, "sm_60"),
            (7, 5, "sm_75"),
            (8, 0, "sm_80"),
            (8, 6, "sm_86"),
            (8, 9, "sm_89"),
            (9, 0, "sm_90"),
            (10, 0, "sm_100"),
            (12, 0, "sm_120"),
        ] {
            assert_eq!(cubin_arch_for(major, minor), expected);
        }
    }

    #[test]
    fn capability_limits_use_conservative_fallbacks() {
        let capabilities = CudaDeviceCapabilities::from_reported_limits(
            (7, 0),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(capabilities.compute_capability(), (7, 0));
        assert_eq!(capabilities.max_threads_per_block, 256);
        assert_eq!(capabilities.max_grid_dim_x(), SAFE_MAX_GRID_DIM_X_FALLBACK);
        assert_eq!(
            capabilities.max_shared_memory_per_block,
            SAFE_SHARED_MEMORY_PER_BLOCK_FALLBACK
        );
        assert_eq!(
            capabilities.max_shared_memory_per_block_optin(),
            SAFE_SHARED_MEMORY_PER_BLOCK_FALLBACK
        );
        assert_eq!(capabilities.multiprocessor_count(), 1);
        assert_eq!(capabilities.l2_cache_size(), 0);
    }

    #[test]
    fn capability_limits_never_reduce_optin_below_default() {
        let capabilities = CudaDeviceCapabilities::from_reported_limits(
            (12, 0),
            Some(1024),
            Some(2_147_483_647),
            Some(64 * 1024),
            Some(48 * 1024),
            Some(200),
            Some(96 * 1024 * 1024),
        );
        assert_eq!(capabilities.max_grid_dim_x(), 2_147_483_647);
        assert_eq!(capabilities.max_shared_memory_per_block_optin(), 64 * 1024);
        assert_eq!(capabilities.multiprocessor_count(), 200);
        assert_eq!(capabilities.l2_cache_size(), 96 * 1024 * 1024);
    }

    #[test]
    fn reduction_launch_is_clamped_to_device_limits() {
        assert_eq!(
            reduction_launch_params(256, 1024, 4, 227 * 1024),
            Some((256, 1024))
        );
        assert_eq!(
            reduction_launch_params(256, 128, 4, 227 * 1024),
            Some((128, 512))
        );
        assert_eq!(reduction_launch_params(256, 1024, 4, 768), Some((128, 512)));
        assert_eq!(reduction_launch_params(256, 1024, 8, 0), None);
    }

    #[test]
    fn dynamic_shared_memory_optin_respects_device_budgets() {
        let default_budget = 48 * 1024;
        // Fits the 48 KB non-opt-in budget: launch as-is, no attribute change.
        assert_eq!(
            dynamic_shared_memory_optin(32 * 1024, default_budget, 227 * 1024),
            Ok(None)
        );
        // Boundary: exactly the default budget still needs no opt-in.
        assert_eq!(
            dynamic_shared_memory_optin(default_budget, default_budget, 100 * 1024),
            Ok(None)
        );
        // Over 48 KB but within a consumer (sm_86/sm_89) ~100 KB opt-in ceiling:
        // must opt the function into the exact request.
        assert_eq!(
            dynamic_shared_memory_optin(64 * 1024, default_budget, 100 * 1024),
            Ok(Some(64 * 1024))
        );
        // Sized for an H200 (227 KB) but launched on a 100 KB consumer card:
        // reject loudly rather than crash at launch.
        assert_eq!(
            dynamic_shared_memory_optin(160 * 1024, default_budget, 100 * 1024),
            Err(())
        );
    }

    /// Nothing is offered to NVRTC that is not a CUDA header directory.
    ///
    /// Two headers qualify a directory, not one. A toolkit install keeps them
    /// together, but the pip wheels split them: `cuda_fp16.h` ships in
    /// `nvidia-cuda-runtime` and the `crt/` tree that `mma.h` includes ships in
    /// `nvidia-cuda-nvcc`. Requiring `cuda_fp16.h` of every directory dropped
    /// the second, and the tensor-core kernels then failed inside NVRTC with
    /// `cannot open source file "crt/mma.h"`.
    #[test]
    fn nvrtc_include_paths_only_returns_cuda_header_dirs() {
        for path in nvrtc_include_paths() {
            let path = Path::new(&path);
            assert!(
                path.join("cuda_fp16.h").is_file() || path.join("crt/mma.h").is_file(),
                "{path:?} carries neither cuda_fp16.h nor crt/mma.h"
            );
        }
    }

    fn maybe_runtime() -> Option<Arc<CudaRuntime>> {
        std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten()
    }

    /// The on-disk kernel cache stores PTX *text* and restores it through a
    /// different driver entry point than a freshly compiled image uses.
    ///
    /// This is the property the whole cache rests on: if a restored module ever
    /// computed something different from a compiled one, every second run of
    /// every model would be silently wrong. A pure round-trip test of the bytes
    /// would not catch that — the module has to actually run.
    #[test]
    fn a_module_restored_from_cached_ptx_computes_what_a_compiled_one_does() {
        const MODULE: &str = "kernel_cache_roundtrip_v1";
        const SOURCE: &str = r#"
extern "C" __global__ void add_seven(float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i + 7.0f;
}
"#;
        let Some(runtime) = maybe_runtime() else {
            eprintln!("skipping cached-PTX equivalence: CUDA runtime unavailable");
            return;
        };
        runtime.bind().unwrap();
        let include_paths = nvrtc_include_paths();
        let before = crate::kernel_cache::kernel_compile_stats();
        let ptx = runtime.nvrtc_ptx(MODULE, SOURCE, &include_paths).unwrap();
        let after = crate::kernel_cache::kernel_compile_stats();

        // The accounting must be wired to the real path. A counter that never
        // moves is worse than no counter: it reports "nothing was compiled",
        // which is exactly the answer a warm cache is supposed to give.
        //
        // The bound is `>= 1`, not `== 1`. These counters are process-global
        // and the harness runs several hundred tests as parallel threads, so
        // other threads resolve their own modules between the two reads. An
        // exact count would be asserting that no other test was running, which
        // is not a property of the code under test. What still holds -- and is
        // what this guards -- is that a counter moved at all.
        assert!(
            (after.compiled - before.compiled) + (after.cache_hits - before.cache_hits) >= 1,
            "resolving a module must advance the compiled or cache-hit counter: \
             {before:?} -> {after:?}"
        );
        if after.compiled > before.compiled {
            assert!(
                after.compile_time > before.compile_time,
                "a recorded compile must carry time: {before:?} -> {after:?}"
            );
        }

        // Whatever the first resolution did, a second one for the same key is a
        // hit: the negative control that separates "the cache is wired up" from
        // "some counter happens to be moving".
        let warm_before = crate::kernel_cache::kernel_compile_stats();
        runtime.nvrtc_ptx(MODULE, SOURCE, &include_paths).unwrap();
        let warm_after = crate::kernel_cache::kernel_compile_stats();
        assert!(
            warm_after.cache_hits > warm_before.cache_hits,
            "re-resolving an already cached module must count a cache hit: \
             {warm_before:?} -> {warm_after:?}"
        );

        // Exactly the bytes `kernel_cache::store` writes, read back exactly the
        // way `nvrtc_ptx` reads them on a hit.
        let stored = ptx.to_src().into_bytes();
        let restored = cudarc::nvrtc::Ptx::from_src(String::from_utf8(stored).unwrap());

        let n = 1024usize;
        let bytes = n * std::mem::size_of::<f32>();
        let run = |ptx: cudarc::nvrtc::Ptx| -> Vec<f32> {
            let module = runtime.context.load_module(ptx).unwrap();
            let function = module.load_function("add_seven").unwrap();
            let out = runtime.alloc_raw(bytes).unwrap();
            let mut builder = runtime.stream().launch_builder(&function);
            let n_u64 = n as u64;
            builder.arg(&out).arg(&n_u64);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap();
            }
            runtime.synchronize().unwrap();
            let mut host = vec![0.0f32; n];
            let host_bytes =
                unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(host_bytes, out) }.unwrap();
            let _ = unsafe { runtime.free_raw(out) };
            host
        };

        let fresh = run(ptx);
        let cached = run(restored);

        assert_eq!(fresh[0], 7.0, "the kernel must have actually run");
        assert_eq!(fresh, cached, "cached PTX must not change the result");
    }

    // Regression guard for the DeepSeek-V2-Lite garbage-decode race: kernels run
    // on the EP's non-default stream, but `cuMemcpyDtoD` issues on the legacy
    // default stream, which is NOT implicitly ordered against a non-blocking
    // compute stream. `dtod` must therefore drain the EP stream before copying,
    // so it never reads bytes a producer kernel is still writing. Without the
    // synchronize this test observes the pre-launch poison values; with it the
    // copy always sees the fully-produced sentinel.
    #[test]
    fn dtod_waits_for_pending_stream_writes() {
        const MODULE: &str = "runtime_dtod_race_test";
        // Each thread spins on the GPU clock (~a few ms) before storing its
        // sentinel, guaranteeing the store is still in flight when the racing
        // default-stream copy would otherwise run.
        const SOURCE: &str = r#"
extern "C" __global__ void slow_fill(float* out, unsigned long long n, long long spin) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long start = clock64();
    while (clock64() - start < spin) { }
    out[i] = 1.0f + (float)(i % 7);
}
"#;
        let Some(runtime) = maybe_runtime() else {
            eprintln!("skipping dtod race regression: CUDA runtime unavailable");
            return;
        };
        let function = runtime.nvrtc_function(MODULE, SOURCE, "slow_fill").unwrap();
        let n = 4096usize;
        let bytes = n * std::mem::size_of::<f32>();
        let src = runtime.alloc_raw(bytes).unwrap();
        let dst = runtime.alloc_raw(bytes).unwrap();

        // Poison the source so a premature (racing) copy is detectable.
        let poison = vec![-999.0f32; n];
        let poison_bytes =
            unsafe { std::slice::from_raw_parts(poison.as_ptr().cast::<u8>(), bytes) };
        // Run several iterations; a race is probabilistic per launch, but the
        // fix must make every iteration correct.
        for _ in 0..8 {
            unsafe { runtime.htod(poison_bytes, src) }.unwrap();
            runtime.synchronize().unwrap();

            // Enqueue the slow producer on the EP stream, then immediately copy
            // WITHOUT an explicit synchronize — `dtod` must order this itself.
            let spin: i64 = 8_000_000;
            let mut builder = runtime.stream().launch_builder(&function);
            let n_u64 = n as u64;
            builder.arg(&src).arg(&n_u64).arg(&spin);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap();
            }
            unsafe { runtime.dtod(src, dst, bytes) }.unwrap();

            let mut out = vec![0.0f32; n];
            let out_bytes =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(out_bytes, dst) }.unwrap();

            for (i, value) in out.iter().enumerate() {
                let expected = 1.0f32 + (i % 7) as f32;
                assert_eq!(
                    *value, expected,
                    "dtod observed unsynchronized/poison data at index {i}: \
                     got {value}, expected {expected} (stream-ordering race)"
                );
            }
        }

        unsafe { runtime.free_raw(src) }.unwrap();
        unsafe { runtime.free_raw(dst) }.unwrap();
    }

    // Companion to the sync-`dtod` guard: a stream-ordered `dtod_async` issued on
    // the EP compute stream (as `copy_reshape` uses for Reshape/Squeeze) must be
    // implicitly ordered after a producer kernel on the same stream, WITHOUT any
    // host synchronize. A later `dtoh` (which drains the stream) must then read
    // the fully-produced sentinel, never the pre-launch poison.
    #[test]
    fn dtod_async_is_ordered_after_same_stream_producer() {
        const MODULE: &str = "runtime_dtod_async_order_test";
        const SOURCE: &str = r#"
extern "C" __global__ void slow_fill(float* out, unsigned long long n, long long spin) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long start = clock64();
    while (clock64() - start < spin) { }
    out[i] = 1.0f + (float)(i % 7);
}
"#;
        let Some(runtime) = maybe_runtime() else {
            eprintln!("skipping dtod_async ordering test: CUDA runtime unavailable");
            return;
        };
        let function = runtime.nvrtc_function(MODULE, SOURCE, "slow_fill").unwrap();
        let n = 4096usize;
        let bytes = n * std::mem::size_of::<f32>();
        let src = runtime.alloc_raw(bytes).unwrap();
        let dst = runtime.alloc_raw(bytes).unwrap();
        let poison = vec![-999.0f32; n];
        let poison_bytes =
            unsafe { std::slice::from_raw_parts(poison.as_ptr().cast::<u8>(), bytes) };

        for _ in 0..8 {
            unsafe { runtime.htod(poison_bytes, src) }.unwrap();
            runtime.synchronize().unwrap();

            let spin: i64 = 8_000_000;
            let mut builder = runtime.stream().launch_builder(&function);
            let n_u64 = n as u64;
            builder.arg(&src).arg(&n_u64).arg(&spin);
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap();
            }
            // Stream-ordered copy: no explicit synchronize, ordering is by stream.
            unsafe { runtime.dtod_async(src, dst, bytes) }.unwrap();

            let mut out = vec![0.0f32; n];
            let out_bytes =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(out_bytes, dst) }.unwrap();

            for (i, value) in out.iter().enumerate() {
                let expected = 1.0f32 + (i % 7) as f32;
                assert_eq!(
                    *value, expected,
                    "dtod_async observed poison at index {i}: got {value}, expected {expected} \
                     (same-stream ordering violated)"
                );
            }
        }

        unsafe { runtime.free_raw(src) }.unwrap();
        unsafe { runtime.free_raw(dst) }.unwrap();
    }

    // Phase-4 compute/transfer overlap — read-after-write ordering.
    //
    // A weight prefetch is an *asynchronous* host→device copy issued on the
    // dedicated transfer stream, while the consuming kernel runs on the separate
    // compute stream. The two non-blocking streams have NO implicit ordering, so
    // the compute kernel must wait on the transfer's completion event before it
    // may read the destination. This test delays the async copy behind a spin
    // kernel on the transfer stream, then relies solely on
    // `record_copy_fence` + `compute_wait_fence` to order the consumer after it.
    // With the event wait the consumer always reads the fully-transferred
    // payload; if the fence were a no-op placeholder the consumer would race
    // ahead and read the pre-transfer poison.
    #[test]
    fn async_prefetch_h2d_event_orders_copy_before_consume() {
        const MODULE: &str = "runtime_async_prefetch_raw_test";
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
        let Some(runtime) = maybe_runtime() else {
            eprintln!("skipping async prefetch RAW test: CUDA runtime unavailable");
            return;
        };
        let spin_delay = runtime
            .nvrtc_function(MODULE, SOURCE, "spin_delay")
            .unwrap();
        let copy_out = runtime.nvrtc_function(MODULE, SOURCE, "copy_out").unwrap();
        let n = 4096usize;
        let bytes = n * std::mem::size_of::<f32>();
        let dst = runtime.alloc_raw(bytes).unwrap();
        let out = runtime.alloc_raw(bytes).unwrap();

        let mut staging = runtime.alloc_pinned(bytes).unwrap();
        let payload: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32).collect();
        staging.as_mut_slice().copy_from_slice(unsafe {
            std::slice::from_raw_parts(payload.as_ptr().cast::<u8>(), bytes)
        });

        for _ in 0..8 {
            // Poison the destination so a premature (racing) read is detectable.
            let poison = vec![-999.0f32; n];
            let poison_bytes =
                unsafe { std::slice::from_raw_parts(poison.as_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.htod(poison_bytes, dst) }.unwrap();
            runtime.synchronize().unwrap();

            // Occupy the transfer stream so the async H2D copy cannot complete
            // immediately, widening the race the event wait must close.
            let spin: i64 = 8_000_000;
            let mut delay = runtime.copy_stream().launch_builder(&spin_delay);
            delay.arg(&spin);
            unsafe { delay.launch(LaunchConfig::for_num_elems(1)).unwrap() };

            // Async prefetch on the transfer stream, then fence it.
            unsafe { runtime.htod_async(staging.as_slice(), dst) }.unwrap();
            let fence = runtime.record_copy_fence().unwrap();

            // Order the compute stream after the transfer, then consume.
            runtime.compute_wait_fence(fence).unwrap();
            let n_u64 = n as u64;
            let mut consume = runtime.stream().launch_builder(&copy_out);
            consume.arg(&dst).arg(&out).arg(&n_u64);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };

            let mut host = vec![0.0f32; n];
            let host_bytes =
                unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(host_bytes, out) }.unwrap();

            for (i, value) in host.iter().enumerate() {
                let expected = 1.0f32 + (i % 7) as f32;
                assert_eq!(
                    *value, expected,
                    "async prefetch consumer read poison at index {i}: got {value}, \
                     expected {expected} (transfer→compute event ordering violated)"
                );
            }
        }

        unsafe { runtime.free_raw(dst) }.unwrap();
        unsafe { runtime.free_raw(out) }.unwrap();
    }

    // Phase-4 double-buffering — write-after-read safety across waves.
    //
    // The executor prefetches wave N+1's weights into the *alternate* of two
    // device staging buffers while wave N's kernel consumes the current one.
    // With only two buffers, buffer B is reused every second wave, so the
    // transfer that refills B for wave N+2 must not overwrite it while wave N's
    // (still-running) consumer is reading it. `copy_wait_fence` makes the
    // transfer stream wait on the consumer's completion event before the reuse
    // copy; `compute_wait_fence` makes each consumer wait on its transfer. Every
    // wave's output must equal that wave's distinct payload; a missing WAR fence
    // would let a later prefetch clobber a buffer mid-read and corrupt an
    // earlier wave's result.
    #[test]
    fn double_buffered_prefetch_is_race_free_across_waves() {
        const MODULE: &str = "runtime_double_buffer_war_test";
        const SOURCE: &str = r#"
extern "C" __global__ void slow_copy(const float* in, float* out, unsigned long long n, long long spin) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long start = clock64();
    while (clock64() - start < spin) { }
    out[i] = in[i];
}
"#;
        let Some(runtime) = maybe_runtime() else {
            eprintln!("skipping double-buffer WAR test: CUDA runtime unavailable");
            return;
        };
        let slow_copy = runtime.nvrtc_function(MODULE, SOURCE, "slow_copy").unwrap();
        let waves = 6usize;
        let n = 2048usize;
        let bytes = n * std::mem::size_of::<f32>();
        let n_u64 = n as u64;
        let spin: i64 = 8_000_000;
        let payload = |w: usize| -> Vec<f32> {
            (0..n)
                .map(|i| 1.0 + (w as f32) * 13.0 + (i % 5) as f32)
                .collect()
        };

        // Two double-buffered device staging regions, poisoned up front.
        let buf = [
            runtime.alloc_raw(bytes).unwrap(),
            runtime.alloc_raw(bytes).unwrap(),
        ];
        let poison = vec![-777.0f32; n];
        let poison_bytes =
            unsafe { std::slice::from_raw_parts(poison.as_ptr().cast::<u8>(), bytes) };
        for b in buf {
            unsafe { runtime.htod(poison_bytes, b) }.unwrap();
        }
        // Per-wave output buffers and pre-filled pinned host payloads.
        let results: Vec<CUdeviceptr> = (0..waves)
            .map(|_| runtime.alloc_raw(bytes).unwrap())
            .collect();
        let pinned: Vec<PinnedStaging> = (0..waves)
            .map(|w| {
                let mut p = runtime.alloc_pinned(bytes).unwrap();
                let src = payload(w);
                p.as_mut_slice().copy_from_slice(unsafe {
                    std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), bytes)
                });
                p
            })
            .collect();
        runtime.synchronize().unwrap();

        let mut copy_fence = [0u64; 2];
        let mut last_compute_fence = [0u64; 2];

        // Prime the first buffer (no prior consumer, so no WAR wait), then run
        // the double-buffered loop.
        unsafe { runtime.htod_async(pinned[0].as_slice(), buf[0]) }.unwrap();
        copy_fence[0] = runtime.record_copy_fence().unwrap();
        for w in 0..waves {
            let cur = w % 2;
            if w + 1 < waves {
                let nxt = (w + 1) % 2;
                // WAR: do not overwrite buffer `nxt` until the prior wave that
                // consumed it has finished (no-op the first time it is used).
                runtime.copy_wait_fence(last_compute_fence[nxt]).unwrap();
                unsafe { runtime.htod_async(pinned[w + 1].as_slice(), buf[nxt]) }.unwrap();
                copy_fence[nxt] = runtime.record_copy_fence().unwrap();
            }
            // RAW: consumer waits on this buffer's transfer, then reads it.
            runtime.compute_wait_fence(copy_fence[cur]).unwrap();
            let mut consume = runtime.stream().launch_builder(&slow_copy);
            let result = results[w];
            let src = buf[cur];
            consume.arg(&src).arg(&result).arg(&n_u64).arg(&spin);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };
            // Mark the buffer busy until this consumer completes so a future
            // reuse prefetch waits for it.
            last_compute_fence[cur] = runtime.record_compute_fence().unwrap();
        }
        runtime.synchronize().unwrap();

        for (w, &result) in results.iter().enumerate() {
            let mut host = vec![0.0f32; n];
            let host_bytes =
                unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(host_bytes, result) }.unwrap();
            let expected = payload(w);
            assert_eq!(
                host, expected,
                "wave {w} output corrupted — a reuse prefetch clobbered its \
                 staging buffer mid-read (write-after-read fence violated)"
            );
        }

        for b in buf {
            unsafe { runtime.free_raw(b) }.unwrap();
        }
        for result in results {
            unsafe { runtime.free_raw(result) }.unwrap();
        }
    }
}
