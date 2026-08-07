//! Live GPU weight paging (WEIGHT_OFFLOAD Phase 3b): allocate a bounded VRAM
//! page for one offloaded `pkg.nxrt::BlockQuantizedMoE` weight tensor, copy its
//! canonical compressed bytes host→device, and hand back an owned device
//! binding a kernel can read.
//!
//! This is the CUDA implementation of the [`LazyDeviceWeightBinder`] seam that
//! Phase 3a stubbed out with [`Phase3aHostOnlyBinder`]. It copies only the
//! selected external-data region bytes (resolved through [`MmapRegionSource`]),
//! never a full host expansion, so it upholds the WEIGHT_OFFLOAD §9 invariant
//! that residency never allocates an unbudgeted full expert/model expansion and
//! keeps compressed blocks in VRAM (§7.3). The copied bytes are the exact
//! canonical backing bytes, so an offloaded weight is byte-identical to the
//! resident-weight path — offload is an optimization, never an output change.
//!
//! Deferred (clean seams left in place): wiring this binder + [`CudaWeightResidency`]
//! into the executor's live MoE dispatch so the fused kernel consumes the device
//! page, and async prefetch overlap (issues #82/#87).

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    LazyDeviceWeightBinder, LazyWeight, LazyWeightBoundary, MmapRegionSource, WeightHandleError,
};
use onnx_runtime_ir::DataType;
use onnx_runtime_memory_governor::{DeviceAllocator, DeviceKey, HolderId, MemoryRole};

use crate::runtime::{CudaRuntime, PinnedStaging, raw_ptr};

/// Process-global weight-offload activity counters. Every [`CudaWeightResidency`]
/// updates these in addition to its own instance stats, so an end-to-end decode
/// driven through an opaque engine (where the residency handle is not reachable)
/// can still be observed — e.g. a token-parity test asserting that paging and
/// eviction actually happened. Instance [`CudaResidencyStats`] remain the precise
/// per-cache view; these are a coarse cross-cache tally.
static GLOBAL_PAGE_INS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_HITS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_EVICTIONS: AtomicU64 = AtomicU64::new(0);
// Time spent filling the host staging buffer from mmap regions. This is a
// host-blocking CPU memcpy span and contains no CUDA synchronization.
static GLOBAL_MATERIALIZE_NS: AtomicU64 = AtomicU64::new(0);
// CUDA-event elapsed time for H2D DMA: start event before cuMemcpyHtoDAsync,
// end event after it, then host-block on the end event to read elapsed time.
static GLOBAL_HTOD_NS: AtomicU64 = AtomicU64::new(0);
// Host-blocking compute-stream synchronize taken before evicting or unmapping
// pages whose VRAM might still be referenced by earlier kernels.
static GLOBAL_ADMIT_SYNC_NS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_STAGING_FILL_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_STAGING_FILL_REGIONS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_STAGING_FILL_CALLS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_MATERIALIZE_FALLBACK_CALLS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_HTOD_BYTES: AtomicU64 = AtomicU64::new(0);
// Host-blocking cuMemAlloc/cuMemFree spans for paged weight buffers.
static GLOBAL_VRAM_ALLOC_NS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_VRAM_FREE_NS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_BUDGET_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_PEAK_RESIDENT_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_PEAK_RESIDENT_CONTENT_BYTES: AtomicU64 = AtomicU64::new(0);

fn add_duration(counter: &AtomicU64, elapsed: Duration) {
    let nanos = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
    counter.fetch_add(nanos, Ordering::Relaxed);
}

/// Snapshot of the process-global weight-offload counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalOffloadStats {
    pub page_ins: u64,
    pub hits: u64,
    pub evictions: u64,
    pub materialize_ns: u64,
    pub htod_ns: u64,
    pub admit_sync_ns: u64,
    pub staging_fill_bytes: u64,
    pub staging_fill_regions: u64,
    pub staging_fill_calls: u64,
    pub materialize_fallback_calls: u64,
    pub htod_bytes: u64,
    pub vram_alloc_ns: u64,
    pub vram_free_ns: u64,
    pub budget_bytes: u64,
    pub peak_resident_bytes: u64,
    pub peak_resident_content_bytes: u64,
}

/// Read the process-global weight-offload counters.
pub fn global_offload_stats() -> GlobalOffloadStats {
    GlobalOffloadStats {
        page_ins: GLOBAL_PAGE_INS.load(Ordering::Relaxed),
        hits: GLOBAL_HITS.load(Ordering::Relaxed),
        evictions: GLOBAL_EVICTIONS.load(Ordering::Relaxed),
        materialize_ns: GLOBAL_MATERIALIZE_NS.load(Ordering::Relaxed),
        htod_ns: GLOBAL_HTOD_NS.load(Ordering::Relaxed),
        admit_sync_ns: GLOBAL_ADMIT_SYNC_NS.load(Ordering::Relaxed),
        staging_fill_bytes: GLOBAL_STAGING_FILL_BYTES.load(Ordering::Relaxed),
        staging_fill_regions: GLOBAL_STAGING_FILL_REGIONS.load(Ordering::Relaxed),
        staging_fill_calls: GLOBAL_STAGING_FILL_CALLS.load(Ordering::Relaxed),
        materialize_fallback_calls: GLOBAL_MATERIALIZE_FALLBACK_CALLS.load(Ordering::Relaxed),
        htod_bytes: GLOBAL_HTOD_BYTES.load(Ordering::Relaxed),
        vram_alloc_ns: GLOBAL_VRAM_ALLOC_NS.load(Ordering::Relaxed),
        vram_free_ns: GLOBAL_VRAM_FREE_NS.load(Ordering::Relaxed),
        budget_bytes: GLOBAL_BUDGET_BYTES.load(Ordering::Relaxed),
        peak_resident_bytes: GLOBAL_PEAK_RESIDENT_BYTES.load(Ordering::Relaxed),
        peak_resident_content_bytes: GLOBAL_PEAK_RESIDENT_CONTENT_BYTES.load(Ordering::Relaxed),
    }
}

/// Reset the process-global weight-offload counters (test observability helper).
pub fn reset_global_offload_stats() {
    GLOBAL_PAGE_INS.store(0, Ordering::Relaxed);
    GLOBAL_HITS.store(0, Ordering::Relaxed);
    GLOBAL_EVICTIONS.store(0, Ordering::Relaxed);
    GLOBAL_MATERIALIZE_NS.store(0, Ordering::Relaxed);
    GLOBAL_HTOD_NS.store(0, Ordering::Relaxed);
    GLOBAL_ADMIT_SYNC_NS.store(0, Ordering::Relaxed);
    GLOBAL_STAGING_FILL_BYTES.store(0, Ordering::Relaxed);
    GLOBAL_STAGING_FILL_REGIONS.store(0, Ordering::Relaxed);
    GLOBAL_STAGING_FILL_CALLS.store(0, Ordering::Relaxed);
    GLOBAL_MATERIALIZE_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    GLOBAL_HTOD_BYTES.store(0, Ordering::Relaxed);
    GLOBAL_VRAM_ALLOC_NS.store(0, Ordering::Relaxed);
    GLOBAL_VRAM_FREE_NS.store(0, Ordering::Relaxed);
    GLOBAL_BUDGET_BYTES.store(0, Ordering::Relaxed);
    GLOBAL_PEAK_RESIDENT_BYTES.store(0, Ordering::Relaxed);
    GLOBAL_PEAK_RESIDENT_CONTENT_BYTES.store(0, Ordering::Relaxed);
}

/// Environment switch that enables the CUDA device residency cache. Reuses the
/// same knob as the CPU host-cache offload path (`onnx_runtime_ep_cpu`) so a
/// single `ONNX_GENAI_WEIGHT_OFFLOAD=1` turns offload on for whichever EP runs.
pub const WEIGHT_OFFLOAD_ENV: &str = onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV;

/// VRAM budget (bytes) for the device residency cache. When unset the residency
/// manager is constructed with a caller-chosen default.
pub const WEIGHT_OFFLOAD_DEVICE_BYTES_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES";

/// Sub-knob (default ON / opt-OUT) selecting the asynchronous, fence-ordered
/// residency page-in over the synchronous `cuMemcpyHtoD`. Set to a truthy value
/// (`1`/`true`/`yes`/`on`, case/whitespace-insensitive) to force async page-in;
/// set to a falsy value (`0`/`false`/`no`/`off`) to force synchronous page-in.
/// Unset uses async because the not-fit WDDM regime needs a prefetchable,
/// fence-ordered H2D path; keeping the old synchronous default made every
/// lookahead request decline before it could overlap the known layer order. This
/// knob only has any effect when weight offload itself is enabled
/// (`ONNX_GENAI_WEIGHT_OFFLOAD=1`); with offload off the resident fast path is
/// untouched and byte-identical regardless.
pub const WEIGHT_OFFLOAD_ASYNC_PAGEIN_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN";

/// Sub-knob (default ON / opt-OUT) selecting scan-resistant residency for dense
/// per-layer weight walks. Set to a falsey value (`0`/`false`/`no`/`off`,
/// case/whitespace-insensitive) to force the old LRU policy for A/B.
///
/// Default is ON because qwen2.5-14b measured 0/6,936 LRU hits at a 6 GB
/// budget, while stable dense residency measured 5,145/6,936 hits (74.18%) and
/// reduced evictions from 6,286 to 0 in the same harness.
pub const WEIGHT_OFFLOAD_SCAN_RESISTANT_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_SCAN_RESISTANT";

/// Opt-in stage-1 path that backs weight pages with stable CUDA VMM virtual
/// addresses instead of per-page-in `cuMemAlloc` allocations.
///
/// This is not the default yet: CUDA VMM physical memory cannot oversubscribe
/// into WDDM shared memory the way `cudaMalloc` can, so enabling it can turn a
/// spill-surviving run into a hard device OOM on Windows.
pub const WEIGHT_OFFLOAD_VMM_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_VMM";

/// Parse [`WEIGHT_OFFLOAD_ASYNC_PAGEIN_ENV`]. Async page-in is **default-on**:
/// unset (`None`) enables it, an explicit falsy value disables it, and truthy
/// values keep it enabled.
pub(crate) fn async_pagein_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => true,
    }
}

/// Parse [`WEIGHT_OFFLOAD_SCAN_RESISTANT_ENV`]. Scan-resistant dense residency
/// defaults ON; explicit falsey values force old LRU for A/B and rollback.
pub(crate) fn scan_resistant_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        None => true,
    }
}

fn truthy_env_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Whether/how the CUDA EP should page offloaded weights into a bounded VRAM
/// residency cache. Disabled by default so the resident fast path is untouched
/// and byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceOffloadPolicy {
    pub enabled: bool,
    /// Explicit VRAM budget in bytes, if the operator pinned one.
    pub device_budget_bytes: Option<u64>,
    /// Use the asynchronous, fence-ordered page-in (default `true` / opt-out).
    /// This is the only path that can prefetch the next known layer while the
    /// current layer runs; set `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=0` to
    /// force the old synchronous demand page-in for A/B measurements.
    pub async_pagein: bool,
    /// Use scan-resistant stable-subset residency for dense per-layer weights.
    /// Default-on / opt-out via `ONNX_GENAI_WEIGHT_OFFLOAD_SCAN_RESISTANT=0`;
    /// MoE boundaries stay on LRU even when this is enabled to avoid regressing
    /// skewed expert selection.
    pub scan_resistant_dense: bool,
    /// Use stable CUDA VMM virtual addresses for paged weights. Stage 1 keeps
    /// graph capture declined until residency guarantees are added.
    pub stable_vmm: bool,
}

impl Default for DeviceOffloadPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            device_budget_bytes: None,
            async_pagein: false,
            scan_resistant_dense: true,
            stable_vmm: false,
        }
    }
}

impl DeviceOffloadPolicy {
    /// Read the policy from the process environment.
    pub fn from_env() -> Self {
        let enabled = std::env::var_os(WEIGHT_OFFLOAD_ENV).is_some_and(|value| value == "1");
        let device_budget_bytes = std::env::var(WEIGHT_OFFLOAD_DEVICE_BYTES_ENV)
            .ok()
            .and_then(|value| parse_budget_bytes(&value));
        // Async page-in defaults ON; an explicit falsy value restores the old
        // synchronous demand-copy path for A/B.
        let async_pagein = async_pagein_from_env_value(
            std::env::var(WEIGHT_OFFLOAD_ASYNC_PAGEIN_ENV)
                .ok()
                .as_deref(),
        );
        let scan_resistant_dense = scan_resistant_from_env_value(
            std::env::var(WEIGHT_OFFLOAD_SCAN_RESISTANT_ENV)
                .ok()
                .as_deref(),
        );
        let stable_vmm = truthy_env_value(std::env::var(WEIGHT_OFFLOAD_VMM_ENV).ok().as_deref());
        Self {
            enabled,
            device_budget_bytes,
            async_pagein,
            scan_resistant_dense,
            stable_vmm,
        }
    }
}

/// Parse a VRAM budget string into bytes, rejecting empty/garbage/zero values.
fn parse_budget_bytes(value: &str) -> Option<u64> {
    match value.trim().parse::<u64>() {
        Ok(bytes) if bytes > 0 => Some(bytes),
        _ => None,
    }
}

/// A snapshot of [`CudaWeightResidency`] activity for observability / tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaResidencyStats {
    /// Budget the cache tries to stay within, in bytes.
    pub budget_bytes: u64,
    /// Bytes currently resident across all cached pages.
    pub resident_bytes: u64,
    /// High-water mark of `resident_bytes`.
    pub peak_resident_bytes: u64,
    /// Canonical content bytes currently resident. This differs from
    /// `resident_bytes` when stable VMM rounds mappings to CUDA granules.
    pub resident_content_bytes: u64,
    /// High-water mark of `resident_content_bytes`.
    pub peak_resident_content_bytes: u64,
    /// Number of pages currently resident.
    pub pages_resident: u64,
    /// H2D page-ins performed (cache misses that allocated + copied a page).
    pub page_ins: u64,
    /// Cache hits that reused an already-resident page (no H2D copy).
    pub hits: u64,
    /// LRU evictions that freed a page's VRAM.
    pub evictions: u64,
}

/// A live VRAM residency page for one offloaded weight tensor.
///
/// Owns a single device allocation holding the tensor's canonical compressed
/// bytes and frees it exactly once on drop. The address is a CUDA device
/// pointer — never dereferenced on the host — exposed through [`Self::device_ptr`]
/// for a consuming kernel's `TensorView`.
pub struct CudaWeightPage {
    storage: CudaWeightPageStorage,
    ptr: CUdeviceptr,
    len: usize,
    dtype: DataType,
    shape: Vec<usize>,
}

enum CudaWeightPageStorage {
    Raw { runtime: Arc<CudaRuntime> },
    StableVmm { slot: Arc<StableVmmSlot> },
}

impl CudaWeightPage {
    /// Allocate a VRAM page and copy `bytes` host→device into it. The bytes are
    /// the canonical (compressed) backing of the tensor, so the page is
    /// byte-identical to a resident upload. Frees the allocation on copy failure.
    pub fn upload(
        runtime: &Arc<CudaRuntime>,
        dtype: DataType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<Self, WeightHandleError> {
        if bytes.is_empty() {
            return Err(WeightHandleError::MissingRegions);
        }
        let alloc_start = std::time::Instant::now();
        let ptr = runtime
            .alloc_raw(bytes.len())
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        add_duration(&GLOBAL_VRAM_ALLOC_NS, alloc_start.elapsed());
        let page = Self {
            storage: CudaWeightPageStorage::Raw {
                runtime: Arc::clone(runtime),
            },
            ptr,
            len: bytes.len(),
            dtype,
            shape,
        };
        // SAFETY: `ptr` owns `bytes.len()` bytes; `page`'s Drop frees it if the
        // copy below fails.
        let copy_start = std::time::Instant::now();
        unsafe { runtime.htod(bytes, ptr) }
            .map_err(|error| WeightHandleError::DeviceBinding(format!("H2D copy: {error}")))?;
        add_duration(&GLOBAL_HTOD_NS, copy_start.elapsed());
        GLOBAL_HTOD_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(page)
    }

    /// Asynchronous, overlap-friendly variant of [`Self::upload`]: allocate a
    /// VRAM page and enqueue a host→device copy of `bytes` on the runtime's
    /// dedicated transfer stream, returning the page plus a **copy fence** the
    /// caller MUST order the consuming compute work after (via
    /// [`CudaRuntime::compute_wait_fence`]). Because the copy is asynchronous,
    /// the source bytes are first staged into a caller-owned page-locked buffer
    /// that MUST outlive the returned fence. Frees the VRAM on any failure.
    pub fn upload_async(
        runtime: &Arc<CudaRuntime>,
        dtype: DataType,
        shape: Vec<usize>,
        bytes: &[u8],
        mut staging: PinnedStaging,
    ) -> Result<(Self, u64, PinnedStaging), WeightHandleError> {
        if bytes.is_empty() {
            return Err(WeightHandleError::MissingRegions);
        }
        if staging.len() < bytes.len() {
            return Err(WeightHandleError::InvalidResident(format!(
                "pinned staging buffer is too small: {} < {}",
                staging.len(),
                bytes.len()
            )));
        }
        let alloc_start = std::time::Instant::now();
        let ptr = runtime
            .alloc_raw(bytes.len())
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        add_duration(&GLOBAL_VRAM_ALLOC_NS, alloc_start.elapsed());
        staging.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        // Own the VRAM `ptr` before enqueuing the copy, so any error below drops
        // `page` and frees it exactly once. The pinned staging remains owned by
        // this function and is returned to the caller on success so it can keep
        // the source alive until the fence completes.
        let page = Self {
            storage: CudaWeightPageStorage::Raw {
                runtime: Arc::clone(runtime),
            },
            ptr,
            len: bytes.len(),
            dtype,
            shape,
        };
        // SAFETY: `dst` (`ptr`) owns `len` bytes; the async source is pinned
        // staging owned by the caller until the returned fence is awaited.
        let staged = &staging.as_slice()[..bytes.len()];
        let copy_start = std::time::Instant::now();
        unsafe { runtime.htod_async(staged, ptr) }.map_err(|error| {
            WeightHandleError::DeviceBinding(format!("async H2D copy: {error}"))
        })?;
        let fence = runtime
            .record_copy_fence()
            .map_err(|error| WeightHandleError::DeviceBinding(format!("copy fence: {error}")))?;
        add_duration(&GLOBAL_HTOD_NS, copy_start.elapsed());
        GLOBAL_HTOD_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok((page, fence, staging))
    }

    /// Asynchronous upload from an already-filled pinned staging buffer.
    ///
    /// The live weight-offload path uses this to copy directly from the package
    /// mmap into reusable pinned staging, avoiding the failure mode where every
    /// page-in first rebuilt a throwaway owned host tensor.
    pub fn upload_staged_async(
        runtime: &Arc<CudaRuntime>,
        dtype: DataType,
        shape: Vec<usize>,
        len: usize,
        staging: PinnedStaging,
    ) -> Result<(Self, u64, PinnedStaging), WeightHandleError> {
        if len == 0 {
            return Err(WeightHandleError::MissingRegions);
        }
        if staging.len() < len {
            return Err(WeightHandleError::InvalidResident(format!(
                "pinned staging buffer is too small: {} < {}",
                staging.len(),
                len
            )));
        }
        let alloc_start = std::time::Instant::now();
        let ptr = runtime
            .alloc_raw(len)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        add_duration(&GLOBAL_VRAM_ALLOC_NS, alloc_start.elapsed());
        let page = Self {
            storage: CudaWeightPageStorage::Raw {
                runtime: Arc::clone(runtime),
            },
            ptr,
            len,
            dtype,
            shape,
        };
        let staged = &staging.as_slice()[..len];
        let copy_ms = unsafe { runtime.htod_async_elapsed_ms(staged, ptr) }.map_err(|error| {
            WeightHandleError::DeviceBinding(format!("measured H2D copy: {error}"))
        })?;
        GLOBAL_HTOD_NS.fetch_add((copy_ms * 1_000_000.0) as u64, Ordering::Relaxed);
        GLOBAL_HTOD_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        Ok((page, 0, staging))
    }

    /// Opaque device pointer to the paged bytes, for a kernel `TensorView`.
    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        raw_ptr(self.ptr)
    }

    /// Number of canonical bytes resident in this VRAM page.
    pub fn len(&self) -> usize {
        self.len
    }

    fn accounting_len(&self) -> usize {
        match &self.storage {
            CudaWeightPageStorage::Raw { .. } => self.len,
            CudaWeightPageStorage::StableVmm { slot } => slot.allocation_bytes,
        }
    }

    /// Whether the page is empty (never true for a validated binding).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Canonical element type of the paged tensor.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Canonical shape of the paged tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

fn fill_staging_from_regions(
    weight: &LazyWeight,
    source: &dyn MmapRegionSource,
    staging: &mut PinnedStaging,
) -> Result<(), WeightHandleError> {
    let total = weight.region_bytes_len();
    if total == 0 {
        return Err(WeightHandleError::MissingRegions);
    }
    if staging.len() < total {
        return Err(WeightHandleError::InvalidResident(format!(
            "pinned staging buffer is too small: {} < {}",
            staging.len(),
            total
        )));
    }
    let mut offset = 0usize;
    for region in &weight.regions {
        let bytes = source.region_bytes(region)?;
        if bytes.len() != region.len {
            return Err(WeightHandleError::DeviceBinding(format!(
                "region source returned {} bytes for a {}-byte region",
                bytes.len(),
                region.len
            )));
        }
        let end = offset.checked_add(bytes.len()).ok_or_else(|| {
            WeightHandleError::InvalidResident("staging byte count overflow".into())
        })?;
        staging.as_mut_slice()[offset..end].copy_from_slice(bytes);
        offset = end;
    }
    GLOBAL_STAGING_FILL_CALLS.fetch_add(1, Ordering::Relaxed);
    GLOBAL_STAGING_FILL_REGIONS.fetch_add(weight.regions.len() as u64, Ordering::Relaxed);
    GLOBAL_STAGING_FILL_BYTES.fetch_add(total as u64, Ordering::Relaxed);
    Ok(())
}

impl std::fmt::Debug for CudaWeightPage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaWeightPage")
            .field("len", &self.len)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

struct StableVmmSlot {
    runtime: Arc<CudaRuntime>,
    allocator: Arc<crate::vmm_allocator::CudaVmmAllocator>,
    ptr: NonNull<u8>,
    allocation_bytes: usize,
    committed: Mutex<bool>,
}

unsafe impl Send for StableVmmSlot {}
unsafe impl Sync for StableVmmSlot {}

impl StableVmmSlot {
    fn new(
        runtime: Arc<CudaRuntime>,
        allocator: Arc<crate::vmm_allocator::CudaVmmAllocator>,
        len: usize,
    ) -> Result<Self, WeightHandleError> {
        let granularity = allocator.granularity();
        let allocation_bytes = len.next_multiple_of(granularity);
        let ptr = allocator
            .allocate_committed(allocation_bytes, granularity, &[])
            .map_err(|error| {
                WeightHandleError::DeviceBinding(format!(
                    "stable weight VA reserve ({allocation_bytes} bytes): {error}"
                ))
            })?;
        Ok(Self {
            runtime,
            allocator,
            ptr,
            allocation_bytes,
            committed: Mutex::new(false),
        })
    }

    fn device_ptr(&self) -> CUdeviceptr {
        self.ptr.as_ptr() as CUdeviceptr
    }

    fn commit(&self, len: usize) -> Result<(), WeightHandleError> {
        self.allocator
            .commit_allocation_range(
                self.ptr,
                self.allocation_bytes,
                self.allocator.granularity(),
                0,
                len,
            )
            .map_err(|error| {
                WeightHandleError::DeviceBinding(format!("stable weight commit: {error}"))
            })?;
        *self.committed.lock().expect("stable slot lock poisoned") = true;
        Ok(())
    }

    fn decommit(&self) {
        let mut committed = self
            .committed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !*committed {
            return;
        }
        let sync_start = std::time::Instant::now();
        if self.runtime.synchronize().is_ok() {
            add_duration(&GLOBAL_ADMIT_SYNC_NS, sync_start.elapsed());
        }
        let _ = self.allocator.decommit_allocation_range(
            self.ptr,
            self.allocation_bytes,
            self.allocator.granularity(),
            0,
            self.allocation_bytes,
        );
        *committed = false;
    }
}

impl Drop for StableVmmSlot {
    fn drop(&mut self) {
        self.decommit();
        // SAFETY: `ptr` came from this allocator, and this slot is the only
        // owner that deallocates the stable virtual span.
        unsafe {
            self.allocator.deallocate(
                self.ptr,
                self.allocation_bytes,
                self.allocator.granularity(),
            );
        }
    }
}

impl Drop for CudaWeightPage {
    fn drop(&mut self) {
        match &self.storage {
            CudaWeightPageStorage::Raw { runtime } => {
                // SAFETY: `ptr` came from this runtime's `alloc_raw` and is freed
                // exactly once here; no alias to it escapes `CudaWeightPage`.
                let free_start = std::time::Instant::now();
                let _ = unsafe { runtime.free_raw(self.ptr) };
                add_duration(&GLOBAL_VRAM_FREE_NS, free_start.elapsed());
            }
            CudaWeightPageStorage::StableVmm { slot } => {
                slot.decommit();
            }
        }
    }
}

/// CUDA Phase-3b device binder: pages one offloaded weight tensor into VRAM.
///
/// Copies the canonical compressed region bytes host→device, so the device page
/// is byte-identical to the resident tensor a stock EP would upload.
pub struct CudaWeightPager<'a, S: MmapRegionSource + ?Sized> {
    runtime: Arc<CudaRuntime>,
    source: &'a S,
}

impl<'a, S: MmapRegionSource + ?Sized> CudaWeightPager<'a, S> {
    pub fn new(runtime: Arc<CudaRuntime>, source: &'a S) -> Self {
        Self { runtime, source }
    }
}

impl<S: MmapRegionSource + ?Sized> LazyDeviceWeightBinder for CudaWeightPager<'_, S> {
    type Binding = CudaWeightPage;

    fn bind_block_quantized_moe(
        &self,
        weight: &LazyWeight,
    ) -> Result<Self::Binding, WeightHandleError> {
        let total = weight.region_bytes_len();
        if total == 0 {
            return Err(WeightHandleError::MissingRegions);
        }

        // Allocate the VRAM page up front, then stream each selected region into
        // its contiguous slot. The page is owned by `CudaWeightPage` on success;
        // on any copy failure we free it before returning so no VRAM leaks.
        let ptr = self
            .runtime
            .alloc_raw(total)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        let page = CudaWeightPage {
            storage: CudaWeightPageStorage::Raw {
                runtime: Arc::clone(&self.runtime),
            },
            ptr,
            len: total,
            dtype: weight.dtype,
            shape: weight.shape.clone(),
        };

        let mut offset: usize = 0;
        for region in &weight.regions {
            let bytes = self.source.region_bytes(region)?;
            if bytes.len() != region.len {
                return Err(WeightHandleError::DeviceBinding(format!(
                    "region source returned {} bytes for a {}-byte region",
                    bytes.len(),
                    region.len
                )));
            }
            let dst = ptr + offset as CUdeviceptr;
            // SAFETY: `dst` lies within the `total`-byte allocation `page` owns
            // ([offset, offset + region.len) with the running sum bounded by
            // `total`); `bytes` covers exactly `region.len` bytes.
            let copy_start = std::time::Instant::now();
            unsafe { self.runtime.htod(bytes, dst) }
                .map_err(|error| WeightHandleError::DeviceBinding(format!("H2D copy: {error}")))?;
            add_duration(&GLOBAL_HTOD_NS, copy_start.elapsed());
            GLOBAL_HTOD_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            offset += region.len;
        }

        Ok(page)
    }
}

/// A bounded-VRAM LRU cache of live device weight pages (WEIGHT_OFFLOAD Phase 3b
/// "page-in + eviction").
///
/// Pages are keyed by a stable caller identifier (e.g. the graph `ValueId` of the
/// offloaded initializer). On a miss the weight's canonical region bytes are
/// streamed host→device through [`CudaWeightPager`]; on a hit the already-resident
/// [`CudaWeightPage`] is reused with no H2D traffic. When admitting a page would
/// exceed the VRAM budget, least-recently-used pages are evicted first.
///
/// Eviction is use-safe: a page is only reclaimed while the cache is its sole
/// owner (`Arc::strong_count == 1`). A page handed out and still referenced by an
/// in-flight kernel is never freed, so a paged weight can never be pulled out from
/// under a running MoE dispatch.
///
/// A single weight larger than the whole budget is still paged in (correctness
/// beats the budget); the cache simply runs transiently over budget for it.
///
/// Page-in currently performs measured, host-blocking H2D copies. The earlier
/// prefetch/in-flight path was removed because it kept VRAM outside the lease
/// and reported non-blocking stream-wait enqueue time as copy waiting.
pub struct CudaWeightResidency {
    runtime: Arc<CudaRuntime>,
    scan_resistant_dense: bool,
    inner: Mutex<ResidencyInner>,
    stable_vmm: Option<Arc<crate::vmm_allocator::CudaVmmAllocator>>,
}

struct ResidencyInner {
    policy: WeightResidencyPolicy,
    pages: HashMap<u64, Arc<CudaWeightPage>>,
    stable_slots: HashMap<u64, Arc<StableVmmSlot>>,
    /// The governor grant this budget came from, when it came from one.
    ///
    /// Held for its `Drop`: releasing the lease is how the tier learns these
    /// bytes are available again. `None` means the budget was chosen locally and
    /// no governor knows about it.
    ///
    /// **Declared after `pages` on purpose.** Rust drops fields in declaration
    /// order, so this releases the entitlement only once the pages holding the
    /// VRAM have been dropped. The other order tells the governor those bytes
    /// are free while they are still held, and hands them to the next requester
    /// on top of memory that has not been returned yet.
    ///
    /// Note this only orders the pages *this cache* still holds. An
    /// `Arc<CudaWeightPage>` handed to a caller keeps its page alive
    /// independently, so a caller outliving the cache still outlives the lease.
    /// See `residency_holds_its_lease_until_its_pages_are_gone`.
    lease: Option<onnx_runtime_memory_governor::MemoryLease>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightEvictionPolicy {
    Lru,
    StableResident,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
struct WeightPolicyAccess {
    hit: bool,
    admitted: bool,
    evicted: Vec<u64>,
}

/// Pure bounded-residency eviction/accounting state.
///
/// This owns only keys, byte sizes, recency order, the byte budget, and cache
/// counters. CUDA page ownership and `Arc` liveness checks stay in
/// [`CudaWeightResidency`], which supplies an evictability predicate when it
/// asks this policy state to make room.
#[derive(Debug)]
struct WeightResidencyPolicy {
    budget: u64,
    resident_bytes: u64,
    peak_resident_bytes: u64,
    resident_content_bytes: u64,
    peak_resident_content_bytes: u64,
    page_ins: u64,
    hits: u64,
    evictions: u64,
    /// LRU order: front = least-recently-used, back = most-recently-used.
    order: Vec<u64>,
    bytes_by_key: HashMap<u64, u64>,
    content_bytes_by_key: HashMap<u64, u64>,
}

impl WeightResidencyPolicy {
    fn new(budget: u64) -> Self {
        Self {
            budget,
            resident_bytes: 0,
            peak_resident_bytes: 0,
            resident_content_bytes: 0,
            peak_resident_content_bytes: 0,
            page_ins: 0,
            hits: 0,
            evictions: 0,
            order: Vec::new(),
            bytes_by_key: HashMap::new(),
            content_bytes_by_key: HashMap::new(),
        }
    }

    #[cfg(test)]
    fn access(
        &mut self,
        key: u64,
        bytes: u64,
        eviction: WeightEvictionPolicy,
    ) -> WeightPolicyAccess {
        if self.bytes_by_key.contains_key(&key) {
            self.record_hit(key);
            return WeightPolicyAccess {
                hit: true,
                admitted: false,
                evicted: Vec::new(),
            };
        }
        if eviction == WeightEvictionPolicy::StableResident && !self.can_fit(bytes) {
            self.record_page_in();
            return WeightPolicyAccess {
                hit: false,
                admitted: false,
                evicted: Vec::new(),
            };
        }
        let evicted = self.evict_to_fit(bytes, eviction, |_| true);
        self.insert_page(key, bytes, bytes);
        WeightPolicyAccess {
            hit: false,
            admitted: true,
            evicted,
        }
    }

    fn can_fit(&self, incoming: u64) -> bool {
        self.resident_bytes.saturating_add(incoming) <= self.budget
    }

    fn touch(&mut self, key: u64) {
        if let Some(position) = self.order.iter().position(|&k| k == key) {
            let key = self.order.remove(position);
            self.order.push(key);
        }
    }

    fn record_hit(&mut self, key: u64) {
        self.touch(key);
        self.hits += 1;
    }

    fn insert_page(&mut self, key: u64, bytes: u64, content_bytes: u64) {
        self.bytes_by_key.insert(key, bytes);
        self.content_bytes_by_key.insert(key, content_bytes);
        self.order.push(key);
        self.resident_bytes += bytes;
        self.peak_resident_bytes = self.peak_resident_bytes.max(self.resident_bytes);
        self.resident_content_bytes += content_bytes;
        self.peak_resident_content_bytes = self
            .peak_resident_content_bytes
            .max(self.resident_content_bytes);
        self.record_page_in();
    }

    fn record_page_in(&mut self) {
        self.page_ins += 1;
    }

    fn evict_to_fit<F>(
        &mut self,
        incoming: u64,
        eviction: WeightEvictionPolicy,
        mut evictable: F,
    ) -> Vec<u64>
    where
        F: FnMut(u64) -> bool,
    {
        let mut evicted = Vec::new();
        while self.resident_bytes.saturating_add(incoming) > self.budget && !self.order.is_empty() {
            let Some(index) = self.next_evictable_index(eviction, &mut evictable) else {
                break;
            };
            let key = self.order.remove(index);
            if let Some(bytes) = self.bytes_by_key.remove(&key) {
                self.resident_bytes = self.resident_bytes.saturating_sub(bytes);
                if let Some(content_bytes) = self.content_bytes_by_key.remove(&key) {
                    self.resident_content_bytes =
                        self.resident_content_bytes.saturating_sub(content_bytes);
                }
                self.evictions += 1;
                evicted.push(key);
            }
        }
        evicted
    }

    fn next_evictable_index<F>(
        &self,
        eviction: WeightEvictionPolicy,
        evictable: &mut F,
    ) -> Option<usize>
    where
        F: FnMut(u64) -> bool,
    {
        match eviction {
            WeightEvictionPolicy::Lru | WeightEvictionPolicy::StableResident => {
                self.order.iter().position(|&key| evictable(key))
            }
        }
    }
}

fn eviction_for_boundary(
    scan_resistant_dense: bool,
    boundary: LazyWeightBoundary,
) -> WeightEvictionPolicy {
    if scan_resistant_dense
        && matches!(
            boundary,
            LazyWeightBoundary::MatMul | LazyWeightBoundary::MatMulNBits
        )
    {
        WeightEvictionPolicy::StableResident
    } else {
        WeightEvictionPolicy::Lru
    }
}

impl CudaWeightResidency {
    /// Build a residency cache with an explicit VRAM `budget_bytes`. This
    /// constructor is synchronous by itself so tests can choose deliberately;
    /// [`DeviceOffloadPolicy::from_env`] supplies the runtime default.
    ///
    /// Prefer [`Self::new_leased`] where a governor exists. A budget invented
    /// here is a second claim on the same VRAM the governor is already handing
    /// out, and neither side can see the other's.
    pub fn new(runtime: Arc<CudaRuntime>, budget_bytes: u64) -> Self {
        GLOBAL_BUDGET_BYTES.store(budget_bytes, Ordering::Relaxed);
        Self {
            runtime,
            scan_resistant_dense: false,
            inner: Mutex::new(ResidencyInner {
                policy: WeightResidencyPolicy::new(budget_bytes),
                lease: None,
                pages: HashMap::new(),
                stable_slots: HashMap::new(),
            }),
            stable_vmm: None,
        }
    }

    /// Build a residency cache whose budget is *leased* from `governor` rather
    /// than chosen here.
    ///
    /// This cache holds device memory for as long as a model is loaded, so a
    /// budget it picks for itself is a second ledger over the same VRAM the
    /// governor is dividing between KV and everything else. Nothing reconciles
    /// the two: grant KV most of an 8 GiB card and let this default to 4 GiB,
    /// and both are individually satisfied while the card is oversubscribed.
    ///
    /// The lease is taken under [`MemoryRole::Weights`], which is what these
    /// bytes are and which already carries the right eviction semantics --
    /// immutable, shareable, and re-readable from the package on disk, so the
    /// cheapest thing to demote under pressure.
    ///
    /// # Errors
    ///
    /// If the tier cannot grant `budget_bytes`. Failing here is the point: it
    /// says the model does not fit *before* pages start being admitted, rather
    /// than letting two budgets agree separately and discovering it at a
    /// `cuMemAlloc` in the middle of generation.
    pub fn new_leased(
        runtime: Arc<CudaRuntime>,
        budget_bytes: u64,
        governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> Result<Self, onnx_runtime_memory_governor::MemoryError> {
        let lease = governor.reserve(
            tier,
            budget_bytes,
            onnx_runtime_memory_governor::MemoryRole::Weights,
            holder,
        )?;
        GLOBAL_BUDGET_BYTES.store(lease.bytes(), Ordering::Relaxed);
        Ok(Self {
            runtime,
            scan_resistant_dense: false,
            inner: Mutex::new(ResidencyInner {
                policy: WeightResidencyPolicy::new(lease.bytes()),
                lease: Some(lease),
                pages: HashMap::new(),
                stable_slots: HashMap::new(),
            }),
            stable_vmm: None,
        })
    }

    /// Bytes this cache is entitled to, and whether that came from a governor.
    ///
    /// `false` means the budget was chosen locally and nothing reconciles it
    /// with any other claim on the same device.
    pub fn budget(&self) -> (u64, bool) {
        let inner = self.inner.lock().expect("residency lock poisoned");
        (inner.policy.budget, inner.lease.is_some())
    }

    /// Replace the locally chosen budget before the cache is governed.
    ///
    /// Automatic `--vram-limit` offload must leave room for native KV and fixed
    /// state. The CUDA EP is constructed before those bytes are known, so the
    /// engine corrects the provisional residency budget after sizing the
    /// session but before adoption. Refusing once governed prevents silently
    /// shrinking a lease while pages may already be resident.
    pub fn set_ungoverned_budget(
        &self,
        budget_bytes: u64,
    ) -> Result<u64, onnx_runtime_memory_governor::MemoryError> {
        let mut inner = self.inner.lock().expect("residency lock poisoned");
        if inner.lease.is_some() {
            return Ok(inner.policy.budget);
        }
        inner.policy.budget = budget_bytes;
        Ok(inner.policy.budget)
    }

    /// Replace a locally chosen budget with one leased from `governor`.
    ///
    /// The execution provider is built before the engine's governor exists, so
    /// the cache starts with the operator's figure or a default. This is where
    /// that becomes a claim the rest of the system can see. Nothing has paged in
    /// yet at that point, so no resident bytes are stranded by the swap.
    ///
    /// Asks for the budget it already had. That figure is what the operator
    /// pinned or what the EP defaulted to; the governor's job here is to say
    /// whether the device can actually afford it alongside everything else, not
    /// to invent a different number.
    ///
    /// # Errors
    ///
    /// If the tier cannot grant it. The budget is left as it was, so a caller
    /// that chooses to continue ungoverned is no worse off than before -- but it
    /// now knows.
    pub fn adopt_governed_budget(
        &self,
        governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> Result<u64, onnx_runtime_memory_governor::MemoryError> {
        if let Some(stable_vmm) = &self.stable_vmm {
            let _ = stable_vmm.adopt_governor(governor, holder);
            let inner = self.inner.lock().expect("residency lock poisoned");
            stable_vmm.precharge_budget(inner.policy.budget)?;
            GLOBAL_BUDGET_BYTES.store(inner.policy.budget, Ordering::Relaxed);
            return Ok(inner.policy.budget);
        }
        let requested = {
            let inner = self.inner.lock().expect("residency lock poisoned");
            if inner.lease.is_some() {
                return Ok(inner.policy.budget);
            }
            inner.policy.budget
        };
        let lease = governor.reserve(
            tier,
            requested,
            onnx_runtime_memory_governor::MemoryRole::Weights,
            holder,
        )?;
        let granted = lease.bytes();
        let mut inner = self.inner.lock().expect("residency lock poisoned");
        inner.policy.budget = granted;
        inner.lease = Some(lease);
        GLOBAL_BUDGET_BYTES.store(granted, Ordering::Relaxed);
        Ok(granted)
    }

    /// Select the asynchronous (default `true`) vs synchronous page-in path.
    pub fn with_async_pagein(self, async_pagein: bool) -> Self {
        let _ = async_pagein;
        self
    }

    /// Select whether dense per-layer weights use scan-resistant residency.
    pub fn with_scan_resistant_dense(mut self, scan_resistant_dense: bool) -> Self {
        self.scan_resistant_dense = scan_resistant_dense;
        self
    }

    /// Use CUDA VMM for weight pages, keeping one stable virtual address per
    /// weight key while mapping/unmapping physical granules on page-in/eviction.
    pub fn with_stable_vmm(mut self) -> Result<Self, onnx_runtime_memory_governor::MemoryError> {
        const RESERVATION_BYTES: usize = 64 << 30;
        let allocator = crate::vmm_allocator::CudaVmmAllocator::detached(
            self.runtime.cuda_context(),
            DeviceKey::device(self.runtime.ordinal()),
            self.runtime.ordinal() as i32,
            RESERVATION_BYTES,
            HolderId::new(716),
            MemoryRole::Weights,
        )?;
        self.stable_vmm = Some(Arc::new(allocator));
        Ok(self)
    }

    /// Return the device page for `key`, paging it in from `source` on a miss and
    /// evicting LRU pages to respect the budget. The returned [`Arc`] keeps the
    /// page resident for as long as the caller holds it.
    pub fn resident<S: MmapRegionSource>(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &S,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        if let Some(hit) = self.get_hit(key) {
            return Ok(hit);
        }
        if self.stable_vmm.is_some() {
            return self.resident_mapped(key, weight, source);
        }
        // Copy region bytes host→device before re-locking so a failed bind never
        // mutates cache accounting.
        let pager = CudaWeightPager::new(Arc::clone(&self.runtime), source);
        let page = Arc::new(pager.bind_block_quantized_moe(weight)?);
        self.admit(key, page, self.eviction_for(weight.boundary))
    }

    /// Live-dispatch entry point backed directly by the package mmap.
    ///
    /// This avoids calling [`LazyWeight::materialize`] on the hot path. On a
    /// not-fit model the same layer weights are paged every token; rebuilding an
    /// owned host tensor for each miss made CPU materialization dominate decode
    /// time.
    pub fn resident_mapped(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn MmapRegionSource,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        if let Some(hit) = self.get_hit(key) {
            return Ok(hit);
        }
        let len = weight.region_bytes_len();
        let mut staging = self
            .runtime
            .alloc_pinned(len)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("pinned alloc: {error}")))?;
        let materialize_start = std::time::Instant::now();
        fill_staging_from_regions(weight, source, &mut staging)?;
        add_duration(&GLOBAL_MATERIALIZE_NS, materialize_start.elapsed());
        if self.stable_vmm.is_some() {
            return self.admit_stable_staged(key, weight, len, staging);
        }
        let (page, _, _staging) = CudaWeightPage::upload_staged_async(
            &self.runtime,
            weight.dtype,
            weight.shape.clone(),
            len,
            staging,
        )?;
        self.admit(key, Arc::new(page), self.eviction_for(weight.boundary))
    }

    /// Live-dispatch entry point: return the device page for `key`, paging it in
    /// on a miss by materializing the weight's canonical (compressed) bytes and
    /// streaming them host→device, with LRU eviction under the VRAM budget. The
    /// materialized bytes are the exact resident backing, so the page is
    /// byte-identical to a stock upload.
    pub fn resident_materialized(
        &self,
        key: u64,
        weight: &LazyWeight,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        if let Some(hit) = self.get_hit(key) {
            return Ok(hit);
        }
        let materialize_start = std::time::Instant::now();
        let resident = weight.materialize()?;
        GLOBAL_MATERIALIZE_FALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
        add_duration(&GLOBAL_MATERIALIZE_NS, materialize_start.elapsed());
        if self.stable_vmm.is_some() {
            let mut staging =
                self.runtime
                    .alloc_pinned(resident.bytes().len())
                    .map_err(|error| {
                        WeightHandleError::DeviceBinding(format!("pinned alloc: {error}"))
                    })?;
            staging.as_mut_slice()[..resident.bytes().len()].copy_from_slice(resident.bytes());
            return self.admit_stable_staged(key, weight, resident.bytes().len(), staging);
        }
        let page = Arc::new(CudaWeightPage::upload(
            &self.runtime,
            resident.dtype,
            resident.shape.clone(),
            resident.bytes(),
        )?);
        self.admit(key, page, self.eviction_for(weight.boundary))
    }

    fn eviction_for(&self, boundary: LazyWeightBoundary) -> WeightEvictionPolicy {
        eviction_for_boundary(self.scan_resistant_dense, boundary)
    }

    fn admit_stable_staged(
        &self,
        key: u64,
        weight: &LazyWeight,
        len: usize,
        staging: PinnedStaging,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        let allocator = self
            .stable_vmm
            .as_ref()
            .expect("stable VMM caller checked")
            .clone();
        let mut inner = self.lock();
        if let Some(existing) = inner.pages.get(&key).cloned() {
            inner.record_hit(key);
            return Ok(existing);
        }
        let slot = match inner.stable_slots.get(&key).cloned() {
            Some(slot) => slot,
            None => {
                let slot = Arc::new(StableVmmSlot::new(
                    Arc::clone(&self.runtime),
                    allocator,
                    len,
                )?);
                inner.stable_slots.insert(key, Arc::clone(&slot));
                slot
            }
        };
        let accounting_bytes = slot.allocation_bytes as u64;
        let content_bytes = len as u64;
        let eviction = self.eviction_for(weight.boundary);
        let mut admit_after_copy = true;
        if !inner.policy.can_fit(accounting_bytes) {
            drop(inner);
            let sync_start = std::time::Instant::now();
            self.runtime.synchronize().map_err(|error| {
                WeightHandleError::DeviceBinding(format!("stream sync: {error}"))
            })?;
            add_duration(&GLOBAL_ADMIT_SYNC_NS, sync_start.elapsed());
            let mut inner_after_sync = self.lock();
            if let Some(existing) = inner_after_sync.pages.get(&key).cloned() {
                inner_after_sync.record_hit(key);
                return Ok(existing);
            }
            if eviction == WeightEvictionPolicy::StableResident {
                inner_after_sync.evict_to_fit(accounting_bytes, WeightEvictionPolicy::Lru);
                admit_after_copy = false;
            } else {
                inner_after_sync.evict_to_fit(accounting_bytes, eviction);
            }
            if !inner_after_sync.policy.can_fit(accounting_bytes) {
                return Err(WeightHandleError::DeviceBinding(format!(
                    "stable VMM weight page requires {accounting_bytes} bytes after granule \
                     rounding, but eviction left {} of {} bytes resident",
                    inner_after_sync.policy.resident_bytes, inner_after_sync.policy.budget
                )));
            }
            drop(inner_after_sync);
        } else {
            drop(inner);
        }
        slot.commit(len)?;
        let staged = &staging.as_slice()[..len];
        let copy_ms = unsafe {
            self.runtime
                .htod_async_elapsed_ms(staged, slot.device_ptr())
        }
        .map_err(|error| {
            slot.decommit();
            WeightHandleError::DeviceBinding(format!("measured H2D copy: {error}"))
        })?;
        GLOBAL_HTOD_NS.fetch_add((copy_ms * 1_000_000.0) as u64, Ordering::Relaxed);
        GLOBAL_HTOD_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        let page = Arc::new(CudaWeightPage {
            storage: CudaWeightPageStorage::StableVmm {
                slot: Arc::clone(&slot),
            },
            ptr: slot.device_ptr(),
            len,
            dtype: weight.dtype,
            shape: weight.shape.clone(),
        });
        if !admit_after_copy {
            let mut inner = self.lock();
            inner.record_bypassed_page_in();
            return Ok(page);
        }
        let mut inner = self.lock();
        inner.insert_page(key, Arc::clone(&page), accounting_bytes, content_bytes);
        Ok(page)
    }

    /// Look up `key`, marking it most-recently-used and counting a hit.
    fn get_hit(&self, key: u64) -> Option<Arc<CudaWeightPage>> {
        let mut inner = self.lock();
        if let Some(page) = inner.pages.get(&key).cloned() {
            inner.record_hit(key);
            Some(page)
        } else {
            None
        }
    }

    /// Drain the transfer stream so a just-uploaded page we are about to drop
    /// cannot have an in-flight async copy still reading its (freeing) VRAM or
    /// pinned staging. A no-op for synchronously-uploaded pages (idle copy stream).
    fn drain_copy_stream(&self) -> Result<(), WeightHandleError> {
        self.runtime.sync_copy_stream().map_err(|error| {
            WeightHandleError::DeviceBinding(format!("transfer stream sync: {error}"))
        })
    }

    /// Insert a freshly paged-in `page` under `key`, evicting LRU pages to fit the
    /// budget.
    ///
    /// The compute stream is drained **only when admitting must evict**, so no
    /// in-flight kernel still references an about-to-be-freed page's VRAM (the
    /// original WAR/reuse guarantee). A page-in that fits the budget frees nothing
    /// and therefore skips the sync, letting its async transfer overlap the
    /// current compute. This sync is one of the quantities reported in the
    /// offload counters because an unexpected host drain can erase prefetch
    /// overlap without changing correctness.
    fn admit(
        &self,
        key: u64,
        page: Arc<CudaWeightPage>,
        eviction: WeightEvictionPolicy,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        let bytes = page.accounting_len() as u64;
        let content_bytes = page.len() as u64;
        {
            let mut inner = self.lock();
            // A concurrent caller may have populated `key` while we paged in;
            // prefer the already-resident page and drop ours (its Drop frees the
            // VRAM + staging — drained first so it never races an in-flight copy).
            if let Some(existing) = inner.pages.get(&key).cloned() {
                inner.record_hit(key);
                drop(inner);
                self.drain_copy_stream()?;
                return Ok(existing);
            }
            // Fits without eviction: nothing is freed, so no consumer drain is
            // needed. Admit directly so the async transfer keeps overlapping.
            if inner.policy.can_fit(bytes) {
                inner.insert_page(key, Arc::clone(&page), bytes, content_bytes);
                return Ok(page);
            }
        }
        // Eviction required: drain in-flight consumers before freeing any page.
        // Weight offload and CUDA graph capture are mutually exclusive (the decode
        // session declines capture whenever offload is enabled), so this
        // synchronize is never capture-illegal.
        let sync_start = std::time::Instant::now();
        self.runtime
            .synchronize()
            .map_err(|error| WeightHandleError::DeviceBinding(format!("stream sync: {error}")))?;
        add_duration(&GLOBAL_ADMIT_SYNC_NS, sync_start.elapsed());
        let mut inner = self.lock();
        // Re-check after releasing the lock for the sync.
        if let Some(existing) = inner.pages.get(&key).cloned() {
            inner.record_hit(key);
            drop(inner);
            self.drain_copy_stream()?;
            return Ok(existing);
        }
        if eviction == WeightEvictionPolicy::StableResident && !inner.policy.can_fit(bytes) {
            inner.record_bypassed_page_in();
            return Ok(page);
        }
        inner.evict_to_fit(bytes, eviction);
        // Eviction is best effort: a page larger than the whole budget, or
        // pinned pages that cannot be evicted, both leave no room. Admitting
        // anyway is what this used to do, and it left the cache physically
        // holding more than it leased -- a total the governor reported as
        // correct while it was not.
        //
        // So ask for the difference first. Growing is a fresh claim and obeys
        // G4, so a refusal leaves the tier and the lease exactly as they were,
        // and the page-in fails instead of the accounting quietly going wrong.
        if let Some(over) = inner
            .policy
            .resident_bytes
            .saturating_add(bytes)
            .checked_sub(inner.policy.budget)
            .filter(|over| *over > 0)
        {
            match inner.lease.as_mut() {
                Some(lease) => {
                    lease.grow(over).map_err(|error| {
                        WeightHandleError::DeviceBinding(format!(
                            "the weight-residency cache needs {over} bytes beyond its \
                             {} byte budget for a {bytes} byte page, and eviction could not \
                             free them: {error}",
                            inner.policy.budget
                        ))
                    })?;
                    inner.policy.budget = inner.policy.budget.saturating_add(over);
                    GLOBAL_BUDGET_BYTES.store(inner.policy.budget, Ordering::Relaxed);
                }
                // No lease means no governor knows about this cache, so there is
                // nothing to ask and nothing whose total this would falsify.
                // Keep the previous behaviour rather than inventing a refusal
                // the operator never asked for.
                None => {
                    inner.policy.budget = inner.policy.budget.saturating_add(over);
                    GLOBAL_BUDGET_BYTES.store(inner.policy.budget, Ordering::Relaxed);
                }
            }
        }
        inner.insert_page(key, Arc::clone(&page), bytes, content_bytes);
        Ok(page)
    }

    /// Snapshot the cache's activity counters.
    pub fn stats(&self) -> CudaResidencyStats {
        let inner = self.lock();
        GLOBAL_BUDGET_BYTES.store(inner.policy.budget, Ordering::Relaxed);
        GLOBAL_PEAK_RESIDENT_BYTES.fetch_max(inner.policy.peak_resident_bytes, Ordering::Relaxed);
        CudaResidencyStats {
            budget_bytes: inner.policy.budget,
            resident_bytes: inner.policy.resident_bytes,
            peak_resident_bytes: inner.policy.peak_resident_bytes,
            resident_content_bytes: inner.policy.resident_content_bytes,
            peak_resident_content_bytes: inner.policy.peak_resident_content_bytes,
            pages_resident: inner.pages.len() as u64,
            page_ins: inner.policy.page_ins,
            hits: inner.policy.hits,
            evictions: inner.policy.evictions,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ResidencyInner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl std::fmt::Debug for CudaWeightResidency {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaWeightResidency")
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl ResidencyInner {
    /// Record a cache hit for `key`: mark it most-recently-used and bump the
    /// per-instance and process-global hit counters.
    fn record_hit(&mut self, key: u64) {
        self.policy.record_hit(key);
        GLOBAL_HITS.fetch_add(1, Ordering::Relaxed);
    }

    /// Insert a freshly paged-in `page` of `bytes` under `key`, updating the
    /// order, residency accounting, and the page-in counters.
    fn insert_page(&mut self, key: u64, page: Arc<CudaWeightPage>, bytes: u64, content_bytes: u64) {
        self.pages.insert(key, page);
        self.policy.insert_page(key, bytes, content_bytes);
        GLOBAL_PEAK_RESIDENT_BYTES.fetch_max(self.policy.peak_resident_bytes, Ordering::Relaxed);
        GLOBAL_PEAK_RESIDENT_CONTENT_BYTES
            .fetch_max(self.policy.peak_resident_content_bytes, Ordering::Relaxed);
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
    }

    /// Count a miss whose freshly paged-in allocation is returned directly to
    /// the caller instead of becoming part of the resident set.
    fn record_bypassed_page_in(&mut self) {
        self.policy.record_page_in();
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
    }

    /// Evict currently-unreferenced pages until admitting `incoming` bytes fits
    /// the budget (best effort; stops when nothing more is evictable).
    fn evict_to_fit(&mut self, incoming: u64, eviction: WeightEvictionPolicy) {
        let evicted = {
            let pages = &self.pages;
            self.policy.evict_to_fit(incoming, eviction, |key| {
                pages
                    .get(&key)
                    .is_some_and(|page| Arc::strong_count(page) == 1)
            })
        };
        for key in evicted {
            if self.pages.remove(&key).is_some() {
                GLOBAL_EVICTIONS.fetch_add(1, Ordering::Relaxed);
                // The page's Drop frees VRAM here when the cache was sole owner.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_policy_defaults_to_disabled() {
        let policy = DeviceOffloadPolicy::default();
        assert!(!policy.enabled);
        assert_eq!(policy.device_budget_bytes, None);
        // The struct default is inert; from-env supplies the offload runtime default.
        assert!(!policy.async_pagein);
        assert!(policy.scan_resistant_dense);
    }

    #[test]
    fn async_pagein_env_is_default_on_with_explicit_opt_out() {
        // Unset => async (default on), so weight prefetch can overlap decode.
        assert!(async_pagein_from_env_value(None));
        // Truthy spellings keep async enabled, case/whitespace-insensitive.
        assert!(async_pagein_from_env_value(Some("1")));
        assert!(async_pagein_from_env_value(Some("true")));
        assert!(async_pagein_from_env_value(Some("YES")));
        assert!(async_pagein_from_env_value(Some("  On ")));
        // Explicit falsey / anything else forces the old synchronous path.
        assert!(!async_pagein_from_env_value(Some("0")));
        assert!(!async_pagein_from_env_value(Some("false")));
        assert!(!async_pagein_from_env_value(Some("")));
        assert!(!async_pagein_from_env_value(Some("maybe")));
    }

    #[test]
    fn scan_resistant_env_defaults_on_with_lru_opt_out() {
        assert!(scan_resistant_from_env_value(None));
        assert!(scan_resistant_from_env_value(Some("1")));
        assert!(scan_resistant_from_env_value(Some("true")));
        assert!(scan_resistant_from_env_value(Some("YES")));
        assert!(scan_resistant_from_env_value(Some("  On ")));
        assert!(!scan_resistant_from_env_value(Some("0")));
        assert!(!scan_resistant_from_env_value(Some("false")));
        assert!(!scan_resistant_from_env_value(Some("NO")));
        assert!(!scan_resistant_from_env_value(Some("  off ")));
        assert!(scan_resistant_from_env_value(Some("")));
        assert!(scan_resistant_from_env_value(Some("maybe")));
    }

    #[test]
    fn stable_vmm_env_is_explicit_opt_in() {
        assert!(!truthy_env_value(None));
        assert!(truthy_env_value(Some("1")));
        assert!(truthy_env_value(Some("true")));
        assert!(truthy_env_value(Some("YES")));
        assert!(truthy_env_value(Some("  on ")));
        assert!(!truthy_env_value(Some("0")));
        assert!(!truthy_env_value(Some("false")));
    }

    #[test]
    fn budget_parsing_rejects_zero_and_garbage() {
        assert_eq!(parse_budget_bytes("1048576"), Some(1_048_576));
        assert_eq!(parse_budget_bytes("  4096 "), Some(4096));
        assert_eq!(parse_budget_bytes("0"), None);
        assert_eq!(parse_budget_bytes(""), None);
        assert_eq!(parse_budget_bytes("lots"), None);
        assert_eq!(parse_budget_bytes("-5"), None);
    }

    /// A locally chosen budget becomes a claim the rest of the system can see.
    ///
    /// This cache used to carry its own budget -- 4 GiB by default, or whatever
    /// `ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES` said -- reconciled with nothing.
    /// Grant the KV pool most of an 8 GiB card and let this default to 4 GiB and
    /// both are individually satisfied while the card is oversubscribed. Nobody
    /// finds out until an allocation fails somewhere unrelated.
    #[test]
    fn adopting_a_governed_budget_makes_the_claim_visible_to_other_holders() {
        use onnx_runtime_memory_governor::{
            HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole, Tier,
        };

        let Ok(runtime) = crate::runtime::CudaRuntime::new(0).map(std::sync::Arc::new) else {
            eprintln!("SKIPPED (no CUDA runtime): the governed weight-budget check did NOT run.");
            return;
        };
        let residency = CudaWeightResidency::new(runtime, 700);
        assert_eq!(
            residency.budget(),
            (700, false),
            "before adoption the budget answers to nobody"
        );

        let governor = LedgerGovernor::new(LeaseLedger::new(1000, 0, 0));
        let granted = residency
            .adopt_governed_budget(&governor, Tier::Device, HolderId::new(4))
            .expect("700 of 1000 is affordable");
        assert_eq!(granted, 700);
        assert_eq!(residency.budget(), (700, true));

        // The point: another holder now sees those bytes are spoken for.
        assert_eq!(governor.available(Tier::Device), 300);
        let refused = governor
            .reserve(Tier::Device, 700, MemoryRole::KvCache, HolderId::new(1))
            .expect_err("the weights already hold 700 of the 1000");
        assert!(matches!(
            refused,
            onnx_runtime_memory_governor::MemoryError::TierExhausted { .. }
        ));
    }

    /// Adopting twice does not charge twice.
    #[test]
    fn a_budget_already_governed_is_not_reserved_again() {
        use onnx_runtime_memory_governor::{
            HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, Tier,
        };

        let Ok(runtime) = crate::runtime::CudaRuntime::new(0).map(std::sync::Arc::new) else {
            eprintln!("SKIPPED (no CUDA runtime): the double-adoption check did NOT run.");
            return;
        };
        let residency = CudaWeightResidency::new(runtime, 400);
        let governor = LedgerGovernor::new(LeaseLedger::new(1000, 0, 0));

        residency
            .adopt_governed_budget(&governor, Tier::Device, HolderId::new(4))
            .expect("first adoption");
        residency
            .adopt_governed_budget(&governor, Tier::Device, HolderId::new(4))
            .expect("second adoption is a no-op, not a second charge");

        assert_eq!(
            governor.available(Tier::Device),
            600,
            "the same budget was charged twice"
        );
    }

    fn drive_repeated_scan(
        policy: &mut WeightResidencyPolicy,
        eviction: WeightEvictionPolicy,
        working_set: u64,
        cycles: u64,
    ) -> u64 {
        let hits_before = policy.hits;
        for _ in 0..cycles {
            for key in 0..working_set {
                let _ = policy.access(key, 1, eviction);
            }
        }
        policy.hits - hits_before
    }

    fn measured_scan_hits(capacity: u64, eviction: WeightEvictionPolicy) -> u64 {
        const WORKING_SET: u64 = 10;
        const MEASURED_CYCLES: u64 = 5;
        let mut policy = WeightResidencyPolicy::new(capacity);

        let _ = drive_repeated_scan(&mut policy, eviction, WORKING_SET, 1);
        drive_repeated_scan(&mut policy, eviction, WORKING_SET, MEASURED_CYCLES)
    }

    #[test]
    fn lru_has_zero_steady_state_hits_across_cyclic_scan_capacity_sweep() {
        const WORKING_SET: u64 = 10;
        const MEASURED_CYCLES: u64 = 5;
        let accesses = WORKING_SET * MEASURED_CYCLES;

        for capacity in [2, 4, 6, 8] {
            let hits = measured_scan_hits(capacity, WeightEvictionPolicy::Lru);
            assert_eq!(
                hits,
                0,
                "LRU should be pessimal for a clean cycle larger than capacity; \
                 B={capacity}/{WORKING_SET}, hit_rate={:.1}%",
                (hits as f64 / accesses as f64) * 100.0
            );
        }
    }

    #[test]
    fn stable_subset_recovers_capacity_fraction_across_cyclic_scan_sweep() {
        const WORKING_SET: u64 = 10;
        const MEASURED_CYCLES: u64 = 5;
        let accesses = WORKING_SET * MEASURED_CYCLES;

        for capacity in [2, 4, 6, 8] {
            let hits = measured_scan_hits(capacity, WeightEvictionPolicy::StableResident);
            assert_eq!(hits, capacity * MEASURED_CYCLES);
            assert_eq!(
                (hits as f64) / (accesses as f64),
                (capacity as f64) / (WORKING_SET as f64),
                "stable-subset residency should recover B/W for B={capacity}/{WORKING_SET}"
            );
        }
    }

    #[test]
    fn stable_subset_matches_lru_when_whole_scan_fits() {
        const WORKING_SET: u64 = 10;
        const MEASURED_CYCLES: u64 = 5;

        for capacity in [10, 12] {
            let lru_hits = measured_scan_hits(capacity, WeightEvictionPolicy::Lru);
            let stable_hits = measured_scan_hits(capacity, WeightEvictionPolicy::StableResident);

            assert_eq!(lru_hits, WORKING_SET * MEASURED_CYCLES);
            assert_eq!(stable_hits, lru_hits);
        }
    }

    #[test]
    fn scan_resistant_mode_leaves_moe_skew_on_lru() {
        let selected = eviction_for_boundary(true, LazyWeightBoundary::QMoe);
        assert_eq!(selected, WeightEvictionPolicy::Lru);

        const CYCLES: u64 = 20;
        let skewed = [0, 0, 0, 0, 1, 0, 1, 2, 0, 3, 0, 1];
        let mut baseline = WeightResidencyPolicy::new(3);
        let mut moe = WeightResidencyPolicy::new(3);
        for _ in 0..CYCLES {
            for &key in &skewed {
                let _ = baseline.access(key, 1, WeightEvictionPolicy::Lru);
                let _ = moe.access(key, 1, selected);
            }
        }

        assert_eq!(moe.hits, baseline.hits);
        assert_eq!(moe.page_ins, baseline.page_ins);
        assert_eq!(moe.evictions, baseline.evictions);
        assert_eq!(moe.hits, 178);
        assert_eq!(moe.page_ins, 62);
        assert_eq!(moe.evictions, 59);
    }

    #[test]
    fn stable_vmm_weight_slots_keep_their_virtual_address_across_eviction() {
        use onnx_runtime_ep_api::{ExternalMmapRegion, LazyWeightBoundary};

        struct Source {
            a: Vec<u8>,
            b: Vec<u8>,
        }

        impl MmapRegionSource for Source {
            fn region_bytes(
                &self,
                region: &ExternalMmapRegion,
            ) -> Result<&[u8], WeightHandleError> {
                let bytes = match region.mapping_id {
                    1 => &self.a,
                    2 => &self.b,
                    other => {
                        return Err(WeightHandleError::DeviceBinding(format!(
                            "unexpected mapping id {other}"
                        )));
                    }
                };
                Ok(&bytes[region.offset..region.offset + region.len])
            }
        }

        let Ok(runtime) = crate::runtime::CudaRuntime::new(0).map(std::sync::Arc::new) else {
            eprintln!("SKIPPED (no CUDA runtime): the stable VMM weight-slot test did NOT run.");
            return;
        };
        crate::vmm_allocator::reset_global_vmm_stats();
        let len = 4096;
        let source = Source {
            a: vec![0x11; len],
            b: vec![0x22; len],
        };
        let make_weight = |mapping_id| {
            LazyWeight::new(
                LazyWeightBoundary::QMoe,
                DataType::Uint8,
                vec![len],
                vec![ExternalMmapRegion {
                    mapping_id,
                    offset: 0,
                    len,
                }],
                {
                    let bytes = if mapping_id == 1 {
                        source.a.clone()
                    } else {
                        source.b.clone()
                    };
                    move || {
                        onnx_runtime_ep_api::ResidentWeight::new(
                            DataType::Uint8,
                            vec![len],
                            bytes.clone(),
                        )
                    }
                },
            )
            .expect("valid lazy weight")
        };
        let residency = CudaWeightResidency::new(Arc::clone(&runtime), 2 * 1024 * 1024)
            .with_stable_vmm()
            .expect("stable VMM residency");
        let weight_a = make_weight(1);
        let weight_b = make_weight(2);
        let raw_counts_before = runtime.allocation_counts();

        let first = residency
            .resident_mapped(1, &weight_a, &source)
            .expect("first page-in");
        let first_ptr = first.device_ptr();
        let mut readback = vec![0u8; len];
        unsafe { runtime.dtoh(&mut readback, first_ptr as CUdeviceptr) }.expect("read first");
        assert!(readback.iter().all(|&byte| byte == 0x11));
        drop(first);

        let second = residency
            .resident_mapped(2, &weight_b, &source)
            .expect("second page-in evicts first");
        assert_ne!(
            second.device_ptr(),
            first_ptr,
            "different keys get different VAs"
        );
        drop(second);

        let remapped = residency
            .resident_mapped(1, &weight_a, &source)
            .expect("first key pages back into its stable VA");
        assert_eq!(
            remapped.device_ptr(),
            first_ptr,
            "the key's captured kernel parameter would remain stable"
        );
        readback.fill(0);
        unsafe { runtime.dtoh(&mut readback, remapped.device_ptr() as CUdeviceptr) }
            .expect("read remapped");
        assert!(readback.iter().all(|&byte| byte == 0x11));

        let stats = residency.stats();
        assert_eq!(stats.page_ins, 3);
        assert_eq!(stats.hits, 0);
        assert!(stats.evictions >= 2);
        let vmm = crate::vmm_allocator::global_vmm_stats();
        assert!(vmm.commits >= 3, "page-ins should map VMM granules");
        assert!(vmm.releases >= 2, "evictions should unmap VMM granules");
        assert_eq!(
            runtime.allocation_counts(),
            raw_counts_before,
            "stable VMM weight page-ins should not churn cuMemAlloc/cuMemFree"
        );
    }
}
