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
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    LazyDeviceWeightBinder, LazyWeight, LazyWeightBoundary, MmapRegionSource, WeightHandleError,
};
use onnx_runtime_ir::DataType;
use onnx_runtime_memory_governor::{DeviceAllocator, Tier};

use crate::pinned_pool::PinnedStagingPool;
use crate::runtime::{CopyCompleted, CudaRuntime, PinnedStaging, raw_ptr};

/// Alignment for stable-VA weight slots (issue #716). The VMM arena rounds
/// commits to the 2 MiB device granule (#776) regardless, so this only governs
/// the reserved VA start; 256 B matches the value used for the pre-#716
/// throwaway carves so slot addresses stay comparably aligned.
const WEIGHT_SLOT_ALIGN: usize = 256;

/// Process-global weight-offload activity counters. These may be reset between
/// benchmark measurement windows while caches remain alive.
static GLOBAL_PAGE_INS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_HITS: AtomicU64 = AtomicU64::new(0);
/// Bytes served from residency without an H2D copy. Pairs with
/// [`GLOBAL_HTOD_BYTES`] to give a byte-weighted hit rate; see
/// [`ResidencyInner::record_hit`] for why the count-based rate misleads.
static GLOBAL_HIT_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_EVICTIONS: AtomicU64 = AtomicU64::new(0);
// Page-ins under scan-resistant admission that could not be admitted to the
// resident set and were handed back to the caller transiently (issue #716:
// a non-zero value since capture arm means the resident set is NOT stable, so
// whole-step CUDA graph capture must stay declined for correctness).
static GLOBAL_BYPASSED_PAGE_INS: AtomicU64 = AtomicU64::new(0);
// Time spent filling the host staging buffer from mmap regions. This is a
// host-blocking CPU memcpy span and contains no CUDA synchronization.
static GLOBAL_MATERIALIZE_NS: AtomicU64 = AtomicU64::new(0);
// CUDA-event elapsed time for H2D DMA: start event before cuMemcpyHtoDAsync,
// end event after it, then host-block on the end event to read elapsed time.
static GLOBAL_HTOD_NS: AtomicU64 = AtomicU64::new(0);
// Host-blocking compute-stream synchronize taken before evicting pages whose
// VRAM might still be referenced by earlier kernels.
static GLOBAL_ADMIT_SYNC_NS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_STAGING_FILL_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_STAGING_FILL_REGIONS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_STAGING_FILL_CALLS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_MATERIALIZE_FALLBACK_CALLS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_HTOD_BYTES: AtomicU64 = AtomicU64::new(0);
// Host-blocking cuMemAlloc/cuMemFree spans for paged weight buffers.
static GLOBAL_VRAM_ALLOC_NS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_VRAM_FREE_NS: AtomicU64 = AtomicU64::new(0);
// Process-lifetime high-water gauge. Resetting activity counters must not write
// it: a concurrent page-in could otherwise be overwritten with a stale value.
static GLOBAL_PEAK_RESIDENT_BYTES: AtomicU64 = AtomicU64::new(0);
// Process-global live-state gauges. Unlike the activity counters above, these
// are changed only when a cache, page, or mapping changes state. Resetting a
// benchmark window must preserve them because hits do not rewrite residency.
static GLOBAL_BUDGET_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_CONTENT_RESIDENT_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_WEIGHT_MAPPED_BYTES: AtomicU64 = AtomicU64::new(0);

fn add_duration(counter: &AtomicU64, elapsed: Duration) {
    let nanos = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
    counter.fetch_add(nanos, Ordering::Relaxed);
}

fn replace_global_budget(old: u64, new: u64) {
    if new >= old {
        GLOBAL_BUDGET_BYTES.fetch_add(new - old, Ordering::Relaxed);
    } else {
        let decrease = old - new;
        let _ = GLOBAL_BUDGET_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(decrease))
        });
    }
}

fn committed_admission_fits(
    required_mapped: u64,
    zone_available: u64,
    required_owned: u64,
    global_available: u64,
) -> bool {
    required_mapped <= zone_available && required_owned <= global_available
}

fn eviction_made_committed_progress(
    before_owned: u64,
    after_owned: u64,
    before_required_owned: u64,
    after_required_owned: u64,
    before_required_mapped: u64,
    after_required_mapped: u64,
) -> bool {
    after_owned < before_owned
        || after_required_owned < before_required_owned
        || after_required_mapped < before_required_mapped
}

/// Snapshot of the process-global weight-offload counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalOffloadStats {
    pub page_ins: u64,
    pub hits: u64,
    /// Bytes served from residency (no H2D copy). With [`Self::htod_bytes`] this
    /// gives the **byte-weighted** hit rate, which is what streaming cost
    /// actually tracks — see [`Self::byte_hit_rate`].
    pub hit_bytes: u64,
    pub evictions: u64,
    pub bypassed_page_ins: u64,
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
    pub content_resident_bytes: u64,
    pub physical_owned_bytes: u64,
    pub mapped_physical_bytes: u64,
    /// Real `cuMemHostAlloc` calls issued by the pinned staging pool (issue
    /// #837). A page-in that reuses a pooled buffer does not increment this, so
    /// on the not-fit streaming path this stays far below `page_ins`.
    pub pinned_alloc_calls: u64,
    /// Page-ins whose pinned staging buffer was served from the pool free-list.
    pub pinned_reuses: u64,
}

impl GlobalOffloadStats {
    /// Fraction of requested weight **bytes** served from residency.
    ///
    /// Prefer this over `hits / (hits + page_ins)` when judging residency policy:
    /// the count-based rate weights a 10 KiB norm the same as an 11 MiB
    /// projection, so it can improve while the bytes actually streamed get
    /// worse. Measured on qwen14b-zp, raising the weight budget moved the count
    /// rate 57.09% -> 81.31% while the byte gap to the streaming floor widened
    /// from 1.78x to 2.30x (#857, #837 item 3).
    ///
    /// `None` when no weight bytes were requested in the window.
    #[must_use]
    pub fn byte_hit_rate(&self) -> Option<f64> {
        let requested = self.hit_bytes.checked_add(self.htod_bytes)?;
        (requested > 0).then(|| self.hit_bytes as f64 / requested as f64)
    }
}

/// Read the process-global weight-offload counters.
pub fn global_offload_stats() -> GlobalOffloadStats {
    GlobalOffloadStats {
        page_ins: GLOBAL_PAGE_INS.load(Ordering::Relaxed),
        hits: GLOBAL_HITS.load(Ordering::Relaxed),
        hit_bytes: GLOBAL_HIT_BYTES.load(Ordering::Relaxed),
        evictions: GLOBAL_EVICTIONS.load(Ordering::Relaxed),
        bypassed_page_ins: GLOBAL_BYPASSED_PAGE_INS.load(Ordering::Relaxed),
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
        content_resident_bytes: GLOBAL_CONTENT_RESIDENT_BYTES.load(Ordering::Relaxed),
        physical_owned_bytes: crate::virtual_memory::total_physical_pool_owned_bytes(),
        mapped_physical_bytes: GLOBAL_WEIGHT_MAPPED_BYTES.load(Ordering::Relaxed),
        pinned_alloc_calls: crate::pinned_pool::global_pinned_alloc_calls(),
        pinned_reuses: crate::pinned_pool::global_pinned_reuses(),
    }
}

/// Reset cumulative weight-offload activity for a new measurement window.
///
/// Live gauges (`budget_bytes`, `content_resident_bytes`,
/// `mapped_physical_bytes`, authority-owned physical bytes, and the
/// process-lifetime `peak_resident_bytes`) are preserved. Caches and mappings
/// may outlive a benchmark warmup reset, and cache hits do not rewrite those
/// values. Benchmarks needing an interval peak must track it locally.
pub fn reset_global_offload_stats() {
    GLOBAL_PAGE_INS.store(0, Ordering::Relaxed);
    GLOBAL_HITS.store(0, Ordering::Relaxed);
    GLOBAL_HIT_BYTES.store(0, Ordering::Relaxed);
    GLOBAL_EVICTIONS.store(0, Ordering::Relaxed);
    GLOBAL_BYPASSED_PAGE_INS.store(0, Ordering::Relaxed);
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
    crate::pinned_pool::reset_pinned_pool_counters();
}

/// Environment switch that enables the CUDA device residency cache. Reuses the
/// same knob as the CPU host-cache offload path (`onnx_runtime_ep_cpu`) so a
/// single `ONNX_GENAI_WEIGHT_OFFLOAD=1` turns offload on for whichever EP runs.
pub const WEIGHT_OFFLOAD_ENV: &str = onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV;

/// VRAM budget (bytes) for the device residency cache. When unset the residency
/// manager is constructed with a caller-chosen default.
pub const WEIGHT_OFFLOAD_DEVICE_BYTES_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES";

/// Sub-knob (default ON / opt-OUT) selecting the asynchronous, fence-ordered
/// residency page-in over the synchronous `cuMemcpyHtoD`. **Unset** uses async.
/// When set, only `1`/`true`/`yes`/`on` (case/whitespace-insensitive) keep async
/// enabled; **every other value — including an empty string, `2`, or `enabled` —
/// selects the synchronous path.** That is deliberate and pinned by
/// `async_pagein_env_parsing`, but it is strictly stronger than "a falsy value
/// disables it", so a typo here silently selects the slow path rather than
/// erroring. Note the asymmetry with [`WEIGHT_OFFLOAD_SCAN_RESISTANT_ENV`],
/// which disables only on an explicitly falsy value and ignores anything else.
///
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

/// Parse [`WEIGHT_OFFLOAD_ASYNC_PAGEIN_ENV`]. Async page-in is **default-on**:
/// unset (`None`) enables it. When a value *is* present, this is opt-**in**, not
/// opt-out — only `1`/`true`/`yes`/`on` keep async enabled, and every other
/// value (including `""` and `"maybe"`, both pinned by
/// `async_pagein_env_parsing`) selects the synchronous path. Deliberate, but
/// note it differs from [`scan_resistant_from_env_value`], which disables only
/// on an explicitly falsy value.
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

/// Whether/how the CUDA EP should page offloaded weights into a bounded VRAM
/// residency cache. Disabled by default so the resident fast path is untouched
/// and byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceOffloadPolicy {
    pub enabled: bool,
    /// An explicit device byte limit selected authority-managed VMM allocation
    /// instead of the WDDM-spill-compatible allocator.
    pub managed_no_spill: bool,
    /// The explicit authority ceiling that selected managed no-spill mode.
    pub managed_limit_bytes: Option<u64>,
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
}

impl Default for DeviceOffloadPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            managed_no_spill: false,
            managed_limit_bytes: None,
            device_budget_bytes: None,
            async_pagein: false,
            scan_resistant_dense: true,
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
        // Async page-in defaults ON when the variable is unset. When it IS set,
        // parsing is opt-in: anything other than 1/true/yes/on restores the old
        // synchronous demand-copy path, so a typo here costs performance
        // silently rather than erroring.
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
        Self {
            enabled,
            managed_no_spill: false,
            managed_limit_bytes: None,
            device_budget_bytes,
            async_pagein,
            scan_resistant_dense,
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
    /// Configured budget in bytes.
    ///
    /// With production VMM pooling, admission uses incremental physical bytes
    /// and the shared governor's headroom. Without VMM, the allocator cannot
    /// truthfully report physical ownership, so this retains content-byte
    /// semantics.
    pub budget_bytes: u64,
    /// Bytes currently resident across all cached pages.
    pub resident_bytes: u64,
    /// High-water mark of `resident_bytes`.
    pub peak_resident_bytes: u64,
    /// Number of pages currently resident.
    pub pages_resident: u64,
    /// H2D page-ins performed (cache misses that allocated + copied a page).
    pub page_ins: u64,
    /// Cache hits that reused an already-resident page (no H2D copy).
    pub hits: u64,
    /// LRU evictions that freed a page's VRAM.
    pub evictions: u64,
    /// Authority-owned physical bytes for the shared VMM pool.
    ///
    /// This is distinct from `resident_bytes`: content can share a granule,
    /// and an unmapped retained handle remains physically owned.
    pub physical_owned_bytes: u64,
    /// Physical granules currently mapped and attributed to this weight zone.
    pub mapped_physical_bytes: u64,
    /// Admission passes that stopped because eviction made no physical or
    /// reusable-pool progress.
    pub admission_no_progress: u64,
}

/// A live VRAM residency page for one offloaded weight tensor.
///
/// Owns a single device allocation holding the tensor's canonical compressed
/// bytes and frees it exactly once on drop. The address is a CUDA device
/// pointer — never dereferenced on the host — exposed through [`Self::device_ptr`]
/// for a consuming kernel's `TensorView`.
pub struct CudaWeightPage {
    runtime: Arc<CudaRuntime>,
    allocation: WeightAllocation,
    ptr: CUdeviceptr,
    len: usize,
    dtype: DataType,
    shape: Vec<usize>,
}

enum WeightAllocation {
    Runtime,
    Retired,
    Vmm {
        allocator: Arc<crate::vmm_allocator::CudaVmmAllocator>,
        allowance: onnx_runtime_memory_governor::MappedAllowance,
        /// When `true`, this page occupies a per-key **stable virtual address
        /// slot** (issue #716). Its Drop unmaps the physical granules but KEEPS
        /// the reserved VA live so the next page-in of the same key reuses the
        /// identical device pointer a captured CUDA graph baked into its nodes.
        /// The VA itself is reclaimed by the arena reservation's Drop at model
        /// unload. When `false`, this is a transient (bypassed) page on its own
        /// throwaway VA, freed outright on Drop — it is never baked into a
        /// captured graph, so no stable address is required.
        stable_slot: bool,
    },
}

/// A reserved-once virtual address slot backing a paged weight `key` (issue
/// #716). The physical granules under `va` are mapped on page-in and unmapped
/// on eviction, but the `va` itself persists for the residency's lifetime so a
/// captured graph that baked this pointer keeps reading the current physical
/// mapping across repeated page-ins.
#[derive(Clone, Copy, Debug)]
struct StableWeightSlot {
    va: CUdeviceptr,
    len: usize,
}

impl CudaWeightPage {
    fn release_allocation(&mut self, synchronize_streams: bool) {
        let allocation = std::mem::replace(&mut self.allocation, WeightAllocation::Retired);
        match allocation {
            WeightAllocation::Runtime => {
                let _ = unsafe { self.runtime.free_raw(self.ptr) };
            }
            WeightAllocation::Retired => {}
            WeightAllocation::Vmm {
                allocator,
                allowance,
                stable_slot,
            } => {
                // VMM unmap does not wait for users of the VA. Normal Drop
                // drains both streams; the eviction batch may do that once
                // up front and retire several pages without repeating it.
                if synchronize_streams
                    && (self.runtime.synchronize().is_err()
                        || self.runtime.copy_stream().synchronize().is_err())
                {
                    self.allocation = WeightAllocation::Vmm {
                        allocator,
                        allowance,
                        stable_slot,
                    };
                    return;
                }
                if let Some(ptr) = NonNull::new(self.ptr as *mut u8) {
                    // Stable slots retain VA for graph-baked pointers. Never
                    // assert here: this remains reachable from Drop.
                    let unmapped = if stable_slot {
                        allocator
                            .decommit_allocation_range(
                                ptr,
                                self.len,
                                WEIGHT_SLOT_ALIGN,
                                0,
                                self.len,
                            )
                            .unwrap_or(0)
                    } else {
                        allocator.deallocate_span(ptr)
                    };
                    allowance.unmap(unmapped);
                    let _ = GLOBAL_WEIGHT_MAPPED_BYTES.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |current| Some(current.saturating_sub(unmapped)),
                    );
                }
            }
        }
    }

    fn retire_after_stream_sync(&mut self) {
        let free_start = std::time::Instant::now();
        self.release_allocation(false);
        add_duration(&GLOBAL_VRAM_FREE_NS, free_start.elapsed());
    }

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
            runtime: Arc::clone(runtime),
            allocation: WeightAllocation::Runtime,
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
            runtime: Arc::clone(runtime),
            allocation: WeightAllocation::Runtime,
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
    ) -> Result<(Self, u64, PinnedStaging, CopyCompleted), WeightHandleError> {
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
            runtime: Arc::clone(runtime),
            allocation: WeightAllocation::Runtime,
            ptr,
            len,
            dtype,
            shape,
        };
        let staged = &staging.as_slice()[..len];
        let (copy_ms, completed) =
            unsafe { runtime.htod_async_elapsed_ms(staged, ptr) }.map_err(|error| {
                WeightHandleError::DeviceBinding(format!("measured H2D copy: {error}"))
            })?;
        GLOBAL_HTOD_NS.fetch_add((copy_ms * 1_000_000.0) as u64, Ordering::Relaxed);
        GLOBAL_HTOD_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        Ok((page, 0, staging, completed))
    }

    /// Opaque device pointer to the paged bytes, for a kernel `TensorView`.
    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        raw_ptr(self.ptr)
    }

    /// Number of canonical bytes resident in this VRAM page.
    pub fn len(&self) -> usize {
        self.len
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

impl Drop for CudaWeightPage {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from this runtime's `alloc_raw` in `bind_block_quantized_moe`
        // and is freed exactly once here; no alias to it escapes `CudaWeightPage`.
        let free_start = std::time::Instant::now();
        self.release_allocation(true);
        add_duration(&GLOBAL_VRAM_FREE_NS, free_start.elapsed());
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
            runtime: Arc::clone(&self.runtime),
            allocation: WeightAllocation::Runtime,
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
/// Production VMM pooling changes the admission unit, not the cache-efficiency
/// unit: the governor admits incremental authority-owned granules while hit
/// rate and resident-content metrics continue to count canonical weight bytes.
/// The non-VMM path intentionally retains content-byte admission until its
/// allocator can report truthful physical ownership.
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
    physical: OnceLock<PhysicalAdmission>,
    /// Reused pinned host staging buffers for weight page-ins. Shared so every
    /// page-in draws from the same bounded free-list instead of page-locking a
    /// fresh buffer per miss (issue #837). See [`crate::pinned_pool`] for the
    /// fence-safety argument.
    staging_pool: Arc<PinnedStagingPool>,
    inner: Mutex<ResidencyInner>,
}

struct PhysicalAdmission {
    allocator: Arc<crate::vmm_allocator::CudaVmmAllocator>,
    governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
}

struct ResidencyInner {
    policy: WeightResidencyPolicy,
    pages: HashMap<u64, Arc<CudaWeightPage>>,
    mapped_allowance: Option<onnx_runtime_memory_governor::MappedAllowance>,
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
    admission_no_progress: u64,
    /// Per-key **stable virtual address slots** (issue #716). A key's slot is
    /// reserved once (VA only, no physical bytes) on its first retained page-in
    /// and reused for every subsequent page-in of that key, so the device
    /// pointer a captured CUDA graph baked stays valid across evict→repage
    /// cycles. Only retained (resident-set) keys get a slot; transient bypass
    /// page-ins keep their own throwaway VA and never appear here.
    slots: HashMap<u64, StableWeightSlot>,
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
    page_ins: u64,
    hits: u64,
    evictions: u64,
    /// LRU order: front = least-recently-used, back = most-recently-used.
    order: Vec<u64>,
    bytes_by_key: HashMap<u64, u64>,
}

impl WeightResidencyPolicy {
    fn new(budget: u64) -> Self {
        Self {
            budget,
            resident_bytes: 0,
            peak_resident_bytes: 0,
            page_ins: 0,
            hits: 0,
            evictions: 0,
            order: Vec::new(),
            bytes_by_key: HashMap::new(),
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
        self.insert_page(key, bytes);
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

    fn insert_page(&mut self, key: u64, bytes: u64) {
        self.bytes_by_key.insert(key, bytes);
        self.order.push(key);
        self.resident_bytes += bytes;
        self.peak_resident_bytes = self.peak_resident_bytes.max(self.resident_bytes);
        self.record_page_in();
    }

    fn remove_page(&mut self, key: u64) -> Option<u64> {
        if let Some(position) = self.order.iter().position(|&candidate| candidate == key) {
            self.order.remove(position);
        }
        let bytes = self.bytes_by_key.remove(&key)?;
        self.resident_bytes = self.resident_bytes.saturating_sub(bytes);
        self.evictions = self.evictions.saturating_add(1);
        Some(bytes)
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
        replace_global_budget(0, budget_bytes);
        Self {
            runtime: Arc::clone(&runtime),
            scan_resistant_dense: false,
            physical: OnceLock::new(),
            staging_pool: PinnedStagingPool::new(Arc::clone(&runtime)),
            inner: Mutex::new(ResidencyInner {
                policy: WeightResidencyPolicy::new(budget_bytes),
                lease: None,
                pages: HashMap::new(),
                mapped_allowance: None,
                admission_no_progress: 0,
                slots: HashMap::new(),
            }),
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
        replace_global_budget(0, lease.bytes());
        Ok(Self {
            runtime: Arc::clone(&runtime),
            scan_resistant_dense: false,
            physical: OnceLock::new(),
            staging_pool: PinnedStagingPool::new(Arc::clone(&runtime)),
            inner: Mutex::new(ResidencyInner {
                policy: WeightResidencyPolicy::new(lease.bytes()),
                lease: Some(lease),
                pages: HashMap::new(),
                mapped_allowance: None,
                admission_no_progress: 0,
                slots: HashMap::new(),
            }),
        })
    }

    /// Bytes this cache is entitled to, and whether that came from a governor.
    ///
    /// `false` means the budget was chosen locally and nothing reconciles it
    /// with any other claim on the same device.
    pub fn budget(&self) -> (u64, bool) {
        let inner = self.inner.lock().expect("residency lock poisoned");
        (
            inner.policy.budget,
            inner.lease.is_some() || inner.mapped_allowance.is_some(),
        )
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
        if inner.lease.is_some() || inner.mapped_allowance.is_some() {
            return Ok(inner.policy.budget);
        }
        let old = inner.policy.budget;
        inner.policy.budget = budget_bytes;
        replace_global_budget(old, budget_bytes);
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
        if let Some(physical) = self.physical.get() {
            if physical.governor.authority_id() != governor.authority_id() {
                return Err(onnx_runtime_memory_governor::MemoryError::InvalidRequest {
                    tier: tier.name(),
                    requested: 0,
                    reason: "VMM weight residency was built with a different physical-memory authority",
                });
            }
            let mut inner = self.lock();
            if inner.mapped_allowance.is_none() {
                inner.mapped_allowance = Some(governor.reserve_mapped_allowance(
                    tier,
                    inner.policy.budget,
                    onnx_runtime_memory_governor::MemoryRole::Weights,
                    holder,
                )?);
            }
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
        let old = inner.policy.budget;
        inner.policy.budget = granted;
        inner.lease = Some(lease);
        replace_global_budget(old, granted);
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

    /// Use the production VMM arena and its existing physical-memory authority
    /// for weight pages. The configured cache budget remains an observability
    /// value; admission is governed by incremental authority-owned bytes, not
    /// by a second private lease.
    pub fn with_vmm_admission(
        self,
        allocator: Arc<crate::vmm_allocator::CudaVmmAllocator>,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    ) -> Result<Self, WeightHandleError> {
        if self.lock().lease.is_some() {
            return Err(WeightHandleError::DeviceBinding(
                "cannot install VMM physical admission on a cache that already holds a \
                 content-byte governor lease"
                    .into(),
            ));
        }
        if allocator.physical_pool_authority() != Some(governor.authority_id()) {
            return Err(WeightHandleError::DeviceBinding(
                "VMM weight residency and its governor must share one physical-memory authority"
                    .into(),
            ));
        }
        let _ = self.physical.set(PhysicalAdmission {
            allocator,
            governor,
        });
        Ok(self)
    }

    pub(crate) fn install_vmm_admission(
        &self,
        allocator: Arc<crate::vmm_allocator::CudaVmmAllocator>,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    ) -> Result<(), WeightHandleError> {
        if self.lock().lease.is_some() {
            return Err(WeightHandleError::DeviceBinding(
                "cannot install VMM physical admission on a cache that already holds a \
                 content-byte governor lease"
                    .into(),
            ));
        }
        if allocator.physical_pool_authority() != Some(governor.authority_id()) {
            return Err(WeightHandleError::DeviceBinding(
                "VMM weight residency and its governor must share one physical-memory authority"
                    .into(),
            ));
        }
        self.physical
            .set(PhysicalAdmission {
                allocator,
                governor,
            })
            .map_err(|_| {
                WeightHandleError::DeviceBinding(
                    "VMM weight residency admission was installed more than once".into(),
                )
            })
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
        if self.physical.get().is_some() {
            let resident = weight.materialize()?;
            GLOBAL_MATERIALIZE_FALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
            let bytes = resident.bytes().to_vec();
            return self.resident_vmm_with(
                key,
                resident.dtype,
                resident.shape.clone(),
                bytes.len(),
                self.eviction_for(weight.boundary),
                move |runtime, ptr| {
                    unsafe { runtime.htod(&bytes, ptr) }.map_err(|error| {
                        WeightHandleError::DeviceBinding(format!("H2D copy: {error}"))
                    })
                },
            );
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
        // Draw a reusable pinned staging buffer from the bounded pool instead of
        // page-locking a fresh `cuMemHostAlloc` per page-in (issue #837). The
        // buffer returns to the pool only after the (host-blocking) H2D copy
        // below completes — see `pinned_pool` for the fence-safety argument.
        let mut staging = self
            .staging_pool
            .acquire(len)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("pinned alloc: {error}")))?;
        let materialize_start = std::time::Instant::now();
        fill_staging_from_regions(weight, source, staging.staging_mut())?;
        add_duration(&GLOBAL_MATERIALIZE_NS, materialize_start.elapsed());
        if self.physical.get().is_some() {
            return self.resident_vmm_with(
                key,
                weight.dtype,
                weight.shape.clone(),
                len,
                self.eviction_for(weight.boundary),
                // `staging` (a `PooledStaging`) is moved into the fill closure.
                // `htod_async_elapsed_ms` host-synchronizes the copy before it
                // returns and yields a `CopyCompleted` witness; `retire` consumes
                // that witness to return the buffer to the pool, so reuse is
                // structurally gated on the copy having completed.
                move |runtime, ptr| {
                    let staged = &staging.as_slice()[..len];
                    let (copy_ms, completed) =
                        unsafe { runtime.htod_async_elapsed_ms(staged, ptr) }.map_err(|error| {
                            WeightHandleError::DeviceBinding(format!("measured H2D copy: {error}"))
                        })?;
                    GLOBAL_HTOD_NS.fetch_add((copy_ms * 1_000_000.0) as u64, Ordering::Relaxed);
                    GLOBAL_HTOD_BYTES.fetch_add(len as u64, Ordering::Relaxed);
                    staging.retire(completed);
                    Ok(())
                },
            );
        }
        // Non-VMM branch: `upload_staged_async` consumes the buffer, performs a
        // host-blocking copy, and hands it back with a `CopyCompleted` witness.
        // Return it to the pool with that witness (copy complete), then admit.
        let raw_staging = staging.into_inner();
        let (page, _, raw_staging, completed) = CudaWeightPage::upload_staged_async(
            &self.runtime,
            weight.dtype,
            weight.shape.clone(),
            len,
            raw_staging,
        )?;
        self.staging_pool.release(raw_staging, completed);
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
        if self.physical.get().is_some() {
            let bytes = resident.bytes().to_vec();
            return self.resident_vmm_with(
                key,
                resident.dtype,
                resident.shape.clone(),
                bytes.len(),
                self.eviction_for(weight.boundary),
                move |runtime, ptr| {
                    unsafe { runtime.htod(&bytes, ptr) }.map_err(|error| {
                        WeightHandleError::DeviceBinding(format!("H2D copy: {error}"))
                    })
                },
            );
        }
        let page = Arc::new(CudaWeightPage::upload(
            &self.runtime,
            resident.dtype,
            resident.shape.clone(),
            resident.bytes(),
        )?);
        self.admit(key, page, self.eviction_for(weight.boundary))
    }

    fn resident_vmm_with<F>(
        &self,
        key: u64,
        dtype: DataType,
        shape: Vec<usize>,
        len: usize,
        eviction: WeightEvictionPolicy,
        fill: F,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError>
    where
        F: FnOnce(&CudaRuntime, CUdeviceptr) -> Result<(), WeightHandleError>,
    {
        let physical = self
            .physical
            .get()
            .expect("VMM residency helper requires physical admission");
        let mut inner = self.lock();
        if let Some(existing) = inner.pages.get(&key).cloned() {
            inner.record_hit(key);
            return Ok(existing);
        }
        let allowance = inner.mapped_allowance.clone().ok_or_else(|| {
            WeightHandleError::DeviceBinding(
                "VMM weight residency has no authority-scoped mapped-byte allowance; \
                     adopt the memory governor before page-in"
                    .into(),
            )
        })?;

        // Issue #716: reserve a per-key **stable virtual address** once and
        // reuse it for every page-in of this key. A captured CUDA graph bakes
        // the device pointer of each weight it reads; reusing the identical VA
        // — with physical granules mapped underneath on page-in and unmapped on
        // eviction — is what lets a captured graph replay correctly after the
        // weight is evicted and paged back in (proven in
        // `vmm_stable_va_weight_slot_gpu`). A first-seen key gets a fresh
        // throwaway reservation; whether that becomes a persistent slot is
        // decided once admission classifies the page as retained vs bypassed.
        let reused_slot = inner.slots.get(&key).copied();
        let ptr = match reused_slot {
            Some(slot) => {
                // A key's byte size is fixed by the weight it names; a differing
                // length means two weights collided on one key, which would
                // corrupt the baked-pointer contract. Refuse rather than remap.
                if slot.len != len {
                    return Err(WeightHandleError::DeviceBinding(format!(
                        "stable weight slot for key {key} was reserved for {} bytes but a \
                         {len}-byte page-in requested it",
                        slot.len
                    )));
                }
                NonNull::new(slot.va as *mut u8).ok_or_else(|| {
                    WeightHandleError::DeviceBinding("stable weight slot has a null VA".into())
                })?
            }
            None => physical
                .allocator
                .allocate_committed(len, WEIGHT_SLOT_ALIGN, &[])
                .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))?,
        };

        let bypass = match self
            .admit_committed_span(&mut inner, physical, &allowance, ptr, len, eviction, fill)
        {
            Ok(bypass) => bypass,
            Err(error) => {
                // A reused slot's VA is persistent (its physical is already
                // decommitted); a fresh throwaway reservation must be released
                // so a failed page-in never leaks address space.
                if reused_slot.is_none() {
                    let _ = physical.allocator.deallocate_span(ptr);
                }
                return Err(error);
            }
        };

        // A reused slot is always persistent. A fresh page that admission
        // retained (not bypassed) becomes a persistent slot; a fresh bypassed
        // page keeps its throwaway VA and is freed outright on Drop, so two
        // concurrent bypass page-ins can never alias one physical granule.
        let stable_slot = reused_slot.is_some() || !bypass;
        if stable_slot && reused_slot.is_none() {
            inner.slots.insert(
                key,
                StableWeightSlot {
                    va: ptr.as_ptr() as CUdeviceptr,
                    len,
                },
            );
        }
        let page = Arc::new(CudaWeightPage {
            runtime: Arc::clone(&self.runtime),
            allocation: WeightAllocation::Vmm {
                allocator: Arc::clone(&physical.allocator),
                allowance: allowance.clone(),
                stable_slot,
            },
            ptr: ptr.as_ptr() as CUdeviceptr,
            len,
            dtype,
            shape,
        });
        if bypass {
            inner.record_bypassed_page_in();
        } else {
            inner.insert_page(key, Arc::clone(&page), len as u64);
        }
        Ok(page)
    }

    /// Commit `len` physical bytes under the reserved VA `ptr`, evicting other
    /// resident pages as `eviction` permits, then run `fill` to copy the weight
    /// bytes into the freshly mapped granules. Returns `true` when the page
    /// could not be admitted to the resident set and is handed back transiently
    /// (a "bypass" under scan-resistant admission), `false` when it is retained.
    ///
    /// This runs entirely under the residency lock (`inner`), so page-ins —
    /// and therefore the eviction-driven `decommit` of any stable slot — are
    /// fully serialized. Whole-step CUDA graph capture is capture-once /
    /// replay-many and performs no page-ins during replay, so no `decommit` of
    /// a baked VA can occur while a replay is in flight; the engine additionally
    /// declines capture whenever a step reports a bypass or eviction (issue
    /// #716), keeping the invariant enforceable rather than advisory.
    #[allow(clippy::too_many_arguments)]
    fn admit_committed_span<F>(
        &self,
        inner: &mut ResidencyInner,
        physical: &PhysicalAdmission,
        allowance: &onnx_runtime_memory_governor::MappedAllowance,
        ptr: NonNull<u8>,
        len: usize,
        eviction: WeightEvictionPolicy,
        fill: F,
    ) -> Result<bool, WeightHandleError>
    where
        F: FnOnce(&CudaRuntime, CUdeviceptr) -> Result<(), WeightHandleError>,
    {
        let mut fill = Some(fill);
        let max_evictions = inner.pages.len();
        let mut evictions = 0usize;
        let mut bypass = false;
        let mut streams_drained = false;
        loop {
            let required_owned = physical
                .allocator
                .incremental_owned_bytes_for_span(ptr, len, 0, len)
                .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))?;
            let required_mapped = physical
                .allocator
                .incremental_mapped_bytes_for_span(ptr, len, 0, len)
                .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))?;
            let global_available = physical.governor.available(Tier::Device);
            let zone_available = allowance.available();
            if committed_admission_fits(
                required_mapped,
                zone_available,
                required_owned,
                global_available,
            ) {
                allowance
                    .try_map(required_mapped)
                    .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))?;
                GLOBAL_WEIGHT_MAPPED_BYTES.fetch_add(required_mapped, Ordering::Relaxed);
                match physical.allocator.try_commit_span(
                    ptr,
                    len,
                    0,
                    len,
                    required_mapped,
                    global_available,
                ) {
                    Ok(commit) => {
                        let excess = required_mapped.saturating_sub(commit.newly_mapped_bytes);
                        if excess > 0 {
                            allowance.unmap(excess);
                            let _ = GLOBAL_WEIGHT_MAPPED_BYTES.fetch_update(
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                                |current| Some(current.saturating_sub(excess)),
                            );
                        }
                        fill.take().expect("VMM page fill runs once")(
                            &self.runtime,
                            ptr.as_ptr() as CUdeviceptr,
                        )?;
                        return Ok(bypass);
                    }
                    Err(error) => {
                        allowance.unmap(required_mapped);
                        let _ = GLOBAL_WEIGHT_MAPPED_BYTES.fetch_update(
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                            |current| Some(current.saturating_sub(required_mapped)),
                        );
                        if evictions >= max_evictions {
                            return Err(WeightHandleError::DeviceBinding(error.to_string()));
                        }
                    }
                }
            } else if eviction == WeightEvictionPolicy::StableResident {
                bypass = true;
            }

            if evictions >= max_evictions {
                return Err(WeightHandleError::DeviceBinding(format!(
                    "weight residency requires {required_mapped} incremental mapped bytes with \
                     {zone_available} bytes of weight-zone headroom and {required_owned} \
                     incremental committed bytes with {global_available} bytes of physical \
                     headroom after {evictions} eviction(s)"
                )));
            }
            let before_owned = physical
                .allocator
                .physical_pool_stats()
                .map_or(0, |stats| stats.snapshot().total_owned_bytes);
            let before_required_owned = required_owned;
            let before_required_mapped = required_mapped;
            let Some(evicted_key) = inner.next_evictable_key(eviction) else {
                return Err(WeightHandleError::DeviceBinding(format!(
                    "weight residency requires {required_mapped} incremental mapped bytes with \
                     {zone_available} bytes of weight-zone headroom and {required_owned} \
                     incremental committed bytes with {global_available} bytes of physical \
                     headroom, and no page is evictable"
                )));
            };
            if !streams_drained {
                let sync_start = std::time::Instant::now();
                self.runtime.synchronize().map_err(|error| {
                    WeightHandleError::DeviceBinding(format!("compute stream sync: {error}"))
                })?;
                self.runtime.copy_stream().synchronize().map_err(|error| {
                    WeightHandleError::DeviceBinding(format!("copy stream sync: {error}"))
                })?;
                add_duration(&GLOBAL_ADMIT_SYNC_NS, sync_start.elapsed());
                streams_drained = true;
            }
            inner.remove_page_after_stream_sync(evicted_key);
            evictions += 1;
            let after_owned = physical
                .allocator
                .physical_pool_stats()
                .map_or(0, |stats| stats.snapshot().total_owned_bytes);
            let after_required_owned = physical
                .allocator
                .incremental_owned_bytes_for_span(ptr, len, 0, len)
                .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))?;
            let after_required_mapped = physical
                .allocator
                .incremental_mapped_bytes_for_span(ptr, len, 0, len)
                .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))?;
            if !eviction_made_committed_progress(
                before_owned,
                after_owned,
                before_required_owned,
                after_required_owned,
                before_required_mapped,
                after_required_mapped,
            ) {
                inner.admission_no_progress = inner.admission_no_progress.saturating_add(1);
            }
        }
    }

    fn eviction_for(&self, boundary: LazyWeightBoundary) -> WeightEvictionPolicy {
        eviction_for_boundary(self.scan_resistant_dense, boundary)
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
        let bytes = page.len() as u64;
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
                inner.insert_page(key, Arc::clone(&page), bytes);
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
                    replace_global_budget(inner.policy.budget - over, inner.policy.budget);
                }
                // No lease means no governor knows about this cache, so there is
                // nothing to ask and nothing whose total this would falsify.
                // Keep the previous behaviour rather than inventing a refusal
                // the operator never asked for.
                None => {
                    inner.policy.budget = inner.policy.budget.saturating_add(over);
                    replace_global_budget(inner.policy.budget - over, inner.policy.budget);
                }
            }
        }
        inner.insert_page(key, Arc::clone(&page), bytes);
        Ok(page)
    }

    /// Snapshot the cache's activity counters.
    pub fn stats(&self) -> CudaResidencyStats {
        let inner = self.lock();
        let physical_owned_bytes = self
            .physical
            .get()
            .and_then(|physical| physical.allocator.physical_pool_stats())
            .map_or(inner.policy.resident_bytes, |stats| {
                stats.snapshot().total_owned_bytes
            });
        let mapped_physical_bytes = inner
            .mapped_allowance
            .as_ref()
            .map_or(inner.policy.resident_bytes, |allowance| {
                allowance.mapped_bytes()
            });
        CudaResidencyStats {
            budget_bytes: inner.policy.budget,
            resident_bytes: inner.policy.resident_bytes,
            peak_resident_bytes: inner.policy.peak_resident_bytes,
            pages_resident: inner.pages.len() as u64,
            page_ins: inner.policy.page_ins,
            hits: inner.policy.hits,
            evictions: inner.policy.evictions,
            physical_owned_bytes,
            mapped_physical_bytes,
            admission_no_progress: inner.admission_no_progress,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ResidencyInner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Whether page-ins go through the stable-virtual-address VMM path (issue
    /// #716). When `true`, every retained weight `key` is served from a
    /// reserved-once device VA whose physical granules are mapped/unmapped
    /// underneath, so a captured CUDA graph that baked a weight pointer stays
    /// valid across evict→repage. This is the gate the decode session uses to
    /// decide that weight offload no longer forces whole-step graph capture
    /// OFF. `false` means the non-VMM `alloc_raw`/`free_raw` path is in use,
    /// which hands out a different pointer per page-in and is capture-illegal.
    pub fn stable_va_paging_active(&self) -> bool {
        self.physical.get().is_some()
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

impl onnx_runtime_memory_governor::ReclaimableMappedHolder for CudaWeightResidency {
    fn allowance(&self) -> onnx_runtime_memory_governor::MappedAllowance {
        self.lock()
            .mapped_allowance
            .clone()
            .expect("registered VMM weight residency must have a mapped allowance")
    }

    fn reclaim_priority(&self) -> u32 {
        0
    }

    fn mapped_bytes(&self) -> u64 {
        self.lock()
            .mapped_allowance
            .as_ref()
            .map_or(0, |allowance| allowance.mapped_bytes())
    }

    fn reclaim_mapped(
        &self,
        target_bytes: u64,
    ) -> Result<
        onnx_runtime_memory_governor::MappedReclaimReport,
        onnx_runtime_memory_governor::MemoryError,
    > {
        let allowance = self.allowance();
        let before = allowance.mapped_bytes();
        let mut inner = self.lock();
        let max_attempts = inner.pages.len();
        let mut attempts = 0usize;
        while before.saturating_sub(allowance.mapped_bytes()) < target_bytes
            && attempts < max_attempts
        {
            let Some(key) = inner.next_evictable_key(WeightEvictionPolicy::Lru) else {
                break;
            };
            inner.remove_page(key);
            attempts += 1;
        }
        let reclaimed = before.saturating_sub(allowance.mapped_bytes());
        Ok(onnx_runtime_memory_governor::MappedReclaimReport {
            target_bytes,
            reclaimed_bytes: reclaimed,
        })
    }
}

impl Drop for CudaWeightResidency {
    fn drop(&mut self) {
        let budget = self
            .inner
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .policy
            .budget;
        let resident = self
            .inner
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .policy
            .resident_bytes;
        replace_global_budget(budget, 0);
        let _ = GLOBAL_CONTENT_RESIDENT_BYTES.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| Some(current.saturating_sub(resident)),
        );
    }
}

impl ResidencyInner {
    /// Record a cache hit for `key`: mark it most-recently-used and bump the
    /// per-instance and process-global hit counters.
    ///
    /// Also accumulates the **bytes** served from residency. The count-based hit
    /// rate is a poor proxy for streaming cost, because the resident set skews
    /// toward many small tensors (norms, biases) while misses skew toward few
    /// large ones. Measured on qwen14b-zp: raising the weight budget moved the
    /// count hit rate 57.09% -> 81.31% while the byte gap to the streaming floor
    /// *widened* from 1.78x to 2.30x (#857). `htod_bytes` drives cost, so the
    /// byte-weighted rate `hit_bytes / (hit_bytes + htod_bytes)` is the metric
    /// residency-policy work must be judged on (#837 item 3).
    fn record_hit(&mut self, key: u64) {
        self.policy.record_hit(key);
        GLOBAL_HITS.fetch_add(1, Ordering::Relaxed);
        if let Some(page) = self.pages.get(&key) {
            GLOBAL_HIT_BYTES.fetch_add(page.len as u64, Ordering::Relaxed);
        }
    }

    /// Insert a freshly paged-in `page` of `bytes` under `key`, updating the
    /// order, residency accounting, and the page-in counters.
    fn insert_page(&mut self, key: u64, page: Arc<CudaWeightPage>, bytes: u64) {
        self.pages.insert(key, page);
        self.policy.insert_page(key, bytes);
        let global_resident = GLOBAL_CONTENT_RESIDENT_BYTES
            .fetch_add(bytes, Ordering::Relaxed)
            .saturating_add(bytes);
        GLOBAL_PEAK_RESIDENT_BYTES.fetch_max(global_resident, Ordering::Relaxed);
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
    }

    /// Count a miss whose freshly paged-in allocation is returned directly to
    /// the caller instead of becoming part of the resident set.
    fn record_bypassed_page_in(&mut self) {
        self.policy.record_page_in();
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_BYPASSED_PAGE_INS.fetch_add(1, Ordering::Relaxed);
    }

    fn next_evictable_key(&self, eviction: WeightEvictionPolicy) -> Option<u64> {
        self.policy
            .next_evictable_index(eviction, &mut |key| {
                self.pages
                    .get(&key)
                    .is_some_and(|page| Arc::strong_count(page) == 1)
            })
            .map(|index| self.policy.order[index])
    }

    fn remove_page(&mut self, key: u64) {
        if self.pages.remove(&key).is_some()
            && let Some(bytes) = self.policy.remove_page(key)
        {
            let _ = GLOBAL_CONTENT_RESIDENT_BYTES.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current| Some(current.saturating_sub(bytes)),
            );
            GLOBAL_EVICTIONS.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn remove_page_after_stream_sync(&mut self, key: u64) {
        let Some(page) = self.pages.remove(&key) else {
            return;
        };
        let bytes = self.policy.remove_page(key);
        if let Ok(mut page) = Arc::try_unwrap(page) {
            page.retire_after_stream_sync();
        }
        if let Some(bytes) = bytes {
            let _ = GLOBAL_CONTENT_RESIDENT_BYTES.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current| Some(current.saturating_sub(bytes)),
            );
            GLOBAL_EVICTIONS.fetch_add(1, Ordering::Relaxed);
        }
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
            if let Some(page) = self.pages.remove(&key) {
                let bytes = page.len() as u64;
                let _ = GLOBAL_CONTENT_RESIDENT_BYTES.fetch_update(
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                    |current| Some(current.saturating_sub(bytes)),
                );
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
    fn byte_hit_rate_diverges_from_the_count_based_rate() {
        // The failure mode this metric exists to catch, from the #857
        // measurement: raising the weight budget moved the count-based hit rate
        // 57.09% -> 81.31% while the bytes actually streamed stayed dominant,
        // because hits skew small (norms, biases) and misses skew large
        // (projections, ~11.9 MB average page-in on qwen14b).
        let stats = GlobalOffloadStats {
            hits: 9,
            page_ins: 1,
            hit_bytes: 9 * 10 * 1024,     // nine 10 KiB tensors
            htod_bytes: 12 * 1024 * 1024, // one 12 MiB tensor
            ..GlobalOffloadStats::default()
        };
        let count_rate = stats.hits as f64 / (stats.hits + stats.page_ins) as f64;
        let byte_rate = stats.byte_hit_rate().expect("bytes were requested");
        assert!(
            (count_rate - 0.90).abs() < 1e-9,
            "count-based rate looks excellent: {count_rate}"
        );
        assert!(
            byte_rate < 0.01,
            "byte-weighted rate tells the truth about streaming cost: {byte_rate}"
        );
    }

    #[test]
    fn byte_hit_rate_is_none_when_no_bytes_were_requested() {
        assert_eq!(GlobalOffloadStats::default().byte_hit_rate(), None);
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
    fn budget_parsing_rejects_zero_and_garbage() {
        assert_eq!(parse_budget_bytes("1048576"), Some(1_048_576));
        assert_eq!(parse_budget_bytes("  4096 "), Some(4096));
        assert_eq!(parse_budget_bytes("0"), None);
        assert_eq!(parse_budget_bytes(""), None);
        assert_eq!(parse_budget_bytes("lots"), None);
        assert_eq!(parse_budget_bytes("-5"), None);
    }

    #[test]
    fn vmm_admission_requires_both_zone_and_global_headroom() {
        assert!(committed_admission_fits(0, 0, 0, 0));
        assert!(
            !committed_admission_fits(2 << 20, 0, 0, 8 << 30),
            "pooled ownership cannot bypass a full weight zone"
        );
        assert!(
            !committed_admission_fits(0, 8 << 30, 2 << 20, 742 << 10),
            "zone room cannot bypass missing global creation headroom"
        );
    }

    #[test]
    fn zero_committed_byte_eviction_is_observable_no_progress() {
        assert!(!eviction_made_committed_progress(8, 8, 4, 4, 4, 4));
        assert!(
            eviction_made_committed_progress(8, 8, 4, 0, 4, 4),
            "returning an owned handle to the pool is useful even though owned bytes stay flat"
        );
        assert!(eviction_made_committed_progress(8, 4, 4, 4, 4, 4));
        assert!(eviction_made_committed_progress(8, 8, 4, 4, 4, 0));
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn vmm_weight_admission_reuses_owned_granules_without_runtime_alloc_free() {
        use onnx_runtime_memory_governor::{
            DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
        };

        unsafe {
            std::env::set_var(
                crate::vmm_allocator::CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
                (64usize << 20).to_string(),
            );
        }
        let Ok(runtime) = CudaRuntime::new(0).map(Arc::new) else {
            eprintln!(
                "SKIPPED (CUDA runtime dependencies unavailable): VMM weight admission GPU test"
            );
            return;
        };
        let baseline_global = global_offload_stats();
        let granule = 2usize << 20;
        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new(
            (granule * 2) as u64,
            0,
            0,
        )));
        let allocator = Arc::new(
            crate::vmm_allocator::CudaVmmAllocator::new(
                runtime.cuda_context(),
                DeviceKey::device(0),
                0,
                64 << 20,
                governor.as_ref(),
                HolderId::new(736),
                MemoryRole::Weights,
            )
            .expect("VMM allocator"),
        );
        let authority: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync> =
            governor.clone();
        let residency = CudaWeightResidency::new(Arc::clone(&runtime), granule as u64)
            .with_vmm_admission(Arc::clone(&allocator), authority)
            .expect("install VMM admission");
        residency
            .adopt_governed_budget(governor.as_ref(), Tier::Device, HolderId::new(736))
            .expect("reserve mapped weight allowance");
        let before = runtime.allocation_counts();

        let first_bytes = vec![0x31u8; granule];
        let first = residency
            .resident_vmm_with(
                1,
                DataType::Uint8,
                vec![granule],
                granule,
                WeightEvictionPolicy::Lru,
                move |runtime, ptr| {
                    unsafe { runtime.htod(&first_bytes, ptr) }
                        .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))
                },
            )
            .expect("first physical page");

        let zone_error = residency
            .resident_vmm_with(
                2,
                DataType::Uint8,
                vec![granule],
                granule,
                WeightEvictionPolicy::StableResident,
                |_, _| panic!("zone refusal must happen before copy"),
            )
            .expect_err("global room cannot bypass a full mapped weight allowance");
        assert!(zone_error.to_string().contains("weight-zone headroom"));
        drop(first);

        let second_bytes = vec![0x42u8; granule];
        let second = residency
            .resident_vmm_with(
                2,
                DataType::Uint8,
                vec![granule],
                granule,
                WeightEvictionPolicy::Lru,
                move |runtime, ptr| {
                    unsafe { runtime.htod(&second_bytes, ptr) }
                        .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))
                },
            )
            .expect("second page reuses the owned handle");
        drop(second);

        let stats = residency.stats();
        assert_eq!(stats.resident_bytes, granule as u64);
        assert_eq!(stats.mapped_physical_bytes, granule as u64);
        assert_eq!(stats.physical_owned_bytes, granule as u64);
        assert_eq!(stats.page_ins, 2);
        assert_eq!(stats.evictions, 1);
        assert_eq!(runtime.allocation_counts(), before);
        assert_eq!(governor.used(Tier::Device), granule as u64);

        let live_before_reset = global_offload_stats();
        assert_eq!(
            live_before_reset.content_resident_bytes,
            baseline_global.content_resident_bytes + granule as u64
        );
        assert_eq!(
            live_before_reset.mapped_physical_bytes,
            baseline_global.mapped_physical_bytes + granule as u64
        );
        assert_eq!(
            live_before_reset.budget_bytes,
            baseline_global.budget_bytes + granule as u64
        );
        assert!(live_before_reset.page_ins > 0);
        assert!(live_before_reset.evictions > 0);

        // Model an earlier lifetime high-water above current live residency.
        // The reset must perform no write to the peak; storing either zero or a
        // sampled live value would make this assertion fail.
        let lifetime_peak = live_before_reset
            .content_resident_bytes
            .saturating_add(granule as u64);
        GLOBAL_PEAK_RESIDENT_BYTES.fetch_max(lifetime_peak, Ordering::Relaxed);
        let live_before_reset = global_offload_stats();

        reset_global_offload_stats();
        let live_after_reset = global_offload_stats();
        assert_eq!(live_after_reset.page_ins, 0);
        assert_eq!(live_after_reset.hits, 0);
        assert_eq!(live_after_reset.evictions, 0);
        assert_eq!(
            live_after_reset.content_resident_bytes,
            live_before_reset.content_resident_bytes
        );
        assert_eq!(
            live_after_reset.mapped_physical_bytes,
            live_before_reset.mapped_physical_bytes
        );
        assert_eq!(
            live_after_reset.budget_bytes,
            live_before_reset.budget_bytes
        );
        assert_eq!(
            live_after_reset.peak_resident_bytes,
            live_before_reset.peak_resident_bytes
        );
        assert!(live_after_reset.peak_resident_bytes >= live_after_reset.content_resident_bytes);

        let second_authority: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync> =
            governor.clone();
        let other = CudaWeightResidency::new(Arc::clone(&runtime), granule as u64)
            .with_vmm_admission(Arc::clone(&allocator), second_authority)
            .expect("install second VMM admission");
        other
            .adopt_governed_budget(governor.as_ref(), Tier::Device, HolderId::new(737))
            .expect("reserve second mapped allowance");
        let _kv = governor
            .reserve(
                Tier::Device,
                granule as u64,
                MemoryRole::KvCache,
                HolderId::new(9),
            )
            .expect("KV consumes remaining physical headroom");
        let global_error = other
            .resident_vmm_with(
                3,
                DataType::Uint8,
                vec![granule],
                granule,
                WeightEvictionPolicy::Lru,
                |_, _| panic!("global refusal must happen before copy"),
            )
            .expect_err("zone room cannot bypass missing global physical headroom");
        assert!(global_error.to_string().contains("physical headroom"));
        drop(_kv);
        drop(other);
        drop(residency);
        let unloaded = global_offload_stats();
        assert_eq!(
            unloaded.content_resident_bytes,
            baseline_global.content_resident_bytes
        );
        assert_eq!(
            unloaded.mapped_physical_bytes,
            baseline_global.mapped_physical_bytes
        );
        assert_eq!(unloaded.budget_bytes, baseline_global.budget_bytes);
        let retained = allocator
            .physical_pool_stats()
            .expect("pool stats after unload")
            .snapshot();
        assert_eq!(retained.mapped_bytes, 0);
        assert_eq!(retained.pooled_unmapped_bytes, granule as u64);
        assert_eq!(retained.total_owned_bytes, granule as u64);
    }

    /// Issue #716: a retained weight `key` keeps a **stable device virtual
    /// address** across an evict→repage cycle. The first page-in reserves the
    /// key's VA slot; evicting it (paging another key under a one-page budget)
    /// unmaps only the physical granule and keeps the VA; paging the key back in
    /// reuses that identical VA. This is the residency-level guarantee that lets
    /// a captured CUDA graph — which baked the weight's device pointer — replay
    /// correctly after the weight was evicted and paged back in.
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn vmm_retained_weight_key_keeps_a_stable_virtual_address_across_repage() {
        use onnx_runtime_memory_governor::{
            DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole,
        };

        unsafe {
            std::env::set_var(
                crate::vmm_allocator::CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
                (64usize << 20).to_string(),
            );
        }
        let Ok(runtime) = CudaRuntime::new(0).map(Arc::new) else {
            eprintln!("SKIPPED (CUDA runtime dependencies unavailable): stable-VA repage GPU test");
            return;
        };
        let granule = 2usize << 20;
        // Two granules of global headroom, but a one-granule weight budget: the
        // second key must evict the first, exercising the evict→repage path.
        let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new(
            (granule * 2) as u64,
            0,
            0,
        )));
        let allocator = Arc::new(
            crate::vmm_allocator::CudaVmmAllocator::new(
                runtime.cuda_context(),
                DeviceKey::device(0),
                0,
                64 << 20,
                governor.as_ref(),
                HolderId::new(716),
                MemoryRole::Weights,
            )
            .expect("VMM allocator"),
        );
        let authority: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync> =
            governor.clone();
        let residency = CudaWeightResidency::new(Arc::clone(&runtime), granule as u64)
            .with_vmm_admission(Arc::clone(&allocator), authority)
            .expect("install VMM admission");
        residency
            .adopt_governed_budget(governor.as_ref(), Tier::Device, HolderId::new(716))
            .expect("reserve mapped weight allowance");

        assert!(
            residency.stable_va_paging_active(),
            "installing VMM admission must activate the stable-VA paging path"
        );

        let page_key_1 = |fill_byte: u8| {
            let bytes = vec![fill_byte; granule];
            residency
                .resident_vmm_with(
                    1,
                    DataType::Uint8,
                    vec![granule],
                    granule,
                    WeightEvictionPolicy::Lru,
                    move |runtime, ptr| {
                        unsafe { runtime.htod(&bytes, ptr) }
                            .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))
                    },
                )
                .expect("page key 1")
        };

        // First page-in of key 1 reserves its stable slot and records its VA.
        let first = page_key_1(0x11);
        let stable_va = first.device_ptr();
        drop(first);

        // Page key 2 under the one-page budget: key 1 is evicted, which unmaps
        // its physical granule but keeps its VA slot reserved.
        let second_bytes = vec![0x22u8; granule];
        let second = residency
            .resident_vmm_with(
                2,
                DataType::Uint8,
                vec![granule],
                granule,
                WeightEvictionPolicy::Lru,
                move |runtime, ptr| {
                    unsafe { runtime.htod(&second_bytes, ptr) }
                        .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))
                },
            )
            .expect("page key 2 evicts key 1");
        assert_ne!(
            second.device_ptr(),
            stable_va,
            "a different key must get a different VA slot"
        );
        drop(second);

        // Page key 1 back in: it must reuse the identical device VA it had
        // before, even though the physical granule underneath was returned to
        // the pool and re-mapped.
        let repaged = page_key_1(0x33);
        assert_eq!(
            repaged.device_ptr(),
            stable_va,
            "issue #716: a retained key must keep its stable VA across evict→repage"
        );

        let stats = residency.stats();
        assert_eq!(stats.page_ins, 3);
        // key 2 evicted key 1; re-paging key 1 evicted key 2 — two evictions.
        assert_eq!(stats.evictions, 2);
        // No churn allocator: physical ownership stayed at one reused granule.
        assert_eq!(stats.physical_owned_bytes, granule as u64);
        drop(repaged);
        drop(residency);
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
}
