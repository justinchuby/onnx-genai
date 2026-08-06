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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    LazyDeviceWeightBinder, LazyWeight, MmapRegionSource, WeightHandleError,
};
use onnx_runtime_ir::DataType;

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
static GLOBAL_PREFETCH_ISSUED: AtomicU64 = AtomicU64::new(0);
static GLOBAL_PREFETCH_DECLINED_GUARD: AtomicU64 = AtomicU64::new(0);
static GLOBAL_PREFETCH_JOINED: AtomicU64 = AtomicU64::new(0);
static GLOBAL_PREFETCH_STAGING_ALLOCS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_PREFETCH_STAGING_REUSES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_MATERIALIZE_NS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_HTOD_NS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_COPY_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_ADMIT_SYNC_NS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_STAGING_FILL_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_STAGING_FILL_REGIONS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_STAGING_FILL_CALLS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_MATERIALIZE_FALLBACK_CALLS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_HTOD_BYTES: AtomicU64 = AtomicU64::new(0);

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
    pub prefetch_issued: u64,
    pub prefetch_declined_guard: u64,
    pub prefetch_joined: u64,
    pub prefetch_staging_allocs: u64,
    pub prefetch_staging_reuses: u64,
    pub materialize_ns: u64,
    pub htod_ns: u64,
    pub copy_wait_ns: u64,
    pub admit_sync_ns: u64,
    pub staging_fill_bytes: u64,
    pub staging_fill_regions: u64,
    pub staging_fill_calls: u64,
    pub materialize_fallback_calls: u64,
    pub htod_bytes: u64,
}

/// Read the process-global weight-offload counters.
pub fn global_offload_stats() -> GlobalOffloadStats {
    GlobalOffloadStats {
        page_ins: GLOBAL_PAGE_INS.load(Ordering::Relaxed),
        hits: GLOBAL_HITS.load(Ordering::Relaxed),
        evictions: GLOBAL_EVICTIONS.load(Ordering::Relaxed),
        prefetch_issued: GLOBAL_PREFETCH_ISSUED.load(Ordering::Relaxed),
        prefetch_declined_guard: GLOBAL_PREFETCH_DECLINED_GUARD.load(Ordering::Relaxed),
        prefetch_joined: GLOBAL_PREFETCH_JOINED.load(Ordering::Relaxed),
        prefetch_staging_allocs: GLOBAL_PREFETCH_STAGING_ALLOCS.load(Ordering::Relaxed),
        prefetch_staging_reuses: GLOBAL_PREFETCH_STAGING_REUSES.load(Ordering::Relaxed),
        materialize_ns: GLOBAL_MATERIALIZE_NS.load(Ordering::Relaxed),
        htod_ns: GLOBAL_HTOD_NS.load(Ordering::Relaxed),
        copy_wait_ns: GLOBAL_COPY_WAIT_NS.load(Ordering::Relaxed),
        admit_sync_ns: GLOBAL_ADMIT_SYNC_NS.load(Ordering::Relaxed),
        staging_fill_bytes: GLOBAL_STAGING_FILL_BYTES.load(Ordering::Relaxed),
        staging_fill_regions: GLOBAL_STAGING_FILL_REGIONS.load(Ordering::Relaxed),
        staging_fill_calls: GLOBAL_STAGING_FILL_CALLS.load(Ordering::Relaxed),
        materialize_fallback_calls: GLOBAL_MATERIALIZE_FALLBACK_CALLS.load(Ordering::Relaxed),
        htod_bytes: GLOBAL_HTOD_BYTES.load(Ordering::Relaxed),
    }
}

/// Reset the process-global weight-offload counters (test observability helper).
pub fn reset_global_offload_stats() {
    GLOBAL_PAGE_INS.store(0, Ordering::Relaxed);
    GLOBAL_HITS.store(0, Ordering::Relaxed);
    GLOBAL_EVICTIONS.store(0, Ordering::Relaxed);
    GLOBAL_PREFETCH_ISSUED.store(0, Ordering::Relaxed);
    GLOBAL_PREFETCH_DECLINED_GUARD.store(0, Ordering::Relaxed);
    GLOBAL_PREFETCH_JOINED.store(0, Ordering::Relaxed);
    GLOBAL_PREFETCH_STAGING_ALLOCS.store(0, Ordering::Relaxed);
    GLOBAL_PREFETCH_STAGING_REUSES.store(0, Ordering::Relaxed);
    GLOBAL_MATERIALIZE_NS.store(0, Ordering::Relaxed);
    GLOBAL_HTOD_NS.store(0, Ordering::Relaxed);
    GLOBAL_COPY_WAIT_NS.store(0, Ordering::Relaxed);
    GLOBAL_ADMIT_SYNC_NS.store(0, Ordering::Relaxed);
    GLOBAL_STAGING_FILL_BYTES.store(0, Ordering::Relaxed);
    GLOBAL_STAGING_FILL_REGIONS.store(0, Ordering::Relaxed);
    GLOBAL_STAGING_FILL_CALLS.store(0, Ordering::Relaxed);
    GLOBAL_MATERIALIZE_FALLBACK_CALLS.store(0, Ordering::Relaxed);
    GLOBAL_HTOD_BYTES.store(0, Ordering::Relaxed);
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

/// Whether/how the CUDA EP should page offloaded weights into a bounded VRAM
/// residency cache. Disabled by default so the resident fast path is untouched
/// and byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DeviceOffloadPolicy {
    pub enabled: bool,
    /// Explicit VRAM budget in bytes, if the operator pinned one.
    pub device_budget_bytes: Option<u64>,
    /// Use the asynchronous, fence-ordered page-in (default `true` / opt-out).
    /// This is the only path that can prefetch the next known layer while the
    /// current layer runs; set `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=0` to
    /// force the old synchronous demand page-in for A/B measurements.
    pub async_pagein: bool,
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
        Self {
            enabled,
            device_budget_bytes,
            async_pagein,
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
    /// Number of pages currently resident.
    pub pages_resident: u64,
    /// H2D page-ins performed (cache misses that allocated + copied a page).
    pub page_ins: u64,
    /// Cache hits that reused an already-resident page (no H2D copy).
    pub hits: u64,
    /// LRU evictions that freed a page's VRAM.
    pub evictions: u64,
    /// Lookahead page-ins issued on the transfer stream.
    pub prefetch_issued: u64,
    /// Lookahead requests declined because admitting them would evict or grow.
    pub prefetch_declined_guard: u64,
    /// Demand page-ins that consumed an in-flight lookahead page.
    pub prefetch_joined: u64,
    /// Pinned staging buffers allocated for prefetch.
    pub prefetch_staging_allocs: u64,
    /// Pinned staging buffers reused from the prefetch pool.
    pub prefetch_staging_reuses: u64,
}

/// A live VRAM residency page for one offloaded weight tensor.
///
/// Owns a single device allocation holding the tensor's canonical compressed
/// bytes and frees it exactly once on drop. The address is a CUDA device
/// pointer — never dereferenced on the host — exposed through [`Self::device_ptr`]
/// for a consuming kernel's `TensorView`.
pub struct CudaWeightPage {
    runtime: Arc<CudaRuntime>,
    ptr: CUdeviceptr,
    len: usize,
    dtype: DataType,
    shape: Vec<usize>,
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
        let ptr = runtime
            .alloc_raw(bytes.len())
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        let page = Self {
            runtime: Arc::clone(runtime),
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
        let ptr = runtime
            .alloc_raw(bytes.len())
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        staging.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        // Own the VRAM `ptr` before enqueuing the copy, so any error below drops
        // `page` and frees it exactly once. The pinned staging remains owned by
        // this function and is returned to the caller on success so it can keep
        // the source alive until the fence completes.
        let page = Self {
            runtime: Arc::clone(runtime),
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
        let ptr = runtime
            .alloc_raw(len)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        let page = Self {
            runtime: Arc::clone(runtime),
            ptr,
            len,
            dtype,
            shape,
        };
        let copy_start = std::time::Instant::now();
        let staged = &staging.as_slice()[..len];
        unsafe { runtime.htod_async(staged, ptr) }.map_err(|error| {
            WeightHandleError::DeviceBinding(format!("async H2D copy: {error}"))
        })?;
        let fence = runtime
            .record_copy_fence()
            .map_err(|error| WeightHandleError::DeviceBinding(format!("copy fence: {error}")))?;
        add_duration(&GLOBAL_HTOD_NS, copy_start.elapsed());
        GLOBAL_HTOD_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        Ok((page, fence, staging))
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
        let _ = unsafe { self.runtime.free_raw(self.ptr) };
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
/// Page-in defaults to the asynchronous, fence-ordered path
/// ([`CudaWeightPage::upload_async`]): the H2D copy is enqueued on the runtime's
/// dedicated transfer stream so it overlaps the in-flight compute, and the
/// consuming compute stream is ordered after it with a completion fence. The
/// full compute-stream drain is now taken **only when admitting must evict** (so
/// eviction never frees a page a prior kernel still reads); a page-in that fits
/// the budget no longer host-blocks, which is what lets the transfer overlap.
/// Set `async_pagein = false` (env `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=0`) to
/// force the old synchronous page-in for A/B comparison.
pub struct CudaWeightResidency {
    runtime: Arc<CudaRuntime>,
    /// Whether misses page in asynchronously (see [`CudaWeightPage::upload_async`]).
    async_pagein: bool,
    inner: Mutex<ResidencyInner>,
}

struct ResidencyInner {
    budget: u64,
    resident_bytes: u64,
    peak_resident_bytes: u64,
    page_ins: u64,
    hits: u64,
    evictions: u64,
    prefetch_issued: u64,
    prefetch_declined_guard: u64,
    prefetch_joined: u64,
    prefetch_staging_allocs: u64,
    prefetch_staging_reuses: u64,
    /// LRU order: front = least-recently-used, back = most-recently-used.
    order: Vec<u64>,
    pages: HashMap<u64, Arc<CudaWeightPage>>,
    in_flight: HashMap<u64, InFlightPage>,
    prefetch_staging_pool: Vec<PinnedStaging>,
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

struct InFlightPage {
    page: Arc<CudaWeightPage>,
    copy_fence: u64,
    staging: PinnedStaging,
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
        Self {
            runtime,
            async_pagein: false,
            inner: Mutex::new(ResidencyInner {
                budget: budget_bytes,
                lease: None,
                resident_bytes: 0,
                peak_resident_bytes: 0,
                page_ins: 0,
                hits: 0,
                evictions: 0,
                prefetch_issued: 0,
                prefetch_declined_guard: 0,
                prefetch_joined: 0,
                prefetch_staging_allocs: 0,
                prefetch_staging_reuses: 0,
                order: Vec::new(),
                pages: HashMap::new(),
                in_flight: HashMap::new(),
                prefetch_staging_pool: Vec::new(),
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
        Ok(Self {
            runtime,
            async_pagein: false,
            inner: Mutex::new(ResidencyInner {
                budget: lease.bytes(),
                lease: Some(lease),
                resident_bytes: 0,
                peak_resident_bytes: 0,
                page_ins: 0,
                hits: 0,
                evictions: 0,
                prefetch_issued: 0,
                prefetch_declined_guard: 0,
                prefetch_joined: 0,
                prefetch_staging_allocs: 0,
                prefetch_staging_reuses: 0,
                order: Vec::new(),
                pages: HashMap::new(),
                in_flight: HashMap::new(),
                prefetch_staging_pool: Vec::new(),
            }),
        })
    }

    /// Bytes this cache is entitled to, and whether that came from a governor.
    ///
    /// `false` means the budget was chosen locally and nothing reconciles it
    /// with any other claim on the same device.
    pub fn budget(&self) -> (u64, bool) {
        let inner = self.inner.lock().expect("residency lock poisoned");
        (inner.budget, inner.lease.is_some())
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
            return Ok(inner.budget);
        }
        inner.budget = budget_bytes;
        Ok(inner.budget)
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
        let requested = {
            let inner = self.inner.lock().expect("residency lock poisoned");
            if inner.lease.is_some() {
                return Ok(inner.budget);
            }
            inner.budget
        };
        let lease = governor.reserve(
            tier,
            requested,
            onnx_runtime_memory_governor::MemoryRole::Weights,
            holder,
        )?;
        let granted = lease.bytes();
        let mut inner = self.inner.lock().expect("residency lock poisoned");
        inner.budget = granted;
        inner.lease = Some(lease);
        Ok(granted)
    }

    /// Select the asynchronous (default `true`) vs synchronous page-in path.
    pub fn with_async_pagein(mut self, async_pagein: bool) -> Self {
        self.async_pagein = async_pagein;
        self
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
        // Copy region bytes host→device before re-locking so a failed bind never
        // mutates cache accounting.
        let pager = CudaWeightPager::new(Arc::clone(&self.runtime), source);
        let page = Arc::new(pager.bind_block_quantized_moe(weight)?);
        self.admit(key, page)
    }

    /// Live-dispatch entry point backed directly by the package mmap.
    ///
    /// This avoids calling [`LazyWeight::materialize`] on the hot path. On a
    /// not-fit model the same layer weights are paged every token; rebuilding an
    /// owned host tensor for each miss made CPU materialization dominate decode
    /// time even when H2D was a small fraction of the step.
    pub fn resident_mapped(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn MmapRegionSource,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        if let Some(prefetched) = self.join_in_flight(key)? {
            return Ok(prefetched);
        }
        if let Some(hit) = self.get_hit(key) {
            return Ok(hit);
        }
        if self.async_pagein {
            let len = weight.region_bytes_len();
            let mut staging = self.take_prefetch_staging(len)?;
            let materialize_start = std::time::Instant::now();
            fill_staging_from_regions(weight, source, &mut staging)?;
            add_duration(&GLOBAL_MATERIALIZE_NS, materialize_start.elapsed());
            let (page, copy_fence, staging) = CudaWeightPage::upload_staged_async(
                &self.runtime,
                weight.dtype,
                weight.shape.clone(),
                len,
                staging,
            )?;
            let admitted = self.admit(key, Arc::new(page))?;
            let wait_start = std::time::Instant::now();
            self.runtime
                .compute_wait_fence(copy_fence)
                .map_err(|error| {
                    WeightHandleError::DeviceBinding(format!("copy fence wait: {error}"))
                })?;
            add_duration(&GLOBAL_COPY_WAIT_NS, wait_start.elapsed());
            self.recycle_prefetch_staging(staging);
            Ok(admitted)
        } else {
            let pager = CudaWeightPager::new(Arc::clone(&self.runtime), source);
            let page = Arc::new(pager.bind_block_quantized_moe(weight)?);
            self.admit(key, page)
        }
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
        if let Some(prefetched) = self.join_in_flight(key)? {
            return Ok(prefetched);
        }
        if let Some(hit) = self.get_hit(key) {
            return Ok(hit);
        }
        let materialize_start = std::time::Instant::now();
        let resident = weight.materialize()?;
        GLOBAL_MATERIALIZE_FALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
        add_duration(&GLOBAL_MATERIALIZE_NS, materialize_start.elapsed());
        if self.async_pagein {
            // Async, fence-ordered page-in: enqueue the H2D on the transfer stream
            // (overlapping the in-flight compute), admit, then order the compute
            // stream after the transfer's completion event so the consuming kernel
            // waits only on the copy — never a full host sync.
            let staging = self.take_prefetch_staging(resident.bytes().len())?;
            let (page, copy_fence, staging) = CudaWeightPage::upload_async(
                &self.runtime,
                resident.dtype,
                resident.shape.clone(),
                resident.bytes(),
                staging,
            )?;
            let admitted = self.admit(key, Arc::new(page))?;
            let wait_start = std::time::Instant::now();
            self.runtime
                .compute_wait_fence(copy_fence)
                .map_err(|error| {
                    WeightHandleError::DeviceBinding(format!("copy fence wait: {error}"))
                })?;
            self.runtime.sync_copy_stream().map_err(|error| {
                WeightHandleError::DeviceBinding(format!("copy stream wait: {error}"))
            })?;
            add_duration(&GLOBAL_COPY_WAIT_NS, wait_start.elapsed());
            self.recycle_prefetch_staging(staging);
            Ok(admitted)
        } else {
            // Legacy synchronous page-in (A/B "before" arm / kill-switch): the
            // blocking `htod` serializes the transfer with compute.
            let page = Arc::new(CudaWeightPage::upload(
                &self.runtime,
                resident.dtype,
                resident.shape.clone(),
                resident.bytes(),
            )?);
            self.admit(key, page)
        }
    }

    /// Best-effort single-weight lookahead page-in. It only engages for the
    /// asynchronous path and keeps at most one over-budget in-flight page. The
    /// page is not admitted to the LRU until demand joins it, so any eviction
    /// remains at the normal demand synchronization point instead of freeing a
    /// page while the previous kernel may still be reading it.
    ///
    /// Depth is intentionally a scheduler knob, not a residency-policy knob:
    /// qwen2.5-0.5b on RTX 4060 at a 1.5 GiB weight budget was swept at 1, 2,
    /// 4, 8, and 16 nodes of lookahead. The requested issue-to-join gap was
    /// reachable and no evictions occurred, but no speedup was established; 16
    /// nodes was worse. Keep the guard conservative unless a larger,
    /// transfer-bound model proves a deeper default helps.
    pub fn prefetch_materialized(
        &self,
        key: u64,
        weight: &LazyWeight,
    ) -> Result<bool, WeightHandleError> {
        if !self.async_pagein {
            return Ok(false);
        }
        {
            let inner = self.lock();
            if inner.pages.contains_key(&key) || inner.in_flight.contains_key(&key) {
                return Ok(false);
            }
        }
        {
            let inner = self.lock();
            if inner.pages.contains_key(&key)
                || inner.in_flight.contains_key(&key)
                || !inner.in_flight.is_empty()
            {
                drop(inner);
                self.record_prefetch_declined_guard();
                return Ok(false);
            }
        }
        let materialize_start = std::time::Instant::now();
        let resident = weight.materialize()?;
        GLOBAL_MATERIALIZE_FALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
        add_duration(&GLOBAL_MATERIALIZE_NS, materialize_start.elapsed());
        let staging = self.take_prefetch_staging(resident.bytes().len())?;
        let (page, copy_fence, staging) = match CudaWeightPage::upload_async(
            &self.runtime,
            resident.dtype,
            resident.shape.clone(),
            resident.bytes(),
            staging,
        ) {
            Ok(uploaded) => uploaded,
            Err(error) => {
                self.record_prefetch_declined_guard();
                return match error {
                    WeightHandleError::DeviceBinding(message)
                        if message.contains("VRAM alloc") || message.contains("out of memory") =>
                    {
                        Ok(false)
                    }
                    other => Err(other),
                };
            }
        };
        let page = Arc::new(page);
        let mut inner = self.lock();
        if inner.pages.contains_key(&key)
            || inner.in_flight.contains_key(&key)
            || !inner.in_flight.is_empty()
        {
            drop(inner);
            self.drain_copy_stream()?;
            self.recycle_prefetch_staging(staging);
            self.record_prefetch_declined_guard();
            return Ok(false);
        }
        inner.record_prefetch_issued();
        inner.in_flight.insert(
            key,
            InFlightPage {
                page,
                copy_fence,
                staging,
            },
        );
        Ok(true)
    }

    /// Best-effort lookahead page-in from the package mmap, avoiding a throwaway
    /// host tensor in the same way as [`Self::resident_mapped`].
    pub fn prefetch_mapped(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn MmapRegionSource,
    ) -> Result<bool, WeightHandleError> {
        if !self.async_pagein {
            return Ok(false);
        }
        {
            let inner = self.lock();
            if inner.pages.contains_key(&key) || inner.in_flight.contains_key(&key) {
                return Ok(false);
            }
        }
        {
            let inner = self.lock();
            if inner.pages.contains_key(&key)
                || inner.in_flight.contains_key(&key)
                || !inner.in_flight.is_empty()
            {
                drop(inner);
                self.record_prefetch_declined_guard();
                return Ok(false);
            }
        }
        let len = weight.region_bytes_len();
        let mut staging = self.take_prefetch_staging(len)?;
        let materialize_start = std::time::Instant::now();
        fill_staging_from_regions(weight, source, &mut staging)?;
        add_duration(&GLOBAL_MATERIALIZE_NS, materialize_start.elapsed());
        let (page, copy_fence, staging) = match CudaWeightPage::upload_staged_async(
            &self.runtime,
            weight.dtype,
            weight.shape.clone(),
            len,
            staging,
        ) {
            Ok(uploaded) => uploaded,
            Err(error) => {
                self.record_prefetch_declined_guard();
                return match error {
                    WeightHandleError::DeviceBinding(message)
                        if message.contains("VRAM alloc") || message.contains("out of memory") =>
                    {
                        Ok(false)
                    }
                    other => Err(other),
                };
            }
        };
        let page = Arc::new(page);
        let mut inner = self.lock();
        if inner.pages.contains_key(&key)
            || inner.in_flight.contains_key(&key)
            || !inner.in_flight.is_empty()
        {
            drop(inner);
            self.drain_copy_stream()?;
            self.recycle_prefetch_staging(staging);
            self.record_prefetch_declined_guard();
            return Ok(false);
        }
        inner.record_prefetch_issued();
        inner.in_flight.insert(
            key,
            InFlightPage {
                page,
                copy_fence,
                staging,
            },
        );
        Ok(true)
    }

    fn join_in_flight(&self, key: u64) -> Result<Option<Arc<CudaWeightPage>>, WeightHandleError> {
        let mut inner = self.lock();
        let Some(prefetch) = inner.in_flight.remove(&key) else {
            return Ok(None);
        };
        let page = Arc::clone(&prefetch.page);
        let copy_fence = prefetch.copy_fence;
        if inner.pages.contains_key(&key) {
            inner.touch(key);
        }
        drop(inner);
        let wait_start = std::time::Instant::now();
        self.runtime
            .compute_wait_fence(copy_fence)
            .map_err(|error| {
                WeightHandleError::DeviceBinding(format!("prefetch fence wait: {error}"))
            })?;
        add_duration(&GLOBAL_COPY_WAIT_NS, wait_start.elapsed());
        let admitted = self.admit(key, page)?;
        let mut inner = self.lock();
        inner.record_prefetch_joined();
        inner.prefetch_staging_pool.push(prefetch.staging);
        Ok(Some(admitted))
    }

    fn take_prefetch_staging(&self, bytes: usize) -> Result<PinnedStaging, WeightHandleError> {
        {
            let mut inner = self.lock();
            if let Some(index) = inner
                .prefetch_staging_pool
                .iter()
                .position(|staging| staging.len() >= bytes)
            {
                inner.prefetch_staging_reuses += 1;
                GLOBAL_PREFETCH_STAGING_REUSES.fetch_add(1, Ordering::Relaxed);
                let staging = inner.prefetch_staging_pool.swap_remove(index);
                drop(inner);
                let wait_start = std::time::Instant::now();
                self.runtime.sync_copy_stream().map_err(|error| {
                    WeightHandleError::DeviceBinding(format!("copy stream wait: {error}"))
                })?;
                add_duration(&GLOBAL_COPY_WAIT_NS, wait_start.elapsed());
                return Ok(staging);
            }
        }
        let staging = self
            .runtime
            .alloc_pinned(bytes)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("pinned alloc: {error}")))?;
        let mut inner = self.lock();
        inner.prefetch_staging_allocs += 1;
        GLOBAL_PREFETCH_STAGING_ALLOCS.fetch_add(1, Ordering::Relaxed);
        Ok(staging)
    }

    fn recycle_prefetch_staging(&self, staging: PinnedStaging) {
        self.lock().prefetch_staging_pool.push(staging);
    }

    fn record_prefetch_declined_guard(&self) {
        let mut inner = self.lock();
        inner.prefetch_declined_guard += 1;
        GLOBAL_PREFETCH_DECLINED_GUARD.fetch_add(1, Ordering::Relaxed);
    }

    /// Look up `key`, marking it most-recently-used and counting a hit.
    fn get_hit(&self, key: u64) -> Option<Arc<CudaWeightPage>> {
        let mut inner = self.lock();
        if inner.in_flight.contains_key(&key) {
            return None;
        }
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
            if inner.resident_bytes.saturating_add(bytes) <= inner.budget {
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
        inner.evict_to_fit(bytes);
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
            .resident_bytes
            .saturating_add(bytes)
            .checked_sub(inner.budget)
            .filter(|over| *over > 0)
        {
            match inner.lease.as_mut() {
                Some(lease) => {
                    lease.grow(over).map_err(|error| {
                        WeightHandleError::DeviceBinding(format!(
                            "the weight-residency cache needs {over} bytes beyond its \
                             {} byte budget for a {bytes} byte page, and eviction could not \
                             free them: {error}",
                            inner.budget
                        ))
                    })?;
                    inner.budget = inner.budget.saturating_add(over);
                }
                // No lease means no governor knows about this cache, so there is
                // nothing to ask and nothing whose total this would falsify.
                // Keep the previous behaviour rather than inventing a refusal
                // the operator never asked for.
                None => inner.budget = inner.budget.saturating_add(over),
            }
        }
        inner.insert_page(key, Arc::clone(&page), bytes);
        Ok(page)
    }

    /// Snapshot the cache's activity counters.
    pub fn stats(&self) -> CudaResidencyStats {
        let inner = self.lock();
        CudaResidencyStats {
            budget_bytes: inner.budget,
            resident_bytes: inner.resident_bytes,
            peak_resident_bytes: inner.peak_resident_bytes,
            pages_resident: inner.pages.len() as u64,
            page_ins: inner.page_ins,
            hits: inner.hits,
            evictions: inner.evictions,
            prefetch_issued: inner.prefetch_issued,
            prefetch_declined_guard: inner.prefetch_declined_guard,
            prefetch_joined: inner.prefetch_joined,
            prefetch_staging_allocs: inner.prefetch_staging_allocs,
            prefetch_staging_reuses: inner.prefetch_staging_reuses,
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
    /// Move `key` to the most-recently-used end of the LRU order.
    fn touch(&mut self, key: u64) {
        if let Some(position) = self.order.iter().position(|&k| k == key) {
            let key = self.order.remove(position);
            self.order.push(key);
        }
    }

    /// Record a cache hit for `key`: mark it most-recently-used and bump the
    /// per-instance and process-global hit counters.
    fn record_hit(&mut self, key: u64) {
        self.touch(key);
        self.hits += 1;
        GLOBAL_HITS.fetch_add(1, Ordering::Relaxed);
    }

    /// Insert a freshly paged-in `page` of `bytes` under `key`, updating the LRU
    /// order, residency accounting, and the page-in counters.
    fn insert_page(&mut self, key: u64, page: Arc<CudaWeightPage>, bytes: u64) {
        self.pages.insert(key, page);
        self.order.push(key);
        self.resident_bytes += bytes;
        self.peak_resident_bytes = self.peak_resident_bytes.max(self.resident_bytes);
        self.page_ins += 1;
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
    }

    fn record_prefetch_issued(&mut self) {
        self.prefetch_issued += 1;
        GLOBAL_PREFETCH_ISSUED.fetch_add(1, Ordering::Relaxed);
    }

    fn record_prefetch_joined(&mut self) {
        self.prefetch_joined += 1;
        GLOBAL_PREFETCH_JOINED.fetch_add(1, Ordering::Relaxed);
    }

    /// Evict least-recently-used, currently-unreferenced pages until admitting
    /// `incoming` bytes fits the budget (best effort; stops when nothing more is
    /// evictable).
    fn evict_to_fit(&mut self, incoming: u64) {
        let mut index = 0;
        while self.resident_bytes.saturating_add(incoming) > self.budget && index < self.order.len()
        {
            let key = self.order[index];
            let evictable = self
                .pages
                .get(&key)
                .is_some_and(|page| Arc::strong_count(page) == 1);
            if evictable {
                if let Some(page) = self.pages.remove(&key) {
                    self.resident_bytes = self.resident_bytes.saturating_sub(page.len() as u64);
                    self.evictions += 1;
                    GLOBAL_EVICTIONS.fetch_add(1, Ordering::Relaxed);
                    // `page`'s Drop frees the VRAM here (sole owner).
                }
                self.order.remove(index);
                // Do not advance `index`: the vector shifted left under it.
            } else {
                index += 1;
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

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
    fn prefetched_materialized_weight_is_joined_by_demand_page_in() {
        let Ok(runtime) = crate::runtime::CudaRuntime::new(0).map(std::sync::Arc::new) else {
            eprintln!("SKIPPED (no CUDA runtime): prefetch join check did NOT run.");
            return;
        };
        let residency =
            CudaWeightResidency::new(Arc::clone(&runtime), 1024).with_async_pagein(true);
        let payload: Arc<[u8]> = Arc::from([9_u8, 8, 7, 6]);
        let materialized = Arc::clone(&payload);
        let weight = LazyWeight::new(
            onnx_runtime_ep_api::LazyWeightBoundary::MatMulNBits,
            DataType::Uint8,
            vec![4],
            vec![onnx_runtime_ep_api::ExternalMmapRegion {
                mapping_id: 0,
                offset: 0,
                len: 4,
            }],
            move || {
                onnx_runtime_ep_api::ResidentWeight::new(
                    DataType::Uint8,
                    vec![4],
                    Arc::clone(&materialized),
                )
            },
        )
        .unwrap();

        residency.prefetch_materialized(7, &weight).unwrap();
        let prefetched = residency.stats();
        assert_eq!(
            prefetched.page_ins, 0,
            "prefetch holds the page in-flight; demand admission owns LRU accounting"
        );
        assert_eq!(prefetched.prefetch_issued, 1);
        assert_eq!(prefetched.prefetch_declined_guard, 0);
        let page = residency.resident_materialized(7, &weight).unwrap();
        let joined = residency.stats();
        assert_eq!(
            joined.page_ins, 1,
            "demand page-in must join the prefetch fence, not start a second copy"
        );
        assert_eq!(joined.prefetch_joined, 1);
        let mut observed = [0_u8; 4];
        unsafe { runtime.dtoh(&mut observed, page.ptr) }.unwrap();
        assert_eq!(&observed, payload.as_ref());
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
    fn prefetch_can_run_one_page_ahead_when_cache_is_full() {
        let Ok(runtime) = crate::runtime::CudaRuntime::new(0).map(std::sync::Arc::new) else {
            eprintln!("SKIPPED (no CUDA runtime): prefetch guard check did NOT run.");
            return;
        };
        let residency = CudaWeightResidency::new(Arc::clone(&runtime), 4).with_async_pagein(true);
        let payload: Arc<[u8]> = Arc::from([1_u8, 2, 3, 4]);
        let materialized = Arc::clone(&payload);
        let weight = LazyWeight::new(
            onnx_runtime_ep_api::LazyWeightBoundary::MatMulNBits,
            DataType::Uint8,
            vec![4],
            vec![onnx_runtime_ep_api::ExternalMmapRegion {
                mapping_id: 0,
                offset: 0,
                len: 4,
            }],
            move || {
                onnx_runtime_ep_api::ResidentWeight::new(
                    DataType::Uint8,
                    vec![4],
                    Arc::clone(&materialized),
                )
            },
        )
        .unwrap();
        let other_payload: Arc<[u8]> = Arc::from([5_u8, 6, 7, 8]);
        let other_materialized = Arc::clone(&other_payload);
        let other = LazyWeight::new(
            onnx_runtime_ep_api::LazyWeightBoundary::MatMulNBits,
            DataType::Uint8,
            vec![4],
            vec![onnx_runtime_ep_api::ExternalMmapRegion {
                mapping_id: 1,
                offset: 0,
                len: 4,
            }],
            move || {
                onnx_runtime_ep_api::ResidentWeight::new(
                    DataType::Uint8,
                    vec![4],
                    Arc::clone(&other_materialized),
                )
            },
        )
        .unwrap();

        let _resident = residency.resident_materialized(1, &weight).unwrap();
        residency.prefetch_materialized(2, &other).unwrap();
        let prefetched = residency.stats();
        assert_eq!(prefetched.prefetch_issued, 1);
        assert_eq!(prefetched.page_ins, 1, "the original resident page remains");
        assert_eq!(prefetched.evictions, 0, "prefetch itself must not evict");

        let _joined = residency.resident_materialized(2, &other).unwrap();
        let joined = residency.stats();
        assert_eq!(joined.prefetch_joined, 1);
        assert_eq!(joined.page_ins, 2);
        assert_eq!(
            joined.evictions, 0,
            "the held first page is not evictable while the test keeps its Arc"
        );
    }
}
