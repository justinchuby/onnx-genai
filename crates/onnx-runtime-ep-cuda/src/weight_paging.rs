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
/// Page-ins served through the asynchronous copy-stream path (#87 Increment 1).
/// A subset of `GLOBAL_PAGE_INS`; lets an opaque end-to-end decode confirm the
/// async prefetch path actually ran rather than silently falling back to sync.
static GLOBAL_ASYNC_PAGE_INS: AtomicU64 = AtomicU64::new(0);
/// Look-ahead prefetches issued (#87 Increment 2): a successor weight's H2D copy
/// started on the copy stream *before* its consuming kernel is reached, so it can
/// overlap the current layer's compute. A subset of `GLOBAL_ASYNC_PAGE_INS`; a
/// token-parity test asserts it is > 0 to prove look-ahead actually ran.
static GLOBAL_LOOK_AHEAD_PREFETCHES: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the process-global weight-offload counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalOffloadStats {
    pub page_ins: u64,
    pub hits: u64,
    pub evictions: u64,
    /// Page-ins that used the async copy-stream + fence path (⊆ `page_ins`).
    pub async_page_ins: u64,
    /// Look-ahead successor prefetches issued (⊆ `async_page_ins`, #87 Inc 2).
    pub look_ahead_prefetches: u64,
}

/// Read the process-global weight-offload counters.
pub fn global_offload_stats() -> GlobalOffloadStats {
    GlobalOffloadStats {
        page_ins: GLOBAL_PAGE_INS.load(Ordering::Relaxed),
        hits: GLOBAL_HITS.load(Ordering::Relaxed),
        evictions: GLOBAL_EVICTIONS.load(Ordering::Relaxed),
        async_page_ins: GLOBAL_ASYNC_PAGE_INS.load(Ordering::Relaxed),
        look_ahead_prefetches: GLOBAL_LOOK_AHEAD_PREFETCHES.load(Ordering::Relaxed),
    }
}

/// Reset the process-global weight-offload counters (test observability helper).
pub fn reset_global_offload_stats() {
    GLOBAL_PAGE_INS.store(0, Ordering::Relaxed);
    GLOBAL_HITS.store(0, Ordering::Relaxed);
    GLOBAL_EVICTIONS.store(0, Ordering::Relaxed);
    GLOBAL_ASYNC_PAGE_INS.store(0, Ordering::Relaxed);
    GLOBAL_LOOK_AHEAD_PREFETCHES.store(0, Ordering::Relaxed);
}

/// Environment switch that enables the CUDA device residency cache. Reuses the
/// same knob as the CPU host-cache offload path (`onnx_runtime_ep_cpu`) so a
/// single `ONNX_GENAI_WEIGHT_OFFLOAD=1` turns offload on for whichever EP runs.
pub const WEIGHT_OFFLOAD_ENV: &str = onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV;

/// VRAM budget (bytes) for the device residency cache. When unset the residency
/// manager is constructed with a caller-chosen default.
pub const WEIGHT_OFFLOAD_DEVICE_BYTES_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES";

/// Environment switch that routes device page-ins through the asynchronous
/// copy-stream + fence path ([`CudaRuntime::htod_async`]) instead of the
/// synchronous [`CudaRuntime::htod`], so a page-in overlaps prior compute (#87
/// Increment 1). Shares the env *name* with the CPU host-cache prefetch knob,
/// but the device path is **opt-in**: unset keeps the byte-identical synchronous
/// page-in, and only `ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH=1` enables async. This
/// conservative default lets the async overlap be A/B'd against the shipped
/// synchronous path without changing the default behavior.
pub const WEIGHT_OFFLOAD_PREFETCH_ENV: &str = onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_PREFETCH_ENV;

/// Look-ahead depth for double-buffered prefetch (#87 Increment 2): how many
/// *upcoming* weights (in the learned decode access order) to page in on the copy
/// stream ahead of the kernel that consumes them, so their H2D overlaps the
/// current layer's compute. Only consulted when [`WEIGHT_OFFLOAD_PREFETCH_ENV`] is
/// `1`. Defaults to [`DEFAULT_PREFETCH_DEPTH`] when unset; `0` degrades to the
/// Increment-1 single-buffer async path (no look-ahead).
pub const WEIGHT_OFFLOAD_PREFETCH_DEPTH_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH_DEPTH";

/// Default look-ahead depth when async prefetch is enabled but the depth knob is
/// unset. Two lets the next weight's transfer overlap the current compute while
/// keeping the extra resident VRAM bounded (current + look-ahead set).
pub const DEFAULT_PREFETCH_DEPTH: usize = 2;

/// Whether/how the CUDA EP should page offloaded weights into a bounded VRAM
/// residency cache. Disabled by default so the resident fast path is untouched
/// and byte-identical.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceOffloadPolicy {
    pub enabled: bool,
    /// Explicit VRAM budget in bytes, if the operator pinned one.
    pub device_budget_bytes: Option<u64>,
    /// Route page-ins through the async copy stream + fence (opt-in, #87 Inc 1).
    pub prefetch: bool,
    /// Look-ahead depth for double-buffered prefetch (#87 Inc 2). `0` = the
    /// Increment-1 single-buffer async path (no look-ahead). Only meaningful when
    /// `prefetch` is set.
    pub prefetch_depth: usize,
}

impl DeviceOffloadPolicy {
    /// Read the policy from the process environment.
    pub fn from_env() -> Self {
        let enabled = std::env::var_os(WEIGHT_OFFLOAD_ENV).is_some_and(|value| value == "1");
        let device_budget_bytes = std::env::var(WEIGHT_OFFLOAD_DEVICE_BYTES_ENV)
            .ok()
            .and_then(|value| parse_budget_bytes(&value));
        // Opt-in: only an explicit `=1` enables async page-in; unset/anything
        // else keeps the synchronous, byte-identical path.
        let prefetch =
            std::env::var_os(WEIGHT_OFFLOAD_PREFETCH_ENV).is_some_and(|value| value == "1");
        // Look-ahead depth is only consulted when async prefetch is on. An
        // explicit value (including 0) wins; unset defaults to DEFAULT_PREFETCH_DEPTH.
        let prefetch_depth = if prefetch {
            std::env::var(WEIGHT_OFFLOAD_PREFETCH_DEPTH_ENV)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(DEFAULT_PREFETCH_DEPTH)
        } else {
            0
        };
        Self {
            enabled,
            device_budget_bytes,
            prefetch,
            prefetch_depth,
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
        unsafe { runtime.htod(bytes, ptr) }
            .map_err(|error| WeightHandleError::DeviceBinding(format!("H2D copy: {error}")))?;
        Ok(page)
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
pub struct CudaWeightPager<'a, S: MmapRegionSource> {
    runtime: Arc<CudaRuntime>,
    source: &'a S,
}

impl<'a, S: MmapRegionSource> CudaWeightPager<'a, S> {
    pub fn new(runtime: Arc<CudaRuntime>, source: &'a S) -> Self {
        Self { runtime, source }
    }
}

impl<S: MmapRegionSource> LazyDeviceWeightBinder for CudaWeightPager<'_, S> {
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
            unsafe { self.runtime.htod(bytes, dst) }
                .map_err(|error| WeightHandleError::DeviceBinding(format!("H2D copy: {error}")))?;
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
pub struct CudaWeightResidency {
    runtime: Arc<CudaRuntime>,
    /// When set, page-ins stream through the async copy stream + fence
    /// ([`CudaRuntime::htod_async`]) so a transfer overlaps prior compute; when
    /// clear, the synchronous [`CudaWeightPage::upload`] path is used (default,
    /// byte-identical). See [`WEIGHT_OFFLOAD_PREFETCH_ENV`].
    prefetch: bool,
    /// Look-ahead depth (#87 Inc 2): number of upcoming weights (in the learned
    /// decode access order) to prefetch on the copy stream ahead of consumption.
    /// `0` = Increment-1 single-buffer async (no look-ahead). Only used when
    /// `prefetch` is set.
    depth: usize,
    inner: Mutex<ResidencyInner>,
}

struct ResidencyInner {
    budget: u64,
    resident_bytes: u64,
    peak_resident_bytes: u64,
    page_ins: u64,
    hits: u64,
    evictions: u64,
    /// LRU order: front = least-recently-used, back = most-recently-used.
    order: Vec<u64>,
    pages: HashMap<u64, Arc<CudaWeightPage>>,
    /// Ring of reusable pinned host staging buffers for async copies. Increment 1
    /// used a single buffer (serialized page-ins); Increment 2 look-ahead can have
    /// up to `depth` transfers in flight, so the ring holds `depth + 1` buffers
    /// and a copy never overwrites a buffer whose prior DMA has not completed
    /// (WAR, enforced by `ring_fence`). Grown on demand to the largest weight.
    staging: Vec<Option<PinnedStaging>>,
    /// Per-ring-slot copy completion fence: before the host refills slot `i`, it
    /// waits on `ring_fence[i]` so it never overwrites bytes an in-flight DMA is
    /// still reading. `0` = already-signalled (slot idle).
    ring_fence: Vec<u64>,
    /// Next ring slot to hand out (round-robin).
    ring_pos: usize,
    /// Learned autoregressive access order → look-ahead successors. Decode replays
    /// the same weight sequence every token, so the first pass records the order
    /// and every subsequent access knows which weights come next.
    lazy_by_key: HashMap<u64, LazyWeight>,
    /// Access order recorded during the first (still-`recording`) pass.
    trace: Vec<u64>,
    /// Successor keys for each key, in access order (up to a useful look-ahead).
    successors: HashMap<u64, Vec<u64>>,
    /// Whether the access trace is still being recorded (first pass).
    recording: bool,
    /// Key → outstanding look-ahead copy fence, awaited on the compute stream when
    /// the weight is finally consumed (RAW) and cleared. A prefetched page is only
    /// safe to free once this fence is drained (handled by the miss path's copy
    /// stream sync before any eviction).
    pending_copy: HashMap<u64, u64>,
}

impl CudaWeightResidency {
    /// Build a residency cache with an explicit VRAM `budget_bytes`. Page-ins are
    /// synchronous (byte-identical fast path).
    pub fn new(runtime: Arc<CudaRuntime>, budget_bytes: u64) -> Self {
        Self::new_with_prefetch_depth(runtime, budget_bytes, false, 0)
    }

    /// Build a residency cache, choosing between the synchronous page-in path and
    /// the asynchronous copy-stream + fence page-in path (#87 Increment 1). The
    /// async path is transparent: output is byte-identical, only the transfer is
    /// overlapped with compute. Look-ahead depth is `0` (single-buffer).
    pub fn new_with_prefetch(runtime: Arc<CudaRuntime>, budget_bytes: u64, prefetch: bool) -> Self {
        Self::new_with_prefetch_depth(runtime, budget_bytes, prefetch, 0)
    }

    /// Build a residency cache with async page-in and a look-ahead `depth` (#87
    /// Increment 2). When `depth >= 1` and `prefetch` is set, serving a weight also
    /// issues copy-stream page-ins for the next `depth` weights in the learned
    /// decode order, so their H2D overlaps the current layer's compute. `depth = 0`
    /// is the Increment-1 single-buffer path. Both are output-identical to the
    /// synchronous path; only transfer scheduling differs.
    pub fn new_with_prefetch_depth(
        runtime: Arc<CudaRuntime>,
        budget_bytes: u64,
        prefetch: bool,
        depth: usize,
    ) -> Self {
        // Ring size: `depth + 1` pinned buffers so a look-ahead wave of `depth`
        // in-flight copies never reuses a buffer whose DMA is still reading.
        let ring = depth.saturating_add(1).max(1);
        Self {
            runtime,
            prefetch,
            depth,
            inner: Mutex::new(ResidencyInner {
                budget: budget_bytes,
                resident_bytes: 0,
                peak_resident_bytes: 0,
                page_ins: 0,
                hits: 0,
                evictions: 0,
                order: Vec::new(),
                pages: HashMap::new(),
                staging: (0..ring).map(|_| None).collect(),
                ring_fence: vec![0; ring],
                ring_pos: 0,
                lazy_by_key: HashMap::new(),
                trace: Vec::new(),
                successors: HashMap::new(),
                recording: true,
                pending_copy: HashMap::new(),
            }),
        }
    }

    /// Whether page-ins use the asynchronous copy-stream path.
    pub fn prefetch_enabled(&self) -> bool {
        self.prefetch
    }

    /// Look-ahead depth (0 when disabled or single-buffer).
    pub fn prefetch_depth(&self) -> usize {
        self.depth
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
        if let Some(hit) = self.get_hit(key)? {
            return Ok(hit);
        }
        // Copy region bytes host→device before re-locking so a failed bind never
        // mutates cache accounting.
        let pager = CudaWeightPager::new(Arc::clone(&self.runtime), source);
        let page = Arc::new(pager.bind_block_quantized_moe(weight)?);
        self.admit(key, page)
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
        // Learn the autoregressive access order so look-ahead knows what comes
        // next. Cheap and idempotent; only used when look-ahead is active.
        if self.prefetch && self.depth >= 1 {
            self.record_access(key, weight);
        }
        if let Some(hit) = self.get_hit(key)? {
            // The upcoming weights may not be resident yet — issue their copies now
            // so they overlap this layer's (and following layers') compute.
            self.prefetch_ahead(key)?;
            return Ok(hit);
        }
        let resident = weight.materialize()?;
        let page = if self.prefetch {
            self.admit_async(
                key,
                resident.dtype,
                resident.shape.clone(),
                resident.bytes(),
            )?
        } else {
            let page = Arc::new(CudaWeightPage::upload(
                &self.runtime,
                resident.dtype,
                resident.shape.clone(),
                resident.bytes(),
            )?);
            self.admit(key, page)?
        };
        // After paging in the current (missed) weight, warm the look-ahead set.
        self.prefetch_ahead(key)?;
        Ok(page)
    }

    /// Record `key` in the learned decode access order and cache a cheap clone of
    /// its [`LazyWeight`] so a later access can materialize it as a look-ahead
    /// successor without the dispatcher handing it to us again. The first pass
    /// records the full order; once a key repeats (the decode sequence is periodic
    /// per token) the successor map is frozen.
    fn record_access(&self, key: u64, weight: &LazyWeight) {
        let mut inner = self.lock();
        inner
            .lazy_by_key
            .entry(key)
            .or_insert_with(|| weight.clone());
        if !inner.recording {
            return;
        }
        // First pass: record the order until a key repeats (decode replays the
        // same weight sequence every token, so the first repeat closes one period).
        if inner.trace.contains(&key) {
            inner.recording = false;
            let depth = self.depth;
            let trace = std::mem::take(&mut inner.trace);
            for i in 0..trace.len() {
                let k = trace[i];
                let mut succ = Vec::with_capacity(depth);
                // Look ahead cyclically so the last weights of a step prefetch the
                // first weights of the next step.
                let mut j = 1;
                while succ.len() < depth && j <= trace.len() {
                    let cand = trace[(i + j) % trace.len()];
                    if cand != k && !succ.contains(&cand) {
                        succ.push(cand);
                    }
                    j += 1;
                }
                inner.successors.insert(k, succ);
            }
        } else {
            inner.trace.push(key);
        }
    }

    /// Issue look-ahead page-ins for the weights that follow `key` in the learned
    /// decode order, up to `depth`, for those not already resident/in-flight. Each
    /// is admitted through [`prefetch_admit`], which evicts only cold pages and
    /// declines when the protected working set leaves no room (budget→depth
    /// fallback to single-buffer), so look-ahead never thrashes a tight budget.
    fn prefetch_ahead(&self, key: u64) -> Result<(), WeightHandleError> {
        if !self.prefetch || self.depth == 0 {
            return Ok(());
        }
        let successors: Vec<u64> = {
            let inner = self.lock();
            inner
                .successors
                .get(&key)
                .map(|succ| succ.to_vec())
                .unwrap_or_default()
        };
        for succ in successors {
            // Materialize outside the lock is not possible (we need the cached
            // handle under the lock); grab the handle, then materialize, then admit.
            let handle = {
                let inner = self.lock();
                if inner.pages.contains_key(&succ) || inner.pending_copy.contains_key(&succ) {
                    None
                } else {
                    inner.lazy_by_key.get(&succ).cloned()
                }
            };
            let Some(handle) = handle else {
                continue;
            };
            let resident = handle.materialize()?;
            self.prefetch_admit(
                succ,
                resident.dtype,
                resident.shape.clone(),
                resident.bytes(),
            )?;
        }
        Ok(())
    }

    /// Look up `key`, marking it most-recently-used and counting a hit. If the page
    /// was brought in by a look-ahead prefetch whose copy fence has not yet been
    /// awaited, order the consuming kernel after that copy (RAW) and clear the
    /// pending fence so the page becomes an ordinary resident page.
    fn get_hit(&self, key: u64) -> Result<Option<Arc<CudaWeightPage>>, WeightHandleError> {
        let (page, pending_fence) = {
            let mut inner = self.lock();
            let Some(page) = inner.pages.get(&key).cloned() else {
                return Ok(None);
            };
            inner.touch(key);
            inner.hits += 1;
            GLOBAL_HITS.fetch_add(1, Ordering::Relaxed);
            (page, inner.pending_copy.remove(&key))
        };
        if let Some(fence) = pending_fence {
            // Non host-blocking cross-stream wait: the consumer kernel (later on the
            // compute stream) observes the fully-transferred bytes, while the copy
            // itself already overlapped earlier compute.
            self.runtime.compute_wait_fence(fence).map_err(|error| {
                WeightHandleError::DeviceBinding(format!("await prefetch fence: {error}"))
            })?;
        }
        Ok(Some(page))
    }

    /// Insert a freshly paged-in `page` under `key`, evicting LRU pages to fit the
    /// budget. Synchronizes the compute stream first so no in-flight kernel still
    /// references an about-to-be-freed page's VRAM. (Offload is mutually exclusive
    /// with CUDA graph capture, so this sync is never capture-illegal.)
    fn admit(
        &self,
        key: u64,
        page: Arc<CudaWeightPage>,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        let bytes = page.len() as u64;
        // Weight offload and CUDA graph capture are mutually exclusive: the decode
        // session declines capture whenever offload is enabled (see
        // `resolve_graph_capture_enabled`), so paging never runs during capture.
        // The synchronize below — and the alloc/copy/free it guards against — are
        // themselves capture-illegal, so we can unconditionally sync here.
        self.runtime
            .synchronize()
            .map_err(|error| WeightHandleError::DeviceBinding(format!("stream sync: {error}")))?;
        // When async page-ins are in use, an evicted page's inbound H2D copy runs
        // on the *transfer* stream, which the compute-stream sync above does not
        // cover. Drain it too so eviction never frees VRAM a copy is still writing
        // into (use-after-free). No-op when no async copy is outstanding.
        if self.prefetch {
            self.runtime.sync_copy_stream().map_err(|error| {
                WeightHandleError::DeviceBinding(format!("transfer stream sync: {error}"))
            })?;
        }
        let mut inner = self.lock();
        // A concurrent caller may have populated `key` while we paged in; prefer
        // the already-resident page and drop ours (its Drop frees the VRAM).
        if let Some(existing) = inner.pages.get(&key).cloned() {
            inner.touch(key);
            inner.hits += 1;
            GLOBAL_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(existing);
        }
        inner.evict_to_fit(bytes, &self.runtime);
        inner.pages.insert(key, Arc::clone(&page));
        inner.order.push(key);
        inner.resident_bytes += bytes;
        inner.peak_resident_bytes = inner.peak_resident_bytes.max(inner.resident_bytes);
        inner.page_ins += 1;
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
        Ok(page)
    }

    /// Asynchronous page-in for a *consumed-now* weight (miss path). Same as the
    /// synchronous path but the H2D runs on the transfer stream and the consuming
    /// kernel is ordered after it with a fence rather than a host block, so the
    /// copy overlaps compute already queued ahead of the consumer (#87 Increment 1).
    ///
    /// Correctness / hazard handling (the one real risk of async offload):
    /// * **RAW** — the copy fence is recorded on the transfer stream and the
    ///   compute stream is made to wait on it (`compute_wait_fence`), so the
    ///   consuming kernel never reads bytes the DMA is still transferring.
    /// * **WAR on the pinned ring** — [`issue_ring_copy`] host-waits a slot's prior
    ///   copy fence before refilling it, so no in-flight DMA is still reading the
    ///   bytes being overwritten.
    /// * **Use-after-free on eviction** — the compute + transfer stream drains
    ///   before `evict_to_fit` guarantee no kernel or copy is still touching an
    ///   allocation handed back to the driver. The transfer drain retires *all*
    ///   outstanding look-ahead copies, so an evicted prefetched page is safe too.
    fn admit_async(
        &self,
        key: u64,
        dtype: DataType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        if bytes.is_empty() {
            return Err(WeightHandleError::MissingRegions);
        }
        let len = bytes.len();
        let mut inner = self.lock();
        // Concurrent-populate check: another caller may have paged `key` in.
        if let Some(existing) = inner.pages.get(&key).cloned() {
            inner.touch(key);
            inner.hits += 1;
            GLOBAL_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(existing);
        }
        // Order device work before mutating VRAM: the compute sync retires kernels
        // that read soon-to-be-evicted pages; the transfer sync retires every
        // outstanding async copy (including in-flight look-ahead prefetches) so
        // evicting/freeing is safe. Both precede the new copy, so this page-in's
        // own transfer is never drained by itself and stays overlappable.
        self.runtime
            .synchronize()
            .map_err(|error| WeightHandleError::DeviceBinding(format!("stream sync: {error}")))?;
        self.runtime.sync_copy_stream().map_err(|error| {
            WeightHandleError::DeviceBinding(format!("transfer stream sync: {error}"))
        })?;
        inner.evict_to_fit(len as u64, &self.runtime);
        let ptr = self
            .runtime
            .alloc_raw(len)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        let page = Arc::new(CudaWeightPage {
            runtime: Arc::clone(&self.runtime),
            ptr,
            len,
            dtype,
            shape,
        });
        // Stage into the pinned ring and launch the async H2D; `page`'s Drop frees
        // `ptr` if staging/copy fails. Consumed immediately, so await the fence now.
        let raw_fence = self.issue_ring_copy(&mut inner, ptr, bytes)?;
        self.runtime
            .compute_wait_fence(raw_fence)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("await fence: {error}")))?;
        inner.pages.insert(key, Arc::clone(&page));
        inner.order.push(key);
        inner.resident_bytes += len as u64;
        inner.peak_resident_bytes = inner.peak_resident_bytes.max(inner.resident_bytes);
        inner.page_ins += 1;
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_ASYNC_PAGE_INS.fetch_add(1, Ordering::Relaxed);
        Ok(page)
    }

    /// Look-ahead page-in (#87 Increment 2): stage a *not-yet-consumed* successor
    /// weight onto the transfer stream so its H2D overlaps the current layer's
    /// compute. The copy fence is *deferred*: recorded now and awaited on the
    /// compute stream only when the weight is finally consumed ([`get_hit`]), which
    /// is what lets the transfer run concurrently with the intervening compute.
    ///
    /// To make room in a full cache it may evict the LRU cold page, so it takes the
    /// same safety drains as the miss path ([`admit_async`]): the compute stream is
    /// synchronized (no kernel still reads an about-to-be-freed page — RAW/WAR/UAF
    /// on compute) and the transfer stream is drained (retires *every* outstanding
    /// look-ahead copy, so an evicted page never has an in-flight DMA writing it —
    /// the multi-fence async-eviction hazard). The one-time sync cost is already
    /// paid by the miss path at the tight budgets where offload runs.
    ///
    /// **Budget→depth policy (thrash avoidance):** prefetch only proceeds when the
    /// *protected* working set — pages currently referenced by a live kernel plus
    /// pages already prefetched-in-flight this round — leaves room for this page
    /// (`protected + len <= budget`). Eviction then only reclaims genuinely cold,
    /// unprotected pages. When the budget cannot hold the working set alongside the
    /// look-ahead page the successor is skipped, degrading gracefully to the
    /// single-buffer (#87 Increment 1) behaviour rather than thrashing.
    fn prefetch_admit(
        &self,
        key: u64,
        dtype: DataType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<(), WeightHandleError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let len = bytes.len();
        let mut inner = self.lock();
        if inner.pages.contains_key(&key) || inner.pending_copy.contains_key(&key) {
            return Ok(());
        }
        // Budget→depth policy: prefetch only if the *protected* working set (pages
        // a live kernel still references, plus in-flight look-ahead pages) leaves
        // room for this page. Otherwise skip → graceful fallback to single-buffer
        // rather than evicting a page we still need this round (thrash).
        let protected: u64 = inner
            .pages
            .iter()
            .filter(|(k, page)| Arc::strong_count(page) > 1 || inner.pending_copy.contains_key(*k))
            .map(|(_, page)| page.len() as u64)
            .sum();
        if protected.saturating_add(len as u64) > inner.budget {
            return Ok(());
        }
        // Making room may evict a cold page, so drain device work first (same
        // guarantees as the miss path): compute sync retires kernels reading an
        // evicted page; transfer sync retires every outstanding look-ahead copy so
        // no in-flight DMA is writing an evicted/reused allocation.
        self.runtime
            .synchronize()
            .map_err(|error| WeightHandleError::DeviceBinding(format!("stream sync: {error}")))?;
        self.runtime.sync_copy_stream().map_err(|error| {
            WeightHandleError::DeviceBinding(format!("transfer stream sync: {error}"))
        })?;
        // Evict only cold, unprotected pages; if that still cannot free enough,
        // skip this prefetch rather than exceed the VRAM budget.
        inner.evict_cold_to_fit(len as u64);
        if inner.resident_bytes.saturating_add(len as u64) > inner.budget {
            return Ok(());
        }
        let ptr = self
            .runtime
            .alloc_raw(len)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        let page = Arc::new(CudaWeightPage {
            runtime: Arc::clone(&self.runtime),
            ptr,
            len,
            dtype,
            shape,
        });
        let raw_fence = self.issue_ring_copy(&mut inner, ptr, bytes)?;
        // Defer the RAW wait to consumption; keep the page resident meanwhile.
        inner.pending_copy.insert(key, raw_fence);
        inner.pages.insert(key, page);
        // Insert at the most-recently-used end so LRU eviction reclaims genuinely
        // cold pages before a freshly prefetched (soon-to-be-consumed) one.
        inner.order.push(key);
        inner.resident_bytes += len as u64;
        inner.peak_resident_bytes = inner.peak_resident_bytes.max(inner.resident_bytes);
        inner.page_ins += 1;
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_ASYNC_PAGE_INS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_LOOK_AHEAD_PREFETCHES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Stage `bytes` into the next pinned ring slot and launch the async H2D into
    /// `ptr`, returning the copy's RAW fence (for the consumer to wait on). WAR on
    /// the ring is enforced here: before refilling a slot the host waits on that
    /// slot's *previous* copy fence, so it never overwrites bytes an in-flight DMA
    /// is still reading. Two fences are recorded per copy — one dedicated to the
    /// slot's WAR throttle (host-waited on the slot's next reuse) and one for the
    /// consumer's RAW ordering — because a fence event is single-use (consumed when
    /// awaited). Caller holds the residency lock.
    fn issue_ring_copy(
        &self,
        inner: &mut ResidencyInner,
        ptr: CUdeviceptr,
        bytes: &[u8],
    ) -> Result<u64, WeightHandleError> {
        let len = bytes.len();
        let ring = inner.staging.len();
        let slot = inner.ring_pos;
        inner.ring_pos = (inner.ring_pos + 1) % ring;
        // WAR: the slot's prior DMA must finish before the host overwrites it.
        let prior = std::mem::replace(&mut inner.ring_fence[slot], 0);
        if prior != 0 {
            self.runtime.host_wait_fence(prior).map_err(|error| {
                WeightHandleError::DeviceBinding(format!("ring WAR wait: {error}"))
            })?;
        }
        if inner.staging[slot]
            .as_ref()
            .is_none_or(|buf| buf.len() < len)
        {
            inner.staging[slot] = None;
            let buffer = self.runtime.alloc_pinned(len).map_err(|error| {
                WeightHandleError::DeviceBinding(format!("pinned staging alloc: {error}"))
            })?;
            inner.staging[slot] = Some(buffer);
        }
        let buffer = inner.staging[slot]
            .as_mut()
            .expect("staging buffer present after ensure");
        buffer.as_mut_slice()[..len].copy_from_slice(bytes);
        // SAFETY: `ptr` owns `len` freshly-allocated bytes; the pinned source stays
        // alive in `inner.staging[slot]` and is not overwritten until this slot's
        // next reuse first host-waits the WAR fence recorded just below.
        unsafe { self.runtime.htod_async(&buffer.as_slice()[..len], ptr) }.map_err(|error| {
            WeightHandleError::DeviceBinding(format!("async H2D copy: {error}"))
        })?;
        inner.ring_fence[slot] = self.runtime.record_copy_fence().map_err(|error| {
            WeightHandleError::DeviceBinding(format!("record WAR fence: {error}"))
        })?;
        let raw_fence = self
            .runtime
            .record_copy_fence()
            .map_err(|error| WeightHandleError::DeviceBinding(format!("record fence: {error}")))?;
        Ok(raw_fence)
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

    /// Evict least-recently-used, currently-unreferenced pages until admitting
    /// `incoming` bytes fits the budget (best effort; stops when nothing more is
    /// evictable). Callers must drain the transfer stream first, so any evicted
    /// page's outstanding look-ahead copy has already retired; here we simply drop
    /// its now-complete pending fence so the one-shot event is released.
    fn evict_to_fit(&mut self, incoming: u64, runtime: &Arc<CudaRuntime>) {
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
                if let Some(fence) = self.pending_copy.remove(&key) {
                    // The caller drained the transfer stream, so this copy is done;
                    // release its one-shot event (no-op if already retired).
                    let _ = runtime.host_wait_fence(fence);
                }
                self.order.remove(index);
                // Do not advance `index`: the vector shifted left under it.
            } else {
                index += 1;
            }
        }
    }

    /// Like [`evict_to_fit`] but only reclaims *cold* pages — never one with an
    /// outstanding look-ahead copy (`pending_copy`), since that page was prefetched
    /// precisely because it is needed soon. Used by the look-ahead path so a
    /// prefetch never evicts a sibling prefetch issued earlier in the same round.
    /// Stops (leaving the cache over budget) when only protected pages remain; the
    /// caller then declines the prefetch rather than exceed VRAM.
    fn evict_cold_to_fit(&mut self, incoming: u64) {
        let mut index = 0;
        while self.resident_bytes.saturating_add(incoming) > self.budget && index < self.order.len()
        {
            let key = self.order[index];
            let evictable = !self.pending_copy.contains_key(&key)
                && self
                    .pages
                    .get(&key)
                    .is_some_and(|page| Arc::strong_count(page) == 1);
            if evictable {
                if let Some(page) = self.pages.remove(&key) {
                    self.resident_bytes = self.resident_bytes.saturating_sub(page.len() as u64);
                    self.evictions += 1;
                    GLOBAL_EVICTIONS.fetch_add(1, Ordering::Relaxed);
                }
                self.order.remove(index);
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
}
