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

/// Snapshot of the process-global weight-offload counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalOffloadStats {
    pub page_ins: u64,
    pub hits: u64,
    pub evictions: u64,
}

/// Read the process-global weight-offload counters.
pub fn global_offload_stats() -> GlobalOffloadStats {
    GlobalOffloadStats {
        page_ins: GLOBAL_PAGE_INS.load(Ordering::Relaxed),
        hits: GLOBAL_HITS.load(Ordering::Relaxed),
        evictions: GLOBAL_EVICTIONS.load(Ordering::Relaxed),
    }
}

/// Reset the process-global weight-offload counters (test observability helper).
pub fn reset_global_offload_stats() {
    GLOBAL_PAGE_INS.store(0, Ordering::Relaxed);
    GLOBAL_HITS.store(0, Ordering::Relaxed);
    GLOBAL_EVICTIONS.store(0, Ordering::Relaxed);
}

/// Environment switch that enables the CUDA device residency cache. Reuses the
/// same knob as the CPU host-cache offload path (`onnx_runtime_ep_cpu`) so a
/// single `ONNX_GENAI_WEIGHT_OFFLOAD=1` turns offload on for whichever EP runs.
pub const WEIGHT_OFFLOAD_ENV: &str = onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV;

/// VRAM budget (bytes) for the device residency cache. When unset the residency
/// manager is constructed with a caller-chosen default.
pub const WEIGHT_OFFLOAD_DEVICE_BYTES_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES";

/// Sub-knob (default OFF / opt-IN) selecting the asynchronous, fence-ordered
/// residency page-in over the synchronous `cuMemcpyHtoD`. Set to a truthy value
/// (`1`/`true`/`yes`/`on`, case/whitespace-insensitive) to enable async page-in;
/// unset or any other value uses the synchronous page-in. Async is opt-in because
/// a measured A/B (qwen3-0.6b-int4, 96MiB budget, eviction/thrash regime) showed
/// sync is FASTER (15.84 vs 12.16 tok/s): when every admit evicts, the async path
/// cannot overlap and still pays a non-overlappable pinned-staging alloc+copy. The
/// async path stays fully intact behind this flag (the A/B "after" arm) and is
/// expected to become a net win once a warm-host materialize cache lands. This
/// knob only has any effect when weight offload itself is enabled
/// (`ONNX_GENAI_WEIGHT_OFFLOAD=1`); with offload off the resident fast path is
/// untouched and byte-identical regardless.
pub const WEIGHT_OFFLOAD_ASYNC_PAGEIN_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN";

/// Pure opt-in parse for [`WEIGHT_OFFLOAD_ASYNC_PAGEIN_ENV`]. Async page-in is
/// **default-off**: only an explicit truthy value (`1`/`true`/`yes`/`on`,
/// case/whitespace-insensitive) enables it; unset (`None`) or any other value
/// keeps the synchronous page-in.
pub(crate) fn async_pagein_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
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
    /// Use the asynchronous, fence-ordered page-in (default `false` / opt-IN).
    /// Enabled only when `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN` is set to a
    /// truthy value (`1`/`true`/`yes`/`on`); otherwise the synchronous page-in
    /// is used, which is faster in the eviction/thrash regime (measured A/B).
    pub async_pagein: bool,
}

impl DeviceOffloadPolicy {
    /// Read the policy from the process environment.
    pub fn from_env() -> Self {
        let enabled = std::env::var_os(WEIGHT_OFFLOAD_ENV).is_some_and(|value| value == "1");
        let device_budget_bytes = std::env::var(WEIGHT_OFFLOAD_DEVICE_BYTES_ENV)
            .ok()
            .and_then(|value| parse_budget_bytes(&value));
        // Async page-in defaults OFF (opt-in); only an explicit truthy value enables it.
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
    /// Owned pinned host staging that backed an asynchronous page-in, kept alive
    /// so an in-flight `cuMemcpyHtoDAsync` never reads freed host memory. `None`
    /// for pages uploaded synchronously (the source is fully consumed before
    /// `htod` returns). Freed on drop — long after the copy has completed.
    staging: Option<PinnedStaging>,
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
            staging: None,
        };
        // SAFETY: `ptr` owns `bytes.len()` bytes; `page`'s Drop frees it if the
        // copy below fails.
        unsafe { runtime.htod(bytes, ptr) }
            .map_err(|error| WeightHandleError::DeviceBinding(format!("H2D copy: {error}")))?;
        Ok(page)
    }

    /// Asynchronous, overlap-friendly variant of [`Self::upload`]: allocate a
    /// VRAM page and enqueue a host→device copy of `bytes` on the runtime's
    /// dedicated transfer stream, returning the page plus a **copy fence** the
    /// caller MUST order the consuming compute work after (via
    /// [`CudaRuntime::compute_wait_fence`]). Because the copy is asynchronous,
    /// the source bytes are first staged into a page-locked buffer that the
    /// returned [`CudaWeightPage`] owns, so the transfer keeps reading valid host
    /// memory until it completes — the staging outlives the page's whole life,
    /// which itself outlives any consumer ordered after the fence. Frees the VRAM
    /// (and staging) on any failure.
    pub fn upload_async(
        runtime: &Arc<CudaRuntime>,
        dtype: DataType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<(Self, u64), WeightHandleError> {
        if bytes.is_empty() {
            return Err(WeightHandleError::MissingRegions);
        }
        let ptr = runtime
            .alloc_raw(bytes.len())
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        let mut staging = runtime
            .alloc_pinned(bytes.len())
            .map_err(|error| WeightHandleError::DeviceBinding(format!("pinned alloc: {error}")))?;
        staging.as_mut_slice().copy_from_slice(bytes);
        // Own both the VRAM `ptr` and the pinned `staging` before enqueuing the
        // copy, so any error below drops `page` and frees both exactly once.
        let page = Self {
            runtime: Arc::clone(runtime),
            ptr,
            len: bytes.len(),
            dtype,
            shape,
            staging: Some(staging),
        };
        // SAFETY: `dst` (`ptr`) owns `len` bytes; the async source is the pinned
        // staging `page` owns, which outlives the transfer (freed only on the
        // page's Drop, well after the copy fence has been awaited).
        let staged = page
            .staging
            .as_ref()
            .expect("staging was just set")
            .as_slice();
        unsafe { runtime.htod_async(staged, ptr) }.map_err(|error| {
            WeightHandleError::DeviceBinding(format!("async H2D copy: {error}"))
        })?;
        let fence = runtime
            .record_copy_fence()
            .map_err(|error| WeightHandleError::DeviceBinding(format!("copy fence: {error}")))?;
        Ok((page, fence))
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
            staging: None,
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
///
/// Page-in defaults to the asynchronous, fence-ordered path
/// ([`CudaWeightPage::upload_async`]): the H2D copy is enqueued on the runtime's
/// dedicated transfer stream so it overlaps the in-flight compute, and the
/// consuming compute stream is ordered after it with a completion fence. The
/// full compute-stream drain is now taken **only when admitting must evict** (so
/// eviction never frees a page a prior kernel still reads); a page-in that fits
/// the budget no longer host-blocks, which is what lets the transfer overlap.
/// Set `async_pagein = true` (env `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=1`) to
/// opt into the asynchronous page-in for A/B comparison; the synchronous page-in
/// is the default (faster in the eviction/thrash regime, measured A/B).
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
    /// LRU order: front = least-recently-used, back = most-recently-used.
    order: Vec<u64>,
    pages: HashMap<u64, Arc<CudaWeightPage>>,
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

impl CudaWeightResidency {
    /// Build a residency cache with an explicit VRAM `budget_bytes`. Page-in is
    /// synchronous by default; chain [`Self::with_async_pagein`] to opt into async.
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
                order: Vec::new(),
                pages: HashMap::new(),
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
                order: Vec::new(),
                pages: HashMap::new(),
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
        let resident = weight.materialize()?;
        if self.async_pagein {
            // Async, fence-ordered page-in: enqueue the H2D on the transfer stream
            // (overlapping the in-flight compute), admit, then order the compute
            // stream after the transfer's completion event so the consuming kernel
            // waits only on the copy — never a full host sync.
            let (page, copy_fence) = CudaWeightPage::upload_async(
                &self.runtime,
                resident.dtype,
                resident.shape.clone(),
                resident.bytes(),
            )?;
            let admitted = self.admit(key, Arc::new(page))?;
            self.runtime
                .compute_wait_fence(copy_fence)
                .map_err(|error| {
                    WeightHandleError::DeviceBinding(format!("copy fence wait: {error}"))
                })?;
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
    /// current compute. (Offload is mutually exclusive with CUDA graph capture, so
    /// this sync is never capture-illegal.)
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
        self.runtime
            .synchronize()
            .map_err(|error| WeightHandleError::DeviceBinding(format!("stream sync: {error}")))?;
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
        // Async page-in is opt-in: the default policy uses the synchronous path.
        assert!(!policy.async_pagein);
    }

    #[test]
    fn async_pagein_env_is_opt_in() {
        // Unset => synchronous (default off).
        assert!(!async_pagein_from_env_value(None));
        // Truthy spellings enable async, case/whitespace-insensitive.
        assert!(async_pagein_from_env_value(Some("1")));
        assert!(async_pagein_from_env_value(Some("true")));
        assert!(async_pagein_from_env_value(Some("YES")));
        assert!(async_pagein_from_env_value(Some("  On ")));
        // Explicit falsey / anything else stays synchronous.
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
}
