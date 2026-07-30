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

use crate::runtime::{CudaRuntime, raw_ptr};

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

/// Whether/how the CUDA EP should page offloaded weights into a bounded VRAM
/// residency cache. Disabled by default so the resident fast path is untouched
/// and byte-identical.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceOffloadPolicy {
    pub enabled: bool,
    /// Explicit VRAM budget in bytes, if the operator pinned one.
    pub device_budget_bytes: Option<u64>,
}

impl DeviceOffloadPolicy {
    /// Read the policy from the process environment.
    pub fn from_env() -> Self {
        let enabled = std::env::var_os(WEIGHT_OFFLOAD_ENV).is_some_and(|value| value == "1");
        let device_budget_bytes = std::env::var(WEIGHT_OFFLOAD_DEVICE_BYTES_ENV)
            .ok()
            .and_then(|value| parse_budget_bytes(&value));
        Self {
            enabled,
            device_budget_bytes,
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
}

impl CudaWeightResidency {
    /// Build a residency cache with an explicit VRAM `budget_bytes`.
    pub fn new(runtime: Arc<CudaRuntime>, budget_bytes: u64) -> Self {
        Self {
            runtime,
            inner: Mutex::new(ResidencyInner {
                budget: budget_bytes,
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
        let page = Arc::new(CudaWeightPage::upload(
            &self.runtime,
            resident.dtype,
            resident.shape.clone(),
            resident.bytes(),
        )?);
        self.admit(key, page)
    }

    /// Look up `key`, marking it most-recently-used and counting a hit.
    fn get_hit(&self, key: u64) -> Option<Arc<CudaWeightPage>> {
        let mut inner = self.lock();
        if let Some(page) = inner.pages.get(&key).cloned() {
            inner.touch(key);
            inner.hits += 1;
            GLOBAL_HITS.fetch_add(1, Ordering::Relaxed);
            Some(page)
        } else {
            None
        }
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
        let mut inner = self.lock();
        // A concurrent caller may have populated `key` while we paged in; prefer
        // the already-resident page and drop ours (its Drop frees the VRAM).
        if let Some(existing) = inner.pages.get(&key).cloned() {
            inner.touch(key);
            inner.hits += 1;
            GLOBAL_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(existing);
        }
        inner.evict_to_fit(bytes);
        inner.pages.insert(key, Arc::clone(&page));
        inner.order.push(key);
        inner.resident_bytes += bytes;
        inner.peak_resident_bytes = inner.peak_resident_bytes.max(inner.resident_bytes);
        inner.page_ins += 1;
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
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
