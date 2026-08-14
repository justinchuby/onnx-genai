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

use cudarc::driver::sys;
use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    ExternalMmapRegion, LazyDeviceWeightBinder, LazyWeight, LazyWeightBoundary, MmapRegionSource,
    WeightHandleError,
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
// Weight **bytes** streamed H2D by bypassed page-ins (issue #837 item 3). Every
// bypass still runs the full host->device copy — so these bytes are already
// inside `GLOBAL_HTOD_BYTES` — but the page is handed back transiently and never
// joins the resident set, so the identical bytes are re-streamed on the next
// decode step. This is the byte-weighted attribution of the bypass count: it
// answers "how much of `htod_bytes` is bypass traffic that residency policy
// left on the table" directly, rather than inferring it from the ~11.9 MB
// average page-in size.
static GLOBAL_BYPASSED_PAGE_IN_BYTES: AtomicU64 = AtomicU64::new(0);
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
// Zero-copy hybrid counters (#864). A cold weight the hybrid declines to copy
// into VRAM is read in place from a `cuMemHostRegister(READ_ONLY|DEVICEMAP)`
// host mapping. `GLOBAL_ZERO_COPY_BINDS` counts the distinct cold weights bound
// this way (each registered once, never re-copied); `GLOBAL_ZERO_COPY_READS`
// and `GLOBAL_ZERO_COPY_BYTES` count every dispatch that reads such a weight,
// so `zero_copy_bytes / emitted_tokens` is the per-step PCIe traffic the cold
// fraction moves — the honest analogue of `htod_bytes_per_token` for the arm
// that never copies. `GLOBAL_HOST_REGISTERED_BYTES` is the page-locked host RAM
// the registrations claim (a live gauge, preserved across window resets).
static GLOBAL_ZERO_COPY_BINDS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_ZERO_COPY_READS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_ZERO_COPY_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_HOST_REGISTERED_BYTES: AtomicU64 = AtomicU64::new(0);
// Distinct bytes actually bound zero-copy (each cold weight counted once at
// bind, not per read). This is the per-step host-mapped read footprint, which
// the safety budget caps — see `ZERO_COPY_SAFE_BUDGET_BYTES`.
static GLOBAL_ZERO_COPY_BOUND_BYTES: AtomicU64 = AtomicU64::new(0);

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
    /// Weight bytes streamed H2D by bypassed page-ins. A subset of
    /// [`Self::htod_bytes`]: these bytes were copied device-ward but the page was
    /// not retained, so they are re-streamed every decode step. See
    /// [`Self::bypassed_byte_share`].
    pub bypassed_page_in_bytes: u64,
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
    /// Distinct cold weights bound zero-copy from host-mapped memory (#864).
    /// Each is `cuMemHostRegister`ed once and never copied into VRAM; a nonzero
    /// value means the zero-copy hybrid is engaged.
    pub zero_copy_binds: u64,
    /// Dispatch reads served from a host-mapped (zero-copy) weight page. Every
    /// decode step re-reads each cold weight over PCIe in place, so this grows
    /// once per cold-weight lookup per step.
    pub zero_copy_reads: u64,
    /// Weight **bytes** read in place from host-mapped memory. This is *not* part
    /// of [`Self::htod_bytes`] (no copy happened); it is the honest per-step PCIe
    /// traffic of the cold fraction. See [`Self::zero_copy_byte_hit_rate`].
    pub zero_copy_bytes: u64,
    /// Page-locked host RAM claimed by zero-copy registrations (a live gauge).
    pub host_registered_bytes: u64,
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

    /// Fraction of streamed weight **bytes** (`htod_bytes`) attributable to
    /// bypassed page-ins — page-ins copied device-ward but never retained, so
    /// they are re-streamed every decode step. This is the byte-weighted
    /// attribution of the bypass count (#837 item 3): a large share means the
    /// residency policy is spending H2D bandwidth on traffic it keeps no benefit
    /// from, and is the first thing to rule in or out when explaining the gap
    /// between `htod_bytes_per_token` and the streaming floor `W - B`.
    ///
    /// `None` when no bytes were streamed in the window.
    #[must_use]
    pub fn bypassed_byte_share(&self) -> Option<f64> {
        (self.htod_bytes > 0).then(|| self.bypassed_page_in_bytes as f64 / self.htod_bytes as f64)
    }

    /// Byte-weighted fraction of requested weight bytes served from **VRAM
    /// residency**, counting host-mapped zero-copy reads as *not* resident
    /// (#864). [`Self::byte_hit_rate`] divides only by copied bytes, so on the
    /// zero-copy hybrid — where the cold fraction is read in place and never
    /// copied — it would report ~100% and hide that the cold bytes still cross
    /// PCIe every step. This denominator adds `zero_copy_bytes` so the number
    /// reflects the true resident share: `hit_bytes / (hit_bytes + htod_bytes +
    /// zero_copy_bytes)`.
    ///
    /// `None` when no weight bytes were requested in the window.
    #[must_use]
    pub fn zero_copy_byte_hit_rate(&self) -> Option<f64> {
        let requested = self
            .hit_bytes
            .checked_add(self.htod_bytes)?
            .checked_add(self.zero_copy_bytes)?;
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
        bypassed_page_in_bytes: GLOBAL_BYPASSED_PAGE_IN_BYTES.load(Ordering::Relaxed),
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
        zero_copy_binds: GLOBAL_ZERO_COPY_BINDS.load(Ordering::Relaxed),
        zero_copy_reads: GLOBAL_ZERO_COPY_READS.load(Ordering::Relaxed),
        zero_copy_bytes: GLOBAL_ZERO_COPY_BYTES.load(Ordering::Relaxed),
        host_registered_bytes: GLOBAL_HOST_REGISTERED_BYTES.load(Ordering::Relaxed),
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
    GLOBAL_BYPASSED_PAGE_IN_BYTES.store(0, Ordering::Relaxed);
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
    // Per-window activity: cold zero-copy reads/bytes. `zero_copy_binds` and
    // `host_registered_bytes` are live gauges (the registrations survive a
    // window reset), so they are preserved exactly like the residency gauges.
    GLOBAL_ZERO_COPY_READS.store(0, Ordering::Relaxed);
    GLOBAL_ZERO_COPY_BYTES.store(0, Ordering::Relaxed);
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

/// Sub-knob (default OFF / opt-IN) selecting **byte-aware** residency on top of
/// scan-resistant dense residency. Set to `1`/`true`/`yes`/`on` to enable.
///
/// Scan-resistant `StableResident` makes a size-blind admission decision: once
/// the budget is full, every subsequent distinct tensor is streamed transiently
/// (a "bypass") regardless of its size, and the resident set stays whatever
/// first-fit tensors happened to land in the remaining headroom — which biases
/// toward *small* tensors, because a small tensor fits the leftover budget while
/// a large one does not. Measured on qwen14b-zp (#837 item 3), that leaves
/// bypasses at 11% of page-in *events* but **44.6% of streamed bytes** (avg
/// bypass 49.9 MB vs 7.8 MB for retained page-ins): the policy streams the large
/// projections transiently and re-streams them every decode step.
///
/// Byte-aware residency instead admits an incoming tensor into the resident set
/// (evicting the *smallest* evictable resident to make room) whenever it is
/// strictly larger than that smallest resident, and evicts smallest-first. This
/// converges the resident set to the top-`B`-bytes tensors, driving the
/// byte-weighted hit rate toward the `B/W` ceiling. Default OFF.
///
/// **EXPERIMENTAL — KNOWN UNSAFE, DO NOT ENABLE (#837 item 3).** A/B measurement
/// on qwen14b-zp (managed streaming, `ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1`)
/// shows that whenever this policy *actually engages* (offload active, i.e. the
/// only regime where it could help), it **violates the token-identity hard
/// constraint**: greedy decode collapses to 3 tokens instead of 16, both with
/// (`ONNX_GENAI_CUDA_GRAPH=1`) and without CUDA-graph capture.
///
/// The failure is **silent numeric corruption, not an admission error** — proven
/// by the profiler's error-propagation structure: `generate_with_callback` is
/// `?`-propagated with `.context("steady measured generation")` *before* the
/// `bail!("generation emitted N tokens")` check, so the observed `bail!` means
/// generation returned `Ok` with only 3 tokens (an early EOS), not that a
/// `WeightHandleError` surfaced. A residency policy decides *what is resident*,
/// not *what is computed*: both the resident-hit and bypass paths fill the
/// correct bytes at a stable per-tensor VA, and every eviction here only targets
/// `strong_count == 1` pages after draining **both** the compute and copy
/// streams. Under those guards a change of eviction *target* (smallest-by-bytes
/// instead of `next_evictable_index`'s front-of-order oldest page) should be
/// value-neutral. That it is not implicates state that depends on eviction
/// *order* — most plausibly the physical granule / retained-handle pool
/// accounting — rather than the policy decision itself. Reproduction with CUDA
/// graph OFF rules out captured-VA baking as the sole cause. Whether this is a
/// bug in the byte-aware admission loop or a **latent defect in the existing
/// offload path** (exposed, not caused, by reordering evictions) was not
/// isolated in #886 and was investigated separately in **#888 — resolved
/// below**.
///
/// **#888 resolution: it is the retain-vs-bypass *flip*, not eviction order.**
/// Byte-aware changes two independent things at once — it starts *retaining*
/// large tensors that the shipped path streams transiently, **and** it evicts
/// the *smallest* resident instead of the front-of-order oldest. These were
/// separated with [`WEIGHT_OFFLOAD_EVICT_ORDER_ENV`], which changes only the
/// eviction victim while keeping the shipped always-bypass decision. On
/// qwen14b-zp, both `mru` (reverse recency) **and** `smallest` (byte-aware's
/// exact victim, under 10,192 evictions) stay **byte-identical** to the LRU
/// baseline with clean ledgers — so decode correctness does **not** depend on
/// eviction order (explanation 1, not 2). The corruption is caused solely by
/// promoting a would-be-bypass tensor into a retained, stable-slot resident that
/// is then served as a *hit* (no re-fill) across steps. It is **not** a
/// copy/compute fence hazard ([`WEIGHT_OFFLOAD_SYNC_BEFORE_FILL_ENV`] draining
/// both streams before every fill does not fix it) and **not** captured-VA
/// baking (graph-OFF still corrupts). One concrete consistency bug in this
/// change was found and confirmed to occur — a *slotted* key that later bypasses
/// gets `stable_slot = true` yet never rejoins `pages`
/// ([`WEIGHT_OFFLOAD_RETAIN_SLOTTED_ENV`] closes it) — but closing it does
/// **not** stop the corruption, so the primary value-corruption path is deeper
/// in retaining/re-admitting large stable-slot tensors (granule-level checksums
/// across steps are the remaining decider). Since the shipped size-blind path
/// never retains large tensors, it is unaffected; a #864 hybrid that pins a
/// *static* hot set (retain once, never churn) avoids this path, whereas any
/// scheme that evicts and re-admits large stable-slot residents inherits the
/// same hazard.
///
/// The count-vs-byte residency gap therefore cannot be closed by an eviction-
/// order change alone; it needs the structural lever deferred by #837 (a
/// dedicated transient staging zone so a large tensor can be handed to the
/// kernel *without* evicting a resident page). Kept default-OFF and wired only
/// so the rejected approach and its evidence are reviewable; the default
/// (size-blind) path is unaffected.
pub const WEIGHT_OFFLOAD_BYTE_AWARE_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_BYTE_AWARE";

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

/// Parse [`WEIGHT_OFFLOAD_BYTE_AWARE_ENV`]. Byte-aware residency defaults **OFF**
/// (opt-in): only `1`/`true`/`yes`/`on` (case/whitespace-insensitive) enable it,
/// every other value — including unset — keeps the size-blind admission path.
pub(crate) fn byte_aware_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

/// Read [`WEIGHT_OFFLOAD_BYTE_AWARE_ENV`] from the process environment. Exposed
/// so the engine's memory-strategy policy can opt this experimental A/B knob in
/// without threading a new field through the whole runtime-application config.
#[must_use]
pub fn byte_aware_residency_from_env() -> bool {
    byte_aware_from_env_value(std::env::var(WEIGHT_OFFLOAD_BYTE_AWARE_ENV).ok().as_deref())
}

/// Sub-knob (default OFF / opt-IN) selecting the **zero-copy hybrid** (#864). On
/// top of the size-blind `StableResident` residency, the cold remainder that
/// would otherwise be streamed transiently (copied into VRAM and evicted every
/// decode step) is instead read *in place* from a
/// `cuMemHostRegister(READ_ONLY|DEVICEMAP)` host mapping — the exact zero-copy
/// path #877/#880 measured at ~5.6 GB/s with bit-identical outputs and CUDA
/// graph capture support.
///
/// The hot resident set is unchanged from `StableResident` (arrival-order
/// first-fit up to the weight budget, retained and never evicted), so the
/// hybrid is a clean A/B against managed streaming: **same resident set, cold
/// reads zero-copy in place instead of copied**. It removes the per-step CPU
/// memcpy into pinned staging, the VRAM commit, the eviction and the
/// synchronize for every cold weight, and pins nothing dynamically — so it
/// never exercises the retain-then-churn path #886/#892 localised as unsafe.
///
/// Only takes effect when weight offload is enabled on the VMM stable-VA path
/// (`ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1`); with offload off, or on the
/// non-VMM path, it is inert and byte-identical. Set to `1`/`true`/`yes`/`on`
/// to enable.
///
/// Host-RAM policy: only the **cold** weights actually read zero-copy are
/// page-locked, at page granularity, bounded to the over-budget fraction
/// (~`W - B`, e.g. ~1.2 GiB on qwen14b-zp), not the whole 16.65 GiB data file.
/// Registration happens once per weight and is never repeated per page-in.
pub const WEIGHT_OFFLOAD_ZERO_COPY_HYBRID_ENV: &str = "ONNX_GENAI_ZERO_COPY_HYBRID";

/// Parse [`WEIGHT_OFFLOAD_ZERO_COPY_HYBRID_ENV`]. Defaults **OFF** (opt-in):
/// only `1`/`true`/`yes`/`on` (case/whitespace-insensitive) enable it.
pub(crate) fn zero_copy_hybrid_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

/// Read [`WEIGHT_OFFLOAD_ZERO_COPY_HYBRID_ENV`] from the process environment.
/// Exposed so the engine's memory-strategy policy can opt this knob in without
/// threading a new field through the whole runtime-application config, exactly
/// like [`byte_aware_residency_from_env`].
#[must_use]
pub fn zero_copy_hybrid_from_env() -> bool {
    zero_copy_hybrid_from_env_value(
        std::env::var(WEIGHT_OFFLOAD_ZERO_COPY_HYBRID_ENV)
            .ok()
            .as_deref(),
    )
}

/// Diagnostic knob (#864): perform the deferred weight's real streaming copy
/// instead of binding its host-mapped device pointer. Isolates the deferral/
/// admission flow from the host-mapped READ. Default OFF.
fn zero_copy_copy_instead() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        zero_copy_hybrid_from_env_value(
            std::env::var("ONNX_GENAI_ZERO_COPY_HYBRID_COPY_INSTEAD")
                .ok()
                .as_deref(),
        )
    })
}

/// Diagnostic knob (#864): print per-bind zero-copy diagnostics. Default OFF.
fn zero_copy_debug() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        zero_copy_hybrid_from_env_value(
            std::env::var("ONNX_GENAI_ZERO_COPY_HYBRID_DEBUG")
                .ok()
                .as_deref(),
        )
    })
}

/// Diagnostic knob (#864): pre-fault every page of the mapping on the CPU before
/// `cuMemHostRegister`, so pinning sees populated pages rather than racing the
/// lazy mmap's demand paging. Default OFF.
fn zero_copy_prefault() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        zero_copy_hybrid_from_env_value(
            std::env::var("ONNX_GENAI_ZERO_COPY_HYBRID_PREFAULT")
                .ok()
                .as_deref(),
        )
    })
}

/// Diagnostic knob (#864): register the mapping with DEVICEMAP only (drop
/// READ_ONLY), to isolate a READ_ONLY-flag correctness problem. Default OFF.
fn zero_copy_no_readonly() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        zero_copy_hybrid_from_env_value(
            std::env::var("ONNX_GENAI_ZERO_COPY_HYBRID_NO_READONLY")
                .ok()
                .as_deref(),
        )
    })
}

/// Outcome of reading a numeric diagnostic knob.
///
/// The three states are kept apart deliberately. Collapsing `Invalid` into
/// `Unset` is what makes the silent-fallback shape dangerous: a sweep that
/// writes `..._BUDGET_BYTES=2GB` (or leaves a digit separator, or a stray
/// character) would fall back to the conservative default and then report "no
/// corruption at 2 GiB" having never tested 2 GiB. A confident wrong answer is
/// worse than an error. This is the same failure as rendering "not determined"
/// as "determined to be unsafe" (#931), and the same family as the
/// `ASYNC_PAGEIN` trap where an unrecognized value silently selected the slow
/// path.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NumericEnv {
    Unset,
    Invalid(String),
    Value(u64),
}

impl NumericEnv {
    /// The configured value, warning on stderr when one was supplied but not
    /// understood, so a measurement taken under it is not mistaken for a
    /// measurement of the value the operator believed they set.
    fn or_default(self, name: &str, default: u64) -> u64 {
        match self {
            NumericEnv::Value(value) => value,
            NumericEnv::Unset => default,
            NumericEnv::Invalid(raw) => {
                eprintln!(
                    "cuda_ep: {name}={raw:?} is not a base-10 byte count; using {default} instead. \
                     Set a plain integer (e.g. 1073741824 for 1 GiB) — this value did NOT take \
                     effect, so any measurement taken under it describes the default."
                );
                default
            }
        }
    }

    fn into_option(self, name: &str) -> Option<u64> {
        match self {
            NumericEnv::Value(value) => Some(value),
            NumericEnv::Unset => None,
            NumericEnv::Invalid(raw) => {
                eprintln!(
                    "cuda_ep: {name}={raw:?} is not a base-10 integer; ignoring it. This value did \
                     NOT take effect."
                );
                None
            }
        }
    }
}

/// Read a numeric diagnostic knob, keeping "unset" and "set but not understood"
/// distinct. See [`NumericEnv`] for why that distinction is load-bearing.
fn parse_numeric_env(name: &str) -> NumericEnv {
    let Ok(raw) = std::env::var(name) else {
        return NumericEnv::Unset;
    };
    match raw.trim().parse::<u64>() {
        Ok(value) => NumericEnv::Value(value),
        Err(_) => NumericEnv::Invalid(raw),
    }
}

/// Diagnostic knob (#864): cap how many weights are actually bound zero-copy
/// (the rest are copied). `Some(1)` isolates a single zero-copy read. Default
/// `None` (unbounded).
fn zero_copy_max_binds() -> Option<u64> {
    const NAME: &str = "ONNX_GENAI_ZERO_COPY_HYBRID_MAX_BINDS";
    static V: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_numeric_env(NAME).into_option(NAME))
}

/// Conservative default safety budget for **distinct** zero-copy bytes bound
/// per residency (256 MiB). This exists because of a measured hardware limit
/// (#864): on an RTX 4060 Laptop under WDDM, device reads through a
/// `cuMemHostRegister(READ_ONLY | DEVICEMAP)` mapping are bit-identical up to
/// ~0.44 GB of distinct host-mapped data read per decode step (32 cold weights
/// verified correct), but **silently corrupt** above that (48 weights / ~0.65 GB
/// collapsed generation 16 → 3 tokens — the #886 signature, but from stale
/// host-mapped reads, not eviction). Individual reads are always correct, so
/// this is an aggregate host-mapped-aperture ceiling, not a per-read fault.
/// The default is set well under the observed-safe ceiling so the opt-in knob
/// can never violate the byte-identical gate; override for investigation only.
const ZERO_COPY_SAFE_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Per-residency distinct zero-copy byte budget (see
/// [`ZERO_COPY_SAFE_BUDGET_BYTES`]). Override via
/// `ONNX_GENAI_ZERO_COPY_HYBRID_BUDGET_BYTES` (e.g. `0` to force copy-only, or a
/// large value to reproduce the corruption on other hardware).
///
/// The default is a **WDDM-derived figure** (#912) and carries no evidence on
/// any other platform; whether the ceiling it guards against exists at all on
/// Linux is unmeasured (#925). Anyone sweeping this on other hardware should
/// confirm the value took effect by checking that `zero_copy_bytes_per_token`
/// tracks it, rather than trusting that the environment was read as intended.
fn zero_copy_budget_bytes() -> u64 {
    const NAME: &str = "ONNX_GENAI_ZERO_COPY_HYBRID_BUDGET_BYTES";
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_numeric_env(NAME).or_default(NAME, ZERO_COPY_SAFE_BUDGET_BYTES))
}

///
/// This exists purely to answer the #888 discriminating question: byte-aware
/// residency (#886) changed *two* independent things at once — it started
/// *retaining* large tensors that used to bypass, **and** it started evicting
/// the *smallest* resident instead of the front-of-order oldest one. If merely
/// changing the eviction victim — while keeping the shipped size-blind
/// always-bypass semantics — is enough to corrupt decode output, the surrounding
/// offload path harbours a latent order-dependent defect (explanation 2). If
/// only the full byte-aware policy (which also flips pages from bypass to
/// retained) corrupts, the defect is specific to that retain change
/// (explanation 1). Default (`lru`, or unset) is byte-identical to the shipped
/// path; every other value is opt-in and experimental.
pub const WEIGHT_OFFLOAD_EVICT_ORDER_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_EVICT_ORDER";

/// Eviction victim ordering for the size-blind admission path. Only the victim
/// changes; the always-bypass decision, the strong-count-1 evictable predicate,
/// and the pre-eviction stream drain are all identical to the shipped path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EvictOrderProbe {
    /// Front-of-order oldest evictable page — the shipped `next_evictable_key`
    /// behaviour. Byte-identical default.
    #[default]
    Lru,
    /// Most-recently-used evictable page (reverse recency). A "trivially
    /// different but still-correct" order: it only ever targets `strong_count
    /// == 1` pages, exactly like LRU.
    Mru,
    /// Smallest evictable page by bytes — the eviction *target* byte-aware uses,
    /// but decoupled from byte-aware's retain-large decision.
    Smallest,
    /// Largest evictable page by bytes.
    Largest,
}

/// Parse [`WEIGHT_OFFLOAD_EVICT_ORDER_ENV`]. Unset or unrecognised keeps the
/// shipped `Lru` order, so the default path stays byte-identical.
pub(crate) fn evict_order_from_env_value(value: Option<&str>) -> EvictOrderProbe {
    match value.map(|value| value.trim().to_ascii_lowercase()) {
        Some(value) => match value.as_str() {
            "mru" | "reverse" => EvictOrderProbe::Mru,
            "smallest" | "small" => EvictOrderProbe::Smallest,
            "largest" | "large" => EvictOrderProbe::Largest,
            _ => EvictOrderProbe::Lru,
        },
        None => EvictOrderProbe::Lru,
    }
}

/// Read [`WEIGHT_OFFLOAD_EVICT_ORDER_ENV`] from the process environment.
#[must_use]
pub fn evict_order_probe_from_env() -> EvictOrderProbe {
    evict_order_from_env_value(
        std::env::var(WEIGHT_OFFLOAD_EVICT_ORDER_ENV)
            .ok()
            .as_deref(),
    )
}

/// #888 diagnostic env: when truthy, drain the compute and copy streams
/// immediately before every H2D page-in fill. Default OFF (byte-identical).
/// Used to classify whether byte-aware residency's corruption is a
/// write-after-read hazard on the shared physical granule pool (would be fixed
/// by this drain) or a pure aliasing/logic bug (would not).
const WEIGHT_OFFLOAD_SYNC_BEFORE_FILL_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_SYNC_BEFORE_FILL";

fn sync_before_fill_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        matches!(
            std::env::var(WEIGHT_OFFLOAD_SYNC_BEFORE_FILL_ENV)
                .ok()
                .as_deref()
                .map(|value| value.trim().to_ascii_lowercase()),
            Some(ref value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
        )
    })
}

/// #888 diagnostic env: when truthy, byte-aware admission never *bypasses* a key
/// that already owns a stable VA slot — it re-retains it into residency instead.
/// Default OFF. This closes the one `stable_slot`/residency disagreement that is
/// reachable only under byte-aware (a once-retained tensor squeezed below the
/// resident set and re-entering as a bypass), letting an A/B on the same binary
/// test whether that disagreement is byte-aware's corruption mechanism.
const WEIGHT_OFFLOAD_RETAIN_SLOTTED_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_RETAIN_SLOTTED";

fn retain_slotted_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        matches!(
            std::env::var(WEIGHT_OFFLOAD_RETAIN_SLOTTED_ENV)
                .ok()
                .as_deref()
                .map(|value| value.trim().to_ascii_lowercase()),
            Some(ref value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
        )
    })
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
    /// Use byte-aware admission on top of scan-resistant residency: keep the
    /// largest tensors resident instead of whatever first-fit smalls landed in
    /// the leftover budget. Default-off / opt-in via
    /// `ONNX_GENAI_WEIGHT_OFFLOAD_BYTE_AWARE=1` (#837 item 3). Has no effect
    /// unless `scan_resistant_dense` is also on, since it only refines the
    /// `StableResident` bypass decision.
    pub byte_aware_residency: bool,
    /// Eviction victim ordering for the size-blind admission path (#888
    /// investigation). Default [`EvictOrderProbe::Lru`] is byte-identical to the
    /// shipped path; other values only change *which* unreferenced page is
    /// evicted for physical room, isolating whether decode correctness depends
    /// on eviction order independently of byte-aware's retain change.
    pub evict_order_probe: EvictOrderProbe,
    /// Enable the zero-copy hybrid (#864): read the cold, over-budget weight
    /// remainder in place from a `cuMemHostRegister(READ_ONLY|DEVICEMAP)` host
    /// mapping instead of copying it into VRAM every decode step. Default off;
    /// see [`WEIGHT_OFFLOAD_ZERO_COPY_HYBRID_ENV`]. Only effective on the VMM
    /// stable-VA managed-streaming path.
    pub zero_copy_hybrid: bool,
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
            byte_aware_residency: false,
            evict_order_probe: EvictOrderProbe::Lru,
            zero_copy_hybrid: false,
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
        let byte_aware_residency =
            byte_aware_from_env_value(std::env::var(WEIGHT_OFFLOAD_BYTE_AWARE_ENV).ok().as_deref());
        let evict_order_probe = evict_order_from_env_value(
            std::env::var(WEIGHT_OFFLOAD_EVICT_ORDER_ENV)
                .ok()
                .as_deref(),
        );
        let zero_copy_hybrid = zero_copy_hybrid_from_env_value(
            std::env::var(WEIGHT_OFFLOAD_ZERO_COPY_HYBRID_ENV)
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
            byte_aware_residency,
            evict_order_probe,
            zero_copy_hybrid,
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
    /// A cold weight read in place from host-mapped memory (#864 zero-copy
    /// hybrid). `ptr` is a `cuMemHostGetDevicePointer` over a
    /// `cuMemHostRegister(READ_ONLY|DEVICEMAP)` region of the weight mmap; no
    /// VRAM is owned, so Drop frees nothing. Unregistration is owned by the
    /// residency's [`HostMapRegistry`], which outlives every page it hands out.
    HostMapped,
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
            WeightAllocation::HostMapped => {
                // The device pointer aliases host-mapped memory owned and
                // unregistered by the residency's `HostMapRegistry`. Nothing to
                // free here; never assert (reachable from Drop).
            }
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

/// Host page size assertion helper for `cuMemHostRegister` (#864). x86-64 mmap
/// bases are 4 KiB page-aligned, which `cuMemHostRegister` requires.
const HOST_REGISTER_PAGE: usize = 4096;
const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;
const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 0x08;

/// Owns the `cuMemHostRegister(READ_ONLY|DEVICEMAP)` registrations backing the
/// zero-copy hybrid (#864) and derives per-weight device pointers from them.
///
/// An **entire mapping** is registered in a single call the first time any cold
/// weight from it is bound. This is the critical correctness property:
/// `cuMemHostGetDevicePointer` only guarantees a device address that is
/// contiguous over the extent of **one** registration, so a weight spanning two
/// separate registrations would map to discontiguous device VAs and a kernel
/// reading it linearly would run off the end (observed as
/// `CUDA_ERROR_ILLEGAL_ADDRESS`). One registration per mapping guarantees every
/// weight in that mapping is fully covered and contiguous.
///
/// Policy / cost: registering a whole mapping page-locks its full host size
/// (the model data file). On this qwen14b-zp box that is ~16.6 GiB of a 63.8
/// GiB host with ~34 GiB free — comfortable here, but a real claim on smaller
/// hosts, which is why the hybrid is gated behind a default-OFF knob. The lock
/// is paid once at first cold touch, never per page-in.
struct HostMapRegistry {
    /// mapping_id → (registered host base address, registered length).
    registered: HashMap<usize, (usize, usize)>,
}

impl HostMapRegistry {
    fn new() -> Self {
        Self {
            registered: HashMap::new(),
        }
    }

    /// Ensure the whole `mapping` is registered `READ_ONLY | DEVICEMAP` (once),
    /// then return the device pointer for `host_ptr`, which must lie inside it.
    /// Registration happens once per mapping and is never repeated, so repeated
    /// page-ins of any weight in the mapping pay nothing.
    fn device_ptr_for(
        &mut self,
        mapping_id: usize,
        mapping: &[u8],
        host_ptr: *const u8,
    ) -> Result<CUdeviceptr, WeightHandleError> {
        if let std::collections::hash_map::Entry::Vacant(entry) = self.registered.entry(mapping_id)
        {
            let base = mapping.as_ptr();
            if (base as usize) & (HOST_REGISTER_PAGE - 1) != 0 {
                return Err(WeightHandleError::DeviceBinding(format!(
                    "mmap base {base:p} is not {HOST_REGISTER_PAGE}-byte page aligned; \
                     cannot host-register for zero-copy"
                )));
            }
            let len = mapping.len();
            // Optional pre-fault (#864 diagnostic): the weight file is mmap'd
            // lazily, so a cold page may not be resident when we register it.
            // Touch every page on the CPU first so `cuMemHostRegister` pins the
            // real, populated physical pages rather than racing demand paging.
            if zero_copy_prefault() {
                let mut acc: u64 = 0;
                let mut off = 0usize;
                while off < len {
                    // SAFETY: `off < len` and `mapping` is a live slice.
                    acc = acc.wrapping_add(unsafe { *mapping.get_unchecked(off) } as u64);
                    off += HOST_REGISTER_PAGE;
                }
                // Prevent the loop from being optimized away.
                std::hint::black_box(acc);
            }
            let flags = if zero_copy_no_readonly() {
                CU_MEMHOSTREGISTER_DEVICEMAP
            } else {
                CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_READ_ONLY
            };
            // SAFETY: `mapping` is a live, page-aligned, read-only weight file
            // mapping owned by the executor's weight store, which outlives this
            // registry. READ_ONLY is sound because weights are immutable;
            // DEVICEMAP makes the device pointer valid. It is registered exactly
            // once per mapping_id (guarded above), so never double-registers.
            unsafe { sys::cuMemHostRegister_v2(base as *mut std::ffi::c_void, len, flags) }
                .result()
                .map_err(|error| {
                    WeightHandleError::DeviceBinding(format!(
                        "cuMemHostRegister(DEVICEMAP) of {len} bytes failed: {error}"
                    ))
                })?;
            entry.insert((base as usize, len));
            GLOBAL_HOST_REGISTERED_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        }
        let mut device_ptr: CUdeviceptr = 0;
        // SAFETY: `host_ptr` lies inside the mapping registered with DEVICEMAP
        // above, so a device pointer for it exists and is contiguous over the
        // whole mapping.
        unsafe {
            sys::cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr as *mut std::ffi::c_void, 0)
        }
        .result()
        .map_err(|error| {
            WeightHandleError::DeviceBinding(format!("cuMemHostGetDevicePointer failed: {error}"))
        })?;
        Ok(device_ptr)
    }
}

impl Drop for HostMapRegistry {
    fn drop(&mut self) {
        for &(base, _len) in self.registered.values() {
            // Best-effort teardown; never assert in Drop. If the mapping was
            // already torn down, unregistration failing is harmless.
            let _ = unsafe { sys::cuMemHostUnregister(base as *mut std::ffi::c_void) };
        }
    }
}

/// Outcome of a single VMM `admit_committed_span` attempt.
enum SpanAdmit {
    /// The span was filled into VRAM. `bypass` is `true` when the page was
    /// streamed transiently (not retained in the resident set).
    Filled { bypass: bool },
    /// Zero-copy hybrid (#864) only: admission would have bypassed (transiently
    /// streamed) this span, so the caller must bind it zero-copy instead. No
    /// eviction, reservation-map, or fill happened — the reserved VA is clean.
    DeferToZeroCopy,
}

/// Outcome of a VMM live page-in attempt (`resident_vmm_with`) under the
/// zero-copy hybrid. Non-hybrid callers only ever observe [`VmmAdmit::Page`].
#[derive(Debug)]
enum VmmAdmit {
    Page(Arc<CudaWeightPage>),
    /// The weight would be bypassed; the hybrid must bind it zero-copy in place
    /// rather than streaming it transiently through VRAM.
    DeferToZeroCopy,
}

impl VmmAdmit {
    /// Unwrap a page from a non-hybrid admission, which can never defer.
    fn expect_page(self) -> Arc<CudaWeightPage> {
        match self {
            VmmAdmit::Page(page) => page,
            VmmAdmit::DeferToZeroCopy => {
                unreachable!("non-hybrid VMM admission never defers to zero-copy")
            }
        }
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
    /// Byte-aware admission (#837 item 3): keep the largest tensors resident by
    /// evicting the smallest evictable resident to admit a strictly-larger
    /// incoming tensor, instead of streaming it transiently. Default false so
    /// the shipped path is byte-identical; opt in via
    /// `ONNX_GENAI_WEIGHT_OFFLOAD_BYTE_AWARE=1`.
    byte_aware: bool,
    /// Eviction victim ordering for the size-blind admission path (#888). Only
    /// changes which unreferenced page is chosen for physical room; the
    /// always-bypass decision is unchanged. Default [`EvictOrderProbe::Lru`] is
    /// byte-identical to the shipped path.
    evict_order_probe: EvictOrderProbe,
    /// Enable the zero-copy hybrid (#864): read the cold, over-budget remainder
    /// in place from host-mapped memory instead of copying it into VRAM every
    /// step. Only effective with `physical` (VMM stable-VA) installed.
    zero_copy_hybrid: bool,
    /// Owns the host registrations and derives device pointers for cold weights
    /// bound zero-copy. A separate lock from `inner` so the cold path never
    /// contends the residency mutex for its (idempotent) registration bookkeeping.
    host_registry: Mutex<HostMapRegistry>,
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
    /// Cold weights bound zero-copy from host-mapped memory (#864 hybrid). Kept
    /// separate from `pages` because these own no VRAM and must never be counted
    /// as resident, evicted, or reported in the residency byte ledger. Each is
    /// bound once and its `Arc` reused for every subsequent decode step (the
    /// device pointer is stable for the residency's lifetime), so steady-state
    /// cold lookups allocate nothing.
    cold_pages: HashMap<u64, Arc<CudaWeightPage>>,
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

    /// Pure CPU model of the byte-aware `StableResident` admission decision the
    /// GPU VMM path implements in `admit_committed_span` (#837 item 3): a page
    /// that does not fit is *retained* only when room can be made by evicting
    /// **strictly smaller** residents (smallest first); otherwise it bypasses.
    /// Refusing to evict an equal-or-larger peer is what keeps the largest
    /// tensors a *stable* resident set instead of thrashing them against each
    /// other. Lets the convergence-to-largest property be regression-tested
    /// without a GPU. The GPU path's physical-eviction loop differs in mechanism
    /// but shares this decision, which is what governs the byte-weighted hit
    /// rate.
    #[cfg(test)]
    fn access_byte_aware(&mut self, key: u64, bytes: u64) -> WeightPolicyAccess {
        if self.bytes_by_key.contains_key(&key) {
            self.record_hit(key);
            return WeightPolicyAccess {
                hit: true,
                admitted: false,
                evicted: Vec::new(),
            };
        }
        if !self.can_fit(bytes) {
            // Room can only be made from residents strictly smaller than the
            // incoming page; equal/larger peers are never displaced.
            let reclaimable: u64 = self
                .bytes_by_key
                .values()
                .filter(|&&resident| resident < bytes)
                .sum();
            let free = self.budget.saturating_sub(self.resident_bytes);
            if free.saturating_add(reclaimable) < bytes {
                self.record_page_in();
                return WeightPolicyAccess {
                    hit: false,
                    admitted: false,
                    evicted: Vec::new(),
                };
            }
        }
        let mut evicted = Vec::new();
        while !self.can_fit(bytes) {
            let Some((&smallest_key, _)) = self
                .bytes_by_key
                .iter()
                .filter(|&(_, &resident)| resident < bytes)
                .min_by_key(|&(_, &resident)| resident)
            else {
                break;
            };
            self.remove_page(smallest_key);
            evicted.push(smallest_key);
        }
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
            byte_aware: false,
            evict_order_probe: EvictOrderProbe::Lru,
            zero_copy_hybrid: false,
            host_registry: Mutex::new(HostMapRegistry::new()),
            physical: OnceLock::new(),
            staging_pool: PinnedStagingPool::new(Arc::clone(&runtime)),
            inner: Mutex::new(ResidencyInner {
                policy: WeightResidencyPolicy::new(budget_bytes),
                lease: None,
                pages: HashMap::new(),
                mapped_allowance: None,
                admission_no_progress: 0,
                slots: HashMap::new(),
                cold_pages: HashMap::new(),
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
            byte_aware: false,
            evict_order_probe: EvictOrderProbe::Lru,
            zero_copy_hybrid: false,
            host_registry: Mutex::new(HostMapRegistry::new()),
            physical: OnceLock::new(),
            staging_pool: PinnedStagingPool::new(Arc::clone(&runtime)),
            inner: Mutex::new(ResidencyInner {
                policy: WeightResidencyPolicy::new(lease.bytes()),
                lease: Some(lease),
                pages: HashMap::new(),
                mapped_allowance: None,
                admission_no_progress: 0,
                slots: HashMap::new(),
                cold_pages: HashMap::new(),
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

    /// Select byte-aware admission (#837 item 3). Only refines the
    /// `StableResident` bypass decision, so it has no effect unless
    /// scan-resistant dense residency is also on.
    pub fn with_byte_aware_residency(mut self, byte_aware: bool) -> Self {
        self.byte_aware = byte_aware;
        self
    }

    /// Select the eviction victim order for the size-blind admission path (#888
    /// investigation). Default [`EvictOrderProbe::Lru`] is byte-identical to the
    /// shipped path; other orders isolate whether decode correctness depends on
    /// eviction order independently of byte-aware's retain-vs-bypass change.
    pub fn with_evict_order_probe(mut self, evict_order_probe: EvictOrderProbe) -> Self {
        self.evict_order_probe = evict_order_probe;
        self
    }

    /// Select the zero-copy hybrid (#864): read the cold, over-budget weight
    /// remainder in place from host-mapped memory instead of copying it into
    /// VRAM every decode step. Only effective once VMM stable-VA admission is
    /// installed; inert (byte-identical) otherwise.
    pub fn with_zero_copy_hybrid(mut self, zero_copy_hybrid: bool) -> Self {
        self.zero_copy_hybrid = zero_copy_hybrid;
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
            return self
                .resident_vmm_with(
                    key,
                    resident.dtype,
                    resident.shape.clone(),
                    bytes.len(),
                    self.eviction_for(weight.boundary),
                    false,
                    move |runtime, ptr| {
                        unsafe { runtime.htod(&bytes, ptr) }.map_err(|error| {
                            WeightHandleError::DeviceBinding(format!("H2D copy: {error}"))
                        })
                    },
                )
                .map(VmmAdmit::expect_page);
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
    ///
    /// When the zero-copy hybrid (#864) is enabled and VMM stable-VA admission
    /// is installed, dispatches to [`Self::resident_mapped_hybrid`]; otherwise
    /// runs the copy-into-VRAM path unchanged.
    pub fn resident_mapped(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn MmapRegionSource,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        if self.zero_copy_hybrid && self.physical.get().is_some() {
            return self.resident_mapped_hybrid(key, weight, source);
        }
        self.resident_mapped_inner(key, weight, source, false)
            .map(VmmAdmit::expect_page)
    }

    /// Copy-into-VRAM live page-in path (the managed-streaming behaviour). The
    /// hybrid reuses this verbatim to pin its static hot set.
    ///
    /// When `hybrid_zero_copy` is set, a weight the VMM admission would bypass
    /// (transiently stream) is not evicted-for and not filled — instead the
    /// call returns [`VmmAdmit::DeferToZeroCopy`] so the caller binds it
    /// zero-copy in place. Non-hybrid callers pass `false` and always receive a
    /// [`VmmAdmit::Page`].
    fn resident_mapped_inner(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn MmapRegionSource,
        hybrid_zero_copy: bool,
    ) -> Result<VmmAdmit, WeightHandleError> {
        if let Some(hit) = self.get_hit(key) {
            return Ok(VmmAdmit::Page(hit));
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
                hybrid_zero_copy,
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
            .map(VmmAdmit::Page)
    }

    /// Zero-copy hybrid live-dispatch path (#864).
    ///
    /// The hot resident set is **exactly** what size-blind `StableResident`
    /// admission retains: a weight is copied into VRAM once and pinned only when
    /// it fits *without evicting* anything. The moment admission would bypass a
    /// weight — i.e. transiently stream it because it does not fit — the hybrid
    /// binds it **zero-copy in place** from a `cuMemHostRegister(READ_ONLY |
    /// DEVICEMAP)` host mapping instead of copying it, and no eviction ever
    /// runs. This is the critical safety property (#886/#892): because the hot
    /// set never evicts and the cold set never enters VRAM, no large weight ever
    /// occupies a stable slot that could later be evicted and re-admitted — the
    /// exact pattern that silently corrupted decode in #886. It is also a clean
    /// A/B against managed streaming: identical hot set, cold remainder read
    /// zero-copy in place rather than copied-and-bypassed every step.
    ///
    /// Cold pages own no VRAM and their device pointer is stable for the
    /// residency's lifetime, so after the first touch every lookup is a cache
    /// hit that allocates nothing.
    fn resident_mapped_hybrid(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn MmapRegionSource,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        // Hot (VRAM-resident) hit.
        if let Some(hit) = self.get_hit(key) {
            return Ok(hit);
        }
        // Cold (host-mapped) hit: already bound zero-copy on an earlier step.
        {
            let inner = self.lock();
            if let Some(cold) = inner.cold_pages.get(&key).cloned() {
                let len = cold.len();
                drop(inner);
                self.record_zero_copy_read(len);
                return Ok(cold);
            }
        }
        let len = weight.region_bytes_len();
        // Attempt a normal VMM page-in. In hybrid mode the admission short-
        // circuits to `DeferToZeroCopy` the instant it would bypass (transiently
        // stream) this weight — before any eviction — so a retained hot weight
        // is only ever one that fit without displacing another. Everything else
        // is bound zero-copy in place below.
        match self.resident_mapped_inner(key, weight, source, true)? {
            VmmAdmit::Page(page) => Ok(page),
            VmmAdmit::DeferToZeroCopy => self.bind_zero_copy(key, weight, source, len),
        }
    }

    /// Bind `key`'s cold weight zero-copy from host-mapped memory, caching the
    /// resulting page so subsequent steps reuse the identical device pointer.
    ///
    /// Falls back to a transient copy-into-VRAM stream (the managed-streaming
    /// bypass, which is byte-identical and never retains a large stable slot)
    /// when the weight's regions are not a single contiguous span — a single
    /// device pointer cannot address a gapped weight; in practice the packed
    /// `MatMulNBits`/`QMoE` blobs are one region.
    fn bind_zero_copy(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn MmapRegionSource,
        len: usize,
    ) -> Result<Arc<CudaWeightPage>, WeightHandleError> {
        // Safety budget (#864): keep distinct zero-copy bytes below the measured
        // host-mapped corruption ceiling on this class of hardware. Over budget,
        // copy the weight (byte-identical managed-streaming bypass) rather than
        // bind it host-mapped — this is what keeps the opt-in knob from ever
        // violating the byte-identical gate. See `ZERO_COPY_SAFE_BUDGET_BYTES`.
        if GLOBAL_ZERO_COPY_BOUND_BYTES
            .load(Ordering::Relaxed)
            .saturating_add(len as u64)
            > zero_copy_budget_bytes()
        {
            return self
                .resident_mapped_inner(key, weight, source, false)
                .map(VmmAdmit::expect_page);
        }
        let device_ptr = match self.zero_copy_device_ptr(weight, source)? {
            Some(ptr) => ptr,
            None => {
                // Non-contiguous weight: fall back to the plain streaming path
                // (hybrid flag off) so it transiently bypasses like managed
                // streaming rather than looping back here.
                return self
                    .resident_mapped_inner(key, weight, source, false)
                    .map(VmmAdmit::expect_page);
            }
        };
        // Diagnostic isolation knob (#864): cap the number of weights actually
        // bound zero-copy; copy the rest. `max=1` tests whether a *single*
        // zero-copy read is correct in the real decode integration (bisecting a
        // scale/aperture fault from a per-read fault). The whole mapping is still
        // registered above, so the device pointer is real either way.
        if zero_copy_max_binds()
            .is_some_and(|max| GLOBAL_ZERO_COPY_BINDS.load(Ordering::Relaxed) >= max)
        {
            return self
                .resident_mapped_inner(key, weight, source, false)
                .map(VmmAdmit::expect_page);
        } // end MAX_BINDS diagnostic cap
        // Diagnostic isolation knob (#864): when set, take the deferral decision
        // exactly as the zero-copy path would, but perform the real streaming
        // copy instead of binding the host-mapped pointer. If output is correct
        // under this knob but wrong without it, the fault is the host-mapped
        // READ (alignment/hardware), not the deferral/admission flow.
        if zero_copy_copy_instead() {
            return self
                .resident_mapped_inner(key, weight, source, false)
                .map(VmmAdmit::expect_page);
        }
        if zero_copy_debug() {
            let host_align = (device_ptr as usize) & 0xff;
            eprintln!(
                "zero_copy_bind: key={key} len={len} dptr=0x{device_ptr:x} dptr_align256={host_align} \
                 regions={} first_off={}",
                weight.regions.len(),
                weight.regions.first().map(|r| r.offset).unwrap_or(0)
            );
        }
        let page = Arc::new(CudaWeightPage {
            runtime: Arc::clone(&self.runtime),
            allocation: WeightAllocation::HostMapped,
            ptr: device_ptr,
            len,
            dtype: weight.dtype,
            shape: weight.shape.clone(),
        });
        {
            let mut inner = self.lock();
            // A concurrent dispatch may have bound the same key; prefer the
            // existing page and drop ours (Drop frees nothing for HostMapped).
            if let Some(existing) = inner.cold_pages.get(&key).cloned() {
                let existing_len = existing.len();
                drop(inner);
                self.record_zero_copy_read(existing_len);
                return Ok(existing);
            }
            inner.cold_pages.insert(key, Arc::clone(&page));
        }
        GLOBAL_ZERO_COPY_BINDS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_ZERO_COPY_BOUND_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        self.record_zero_copy_read(len);
        Ok(page)
    }

    /// Resolve a device pointer for `weight`'s contiguous host-mapped span, or
    /// `Ok(None)` when the weight is not a single contiguous region and cannot
    /// be addressed by one pointer.
    fn zero_copy_device_ptr(
        &self,
        weight: &LazyWeight,
        source: &dyn MmapRegionSource,
    ) -> Result<Option<CUdeviceptr>, WeightHandleError> {
        let Some(first) = weight.regions.first() else {
            return Ok(None);
        };
        let mapping_id = first.mapping_id;
        let mut expected = first.offset;
        for region in &weight.regions {
            if region.mapping_id != mapping_id || region.offset != expected {
                return Ok(None);
            }
            expected = match expected.checked_add(region.len) {
                Some(next) => next,
                None => return Ok(None),
            };
        }
        let span = ExternalMmapRegion {
            mapping_id,
            offset: first.offset,
            len: weight.region_bytes_len(),
        };
        let bytes = source.region_bytes(&span)?;
        let host_ptr = bytes.as_ptr();
        // The whole mapping must be registered in one call so the weight's
        // device pointer is contiguous over its full length (a per-weight
        // registration is only contiguous within itself, so a weight spanning
        // two registrations would read off the end — `CUDA_ERROR_ILLEGAL_ADDRESS`).
        let Some(mapping) = source.full_mapping_bytes(mapping_id) else {
            return Ok(None);
        };
        let device_ptr = self
            .host_registry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .device_ptr_for(mapping_id, mapping, host_ptr)?;
        Ok(Some(device_ptr))
    }

    /// Account one dispatch read of a cold host-mapped weight: the bytes cross
    /// PCIe in place this step, so they are the honest per-step cold traffic —
    /// tracked separately from `htod_bytes` (no copy happened).
    fn record_zero_copy_read(&self, len: usize) {
        GLOBAL_ZERO_COPY_READS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_ZERO_COPY_BYTES.fetch_add(len as u64, Ordering::Relaxed);
    }

    /// Resolve a weight on a miss by materializing the weight's canonical
    /// (compressed) bytes and streaming them host→device, with LRU eviction
    /// under the VRAM budget. The materialized bytes are the exact resident
    /// backing, so the page is byte-identical to a stock upload.
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
            return self
                .resident_vmm_with(
                    key,
                    resident.dtype,
                    resident.shape.clone(),
                    bytes.len(),
                    self.eviction_for(weight.boundary),
                    false,
                    move |runtime, ptr| {
                        unsafe { runtime.htod(&bytes, ptr) }.map_err(|error| {
                            WeightHandleError::DeviceBinding(format!("H2D copy: {error}"))
                        })
                    },
                )
                .map(VmmAdmit::expect_page);
        }
        let page = Arc::new(CudaWeightPage::upload(
            &self.runtime,
            resident.dtype,
            resident.shape.clone(),
            resident.bytes(),
        )?);
        self.admit(key, page, self.eviction_for(weight.boundary))
    }

    #[allow(clippy::too_many_arguments)]
    fn resident_vmm_with<F>(
        &self,
        key: u64,
        dtype: DataType,
        shape: Vec<usize>,
        len: usize,
        eviction: WeightEvictionPolicy,
        hybrid_zero_copy: bool,
        fill: F,
    ) -> Result<VmmAdmit, WeightHandleError>
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
            return Ok(VmmAdmit::Page(existing));
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

        let bypass = match self.admit_committed_span(
            &mut inner,
            physical,
            &allowance,
            ptr,
            len,
            eviction,
            reused_slot.is_some(),
            hybrid_zero_copy,
            fill,
        ) {
            Ok(SpanAdmit::Filled { bypass }) => bypass,
            Ok(SpanAdmit::DeferToZeroCopy) => {
                // Hybrid mode short-circuited before any eviction or fill: this
                // weight would be transiently streamed, so the caller binds it
                // zero-copy instead. A deferred weight never had a stable slot
                // (bypass keys never do), so `reused_slot` is always None here;
                // release the fresh throwaway VA and report the deferral.
                if reused_slot.is_none() {
                    let _ = physical.allocator.deallocate_span(ptr);
                }
                return Ok(VmmAdmit::DeferToZeroCopy);
            }
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
        // #888 diagnostic: a *slotted* key that admission decides to *bypass*
        // is the one place `stable_slot` (drives Drop) and residency (drives
        // hits) disagree — the page is decommitted-but-VA-kept yet never joins
        // `pages`. This case is unreachable on the shipped size-blind path
        // (slotted keys are the retained smalls, which stay resident) but
        // reachable under byte-aware, where a once-retained large tensor can be
        // squeezed below the resident set and re-enter as a bypass. Emit once so
        // its occurrence during a corrupting run is observable even though the
        // harness bails at the 3-token early-EOS.
        if reused_slot.is_some() && bypass {
            static SLOTTED_BYPASS_SEEN: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !SLOTTED_BYPASS_SEEN.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "weight_paging_diag[#888]: slotted-key bypass occurred (key={key}, \
                     len={len}) — stable_slot/residency disagreement reachable under byte-aware"
                );
            }
        }
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
            inner.record_bypassed_page_in(len as u64);
        } else {
            inner.insert_page(key, Arc::clone(&page), len as u64);
        }
        Ok(VmmAdmit::Page(page))
    }

    /// Commit `len` physical bytes under the reserved VA `ptr`, evicting other
    /// resident pages as `eviction` permits, then run `fill` to copy the weight
    /// bytes into the freshly mapped granules. Returns
    /// [`SpanAdmit::Filled`] with `bypass = true` when the page could not be
    /// admitted to the resident set and is handed back transiently (a "bypass"
    /// under scan-resistant admission), `bypass = false` when it is retained.
    ///
    /// When `hybrid_zero_copy` is set (#864), the first time admission would
    /// bypass this span the method returns [`SpanAdmit::DeferToZeroCopy`]
    /// **before** evicting or filling, so the caller can bind the weight
    /// zero-copy in place. This is what keeps the hybrid's hot set static: no
    /// eviction ever runs on behalf of a cold weight, so no retained weight is
    /// evicted and re-admitted (the #886 corruption pattern).
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
        has_stable_slot: bool,
        hybrid_zero_copy: bool,
        fill: F,
    ) -> Result<SpanAdmit, WeightHandleError>
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
                        // #888 diagnostic: optionally drain the compute stream
                        // immediately before the H2D fill so no in-flight kernel
                        // can still be reading a physical granule this span just
                        // (re)committed from the shared pool. If enabling this
                        // makes byte-aware residency correct, the corruption is a
                        // write-after-read hazard on the shared granule pool
                        // exposed by retaining (and re-committing) large tensors,
                        // not a pure aliasing/logic bug. Default OFF /
                        // byte-identical.
                        if sync_before_fill_enabled() {
                            self.runtime.synchronize().map_err(|error| {
                                WeightHandleError::DeviceBinding(format!(
                                    "sync-before-fill compute drain: {error}"
                                ))
                            })?;
                            self.runtime.copy_stream().synchronize().map_err(|error| {
                                WeightHandleError::DeviceBinding(format!(
                                    "sync-before-fill copy drain: {error}"
                                ))
                            })?;
                        }
                        fill.take().expect("VMM page fill runs once")(
                            &self.runtime,
                            ptr.as_ptr() as CUdeviceptr,
                        )?;
                        return Ok(SpanAdmit::Filled { bypass });
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
                // Size-blind `StableResident` always bypasses here: once the
                // budget is full every distinct tensor streams transiently,
                // which biases residency toward small first-fit tensors and
                // streams the large projections every step (#837 item 3).
                //
                // Byte-aware admission instead retains the incoming page (falls
                // through to evict the *smallest* resident below) whenever it is
                // strictly larger than the smallest evictable resident, so the
                // resident set converges to the top-`B`-bytes tensors. A page no
                // larger than the smallest resident still bypasses, so a small
                // tensor never displaces a large one to become resident.
                //
                // NOTE (#837 item 3): `self.byte_aware` is EXPERIMENTAL and
                // measured UNSAFE — when this branch actually engages it
                // corrupts decode output (token-identity failure). It is
                // default-OFF and never set on the shipped path; see
                // `WEIGHT_OFFLOAD_BYTE_AWARE_ENV` for the measured evidence.
                bypass = !(self.byte_aware
                    && (has_stable_slot && retain_slotted_enabled()
                        || inner
                            .smallest_evictable()
                            .is_some_and(|(_, smallest)| (len as u64) > smallest)));
            }

            // Zero-copy hybrid (#864): the instant admission would need to evict
            // to place this weight, hand it back to be bound zero-copy in place —
            // *before* any eviction runs, for every eviction policy. This is the
            // safety hinge: no cold weight ever evicts a retained hot page, so no
            // large weight occupies a stable slot that is later evicted and
            // re-admitted (the #886 corruption). It also means a hybrid hot page
            // is only ever one that fit without eviction. `evictions` is always 0
            // here under the hybrid because the first non-fit defers immediately.
            if hybrid_zero_copy {
                return Ok(SpanAdmit::DeferToZeroCopy);
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
            // Byte-aware admission evicts the smallest evictable resident so a
            // large incoming tensor displaces cheap smalls rather than another
            // large tensor; size-blind residency evicts by `evict_order_probe`,
            // which is front-of-order LRU on the shipped default path and only
            // varies under the #888 eviction-order investigation knob.
            let evicted_key = if self.byte_aware {
                inner.smallest_evictable().map(|(key, _)| key)
            } else {
                inner.evictable_key_by_probe(self.evict_order_probe, eviction)
            };
            let Some(evicted_key) = evicted_key else {
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
            inner.record_bypassed_page_in(bytes);
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
    fn record_bypassed_page_in(&mut self, bytes: u64) {
        self.policy.record_page_in();
        GLOBAL_PAGE_INS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_BYPASSED_PAGE_INS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_BYPASSED_PAGE_IN_BYTES.fetch_add(bytes, Ordering::Relaxed);
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

    /// Smallest evictable resident page as `(key, bytes)`, or `None` when no
    /// page is evictable.
    ///
    /// "Evictable" is the same predicate [`Self::next_evictable_key`] uses: the
    /// cache must be the page's sole owner (`Arc::strong_count == 1`), so a page
    /// still read by an in-flight kernel is never a candidate. Byte-aware
    /// admission uses this to displace only the cheapest resident, protecting
    /// the large tensors that dominate streaming cost (#837 item 3).
    fn smallest_evictable(&self) -> Option<(u64, u64)> {
        let mut best: Option<(u64, u64)> = None;
        for (&key, &bytes) in &self.policy.bytes_by_key {
            let evictable = self
                .pages
                .get(&key)
                .is_some_and(|page| Arc::strong_count(page) == 1);
            if evictable && best.is_none_or(|(_, best_bytes)| bytes < best_bytes) {
                best = Some((key, bytes));
            }
        }
        best
    }

    /// Select the eviction victim for the size-blind admission path under the
    /// #888 eviction-order probe. [`EvictOrderProbe::Lru`] delegates to
    /// [`Self::next_evictable_key`], so the default path is byte-identical to
    /// the shipped code. Every variant applies the identical "evictable" filter
    /// (`Arc::strong_count == 1`), so no order can ever target a page still read
    /// by an in-flight kernel — the only thing that changes is *which*
    /// unreferenced page is freed for physical room.
    fn evictable_key_by_probe(
        &self,
        probe: EvictOrderProbe,
        eviction: WeightEvictionPolicy,
    ) -> Option<u64> {
        let is_evictable = |key: u64| -> bool {
            self.pages
                .get(&key)
                .is_some_and(|page| Arc::strong_count(page) == 1)
        };
        match probe {
            EvictOrderProbe::Lru => self.next_evictable_key(eviction),
            EvictOrderProbe::Mru => self
                .policy
                .order
                .iter()
                .rev()
                .copied()
                .find(|&k| is_evictable(k)),
            EvictOrderProbe::Smallest => self.smallest_evictable().map(|(key, _)| key),
            EvictOrderProbe::Largest => {
                let mut best: Option<(u64, u64)> = None;
                for (&key, &bytes) in &self.policy.bytes_by_key {
                    if is_evictable(key) && best.is_none_or(|(_, best_bytes)| bytes > best_bytes) {
                        best = Some((key, bytes));
                    }
                }
                best.map(|(key, _)| key)
            }
        }
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
    fn bypassed_byte_share_attributes_streamed_bytes() {
        // No bytes streamed -> no attribution, never a divide-by-zero.
        assert_eq!(GlobalOffloadStats::default().bypassed_byte_share(), None);
        // Bypass bytes are a subset of htod_bytes: 3 MiB of a 12 MiB stream.
        let stats = GlobalOffloadStats {
            htod_bytes: 12 * 1024 * 1024,
            bypassed_page_in_bytes: 3 * 1024 * 1024,
            ..GlobalOffloadStats::default()
        };
        let share = stats.bypassed_byte_share().expect("bytes were streamed");
        assert!(
            (share - 0.25).abs() < 1e-9,
            "one quarter of stream: {share}"
        );
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
    fn byte_aware_env_defaults_off_and_opts_in() {
        // Opt-in: unset and every non-truthy value stay OFF, so the shipped path
        // is byte-identical unless the operator explicitly enables the A/B knob.
        assert!(!byte_aware_from_env_value(None));
        assert!(!byte_aware_from_env_value(Some("0")));
        assert!(!byte_aware_from_env_value(Some("false")));
        assert!(!byte_aware_from_env_value(Some("")));
        assert!(!byte_aware_from_env_value(Some("maybe")));
        // Only the canonical truthy spellings enable it.
        assert!(byte_aware_from_env_value(Some("1")));
        assert!(byte_aware_from_env_value(Some("true")));
        assert!(byte_aware_from_env_value(Some("YES")));
        assert!(byte_aware_from_env_value(Some("  On ")));
    }

    #[test]
    fn zero_copy_hybrid_env_defaults_off_and_opts_in() {
        // Opt-in, default OFF: the hybrid is byte-identical to the shipped path
        // until an operator explicitly enables it (#864). Unset and every
        // non-truthy spelling stay OFF.
        assert!(!zero_copy_hybrid_from_env_value(None));
        assert!(!zero_copy_hybrid_from_env_value(Some("0")));
        assert!(!zero_copy_hybrid_from_env_value(Some("false")));
        assert!(!zero_copy_hybrid_from_env_value(Some("")));
        assert!(!zero_copy_hybrid_from_env_value(Some("maybe")));
        // Only the canonical truthy spellings enable it.
        assert!(zero_copy_hybrid_from_env_value(Some("1")));
        assert!(zero_copy_hybrid_from_env_value(Some("true")));
        assert!(zero_copy_hybrid_from_env_value(Some("YES")));
        assert!(zero_copy_hybrid_from_env_value(Some("  On ")));
    }

    #[test]
    fn zero_copy_safe_budget_is_below_measured_corruption_ceiling() {
        // #864: distinct host-mapped reads were byte-identical at 32 cold
        // weights (~0.44 GB/step) and corrupted at 48 (~0.65 GB/step). The
        // default safety budget must sit strictly under the observed-safe
        // ceiling so the opt-in knob can never violate the byte-identical gate.
        const OBSERVED_SAFE_BYTES: u64 = 436_633_600; // 32 cold weights, measured
        assert!(
            ZERO_COPY_SAFE_BUDGET_BYTES < OBSERVED_SAFE_BYTES,
            "default budget {ZERO_COPY_SAFE_BUDGET_BYTES} must be under the measured-safe ceiling"
        );
        assert_eq!(ZERO_COPY_SAFE_BUDGET_BYTES, 256 * 1024 * 1024);
    }

    #[test]
    fn numeric_env_keeps_unset_and_unparseable_distinct() {
        // The hazard is a measurement one. A sweep that writes `2GB`, or leaves
        // a digit separator or a stray character, would silently fall back to
        // the conservative default -- and then report "no corruption at 2 GiB"
        // having never tested 2 GiB. A confident wrong answer is worse than an
        // error.
        //
        // The assertion that matters is that `Invalid` is distinguishable from
        // `Unset`. The previous implementation was `.parse().ok()`, which
        // collapsed both to `None`, so a test that only checked "bad input does
        // not yield a value" would have passed against it unchanged and proved
        // nothing. Same shape as rendering "not determined" as "determined to
        // be unsafe" (#931).
        const NAME: &str = "ONNX_GENAI_TEST_NUMERIC_ENV_PROBE";

        // SAFETY: single-threaded test-local variable with a unique name; it is
        // set and removed within this test and read by nothing else.
        unsafe {
            std::env::remove_var(NAME);
        }
        assert_eq!(parse_numeric_env(NAME), NumericEnv::Unset);

        // SAFETY: as above.
        unsafe {
            std::env::set_var(NAME, " 1073741824 ");
        }
        assert_eq!(
            parse_numeric_env(NAME),
            NumericEnv::Value(1_073_741_824),
            "a plain integer, surrounding whitespace included, must be honoured"
        );

        for bad in ["2GB", "1_073_741_824", "0x10", "", "1.5", "-1"] {
            // SAFETY: as above.
            unsafe {
                std::env::set_var(NAME, bad);
            }
            assert_eq!(
                parse_numeric_env(NAME),
                NumericEnv::Invalid(bad.to_string()),
                "{bad:?} must be reported as supplied-but-unusable, not as absent"
            );
        }

        // The default substitution still happens -- the point is that it is
        // announced rather than silent.
        assert_eq!(parse_numeric_env(NAME).or_default(NAME, 4096), 4096);
        assert_eq!(parse_numeric_env(NAME).into_option(NAME), None);

        // SAFETY: as above.
        unsafe {
            std::env::remove_var(NAME);
        }
        assert_eq!(parse_numeric_env(NAME).or_default(NAME, 4096), 4096);
    }

    #[test]
    fn evict_order_env_defaults_lru_and_parses_variants() {
        // Unset and unrecognised keep the shipped front-of-order LRU victim, so
        // the default path is byte-identical (#888).
        assert_eq!(evict_order_from_env_value(None), EvictOrderProbe::Lru);
        assert_eq!(evict_order_from_env_value(Some("")), EvictOrderProbe::Lru);
        assert_eq!(
            evict_order_from_env_value(Some("lru")),
            EvictOrderProbe::Lru
        );
        assert_eq!(
            evict_order_from_env_value(Some("nonsense")),
            EvictOrderProbe::Lru
        );
        // Opt-in experimental orders (case/whitespace-insensitive).
        assert_eq!(
            evict_order_from_env_value(Some("mru")),
            EvictOrderProbe::Mru
        );
        assert_eq!(
            evict_order_from_env_value(Some(" Reverse ")),
            EvictOrderProbe::Mru
        );
        assert_eq!(
            evict_order_from_env_value(Some("SMALLEST")),
            EvictOrderProbe::Smallest
        );
        assert_eq!(
            evict_order_from_env_value(Some("large")),
            EvictOrderProbe::Largest
        );
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
                false,
                move |runtime, ptr| {
                    unsafe { runtime.htod(&first_bytes, ptr) }
                        .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))
                },
            )
            .map(VmmAdmit::expect_page)
            .expect("first physical page");

        let zone_error = residency
            .resident_vmm_with(
                2,
                DataType::Uint8,
                vec![granule],
                granule,
                WeightEvictionPolicy::StableResident,
                false,
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
                false,
                move |runtime, ptr| {
                    unsafe { runtime.htod(&second_bytes, ptr) }
                        .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))
                },
            )
            .map(VmmAdmit::expect_page)
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
                false,
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
                    false,
                    move |runtime, ptr| {
                        unsafe { runtime.htod(&bytes, ptr) }
                            .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))
                    },
                )
                .map(VmmAdmit::expect_page)
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
                false,
                move |runtime, ptr| {
                    unsafe { runtime.htod(&second_bytes, ptr) }
                        .map_err(|error| WeightHandleError::DeviceBinding(error.to_string()))
                },
            )
            .map(VmmAdmit::expect_page)
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

    /// The core #837 item-3 claim, modelled on CPU: when small tensors are
    /// touched *before* the large ones each cycle, size-blind `StableResident`
    /// lets the smalls win the first-fit race and crowds the large projections
    /// out of the budget, streaming them every step. Byte-aware admission
    /// evicts those smalls to seat the large tensors, so its **byte-weighted**
    /// hit rate is materially higher and it streams strictly fewer bytes.
    ///
    /// This exercises the admission *decision function* in isolation — an
    /// abstract residency model with no CUDA-graph capture, no physical granule
    /// churn, and no in-flight-read lifecycle. It confirms the size-greedy rule
    /// does what it claims *on paper*. It does **not** validate the real GPU
    /// path: the on-device A/B (see `WEIGHT_OFFLOAD_BYTE_AWARE_ENV`) shows the
    /// same rule corrupts decode output when it actually engages, which is why
    /// the knob is default-OFF and unshipped. Keep both facts together.
    #[test]
    fn byte_aware_beats_stable_subset_when_smalls_crowd_out_larges() {
        // Six 1-byte "norms/biases" touched first, then two 10-byte
        // "projections". W = 26 bytes; budget seats both projections (20) plus
        // two norms with room to spare, but only if the norms do not squat it.
        let order: [(u64, u64); 8] = [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 10),
            (7, 10),
        ];
        const BUDGET: u64 = 22;
        const WARMUP: u64 = 4;
        const MEASURED: u64 = 6;

        let byte_hits = |byte_aware: bool| -> (u64, u64) {
            let mut policy = WeightResidencyPolicy::new(BUDGET);
            let mut hit_bytes = 0u64;
            let mut streamed_bytes = 0u64;
            for cycle in 0..(WARMUP + MEASURED) {
                let measuring = cycle >= WARMUP;
                for (key, bytes) in order {
                    let access = if byte_aware {
                        policy.access_byte_aware(key, bytes)
                    } else {
                        policy.access(key, bytes, WeightEvictionPolicy::StableResident)
                    };
                    if measuring {
                        if access.hit {
                            hit_bytes += bytes;
                        } else {
                            streamed_bytes += bytes;
                        }
                    }
                }
            }
            (hit_bytes, streamed_bytes)
        };

        let (blind_hit, blind_stream) = byte_hits(false);
        let (aware_hit, aware_stream) = byte_hits(true);
        let blind_rate = blind_hit as f64 / (blind_hit + blind_stream) as f64;
        let aware_rate = aware_hit as f64 / (aware_hit + aware_stream) as f64;

        assert!(
            aware_rate > blind_rate + 0.15,
            "byte-aware residency must materially raise the byte-weighted hit rate when \
             smalls crowd out larges: blind={blind_rate:.3} aware={aware_rate:.3}"
        );
        // The point is a smaller `htod_bytes`, not just a nicer count-based report.
        assert!(
            aware_stream < blind_stream,
            "byte-aware residency must stream fewer bytes: \
             blind={blind_stream} aware={aware_stream}"
        );
    }

    /// The honest limit of the policy, also locked in: when the *large* tensors
    /// alone exceed the budget, no admission order can hold them all, so
    /// byte-aware must not thrash the large set against itself (strictly-smaller
    /// eviction) and must not regress below size-blind `StableResident`. This is
    /// the case that a naive "evict the smallest to admit any larger page" rule
    /// gets wrong, and the reason the policy refuses to displace equal/larger
    /// peers (#837 item 3).
    #[test]
    fn byte_aware_does_not_thrash_when_larges_exceed_budget() {
        // Five 10-byte projections (50 bytes) against a 22-byte budget: at most
        // two can ever be resident, and a clean cyclic scan of five is pessimal
        // for any evict-to-admit rule.
        let order: [(u64, u64); 5] = [(0, 10), (1, 10), (2, 10), (3, 10), (4, 10)];
        const BUDGET: u64 = 22;
        const WARMUP: u64 = 4;
        const MEASURED: u64 = 6;

        let steady_stream = |byte_aware: bool| -> u64 {
            let mut policy = WeightResidencyPolicy::new(BUDGET);
            let mut streamed = 0u64;
            for cycle in 0..(WARMUP + MEASURED) {
                for (key, bytes) in order {
                    let access = if byte_aware {
                        policy.access_byte_aware(key, bytes)
                    } else {
                        policy.access(key, bytes, WeightEvictionPolicy::StableResident)
                    };
                    if cycle >= WARMUP && !access.hit {
                        streamed += bytes;
                    }
                }
            }
            streamed
        };

        assert!(
            steady_stream(true) <= steady_stream(false),
            "byte-aware must not stream more than size-blind when larges exceed budget: \
             aware={} blind={}",
            steady_stream(true),
            steady_stream(false)
        );
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
