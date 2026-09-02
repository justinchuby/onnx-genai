//! Bounded reuse pool for page-locked (pinned) host staging buffers.
//!
//! The live weight-offload page-in path (`CudaWeightResidency::resident_mapped`)
//! copies weight bytes host→device from a pinned staging buffer. Historically it
//! allocated a fresh buffer with `cuMemHostAlloc` and freed it with
//! `cuMemFreeHost` on *every* page-in. On a not-fit model the same layer weights
//! are paged every token (issue #837: 5,535 page-ins for 16 tokens), so that is
//! 5,535 page-lock / page-unlock pairs. `cuMemHostAlloc`/`cuMemFreeHost` take a
//! kernel lock to pin/unpin OS pages and tend to serialize against the device,
//! which defeats the async, fence-ordered page-in path the offload design
//! otherwise implements correctly.
//!
//! This pool keeps a small free-list of already-pinned buffers and hands them
//! back out, so a steady-state decode pays the page-lock cost a handful of times
//! instead of once per page-in.
//!
//! # Lifetime / fence safety
//!
//! A staging buffer is the **source** of a host→device copy and must not be
//! reused or freed until that copy has *completed*, not merely been enqueued.
//! The offload page-in path enforces this structurally: every copy on that path
//! is issued through [`CudaRuntime::htod_async_elapsed_ms`], which records a
//! CUDA end event after the `cuMemcpyHtoDAsync` and **synchronizes that event on
//! the host before returning** (see `cudarc` `Event::elapsed_ms`, which calls
//! `end.synchronize()`). The copy's DMA read of the staging buffer is therefore
//! complete on the host timeline before the fill/upload call returns.
//!
//! This ordering is enforced **structurally, not by comment**. The reuse path
//! ([`PinnedStagingPool::release`] and [`PooledStaging::retire`]) requires a
//! [`CopyCompleted`] witness, and that witness can only be minted by a
//! host-synchronizing copy primitive in the `runtime` module (today
//! [`crate::runtime::CudaRuntime::htod_async_elapsed_ms`]). A buffer returns to
//! the pool for reuse only by presenting that witness, so a future switch to a
//! non-blocking `htod_async` + deferred fence produces no witness at the reuse
//! site and **fails to compile** until the author threads a completion witness
//! through after awaiting the fence. [`PooledStaging`]'s `Drop` is only a
//! leak-safe fallback: it *frees* the buffer (never reuses it), so a dropped-
//! without-witness buffer costs a re-allocation — caught by the
//! `pinned_alloc_calls` counter and its regression test — but can never cause
//! silent reuse-while-in-flight corruption.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::byte_telemetry::{EventSpec, ObservedBoundary, ObservedCategory, ObservedStatus};
use crate::runtime::{CopyCompleted, CudaRuntime, PinnedStaging};

/// Number of times the pool actually called `cuMemHostAlloc` (a pinned-buffer
/// page-lock), as opposed to satisfying a request from the free-list. A
/// regression guard asserts this stays far below the page-in count.
static GLOBAL_PINNED_ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

/// Number of times the pool satisfied a request by reusing a free-listed buffer.
static GLOBAL_PINNED_REUSES: AtomicU64 = AtomicU64::new(0);

/// Read the process-global count of real pinned host allocations issued by the
/// staging pool.
pub fn global_pinned_alloc_calls() -> u64 {
    GLOBAL_PINNED_ALLOC_CALLS.load(Ordering::Relaxed)
}

/// Read the process-global count of pinned staging reuses served by the pool.
pub fn global_pinned_reuses() -> u64 {
    GLOBAL_PINNED_REUSES.load(Ordering::Relaxed)
}

/// Reset the pinned-pool counters for a new measurement window. The retained
/// buffers themselves are intentionally *not* dropped: they are live host
/// memory that survives a benchmark warmup reset the same way the residency
/// pages do.
pub(crate) fn reset_pinned_pool_counters() {
    GLOBAL_PINNED_ALLOC_CALLS.store(0, Ordering::Relaxed);
    GLOBAL_PINNED_REUSES.store(0, Ordering::Relaxed);
}

/// Default cap on the number of retained free buffers.
const DEFAULT_MAX_BUFFERS: usize = 8;

/// Default cap on total retained pinned host bytes (512 MiB). Page-ins on the
/// offload path are host-synchronous and serialized under the residency lock, so
/// at most one staging buffer is live at a time and the pool converges to a
/// single retained buffer sized to the largest page-in (tens of MiB for
/// qwen14b). This ceiling only bounds a pathological worst case.
const DEFAULT_MAX_BYTES: usize = 512 * 1024 * 1024;

/// Bounded free-list of pinned host staging buffers reused across weight
/// page-ins. See the module docs for the fence-safety argument.
pub struct PinnedStagingPool {
    runtime: Arc<CudaRuntime>,
    free: Mutex<Vec<PinnedStaging>>,
    max_buffers: usize,
    max_bytes: usize,
    /// Per-instance count of real `cuMemHostAlloc` calls, for deterministic
    /// testing independent of the process-global counter.
    alloc_calls: AtomicU64,
    /// Per-instance count of reuses served from the free-list.
    reuses: AtomicU64,
}

impl PinnedStagingPool {
    /// Build a pool bound to `runtime` with the default retention caps.
    pub fn new(runtime: Arc<CudaRuntime>) -> Arc<Self> {
        Self::with_bounds(runtime, DEFAULT_MAX_BUFFERS, DEFAULT_MAX_BYTES)
    }

    /// Build a pool with explicit retention bounds (used by tests).
    pub fn with_bounds(
        runtime: Arc<CudaRuntime>,
        max_buffers: usize,
        max_bytes: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            runtime,
            free: Mutex::new(Vec::new()),
            max_buffers: max_buffers.max(1),
            max_bytes,
            alloc_calls: AtomicU64::new(0),
            reuses: AtomicU64::new(0),
        })
    }

    /// Acquire a staging buffer with capacity of at least `len` bytes.
    ///
    /// Reuses the smallest free-listed buffer that already fits `len`
    /// (reuse-if-large-enough), or page-locks a fresh `len`-byte buffer when
    /// none fits. The returned [`PooledStaging`] returns its buffer to the pool
    /// on `Drop`.
    pub fn acquire(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<PooledStaging, onnx_runtime_ep_api::EpError> {
        let reused = {
            let mut free = self.free.lock().expect("pinned staging pool poisoned");
            let mut best: Option<usize> = None;
            for (idx, buffer) in free.iter().enumerate() {
                if buffer.len() >= len {
                    match best {
                        Some(b) if free[b].len() <= buffer.len() => {}
                        _ => best = Some(idx),
                    }
                }
            }
            best.map(|idx| free.swap_remove(idx))
        };
        let staging = match reused {
            Some(staging) => {
                self.reuses.fetch_add(1, Ordering::Relaxed);
                GLOBAL_PINNED_REUSES.fetch_add(1, Ordering::Relaxed);
                self.runtime.observe_bytes(EventSpec::new(
                    ObservedCategory::HostAllocation,
                    ObservedBoundary::PinnedHostReuse,
                    ObservedStatus::Reclaimed,
                    staging.len() as u64,
                ));
                staging
            }
            None => {
                let staging = self.runtime.alloc_pinned(len)?;
                self.alloc_calls.fetch_add(1, Ordering::Relaxed);
                GLOBAL_PINNED_ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                staging
            }
        };
        Ok(PooledStaging {
            pool: Arc::clone(self),
            staging: Some(staging),
        })
    }

    /// Return a raw staging buffer to the free-list for reuse, subject to the
    /// retention bounds. A buffer that would exceed either bound is dropped,
    /// which frees its pinned host pages.
    ///
    /// Consuming a [`CopyCompleted`] witness is the compile-time proof that the
    /// buffer's most recent host→device copy has *completed* (not merely been
    /// enqueued), so reuse cannot race an in-flight DMA read. The witness is
    /// produced only by a host-synchronizing copy primitive in `runtime`.
    pub fn release(&self, staging: PinnedStaging, _completed: CopyCompleted) {
        let mut free = self.free.lock().expect("pinned staging pool poisoned");
        let retained: usize = free.iter().map(PinnedStaging::len).sum();
        if free.len() < self.max_buffers && retained.saturating_add(staging.len()) <= self.max_bytes
        {
            free.push(staging);
        }
        // Otherwise `staging` drops here, freeing its pinned host pages.
    }

    /// Number of buffers currently retained on the free-list (for tests).
    #[cfg(test)]
    pub fn free_len(&self) -> usize {
        self.free
            .lock()
            .expect("pinned staging pool poisoned")
            .len()
    }

    /// Per-instance count of real pinned allocations issued so far.
    pub fn alloc_calls(&self) -> u64 {
        self.alloc_calls.load(Ordering::Relaxed)
    }

    /// Per-instance count of reuses served from the free-list so far.
    pub fn reuses(&self) -> u64 {
        self.reuses.load(Ordering::Relaxed)
    }

    /// Whether this pool's retention bounds can hold `count` concurrently
    /// live buffers of `len` bytes each without evicting one of them on
    /// release.
    ///
    /// A look-ahead prefetch (issue #82 BlockQuantizedMoE prefill prefetch)
    /// genuinely needs **two** same-size buffers alive at once in steady
    /// state: the buffer backing the just-issued prefetch for the next
    /// boundary, and the buffer the current boundary's own promoted prefetch
    /// releases moments later (see
    /// `weight_paging.rs::prefetch_block_quantized_moe`'s call-order note).
    /// If the pool cannot retain both, one of the two is always evicted on
    /// release, so every steady-state cycle pays a fresh
    /// `cuMemHostAlloc`/`cuMemFreeHost` pair for whichever buffer the pool
    /// drops -- reintroducing issue #837's exact cost for a path meant to
    /// avoid it, and potentially costing more than the transfer it hides.
    /// Callers use this to decline a prefetch up front rather than discover
    /// the regression empirically.
    pub fn can_retain_concurrent(&self, len: usize, count: usize) -> bool {
        count <= self.max_buffers && len.saturating_mul(count) <= self.max_bytes
    }
}

/// A pinned staging buffer borrowed from a [`PinnedStagingPool`].
///
/// The buffer returns to the pool for reuse **only** via
/// [`PooledStaging::retire`], which requires a [`CopyCompleted`] witness proving
/// the buffer's host→device copy has completed. `Drop` is a leak-safe fallback:
/// it *frees* the buffer rather than returning it to the pool, so forgetting to
/// `retire` costs a re-allocation (observable via the `pinned_alloc_calls`
/// counter) but can never reuse a buffer while a copy is still reading it.
///
/// `Drop` never asserts (an assert in `Drop` triggers
/// `STATUS_STACK_BUFFER_OVERRUN` during unwind in this codebase).
pub struct PooledStaging {
    pool: Arc<PinnedStagingPool>,
    staging: Option<PinnedStaging>,
}

impl PooledStaging {
    /// Mutable view of the underlying pinned buffer, for filling before a copy.
    pub fn staging_mut(&mut self) -> &mut PinnedStaging {
        self.staging
            .as_mut()
            .expect("pooled staging present until into_inner/retire/Drop")
    }

    /// Read-only view of the pinned bytes.
    pub fn as_slice(&self) -> &[u8] {
        self.staging
            .as_ref()
            .expect("pooled staging present until into_inner/retire/Drop")
            .as_slice()
    }

    /// Take ownership of the raw buffer, cancelling the automatic drop. Used by
    /// the non-VMM upload path, which consumes and hands the buffer back so the
    /// caller can `release` it (with a completion witness) after the copy.
    pub fn into_inner(mut self) -> PinnedStaging {
        self.staging
            .take()
            .expect("pooled staging present until into_inner/retire/Drop")
    }

    /// Return this buffer to the pool for reuse. Requires a [`CopyCompleted`]
    /// witness proving the buffer's host→device copy has completed, so reuse is
    /// structurally ordered after copy completion.
    pub fn retire(mut self, completed: CopyCompleted) {
        if let Some(staging) = self.staging.take() {
            self.pool.release(staging, completed);
        }
    }
}

impl Drop for PooledStaging {
    fn drop(&mut self) {
        // Leak-safe fallback only: free the buffer, never return it to the pool.
        // Reuse must go through `retire`, which requires a `CopyCompleted`
        // witness. Never assert here (STATUS_STACK_BUFFER_OVERRUN on unwind).
        drop(self.staging.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> Option<Arc<CudaRuntime>> {
        CudaRuntime::new(0).ok().map(Arc::new)
    }

    /// The core regression guard for issue #837: a steady stream of same-size
    /// page-ins must not page-lock a fresh pinned buffer each time. Draining and
    /// returning `n` buffers of the same size must issue exactly one real
    /// `cuMemHostAlloc`, with every subsequent acquire served by reuse.
    #[test]
    fn reused_buffers_do_not_reallocate_per_page_in() {
        let Some(runtime) = runtime() else {
            eprintln!("SKIPPED (no CUDA runtime): pinned-pool reuse guard did NOT run.");
            return;
        };
        let pool = PinnedStagingPool::new(runtime);
        let len = 4 * 1024 * 1024;
        let page_ins = 64u64;
        for _ in 0..page_ins {
            let mut staging = pool.acquire(len).unwrap();
            // Touch the buffer the way a page-in fills it before its copy.
            staging.staging_mut().as_mut_slice()[..len].fill(0xAB);
            staging.retire(CopyCompleted::new_for_test()); // returns to pool
        }
        assert_eq!(
            pool.alloc_calls(),
            1,
            "a per-page-in pinned allocation silently returned: {page_ins} page-ins should \
             reuse one pooled buffer, not re-alloc"
        );
        assert_eq!(pool.reuses(), page_ins - 1);
        assert!(
            pool.alloc_calls() < page_ins,
            "pinned allocations ({}) must stay far below page-ins ({page_ins})",
            pool.alloc_calls()
        );
    }

    /// Reuse-if-large-enough: a request smaller than a retained buffer reuses
    /// it; a larger request allocates.
    #[test]
    fn reuse_if_large_enough_then_grow() {
        let Some(runtime) = runtime() else {
            eprintln!("SKIPPED (no CUDA runtime): pinned-pool sizing guard did NOT run.");
            return;
        };
        let pool = PinnedStagingPool::new(runtime);
        let big = pool.acquire(8 * 1024 * 1024).unwrap();
        big.retire(CopyCompleted::new_for_test());
        assert_eq!(pool.alloc_calls(), 1);
        // Smaller request fits the retained 8 MiB buffer — reuse, no alloc.
        let small = pool.acquire(1024 * 1024).unwrap();
        assert_eq!(pool.alloc_calls(), 1);
        assert_eq!(pool.reuses(), 1);
        small.retire(CopyCompleted::new_for_test());
        // A larger request cannot reuse the 8 MiB buffer — a fresh alloc.
        let bigger = pool.acquire(16 * 1024 * 1024).unwrap();
        assert_eq!(pool.alloc_calls(), 2);
        bigger.retire(CopyCompleted::new_for_test());
    }

    /// Retention is bounded: releasing beyond `max_buffers` drops the surplus
    /// rather than growing the free-list without limit.
    #[test]
    fn retention_is_bounded_by_buffer_count() {
        let Some(runtime) = runtime() else {
            eprintln!("SKIPPED (no CUDA runtime): pinned-pool bound guard did NOT run.");
            return;
        };
        let max_buffers = 2;
        let pool = PinnedStagingPool::with_bounds(runtime, max_buffers, usize::MAX);
        // Hold three distinct buffers live, then drop them all so all three try
        // to return at once.
        let a = pool.acquire(1024).unwrap();
        let b = pool.acquire(1024).unwrap();
        let c = pool.acquire(1024).unwrap();
        a.retire(CopyCompleted::new_for_test());
        b.retire(CopyCompleted::new_for_test());
        c.retire(CopyCompleted::new_for_test());
        assert!(
            pool.free_len() <= max_buffers,
            "free-list grew past its bound: {} > {max_buffers}",
            pool.free_len()
        );
    }

    /// Retention is bounded by total bytes: a buffer that would push retained
    /// pinned bytes over the ceiling is dropped, not kept.
    #[test]
    fn retention_is_bounded_by_bytes() {
        let Some(runtime) = runtime() else {
            eprintln!("SKIPPED (no CUDA runtime): pinned-pool byte-bound guard did NOT run.");
            return;
        };
        // Ceiling admits one 1 MiB buffer but not two.
        let pool = PinnedStagingPool::with_bounds(runtime, 8, 1024 * 1024 + 1);
        let a = pool.acquire(1024 * 1024).unwrap();
        let b = pool.acquire(1024 * 1024).unwrap();
        a.retire(CopyCompleted::new_for_test());
        b.retire(CopyCompleted::new_for_test());
        assert_eq!(
            pool.free_len(),
            1,
            "byte ceiling must cap retained pinned host memory"
        );
    }
}
