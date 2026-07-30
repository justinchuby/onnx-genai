// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Lock-free KV page telemetry that can be read while a generation is running.
//!
//! # Why this exists
//!
//! Paged-KV allocation is a *during-generation* phenomenon: blocks fill, are
//! shared, and are evicted as tokens are produced. Sampling the pool between
//! requests shows a pool at rest, which is not a weaker version of that story
//! but the opposite of it.
//!
//! Reading it during generation was structurally impossible before this type.
//! Both driver loops run generation inline, so a command round-trip cannot be
//! serviced while a generation is in flight; the per-token callback cannot help
//! either, because the engine is mutably borrowed for the duration; and there
//! was nothing to share, because the page table is owned outright rather than
//! behind an `Arc`.
//!
//! # Why atomics rather than a snapshot channel
//!
//! [`PageTable::usage`](crate::page_table::PageTable::usage) is `O(pages)` and
//! allocates: it walks every page, builds a `BTreeMap` of reference counts, a
//! per-tier `Vec`, a per-sequence `Vec`, and sorts. Against a measured 14,612
//! pages that is far too expensive to run per decode step.
//!
//! These counters are instead maintained incrementally at the few sites that
//! change them, so publishing is a handful of relaxed stores: wait-free, no
//! allocation, and cheaper in the decode loop than servicing a command would
//! have been. Readers never touch the driver thread at all.
//!
//! Relaxed ordering is deliberate. These are independent gauges for human
//! display, not a consistent set: a reader may observe a snapshot in which
//! `pages_in_use` and `pages_shared` come from different instants. That is
//! acceptable at a 4 Hz refresh, and paying for stronger ordering on a decode
//! hot path to make a dashboard's two numbers agree would be a bad trade.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Live, lock-free view of a page pool.
///
/// Held as `Option<Arc<KvTelemetry>>` so it costs nothing when absent: the
/// non-demo path pays one `Option` check per mutation and no stores.
#[derive(Debug, Default)]
pub struct KvTelemetry {
    pages_in_use: AtomicUsize,
    pages_shared: AtomicUsize,
    hot_capacity: AtomicUsize,
    page_size: AtomicUsize,
    allocations: AtomicU64,
    allocation_failures: AtomicU64,
    frees: AtomicU64,
    hot_evictions: AtomicU64,
    prefix_evictions: AtomicU64,
}

/// A plain-data read of [`KvTelemetry`], safe to serialise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvTelemetrySnapshot {
    /// Pages with at least one reference, on any tier.
    ///
    /// May exceed `hot_capacity`: eviction demotes a page to the cold tier by
    /// changing its device, and does not drop the reference, so an evicted page
    /// is still in use. Callers deriving a utilisation ratio must account for
    /// that rather than assuming the value is bounded by capacity.
    pub pages_in_use: usize,
    /// Pages with more than one reference, i.e. genuinely shared by
    /// copy-on-write or prefix reuse.
    pub pages_shared: usize,
    /// Hot-tier live page capacity.
    pub hot_capacity: usize,
    /// Token slots per page.
    pub page_size: usize,
    pub allocations: u64,
    /// Allocations that found no page: the pool was exhausted. The honest
    /// signal that the pool is under real pressure.
    pub allocation_failures: u64,
    pub frees: u64,
    pub hot_evictions: u64,
    pub prefix_evictions: u64,
}

impl KvTelemetry {
    /// Records the pool's fixed geometry. Called once, at construction.
    pub(crate) fn set_geometry(&self, hot_capacity: usize, page_size: usize) {
        self.hot_capacity.store(hot_capacity, Ordering::Relaxed);
        self.page_size.store(page_size, Ordering::Relaxed);
    }

    /// Seeds the live gauges from a directly-computed count.
    ///
    /// Used when attaching to an already-warm pool, so the gauges do not start
    /// from a zero that was never true.
    pub(crate) fn set_live_gauges(&self, in_use: usize, shared: usize) {
        self.pages_in_use.store(in_use, Ordering::Relaxed);
        self.pages_shared.store(shared, Ordering::Relaxed);
    }

    /// Adjusts the live gauges for a single page's reference-count transition.
    ///
    /// Both gauges are edge-triggered rather than recomputed: a page enters
    /// `in_use` when its count leaves zero and leaves when it returns, and
    /// enters `shared` when its count passes one in either direction.
    pub(crate) fn note_ref_count_change(&self, old: u32, new: u32) {
        match (old == 0, new == 0) {
            (true, false) => {
                self.pages_in_use.fetch_add(1, Ordering::Relaxed);
            }
            (false, true) => {
                decrement(&self.pages_in_use);
            }
            _ => {}
        }
        match (old > 1, new > 1) {
            (false, true) => {
                self.pages_shared.fetch_add(1, Ordering::Relaxed);
            }
            (true, false) => {
                decrement(&self.pages_shared);
            }
            _ => {}
        }
    }

    /// Mirrors the cumulative counters.
    ///
    /// Copies the whole set rather than incrementing individually so a counter
    /// added later cannot be silently omitted from the published view.
    pub(crate) fn publish_counters(&self, stats: &crate::page_table::PageStats) {
        self.allocations.store(stats.allocations, Ordering::Relaxed);
        self.allocation_failures
            .store(stats.allocation_failures, Ordering::Relaxed);
        self.frees.store(stats.frees, Ordering::Relaxed);
        self.hot_evictions
            .store(stats.hot_evictions, Ordering::Relaxed);
        self.prefix_evictions
            .store(stats.prefix_evictions, Ordering::Relaxed);
    }

    /// Reads every counter. Never blocks, and never touches the decode thread.
    pub fn snapshot(&self) -> KvTelemetrySnapshot {
        KvTelemetrySnapshot {
            pages_in_use: self.pages_in_use.load(Ordering::Relaxed),
            pages_shared: self.pages_shared.load(Ordering::Relaxed),
            hot_capacity: self.hot_capacity.load(Ordering::Relaxed),
            page_size: self.page_size.load(Ordering::Relaxed),
            allocations: self.allocations.load(Ordering::Relaxed),
            allocation_failures: self.allocation_failures.load(Ordering::Relaxed),
            frees: self.frees.load(Ordering::Relaxed),
            hot_evictions: self.hot_evictions.load(Ordering::Relaxed),
            prefix_evictions: self.prefix_evictions.load(Ordering::Relaxed),
        }
    }
}

/// Saturating decrement. A gauge that underflows to `usize::MAX` would render
/// as a catastrophic-looking number from a trivial accounting slip, which is a
/// worse failure than being one off.
fn decrement(counter: &AtomicUsize) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_sub(1))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use crate::page_table::PageTable;
    use std::sync::Arc;

    const GPU: Device = Device::Gpu(0);

    fn pool(pages: usize) -> (PageTable, Arc<KvTelemetry>) {
        let mut table = PageTable::new(16, pages);
        let telemetry = Arc::new(KvTelemetry::default());
        table.attach_telemetry(Arc::clone(&telemetry));
        (table, telemetry)
    }

    /// The incrementally-maintained gauges must equal a direct count of the
    /// pages. This is the safety net for the whole design: the gauges are
    /// edge-triggered, so a missed transition would drift silently and forever.
    fn assert_no_drift(table: &PageTable, telemetry: &KvTelemetry, context: &str) {
        let (in_use, shared) = table.live_page_counts();
        let snapshot = telemetry.snapshot();
        assert_eq!(
            (snapshot.pages_in_use, snapshot.pages_shared),
            (in_use, shared),
            "telemetry drifted from the page table after {context}"
        );
    }

    #[test]
    fn geometry_is_published_on_attach() {
        let (_table, telemetry) = pool(8);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.hot_capacity, 8);
        assert_eq!(snapshot.page_size, 16);
    }

    #[test]
    fn allocate_and_free_track_pages_in_use() {
        let (mut table, telemetry) = pool(8);
        assert_eq!(telemetry.snapshot().pages_in_use, 0);

        let first = table.allocate(GPU).expect("pool has capacity");
        let second = table.allocate(GPU).expect("pool has capacity");
        assert_eq!(telemetry.snapshot().pages_in_use, 2);
        assert_no_drift(&table, &telemetry, "two allocations");

        table.free(first);
        assert_eq!(telemetry.snapshot().pages_in_use, 1);
        table.free(second);
        assert_eq!(telemetry.snapshot().pages_in_use, 0);
        assert_no_drift(&table, &telemetry, "freeing both");
    }

    /// A retained page is shared, and stays in use until *every* reference is
    /// dropped. Freeing it once must move `shared` without moving `in_use`.
    #[test]
    fn retain_tracks_shared_pages_independently_of_in_use() {
        let (mut table, telemetry) = pool(8);
        let page = table.allocate(GPU).expect("pool has capacity");
        assert_eq!(telemetry.snapshot().pages_shared, 0);

        assert!(table.retain(page));
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_shared, 1);
        assert_eq!(snapshot.pages_in_use, 1, "sharing does not add a page");
        assert_no_drift(&table, &telemetry, "retain");

        table.free(page);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_shared, 0, "one reference left, so unshared");
        assert_eq!(snapshot.pages_in_use, 1, "still referenced once");
        assert_no_drift(&table, &telemetry, "first free of a shared page");

        table.free(page);
        assert_eq!(telemetry.snapshot().pages_in_use, 0);
        assert_no_drift(&table, &telemetry, "final free");
    }

    /// Deep sharing must not double-count: `shared` is "more than one
    /// reference", not "number of references".
    #[test]
    fn repeated_retain_counts_the_page_once() {
        let (mut table, telemetry) = pool(8);
        let page = table.allocate(GPU).expect("pool has capacity");
        for _ in 0..5 {
            assert!(table.retain(page));
        }
        assert_eq!(telemetry.snapshot().pages_shared, 1);
        assert_no_drift(&table, &telemetry, "five retains");

        for _ in 0..5 {
            table.free(page);
        }
        assert_eq!(telemetry.snapshot().pages_shared, 0);
        assert_no_drift(&table, &telemetry, "five frees");
    }

    /// Allocating past `hot_capacity` does **not** fail: the pool demotes the
    /// least-recently-used page to the cold tier and grows. Two consequences
    /// that any consumer of this telemetry must handle:
    ///
    /// 1. `pages_in_use` can exceed `hot_capacity`, because demotion changes a
    ///    page's device without dropping its reference. A naive
    ///    `pages_in_use / hot_capacity` utilisation ratio will exceed 1.0.
    /// 2. `allocation_failures` stays at zero under this kind of pressure, so
    ///    it is not the signal that the pool is full. `hot_evictions` is.
    #[test]
    fn allocating_past_capacity_evicts_and_grows_rather_than_failing() {
        let (mut table, telemetry) = pool(2);
        assert!(table.allocate(GPU).is_some());
        assert!(table.allocate(GPU).is_some());
        assert!(
            table.allocate(GPU).is_some(),
            "the pool grows by demoting to the cold tier"
        );

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.allocations, 3);
        assert_eq!(
            snapshot.allocation_failures, 0,
            "pressure surfaces as eviction, not failure"
        );
        assert!(
            snapshot.hot_evictions > 0,
            "demotion to the cold tier is the real pressure signal"
        );
        assert!(
            snapshot.pages_in_use > snapshot.hot_capacity,
            "in_use is unbounded by hot_capacity once pages are demoted"
        );
        assert_no_drift(&table, &telemetry, "allocating past capacity");
    }

    #[test]
    fn frees_counter_only_moves_when_the_last_reference_drops() {
        let (mut table, telemetry) = pool(8);
        let page = table.allocate(GPU).expect("pool has capacity");
        table.retain(page);

        table.free(page);
        assert_eq!(
            telemetry.snapshot().frees,
            0,
            "a page with references left has not been freed"
        );
        table.free(page);
        assert_eq!(telemetry.snapshot().frees, 1);
    }

    /// A deterministic churn of interleaved allocate/retain/free. The point is
    /// the drift assertion after *every* operation: a transition missed only in
    /// an uncommon ordering would be invisible to the focused tests above.
    #[test]
    fn interleaved_churn_never_drifts() {
        let (mut table, telemetry) = pool(16);
        let mut live: Vec<u32> = Vec::new();

        for step in 0..200_u32 {
            match step % 4 {
                0 | 1 => {
                    if let Some(page) = table.allocate(GPU) {
                        live.push(page);
                    }
                }
                2 => {
                    if let Some(&page) = live.get((step as usize / 7) % live.len().max(1)) {
                        table.retain(page);
                        live.push(page);
                    }
                }
                _ => {
                    if !live.is_empty() {
                        let index = (step as usize / 3) % live.len();
                        let page = live.swap_remove(index);
                        table.free(page);
                    }
                }
            }
            assert_no_drift(&table, &telemetry, &format!("step {step}"));
        }
    }

    /// Reattaching to a warm pool must adopt its occupancy rather than
    /// restarting from a zero that was never true.
    #[test]
    fn attaching_to_a_warm_pool_adopts_current_occupancy() {
        let mut table = PageTable::new(16, 8);
        let first = table.allocate(GPU).expect("pool has capacity");
        table.allocate(GPU).expect("pool has capacity");
        table.retain(first);

        let telemetry = Arc::new(KvTelemetry::default());
        table.attach_telemetry(Arc::clone(&telemetry));

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_in_use, 2);
        assert_eq!(snapshot.pages_shared, 1);
        assert_eq!(snapshot.allocations, 2, "prior activity is adopted too");
        assert_no_drift(&table, &telemetry, "attaching to a warm pool");
    }

    /// The pool must behave identically with no telemetry attached; this is the
    /// default path for every non-demo caller.
    #[test]
    fn pool_without_telemetry_still_works() {
        let mut table = PageTable::new(16, 4);
        let page = table.allocate(GPU).expect("pool has capacity");
        table.retain(page);
        table.free(page);
        table.free(page);
        assert_eq!(table.live_page_counts(), (0, 0));
    }

    /// The point of the whole design: a reader on another thread observes the
    /// pool changing *while* it is being mutated, without a lock, a channel, or
    /// any cooperation from the mutating thread. This is what a command
    /// round-trip could not do, because generation runs inline on that thread.
    #[test]
    fn telemetry_is_readable_from_another_thread_during_mutation() {
        use std::sync::atomic::{AtomicBool, Ordering as O};
        use std::thread;

        let (mut table, telemetry) = pool(64);
        let stop = Arc::new(AtomicBool::new(false));

        let reader_telemetry = Arc::clone(&telemetry);
        let reader_stop = Arc::clone(&stop);
        let reader = thread::spawn(move || {
            let mut observed_nonzero = false;
            let mut reads = 0_u64;
            while !reader_stop.load(O::Relaxed) {
                let snapshot = reader_telemetry.snapshot();
                if snapshot.pages_in_use > 0 {
                    observed_nonzero = true;
                }
                assert_eq!(snapshot.hot_capacity, 64, "geometry must stay stable");
                reads += 1;
            }
            (observed_nonzero, reads)
        });

        let mut live = Vec::new();
        for _ in 0..2_000 {
            for _ in 0..16 {
                if let Some(page) = table.allocate(GPU) {
                    live.push(page);
                }
            }
            while let Some(page) = live.pop() {
                table.free(page);
            }
        }

        stop.store(true, O::Relaxed);
        let (observed_nonzero, reads) = reader.join().expect("reader thread panicked");

        assert!(reads > 0, "reader never ran");
        assert!(
            observed_nonzero,
            "reader never saw the pool occupied, so it was not observing live state"
        );
        assert_no_drift(&table, &telemetry, "concurrent reads");
    }

    /// A cloned pool is a *separate* pool. If it inherited the mirror, both
    /// would publish into one set of gauges and every counter would double.
    #[test]
    fn cloning_a_pool_does_not_share_its_telemetry() {
        let (mut table, telemetry) = pool(8);
        table.allocate(GPU).expect("pool has capacity");
        assert_eq!(telemetry.snapshot().pages_in_use, 1);

        let mut cloned = table.clone();
        cloned.allocate(GPU).expect("clone has capacity");
        cloned.allocate(GPU).expect("clone has capacity");

        assert_eq!(
            telemetry.snapshot().pages_in_use,
            1,
            "the clone must not publish into the original's gauges"
        );
        assert_no_drift(&table, &telemetry, "cloning the pool");
    }

    /// Block ordering must be stable across polls. `pages` is a `HashMap`, so
    /// unsorted iteration would reshuffle the block table on every refresh and
    /// render motion that never happened.
    #[test]
    fn block_window_ordering_is_stable_across_calls() {
        let (mut table, _telemetry) = pool(32);
        for _ in 0..20 {
            table.allocate(GPU).expect("pool has capacity");
        }

        let first: Vec<_> = table.block_window(0, 20).iter().map(|b| b.id).collect();
        for _ in 0..10 {
            let again: Vec<_> = table.block_window(0, 20).iter().map(|b| b.id).collect();
            assert_eq!(first, again, "block order must not change between polls");
        }
        assert!(
            first.windows(2).all(|w| w[0] < w[1]),
            "must be ordered by id"
        );
    }

    #[test]
    fn block_window_pages_through_without_gaps_or_repeats() {
        let (mut table, _telemetry) = pool(16);
        for _ in 0..16 {
            table.allocate(GPU).expect("pool has capacity");
        }
        let total = table.total_pages();
        assert!(total >= 16);

        let mut paged = Vec::new();
        let mut start = 0;
        while start < total {
            let window = table.block_window(start, 5);
            assert!(window.len() <= 5);
            paged.extend(window.iter().map(|b| b.id));
            start += 5;
        }

        let whole: Vec<_> = table.block_window(0, total).iter().map(|b| b.id).collect();
        assert_eq!(paged, whole, "paging must cover exactly the whole pool");
    }

    #[test]
    fn block_window_beyond_the_end_is_empty_rather_than_panicking() {
        let (mut table, _telemetry) = pool(4);
        table.allocate(GPU).expect("pool has capacity");
        assert!(table.block_window(10_000, 10).is_empty());
        assert!(table.block_window(0, 0).is_empty());
    }

    /// The block table exists to show fragmentation and sharing, so those two
    /// fields must reflect reality rather than pool-level aggregates.
    #[test]
    fn block_window_reports_per_page_sharing() {
        let (mut table, _telemetry) = pool(8);
        let shared = table.allocate(GPU).expect("pool has capacity");
        let solo = table.allocate(GPU).expect("pool has capacity");
        table.retain(shared);

        let blocks = table.block_window(0, 8);
        let shared_block = blocks.iter().find(|b| b.id == shared).expect("present");
        let solo_block = blocks.iter().find(|b| b.id == solo).expect("present");

        assert_eq!(shared_block.ref_count, 2);
        assert_eq!(solo_block.ref_count, 1);
    }

    #[test]
    fn gauges_saturate_rather_than_underflowing() {
        let telemetry = KvTelemetry::default();
        telemetry.note_ref_count_change(1, 0);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_in_use, 0, "must not wrap to usize::MAX");
        assert_eq!(snapshot.pages_shared, 0);
    }
}
