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

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};

/// Whether the paged pool is the mechanism the decoder actually uses.
///
/// `Unknown` is a real and necessary state: the driver picks the decode path
/// asynchronously at startup, so a poll can genuinely arrive before the answer
/// exists. Collapsing it into `NotApplicable` turns "we don't know yet" into a
/// confident wrong claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applicability {
    Unknown = 0,
    Applicable = 1,
    NotApplicable = 2,
}

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
    /// Per-page mirror, one packed `AtomicU64` per page.
    ///
    /// A block table needs per-page detail, but the page table lives inside the
    /// engine on the driver thread and is unreachable from an HTTP handler.
    /// Copying it out would need either a command round-trip (impossible during
    /// an inline generation) or a lock on the decode path. Instead each page
    /// gets one atomic word, written O(1) at the exact site that changed it, so
    /// a reader always sees live per-page state without any coordination.
    ///
    /// Sized once, on attach, because the pool's capacity is not known when the
    /// telemetry is constructed.
    blocks: OnceLock<Box<[AtomicU64]>>,
    /// Pages the POOL has, as opposed to the number this mirror can describe.
    ///
    /// Kept because `size_blocks` caps the mirror and used to discard the
    /// requested size, which destroyed the only evidence that anything had
    /// been truncated. A client comparing `pages_in_use` (a true reading of
    /// the whole pool) against the mirror length (a capped one) would see
    /// pages-in-use exceed the total and have no way to learn why.
    pool_blocks: AtomicUsize,
    /// Whether the paged pool this mirrors is the one the decoder actually
    /// uses. Continuous batching and paged KV are mutually exclusive, so on a
    /// batching model every counter here is a truthful reading of a pool that
    /// is never consulted.
    applicability: AtomicU8,
    /// Why `applicability` is `NotApplicable`. Meaningless in any other state.
    ///
    /// Separate from `applicability` because that is stored as a plain `as u8`
    /// discriminant and so cannot carry a payload. Kept in lockstep by the fact
    /// that the only way to reach `NotApplicable` is
    /// [`set_not_applicable`](KvTelemetry::set_not_applicable), which demands a
    /// reason.
    not_applicable_reason: AtomicU8,
}

/// Why a paged pool is not the mechanism in play.
///
/// Two genuinely different facts reach the same `NotApplicable` state, and they
/// have different explanations. Reporting one wording for both would state the
/// wrong mechanism with full confidence -- the failure this module's
/// [`Applicability`] tri-state was introduced to stop, one level down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KvNotApplicable {
    /// The decoder batches continuously, which is mutually exclusive with
    /// paged KV. The pool is fully allocated and simply never consulted.
    ContinuousBatching = 0,
    /// This engine's KV cache owns no paged tensor storage, so its page table
    /// keeps bookkeeping the decoder can never use.
    CacheCannotPage = 1,
}

impl KvNotApplicable {
    /// The client-facing explanation. Owned here, beside the variant, so a new
    /// cause cannot be added without its wording.
    pub const fn detail(self) -> &'static str {
        match self {
            Self::ContinuousBatching => {
                "this model uses continuous batching, which is mutually exclusive \
                 with paged KV; the page pool exists but the decoder never uses it"
            }
            Self::CacheCannotPage => {
                "this model's KV cache holds no paged tensor storage, so the page \
                 table is bookkeeping the decoder never consults"
            }
        }
    }
}

/// Packs a page's mutable state into one atomic word so a reader gets a
/// self-consistent view of a single page without a lock.
///
/// Layout: `ref_count` in bits 0..16, `filled_slots` in 16..32, `tier` in
/// 32..40, and a present flag in bit 40. Saturating rather than wrapping: a
/// page with more than 65,535 references would be a bug, and displaying a
/// wrapped 0 would hide it.
fn pack_block(ref_count: u32, filled_slots: usize, tier: u8) -> u64 {
    let ref_count = ref_count.min(u32::from(u16::MAX)) as u64;
    let filled = (filled_slots.min(u16::MAX as usize)) as u64;
    ref_count | (filled << 16) | ((tier as u64) << 32) | (1 << 40)
}

/// One page's live state, as read from the mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockState {
    pub page_id: u32,
    pub ref_count: u32,
    pub filled_slots: usize,
    /// `0` = hot tier, `1` = cold tier.
    pub tier: u8,
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

    /// Declares that this pool *is* the one the decoder uses.
    ///
    /// Must be set by whoever knows the decode path -- which is only knowable
    /// once the driver has chosen one, so this is set asynchronously at
    /// startup.
    pub fn set_applicable(&self) {
        self.applicability
            .store(Applicability::Applicable as u8, Ordering::Relaxed);
    }

    /// Declares that this pool is *not* the mechanism in play, and why.
    ///
    /// The reason is required rather than optional on purpose. There is more
    /// than one way to reach this state and they have different explanations,
    /// so a caller that could omit the reason would be choosing, silently,
    /// whichever wording happened to be hardcoded downstream.
    pub fn set_not_applicable(&self, reason: KvNotApplicable) {
        self.not_applicable_reason
            .store(reason as u8, Ordering::Relaxed);
        // Release, paired with the Acquire in `not_applicable_reason()`: a
        // reader that observes `NotApplicable` must also observe the reason
        // stored just above it. Under Relaxed it could see the new state with
        // the default reason and report the wrong mechanism -- which is the
        // precise bug this enum exists to prevent, reintroduced as a data race.
        self.applicability
            .store(Applicability::NotApplicable as u8, Ordering::Release);
    }

    /// Why this pool is not in play, or `None` unless it is `NotApplicable`.
    pub fn not_applicable_reason(&self) -> Option<KvNotApplicable> {
        if self.applicability.load(Ordering::Acquire) != Applicability::NotApplicable as u8 {
            return None;
        }
        match self.not_applicable_reason.load(Ordering::Relaxed) {
            1 => Some(KvNotApplicable::CacheCannotPage),
            _ => Some(KvNotApplicable::ContinuousBatching),
        }
    }

    /// Whether these counters describe a mechanism that is actually in play.
    ///
    /// **Check this before rendering anything.** The counters are all honest
    /// reads of a real structure even when this is `false`, which is exactly
    /// what makes them dangerous: `hot_capacity` is *non-zero* on a batching
    /// model, so it survives any "is this hardcoded?" audit while describing a
    /// pool the decoder never touches. A non-zero value is not evidence that a
    /// mechanism is in use.
    pub fn is_applicable(&self) -> bool {
        self.applicability() == Applicability::Applicable
    }

    /// Whether the decode path has been chosen yet, and if so which.
    ///
    /// Tri-state on purpose. A plain bool defaulting to `false` meant that a
    /// poll arriving before the driver finished starting got the confident
    /// answer "not applicable: this model uses continuous batching" -- on a
    /// *paged* model, that is not a missing answer, it is the opposite of the
    /// truth, stated authoritatively. `Unknown` is reported as `pending`, which
    /// is a claim we can always stand behind.
    pub fn applicability(&self) -> Applicability {
        match self.applicability.load(Ordering::Relaxed) {
            1 => Applicability::Applicable,
            2 => Applicability::NotApplicable,
            _ => Applicability::Unknown,
        }
    }

    /// Sizes the per-page mirror. Called once, on attach.
    ///
    /// Capped so a pool that grows via cold-tier offload cannot make this
    /// allocation unbounded; pages beyond the cap are simply not mirrored,
    /// which a reader detects as an absent block rather than a wrong one.
    pub(crate) fn size_blocks(&self, pages: usize) {
        // Recorded BEFORE the cap. The cap is correct; discarding what it
        // capped is what made the truncation unobservable.
        self.pool_blocks.store(pages, Ordering::Relaxed);
        let capped = pages.min(Self::MAX_MIRRORED_BLOCKS);
        let _ = self.blocks.set(
            (0..capped)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
    }

    /// Upper bound on mirrored pages.
    ///
    /// A block table of more than this many cells is a texture rather than a
    /// visualisation, so there is nothing to gain from mirroring further.
    pub const MAX_MIRRORED_BLOCKS: usize = 4096;

    /// Publishes one page's state. `O(1)`, one relaxed store, no allocation.
    pub(crate) fn note_block(&self, page_id: u32, ref_count: u32, filled_slots: usize, tier: u8) {
        let Some(blocks) = self.blocks.get() else {
            return;
        };
        let Some(slot) = blocks.get(page_id as usize) else {
            return;
        };
        slot.store(pack_block(ref_count, filled_slots, tier), Ordering::Relaxed);
    }

    /// Reads a window of the block mirror, DENSE and indexed by position.
    ///
    /// `result[i]` describes page `start + i`, always, and the returned length
    /// is exactly the number of page ids examined (the window clamped to the
    /// mirror). Position is therefore structural rather than conventional: a
    /// renderer can draw `result[i]` into cell `i` and be correct by
    /// construction.
    ///
    /// This density is the contract, not an implementation detail. The previous
    /// version skipped never-written pages, which made the array SPARSE -- and
    /// a sparse array silently breaks a grid, because the first page to be
    /// written shifts every later block one cell along. That renders as a
    /// hundred blocks migrating when in fact one was allocated, which is
    /// exactly the false motion the id-indexed mirror exists to prevent. The
    /// ordering guarantee was real and tested; it protected ORDER, and a grid
    /// draws by POSITION. Those are the same property only in a dense array.
    ///
    /// `None` means the page has NEVER BEEN WRITTEN -- a genuine absence of
    /// observation, not a state of the pool. It is NOT "free": a page that was
    /// used and released keeps its present flag and comes back as `Some` with
    /// `ref_count: 0`, because that is a measurement -- we looked, and it was
    /// empty. Collapsing those two into one absence would make "we have never
    /// seen this page" indistinguishable from "this page is free right now".
    pub fn block_window(&self, start: usize, count: usize) -> Vec<Option<BlockState>> {
        let Some(blocks) = self.blocks.get() else {
            return Vec::new();
        };
        blocks
            .iter()
            .enumerate()
            .skip(start)
            .take(count)
            .map(|(page_id, slot)| {
                let packed = slot.load(Ordering::Relaxed);
                if packed & (1 << 40) == 0 {
                    return None;
                }
                Some(BlockState {
                    page_id: page_id as u32,
                    ref_count: (packed & 0xFFFF) as u32,
                    filled_slots: ((packed >> 16) & 0xFFFF) as usize,
                    tier: ((packed >> 32) & 0xFF) as u8,
                })
            })
            .collect()
    }

    /// How many pages the mirror can describe.
    pub fn mirrored_block_capacity(&self) -> usize {
        self.blocks.get().map_or(0, |blocks| blocks.len())
    }

    /// Pages the pool holds, whether or not this mirror can describe them all.
    ///
    /// Distinct from [`Self::mirrored_block_capacity`], and the difference is
    /// the whole point: `pages_in_use` is measured against THIS number, so a
    /// client handed only the mirror length can compute an occupancy greater
    /// than one and be unable to tell whether that is overload or truncation.
    pub fn pool_block_count(&self) -> usize {
        self.pool_blocks.load(Ordering::Relaxed)
    }

    /// Whether the mirror describes fewer pages than the pool holds.
    pub fn block_mirror_is_truncated(&self) -> bool {
        self.pool_block_count() > self.mirrored_block_capacity()
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

    /// A page size this system never uses anywhere else, chosen deliberately.
    ///
    /// Every fixture in this module used to say `16`, which is precisely the
    /// `page_size` a live demo server reports on `/v1/debug/kv/blocks`. A
    /// fixture whose value equals the value production runs at cannot tell a
    /// *published* geometry apart from a *hardcoded* one: both produce `16`.
    /// That was not a hypothetical. Replacing the store in `set_geometry` with
    /// a literal `16` — discarding the parameter outright — left all 126 tests
    /// in this crate green, and the server suite green with it.
    ///
    /// `7` is a value no config, default, or model in this repository produces,
    /// so a constant of any value now fails somewhere.
    const FIXTURE_PAGE_SIZE: usize = 7;

    fn pool(pages: usize) -> (PageTable, Arc<KvTelemetry>) {
        let mut table = PageTable::new(FIXTURE_PAGE_SIZE, pages);
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

    /// Geometry must be *carried* from the pool to the telemetry, not merely
    /// present in it.
    ///
    /// Asserting a single pool reports the number that pool was built with is
    /// weaker than it looks: it holds just as well if the publisher ignores its
    /// argument and stores a constant that happens to match. The only way to
    /// separate carrying from coincidence is to publish two different
    /// geometries and require the two snapshots to disagree — no single
    /// constant can satisfy both arms at once.
    #[test]
    fn geometry_is_carried_from_the_pool_and_not_a_constant() {
        let (_small, small) = pool(8);
        assert_eq!(small.snapshot().hot_capacity, 8);
        assert_eq!(small.snapshot().page_size, FIXTURE_PAGE_SIZE);

        let mut wide = PageTable::new(FIXTURE_PAGE_SIZE * 3, 5);
        let wide_telemetry = Arc::new(KvTelemetry::default());
        wide.attach_telemetry(Arc::clone(&wide_telemetry));

        assert_eq!(wide_telemetry.snapshot().hot_capacity, 5);
        assert_eq!(wide_telemetry.snapshot().page_size, FIXTURE_PAGE_SIZE * 3);

        // The load-bearing half: two pools alive at once must not agree.
        assert_ne!(
            small.snapshot().page_size,
            wide_telemetry.snapshot().page_size,
            "both pools report the same page size, so the geometry is a \
             constant rather than a measurement of the pool"
        );
        assert_ne!(
            small.snapshot().hot_capacity,
            wide_telemetry.snapshot().hot_capacity,
            "both pools report the same capacity, so the capacity is a \
             constant rather than a measurement of the pool"
        );
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

    /// The block mirror is what makes a per-page view readable from an HTTP
    /// handler at all: the page table itself is owned by the driver thread and
    /// mutably borrowed for the whole of a generation.
    #[test]
    fn block_mirror_reflects_allocation_and_sharing() {
        let (mut table, telemetry) = pool(8);
        // The pool preallocates its pages, so a block table shows all eight
        // from the start, every one free. That is the correct picture: the
        // blocks exist, and seeing them fill is the point of the view.
        let initial = telemetry.block_window(0, 8);
        assert_eq!(initial.len(), 8);
        assert!(
            initial
                .iter()
                .all(|b| b.expect("preallocated pages are observed").ref_count == 0),
            "an untouched pool holds no references"
        );

        let first = table.allocate(GPU).expect("pool has capacity");
        let second = table.allocate(GPU).expect("pool has capacity");
        table.retain(first);

        let blocks = telemetry.block_window(0, 8);
        assert_eq!(blocks.len(), 8);
        assert_eq!(
            blocks
                .iter()
                .filter(|b| b.is_some_and(|b| b.ref_count > 0))
                .count(),
            2,
            "exactly the two allocated pages are in use"
        );

        let a = blocks[first as usize].expect("an allocated page is observed");
        let b = blocks[second as usize].expect("an allocated page is observed");
        assert_eq!(a.ref_count, 2, "retained page is shared");
        assert_eq!(b.ref_count, 1);
        assert_eq!(a.tier, 0, "freshly allocated pages are hot");
    }

    /// A freed page must report as free, not linger as occupied.
    #[test]
    fn block_mirror_follows_a_page_back_to_zero_references() {
        let (mut table, telemetry) = pool(8);
        let page = table.allocate(GPU).expect("pool has capacity");
        table.free(page);

        let block = telemetry.block_window(0, 8)[page as usize]
            .expect("a freed page is still a real page, reported with zero refs");
        assert_eq!(block.ref_count, 0);
    }

    /// Block ids are the mirror's indices, so a block cannot move between
    /// polls. This is stronger than sorting: there is no ordering step to get
    /// wrong.
    #[test]
    fn block_mirror_ordering_is_positional_and_stable() {
        let (mut table, telemetry) = pool(16);
        for _ in 0..10 {
            table.allocate(GPU).expect("pool has capacity");
        }
        let ids = |w: Vec<Option<BlockState>>| -> Vec<Option<u32>> {
            w.iter().map(|b| b.map(|b| b.page_id)).collect()
        };
        let first = ids(telemetry.block_window(0, 16));
        for _ in 0..5 {
            assert_eq!(first, ids(telemetry.block_window(0, 16)));
        }
        // Position IS the page id, so the window is dense and ascending with
        // no ordering step that could get it wrong.
        for (offset, id) in first.iter().enumerate() {
            assert_eq!(*id, Some(offset as u32));
        }
    }

    #[test]
    fn block_window_is_bounded_and_pages_cleanly() {
        let (mut table, telemetry) = pool(16);
        for _ in 0..16 {
            table.allocate(GPU).expect("pool has capacity");
        }
        assert_eq!(telemetry.mirrored_block_capacity(), 16);
        assert_eq!(telemetry.block_window(0, 5).len(), 5);
        assert_eq!(telemetry.block_window(14, 5).len(), 2, "clamps at the end");
        assert!(telemetry.block_window(100, 5).is_empty());
    }

    /// The mirror is capped so a pool that grows via cold-tier offload cannot
    /// make the allocation unbounded.
    #[test]
    fn block_mirror_is_capped() {
        let telemetry = KvTelemetry::default();
        telemetry.size_blocks(KvTelemetry::MAX_MIRRORED_BLOCKS * 4);
        assert_eq!(
            telemetry.mirrored_block_capacity(),
            KvTelemetry::MAX_MIRRORED_BLOCKS
        );
    }

    /// The cap must not destroy the evidence that it fired.
    ///
    /// `size_blocks` used to keep only the capped length, so a pool larger than
    /// the mirror became indistinguishable from one that fit exactly. No
    /// endpoint downstream could recover the difference, which is what let the
    /// block table publish a total smaller than its own `pages_in_use`.
    #[test]
    fn a_capped_mirror_still_remembers_the_pool_it_could_not_hold() {
        let telemetry = KvTelemetry::default();
        let oversized = KvTelemetry::MAX_MIRRORED_BLOCKS * 3;
        telemetry.size_blocks(oversized);

        assert_eq!(
            telemetry.mirrored_block_capacity(),
            KvTelemetry::MAX_MIRRORED_BLOCKS,
            "the cap must still bite"
        );
        assert_eq!(
            telemetry.pool_block_count(),
            oversized,
            "the requested size must survive the cap"
        );
        assert!(telemetry.block_mirror_is_truncated());
    }

    /// A pool that fits must not claim truncation, or the flag is decorative.
    #[test]
    fn a_mirror_that_holds_the_whole_pool_reports_no_truncation() {
        let telemetry = KvTelemetry::default();
        telemetry.size_blocks(40);

        assert_eq!(telemetry.pool_block_count(), 40);
        assert_eq!(telemetry.mirrored_block_capacity(), 40);
        assert!(!telemetry.block_mirror_is_truncated());
    }

    /// Packing must round-trip, including the boundary values that would
    /// corrupt neighbouring fields if a shift or mask were wrong.
    #[test]
    fn packed_block_fields_do_not_bleed_into_each_other() {
        let telemetry = KvTelemetry::default();
        telemetry.size_blocks(4);
        telemetry.note_block(0, u16::MAX as u32, u16::MAX as usize, 1);
        telemetry.note_block(1, 0, 0, 0);

        let blocks = telemetry.block_window(0, 4);
        // DENSE: four page ids examined, four entries. Two are observed and
        // two have never been written. Before this was dense the same call
        // returned LENGTH 2, so `blocks[1]` was page 1 only by coincidence --
        // and had page 0 been the unwritten one, `blocks[0]` would have been
        // page 1 while rendering into page 0's cell.
        assert_eq!(blocks.len(), 4, "the window is dense over the ids examined");
        let zero = blocks[0].expect("page 0 was written");
        let one = blocks[1].expect("page 1 was written");
        assert_eq!(zero.page_id, 0);
        assert_eq!(one.page_id, 1);
        assert_eq!(zero.ref_count, u16::MAX as u32);
        assert_eq!(zero.filled_slots, u16::MAX as usize);
        assert_eq!(zero.tier, 1);
        assert_eq!(one.ref_count, 0);
        assert_eq!(one.filled_slots, 0);
        assert_eq!(one.tier, 0);
        assert!(
            blocks[2].is_none() && blocks[3].is_none(),
            "a page never written is an absence of observation, and must not \
             be reported as a page with zero references -- that would be a \
             measurement we never took"
        );
    }

    /// D256: a page appearing must not move the pages after it.
    ///
    /// This is the defect the dense window exists to prevent, driven exactly
    /// as the demo drives it: a mirror larger than the set of pages written so
    /// far, filling up over time. With a sparse window every later block
    /// shifts one cell each time an earlier page is first written, so a grid
    /// renders a hundred blocks migrating when one was allocated.
    #[test]
    fn a_page_becoming_observed_does_not_shift_the_pages_after_it() {
        let telemetry = KvTelemetry::default();
        telemetry.size_blocks(6);
        telemetry.note_block(3, 1, 4, 0);
        telemetry.note_block(5, 2, 8, 1);

        let position_of = |w: &[Option<BlockState>], id: u32| {
            w.iter()
                .position(|b| b.is_some_and(|b| b.page_id == id))
                .expect("page is observed")
        };

        let before = telemetry.block_window(0, 6);
        assert_eq!(position_of(&before, 3), 3);
        assert_eq!(position_of(&before, 5), 5);

        // A page EARLIER in the window is written for the first time. Under a
        // sparse window this inserts an element and pushes 3 and 5 along.
        telemetry.note_block(0, 1, 2, 0);
        let after = telemetry.block_window(0, 6);

        assert_eq!(
            after.len(),
            before.len(),
            "the window length is the ids examined, not the ids observed"
        );
        assert_eq!(
            position_of(&after, 3),
            3,
            "page 3 moved when page 0 was allocated; a block table that \
             reshuffles renders motion that never happened"
        );
        assert_eq!(
            position_of(&after, 5),
            5,
            "page 5 moved when page 0 was allocated"
        );
    }

    #[test]
    fn gauges_saturate_rather_than_underflowing() {
        let telemetry = KvTelemetry::default();
        telemetry.note_ref_count_change(1, 0);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_in_use, 0, "must not wrap to usize::MAX");
        assert_eq!(snapshot.pages_shared, 0);
    }

    /// A pool nobody has classified must not claim the decoder skips it.
    ///
    /// This is the test that would have caught the original bug. `applicable`
    /// was a bool defaulting to `false`, and `false` rendered as the confident
    /// sentence "this model uses continuous batching" -- so during the window
    /// before the driver finished starting, a paged model asserted the exact
    /// opposite of the truth. The bug was invisible to every existing test
    /// because the driver almost always won the race.
    #[test]
    fn an_unclassified_pool_says_unknown_rather_than_not_applicable() {
        let telemetry = KvTelemetry::default();
        assert_eq!(telemetry.applicability(), Applicability::Unknown);
        assert!(
            !telemetry.is_applicable(),
            "unknown must still gate rendering"
        );
        assert_ne!(
            telemetry.applicability(),
            Applicability::NotApplicable,
            "not knowing yet is not the same claim as knowing it does not apply"
        );
    }

    #[test]
    fn classifying_a_pool_moves_it_off_unknown_in_both_directions() {
        let paged = KvTelemetry::default();
        paged.set_applicable();
        assert_eq!(paged.applicability(), Applicability::Applicable);
        assert!(paged.is_applicable());
        assert_eq!(
            paged.not_applicable_reason(),
            None,
            "an applicable pool has no reason for not applying"
        );

        let batching = KvTelemetry::default();
        batching.set_not_applicable(KvNotApplicable::ContinuousBatching);
        assert_eq!(batching.applicability(), Applicability::NotApplicable);
        assert!(!batching.is_applicable());
    }

    /// The two causes of `NotApplicable` must not share one explanation.
    ///
    /// Before this, the block-table route hardcoded the continuous-batching
    /// sentence for every `NotApplicable` pool. A model whose KV cache simply
    /// holds no paged tensors would have been told, with total confidence, that
    /// it was using a batching mechanism it may not have. A wrong mechanism
    /// stated fluently is worse than no explanation: it terminates the reader's
    /// inquiry.
    #[test]
    fn each_cause_of_not_applicable_explains_itself_differently() {
        let batching = KvTelemetry::default();
        batching.set_not_applicable(KvNotApplicable::ContinuousBatching);
        let unpaged = KvTelemetry::default();
        unpaged.set_not_applicable(KvNotApplicable::CacheCannotPage);

        assert_eq!(
            batching.not_applicable_reason(),
            Some(KvNotApplicable::ContinuousBatching)
        );
        assert_eq!(
            unpaged.not_applicable_reason(),
            Some(KvNotApplicable::CacheCannotPage)
        );

        let batching_detail = KvNotApplicable::ContinuousBatching.detail();
        let unpaged_detail = KvNotApplicable::CacheCannotPage.detail();
        assert_ne!(
            batching_detail, unpaged_detail,
            "two different facts must not be reported with one sentence"
        );
        assert!(
            !unpaged_detail.contains("continuous batching"),
            "a cache that cannot page must not be explained as batching: {unpaged_detail}"
        );
        assert!(
            batching_detail.contains("continuous batching"),
            "the batching cause must still name batching: {batching_detail}"
        );
    }

    /// `Unknown` carries no reason, so a reader cannot render an explanation
    /// for a state that has not been decided yet.
    #[test]
    fn an_undecided_pool_offers_no_explanation_to_render() {
        let telemetry = KvTelemetry::default();
        assert_eq!(
            telemetry.not_applicable_reason(),
            None,
            "a pending pool must not hand out the default reason"
        );
    }
}
