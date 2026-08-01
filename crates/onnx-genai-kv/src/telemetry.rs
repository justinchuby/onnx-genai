// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Lock-free KV page telemetry that can be read while a generation is running.
//!
//! # Why this exists
//!
//! Paged-KV allocation is a *during-generation* phenomenon: pages fill, are
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
//! acceptable for monitoring, and paying for stronger ordering on a decode hot
//! path to make two displayed numbers agree would be a bad trade.
//!
//! # Scope
//!
//! This module publishes **aggregate** pool state only. A per-page mirror, which
//! a block-table view would need, is deliberately not here: it is a larger
//! allocation with its own truncation semantics, and it has no consumer until a
//! per-page view exists to read it.

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
/// path that does not observe it pays one `Option` check per mutation and no
/// stores.
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

    /// A short, stable machine token for the reason.
    ///
    /// Separate from [`detail`](Self::detail) because a client that branches on
    /// the cause must not have to pattern-match on English prose.
    pub const fn code(self) -> &'static str {
        match self {
            Self::ContinuousBatching => "continuous-batching",
            Self::CacheCannotPage => "cache-cannot-page",
        }
    }
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
    /// **Check this before reporting anything.** The counters are all honest
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

    /// Overwrites the live gauges wholesale.
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

    #[test]
    fn applicability_starts_unknown_not_not_applicable() {
        // The whole point of the tri-state: a poll before the driver has chosen
        // a decode path must not be answered with a confident "not applicable".
        let telemetry = KvTelemetry::default();
        assert_eq!(telemetry.applicability(), Applicability::Unknown);
        assert!(!telemetry.is_applicable());
        assert_eq!(telemetry.not_applicable_reason(), None);
    }

    #[test]
    fn not_applicable_carries_its_reason() {
        for reason in [
            KvNotApplicable::ContinuousBatching,
            KvNotApplicable::CacheCannotPage,
        ] {
            let telemetry = KvTelemetry::default();
            telemetry.set_not_applicable(reason);
            assert_eq!(telemetry.applicability(), Applicability::NotApplicable);
            assert!(!telemetry.is_applicable());
            assert_eq!(telemetry.not_applicable_reason(), Some(reason));
        }
    }

    #[test]
    fn each_not_applicable_reason_has_distinct_wording_and_code() {
        // Guards the "one wording for two causes" failure this enum exists to
        // prevent: adding a variant without its own explanation must not be
        // possible silently.
        let a = KvNotApplicable::ContinuousBatching;
        let b = KvNotApplicable::CacheCannotPage;
        assert_ne!(a.detail(), b.detail());
        assert_ne!(a.code(), b.code());
        assert!(!a.detail().is_empty() && !b.detail().is_empty());
    }

    #[test]
    fn applicable_reports_no_reason() {
        let telemetry = KvTelemetry::default();
        telemetry.set_applicable();
        assert_eq!(telemetry.applicability(), Applicability::Applicable);
        assert!(telemetry.is_applicable());
        assert_eq!(
            telemetry.not_applicable_reason(),
            None,
            "a reason is meaningless unless the state is NotApplicable"
        );
    }

    #[test]
    fn ref_count_transitions_drive_both_gauges() {
        let telemetry = KvTelemetry::default();

        // 0 -> 1: now in use, not yet shared.
        telemetry.note_ref_count_change(0, 1);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_in_use, 1);
        assert_eq!(snapshot.pages_shared, 0);

        // 1 -> 2: still one page in use, but now shared.
        telemetry.note_ref_count_change(1, 2);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_in_use, 1);
        assert_eq!(snapshot.pages_shared, 1);

        // 2 -> 1: no longer shared, still in use.
        telemetry.note_ref_count_change(2, 1);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_in_use, 1);
        assert_eq!(snapshot.pages_shared, 0);

        // 1 -> 0: released.
        telemetry.note_ref_count_change(1, 0);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_in_use, 0);
        assert_eq!(snapshot.pages_shared, 0);
    }

    #[test]
    fn gauges_saturate_instead_of_underflowing() {
        // An accounting slip should read as 0, not as usize::MAX, which would
        // look like a catastrophic leak.
        let telemetry = KvTelemetry::default();
        telemetry.note_ref_count_change(1, 0);
        telemetry.note_ref_count_change(2, 1);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.pages_in_use, 0);
        assert_eq!(snapshot.pages_shared, 0);
    }

    #[test]
    fn geometry_and_live_gauges_are_published() {
        let telemetry = KvTelemetry::default();
        telemetry.set_geometry(512, 16);
        telemetry.set_live_gauges(40, 7);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.hot_capacity, 512);
        assert_eq!(snapshot.page_size, 16);
        assert_eq!(snapshot.pages_in_use, 40);
        assert_eq!(snapshot.pages_shared, 7);
    }
}
