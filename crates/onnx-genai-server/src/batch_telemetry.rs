//! Live occupancy of the batch the decoder is actually stepping.
//!
//! # Why this exists when `metrics::REGISTRY.batch_size` already counts things
//!
//! `REGISTRY.batch_size` is incremented by `GenerationMetrics::start()`, once
//! per *HTTP generation*, and decremented on drop. It is a truthful count of
//! requests being served. It is *not* the width of the batch, and the two are
//! bounded by different limits:
//!
//! ```text
//!   REGISTRY.batch_size      bounded by  max_queue_depth   (admission, 256)
//!   effective_batch_capacity  =  min(max_batch, max_queue_depth)      (4)
//! ```
//!
//! Pairing them produces a fraction whose numerator and denominator count
//! different populations. Measured on a live demo server: six concurrent
//! requests against a reported capacity of four yielded `batch_in_flight = 6,
//! batch_capacity = 4`. The ratio was clamped to `1.0` and looked fine; the
//! raw pair renders as **"6 of 4"**. The clamp was concealing an
//! incommensurable pair rather than a saturated server.
//!
//! # The invariant this type exists to carry
//!
//! Both terms are read from the same object at the same instant. In the
//! continuous batch manager `max_batch()` is `rows.len()` and `active_len()`
//! counts occupied slots of that same array, so `active <= capacity` is
//! structural: there is no way to occupy more slots than exist. The
//! per-request paths publish a capacity of one and hold at most one row, for
//! the same reason.
//!
//! So an over-unity ratio is not forbidden here, it is unrepresentable.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A batch occupancy reading in which both terms came from the same source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchOccupancy {
    /// Rows the decoder is stepping right now.
    pub(crate) active: u64,
    /// Requests admitted and waiting for a free row.
    pub(crate) queued: u64,
    /// Rows that exist to be filled. Never smaller than `active`.
    pub(crate) capacity: u64,
}

/// Publishes batch occupancy from a driver thread to the status route.
///
/// Deliberately not routed through `DriverCommand`: both driver loops run
/// generation inline, so a command cannot be serviced mid-generation, and
/// occupancy is only interesting *during* one. This mirrors `KvTelemetry`.
#[derive(Debug, Default)]
pub(crate) struct BatchTelemetry {
    active: AtomicU64,
    queued: AtomicU64,
    capacity: AtomicU64,
    /// Widest batch this driver has ever assembled.
    ///
    /// The instantaneous `active` reading cannot answer "did continuous
    /// batching actually overlap anything?": the loop publishes `(0, 0,
    /// capacity)` when the batch drains, so by the time anyone reads it the
    /// evidence is gone. A poller can miss the overlap entirely by sampling
    /// between batches, which makes a working server indistinguishable from a
    /// serialising one at exactly the moment the distinction matters.
    ///
    /// Monotonic and never reset: this is a claim about what the process has
    /// demonstrated, not about what it is doing now.
    peak_active: AtomicU64,
    /// False until a driver loop publishes. Distinguishes "this build never
    /// reached a batching path" from "the batch is empty", which are the same
    /// zeroes and very different facts.
    observed: AtomicBool,
}

impl BatchTelemetry {
    /// Records one reading. Callers must pass terms read from a single source
    /// at a single instant; that co-location is the whole point of the type.
    pub(crate) fn publish(&self, active: usize, queued: usize, capacity: usize) {
        debug_assert!(
            active <= capacity,
            "batch occupancy {active} exceeds capacity {capacity}; the terms \
             are not from the same source"
        );
        self.active.store(active as u64, Ordering::Relaxed);
        self.queued.store(queued as u64, Ordering::Relaxed);
        self.capacity.store(capacity as u64, Ordering::Relaxed);
        self.peak_active.fetch_max(active as u64, Ordering::Relaxed);
        self.observed.store(true, Ordering::Release);
    }

    /// The widest batch assembled so far, or `None` if no loop has published.
    pub(crate) fn peak_active(&self) -> Option<u64> {
        if !self.observed.load(Ordering::Acquire) {
            return None;
        }
        Some(self.peak_active.load(Ordering::Relaxed))
    }

    /// Returns `None` until a driver loop has published, so the route omits
    /// the fields rather than serving a zero nobody measured.
    pub(crate) fn snapshot(&self) -> Option<BatchOccupancy> {
        if !self.observed.load(Ordering::Acquire) {
            return None;
        }
        let capacity = self.capacity.load(Ordering::Relaxed);
        let active = self.active.load(Ordering::Relaxed);
        Some(BatchOccupancy {
            // Clamped defensively so a torn read across the three relaxed
            // loads can never publish an over-unity pair to a client. The
            // source cannot produce one; this makes the wire agree even
            // mid-update.
            active: active.min(capacity),
            queued: self.queued.load(Ordering::Relaxed),
            capacity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unpublished gauge must be absent, not zero. A zero here would be
    /// indistinguishable from an idle batch, and the demo's whole claim is
    /// that its numbers are measured.
    #[test]
    fn occupancy_is_absent_until_a_driver_publishes() {
        let telemetry = BatchTelemetry::default();
        assert_eq!(telemetry.snapshot(), None);

        telemetry.publish(0, 0, 4);
        assert_eq!(
            telemetry.snapshot(),
            Some(BatchOccupancy {
                active: 0,
                queued: 0,
                capacity: 4,
            }),
            "an idle batch is a measurement and must be distinguishable from \
             never having run"
        );
    }

    /// The defect this module exists to prevent, stated as a test: the pair on
    /// the wire must never read "6 of 4".
    #[test]
    fn a_torn_read_cannot_publish_more_rows_than_exist() {
        let telemetry = BatchTelemetry::default();
        telemetry.publish(4, 2, 4);
        // Simulate the interleaving a relaxed reader can observe: capacity
        // updated to a narrower batch before the matching active store.
        telemetry.capacity.store(2, Ordering::Relaxed);

        let seen = telemetry.snapshot().expect("published");
        assert!(
            seen.active <= seen.capacity,
            "wire published {} of {}, which renders as an over-capacity \
             fraction",
            seen.active,
            seen.capacity
        );
    }

    /// Queued is reported alongside rather than folded into `active`, because
    /// a request waiting for a row is a different fact from one being decoded
    /// -- and it is the fact that makes queueing visible in the demo.
    #[test]
    fn queued_requests_are_reported_beside_active_rows_not_inside_them() {
        let telemetry = BatchTelemetry::default();
        telemetry.publish(4, 2, 4);
        let seen = telemetry.snapshot().expect("published");
        assert_eq!((seen.active, seen.queued, seen.capacity), (4, 2, 4));
    }
}
