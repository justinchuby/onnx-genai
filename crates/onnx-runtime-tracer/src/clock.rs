//! The shared monotonic [`TraceClock`] and [`TraceSessionId`] (§48.3).
//!
//! Both the runtime layer and the genai layer share a single [`TraceClock`] so
//! that events recorded on different threads (or in different layers) land on
//! one timeline. The clock is fixed at construction: it captures a monotonic
//! [`Instant`] epoch and reports microseconds elapsed from it. A
//! [`TraceSessionId`] tags every context so traces from separate runs stay
//! distinguishable when merged.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// A monotonic clock anchored at a fixed epoch, shared across a trace.
///
/// Reused (renamed) from the Phase-1 tracer epoch. Clone-share it through an
/// [`Arc`](std::sync::Arc) so every layer stamps timestamps against the same
/// origin. All timestamps are **microseconds** since the epoch.
#[derive(Debug)]
pub struct TraceClock {
    epoch: Instant,
}

impl TraceClock {
    /// Create a clock whose epoch is the moment of this call.
    #[must_use]
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }

    /// The epoch [`Instant`] this clock measures against.
    #[must_use]
    pub fn epoch(&self) -> Instant {
        self.epoch
    }

    /// Microseconds elapsed from the epoch to *now*.
    #[must_use]
    pub fn now_micros(&self) -> u64 {
        self.micros_at(Instant::now())
    }

    /// Microseconds elapsed from the epoch to `at`.
    ///
    /// Saturates at zero for the (impossible for a monotonic clock, but
    /// defensive) case of an instant before the epoch.
    #[must_use]
    pub fn micros_at(&self, at: Instant) -> u64 {
        at.saturating_duration_since(self.epoch).as_micros() as u64
    }
}

impl Default for TraceClock {
    fn default() -> Self {
        Self::new()
    }
}

/// A process-unique identifier for one tracing session (§48.3).
///
/// Ids are drawn from a monotonic per-process counter, so distinct
/// [`TraceContext`](crate::TraceContext)s never collide within a process. The
/// value is opaque; only its identity and [`Display`](std::fmt::Display) matter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TraceSessionId(u64);

impl TraceSessionId {
    /// Allocate the next session id for this process.
    #[must_use]
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Construct a session id from an explicit value (e.g. to correlate with an
    /// external trace).
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// The raw numeric value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for TraceSessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
