//! The shared monotonic [`TraceClock`] and [`TraceSessionId`] (§48.3).
//!
//! Both the runtime layer and the genai layer share a single [`TraceClock`] so
//! that events recorded on different threads (or in different layers) land on
//! one timeline. The clock is fixed at construction: it captures a monotonic
//! [`Instant`] epoch and reports microseconds elapsed from it. A
//! [`TraceSessionId`] tags every context so traces from separate runs stay
//! distinguishable when merged.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// This library's fixed tie between the monotonic clock and absolute UNIX time.
///
/// Sampled once, on first use. Every later reading adds monotonic elapsed time
/// to it, so readings are monotonic *and* absolute.
fn origin() -> &'static (Instant, u64) {
    static ORIGIN: OnceLock<(Instant, u64)> = OnceLock::new();
    ORIGIN.get_or_init(|| {
        let instant = Instant::now();
        let unix_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_micros() as u64)
            .unwrap_or(0);
        (instant, unix_us)
    })
}

/// The current time in **absolute UNIX microseconds**, on a monotonic basis.
///
/// This is the one time base every layer stamps against, and the reason a trace
/// from the host and a trace from a plugin execution provider can be read on a
/// single timeline without negotiating an offset.
///
/// A plain [`Instant`] epoch cannot do that job. `Instant` is monotonic but its
/// origin is arbitrary and private to whoever called [`Instant::now`], and a
/// plugin loaded as a dynamic library links its own copy of this crate with its
/// own statics — so a process-global epoch would still not be shared. Anchoring
/// to UNIX time gives every copy the same origin.
///
/// Elapsed time still comes from the monotonic clock, so durations are immune
/// to wall-clock adjustments; only the origin is absolute. The two clocks are
/// read one after the other when [`origin`] initialises, so independently
/// initialised copies can disagree by that sampling gap — well under a
/// microsecond, against spans measured in microseconds and up.
#[must_use]
pub fn absolute_now_us() -> u64 {
    let (instant, unix_us) = origin();
    unix_us.saturating_add(instant.elapsed().as_micros() as u64)
}

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

    /// The current time on the shared absolute axis, in microseconds.
    ///
    /// Reports [`absolute_now_us`] rather than time since this clock's own
    /// construction, so two clocks built at different moments — in the host and
    /// in a plugin execution provider — still place the same instant at the
    /// same number.
    #[must_use]
    pub fn now_micros(&self) -> u64 {
        absolute_now_us()
    }

    /// The time of `at` on the shared absolute axis, in microseconds.
    ///
    /// `at` must come from this process's monotonic clock; it is converted by
    /// how long ago it was, so it lands on the same axis as [`now_micros`].
    #[must_use]
    pub fn micros_at(&self, at: Instant) -> u64 {
        let now = Instant::now();
        let ago = now.saturating_duration_since(at).as_micros() as u64;
        absolute_now_us().saturating_sub(ago)
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

/// The process id every sink stamps on its events.
///
/// A Chrome trace groups lanes by `pid` first, so two sinks that disagree here
/// render as two unrelated processes and one cannot be shown nested inside the
/// other — however well their timestamps line up. This is the single answer,
/// so an engine span and an operator span land in the same process.
#[must_use]
pub fn process_id() -> u64 {
    u64::from(std::process::id())
}

/// A small, stable lane number for the calling OS thread.
///
/// Shared for the same reason as [`process_id`]: two sinks numbering threads
/// independently both start at 0, so unrelated threads collide on one lane and
/// the same thread appears as two.
#[must_use]
pub fn thread_lane_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        static LANE: u64 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    LANE.with(|lane| *lane)
}
