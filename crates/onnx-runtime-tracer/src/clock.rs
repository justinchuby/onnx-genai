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

/// Microseconds on the operating system's own monotonic clock.
///
/// The origin is whatever the OS counts from -- boot, on both platforms here.
/// That is the point: it is the *same* origin for every library, thread and
/// process on the machine, so two readings are comparable because they are
/// readings of one clock, with nothing sampled or agreed between them.
#[cfg(unix)]
#[allow(unsafe_code)]
fn monotonic_now_us() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes a `timespec` through the pointer given and
    // touches nothing else; `ts` is a live local of exactly that type. The call
    // borrows nothing and can fail only by returning non-zero, which is checked.
    let read = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    if read != 0 {
        return 0;
    }
    (ts.tv_sec as u64)
        .saturating_mul(1_000_000)
        .saturating_add((ts.tv_nsec as u64) / 1_000)
}

/// Microseconds on the operating system's own monotonic clock (Windows).
#[cfg(windows)]
#[allow(unsafe_code)]
fn monotonic_now_us() -> u64 {
    use windows_sys::Win32::System::Performance::{
        QueryPerformanceCounter, QueryPerformanceFrequency,
    };
    let mut frequency = 0i64;
    let mut counter = 0i64;
    // SAFETY: both calls write a single `i64` through the pointer given and
    // touch nothing else; both locals are live and of exactly that type. Each
    // returns zero on failure, which is checked before the value is used.
    let ok = unsafe {
        QueryPerformanceFrequency(&mut frequency) != 0 && QueryPerformanceCounter(&mut counter) != 0
    };
    if !ok || frequency <= 0 {
        return 0;
    }
    ((i128::from(counter) * 1_000_000) / i128::from(frequency)) as u64
}

/// Where this library first saw the monotonic and wall clocks agree.
///
/// Sampled once, and used only to present timestamps as dates.
fn unix_anchor() -> &'static (u64, u64) {
    static ANCHOR: OnceLock<(u64, u64)> = OnceLock::new();
    ANCHOR.get_or_init(|| {
        let monotonic = monotonic_now_us();
        let unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_micros() as u64)
            .unwrap_or(0);
        (monotonic, unix)
    })
}

/// Present a trace timestamp as wall-clock UNIX microseconds.
///
/// For display only. The mapping is sampled once per library, so two libraries
/// can disagree by that sampling gap and by any wall-clock adjustment between
/// their first reads. Alignment deliberately does not depend on it.
#[must_use]
pub fn monotonic_to_unix_us(trace_us: u64) -> u64 {
    let (anchor_monotonic, anchor_unix) = *unix_anchor();
    // Signed, because the anchor is established on first use and a trace is
    // normally converted *after* it was captured -- so most timestamps are
    // behind the anchor, not ahead of it. Subtracting with saturation instead
    // clamped every one of those to the anchor itself, collapsing a 55ms span
    // to 28us.
    let offset = i128::from(trace_us) - i128::from(anchor_monotonic);
    let unix = i128::from(anchor_unix) + offset;
    u64::try_from(unix.max(0)).unwrap_or(u64::MAX)
}

/// The current time on the shared trace axis, in microseconds.
///
/// The one time base every layer stamps against, and the reason a trace from
/// the host and one from a plugin execution provider can be read together with
/// no offset to negotiate. It reads the operating system's monotonic clock, so
/// the origin belongs to the machine rather than to whoever initialised first.
///
/// Both alternatives were tried and are worse:
///
/// * An [`Instant`] epoch is monotonic but its origin is private to the caller.
///   A plugin dylib links its own copy of this crate with its own statics, so
///   even a process-global epoch is not shared with it.
/// * Anchoring each copy to UNIX time by sampling both clocks makes them agree
///   only while the wall clock behaves. Step it between two libraries' first
///   reads -- one NTP correction -- and they disagree by that step for the life
///   of the process. That was the previous implementation.
///
/// Handing an offset across the plugin ABI fixes neither, because a provider
/// loaded by ONNX Runtime is reached through no interface of ours at all.
///
/// Timestamps are microseconds since boot, not since the UNIX epoch; use
/// [`monotonic_to_unix_us`] to show them as dates.
#[must_use]
pub fn absolute_now_us() -> u64 {
    monotonic_now_us()
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

#[cfg(test)]
mod shared_axis_tests {
    use super::*;

    /// The axis must advance, and must do so at wall-clock rate.
    #[test]
    fn the_trace_axis_advances_in_real_microseconds() {
        let before = absolute_now_us();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let after = absolute_now_us();
        let elapsed = after.saturating_sub(before);
        assert!(
            (15_000..500_000).contains(&elapsed),
            "20ms of sleep measured as {elapsed}us on the trace axis"
        );
    }

    /// Readings must be comparable no matter who takes them.
    ///
    /// This is the property the whole design exists for: a plugin execution
    /// provider links its own copy of this crate with its own statics, so any
    /// per-copy epoch would put its spans somewhere else entirely. Threads are
    /// the closest a unit test can get to that; the real check is that nothing
    /// here is per-copy state.
    #[test]
    fn readings_from_different_threads_share_one_origin() {
        let start = absolute_now_us();
        let from_thread = std::thread::spawn(absolute_now_us)
            .join()
            .expect("the reading thread must not panic");
        let end = absolute_now_us();
        assert!(
            start <= from_thread && from_thread <= end,
            "a reading from another thread ({from_thread}) fell outside the \
             interval it was taken in ({start}..{end}), so the two are not on \
             one axis"
        );
    }

    /// Converting must preserve intervals on both sides of the anchor.
    ///
    /// The anchor is established on first use, so a trace exported after
    /// capture has *every* timestamp behind it. Saturating subtraction sent
    /// all of those to the anchor itself, turning a 55ms span into 28us --
    /// a timeline where nothing appeared to take any time.
    #[test]
    fn converting_preserves_intervals_before_and_after_the_anchor() {
        let early = absolute_now_us();
        std::thread::sleep(std::time::Duration::from_millis(20));
        // First conversion, which is what establishes the anchor.
        let mapped_early = monotonic_to_unix_us(early);
        let late = absolute_now_us();
        let mapped_late = monotonic_to_unix_us(late);

        let raw_gap = late - early;
        let mapped_gap = mapped_late - mapped_early;
        assert!(
            mapped_gap.abs_diff(raw_gap) <= 2,
            "a {raw_gap}us interval converted to {mapped_gap}us; conversion \
             must not compress time"
        );
        assert!(
            mapped_early < mapped_late,
            "conversion reordered two timestamps"
        );
    }

    /// The wall-clock mapping is for display, and must round-trip its anchor.
    #[test]
    fn the_unix_mapping_lands_near_the_present() {
        let now = absolute_now_us();
        let as_unix = monotonic_to_unix_us(now);
        let actual = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the system clock must be after 1970")
            .as_micros() as u64;
        let skew = as_unix.abs_diff(actual);
        assert!(
            skew < 5_000_000,
            "the trace axis mapped to {as_unix}, {skew}us away from the wall \
             clock's {actual}"
        );
    }
}
