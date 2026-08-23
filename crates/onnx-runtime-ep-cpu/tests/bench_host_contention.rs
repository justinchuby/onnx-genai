//! Tests for `benches/common/host_contention.rs`.
//!
//! They live here rather than in a `#[cfg(test)]` module beside the code for a
//! specific reason. Every bench target in this crate is declared
//! `harness = false`, so `cargo test --bench <name>` *runs the benchmark* and
//! never runs a test module inside it, and when such a module is compiled as
//! part of a `harness = false` bench its `#[test]` functions are stripped. Tests
//! written next to that module are therefore compiled-and-discarded: they can
//! only ever pass, because they never run. That is exactly the failure the
//! module itself exists to prevent one level up -- a number that was never
//! measured -- so it would be a poor place to keep its own tests.

#[path = "../benches/common/host_contention.rs"]
mod host_contention;

use host_contention::{AllowedCpus, ContentionSnapshot, clock_tick_hz, contention, snapshot};
use std::time::{Duration, Instant};

/// One core-second's worth of jiffies.
fn jiffies(core_seconds: f64) -> u64 {
    (clock_tick_hz() * core_seconds) as u64
}

/// An unmeasurable window must report `measured=false`, never a clean zero. A
/// zero would read as "we checked and the set was quiet".
#[test]
fn a_missing_snapshot_is_reported_as_unmeasured_rather_than_clean() {
    let c = contention(None, None);
    assert!(!c.measured);
    assert!(!c.is_contended());
    assert_eq!(c.foreign_pct, 0.0);
}

/// The arithmetic, on synthetic snapshots so it does not depend on whatever the
/// host happened to be doing while the suite ran.
#[test]
fn foreign_cpu_is_total_busy_on_the_set_minus_our_own() {
    let taken = Instant::now();
    let before = ContentionSnapshot::from_parts(taken, vec![0, 1], vec![0, 0], 0);
    // One second of wall, during which the two-CPU set accrued 1.5 core-seconds
    // of busy while we accrued 1.0 -- so half a core was somebody else.
    let after = ContentionSnapshot::from_parts(
        taken + Duration::from_secs(1),
        vec![0, 1],
        vec![jiffies(1.0), jiffies(0.5)],
        jiffies(1.0),
    );
    let c = contention(Some(&before), Some(&after));
    assert!(c.measured);
    assert!(
        (c.foreign_pct - 50.0).abs() < 1.0,
        "expected ~50% of one core foreign, got {}",
        c.foreign_pct
    );
    assert!(
        (c.total_pct - 150.0).abs() < 1.0,
        "expected ~150% of one core total, got {}",
        c.total_pct
    );
    assert!(c.is_contended());
}

/// A window we had entirely to ourselves must not be flagged, or the column
/// cries wolf on every clean run and stops being read.
#[test]
fn a_window_we_had_entirely_to_ourselves_is_not_flagged() {
    let taken = Instant::now();
    let before = ContentionSnapshot::from_parts(taken, vec![0, 1], vec![0, 0], 0);
    let after = ContentionSnapshot::from_parts(
        taken + Duration::from_secs(1),
        vec![0, 1],
        vec![jiffies(1.0), jiffies(1.0)],
        jiffies(2.0),
    );
    let c = contention(Some(&before), Some(&after));
    assert!(c.measured);
    assert!(!c.is_contended(), "foreign_pct was {}", c.foreign_pct);
}

/// Coarse jiffy accounting across two non-simultaneous reads can make an idle
/// set difference to a small negative. That must clamp rather than surface as a
/// negative contention, which would read as a bug in the workload.
#[test]
fn sampling_skew_clamps_instead_of_reporting_negative_contention() {
    let taken = Instant::now();
    let before = ContentionSnapshot::from_parts(taken, vec![0], vec![0], 0);
    let after = ContentionSnapshot::from_parts(
        taken + Duration::from_secs(1),
        vec![0],
        vec![jiffies(0.9)],
        jiffies(1.0),
    );
    let c = contention(Some(&before), Some(&after));
    assert!(c.measured);
    assert_eq!(c.foreign_pct, 0.0);
}

/// A mask that moved under the window must not be differenced -- the two ends
/// would be taken over different core sets.
///
/// This is a live case, not a hypothetical one, and it produced a real false
/// positive: the EP narrows the process's affinity when it builds the decode
/// pool, so a mask read before the first decode is the whole machine. Scoped to
/// that stale mask, a spinner pinned to a CPU this process can never run on was
/// reported as 39.7% contention. Re-reading the mask inside each snapshot fixed
/// it; this test pins the invariant that makes such a mismatch detectable.
#[test]
fn a_mask_that_moved_under_the_window_is_reported_as_unmeasured() {
    let taken = Instant::now();
    let before = ContentionSnapshot::from_parts(taken, vec![0, 1, 2, 3], vec![0, 0, 0, 0], 0);
    let after = ContentionSnapshot::from_parts(
        taken + Duration::from_secs(1),
        vec![0, 2],
        vec![jiffies(1.0), jiffies(1.0)],
        0,
    );
    let c = contention(Some(&before), Some(&after));
    assert!(
        !c.measured,
        "differencing across a changed mask compares different core sets"
    );
}

/// The real mask must be readable on the host running this suite, since every
/// other reading is scoped to it.
#[test]
#[cfg(target_os = "linux")]
fn the_allowed_cpu_set_is_readable_and_non_empty() {
    let allowed = AllowedCpus::current().expect("sched_getaffinity must work on Linux");
    assert!(!allowed.is_empty());
    assert!(!allowed.label().is_empty());
    assert!(snapshot().is_some(), "/proc/stat must be readable");
}

/// End to end against the live host: two real snapshots around a window in which
/// this process deliberately burns a core must attribute that core to *us*.
///
/// Deliberately asserts only the direction a busy host cannot break. A co-tenant
/// can push `foreign_pct` up at any moment, so an upper bound on it would be
/// flaky; our own core-second, though, must always appear in `total_pct`.
#[test]
#[cfg(target_os = "linux")]
fn our_own_cpu_burn_is_counted_in_the_total_for_the_set() {
    let before = snapshot().expect("snapshot");
    let deadline = Instant::now() + Duration::from_millis(600);
    let mut sink = 0u64;
    while Instant::now() < deadline {
        sink = sink.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    std::hint::black_box(sink);
    let after = snapshot().expect("snapshot");

    let c = contention(Some(&before), Some(&after));
    assert!(c.measured);
    assert!(
        c.total_pct > 50.0,
        "we burned ~one core for the window, so the total must see it; got {}",
        c.total_pct
    );
}
