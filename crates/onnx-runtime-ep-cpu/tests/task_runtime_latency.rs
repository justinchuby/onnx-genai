//! A measurement harness for the CPU task runtime's dispatch latency.
//!
//! This is the experiment that motivated the runtime. Rayon parks its workers
//! aggressively, so a fan-out issued immediately after another one is cheap
//! while a fan-out issued after even a short idle gap pays a full wake. Decode
//! is exactly the second shape: a handful of microseconds of serial work
//! between every parallel region.
//!
//! The tests here are `#[ignore]`d because they are measurements, not
//! assertions -- they report numbers to stdout and are meant to be run
//! deliberately on a quiet machine:
//!
//! ```text
//! cargo test -p onnx-runtime-ep-cpu --release --test task_runtime_latency \
//!     -- --ignored --nocapture
//! ```
//!
//! The one thing they do assert is the property the runtime exists to
//! guarantee: a *decode-shaped* gap must not cost dramatically more than a
//! back-to-back dispatch. Gaps longer than the spin window are supposed to
//! park -- that is the whole point of the window closing -- so those are
//! reported rather than asserted.
//!
//! Both live in one test function because they measure the same shared pool
//! and would otherwise run concurrently and interleave.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use onnx_runtime_ep_cpu::task_runtime;

/// Gaps to probe, in microseconds. Zero means back-to-back.
const GAPS_US: [u64; 6] = [0, 5, 20, 100, 500, 2000];

/// The longest gap that still counts as "inside a decode step".
///
/// The serial stretch between two parallel regions in decode is single-digit
/// to low-tens of microseconds. Past this the runtime is *supposed* to park.
const DECODE_GAP_US: u64 = 100;

/// How many fan-outs each sample averages over.
const ROUNDS: usize = 400;

/// Spins for roughly `micros`, without parking, the way a serial stretch of
/// decode between two parallel regions would.
fn busy_gap(micros: u64) {
    if micros == 0 {
        return;
    }
    let until = Instant::now() + Duration::from_micros(micros);
    while Instant::now() < until {
        std::hint::spin_loop();
    }
}

/// Times a minimal fan-out: enough tasks to reach every worker, almost no work
/// in each, so what is measured is wake-up and hand-back, not the body.
fn timed_fanout(scratch: &mut [u64]) -> Duration {
    let started = Instant::now();
    task_runtime::chunk_runs_mut(scratch, 1, 1, |_, run| {
        for slot in run {
            *slot = slot.wrapping_add(1);
        }
    });
    started.elapsed()
}

fn percentile(sorted: &[Duration], fraction: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

/// Measures dispatch latency at one gap, returning (p50, p90) over [`ROUNDS`].
fn sample_at_gap(gap_us: u64, width: usize) -> (Duration, Duration) {
    let mut scratch = vec![0u64; width.max(2) * 4];
    // Warm up so thread creation and first-touch are not in the sample.
    for _ in 0..64 {
        timed_fanout(&mut scratch);
    }
    let mut samples = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        busy_gap(gap_us);
        samples.push(timed_fanout(&mut scratch));
    }
    samples.sort_unstable();
    (percentile(&samples, 0.5), percentile(&samples, 0.9))
}

#[test]
#[ignore = "measurement harness; run deliberately on a quiet machine"]
fn report_dispatch_latency() {
    report_dispatch_latency_against_idle_gap();
    report_latency_under_concurrent_sessions();
}

fn report_dispatch_latency_against_idle_gap() {
    let width = task_runtime::testing::pool_width();
    println!("\ndispatch latency vs idle gap ({width} workers, {ROUNDS} rounds each)");
    println!("{:>10}  {:>12}  {:>12}", "gap", "p50", "p90");

    let mut results = Vec::new();
    for gap_us in GAPS_US {
        let (p50, p90) = sample_at_gap(gap_us, width);
        println!(
            "{gap_us:>8}us  {:>10.1}us  {:>10.1}us",
            as_us(p50),
            as_us(p90)
        );
        results.push((gap_us, p50));
    }

    if width < 2 {
        println!("(single worker: nothing to wake, so the gap cannot matter)");
        return;
    }

    let back_to_back = results[0].1;
    let (worst_gap, worst) = results
        .iter()
        .filter(|(gap, _)| *gap > 0 && *gap <= DECODE_GAP_US)
        .max_by_key(|(_, p50)| *p50)
        .copied()
        .expect("at least one decode-shaped gapped sample");

    // The point of an adaptive spin window: a decode-shaped gap should not
    // turn a dispatch into a futex round trip. Three times back-to-back is a
    // loose bound -- the Rayon baseline this replaced was 3.4x at a 20us gap
    // and worse beyond -- but it is tight enough to catch a pool that parks
    // immediately.
    let budget = back_to_back * 3 + Duration::from_micros(10);
    assert!(
        worst <= budget,
        "a {worst_gap}us gap cost {:.1}us at p50 against {:.1}us back-to-back \
         (budget {:.1}us): the pool is parking instead of spinning",
        as_us(worst),
        as_us(back_to_back),
        as_us(budget),
    );
}

fn as_us(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

fn report_latency_under_concurrent_sessions() {
    let width = task_runtime::testing::pool_width();
    println!("\ndispatch latency vs concurrent sessions ({width} workers)");
    println!("{:>10}  {:>12}  {:>12}", "sessions", "p50", "p90");

    for sessions in [1usize, 2, 4, 8] {
        let barrier = Arc::new(Barrier::new(sessions));
        let total_p50 = Arc::new(AtomicU64::new(0));
        let total_p90 = Arc::new(AtomicU64::new(0));
        std::thread::scope(|scope| {
            for _ in 0..sessions {
                let barrier = Arc::clone(&barrier);
                let total_p50 = Arc::clone(&total_p50);
                let total_p90 = Arc::clone(&total_p90);
                scope.spawn(move || {
                    let mut scratch = vec![0u64; width.max(2) * 4];
                    for _ in 0..64 {
                        timed_fanout(&mut scratch);
                    }
                    barrier.wait();
                    let mut samples = Vec::with_capacity(ROUNDS);
                    for _ in 0..ROUNDS {
                        busy_gap(20);
                        samples.push(timed_fanout(&mut scratch));
                    }
                    samples.sort_unstable();
                    total_p50.fetch_add(
                        percentile(&samples, 0.5).as_nanos() as u64,
                        Ordering::Relaxed,
                    );
                    total_p90.fetch_add(
                        percentile(&samples, 0.9).as_nanos() as u64,
                        Ordering::Relaxed,
                    );
                });
            }
        });
        let mean_p50 = Duration::from_nanos(total_p50.load(Ordering::Relaxed) / sessions as u64);
        let mean_p90 = Duration::from_nanos(total_p90.load(Ordering::Relaxed) / sessions as u64);
        println!(
            "{sessions:>10}  {:>10.1}us  {:>10.1}us",
            as_us(mean_p50),
            as_us(mean_p90)
        );
    }

    let counters = task_runtime::testing::counters();
    println!(
        "slot exhaustion: {} of {} dispatches fell back to serial",
        counters.slot_exhausted, counters.dispatches
    );
}
