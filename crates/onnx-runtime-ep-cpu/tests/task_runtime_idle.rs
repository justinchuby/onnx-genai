//! Resource bounds for the CPU task runtime's native pool.
//!
//! A spinning pool is a liability if it spins forever: it burns a core per
//! worker in a process that is between tokens, between requests, or simply
//! done. These tests pin down the two costs that a user would notice --
//! background CPU and resident memory -- rather than the throughput the rest of
//! the suite covers.
//!
//! They live in their own integration test binary, and in a *single* test
//! function, on purpose. Both measure a whole-process quantity, so anything
//! else running fan-outs in the same process would make them meaningless --
//! including each other. As two `#[test]`s they ran concurrently and the
//! RSS test's four thousand fan-outs showed up as 1.31s of "idle" CPU.

use std::time::{Duration, Instant};

use onnx_runtime_ep_cpu::task_runtime;

/// How long the pool is left alone before we measure its idle cost.
///
/// Comfortably longer than the pool's maximum spin window (500us), so a
/// correctly parked pool contributes nothing to the sample.
const IDLE_WINDOW: Duration = Duration::from_millis(750);

/// Reads this process's CPU time, user + system, across all threads.
#[cfg(target_os = "linux")]
fn process_cpu_time() -> Option<Duration> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The comm field can contain spaces and parentheses, so fields are counted
    // from the last ')' rather than from the start of the line.
    let after_comm = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    // utime and stime are fields 14 and 15 of the full line; the split above
    // dropped pid and comm and left a leading state field, so they are at 11
    // and 12 here.
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    let ticks_per_second = 100u64;
    let ticks = utime + stime;
    Some(Duration::from_nanos(
        ticks * (1_000_000_000 / ticks_per_second),
    ))
}

#[cfg(not(target_os = "linux"))]
fn process_cpu_time() -> Option<Duration> {
    None
}

/// Reads this process's resident set size in bytes.
#[cfg(target_os = "linux")]
fn resident_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(resident_pages * 4096)
}

#[cfg(not(target_os = "linux"))]
fn resident_bytes() -> Option<u64> {
    None
}

/// Runs a fan-out big enough to wake every worker, so the pool is warm.
fn wake_the_pool() {
    let width = task_runtime::testing::pool_width();
    let mut data = vec![0u64; 1 << 20];
    task_runtime::chunk_runs_mut(&mut data, 1 << 12, 1, |_, run| {
        for slot in run {
            *slot = slot.wrapping_add(1);
        }
    });
    std::hint::black_box(&data);
    assert!(width >= 1);
}

#[test]
fn an_idle_pool_is_cheap_and_a_busy_one_does_not_grow() {
    idle_costs_no_cpu();
    a_warm_pool_does_not_grow_its_footprint();
}

/// Phase one: the pool must give the cores back when the work stops.
fn idle_costs_no_cpu() {
    let Some(before_wake) = process_cpu_time() else {
        eprintln!("skipping: no per-process CPU accounting on this platform");
        return;
    };

    // Warm the pool first. Thread creation and the first dispatch legitimately
    // cost CPU; what we care about is what happens once the work has stopped.
    for _ in 0..8 {
        wake_the_pool();
    }
    let after_wake = process_cpu_time().expect("CPU accounting vanished mid-test");
    let working = after_wake.saturating_sub(before_wake);

    let started = Instant::now();
    std::thread::sleep(IDLE_WINDOW);
    let elapsed = started.elapsed();
    let idle = process_cpu_time()
        .expect("CPU accounting vanished mid-test")
        .saturating_sub(after_wake);

    let width = task_runtime::testing::pool_width();
    // If every worker spun for its whole maximum window once and then parked,
    // that is 500us each. Ten times that is still nothing next to the wall
    // clock, and is far below what a spinning pool would burn.
    let budget = Duration::from_micros(500) * 10 * width as u32;
    assert!(
        idle <= budget,
        "an idle pool of {width} workers burned {idle:?} of CPU over {elapsed:?} \
         (budget {budget:?}); it did {working:?} of actual work beforehand"
    );
}

/// Phase two: the dispatch path must not allocate its way into the heap.
fn a_warm_pool_does_not_grow_its_footprint() {
    let Some(before) = resident_bytes() else {
        eprintln!("skipping: no resident-set accounting on this platform");
        return;
    };

    // Spawn the threads and touch their stacks.
    wake_the_pool();
    let warm = resident_bytes().expect("RSS accounting vanished mid-test");

    // Then hammer it. A dispatch path that allocated -- a closure box, a task
    // vector, a channel message -- would show up here as steady growth.
    for _ in 0..4_000 {
        let mut data = vec![0u64; 1 << 14];
        task_runtime::chunk_runs_mut(&mut data, 1 << 8, 1, |_, run| {
            for slot in run {
                *slot = slot.wrapping_add(1);
            }
        });
        std::hint::black_box(&data);
    }
    let after = resident_bytes().expect("RSS accounting vanished mid-test");

    let width = task_runtime::testing::pool_width();
    let growth = after.saturating_sub(warm);
    // 1 MiB of slack per worker covers page-in of stacks the first dispatch
    // did not touch and allocator arenas per thread. Anything genuinely
    // leaking per dispatch would clear this many times over in 4000 rounds.
    let budget = 1024 * 1024 * width as u64;
    assert!(
        growth <= budget,
        "4000 fan-outs grew RSS by {growth} bytes with {width} workers \
         (budget {budget}); RSS was {before} at start, {warm} once warm, {after} at end"
    );
}
