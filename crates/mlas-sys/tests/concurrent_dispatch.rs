//! Regression guard for overlapping `parallel_for` dispatches (#1685).
//!
//! # The defect this pins
//!
//! [`WorkStealingThreadPool`] publishes one job at a time: the dispatcher fills
//! the shared `LoopCounter` shards, bumps the epoch, joins in as worker 0, and
//! then clears the job. `wait_for_completion` returns as soon as `remaining`
//! hits zero — that is, as soon as the last *block* has been executed.
//!
//! But a worker is not finished when the last block is: it is still inside
//! `run_job`, holding a by-value copy of the old `Job` and looping in
//! `claim_iterations` against the shared shards. Without a second wait, the
//! dispatcher returned and released the dispatch lock while that worker was
//! still in the loop. The next dispatch — from another thread — then
//! republished the shards, and the straggler claimed a block belonging to the
//! *new* job while still holding the *old* job's closure pointer and bounds.
//!
//! Two things went wrong at once, and both were observed in #1685:
//!
//! 1. The straggler decremented the **new** job's `remaining` without running
//!    the new job's closure over that block, so `wait_for_completion` returned
//!    early and the dispatch finished with a partition **never executed**. In
//!    MLAS that is a `beta = 0` SGEMM leaving part of `C` unwritten — the
//!    reported eight-row hole of exact `0.0` inside one attention tile.
//! 2. The straggler invoked the **old** closure with an index from the new
//!    range. For MLAS that is `task(stale_ctx, tid)` with a partition index
//!    past the end of the previous GEMM's output, which writes `C` through raw
//!    pointers — the reported SIGSEGV.
//!
//! `wait_for_workers` closes both: the dispatcher now waits until every worker
//! has left `run_job` and acknowledged the epoch before clearing the job and
//! releasing the lock.
//!
//! # Why this test is shaped the way it is
//!
//! The defect is invisible to a single-threaded sweep, however long: it needs
//! two dispatches to *overlap*. The original reproducer was a whole test binary
//! failing about 3% of the time, which is far too coarse to bisect. Driving
//! several dispatcher threads at one pool reproduces it in well under a second,
//! and the assertion is on lost work rather than on values, so it cannot be
//! satisfied by a coincidence.
//!
//! Verified live: with the `wait_for_workers` call removed, this test fails
//! immediately — either on unwritten slots or on the stale closure panicking
//! with an out-of-range slice index, which is the safe-Rust shadow of the
//! SIGSEGV MLAS suffered.

use mlas_sys::WorkStealingThreadPool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Overlapping dispatches must never return with work unexecuted.
///
/// Each round writes a tag unique to `(dispatcher, round)` into a fresh buffer.
/// A slot still holding its initial `0` means `parallel_for` returned before
/// that block ran — the exact failure MLAS reported as an unwritten `C`.
#[test]
fn overlapping_dispatches_never_return_with_work_unexecuted() {
    // Four dispatchers against an eight-thread pool: enough contention that
    // dispatches interleave, small enough to stay well inside CI's budget.
    let pool = Arc::new(WorkStealingThreadPool::new(8).expect("pool must start"));
    let holes = Arc::new(AtomicUsize::new(0));
    let unwritten = Arc::new(AtomicUsize::new(0));
    let dispatchers = 4usize;
    let rounds = 1500usize;
    // Comfortably more iterations than blocks-per-worker, so the loop counters
    // are claimed dynamically rather than in one inline chunk.
    let len = 64usize;

    std::thread::scope(|scope| {
        for dispatcher in 0..dispatchers {
            let pool = Arc::clone(&pool);
            let holes = Arc::clone(&holes);
            let unwritten = Arc::clone(&unwritten);
            scope.spawn(move || {
                for round in 0..rounds {
                    let tag = dispatcher * rounds + round + 1;
                    let slots: Vec<AtomicUsize> = (0..len).map(|_| AtomicUsize::new(0)).collect();
                    pool.parallel_for(0, len, 1, |begin, end| {
                        for slot in &slots[begin..end] {
                            slot.store(tag, Ordering::Relaxed);
                        }
                    });
                    let missed = slots
                        .iter()
                        .filter(|slot| slot.load(Ordering::Relaxed) != tag)
                        .count();
                    if missed > 0 {
                        holes.fetch_add(1, Ordering::Relaxed);
                        unwritten.fetch_add(missed, Ordering::Relaxed);
                    }
                }
            });
        }
    });

    let dispatches = dispatchers * rounds;
    assert_eq!(
        holes.load(Ordering::Relaxed),
        0,
        "{} of {dispatches} dispatches returned with work unexecuted ({} slots total); \
         a straggler from the previous epoch consumed this job's loop counters",
        holes.load(Ordering::Relaxed),
        unwritten.load(Ordering::Relaxed),
    );
}

/// The pool's contract is that `parallel_for` does not return until every
/// worker has left the closure. That is not a nicety: it is what makes the
/// stack-borrowing API sound, because the dispatcher clears the shared `Job`
/// (and drops the borrow the closure captured) the instant it returns.
///
/// Assert it directly — no worker may still be inside the body once the
/// dispatch has returned.
///
/// # Why the watchdog
///
/// Under the pre-fix pool this property does not merely fail, it *deadlocks*:
/// a straggler decrements the next job's `remaining` past zero, the counter
/// wraps, and `wait_for_completion` spins forever. A regression test that
/// hangs is worse than one that fails, so the work runs on its own thread and
/// the assertion is on finishing at all. Verified against the falsifier: with
/// `wait_for_workers` removed this reports the deadlock instead of hanging CI.
#[test]
fn parallel_for_returns_only_after_every_worker_has_left_the_closure() {
    let (done, finished) = std::sync::mpsc::channel::<Result<usize, String>>();
    std::thread::Builder::new()
        .name("ws-closure-contract".into())
        .spawn(move || {
            let pool = WorkStealingThreadPool::new(8).expect("pool must start");
            let inside = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let mut verdict = Ok(0);
            for _ in 0..2000 {
                pool.parallel_for(0, 64, 1, |begin, end| {
                    let now = inside.fetch_add(1, Ordering::AcqRel) + 1;
                    peak.fetch_max(now, Ordering::AcqRel);
                    std::hint::black_box(end - begin);
                    inside.fetch_sub(1, Ordering::AcqRel);
                });
                let still_running = inside.load(Ordering::Acquire);
                if still_running != 0 {
                    verdict = Err(format!(
                        "parallel_for returned while {still_running} worker(s) were still \
                         inside the closure; the borrow it hands out would outlive the dispatch"
                    ));
                    break;
                }
            }
            let _ = done.send(verdict.map(|_| peak.load(Ordering::Acquire)));
        })
        .expect("watchdog thread must start");

    match finished.recv_timeout(std::time::Duration::from_secs(120)) {
        Ok(Ok(peak)) => assert!(
            peak > 1,
            "no two workers were ever in the closure at once, so this run never \
             exercised the concurrency it claims to guard"
        ),
        Ok(Err(message)) => panic!("{message}"),
        Err(_) => panic!(
            "the dispatch loop never finished: the pool deadlocked, which is what \
             happens when a straggler from a retired epoch decrements the next job's \
             completion counter past zero"
        ),
    }
}
