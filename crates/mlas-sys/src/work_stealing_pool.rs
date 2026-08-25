//! Persistent, low-overhead work-stealing pool for decode-sized parallel-for.
//!
//! This is the Rust-side prototype of the ORT/Eigen design: workers are created
//! once, stay in a spin-then-park loop between dispatches, and each parallel-for
//! publishes ORT-style `LoopCounter` shards that workers dynamically claim from.
//! The API accepts stack-borrowing closures because `parallel_for` does not
//! return until every worker has left the closure.

use std::cell::UnsafeCell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

type JobFn = unsafe fn(*const (), usize, usize);

const SPIN_LOOP_BUDGET: usize = 1 << 12;
const YIELD_ROUNDS: usize = 64;
const PARK_TIMEOUT: Duration = Duration::from_micros(50);
const MAX_LOOP_COUNTER_SHARDS: usize = 8;

/// How long [`WorkStealingThreadPool::new`] waits for every spawned worker to
/// reach its loop before giving up.
///
/// Deliberately far beyond any real startup: a worker announces on the first
/// line of `worker_loop`, so the only thing between `spawn` returning `Ok` and
/// the announcement is the OS scheduling the new thread once. A wait that
/// reaches this deadline is reporting a pool that can no longer become ready,
/// not a slow one. Matches `onnx-runtime-ep-cpu`'s `POOL_READY_TIMEOUT`, which
/// is the same backstop against the same failure.
const POOL_READY_TIMEOUT: Duration = Duration::from_secs(120);

// Both knobs below are thread-scoped rather than global, so injecting a fault
// into one test cannot reach a pool another test is building concurrently.
#[cfg(test)]
thread_local! {
    /// Test override for [`POOL_READY_TIMEOUT`], in milliseconds; `0` means
    /// "use the real one".
    pub(super) static POOL_READY_TIMEOUT_MS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Test fault injection: the worker index that must never announce
    /// readiness, or `usize::MAX` for none.
    pub(super) static HOLD_WORKER_BEFORE_READY: std::cell::Cell<usize> =
        const { std::cell::Cell::new(usize::MAX) };

    /// Test injection: how long the held worker ignores the shutdown flag, in
    /// milliseconds; `0` honours it immediately.
    ///
    /// The held worker parks on `shared.shutdown`, which the abandon path
    /// sets, so it is *cooperative*: a `join()` restored to that path would
    /// return promptly and leave the suite green while production -- against
    /// the genuinely wedged worker this backstop exists for -- reintroduced
    /// the unbounded wait. This knob models the worker that does not
    /// cooperate, so the no-join contract has an oracle.
    pub(super) static HOLD_WORKER_IGNORES_SHUTDOWN_MS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };

    /// Test injection of a *contended* yield, in microseconds; `0` yields
    /// normally. The every-yield clock check is invisible when yields are
    /// cheap: an uncontended `yield_now` costs ~1.2 us, so a stride of 64
    /// moves the deadline by ~78 us and no assertion against a millisecond
    /// deadline can see it. Injecting the yield *cost* makes the regime
    /// deterministic without manufacturing real contention.
    pub(super) static READY_SLOW_YIELD_US: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Counts the barrier's yields, so a test can discriminate an every-yield
    /// clock check from a strided one by the count rather than by wall clock.
    pub(super) static READY_YIELDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Test injection: on the first deadline expiry, publish `ready ==
    /// worker_count` immediately before the barrier re-reads it, reproducing
    /// the lost race the re-read exists to absorb.
    pub(super) static FORCE_READY_RACE_AT_DEADLINE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Test observation of the held worker's deaf window, tagged by generation.
///
/// Generation-tagged rather than a pair of booleans because a held worker from
/// an *earlier* test outlives the constructor that abandoned it: it is released
/// by the shutdown store, but the store recording that it left runs on the
/// worker's thread and can be descheduled past the next test's reset. Two
/// booleans would then report the previous test's exit as this one's, which is
/// a false failure that no amount of resetting can close. A generation only
/// ever matches the test that issued it.
#[cfg(test)]
pub(super) static DEAF_GENERATION: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub(super) static DEAF_ENTERED_GENERATION: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub(super) static DEAF_LEFT_GENERATION: AtomicUsize = AtomicUsize::new(0);

/// Yield the way the barrier does, counting it and optionally making it
/// expensive. Free in production: the whole body is `#[cfg(test)]`.
fn ready_yield() {
    #[cfg(test)]
    {
        READY_YIELDS.with(|cell| cell.set(cell.get().saturating_add(1)));
        let slow_us = READY_SLOW_YIELD_US.with(std::cell::Cell::get);
        if slow_us > 0 {
            thread::sleep(Duration::from_micros(slow_us));
            return;
        }
    }
    thread::yield_now();
}

/// Read on the builder thread, which is where the barrier runs.
fn pool_ready_timeout() -> Duration {
    #[cfg(test)]
    {
        let ms = POOL_READY_TIMEOUT_MS.with(std::cell::Cell::get);
        if ms > 0 {
            return Duration::from_millis(ms);
        }
    }
    POOL_READY_TIMEOUT
}

#[repr(align(128))]
struct PaddedAtomicUsize(AtomicUsize);

impl PaddedAtomicUsize {
    fn new(value: usize) -> Self {
        Self(AtomicUsize::new(value))
    }

    fn load(&self, ordering: Ordering) -> usize {
        self.0.load(ordering)
    }

    fn store(&self, value: usize, ordering: Ordering) {
        self.0.store(value, ordering);
    }

    fn fetch_add(&self, value: usize, ordering: Ordering) -> usize {
        self.0.fetch_add(value, ordering)
    }
}

struct LoopCounterShard {
    next: PaddedAtomicUsize,
    end: PaddedAtomicUsize,
}

impl LoopCounterShard {
    fn new() -> Self {
        Self {
            next: PaddedAtomicUsize::new(0),
            end: PaddedAtomicUsize::new(0),
        }
    }
}

#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: Option<JobFn>,
    begin: usize,
    end: usize,
    grain: usize,
    num_shards: usize,
    work_items: usize,
    claims: usize,
}

impl Job {
    const fn empty() -> Self {
        Self {
            data: std::ptr::null(),
            call: None,
            begin: 0,
            end: 0,
            grain: 1,
            num_shards: 1,
            work_items: 0,
            claims: 0,
        }
    }
}

struct Shared {
    epoch: AtomicUsize,
    ready: AtomicUsize,
    observed: AtomicUsize,
    remaining: AtomicUsize,
    active: AtomicUsize,
    shutdown: AtomicBool,
    panicked: AtomicBool,
    shards: Vec<LoopCounterShard>,
    thread_count: usize,
    job: UnsafeCell<Job>,
}

unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// Persistent, spin-then-park, per-worker-queue parallel-for pool.
///
/// The pool is intentionally small and synchronous: at most one `parallel_for`
/// may run at a time, and the caller blocks until all worker threads complete.
/// Work is published exactly like ORT's `ParallelForFixedBlockSizeScheduling`:
/// at most one work item per pool lane, up to eight `LoopCounter` shards, and a
/// fixed block size (`grain`). Each work item starts at its home shard and
/// atomically claims the next block, then scans the remaining shards. A delayed
/// worker therefore cannot strand a static range behind a barrier.
pub struct WorkStealingThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    worker_threads: Vec<Thread>,
    dispatch_lock: Mutex<()>,
}

impl WorkStealingThreadPool {
    /// Create a pool with `thread_count` total threads, including the caller.
    pub fn new(thread_count: usize) -> std::io::Result<Self> {
        assert!(thread_count > 0, "thread_count must be non-zero");
        let worker_count = thread_count.saturating_sub(1);

        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            ready: AtomicUsize::new(0),
            observed: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            shards: (0..MAX_LOOP_COUNTER_SHARDS)
                .map(|_| LoopCounterShard::new())
                .collect(),
            thread_count,
            job: UnsafeCell::new(Job::empty()),
        });

        let mut workers = Vec::with_capacity(worker_count);
        // Latch the fault-injection knob on the builder thread: it is
        // thread-local to the injecting test, and a worker runs on its own
        // thread where it is always at its default.
        #[cfg(test)]
        let held_worker = HOLD_WORKER_BEFORE_READY.with(std::cell::Cell::get);
        #[cfg(test)]
        let hold_ignores_shutdown_ms = HOLD_WORKER_IGNORES_SHUTDOWN_MS.with(std::cell::Cell::get);
        #[cfg(test)]
        let force_ready_race = FORCE_READY_RACE_AT_DEADLINE.with(std::cell::Cell::get);
        #[cfg(test)]
        let deaf_generation = DEAF_GENERATION.load(Ordering::SeqCst);
        for worker_id in 0..worker_count {
            let shared = Arc::clone(&shared);
            let queue_id = worker_id + 1;
            workers.push(
                thread::Builder::new()
                    .name(format!("mlas-sys-ws-{worker_id}"))
                    .spawn(move || {
                        #[cfg(test)]
                        if worker_id == held_worker {
                            // Not a panic: a panicking worker drops its
                            // `Arc<Shared>`, a tidier failure than the one being
                            // reproduced. The defect is a worker that exists,
                            // holds its share of the pool, and never announces.
                            //
                            // `HOLD_WORKER_IGNORES_SHUTDOWN_MS` makes it deaf to
                            // the shutdown flag for a bounded window first,
                            // which is the only shape that can catch a `join()`
                            // returning to the abandon path.
                            let deaf_until =
                                Instant::now() + Duration::from_millis(hold_ignores_shutdown_ms);
                            DEAF_ENTERED_GENERATION.store(deaf_generation, Ordering::SeqCst);
                            while !shared.shutdown.load(Ordering::Acquire)
                                || Instant::now() < deaf_until
                            {
                                thread::sleep(Duration::from_millis(1));
                            }
                            DEAF_LEFT_GENERATION.store(deaf_generation, Ordering::SeqCst);
                            return;
                        }
                        worker_loop(shared, queue_id);
                    })?,
            );
        }

        let worker_threads: Vec<Thread> = workers
            .iter()
            .map(|worker| worker.thread().clone())
            .collect();
        // Spin briefly, then *yield*, then give up. A pure `spin_loop` here
        // waits on threads that need a core to make progress, so where the
        // builder and its workers contend for the same CPU the spinner starves
        // the very workers it is waiting for -- livelock, not slow progress,
        // and most reachable exactly where cores are scarcest (an emulated
        // target, a cpuset-confined process, a loaded host). A worker that dies
        // before announcing makes the condition permanently unsatisfiable, and
        // spinning on it burns a core forever while looking exactly like work.
        //
        // The deadline is a liveness backstop, not a performance bound.
        let ready_since = Instant::now();
        let mut spins = 0usize;
        while shared.ready.load(Ordering::Acquire) < worker_count {
            spins = spins.wrapping_add(1);
            if spins < SPIN_LOOP_BUDGET {
                std::hint::spin_loop();
            } else {
                ready_yield();
                // Every yield, not on a stride: a yield under contention costs
                // microseconds to milliseconds, so a stride of N multiplies the
                // deadline's granularity by N yields of an already-starved
                // thread.
                if ready_since.elapsed() >= pool_ready_timeout() {
                    #[cfg(test)]
                    if force_ready_race {
                        shared.ready.store(worker_count, Ordering::Release);
                    }
                    let ready = shared.ready.load(Ordering::Acquire);
                    // Two separate reads, so a worker can announce between
                    // them; accept that pool rather than tearing down a healthy
                    // one.
                    if ready >= worker_count {
                        break;
                    }
                    // Release the workers that did start, using the same
                    // publish-then-wake sequence as `shutdown`. Deliberately
                    // not joined: this build has a worker that never reached
                    // its loop, and blocking on it would reintroduce the
                    // unbounded wait this backstop exists to remove.
                    shared.shutdown.store(true, Ordering::SeqCst);
                    shared.epoch.fetch_add(1, Ordering::Release);
                    for thread in &worker_threads {
                        thread.unpark();
                    }
                    drop(workers);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!(
                            "work-stealing pool never became ready: {ready} of \
                             {worker_count} workers announced within {:?}",
                            pool_ready_timeout()
                        ),
                    ));
                }
            }
        }
        Ok(Self {
            shared,
            workers,
            worker_threads,
            dispatch_lock: Mutex::new(()),
        })
    }

    /// Degree of parallelism, including the caller thread.
    pub fn thread_count(&self) -> usize {
        self.shared.thread_count
    }

    /// Run `body(start, end)` over `[begin, end)` using chunks of at least `grain`.
    ///
    /// A single-chunk range runs inline, matching ORT's trivial-loop fast path.
    pub fn parallel_for<F>(&self, begin: usize, end: usize, grain: usize, body: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        assert!(begin <= end, "parallel_for begin must be <= end");
        assert!(grain > 0, "parallel_for grain must be non-zero");
        let len = end - begin;
        if len == 0 {
            return;
        }
        if len <= grain {
            body(begin, end);
            return;
        }

        let num_blocks = len / grain;
        let claims = len.div_ceil(grain);
        let num_shards = loop_counter_shards(len, self.thread_count(), grain);
        let work_items = self.thread_count().min(num_blocks.max(1));
        let _dispatch_guard = self.dispatch_lock.lock().unwrap();
        self.shared.panicked.store(false, Ordering::Relaxed);

        let blocks_per_shard = num_blocks / num_shards;
        let iterations_per_shard = blocks_per_shard * grain;
        for (shard_idx, shard) in self.shared.shards.iter().enumerate() {
            if shard_idx < num_shards {
                let start = shard_idx * iterations_per_shard;
                let end = if shard_idx + 1 == num_shards {
                    len
                } else {
                    (shard_idx + 1) * iterations_per_shard
                };
                shard.next.store(start, Ordering::Relaxed);
                shard.end.store(end, Ordering::Relaxed);
            } else {
                shard.next.store(0, Ordering::Relaxed);
                shard.end.store(0, Ordering::Relaxed);
            }
        }

        unsafe fn call<F>(data: *const (), begin: usize, end: usize)
        where
            F: Fn(usize, usize) + Sync,
        {
            let body = unsafe { &*(data as *const F) };
            body(begin, end);
        }

        unsafe {
            *self.shared.job.get() = Job {
                data: &body as *const F as *const (),
                call: Some(call::<F>),
                begin,
                end,
                grain,
                num_shards,
                work_items,
                claims,
            };
        }

        self.shared.observed.store(0, Ordering::Relaxed);
        self.shared.remaining.store(claims, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for worker in self
            .worker_threads
            .iter()
            .take(work_items.saturating_sub(1))
        {
            worker.unpark();
        }

        let caller_result = panic::catch_unwind(AssertUnwindSafe(|| run_job(&self.shared, 0)));
        if caller_result.is_err() {
            self.shared.panicked.store(true, Ordering::Release);
        }
        wait_for_completion(&self.shared);
        wait_for_workers(&self.shared);
        if self.shared.panicked.load(Ordering::Acquire) {
            wait_for_active(&self.shared);
        }

        unsafe {
            *self.shared.job.get() = Job::empty();
        }

        if caller_result.is_err() || self.shared.panicked.load(Ordering::Acquire) {
            panic!("WorkStealingThreadPool worker panicked");
        }
    }
}

impl Drop for WorkStealingThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for worker in &self.worker_threads {
            worker.unpark();
        }
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

fn worker_loop(shared: Arc<Shared>, worker_id: usize) {
    let mut seen_epoch = shared.epoch.load(Ordering::Acquire);
    shared.ready.fetch_add(1, Ordering::Release);
    loop {
        let epoch = wait_for_epoch(&shared, seen_epoch);
        seen_epoch = epoch;
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }

        let result = panic::catch_unwind(AssertUnwindSafe(|| run_job(&shared, worker_id)));
        if result.is_err() {
            shared.panicked.store(true, Ordering::Release);
        }
        shared.observed.fetch_add(1, Ordering::Release);
    }
}

fn wait_for_epoch(shared: &Shared, seen_epoch: usize) -> usize {
    let mut rounds = 0;
    loop {
        for _ in 0..SPIN_LOOP_BUDGET {
            let epoch = shared.epoch.load(Ordering::Acquire);
            if epoch != seen_epoch {
                return epoch;
            }
            std::hint::spin_loop();
        }
        if rounds < YIELD_ROUNDS {
            rounds += 1;
            thread::yield_now();
        } else {
            thread::park_timeout(PARK_TIMEOUT);
        }
    }
}

fn wait_for_completion(shared: &Shared) {
    let mut spins = 0usize;
    while shared.remaining.load(Ordering::Acquire) != 0 && !shared.panicked.load(Ordering::Acquire)
    {
        if spins < SPIN_LOOP_BUDGET {
            spins += 1;
            std::hint::spin_loop();
        } else {
            thread::yield_now();
        }
    }
}

fn wait_for_active(shared: &Shared) {
    while shared.active.load(Ordering::Acquire) != 0 {
        std::hint::spin_loop();
    }
}

/// Blocks until every worker has *left* `run_job`, not merely until the last
/// block has run.
///
/// This looks redundant next to [`wait_for_completion`] and is not. Completion
/// only means `remaining` reached zero — the worker that decremented it is
/// still inside `run_job`, holding a by-value copy of the old [`Job`] and
/// looping in `claim_iterations` against the *shared* counters. If the
/// dispatcher returned there and released `dispatch_lock`, the next
/// `parallel_for` (another rayon thread, since MLAS fans out under rayon in
/// `sdpa_f32_fast`) would republish the loop bounds and bump the epoch while
/// that straggler was still live. The straggler would then claim a block of
/// the *new* range and:
///
/// 1. decrement the new job's `remaining` without running the new closure, so
///    `wait_for_completion` returns early and a partition never executes —
///    leaving a `beta = 0` SGEMM's output rows unwritten; and
/// 2. invoke the *old* closure with an index from the new range, writing
///    through raw pointers past the end of the previous GEMM's `C`.
///
/// Those are exactly the two symptoms of #1685 (an 8-row hole of exact `0.0`
/// in SDPA output, and an intermittent SIGSEGV). Waiting for
/// `observed == worker_count` before the caller clears the `Job` and drops the
/// lock closes both. `crates/mlas-sys/tests/concurrent_dispatch.rs` fails
/// within milliseconds if this call is removed.
fn wait_for_workers(shared: &Shared) {
    let worker_count = shared.thread_count.saturating_sub(1);
    while shared.observed.load(Ordering::Acquire) != worker_count {
        std::hint::spin_loop();
    }
}

fn run_job(shared: &Shared, worker_id: usize) {
    let job = unsafe { *shared.job.get() };
    let Some(call) = job.call else {
        return;
    };
    if worker_id >= job.work_items {
        return;
    }

    let home_shard = worker_id % job.num_shards;
    let mut shard = home_shard;
    while !shared.panicked.load(Ordering::Acquire) {
        let Some((iter_begin, iter_end)) =
            claim_iterations(shared, home_shard, &mut shard, job.grain, job.num_shards)
        else {
            break;
        };
        debug_assert!(iter_begin < job.end - job.begin);
        debug_assert!(iter_begin / job.grain <= job.claims);
        let begin = job.begin + iter_begin;
        let end = job.begin + iter_end.min(job.end - job.begin);
        shared.active.fetch_add(1, Ordering::AcqRel);
        let result = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            call(job.data, begin, end);
        }));
        shared.active.fetch_sub(1, Ordering::AcqRel);
        shared.remaining.fetch_sub(1, Ordering::Release);
        if result.is_err() {
            shared.panicked.store(true, Ordering::Release);
            break;
        }
    }
}

fn claim_iterations(
    shared: &Shared,
    home_shard: usize,
    shard: &mut usize,
    block_size: usize,
    num_shards: usize,
) -> Option<(usize, usize)> {
    loop {
        let counter = &shared.shards[*shard];
        let end = counter.end.load(Ordering::Relaxed);
        if counter.next.load(Ordering::Relaxed) < end {
            let start = counter.next.fetch_add(block_size, Ordering::AcqRel);
            if start < end {
                return Some((start, (start + block_size).min(end)));
            }
        }

        *shard = (*shard + 1) % num_shards;
        if *shard == home_shard {
            return None;
        }
    }
}

fn loop_counter_shards(
    num_iterations: usize,
    degree_of_parallelism: usize,
    block_size: usize,
) -> usize {
    let num_blocks = num_iterations / block_size;
    let mut num_shards = if num_blocks == 0 {
        1
    } else {
        num_blocks.min(MAX_LOOP_COUNTER_SHARDS)
    };
    num_shards = num_shards.min(degree_of_parallelism.max(1));
    num_shards.max(1)
}

#[cfg(test)]
mod tests {
    use super::WorkStealingThreadPool;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parallel_for_visits_every_item_once() {
        let pool = WorkStealingThreadPool::new(4).unwrap();
        let hits = (0..257).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
        pool.parallel_for(0, hits.len(), 7, |begin, end| {
            for hit in &hits[begin..end] {
                hit.fetch_add(1, Ordering::Relaxed);
            }
        });

        for hit in hits {
            assert_eq!(hit.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn parallel_for_accepts_stack_borrows() {
        let pool = WorkStealingThreadPool::new(3).unwrap();
        let input = (0..128).collect::<Vec<usize>>();
        let sum = AtomicUsize::new(0);

        pool.parallel_for(0, input.len(), 5, |begin, end| {
            let local = input[begin..end].iter().sum::<usize>();
            sum.fetch_add(local, Ordering::Relaxed);
        });

        assert_eq!(sum.load(Ordering::Relaxed), input.iter().sum::<usize>());
    }

    #[test]
    fn parallel_for_repeated_dispatches_do_not_miss_epochs() {
        let pool = WorkStealingThreadPool::new(4).unwrap();
        let calls = AtomicUsize::new(0);
        for _ in 0..1000 {
            pool.parallel_for(0, 64, 8, |_, _| {
                calls.fetch_add(1, Ordering::Relaxed);
            });
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1000 * 8);
    }
}

/// The readiness backstop: a pool that cannot become ready must report an error
/// rather than hold the machine at full occupancy indefinitely.
#[cfg(test)]
mod ready_backstop_tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// The deadline the wedge test and its negative control share.
    ///
    /// One constant rather than two literals: the control's entire claim is
    /// that *the same* deadline builds a healthy pool, so if the two numbers
    /// can drift apart the control silently stops controlling for anything.
    const PAIRED_DEADLINE_MS: u64 = 1_000;

    /// Serialises the readiness-injection tests against each other.
    ///
    /// The knobs are thread-locals, but the thing they measure is not: an
    /// injected deadline is milliseconds of wall clock, and a second test
    /// building its own pool at the same time competes for the same cores, so
    /// a healthy worker can miss the deadline because of the *other* test's
    /// scheduling. That red is indistinguishable from the defect these tests
    /// exist to catch, which makes it the worst kind -- the cheapest way to
    /// make it stop is to widen the deadline until the test cannot fail.
    static INJECTION: Mutex<()> = Mutex::new(());

    /// Sets the readiness knobs for the test and clears them afterwards.
    ///
    /// Holds [`INJECTION`] for exactly that window, bound to the guard rather
    /// than left to each call site to remember.
    struct Injected {
        _serialise: MutexGuard<'static, ()>,
        generation: usize,
    }

    impl Injected {
        fn new(timeout_ms: u64, held: usize) -> Self {
            let serialise = INJECTION
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            POOL_READY_TIMEOUT_MS.with(|cell| cell.set(timeout_ms));
            HOLD_WORKER_BEFORE_READY.with(|cell| cell.set(held));
            let generation = DEAF_GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
            Self {
                _serialise: serialise,
                generation,
            }
        }

        fn hold_ignores_shutdown_ms(self, ms: u64) -> Self {
            HOLD_WORKER_IGNORES_SHUTDOWN_MS.with(|cell| cell.set(ms));
            self
        }

        fn slow_yield_us(self, us: u64) -> Self {
            READY_SLOW_YIELD_US.with(|cell| cell.set(us));
            self
        }

        fn force_ready_race(self) -> Self {
            FORCE_READY_RACE_AT_DEADLINE.with(|cell| cell.set(true));
            self
        }

        fn reset_yields(self) -> Self {
            READY_YIELDS.with(|cell| cell.set(0));
            self
        }

        fn yields(&self) -> u64 {
            READY_YIELDS.with(std::cell::Cell::get)
        }
    }

    impl Drop for Injected {
        fn drop(&mut self) {
            POOL_READY_TIMEOUT_MS.with(|cell| cell.set(0));
            HOLD_WORKER_BEFORE_READY.with(|cell| cell.set(usize::MAX));
            HOLD_WORKER_IGNORES_SHUTDOWN_MS.with(|cell| cell.set(0));
            READY_SLOW_YIELD_US.with(|cell| cell.set(0));
            READY_YIELDS.with(|cell| cell.set(0));
            FORCE_READY_RACE_AT_DEADLINE.with(|cell| cell.set(false));
        }
    }

    /// A worker that never announces must fail the build, not hang it.
    ///
    /// Before this change the constructor ran `while ready != worker_count {
    /// spin_loop() }` with no yield, no bound and no exit, so one wedged worker
    /// held the builder's core at 100% for the life of the process --
    /// indistinguishable from work.
    #[test]
    fn a_worker_that_never_announces_fails_the_build_instead_of_hanging() {
        let injected = Injected::new(PAIRED_DEADLINE_MS, 0);
        let started = Instant::now();
        let err = WorkStealingThreadPool::new(4)
            .err()
            .expect("a pool with a worker that never announces was reported as built");
        let elapsed = started.elapsed();
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut, "{err}");
        // "fewer than all of 3" rather than the exact "2 of 3": under emulation
        // or contention a healthy worker can also still be short of announcing
        // when the deadline fires, and a test that reds on that is reporting
        // the host rather than the backstop.
        let message = err.to_string();
        let announced = message
            .split(" of 3 workers announced")
            .next()
            .and_then(|head| head.rsplit(' ').next())
            .and_then(|count| count.parse::<usize>().ok());
        match announced {
            Some(count) => assert!(
                count < 3,
                "the backstop fired reporting all 3 workers announced, which is \
                 not a wedge at all: {err}"
            ),
            None => panic!(
                "the diagnostic does not say how far the pool got, which is the \
                 only thing that distinguishes this from a spawn failure: {err}"
            ),
        }
        assert!(
            elapsed < Duration::from_secs(30),
            "the build took {elapsed:?} against a {PAIRED_DEADLINE_MS}ms deadline, \
             so it is not the deadline that ended it"
        );
        drop(injected);
    }

    /// The negative control: the same deadline must build a healthy pool, so
    /// the test above cannot be passing because the backstop fires on every
    /// build.
    #[test]
    fn the_same_deadline_builds_a_healthy_pool() {
        let injected = Injected::new(PAIRED_DEADLINE_MS, usize::MAX);
        let pool = WorkStealingThreadPool::new(4).expect("healthy pool must build");
        assert_eq!(pool.thread_count(), 4);
        let calls = AtomicUsize::new(0);
        pool.parallel_for(0, 64, 1, |begin, end| {
            calls.fetch_add(end - begin, Ordering::Relaxed);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 64);
        drop(injected);
    }

    /// The deadline must be consulted on every yield, not on a stride.
    ///
    /// The discriminator is the yield *count*, not elapsed time. A wall-clock
    /// bound has to sit above the healthy path and below the strided signature,
    /// an interval contention can close from below -- at which point the test
    /// either reds on the host or gets widened past the signature and silently
    /// stops catching the stride. A stride of N cannot leave before its first
    /// multiple of N, and a slow host only makes each yield cost *more* wall
    /// clock, moving the count away from the failing value rather than towards
    /// it.
    #[test]
    fn the_deadline_is_checked_on_every_yield_not_on_a_stride() {
        let injected = Injected::new(100, 0).slow_yield_us(12_000).reset_yields();
        WorkStealingThreadPool::new(3)
            .err()
            .expect("the injected hold must fail the build");
        let yields = injected.yields();
        assert!(
            yields > 0,
            "no yield happened, so the stride this test is about was never reached"
        );
        // `thread::sleep` is a floor, never early, so a correct barrier cannot
        // exceed ceil(deadline / yield_cost) = ceil(100/12) = 9 yields on any
        // host; a slower host makes each yield cost more and drives the count
        // *down*. 3x that is a bound only a strided check can cross -- the
        // sibling barrier's stride of 64 lands at 64-65 -- and it needs no
        // stride constant of its own to state.
        const PREDICTED_YIELDS: u64 = 9;
        assert!(
            yields <= 3 * PREDICTED_YIELDS,
            "the barrier yielded {yields} times at 12ms per yield against a \
             100ms deadline, more than 3x the {PREDICTED_YIELDS} an every-yield \
             clock check can take, so it cannot have consulted the clock on each \
             one; that is the signature of a strided check"
        );
        drop(injected);
    }

    /// The abandoned workers must not be joined, and a cooperative fault
    /// cannot show that.
    ///
    /// The abandon path drops the handles unjoined on purpose: joining would
    /// block on precisely the worker that is not coming. Every other test here
    /// injects a worker that parks on `shared.shutdown` -- the same flag that
    /// path sets -- so it *is* coming, and a restored `join()` would return
    /// promptly and leave the suite green while production hung.
    ///
    /// The oracle is the worker's own deaf flag rather than elapsed time, so
    /// there is no absolute threshold for a loaded or emulated host to breach:
    /// correct code returns *during* the deaf window, a restored join returns
    /// only after it closes, and both sides scale with the host.
    #[test]
    fn an_abandoned_worker_that_ignores_shutdown_does_not_delay_the_builder() {
        let injected = Injected::new(PAIRED_DEADLINE_MS, 0).hold_ignores_shutdown_ms(8_000);
        WorkStealingThreadPool::new(3)
            .err()
            .expect("the injected hold must fail the build");
        assert_eq!(
            DEAF_ENTERED_GENERATION.load(Ordering::SeqCst),
            injected.generation,
            "the injected worker never reached its deaf window, so this test \
             observed nothing about the abandon path"
        );
        assert_ne!(
            DEAF_LEFT_GENERATION.load(Ordering::SeqCst),
            injected.generation,
            "the constructor returned only after the deaf worker finished, \
             which is the signature of the abandon path joining the handles it \
             must only drop"
        );
        drop(injected);
    }

    /// A worker that announces inside the deadline's race window must be kept,
    /// not torn down.
    ///
    /// The barrier reads `ready` twice -- once in the loop condition, once
    /// after the deadline fires -- so a worker can announce between them.
    /// Without the re-read the pool is destroyed and the diagnostic says the
    /// self-contradictory "N of N workers announced ... never became ready".
    /// The knob publishes the announcement in exactly that window, which no
    /// amount of timing could do reliably.
    #[test]
    fn a_worker_that_announces_in_the_race_window_is_not_torn_down() {
        let injected = Injected::new(50, 0).force_ready_race();
        let pool = WorkStealingThreadPool::new(3)
            .expect("a pool whose workers all announced must not be torn down");
        assert_eq!(pool.thread_count(), 3);
        drop(pool);
        drop(injected);
    }
}
