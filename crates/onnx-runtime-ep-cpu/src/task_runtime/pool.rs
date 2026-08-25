//! The native side of the CPU task runtime: a persistent, multi-dispatcher,
//! dynamically-claimed thread pool.
//!
//! This is the pool [`super`] uses when there is no host runtime pool to borrow
//! — the native `InferenceSession`, unit tests, and any embedder that is not
//! ORT. Inside an ORT plugin-EP compute call the façade routes to
//! `KernelContext_ParallelFor` instead and this pool stays parked, because two
//! spinning pools on one machine is the failure mode the whole exercise exists
//! to remove.
//!
//! # What makes it different from [`crate::decode_spmd`]
//!
//! `decode_spmd` is an SPMD broadcast barrier: one job slot, one barrier, one
//! shard per worker, statically assigned. That is the right shape for the decode
//! GEMM, whose shards are equal-cost and whose weight placement must line up
//! with a fixed row split. It is the wrong shape for everything else:
//!
//! * **One dispatcher.** A second concurrent dispatcher loses the claim and runs
//!   its whole fan-out inline (#1184 made that *safe*; it did not make it
//!   *parallel*). Two sessions decoding in one process therefore serialise.
//! * **Static shards.** A worker that is late strands its whole shard behind the
//!   barrier; nobody can take it.
//! * **A fixed 500 µs blocktime.** Right for tight decode, pure waste for a
//!   session that runs one graph a second.
//!
//! This pool fixes all three: `SLOT_COUNT` independent job slots so concurrent
//! dispatchers get real parallelism, per-task dynamic claiming so a late worker
//! strands at most one task, and a per-worker adaptive spin window that grows
//! while dispatches keep arriving and decays to a park when they stop.
//!
//! # The claim protocol, and why it is sound
//!
//! Each slot holds a countdown `claim` cursor and a `remaining` completion
//! counter. Publishing writes the job, stores `remaining = total`, then
//! **releases** `claim = total`; a participant takes a task with
//! `claim.fetch_sub(1)` and owns index `left - 1` when `left >= 1`.
//!
//! The one non-obvious safety argument is why a participant may dereference the
//! borrowed closure after claiming:
//!
//! > A successful claim (`left >= 1`) means that task had not been taken by
//! > anyone, so its completion is still outstanding, so `remaining > 0`, so the
//! > dispatcher is still blocked in `dispatch` — and the dispatcher is the
//! > owner of the closure's stack frame. The claim's `Acquire` pairs with the
//! > publisher's `Release` store of `claim`, which is ordered after the job
//! > write, so the job the claimant reads is the job whose counter it just
//! > decremented, even if the slot was recycled by a different dispatcher in
//! > between.
//!
//! The completion side closes the loop: the last `remaining.fetch_sub(1,
//! Release)` synchronises with the dispatcher's `Acquire` load of zero, so every
//! participant's read of the job pointer happens-before the dispatcher returns
//! and the frame dies.
//!
//! # Cost
//!
//! Nothing on the dispatch path allocates. A dispatch is: one plain write of a
//! two-word job, two relaxed stores, one release store, one `fetch_or`, one
//! `fetch_add` and one futex wake — then arithmetic. Task ranges are *computed*
//! from the task index rather than materialised into a `Vec`, so a 4096-task
//! fan-out and a 2-task fan-out allocate the same zero bytes.

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Independent job slots, i.e. how many dispatchers can be in flight before one
/// has to run its fan-out inline.
///
/// Eight is chosen against the concurrency this has to survive rather than
/// against a benchmark: a multi-model server runs one dispatcher per concurrent
/// `Run`, and past eight the workers' scan of the active mask starts to cost
/// more than the ninth dispatcher gains. Exhaustion is not a failure — the
/// caller runs serially and [`PoolCounters::slot_exhausted`] records it, so the
/// choice is observable rather than silent.
pub(crate) const SLOT_COUNT: usize = 8;

/// Shortest adaptive spin window a worker will hold a core for before parking.
///
/// Not zero: a worker that parks the instant a region ends pays a futex wake on
/// the next one, and the measured cost of that is the entire problem (§26.1 of
/// `docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md`: a 20 µs gap
/// between regions took an isolated elementwise region from 67 µs to 226 µs).
const MIN_SPIN: Duration = Duration::from_micros(20);

/// Longest adaptive spin window.
///
/// Bounds idle CPU after a burst ends: every worker parks within one window of
/// the last dispatch, so a process that stops inferencing returns to ~0% CPU in
/// under a millisecond. Matches `decode_spmd`'s fixed `KMP_BLOCKTIME` analogue,
/// which is the value that regime was tuned to — the difference here is that
/// this is the *ceiling* of an adaptive window rather than a constant.
const MAX_SPIN: Duration = Duration::from_micros(500);

/// Pure `spin_loop` iterations before a spinning worker starts yielding.
const SPIN_LOOP_BUDGET: u32 = 1 << 12;

/// Spin iterations between wall-clock reads *during the pure-`spin_loop` phase*,
/// so `Instant::now()` (~20 ns) is amortised rather than paid every iteration.
/// It deliberately does not gate the yield phase: a yield costs microseconds to
/// milliseconds under contention, so a stride there would multiply the spin
/// window's granularity by 64 yields of an already-starved thread (#1825).
const CLOCK_CHECK_STRIDE: u32 = 1 << 6;

/// Dispatcher spins between `yield_now` calls while waiting for stragglers.
const DISPATCHER_YIELD_STRIDE: u32 = 1 << 12;

/// How long [`TaskPool::new`]'s readiness barrier waits for every spawned
/// worker to reach its loop before failing loud.
///
/// Deliberately far beyond any real startup: a worker announces on the first
/// line of `worker_loop`, so the only thing between `spawn` returning `Ok` and
/// the announcement is the OS scheduling the new thread once. Even under
/// `qemu-user` on a host at 78 load that is milliseconds, not minutes. A wait
/// that reaches this deadline is therefore reporting a pool that can no longer
/// become ready, not a slow one.
///
/// Matches `decode_spmd`'s [`crate::decode_spmd`] barrier deliberately: the two
/// are the same backstop against the same failure, and a reader who has
/// understood one should not have to re-derive the other's bound.
const POOL_READY_TIMEOUT: Duration = Duration::from_secs(120);

// Scoped to the thread that builds the pool, not global, for the reason
// `decode_spmd` documents at length: a global knob is read by every
// concurrently-building pool in the same test binary, so injecting a fault into
// one test reaches pools belonging to unrelated tests. A `Mutex` does not fix
// that, because the tests being corrupted never take it.
#[cfg(test)]
thread_local! {
    /// Test override for [`POOL_READY_TIMEOUT`], in milliseconds; `0` means
    /// "use the real one". A liveness backstop is only testable if the test
    /// does not have to wait out the production deadline to observe it.
    static POOL_READY_TIMEOUT_MS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Test fault injection: the worker index that must never announce
    /// readiness, or `usize::MAX` for none. Reproduces the one failure the
    /// barrier cannot distinguish from slowness -- a worker that will never
    /// arrive.
    static HOLD_WORKER_BEFORE_READY: std::cell::Cell<usize> =
        const { std::cell::Cell::new(usize::MAX) };

    /// Test injection: how long the held worker ignores the shutdown flag, in
    /// milliseconds; `0` honours it immediately.
    ///
    /// The held worker above parks on `shared.shutdown`, which
    /// `abandon_unready_workers` sets. That makes it *cooperative*, so a
    /// `join()` re-added to the abandon path would return promptly and every
    /// test would still pass -- while in production, against the genuinely
    /// wedged worker this backstop exists for, that same join reintroduces the
    /// unbounded wait. This knob models the worker that does not cooperate, so
    /// the no-join contract has an oracle. Bounded rather than permanent: a
    /// thread that never exits would leak into the rest of the test binary and
    /// trip Miri's remaining-threads check at exit, and a bounded stubbornness
    /// discriminates just as sharply.
    static HOLD_WORKER_IGNORES_SHUTDOWN_MS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Test injection: on the first deadline expiry, publish `ready ==
    /// workers` immediately before the barrier re-reads it, reproducing the
    /// lost race the re-read exists to absorb. No amount of timing could
    /// arrange that window reliably.
    static FORCE_READY_RACE_AT_DEADLINE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Test injection of the *other* case: every worker announces, but late.
    /// Without it the healthy-pool assertion is vacuous, because real workers
    /// announce inside the spin budget and the deadline is never consulted --
    /// so the test would pass with a 1 ms bound just as happily as a 120 s one.
    static DELAY_WORKER_BEFORE_READY_MS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Test injection of a *contended* yield, in microseconds; `0` yields
    /// normally. The every-yield clock check is invisible when yields are
    /// cheap: an uncontended `yield_now` costs ~1.2 us, so a stride of 64 moves
    /// the deadline by ~78 us and no assertion against a millisecond deadline
    /// can see it. The production measurement behind #1933 recorded ~7 ms per
    /// yield on a contended pair of CPUs -- three orders of magnitude larger --
    /// which is what turns a stride into a 3x deadline overrun. Manufacturing
    /// real contention in a unit test is the kind of load-dependent arrangement
    /// that flakes; injecting the yield *cost* makes the regime deterministic.
    static READY_SLOW_YIELD_US: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Yields the readiness barrier performed on this thread. The anti-vacuity
    /// observable for the slow-but-healthy case: without it that test passes
    /// whether or not the wait ever left its pure-spin phase, which is the
    /// phase the fix is about.
    static READY_YIELDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The part of a task-runtime worker's thread name that survives `/proc`.
///
/// Linux caps `comm` at 15 bytes plus its NUL, so a name longer than that is
/// truncated before anything can read it back. Shared with the spawn site so
/// the two cannot drift: a `/proc` scan filtered on a name is a *filter*, and a
/// filter that matches nothing reports "no threads" -- which is the *passing*
/// answer for the teardown test that reads it. `the_worker_name_survives_proc_truncation`
/// is the tripwire on that.
const TASK_THREAD_NAME_PREFIX: &str = "nxrt-task";

/// Linux's `comm` field width, excluding the terminating NUL (`TASK_COMM_LEN -
/// 1`). Not configurable and not queryable at runtime; see `man 5 proc`.
#[cfg(test)]
const COMM_MAX_BYTES: usize = 15;

/// The readiness barrier's yield, with the injected cost applied. Runs on the
/// builder thread, which is the test's own thread, so the thread-local scoping
/// makes this answer the injecting test's question rather than some other
/// test's.
fn ready_yield() {
    thread::yield_now();
    #[cfg(test)]
    {
        READY_YIELDS.with(|count| count.set(count.get() + 1));
        let us = READY_SLOW_YIELD_US.with(std::cell::Cell::get);
        if us > 0 {
            thread::sleep(Duration::from_micros(us));
        }
    }
}

/// Test observation of the held worker's deaf window, tagged by generation.
///
/// Generation-tagged rather than a pair of booleans because a held worker from
/// an *earlier* test outlives the constructor that abandoned it: it is released
/// by the shutdown store, but the store recording that it left runs on the
/// worker's thread and can be descheduled past the next test's reset. Two
/// booleans would then report the previous test's exit as this one's -- a false
/// failure no amount of resetting can close. A generation only ever matches the
/// test that issued it.
#[cfg(test)]
static DEAF_GENERATION: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static DEAF_ENTERED_GENERATION: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static DEAF_LEFT_GENERATION: AtomicUsize = AtomicUsize::new(0);

/// Read on the builder thread only, which is where the barrier runs, so the
/// thread-local scoping above answers the injecting test's question rather than
/// some other test's.
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

/// Cache-line padding. The claim cursor, the completion counter and the epoch
/// are each written by every participant of a dispatch; sharing a line would
/// turn one fan-out into three lines' worth of coherency traffic per task.
#[repr(align(128))]
struct Padded<T>(T);

/// A type-erased fan-out body: run one task index.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

/// `call` is never invoked for an unpublished slot: a participant only reads the
/// job after a successful claim, and an unpublished slot's cursor is zero.
unsafe fn never_called(_: *const (), _: usize) {
    unreachable!("an unpublished CPU task-runtime slot was claimed");
}

impl Job {
    const fn empty() -> Self {
        Self {
            data: std::ptr::null(),
            call: never_called,
        }
    }
}

/// One dispatcher's in-flight fan-out.
struct Slot {
    /// Countdown claim cursor. `fetch_sub` returning `left >= 1` claims task
    /// index `left - 1`. Reset *absolutely* at publish, never incrementally, so
    /// the negative drift left by losing claimants cannot accumulate.
    claim: Padded<AtomicIsize>,
    /// Tasks whose body has not finished. The dispatcher waits for zero.
    remaining: Padded<AtomicUsize>,
    /// Valid while `remaining > 0`; see the module docs for why that is enough.
    job: UnsafeCell<Job>,
    /// Set when a task body panicked, so the dispatcher can re-raise on its own
    /// thread instead of letting the unwind cross a worker boundary.
    panicked: AtomicBool,
}

impl Slot {
    fn new() -> Self {
        Self {
            claim: Padded(AtomicIsize::new(0)),
            remaining: Padded(AtomicUsize::new(0)),
            job: UnsafeCell::new(Job::empty()),
            panicked: AtomicBool::new(false),
        }
    }
}

/// Observable pool behaviour, for tests and for diagnostics. Every counter is
/// monotonic and relaxed; they exist to make assertions about *scheduling*
/// possible without timing, which is the only way to test a scheduler on a
/// shared runner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolCounters {
    /// Fan-outs published to the pool.
    pub dispatches: u64,
    /// Task bodies run by any participant.
    pub tasks: u64,
    /// Task bodies run by the dispatching thread itself.
    pub tasks_by_dispatcher: u64,
    /// Fan-outs that found no free slot and were declined back to the caller.
    pub slot_exhausted: u64,
    /// Times a worker futex-parked.
    pub parks: u64,
    /// Times a worker's spin window caught a dispatch before parking.
    pub spin_hits: u64,
    /// Task bodies that panicked.
    pub panics: u64,
    /// Fan-outs whose dispatcher outran its own slot and had to wait for a
    /// straggler claimed by someone else.
    pub straggler_waits: u64,
    /// Times a waiting dispatcher exhausted `DISPATCHER_YIELD_STRIDE` spins and
    /// yielded. Only reachable when the straggler's claimant is descheduled,
    /// so this is a direct oversubscription signal.
    pub straggler_yields: u64,
    /// `sched_yield` calls made by *workers* in the second phase of their spin
    /// window, after `SPIN_LOOP_BUDGET` pure `spin_loop`s and before parking.
    ///
    /// Separate from [`Self::straggler_yields`], which counts the dispatcher's
    /// side and is already rate-limited by `DISPATCHER_YIELD_STRIDE`. This one
    /// is not rate-limited at all: the phase yields on every iteration, which
    /// is the shape #2072 measured at 2.61 of 16 cores of kernel time in
    /// `decode_spmd`. Whether it costs the same here is #2075, and this counter
    /// exists so that question can be answered from a release build instead of
    /// from the resemblance between two loops.
    ///
    /// A yield rate is only interpretable against the waits that could have
    /// produced it, so read it with [`Self::parks`] and [`Self::spin_hits`]:
    /// those two count spin windows that ended, and this counts yields spent
    /// inside them.
    pub spin_yields: u64,
}

#[derive(Default)]
struct AtomicCounters {
    dispatches: AtomicU64,
    tasks: AtomicU64,
    tasks_by_dispatcher: AtomicU64,
    slot_exhausted: AtomicU64,
    straggler_waits: AtomicU64,
    straggler_yields: AtomicU64,
    spin_yields: AtomicU64,
    parks: AtomicU64,
    spin_hits: AtomicU64,
    panics: AtomicU64,
}

impl AtomicCounters {
    fn snapshot(&self) -> PoolCounters {
        PoolCounters {
            dispatches: self.dispatches.load(Ordering::Relaxed),
            tasks: self.tasks.load(Ordering::Relaxed),
            tasks_by_dispatcher: self.tasks_by_dispatcher.load(Ordering::Relaxed),
            slot_exhausted: self.slot_exhausted.load(Ordering::Relaxed),
            straggler_waits: self.straggler_waits.load(Ordering::Relaxed),
            straggler_yields: self.straggler_yields.load(Ordering::Relaxed),
            spin_yields: self.spin_yields.load(Ordering::Relaxed),
            parks: self.parks.load(Ordering::Relaxed),
            spin_hits: self.spin_hits.load(Ordering::Relaxed),
            panics: self.panics.load(Ordering::Relaxed),
        }
    }
}

struct Shared {
    slots: Vec<Slot>,
    /// Bit set = slot free for a dispatcher to claim.
    free: AtomicU32,
    /// Bit set = slot holds a published job. Workers scan this.
    active: Padded<AtomicU32>,
    /// Bumped once per publish and futex-waited on by parked workers, so a
    /// dispatch is one `wake_all` syscall rather than an O(workers) unpark.
    ///
    /// Workers watch the *epoch*, not `active`: `active` stays set for the whole
    /// life of a job, so a worker that has drained every claimable task would
    /// otherwise re-scan in a tight loop until the last straggler finished.
    epoch: Padded<AtomicU32>,
    shutdown: AtomicBool,
    ready: AtomicUsize,
    counters: AtomicCounters,
}

// SAFETY: `slots[i].job` is only read by a participant holding a claim on slot
// `i`, which the module docs prove keeps the pointee alive; every other field is
// an atomic.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

thread_local! {
    /// Set while this thread is inside a task body, on workers and on the
    /// dispatcher alike. A nested fan-out reaching the façade sees this and
    /// stays serial rather than nesting a second region inside the first.
    static IN_TASK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread is currently executing a task body.
pub fn in_task() -> bool {
    IN_TASK.with(std::cell::Cell::get)
}

struct TaskGuard(bool);

impl TaskGuard {
    fn enter() -> Self {
        Self(IN_TASK.with(|c| c.replace(true)))
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        IN_TASK.with(|c| c.set(self.0));
    }
}

impl Shared {
    /// Take exclusive ownership of a free slot, preferring `hint` so concurrent
    /// dispatchers spread across slots instead of contending on slot 0.
    fn claim_slot(&self, hint: usize) -> Option<usize> {
        let start = hint % SLOT_COUNT;
        let mut free = self.free.load(Ordering::Relaxed);
        loop {
            let index = (0..SLOT_COUNT)
                .map(|offset| (start + offset) % SLOT_COUNT)
                .find(|&i| free & (1 << i) != 0)?;
            let bit = 1u32 << index;
            match self.free.compare_exchange_weak(
                free,
                free & !bit,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(index),
                Err(current) => free = current,
            }
        }
    }

    fn release_slot(&self, index: usize) {
        self.free.fetch_or(1 << index, Ordering::Release);
    }

    /// Claim and run tasks from `index` until nothing is claimable, returning
    /// how many ran.
    fn run_slot(&self, index: usize) -> u64 {
        let slot = &self.slots[index];
        let mut ran = 0u64;
        loop {
            // Non-mutating pre-check. Without it a worker whose slot is active
            // but exhausted would drive the cursor negative once per scan, and
            // the outer loop can scan very fast while a long straggler task
            // finishes.
            if slot.claim.0.load(Ordering::Acquire) < 1 {
                break;
            }
            let left = slot.claim.0.fetch_sub(1, Ordering::Acquire);
            if left < 1 {
                break;
            }
            let task = (left - 1) as usize;
            // SAFETY: the claim above proves this task was outstanding, so
            // `remaining > 0` and the dispatcher is still parked inside
            // `dispatch` holding the closure's frame alive. The `Acquire` pairs
            // with the `Release` store of `claim` in `dispatch`, which is
            // ordered after the job write.
            let job = unsafe { *slot.job.get() };
            let outcome = {
                let _task = TaskGuard::enter();
                // A panic must not escape into another dispatcher's frame, and
                // must not skip the completion decrement below — that would hang
                // the dispatcher forever.
                // SAFETY: `job.data` was produced from a live `&F` by
                // `dispatch`, and `job.call` is that `F`'s monomorphised
                // trampoline.
                catch_unwind(AssertUnwindSafe(|| unsafe { (job.call)(job.data, task) }))
            };
            if outcome.is_err() {
                slot.panicked.store(true, Ordering::Relaxed);
                self.counters.panics.fetch_add(1, Ordering::Relaxed);
            }
            // Counted *before* the completion decrement, not batched after the
            // loop: the dispatcher is released by that decrement and may read
            // the counters immediately, while this worker is still inside its
            // claim loop. Publishing per task puts every increment behind the
            // `Release` the dispatcher's `Acquire` pairs with, so a snapshot
            // taken after a fan-out returns can never be missing one of its
            // tasks.
            self.counters.tasks.fetch_add(1, Ordering::Relaxed);
            slot.remaining.0.fetch_sub(1, Ordering::Release);
            ran += 1;
        }
        ran
    }

    /// Run every claimable task in every active slot, starting at `hint`.
    fn drain(&self, hint: usize) -> u64 {
        let active = self.active.0.load(Ordering::Acquire);
        if active == 0 {
            return 0;
        }
        let mut ran = 0;
        for offset in 0..SLOT_COUNT {
            let index = (hint + offset) % SLOT_COUNT;
            if active & (1 << index) != 0 {
                ran += self.run_slot(index);
            }
        }
        ran
    }

    fn worker_loop(self: &Arc<Self>, worker_id: usize) {
        self.ready.fetch_add(1, Ordering::Release);
        let mut spin_window = MIN_SPIN;
        let mut last_epoch = self.epoch.0.load(Ordering::Acquire);
        loop {
            if self.shutdown.load(Ordering::Acquire) {
                return;
            }
            let ran = self.drain(worker_id);
            let epoch = self.epoch.0.load(Ordering::Acquire);
            if ran > 0 || epoch != last_epoch {
                last_epoch = epoch;
                if ran > 0 {
                    spin_window = (spin_window * 2).min(MAX_SPIN);
                }
                continue;
            }

            // Bounded active spin: catch the next dispatch without a syscall.
            let caught = match self.spin_for_dispatch(last_epoch, spin_window) {
                SpinOutcome::Shutdown => return,
                SpinOutcome::Caught => true,
                SpinOutcome::Expired => false,
            };
            if caught {
                self.counters.spin_hits.fetch_add(1, Ordering::Relaxed);
                // The window paid for itself; hold a core longer next time.
                spin_window = (spin_window * 2).min(MAX_SPIN);
                continue;
            }

            // Genuinely idle: give the core back. `atomic_wait::wait` re-checks
            // the epoch under the futex guard, so a publish racing this arm
            // returns immediately instead of sleeping through the wake.
            self.counters.parks.fetch_add(1, Ordering::Relaxed);
            atomic_wait::wait(&self.epoch.0, last_epoch);
            // The window did not pay for itself; shrink it so a mostly-idle
            // process converges on parking rather than on burning a core.
            spin_window = (spin_window / 2).max(MIN_SPIN);
        }
    }

    /// One bounded spin window: wait for the epoch to move past `last_epoch`,
    /// giving up after `spin_window`.
    ///
    /// Split out of [`Self::worker_loop`] so the deadline policy can be driven
    /// from a test thread. It cannot be exercised through the pool: a real
    /// worker runs this on a thread the test does not own, so the only
    /// observable from outside is wall-clock idle CPU, which is not
    /// discriminating on a shared box. Nothing else calls it.
    fn spin_for_dispatch(&self, last_epoch: u32, spin_window: Duration) -> SpinOutcome {
        let start = Instant::now();
        let mut spins = 0u32;
        let mut yields = 0u64;
        let outcome = loop {
            if self.shutdown.load(Ordering::Acquire) {
                break SpinOutcome::Shutdown;
            }
            if self.epoch.0.load(Ordering::Acquire) != last_epoch {
                break SpinOutcome::Caught;
            }
            spins = spins.wrapping_add(1);
            if spins < SPIN_LOOP_BUDGET {
                std::hint::spin_loop();
                // The stride belongs here and only here: a `spin_loop`
                // iteration costs nanoseconds, so an unamortised
                // `Instant::now()` would dominate the phase. And the check
                // has to run here at all, because at the converged idle
                // window (`MIN_SPIN`, 20us) the deadline expires *before*
                // the spin phase ends: 4096 `spin_loop`s measure 130us on
                // this host.
                if spins.is_multiple_of(CLOCK_CHECK_STRIDE) && start.elapsed() >= spin_window {
                    break SpinOutcome::Expired;
                }
            } else {
                spin_yield();
                yields += 1;
                // ...and must not apply here. A yield costs microseconds to
                // milliseconds under contention, so a stride of 64
                // multiplies the window's granularity by 64 yields of an
                // already-starved thread -- see #1825, which made this
                // correction in `decode_spmd`'s readiness barrier. It bites
                // whenever the window outlasts the spin phase, which is the
                // upper part of the grown range rather than all of it: 4096
                // `spin_loop`s measure 130us on this host, so windows from
                // ~160us up -- including the `MAX_SPIN` (500us) ceiling a
                // busy steady state converges on -- reach this phase, while
                // the shorter windows above the floor expire in the spin
                // phase above and never arrive here. `MAX_SPIN`'s contract
                // is that a process which stops inferencing returns to ~0%
                // CPU in under a millisecond; a stride-gated check cannot
                // honour that under exactly the load that makes it matter:
                // measured, a yield with four runnable siblings on one core
                // costs 11.2ms, so a 64-yield stride holds the core 717ms
                // past a 500us window.
                if start.elapsed() >= spin_window {
                    break SpinOutcome::Expired;
                }
            }
        };
        // Recorded on exit rather than per iteration: a thread-local bump
        // inside the spin phase would be a sizeable fraction of a ~31ns
        // `spin_loop` and would distort the very phase under test.
        #[cfg(test)]
        SPIN_COUNT.with(|c| c.set(spins as u64));
        // Same reasoning, and the reason this is a local `u64` rather than a
        // `fetch_add` per yield: an instrument for #2075 that added a shared
        // atomic to every iteration of the loop it is measuring would be
        // changing the contention it exists to report. One relaxed add per
        // window is off the measured path.
        if yields > 0 {
            self.counters
                .spin_yields
                .fetch_add(yields, Ordering::Relaxed);
        }
        outcome
    }
}

/// Why a spin window ended. `Expired` and `Caught` drive opposite window
/// adjustments, so a test that cannot tell them apart cannot tell "gave the
/// core back late" from "gave it back for the right reason".
#[derive(Debug, PartialEq, Eq)]
enum SpinOutcome {
    /// The epoch moved: a dispatch arrived within the window.
    Caught,
    /// The window elapsed with no dispatch. The caller parks.
    Expired,
    /// The pool is shutting down.
    Shutdown,
}

#[cfg(test)]
thread_local! {
    /// Injected cost of one yield in [`spin_yield`], in microseconds. Models
    /// the contended regime, where a yield costs microseconds to milliseconds
    /// rather than the ~1.2us an uncontended one costs, without requiring the
    /// test to actually contend a shared box.
    ///
    /// Thread-scoped, and that is what makes it sound: a real pool's workers
    /// run `spin_for_dispatch` on threads that never set this and so always
    /// read `0`. Only a test calling it directly on its own thread reaches the
    /// injection.
    static SLOW_YIELD_US: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// Yields injected on this thread. This is the observable for the stride,
    /// and it is chosen because it is *monotone in the right direction under
    /// load*: a starved thread accumulates wall time faster per yield, so it
    /// crosses the deadline in FEWER yields, never more. A wall-clock
    /// observable ("was it parked at T?") is not monotone that way and goes
    /// flaky beside 1700 siblings.
    static YIELD_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

    /// `spin_loop` iterations the last [`Shared::spin_for_dispatch`] on this
    /// thread ran before returning. The observable for the *spin*-phase
    /// deadline, and monotone in the safe direction for the same reason
    /// [`YIELD_COUNT`] is: a preempted thread accumulates wall time faster per
    /// iteration, so it crosses the deadline in FEWER spins, never more. Load
    /// can only push the guard below towards passing.
    static SPIN_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The yield in the spin loop's second phase, with a test-only injected cost.
///
/// In a non-test build this is `thread::yield_now()` and nothing else.
fn spin_yield() {
    #[cfg(test)]
    {
        let us = SLOW_YIELD_US.with(std::cell::Cell::get);
        if us > 0 {
            YIELD_COUNT.with(|c| c.set(c.get() + 1));
            thread::sleep(Duration::from_micros(us));
        }
    }
    thread::yield_now();
}

/// A persistent, multi-dispatcher, dynamically-claimed CPU task pool.
pub struct TaskPool {
    shared: Arc<Shared>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    /// Worker threads, excluding the dispatching thread (which participates).
    workers: usize,
}

impl TaskPool {
    /// Build a pool whose fan-outs run across `width` threads in total,
    /// including the dispatching thread. `width <= 1` builds a pool that always
    /// declines, and creates no threads.
    pub fn new(width: usize) -> Self {
        let requested = width.saturating_sub(1);
        let shared = Arc::new(Shared {
            slots: (0..SLOT_COUNT).map(|_| Slot::new()).collect(),
            free: AtomicU32::new((1u32 << SLOT_COUNT) - 1),
            active: Padded(AtomicU32::new(0)),
            epoch: Padded(AtomicU32::new(0)),
            shutdown: AtomicBool::new(false),
            ready: AtomicUsize::new(0),
            counters: AtomicCounters::default(),
        });
        let mut handles = Vec::with_capacity(requested);
        // Latch the fault-injection knobs here, on the builder thread, rather
        // than reading them from inside the worker closure: the knobs are
        // thread-local to the *injecting* test, and a worker runs on its own
        // thread where they are always at their defaults.
        #[cfg(test)]
        let held_worker = HOLD_WORKER_BEFORE_READY.with(std::cell::Cell::get);
        #[cfg(test)]
        let hold_ignores_shutdown_ms = HOLD_WORKER_IGNORES_SHUTDOWN_MS.with(std::cell::Cell::get);
        #[cfg(test)]
        let force_ready_race = FORCE_READY_RACE_AT_DEADLINE.with(std::cell::Cell::get);
        #[cfg(test)]
        let deaf_generation = DEAF_GENERATION.load(Ordering::SeqCst);
        #[cfg(test)]
        let ready_delay_ms = DELAY_WORKER_BEFORE_READY_MS.with(std::cell::Cell::get);
        for worker_id in 0..requested {
            let shared = Arc::clone(&shared);
            match thread::Builder::new()
                .name(format!("{TASK_THREAD_NAME_PREFIX}-{worker_id}"))
                .spawn(move || {
                    #[cfg(test)]
                    {
                        if ready_delay_ms > 0 {
                            thread::sleep(Duration::from_millis(ready_delay_ms));
                        }
                        if worker_id == held_worker {
                            // Not a panic: a panicking worker unwinds and its
                            // `Arc<Shared>` drops, which is a *tidier* failure
                            // than the one being reproduced. The defect is a
                            // worker that exists, holds its share of the pool,
                            // and never announces -- so park it until the
                            // builder's shutdown flag releases it.
                            //
                            // `HOLD_WORKER_IGNORES_SHUTDOWN_MS` makes it
                            // deaf to that flag for a bounded window first,
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
                    }
                    shared.worker_loop(worker_id);
                }) {
                Ok(handle) => handles.push(handle),
                // A host that will not give us a thread is a host we run
                // serially on. Never a hard failure: this is a performance
                // facility, not a correctness one.
                Err(_) => break,
            }
        }
        let workers = handles.len();
        // Wait for every worker to reach its loop before any dispatch can
        // publish, so a fan-out cannot be published into a pool that has not
        // started watching the epoch yet.
        //
        // Spin briefly, then *yield*, then give up loudly. All three matter,
        // and the first two are what this loop was missing:
        //
        // * A pure `spin_loop` waits on threads that need a core to make
        //   progress. Where the builder and its workers contend for the same
        //   CPU, the spinner starves the very workers it is waiting for --
        //   livelock, not slow progress -- and it is most reachable exactly
        //   where cores are scarcest: an emulated target (`qemu-user` runs
        //   every guest thread as a host thread and gives the spinner no
        //   reason to yield), a cpuset-confined process, or a test harness
        //   building several pools at once on a loaded box.
        // * A worker that dies or wedges before announcing makes the condition
        //   permanently unsatisfiable, and spinning on it burns every core it
        //   holds *forever*. That is strictly worse than deadlocking quietly,
        //   because full occupancy is indistinguishable from work: an aarch64
        //   suite hung this way for 5h40m at ~18 cores and was noticed only by
        //   someone reading `/proc`.
        //
        // The deadline is a liveness backstop, not a performance bound; see
        // [`POOL_READY_TIMEOUT`].
        let ready_since = Instant::now();
        let mut spins = 0u32;
        while shared.ready.load(Ordering::Acquire) < workers {
            spins = spins.wrapping_add(1);
            if spins < SPIN_LOOP_BUDGET {
                std::hint::spin_loop();
            } else {
                ready_yield();
                // Check the clock on *every* yield, not on `CLOCK_CHECK_STRIDE`.
                // The stride is right for the spin phase, where an iteration
                // costs nanoseconds and `Instant::now()` would dominate; it is
                // wrong here, because a yield under contention costs
                // microseconds to milliseconds and a stride of N multiplies the
                // deadline's granularity by N yields of an already-starved
                // thread. #1933 made exactly this correction to `decode_spmd`'s
                // barrier after a build 3x over its deadline completed as if
                // healthy.
                if ready_since.elapsed() >= pool_ready_timeout() {
                    #[cfg(test)]
                    if force_ready_race {
                        shared.ready.store(workers, Ordering::Release);
                    }
                    let ready = shared.ready.load(Ordering::Acquire);
                    // The loop condition and this load are two separate reads,
                    // so a worker can announce between them. Accept that pool
                    // rather than tearing down a healthy one and reporting the
                    // self-contradictory "N of N announced ... never became
                    // ready".
                    if ready >= workers {
                        break;
                    }
                    Self::abandon_unready_workers(&shared, handles);
                    panic!(
                        "CPU task-runtime pool never became ready: {ready} of {workers} \
                         workers announced within {:?}. A worker most likely died or \
                         wedged before entering its loop; failing here rather than \
                         spinning on a condition that can no longer be satisfied.",
                        pool_ready_timeout()
                    );
                }
            }
        }
        Self {
            shared,
            handles: Mutex::new(handles),
            workers,
        }
    }

    /// Release the workers of a pool that never became ready, then hand their
    /// handles back to be dropped.
    ///
    /// The barrier above is about to panic, and `TaskPool::new` has not
    /// returned, so there is no `Drop` to run: without this, every worker that
    /// *did* start keeps spinning its adaptive window forever and the failure
    /// path leaves behind exactly the burning-core condition the deadline
    /// exists to end.
    ///
    /// Uses the same publish-then-wake sequence as [`Self::shutdown`], because
    /// the stop flag alone is not enough -- a worker already parked on the
    /// futex never re-reads it and would linger for the life of the process.
    ///
    /// Deliberately **not** joined. A build that failed this way has at least
    /// one worker that never reached its loop; blocking on it here would
    /// reintroduce the unbounded wait this whole path exists to remove. Woken
    /// threads observe `shutdown` and exit on their own, and the `Arc<Shared>`
    /// keeps what they touch alive until the last of them is gone.
    fn abandon_unready_workers(shared: &Arc<Shared>, handles: Vec<JoinHandle<()>>) {
        shared.shutdown.store(true, Ordering::SeqCst);
        shared.epoch.0.fetch_add(1, Ordering::Release);
        atomic_wait::wake_all(&shared.epoch.0);
        drop(handles);
    }

    /// Threads a fan-out can run across, including the dispatching thread.
    pub fn width(&self) -> usize {
        self.workers + 1
    }

    /// Snapshot of this pool's counters.
    pub fn counters(&self) -> PoolCounters {
        self.shared.counters.snapshot()
    }

    /// Run `body(0..total)` across the pool and block until every index has run.
    ///
    /// Returns `false` without running anything when the pool cannot serve the
    /// fan-out — no workers, shut down, or every slot busy — so the caller can
    /// run it serially. Never partially runs a declined fan-out.
    ///
    /// # Panics
    ///
    /// Re-raises on the calling thread if any task body panicked, after every
    /// task has finished and the slot has been returned. Unlike
    /// [`crate::decode_spmd`], a panic does **not** poison the pool: the slot is
    /// recycled and later fan-outs are unaffected.
    #[must_use]
    pub fn dispatch<F>(&self, total: usize, body: &F) -> bool
    where
        F: Fn(usize) + Sync,
    {
        if total == 0 {
            return true;
        }
        if self.workers == 0 || self.shared.shutdown.load(Ordering::Acquire) {
            return false;
        }
        unsafe fn call<F>(data: *const (), task: usize)
        where
            F: Fn(usize) + Sync,
        {
            // SAFETY: `data` came from a live `&F` whose frame outlives every
            // claim; see the module docs.
            let body = unsafe { &*data.cast::<F>() };
            body(task);
        }

        // Spread concurrent dispatchers across slots.
        let Some(index) = self.shared.claim_slot(slot_hint()) else {
            self.shared
                .counters
                .slot_exhausted
                .fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let slot = &self.shared.slots[index];
        // SAFETY: the slot was just taken out of the free mask, so this
        // dispatcher exclusively owns it and no participant can hold a claim on
        // it (its cursor is 0 until the release store below).
        unsafe {
            *slot.job.get() = Job {
                data: std::ptr::from_ref(body).cast(),
                call: call::<F>,
            };
        }
        slot.panicked.store(false, Ordering::Relaxed);
        slot.remaining.0.store(total, Ordering::Relaxed);
        // Release: everything above is visible to whoever claims a task below.
        slot.claim.0.store(total as isize, Ordering::Release);
        let bit = 1u32 << index;
        self.shared.active.0.fetch_or(bit, Ordering::Release);
        self.shared.epoch.0.fetch_add(1, Ordering::Release);
        atomic_wait::wake_all(&self.shared.epoch.0);
        self.shared
            .counters
            .dispatches
            .fetch_add(1, Ordering::Relaxed);

        // The dispatcher is a participant, not an idle waiter: it claims tasks
        // from its own slot until they run out, then waits for stragglers.
        let ran = self.shared.run_slot(index);
        self.shared
            .counters
            .tasks_by_dispatcher
            .fetch_add(ran, Ordering::Relaxed);

        let mut spins = 0u32;
        if slot.remaining.0.load(Ordering::Acquire) != 0 {
            self.shared
                .counters
                .straggler_waits
                .fetch_add(1, Ordering::Relaxed);
        }
        while slot.remaining.0.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(DISPATCHER_YIELD_STRIDE) {
                self.shared
                    .counters
                    .straggler_yields
                    .fetch_add(1, Ordering::Relaxed);
                // Only reached under oversubscription, where a claimant is
                // descheduled; yielding lets it finish instead of us spinning
                // against it.
                thread::yield_now();
            }
        }

        self.shared.active.0.fetch_and(!bit, Ordering::Release);
        // Leave the cursor non-claimable for any straggler still scanning, then
        // hand the slot back.
        slot.claim.0.store(0, Ordering::Relaxed);
        let panicked = slot.panicked.load(Ordering::Acquire);
        self.shared.release_slot(index);
        assert!(
            !panicked,
            "a CPU task-runtime body panicked; re-raised on the dispatching \
             thread after every task finished"
        );
        true
    }

    /// Stop the workers and join them. Idempotent, and safe to call while other
    /// threads are dispatching: a fan-out already published is completed by its
    /// dispatcher (which participates), and later ones are declined.
    pub fn shutdown(&self) {
        let handles: Vec<JoinHandle<()>> = {
            let mut guard = match self.handles.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            std::mem::take(&mut *guard)
        };
        if handles.is_empty() {
            return;
        }
        self.shared.shutdown.store(true, Ordering::SeqCst);
        // Bump before waking so a worker that raced into `wait` sees a changed
        // epoch under the futex guard and does not sleep through the shutdown.
        self.shared.epoch.0.fetch_add(1, Ordering::Release);
        atomic_wait::wake_all(&self.shared.epoch.0);
        for handle in handles {
            let _ = handle.join();
        }
        // Only now is it true that no worker can claim a task, which is what
        // makes a later `dispatch` safe to decline rather than serve.
        self.workers_stopped();
    }

    fn workers_stopped(&self) {
        // `workers` is immutable after construction, so "stopped" is expressed
        // through the shutdown flag that `dispatch` already checks. This exists
        // to keep that reasoning in one named place.
        debug_assert!(self.shared.shutdown.load(Ordering::Acquire));
    }
}

impl Drop for TaskPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// A cheap, stable, per-thread slot preference.
fn slot_hint() -> usize {
    thread_local! {
        static HINT: usize = {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            NEXT.fetch_add(1, Ordering::Relaxed)
        };
    }
    HINT.with(|hint| *hint)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spin window's deadline must also be evaluated *inside* the pure
    /// `spin_loop` phase, not only once that phase is exhausted.
    ///
    /// #1868 did two separable things at this site: it removed a stride from
    /// the yield phase (guarded by the test below, #2022) and it *added* this
    /// check to the spin phase. The second half was unguarded — deleting it
    /// outright left the whole crate green at **1723 passed / 0 failed**.
    ///
    /// It matters at the bottom of the window's range, which is where a
    /// mostly-idle process lives: the window halves on every park and floors
    /// at `MIN_SPIN` (20us), while the spin phase runs 4096 `spin_loop`s,
    /// measured at 130us on this host. Without a check inside that phase a
    /// worker told to release its core after 20us cannot even look at the
    /// clock until 130us have passed — a 6.5x overshoot at exactly the idle
    /// floor the window exists to bound, and on a shared box that surplus
    /// lands on a co-tenant.
    ///
    /// The observable is the spin count rather than elapsed time because it is
    /// monotone in the safe direction: a preempted thread crosses the deadline
    /// in fewer spins, never more, so load can only push this towards passing.
    /// It also survives the interpreter — under Miri 4096 iterations take far
    /// longer than 20us, so the correct code exits at the first stride
    /// boundary and the defective code still runs the whole budget.
    #[test]
    fn the_spin_window_deadline_is_also_evaluated_during_the_pure_spin_phase() {
        // Width 1 spawns no threads, so the epoch cannot move under us and the
        // only way out of the loop is the deadline.
        let pool = TaskPool::new(1);
        let last_epoch = pool.shared.epoch.0.load(Ordering::Acquire);
        SLOW_YIELD_US.with(|c| c.set(0));
        SPIN_COUNT.with(|c| c.set(0));

        let outcome = pool.shared.spin_for_dispatch(last_epoch, MIN_SPIN);
        let spins = SPIN_COUNT.with(std::cell::Cell::get);

        assert_eq!(
            outcome,
            SpinOutcome::Expired,
            "no dispatch was published, so the window must have expired"
        );
        // The first spin-phase evaluation is at the first stride boundary, so
        // anything below it did not leave through the deadline at all.
        assert!(
            spins >= u64::from(CLOCK_CHECK_STRIDE),
            "left after {spins} spins, fewer than one {CLOCK_CHECK_STRIDE}-iteration \
             stride: the loop did not exit through the spin-phase deadline"
        );
        assert!(
            spins < u64::from(SPIN_LOOP_BUDGET),
            "ran the whole {SPIN_LOOP_BUDGET}-iteration spin budget against a \
             {MIN_SPIN:?} window, i.e. the deadline was not evaluated during the \
             pure-spin phase at all and could not be honoured until that phase \
             ended -- ~130us on this host against a window of {MIN_SPIN:?}"
        );
    }

    /// `spin_yields` must count exactly the yields the second phase performed,
    /// and nothing from the pure-`spin_loop` phase.
    ///
    /// This counter is the instrument #2075 needs, and #2075 is the question
    /// "does `decode_spmd`'s yield tax (2.61 of 16 cores in kernel time, fixed
    /// by #2072) transfer to this loop?". An instrument answering that must be
    /// trustworthy before any number it produces is quoted, so this asserts
    /// the count *exactly* rather than asserting it is non-zero: a
    /// `> 0` check passes just as happily if the bump were moved into the spin
    /// phase, which would over-report the yield rate by ~4096x and manufacture
    /// the very tax the issue is trying to detect.
    ///
    /// The assertion is an *iff* on whether the yield phase was reached, which
    /// is what makes it load-independent. Reaching that phase is inherently
    /// timing-dependent — it needs 4096 `spin_loop`s (~130us here) to complete
    /// inside the window — so under Miri, or on a saturated runner, the window
    /// expires in the spin phase instead. Rather than skip there (a test that
    /// silently stops testing is worse than no test), the other branch asserts
    /// the complementary property: no yield phase, no yields counted. Exactly
    /// one branch runs on any host and both are real, so this cannot go quiet.
    #[test]
    fn the_spin_phase_yield_counter_counts_exactly_the_yields_the_window_performed() {
        // Width 1 spawns no threads, so nothing can move the epoch or touch
        // these counters concurrently: the only way out is the deadline, and
        // the only writer is this thread.
        let pool = TaskPool::new(1);
        let last_epoch = pool.shared.epoch.0.load(Ordering::Acquire);
        SLOW_YIELD_US.with(|c| c.set(0));
        SPIN_COUNT.with(|c| c.set(0));

        let before = pool.counters().spin_yields;
        // `MAX_SPIN` rather than an arbitrary large window: it is the ceiling a
        // busy steady state converges on, so this measures the production
        // worst case rather than a synthetic one.
        let outcome = pool.shared.spin_for_dispatch(last_epoch, MAX_SPIN);
        let after = pool.counters().spin_yields;
        let spins = SPIN_COUNT.with(std::cell::Cell::get);
        let yields = after - before;

        assert_eq!(
            outcome,
            SpinOutcome::Expired,
            "no dispatch was published, so the window must have expired"
        );

        if spins >= u64::from(SPIN_LOOP_BUDGET) {
            // Yields run for spins in `SPIN_LOOP_BUDGET..=spins`, inclusive at
            // both ends.
            let expected = spins - u64::from(SPIN_LOOP_BUDGET) + 1;
            assert_eq!(
                yields, expected,
                "the window ran {spins} spins, so it yielded on {expected} of them \
                 (every iteration from {SPIN_LOOP_BUDGET} onwards), but the counter \
                 recorded {yields}: it is not counting one yield per yield-phase \
                 iteration, so any yield rate read from it is wrong"
            );
        } else {
            assert_eq!(
                yields, 0,
                "the window expired after {spins} spins, before the \
                 {SPIN_LOOP_BUDGET}-iteration budget was exhausted, so the yield \
                 phase was never reached and no yield was performed -- but the \
                 counter recorded {yields}, meaning it is counting something \
                 other than yields"
            );
        }
    }

    /// The spin window's deadline must be re-read on every yield.
    ///
    /// #1868 corrected this loop to check the clock on each yield in the second
    /// phase while keeping the stride in the pure-`spin_loop` phase, and its
    /// sibling site in `decode_spmd` got a test. This one did not: reverting
    /// the split here left the whole crate green, so the fix was held in place
    /// by nothing but the comment next to it.
    ///
    /// `MAX_SPIN`'s contract is that a process which stops inferencing returns
    /// to ~0% CPU in under a millisecond. A stride-gated check cannot honour
    /// that: `SPIN_LOOP_BUDGET` (4096) is an exact multiple of
    /// `CLOCK_CHECK_STRIDE` (64), so the yield phase begins on a stride
    /// boundary and the next evaluation is 64 yields away — under the
    /// contention that makes a yield expensive, that is the difference between
    /// releasing a core and holding it for most of a second. On a shared box
    /// that cost lands on a co-tenant, which is why this is worth a test rather
    /// than a comment.
    #[test]
    // Miri makes the premise false rather than the assertion wrong. The test
    // needs the spin phase to be short against the window -- 4096
    // `spin_loop`s measure 128us natively against a 200ms window -- but under
    // the interpreter those 4096 iterations and their strided `Instant::now()`
    // calls outlast the window, so the yield phase is entered already expired
    // and exits at yield 0. Both the strided and the every-yield form do that,
    // so the test discriminates nothing there. It says so itself: the first CI
    // run of this test failed under Miri on its own non-vacuity assertion
    // ("left the yield phase after 0 yield(s) ... it is not a pass") rather
    // than passing emptily, which is the behaviour that earned it this
    // attribute instead of a silent green. Same reason as
    // `workers_park_when_idle_and_wake_again` above: a wall-clock policy, not
    // a memory-model one.
    #[cfg_attr(miri, ignore = "spin-vs-window ratio is wall-clock, not emulated")]
    fn the_spin_window_deadline_is_evaluated_on_every_yield_not_on_a_stride() {
        /// Cost of one injected yield: the contended regime.
        const YIELD: Duration = Duration::from_millis(10);
        /// Long enough that the first yield's check does not already satisfy it
        /// — otherwise both forms exit on yield 1 and nothing is measured — and
        /// far longer than the ~128us the 4096-iteration spin phase costs, so
        /// the yield phase is certain to be reached.
        const WINDOW: Duration = Duration::from_millis(200);
        /// One stride. The strided form cannot exit in fewer yields than this.
        const STRIDE: u64 = CLOCK_CHECK_STRIDE as u64;

        // Width 1 spawns no threads, so this `Shared` is driven only by the
        // call below and its epoch cannot move under us.
        let pool = TaskPool::new(1);
        let last_epoch = pool.shared.epoch.0.load(Ordering::Acquire);
        SLOW_YIELD_US.with(|c| c.set(YIELD.as_micros() as u64));
        YIELD_COUNT.with(|c| c.set(0));

        let outcome = pool.shared.spin_for_dispatch(last_epoch, WINDOW);

        let yields = YIELD_COUNT.with(std::cell::Cell::get);
        // Cleared before asserting: a panic below would otherwise leave the
        // knob set for whatever else libtest runs on this thread.
        SLOW_YIELD_US.with(|c| c.set(0));
        YIELD_COUNT.with(|c| c.set(0));

        assert_eq!(
            outcome,
            SpinOutcome::Expired,
            "no dispatch was published, so the window must have expired"
        );
        // Non-vacuity, asserted rather than hoped for: if the deadline had
        // already passed when the yield phase began, both the strided and the
        // every-yield form exit on the first yield and this test discriminates
        // nothing. That is not a pass.
        assert!(
            yields >= 2,
            "inconclusive: left the yield phase after {yields} yield(s), so the \
             {WINDOW:?} window had already elapsed before the second check. \
             This test cannot tell a strided clock read from an unstrided one \
             in that regime — it is not a pass"
        );
        assert!(
            yields < STRIDE,
            "left the yield phase after {yields} yields against a {WINDOW:?} \
             window and {YIELD:?} yields, i.e. the clock was not re-read on \
             every yield: a stride of {STRIDE} would exit no sooner than its \
             next boundary, holding the core long past the window"
        );
    }

    #[test]
    fn every_index_runs_exactly_once() {
        let pool = TaskPool::new(4);
        for total in [1usize, 2, 3, 7, 64, 1000] {
            let seen: Vec<AtomicU32> = (0..total).map(|_| AtomicU32::new(0)).collect();
            let ok = pool.dispatch(total, &|i: usize| {
                seen[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(ok, "pool declined a {total}-task fan-out");
            for (i, count) in seen.iter().enumerate() {
                assert_eq!(count.load(Ordering::Relaxed), 1, "index {i} of {total}");
            }
        }
    }

    /// A dispatcher that outruns its own slot must record the wait.
    ///
    /// `straggler_waits` is what distinguishes "this fan-out was absorbed by
    /// the dispatcher" from "the dispatcher finished its share and then blocked
    /// on someone else's task", and only the second shape can put a scheduler
    /// timeslice into the tail -- which is why the counter exists.
    ///
    /// Which thread ends up holding the last task is genuinely the scheduler's
    /// choice, so a single fan-out cannot be forced into the second shape. The
    /// property under test is therefore the aggregate one: over many fan-outs
    /// wide enough to require every worker, the dispatcher must at some point
    /// have been the one left waiting. Retrying rather than sleeping keeps this
    /// deterministic in the sense that matters -- it cannot pass for the wrong
    /// reason, and it does not encode a timing assumption.
    #[test]
    fn a_dispatcher_waiting_for_a_straggler_records_it() {
        let pool = TaskPool::new(4);
        if pool.width() < 2 {
            return;
        }
        let before = pool.shared.counters.snapshot().straggler_waits;
        let wanted = pool.width();
        for _ in 0..200 {
            let arrived = AtomicUsize::new(0);
            assert!(
                pool.dispatch(wanted, &|_| rendezvous(&arrived, wanted)),
                "pool declined a width-sized fan-out"
            );
            if pool.shared.counters.snapshot().straggler_waits > before {
                return;
            }
        }
        panic!(
            "200 fan-outs that every worker had to join, and the dispatcher was \
             never recorded as waiting for one of them"
        );
    }

    #[test]
    fn a_zero_task_fanout_is_a_no_op() {
        let pool = TaskPool::new(4);
        assert!(pool.dispatch(0, &|_| unreachable!()));
    }

    #[test]
    fn a_single_thread_pool_declines() {
        let pool = TaskPool::new(1);
        assert_eq!(pool.width(), 1);
        assert!(!pool.dispatch(8, &|_| unreachable!()));
    }

    /// Blocks until `arrived` reaches `wanted` participants, or the deadline
    /// passes.
    ///
    /// This is how a scheduler test asserts "more than one thread ran this"
    /// without asserting on elapsed time. A dispatcher drains its own slot
    /// greedily -- correctly so, since waking a worker for four microseconds of
    /// work is a loss -- which means a fan-out of trivial tasks legitimately
    /// runs entirely on the calling thread. Making each task *wait* for company
    /// removes the race instead of papering over it with a sleep: whoever runs
    /// first parks in the body, and the next participant to claim a task
    /// releases it.
    fn rendezvous(arrived: &AtomicUsize, wanted: usize) {
        arrived.fetch_add(1, Ordering::AcqRel);
        let deadline = Instant::now() + Duration::from_secs(10);
        while arrived.load(Ordering::Acquire) < wanted && Instant::now() < deadline {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn workers_actually_participate() {
        // The point of the pool. Asserting on *thread identity* rather than on
        // elapsed time keeps this meaningful on a contended runner.
        let pool = TaskPool::new(4);
        let arrived = AtomicUsize::new(0);
        let threads = Mutex::new(std::collections::HashSet::new());
        let ok = pool.dispatch(512, &|_| {
            rendezvous(&arrived, 2);
            threads.lock().unwrap().insert(thread::current().id());
        });
        assert!(ok);
        assert!(
            threads.lock().unwrap().len() > 1,
            "every task ran on the dispatching thread; the pool did no work"
        );
    }

    #[test]
    fn concurrent_dispatchers_all_make_progress() {
        // decode_spmd's single job slot forces a second concurrent dispatcher to
        // run its whole fan-out inline. This pool must not.
        let pool = Arc::new(TaskPool::new(8));
        let before = pool.counters();
        let dispatchers = 4u64;
        let per = 256usize;
        thread::scope(|scope| {
            for _ in 0..dispatchers {
                let pool = Arc::clone(&pool);
                scope.spawn(move || {
                    for _ in 0..20 {
                        let seen: Vec<AtomicU32> = (0..per).map(|_| AtomicU32::new(0)).collect();
                        assert!(pool.dispatch(per, &|i: usize| {
                            seen[i].fetch_add(1, Ordering::Relaxed);
                        }));
                        assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
                    }
                });
            }
        });
        let after = pool.counters();
        assert_eq!(after.dispatches - before.dispatches, dispatchers * 20);
        assert_eq!(
            after.tasks - before.tasks,
            dispatchers * 20 * per as u64,
            "tasks run does not match tasks published"
        );
        // Deliberately no assertion that a *worker* ran some of this. A
        // dispatcher drains its own slot greedily, so "at least one task landed
        // off-thread" is a scheduling outcome, not an invariant, and asserting
        // it here would be the timing assertion `workers_actually_participate`
        // exists to avoid. What this test owns is that concurrent dispatchers
        // do not corrupt each other: every task runs exactly once and the
        // totals add up, both checked above.
    }

    #[test]
    fn slot_exhaustion_declines_instead_of_blocking() {
        // More concurrent dispatchers than slots must degrade to "run it
        // yourself", never to a hang and never to two dispatchers sharing one
        // slot's cursor.
        let pool = Arc::new(TaskPool::new(4));
        let hold = Arc::new(AtomicBool::new(true));
        let declined = Arc::new(AtomicUsize::new(0));
        // Count how many holder dispatches have actually entered their body, so
        // the probe below runs only once every slot is provably occupied.
        // Sleeping a fixed 50 ms instead assumes the OS scheduled all
        // `SLOT_COUNT` holder threads inside that window -- which fails on a
        // contended box (e.g. the rest of the test suite running in parallel),
        // leaving slots free, the probe succeeding, and the assertion flaking.
        // Constructing the precondition removes the race rather than papering
        // it over.
        let held = Arc::new(AtomicUsize::new(0));
        thread::scope(|scope| {
            for _ in 0..SLOT_COUNT {
                let pool = Arc::clone(&pool);
                let hold = Arc::clone(&hold);
                let held = Arc::clone(&held);
                scope.spawn(move || {
                    // One increment per holder even though the fan-out has two
                    // tasks and a pool worker may run the second.
                    let counted = AtomicBool::new(false);
                    let _ = pool.dispatch(2, &|_| {
                        if !counted.swap(true, Ordering::Relaxed) {
                            held.fetch_add(1, Ordering::Release);
                        }
                        while hold.load(Ordering::Relaxed) {
                            std::hint::spin_loop();
                        }
                    });
                });
            }
            // Wait until every holder is provably inside its body -- i.e. every
            // slot is claimed -- then prove the next dispatcher is declined
            // rather than blocked. A generous deadline keeps a starved runner
            // from hanging the suite forever.
            let deadline = Instant::now() + Duration::from_secs(10);
            while held.load(Ordering::Acquire) < SLOT_COUNT && Instant::now() < deadline {
                std::hint::spin_loop();
            }
            assert_eq!(
                held.load(Ordering::Acquire),
                SLOT_COUNT,
                "not every holder took a slot within the deadline"
            );
            for _ in 0..4 {
                if !pool.dispatch(2, &|_| {}) {
                    declined.fetch_add(1, Ordering::Relaxed);
                }
            }
            hold.store(false, Ordering::Relaxed);
        });
        assert!(
            declined.load(Ordering::Relaxed) > 0 || pool.counters().slot_exhausted > 0,
            "expected at least one declined fan-out once all {SLOT_COUNT} slots were held"
        );
    }

    #[test]
    fn a_panicking_body_is_re_raised_and_does_not_poison_the_pool() {
        let pool = TaskPool::new(4);
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = pool.dispatch(64, &|i: usize| {
                assert!(i != 17, "task 17 says no");
            });
        }));
        assert!(result.is_err(), "the dispatcher did not re-raise");
        assert_eq!(pool.counters().panics, 1);
        // The pool is still usable — the whole point of not poisoning.
        let seen: Vec<AtomicU32> = (0..32).map(|_| AtomicU32::new(0)).collect();
        assert!(pool.dispatch(32, &|i: usize| {
            seen[i].fetch_add(1, Ordering::Relaxed);
        }));
        assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn every_task_completes_even_when_many_panic() {
        // The completion counter must be decremented on the unwind path too, or
        // the dispatcher hangs forever.
        let pool = TaskPool::new(4);
        let ran = AtomicUsize::new(0);
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = pool.dispatch(128, &|i: usize| {
                ran.fetch_add(1, Ordering::Relaxed);
                assert!(!i.is_multiple_of(3), "every third task");
            });
        }));
        assert!(result.is_err());
        assert_eq!(ran.load(Ordering::Relaxed), 128);
    }

    #[test]
    fn a_nested_fanout_sees_that_it_is_inside_a_task() {
        let pool = TaskPool::new(4);
        let nested_ok = AtomicBool::new(true);
        assert!(pool.dispatch(32, &|_| {
            if !in_task() {
                nested_ok.store(false, Ordering::Relaxed);
            }
        }));
        assert!(nested_ok.load(Ordering::Relaxed));
        assert!(!in_task(), "the guard leaked out of the dispatch");
    }

    #[test]
    fn shutdown_is_idempotent_and_declines_afterwards() {
        let pool = TaskPool::new(4);
        assert!(pool.dispatch(16, &|_| {}));
        pool.shutdown();
        pool.shutdown();
        assert!(
            !pool.dispatch(16, &|_| {}),
            "a shut-down pool must decline, not hang"
        );
    }

    #[test]
    // Miri's clock is emulated, and the adaptive window is defined in wall time:
    // a spinning worker advances the emulated clock only a sliver per iteration,
    // so `MAX_SPIN` is effectively unreachable and the park never happens. The
    // rest of this module runs under Miri, which is where the value is -- this
    // one test asserts a timing behaviour rather than a memory-model one.
    #[cfg_attr(miri, ignore = "adaptive spin window is wall-clock, not emulated")]
    fn workers_park_when_idle_and_wake_again() {
        // The idle-CPU bound: after the adaptive window decays, workers must
        // reach the futex, and a later dispatch must still be served.
        let pool = TaskPool::new(4);
        assert!(pool.dispatch(64, &|_| {}));
        // Poll rather than sleep a fixed span. The window is `MAX_SPIN` at
        // worst, but a loaded runner can delay a worker's *observation* of it
        // for far longer than the window itself, and a fixed sleep turns that
        // into a flake.
        let deadline = Instant::now() + Duration::from_secs(10);
        while pool.counters().parks == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        let parked = pool.counters().parks;
        assert!(parked > 0, "no worker parked while idle");
        let seen: Vec<AtomicU32> = (0..64).map(|_| AtomicU32::new(0)).collect();
        assert!(pool.dispatch(64, &|i: usize| {
            seen[i].fetch_add(1, Ordering::Relaxed);
        }));
        assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn back_to_back_dispatches_are_caught_by_the_spin_window() {
        // The §26.1 regime: many regions microseconds apart. Workers should
        // catch them by spinning, not by a futex round trip per region.
        let pool = TaskPool::new(4);
        for _ in 0..2000 {
            assert!(pool.dispatch(16, &|_| {}));
        }
        let counters = pool.counters();
        assert_eq!(counters.dispatches, 2000);
        assert!(
            counters.parks < counters.dispatches,
            "workers parked on {} of {} back-to-back dispatches",
            counters.parks,
            counters.dispatches
        );
    }

    #[test]
    fn the_claim_cursor_does_not_drift_across_dispatches() {
        // Losing claimants drive the cursor negative; publishing must reset it
        // absolutely so the drift cannot accumulate into a slot that can never
        // hand out work again.
        let pool = TaskPool::new(8);
        for round in 0..200 {
            let seen: Vec<AtomicU32> = (0..8).map(|_| AtomicU32::new(0)).collect();
            assert!(pool.dispatch(8, &|i: usize| {
                seen[i].fetch_add(1, Ordering::Relaxed);
            }));
            assert!(
                seen.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                "round {round} lost or duplicated a task"
            );
        }
    }
}

/// The readiness backstop: a pool that cannot become ready must fail loudly
/// rather than hold the machine at full occupancy indefinitely.
///
/// Separate module so the fault-injection knobs are reset by one guard type
/// rather than by hand in each test. Forgetting a reset leaks a fault into
/// every later pool this thread builds, which libtest reuses.
#[cfg(test)]
mod ready_backstop_tests {
    use super::*;
    use std::sync::MutexGuard;

    /// Sets the readiness knobs for the duration of a test and clears them
    /// afterwards, including on the unwinding path the fault tests take.
    ///
    /// Also holds [`INJECTION`] for exactly that window. The knobs themselves
    /// are thread-locals, but two things these tests depend on are not, so the
    /// guard is bound to the injection rather than left to each call site to
    /// remember.
    struct Injected {
        _serialise: MutexGuard<'static, ()>,
        generation: usize,
    }

    /// Serialises the readiness-injection tests against each other.
    ///
    /// Two process-global resources make them unsafe to interleave. First,
    /// `std::panic::{take_hook, set_hook}` is process-wide, so a concurrent
    /// test restores the default hook inside another's silenced window and the
    /// expected panic escapes into the log attributed to the wrong test.
    /// Second, an injected deadline is measured in milliseconds of wall clock:
    /// a second test building its own pool at the same time competes for the
    /// same cores, so a healthy worker can miss a 100 ms deadline because of
    /// the *other* test's scheduling. That failure is indistinguishable from
    /// the defect these tests exist to catch, which makes it the worst kind --
    /// the cheapest way to make it stop is to raise the deadline until the
    /// test can no longer fail.
    static INJECTION: Mutex<()> = Mutex::new(());

    impl Injected {
        fn timeout_ms(ms: u64) -> Self {
            let serialise = INJECTION
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            POOL_READY_TIMEOUT_MS.with(|cell| cell.set(ms));
            let generation = DEAF_GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
            Self {
                _serialise: serialise,
                generation,
            }
        }

        fn hold_worker(self, index: usize) -> Self {
            HOLD_WORKER_BEFORE_READY.with(|cell| cell.set(index));
            self
        }

        fn hold_ignores_shutdown_ms(self, ms: u64) -> Self {
            HOLD_WORKER_IGNORES_SHUTDOWN_MS.with(|cell| cell.set(ms));
            self
        }

        fn force_ready_race(self) -> Self {
            FORCE_READY_RACE_AT_DEADLINE.with(|cell| cell.set(true));
            self
        }

        fn delay_ms(self, ms: u64) -> Self {
            DELAY_WORKER_BEFORE_READY_MS.with(|cell| cell.set(ms));
            self
        }

        fn slow_yield_us(self, us: u64) -> Self {
            READY_SLOW_YIELD_US.with(|cell| cell.set(us));
            self
        }

        fn yields(&self) -> u64 {
            READY_YIELDS.with(std::cell::Cell::get)
        }

        fn reset_yields(self) -> Self {
            READY_YIELDS.with(|cell| cell.set(0));
            self
        }
    }

    impl Drop for Injected {
        fn drop(&mut self) {
            POOL_READY_TIMEOUT_MS.with(|cell| cell.set(0));
            HOLD_WORKER_BEFORE_READY.with(|cell| cell.set(usize::MAX));
            HOLD_WORKER_IGNORES_SHUTDOWN_MS.with(|cell| cell.set(0));
            FORCE_READY_RACE_AT_DEADLINE.with(|cell| cell.set(false));
            DELAY_WORKER_BEFORE_READY_MS.with(|cell| cell.set(0));
            READY_SLOW_YIELD_US.with(|cell| cell.set(0));
            READY_YIELDS.with(|cell| cell.set(0));
        }
    }

    /// Build a pool with the panic hook silenced, reporting the message.
    ///
    /// Returns `Ok(())` rather than the pool: every caller here is asserting a
    /// failure, and a successfully built pool is dropped (and so shut down and
    /// joined) before the assertion fires, rather than being leaked into the
    /// panic message.
    fn build_expecting_failure(width: usize) -> Result<(), String> {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = catch_unwind(AssertUnwindSafe(|| TaskPool::new(width)));
        std::panic::set_hook(previous);
        outcome.map(drop).map_err(|payload| {
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string())
        })
    }

    /// A worker that never announces must fail the build, not hang it.
    ///
    /// This is the whole point of the change: before it, the constructor's
    /// `while ready < workers { spin_loop() }` had no yield, no bound and no
    /// exit, so one wedged worker held the builder's core at 100% for as long
    /// as the process lived -- indistinguishable from work, which is why an
    /// aarch64 suite could sit in it for hours.
    #[test]
    fn a_worker_that_never_announces_fails_the_build_instead_of_hanging() {
        let injected = Injected::timeout_ms(PAIRED_DEADLINE_MS).hold_worker(0);
        let started = Instant::now();
        let message = build_expecting_failure(4).expect_err(
            "a pool with a worker that never announces was reported as successfully built",
        );
        let elapsed = started.elapsed();
        assert!(
            message.contains("never became ready"),
            "the build failed for some other reason than the readiness backstop: {message}"
        );
        // The count in the message is what tells a reader this was a wedged
        // worker rather than, say, a spawn failure. `4` means dispatcher plus
        // three workers, one of which is held.
        //
        // Asserted as "fewer than all of 3" rather than the exact "2 of 3":
        // under emulation or contention a healthy worker can also still be
        // short of announcing when the deadline fires, and a test that reds on
        // that is reporting the host, not the backstop. The claim that
        // separates a wedge from a spawn failure is that the pool had its
        // workers and they did not all announce.
        let announced = message
            .split(" of 3 workers announced")
            .next()
            .and_then(|head| head.rsplit(' ').next())
            .and_then(|count| count.parse::<usize>().ok());
        match announced {
            Some(count) => assert!(
                count < 3,
                "the backstop fired reporting all 3 workers announced, which is \
                 not a wedge at all: {message}"
            ),
            None => panic!(
                "the diagnostic does not say how far the pool got, which is the \
                 only thing that distinguishes this from a spawn failure: {message}"
            ),
        }
        assert!(
            elapsed < Duration::from_secs(30),
            "the build took {elapsed:?} against a {PAIRED_DEADLINE_MS}ms \
             deadline, so it is not the deadline that ended it"
        );
        drop(injected);
    }

    /// The deadline the wedge test and its negative control share.
    ///
    /// One constant rather than two literals: the control's entire claim is
    /// that *the same* deadline builds a healthy pool, so if the two numbers
    /// can drift apart the control silently stops controlling for anything.
    /// Sized ~1000x a healthy announce (a `fetch_add` per worker) so cross-test
    /// core contention cannot starve a healthy build past it, while staying
    /// short enough that the wedge test does not pad the suite.
    const PAIRED_DEADLINE_MS: u64 = 1_000;

    /// ...and the wait it does must actually be bounded by the deadline, not
    /// merely by the fault clearing itself.
    ///
    /// The negative control for the test above: without the hold, the same
    /// deadline must build a pool. A backstop that fired on every build would
    /// pass the test above for the wrong reason.
    #[test]
    fn the_same_deadline_builds_a_healthy_pool() {
        let injected = Injected::timeout_ms(PAIRED_DEADLINE_MS);
        let pool = TaskPool::new(4);
        assert_eq!(pool.width(), 4);
        assert!(pool.dispatch(16, &|_| {}));
        drop(injected);
    }

    /// The abandoned workers must not be joined, and a cooperative fault
    /// cannot show that.
    ///
    /// `abandon_unready_workers` drops the handles unjoined on purpose:
    /// joining would block on precisely the worker that is not coming, which
    /// is the unbounded wait this whole change removes. Every other test here
    /// injects a worker that parks on `shared.shutdown` -- the same flag the
    /// abandon path sets -- so it *is* coming, and a `join()` restored to that
    /// path would return promptly and leave the suite green while production
    /// hung. This test makes the held worker deaf to shutdown for 4 s, an
    /// order of magnitude past the deadline, so a restored join shows up as
    /// the constructor taking seconds instead of milliseconds.
    ///
    /// The stubbornness is bounded rather than permanent so the thread is gone
    /// by the end of the test rather than leaking into the rest of the binary.
    /// That costs nothing: 4 s against a 250 ms deadline discriminates by 16x,
    /// and the assertion is on the *builder*, which under the correct code
    /// never waits for that thread at all.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "leaves a deliberately deaf worker alive at exit, which Miri's remaining-threads check reports"
    )]
    fn an_abandoned_worker_that_ignores_shutdown_does_not_delay_the_builder() {
        let injected = Injected::timeout_ms(PAIRED_DEADLINE_MS)
            .hold_worker(0)
            .hold_ignores_shutdown_ms(8_000);
        build_expecting_failure(3).expect_err("the injected hold must fail the build");
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
        let injected = Injected::timeout_ms(50).hold_worker(0).force_ready_race();
        let pool = TaskPool::new(3);
        assert_eq!(
            pool.width(),
            3,
            "a pool whose workers had all announced by the re-read was torn down"
        );
        drop(pool);
        drop(injected);
    }

    /// Slow is not the same as broken: a pool whose workers announce late --
    /// which is what emulation and a loaded host produce -- must still build.
    ///
    /// The yield counter is the anti-vacuity guard. Real workers announce
    /// inside `SPIN_LOOP_BUDGET`, so without an injected delay this test never
    /// reaches the yield-and-check phase at all and would pass identically if
    /// that phase were deleted.
    ///
    /// Not run under Miri, and the guard is the reason. The claim being made
    /// is a *race* between wall-clock delay and spin iterations: 150 ms must
    /// outlast `SPIN_LOOP_BUDGET` spins. Miri's clock is virtual and its
    /// scheduler is not the OS's, so a spin costs no time there, the delay
    /// elapses inside the budget, and the workers announce before the builder
    /// ever yields -- the guard fails on a perfectly healthy build. Weakening
    /// it to accommodate that would delete the only thing keeping this test
    /// from passing with the yield phase removed, so the test opts out of the
    /// one environment where its precondition cannot hold instead.
    ///
    /// The sibling tests need no such opt-out and deliberately keep none: they
    /// inject a *hold*, so the builder exhausts the spin budget no matter what
    /// a spin costs, and reaching the yield path does not depend on the clock.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "races an injected wall-clock delay against the spin budget; Miri's virtual clock cannot represent that"
    )]
    fn a_pool_whose_workers_are_merely_slow_still_builds() {
        let injected = Injected::timeout_ms(30_000).delay_ms(150).reset_yields();
        let pool = TaskPool::new(4);
        assert_eq!(pool.width(), 4);
        assert!(pool.dispatch(16, &|_| {}));
        assert!(
            injected.yields() > 0,
            "the barrier never left its pure-spin phase, so this test says \
             nothing about the yield-and-deadline path it exists to exercise"
        );
        drop(injected);
    }

    /// The deadline must be consulted on every yield, not on a stride.
    ///
    /// #1933's defect, ported here with its regression test: with a stride of
    /// N, a starved builder evaluates the clock once per N yields, and a yield
    /// under real contention costs milliseconds. The injected yield cost makes
    /// that regime deterministic -- 12 ms per yield against a 100 ms deadline
    /// means a stride of 64 would overrun by ~64x before noticing.
    ///
    /// The discriminator is the *yield count*, not elapsed time, and that is
    /// deliberate. A wall-clock bound has to sit above the healthy path
    /// (~9 yields x 12 ms ~= 110 ms, plus whatever the host adds) and below the
    /// strided signature (~64 x 12 ms ~= 770 ms) -- an interval that contention
    /// can close from below, at which point the test either reds on the host or
    /// gets widened past 770 ms and silently stops catching the stride. The
    /// yield count has no such window: the correct code leaves after
    /// `deadline / yield_cost` yields, a stride of N cannot leave before its
    /// first multiple of N, and a slow host only makes each yield cost *more*
    /// wall clock, so the count falls further from the failing value rather
    /// than towards it. It is the property itself rather than a proxy for it.
    #[test]
    fn the_deadline_is_checked_on_every_yield_not_on_a_stride() {
        let injected = Injected::timeout_ms(100)
            .hold_worker(0)
            .slow_yield_us(12_000)
            .reset_yields();
        let started = Instant::now();
        build_expecting_failure(3).expect_err("the injected hold must fail the build");
        let elapsed = started.elapsed();
        let yields = injected.yields();
        assert!(
            yields > 0,
            "no yield happened, so the stride this test is about was never reached"
        );
        assert!(
            yields < u64::from(CLOCK_CHECK_STRIDE),
            "the barrier yielded {yields} times at 12ms per yield against a \
             100ms deadline, so it cannot have consulted the clock on each one; \
             not leaving before yield {CLOCK_CHECK_STRIDE} is the signature of a \
             strided check"
        );
        // Liveness only, deliberately far above both regimes: the claim about
        // *granularity* is the assertion above, and duplicating it as a tight
        // time bound would just reintroduce the window that bound cannot hold.
        assert!(
            elapsed < Duration::from_secs(30),
            "the barrier never left its 100ms deadline at all ({elapsed:?} \
             across {yields} yields)"
        );
        drop(injected);
    }

    /// The name workers are spawned with must still be recognisable after
    /// Linux truncates it.
    ///
    /// Not arch-gated: the identity checked is string arithmetic, and the drift
    /// it guards against -- someone renaming the spawn site -- happens on
    /// whatever platform the rename is written on, not on the one that reads
    /// `/proc`. A filter that matches nothing does not fail; it returns "no
    /// threads", which is the *passing* answer for the teardown test below.
    #[test]
    fn the_worker_name_survives_proc_truncation() {
        assert!(
            TASK_THREAD_NAME_PREFIX.len() <= COMM_MAX_BYTES,
            "the prefix the survivor scan filters on is itself truncated by \
             comm, so the scan can never match a worker and its count is \
             always the passing zero"
        );
    }

    /// Count the task-runtime workers among a sequence of `comm` reads, or
    /// report why the sequence cannot be counted.
    ///
    /// Takes the reads as an argument rather than performing them, so the blind
    /// cases have deterministic negative controls: `/proc` cannot be made to
    /// fail on demand, and a fail-closed policy exercised only by the happy
    /// path is a policy nobody has tested.
    #[cfg(target_os = "linux")]
    fn tally_worker_comms(
        reads: impl IntoIterator<Item = std::io::Result<String>>,
    ) -> Result<usize, String> {
        let mut named = 0usize;
        let mut workers = 0usize;
        for read in reads {
            match read {
                Ok(comm) => {
                    named += 1;
                    workers += usize::from(comm.trim().starts_with(TASK_THREAD_NAME_PREFIX));
                }
                // The task exited between listing the directory and reading its
                // `comm`. Benign, and the only benign error here: a thread that
                // has gone is a thread that is not running.
                Err(err) if err.raw_os_error() == Some(libc::ESRCH) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(format!("a task's comm is unreadable: {err}")),
            }
        }
        // A listable directory does not imply readable entries, and it is the
        // entries the filter depends on: a `/proc` that enumerates tasks but
        // refuses every `comm` yields the same `0` an empty process would. The
        // calling thread is always in there and cannot exit under itself, so
        // reading no name at all is impossible unless the scan is blind.
        if named == 0 {
            return Err(
                "no task's comm could be read at all, not even the calling thread's".to_string(),
            );
        }
        Ok(workers)
    }

    /// Live task-runtime worker threads in this process, or a panic saying why
    /// the count is not evidence.
    #[cfg(target_os = "linux")]
    fn live_task_worker_threads() -> usize {
        let entries = std::fs::read_dir("/proc/self/task")
            .unwrap_or_else(|err| panic!("/proc/self/task could not be listed: {err}"));
        match tally_worker_comms(
            entries
                .filter_map(Result::ok)
                .map(|entry| std::fs::read_to_string(entry.path().join("comm"))),
        ) {
            Ok(workers) => workers,
            Err(reason) => panic!(
                "the surviving-worker scan went blind, so its count is not \
                 evidence either way: {reason}"
            ),
        }
    }

    /// A scan that saw nothing must never be reported as "no survivors".
    ///
    /// `0` is the passing value for the teardown test, so an instrument whose
    /// failure value equals its passing value cannot report the failure it
    /// exists to catch.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_scan_that_read_nothing_is_never_reported_as_no_survivors() {
        let empty: [std::io::Result<String>; 0] = [];
        assert!(
            tally_worker_comms(empty).is_err(),
            "a scan that read no task at all reported zero survivors, which is \
             the same answer a clean teardown gives"
        );
        let all_denied = (0..4)
            .map(|_| Err::<String, _>(std::io::Error::from(std::io::ErrorKind::PermissionDenied)));
        assert!(tally_worker_comms(all_denied).is_err());
    }

    /// The real scan must be able to see this process, on any host CI uses.
    #[test]
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore = "Miri has no /proc")]
    fn the_live_scan_can_see_this_process() {
        let entries = std::fs::read_dir("/proc/self/task").expect("list /proc/self/task");
        assert!(
            tally_worker_comms(
                entries
                    .filter_map(Result::ok)
                    .map(|entry| std::fs::read_to_string(entry.path().join("comm"))),
            )
            .is_ok(),
            "the surviving-worker scan cannot read this process's own threads, \
             so the teardown test below would be vacuous"
        );
    }

    /// Selects child mode for [`ready_leak_child`].
    #[cfg(target_os = "linux")]
    const READY_LEAK_CHILD_ENV: &str = "ONNX_GENAI_TEST_TASK_POOL_READY_LEAK_CHILD";

    /// Child half of [`a_failed_build_leaves_no_workers_running`].
    #[test]
    #[ignore = "child process driven by a_failed_build_leaves_no_workers_running"]
    #[cfg(target_os = "linux")]
    fn ready_leak_child() {
        if std::env::var(READY_LEAK_CHILD_ENV).is_err() {
            return;
        }
        // Prove the instrument can count a live worker before its zero is used
        // as evidence. `0` is the *passing* answer below, so a scan that can
        // never see a worker agrees that nothing survived -- the exact shape of
        // a check that cannot fail.
        {
            let probe = TaskPool::new(4);
            let seen = live_task_worker_threads();
            assert!(
                seen >= 3,
                "the scan saw {seen} workers while a 4-wide pool was alive, so \
                 its zero below would not be evidence of anything"
            );
            drop(probe);
        }
        let injected = Injected::timeout_ms(250).hold_worker(0);
        build_expecting_failure(4).expect_err("the injected hold must fail the build");
        drop(injected);
        // Poll rather than sleep a fixed interval: the claim is that they
        // leave, not that they leave within one arbitrary instant. A worker
        // parked on the futex and never woken never leaves, so this reports the
        // steady state either way.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut live = live_task_worker_threads();
        while live > 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
            live = live_task_worker_threads();
        }
        println!("task_threads_after={live}");
    }

    /// A failed build must not leave its surviving workers running.
    ///
    /// The backstop's whole purpose is to stop holding a machine, so panicking
    /// while leaving spinning workers behind would only relocate the defect --
    /// and relocate it into a constructor that never returned, so no `Drop`
    /// will ever clean up after it.
    ///
    /// Runs in a child process because thread identity cannot be established
    /// in-process: every pool in the binary names its workers `nxrt-task-N`,
    /// so a concurrently-running test's pool is indistinguishable from this
    /// one's. A child owns all of its threads, which makes the count exact.
    #[test]
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore = "Miri cannot spawn the child process this needs")]
    fn a_failed_build_leaves_no_workers_running() {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.arg("--exact")
            .arg("task_runtime::pool::ready_backstop_tests::ready_leak_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .arg("--ignored")
            .env(READY_LEAK_CHILD_ENV, "1");
        let output = cmd.output().expect("run readiness-leak child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let live: usize = stdout
            .split_once("task_threads_after=")
            .map(|(_, rest)| {
                rest.chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
            })
            .and_then(|digits| digits.parse().ok())
            .unwrap_or_else(|| {
                panic!(
                    "child never reported its worker-thread count; stdout: {stdout}\n\
                     stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                )
            });
        assert_eq!(
            live, 0,
            "a failed build left {live} task-runtime worker(s) running; panicking \
             while holding threads relocates the defect rather than fixing it"
        );
    }
}
