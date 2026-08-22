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

/// Spin iterations between wall-clock reads, so `Instant::now()` (~20 ns) is
/// amortised rather than paid every iteration.
const CLOCK_CHECK_STRIDE: u32 = 1 << 6;

/// Dispatcher spins between `yield_now` calls while waiting for stragglers.
const DISPATCHER_YIELD_STRIDE: u32 = 1 << 12;

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
}

#[derive(Default)]
struct AtomicCounters {
    dispatches: AtomicU64,
    tasks: AtomicU64,
    tasks_by_dispatcher: AtomicU64,
    slot_exhausted: AtomicU64,
    straggler_waits: AtomicU64,
    straggler_yields: AtomicU64,
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
            let start = Instant::now();
            let mut spins = 0u32;
            let mut caught = false;
            loop {
                if self.shutdown.load(Ordering::Acquire) {
                    return;
                }
                if self.epoch.0.load(Ordering::Acquire) != last_epoch {
                    caught = true;
                    break;
                }
                spins = spins.wrapping_add(1);
                if spins < SPIN_LOOP_BUDGET {
                    std::hint::spin_loop();
                } else {
                    thread::yield_now();
                }
                if spins.is_multiple_of(CLOCK_CHECK_STRIDE) && start.elapsed() >= spin_window {
                    break;
                }
            }
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
        for worker_id in 0..requested {
            let shared = Arc::clone(&shared);
            match thread::Builder::new()
                .name(format!("nxrt-task-{worker_id}"))
                .spawn(move || shared.worker_loop(worker_id))
            {
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
        while shared.ready.load(Ordering::Acquire) < workers {
            std::hint::spin_loop();
        }
        Self {
            shared,
            handles: Mutex::new(handles),
            workers,
        }
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
