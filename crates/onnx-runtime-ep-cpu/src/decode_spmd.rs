//! Persistent SPMD decode pool: one hot worker set joined by a lightweight
//! reusable barrier, replacing the ~141 per-token Rayon fork-join regions.
//!
//! # Why
//!
//! Native M=1 int4 decode issues ~141 `MatMulNBits` projections per token, each
//! run today as a *separate* Rayon parallel region (`par_chunks_mut(..).for_each`
//! -- see [`crate::kernels::matmul_nbits::parallel_output_rows`]). Even with the
//! pool kept hot by [`crate::kernels::matmul_nbits::with_decode_pool_scope`],
//! every projection still pays Rayon's per-region machinery: task publication on
//! the crossbeam deque, work-stealing coordination, `crossbeam-epoch` memory
//! reclamation, and a join latch. Profiling attributes ~27% of the decode step
//! to this fork-join glue, and it is exactly the term that makes >32 cross-socket
//! threads regress.
//!
//! This module keeps a fixed set of worker threads hot-then-parked and drives
//! them with a hand-rolled **sense-reversing broadcast + counting barrier**: a
//! per-node sense counter bump publishes the op, workers observe their node's
//! sense advance, run their pre-assigned output-row shard, and decrement a
//! per-node completion counter; the dispatcher spins on those counters. Workers
//! wait with a KMP_BLOCKTIME-style bounded active spin then park on a futex
//! (`atomic-wait`), so a single `wake` per node releases the whole node in one
//! syscall and idle workers yield the CPU. No per-op allocation, no deque, no
//! epoch GC -- just a handful of atomics per projection. An unwind-only
//! completion guard still decrements the counter if a worker panics, poisons the
//! pool, and makes the dispatcher report an actionable panic instead of hanging.
//!
//! # Two-level, NUMA-aware (mirrors `numa-split`)
//!
//! To use both sockets' memory bandwidth without a toxic flat cross-socket
//! barrier, workers are split into per-node groups (16+16 on a 2-node host),
//! each pinned to its node and reading a node-local first-touched weight shard,
//! exactly like [`crate::decode_numa`]. Row-sharding a GEMV is exactly
//! associative -- each output row is an independent dot product over the whole K
//! dimension -- so concatenating the per-worker row slices reproduces the flat
//! result bit-for-bit, with no cross-row/-node reduction. The only cross-socket
//! traffic per op is the dispatcher reading each node's own completion counter
//! (one line per node), not an N-way shared barrier.
//!
//! # Generality (rule 2)
//!
//! Topology is queried at runtime, never hardcoded. On a single-node host, a
//! non-NUMA machine, or a platform without CPU pinning, the pool degrades to one
//! unpinned worker group -- it still replaces the per-op Rayon barrier with the
//! lightweight one, and stays correct.
//!
//! # Deterministic by default, with opt-in adaptive calibration (rule 5)
//!
//! The pool is **on by default** -- the deterministic, predictable choice.
//! `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` selects the policy:
//!
//! * **unset (the default) or `=1`**: the persistent SPMD pool is always used.
//!   No host probing, no calibration, fully deterministic. The same prompt at
//!   temperature 0 always follows the same floating-point reduction order.
//! * **`=auto`**: opt-in load-adaptive calibration. A runtime heuristic times
//!   the pool and flat paths on the live workload and keeps the faster one
//!   (see [`Calibrator`]). Under co-tenant load the flat path usually wins;
//!   on a quiet host the pool wins. The tradeoff is that the selected path
//!   depends on transient system state, so results are not reproducible across
//!   runs on differently-loaded machines.
//! * **`=0`**: explicit opt-out; the decode path stays on the flat legacy pool.
//!
//! **The path is frozen once committed.** Whether the pool was selected by the
//! default, by `=1`, or by the adaptive calibrator, the path stays fixed for
//! the lifetime of the generation. The flat and pool paths use different
//! floating-point reduction orders (single-threaded vs partitioned parallel),
//! so switching mid-generation changes logits and can produce different tokens
//! under greedy decode.
//!
//! The worker count is
//! [`crate::kernels::matmul_nbits::configured_persistent_decode_threads`] (about
//! half the logical CPUs); a `THREADS=0` opt-out leaves the decode path unchanged.
//!
//! # Precedence when on (default or `=1`) vs the affinity control
//!
//! When the pool is on (the default, or `=1`), the decode strategy precedence is,
//! highest first:
//!
//! 1. **`ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split`** -- the explicit multi-node
//!    split wins when its two-level layout can be built (the mutually-exclusive
//!    selection vs the persistent pool is reported once).
//! 2. **Persistent SPMD pool** (default or `=1`) -- its own per-node pinning applies.
//! 3. **Flat Rayon + auto-`compact`** legacy path -- reached by `=0` (Off). Under
//!    `=auto` (adaptive), an explicit `numa-split` affinity likewise takes
//!    precedence over calibration (the user picked a specific strategy), so the
//!    adaptive calibrator measures the persistent SPMD pool against the flat path
//!    only.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::decode_affinity::{NodeShard, NumaTopology};
use crate::kernels::matmul_nbits::output_chunk_len_for;

/// Environment switch selecting the persistent SPMD decode pool policy:
/// **unset (the default) or `=1`** uses the persistent SPMD pool deterministically
/// (no host probing); `=auto` opts in to load-adaptive calibration (see
/// [`Calibrator`]); `=0` forces the flat legacy path. The pool beats the flat
/// path on a quiet or dedicated host; under heavy co-tenant load the flat path
/// degrades more gracefully, which is why the adaptive mode exists -- but the
/// default prioritises predictability over adaptation.
/// See `.squad/decisions.md` (Voight 2026-07-24; Chu 2026-07-27 opt-in adaptive).
pub const PERSISTENT_POOL_ENV: &str = "ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL";
/// Selects how the persistent decode pool assigns output-column work inside an
/// op. Unset / `fixed` keeps the existing deterministic one-shard-per-worker
/// SPMD split; `steal` decomposes the op into coarser tiles and lets resident
/// workers claim tiles dynamically, so a delayed worker does not strand a whole
/// static shard behind the per-op barrier.
pub const DECODE_SCHEDULE_ENV: &str = "ONNX_GENAI_CPU_DECODE_SCHEDULE";

/// Bounded active-spin window before a worker parks, mirroring the
/// **LLVM/Intel OpenMP runtime `KMP_BLOCKTIME`** design: after a fork/join the
/// worker busy-waits ("blocktime") for the next barrier release, and only sleeps
/// on a futex once genuinely idle for longer than the window. This is the gold
/// standard for exactly this OMP-style fine-grained fork/join HPC workload.
///
/// The window is **load-bearing, not a bad habit**: decode fires ~400 fork/join
/// barriers per token, microseconds apart, so a worker must catch the next
/// barrier while spinning. Parking inside the active path would pay a futex wake
/// (~1-5 us) on every barrier and tank throughput. The window is sized to span
/// the inter-op / inter-token gaps of tight decode (so workers effectively never
/// park mid-generation) yet expire quickly once a generation ends, so idle CPU
/// returns to ~0 between requests -- the spin is bounded to genuinely-active
/// decode, not "unconditional". Tunable via [`decode_blocktime`].
///
/// Default 500 us: comfortably longer than the ~microsecond inter-op gap and the
/// serial dispatcher glue between barriers, and shorter than any human-scale idle
/// gap between requests. Analogous to (though far shorter than) OpenMP's 200 ms
/// default, which targets coarse parallel regions rather than per-token decode.
const DEFAULT_BLOCKTIME: Duration = Duration::from_micros(500);

/// Environment variable naming the worker active-spin window, in microseconds.
const DECODE_BLOCKTIME_ENV: &str = "ONNX_GENAI_CPU_DECODE_BLOCKTIME_US";

/// Pure `spin_loop` iterations at the start of the active window before the
/// worker begins yielding the core to co-tenants between clock checks. A
/// crossbeam-`Backoff`-style ramp: hammer the sense line first to catch the
/// common immediately-ready case with the lowest latency, then relax to
/// `yield_now` so a busy host can schedule other work while we finish out the
/// blocktime window. Sized (~4096 spins, a few microseconds) to cover the
/// typical inter-op dispatcher gap so back-to-back decode barriers are caught by
/// pure spinning and only genuinely idle gaps ramp into yielding then parking.
const SPIN_LOOP_BUDGET: u32 = 1 << 12;

/// Spin iterations between wall-clock checks, so `Instant::now()` (a vDSO read,
/// ~20 ns) is amortised over the hot spin loop rather than read every iteration.
const CLOCK_CHECK_STRIDE: u32 = 1 << 6;

/// KMP_BLOCKTIME analog: the active-spin window a worker holds a core before
/// parking on the futex, read once. `ONNX_GENAI_CPU_DECODE_BLOCKTIME_US` sets it
/// in microseconds; `0` parks as soon as the sense line is not already advanced
/// (maximally polite, higher wake latency). Unset uses [`DEFAULT_BLOCKTIME`].
fn decode_blocktime() -> Duration {
    static V: OnceLock<Duration> = OnceLock::new();
    *V.get_or_init(|| parse_decode_blocktime(std::env::var(DECODE_BLOCKTIME_ENV).ok().as_deref()))
}

/// Parse a [`DECODE_BLOCKTIME_ENV`] value, falling back to
/// [`DEFAULT_BLOCKTIME`] for anything unset or unparseable.
///
/// Split out from [`decode_blocktime`] so the policy is testable without the
/// environment. That is not a stylistic preference: `decode_blocktime` latches
/// into a `OnceLock` on first `worker_wait`, so a test that set the variable and
/// called it twice would compare the first value against itself and pass while
/// measuring nothing -- the same latched-control defect as #1736. Testing the
/// pure function cannot silently degrade that way, and `cargo test` runs a
/// binary's tests on parallel threads where `set_var` racing another thread's
/// `getenv` is a data race Rust 2024 forbids outright.
fn parse_decode_blocktime(raw: Option<&str>) -> Duration {
    raw.and_then(|raw| raw.trim().parse::<u64>().ok())
        .map_or(DEFAULT_BLOCKTIME, Duration::from_micros)
}

/// Spin iterations the (single, never-idle) dispatcher busy-waits on the
/// completion counters before yielding. The dispatcher runs the barrier inline
/// and needs the workers' results the instant they land, so it spins rather than
/// parks; the yield after the budget only lets a descheduled straggler worker get
/// a core under oversubscription.
const DISPATCHER_SPIN_BEFORE_YIELD: u32 = 1 << 12;

/// Default number of dynamic tiles per resident worker in work-stealing mode.
/// One tile per worker preserves the coarse MLAS QNBit shard size that made
/// fixed-SPMD fast, while Deckard's pool can still steal a not-yet-claimed tile
/// from a delayed worker. Finer 2x/3x tiling improved theoretical steal
/// opportunities but split Qwen3 projection shards too narrowly and regressed
/// measured throughput.
const DEFAULT_STEAL_TILES_PER_WORKER: usize = 1;
/// Minimum output columns in one dynamic tile. Keeps MLAS/KAI/hand GEMV calls
/// coarse enough that the extra atomic `fetch_add` scheduling is amortized.
const MIN_STEAL_OUTPUTS_PER_TASK: usize = 32;

/// Cache-line pad so per-node completion counters and per-worker park flags do
/// not false-share (which would reintroduce cross-socket coherency traffic).
#[repr(align(128))]
struct Padded<T>(T);

/// A type-erased decode job: run the shard for the given global worker index.
/// The data pointer is only dereferenced between [`SharedState::publish`] and the
/// matching dispatcher wait, so the borrowed closure always outlives its use.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

/// State shared between the dispatcher (the engine thread running the forward)
/// and the persistent worker threads.
struct SharedState {
    /// Per-node sense counter, bumped once per dispatched op; a node's workers
    /// wait for *their* node's counter to advance. This is a **sense-reversing
    /// barrier** generalised from a 1-bit phase to a monotonic `u32`: each op has
    /// a strictly-increasing sense value, so barrier reuse across the ~400
    /// barriers/token can never race (no ABA -- wrap needs 2^32 barriers while a
    /// worker sleeps through exactly that many, which is impossible). Splitting it
    /// **per node** is the key locality fix: every worker spins on / futex-waits
    /// on its own node's line, so there is no single shared cache line ping-ponging
    /// across the UPI link between sockets.
    node_sense: Vec<Padded<AtomicU32>>,
    /// The current op, published before the sense counters bump and read after the
    /// bump is observed (release/acquire pairing on `node_sense`).
    job: UnsafeCell<Option<Job>>,
    /// Outstanding worker acknowledgements for the current op, one counter per
    /// node so the dispatcher only reads each node's own (mostly node-local)
    /// line instead of an N-way shared barrier.
    node_pending: Vec<Padded<AtomicUsize>>,
    /// The node each global worker index belongs to (drives which pending
    /// counter it decrements and which sense line it waits on).
    worker_node: Vec<usize>,
    /// Count of workers that have entered their loop and are ready to receive
    /// ops. `build` blocks until this reaches `total_workers` so no dispatch can
    /// race a not-yet-started worker (which would miss the op and hang the
    /// barrier).
    ready: AtomicUsize,
    /// Nonzero after a worker panics while running an op (`worker_index + 1`).
    /// A poisoned pool rejects this and every later dispatch instead of hanging
    /// forever waiting for a worker that has unwound.
    poisoned_worker: AtomicUsize,
    /// Set while one thread owns the publish/wait protocol above.
    ///
    /// `job` is a **single slot** and the `node_pending` counters are a single
    /// barrier, so the whole `publish` -> `wait` sequence is one critical
    /// section over the entire worker set. Two threads running it at once would
    /// overwrite each other's job pointer and both sets of workers would run
    /// whichever closure landed last -- producing *silently wrong tensors*
    /// rather than a crash or a hang, since each shard writes a disjoint output
    /// region and the barrier still balances.
    ///
    /// [`SpmdDecodePools::dispatch`] claims this with a compare-exchange and
    /// runs the shards inline when the claim fails, so a second dispatcher
    /// degrades to serial instead of corrupting the first. This also makes a
    /// re-entrant dispatch (a shard closure that itself dispatches) run inline
    /// rather than deadlock against the barrier it is already inside.
    ///
    /// Padded onto its own line, which is not optional. It is written twice per
    /// op by the dispatcher while every worker polls `shutdown` in its wait
    /// loop, so sharing a line with those fields makes each claim invalidate
    /// all 16 workers' copies. Empty 16-worker dispatch, best of 7 x 20k,
    /// three alternating rounds:
    ///
    /// | variant | ns/dispatch |
    /// |---|---|
    /// | ungated (pre-fix) | 908 / 903 / 907 |
    /// | gated, padded | 898 / 881 / 898 |
    /// | gated, unpadded | 1452 / 1497 / 1432 |
    ///
    /// Padded, the gate costs nothing measurable. Unpadded it costs ~60%, which
    /// at ~400 barriers per token is ~230 us/token of pure coherency traffic.
    dispatching: Padded<AtomicBool>,
    shutdown: AtomicBool,
}

// SAFETY: `job` is a raw pointer guarded by the publish/observe protocol on
// `node_sense`; it is only read by workers while the dispatcher blocks in
// `dispatch`, so the pointee outlives every access. All other fields are atomics.
unsafe impl Sync for SharedState {}
unsafe impl Send for SharedState {}

impl SharedState {
    /// Publish `job` for `node_pending[node] = counts[node]` workers, then wake
    /// each node's sleeping workers with a single futex `wake` per node. Must be
    /// paired with [`SharedState::wait`].
    fn publish(&self, job: Job, counts: &[usize]) {
        // Publish the job pointer, then the per-node counts, before the sense
        // bumps make them visible to workers.
        unsafe {
            *self.job.get() = Some(job);
        }
        for (counter, &count) in self.node_pending.iter().zip(counts) {
            counter.0.store(count, Ordering::Release);
        }
        // Advance each node's sense (Release) so the job + count writes above are
        // visible to any worker that observes the new sense (Acquire), then wake
        // that node's parked workers. `wake` is a single futex_wake(i32::MAX) per
        // node -- one syscall releases the whole node, not an O(workers) unpark
        // fan-out. A worker that raced into parking re-checks the sense under the
        // futex guard, so this ordering (bump-then-wake) loses no wakeup: if the
        // worker armed `wait(last_seen)` before this bump it wakes here; if it
        // armed after, `wait` sees the advanced sense and returns without sleeping.
        for (node, sense) in self.node_sense.iter().enumerate() {
            if counts.get(node).copied().unwrap_or(0) == 0 {
                continue;
            }
            sense.0.fetch_add(1, Ordering::Release);
            atomic_wait::wake_all(&sense.0);
        }
    }

    /// Spin-wait until every node's workers have finished the published op.
    ///
    /// There is exactly one dispatcher at a time -- enforced by
    /// [`SharedState::dispatching`], which callers claim through
    /// [`DispatchClaim::try_claim`] before publishing. It is never idle and
    /// needs the results the instant they land, so it spins (with a yield
    /// backstop for stragglers under oversubscription) rather than parking.
    fn wait(&self) {
        let mut spins = 0u32;
        loop {
            let done = self
                .node_pending
                .iter()
                .all(|counter| counter.0.load(Ordering::Acquire) == 0);
            if done {
                return;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins >= DISPATCHER_SPIN_BEFORE_YIELD {
                thread::yield_now();
            }
        }
    }

    /// Wait for the next op published to `node` (its sense advancing past
    /// `last_seen`) or for shutdown, then return the observed sense value; the
    /// caller re-checks `shutdown`.
    ///
    /// KMP_BLOCKTIME-style bounded active spin then futex park. Phase 1 busy-waits
    /// (crossbeam-`Backoff`-style `spin_loop` ramp into `yield_now`) for up to
    /// `blocktime`, catching the common immediately-ready barrier release with the
    /// lowest latency. Phase 2 parks on the futex once idle past the window.
    ///
    /// No lost wakeup: `atomic_wait::wait(sense, last_seen)` sleeps only while the
    /// sense still equals `last_seen`; the kernel re-checks it under the futex
    /// bucket lock, so a [`SharedState::publish`] bump that races the arm makes
    /// `wait` return immediately instead of sleeping. The Acquire load pairs with
    /// publish's Release bump to make the job pointer and pending counts visible.
    fn worker_wait(&self, node: usize, last_seen: u32, blocktime: Duration) -> u32 {
        let sense = &self.node_sense[node].0;
        // Phase 1: bounded active spin (blocktime), spin_loop ramping into yield.
        let mut spins = 0u32;
        let start = Instant::now();
        loop {
            let current = sense.load(Ordering::Acquire);
            if current != last_seen || self.shutdown.load(Ordering::Acquire) {
                return current;
            }
            spins = spins.wrapping_add(1);
            if spins < SPIN_LOOP_BUDGET {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
                if spins.is_multiple_of(CLOCK_CHECK_STRIDE) && start.elapsed() >= blocktime {
                    break;
                }
            }
        }
        // Phase 2: park on the futex until the sense advances (or shutdown wakes
        // us via its own sense bump). Re-check under the guard for no lost wakeup.
        loop {
            let current = sense.load(Ordering::Acquire);
            if current != last_seen || self.shutdown.load(Ordering::Acquire) {
                return current;
            }
            atomic_wait::wait(sense, last_seen);
        }
    }

    fn panic_if_poisoned(&self) {
        let poisoned = self.poisoned_worker.load(Ordering::Acquire);
        if poisoned != 0 {
            let worker = poisoned - 1;
            panic!(
                "persistent SPMD decode worker {worker} panicked while executing a decode op; \
                 the pool is poisoned and cannot continue. Disable \
                 ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL or restart the process"
            );
        }
    }
}

/// Exclusive ownership of the pool's single job slot and barrier.
///
/// Claiming and releasing live together here so the two halves cannot drift
/// apart: the only way to construct one is [`DispatchClaim::try_claim`], and
/// `Drop` releases on every exit including an unwind. A leaked claim would send
/// every later dispatch inline forever, so the panic path matters.
struct DispatchClaim<'a> {
    shared: &'a SharedState,
}

impl<'a> DispatchClaim<'a> {
    /// Take exclusive ownership, or `None` if another thread already holds it.
    ///
    /// Acquire on success pairs with the Release in `Drop`, so the new owner
    /// observes the previous op's completion before overwriting the job slot.
    fn try_claim(shared: &'a SharedState) -> Option<Self> {
        shared
            .dispatching
            .0
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| Self { shared })
    }
}

impl Drop for DispatchClaim<'_> {
    fn drop(&mut self) {
        self.shared.dispatching.0.store(false, Ordering::Release);
    }
}

/// A persistent SPMD decode pool: hot worker threads plus the shared barrier
/// state that drives them.
pub struct SpmdDecodePools {
    shared: Option<Arc<SharedState>>,
    /// Owned join handles, held behind a `Mutex` so [`SpmdDecodePools::shutdown`]
    /// can join the workers through a shared `&self` (the pool is reached as a
    /// `&'static` through [`pools`]). Only touched at teardown, never on the hot
    /// path. Drained on the first shutdown so the operation is idempotent.
    join_handles: Mutex<Vec<JoinHandle<()>>>,
    /// Deckard's Eigen-style persistent work-stealing pool. In `Steal` mode this
    /// is the only resident worker set: decode ops publish coarse MLAS/KAI tiles
    /// directly to this pool so delayed workers do not strand a static SPMD
    /// shard behind a hard barrier.
    #[cfg(feature = "mlas")]
    work_stealing_pool: Option<mlas_sys::WorkStealingThreadPool>,
    /// Compute shards assigned to each node, node-major, matching global worker
    /// index order (shards `0..counts[0]` are node 0, and so on).
    ///
    /// This is the *partition* width. It exceeds the number of spawned worker
    /// threads by one on the last node when [`Self::dispatcher_shard`] is set,
    /// because the dispatcher computes that shard itself.
    node_worker_counts: Vec<usize>,
    /// Worker *threads* per node, node-major. Equal to [`Self::node_worker_counts`]
    /// except on the last node when the dispatcher owns a shard.
    ///
    /// This is what [`SharedState::publish`] counts down: only spawned threads
    /// decrement `node_pending`, so the dispatcher's shard must not be included
    /// or [`SharedState::wait`] would never observe zero.
    node_thread_counts: Vec<usize>,
    /// The global shard index the dispatcher computes inline, if any.
    ///
    /// Always the last index (`total_workers - 1`), which keeps the spawned
    /// threads on a contiguous `0..total_threads` range so their global indices,
    /// their node assignment and their CPU pinning are all unchanged.
    dispatcher_shard: Option<usize>,
    total_workers: usize,
    schedule: DecodeSchedule,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeSchedule {
    Fixed,
    Steal,
}

fn decode_schedule_from_raw(raw: Option<&str>) -> DecodeSchedule {
    match raw.map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("fixed") => DecodeSchedule::Fixed,
        Some(value) if value.eq_ignore_ascii_case("spmd") => DecodeSchedule::Fixed,
        #[cfg(feature = "mlas")]
        Some(value) if value.eq_ignore_ascii_case("steal") => DecodeSchedule::Steal,
        #[cfg(feature = "mlas")]
        Some(value) if value.eq_ignore_ascii_case("work-stealing") => DecodeSchedule::Steal,
        _ => DecodeSchedule::Fixed,
    }
}

fn decode_schedule() -> DecodeSchedule {
    decode_schedule_from_raw(std::env::var(DECODE_SCHEDULE_ENV).ok().as_deref())
}

fn steal_tiles_per_worker() -> usize {
    static TILES: OnceLock<usize> = OnceLock::new();
    *TILES.get_or_init(|| {
        std::env::var("ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_STEAL_TILES_PER_WORKER)
    })
}

impl SpmdDecodePools {
    /// Build the pool from per-node worker shards. Global worker indices are
    /// laid out node-major (node 0's workers first) so row segments and weight
    /// placement line up with the node assignment.
    ///
    /// `dispatcher_shard` adds one compute shard, owned by the dispatcher rather
    /// than by a spawned thread. See [`SpmdDecodePools::dispatcher_shard`].
    fn build(shards: &[NodeShard], dispatcher_shard: bool) -> Self {
        Self::build_with_schedule(shards, decode_schedule(), dispatcher_shard)
    }

    fn build_with_schedule(
        shards: &[NodeShard],
        schedule: DecodeSchedule,
        dispatcher_shard: bool,
    ) -> Self {
        let node_count = shards.len();
        let mut worker_node = Vec::new();
        let mut node_thread_counts = Vec::with_capacity(node_count);
        // Global (index, pinned cpu) assignment, node-major.
        let mut assignment: Vec<(usize, Option<usize>)> = Vec::new();
        for (node_position, shard) in shards.iter().enumerate() {
            node_thread_counts.push(shard.workers);
            for worker in 0..shard.workers {
                worker_node.push(node_position);
                let cpu = shard.cpus.get(worker % shard.cpus.len().max(1)).copied();
                assignment.push((node_position, cpu));
            }
        }
        let total_threads = assignment.len();
        // The dispatcher's shard is the last global index, so the spawned threads
        // keep the contiguous `0..total_threads` range they already had.
        let dispatcher_shard = (dispatcher_shard && node_count > 0).then_some(total_threads);
        let mut node_worker_counts = node_thread_counts.clone();
        if dispatcher_shard.is_some()
            && let Some(last) = node_worker_counts.last_mut()
        {
            *last += 1;
        }
        let total_workers = total_threads + usize::from(dispatcher_shard.is_some());

        #[cfg(feature = "mlas")]
        if schedule == DecodeSchedule::Steal {
            // The work-stealing pool is its own executor with no inline
            // dispatcher, so there is no reserved CPU to reclaim: it spawns a
            // thread per shard and the dispatcher never computes.
            return Self {
                shared: None,
                join_handles: Mutex::new(Vec::new()),
                work_stealing_pool: Some(
                    mlas_sys::WorkStealingThreadPool::new(total_threads)
                        .expect("spawn persistent work-stealing decode pool"),
                ),
                node_worker_counts: node_thread_counts.clone(),
                node_thread_counts,
                dispatcher_shard: None,
                total_workers: total_threads,
                schedule,
            };
        }

        let shared = Arc::new(SharedState {
            node_sense: (0..node_count).map(|_| Padded(AtomicU32::new(0))).collect(),
            job: UnsafeCell::new(None),
            node_pending: (0..node_count)
                .map(|_| Padded(AtomicUsize::new(0)))
                .collect(),
            worker_node,
            ready: AtomicUsize::new(0),
            poisoned_worker: AtomicUsize::new(0),
            dispatching: Padded(AtomicBool::new(false)),
            shutdown: AtomicBool::new(false),
        });

        let mut handles = Vec::with_capacity(total_threads);
        for (global_index, (node_position, cpu)) in assignment.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("onnx-genai-spmd-n{node_position}-{global_index}"))
                .spawn(move || {
                    if let Some(cpu) = cpu
                        && let Err(message) = crate::decode_affinity::pin_current_thread_to_cpu(cpu)
                    {
                        report_spmd_fallback(&format!(
                            "worker {global_index} could not pin to cpu {cpu}: {message}"
                        ));
                    }
                    worker_loop(shared, global_index);
                })
                .expect("spawn persistent SPMD decode worker");
            handles.push(handle);
        }

        // Block until every worker has entered its loop and is waiting for ops.
        // Without this, a dispatch issued before a worker starts would set the
        // op's pending count for that worker, which would never arrive to
        // decrement it -- hanging the barrier. This counts spawned *threads*,
        // not compute shards: the dispatcher's shard has no thread to wait for.
        while shared.ready.load(Ordering::Acquire) < total_threads {
            std::hint::spin_loop();
        }

        // Snapshot the worker `Thread`s for teardown join only; the hot dispatch
        // path wakes workers via the per-node futex, not per-thread `unpark`.
        Self {
            shared: Some(shared),
            join_handles: Mutex::new(handles),
            #[cfg(feature = "mlas")]
            work_stealing_pool: None,
            node_worker_counts,
            node_thread_counts,
            dispatcher_shard,
            total_workers,
            schedule,
        }
    }

    /// Total decode workers across all node groups.
    pub fn total_workers(&self) -> usize {
        self.total_workers
    }

    /// Number of node groups in the layout.
    pub fn node_count(&self) -> usize {
        self.node_worker_counts.len()
    }

    fn uses_work_stealing(&self) -> bool {
        self.schedule == DecodeSchedule::Steal
    }

    /// Signal every worker to stop and **join** them. Idempotent: the join
    /// handles are drained on the first call, so a second call (e.g. `Drop`
    /// after an explicit [`shutdown_pools`]) is a no-op.
    ///
    /// Reachable through a shared `&self` (the process-wide pool is handed out as
    /// a `&'static` by [`pools`]); the join handles live behind a `Mutex` so this
    /// can consume them without owning the pool. This is a teardown-only call and
    /// must not race a live decode dispatch -- after it returns the workers are
    /// gone and the pool must not be dispatched to again.
    pub fn shutdown(&self) {
        if self.shared.is_none() {
            return;
        }
        let Some(shared) = self.shared.as_ref() else {
            return;
        };
        // Take the handles first so concurrent callers can't double-join; if
        // another thread already drained them, there is nothing to do.
        let handles: Vec<JoinHandle<()>> = {
            let mut guard = self
                .join_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.drain(..).collect()
        };
        if handles.is_empty() {
            return;
        }
        // Publish the stop flag, then bump every node's sense so spinning workers
        // observe the change and re-check `shutdown`, and futex-wake any parked
        // worker so it leaves the park. The bump-then-wake ordering (mirroring the
        // dispatch path) wakes a worker that raced into parking: it either sees
        // the advanced sense under the futex guard or is woken by `wake_all`.
        shared.shutdown.store(true, Ordering::SeqCst);
        for sense in &shared.node_sense {
            sense.0.fetch_add(1, Ordering::Release);
            atomic_wait::wake_all(&sense.0);
        }
        for handle in handles {
            let _ = handle.join();
        }
    }

    /// Run every shard of `job` on this thread, in global worker order.
    ///
    /// The shard closures partition their output into disjoint regions indexed
    /// by global worker index, so running all of them on one thread is exactly
    /// the same computation as broadcasting them -- only serial. This is the
    /// fallback when the pool's single publish/wait slot is already claimed.
    fn dispatch_inline<F>(&self, job: &F)
    where
        F: Fn(usize) + Sync,
    {
        for global_index in 0..self.total_workers {
            job(global_index);
        }
    }

    /// Broadcast `job` to the workers and block until all have finished.
    ///
    /// `job(global_worker_index)` runs the shard owned by that worker. The
    /// dispatcher (this thread) does not compute; it only publishes and waits,
    /// mirroring an external `pool.install` where the caller blocks.
    ///
    /// The pool has one job slot and one barrier, so it can serve one
    /// dispatcher at a time. When another thread already owns them this runs
    /// the shards inline instead of waiting or racing -- see
    /// [`SharedState::dispatching`].
    fn dispatch<F>(&self, job: &F)
    where
        F: Fn(usize) + Sync,
    {
        #[cfg(feature = "mlas")]
        if let Some(pool) = &self.work_stealing_pool {
            pool.parallel_for(0, self.total_workers, 1, |begin, end| {
                for global_index in begin..end {
                    job(global_index);
                }
            });
            return;
        }
        let shared = self
            .shared
            .as_ref()
            .expect("fixed SPMD dispatch requires shared worker state");
        shared.panic_if_poisoned();
        unsafe fn call<F>(data: *const (), global_index: usize)
        where
            F: Fn(usize) + Sync,
        {
            // SAFETY: `data` came from a live `&F`; synchronous dispatch keeps
            // that borrow alive until every worker acknowledges this op.
            let job = unsafe { &*data.cast::<F>() };
            job(global_index);
        }
        // The pool serves one dispatcher at a time; a second one runs the same
        // shards inline rather than racing for the slot.
        let Some(claim) = DispatchClaim::try_claim(shared) else {
            self.dispatch_inline(job);
            return;
        };
        let job_ptr = Job {
            data: std::ptr::from_ref(job).cast(),
            call: call::<F>,
        };
        shared.publish(job_ptr, &self.node_thread_counts);
        // Compute the dispatcher's own shard, when it has one, on the CPU the
        // headroom reservation kept free. `catch_unwind` is load-bearing rather
        // than defensive: the workers are already running against a `Job` that
        // borrows `job` off *this* stack frame, so unwinding straight out of
        // here would free the pointee while they still read it. Catch, complete
        // the barrier, then resume the panic.
        let unwind = self.dispatcher_shard.and_then(|global_index| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(global_index))).err()
        });
        shared.wait();
        drop(claim);
        if let Some(payload) = unwind {
            // A worker panic during the same op stays latched in
            // `poisoned_worker` and is reported by the next dispatch's
            // `panic_if_poisoned`, which runs before it publishes anything. The
            // dispatcher's own payload wins this frame because it is the one
            // carrying the caller's stack; neither is lost, and `wait` above
            // already guaranteed no worker is still running.
            std::panic::resume_unwind(payload);
        }
        shared.panic_if_poisoned();
    }

    /// Split `n` output rows across the node groups proportionally to their
    /// worker counts (contiguous, non-overlapping, last node absorbs the
    /// remainder), matching [`crate::decode_numa`] so weight placement and
    /// compute dispatch always line up.
    fn node_row_lengths(&self, n: usize) -> Vec<usize> {
        let node_count = self.node_worker_counts.len();
        let mut lengths = Vec::with_capacity(node_count);
        let mut assigned = 0;
        for (position, &node_workers) in self.node_worker_counts.iter().enumerate() {
            let rows = if position + 1 == node_count {
                n - assigned
            } else {
                n.saturating_mul(node_workers) / self.total_workers
            };
            assigned += rows;
            lengths.push(rows);
        }
        lengths
    }

    /// Contiguous `(start, len)` output-row segment for each global worker index,
    /// node-major: a node's rows are split evenly across that node's workers.
    fn worker_row_segments(&self, n: usize) -> Vec<(usize, usize)> {
        let node_lengths = self.node_row_lengths(n);
        let mut segments = Vec::with_capacity(self.total_workers);
        let mut node_start = 0;
        for (&node_len, &node_workers) in node_lengths.iter().zip(&self.node_worker_counts) {
            let base = node_len / node_workers;
            let remainder = node_len % node_workers;
            let mut offset = node_start;
            for worker in 0..node_workers {
                let len = base + usize::from(worker < remainder);
                segments.push((offset, len));
                offset += len;
            }
            node_start += node_len;
        }
        segments
    }

    /// [`Self::worker_row_segments`] with every interior boundary snapped to a
    /// multiple of `align`. The per-worker split is computed exactly as the
    /// unaligned version (so node-major ordering and weight placement still line
    /// up), then each cumulative boundary except the final `n` is rounded to the
    /// nearest multiple of `align`, kept monotonic and in `[0, n]`. The result
    /// still covers `0..n` exactly once; a boundary collision can leave a worker
    /// with a zero-length segment (it simply runs no work), which the dispatch
    /// and shard-build paths already tolerate.
    ///
    /// `align <= 1` is the identity (returns the unaligned segments): callers
    /// whose per-column arithmetic is partition-independent pass `1`.
    fn worker_row_segments_aligned(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        let base = self.worker_row_segments(n);
        if align <= 1 {
            return base;
        }
        let mut segments = Vec::with_capacity(base.len());
        let mut prev_boundary = 0;
        let mut cumulative = 0;
        let last = base.len().saturating_sub(1);
        for (index, &(_, len)) in base.iter().enumerate() {
            cumulative += len;
            let boundary = if index == last {
                // The final boundary is always `n`, even when `n` is not a
                // multiple of `align`: the last shard's start is aligned, so its
                // trailing partial N-tile matches the full-width call's tail.
                n
            } else {
                // Round the ideal cumulative boundary to the nearest multiple of
                // `align`, staying monotonic and within bounds.
                let rounded = ((cumulative + align / 2) / align) * align;
                rounded.clamp(prev_boundary, n)
            };
            segments.push((prev_boundary, boundary - prev_boundary));
            prev_boundary = boundary;
        }
        segments
    }

    fn work_stealing_segments_aligned(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        if n == 0 {
            return vec![(0, 0)];
        }
        let align = align.max(1);
        let min_tile = MIN_STEAL_OUTPUTS_PER_TASK.div_ceil(align) * align;
        let max_by_size = n.div_ceil(min_tile).max(1);
        let target = self
            .total_workers
            .saturating_mul(steal_tiles_per_worker())
            .max(1)
            .min(max_by_size)
            .min(n);
        if target <= self.total_workers {
            return self.worker_row_segments_aligned(n, align);
        }
        let mut segments = Vec::with_capacity(target);
        let mut prev = 0usize;
        for index in 0..target {
            let boundary = if index + 1 == target {
                n
            } else {
                let ideal = n.saturating_mul(index + 1).div_ceil(target);
                let rounded = ((ideal + align / 2) / align) * align;
                rounded.clamp(prev, n)
            };
            segments.push((prev, boundary - prev));
            prev = boundary;
        }
        segments
    }

    fn output_segments_aligned(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        if self.uses_work_stealing() {
            self.work_stealing_segments_aligned(n, align)
        } else {
            self.worker_row_segments_aligned(n, align)
        }
    }

    /// Shard `result`'s output rows across the workers and run `compute` on each
    /// worker's contiguous slice under one lightweight barrier.
    ///
    /// `compute(output_start, outputs)` fills the rows
    /// `output_start .. output_start + outputs.len()` -- the same closure the
    /// flat path hands to `par_chunks_mut`, so the arithmetic is identical.
    /// Tiny ops (below the flat path's parallelization threshold) run serially
    /// on the dispatcher, so the same set of ops parallelize as before.
    ///
    /// The serial threshold is sized from [`Self::total_workers`], the executor
    /// that will actually run the fan-out, and *not* from the ambient Rayon
    /// width -- this pool is not a Rayon pool. See [`output_chunk_len_for`].
    pub fn dispatch_output_rows<F>(&self, result: &mut [f32], k: usize, compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        if self.total_workers <= 1 || output_chunk_len_for(self.total_workers, n, k) >= n {
            compute(0, result);
            return;
        }
        if self.uses_work_stealing() {
            self.dispatch_rows_work_stealing(result, 1, compute);
            return;
        }
        self.dispatch_rows_across_workers(result, &compute);
    }

    /// Public view of the contiguous `(start, len)` output-column segment each
    /// global worker owns when a length-`n` GEMV output is sharded across the
    /// pool. Callers that pre-partition a weight along N (e.g. one MLAS SQNBit
    /// packed shard per worker) use this to build shards that line up exactly
    /// with [`Self::dispatch_output_rows_indexed`].
    ///
    /// Every segment boundary is snapped to a multiple of `align` (the last,
    /// `n`-terminated segment excepted). This matters for kernels whose SIMD
    /// column-tiling is *not* bit-stable across an arbitrary N-partition: MLAS's
    /// SQNBit GEMV processes output columns in fixed-width N-tiles, so a shard
    /// boundary that falls *mid-tile* forces MLAS's remainder path to reduce a
    /// block-sum in a different order than the full-width call, shifting that
    /// column by ~1 ULP. Aligning every interior boundary to the N-tile width
    /// keeps every tile whole inside a single shard, so each shard reproduces
    /// the full-width tiling exactly and the concatenated output is
    /// bit-identical to the unsharded call (verified `max_ulp = 0`). Pass
    /// `align = 1` for kernels whose per-column result is already
    /// partition-independent (e.g. the hand int4/int8 GEMV).
    pub fn output_column_segments(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        self.output_segments_aligned(n, align)
    }

    /// Like [`Self::dispatch_output_rows`], but hands each worker its global
    /// index alongside its output slice and always dispatches across the pool
    /// (no serial-threshold short-circuit), so a caller can select the matching
    /// pre-partitioned weight shard (`compute(global_index, output_start,
    /// outputs)`). `result.len()` must equal `n` passed to
    /// [`Self::output_column_segments`], and `align` must match so the dispatch
    /// segments line up byte-for-byte with the caller's pre-built shards; each
    /// worker writes only its own segment, so the concatenated result is
    /// bit-identical to the single-worker path.
    pub fn dispatch_output_rows_indexed<F>(&self, result: &mut [f32], align: usize, compute: &F)
    where
        F: Fn(usize, usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        let segments = self.output_segments_aligned(n, align);
        let table = RowTable {
            base: result.as_mut_ptr(),
            segments: &segments,
        };
        let table = &table;
        if self.uses_work_stealing() {
            let next = AtomicUsize::new(0);
            let next = &next;
            #[cfg(feature = "mlas")]
            if let Some(pool) = &self.work_stealing_pool {
                pool.parallel_for(0, segments.len(), 1, |begin, end| {
                    for task_index in begin..end {
                        let (start, len) = table.segments[task_index];
                        if len == 0 {
                            continue;
                        }
                        // SAFETY: dynamic segments are disjoint, in-bounds
                        // column ranges; each task index is claimed once by
                        // exactly one work-stealing chunk.
                        let outputs =
                            unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
                        compute(task_index, start, outputs);
                    }
                });
                return;
            }
            let job = move |_global_index: usize| loop {
                let task_index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(start, len)) = table.segments.get(task_index) else {
                    break;
                };
                if len == 0 {
                    continue;
                }
                // SAFETY: dynamic segments are disjoint, in-bounds column ranges;
                // each task index is claimed once by exactly one worker.
                let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
                compute(task_index, start, outputs);
            };
            self.dispatch(&job);
            return;
        }
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            if len == 0 {
                return;
            }
            // SAFETY: `worker_row_segments` produces disjoint, in-bounds column
            // ranges covering `0..n` exactly once, so each worker's slice never
            // aliases another's.
            let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
            compute(global_index, start, outputs);
        };
        self.dispatch(&job);
    }

    /// Shard `result`'s `num_rows` fixed-width rows (each `row_len` elements)
    /// across the resident workers and run `compute(row_index, row_slice)` on
    /// each whole row under one lightweight barrier.
    ///
    /// Unlike [`Self::dispatch_output_rows`] (which shards a GEMV's scalar output
    /// rows), this keeps every `row_len`-element row intact on a single worker,
    /// so a caller whose per-row closure needs the full contiguous row (e.g. an
    /// attention head's output vector) can run on the persistent decode pool
    /// instead of a second, contending thread pool. Rows are handed out
    /// contiguously, so concatenating the per-worker slices reproduces the
    /// single-threaded result bit-for-bit (each row is independent).
    pub fn dispatch_output_row_blocks<F>(
        &self,
        result: &mut [f32],
        row_len: usize,
        num_rows: usize,
        compute: &F,
    ) where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        debug_assert_eq!(result.len(), row_len.saturating_mul(num_rows));
        if self.total_workers <= 1 || num_rows <= 1 || row_len == 0 {
            for row in 0..num_rows {
                compute(row, &mut result[row * row_len..(row + 1) * row_len]);
            }
            return;
        }
        let segments = self.worker_row_segments(num_rows);
        let table = RowBlockTable {
            base: result.as_mut_ptr(),
            row_len,
            segments: &segments,
        };
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            for row in start..start + len {
                // SAFETY: `worker_row_segments` produces disjoint, in-bounds row
                // ranges covering `0..num_rows` exactly once, so each worker's
                // `[row*row_len, (row+1)*row_len)` slice never aliases another's.
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        table.base.add(row * table.row_len),
                        table.row_len,
                    )
                };
                compute(row, slice);
            }
        };
        self.dispatch(&job);
    }

    /// Run `num_tasks` independent subtasks across the resident workers under one
    /// lightweight barrier: each worker runs a contiguous range of task indices
    /// and invokes `compute(task_index)` for each.
    ///
    /// Unlike [`Self::dispatch_output_row_blocks`] this partitions only the
    /// *index space*, not a shared result buffer — the caller's closure is
    /// responsible for writing disjoint scratch per task index (flash-decoding
    /// writes one `(max, sum, value-accumulator)` partial per task). The
    /// contiguous, non-overlapping partition is exactly
    /// [`Self::worker_row_segments`], so every index in `0..num_tasks` runs on
    /// exactly one worker, once.
    pub fn dispatch_index_tasks<F>(&self, num_tasks: usize, compute: &F)
    where
        F: Fn(usize) + Sync,
    {
        if self.total_workers <= 1 || num_tasks <= 1 {
            for task in 0..num_tasks {
                compute(task);
            }
            return;
        }
        let segments = self.worker_row_segments(num_tasks);
        let segments = &segments;
        let job = move |global_index: usize| {
            let (start, len) = segments[global_index];
            for task in start..start + len {
                compute(task);
            }
        };
        self.dispatch(&job);
    }

    /// Broadcast the output-row shards to every worker under one barrier,
    /// unconditionally (no serial-threshold check). The public
    /// [`Self::dispatch_output_rows`] applies the threshold before calling this;
    /// tests exercise the multi-worker path directly through it.
    fn dispatch_rows_across_workers<F>(&self, result: &mut [f32], compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        let segments = self.worker_row_segments(n);
        let table = RowTable {
            base: result.as_mut_ptr(),
            segments: &segments,
        };
        // Bind a reference so the `move` closure captures the whole `RowTable`
        // (which carries the manual `Sync` impl) rather than its raw pointer
        // field individually (disjoint capture does not reach through a
        // reference).
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            if len == 0 {
                return;
            }
            // SAFETY: `worker_row_segments` produces disjoint, in-bounds row
            // ranges covering `0..n` exactly once, and each worker touches only
            // its own segment, so these mutable slices never alias.
            let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
            compute(start, outputs);
        };
        self.dispatch(&job);
    }

    fn dispatch_rows_work_stealing<F>(&self, result: &mut [f32], align: usize, compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let segments = self.work_stealing_segments_aligned(result.len(), align);
        let table = RowTable {
            base: result.as_mut_ptr(),
            segments: &segments,
        };
        let table = &table;
        #[cfg(feature = "mlas")]
        if let Some(pool) = &self.work_stealing_pool {
            pool.parallel_for(0, segments.len(), 1, |begin, end| {
                for task_index in begin..end {
                    let (start, len) = table.segments[task_index];
                    if len == 0 {
                        continue;
                    }
                    // SAFETY: each dynamic tile is a disjoint, in-bounds row
                    // range, and the work-stealing pool hands every tile chunk
                    // to exactly one worker.
                    let outputs =
                        unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
                    compute(start, outputs);
                }
            });
            return;
        }
        let next = AtomicUsize::new(0);
        let next = &next;
        let job = move |_global_index: usize| loop {
            let task_index = next.fetch_add(1, Ordering::Relaxed);
            let Some(&(start, len)) = table.segments.get(task_index) else {
                break;
            };
            if len == 0 {
                continue;
            }
            // SAFETY: each dynamic tile is a disjoint, in-bounds row range, and
            // the atomic cursor hands every tile to exactly one worker.
            let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
            compute(start, outputs);
        };
        self.dispatch(&job);
    }

    /// Copy `src` into a fresh buffer whose per-node row shards are first-touched
    /// on their owning node, so each worker later streams node-local memory.
    ///
    /// `src` is a row-major `[n, stride]` weight component; the row split matches
    /// [`Self::worker_row_segments`] exactly so it lines up with dispatch.
    pub fn place_rows<T: Copy + Send + Sync>(&self, src: &[T], n: usize) -> Vec<T> {
        if n == 0 || src.is_empty() || self.total_workers <= 1 {
            return src.to_vec();
        }
        let stride = src.len() / n;
        debug_assert_eq!(stride * n, src.len());
        let mut dst: Vec<T> = Vec::with_capacity(src.len());
        // Leave the buffer uninitialized on purpose: zero-filling here would
        // fault every page onto the dispatcher's node, defeating the node-local
        // first-touch performed by the pinned workers below.
        // SAFETY: `T: Copy` has no `Drop`, capacity is exactly `src.len()`, and
        // every element is overwritten by the per-worker `copy_from_slice`
        // (`worker_row_segments` covers `0..n` exactly once) before the buffer is
        // read.
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len());
        }
        let segments = self.worker_row_segments(n);
        let table = CopyTable {
            dst: dst.as_mut_ptr(),
            src: src.as_ptr(),
            stride,
            segments: &segments,
        };
        // Capture the whole `CopyTable` (manual `Sync`) rather than its raw
        // pointer fields individually.
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            if len == 0 {
                return;
            }
            // SAFETY: disjoint, in-bounds `[start, start+len)` row ranges (in
            // units of `stride`), covering every row exactly once; the pinned
            // worker's write faults these destination pages onto its own node.
            unsafe {
                let dst = table.dst.add(start * table.stride);
                let src = table.src.add(start * table.stride);
                std::ptr::copy_nonoverlapping(src, dst, len * table.stride);
            }
        };
        self.dispatch(&job);
        dst
    }
}

impl Drop for SpmdDecodePools {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Row-major output view handed to a dispatched compute job. Disjoint per-worker
/// segments make the raw base pointer safe to share.
struct RowTable<'a> {
    base: *mut f32,
    segments: &'a [(usize, usize)],
}
// SAFETY: each global worker index reads only its own disjoint row segment.
unsafe impl Sync for RowTable<'_> {}

/// Output view for fixed-width row-block dispatch: `base` is a `[num_rows,
/// row_len]` row-major buffer, and each worker owns the disjoint row range
/// `segments[worker]`.
struct RowBlockTable<'a> {
    base: *mut f32,
    row_len: usize,
    segments: &'a [(usize, usize)],
}
// SAFETY: each global worker index writes only its own disjoint row range.
unsafe impl Sync for RowBlockTable<'_> {}

/// Source/destination view for node-local weight placement.
struct CopyTable<'a, T> {
    dst: *mut T,
    src: *const T,
    stride: usize,
    segments: &'a [(usize, usize)],
}
// SAFETY: each worker copies only its own disjoint row range.
unsafe impl<T: Send + Sync> Sync for CopyTable<'_, T> {}

/// Ensures a worker always acknowledges the current op while making the normal
/// path no more expensive than the existing atomic decrement. `complete`
/// forgets the guard after decrementing; only unwinding executes `Drop`, which
/// poisons the pool before decrementing so the dispatcher cannot miss the panic.
struct WorkerCompletion<'a> {
    shared: &'a SharedState,
    node: usize,
    global_index: usize,
}

impl WorkerCompletion<'_> {
    fn complete(self) {
        self.shared.node_pending[self.node]
            .0
            .fetch_sub(1, Ordering::AcqRel);
        std::mem::forget(self);
    }
}

impl Drop for WorkerCompletion<'_> {
    fn drop(&mut self) {
        self.shared
            .poisoned_worker
            .compare_exchange(
                0,
                self.global_index + 1,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .ok();
        self.shared.node_pending[self.node]
            .0
            .fetch_sub(1, Ordering::AcqRel);
    }
}

/// The persistent worker main loop: wait for a published op, run this worker's
/// shard, acknowledge, repeat until shutdown.
fn worker_loop(shared: Arc<SharedState>, global_index: usize) {
    let node = shared.worker_node[global_index];
    // Track this node's sense line: 0 until the first op is published. Announce
    // readiness only after establishing the baseline; the dispatcher blocks in
    // `build` until every worker has done this, so no op can be published before
    // this worker is waiting for it.
    let mut last_seen: u32 = 0;
    let blocktime = decode_blocktime();
    shared.ready.fetch_add(1, Ordering::AcqRel);
    loop {
        // Bounded active spin (blocktime) then futex park; returns the observed
        // sense, or an unchanged value if shutdown was seen (re-checked below).
        let current = shared.worker_wait(node, last_seen, blocktime);
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        if current == last_seen {
            // Woken without a new op and not shutting down (spurious futex wake);
            // resume waiting on the same sense.
            continue;
        }
        last_seen = current;
        // Read and run the published op. The acquire on the node sense in
        // `worker_wait` established visibility of the job pointer and the pending
        // counts.
        // SAFETY: the dispatcher keeps the pointee alive until every node
        // counter reaches zero, i.e. until after this worker acknowledges below.
        let job = unsafe { (*shared.job.get()).expect("published SPMD job") };
        let completion = WorkerCompletion {
            shared: &shared,
            node,
            global_index,
        };
        // SAFETY: `dispatch` keeps the closure alive until this worker
        // acknowledges through `completion`.
        unsafe { (job.call)(job.data, global_index) };
        completion.complete();
    }
}

/// The process-wide persistent SPMD layout. `None` (once initialized) means the
/// mode opted out or the safe auto-enable gate declined. Held at module scope so
/// [`shutdown_pools`] can *peek* (join the workers if the pool was ever built)
/// without forcing a build.
static POOLS: OnceLock<Option<SpmdDecodePools>> = OnceLock::new();

/// The label describing the active decode path, set once at pool build time.
/// Queryable via [`decode_path_label`] so callers can inspect which strategy is
/// active without parsing stderr or comparing throughput numbers.
static DECODE_PATH_LABEL: OnceLock<&'static str> = OnceLock::new();

/// Human-readable label for the decode path that was selected. Returns the
/// path name once the pool initialization has run, or `"unresolved"` if it
/// has not been queried yet.
///
/// Examples:
/// - `"spmd-pool"` — the persistent SPMD pool (default or `=1`)
/// - `"adaptive"` — load-adaptive calibration (`=auto`)
/// - `"flat"` — the flat legacy path (`=0` or fallback)
/// - `"unresolved"` — pool initialization has not run yet
pub fn decode_path_label() -> &'static str {
    DECODE_PATH_LABEL.get().copied().unwrap_or("unresolved")
}

/// The width a decode request asked for, what it actually got, and which path
/// served it.
///
/// Three separate mechanisms can quietly hand back fewer compute lanes than
/// were requested: the pre-clamp to `available_parallelism` in
/// `resolve_persistent_decode_threads_with_override`,
/// [`reserve_split_headroom`], and the single-CPU-cpuset fallback in
/// [`build_from_env`]. Only the last reports through [`report_spmd_fallback`],
/// which is itself `tracing::debug!` or `NXRT_CALIB_DEBUG`-gated; the other two
/// log nothing at all. In a default benchmark run none of them are visible, so a
/// `t=N` row in a results table is a *label* rather than a measurement of width
/// `N`. This is the read that turns it back into a measurement.
///
/// [`reserve_single_group_headroom`] is deliberately *not* in that list. It does
/// reduce the number of spawned threads, but it only runs in the single-group
/// case, which is exactly the case where `dispatcher_owns_a_shard` is true, so
/// the lane it takes is added straight back by the dispatcher's own shard. A
/// 2-lane budget on a 2-CPU cpuset spawns one thread and still reports
/// `realized = 2`, measured, because two lanes really do compute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeWidth {
    /// Compute lanes asked for: the explicit `ONNX_GENAI_CPU_DECODE_THREADS`
    /// budget, or the default physical-core budget when unset. `None` means
    /// either the width was never resolved (no decode has run yet) or the
    /// caller opted out with `=0`, which is what `path` disambiguates.
    pub requested: Option<usize>,
    /// Compute lanes actually available. `None` until a decode path is chosen.
    ///
    /// On the persistent pool this is exact and includes the dispatcher's own
    /// shard. On the flat path there is no fixed layout to report, so it is the
    /// width of the Rayon pool **the calling thread is on** -- meaningful inside
    /// `with_decode_pool_scope`, and the global width outside it. Same
    /// convention as [`crate::kernels::matmul_nbits::active_decode_worker_count`].
    pub realized: Option<usize>,
    /// The selected path, as [`decode_path_label`] reports it.
    pub path: &'static str,
}

impl DecodeWidth {
    /// Whether the realized width matches what was asked for.
    ///
    /// `false` whenever a request was silently reduced, and also whenever the
    /// width has not been resolved yet -- a harness that asserts on this cannot
    /// accidentally pass by querying too early.
    pub fn is_as_requested(&self) -> bool {
        match (self.requested, self.realized) {
            (Some(requested), Some(realized)) => requested == realized,
            _ => false,
        }
    }
}

/// Report the requested vs realized decode width and the path that served it.
///
/// Deliberately **does not** build the pool: it reads the already-initialized
/// statics, so calling it cannot change which path a process takes or when it is
/// chosen. A harness should run at least one decode step first and then assert;
/// before that the fields read `None` / `"unresolved"` and
/// [`DecodeWidth::is_as_requested`] is `false`.
pub fn decode_width() -> DecodeWidth {
    let requested = REQUESTED_WIDTH.get().copied().flatten();
    let realized = match POOLS.get() {
        Some(Some(pools)) => Some(pools.total_workers()),
        // The pool was resolved and declined: decode is on the flat path, whose
        // width is whatever Rayon pool the caller is standing in.
        Some(None) => Some(rayon::current_num_threads()),
        None => None,
    };
    DecodeWidth {
        requested,
        realized,
        path: decode_path_label(),
    }
}

/// The width the caller asked for, recorded when the pool is resolved and before
/// any of the reductions that can shrink it.
///
/// When an explicit request exists this is the *unclamped*
/// [`crate::kernels::matmul_nbits::decode_thread_budget`], not the resolved
/// width `build_from_env` receives. The resolver already clamps to
/// `available_parallelism`, so recording its output would report a request of 8
/// inside a 2-CPU cpuset as `2 requested, 2 realized` -- satisfied. That is
/// exactly the silent mislabelling this read exists to expose, so the comparison
/// has to be against what the user actually asked for.
///
/// With no explicit request this falls back to the resolved default width, which
/// under default policy *is* the intended width -- there is no request for it to
/// disagree with, so [`DecodeWidth::is_as_requested`] is meaningful on a
/// default-configured run rather than trivially `false`.
///
/// `None` only while the pool is still unresolved, or when the caller opted out
/// with `=0` (which never reaches the latch in [`pools`]).
static REQUESTED_WIDTH: OnceLock<Option<usize>> = OnceLock::new();

/// The lazily built persistent SPMD layout, or `None` when the mode is opted out
/// or the safe auto-enable gate declines. Built once and reused for the whole
/// process.
pub fn pools() -> Option<&'static SpmdDecodePools> {
    POOLS
        .get_or_init(|| {
            let threads = default_threads();
            // Latched here rather than inside `build_from_env` so the recorded
            // request can only ever describe the pool that `POOLS` actually
            // holds. `build_from_env` is `pub`; latching there would let a direct
            // call record a width that no realized pool matches -- reintroducing
            // the exact "the t=N label is a lie" failure this read exists to
            // catch, through the front door.
            //
            // Prefer the unclamped `decode_thread_budget()`, falling back to
            // `threads` only when nobody asked for a specific width. Under
            // default policy the clamped width *is* the intended width, so there
            // is no request for it to disagree with.
            REQUESTED_WIDTH
                .get_or_init(|| crate::kernels::matmul_nbits::decode_thread_budget().or(threads));
            build_from_env(threads)
        })
        .as_ref()
}

/// Signal the persistent pool's workers to stop and **join** them, if the pool
/// was ever built. Idempotent and safe to call when the pool was never built (it
/// only inspects the already-initialized static; it never forces a build).
///
/// The persistent pool lives in a module-level `static`, and Rust does **not**
/// run `Drop` on statics at process exit. Without an explicit join the pool's
/// hot worker threads stay alive (spinning or parked on `Arc<SharedState>`)
/// while the process tears down its runtime. On weakly-ordered targets (notably
/// native Windows ARM64) a worker still touching that shared state during
/// teardown can fault the whole process with `STATUS_ACCESS_VIOLATION`
/// (0xC0000005) and an empty stderr -- not a Rust panic. Calling this before the
/// process exits makes teardown deterministic: workers observe the stop flag,
/// return, and are joined, so none is live during runtime teardown.
///
/// This is a teardown-only operation: it takes no locks on the decode hot path
/// and must not be called between decode ops (it stops the workers).
pub fn shutdown_pools() {
    if let Some(Some(pool)) = POOLS.get() {
        pool.shutdown();
    }
}

/// Resolve the persistent pool's worker count. Honors `ONNX_GENAI_CPU_DECODE_THREADS`
/// when set (`0` opts out); when unset it uses the persistent-specific default
/// (about half the logical CPUs), *not* the flat pool's eight-worker ceiling --
/// see [`crate::kernels::matmul_nbits::configured_persistent_decode_threads`].
fn default_threads() -> Option<usize> {
    crate::kernels::matmul_nbits::configured_persistent_decode_threads()
}

/// How the persistent pool was selected, parsed from `PERSISTENT_POOL_ENV`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PersistenceMode {
    /// `=0`: explicit opt-out; the decode path stays on the flat legacy pool.
    Off,
    /// Unset (or `=1`, or an unrecognized value): the default. The persistent
    /// SPMD pool is always used, deterministically -- no host probing.
    On,
    /// `=auto`: opt-in load-adaptive calibration. The pool is built but the
    /// [`Calibrator`] times the live decode step both ways and keeps the faster
    /// path. Under co-tenant load the flat path usually wins.
    Adaptive,
}

/// Parse the persistence mode from the raw env value (`None` = unset). Unset,
/// `=1`, and any unrecognized value map to `On` (the deterministic pool
/// default); `=0` is the explicit opt-out; `=auto` enables load-adaptive
/// calibration.
pub(crate) fn persistence_mode_from_raw(raw: Option<&str>) -> PersistenceMode {
    match raw.map(str::trim) {
        Some("0") => PersistenceMode::Off,
        Some(v) if v.eq_ignore_ascii_case("auto") => PersistenceMode::Adaptive,
        _ => PersistenceMode::On,
    }
}

/// Default when [`PERSISTENT_POOL_ENV`] is unset. `true` (the historical
/// default) builds the persistent SPMD pool; a host that never enters an SPMD
/// decode scope sets this to `false`.
static PERSISTENT_POOL_DEFAULT: AtomicBool = AtomicBool::new(true);

/// Opt this process out of (or back into) the persistent SPMD decode pool when
/// [`PERSISTENT_POOL_ENV`] is unset.
///
/// The pool only pays off for a host that drives decode *inside* an SPMD scope
/// (`onnx-genai-engine`'s native decode loop): resident workers, one barrier
/// per projection, no fork/join. A host that never enters such a scope -- the
/// plugin EP, where ONNX Runtime owns the graph and the threads -- gets only
/// the costs: resident workers competing with ORT's own intra-op pool, and an
/// MLAS weight partitioned into one shard per persistent decode worker
/// (`available_parallelism() / 2` by default), which caps an unscoped decode
/// GEMV at that worker count however many threads the host has.
///
/// Measured on the plugin path (32 vCPU AMD EPYC 9V74, MLAS build, int4
/// block-32, K=N=2048, M=1, p50 of 41 runs, each backend alone in its process):
/// 0.376 ms with the pool built against 0.092 ms without it, with ONNX
/// Runtime's own CPU EP at 0.097 ms.
///
/// An explicit `PERSISTENT_POOL_ENV` setting always wins. Must be called before
/// the first [`pools()`] query; afterwards the built layout is fixed for the
/// process. `Relaxed` is sufficient because the only supported caller writes
/// this during library initialization, before the host has created a session or
/// spawned the threads that read it -- that handoff is the synchronisation.
pub fn set_persistent_decode_pool_default(enabled: bool) {
    PERSISTENT_POOL_DEFAULT.store(enabled, Ordering::Relaxed);
}

/// Resolve the persistence mode from the raw env value and the process default.
/// An explicit setting always wins; the default only decides the unset case.
/// Pure so the precedence is unit-tested without env races.
pub(crate) fn resolve_persistence_mode(
    raw: Option<&str>,
    default_enabled: bool,
) -> PersistenceMode {
    match raw {
        Some(value) => persistence_mode_from_raw(Some(value)),
        None if !default_enabled => PersistenceMode::Off,
        None => persistence_mode_from_raw(None),
    }
}

fn persistence_mode() -> PersistenceMode {
    resolve_persistence_mode(
        std::env::var(PERSISTENT_POOL_ENV).ok().as_deref(),
        PERSISTENT_POOL_DEFAULT.load(Ordering::Relaxed),
    )
}

/// Whether a persistence mode **builds** the persistent SPMD pool. Both `On`
/// (the default or `=1`) and `Adaptive` (`=auto`) build it: `On` always
/// dispatches to it, and `Adaptive` needs it available so calibration can time
/// the real workload on it. Only `Off` (`=0`) never builds it. Pure so the
/// gating is unit-tested without env races.
fn pool_mode_builds(mode: PersistenceMode) -> bool {
    matches!(mode, PersistenceMode::On | PersistenceMode::Adaptive)
}

/// Whether a persistence mode **unconditionally** dispatches to the pool (no
/// calibration): `On` (the default or `=1`). `Adaptive` builds the pool but
/// lets the [`Calibrator`] pick per step; `Off` never uses it.
fn pool_mode_forces(mode: PersistenceMode) -> bool {
    matches!(mode, PersistenceMode::On)
}

/// Whether the persistent pool is unconditionally dispatched to (the default,
/// or `PERSISTENT_POOL=1`). Used to keep the `numa-split` mutual-exclusion
/// diagnostic scoped to users who actually asked for the persistent pool, to
/// make dense-f32 decode still eligible for the pool, and to skip calibration.
pub(crate) fn is_forced() -> bool {
    pool_mode_forces(persistence_mode())
}

/// Build the persistent SPMD layout when the mode builds it (`On` -- the default
/// or `=1` -- and `Adaptive` `=auto`); `=0` (Off) or `THREADS=0` return `None`
/// so decode stays on the flat path. Under `Adaptive` the pool is built but only
/// *used* when calibration adopts it (see [`Calibrator`]); under `On` it is
/// always used.
///
/// Two or more usable NUMA nodes yield the two-level node-pinned layout; a
/// single-node host, a non-NUMA machine, or a platform without pinning yields a
/// single unpinned worker group (still the lightweight barrier, still correct).
pub fn build_from_env(threads: Option<usize>) -> Option<SpmdDecodePools> {
    // Build for `On` (the default or `=1`) and `Adaptive` (`=auto`). `On` always
    // dispatches to the pool; `Adaptive` needs the pool available so the
    // calibrator can time the live decode step on it. `Off` (`=0`) and `THREADS=0`
    // leave decode on the flat Rayon path. See `PERSISTENT_POOL_ENV`.
    let mode = persistence_mode();
    if !pool_mode_builds(mode) {
        // An explicit `=0` opt-out is a decision, not an unresolved state: decode
        // is definitively on the flat path. Without this the label reads
        // "unresolved" forever and a harness cannot tell an opt-out apart from a
        // process that has not decoded yet.
        DECODE_PATH_LABEL.get_or_init(|| "flat");
        return None;
    }
    // Adaptive defers to an explicit decode-affinity request: if the user set
    // `ONNX_GENAI_CPU_DECODE_AFFINITY` (numa-split, compact, node:N, off, ...),
    // they picked a specific strategy, so Adaptive does not build/calibrate the
    // persistent pool and lets that request drive decode (numa-split via
    // `numa_pools`, everything else via the flat path + `plan_decode_affinity`).
    // `On` still builds the pool and keeps its documented precedence.
    if matches!(mode, PersistenceMode::Adaptive) && explicit_decode_affinity_set() {
        // Also a decision: the user's affinity request drives decode via
        // `numa_pools` or the flat path, so record that rather than leaving the
        // label unresolved.
        DECODE_PATH_LABEL.get_or_init(|| "flat");
        return None;
    }
    let Some(total) = threads else {
        report_spmd_fallback(
            "ONNX_GENAI_CPU_DECODE_THREADS=0 opts out of the bounded pool; the persistent \
             SPMD pool needs a bounded worker count -- leaving the decode path unchanged",
        );
        return None;
    };
    if total == 0 {
        DECODE_PATH_LABEL.get_or_init(|| "flat");
        return None;
    }
    // Cpuset-aware dispatcher headroom: when the requested spinning-worker count
    // would occupy *every* CPU the process is allowed to run on, a single
    // allowed CPU leaves no core for the inline dispatcher (the engine thread
    // that publishes each op and spins on the per-node completion counters).
    // With N spinning workers pinned across all N allowed CPUs the dispatcher is
    // migrated onto a busy worker core and can no longer publish jobs / read
    // counters promptly, collapsing throughput ~20-60x (measured 1.47 tok/s at
    // 32 workers on `taskset -c 0-31` vs ~29 tok/s once one CPU is spare). If the
    // whole allowed cpuset is a single CPU there is no core to reserve, so the
    // pool cannot run without starving itself -- fall back to the flat path.
    if let Some(allowed) = crate::decode_affinity::allowed_cpus()
        && allowed.len() == 1
    {
        report_spmd_fallback(
            "the process is confined to a single CPU (cpuset/taskset), which leaves no core \
             for the inline dispatcher alongside a spinning worker -- leaving decode on the \
             flat path instead of starving the persistent SPMD pool",
        );
        return None;
    }
    report_pool_built(mode);
    let shards = node_shards(total);
    // The dispatcher computes a shard exactly when the headroom reservation took
    // a worker away to keep its CPU free. Then `total - 1` pinned threads plus
    // the dispatcher give `total` compute lanes on `total` CPUs -- the requested
    // width, with the anti-starvation reservation still in force.
    //
    // This is the common case, not a corner case: `bound_process_to_decode_budget`
    // confines the process to exactly `total` CPUs at EP initialization, so an
    // explicit budget *always* arrives here fully subscribed. Without the
    // dispatcher shard, `ONNX_GENAI_CPU_DECODE_THREADS=N` silently buys `N-1`
    // lanes -- and at `N=2` that is one lane, which trips the `total_workers <= 1`
    // serial short-circuit and makes the knob indistinguishable from `=1`.
    //
    // Restricted to the single-group layout. On a NUMA split the dispatcher's
    // node is not known at build time, so handing it a shard could pull that
    // shard's weights across sockets; those layouts keep the previous behavior.
    Some(SpmdDecodePools::build(
        &shards,
        dispatcher_owns_a_shard(&shards, total),
    ))
}

/// Whether the inline dispatcher should own a compute shard.
///
/// True exactly when the single-group headroom reservation took a worker away
/// to keep a CPU free for the dispatcher: giving that CPU back a compute lane
/// restores the requested width without re-pinning a worker to every core. A
/// group that already had headroom is left alone, or the pool would be one lane
/// wider than the user asked for.
fn dispatcher_owns_a_shard(shards: &[NodeShard], requested: usize) -> bool {
    shards.len() == 1 && shards[0].workers < requested
}

/// Whether `ONNX_GENAI_CPU_DECODE_AFFINITY` is set to a non-empty value.
/// Adaptive calibration only engages when it is unset, so an explicit affinity
/// request is honored on the flat/numa path exactly as before.
fn explicit_decode_affinity_set() -> bool {
    std::env::var(crate::decode_affinity::DECODE_AFFINITY_ENV)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

/// Resolve the node shards for `total` workers: the multi-node split when the
/// (cpuset-restricted) topology exposes >=2 nodes, otherwise a single group.
///
/// In both cases the pinned-worker count is capped so the inline dispatcher
/// always keeps at least one allowed CPU free (see [`DISPATCHER_RESERVED_CPUS`]
/// and [`reserve_single_group_headroom`] / [`reserve_split_headroom`]). Without
/// that reservation a fully-subscribed cpuset (workers == allowed CPUs) pins a
/// spinning worker on every core, starving the dispatcher and collapsing
/// throughput.
fn node_shards(total: usize) -> Vec<NodeShard> {
    let allowed = crate::decode_affinity::allowed_cpus();
    if let Some(topology) = NumaTopology::detect() {
        let topology = topology.restrict_to_allowed(allowed.as_deref());
        if let Some(mut shards) = topology.split_workers(total) {
            // Reserve a dispatcher core on every node so the engine thread has a
            // free core on whichever socket the scheduler places it, and each
            // node's completion counter can be read without contending with a
            // pinned spinning worker.
            reserve_split_headroom(&mut shards);
            return shards;
        }
    }
    // Single-node / non-NUMA / no-pinning fallback: one group. Pin to the
    // process's allowed CPUs when known (best-effort), else leave unpinned.
    let cpus = allowed.unwrap_or_default();
    let workers = reserve_single_group_headroom(total, cpus.len());
    vec![NodeShard {
        index: 0,
        cpus,
        workers,
    }]
}

/// Allowed CPUs kept free for the inline dispatcher per pinned worker group.
///
/// The dispatcher is a single thread, so one spare core is sufficient for it to
/// run; reserving *per node* (see [`reserve_split_headroom`]) rather than once
/// globally guarantees the spare core lands on whichever socket the scheduler
/// places the dispatcher on, and keeps every node's completion-counter reads
/// unblocked on a NUMA-split layout. One is enough because the workers already
/// plateau around half the logical CPUs (memory-bandwidth bound), so giving up
/// the single highest-index core costs nothing measurable while removing the
/// starvation cliff.
const DISPATCHER_RESERVED_CPUS: usize = 1;

/// Cap a single pinned worker group so at least [`DISPATCHER_RESERVED_CPUS`]
/// allowed CPU stays free for the inline dispatcher.
///
/// `allowed_count == 0` means the allowed set is unknown, so workers run
/// unpinned and cannot starve the dispatcher by occupying every core -- the
/// count is returned unchanged. When there is already headroom
/// (`total < allowed_count`) the count is likewise unchanged, so this is a
/// strict no-op unless the group would otherwise be fully subscribed. The result
/// is floored at one worker: even a user who sets
/// `ONNX_GENAI_CPU_DECODE_THREADS=N` on an exactly-N-CPU cpuset gets N-1 pinned
/// workers (never zero); the single-CPU case is handled earlier by falling back
/// to the flat path in [`build_from_env`].
fn reserve_single_group_headroom(total: usize, allowed_count: usize) -> usize {
    if allowed_count == 0 || total < allowed_count {
        return total;
    }
    allowed_count
        .saturating_sub(DISPATCHER_RESERVED_CPUS)
        .max(1)
}

/// Cap each NUMA-split shard so at least [`DISPATCHER_RESERVED_CPUS`] CPU of that
/// node stays free for the inline dispatcher. Only nodes that would be fully
/// subscribed (`workers == node CPUs`) are reduced; a node that already has a
/// spare core is left untouched, so this is a no-op wherever headroom exists.
/// Each shard keeps at least one worker.
fn reserve_split_headroom(shards: &mut [NodeShard]) {
    for shard in shards.iter_mut() {
        let cap = shard
            .cpus
            .len()
            .saturating_sub(DISPATCHER_RESERVED_CPUS)
            .max(1);
        shard.workers = shard.workers.min(cap);
    }
}

/// Log the first persistent-pool fallback/pinning problem once so a restricted
/// or unsupported host surfaces the reason without spamming every worker.
/// Emitted as `tracing::debug!` when the `tracing` feature is enabled, or
/// gated behind `NXRT_CALIB_DEBUG` otherwise.
fn report_spmd_fallback(message: &str) {
    DECODE_PATH_LABEL.get_or_init(|| "flat");
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        #[cfg(feature = "tracing")]
        tracing_crate::debug!(path = "flat", reason = %message, "cpu decode pool fallback");
        #[cfg(not(feature = "tracing"))]
        if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
            eprintln!("onnx-genai: persistent SPMD decode pool: {message}");
        }
    }
}

/// Record the selected decode path in the queryable [`DECODE_PATH_LABEL`] static
/// so callers can inspect it via [`decode_path_label`]. Emitted as
/// `tracing::debug!` when the `tracing` feature is enabled (visible to any
/// subscriber at `debug` level or below), or gated behind `NXRT_CALIB_DEBUG`
/// otherwise. See `docs/architecture/ERROR_AND_LOGGING_CONVENTIONS.md` for level guidance.
fn report_pool_built(mode: PersistenceMode) {
    let label = match mode {
        PersistenceMode::On if decode_schedule() == DecodeSchedule::Steal => "work-stealing-pool",
        PersistenceMode::On => "spmd-pool",
        PersistenceMode::Adaptive => "adaptive",
        PersistenceMode::Off => "flat",
    };
    DECODE_PATH_LABEL.get_or_init(|| label);

    #[cfg(feature = "tracing")]
    {
        let workers = POOLS
            .get()
            .and_then(|p| p.as_ref())
            .map(|p| p.total_workers())
            .unwrap_or(0);
        tracing_crate::debug!(path = label, workers, "cpu decode path selected");
    }
    #[cfg(not(feature = "tracing"))]
    if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
        static REPORTED: OnceLock<()> = OnceLock::new();
        if REPORTED.set(()).is_ok() {
            match mode {
                PersistenceMode::On => eprintln!(
                    "NXRT_CALIB: decode path = persistent SPMD pool (default). \
                     Set ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=auto for load-adaptive \
                     selection, =0 for the flat legacy path"
                ),
                PersistenceMode::Adaptive => eprintln!(
                    "NXRT_CALIB: decode path = adaptive \
                     (ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=auto); \
                     the initial decode steps are timed both ways and the faster path is \
                     kept permanently. Set =1 or unset for the deterministic pool, =0 for flat"
                ),
                PersistenceMode::Off => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive calibration (opt-in via =auto): pick the pool or the flat path by
// measuring the live decode step both ways, keeping the faster one.
// ---------------------------------------------------------------------------

/// Which decode path a single adaptive-mode decode step should take. Both paths
/// are token-exact, so the choice never changes the emitted tokens -- only how
/// fast the step runs. The flat and pool paths use different floating-point
/// reduction orders, so switching between them can produce different logits under
/// greedy decode. The calibrator freezes the path once committed to avoid
/// mid-generation non-determinism.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AutoPath {
    /// Dispatch this step's projections to the persistent SPMD pool.
    Pool,
    /// Run this step on the flat Rayon decode pool (the safe default under load).
    Flat,
}

/// Decode steps spent on the pool before the first measurement, so the pool's
/// one-time constant weights are prepacked and node-locally first-touched (see
/// [`SpmdDecodePools::place_rows`]) and caches are warm before it is timed.
/// Without this the pool would be measured reading cross-node memory and unfairly
/// lose; it is a handful of steps, amortized over the whole generation.
const CALIB_WARMUP_STEPS: u64 = 2;
/// Samples collected per path during a probe. The decision uses the median, so an
/// odd count with a small majority rejects a single load-spike outlier while
/// keeping the probe short (a probe costs `2 * CALIB_PROBE_SAMPLES` real steps,
/// half of them possibly-slower pool steps).
const CALIB_PROBE_SAMPLES: usize = 5;
/// Hysteresis margin: the pool is adopted only when its median step time is at
/// least this percent faster than the flat path. Biases toward the flat path (the
/// regression-safe default) and prevents flapping when the two paths are close.
const CALIB_SWITCH_MARGIN_PCT: u64 = 8;
/// Samples discarded at the start of each probe block. The persistent pool's
/// worker threads keep spinning for a short while after their last dispatch, so
/// the *first* flat step after a pool step (and vice-versa) is polluted by the
/// other path's threads still winding down. Discarding the transition sample
/// makes each block measure its path in isolation -- critical because measuring
/// flat while pool workers are still hot makes flat look slow and would bias the
/// choice *toward* the pool (the regression). See the block-ordered probe below.
const CALIB_PROBE_DISCARD: usize = 1;
/// A probe block is treated as **contended** (a co-tenant burst landed on it, so
/// its median overstates that path's true cost) when its fast-half spread
/// `(median - min) / median` exceeds this percent. Every decode step does the
/// identical M=1 work, so an *uncontended* block -- whether uniformly fast (idle)
/// or uniformly slow (sustained load) -- has a tiny spread; only a transient burst
/// that hits some, but not all, of the block's steps inflates it. Sized above the
/// observed clean-block jitter (flat blocks measured ~1-7%; a burst-poisoned pool
/// block measured ~28%) so sustained load is *not* misread as a burst (it commits
/// flat as before) while a transient burst *is* caught.
const CALIB_CONTENDED_SPREAD_PCT: u64 = 15;
/// Maximum times a single recalibration re-collects its probe when a block looks
/// contended before giving up and committing on the (possibly noisy) median. A
/// small bound guarantees the calibrator always commits within a handful of
/// probes -- a genuinely (uniformly) loaded host re-probes at most this many times
/// then correctly commits flat, so re-probing can never loop or sustain overhead.
const CALIB_MAX_REPROBES: u32 = 3;

/// The calibration probe measures the two paths in **separate contiguous blocks**
/// (all flat, then all pool) rather than interleaving them, so a just-finished
/// pool step's still-spinning workers never pollute a flat measurement. Flat is
/// measured first, while the pool is quiesced (parked), which is exactly the
/// steady state a committed-flat adaptive run experiences.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CalibPhase {
    /// Warm the pool (prepack + node-local first-touch) before any measurement.
    Warmup,
    /// Collect the flat block, with the pool quiesced (its worker threads parked).
    ProbeFlat,
    /// Collect the pool block.
    ProbePool,
    /// Run the committed path until the recalibration period elapses.
    Committed,
}

/// Runtime calibrate-and-pick state machine for `Adaptive` mode (`=auto`).
///
/// # Why this cannot regress under load
///
/// * The committed path starts as -- and defaults back to -- [`AutoPath::Flat`],
///   today's flat decode path. A host that never lets the pool win keeps running
///   exactly the flat path.
/// * The pool is adopted only when its *measured* median step time beats the flat
///   path's by [`CALIB_SWITCH_MARGIN_PCT`]. Under co-tenant load the spinning
///   pool is slower, so it never clears the bar and the flat path stays committed.
/// * Flat and pool are measured in **separate blocks** (flat first, pool
///   quiesced), with the transition sample discarded ([`CALIB_PROBE_DISCARD`]),
///   so a pool step's still-spinning workers cannot make the flat measurement
///   look slow. (An interleaved probe *did* mis-commit to the pool under load;
///   see `.squad/decisions.md`, Hudson 2026-07-24.)
/// * The only pool work done while the flat path is committed is a bounded probe
///   (`<= CALIB_PROBE_SAMPLES` pool steps during calibration, plus a one-time
///   warmup), so the worst case is a small number of possibly-slower steps during
///   the initial calibration -- never a sustained regression.
/// * A probe block hit by a *transient* co-tenant burst (non-uniform samples --
///   see [`block_contended`]) is discarded and re-collected (bounded by
///   [`CALIB_MAX_REPROBES`]) rather than committed, so a momentary spike on the
///   pool block no longer locks the slower flat path in permanently. A
///   *uniformly* slow block (genuine sustained load) is never flagged, so this
///   only ever avoids acting on unreliable data -- it cannot regress a clean or a
///   steadily-loaded measurement.
///
/// # Path freezing
///
/// Once the calibrator commits to a path (pool or flat), it stays committed
/// permanently. The flat and pool paths use different floating-point reduction
/// orders, so switching mid-generation changes logits and produces different
/// tokens under greedy decode. A host that becomes loaded after commitment will
/// run the (now-suboptimal) pool path for the rest of the session; this is
/// the correct trade-off because deterministic output is more important than
/// adapting to load changes. Use `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=0` to
/// force flat if the host is known to be loaded.
///
/// The decision logic is pure (no threads, no clock; the only env read is an
/// optional `NXRT_CALIB_DEBUG` diagnostic print that never affects the choice), so
/// it is unit-tested deterministically by feeding synthetic per-path samples.
struct Calibrator {
    phase: CalibPhase,
    warmup_left: u64,
    discard_left: usize,
    pool_ns: Vec<u64>,
    flat_ns: Vec<u64>,
    committed: AutoPath,
    reprobes_left: u32,
}

impl Calibrator {
    fn new() -> Self {
        Self {
            phase: CalibPhase::Warmup,
            warmup_left: CALIB_WARMUP_STEPS,
            discard_left: 0,
            pool_ns: Vec::with_capacity(CALIB_PROBE_SAMPLES),
            flat_ns: Vec::with_capacity(CALIB_PROBE_SAMPLES),
            // Default to the flat path: the safe, no-regression baseline that a
            // host which never lets the pool win keeps forever.
            committed: AutoPath::Flat,
            reprobes_left: CALIB_MAX_REPROBES,
        }
    }

    /// The path the next decode step should take. Warmup uses the pool (to place
    /// weights node-locally); the probe runs the flat block then the pool block;
    /// a committed phase returns the committed path.
    fn choose(&self) -> AutoPath {
        match self.phase {
            CalibPhase::Warmup | CalibPhase::ProbePool => AutoPath::Pool,
            CalibPhase::ProbeFlat => AutoPath::Flat,
            CalibPhase::Committed => self.committed,
        }
    }

    /// Feed back the measured wall time (nanoseconds) of a step that took `path`.
    fn record(&mut self, path: AutoPath, ns: u64) {
        match self.phase {
            CalibPhase::Warmup => {
                self.warmup_left = self.warmup_left.saturating_sub(1);
                if self.warmup_left == 0 {
                    self.enter_flat_probe();
                }
            }
            CalibPhase::ProbeFlat => {
                if path == AutoPath::Flat {
                    self.push_sample_or_discard(ns, true);
                }
                if self.flat_ns.len() >= CALIB_PROBE_SAMPLES {
                    self.enter_pool_probe();
                }
            }
            CalibPhase::ProbePool => {
                if path == AutoPath::Pool {
                    self.push_sample_or_discard(ns, false);
                }
                if self.pool_ns.len() >= CALIB_PROBE_SAMPLES {
                    self.commit_from_samples();
                }
            }
            CalibPhase::Committed => {
                // Path is frozen once committed. The flat and pool paths use
                // different FP reduction orders, so switching mid-generation
                // produces non-deterministic tokens under greedy decode.
            }
        }
    }

    /// Record a sample into the current block, discarding the leading transition
    /// sample(s) so the other path's winding-down threads do not pollute it.
    fn push_sample_or_discard(&mut self, ns: u64, flat: bool) {
        if self.discard_left > 0 {
            self.discard_left -= 1;
            return;
        }
        let block = if flat {
            &mut self.flat_ns
        } else {
            &mut self.pool_ns
        };
        if block.len() < CALIB_PROBE_SAMPLES {
            block.push(ns);
        }
    }

    fn enter_flat_probe(&mut self) {
        self.phase = CalibPhase::ProbeFlat;
        self.flat_ns.clear();
        self.pool_ns.clear();
        self.discard_left = CALIB_PROBE_DISCARD;
    }

    fn enter_pool_probe(&mut self) {
        self.phase = CalibPhase::ProbePool;
        self.discard_left = CALIB_PROBE_DISCARD;
    }

    fn commit_from_samples(&mut self) {
        let pool = median_ns(&mut self.pool_ns);
        let flat = median_ns(&mut self.flat_ns);
        // A transient co-tenant burst that lands on part of a probe block inflates
        // that block's median above the path's true cost, which can lock in the
        // wrong path for the whole recalibration period (observed: a burst on the
        // pool block poisoned its median and stuck decode on the flat path for 600
        // steps). When a block looks contended (non-uniform -- see
        // `CALIB_CONTENDED_SPREAD_PCT`) and re-probe budget remains, discard this
        // probe and re-collect rather than commit on unreliable data. A uniformly
        // slow block (genuine sustained load) is NOT flagged, so a loaded host
        // still commits flat exactly as before -- this only ever avoids acting on
        // a burst-poisoned probe, so it cannot regress a clean measurement.
        let contended =
            block_contended(pool, &self.pool_ns) || block_contended(flat, &self.flat_ns);
        if contended && self.reprobes_left > 0 {
            self.reprobes_left -= 1;
            if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
                eprintln!(
                    "NXRT_CALIB: contended probe (pool_median={}us flat_median={}us) -> re-probe ({} left)",
                    pool / 1000,
                    flat / 1000,
                    self.reprobes_left,
                );
            }
            self.enter_flat_probe();
            return;
        }
        // Adopt the pool only when it is at least CALIB_SWITCH_MARGIN_PCT faster:
        // pool <= flat * (100 - margin) / 100. Use u128 so the multiply cannot
        // overflow for pathologically large samples.
        let pool_scaled = u128::from(pool) * 100;
        let flat_scaled = u128::from(flat) * u128::from(100 - CALIB_SWITCH_MARGIN_PCT);
        self.committed = if pool_scaled <= flat_scaled {
            AutoPath::Pool
        } else {
            AutoPath::Flat
        };
        if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
            eprintln!(
                "NXRT_CALIB: pool_median={}us flat_median={}us margin={}% -> {:?} (pool_samples={:?} flat_samples={:?})",
                pool / 1000,
                flat / 1000,
                CALIB_SWITCH_MARGIN_PCT,
                self.committed,
                self.pool_ns.iter().map(|n| n / 1000).collect::<Vec<_>>(),
                self.flat_ns.iter().map(|n| n / 1000).collect::<Vec<_>>(),
            );
        }
        self.phase = CalibPhase::Committed;
        // Path is frozen permanently — no re-probe. See the "Path freezing"
        // section in the struct-level docs.
        self.reprobes_left = CALIB_MAX_REPROBES;
        self.pool_ns.clear();
        self.flat_ns.clear();
    }
}

/// Whether a probe block was contended by a *transient* burst: its fast-half
/// spread `(median - min) / median` exceeds [`CALIB_CONTENDED_SPREAD_PCT`]. Every
/// M=1 decode step does identical work, so an uncontended block -- uniformly fast
/// (idle) or uniformly slow (sustained load) -- has a tiny spread; only a burst
/// that hits some steps but not others pulls the median well above the min. Empty
/// or single-sample blocks are never flagged (no spread to measure). Pure, so the
/// gate is unit-tested without a clock.
fn block_contended(median: u64, samples: &[u64]) -> bool {
    let Some(&min) = samples.iter().min() else {
        return false;
    };
    if median <= min {
        return false;
    }
    u128::from(median - min) * 100 > u128::from(median) * u128::from(CALIB_CONTENDED_SPREAD_PCT)
}

/// Median of the samples (upper-middle for an even count). `u64::MAX` for an empty
/// slice so an unmeasured path never looks like the fast choice.
fn median_ns(samples: &mut [u64]) -> u64 {
    if samples.is_empty() {
        return u64::MAX;
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn calibrator() -> &'static Mutex<Calibrator> {
    static CALIBRATOR: OnceLock<Mutex<Calibrator>> = OnceLock::new();
    CALIBRATOR.get_or_init(|| Mutex::new(Calibrator::new()))
}

/// The path the next adaptive-mode decode step should take (see [`Calibrator`]).
pub(crate) fn auto_choose_path() -> AutoPath {
    calibrator()
        .lock()
        .map(|calib| calib.choose())
        // A poisoned lock (a panic in a prior decode step) should never change
        // tokens or crash decode -- fall back to the safe flat path.
        .unwrap_or(AutoPath::Flat)
}

/// Feed the measured wall time of an adaptive-mode decode step back to calibration.
pub(crate) fn auto_record_sample(path: AutoPath, elapsed: Duration) {
    let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    if let Ok(mut calib) = calibrator().lock() {
        calib.record(path, ns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host that never enters an SPMD decode scope (the plugin EP) turns the
    /// pool off by default, and an explicit environment setting still wins in
    /// both directions -- otherwise opting the plugin out would take the pool
    /// away from anyone who asked for it by name.
    #[test]
    fn process_default_decides_only_the_unset_case() {
        assert_eq!(
            resolve_persistence_mode(None, true),
            PersistenceMode::On,
            "unset with the historical default builds the pool"
        );
        assert_eq!(
            resolve_persistence_mode(None, false),
            PersistenceMode::Off,
            "unset with the pool opted out leaves decode on the flat path"
        );
        assert_eq!(
            resolve_persistence_mode(Some("1"), false),
            PersistenceMode::On,
            "an explicit =1 wins over a process opt-out"
        );
        assert_eq!(
            resolve_persistence_mode(Some("auto"), false),
            PersistenceMode::Adaptive,
            "an explicit =auto wins over a process opt-out"
        );
        assert_eq!(
            resolve_persistence_mode(Some("0"), true),
            PersistenceMode::Off,
            "an explicit =0 wins over the historical default"
        );
    }

    fn two_group_pool() -> SpmdDecodePools {
        let shards = vec![
            NodeShard {
                index: 0,
                cpus: vec![],
                workers: 2,
            },
            NodeShard {
                index: 1,
                cpus: vec![],
                workers: 2,
            },
        ];
        SpmdDecodePools::build_with_schedule(&shards, DecodeSchedule::Fixed, false)
    }

    fn single_group_pool(workers: usize) -> SpmdDecodePools {
        single_group_pool_with_schedule(workers, DecodeSchedule::Fixed)
    }

    fn single_group_pool_with_schedule(
        workers: usize,
        schedule: DecodeSchedule,
    ) -> SpmdDecodePools {
        let shards = vec![NodeShard {
            index: 0,
            cpus: vec![],
            workers,
        }];
        SpmdDecodePools::build_with_schedule(&shards, schedule, false)
    }

    /// A single group of `threads` spawned workers plus a dispatcher-owned
    /// shard: the layout an explicit `ONNX_GENAI_CPU_DECODE_THREADS=threads + 1`
    /// budget produces once the headroom reservation has taken its CPU back.
    fn dispatcher_shard_pool(threads: usize) -> SpmdDecodePools {
        let shards = vec![NodeShard {
            index: 0,
            cpus: vec![],
            workers: threads,
        }];
        SpmdDecodePools::build_with_schedule(&shards, DecodeSchedule::Fixed, true)
    }

    #[test]
    fn decode_schedule_parses_env_values() {
        assert_eq!(decode_schedule_from_raw(None), DecodeSchedule::Fixed);
        assert_eq!(decode_schedule_from_raw(Some("")), DecodeSchedule::Fixed);
        assert_eq!(
            decode_schedule_from_raw(Some("fixed")),
            DecodeSchedule::Fixed
        );
        let expected_steal = if cfg!(feature = "mlas") {
            DecodeSchedule::Steal
        } else {
            DecodeSchedule::Fixed
        };
        assert_eq!(decode_schedule_from_raw(Some("steal")), expected_steal);
        assert_eq!(
            decode_schedule_from_raw(Some(" work-stealing ")),
            expected_steal
        );
        assert_eq!(
            decode_schedule_from_raw(Some("bogus")),
            DecodeSchedule::Fixed
        );
    }

    #[test]
    fn an_unset_or_unparseable_blocktime_uses_the_default() {
        assert_eq!(parse_decode_blocktime(None), DEFAULT_BLOCKTIME);
        // Every rejected spelling must land on the default rather than on zero:
        // silently parking immediately is a real behaviour change, not a
        // conservative fallback.
        for raw in [
            "",
            "   ",
            "abc",
            "-1",
            "1.5",
            "500us",
            "18446744073709551616",
        ] {
            assert_eq!(
                parse_decode_blocktime(Some(raw)),
                DEFAULT_BLOCKTIME,
                "{raw:?} is not a microsecond count and must fall back"
            );
        }
    }

    #[test]
    fn an_explicit_zero_blocktime_parks_immediately_rather_than_taking_the_default() {
        // Load-bearing, and the reason this policy is worth a test at all: the
        // sibling knob `steal_tiles_per_worker` filters `> 0` and treats zero as
        // "unset". Blocktime must not, because `0` is the one setting that makes
        // a worker park without spinning -- the maximally-polite mode used to
        // measure park/wake latency. Folding it into the default would silently
        // give every such run a 500us spin window and make the measurement
        // impossible while still appearing to work.
        assert_eq!(parse_decode_blocktime(Some("0")), Duration::ZERO);
        assert_ne!(parse_decode_blocktime(Some("0")), DEFAULT_BLOCKTIME);
    }

    #[test]
    fn a_blocktime_is_read_as_microseconds_and_tolerates_surrounding_space() {
        assert_eq!(
            parse_decode_blocktime(Some("250")),
            Duration::from_micros(250)
        );
        assert_eq!(
            parse_decode_blocktime(Some(" 250\n")),
            Duration::from_micros(250)
        );
        // Pins the unit. Reading the same digits as millis or nanos would be a
        // 1000x error in either direction and every one of the tests above would
        // still pass.
        assert_ne!(
            parse_decode_blocktime(Some("250")),
            Duration::from_millis(250)
        );
        assert_ne!(
            parse_decode_blocktime(Some("250")),
            Duration::from_nanos(250)
        );
    }

    #[test]
    fn reserve_single_group_headroom_frees_a_dispatcher_cpu_when_fully_subscribed() {
        // The pathological forced case: requested workers == allowed CPUs (e.g.
        // `taskset -c 0-31` with THREADS=32). One CPU must be reserved for the
        // inline dispatcher, so 31 workers are pinned and the highest-index
        // allowed CPU stays free (workers pin to cpus[0..workers] round-robin).
        assert_eq!(reserve_single_group_headroom(32, 32), 31);
        // Oversubscription (workers > allowed) is likewise capped to allowed - 1.
        assert_eq!(reserve_single_group_headroom(40, 32), 31);
        // Even an explicit THREADS=N on an exactly-N-CPU cpuset still reserves the
        // dispatcher core: N-1 workers, never N and never zero.
        assert_eq!(reserve_single_group_headroom(2, 2), 1);
    }

    #[test]
    fn reserve_single_group_headroom_is_a_noop_when_headroom_exists_or_affinity_unknown() {
        // Requested workers < allowed CPUs: genuine headroom already exists, so
        // the count is unchanged (the numa-split / flat paths are untouched too).
        assert_eq!(reserve_single_group_headroom(16, 32), 16);
        assert_eq!(reserve_single_group_headroom(31, 32), 31);
        // allowed_count == 0 means the allowed set is unknown; workers run
        // unpinned and cannot occupy every core, so nothing is capped.
        assert_eq!(reserve_single_group_headroom(32, 0), 32);
        assert_eq!(reserve_single_group_headroom(1, 0), 1);
    }

    #[test]
    fn reserve_split_headroom_reserves_one_cpu_per_node_only_when_fully_subscribed() {
        // Both nodes fully subscribed (workers == node CPUs): each is reduced by
        // one so every socket keeps a free core for the dispatcher, whichever
        // node it lands on. A node with existing headroom is left untouched.
        let mut shards = vec![
            NodeShard {
                index: 0,
                cpus: (0..16).collect(),
                workers: 16,
            },
            NodeShard {
                index: 1,
                cpus: (16..32).collect(),
                workers: 10,
            },
        ];
        reserve_split_headroom(&mut shards);
        assert_eq!(shards[0].workers, 15);
        assert_eq!(shards[1].workers, 10);
        // CPU lists are preserved; only the pinned-worker count shrinks, so the
        // reserved (highest-index) CPU of a capped node stays unpinned.
        assert_eq!(shards[0].cpus.len(), 16);
    }

    #[test]
    fn reserve_split_headroom_floors_at_one_worker_per_node() {
        let mut shards = vec![NodeShard {
            index: 0,
            cpus: vec![7],
            workers: 1,
        }];
        reserve_split_headroom(&mut shards);
        assert_eq!(shards[0].workers, 1);
    }

    #[test]
    fn node_row_lengths_split_proportionally_and_cover_all_rows() {
        let pool = two_group_pool();
        assert_eq!(pool.node_row_lengths(100), vec![50, 50]);
        assert_eq!(pool.node_row_lengths(101), vec![50, 51]);
        assert_eq!(pool.node_row_lengths(1), vec![0, 1]);
        assert_eq!(pool.node_row_lengths(0), vec![0, 0]);
    }

    #[test]
    fn worker_row_segments_are_disjoint_and_cover_every_row() {
        let pool = two_group_pool();
        let n = 37usize;
        let segments = pool.worker_row_segments(n);
        assert_eq!(segments.len(), pool.total_workers());
        // Contiguous, non-overlapping, covering exactly 0..n.
        let mut expected_start = 0;
        for (start, len) in &segments {
            assert_eq!(*start, expected_start);
            expected_start += len;
        }
        assert_eq!(expected_start, n);
    }

    #[test]
    fn worker_row_segments_aligned_snaps_interior_boundaries_and_covers_every_row() {
        // Every interior boundary must be a multiple of `align`; the segments
        // must still be contiguous, disjoint, and cover exactly 0..n. This is
        // the invariant the MLAS SQNBit decode shard path relies on to keep each
        // N-tile whole (and thus bit-identical to the full-width call).
        let pool = single_group_pool(3);
        for &align in &[4usize, 16] {
            for &n in &[97usize, 128, 151936, 1, 0, 5, 17] {
                let segments = pool.worker_row_segments_aligned(n, align);
                assert_eq!(segments.len(), pool.total_workers());
                let mut expected_start = 0;
                for (index, &(start, len)) in segments.iter().enumerate() {
                    assert_eq!(start, expected_start, "n={n} align={align} seg {index}");
                    // Interior boundaries (every start past the first) must be
                    // align-aligned; the final segment may end at an unaligned n.
                    assert_eq!(
                        start % align,
                        0,
                        "n={n} align={align}: segment start {start} not aligned"
                    );
                    expected_start += len;
                }
                assert_eq!(expected_start, n, "n={n} align={align}: must cover 0..n");
            }
        }
    }

    #[test]
    fn worker_row_segments_aligned_is_identity_for_align_one() {
        let pool = two_group_pool();
        for &n in &[0usize, 1, 37, 100, 101] {
            assert_eq!(
                pool.worker_row_segments_aligned(n, 1),
                pool.worker_row_segments(n),
                "align=1 must reproduce the unaligned split (n={n})"
            );
        }
    }

    #[test]
    fn work_stealing_segments_preserve_coarse_default_tiles() {
        let pool = single_group_pool_with_schedule(4, DecodeSchedule::Steal);
        let segments = pool.output_column_segments(2048, 16);
        assert!(
            segments.len() >= pool.total_workers(),
            "work-stealing mode must expose at least one stealable tile per worker"
        );
        let mut expected_start = 0;
        for (start, len) in segments {
            assert_eq!(start, expected_start);
            assert_eq!(start % 16, 0);
            assert!(len >= MIN_STEAL_OUTPUTS_PER_TASK || start + len == 2048);
            expected_start += len;
        }
        assert_eq!(expected_start, 2048);
    }

    #[test]
    fn dispatch_output_rows_fans_out_under_a_narrow_ambient_rayon_pool() {
        // The regression guard for the call site itself: `dispatch_output_rows`
        // must decide serial-vs-parallel from *its own* `total_workers`, not
        // from whatever Rayon pool happens to be installed on the calling
        // thread. Sizing it from the ambient width means a narrow ambient pool
        // (`RAYON_NUM_THREADS=1`, or any caller inside a one-thread install)
        // silently collapses the whole dispatch onto the dispatcher thread
        // while every SPMD worker sits spinning -- measured at 14.2x on the
        // qwen int4 decode loop.
        //
        // Reverting the call site to the ambient-width rule turns this test
        // red; the `output_chunk_len_for` unit tests alone would not notice.
        let pool = single_group_pool(4);
        assert!(
            pool.total_workers > 1,
            "the test pool must be wide enough to fan out"
        );
        let n = 4096usize;
        let k = 1024usize;

        let threads: std::sync::Mutex<std::collections::HashSet<thread::ThreadId>> =
            std::sync::Mutex::new(std::collections::HashSet::new());
        let compute = |output_start: usize, outputs: &mut [f32]| {
            threads
                .lock()
                .expect("thread-id set")
                .insert(thread::current().id());
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32;
            }
        };

        // The ambient pool must be genuinely one thread wide *on the thread that
        // calls the dispatch*, which is what `install` guarantees.
        let narrow = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread ambient pool");
        let mut sharded = vec![0.0f32; n];
        narrow.install(|| pool.dispatch_output_rows(&mut sharded, k, &compute));

        // A serial short-circuit runs `compute` exactly once, on one thread; a
        // fan-out runs it once per worker. Counting distinct threads therefore
        // separates the two regardless of which thread drove the dispatch.
        let observed = threads.lock().expect("thread-id set").len();
        assert!(
            observed > 1,
            "a {}-worker SPMD pool must still fan out when the ambient Rayon \
             pool is one thread wide, but `compute` ran on {observed} thread(s)",
            pool.total_workers
        );
        for (index, value) in sharded.iter().enumerate() {
            assert_eq!(*value, index as f32, "row {index} was not written once");
        }
    }

    /// The regression guard for #1746. `bound_process_to_decode_budget` confines
    /// the process to exactly `N` CPUs, so `reserve_single_group_headroom(N, N)`
    /// always fires and leaves `N-1` pinned workers. Without a dispatcher-owned
    /// shard the user's budget buys `N-1` compute lanes -- and at `N=2` that is
    /// one lane, which trips the `total_workers <= 1` serial short-circuit and
    /// makes `ONNX_GENAI_CPU_DECODE_THREADS=2` indistinguishable from `=1`.
    ///
    /// Deleting the dispatcher shard turns this red at every width.
    #[test]
    fn a_dispatcher_owned_shard_restores_the_requested_width() {
        for threads in 1..=4usize {
            let pool = dispatcher_shard_pool(threads);
            assert_eq!(
                pool.total_workers(),
                threads + 1,
                "{threads} pinned workers plus the dispatcher must be {} lanes",
                threads + 1
            );
            assert_eq!(
                pool.node_thread_counts.iter().sum::<usize>(),
                threads,
                "only {threads} threads may be spawned; the extra lane is the dispatcher"
            );
            pool.shutdown();
        }
    }

    /// The `N=2` case specifically: one pinned worker plus the dispatcher must
    /// actually execute on two threads, not take the serial short-circuit.
    #[test]
    fn a_two_lane_budget_fans_out_across_the_dispatcher_and_one_worker() {
        let pool = dispatcher_shard_pool(1);
        let n = 4096usize;
        let k = 1024usize;
        assert!(
            output_chunk_len_for(pool.total_workers(), n, k) < n,
            "the shape must be past the serial gate for this to test fan-out"
        );

        let threads: std::sync::Mutex<std::collections::HashSet<thread::ThreadId>> =
            std::sync::Mutex::new(std::collections::HashSet::new());
        let compute = |output_start: usize, outputs: &mut [f32]| {
            threads
                .lock()
                .expect("thread-id set")
                .insert(thread::current().id());
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_output_rows(&mut sharded, k, &compute);

        let observed = threads.lock().expect("thread-id set").len();
        assert_eq!(
            observed, 2,
            "a one-worker pool with a dispatcher shard must compute on two \
             threads, but `compute` ran on {observed}"
        );
        for (index, value) in sharded.iter().enumerate() {
            assert_eq!(*value, index as f32, "row {index} was not written once");
        }
        pool.shutdown();
    }

    /// The dispatcher's shard must cover its rows exactly once alongside the
    /// worker shards -- an off-by-one in the partition would silently drop or
    /// double-write rows rather than fail loudly.
    #[test]
    fn a_dispatcher_owned_shard_covers_its_rows_exactly_once() {
        for threads in 1..=4usize {
            for n in [1usize, 7, 64, 1000, 4096] {
                let pool = dispatcher_shard_pool(threads);
                let segments = pool.worker_row_segments(n);
                assert_eq!(
                    segments.len(),
                    threads + 1,
                    "one segment per lane including the dispatcher"
                );
                let mut covered = vec![0u32; n];
                for (start, len) in segments {
                    for hits in &mut covered[start..start + len] {
                        *hits += 1;
                    }
                }
                assert!(
                    covered.iter().all(|&hits| hits == 1),
                    "threads={threads} n={n}: rows must be covered exactly once"
                );
                pool.shutdown();
            }
        }
    }

    /// A panic in the *dispatcher's* shard must still complete the barrier
    /// before unwinding.
    ///
    /// The published `Job` borrows the closure off the dispatcher's stack frame
    /// and the workers are already running against it, so unwinding straight out
    /// of the inline shard would free the pointee while they still read it --
    /// a use-after-free, not merely a hang. The `catch_unwind` in `dispatch`
    /// exists for this; removing it makes this test fail under Miri.
    #[test]
    fn a_panic_in_the_dispatcher_shard_still_waits_for_the_workers() {
        let pool = dispatcher_shard_pool(2);
        let dispatcher_lane = pool.total_workers() - 1;
        let finished = Arc::new(AtomicUsize::new(0));

        let observed = {
            let finished = Arc::clone(&finished);
            let job = move |global_index: usize| {
                if global_index == dispatcher_lane {
                    panic!("dispatcher shard failed");
                }
                // Give the dispatcher every chance to unwind first.
                thread::sleep(Duration::from_millis(20));
                finished.fetch_add(1, Ordering::Release);
            };
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.dispatch(&job)))
        };

        assert!(
            observed.is_err(),
            "the dispatcher's panic must propagate to the caller"
        );
        assert_eq!(
            finished.load(Ordering::Acquire),
            dispatcher_lane,
            "every worker shard must have completed before the panic escaped"
        );
        pool.shutdown();
    }

    /// The dispatcher shard is only claimed on the single-group layout, and only
    /// when the headroom reservation actually took a worker away. A budget with
    /// spare CPUs must be unchanged, or it would over-subscribe the request.
    /// `decode_width` must never force the pool to build. Reading it is what a
    /// harness does *before* deciding whether the run is trustworthy, so if the
    /// read itself resolved the path it would both change process behaviour and
    /// destroy the "not resolved yet" signal that keeps an early assertion from
    /// passing vacuously.
    #[test]
    fn reading_the_decode_width_does_not_resolve_the_decode_path() {
        let before = POOLS.get().is_some();
        let width = decode_width();
        assert_eq!(
            POOLS.get().is_some(),
            before,
            "decode_width must not build the pool as a side effect"
        );
        if !before {
            assert_eq!(width.realized, None);
            assert!(
                !width.is_as_requested(),
                "an unresolved width must not report as satisfied, or a harness \
                 that asserts too early passes for the wrong reason"
            );
        }
    }

    /// The whole point of the read: a request that was silently reduced must not
    /// report as satisfied. Covers the reduction Roy's `t=N` rows could not see.
    #[test]
    fn a_reduced_width_does_not_report_as_requested() {
        let reduced = DecodeWidth {
            requested: Some(16),
            realized: Some(15),
            path: "spmd-pool",
        };
        assert!(
            !reduced.is_as_requested(),
            "16 requested and 15 realized is exactly the case a t=16 row would \
             otherwise label as a width-16 measurement"
        );

        let honored = DecodeWidth {
            requested: Some(16),
            realized: Some(16),
            path: "spmd-pool",
        };
        assert!(honored.is_as_requested());

        // Dropping to the flat path is the other silent reduction: the width
        // may even look larger, but it is not the pool that was asked for.
        let flat = DecodeWidth {
            requested: Some(2),
            realized: Some(32),
            path: "flat",
        };
        assert!(!flat.is_as_requested());
    }

    /// An unresolved width must stay unsatisfied no matter which half is
    /// missing, so a harness cannot pass by asserting before it has decoded.
    #[test]
    fn a_half_known_width_is_never_satisfied() {
        for (requested, realized) in [(Some(4), None), (None, Some(4)), (None, None)] {
            let width = DecodeWidth {
                requested,
                realized,
                path: "unresolved",
            };
            assert!(
                !width.is_as_requested(),
                "requested={requested:?} realized={realized:?} must not report satisfied"
            );
        }
    }

    #[test]
    fn the_dispatcher_only_claims_a_shard_when_headroom_was_reserved() {
        let group = |workers| {
            vec![NodeShard {
                index: 0,
                cpus: vec![],
                workers,
            }]
        };

        // Fully subscribed: `reserve_single_group_headroom` gave up a worker to
        // keep the dispatcher's CPU free, so the dispatcher computes that lane.
        assert_eq!(reserve_single_group_headroom(4, 4), 3);
        assert!(dispatcher_owns_a_shard(&group(3), 4));

        // Headroom already existed: no CPU was reserved, so claiming a shard
        // would make the pool one lane wider than the budget allows.
        assert_eq!(reserve_single_group_headroom(4, 32), 4);
        assert!(!dispatcher_owns_a_shard(&group(4), 4));

        // A NUMA split keeps the previous behavior: the dispatcher's node is not
        // known at build time, so its shard could land cross-socket.
        let split = vec![
            NodeShard {
                index: 0,
                cpus: vec![],
                workers: 3,
            },
            NodeShard {
                index: 1,
                cpus: vec![],
                workers: 3,
            },
        ];
        assert!(!dispatcher_owns_a_shard(&split, 8));

        let pool = single_group_pool(4);
        assert_eq!(pool.total_workers(), 4);
        assert!(pool.dispatcher_shard.is_none());
        pool.shutdown();
    }

    const BUDGET_LANE_CHILD_ENV: &str = "ONNX_GENAI_TEST_BUDGET_LANE_CHILD";
    const BUDGET_LANE_CONFINE_ENV: &str = "ONNX_GENAI_TEST_BUDGET_LANE_CONFINE";
    const BUDGET_LANE_MARKER: &str = "BUDGET_LANE_RESULT=";
    const BUDGET_LANE_SKIP_MARKER: &str = "BUDGET_LANE_SKIP=";

    /// End-to-end guard for #1746, across the process boundary the defect lived
    /// in.
    ///
    /// The off-by-one was not in either half. `bound_process_to_decode_budget`
    /// correctly confines the process to `N` CPUs, and
    /// `reserve_single_group_headroom` correctly keeps one allowed CPU free for
    /// the dispatcher. It only appears when they *compose*: the confinement
    /// makes the group fully subscribed, so the reservation always fires and the
    /// budget silently buys `N-1` lanes.
    ///
    /// A subprocess per budget is required, not stylistic. Both halves latch --
    /// `PROCESS_BUDGET_BOUND` is a `OnceLock` and the pool is a `OnceLock` --
    /// and the child mutates process-wide CPU affinity, which would poison the
    /// test runner. Unit tests on either half in isolation cannot see this;
    /// that is precisely how it shipped.
    #[test]
    fn an_explicit_budget_buys_its_full_width_end_to_end() {
        if std::env::var_os(BUDGET_LANE_CHILD_ENV).is_some() {
            return; // The child arm runs as its own test below.
        }
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        // A budget of 1 confines the process to one CPU, which `build_from_env`
        // deliberately declines (no core left for the dispatcher), so the
        // smallest budget that builds a pool is 2.
        let budgets: Vec<usize> = [2usize, 4, 8]
            .into_iter()
            .filter(|&n| n <= available)
            .collect();
        if budgets.is_empty() {
            eprintln!("skipped: needs >= 2 CPUs, host reports {available}");
            return;
        }

        for budget in budgets {
            let lane = run_budget_lane_child(budget, None).expect("the unconfined arm never skips");
            let (allowed, nodes, threads, lanes) =
                (lane.allowed, lane.nodes, lane.threads, lane.lanes);
            let (requested, realized, path) = (lane.requested, lane.realized, lane.path.as_str());
            if nodes != 1 {
                // A budget that straddles NUMA nodes takes the split path, which
                // keeps the previous per-node reservation. Not this test's case.
                eprintln!("budget {budget}: skipped, {nodes}-node split layout");
                continue;
            }
            // The introspection must agree with the pool it describes, on the
            // real production path. A `t=N` row is only a width-N measurement if
            // this holds, which is the whole reason the read exists.
            assert_eq!(
                (requested, realized, path),
                (Some(budget), Some(budget), "spmd-pool"),
                "budget {budget}: decode_width must report the realized width and \
                 path, got requested={requested:?} realized={realized:?} path={path}"
            );
            assert_eq!(
                lanes, budget,
                "ONNX_GENAI_CPU_DECODE_THREADS={budget} must buy {budget} compute lanes, \
                 got {lanes} ({threads} pinned threads on {allowed} allowed CPUs)"
            );
            assert!(
                threads >= 1 && threads <= budget,
                "budget {budget}: {threads} pinned threads is outside the budget"
            );
            if allowed == budget {
                // The confinement landed, so the reservation must have fired and
                // the dispatcher must be covering the lane it freed.
                assert_eq!(
                    threads,
                    budget - 1,
                    "budget {budget}: a fully-subscribed group must leave one CPU \
                     free for the dispatcher"
                );
            }
        }
    }

    /// One decoded `BUDGET_LANE_MARKER` line from a lane child.
    struct BudgetLane {
        allowed: usize,
        nodes: usize,
        threads: usize,
        lanes: usize,
        requested: Option<usize>,
        realized: Option<usize>,
        path: String,
    }

    /// Spawns the lane child at `budget`, optionally pre-confining it to
    /// `confine` CPUs, and decodes its one result line.
    fn run_budget_lane_child(budget: usize, confine: Option<usize>) -> Option<BudgetLane> {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.arg("--exact")
            .arg("decode_spmd::tests::budget_lane_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .arg("--ignored")
            .env(BUDGET_LANE_CHILD_ENV, budget.to_string())
            .env(
                crate::kernels::matmul_nbits::DECODE_THREADS_ENV,
                budget.to_string(),
            )
            .env(PERSISTENT_POOL_ENV, "1")
            // The child asserts an exact path label, so any ambient variable that
            // can change which path is chosen has to be cleared, not inherited.
            // `steal` would relabel the path "work-stealing-pool" and fail the
            // assertion for a reason unrelated to decode width.
            .env_remove(DECODE_SCHEDULE_ENV)
            .env_remove(crate::decode_affinity::DECODE_AFFINITY_ENV);
        if let Some(confine) = confine {
            cmd.env(BUDGET_LANE_CONFINE_ENV, confine.to_string());
        } else {
            cmd.env_remove(BUDGET_LANE_CONFINE_ENV);
        }
        let output = cmd.output().expect("run decode-budget lane child");
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        assert!(
            output.status.success(),
            "budget {budget} child failed ({}):\nstdout:\n{stdout}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        if let Some(reason) = stdout.lines().find_map(|line| {
            line.split_once(BUDGET_LANE_SKIP_MARKER)
                .map(|(_, r)| r.trim())
        }) {
            eprintln!("budget {budget}: child skipped: {reason}");
            return None;
        }
        let encoded = stdout
            .lines()
            .find_map(|line| {
                line.split_once(BUDGET_LANE_MARKER)
                    .map(|(_, rest)| rest.trim())
            })
            .unwrap_or_else(|| panic!("budget {budget} child emitted no result:\n{stdout}"));
        let fields: Vec<&str> = encoded.split(',').collect();
        assert_eq!(fields.len(), 7, "malformed lane result {encoded:?}");
        let num = |i: usize| fields[i].parse::<usize>().unwrap();
        // The child encodes an absent width as 0, which is never a valid width.
        let opt = |i: usize| Some(num(i)).filter(|&n| n != 0);
        Some(BudgetLane {
            allowed: num(0),
            nodes: num(1),
            threads: num(2),
            lanes: num(3),
            requested: opt(4),
            realized: opt(5),
            path: fields[6].trim().to_owned(),
        })
    }

    /// The counterpart to [`an_explicit_budget_buys_its_full_width_end_to_end`]:
    /// when the host *cannot* honour the request, the read must say so.
    ///
    /// This is the case that makes the introspection worth having. A published
    /// `t=N` row is a label, not a measurement, and the ways a request gets
    /// silently reduced -- `reserve_single_group_headroom`,
    /// `reserve_split_headroom`, and the single-CPU cpuset branch that drops
    /// decode to the flat path -- all report only through `report_spmd_fallback`,
    /// which is `tracing::debug!`/`NXRT_CALIB_DEBUG`-gated and therefore invisible
    /// in a default benchmark run.
    ///
    /// Requesting 8 lanes inside a 2-CPU cpuset is that situation made
    /// deterministic. The exact realized width is a policy detail this test
    /// deliberately does not pin; what it pins is that the read never claims the
    /// request was honoured when it was not.
    #[test]
    fn a_budget_the_host_cannot_honour_never_reports_as_requested() {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        const CONFINE: usize = 2;
        const REQUEST: usize = 8;
        if available < REQUEST {
            eprintln!("skipped: needs >= {REQUEST} CPUs, host reports {available}");
            return;
        }
        let Some(lane) = run_budget_lane_child(REQUEST, Some(CONFINE)) else {
            return; // No process-wide affinity mask on this host.
        };
        assert!(
            lane.allowed <= CONFINE,
            "the confinement must land, got {} allowed CPUs",
            lane.allowed
        );
        assert_eq!(
            lane.requested,
            Some(REQUEST),
            "the requested width must survive the reduction -- without it there is \
             nothing to compare the realized width against"
        );
        assert!(
            lane.realized.is_none_or(|r| r < REQUEST),
            "a {CONFINE}-CPU cpuset cannot deliver {REQUEST} lanes, but the read \
             reported {:?}",
            lane.realized
        );
        // The whole contract in one line: this must be false whenever the label
        // and the reality disagree.
        let width = DecodeWidth {
            requested: lane.requested,
            realized: lane.realized,
            path: "",
        };
        assert!(
            !width.is_as_requested(),
            "requested={:?} realized={:?} must not report as requested",
            lane.requested,
            lane.realized
        );
        // And the pool, if one was built, must agree with what the read said.
        if lane.path == "spmd-pool" {
            assert_eq!(
                lane.realized,
                Some(lane.lanes),
                "the read must report the pool's own lane count"
            );
        }
        eprintln!(
            "requested={REQUEST} in a {CONFINE}-CPU cpuset -> allowed={} realized={:?} path={}",
            lane.allowed, lane.realized, lane.path
        );
    }

    /// The child arm of [`an_explicit_budget_buys_its_full_width_end_to_end`].
    /// `#[ignore]`d so only the parent's explicit `--ignored --exact` run
    /// executes it; it confines the process's CPU affinity and must not run in
    /// the shared runner.
    #[test]
    #[ignore = "spawned by an_explicit_budget_buys_its_full_width_end_to_end"]
    fn budget_lane_child() {
        let Some(budget) = std::env::var_os(BUDGET_LANE_CHILD_ENV) else {
            return;
        };
        let budget: usize = budget.to_string_lossy().parse().expect("budget");
        // Optional pre-confinement, used by the reduced-width arm to make the
        // host unable to honour the request. Applied before
        // `bound_process_to_decode_budget` so the budget logic sees the smaller
        // cpuset, exactly as an external `taskset` or container limit would
        // present it.
        if let Some(confined) = std::env::var_os(BUDGET_LANE_CONFINE_ENV) {
            let confined: usize = confined.to_string_lossy().parse().expect("confine");
            let cpus: Vec<usize> = (0..confined).collect();
            if let Err(err) = crate::decode_affinity::set_current_thread_affinity(&cpus) {
                // Only Linux implements a process-wide mask, so on other hosts
                // there is no way to manufacture the reduction. Report a skip
                // rather than failing: the arm is about the read's behaviour
                // under reduction, not about affinity support.
                println!("{BUDGET_LANE_SKIP_MARKER}{err}");
                return;
            }
        }
        // `pools()` reads the budget from the environment, so confirm the env the
        // parent set agrees with the budget it asked for. Checked against the
        // *unclamped* accessor, since under confinement the resolved width is
        // deliberately smaller than the request.
        assert_eq!(
            crate::kernels::matmul_nbits::decode_thread_budget(),
            Some(budget),
            "the child's environment must carry the budget the parent requested"
        );
        // Exactly the production sequence: EP `initialize()` bounds the process,
        // then the pool is built lazily at first decode.
        crate::kernels::matmul_nbits::bound_process_to_decode_budget();
        let allowed = crate::decode_affinity::allowed_cpus().map_or(0, |c| c.len());
        // Deliberately the global `pools()` entry rather than `build_from_env`
        // directly: that is what a real decode reaches, and it is the only way
        // `decode_width` sees a resolved path -- so this arm checks the
        // introspection a harness will assert on, not a parallel construction of
        // it that could agree while production disagrees.
        let pool = pools();
        let (nodes, threads, lanes) = pool.map_or((0, 0, 0), |p| {
            (
                p.node_count(),
                p.node_thread_counts.iter().sum::<usize>(),
                p.total_workers(),
            )
        });
        let width = decode_width();
        println!(
            "{BUDGET_LANE_MARKER}{allowed},{nodes},{threads},{lanes},{},{},{}",
            width.requested.unwrap_or(0),
            width.realized.unwrap_or(0),
            width.path,
        );
        shutdown_pools();
    }

    #[test]
    fn dispatch_output_rows_matches_flat_computation() {
        let pool = two_group_pool();
        let n = 101usize;
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32 * 2.5 - 3.0;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_rows_across_workers(&mut sharded, &compute);
        let mut flat = vec![0.0f32; n];
        compute(0, &mut flat);
        assert_eq!(sharded, flat);
    }

    #[test]
    fn work_stealing_dispatch_output_rows_matches_flat_computation() {
        let pool = single_group_pool_with_schedule(4, DecodeSchedule::Steal);
        let n = 2048usize;
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                let row = output_start + offset;
                *out = row as f32 * 1.25 - 7.0;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_output_rows(&mut sharded, 1024, &compute);
        let mut flat = vec![0.0f32; n];
        compute(0, &mut flat);
        assert_eq!(sharded, flat);
    }

    #[test]
    fn dispatch_preserves_per_row_reduction_bit_for_bit() {
        // Mirror the real GEMV: each output row is a full-K f32 dot product. Row
        // sharding must not reorder the per-row accumulation, so the SPMD result
        // must be *byte-for-byte* identical to a single-threaded reference (this
        // is the parity invariant the greedy-token equality relies on).
        let pool = two_group_pool();
        let n = 257usize;
        let k = 320usize;
        // Deterministic pseudo-random-ish weights/activation, mixed signs/scales.
        let activation: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.031_25)
            .collect();
        let weight = |row: usize, col: usize| -> f32 {
            (((row * 131 + col * 17) % 251) as f32 - 125.0) * 0.007_812_5
        };
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                let row = output_start + offset;
                let mut acc = 0.0f32;
                for (col, &a) in activation.iter().enumerate() {
                    acc += a * weight(row, col);
                }
                *out = acc;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_rows_across_workers(&mut sharded, &compute);
        let mut reference = vec![0.0f32; n];
        compute(0, &mut reference);
        assert_eq!(
            sharded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "row-sharded dispatch must be bit-identical to the serial reference"
        );
    }

    #[test]
    fn dispatch_output_row_blocks_matches_flat_computation() {
        // Fixed-width row blocks (mirrors GroupQueryAttention's per-head output
        // rows): every `row_len`-element row is computed whole on one worker, so
        // the sharded result must equal the single-threaded reference row-for-row
        // and bit-for-bit (rows are independent).
        for (num_rows, row_len) in [
            (28usize, 128usize),
            (3, 128),
            (1, 64),
            (5, 3),
            (37, 1),
            (0, 8),
        ] {
            let pool = two_group_pool();
            let compute = |row_index: usize, row: &mut [f32]| {
                for (offset, out) in row.iter_mut().enumerate() {
                    // Order-sensitive accumulation to catch any row reordering.
                    let mut acc = 0.0f32;
                    for step in 0..=offset {
                        acc += (row_index * 7 + step) as f32 * 0.015_625 - 1.0;
                    }
                    *out = acc;
                }
            };
            let mut sharded = vec![0.0f32; num_rows * row_len];
            pool.dispatch_output_row_blocks(&mut sharded, row_len, num_rows, &compute);
            let mut reference = vec![0.0f32; num_rows * row_len];
            for row in 0..num_rows {
                compute(row, &mut reference[row * row_len..(row + 1) * row_len]);
            }
            assert_eq!(
                sharded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row-block dispatch must be bit-identical to the serial reference \
                 (num_rows={num_rows}, row_len={row_len})"
            );
        }
    }

    #[test]
    fn dispatch_is_reusable_across_many_ops() {
        // Exercises the barrier repeatedly: every worker must re-arm and the
        // dispatcher must observe completion each time (regression guard for
        // the sequence/pending protocol).
        let pool = single_group_pool(4);
        for round in 0..200usize {
            let n = 53usize;
            let compute = move |output_start: usize, outputs: &mut [f32]| {
                for (offset, out) in outputs.iter_mut().enumerate() {
                    *out = (round * 1000 + output_start + offset) as f32;
                }
            };
            let mut got = vec![0.0f32; n];
            pool.dispatch_rows_across_workers(&mut got, &compute);
            let mut want = vec![0.0f32; n];
            compute(0, &mut want);
            assert_eq!(got, want, "round {round}");
        }
    }

    #[test]
    fn build_then_immediate_dispatch_never_hangs() {
        // Regression guard: a dispatch issued right after `build` must not race
        // a not-yet-started worker (which would hang the barrier). Rebuild a
        // fresh pool and dispatch across all workers immediately, many times.
        for _ in 0..40usize {
            let pool = single_group_pool(6);
            let n = 61usize;
            let compute = |output_start: usize, outputs: &mut [f32]| {
                for (offset, out) in outputs.iter_mut().enumerate() {
                    *out = (output_start + offset) as f32;
                }
            };
            let mut got = vec![-1.0f32; n];
            pool.dispatch_rows_across_workers(&mut got, &compute);
            let mut want = vec![0.0f32; n];
            compute(0, &mut want);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn place_rows_preserves_bytes() {
        let pool = two_group_pool();
        let n = 7usize;
        let stride = 4usize;
        let src: Vec<u8> = (0..(n * stride) as u8).collect();
        assert_eq!(pool.place_rows(&src, n), src);

        let scales: Vec<f32> = (0..n).map(|row| row as f32 * 0.5).collect();
        assert_eq!(pool.place_rows(&scales, n), scales);
    }

    #[test]
    fn tiny_ops_run_serially_but_correctly() {
        // Below the parallelization threshold the op runs on the dispatcher;
        // the result must still be correct.
        let pool = single_group_pool(8);
        let n = 3usize;
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32;
            }
        };
        let mut got = vec![0.0f32; n];
        pool.dispatch_output_rows(&mut got, 4096, &compute);
        assert_eq!(got, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn panicking_worker_poison_is_reported_without_hanging() {
        let pool = single_group_pool(4);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.dispatch(&|worker| {
                assert_ne!(worker, 2, "intentional SPMD worker panic");
            });
        }));
        let panic = result.expect_err("dispatcher must report a worker panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            message.contains("persistent SPMD decode worker 2 panicked")
                && message.contains("pool is poisoned"),
            "unexpected dispatcher diagnostic: {message}"
        );
    }

    #[test]
    fn persistence_mode_parses_env_values() {
        // Mode parsing (pool is the default): unset -> On (pool), `0` -> Off
        // (flat path), `1` -> On (pool), `auto` -> Adaptive (calibrated).
        // Whitespace is trimmed; unrecognized values map to On (the pool default).
        assert_eq!(persistence_mode_from_raw(None), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some("")), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some("   ")), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some("0")), PersistenceMode::Off);
        assert_eq!(persistence_mode_from_raw(Some(" 0 ")), PersistenceMode::Off);
        assert_eq!(persistence_mode_from_raw(Some("1")), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some(" 1 ")), PersistenceMode::On);
        // `auto` (case-insensitive) enables load-adaptive calibration.
        assert_eq!(
            persistence_mode_from_raw(Some("auto")),
            PersistenceMode::Adaptive
        );
        assert_eq!(
            persistence_mode_from_raw(Some(" Auto ")),
            PersistenceMode::Adaptive
        );
        assert_eq!(
            persistence_mode_from_raw(Some("AUTO")),
            PersistenceMode::Adaptive
        );
        // Unknown values map to On (pool), never a surprise flat path.
        assert_eq!(persistence_mode_from_raw(Some("true")), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some("2")), PersistenceMode::On);

        // The pool is BUILT for `On` (default/`=1`) and `Adaptive` (`=auto`);
        // only `Off` (`=0`) skips building. On is forced (no calibration).
        assert!(pool_mode_builds(persistence_mode_from_raw(Some("1"))));
        assert!(pool_mode_builds(persistence_mode_from_raw(None)));
        assert!(pool_mode_builds(persistence_mode_from_raw(Some("auto"))));
        assert!(!pool_mode_builds(persistence_mode_from_raw(Some("0"))));
        assert!(pool_mode_forces(persistence_mode_from_raw(Some("1"))));
        assert!(pool_mode_forces(persistence_mode_from_raw(None)));
        assert!(!pool_mode_forces(persistence_mode_from_raw(Some("auto"))));
    }

    #[test]
    fn default_selects_pool_without_probing() {
        // The default (unset) must deterministically select the pool with no
        // calibration — the core guarantee of the opt-in adaptive design.
        let mode = persistence_mode_from_raw(None);
        assert_eq!(mode, PersistenceMode::On, "default must be On (pool)");
        assert!(
            pool_mode_forces(mode),
            "default must force the pool (no calibration)"
        );
        assert!(pool_mode_builds(mode), "default must build the pool");
    }

    #[test]
    fn adaptive_flag_enables_calibration() {
        // `=auto` must enable adaptive calibration: build the pool but do NOT
        // force it — the calibrator decides.
        let mode = persistence_mode_from_raw(Some("auto"));
        assert_eq!(mode, PersistenceMode::Adaptive);
        assert!(
            pool_mode_builds(mode),
            "adaptive must build the pool for calibration"
        );
        assert!(
            !pool_mode_forces(mode),
            "adaptive must NOT force the pool — the calibrator picks"
        );
    }

    #[test]
    fn on_and_adaptive_build_the_pool_but_only_on_dispatches_unconditionally() {
        // The pool is built for `On` (the default) and `Adaptive` (`=auto`);
        // `Off` (`=0`) never builds it. `On` always dispatches (no calibration);
        // `Adaptive` lets the calibrator pick.
        assert!(pool_mode_builds(PersistenceMode::On));
        assert!(pool_mode_builds(PersistenceMode::Adaptive));
        assert!(!pool_mode_builds(PersistenceMode::Off));

        assert!(pool_mode_forces(PersistenceMode::On));
        assert!(!pool_mode_forces(PersistenceMode::Adaptive));
        assert!(!pool_mode_forces(PersistenceMode::Off));

        // The default env value (unset) maps to On: built, forced, not calibrated.
        assert!(pool_mode_builds(persistence_mode_from_raw(None)));
        assert!(pool_mode_forces(persistence_mode_from_raw(None)));
        assert!(!pool_mode_builds(persistence_mode_from_raw(Some("0"))));
        assert!(pool_mode_builds(persistence_mode_from_raw(Some("2"))));
        assert!(pool_mode_forces(persistence_mode_from_raw(Some("1"))));
    }

    /// Drive the calibrator from its current phase until it commits, feeding each
    /// step the per-path time it chose. Returns once the committed phase is reached.
    fn drive_to_commit(calib: &mut Calibrator, pool_ns: u64, flat_ns: u64) {
        for _ in 0..100_000 {
            if calib.phase == CalibPhase::Committed {
                return;
            }
            let path = calib.choose();
            let ns = match path {
                AutoPath::Pool => pool_ns,
                AutoPath::Flat => flat_ns,
            };
            calib.record(path, ns);
        }
        panic!("calibrator never reached the committed phase");
    }

    /// Fresh calibrator driven through warmup + one probe with the given per-path
    /// step times; returns the committed decision.
    fn run_one_probe(pool_ns: u64, flat_ns: u64) -> Calibrator {
        let mut calib = Calibrator::new();
        drive_to_commit(&mut calib, pool_ns, flat_ns);
        calib
    }

    #[test]
    fn calibrator_defaults_to_flat_before_any_measurement() {
        // The no-regression baseline: a fresh calibrator's committed path is the
        // flat path, so a host that never lets the pool win runs exactly today's
        // flat decode path.
        let calib = Calibrator::new();
        assert_eq!(calib.committed, AutoPath::Flat);
    }

    #[test]
    fn calibrator_probe_measures_flat_block_before_pool_block() {
        // Warmup runs on the pool (node-local placement), then the flat block is
        // measured first (pool quiesced), then the pool block -- never interleaved,
        // so a hot pool step cannot pollute a flat measurement.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            assert_eq!(calib.choose(), AutoPath::Pool);
            assert_eq!(calib.phase, CalibPhase::Warmup);
            calib.record(AutoPath::Pool, 1_000);
        }
        assert_eq!(calib.phase, CalibPhase::ProbeFlat);
        // The whole flat block chooses Flat.
        while calib.phase == CalibPhase::ProbeFlat {
            assert_eq!(calib.choose(), AutoPath::Flat);
            calib.record(AutoPath::Flat, 100);
        }
        assert_eq!(calib.phase, CalibPhase::ProbePool);
        // The whole pool block chooses Pool.
        while calib.phase == CalibPhase::ProbePool {
            assert_eq!(calib.choose(), AutoPath::Pool);
            calib.record(AutoPath::Pool, 100);
        }
        assert_eq!(calib.phase, CalibPhase::Committed);
    }

    #[test]
    fn calibrator_probe_discards_the_transition_sample() {
        // The first sample of each block is discarded so the other path's
        // winding-down threads do not pollute it: only CALIB_PROBE_SAMPLES land.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 1);
        }
        assert_eq!(calib.phase, CalibPhase::ProbeFlat);
        assert_eq!(calib.discard_left, CALIB_PROBE_DISCARD);
        while calib.phase == CalibPhase::ProbeFlat {
            calib.record(AutoPath::Flat, 100);
        }
        assert_eq!(calib.flat_ns.len(), CALIB_PROBE_SAMPLES);
    }

    #[test]
    fn calibrator_commits_pool_only_when_clearly_faster() {
        // Pool 20% faster than flat clears the 8% hysteresis margin -> adopt pool.
        let calib = run_one_probe(80, 100);
        assert_eq!(calib.phase, CalibPhase::Committed);
        assert_eq!(calib.committed, AutoPath::Pool);
        assert_eq!(calib.choose(), AutoPath::Pool);
    }

    #[test]
    fn calibrator_stays_flat_when_pool_slower_simulating_contention() {
        // Regression guard: under (simulated) co-tenant load the spinning pool is
        // slower, so its median probe time loses and the flat path stays
        // committed -- decode behaves exactly like today's flat path. This is the
        // property that makes "never regress under load" a measured guarantee.
        let calib = run_one_probe(200, 100);
        assert_eq!(calib.committed, AutoPath::Flat);
        assert_eq!(calib.choose(), AutoPath::Flat);
    }

    #[test]
    fn calibrator_stays_flat_within_the_hysteresis_margin() {
        // Pool only ~5% faster (< 8% margin): not worth switching, avoids flapping
        // and keeps the safe flat default.
        let calib = run_one_probe(95, 100);
        assert_eq!(calib.committed, AutoPath::Flat);
    }

    #[test]
    fn calibrator_stays_committed_permanently_after_initial_probe() {
        // Adopt the pool on a quiet probe, then verify the path stays frozen
        // indefinitely — no re-probe, no mid-generation switching.
        let mut calib = run_one_probe(80, 100);
        assert_eq!(calib.committed, AutoPath::Pool);
        // Feed many more steps than the old CALIB_RECAL_PERIOD (600):
        // the path must never leave Committed.
        for _ in 0..2000 {
            assert_eq!(calib.choose(), AutoPath::Pool);
            calib.record(AutoPath::Pool, 80);
            assert_eq!(calib.phase, CalibPhase::Committed);
        }
    }

    #[test]
    fn calibrator_probe_median_rejects_a_single_load_spike() {
        // The pool is genuinely faster (median 80) but one probe sample spikes
        // under a transient stall; the median ignores the outlier and still adopts
        // the pool, so a single blip does not cost the win.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 80);
        }
        // Flat block: all 100.
        while calib.phase == CalibPhase::ProbeFlat {
            calib.record(AutoPath::Flat, 100);
        }
        // Pool block: mostly fast with one spike; the (discarded) transition
        // sample plus four 80s and one huge spike keep the median at 80.
        let mut pool_samples = [80u64, 80, 80, 80, 80, 100_000].into_iter();
        while calib.phase == CalibPhase::ProbePool {
            calib.record(AutoPath::Pool, pool_samples.next().unwrap_or(80));
        }
        assert_eq!(calib.committed, AutoPath::Pool);
    }

    /// Drive a fresh calibrator through warmup + a flat block (all `flat`) + a pool
    /// block whose *stored* samples are `pool_block` (a leading discard sample is
    /// fed automatically). Returns the calibrator after the single probe resolves
    /// (either committed or bounced back to `ProbeFlat` by a re-probe).
    fn probe_once(flat: u64, pool_block: &[u64]) -> Calibrator {
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 1_000);
        }
        while calib.phase == CalibPhase::ProbeFlat {
            calib.record(AutoPath::Flat, flat);
        }
        // The first pool record is the discarded transition sample; the rest are
        // the block's stored samples.
        let mut feed = std::iter::once(9_999u64).chain(pool_block.iter().copied());
        while calib.phase == CalibPhase::ProbePool {
            calib.record(
                AutoPath::Pool,
                feed.next().unwrap_or(pool_block[pool_block.len() - 1]),
            );
        }
        calib
    }

    #[test]
    fn block_contended_flags_a_nonuniform_block_only() {
        // Uniformly fast (idle) or uniformly slow (sustained load) blocks have a
        // tiny fast-half spread and are NOT flagged; a burst that inflates some
        // samples above the min pushes (median - min)/median over the threshold.
        assert!(!block_contended(100, &[100, 100, 100, 100, 100]));
        assert!(!block_contended(300, &[300, 300, 300, 300, 300])); // sustained load
        assert!(!block_contended(100, &[100, 100, 100, 100, 100_000])); // lone high outlier
        assert!(block_contended(420, &[300, 300, 420, 420, 420])); // 28% spread
        // Empty / single-sample blocks have no spread to measure.
        assert!(!block_contended(u64::MAX, &[]));
        assert!(!block_contended(50, &[50]));
    }

    #[test]
    fn calibrator_reprobes_a_burst_contaminated_pool_block_instead_of_committing() {
        // The observed failure: a transient co-tenant burst lands on part of the
        // pool block, inflating its median above the pool's true (fast) cost. The
        // calibrator must NOT lock in a decision from that block -- it re-probes.
        let mut calib = probe_once(100, &[300, 300, 420, 420, 420]);
        assert_eq!(
            calib.phase,
            CalibPhase::ProbeFlat,
            "must re-probe, not commit"
        );
        assert_eq!(calib.reprobes_left, CALIB_MAX_REPROBES - 1);
        // A clean re-probe now measures the pool as genuinely faster and adopts it,
        // recovering the throughput the poisoned probe would have thrown away.
        drive_to_commit(&mut calib, 80, 100);
        assert_eq!(calib.committed, AutoPath::Pool);
    }

    #[test]
    fn calibrator_commits_flat_under_uniform_sustained_load_without_reprobing() {
        // No-regression guarantee: genuine sustained load makes every pool step
        // uniformly slow (low spread), so the block is NOT treated as a burst -- the
        // calibrator commits flat immediately, exactly as before this robustness
        // change. Sustained load can never be misread as a transient burst.
        let calib = probe_once(100, &[300, 300, 300, 300, 300]);
        assert_eq!(calib.phase, CalibPhase::Committed);
        assert_eq!(calib.committed, AutoPath::Flat);
        assert_eq!(
            calib.reprobes_left, CALIB_MAX_REPROBES,
            "no re-probe was spent"
        );
    }

    #[test]
    fn calibrator_reprobe_budget_is_bounded_and_then_commits() {
        // A pathologically noisy host that never yields a clean probe must still
        // commit within CALIB_MAX_REPROBES re-probes (here on the median, which is
        // pool-slow -> flat), so re-probing can never loop or stall decode.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 1_000);
        }
        let mut reprobes = 0u32;
        // Keep feeding contended blocks; count the re-probes until it commits.
        for _ in 0..(CALIB_MAX_REPROBES + 5) {
            while calib.phase == CalibPhase::ProbeFlat {
                calib.record(AutoPath::Flat, 100);
            }
            let mut feed = [9_999u64, 300, 300, 420, 420, 420].into_iter();
            while calib.phase == CalibPhase::ProbePool {
                calib.record(AutoPath::Pool, feed.next().unwrap_or(420));
            }
            if calib.phase == CalibPhase::Committed {
                break;
            }
            reprobes += 1;
        }
        assert_eq!(calib.phase, CalibPhase::Committed);
        assert_eq!(reprobes, CALIB_MAX_REPROBES);
        assert_eq!(calib.committed, AutoPath::Flat);
    }

    #[test]
    fn median_ns_picks_the_middle_and_guards_empty() {
        assert_eq!(median_ns(&mut []), u64::MAX);
        assert_eq!(median_ns(&mut [5]), 5);
        assert_eq!(median_ns(&mut [30, 10, 20]), 20);
        assert_eq!(median_ns(&mut [10, 40, 20, 30]), 30);
    }
}

/// The pool exposes one job slot and one barrier. Before the claim gate, two
/// threads dispatching at once overwrote each other's job pointer and each
/// other's pending counts. Both failure modes were observed by deleting the
/// gate and re-running these tests: shards running the *other* thread's
/// closure (silently wrong tensors, no crash), and -- more often -- the
/// pending counters never draining, hanging `wait` forever. The pool spins in
/// `wait`, so the hang burns every core until the process is killed.
///
/// Only the last two tests here are falsifiers for the gate itself: the first
/// three hold no contention, so they pass with or without it. They cover the
/// RAII guard's own contract (exclusivity, release-on-unwind) and the ordinary
/// single-dispatcher partition, which is worth pinning separately.
#[cfg(test)]
mod dispatch_claim_tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicU32;

    /// A dispatch whose shards all run, exactly once each, is the property the
    /// single job slot silently violated under a concurrent second dispatcher.
    fn assert_each_shard_ran_once(counts: &[AtomicU32]) {
        for (index, count) in counts.iter().enumerate() {
            assert_eq!(
                count.load(Ordering::Relaxed),
                1,
                "shard {index} ran {} times, expected exactly 1",
                count.load(Ordering::Relaxed)
            );
        }
    }

    #[test]
    fn an_uncontended_dispatch_releases_the_claim_for_the_next_one() {
        let shared = SharedState {
            node_sense: Vec::new(),
            job: UnsafeCell::new(None),
            node_pending: Vec::new(),
            worker_node: Vec::new(),
            ready: AtomicUsize::new(0),
            poisoned_worker: AtomicUsize::new(0),
            dispatching: Padded(AtomicBool::new(false)),
            shutdown: AtomicBool::new(false),
        };
        {
            let claim = DispatchClaim::try_claim(&shared)
                .expect("an unclaimed pool must hand out the claim");
            assert!(
                DispatchClaim::try_claim(&shared).is_none(),
                "a held claim must be exclusive; a second claimant would \
                 publish into the same job slot"
            );
            drop(claim);
        }
        assert!(
            DispatchClaim::try_claim(&shared).is_some(),
            "a released claim must let the next dispatch use the pool; leaking \
             it would send every later dispatch inline forever"
        );
    }

    #[test]
    fn a_panic_inside_the_critical_section_still_releases_the_claim() {
        let shared = SharedState {
            node_sense: Vec::new(),
            job: UnsafeCell::new(None),
            node_pending: Vec::new(),
            worker_node: Vec::new(),
            ready: AtomicUsize::new(0),
            poisoned_worker: AtomicUsize::new(0),
            dispatching: Padded(AtomicBool::new(false)),
            shutdown: AtomicBool::new(false),
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _claim = DispatchClaim::try_claim(&shared).expect("claim available");
            panic!("a worker poisoned the pool");
        }));
        assert!(result.is_err());
        assert!(
            DispatchClaim::try_claim(&shared).is_some(),
            "unwinding out of the critical section must release the claim"
        );
    }

    #[test]
    fn every_shard_of_a_dispatch_runs_exactly_once() {
        let Some(pools) = pools() else {
            return;
        };
        let counts: Vec<AtomicU32> = (0..pools.total_workers)
            .map(|_| AtomicU32::new(0))
            .collect();
        pools.dispatch(&|index: usize| {
            counts[index].fetch_add(1, Ordering::Relaxed);
        });
        assert_each_shard_ran_once(&counts);
    }

    /// The regression itself: two threads dispatching at the same instant. On
    /// the unguarded pool the loser's shards ran the winner's closure, so its
    /// own counters came back short (and the winner's over-counted). With the
    /// gate the loser runs inline and both see a complete, exact partition.
    #[test]
    fn two_threads_dispatching_at_once_each_get_their_own_complete_partition() {
        let Some(pools) = pools() else {
            return;
        };
        let workers = pools.total_workers;
        let start = Barrier::new(2);
        let left: Vec<AtomicU32> = (0..workers).map(|_| AtomicU32::new(0)).collect();
        let right: Vec<AtomicU32> = (0..workers).map(|_| AtomicU32::new(0)).collect();
        thread::scope(|scope| {
            for counts in [&left, &right] {
                scope.spawn(|| {
                    start.wait();
                    for _ in 0..64 {
                        pools.dispatch(&|index: usize| {
                            counts[index].fetch_add(1, Ordering::Relaxed);
                        });
                    }
                });
            }
        });
        for (side, counts) in [("left", &left), ("right", &right)] {
            for (index, count) in counts.iter().enumerate() {
                assert_eq!(
                    count.load(Ordering::Relaxed),
                    64,
                    "{side} shard {index} ran {} times across 64 dispatches, \
                     expected 64 -- a shard that ran the other thread's closure \
                     is the silent-corruption failure this gate prevents",
                    count.load(Ordering::Relaxed)
                );
            }
        }
    }

    /// A shard closure that dispatches again used to deadlock: the inner
    /// `publish` overwrote the job the outer barrier was still waiting on. The
    /// gate turns re-entrancy into inline execution.
    #[test]
    fn a_reentrant_dispatch_runs_inline_instead_of_deadlocking() {
        let Some(pools) = pools() else {
            return;
        };
        let workers = pools.total_workers;
        let inner: Vec<AtomicU32> = (0..workers).map(|_| AtomicU32::new(0)).collect();
        let outer = AtomicU32::new(0);
        pools.dispatch(&|index: usize| {
            if index != 0 {
                return;
            }
            outer.fetch_add(1, Ordering::Relaxed);
            pools.dispatch(&|nested: usize| {
                inner[nested].fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(outer.load(Ordering::Relaxed), 1);
        assert_each_shard_ran_once(&inner);
    }
}
