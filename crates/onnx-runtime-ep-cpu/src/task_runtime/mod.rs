//! The CPU task runtime: one place that decides *how* a CPU kernel's work is
//! spread across threads, and one place that owns the threads when it has to
//! own them.
//!
//! # Why this exists
//!
//! Before this module every parallel CPU kernel called `rayon` directly — 198
//! fan-out sites across the CPU EP and the native session. That has two costs
//! the ORT A/B grid measures directly (see
//! `docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md` §26):
//!
//! 1. **Park latency.** Rayon parks its workers between regions. An isolated
//!    elementwise region measured 67 µs back-to-back and **226 µs** when the
//!    previous region ended 20 µs earlier. ORT's intra-op pool spins
//!    (`ALLOW_INTRA_OP_SPINNING`) and does not pay this. Decode is *exactly* the
//!    gapped regime: a few microseconds of graph plumbing between every op.
//! 2. **Oversubscription.** Inside an ORT plugin-EP compute call, ORT's own
//!    intra-op pool already owns the machine. Spinning a second pool of our own
//!    on the same cores measured 3–10× slower than borrowing ORT's.
//!
//! So the runtime picks between three backends per fan-out, and the choice is
//! returned to the caller ([`Backend`]) so tests can assert on the *decision*
//! rather than on elapsed time:
//!
//! * [`Backend::Serial`] — the work is too small to split, or we are already
//!   inside a task (nesting a region inside a region is how you get quadratic
//!   thread counts).
//! * [`Backend::Host`] — we are inside a plugin-EP compute call and ORT's pool
//!   has been observed running our bodies, so hand the fan-out to
//!   `KernelContext_ParallelFor`. One pool on the machine, and it is the one the
//!   embedder configured.
//! * [`Backend::Native`] — nobody else owns the machine, so use [`pool`]'s
//!   persistent adaptive-spin pool.
//!
//! # What it is not
//!
//! It is not a general-purpose work-stealing runtime. There is no `join`, no
//! futures, no nested parallelism: every fan-out is a flat, bounded, blocking
//! `for_each` over a known index space, which is what every CPU kernel in this
//! crate actually needs. That restriction is what makes the native pool
//! allocation-free and its lifetime argument short enough to check by hand.

pub mod pool;

use std::num::NonZeroUsize;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use onnx_runtime_ep_api::host_parallel;

pub use pool::{PoolCounters, TaskPool, in_task};

/// Tasks published per participating thread.
///
/// Not one: equal-sized tasks finish at unequal times (SMT siblings, a
/// neighbouring process, one row of a matrix that happens to be resident), and a
/// pool that hands out exactly one task per thread strands the whole fan-out
/// behind the slowest one. Two gives the dynamic claim protocol something to
/// rebalance with while keeping per-task overhead — a `fetch_sub` and an
/// indirect call — at half a percent of a grain-sized task.
///
/// Not four or eight: past two the tail gets no shorter (the ragged edge is
/// already one task deep) and the claim cursor's cache line gets hotter.
const TASKS_PER_LANE: usize = 2;

/// Overrides the runtime's thread budget process-wide. Zero means "unset".
static TASK_THREADS_OVERRIDE: AtomicUsize = AtomicUsize::new(0);

/// Thread budget for the native pool, when the decode budget is not the right
/// answer. Rarely needed; the decode budget governs by default so that a single
/// knob still sizes the whole engine.
const TASK_THREADS_ENV: &str = "ONNX_GENAI_CPU_TASK_THREADS";

/// Which backend a fan-out actually ran on.
///
/// Returned rather than logged because scheduling is the thing under test here,
/// and a test that asserts "this ran in parallel" by timing it is a test that
/// fails on a busy CI runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Ran inline on the calling thread.
    Serial,
    /// Ran on the host runtime's pool (ORT `KernelContext_ParallelFor`).
    Host,
    /// Ran on this crate's persistent pool.
    Native,
}

/// Sets the native pool's thread budget for the process.
///
/// Takes precedence over [`TASK_THREADS_ENV`] and over the decode budget. Must
/// be called before the first fan-out: the pool is built once and keeps its
/// width for the process lifetime, so a later call is ignored.
///
/// # Errors
///
/// Returns `Err` for `Some(0)`, which would ask for a pool that cannot run
/// anything. Use `None` to clear the override.
pub fn set_task_thread_budget(threads: Option<usize>) -> Result<(), &'static str> {
    if threads == Some(0) {
        return Err("CPU task thread budget must be greater than zero");
    }
    TASK_THREADS_OVERRIDE.store(threads.unwrap_or(0), Ordering::Release);
    Ok(())
}

/// Resolves how many threads a fan-out may use, including the caller's.
///
/// Precedence: programmatic override, then [`TASK_THREADS_ENV`], then the CPU
/// decode budget (`ONNX_GENAI_CPU_DECODE_THREADS` /
/// [`crate::kernels::matmul_nbits::set_decode_thread_budget`]), then the
/// machine. Honouring the decode budget keeps one knob sizing the whole engine:
/// `scripts/ort_ab/ab.py --threads N` sets it, and a task runtime that ignored
/// it would report thread-scaling numbers for a width nobody asked for.
///
/// **The SMT cap applies to the inferred widths, not to the asked-for one.**
/// [`TASK_THREADS_ENV`] and [`set_task_thread_budget`] name *this* pool, so a
/// caller that sets them has said how many threads they want it to spin and
/// gets exactly that. The decode budget and `available_parallelism` are
/// inferences -- one is about a different pool, the other is about the machine
/// -- and both are expressed in logical CPUs, so both are capped to the
/// *physical* cores the process may run on
/// ([`crate::core_topology::cap_spinning_workers`]).
///
/// The cap exists because spinning workers are the workload where SMT siblings
/// hurt most: two spinners on one core do not double throughput, they halve each
/// other's issue rate while burning the same power. But that is only true once
/// the machine is wide enough to saturate its memory system. Below that, the
/// sibling threads *are* the parallelism, and they hide memory latency rather
/// than compete for issue slots.
///
/// So the cap only bites above [`SMT_CAP_FLOOR`] hardware threads. The crossover
/// was measured, not assumed -- same binary, one arm with the explicit knob set
/// to the uncapped width, on RoPE and Softmax cells:
///
/// | logical CPUs | physical | capped wins | uncapped wins |
/// |---|---|---|---|
/// | 2  | 1 | 0/4 | 4/4, by 14-19% |
/// | 4  | 2 | 0/4 | 4/4, by 3-25%  |
/// | 8  | 4 | 1/4 | 3/4, by 10-26% |
/// | 16 | 8 | 3/4, by 12-45% | 1/4 |
/// | 32 | 16 | 3/4, by 12-36% | 1/4 |
///
/// The one cell that prefers SMT at every width is the largest bandwidth-bound
/// softmax; the one that prefers the cap at every width is the smallest RoPE.
/// That is the same story from both ends: latency-bound work likes siblings,
/// issue-bound work does not.
fn resolve_width() -> usize {
    let asked = NonZeroUsize::new(TASK_THREADS_OVERRIDE.load(Ordering::Acquire))
        .map(NonZeroUsize::get)
        .or_else(|| {
            std::env::var(TASK_THREADS_ENV)
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
                .filter(|threads| *threads > 0)
        });
    if let Some(asked) = asked {
        return asked;
    }
    let inferred = crate::kernels::matmul_nbits::decode_thread_budget()
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .map(NonZeroUsize::get)
        })
        .unwrap_or(1);
    smt_cap(
        inferred,
        crate::core_topology::cap_spinning_workers(inferred),
    )
}

/// The width below which the SMT cap is not applied at all.
///
/// Empirical, and from a single microarchitecture (AMD EPYC 9V74, 16 physical /
/// 32 logical). It is the point where halving the width started winning more
/// cells than it lost. Anyone porting this to a machine with a very different
/// memory system should re-run the experiment in [`resolve_width`]'s table
/// rather than trust the number.
const SMT_CAP_FLOOR: usize = 8;

/// Applies the physical-core cap, but never below [`SMT_CAP_FLOOR`] workers.
///
/// Split out from [`resolve_width`] so the policy is testable without a machine
/// that has the topology in question.
fn smt_cap(inferred: usize, capped: usize) -> usize {
    capped.max(1).max(inferred.min(SMT_CAP_FLOOR))
}

/// The process-wide native pool, built on first use.
static POOL: OnceLock<TaskPool> = OnceLock::new();

/// The process-wide native pool, built on first use.
fn global_pool() -> &'static TaskPool {
    POOL.get_or_init(|| TaskPool::new(resolve_width()))
}

/// The pool if some kernel has already built it, and `None` if none has.
///
/// The distinction matters to anything that only wants to *observe*. Building
/// the pool spawns `width - 1` worker threads, so an observer that reaches it
/// through [`global_pool`] manufactures the threads it is about to report --
/// and on a route that never dispatches, every one of them is an artefact of
/// the measurement.
fn built_pool() -> Option<&'static TaskPool> {
    POOL.get()
}

/// Threads the native pool can spread a fan-out across, including the caller's.
///
/// Builds the pool if it does not exist yet, so a kernel can size a partition
/// against it before dispatching.
pub fn width() -> usize {
    if testing::forced_serial() {
        return 1;
    }
    global_pool().width()
}

/// How many tasks a fan-out of `total` items with a minimum grain of
/// `min_grain` should be split into, or `None` to run it serially.
///
/// Split out as a pure function because grain policy is the part most likely to
/// be wrong, and this way it can be tested without threads.
fn plan_tasks(total: usize, min_grain: usize, width: usize) -> Option<usize> {
    if total == 0 || width <= 1 {
        return None;
    }
    let min_grain = min_grain.max(1);
    // Floor, not ceiling: every task must be *at least* `min_grain` items, or a
    // caller's "do not split below this or the fixed cost dominates" is not
    // honoured.
    let by_grain = total / min_grain;
    let tasks = by_grain.min(width.saturating_mul(TASKS_PER_LANE));
    (tasks > 1).then_some(tasks)
}

/// Runs `body(start, end)` over a partition of `0..total` and blocks until every
/// item has been covered exactly once.
///
/// `min_grain` is the smallest number of items worth handing to another thread.
/// It is a floor, not a target: the runtime will use a larger grain when `total`
/// is big, and will run serially when `total` cannot be split into at least two
/// tasks of that size.
///
/// Returns which [`Backend`] served the fan-out.
///
/// # Panics
///
/// Propagates a panic from `body` to the caller, after every other task has
/// finished (see [`TaskPool::dispatch`]).
pub fn for_each_range<F>(total: usize, min_grain: usize, body: F) -> Backend
where
    F: Fn(usize, usize) + Sync,
{
    if total == 0 {
        return Backend::Serial;
    }
    // Already inside a task on either pool: splitting again would put a second
    // region inside the first, and every thread in the outer region would try
    // it at once.
    if testing::forced_serial() || host_parallel::in_host_task() || in_task() {
        body(0, total);
        return Backend::Serial;
    }

    let host = host_parallel::current();
    // `prefer_host` mutates probe state and must be asked exactly once per
    // decision — see its docs.
    let use_host = host.is_some_and(|host| host.prefer_host());
    let width = if use_host {
        // The host's width is not observable through the C API, so plan against
        // the machine and let ORT's pool decide how many of its threads to put
        // on it. Over-splitting is cheap there; under-splitting is not.
        std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1)
    } else {
        global_pool().width()
    };

    let Some(tasks) = plan_tasks(total, min_grain, width) else {
        body(0, total);
        return Backend::Serial;
    };

    let run_task = |task: usize| {
        let (start, end) = task_range(total, tasks, task);
        if start < end {
            body(start, end);
        }
    };

    if use_host && let Some(host) = host {
        return run_on_host(&host, tasks, &run_task);
    }

    if global_pool().dispatch(tasks, &run_task) {
        Backend::Native
    } else {
        // Every slot busy, or a pool that could not get threads. Correctness
        // never depends on the pool; only speed does.
        body(0, total);
        Backend::Serial
    }
}

/// Outlined on purpose.
///
/// Inlining the host arm into the caller repartitions codegen units and has
/// measured real regressions on paths that never take it — `Relu` at 1 Mi lost
/// 34% with no change in which branch ran (see `kernels::simd_activations`,
/// which discovered this). Keep the cold arm in its own function.
#[inline(never)]
fn run_on_host(
    host: &host_parallel::HostParallel,
    tasks: usize,
    run_task: &(dyn Fn(usize) + Sync),
) -> Backend {
    host.run(tasks, run_task);
    Backend::Host
}

/// The half-open item range task `task` of `tasks` covers.
///
/// Remainder items go to the first `total % tasks` tasks rather than all onto
/// the last one, so the longest task is one item longer than the shortest
/// instead of `tasks - 1` longer.
fn task_range(total: usize, tasks: usize, task: usize) -> (usize, usize) {
    let base = total / tasks;
    let extra = total % tasks;
    let start = task * base + task.min(extra);
    let len = base + usize::from(task < extra);
    (start, start + len)
}

/// Runs `body(first_chunk_index, run)` over `data` split into fixed-size chunks,
/// where each task receives its whole contiguous **run** of chunks at once.
///
/// Use this when the per-call cost is not negligible — an MLAS entry point, a
/// packing step, anything with a prologue. Splitting `data` into many small
/// chunks and calling the body once per chunk pays that prologue once per chunk;
/// this pays it once per task while still giving the runtime many chunks to
/// balance with.
///
/// The run's length is a whole number of chunks except for the last task, whose
/// final chunk is short when `chunk` does not divide `data.len()`.
///
/// Returns which [`Backend`] served the fan-out.
///
/// # Panics
///
/// Panics if `chunk` is zero, and propagates a panic from `body`.
pub fn chunk_runs_mut<T, F>(
    data: &mut [T],
    chunk: usize,
    min_chunks_per_task: usize,
    body: F,
) -> Backend
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    assert!(chunk > 0, "chunk size must be greater than zero");
    let len = data.len();
    let chunks = len.div_ceil(chunk);
    if chunks == 0 {
        return Backend::Serial;
    }
    // SAFETY (established here, relied on inside the closure): every task gets a
    // disjoint half-open chunk range, and `for_each_range` covers `0..chunks`
    // exactly once with no overlap, so no two tasks reconstruct overlapping
    // slices. The pointer outlives the fan-out because `for_each_range` blocks.
    let base = SendMutPtr(data.as_mut_ptr());
    for_each_range(chunks, min_chunks_per_task.max(1), move |start, end| {
        let offset = start * chunk;
        let run = (end * chunk).min(len) - offset;
        // SAFETY: `offset + run <= len`, and `start..end` is visited by exactly
        // one task, so this `&mut` is unique for its lifetime.
        let slice = unsafe { std::slice::from_raw_parts_mut(base.as_ptr().add(offset), run) };
        body(start, slice);
    })
}

/// Runs `body(chunk_index, chunk)` over `data.chunks_mut(chunk)` in parallel.
///
/// The common shape in this crate: a `[T]` output split into fixed-size rows,
/// each written by one thread. `min_chunks_per_task` is the smallest number of
/// chunks worth handing to another thread.
///
/// Reach for [`chunk_runs_mut`] instead when the body has a per-call prologue
/// worth amortising over a task's whole run.
///
/// Returns which [`Backend`] served the fan-out.
///
/// # Panics
///
/// Panics if `chunk` is zero, and propagates a panic from `body`.
pub fn chunks_mut<T, F>(
    data: &mut [T],
    chunk: usize,
    min_chunks_per_task: usize,
    body: F,
) -> Backend
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    chunk_runs_mut(data, chunk, min_chunks_per_task, |first, run| {
        for (offset, piece) in run.chunks_mut(chunk).enumerate() {
            body(first + offset, piece);
        }
    })
}

/// A `*mut T` that may cross into a task body.
///
/// `Send`/`Sync` are asserted by the caller in [`chunks_mut`], which is the only
/// constructor: disjointness is proved there by the partition, not here.
///
/// `Clone`/`Copy` are written out rather than derived because the derive would
/// add a `T: Copy` bound, and the pointee is very often not `Copy`.
struct SendMutPtr<T>(*mut T);

impl<T> Clone for SendMutPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendMutPtr<T> {}

impl<T> SendMutPtr<T> {
    /// Reads the pointer through a by-value method so a closure that uses it
    /// captures the whole wrapper rather than the bare `*mut T` field — the
    /// latter is what edition-2021 disjoint capture would do, and a bare
    /// pointer is not `Sync`.
    fn as_ptr(self) -> *mut T {
        self.0
    }
}
// SAFETY: see `chunks_mut`.
unsafe impl<T: Send> Send for SendMutPtr<T> {}
// SAFETY: see `chunks_mut`.
unsafe impl<T: Send> Sync for SendMutPtr<T> {}

/// Deterministic hooks for tests.
///
/// Scoped and thread-local rather than environment variables: a test that has to
/// set an env var cannot run next to another test in the same process, and an
/// env-var hook is a production code path that happens to be undocumented.
pub mod testing {
    use super::{Backend, PoolCounters, TaskPool, global_pool};
    use std::cell::Cell;

    thread_local! {
        static FORCE_SERIAL: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) fn forced_serial() -> bool {
        FORCE_SERIAL.with(Cell::get)
    }

    /// Forces every fan-out on this thread to [`Backend::Serial`] until the
    /// returned guard drops.
    ///
    /// For proving that a kernel's parallel and serial paths agree bit for bit,
    /// which is the property that actually matters about a scheduler change.
    #[must_use]
    pub fn force_serial() -> ForceSerial {
        ForceSerial(FORCE_SERIAL.with(|c| c.replace(true)))
    }

    /// Restores the previous backend policy on drop.
    pub struct ForceSerial(bool);

    impl Drop for ForceSerial {
        fn drop(&mut self) {
            FORCE_SERIAL.with(|c| c.set(self.0));
        }
    }

    /// Counters for the process-wide native pool, or zeroes if no kernel has
    /// built it yet.
    ///
    /// Monotonic, so a test takes a snapshot before and after and asserts on the
    /// difference; there is deliberately no reset, because resetting a
    /// process-wide counter makes concurrent tests lie to each other.
    ///
    /// Deliberately does **not** build the pool. It used to, and that made it an
    /// instrument that changed what it measured: a benchmark taking a "before"
    /// snapshot spawned `width - 1` workers, then reported a model that never
    /// dispatched as having an idle pool. Measured on an fp32 model, which runs
    /// on Rayon and touches this pool nowhere, the snapshot alone created 15
    /// threads and they showed up as parks in the steady window. Zeroes are the
    /// honest answer for a pool that does not exist, and the delta a caller
    /// computes is unaffected: if the pool is built between the two snapshots,
    /// the "before" it missed was zero anyway.
    #[must_use]
    pub fn counters() -> PoolCounters {
        super::built_pool().map_or_else(PoolCounters::default, TaskPool::counters)
    }

    /// Threads the process-wide native pool can use, including the caller's.
    #[must_use]
    pub fn pool_width() -> usize {
        global_pool().width()
    }

    /// A private pool of exactly `width` threads, for tests that need to assert
    /// on counters without another test's fan-outs moving them.
    #[must_use]
    pub fn isolated_pool(width: usize) -> TaskPool {
        TaskPool::new(width)
    }

    /// Which backend a fan-out of this shape *would* use, without running it.
    #[must_use]
    pub fn planned_backend(total: usize, min_grain: usize) -> Backend {
        if total == 0 || forced_serial() || super::in_task() {
            return Backend::Serial;
        }
        if onnx_runtime_ep_api::host_parallel::in_host_task() {
            return Backend::Serial;
        }
        match super::plan_tasks(total, min_grain, global_pool().width()) {
            Some(_) => Backend::Native,
            None => Backend::Serial,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, AtomicUsize};

    #[test]
    fn task_ranges_tile_the_space_exactly() {
        for total in [1usize, 2, 5, 17, 64, 1000, 1001] {
            for tasks in 1..=17usize {
                let mut covered = vec![0u32; total];
                let mut previous_end = 0;
                for task in 0..tasks {
                    let (start, end) = task_range(total, tasks, task);
                    assert!(start <= end, "{total}/{tasks} task {task} is reversed");
                    assert_eq!(start, previous_end, "{total}/{tasks} task {task} gap");
                    for slot in &mut covered[start..end] {
                        *slot += 1;
                    }
                    previous_end = end;
                }
                assert_eq!(previous_end, total, "{total}/{tasks} did not reach the end");
                assert!(covered.iter().all(|&c| c == 1), "{total}/{tasks} overlap");
            }
        }
    }

    #[test]
    fn task_ranges_are_balanced_to_within_one_item() {
        // A partition that gives the last task the whole remainder makes the
        // fan-out as slow as its longest task; this is the property that stops
        // that.
        for total in [10usize, 100, 999, 1_000_000] {
            for tasks in [2usize, 3, 7, 16, 32] {
                let lengths: Vec<usize> = (0..tasks)
                    .map(|task| {
                        let (start, end) = task_range(total, tasks, task);
                        end - start
                    })
                    .collect();
                let min = lengths.iter().copied().min().unwrap();
                let max = lengths.iter().copied().max().unwrap();
                assert!(max - min <= 1, "{total}/{tasks} lengths {lengths:?}");
            }
        }
    }

    #[test]
    fn grain_is_a_floor_not_a_target() {
        // 1000 items, grain 100, 32 lanes: grain wins, so 10 tasks, not 64.
        assert_eq!(plan_tasks(1000, 100, 32), Some(10));
        // 1_000_000 items, grain 100, 4 lanes: the lane cap wins.
        assert_eq!(plan_tasks(1_000_000, 100, 4), Some(4 * TASKS_PER_LANE));
        // Below two grains, serial.
        assert_eq!(plan_tasks(199, 100, 32), None);
        assert_eq!(plan_tasks(200, 100, 32), Some(2));
        // Degenerate inputs never split.
        assert_eq!(plan_tasks(0, 1, 32), None);
        assert_eq!(plan_tasks(1000, 100, 1), None);
        // A zero grain must not divide by zero.
        assert_eq!(plan_tasks(8, 0, 4), Some(4 * TASKS_PER_LANE));
    }

    #[test]
    fn every_item_is_visited_exactly_once() {
        for total in [1usize, 3, 64, 5000] {
            let seen: Vec<AtomicU32> = (0..total).map(|_| AtomicU32::new(0)).collect();
            let backend = for_each_range(total, 1, |start, end| {
                for slot in &seen[start..end] {
                    slot.fetch_add(1, Ordering::Relaxed);
                }
            });
            assert!(matches!(backend, Backend::Serial | Backend::Native));
            assert!(
                seen.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                "total {total} missed or duplicated an item"
            );
        }
    }

    /// Concurrent tasks reconstructing `&mut` slices from one shared base must
    /// retag **only their own range**.
    ///
    /// This lives in the lib, not beside the fan-out benchmark that actually
    /// uses the shape, because CI's Miri lane runs
    /// `-p onnx-runtime-ep-cpu --lib task_runtime::` — integration tests under
    /// `tests/` are never checked. #1377 shipped a whole-row retag in
    /// `tests/task_runtime_latency.rs` and it survived precisely because
    /// nothing under Miri exercised the pattern. Fixing that instance without
    /// putting the shape under Miri would leave the same gap open for the next
    /// one.
    ///
    /// **Falsifier.** Widen the `from_raw_parts_mut` below to the whole row and
    /// index `[start..end]` afterwards:
    ///
    /// ```ignore
    /// let all = unsafe { std::slice::from_raw_parts_mut(base.as_ptr(), WIDTH) };
    /// for slot in &mut all[start..end] { *slot += 1; }
    /// ```
    ///
    /// and this test fails under Stacked Borrows. The *stores* stay disjoint —
    /// what conflicts is the retag, which claims all of `WIDTH` in every task,
    /// so each task pops the previous task's tag. The rendezvous below is what
    /// makes that deterministic rather than a matter of interleaving luck: it
    /// holds every task's reference live at the same time before any of them
    /// writes.
    #[test]
    fn concurrent_tasks_retag_only_their_own_range() {
        if testing::pool_width() < 2 {
            // Nothing to alias against on a single-lane pool.
            return;
        }
        const WIDTH: usize = 16;
        // Small and spin-bounded: this runs under Miri, which is ~2 orders of
        // magnitude slower and would otherwise dominate the lane's runtime.
        const SPIN_LIMIT: u32 = 10_000;

        let mut row = vec![0u64; WIDTH];
        let base = SendMutPtr(row.as_mut_ptr());
        let live = AtomicUsize::new(0);

        let backend = for_each_range(WIDTH, 1, |start, end| {
            // SAFETY: `for_each_range` visits `start..end` in exactly one task,
            // so narrowing before the retag makes this `&mut` unique for its
            // lifetime. This is the same shape `parallel_output_rows_repeated`
            // uses in production.
            let slots =
                unsafe { std::slice::from_raw_parts_mut(base.as_ptr().add(start), end - start) };

            // Publish that this task holds its reference, then wait for company
            // so several retags are live simultaneously. Bounded, so a pool that
            // ran the body serially still finishes instead of hanging.
            live.fetch_add(1, Ordering::AcqRel);
            for _ in 0..SPIN_LIMIT {
                if live.load(Ordering::Acquire) >= 2 {
                    break;
                }
                std::hint::spin_loop();
            }

            for slot in slots {
                *slot += 1;
            }
        });

        // Self-check: a canary that silently degrades to one serial task would
        // pass whether or not the retag is correct. Under Miri that is the
        // default outcome unless `-Zmiri-num-cpus` is set, which is exactly how
        // #1377's whole-row retag survived a lane that claimed to cover this
        // module. Assert the fan-out actually happened.
        assert!(
            matches!(backend, Backend::Native),
            "expected a real fan-out, got {backend:?}; with pool_width \u{003e}= 2 this test \
             must not degenerate to a single task or it stops checking anything"
        );
        assert!(
            row.iter().all(|&visits| visits == 1),
            "every slot must be written exactly once: {row:?}"
        );
    }

    #[test]
    fn a_large_fanout_uses_the_native_pool_and_more_than_one_thread() {
        if testing::pool_width() < 2 {
            // A single-core machine has nothing to assert.
            return;
        }
        // Wait for company inside the body rather than timing the fan-out: the
        // dispatcher drains its own slot greedily, so a fan-out of trivial tasks
        // can legitimately finish before a parked worker wakes. See
        // `pool::tests::rendezvous`.
        let arrived = AtomicUsize::new(0);
        let threads = Mutex::new(std::collections::HashSet::new());
        let backend = for_each_range(1 << 16, 1, |_, _| {
            arrived.fetch_add(1, Ordering::AcqRel);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while arrived.load(Ordering::Acquire) < 2 && std::time::Instant::now() < deadline {
                std::hint::spin_loop();
            }
            threads.lock().unwrap().insert(std::thread::current().id());
        });
        assert_eq!(backend, Backend::Native);
        assert!(
            threads.lock().unwrap().len() > 1,
            "the native backend ran everything on the calling thread"
        );
    }

    #[test]
    fn a_small_fanout_stays_serial() {
        let calls = AtomicUsize::new(0);
        let backend = for_each_range(64, 1024, |start, end| {
            calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!((start, end), (0, 64));
        });
        assert_eq!(backend, Backend::Serial);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn an_empty_fanout_never_calls_the_body() {
        let backend = for_each_range(0, 1, |_, _| unreachable!());
        assert_eq!(backend, Backend::Serial);
    }

    #[test]
    fn nested_fanouts_do_not_split_again() {
        if testing::pool_width() < 2 {
            return;
        }
        let nested_serial = AtomicUsize::new(0);
        let nested_total = AtomicUsize::new(0);
        let backend = for_each_range(1 << 14, 1, |_, _| {
            let inner = for_each_range(1 << 14, 1, |_, _| {});
            nested_total.fetch_add(1, Ordering::Relaxed);
            if inner == Backend::Serial {
                nested_serial.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(backend, Backend::Native);
        assert_eq!(
            nested_serial.load(Ordering::Relaxed),
            nested_total.load(Ordering::Relaxed),
            "a nested fan-out split a second time"
        );
    }

    #[test]
    fn force_serial_is_scoped_and_deterministic() {
        {
            let _serial = testing::force_serial();
            let calls = AtomicUsize::new(0);
            let backend = for_each_range(1 << 20, 1, |_, _| {
                calls.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(backend, Backend::Serial);
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert_eq!(width(), 1);
        }
        assert_eq!(testing::planned_backend(1 << 20, 1), {
            if testing::pool_width() > 1 {
                Backend::Native
            } else {
                Backend::Serial
            }
        });
    }

    #[test]
    fn chunks_mut_writes_every_chunk_exactly_once() {
        for (len, chunk) in [(0usize, 4usize), (1, 4), (16, 4), (17, 4), (100_000, 7)] {
            let mut data = vec![0u32; len];
            let backend = chunks_mut(&mut data, chunk, 1, |index, slice| {
                for (offset, value) in slice.iter_mut().enumerate() {
                    *value = (index * chunk + offset) as u32 + 1;
                }
            });
            assert!(matches!(backend, Backend::Serial | Backend::Native));
            let expected: Vec<u32> = (0..len).map(|i| i as u32 + 1).collect();
            assert_eq!(data, expected, "len {len} chunk {chunk}");
        }
    }

    #[test]
    fn chunks_mut_matches_the_serial_path_bit_for_bit() {
        let len = 100_003;
        let mut parallel = vec![0.0f32; len];
        let mut serial = vec![0.0f32; len];
        let fill = |index: usize, slice: &mut [f32]| {
            for (offset, value) in slice.iter_mut().enumerate() {
                *value = (index as f32).mul_add(0.5, offset as f32 * 0.25);
            }
        };
        chunks_mut(&mut parallel, 13, 1, fill);
        {
            let _serial = testing::force_serial();
            chunks_mut(&mut serial, 13, 1, fill);
        }
        assert_eq!(parallel, serial);
    }

    #[test]
    fn chunk_runs_are_whole_contiguous_spans() {
        // The property `chunks_mut` cannot offer: one body call per task, over
        // the task's entire run, so a per-call prologue is paid once.
        for (len, chunk) in [(0usize, 4usize), (1, 4), (16, 4), (17, 4), (100_000, 7)] {
            let mut data = vec![0u32; len];
            let calls = AtomicUsize::new(0);
            chunk_runs_mut(&mut data, chunk, 1, |first, run| {
                calls.fetch_add(1, Ordering::Relaxed);
                assert!(
                    run.len() % chunk == 0 || first * chunk + run.len() == len,
                    "a non-final run of {} is not a whole number of {chunk}-chunks",
                    run.len()
                );
                for (offset, value) in run.iter_mut().enumerate() {
                    *value = (first * chunk + offset) as u32 + 1;
                }
            });
            let expected: Vec<u32> = (0..len).map(|i| i as u32 + 1).collect();
            assert_eq!(data, expected, "len {len} chunk {chunk}");
            let chunks = len.div_ceil(chunk);
            assert!(
                calls.load(Ordering::Relaxed) <= chunks.max(1),
                "len {len} chunk {chunk} called the body more often than there are chunks"
            );
        }
    }

    #[test]
    #[should_panic(expected = "chunk size must be greater than zero")]
    fn chunks_mut_rejects_a_zero_chunk() {
        let mut data = [0u8; 4];
        let _ = chunks_mut(&mut data, 0, 1, |_, _| {});
    }

    #[test]
    fn the_smt_cap_only_bites_on_wide_machines() {
        // Narrow machines keep every hardware thread: the siblings are the only
        // parallelism there is, and measured they win.
        assert_eq!(smt_cap(2, 1), 2);
        assert_eq!(smt_cap(4, 2), 4);
        assert_eq!(smt_cap(8, 4), 8);
        // Wide machines get capped to their physical cores.
        assert_eq!(smt_cap(16, 8), 8);
        assert_eq!(smt_cap(32, 16), 16);
        // A machine with no SMT is unaffected either way.
        assert_eq!(smt_cap(16, 16), 16);
        // A genuinely single-threaded budget stays single-threaded.
        assert_eq!(smt_cap(1, 1), 1);
        // A cap that is somehow zero still leaves us a usable width.
        assert_eq!(smt_cap(1, 0), 1);
        // The cap never invents threads the caller did not ask for.
        for inferred in 1..64usize {
            for capped in 0..=inferred {
                assert!(smt_cap(inferred, capped) <= inferred);
            }
        }
    }

    #[test]
    fn an_inferred_width_never_exceeds_the_physical_cores_we_may_use() {
        // The process-wide pool is built from an inferred width unless a test
        // process sets the explicit knob, which none of them do.
        if std::env::var_os(TASK_THREADS_ENV).is_some() {
            return;
        }
        let width = testing::pool_width();
        assert!(width >= 1);
        if let Some(cores) = crate::core_topology::allowed_physical_cores() {
            assert!(
                width <= cores.max(SMT_CAP_FLOOR),
                "the pool spins {width} workers on {cores} physical cores"
            );
        }
    }

    #[test]
    fn the_task_thread_budget_rejects_zero() {
        assert!(set_task_thread_budget(Some(0)).is_err());
    }

    /// Observing the counters must not build the pool.
    ///
    /// This is the instrument-perturbs-the-measurement case, and it had already
    /// happened: `testing::counters()` reached the pool through `global_pool()`,
    /// so a benchmark's "before" snapshot spawned `width - 1` workers before the
    /// model ran. On an fp32 model -- which fans out on Rayon and never touches
    /// this pool -- that manufactured 15 threads and then reported them parking,
    /// so the harness showed an idle native pool for a route with no native pool
    /// in it.
    ///
    /// Runs in a child process because the property is only observable in a
    /// process where nothing has built the pool yet. In the shared test binary
    /// any earlier test may have built it, and then this assertion holds
    /// vacuously -- it would pass just as happily against the defect it exists
    /// to catch. A child running this one test `--exact` is virgin by
    /// construction.
    #[test]
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore = "Miri cannot spawn the child process this needs")]
    fn observing_the_counters_does_not_build_the_pool() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("task_runtime::tests::counters_observer_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .arg("--ignored")
            .env(COUNTERS_OBSERVER_CHILD_ENV, "1")
            .output()
            .expect("run counters-observer child");
        assert!(
            output.status.success(),
            "child failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    const COUNTERS_OBSERVER_CHILD_ENV: &str = "ONNX_GENAI_COUNTERS_OBSERVER_CHILD";

    /// Child half of [`observing_the_counters_does_not_build_the_pool`].
    #[test]
    #[ignore = "spawned by its parent test"]
    #[cfg(target_os = "linux")]
    fn counters_observer_child() {
        if std::env::var(COUNTERS_OBSERVER_CHILD_ENV).is_err() {
            return;
        }
        let threads = || {
            std::fs::read_dir("/proc/self/task")
                .expect("read /proc/self/task")
                .count()
        };

        let before = threads();
        let snapshot = testing::counters();
        // Thread creation is asynchronous from the observer's point of view, so
        // a count taken immediately could miss workers that are still being
        // spawned and would report the defect as absent.
        std::thread::sleep(std::time::Duration::from_millis(300));
        let after = threads();

        assert_eq!(
            before,
            after,
            "observing the counters spawned {} thread(s): the snapshot built the \
             pool, so any measurement taken around it includes workers the \
             measurement itself created",
            after.saturating_sub(before)
        );
        assert_eq!(
            snapshot,
            PoolCounters::default(),
            "no kernel has dispatched in this process, so the honest answer is \
             zeroes; a non-zero row here describes a pool that only exists \
             because it was asked about"
        );

        // Anti-vacuity: the assertions above must be capable of failing. If
        // building the pool did not create threads, the count above would be
        // stable whether or not `counters()` built it, and this test would pass
        // against the defect. Proving the observable is live costs one pool.
        let built = super::width();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let with_pool = threads();
        assert!(
            with_pool > after,
            "building the pool (width {built}) created no threads, so the \
             thread-count observable above cannot detect the defect this test \
             exists to catch"
        );
        println!("observer_child before={before} after={after} with_pool={with_pool}");
    }
}
