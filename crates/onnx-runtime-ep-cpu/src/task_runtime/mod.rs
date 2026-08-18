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
/// machine. One budget knob for the whole engine is the property worth keeping:
/// `scripts/ort_ab/ab.py --threads N` sets the decode budget, and a task runtime
/// that ignored it would report thread-scaling numbers for a width nobody asked
/// for.
///
/// Whatever comes out is then capped to the number of *physical* cores the
/// process may run on ([`crate::core_topology::cap_spinning_workers`]). Spinning
/// workers are the one workload where SMT siblings are strictly negative: two
/// spinners on one core do not double throughput, they halve each other's issue
/// rate while burning the same power.
fn resolve_width() -> usize {
    let requested = NonZeroUsize::new(TASK_THREADS_OVERRIDE.load(Ordering::Acquire))
        .map(NonZeroUsize::get)
        .or_else(|| {
            std::env::var(TASK_THREADS_ENV)
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
                .filter(|threads| *threads > 0)
        })
        .or_else(crate::kernels::matmul_nbits::decode_thread_budget)
        .or_else(|| std::thread::available_parallelism().ok().map(NonZeroUsize::get))
        .unwrap_or(1);
    crate::core_topology::cap_spinning_workers(requested).max(1)
}

/// The process-wide native pool, built on first use.
fn global_pool() -> &'static TaskPool {
    static POOL: OnceLock<TaskPool> = OnceLock::new();
    POOL.get_or_init(|| TaskPool::new(resolve_width()))
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

    if use_host
        && let Some(host) = host
    {
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

/// Runs `body(chunk_index, chunk)` over `data.chunks_mut(chunk)` in parallel.
///
/// The common shape in this crate: a `[T]` output split into fixed-size rows,
/// each written by one thread. `min_chunks_per_task` is the smallest number of
/// chunks worth handing to another thread.
///
/// Returns which [`Backend`] served the fan-out.
///
/// # Panics
///
/// Panics if `chunk` is zero, and propagates a panic from `body`.
pub fn chunks_mut<T, F>(data: &mut [T], chunk: usize, min_chunks_per_task: usize, body: F) -> Backend
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
        for index in start..end {
            let offset = index * chunk;
            let this = chunk.min(len - offset);
            // SAFETY: `offset + this <= len`, and `index` is visited by exactly
            // one task, so this `&mut` is unique for its lifetime.
            let slice = unsafe { std::slice::from_raw_parts_mut(base.as_ptr().add(offset), this) };
            body(index, slice);
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

    /// Counters for the process-wide native pool.
    ///
    /// Monotonic, so a test takes a snapshot before and after and asserts on the
    /// difference; there is deliberately no reset, because resetting a
    /// process-wide counter makes concurrent tests lie to each other.
    #[must_use]
    pub fn counters() -> PoolCounters {
        global_pool().counters()
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
    #[should_panic(expected = "chunk size must be greater than zero")]
    fn chunks_mut_rejects_a_zero_chunk() {
        let mut data = [0u8; 4];
        let _ = chunks_mut(&mut data, 0, 1, |_, _| {});
    }

    #[test]
    fn the_resolved_width_never_exceeds_the_physical_cores_we_may_use() {
        let width = testing::pool_width();
        assert!(width >= 1);
        if let Some(cores) = crate::core_topology::allowed_physical_cores() {
            assert!(
                width <= cores,
                "the pool spins {width} workers on {cores} physical cores"
            );
        }
    }

    #[test]
    fn the_task_thread_budget_rejects_zero() {
        assert!(set_task_thread_budget(Some(0)).is_err());
    }
}
