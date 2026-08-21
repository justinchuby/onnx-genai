//! The CPU task runtime must not allocate on the dispatch path.
//!
//! A fan-out happens once per parallel region, and a decode step is hundreds of
//! regions: an allocation per region is an allocator lock per region, on every
//! thread, in the exact regime (many small regions, microseconds apart) this
//! runtime exists to make fast. Worse, it is *invisible* in a wall-clock
//! benchmark on an idle machine and shows up only under concurrent sessions,
//! which is where a serving process lives.
//!
//! So the property is checked directly, with a counting allocator, rather than
//! inferred from a timing. This lives in an integration test because a
//! `#[global_allocator]` replaces the allocator for the whole binary, which is
//! not something a unit test can scope.
//!
//! # Why the counter is scoped to participating threads
//!
//! A `#[global_allocator]` sees *every* thread in the process, including
//! libtest's own harness thread. That thread is not idle while the test runs:
//! it keeps a `HashMap` of running tests, and the insert for *this* test can
//! land after the test body has already started. On an idle machine the harness
//! wins that race and the test passes; under load the body arms first and
//! counts libtest's `HashMap` resize as if it were ours. That was #1660 --
//! reproduced by running eight copies pinned to two cores, and confirmed by
//! capturing the backtrace of the first armed allocation, which was
//! `test::run_tests` -> `HashMap::insert` -> `hashbrown` resize, on a thread
//! that was not the test thread.
//!
//! The fix is not to tolerate a few allocations -- that would let a genuine
//! per-dispatch allocation hide under the allowance. It is to count only the
//! threads the property is actually about: the dispatching thread and every
//! worker in the pool. Those are registered *before* the counter is armed, and
//! the registration doubles as proof that the warm-up really did reach every
//! worker, which the previous version only assumed.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use onnx_runtime_ep_cpu::task_runtime;

/// Counts allocations while armed, on registered threads only.
struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
/// Allocations seen while armed on threads that are *not* part of the dispatch
/// path. Reported, never asserted on: they are someone else's business, and a
/// non-zero count is useful evidence that this scoping is still doing work.
static FOREIGN_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
/// Distinct threads that have executed a task body during warm-up.
static PARTICIPANTS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Set on the dispatching thread and on every pool worker, during warm-up.
    ///
    /// Const-initialised and `Drop`-free on purpose: such a thread-local is a
    /// direct TLS access with no lazy initialisation and no destructor
    /// registration, so reading it from inside the global allocator can neither
    /// allocate nor recurse.
    static IS_PARTICIPANT: Cell<bool> = const { Cell::new(false) };
}

/// Registers this thread as part of the dispatch path, at most once.
fn join_participants() {
    IS_PARTICIPANT.with(|flag| {
        if !flag.get() {
            flag.set(true);
            PARTICIPANTS.fetch_add(1, Ordering::Relaxed);
        }
    });
}

fn counts_here() -> bool {
    IS_PARTICIPANT.with(Cell::get)
}

/// Records one allocation, attributed to the dispatch path or not.
#[inline]
fn record() {
    if !ARMED.load(Ordering::Relaxed) {
        return;
    }
    if counts_here() {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    } else {
        FOREIGN_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards to `System` unchanged; the counters are the only
// addition and they touch no allocator state.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `body` with the allocation counter armed and returns what it counted on
/// the dispatch path.
fn count_allocations(body: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    FOREIGN_ALLOCATIONS.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::SeqCst);
    body();
    ARMED.store(false, Ordering::SeqCst);
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// Spins for roughly `micros`, without sleeping.
///
/// A task body that returns instantly tends to be re-claimed by the same worker
/// before its peers wake, so a warm-up built from instant tasks can leave most
/// of the pool untouched. Holding each task briefly forces the fan-out to
/// spread. This runs before the counter is armed, so its cost does not matter.
fn spin_for(micros: u64) {
    let until = std::time::Duration::from_micros(micros);
    let start = std::time::Instant::now();
    while start.elapsed() < until {
        std::hint::spin_loop();
    }
}

/// Dispatches until every thread in the pool has run a task body, so the armed
/// window measures a fully warm runtime and knows exactly which threads belong
/// to it.
fn warm_every_worker(width: usize) {
    join_participants();
    // Generous bound: the pool claims tasks dynamically, so reaching the last
    // worker is a race the warm-up wins by repetition rather than by fiat.
    for _ in 0..2_000 {
        if PARTICIPANTS.load(Ordering::Relaxed) >= width {
            return;
        }
        // At least one task per thread, each held long enough that idle workers
        // claim their own instead of losing every task to the fastest one.
        task_runtime::for_each_range(width * 64, 1, |_, _| {
            join_participants();
            spin_for(50);
        });
    }
}

#[test]
fn the_dispatch_path_never_allocates() {
    // Warm the runtime first: building the pool allocates (thread stacks, the
    // slot array, the join handles) and is meant to, exactly once per process.
    let mut scratch = vec![0.0f32; 1 << 16];
    let mut backends = Vec::new();
    for _ in 0..8 {
        backends.push(task_runtime::for_each_range(1 << 16, 1, |_, _| {}));
        backends.push(task_runtime::chunks_mut(&mut scratch, 64, 1, |_, _| {}));
    }
    backends.clear();

    let width = task_runtime::testing::pool_width();
    warm_every_worker(width);
    let participants = PARTICIPANTS.load(Ordering::Relaxed);
    assert_eq!(
        participants, width,
        "warm-up reached {participants} of {width} pool threads, so an unwarmed \
         thread could allocate inside the armed window and go uncounted"
    );

    let before = task_runtime::testing::counters();

    let allocations = count_allocations(|| {
        for round in 0..1000usize {
            // Vary the shape so the runtime takes every arm: serial (too small),
            // native, and a ragged partition that does not divide evenly.
            let total = 1 + round * 37;
            task_runtime::for_each_range(total, 8, |start, end| {
                assert!(start <= end);
            });
            task_runtime::chunks_mut(&mut scratch, 1 + round % 97, 1, |_, slab| {
                if let Some(first) = slab.first_mut() {
                    *first = 0.0;
                }
            });
        }
    });

    let after = task_runtime::testing::counters();
    let foreign = FOREIGN_ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(
        allocations, 0,
        "the dispatch path allocated {allocations} times across 2000 fan-outs \
         ({foreign} further allocations came from threads outside the dispatch \
         path and were not counted)"
    );
    // Nothing may join the dispatch path once the measurement has begun: if
    // something did, its allocations went uncounted and the zero above would
    // mean less than it appears.
    assert_eq!(
        PARTICIPANTS.load(Ordering::Relaxed),
        width,
        "a thread joined the dispatch path during the measurement"
    );
    assert_eq!(
        task_runtime::testing::pool_width(),
        width,
        "the pool changed width during the measurement"
    );
    // Guard against the test passing because nothing was actually dispatched.
    if width > 1 {
        assert!(
            after.dispatches > before.dispatches,
            "no fan-out reached the native pool, so the measurement proves nothing"
        );
        assert!(after.tasks > before.tasks);
    }
}
