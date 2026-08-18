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

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use onnx_runtime_ep_cpu::task_runtime;

/// Counts allocations while armed, on every thread.
struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to `System` unchanged; the counters are the only
// addition and they touch no allocator state.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `body` with the allocation counter armed and returns what it counted.
///
/// Arming is process-wide, so this must be the only thing running. Cargo runs
/// integration tests in one binary with a thread per test, hence the single
/// `#[test]` below: two armed tests would count each other's allocations.
fn count_allocations(body: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::SeqCst);
    body();
    ARMED.store(false, Ordering::SeqCst);
    ALLOCATIONS.load(Ordering::Relaxed)
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
    assert_eq!(
        allocations, 0,
        "the dispatch path allocated {allocations} times across 2000 fan-outs"
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
