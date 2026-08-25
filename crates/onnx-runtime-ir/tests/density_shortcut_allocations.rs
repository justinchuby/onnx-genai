//! The density shortcut in `is_dense`, observed without a timer.
//!
//! A shortcut is semantically transparent: deleting it cannot change any
//! answer, so no correctness test can notice it is gone. That is normally where
//! the argument stops, and the only falsifier left is an instruction count.
//!
//! It does not have to stop there. Above `INLINE_RANK` the general path spills
//! its `(stride, extent)` list to the heap, so for a rank-9 tensor the shortcut
//! is the difference between **zero** allocations and some — a behavioural
//! difference, observable with a counting allocator and no timing at all.
//!
//! Scope, stated honestly: this guards rank > `INLINE_RANK` only. At rank <= 8
//! the general path uses stack storage and allocates nothing either way, so a
//! shortcut narrowed to fire only at low rank would still slip past. It is a
//! partial guard, not a complete one -- but it turns "only a benchmark can catch
//! this" into "a test catches full removal", which is worth having.
//!
//! Lives in `tests/` because the crate is `#![forbid(unsafe_code)]` and a global
//! allocator cannot be written without `unsafe`. Integration tests are separate
//! crates, so the lint does not reach here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) == 1 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Count allocations performed by `f`, on this thread, with nothing else running.
fn allocations_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(1, Ordering::Relaxed);
    let out = f();
    COUNTING.store(0, Ordering::Relaxed);
    (out, ALLOCS.load(Ordering::Relaxed))
}

/// Above `INLINE_RANK` the shortcut is the difference between zero heap traffic
/// and some. Deleting it makes the first assertion fail.
#[test]
fn the_density_shortcut_keeps_high_rank_queries_off_the_heap() {
    // Rank 9 > INLINE_RANK (8), contiguous, no empty extent: the shortcut fires.
    let shape = [2usize, 1, 2, 1, 2, 1, 2, 1, 2];
    let strides = onnx_runtime_ir::compute_contiguous_strides(&shape);

    let (dense, allocs) = allocations_during(|| onnx_runtime_ir::is_dense(&shape, &strides));
    assert!(dense, "premise: this layout is dense");
    assert_eq!(
        allocs, 0,
        "the shortcut should answer a rank-9 contiguous query without touching \
         the heap; {allocs} allocation(s) means it did not fire and the general \
         path spilled its pair list"
    );

    // Anchor: the general path really does allocate at this rank, so the
    // assertion above is sensitive rather than vacuously true. A permuted
    // layout is still dense but is not row-major, so the shortcut declines it
    // and the sort runs.
    let mut permuted_shape = shape;
    permuted_shape.swap(0, 8);
    let mut permuted_strides = strides.clone();
    permuted_strides.swap(0, 8);
    let (dense, allocs) =
        allocations_during(|| onnx_runtime_ir::is_dense(&permuted_shape, &permuted_strides));
    assert!(
        dense,
        "premise: a permutation of a dense layout is still dense"
    );
    assert!(
        allocs > 0,
        "anchor failed: the general path was expected to allocate at rank 9, so \
         the zero-allocation assertion above proves nothing"
    );
}
