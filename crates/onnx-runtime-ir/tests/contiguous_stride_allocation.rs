//! Checked geometry must preserve allocation-free filling of existing storage.
use onnx_runtime_ir::{ContiguousStridePlan, compute_contiguous_strides};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
thread_local! {
    static ARMED:Cell<bool>=const{Cell::new(false)};
    static ALLOCS:Cell<usize>=const{Cell::new(0)};
}
struct Allocator;
fn record() {
    if ARMED.with(Cell::get) {
        ALLOCS.with(|n| n.set(n.get() + 1));
    }
}
// SAFETY: all allocation operations forward unchanged to System; observation
// uses allocation-free, constant-initialized, drop-free thread-local cells.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn observed(body: impl FnOnce()) -> usize {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ARMED.with(|a| a.set(false));
        }
    }
    ALLOCS.with(|n| n.set(0));
    ARMED.with(|a| a.set(true));
    let reset = Reset;
    body();
    drop(reset);
    ALLOCS.with(Cell::get)
}
#[test]
fn checked_planning_and_reused_storage_have_no_allocations_with_a_real_positive_control() {
    assert!(
        observed(|| {
            std::hint::black_box(compute_contiguous_strides(&[2, 3, 4]).unwrap());
        }) > 0,
        "allocating constructor must exercise the observer"
    );
    let high = [2usize; 16];
    let shapes: [&[usize]; 6] = [&[], &[2], &[2, 3, 4], &[usize::MAX, 2], &[2, 0, 3], &high];
    let mut out = [0i64; 16];
    let pointer = out.as_ptr();
    for shape in shapes {
        let expected = compute_contiguous_strides(shape).unwrap();
        let allocations = observed(|| {
            for _ in 0..4096 {
                let plan = ContiguousStridePlan::new(shape).unwrap();
                plan.write(&mut out[..shape.len()]).unwrap();
                std::hint::black_box(&out);
            }
        });
        assert_eq!(
            allocations,
            0,
            "rank {}: no temporary container for planning/filling",
            shape.len()
        );
        assert_eq!(&out[..shape.len()], expected);
        assert_eq!(out.as_ptr(), pointer);
    }
}
