//! Count actual native Pool execution allocations on its calling thread.
//! LpPool cannot route to MLAS/BNNS, so this reaches the same generic window
//! walker in both feature configurations. Caller-owned tensors and factory
//! construction are outside the observation; the complete execute call is in.
#![cfg(feature = "ops-cnn")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use onnx_runtime_ep_api::{
    DeviceId, DevicePtr, DevicePtrMut, KernelFactory, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::kernels::pooling::LpPoolFactory;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId};

thread_local! {
    // Constant, drop-free TLS cannot initialize/destruct or allocate recursively
    // inside the allocator. Other libtest threads are deliberately unobserved.
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
struct CountingAllocator;
fn record() {
    if ARMED.with(Cell::get) {
        CALLS.with(|count| count.set(count.get() + 1));
    }
}
// SAFETY: operations forward the original layout/pointers unchanged to System;
// observation only accesses constant, allocation-free, drop-free thread locals.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn observed(body: impl FnOnce()) -> usize {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            ARMED.with(|flag| flag.set(false));
        }
    }
    CALLS.with(|count| count.set(0));
    ARMED.with(|flag| flag.set(true));
    let guard = Disarm;
    body();
    drop(guard);
    CALLS.with(Cell::get)
}

#[test]
fn pool_window_allocations_do_not_scale_with_output_or_kernel_elements() {
    let control = observed(|| {
        std::hint::black_box(vec![0u8; 1024]);
    });
    assert!(
        control > 0,
        "the allocation probe must observe its positive control"
    );
    let mut counts = Vec::new();
    for length in [64, 4096] {
        for width in [1, 17] {
            let x = vec![1.0f32; length];
            let shape = [1, 1, length];
            let strides = [length as i64, length as i64, 1];
            let out_len = length - width + 1;
            let out_shape = [1, 1, out_len];
            let out_strides = [out_len as i64, out_len as i64, 1];
            let mut y = vec![f32::NAN; out_len];
            let mut node = Node::new(NodeId(0), "LpPool", vec![], vec![]);
            node.attributes
                .insert("kernel_shape".into(), Attribute::Ints(vec![width as i64]));
            node.attributes.insert("p".into(), Attribute::Int(1));
            let kernel = LpPoolFactory.create(&node, &[shape.to_vec()]).unwrap();
            let mut run = || {
                kernel
                    .execute(
                        &[TensorView::new(
                            DevicePtr(x.as_ptr().cast()),
                            DataType::Float32,
                            &shape,
                            &strides,
                            DeviceId::cpu(),
                        )],
                        &mut [TensorMut::new(
                            DevicePtrMut(y.as_mut_ptr().cast()),
                            DataType::Float32,
                            &out_shape,
                            &out_strides,
                            DeviceId::cpu(),
                        )],
                    )
                    .unwrap();
            };
            let cold = observed(&mut run);
            let warm = observed(&mut run);
            assert_eq!(cold, warm, "workspace must not hide in a warmed cache");
            // A missing/no-op execute cannot pass by allocating nothing.
            assert!(y.iter().all(|&value| value == width as f32));
            assert!(
                cold > 0,
                "the execute boundary still materializes its output values"
            );
            counts.push(cold);
            println!(
                "length={length} kernel={width} cold_allocations={cold} warm_allocations={warm}"
            );
        }
    }
    assert!(
        counts.iter().all(|count| *count == counts[0]),
        "Pool window traversal must reuse rank-sized scratch, not allocate per output/sample: {counts:?}"
    );
}
