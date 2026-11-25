//! Actual native Conv execute allocation counts on serial ARM64 geometry.
//! MLAS builds select a different Conv implementation and are deliberately excluded.
#![cfg(all(target_arch = "aarch64", feature = "ops-cnn", not(feature = "mlas")))]

use onnx_runtime_ep_api::{
    DeviceId, DevicePtr, DevicePtrMut, KernelFactory, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::kernels::conv::ConvFactory;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, compute_contiguous_strides};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    // Constant, drop-free TLS cannot initialize/destruct or allocate recursively
    // inside the allocator. Other libtest threads are deliberately unobserved.
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static TARGET_BYTES: Cell<usize> = const { Cell::new(0) };
    static MATCHES: Cell<usize> = const { Cell::new(0) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
struct CountingAllocator;
fn record(bytes: usize) {
    if ARMED.with(Cell::get) {
        CALLS.with(|count| count.set(count.get() + 1));
        if bytes == TARGET_BYTES.with(Cell::get) {
            MATCHES.with(|count| count.set(count.get() + 1));
        }
    }
}
// SAFETY: operations forward the original layout/pointers unchanged to System;
// observation only accesses constant, allocation-free, drop-free thread locals.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn observed(bytes: usize, body: impl FnOnce()) -> (usize, usize) {
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            ARMED.with(|flag| flag.set(false));
        }
    }
    CALLS.with(|count| count.set(0));
    MATCHES.with(|count| count.set(0));
    TARGET_BYTES.with(|target| target.set(bytes));
    ARMED.with(|flag| flag.set(true));
    let guard = Disarm;
    body();
    drop(guard);
    (CALLS.with(Cell::get), MATCHES.with(Cell::get))
}

#[test]
fn native_conv_im2col_buffer_is_optional_and_reused_across_batches_and_groups() {
    let control = observed(280, || {
        std::hint::black_box(vec![0u8; 280]);
    });
    assert!(
        control.0 > 0 && control.1 == 1,
        "allocation-size positive control must execute: {control:?}"
    );
    for group in [1, 2] {
        for pointwise in [false, true] {
            // group1 is dilated to exclude BNNS. A 1x1 kernel with dilation2
            // deliberately retains the existing non-pointwise classification.
            // group2 has multiple channels/output channels, excluding depthwise.
            if group == 1 && pointwise {
                continue;
            }
            let (height, width, kh, kw, dilation) = if group == 1 {
                (7, 9, 2, 3, 2)
            } else {
                (5, 7, 1, 1, if pointwise { 1 } else { 2 })
            };
            let oh = height - (kh - 1) * dilation;
            let ow = width - (kw - 1) * dilation;
            let col_bytes = 2 * kh * kw * oh * ow * 4;
            let mut totals = Vec::new();
            for batch in [0, 1, 2, 17] {
                let x_shape = [batch, group * 2, height, width];
                let w_shape = [group * 3, 2, kh, kw];
                let y_shape = [batch, group * 3, oh, ow];
                let xs = compute_contiguous_strides(&x_shape);
                let ws = compute_contiguous_strides(&w_shape);
                let ys = compute_contiguous_strides(&y_shape);
                let x = vec![1f32; x_shape.iter().product()];
                let w = vec![1f32; w_shape.iter().product()];
                let mut y = vec![f32::NAN; y_shape.iter().product()];
                let mut node = Node::new(NodeId(0), "Conv", vec![], vec![]);
                node.attributes
                    .insert("group".into(), Attribute::Int(group as i64));
                node.attributes.insert(
                    "dilations".into(),
                    Attribute::Ints(vec![dilation as i64; 2]),
                );
                let kernel = ConvFactory
                    .create(&node, &[x_shape.to_vec(), w_shape.to_vec()])
                    .unwrap();
                let mut run = || {
                    kernel
                        .execute(
                            &[
                                TensorView::new(
                                    DevicePtr(x.as_ptr().cast()),
                                    DataType::Float32,
                                    &x_shape,
                                    &xs,
                                    DeviceId::cpu(),
                                ),
                                TensorView::new(
                                    DevicePtr(w.as_ptr().cast()),
                                    DataType::Float32,
                                    &w_shape,
                                    &ws,
                                    DeviceId::cpu(),
                                ),
                            ],
                            &mut [TensorMut::new(
                                DevicePtrMut(y.as_mut_ptr().cast()),
                                DataType::Float32,
                                &y_shape,
                                &ys,
                                DeviceId::cpu(),
                            )],
                        )
                        .unwrap()
                };
                run(); // Exclude one-time runtime initialization, not a scratch cache.
                let (total, cols) = observed(col_bytes, &mut run);
                assert!(y.iter().all(|&v| v == (2 * kh * kw) as f32));
                assert_eq!(
                    cols,
                    usize::from(batch != 0 && !pointwise),
                    "group={group} pointwise={pointwise} batch={batch} total={total}"
                );
                if batch != 0 {
                    assert!(
                        total > 0,
                        "nonempty native execution still owns staging storage"
                    );
                }
                println!(
                    "group={group} pointwise={pointwise} batch={batch} col_bytes={col_bytes} col_allocations={cols} execute_allocations={total}"
                );
                if batch != 0 {
                    totals.push(total);
                }
            }
            assert!(
                totals.iter().all(|&n| n == totals[0]),
                "complete serial execute allocations must not grow with batches: {totals:?}"
            );
        }
    }
}
