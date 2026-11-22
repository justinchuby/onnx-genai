//! Count complete QLinearMatMul execute calls on ARM64's borrowed native GEMM.
//! Per-row A quantization cannot route to MLAS, even in reference builds. The
//! small work stays serial; inputs, outputs and factory setup are unobserved.
#![cfg(all(target_arch = "aarch64", feature = "ops-quantized"))]

use onnx_runtime_ep_api::{
    DeviceId, DevicePtr, DevicePtrMut, KernelFactory, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::{DerivedWeightMemory, kernels::qlinear_matmul::QLinearMatMulFactory};
use onnx_runtime_ir::{DataType, Node, NodeId, compute_contiguous_strides};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

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

fn view<'a>(
    ptr: *const std::ffi::c_void,
    dtype: DataType,
    shape: &'a [usize],
    strides: &'a [i64],
) -> TensorView<'a> {
    TensorView::new(DevicePtr(ptr), dtype, shape, strides, DeviceId::cpu())
}

#[test]
fn native_qgemm_compensation_allocations_do_not_scale_with_batches() {
    assert!(
        observed(|| {
            std::hint::black_box(vec![0u8; 1024]);
        }) > 0,
        "the allocation probe must observe its positive control"
    );
    for b_zero in [0u8, 1] {
        let mut counts = Vec::new();
        for batches in [1, 2, 17] {
            let a_shape = [batches, 3, 7];
            let a_strides = compute_contiguous_strides(&a_shape);
            let b_shape = [7, 9];
            let b_strides = compute_contiguous_strides(&b_shape);
            let quant_shape = [batches, 3, 1];
            let quant_strides = compute_contiguous_strides(&quant_shape);
            let y_shape = [batches, 3, 9];
            let y_strides = compute_contiguous_strides(&y_shape);
            let a = vec![129u8; batches * 3 * 7];
            let b = [2u8; 7 * 9];
            let a_scale: Vec<f32> = (0..batches).flat_map(|_| [1., 2., 3.]).collect();
            let a_zero = vec![128u8; batches * 3];
            let b_scale = [1f32];
            let b_zero_value = [b_zero];
            let y_scale = [1f32];
            let y_zero = [128u8];
            let mut y = vec![0u8; batches * 3 * 9];
            let inputs = [
                view(a.as_ptr().cast(), DataType::Uint8, &a_shape, &a_strides),
                view(
                    a_scale.as_ptr().cast(),
                    DataType::Float32,
                    &quant_shape,
                    &quant_strides,
                ),
                view(
                    a_zero.as_ptr().cast(),
                    DataType::Uint8,
                    &quant_shape,
                    &quant_strides,
                ),
                view(b.as_ptr().cast(), DataType::Int8, &b_shape, &b_strides),
                view(b_scale.as_ptr().cast(), DataType::Float32, &[], &[]),
                view(b_zero_value.as_ptr().cast(), DataType::Int8, &[], &[]),
                view(y_scale.as_ptr().cast(), DataType::Float32, &[], &[]),
                view(y_zero.as_ptr().cast(), DataType::Uint8, &[], &[]),
            ];
            let kernel = QLinearMatMulFactory::new(false, DerivedWeightMemory::default())
                .create(&Node::new(NodeId(0), "QLinearMatMul", vec![], vec![]), &[])
                .unwrap();
            let mut run = || {
                kernel
                    .execute(
                        &inputs,
                        &mut [TensorMut::new(
                            DevicePtrMut(y.as_mut_ptr().cast()),
                            DataType::Uint8,
                            &y_shape,
                            &y_strides,
                            DeviceId::cpu(),
                        )],
                    )
                    .unwrap()
            };
            run();
            let calls = observed(&mut run);
            assert!(
                calls > 0,
                "the complete execute still allocates parameter/geometry storage"
            );
            for (index, &value) in y.iter().enumerate() {
                let row = (index / 9) % 3;
                let expected = 128 + 7 * (2 - b_zero) * (row as u8 + 1);
                assert_eq!(value, expected, "batches={batches} row={row} zero={b_zero}");
            }
            println!("b_zero={b_zero} batches={batches} execute_allocations={calls}");
            counts.push(calls);
        }
        assert!(
            counts.iter().all(|&count| count == counts[0]),
            "native compensation must not allocate per batch: b_zero={b_zero} counts={counts:?}"
        );
    }
}
