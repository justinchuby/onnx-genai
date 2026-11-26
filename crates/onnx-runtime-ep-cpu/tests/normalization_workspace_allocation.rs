//! Real allocator evidence that normalization snapshots reuse admitted storage.
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, KernelFactory, OutputMetadata, TensorMetadata, TensorMut, TensorView,
    WorkspaceView,
};
use onnx_runtime_ep_cpu::{
    dtype::{FloatElem, NumericElem, to_dense},
    kernels::lp_normalization::LpNormalizationFactory,
};
use onnx_runtime_ir::{Attribute, DeviceId, Node, NodeId};
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
        if bytes >= TARGET_BYTES.with(Cell::get) {
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
fn lpnorm_snapshot_has_no_matrix_sized_heap_allocation_and_keeps_one_lane() {
    fn check<T: NumericElem + FloatElem + Default>() {
        let shape = [16, 16];
        let strides = [16, 1];
        let source = (0..256)
            .map(|i| T::from_f32_scalar((i % 17) as f32 - 7.25))
            .collect::<Vec<_>>();
        let mut output = vec![T::default(); 256];
        let bytes = 256 * size_of::<T>();
        let x = TensorView::new(
            DevicePtr(source.as_ptr().cast()),
            <T as NumericElem>::DTYPE,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let y = TensorMut::new(
            DevicePtrMut(output.as_mut_ptr().cast()),
            <T as NumericElem>::DTYPE,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let control = observed(bytes, || {
            drop(std::hint::black_box(to_dense::<T>(&x).unwrap()))
        });
        assert!(
            control.1 > 0,
            "counter must observe the former allocating materialization: {control:?}"
        );
        let mut node = Node::new(NodeId(0), "LpNormalization", vec![], vec![]);
        node.attributes.insert("axis".into(), Attribute::Int(1));
        node.attributes.insert("p".into(), Attribute::Int(2));
        let kernel = LpNormalizationFactory.create(&node, &[]).unwrap();
        let requirement = kernel
            .workspace_requirement(
                &[TensorMetadata::new(x.dtype, x.shape, true)],
                &[OutputMetadata::from_view(&y)],
            )
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(requirement.bytes, bytes as u64);
        let mut backing = vec![0u128; bytes.div_ceil(16)];
        let workspace = WorkspaceView::new(DevicePtrMut(backing.as_mut_ptr().cast()), bytes);
        let mut outputs = [y];
        kernel
            .execute_with_workspace(&[x], &mut outputs, Some(workspace))
            .unwrap();
        let count = observed(bytes, || {
            for _ in 0..32 {
                kernel
                    .execute_with_workspace(
                        std::hint::black_box(&[x]),
                        std::hint::black_box(&mut outputs),
                        Some(workspace),
                    )
                    .unwrap();
            }
        });
        assert_eq!(
            count.1, 0,
            "the admitted snapshot must replace every matrix-sized input allocation: {count:?}"
        );
        for row in output.chunks_exact(16) {
            let sum = row
                .iter()
                .map(|v| {
                    let v = v.to_f32();
                    v * v
                })
                .sum::<f32>();
            assert!((sum - 1.).abs() < 0.01);
        }
    }
    check::<f32>();
    check::<f64>();
    check::<half::f16>();
    check::<half::bf16>();
}

#[cfg(feature = "ops-cnn")]
#[test]
fn mvn_direct_and_snapshot_routes_avoid_matrix_sized_heap_allocations() {
    use onnx_runtime_ep_cpu::kernels::mean_variance_normalization::MeanVarianceNormalizationFactory;
    fn check<T: NumericElem + FloatElem + Default>() {
        let shape = [16, 16];
        let strides = [16, 1];
        let source = (0..256)
            .map(|i| T::from_f32_scalar((i % 17) as f32 - 7.25))
            .collect::<Vec<_>>();
        let mut output = vec![T::default(); 512];
        let bytes = 256 * size_of::<T>();
        let x = TensorView::new(
            DevicePtr(source.as_ptr().cast()),
            <T as NumericElem>::DTYPE,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let mut node = Node::new(NodeId(0), "MeanVarianceNormalization", vec![], vec![]);
        node.attributes
            .insert("axes".into(), Attribute::Ints(vec![1]));
        node.attributes
            .insert("epsilon".into(), Attribute::Float(0.125));
        let k = MeanVarianceNormalizationFactory.create(&node, &[]).unwrap();
        let mut scratch = vec![0u128; bytes.div_ceil(16)];
        let ws = WorkspaceView::new(DevicePtrMut(scratch.as_mut_ptr().cast()), bytes);
        let control = observed(bytes, || {
            drop(std::hint::black_box(to_dense::<T>(&x).unwrap()))
        });
        assert!(
            control.1 > 0,
            "allocating-copy positive control must be observed: {control:?}"
        );
        for strided in [false, true] {
            let strides = if strided { [32, 2] } else { [16, 1] };
            let mut y = TensorMut::new(
                DevicePtrMut(output.as_mut_ptr().cast()),
                <T as NumericElem>::DTYPE,
                &shape,
                &strides,
                DeviceId::cpu(),
            );
            let request = k
                .workspace_requirement_for_execution(
                    &[x],
                    std::slice::from_ref(&y),
                    &[TensorMetadata::new(x.dtype, x.shape, true)],
                    &[OutputMetadata::from_view(&y)],
                    None,
                )
                .unwrap()
                .require_ready()
                .unwrap();
            assert_eq!(request.bytes, if strided { bytes as u64 } else { 0 });
            let workspace = if strided { Some(ws) } else { None };
            k.execute_with_workspace(&[x], std::slice::from_mut(&mut y), workspace)
                .unwrap();
            let counts = observed(bytes, || {
                for _ in 0..32 {
                    k.execute_with_workspace(
                        std::hint::black_box(&[x]),
                        std::slice::from_mut(&mut y),
                        workspace,
                    )
                    .unwrap();
                }
            });
            assert_eq!(
                counts.1, 0,
                "serial MVN route strided={strided}: {counts:?}"
            );
            for row in 0..16 {
                let sum = (0..16)
                    .map(|column| {
                        output[(row * 16 + column) * if strided { 2 } else { 1 }].to_f32()
                    })
                    .sum::<f32>();
                assert!(sum.abs() < 0.05);
            }
        }
    }
    check::<f32>();
    check::<f64>();
    check::<half::f16>();
    check::<half::bf16>();
}
