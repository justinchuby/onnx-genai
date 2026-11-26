//! Real allocator evidence for borrowed affine expert-bank storage.
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, OutputMetadata, TensorMetadata, TensorMut,
    TensorView, WorkspaceView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId};
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
fn moe_payload_allocations_are_funded_by_one_workspace() {
    let (t, e, h, i, k) = (2usize, 128usize, 64usize, 64usize, 32usize);
    let provider = CpuExecutionProvider::new();
    let mut node = Node::new(
        NodeId(0),
        onnx_runtime_optimizer::affine_moe::OP,
        vec![],
        vec![],
    );
    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
    for (key, value) in [
        ("k", k),
        ("first_block", 2),
        ("first_zero", 7),
        ("down_block", 2),
        ("down_zero", 7),
    ] {
        node.attributes
            .insert(key.into(), Attribute::Int(value as i64));
    }
    for (key, value) in [("alpha", 1.3), ("beta", 0.2), ("limit", 7.)] {
        node.attributes.insert(key.into(), Attribute::Float(value));
    }
    let kernel = provider.get_kernel(&node, &[], 1).unwrap();
    let x = vec![0.25f32; t * h];
    let router = vec![1f32; t * e];
    let w1 = vec![0x88u8; e * 2 * i * h / 2];
    let s1 = vec![0.125f32; e * 2 * i * h / 2];
    let z1 = vec![0x77u8; e * 2 * i * (h / 2).div_ceil(2)];
    let w2 = vec![0x88u8; e * h * i / 2];
    let s2 = vec![0.25f32; e * h * i / 2];
    let z2 = vec![0x77u8; e * h * (i / 2).div_ceil(2)];
    let shapes = [
        vec![t, h],
        vec![t, e],
        vec![e, 2 * i, h / 2],
        vec![e, 2 * i, h / 2],
        vec![e, 2 * i, (h / 2).div_ceil(2)],
        vec![e, h, i / 2],
        vec![e, h, i / 2],
        vec![e, h, (i / 2).div_ceil(2)],
    ];
    let strides = shapes
        .iter()
        .map(|s| onnx_runtime_ir::compute_contiguous_strides(s))
        .collect::<Vec<_>>();
    let pointers = [
        x.as_ptr().cast(),
        router.as_ptr().cast(),
        w1.as_ptr().cast(),
        s1.as_ptr().cast(),
        z1.as_ptr().cast(),
        w2.as_ptr().cast(),
        s2.as_ptr().cast(),
        z2.as_ptr().cast(),
    ];
    let inputs = (0..8)
        .map(|j| {
            TensorView::new(
                DevicePtr(pointers[j]),
                if [2, 4, 5, 7].contains(&j) {
                    DataType::Uint8
                } else {
                    DataType::Float32
                },
                &shapes[j],
                &strides[j],
                onnx_runtime_ir::DeviceId::cpu(),
            )
        })
        .collect::<Vec<_>>();
    let mut y = vec![0f32; t * h];
    let output_shape = [t, h];
    let output_strides = [h as i64, 1];
    let mut output = TensorMut::new(
        DevicePtrMut(y.as_mut_ptr().cast()),
        DataType::Float32,
        &output_shape,
        &output_strides,
        onnx_runtime_ir::DeviceId::cpu(),
    );
    let metadata = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
        .collect::<Vec<_>>();
    let facts = [OutputMetadata::from_view(&output)];
    let req = kernel
        .workspace_requirement_for_execution(
            &inputs,
            std::slice::from_ref(&output),
            &metadata,
            &facts,
            None,
        )
        .unwrap()
        .require_ready()
        .unwrap();
    let mut backing = vec![0u128; (req.bytes as usize).div_ceil(16)];
    let lane = WorkspaceView::new(
        DevicePtrMut(backing.as_mut_ptr().cast()),
        req.bytes as usize,
    );
    // Independent positive controls for every former payload Vec. black_box
    // prevents allocation elimination; counters observe real allocator calls.
    for bytes in [
        t * h * 4,
        t * e * 4,
        e * size_of::<usize>(),
        k * 4,
        k * 4,
        i * 4,
    ] {
        assert!(
            observed(k * 4, || {
                std::hint::black_box(vec![0u8; bytes]);
            })
            .1 > 0
        );
    }
    for _ in 0..4 {
        let (_, payloads) = observed(k * 4, || {
            kernel
                .execute_with_workspace(&inputs, std::slice::from_mut(&mut output), Some(lane))
                .unwrap()
        });
        assert_eq!(
            payloads, 0,
            "no payload-sized heap allocation during execution"
        );
    }
    assert!(y.iter().all(|v| v.is_finite()));
}
