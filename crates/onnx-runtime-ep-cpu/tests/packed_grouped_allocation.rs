//! Real allocator evidence for one packed grouped execution workspace.
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

struct Backing {
    words: Vec<u128>,
    shape: [usize; 3],
    strides: [i64; 3],
    dtype: DataType,
    base: usize,
    offset: usize,
}
impl Backing {
    fn new(dtype: DataType, shape: [usize; 3], bytes: &[u8], mode: usize) -> Self {
        let [e, n, c] = shape;
        let width = dtype.byte_size();
        let dense = mode == 0 || mode == 4;
        let step = if dense { 1 } else { 2 };
        let row = if dense { c } else { 2 * c + 1 };
        let expert = if dense { n * row } else { n * row + 3 };
        let reversed = matches!(mode, 2 | 3);
        let origin = if reversed && c > 0 { (c - 1) * step } else { 0 };
        let base = usize::from(mode == 3 || mode == 4);
        let strides = [
            expert as i64,
            row as i64,
            if reversed {
                -(step as i64)
            } else {
                step as i64
            },
        ];
        let span = (e.saturating_sub(1) * expert
            + n.saturating_sub(1) * row
            + c.saturating_sub(1) * step
            + 1)
            * width
            + base;
        let mut words = vec![0xa5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5u128; span.div_ceil(16).max(1)];
        // SAFETY: byte writes preserve valid u128 storage and remain within the
        // allocated span. No tensor view or typed value reference exists yet.
        let storage = unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 16)
        };
        for i in 0..e {
            for j in 0..n {
                for k in 0..c {
                    let logical = ((i * n + j) * c + k) * width;
                    let offset = (origin as i64
                        + i as i64 * strides[0]
                        + j as i64 * strides[1]
                        + k as i64 * strides[2]) as usize
                        * width
                        + base;
                    storage[offset..offset + width]
                        .copy_from_slice(&bytes[logical..logical + width]);
                }
            }
        }
        Self {
            words,
            shape,
            strides,
            dtype,
            base,
            offset: origin * width,
        }
    }
    fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(unsafe { self.words.as_ptr().cast::<u8>().add(self.base).cast() }),
            self.dtype,
            &self.shape,
            &self.strides,
            onnx_runtime_ir::DeviceId::cpu(),
        )
        .with_byte_offset(self.offset)
    }
}

#[test]
fn packed_projection_payloads_reuse_one_lane_for_all_supported_input_layouts() {
    let (m, s, e, n, k) = (4usize, 8usize, 4usize, 64usize, 64usize);
    let routes_count = m * s;
    let mut node = Node::new(
        NodeId(0),
        onnx_runtime_ir::quantized_grouped::PACKED_OP,
        vec![],
        vec![],
    );
    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
    node.attributes
        .insert("format".into(), Attribute::String("gguf:q4_0".into()));
    node.attributes.insert(
        "shape".into(),
        Attribute::Ints(vec![e as i64, n as i64, k as i64]),
    );
    let kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &[], 1)
        .unwrap();
    assert!(kernel.supports_strided_input(0) && kernel.supports_strided_input(2));
    assert!(!kernel.supports_strided_input(1));
    let activation = vec![0.25f32; m * k];
    let routes = (0..routes_count)
        .map(|i| (i % e) as i64)
        .collect::<Vec<_>>();
    let xbytes = activation
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    let rbytes = routes
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    let packed_bytes = e * n * (k / 32) * 18;
    let threshold = k * 4;
    for size in [
        m * k * 4,
        routes_count * size_of::<i64>(),
        routes_count * size_of::<usize>(),
        m * s * n * 4,
        k * 4,
    ] {
        assert!(
            observed(threshold, || {
                std::hint::black_box(vec![0u8; size]);
            })
            .1 > 0
        );
    }
    let mut result = vec![123f32; m * s * n];
    for mode in [0, 1, 3, 4] {
        let x = Backing::new(DataType::Float32, [1, m, k], &xbytes, mode);
        let routes = Backing::new(DataType::Int64, [1, m, s], &rbytes, mode);
        let mut bank = vec![0xccu8; usize::from(mode == 4)];
        for _ in 0..e * n * (k / 32) {
            bank.extend_from_slice(&[0, 0x3c]);
            bank.extend_from_slice(&[0x99; 16]);
        }
        let xshape = [m, k];
        let xstrides = [x.strides[1], x.strides[2]];
        let rshape = [m, s];
        let rstrides = [routes.strides[1], routes.strides[2]];
        let bshape = [packed_bytes];
        let xview = TensorView::new(
            x.view().data,
            DataType::Float32,
            &xshape,
            &xstrides,
            onnx_runtime_ir::DeviceId::cpu(),
        )
        .with_byte_offset(x.offset);
        let rview = TensorView::new(
            routes.view().data,
            DataType::Int64,
            &rshape,
            &rstrides,
            onnx_runtime_ir::DeviceId::cpu(),
        )
        .with_byte_offset(routes.offset);
        let bview = TensorView::new(
            DevicePtr(bank.as_ptr().cast()),
            DataType::Uint8,
            &bshape,
            &[1],
            onnx_runtime_ir::DeviceId::cpu(),
        )
        .with_byte_offset(usize::from(mode == 4));
        let inputs = [xview, bview, rview];
        let shape = [m, s, n];
        let strides = [(s * n) as i64, n as i64, 1];
        let mut out = TensorMut::new(
            DevicePtrMut(result.as_mut_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        let metadata = inputs
            .iter()
            .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
            .collect::<Vec<_>>();
        let req = kernel
            .workspace_requirement_for_execution(
                &inputs,
                std::slice::from_ref(&out),
                &metadata,
                &[OutputMetadata::from_view(&out)],
                None,
            )
            .unwrap()
            .require_ready()
            .unwrap();
        let mut backing = vec![0xaaaaaaaaaaaaaaaau128; (req.bytes as usize).div_ceil(16)];
        let lane = WorkspaceView::new(
            DevicePtrMut(backing.as_mut_ptr().cast()),
            req.bytes as usize,
        );
        for _ in 0..4 {
            let (_, payloads) = observed(threshold, || {
                kernel
                    .execute_with_workspace(&inputs, std::slice::from_mut(&mut out), Some(lane))
                    .unwrap()
            });
            assert_eq!(
                payloads, 0,
                "mode={mode}: no payload-sized heap allocations during execution"
            );
        }
        assert!(
            result.iter().all(|&v| v == 16.),
            "mode={mode}, first outputs {:?}",
            &result[..8]
        );
    }
}
