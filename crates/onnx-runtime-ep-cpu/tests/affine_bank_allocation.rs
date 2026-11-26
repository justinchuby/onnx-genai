//! Real allocator evidence for borrowed affine expert-bank storage.
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, OutputMetadata, TensorMetadata, TensorMut,
    TensorView, WorkspaceView,
};
use onnx_runtime_ep_cpu::{CpuExecutionProvider, dtype::to_dense};
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
fn affine_grouped_projection_has_no_bank_sized_copies_for_any_bank_layout() {
    let (experts, width, depth) = (64usize, 128usize, 64usize);
    let packed = depth / 2;
    let codes = (0..experts * width * packed)
        .map(|i| (i * 29 + 17) as u8)
        .collect::<Vec<_>>();
    let zeros = (0..experts * width * packed)
        .map(|i| (i * 37 + 3) as u8)
        .collect::<Vec<_>>();
    let scales = (0..experts * width * depth)
        .map(|i| (1 + (i % 7)) as f32 * 0.125)
        .collect::<Vec<_>>();
    let scale_bytes = scales
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    let activation = (0..depth)
        .map(|i| (i as f32 % 7. - 3.) * 0.25)
        .collect::<Vec<_>>();
    let groups = [3i64];
    let mut output = vec![0f32; width];
    let mut node = Node::new(
        NodeId(0),
        onnx_runtime_ir::quantized_grouped::OP,
        vec![],
        vec![],
    );
    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
    for (name, value) in [("depth", depth as i64), ("block_size", 1), ("zero", 7)] {
        node.attributes.insert(name.into(), Attribute::Int(value));
    }
    let kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &[], 1)
        .unwrap();
    for mode in [0, 1, 3, 4] {
        let c = Backing::new(DataType::Uint8, [experts, width, packed], &codes, mode);
        let s = Backing::new(
            DataType::Float32,
            [experts, width, depth],
            &scale_bytes,
            mode,
        );
        let z = Backing::new(DataType::Uint8, [experts, width, packed], &zeros, mode);
        let inputs = [
            TensorView::new(
                DevicePtr(activation.as_ptr().cast()),
                DataType::Float32,
                &[1, 64],
                &[64, 1],
                onnx_runtime_ir::DeviceId::cpu(),
            ),
            c.view(),
            TensorView::new(
                DevicePtr(groups.as_ptr().cast()),
                DataType::Int64,
                &[1, 1],
                &[1, 1],
                onnx_runtime_ir::DeviceId::cpu(),
            ),
            s.view(),
            z.view(),
        ];
        assert!(kernel.supports_strided_input(1));
        assert!(kernel.supports_strided_input(3));
        assert!(kernel.supports_strided_input(4));
        let threshold = experts * width * packed;
        for slot in [1, 3, 4] {
            let control = observed(threshold, || {
                if slot == 3 {
                    drop(std::hint::black_box(
                        to_dense::<f32>(&inputs[slot]).unwrap(),
                    ));
                } else {
                    drop(std::hint::black_box(to_dense::<u8>(&inputs[slot]).unwrap()));
                }
            });
            assert!(
                control.1 > 0,
                "counter must detect allocating bank copy slot={slot}"
            );
        }
        let mut y = TensorMut::new(
            DevicePtrMut(output.as_mut_ptr().cast()),
            DataType::Float32,
            &[1, 1, 128],
            &[128, 128, 1],
            onnx_runtime_ir::DeviceId::cpu(),
        );
        let metadata = inputs
            .iter()
            .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
            .collect::<Vec<_>>();
        let facts = [OutputMetadata::from_view(&y)];
        let request = kernel
            .workspace_requirement(&metadata, &facts)
            .unwrap()
            .require_ready()
            .unwrap();
        let mut backing = vec![0u128; (request.bytes as usize).div_ceil(16)];
        let workspace = WorkspaceView::new(
            DevicePtrMut(backing.as_mut_ptr().cast()),
            request.bytes as usize,
        );
        kernel
            .execute_with_workspace(&inputs, std::slice::from_mut(&mut y), Some(workspace))
            .unwrap();
        let counts = observed(threshold, || {
            for _ in 0..4 {
                kernel
                    .execute_with_workspace(
                        std::hint::black_box(&inputs),
                        std::slice::from_mut(&mut y),
                        Some(workspace),
                    )
                    .unwrap();
            }
        });
        assert_eq!(counts.1, 0, "mode={mode}: no bank-sized copies: {counts:?}");
        for (n, &value) in output.iter().enumerate() {
            let row = 3 * width + n;
            let mut expected = 0f32;
            for (column, &x) in activation.iter().enumerate() {
                let q = (codes[row * packed + column / 2] >> ((column % 2) * 4)) & 15;
                let zero = (zeros[row * packed + column / 2] >> ((column % 2) * 4)) & 15;
                let weight =
                    f32::from(i16::from(q) - i16::from(zero)) * scales[row * depth + column];
                expected += x * weight;
            }
            assert_eq!(value.to_bits(), expected.to_bits(), "mode={mode} n={n}");
        }
    }
}

#[test]
fn affine_grouped_payloads_are_admitted_once_without_heap_copies() {
    let (m, s, e, n, k) = (4usize, 8usize, 4usize, 64usize, 32usize);
    let r = m * s;
    let blocks = k / 2;
    let mut node = Node::new(
        NodeId(0),
        onnx_runtime_ir::quantized_grouped::OP,
        vec![],
        vec![],
    );
    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
    for (name, value) in [("depth", k), ("block_size", 2), ("zero", 7)] {
        node.attributes
            .insert(name.into(), Attribute::Int(value as i64));
    }
    let kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &[], 1)
        .unwrap();
    let activation = vec![0.25f32; m * k];
    let routes = (0..r).map(|i| (i % e) as i64).collect::<Vec<_>>();
    let codes = vec![0x88; e * n * k / 2];
    let scales = vec![1f32; e * n * blocks];
    let zeros = vec![0x77; e * n * blocks.div_ceil(2)];
    let xbytes = activation
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    let rbytes = routes
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    let sbytes = scales
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    let mut result = vec![0f32; m * s * n];
    let threshold = k * 4;
    for bytes in [
        m * k * 4,
        r * size_of::<i64>(),
        r * size_of::<usize>(),
        m * s * n * 4,
        k * 4,
    ] {
        assert!(
            observed(threshold, || {
                std::hint::black_box(vec![0u8; bytes]);
            })
            .1 > 0
        );
    }
    for mode in [0, 1, 3, 4] {
        let x = Backing::new(DataType::Float32, [1, m, k], &xbytes, mode);
        let routes = Backing::new(DataType::Int64, [1, m, s], &rbytes, mode);
        let weights = Backing::new(DataType::Uint8, [e, n, k / 2], &codes, mode);
        let scales = Backing::new(DataType::Float32, [e, n, blocks], &sbytes, mode);
        let zeros = Backing::new(DataType::Uint8, [e, n, blocks.div_ceil(2)], &zeros, mode);
        let xs = [m, k];
        let xstep = [x.strides[1], x.strides[2]];
        let rs = [m, s];
        let rstep = [routes.strides[1], routes.strides[2]];
        let xview = TensorView::new(
            x.view().data,
            DataType::Float32,
            &xs,
            &xstep,
            onnx_runtime_ir::DeviceId::cpu(),
        )
        .with_byte_offset(x.offset);
        let rview = TensorView::new(
            routes.view().data,
            DataType::Int64,
            &rs,
            &rstep,
            onnx_runtime_ir::DeviceId::cpu(),
        )
        .with_byte_offset(routes.offset);
        let inputs = [xview, weights.view(), rview, scales.view(), zeros.view()];
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
            .workspace_requirement(&metadata, &[OutputMetadata::from_view(&out)])
            .unwrap()
            .require_ready()
            .unwrap();
        let mut backing = vec![0xaaaaaaaaaaaaaaaau128; (req.bytes as usize).div_ceil(16)];
        let lane = WorkspaceView::new(
            DevicePtrMut(backing.as_mut_ptr().cast()),
            req.bytes as usize,
        );
        for _ in 0..4 {
            let (_, copies) = observed(threshold, || {
                kernel
                    .execute_with_workspace(&inputs, std::slice::from_mut(&mut out), Some(lane))
                    .unwrap()
            });
            assert_eq!(
                copies, 0,
                "mode={mode}: no payload-sized allocations during execution"
            );
        }
        assert!(result.iter().all(|&v| v == 8.));
    }
}
