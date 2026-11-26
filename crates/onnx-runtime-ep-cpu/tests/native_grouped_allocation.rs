//! Real allocator evidence for one native grouped execution workspace.
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, OutputMetadata, TensorMetadata, TensorMut,
    TensorView, WorkspaceView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{DataType, Node, NodeId};
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
fn admitted_native_snapshots_keep_float_borrows_and_replace_payload_allocations() {
    let provider = CpuExecutionProvider::new();
    let (m, s, e, n, k) = (4, 8, 4, 64, 64);
    let threshold = 256;
    let node = Node::new(NodeId(0), "GroupedMatMul", vec![], vec![]);
    let kernel = provider.get_kernel(&node, &[], 29).unwrap();
    for index in 0..4 {
        assert!(kernel.supports_strided_input(index));
    }
    for dt in [
        DataType::Float32,
        DataType::Float64,
        DataType::Float16,
        DataType::BFloat16,
    ] {
        let width = dt.byte_size();
        let acc_width = if dt == DataType::Float64 { 8 } else { 4 };
        let encode = |value: f32, count: usize| {
            let one = match dt {
                DataType::Float64 => f64::from(value).to_ne_bytes().to_vec(),
                DataType::Float16 => half::f16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
                DataType::BFloat16 => half::bf16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
                _ => value.to_ne_bytes().to_vec(),
            };
            one.repeat(count)
        };
        for size in [
            m * k * acc_width,
            e * k * n * acc_width,
            e * n * acc_width,
            m * s * 8,
            m * s * n * acc_width,
        ] {
            assert!(
                observed(threshold, || {
                    std::hint::black_box(vec![0u8; size]);
                })
                .1 > 0
            );
        }
        for mode in [0, 1, 3, 4] {
            let x = Backing::new(dt, [1, m, k], &encode(0.25, m * k), mode);
            let w = Backing::new(dt, [e, k, n], &encode(1., e * k * n), mode);
            let r = Backing::new(
                DataType::Int64,
                [1, m, s],
                &(0..m * s)
                    .flat_map(|i| ((i % e) as i64).to_ne_bytes())
                    .collect::<Vec<_>>(),
                mode,
            );
            let b = Backing::new(dt, [1, e, n], &encode(0.5, e * n), mode);
            let xs = [m, k];
            let xst = [x.strides[1], x.strides[2]];
            let rs = [m, s];
            let rst = [r.strides[1], r.strides[2]];
            let bs = [e, n];
            let bst = [b.strides[1], b.strides[2]];
            fn view<'a>(v: &Backing, shape: &'a [usize], strides: &'a [i64]) -> TensorView<'a> {
                TensorView::new(
                    DevicePtr(v.words.as_ptr().cast::<u8>().wrapping_add(v.base).cast()),
                    v.dtype,
                    shape,
                    strides,
                    onnx_runtime_ir::DeviceId::cpu(),
                )
                .with_byte_offset(v.offset)
            }
            let inputs = [
                view(&x, &xs, &xst),
                w.view(),
                view(&r, &rs, &rst),
                view(&b, &bs, &bst),
            ];
            let mut output = vec![0x33u128; (m * s * n * width).div_ceil(16)];
            let shape = [m, s, n];
            let strides = [(s * n) as i64, n as i64, 1];
            let mut out = TensorMut::new(
                DevicePtrMut(output.as_mut_ptr().cast()),
                dt,
                &shape,
                &strides,
                onnx_runtime_ir::DeviceId::cpu(),
            );
            let metadata = inputs
                .iter()
                .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
                .collect::<Vec<_>>();
            let facts = [OutputMetadata::from_view(&out)];
            let req = kernel
                .workspace_requirement_for_execution(
                    &inputs,
                    std::slice::from_ref(&out),
                    &metadata,
                    &facts,
                    None,
                )
                .unwrap()
                .require_ready()
                .unwrap();
            let snapshots = if matches!(dt, DataType::Float32 | DataType::Float64) && mode == 0 {
                0
            } else {
                m * k + e * k * n + e * n
            };
            let expected = ((m * s * n + snapshots) * acc_width)
                .next_multiple_of(align_of::<usize>())
                + m * s * size_of::<usize>();
            assert_eq!(req.bytes, expected as u64);
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
                assert_eq!(payloads, 0, "{dt:?}/{mode}");
            }
            let actual = unsafe {
                std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), m * s * n * width)
            };
            assert_eq!(actual, encode(16.5, m * s * n));
        }
    }
}
