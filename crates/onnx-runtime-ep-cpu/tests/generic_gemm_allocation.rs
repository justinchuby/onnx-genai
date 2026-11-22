//! Probe the real Generic MatMul boundary: warmed column-parallel execution
//! must not allocate weight-sized strip copies. Small geometry, row-reference
//! metadata and caller/result output allocations are deliberately outside this
//! claim. The counter includes the dispatching thread and every Rayon worker.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use onnx_runtime_ep_api::{DeviceId, DevicePtr, DevicePtrMut, Kernel, TensorMut, TensorView};
use onnx_runtime_ep_cpu::{backend::CpuBackend, kernels::matmul::MatMulKernel};
use onnx_runtime_ir::DataType;

const LARGE_ALLOCATION: usize = 64 * 1024;
static ARMED: AtomicBool = AtomicBool::new(false);
static LARGE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PARTICIPANTS: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    // Constant, drop-free TLS is safe to read inside the allocator: no lazy
    // initialization, destructor registration or recursive allocation.
    static PARTICIPANT: Cell<bool> = const { Cell::new(false) };
}

fn record(bytes: usize) {
    if bytes >= LARGE_ALLOCATION && ARMED.load(Ordering::Relaxed) && PARTICIPANT.with(Cell::get) {
        LARGE_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}
struct ProbeAllocator;
// SAFETY: all allocation operations forward to System with unchanged layouts
// and pointers; observation only updates allocation-free TLS/atomic counters.
unsafe impl GlobalAlloc for ProbeAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: ProbeAllocator = ProbeAllocator;

fn observed_bytes(body: impl FnOnce()) -> usize {
    LARGE_BYTES.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::SeqCst);
    body();
    ARMED.store(false, Ordering::SeqCst);
    LARGE_BYTES.load(Ordering::Relaxed)
}

#[test]
fn generic_matmul_borrows_weight_columns_without_large_allocations() {
    const TEST: &str = "generic_matmul_borrows_weight_columns_without_large_allocations";
    // Set the backend only on a child process, avoiding environment mutation
    // races with other libtest threads or the application's own default backend.
    if std::env::var_os("NXRT_GENERIC_GEMM_ALLOCATION_CHILD").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture"])
            .env("NXRT_GENERIC_GEMM_ALLOCATION_CHILD", "1")
            .env("NXRT_CPU_GEMM_BACKEND", "generic")
            .output()
            .expect("launch the isolated Generic MatMul allocation probe");
        assert!(
            output.status.success(),
            "Generic allocation child failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
        print!("{}", String::from_utf8_lossy(&output.stdout));
        return;
    }
    assert_eq!(CpuBackend::auto_detect(), CpuBackend::Generic);
    PARTICIPANT.with(|flag| flag.set(true));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .start_handler(|_| {
            PARTICIPANT.with(|flag| flag.set(true));
            PARTICIPANTS.fetch_add(1, Ordering::Relaxed);
        })
        .build()
        .unwrap();
    pool.broadcast(|_| assert!(PARTICIPANT.with(Cell::get)));
    assert_eq!(
        PARTICIPANTS.load(Ordering::Relaxed),
        4,
        "all measured workers must be registered"
    );

    let (m, k, n) = (3usize, 1025usize, 257usize);
    let a: Vec<f32> = (0..m * k).map(|i| [1.0, 2.0, -1.0][i / k]).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % n) % 9) as f32 * 0.25).collect();
    let mut y = vec![0.0f32; m * n];
    let a_shape = [m, k];
    let b_shape = [k, n];
    let y_shape = [m, n];
    let a_strides = [k as i64, 1];
    let b_strides = [n as i64, 1];
    let mut kernel = MatMulKernel::default();
    kernel.set_constant_inputs(&[false, true]);
    let mut run = || {
        // The owned vectors outlive each view, and y is exclusively borrowed
        // until this synchronous execution (including every worker) returns.
        pool.install(|| {
            let inputs = [
                TensorView::new(
                    DevicePtr(a.as_ptr().cast()),
                    DataType::Float32,
                    &a_shape,
                    &a_strides,
                    DeviceId::cpu(),
                ),
                TensorView::new(
                    DevicePtr(b.as_ptr().cast()),
                    DataType::Float32,
                    &b_shape,
                    &b_strides,
                    DeviceId::cpu(),
                ),
            ];
            kernel
                .execute(
                    &inputs,
                    &mut [TensorMut::new(
                        DevicePtrMut(y.as_mut_ptr().cast()),
                        DataType::Float32,
                        &y_shape,
                        &b_strides,
                        DeviceId::cpu(),
                    )],
                )
                .unwrap()
        });
    };
    let weight_bytes = b.len() * std::mem::size_of::<f32>();
    assert!(
        weight_bytes / 4 > LARGE_ALLOCATION,
        "old per-worker B strips must exceed the probe threshold"
    );
    let control = observed_bytes(|| {
        pool.install(|| {
            std::hint::black_box(vec![0u8; weight_bytes]);
        })
    });
    assert!(
        control >= weight_bytes,
        "the armed allocator must observe a deliberate weight-sized worker allocation"
    );
    let cold = observed_bytes(&mut run);
    assert_eq!(
        cold, 0,
        "Generic column GEMM allocated {cold} large bytes on the first call; borrowing must not hide a retained weight copy"
    );
    for _ in 0..4 {
        run();
    }
    let measured = observed_bytes(|| {
        for _ in 0..4 {
            run();
        }
    });
    assert_eq!(
        measured, 0,
        "Generic column GEMM allocated {measured} large bytes; borrow B and output rows instead of copying matrix strips"
    );
    println!(
        "registered_workers=4 threshold={LARGE_ALLOCATION} positive_control_bytes={control} cold_large_bytes={cold} measured_large_bytes={measured}"
    );
    for row in 0..m {
        for col in 0..n {
            // The original accumulator starts at +0, including a negative
            // row multiplied by an all-zero weight column.
            let expected = 0.0 + k as f32 * [1.0, 2.0, -1.0][row] * (col % 9) as f32 * 0.25;
            assert_eq!(y[row * n + col].to_bits(), expected.to_bits());
        }
    }
}
