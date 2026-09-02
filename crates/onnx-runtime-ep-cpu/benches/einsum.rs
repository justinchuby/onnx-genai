mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::OsString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::{FloatDType, Tensor, assert_close, float_values, make_kernel, process_cpu_time};
use criterion::{BenchmarkId, Criterion, Throughput, black_box};
use onnx_runtime_ep_api::{Kernel, KernelFactory};
use onnx_runtime_ep_cpu::kernels::einsum::{
    EINSUM_MODE_ENV, EinsumFactory, benchmark_scratch_capacity_bytes,
};
use onnx_runtime_ir::{Attribute, Node, NodeId};

struct CountingAllocator;

static ALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn reset_allocations() {
    ALLOCATION_CALLS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn allocations() -> (u64, u64) {
    (
        ALLOCATION_CALLS.load(Ordering::Relaxed),
        ALLOCATED_BYTES.load(Ordering::Relaxed),
    )
}

#[derive(Clone)]
struct Case {
    name: &'static str,
    equation: &'static str,
    input_shapes: Vec<Vec<usize>>,
    output_shape: Vec<usize>,
    dtype: FloatDType,
    tolerance: f32,
}

fn kernel_with_mode(case: &Case, mode: &str) -> (Box<dyn Kernel>, Duration) {
    struct EnvGuard(Option<OsString>);
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match self.0.take() {
                    Some(value) => std::env::set_var(EINSUM_MODE_ENV, value),
                    None => std::env::remove_var(EINSUM_MODE_ENV),
                }
            }
        }
    }

    let guard = EnvGuard(std::env::var_os(EINSUM_MODE_ENV));
    unsafe { std::env::set_var(EINSUM_MODE_ENV, mode) };
    let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
    node.attributes.insert(
        "equation".into(),
        Attribute::String(case.equation.as_bytes().to_vec()),
    );
    let start = Instant::now();
    let kernel = EinsumFactory
        .create(&node, &case.input_shapes)
        .expect("Einsum kernel construction must succeed");
    let elapsed = start.elapsed();
    drop(guard);
    (kernel, elapsed)
}

fn inputs(case: &Case) -> Vec<Tensor> {
    case.input_shapes
        .iter()
        .enumerate()
        .map(|(input, shape)| {
            let len = shape.iter().product();
            let values = float_values(len)
                .into_iter()
                .enumerate()
                .map(|(index, value)| value + input as f32 * 0.03125 + (index % 7) as f32 * 0.001)
                .collect::<Vec<_>>();
            Tensor::floats(case.dtype, shape, &values)
        })
        .collect()
}

fn validate_case(case: &Case, optimized: &dyn Kernel, reference: &dyn Kernel, inputs: &[Tensor]) {
    let views = inputs.iter().map(Tensor::view).collect::<Vec<_>>();
    let mut fast = Tensor::zeros(case.dtype, &case.output_shape);
    let mut oracle = Tensor::zeros(case.dtype, &case.output_shape);
    optimized
        .execute(&views, &mut [fast.view_mut()])
        .expect("optimized Einsum must execute");
    reference
        .execute(&views, &mut [oracle.view_mut()])
        .expect("reference Einsum must execute");
    let fast = fast.to_f32();
    let oracle = oracle.to_f32();
    assert!(
        oracle.iter().any(|value| *value != 0.0),
        "{} oracle is all-zero; the benchmark correctness check would be vacuous",
        case.name
    );
    assert_close(&fast, &oracle, case.tolerance);
}

fn bench_einsum(c: &mut Criterion) {
    common::init_decode_topology();
    let cases = [
        Case {
            name: "gemm_small",
            equation: "ik,kj->ij",
            input_shapes: vec![vec![4, 16], vec![16, 4]],
            output_shape: vec![4, 4],
            dtype: FloatDType::F32,
            tolerance: 1e-4,
        },
        Case {
            name: "gemm_friendly",
            equation: "ik,kj->ij",
            input_shapes: vec![vec![32, 256], vec![256, 256]],
            output_shape: vec![32, 256],
            dtype: FloatDType::F32,
            tolerance: 2e-3,
        },
        Case {
            name: "gemm_large",
            equation: "ik,kj->ij",
            input_shapes: vec![vec![64, 512], vec![512, 256]],
            output_shape: vec![64, 256],
            dtype: FloatDType::F32,
            tolerance: 5e-3,
        },
        Case {
            name: "transpose_required",
            equation: "ik,jk->ij",
            input_shapes: vec![vec![32, 256], vec![128, 256]],
            output_shape: vec![32, 128],
            dtype: FloatDType::F32,
            tolerance: 2e-3,
        },
        Case {
            name: "broadcast_bmm",
            equation: "...mk,...kn->...mn",
            input_shapes: vec![vec![4, 16, 128], vec![1, 128, 64]],
            output_shape: vec![4, 16, 64],
            dtype: FloatDType::F32,
            tolerance: 2e-3,
        },
        Case {
            name: "gemm_f16",
            equation: "ik,kj->ij",
            input_shapes: vec![vec![32, 256], vec![256, 128]],
            output_shape: vec![32, 128],
            dtype: FloatDType::F16,
            tolerance: 0.25,
        },
        Case {
            name: "gemm_bf16",
            equation: "ik,kj->ij",
            input_shapes: vec![vec![32, 256], vec![256, 128]],
            output_shape: vec![32, 128],
            dtype: FloatDType::Bf16,
            tolerance: 1.0,
        },
        Case {
            name: "reduction",
            equation: "ij->i",
            input_shapes: vec![vec![512, 512]],
            output_shape: vec![512],
            dtype: FloatDType::F32,
            tolerance: 1e-3,
        },
        Case {
            name: "elementwise_product",
            equation: "ij,ij->ij",
            input_shapes: vec![vec![512, 512], vec![512, 512]],
            output_shape: vec![512, 512],
            dtype: FloatDType::F32,
            tolerance: 1e-6,
        },
        Case {
            name: "multi_axis_permute",
            equation: "abxy,xycd->dcab",
            input_shapes: vec![vec![4, 4, 8, 8], vec![8, 8, 4, 4]],
            output_shape: vec![4, 4, 4, 4],
            dtype: FloatDType::F32,
            tolerance: 2e-3,
        },
        Case {
            name: "diagonal",
            equation: "ii->i",
            input_shapes: vec![vec![1024, 1024]],
            output_shape: vec![1024],
            dtype: FloatDType::F32,
            tolerance: 0.0,
        },
    ];

    let mut group = c.benchmark_group("einsum");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for case in cases {
        let tensors = inputs(&case);
        let views = tensors.iter().map(Tensor::view).collect::<Vec<_>>();
        let (optimized, optimized_setup) = kernel_with_mode(&case, "optimized");
        let (reference, reference_setup) = kernel_with_mode(&case, "reference");
        validate_case(&case, &*optimized, &*reference, &tensors);
        let workspace = benchmark_scratch_capacity_bytes(&*optimized).unwrap_or(0);
        let mut allocation_probe_output = Tensor::zeros(case.dtype, &case.output_shape);
        reset_allocations();
        optimized
            .execute(&views, &mut [allocation_probe_output.view_mut()])
            .expect("allocation probe must execute");
        let (allocation_calls, allocated_bytes) = allocations();
        eprintln!(
            "EINSUM_SETUP case={} optimized_us={:.3} reference_us={:.3} \
             reusable_workspace_bytes={workspace} steady_allocations={allocation_calls} \
             steady_allocated_bytes={allocated_bytes}",
            case.name,
            optimized_setup.as_secs_f64() * 1e6,
            reference_setup.as_secs_f64() * 1e6,
        );
        group.throughput(Throughput::Elements(
            case.output_shape.iter().product::<usize>() as u64,
        ));

        let mut optimized_output = Tensor::zeros(case.dtype, &case.output_shape);
        group.bench_with_input(
            BenchmarkId::new(case.name, "optimized"),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    optimized
                        .execute(
                            black_box(&views),
                            black_box(&mut [optimized_output.view_mut()]),
                        )
                        .unwrap()
                });
            },
        );

        let mut reference_output = Tensor::zeros(case.dtype, &case.output_shape);
        group.bench_with_input(
            BenchmarkId::new(case.name, "reference_f64"),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    reference
                        .execute(
                            black_box(&views),
                            black_box(&mut [reference_output.view_mut()]),
                        )
                        .unwrap()
                });
            },
        );
    }
    group.finish();

    let view_case = Case {
        name: "view_permutation",
        equation: "abc->bca",
        input_shapes: vec![vec![32, 64, 128]],
        output_shape: vec![64, 128, 32],
        dtype: FloatDType::F32,
        tolerance: 0.0,
    };
    let view_input = inputs(&view_case);
    let view_input = view_input[0].view();
    let (view_kernel, view_setup) = kernel_with_mode(&view_case, "optimized");
    let specs = view_kernel
        .view_outputs(
            std::slice::from_ref(&view_input),
            std::slice::from_ref(&view_case.output_shape),
            1,
        )
        .expect("view-only Einsum must return a zero-copy output");
    assert_eq!(specs[0].strides, [128, 1, 8192]);
    reset_allocations();
    let allocation_probe = view_kernel
        .view_outputs(
            std::slice::from_ref(&view_input),
            std::slice::from_ref(&view_case.output_shape),
            1,
        )
        .expect("view route must stay reachable");
    black_box(&allocation_probe);
    let (view_allocations, view_allocated_bytes) = allocations();
    eprintln!(
        "EINSUM_SETUP case=view_permutation optimized_us={:.3} reference_us=n/a \
         reusable_workspace_bytes=0 steady_allocations={view_allocations} \
         steady_allocated_bytes={view_allocated_bytes}",
        view_setup.as_secs_f64() * 1e6,
    );
    let mut views = c.benchmark_group("einsum_view");
    views.sample_size(10);
    views.bench_function("permutation_metadata_32x64x128", |bencher| {
        bencher.iter(|| {
            black_box(
                view_kernel
                    .view_outputs(
                        black_box(std::slice::from_ref(&view_input)),
                        black_box(std::slice::from_ref(&view_case.output_shape)),
                        1,
                    )
                    .expect("view route must stay reachable"),
            )
        });
    });
    views.finish();

    // MatMul cannot route through Einsum, so this is the control arm for host
    // stability and the "do not regress mature MatMul" gate.
    let (m, k, n) = (32usize, 256usize, 256usize);
    let a = Tensor::floats(FloatDType::F32, &[m, k], &float_values(m * k));
    let b = Tensor::floats(FloatDType::F32, &[k, n], &float_values(k * n));
    let mut output = Tensor::zeros(FloatDType::F32, &[m, n]);
    let matmul = make_kernel("MatMul", [], &[vec![m, k], vec![k, n]], 24);
    reset_allocations();
    matmul
        .execute(&[a.view(), b.view()], &mut [output.view_mut()])
        .expect("MatMul allocation control must execute");
    let (control_allocations, control_allocated_bytes) = allocations();
    eprintln!(
        "EINSUM_CONTROL_ALLOC steady_allocations={control_allocations} \
         steady_allocated_bytes={control_allocated_bytes}"
    );
    let mut control = c.benchmark_group("einsum_control");
    control.sample_size(10);
    control.warm_up_time(Duration::from_secs(1));
    control.measurement_time(Duration::from_secs(2));
    control.bench_function("matmul_32x256x256", |bencher| {
        let cpu_start = process_cpu_time();
        bencher.iter(|| {
            matmul
                .execute(
                    black_box(&[a.view(), b.view()]),
                    black_box(&mut [output.view_mut()]),
                )
                .unwrap()
        });
        if let (Some(start), Some(end)) = (cpu_start, process_cpu_time()) {
            let cpu = end.since(start);
            eprintln!(
                "EINSUM_CONTROL_CPU user_s={:.6} sys_s={:.6} total_s={:.6}",
                cpu.user_s,
                cpu.sys_s,
                cpu.total_s()
            );
        }
    });
    control.finish();
}

fn main() {
    // Confinement must be established before the first contention snapshot;
    // otherwise the before/after masks differ and hostmon correctly reports the
    // whole sweep as unmeasured.
    common::init_decode_topology();
    let host_lock = common::open_host_lock_window();
    let contention_before = onnx_runtime_hostmon::snapshot();
    let cpu_before = process_cpu_time();
    let wall_start = Instant::now();

    let mut criterion = Criterion::default().configure_from_args();
    bench_einsum(&mut criterion);
    criterion.final_summary();

    let wall_s = wall_start.elapsed().as_secs_f64();
    let cpu = cpu_before
        .zip(process_cpu_time())
        .map(|(before, after)| after.since(before));
    let contention_after = onnx_runtime_hostmon::snapshot();
    let contention =
        onnx_runtime_hostmon::contention(contention_before.as_ref(), contention_after.as_ref());
    match cpu {
        Some(cpu) => println!(
            "EINSUM_SWEEP wall_s={wall_s:.6} cpu_user_s={:.6} cpu_sys_s={:.6} \
             process_efficiency={:.4} foreign_pct={} sibling_peak_pct={}",
            cpu.user_s,
            cpu.sys_s,
            cpu.total_s() / wall_s.max(f64::MIN_POSITIVE),
            if contention.measured {
                format!("{:.1}", contention.foreign_pct)
            } else {
                "n/a".into()
            },
            if contention.siblings_known {
                format!("{:.1}", contention.sibling_peak_pct)
            } else {
                "n/a".into()
            },
        ),
        None => println!(
            "EINSUM_SWEEP wall_s={wall_s:.6} cpu_user_s=n/a cpu_sys_s=n/a \
             process_efficiency=n/a foreign_pct={} sibling_peak_pct={}",
            if contention.measured {
                format!("{:.1}", contention.foreign_pct)
            } else {
                "n/a".into()
            },
            if contention.siblings_known {
                format!("{:.1}", contention.sibling_peak_pct)
            } else {
                "n/a".into()
            },
        ),
    }
    common::report_host_lock(host_lock);
}
