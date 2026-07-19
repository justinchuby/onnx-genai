mod common;

use common::{FloatDType, Tensor, float_values, make_kernel};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use onnx_runtime_ir::Attribute;
use rayon::{ThreadPool, ThreadPoolBuilder};

const FLOAT_DTYPES: [FloatDType; 3] = [FloatDType::F32, FloatDType::F16, FloatDType::Bf16];
const MATCHED_THREAD_COUNTS: [usize; 2] = [1, 8];

fn thread_pool(threads: usize) -> ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("benchmark Rayon pool must build")
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    for (size, shape) in [
        ("small", vec![1_024]),
        ("medium", vec![256, 1_024]),
        ("large", vec![1_024, 4_096]),
    ] {
        let len = shape.iter().product();
        let width = shape[shape.len() - 1];
        group.throughput(Throughput::Elements(len as u64));
        for dtype in FLOAT_DTYPES {
            let a = Tensor::floats(dtype, &shape, &float_values(len));
            let b = Tensor::floats(dtype, &[width], &float_values(width));
            let mut output = Tensor::zeros(dtype, &shape);
            let kernel = make_kernel("Add", [], &[shape.clone(), vec![width]], 13);
            group.bench_with_input(
                BenchmarkId::new(format!("{size}/{}/threads=1-internal", dtype.name()), len),
                &(),
                |bencher, _| {
                    bencher.iter(|| {
                        kernel
                            .execute(
                                black_box(&[a.view(), b.view()]),
                                black_box(&mut [output.view_mut()]),
                            )
                            .unwrap()
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_reduce_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_mean");
    for (size, shape) in [
        ("small", vec![32, 128]),
        ("medium", vec![128, 512]),
        ("large", vec![256, 1_024]),
    ] {
        let len = shape.iter().product();
        group.throughput(Throughput::Elements(len as u64));
        let input = Tensor::floats(FloatDType::F32, &shape, &float_values(len));
        let mut output = Tensor::zeros(FloatDType::F32, &[shape[0], 1]);
        let kernel = make_kernel(
            "ReduceMean",
            [
                ("axes", Attribute::Ints(vec![1])),
                ("keepdims", Attribute::Int(1)),
            ],
            &[shape.clone()],
            13,
        );
        group.bench_function(
            BenchmarkId::new(format!("{size}/f32/threads=1-internal"), len),
            |bencher| {
                bencher.iter(|| {
                    kernel
                        .execute(
                            black_box(&[input.view()]),
                            black_box(&mut [output.view_mut()]),
                        )
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_gather(c: &mut Criterion) {
    let mut group = c.benchmark_group("gather");
    for (size, rows, columns, index_count) in [
        ("small", 4_096, 128, 32),
        ("medium", 16_384, 256, 128),
        ("large", 32_768, 512, 256),
    ] {
        let shape = vec![rows, columns];
        let indices_values = (0..index_count)
            .map(|i| ((i * 97) % rows) as i64)
            .collect::<Vec<_>>();
        let indices = Tensor::i64(&[index_count], &indices_values);
        group.throughput(Throughput::Elements((index_count * columns) as u64));
        for dtype in FLOAT_DTYPES {
            let data = Tensor::floats(dtype, &shape, &float_values(rows * columns));
            let mut output = Tensor::zeros(dtype, &[index_count, columns]);
            let kernel = make_kernel(
                "Gather",
                [("axis", Attribute::Int(0))],
                &[shape.clone(), vec![index_count]],
                13,
            );
            group.bench_function(
                BenchmarkId::new(
                    format!("{size}/{}/threads=1-internal", dtype.name()),
                    index_count * columns,
                ),
                |bencher| {
                    bencher.iter(|| {
                        kernel
                            .execute(
                                black_box(&[data.view(), indices.view()]),
                                black_box(&mut [output.view_mut()]),
                            )
                            .unwrap()
                    });
                },
            );
        }
    }
    group.finish();
}

/// Benchmark one explicitly requested GEMM backend without leaving global
/// process environment state behind. Criterion invokes these iterations serially.
fn with_gemm_backend<T>(backend: &str, f: impl FnOnce() -> T) -> T {
    struct EnvGuard(Option<std::ffi::OsString>);

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match self.0.take() {
                    Some(value) => std::env::set_var("NXRT_CPU_GEMM_BACKEND", value),
                    None => std::env::remove_var("NXRT_CPU_GEMM_BACKEND"),
                }
            }
        }
    }

    let guard = EnvGuard(std::env::var_os("NXRT_CPU_GEMM_BACKEND"));
    unsafe { std::env::set_var("NXRT_CPU_GEMM_BACKEND", backend) };
    let result = f();
    drop(guard);
    result
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    let mut backends = vec!["generic"];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    backends.push("simd");
    #[cfg(feature = "mlas")]
    backends.push("mlas");

    for (size, m, k, n) in [
        ("small", 1, 256, 256),
        ("medium", 32, 512, 512),
        ("large", 32, 1_024, 1_024),
    ] {
        group.throughput(Throughput::Elements((m * n) as u64));
        for backend in &backends {
            for dtype in FLOAT_DTYPES {
                for threads in MATCHED_THREAD_COUNTS {
                    let pool = thread_pool(threads);
                    let a = Tensor::floats(dtype, &[m, k], &float_values(m * k));
                    let b = Tensor::floats(dtype, &[k, n], &float_values(k * n));
                    let mut output = Tensor::zeros(dtype, &[m, n]);
                    let mut kernel = make_kernel("MatMul", [], &[vec![m, k], vec![k, n]], 13);
                    group.bench_function(
                        BenchmarkId::new(
                            format!("{size}/{backend}/{}/threads={threads}", dtype.name()),
                            format!("{m}x{k}x{n}"),
                        ),
                        |bencher| {
                            bencher.iter(|| {
                                let a = &a;
                                let b = &b;
                                let output = &mut output;
                                let kernel = &mut kernel;
                                pool.install(move || {
                                    with_gemm_backend(backend, || {
                                        kernel
                                            .execute(
                                                black_box(&[a.view(), b.view()]),
                                                black_box(&mut [output.view_mut()]),
                                            )
                                            .unwrap()
                                    })
                                })
                            });
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

criterion_group!(
    kernel_benches,
    bench_add,
    bench_reduce_mean,
    bench_gather,
    bench_matmul
);
criterion_main!(kernel_benches);
