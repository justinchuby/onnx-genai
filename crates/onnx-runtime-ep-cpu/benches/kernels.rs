mod common;

use common::{FloatDType, Tensor, float_values, make_kernel};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use onnx_runtime_ep_api::{ExecutionProvider, Kernel};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cpu::kernels::block_dequant::{decode_e2m1, decode_e8m0_scale};
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, ValueId};
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
    // Match the decode thread topology a served session runs in (#1749).
    // Idempotent, and called from every group member rather than just the
    // first so coverage does not silently ride on `criterion_group!` order.
    common::init_decode_topology();
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
    // Match the decode thread topology a served session runs in (#1749).
    // Idempotent, and called from every group member rather than just the
    // first so coverage does not silently ride on `criterion_group!` order.
    common::init_decode_topology();
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
            std::slice::from_ref(&shape),
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
    // Match the decode thread topology a served session runs in (#1749).
    // Idempotent, and called from every group member rather than just the
    // first so coverage does not silently ride on `criterion_group!` order.
    common::init_decode_topology();
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
    // Match the decode thread topology a served session runs in (#1749).
    // Idempotent, and called from every group member rather than just the
    // first so coverage does not silently ride on `criterion_group!` order.
    common::init_decode_topology();
    let mut group = c.benchmark_group("matmul");
    let backends = &[
        "generic",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "simd",
        #[cfg(feature = "mlas")]
        "mlas",
    ];

    for (size, m, k, n) in [
        ("small", 1, 256, 256),
        ("medium", 32, 512, 512),
        ("large", 32, 1_024, 1_024),
    ] {
        group.throughput(Throughput::Elements((m * n) as u64));
        for backend in backends {
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

fn block_quantized_matmul_kernel(k: usize, n: usize) -> Box<dyn Kernel> {
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMatMul",
        vec![Some(ValueId(0)), Some(ValueId(1)), None, None],
        vec![],
    );
    node.domain = "pkg.nxrt".into();
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes
        .insert("format".into(), Attribute::String(b"mxfp4".to_vec()));
    node.attributes
        .insert("block_layout_version".into(), Attribute::Int(1));
    CpuExecutionProvider::new()
        .get_kernel(&node, &[], 1)
        .expect("CPU EP must register BlockQuantizedMatMul")
}

fn packed_mxfp4(n: usize, k: usize) -> Vec<u8> {
    const MXFP4_BLOCK_BYTES: usize = 17;
    let blocks = k.div_ceil(32);
    let mut packed = vec![0u8; n * blocks * MXFP4_BLOCK_BYTES];
    for output in 0..n {
        for block in 0..blocks {
            let start = (output * blocks + block) * MXFP4_BLOCK_BYTES;
            packed[start] = 127;
            for byte in 0..16 {
                let low = ((output + block + byte) & 0x0f) as u8;
                let high = ((output * 3 + block + byte) & 0x0f) as u8;
                packed[start + 1 + byte] = low | (high << 4);
            }
        }
    }
    packed
}

fn packed_mxfp4_experts(experts: usize, out_features: usize, in_features: usize) -> Vec<u8> {
    packed_mxfp4(experts * out_features, in_features)
}

fn dequantize_mxfp4_kn(n: usize, k: usize, packed: &[u8]) -> Vec<f32> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 17;
    let blocks = k.div_ceil(QK);
    assert_eq!(packed.len(), n * blocks * BLOCK_BYTES);
    let mut dense = vec![0.0f32; k * n];
    for output in 0..n {
        for block in 0..blocks {
            let packed_block = &packed[(output * blocks + block) * BLOCK_BYTES..][..BLOCK_BYTES];
            let scale = decode_e8m0_scale(packed_block[0]);
            for offset in 0..QK.min(k - block * QK) {
                let byte = packed_block[1 + offset % 16];
                let code = if offset < 16 { byte } else { byte >> 4 };
                dense[(block * QK + offset) * n + output] = decode_e2m1(code) * scale;
            }
        }
    }
    dense
}

fn bench_block_quantized_matmul_cache(c: &mut Criterion) {
    // Match the decode thread topology a served session runs in (#1749).
    // Idempotent, and called from every group member rather than just the
    // first so coverage does not silently ride on `criterion_group!` order.
    common::init_decode_topology();
    let mut group = c.benchmark_group("block_quantized_matmul_cached_dense");
    group.sample_size(15);
    let (m, k, n) = (1usize, 1_024usize, 1_024usize);
    let blocks = k.div_ceil(32);
    let packed = packed_mxfp4(n, k);
    let a = Tensor::floats(FloatDType::F32, &[m, k], &float_values(m * k));
    let b = Tensor::u8(&[n, blocks, 17], &packed);
    let absent_scale = onnx_runtime_ep_api::TensorView::absent(DataType::Float8E8M0);
    let absent_bias = onnx_runtime_ep_api::TensorView::absent(DataType::Float32);
    group.throughput(Throughput::Elements((m * n) as u64));

    // Forced-cold payload path: setup is outside Criterion, but every measured
    // iteration treats packed_B as nonconstant and re-expands it.
    let mut uncached_output = Tensor::zeros(FloatDType::F32, &[m, n]);
    let uncached_kernel = block_quantized_matmul_kernel(k, n);
    group.bench_function(
        BenchmarkId::new("mxfp4_uncached_dequant_each_call", format!("{m}x{k}x{n}")),
        |bencher| {
            bencher.iter(|| {
                uncached_kernel
                    .execute(
                        black_box(&[a.view(), b.view(), absent_scale, absent_bias]),
                        black_box(&mut [uncached_output.view_mut()]),
                    )
                    .unwrap()
            });
        },
    );

    // Proxy for the pre-patch OnceLock steady state: the packed weight is
    // expanded before timing, the dense MatMul kernel is prewarmed, and measured
    // iterations perform no quantized cache lookup/hash/copy/dequantization.
    let dense_b_values = dequantize_mxfp4_kn(n, k, &packed);
    let dense_b = Tensor::floats(FloatDType::F32, &[k, n], &dense_b_values);
    let mut once_like_output = Tensor::zeros(FloatDType::F32, &[m, n]);
    let mut once_like_kernel = make_kernel("MatMul", [], &[vec![m, k], vec![k, n]], 13);
    once_like_kernel.set_constant_inputs(&[false, true]);
    once_like_kernel
        .execute(
            &[a.view(), dense_b.view()],
            &mut [once_like_output.view_mut()],
        )
        .expect("prewarm pre-expanded dense MatMul baseline");
    group.bench_function(
        BenchmarkId::new(
            "mxfp4_preexpanded_dense_oncelock_like_proxy",
            format!("{m}x{k}x{n}"),
        ),
        |bencher| {
            bencher.iter(|| {
                once_like_kernel
                    .execute(
                        black_box(&[a.view(), dense_b.view()]),
                        black_box(&mut [once_like_output.view_mut()]),
                    )
                    .unwrap()
            });
        },
    );

    let mut cached_output = Tensor::zeros(FloatDType::F32, &[m, n]);
    let mut cached_kernel = block_quantized_matmul_kernel(k, n);
    cached_kernel.set_constant_inputs(&[false, true, false, false]);
    // Warm boundary: the first hash/dequant/cache insertion completes before
    // Criterion starts; every measured iteration is a stable-identity LRU hit.
    cached_kernel
        .execute(
            &[a.view(), b.view(), absent_scale, absent_bias],
            &mut [cached_output.view_mut()],
        )
        .expect("prewarm cached dense weight");
    group.bench_function(
        BenchmarkId::new("mxfp4_cached_dense_repeated_call", format!("{m}x{k}x{n}")),
        |bencher| {
            bencher.iter(|| {
                cached_kernel
                    .execute(
                        black_box(&[a.view(), b.view(), absent_scale, absent_bias]),
                        black_box(&mut [cached_output.view_mut()]),
                    )
                    .unwrap()
            });
        },
    );
    group.finish();
}

fn block_quantized_moe_kernel(top_k: usize) -> Box<dyn Kernel> {
    let mut inputs = vec![None; 12];
    for index in [0usize, 1, 2, 4] {
        inputs[index] = Some(ValueId(index as u32));
    }
    let mut node = Node::new(NodeId(0), "BlockQuantizedMoE", inputs, vec![]);
    node.domain = "pkg.nxrt".into();
    node.attributes
        .insert("k".into(), Attribute::Int(top_k as i64));
    node.attributes.insert(
        "activation_type".into(),
        Attribute::String(b"identity".to_vec()),
    );
    node.attributes
        .insert("fc1_format".into(), Attribute::String(b"mxfp4".to_vec()));
    node.attributes
        .insert("fc2_format".into(), Attribute::String(b"mxfp4".to_vec()));
    node.attributes
        .insert("block_layout_version".into(), Attribute::Int(1));
    CpuExecutionProvider::new()
        .get_kernel(&node, &[], 1)
        .expect("CPU EP must register BlockQuantizedMoE")
}

fn bench_block_quantized_moe_cache(c: &mut Criterion) {
    // Match the decode thread topology a served session runs in (#1749).
    // Idempotent, and called from every group member rather than just the
    // first so coverage does not silently ride on `criterion_group!` order.
    common::init_decode_topology();
    let mut group = c.benchmark_group("block_quantized_moe_cached_dense");
    group.sample_size(15);
    let (rows, hidden, inter, experts, top_k) = (1usize, 256usize, 256usize, 4usize, 1usize);
    let hidden_blocks = hidden.div_ceil(32);
    let inter_blocks = inter.div_ceil(32);
    let input = Tensor::floats(
        FloatDType::F32,
        &[rows, hidden],
        &float_values(rows * hidden),
    );
    let logits = Tensor::floats(FloatDType::F32, &[rows, experts], &[4.0, -1.0, -2.0, -3.0]);
    let fc1_values = packed_mxfp4_experts(experts, inter, hidden);
    let fc2_values = packed_mxfp4_experts(experts, hidden, inter);
    let fc1 = Tensor::u8(&[experts, inter, hidden_blocks, 17], &fc1_values);
    let fc2 = Tensor::u8(&[experts, hidden, inter_blocks, 17], &fc2_values);
    let absent_f32 = onnx_runtime_ep_api::TensorView::absent(DataType::Float32);
    let absent_u8 = onnx_runtime_ep_api::TensorView::absent(DataType::Uint8);
    let absent_scale = onnx_runtime_ep_api::TensorView::absent(DataType::Float8E8M0);
    group.throughput(Throughput::Elements((rows * hidden) as u64));

    // Every measured iteration re-expands the routed expert projections.
    let mut uncached_output = Tensor::zeros(FloatDType::F32, &[rows, hidden]);
    let uncached_kernel = block_quantized_moe_kernel(top_k);
    group.bench_function(
        BenchmarkId::new(
            "mxfp4_uncached_expert_dequant_each_call",
            format!("rows={rows},H={hidden},I={inter},E={experts},top_k={top_k}"),
        ),
        |bencher| {
            bencher.iter(|| {
                uncached_kernel
                    .execute(
                        black_box(&[
                            input.view(),
                            logits.view(),
                            fc1.view(),
                            absent_f32,
                            fc2.view(),
                            absent_f32,
                            absent_u8,
                            absent_f32,
                            absent_f32,
                            absent_scale,
                            absent_scale,
                            absent_scale,
                        ]),
                        black_box(&mut [uncached_output.view_mut()]),
                    )
                    .unwrap()
            });
        },
    );

    let mut cached_output = Tensor::zeros(FloatDType::F32, &[rows, hidden]);
    let mut cached_kernel = block_quantized_moe_kernel(top_k);
    cached_kernel.set_constant_inputs(&[
        false, false, true, false, true, false, false, false, false, false, false, false,
    ]);
    // Prewarm the routed expert before Criterion; measured calls are cache hits.
    cached_kernel
        .execute(
            &[
                input.view(),
                logits.view(),
                fc1.view(),
                absent_f32,
                fc2.view(),
                absent_f32,
                absent_u8,
                absent_f32,
                absent_f32,
                absent_scale,
                absent_scale,
                absent_scale,
            ],
            &mut [cached_output.view_mut()],
        )
        .expect("prewarm cached expert weights");
    group.bench_function(
        BenchmarkId::new(
            "mxfp4_cached_dense_expert_repeated_call",
            format!("rows={rows},H={hidden},I={inter},E={experts},top_k={top_k}"),
        ),
        |bencher| {
            bencher.iter(|| {
                cached_kernel
                    .execute(
                        black_box(&[
                            input.view(),
                            logits.view(),
                            fc1.view(),
                            absent_f32,
                            fc2.view(),
                            absent_f32,
                            absent_u8,
                            absent_f32,
                            absent_f32,
                            absent_scale,
                            absent_scale,
                            absent_scale,
                        ]),
                        black_box(&mut [cached_output.view_mut()]),
                    )
                    .unwrap()
            });
        },
    );
    group.finish();
}

criterion_group!(
    kernel_benches,
    bench_add,
    bench_reduce_mean,
    bench_gather,
    bench_matmul,
    bench_block_quantized_matmul_cache,
    bench_block_quantized_moe_cache
);
criterion_main!(kernel_benches);
