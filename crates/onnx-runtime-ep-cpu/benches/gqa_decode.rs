//! M=1 `GroupQueryAttention` decode benchmarks at production model geometries.
//!
//! The decode attention window is a GEMV: roughly two flops per loaded float,
//! so the cost is KV-cache traffic, not arithmetic. Scoring one query head at a
//! time streams the shared KV head once **per query head**, i.e. `group =
//! num_heads / kv_num_heads` times per layer per token. `sdpa_decode_group`
//! streams it once for the whole group instead.
//!
//! Two levels are measured:
//!
//! * `sdpa_group_vs_row` — the SDPA core alone, single-threaded, so the
//!   KV-traffic effect is visible without any scheduling noise. `row` calls
//!   `sdpa_decode_row` once per query head (the pre-fusion schedule); `group`
//!   makes the single fused call. Both produce bit-identical output.
//! * `gqa_kernel_decode` — the whole `com.microsoft::GroupQueryAttention`
//!   kernel through the EP registry on a fixed-width pool, so the model-visible
//!   op time (KV append and output transpose included) is covered too. The
//!   fusion is gated on the decode pool staying saturated *and* on the attended
//!   KV exceeding last-level cache, so only the wide-KV geometries at the long
//!   context lengths take it; the rest measure the unchanged per-head path.
//!   Set `ONNX_GENAI_GQA_GROUP_FUSED=0` to force the per-head path for an A/B
//!   (the flag latches once per process, so run one process per setting).
//!
//! Geometries are real decoder configurations, named for their source model, so
//! the numbers map onto shipped shapes rather than a favourable synthetic.

mod common;

use common::{FloatDType, Tensor};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
use onnx_runtime_ep_api::{ExecutionProvider, Kernel};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cpu::kernels::sdpa::{SoftmaxExp, sdpa_decode_group, sdpa_decode_row};
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Node, NodeId, compute_contiguous_strides};
use rayon::ThreadPoolBuilder;

/// `(label, num_heads, kv_num_heads, head_size)` for shipped decoder configs.
/// `group = num_heads / kv_num_heads` is the KV re-read factor the fusion
/// removes, so the set spans MHA (1), a narrow group (2) and wide groups (4, 7).
const GEOMETRIES: [(&str, usize, usize, usize); 4] = [
    ("qwen2.5-0.5b", 14, 2, 64),
    ("qwen3-0.6b", 16, 8, 128),
    ("llama-3.1-8b", 32, 8, 128),
    ("qwen2.5-7b", 28, 4, 128),
];

/// Past-KV lengths spanning short chat context through long context. The
/// fusion gate opens at 8 MiB of attended K+V per KV head, which at
/// `head_size = 128` is 8192 tokens — so the sweep brackets the threshold.
const KV_LENGTHS: [usize; 4] = [2_048, 4_096, 8_192, 16_384];

fn pseudo_random(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i.wrapping_mul(2_654_435_761).wrapping_add(seed * 40_503)) % 65_536;
            (x as f32 / 65_536.0) - 0.5
        })
        .collect()
}

fn bind_decode_pool_width(threads: usize) {
    // The GQA decode path fans work out on the ambient Rayon pool, so pin the
    // global pool to the bounded decode width (`MAX_TOPOLOGY_DECODE_THREADS`)
    // instead of letting Criterion inherit all host threads. `build_global` is
    // one-shot; a second call is an error we can ignore.
    let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();
}

/// SDPA core only: `group` successive `sdpa_decode_row` calls versus one
/// `sdpa_decode_group` call over the same KV window. Single-threaded on purpose
/// — this isolates the memory-traffic difference from the decode scheduler.
fn bench_sdpa_group_vs_row(c: &mut Criterion) {
    let mut group_bench = c.benchmark_group("sdpa_group_vs_row");
    for (label, num_heads, kv_num_heads, head_size) in GEOMETRIES {
        let group = num_heads / kv_num_heads;
        if group == 1 {
            continue;
        }
        for kv_seq in KV_LENGTHS {
            let q = pseudo_random(group * head_size, 1);
            let k = pseudo_random(kv_seq * head_size, 2);
            let v = pseudo_random(kv_seq * head_size, 3);
            let scale = 1.0 / (head_size as f32).sqrt();
            let mut out = vec![0.0f32; group * head_size];
            let mut scores = Vec::new();

            // Bytes of KV actually streamed by the fused path (K and V once).
            group_bench.throughput(Throughput::Bytes(
                (2 * kv_seq * head_size * std::mem::size_of::<f32>()) as u64,
            ));
            group_bench.bench_function(
                BenchmarkId::new(format!("{label}/group={group}/row"), kv_seq),
                |bencher| {
                    bencher.iter(|| {
                        for g in 0..group {
                            sdpa_decode_row(
                                black_box(&q[g * head_size..(g + 1) * head_size]),
                                black_box(&k),
                                black_box(&v),
                                kv_seq,
                                0,
                                kv_seq,
                                scale,
                                None,
                                SoftmaxExp::F64Intermediate,
                                &mut out[g * head_size..(g + 1) * head_size],
                            );
                        }
                    });
                },
            );
            group_bench.bench_function(
                BenchmarkId::new(format!("{label}/group={group}/group"), kv_seq),
                |bencher| {
                    bencher.iter(|| {
                        sdpa_decode_group(
                            black_box(&q),
                            black_box(&k),
                            black_box(&v),
                            kv_seq,
                            group,
                            head_size,
                            head_size,
                            0,
                            kv_seq,
                            scale,
                            None,
                            SoftmaxExp::F64Intermediate,
                            &mut out,
                            &mut scores,
                        );
                    });
                },
            );
        }
    }
    group_bench.finish();
}

/// Build one decode-step `GroupQueryAttention` call: `q_seq = 1`, a past cache
/// of `past_len` tokens, and the `present` buffers sized for `past_len + 1`.
/// Returns the input tensors, their shapes, and the output tensors.
#[allow(clippy::type_complexity)]
fn decode_inputs(
    num_heads: usize,
    kv_num_heads: usize,
    head_size: usize,
    past_len: usize,
) -> (Vec<Tensor>, Vec<Vec<usize>>, Vec<Tensor>) {
    let total = past_len + 1;
    let shapes = vec![
        vec![1, 1, num_heads * head_size],
        vec![1, 1, kv_num_heads * head_size],
        vec![1, 1, kv_num_heads * head_size],
        vec![1, kv_num_heads, past_len, head_size],
        vec![1, kv_num_heads, past_len, head_size],
        vec![1],
        vec![1],
    ];
    let past_len_elems = kv_num_heads * past_len * head_size;
    let inputs = vec![
        Tensor::floats(
            FloatDType::F32,
            &shapes[0],
            &pseudo_random(num_heads * head_size, 11),
        ),
        Tensor::floats(
            FloatDType::F32,
            &shapes[1],
            &pseudo_random(kv_num_heads * head_size, 12),
        ),
        Tensor::floats(
            FloatDType::F32,
            &shapes[2],
            &pseudo_random(kv_num_heads * head_size, 13),
        ),
        Tensor::floats(
            FloatDType::F32,
            &shapes[3],
            &pseudo_random(past_len_elems, 14),
        ),
        Tensor::floats(
            FloatDType::F32,
            &shapes[4],
            &pseudo_random(past_len_elems, 15),
        ),
        Tensor::i32(&shapes[5], &[past_len as i32]),
        Tensor::i32(&shapes[6], &[total as i32]),
    ];
    let outputs = vec![
        Tensor::zeros(FloatDType::F32, &[1, 1, num_heads * head_size]),
        Tensor::zeros(FloatDType::F32, &[1, kv_num_heads, total, head_size]),
        Tensor::zeros(FloatDType::F32, &[1, kv_num_heads, total, head_size]),
    ];
    (inputs, shapes, outputs)
}

fn gqa_kernel(
    num_heads: usize,
    kv_num_heads: usize,
    input_shapes: &[Vec<usize>],
) -> Box<dyn Kernel> {
    let mut node = Node::new(NodeId(0), "GroupQueryAttention", vec![], vec![]);
    node.domain = "com.microsoft".into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(num_heads as i64));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(kv_num_heads as i64));
    CpuExecutionProvider::new()
        .get_kernel(&node, input_shapes, 1)
        .expect("CPU EP must register com.microsoft::GroupQueryAttention")
}

/// Whole-kernel decode through the EP registry. `ONNX_GENAI_GQA_GROUP_FUSED`
/// latches on first read, so a single process measures a single setting: run
/// the bench twice (once with the variable set to `0`, once to `1`) to get the
/// A/B. The current setting is reported in the benchmark id.
fn bench_gqa_kernel_decode(c: &mut Criterion) {
    let fused = std::env::var("ONNX_GENAI_GQA_GROUP_FUSED").unwrap_or_else(|_| "default".into());
    let mut group_bench = c.benchmark_group("gqa_kernel_decode");
    // 8 workers is the bounded decode-pool width the runtime uses for M=1.
    bind_decode_pool_width(8);
    for (label, num_heads, kv_num_heads, head_size) in GEOMETRIES {
        let group = num_heads / kv_num_heads;
        for kv_seq in KV_LENGTHS {
            let past_len = kv_seq - 1;
            let (inputs, shapes, mut outputs) =
                decode_inputs(num_heads, kv_num_heads, head_size, past_len);
            let kernel = gqa_kernel(num_heads, kv_num_heads, &shapes);
            group_bench.throughput(Throughput::Bytes(
                (2 * kv_num_heads * kv_seq * head_size * std::mem::size_of::<f32>()) as u64,
            ));
            let name = format!("{label}/group={group}/fused={fused}");
            group_bench.bench_function(BenchmarkId::new(name, kv_seq), |bencher| {
                bencher.iter(|| {
                    let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
                    let mut muts: Vec<_> = outputs.iter_mut().map(Tensor::view_mut).collect();
                    kernel
                        .execute(black_box(&views), black_box(&mut muts))
                        .expect("GQA decode must succeed");
                });
            });
        }
    }
    group_bench.finish();
}

/// Half-precision KV append A/B: aliased `present == past` versus the
/// distinct-buffer copy path, at the same shipped geometries.
///
/// This is the measurement behind `detect_inplace_kv_half`. Both arms run the
/// identical kernel on the identical geometry and produce identical numbers;
/// the only difference is whether the caller bound `present` onto `past`.
///
/// * `copy` — distinct past and present buffers. `fill_present` widens the whole
///   `f16`/`bf16` history into f32 scratch, then the output writer narrows the
///   whole f32 scratch back out. Two full passes over the cache per token.
/// * `append` — `present` aliases `past` at physical capacity. Only the new
///   token's rows are narrowed in; the history is already resident. The
///   full-history narrow disappears, leaving one widen for the attention core.
///
/// The gap is pure memory traffic and therefore grows with context length, which
/// is why the sweep runs to 16 K tokens rather than stopping at a chat-sized
/// window. Throughput is reported over the K+V bytes the attention core must
/// read either way, so the ratio between arms reads directly as the fraction of
/// cache traffic the append removes.
fn bench_gqa_half_kv_append(c: &mut Criterion) {
    let mut group_bench = c.benchmark_group("gqa_half_kv_append");
    bind_decode_pool_width(8);
    for (label, num_heads, kv_num_heads, head_size) in GEOMETRIES {
        for kv_seq in KV_LENGTHS {
            let past_len = kv_seq - 1;
            let total = kv_seq;
            let cpu = DeviceId::cpu();
            let dt = DataType::Float16;

            let q = Tensor::floats(
                FloatDType::F32,
                &[1, 1, num_heads * head_size],
                &pseudo_random(num_heads * head_size, 11),
            );
            let cur_k = Tensor::floats(
                FloatDType::F32,
                &[1, 1, kv_num_heads * head_size],
                &pseudo_random(kv_num_heads * head_size, 12),
            );
            let cur_v = Tensor::floats(
                FloatDType::F32,
                &[1, 1, kv_num_heads * head_size],
                &pseudo_random(kv_num_heads * head_size, 13),
            );
            let seqlens = Tensor::i32(&[1], &[past_len as i32]);
            let tsl = Tensor::i32(&[1], &[total as i32]);
            let mut out = Tensor::zeros(FloatDType::F32, &[1, 1, num_heads * head_size]);

            // --- copy arm: distinct past (past_len) and present (total). ---
            let past_shape = vec![1, kv_num_heads, past_len, head_size];
            let present_shape = vec![1, kv_num_heads, total, head_size];
            let past_k = Tensor::floats(
                FloatDType::F16,
                &past_shape,
                &pseudo_random(kv_num_heads * past_len * head_size, 14),
            );
            let past_v = Tensor::floats(
                FloatDType::F16,
                &past_shape,
                &pseudo_random(kv_num_heads * past_len * head_size, 15),
            );
            let mut present_k = Tensor::zeros(FloatDType::F16, &present_shape);
            let mut present_v = Tensor::zeros(FloatDType::F16, &present_shape);
            let shapes: Vec<Vec<usize>> = vec![
                vec![1, 1, num_heads * head_size],
                vec![1, 1, kv_num_heads * head_size],
                vec![1, 1, kv_num_heads * head_size],
                past_shape.clone(),
                past_shape.clone(),
                vec![1],
                vec![1],
            ];
            let kernel = gqa_kernel(num_heads, kv_num_heads, &shapes);
            group_bench.throughput(Throughput::Bytes(
                (2 * kv_num_heads * kv_seq * head_size * std::mem::size_of::<u16>()) as u64,
            ));
            group_bench.bench_function(
                BenchmarkId::new(format!("{label}/f16/copy"), kv_seq),
                |bencher| {
                    bencher.iter(|| {
                        let views = vec![
                            q.view(),
                            cur_k.view(),
                            cur_v.view(),
                            past_k.view(),
                            past_v.view(),
                            seqlens.view(),
                            tsl.view(),
                        ];
                        let mut muts =
                            vec![out.view_mut(), present_k.view_mut(), present_v.view_mut()];
                        kernel
                            .execute(black_box(&views), black_box(&mut muts))
                            .expect("GQA half copy path must succeed");
                    });
                },
            );

            // --- append arm: one capacity buffer bound as both past and present.
            // Raw pointers mirror the executor's device-binding wiring, which is
            // the only way to express `present == past` through the kernel ABI.
            let cap_elems = kv_num_heads * total * head_size;
            let mut cap_k: Vec<u16> = pseudo_random(cap_elems, 14)
                .iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect();
            let mut cap_v: Vec<u16> = pseudo_random(cap_elems, 15)
                .iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect();
            let cap_shape = vec![1, kv_num_heads, total, head_size];
            let cap_strides = compute_contiguous_strides(&cap_shape);
            let kp = cap_k.as_mut_ptr().cast::<std::ffi::c_void>();
            let vp = cap_v.as_mut_ptr().cast::<std::ffi::c_void>();
            let cap_shapes: Vec<Vec<usize>> = vec![
                vec![1, 1, num_heads * head_size],
                vec![1, 1, kv_num_heads * head_size],
                vec![1, 1, kv_num_heads * head_size],
                cap_shape.clone(),
                cap_shape.clone(),
                vec![1],
                vec![1],
            ];
            let cap_kernel = gqa_kernel(num_heads, kv_num_heads, &cap_shapes);
            group_bench.bench_function(
                BenchmarkId::new(format!("{label}/f16/append"), kv_seq),
                |bencher| {
                    bencher.iter(|| {
                        let views = vec![
                            q.view(),
                            cur_k.view(),
                            cur_v.view(),
                            TensorView::new(DevicePtr(kp), dt, &cap_shape, &cap_strides, cpu),
                            TensorView::new(DevicePtr(vp), dt, &cap_shape, &cap_strides, cpu),
                            seqlens.view(),
                            tsl.view(),
                        ];
                        let mut muts = vec![
                            out.view_mut(),
                            TensorMut::new(DevicePtrMut(kp), dt, &cap_shape, &cap_strides, cpu),
                            TensorMut::new(DevicePtrMut(vp), dt, &cap_shape, &cap_strides, cpu),
                        ];
                        cap_kernel
                            .execute(black_box(&views), black_box(&mut muts))
                            .expect("GQA half append path must succeed");
                    });
                },
            );
        }
    }
    group_bench.finish();
}

criterion_group!(
    benches,
    bench_sdpa_group_vs_row,
    bench_gqa_kernel_decode,
    bench_gqa_half_kv_append
);
criterion_main!(benches);
