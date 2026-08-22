//! Paired benchmark for the SIMD SDPA path against the scalar reference.
//!
//! `sdpa_f32` is the adapter-facing entry point used by `Attention`,
//! `MultiHeadAttention`, `com.microsoft.Attention` and `VarLenAttention`. Before
//! this bench's accompanying change, x86 always fell through to
//! `sdpa_f32_scalar`; aarch64 took the vectorised body. Both arms are measured
//! here so the ratio is meaningful on whatever host runs it — on a pre-AVX2 x86
//! machine the two arms are the same code and the ratio is ~1.0 by
//! construction.
//!
//! Shapes are real encoder/decoder geometries rather than round numbers:
//! Whisper-base encoder, BERT-base, ViT-L/14 and a 32/8/128 GQA decode step.

mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use onnx_runtime_ep_cpu::kernels::sdpa::{
    NoBias, NoMask, ScaleMode, SdpaConfig, SdpaTensors, sdpa_f32, sdpa_f32_scalar,
};

struct Geometry {
    name: &'static str,
    num_heads: usize,
    num_kv_heads: usize,
    q_seq: usize,
    kv_seq: usize,
    head_size: usize,
}

const GEOMETRIES: [Geometry; 4] = [
    Geometry {
        name: "whisper-base-enc/8x8x64",
        num_heads: 8,
        num_kv_heads: 8,
        q_seq: 1,
        kv_seq: 1500,
        head_size: 64,
    },
    Geometry {
        name: "bert-base/12x12x64",
        num_heads: 12,
        num_kv_heads: 12,
        q_seq: 1,
        kv_seq: 512,
        head_size: 64,
    },
    Geometry {
        name: "vit-l-14/16x16x64",
        num_heads: 16,
        num_kv_heads: 16,
        q_seq: 1,
        kv_seq: 257,
        head_size: 64,
    },
    Geometry {
        name: "gqa-decode/32x8x128",
        num_heads: 32,
        num_kv_heads: 8,
        q_seq: 1,
        kv_seq: 2048,
        head_size: 128,
    },
];

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s >> 30;
            s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            s ^= s >> 27;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn bench_sdpa_simd(c: &mut Criterion) {
    // Match the decode thread topology a served session runs in (#1749).
    // Idempotent, and called from every group member rather than just the
    // first so coverage does not silently ride on `criterion_group!` order.
    common::init_decode_topology();
    let mut group = c.benchmark_group("sdpa_f32");
    for geo in &GEOMETRIES {
        let batch = 1usize;
        let q = values(
            batch * geo.num_heads * geo.q_seq * geo.head_size,
            0x51D9A001,
        );
        let k = values(
            batch * geo.num_kv_heads * geo.kv_seq * geo.head_size,
            0x51D9A002,
        );
        let v = values(
            batch * geo.num_kv_heads * geo.kv_seq * geo.head_size,
            0x51D9A003,
        );
        let tensors = SdpaTensors {
            q: &q,
            k: &k,
            v: &v,
            batch,
            num_heads: geo.num_heads,
            num_kv_heads: geo.num_kv_heads,
            q_seq: geo.q_seq,
            kv_seq: geo.kv_seq,
            head_size: geo.head_size,
            v_head_size: geo.head_size,
        };
        let cfg = SdpaConfig {
            scale: ScaleMode::PostDot(1.0 / (geo.head_size as f32).sqrt()),
            softcap: None,
            causal: false,
            past_seq: 0,
            causal_fill: f32::NEG_INFINITY,
        };
        let mut y = vec![0.0f32; batch * geo.num_heads * geo.q_seq * geo.head_size];

        group.bench_with_input(BenchmarkId::new("dispatch", geo.name), &(), |bencher, _| {
            bencher.iter(|| {
                sdpa_f32(
                    black_box(&tensors),
                    black_box(&cfg),
                    &NoBias,
                    &NoMask,
                    black_box(&mut y),
                    None,
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("scalar", geo.name), &(), |bencher, _| {
            bencher.iter(|| {
                sdpa_f32_scalar(
                    black_box(&tensors),
                    black_box(&cfg),
                    &NoBias,
                    &NoMask,
                    black_box(&mut y),
                    None,
                )
            });
        });
    }
    group.finish();
}

criterion_group!(sdpa_benches, bench_sdpa_simd);
criterion_main!(sdpa_benches);
