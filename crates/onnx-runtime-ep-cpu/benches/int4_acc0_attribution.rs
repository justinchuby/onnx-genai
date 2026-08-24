//! Attribution bench for the f32-expanded int4 decode GEMV (`gemv_nk`).
//!
//! # What this was built to test, and what it actually found
//!
//! It was built to test the hypothesis that `gemv_nk` is slow because the
//! weight it reads is an expanded f32 `[N, K]` cache -- 4 bytes per weight
//! where the packed form is 0.5, so 8x the memory traffic in a regime (m=1)
//! where every weight is read once for a single FMA.
//!
//! **That hypothesis was wrong**, and this bench is what disproved it:
//!
//! - `nk_serial`, the shape the kernel had, achieves only **3.8-4.0 GB/s**
//!   against a measured ~31-36 GB/s per-CCX ceiling. It was never anywhere
//!   near bandwidth-bound.
//! - Breaking the reduction chain alone -- identical memory traffic, identical
//!   weight layout -- is worth **5.3x-9.9x**, and lifts the same loop to
//!   20-39 GB/s, which *is* at the roofline.
//!
//! So the cost was the dependency chain, not the traffic. `f32` addition is not
//! associative, so `.map(|(&a, &b)| a * b).sum()` forces one serial
//! accumulator; the loop cannot vectorize and issues one FMA per FMA *latency*.
//!
//! The two `packed_*` arms are the control that keeps the conclusion honest,
//! and they are a **negative result**: dequantizing packed nibbles on the fly
//! is 0.08x-0.15x, i.e. 6-12x *slower*, because a scalar nibble unpack costs
//! far more than the traffic it saves. They are deliberately naive scalar
//! reference implementations, so they bound the value of the idea in this form
//! -- they do not prove a SIMD unpack could not win, only that the traffic
//! argument alone does not carry it.
//!
//! # Reading the arms
//!
//! - `nk_serial`    -- f32 weight, serial `.sum()`. The shape `gemv_nk` had.
//! - `nk_acc16`     -- f32 weight, 16 accumulators. Isolates the chain cost.
//! - `packed_serial`-- packed nibbles, serial `.sum()`.
//! - `packed_acc16` -- packed nibbles, 16 accumulators.
//!
//! Four cells rather than a chain of differences, so the two factors can be
//! read as a 2x2 instead of assuming they are independent.
//!
//! # Scope
//!
//! `gemv_nk` is *not* the production default route. Default int4 decode
//! (`accuracy_level = 0`) borrows the packed weight in place via
//! `borrowed_affine_int4_matmul_nblock` and never builds the f32 cache. This
//! path is reached by `accuracy_level = 1`, by grouped quantization (`g_idx`),
//! and by non-contiguous weights. Confirmed by instrumenting `gemv_nk` and
//! observing zero calls at `accuracy_level` 0 and 4, and calls at 1.
//!
//! Run:
//! ```text
//! cargo bench -p onnx-runtime-ep-cpu --bench int4_acc0_attribution
//! ```

mod common;

use std::time::Instant;

/// Shapes taken from the projections that dominate llama/qwen decode.
const SHAPES: &[(&str, usize, usize)] = &[
    ("qwen3_0.6b_qkv", 2048, 1024),
    ("qwen3_0.6b_mlp_up", 3072, 1024),
    ("llama3_8b_qkv", 4096, 4096),
    ("llama3_8b_mlp_down", 4096, 14336),
];

const BLOCK: usize = 32;

/// Deterministic weights/activations. A fixed LCG keeps every arm on identical
/// data without shipping a fixture.
fn fill(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn fill_nibbles(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// The production default: serial f32 reduction over an expanded f32 weight.
fn nk_serial(activation: &[f32], weight_nk: &[f32], result: &mut [f32], k: usize) {
    for (output, weight) in result.iter_mut().zip(weight_nk.chunks_exact(k)) {
        *output = activation.iter().zip(weight).map(|(&a, &b)| a * b).sum();
    }
}

/// Same memory traffic, but with the reduction chain broken into 16 independent
/// accumulators so the loop can vectorize.
fn nk_acc16(activation: &[f32], weight_nk: &[f32], result: &mut [f32], k: usize) {
    const LANES: usize = 16;
    for (output, weight) in result.iter_mut().zip(weight_nk.chunks_exact(k)) {
        let mut acc = [0.0f32; LANES];
        let (w_chunks, w_tail) = weight.as_chunks::<LANES>();
        let (a_chunks, a_tail) = activation.as_chunks::<LANES>();
        for (w, a) in w_chunks.iter().zip(a_chunks) {
            for lane in 0..LANES {
                acc[lane] += w[lane] * a[lane];
            }
        }
        let mut tail = 0.0f32;
        for (w, a) in w_tail.iter().zip(a_tail) {
            tail += *w * *a;
        }
        *output = tail + acc.iter().sum::<f32>();
    }
}

/// Packed nibbles dequantized on the fly, 16 accumulators.
///
/// Uses the same algebraic form the 8-bit acc0 GEMV already uses:
/// `sum_block scale * (q . a) - (scale * zp) * sum_block(a)`, which is the
/// dequantized `sum((q - zp) * scale * a)` rearranged so the per-weight work is
/// a plain multiply-accumulate on the raw nibble.
fn packed_acc16(
    activation: &[f32],
    packed: &[u8],
    scales: &[f32],
    zero_points: &[f32],
    result: &mut [f32],
    k: usize,
    block: usize,
) {
    const LANES: usize = 16;
    let blocks = k / block;
    let blob = block / 2;
    for (n, output) in result.iter_mut().enumerate() {
        let row = &packed[n * blocks * blob..(n + 1) * blocks * blob];
        let mut total = 0.0f32;
        for b in 0..blocks {
            let bytes = &row[b * blob..(b + 1) * blob];
            let a = &activation[b * block..(b + 1) * block];
            let scale = scales[n * blocks + b];
            let zp = zero_points[n * blocks + b];
            let mut acc = [0.0f32; LANES];
            let mut asum = [0.0f32; LANES];
            for (i, &byte) in bytes.iter().enumerate() {
                let lo = (byte & 0x0f) as f32;
                let hi = (byte >> 4) as f32;
                acc[(2 * i) % LANES] += lo * a[2 * i];
                acc[(2 * i + 1) % LANES] += hi * a[2 * i + 1];
                asum[(2 * i) % LANES] += a[2 * i];
                asum[(2 * i + 1) % LANES] += a[2 * i + 1];
            }
            let dot = acc.iter().sum::<f32>();
            let sum_a = asum.iter().sum::<f32>();
            total += scale * dot - scale * zp * sum_a;
        }
        *output = total;
    }
}

/// Packed nibbles, serial reduction: the fourth cell of the 2x2.
fn packed_serial(
    activation: &[f32],
    packed: &[u8],
    scales: &[f32],
    zero_points: &[f32],
    result: &mut [f32],
    k: usize,
    block: usize,
) {
    let blocks = k / block;
    let blob = block / 2;
    for (n, output) in result.iter_mut().enumerate() {
        let row = &packed[n * blocks * blob..(n + 1) * blocks * blob];
        let mut total = 0.0f32;
        for b in 0..blocks {
            let bytes = &row[b * blob..(b + 1) * blob];
            let a = &activation[b * block..(b + 1) * block];
            let scale = scales[n * blocks + b];
            let zp = zero_points[n * blocks + b];
            let mut dot = 0.0f32;
            let mut sum_a = 0.0f32;
            for (i, &byte) in bytes.iter().enumerate() {
                dot += (byte & 0x0f) as f32 * a[2 * i];
                dot += (byte >> 4) as f32 * a[2 * i + 1];
                sum_a += a[2 * i] + a[2 * i + 1];
            }
            total += scale * dot - scale * zp * sum_a;
        }
        *output = total;
    }
}

fn bench(label: &str, reps: usize, mut f: impl FnMut()) -> f64 {
    // Warm caches and let the frequency settle before the measured window.
    for _ in 0..3 {
        f();
    }
    let mut best = f64::MAX;
    for _ in 0..reps {
        let start = Instant::now();
        f();
        best = best.min(start.elapsed().as_secs_f64());
    }
    let _ = label;
    best
}

fn main() {
    // Match the decode thread topology a served session runs in (#1749).
    common::init_decode_topology();
    // Opened before anything else runs, so the window covers warmup too: a
    // warmup that shared cores with somebody else's run leaves caches and
    // frequency in a state the timed region inherits.
    let host_lock = common::open_host_lock_window();

    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);

    println!(
        "{:<20} {:>6} {:>6} {:>11} {:>11} {:>11} {:>11} {:>8} {:>8}",
        "shape", "N", "K", "nk_serial", "nk_acc16", "pk_serial", "pk_acc16", "chain", "traffic"
    );
    println!("{}", "-".repeat(108));

    for &(name, n, k) in SHAPES {
        let blocks = k / BLOCK;
        let activation = fill(1, k);
        let weight_nk = fill(2, n * k);
        let packed = fill_nibbles(3, n * blocks * (BLOCK / 2));
        let scales = fill(4, n * blocks);
        let zero_points = vec![8.0f32; n * blocks];
        let mut result = vec![0.0f32; n];

        let t_nk_serial = bench("nk_serial", reps, || {
            nk_serial(&activation, &weight_nk, &mut result, k);
            std::hint::black_box(&result);
        });
        let t_nk_acc16 = bench("nk_acc16", reps, || {
            nk_acc16(&activation, &weight_nk, &mut result, k);
            std::hint::black_box(&result);
        });
        let t_pk_serial = bench("packed_serial", reps, || {
            packed_serial(
                &activation,
                &packed,
                &scales,
                &zero_points,
                &mut result,
                k,
                BLOCK,
            );
            std::hint::black_box(&result);
        });
        let t_pk_acc16 = bench("packed_acc16", reps, || {
            packed_acc16(
                &activation,
                &packed,
                &scales,
                &zero_points,
                &mut result,
                k,
                BLOCK,
            );
            std::hint::black_box(&result);
        });

        println!(
            "{name:<20} {n:>6} {k:>6} {:>10.3}m {:>10.3}m {:>10.3}m {:>10.3}m {:>7.2}x {:>7.2}x",
            t_nk_serial * 1e3,
            t_nk_acc16 * 1e3,
            t_pk_serial * 1e3,
            t_pk_acc16 * 1e3,
            t_nk_serial / t_nk_acc16,
            t_nk_acc16 / t_pk_acc16,
        );

        let f32_bytes = (n * k * 4) as f64;
        let packed_bytes = (n * blocks * (BLOCK / 2) + n * blocks * 8) as f64;
        println!(
            "  bytes: f32={:.1}MB packed={:.1}MB ({:.1}x)   achieved GB/s: nk_serial={:.1} nk_acc16={:.1} pk_acc16={:.1}",
            f32_bytes / 1e6,
            packed_bytes / 1e6,
            f32_bytes / packed_bytes,
            f32_bytes / t_nk_serial / 1e9,
            f32_bytes / t_nk_acc16 / 1e9,
            packed_bytes / t_pk_acc16 / 1e9,
        );
    }

    // Last, so the second reading covers everything above it.
    common::report_host_lock(host_lock);
}
