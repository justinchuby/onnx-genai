//! Production-path A/B harness for the **8-bit** `MatMulNBits` prefill routes.
//!
//! Drives the real CPU EP kernel through `ExecutionProvider::get_kernel` +
//! `Kernel::execute`, so what is timed is the path production takes — the
//! dispatch decision included — rather than a kernel function called directly.
//!
//! Two arms, both reachable from one build by environment alone:
//!
//! | arm | how | route |
//! |---|---|---|
//! | fused GEBP | default | dequant fused into the packed-panel GEMM |
//! | dequant + GEMM | `ONNX_GENAI_CPU_MM_INT8_GEBP=0` | materialize a `k x n` f32 weight per call, then SGEMM |
//!
//! The second arm is what shipped before: on a native (non-MLAS) build the
//! 8-bit prefill has no borrowed route at all, so it expands the whole weight
//! to f32 in `[k, n]` order — four bytes written for every byte read, at stride
//! `n`, on **every** call, because nothing caches that layout.
//!
//! Both a cold phase (fresh kernel per rep — what time-to-first-token pays) and
//! a steady phase (warmed kernel) are reported. Neither arm caches anything
//! here, so the two columns should agree; a divergence would mean one of them
//! grew a hidden residency.
//!
//! `digest` is an order-sensitive hash of the whole output. `xarm` compares
//! this arm's full output against the *other* arm's, computed in the same
//! process by flipping the switch: that is the assignment == execution
//! evidence, since it shows the route the timings separated by 17x is the one
//! that produced these numbers. `max_rel` is measured against a
//! dequantize-then-naive-GEMM oracle computed in f64.
//!
//! Run with:
//! ```text
//! cargo bench -p onnx-runtime-ep-cpu --bench int8_prefill_route_ab
//! ONNX_GENAI_CPU_MM_INT8_GEBP=0 cargo bench -p onnx-runtime-ep-cpu --bench int8_prefill_route_ab
//! ```
//! `PROBE_SHAPE=small|big|wide` picks one shape; `PROBE_MS=prefill|cross` picks
//! a row sweep, or pass an explicit comma-separated list such as
//! `PROBE_MS=2,3,4`.

mod common;

use std::time::Instant;

use common::Tensor;
use onnx_runtime_ep_api::{ExecutionProvider, Kernel};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, Node, NodeId};

fn packed_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

fn floats(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.0137 + seed).sin() * 0.5)
        .collect()
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Order-sensitive digest, so two routes that agree numerically still show
/// different values when they reduce in different orders.
fn digest(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &value in values {
        hash ^= u64::from(value.to_bits());
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn build_kernel(k: usize, n: usize, block_size: usize, m: usize) -> Box<dyn Kernel> {
    let blocks = k.div_ceil(block_size);
    // 8 bits is one byte per weight, so the blob is the block itself.
    let shapes = vec![vec![m, k], vec![n, blocks, block_size], vec![n, blocks]];
    let mut node = Node::new(NodeId(0), "MatMulNBits", vec![], vec![]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("K", Attribute::Int(k as i64)),
        ("N", Attribute::Int(n as i64)),
        ("bits", Attribute::Int(8)),
        ("block_size", Attribute::Int(block_size as i64)),
        ("accuracy_level", Attribute::Int(0)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &shapes, 1)
        .expect("CPU EP must register MatMulNBits");
    kernel.set_constant_inputs(&[false, true, true]);
    kernel
}

/// The quantized operand, as the oracle reads it.
struct Weight<'a> {
    packed: &'a [u8],
    scales: &'a [f32],
    k: usize,
    blocks: usize,
    block_size: usize,
}

/// `sum_p a[row, p] * (packed[col, p] - 128) * scale[col, block(p)]`, in f64.
fn oracle_row(activations: &[f32], weight: &Weight<'_>, row: usize, col: usize) -> f64 {
    let Weight {
        packed,
        scales,
        k,
        blocks,
        block_size,
    } = *weight;
    (0..k)
        .map(|p| {
            let scale = f64::from(scales[col * blocks + p / block_size]);
            let weight = (f64::from(packed[col * blocks * block_size + p]) - 128.0) * scale;
            f64::from(activations[row * k + p]) * weight
        })
        .sum()
}

fn main() {
    // Match the decode thread topology a served session runs in (#1749).
    common::init_decode_topology();
    // Opened before anything else runs, so the window covers warmup too: a
    // warmup that shared cores with somebody else's run leaves caches and
    // frequency in a state the timed region inherits.
    let host_lock = common::open_host_lock_window();

    let block_size = 32usize;
    let shapes: Vec<(usize, usize)> = match std::env::var("PROBE_SHAPE").as_deref() {
        Ok("big") => vec![(4096, 11008)],
        Ok("small") => vec![(2048, 2048)],
        // The `MatMulNBits` shape the assignment matrix measures against ORT.
        Ok("wide") => vec![(3584, 3584)],
        _ => vec![(2048, 2048), (3584, 3584), (4096, 11008)],
    };
    let gebp = std::env::var("ONNX_GENAI_CPU_MM_INT8_GEBP").unwrap_or_else(|_| "default".into());
    let other_arm = if gebp == "0" { "1" } else { "0" };
    println!("int8_gebp={gebp} threads={}", rayon::current_num_threads());
    println!(
        "{:>6} {:>6} {:>5} {:>10} {:>10} {:>9} {:>9} {:>10} {:>20}",
        "k", "n", "m", "cold_ms", "steady_ms", "gflops", "max_rel", "xarm", "digest"
    );
    for &(k, n) in shapes.iter() {
        let blocks = k.div_ceil(block_size);
        let packed = packed_bytes(n * blocks * block_size, 7);
        let scales = floats(n * blocks, 0.3)
            .into_iter()
            .map(|v| v.abs().max(0.01) * 0.02)
            .collect::<Vec<_>>();
        let b = Tensor::u8(&[n, blocks, block_size], &packed);
        let scales_t = Tensor::floats(common::FloatDType::F32, &[n, blocks], &scales);
        let ms: Vec<usize> = match std::env::var("PROBE_MS").as_deref() {
            Ok("prefill") => vec![8, 64, 256],
            Ok("cross") => vec![1, 2, 3, 4, 6, 8, 16, 32],
            Ok(list) if list.contains(',') => list
                .split(',')
                .map(|value| value.trim().parse().expect("PROBE_MS row count"))
                .collect(),
            _ => vec![1, 4, 64, 256, 512],
        };
        for &m in &ms {
            let activation_values = floats(m * k, 1.1);
            let a = Tensor::floats(common::FloatDType::F32, &[m, k], &activation_values);
            let mut out = Tensor::zeros(common::FloatDType::F32, &[m, n]);
            let ins = vec![a.view(), b.view(), scales_t.view()];

            let cold = median(
                (0..3)
                    .map(|_| {
                        let kernel = build_kernel(k, n, block_size, m);
                        let start = Instant::now();
                        kernel
                            .execute(&ins, &mut [out.view_mut()])
                            .expect("execute");
                        start.elapsed().as_secs_f64() * 1e3
                    })
                    .collect(),
            );

            let kernel = build_kernel(k, n, block_size, m);
            for _ in 0..2 {
                kernel.execute(&ins, &mut [out.view_mut()]).expect("warmup");
            }
            let steady = median(
                (0..7)
                    .map(|_| {
                        let start = Instant::now();
                        kernel
                            .execute(&ins, &mut [out.view_mut()])
                            .expect("execute");
                        start.elapsed().as_secs_f64() * 1e3
                    })
                    .collect(),
            );

            let got = out.to_f32();
            // A few probes, not the whole matrix: the oracle is O(k) per
            // element in f64 and would dominate the run at these shapes.
            let mut max_rel = 0.0f64;
            let weight = Weight {
                packed: &packed,
                scales: &scales,
                k,
                blocks,
                block_size,
            };
            for &(row, col) in &[(0usize, 0usize), (m / 2, n / 2), (m - 1, n - 1)] {
                let want = oracle_row(&activation_values, &weight, row, col);
                let have = f64::from(got[row * n + col]);
                let rel = (have - want).abs() / (1.0 + want.abs());
                max_rel = max_rel.max(rel);
            }
            // Cross-arm conformance: run the *other* route in this same
            // process and compare the full output, so the numbers reported
            // above are provably the ones the timed route produced.
            // SAFETY: single-threaded here; no worker is reading the
            // environment while this executes.
            unsafe { std::env::set_var("ONNX_GENAI_CPU_MM_INT8_GEBP", other_arm) };
            let mut other_out = Tensor::zeros(common::FloatDType::F32, &[m, n]);
            build_kernel(k, n, block_size, m)
                .execute(&ins, &mut [other_out.view_mut()])
                .expect("execute");
            match gebp.as_str() {
                "default" => unsafe { std::env::remove_var("ONNX_GENAI_CPU_MM_INT8_GEBP") },
                value => unsafe { std::env::set_var("ONNX_GENAI_CPU_MM_INT8_GEBP", value) },
            }
            let other = other_out.to_f32();
            let xarm = got
                .iter()
                .zip(other.iter())
                .map(|(a, b)| f64::from((a - b).abs()) / (1.0 + f64::from(b.abs())))
                .fold(0.0f64, f64::max);
            let bitexact = got == other;
            let xarm = if bitexact {
                "bitexact".to_string()
            } else {
                format!("{xarm:.1e}")
            };

            let gflops = (2.0 * m as f64 * k as f64 * n as f64) / (steady * 1e6);
            let digest = digest(&got);
            println!(
                "{k:>6} {n:>6} {m:>5} {cold:>10.3} {steady:>10.3} {gflops:>9.2} \
                 {max_rel:>9.2e} {xarm:>10} {digest:>20}"
            );
        }
    }

    // Last, so the second reading covers everything above it.
    common::report_host_lock(host_lock);
}
