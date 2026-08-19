//! Production-path A/B harness for the `M == 1` **decode** MatMul routes on
//! contiguous `f16`/`bf16` weights.
//!
//! Drives the real CPU EP kernel through `ExecutionProvider::get_kernel` +
//! `Kernel::execute`, so what is timed is the path production takes — the
//! dispatch decision and the narrowing of the output included — rather than a
//! kernel function called directly.
//!
//! Three arms, all reachable from one build by environment alone:
//!
//! | arm | how | route |
//! |---|---|---|
//! | GEMV | `ONNX_GENAI_CPU_MM_HALF_GEBP=0` | read `B` in place in `[K, N]` order, no packing, no copy |
//! | fused GEBP | `ONNX_GENAI_CPU_MM_HALF_GEMV=0` | widen `B` into packed L1 panels (the prefill kernel) |
//! | blocked | `ONNX_GENAI_CPU_MM_HALF_GEMV=0 ONNX_GENAI_CPU_MM_HALF_GEBP=0` | the row-blocked half GEMM |
//!
//! Note that **neither** single-knob arm is the default: with no environment
//! set, the shipped routing picks per shape -- GEMV below
//! `HALF_DECODE_GEBP_MIN_WEIGHT`, fused GEBP at or above it -- so an unset run
//! is a fourth, *shipped* arm that agrees with the GEMV column on small
//! weights and with the GEBP column on large ones. The output digest is what
//! says which one it took.
//!
//! Decode is bandwidth-bound, not flop-bound: at `M == 1` every weight element
//! is touched exactly once, so the figure of merit is **GB/s of weight read**,
//! reported next to the `2*K*N` GFLOP/s for continuity with the prefill sheet.
//! `bytes` counts only the `2 * K * N` weight, which is what both the roofline
//! and the packing arms are actually limited by.
//!
//! An `f32` control row runs the same shapes through the f32 path. No arm here
//! can move it, so it is the check that a difference between the half arms is
//! the route and not the machine.
//!
//! Conformance is reported per row as `max_rel`: the largest relative
//! deviation from the same GEMM run in `f32` on the *same* (already narrowed)
//! operand values, through the same production kernel.
//!
//! Run with:
//! ```text
//! ONNX_GENAI_CPU_MM_HALF_GEBP=0 cargo bench -p onnx-runtime-ep-cpu --bench half_decode_gemv_ab
//! ONNX_GENAI_CPU_MM_HALF_GEMV=0 cargo bench -p onnx-runtime-ep-cpu --bench half_decode_gemv_ab
//! ONNX_GENAI_CPU_MM_HALF_GEMV=0 ONNX_GENAI_CPU_MM_HALF_GEBP=0 \
//!     cargo bench -p onnx-runtime-ep-cpu --bench half_decode_gemv_ab
//! cargo bench -p onnx-runtime-ep-cpu --bench half_decode_gemv_ab  # shipped routing
//! ```
//! `PROBE_SHAPE=mlp|lm_head|square` picks one shape.

mod common;

use std::time::Instant;

use common::{FloatDType, Tensor};
use onnx_runtime_ep_api::{ExecutionProvider, Kernel};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Node, NodeId};

fn floats(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.0137 + seed).sin() * 0.5)
        .collect()
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Order-sensitive digest of the output, so the arms are compared on values
/// and not only on time.
fn digest(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (index, value) in values.iter().enumerate() {
        hash ^= u64::from(value.to_bits()) ^ (index as u64);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn build_kernel(m: usize, k: usize, n: usize) -> Box<dyn Kernel> {
    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &[vec![m, k], vec![k, n]], 1)
        .expect("CPU EP must register MatMul");
    kernel.set_constant_inputs(&[false, true]);
    kernel
}

/// Same GEMM in `f32` on the already-narrowed operand values, through the same
/// production kernel: the conformance reference every half arm is measured
/// against.
fn reference(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Vec<f32> {
    let a32 = Tensor::floats(FloatDType::F32, &[m, k], &a.to_f32());
    let b32 = Tensor::floats(FloatDType::F32, &[k, n], &b.to_f32());
    let mut out = Tensor::zeros(FloatDType::F32, &[m, n]);
    build_kernel(m, k, n)
        .execute(&[a32.view(), b32.view()], &mut [out.view_mut()])
        .expect("reference execute");
    out.to_f32()
}

fn max_relative_deviation(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / (1.0 + w.abs()))
        .fold(0.0f32, f32::max)
}

fn main() {
    // Shapes a decode step actually issues: a 7B-class MLP projection, a
    // vocabulary head (the widest weight in the graph), and a square control.
    let shapes: Vec<(&str, usize, usize)> = match std::env::var("PROBE_SHAPE").as_deref() {
        Ok("mlp") => vec![("mlp", 4096, 11008)],
        Ok("lm_head") => vec![("lm_head", 896, 151936)],
        Ok("square") => vec![("square", 2048, 2048)],
        Ok("tiny") => vec![("attn_out", 1024, 768), ("small", 512, 512)],
        // A `k = 4096` column sweep across the GEMV/GEBP crossover: 4.2M,
        // 8.4M, 16.8M, 33.6M and 45.1M weight elements.
        Ok("cross") => vec![("w17M", 4096, 4096), ("w34M", 4096, 8192)],
        Ok("sweep") => vec![
            ("w4M", 4096, 1024),
            ("w8M", 4096, 2048),
            ("w17M", 4096, 4096),
            ("w34M", 4096, 8192),
            ("w45M", 4096, 11008),
        ],
        _ => vec![
            ("attn_out", 1024, 768),
            ("square", 2048, 2048),
            ("mlp", 4096, 11008),
            ("lm_head", 896, 151936),
        ],
    };
    println!(
        "half_gemv={} half_gebp={}",
        std::env::var("ONNX_GENAI_CPU_MM_HALF_GEMV").unwrap_or_else(|_| "default(on)".into()),
        std::env::var("ONNX_GENAI_CPU_MM_HALF_GEBP").unwrap_or_else(|_| "default(on)".into())
    );
    println!(
        "{:>6} {:>8} {:>6} {:>7} {:>10} {:>10} {:>9} {:>8} {:>10} {:>18}",
        "dtype", "shape", "k", "n", "cold_ms", "steady_ms", "GB/s", "gflops", "max_rel", "digest"
    );
    let m = 1;
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::Bf16] {
        for &(label, k, n) in &shapes {
            let b = Tensor::floats(dtype, &[k, n], &floats(k * n, 0.3));
            let a = Tensor::floats(dtype, &[m, k], &floats(m * k, 1.1));
            let mut out = Tensor::zeros(dtype, &[m, n]);
            let ins = vec![a.view(), b.view()];

            let cold = median(
                (0..3)
                    .map(|_| {
                        let kernel = build_kernel(m, k, n);
                        let start = Instant::now();
                        kernel
                            .execute(&ins, &mut [out.view_mut()])
                            .expect("execute");
                        start.elapsed().as_secs_f64() * 1e3
                    })
                    .collect(),
            );

            let kernel = build_kernel(m, k, n);
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
            let weight_bytes = (k * n * dtype.size_of()) as f64;
            let gbps = weight_bytes / (steady * 1e6);
            let gflops = (2.0 * m as f64 * k as f64 * n as f64) / (steady * 1e6);
            let got = out.to_f32();
            println!(
                "{:>6} {label:>8} {k:>6} {n:>7} {cold:>10.3} {steady:>10.3} {gbps:>9.1} \
                 {gflops:>8.2} {:>10.2e} {:>18}",
                dtype.name(),
                max_relative_deviation(&got, &reference(&a, &b, m, k, n)),
                digest(&got)
            );
        }
    }
}
