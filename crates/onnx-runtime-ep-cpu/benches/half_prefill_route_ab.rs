//! Production-path A/B harness for the contiguous `f16`/`bf16` **prefill**
//! MatMul routes.
//!
//! Drives the real CPU EP kernel through `ExecutionProvider::get_kernel` +
//! `Kernel::execute`, so what is timed is the path production takes — the
//! dispatch decision and the narrowing of the output included — rather than a
//! kernel function called directly.
//!
//! Two arms, selected by environment so one build measures both (no
//! cross-build comparison, no rebuild between arms):
//!
//! | arm | how | route |
//! |---|---|---|
//! | fused GEBP | default | widen `B` straight into the packed L1 panel, one pass over the weight |
//! | blocked | `ONNX_GENAI_CPU_MM_HALF_GEBP=0` | the previous row-blocked half GEMM |
//!
//! The blocked route splits only over rows of C and re-widens/re-packs the
//! whole weight per row block, so its weight traffic scales with `m`; the
//! point of the fused route is to make that traffic independent of `m` while
//! keeping the operands in 16-bit storage (no resident f32 weight copy).
//!
//! An `f32` control row runs the same shapes through the f32 SGEMM. Neither arm
//! can move it, so it is the check that a difference between the two half arms
//! is the half route and not the machine.
//!
//! Both a cold phase (fresh kernel per rep — what time-to-first-token pays) and
//! a steady phase (warmed kernel) are reported, because a route that caches
//! moves cost between them; neither route here caches, so the two columns
//! should track.
//!
//! Conformance is reported per row as `max_rel`: the largest relative
//! deviation from the same GEMM run in `f32` on the *same* (already narrowed)
//! operand values, through the same production kernel. The two arms sum in
//! different orders, so their digests differ by design; what must hold is that
//! each stays at half-precision rounding distance from the f32 result, and the
//! `f32` control rows must read exactly `0`.
//!
//! Run with:
//! ```text
//! cargo bench -p onnx-runtime-ep-cpu --bench half_prefill_route_ab
//! ONNX_GENAI_CPU_MM_HALF_GEBP=0 cargo bench -p onnx-runtime-ep-cpu --bench half_prefill_route_ab
//! ```
//! `PROBE_SHAPE=small|big` picks one shape; `PROBE_MS=prefill|cross` picks a
//! row sweep.

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

/// Order-sensitive digest of the output, so the two arms are compared on
/// values and not only on time.
fn digest(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (index, value) in values.iter().enumerate() {
        hash ^= u64::from(value.to_bits()) ^ (index as u64);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Same GEMM in `f32` on the already-narrowed operand values, through the same
/// production kernel: the conformance reference both half arms are measured
/// against.
fn reference(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Vec<f32> {
    let a32 = Tensor::floats(FloatDType::F32, &[m, k], &a.to_f32());
    let b32 = Tensor::floats(FloatDType::F32, &[k, n], &b.to_f32());
    let mut out = Tensor::zeros(FloatDType::F32, &[m, n]);
    build_kernel(FloatDType::F32, m, k, n)
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

fn build_kernel(dtype: FloatDType, m: usize, k: usize, n: usize) -> Box<dyn Kernel> {
    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &[vec![m, k], vec![k, n]], 1)
        .expect("CPU EP must register MatMul");
    kernel.set_constant_inputs(&[false, true]);
    let _ = dtype;
    kernel
}

fn main() {
    // Match the decode thread topology a served session runs in (#1749).
    common::init_decode_topology();

    let shapes: Vec<(usize, usize)> = match std::env::var("PROBE_SHAPE").as_deref() {
        Ok("big") => vec![(4096, 11008)],
        Ok("small") => vec![(2048, 2048)],
        _ => vec![(2048, 2048), (4096, 11008)],
    };
    let ms: Vec<usize> = match std::env::var("PROBE_MS").as_deref() {
        Ok("prefill") => vec![8, 64, 256],
        Ok("cross") => vec![1, 2, 4, 8, 16, 32],
        _ => vec![1, 8, 64, 256],
    };
    println!(
        "half_gebp={}",
        std::env::var("ONNX_GENAI_CPU_MM_HALF_GEBP").unwrap_or_else(|_| "default(on)".into())
    );
    println!(
        "{:>6} {:>6} {:>6} {:>5} {:>10} {:>10} {:>10} {:>10} {:>18}",
        "dtype", "k", "n", "m", "cold_ms", "steady_ms", "gflops", "max_rel", "digest"
    );
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::Bf16] {
        for &(k, n) in &shapes {
            let b = Tensor::floats(dtype, &[k, n], &floats(k * n, 0.3));
            for &m in &ms {
                let a = Tensor::floats(dtype, &[m, k], &floats(m * k, 1.1));
                let mut out = Tensor::zeros(dtype, &[m, n]);
                let ins = vec![a.view(), b.view()];

                let cold = median(
                    (0..3)
                        .map(|_| {
                            let kernel = build_kernel(dtype, m, k, n);
                            let start = Instant::now();
                            kernel
                                .execute(&ins, &mut [out.view_mut()])
                                .expect("execute");
                            start.elapsed().as_secs_f64() * 1e3
                        })
                        .collect(),
                );

                let kernel = build_kernel(dtype, m, k, n);
                for _ in 0..2 {
                    kernel.execute(&ins, &mut [out.view_mut()]).expect("warmup");
                }
                let steady = median(
                    (0..5)
                        .map(|_| {
                            let start = Instant::now();
                            kernel
                                .execute(&ins, &mut [out.view_mut()])
                                .expect("execute");
                            start.elapsed().as_secs_f64() * 1e3
                        })
                        .collect(),
                );
                let gflops = (2.0 * m as f64 * k as f64 * n as f64) / (steady * 1e6);
                let got = out.to_f32();
                println!(
                    "{:>6} {k:>6} {n:>6} {m:>5} {cold:>10.3} {steady:>10.3} {gflops:>10.2} \
                     {:>10.2e} {:>18}",
                    dtype.name(),
                    max_relative_deviation(&got, &reference(&a, &b, m, k, n)),
                    digest(&got)
                );
            }
        }
    }
}
