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
//! `HALF_PREFILL_GEBP_MIN_WEIGHT`, fused GEBP at or above it -- so an unset run
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
use onnx_runtime_ir::{Attribute, Node, NodeId};

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

/// `Gemm` with `transB = 1` -- the `[N, K]` stored order every `nn.Linear`
/// export produces, and the one `MatMul` cannot express.
///
/// The `[K, N]` arm above cannot reach the `[N, K]` GEMV at all, so without
/// this the transposed route had no benchmark row. That is the coverage gap
/// that let a `bf16` decode sit on the blocked GEMM unmeasured.
fn build_gemm_transb(m: usize, k: usize, n: usize) -> Box<dyn Kernel> {
    let mut node = Node::new(NodeId(0), "Gemm", vec![], vec![]);
    node.attributes.insert("transB".into(), Attribute::Int(1));
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &[vec![m, k], vec![n, k]], 1)
        .expect("CPU EP must register Gemm");
    kernel.set_constant_inputs(&[false, true]);
    kernel
}

/// `FusedMatMulBias` -- the optimizer's fusion of `MatMul + Add(bias)`, and the
/// form most real projections actually take, because almost every `nn.Linear`
/// has a bias.
///
/// It had **no row in any benchmark** before #1702. That is why its divergence
/// from `MatMul` survived #1687: the two arms above cover `MatMul` and `Gemm`,
/// so a reader would reasonably conclude every decode GEMV was measured, while
/// the operator carrying most of a model's projections was not exercised at
/// all. A gate with no row below/at/above it is not a measured gate.
fn build_fused_matmul_bias(m: usize, k: usize, n: usize) -> Box<dyn Kernel> {
    let mut node = Node::new(NodeId(0), "FusedMatMulBias", vec![], vec![]);
    // The fusion lives in `com.microsoft`, not the default domain -- a plain
    // `Node::new` gets `NoEpForOp` here, which is itself a small piece of
    // evidence that nothing had ever built this kernel from a bench.
    node.domain = "com.microsoft".to_string();
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &[vec![m, k], vec![k, n], vec![n]], 1)
        .expect("CPU EP must register FusedMatMulBias");
    kernel.set_constant_inputs(&[false, true, true]);
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
    // Match the decode thread topology a served session runs in (#1749).
    common::init_decode_topology();
    // Opened before anything else runs, so the window covers warmup too: a
    // warmup that shared cores with somebody else's run leaves caches and
    // frequency in a state the timed region inherits.
    let host_lock = common::open_host_lock_window();

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
        Ok("low") => vec![
            ("k1024n768", 1024, 768),
            ("k1024n1024", 1024, 1024),
            ("k1024n2048", 1024, 2048),
            ("k2048n1024", 2048, 1024),
            ("k512n4096", 512, 4096),
        ],
        Ok("full") => vec![
            ("k1024n768", 1024, 768),
            ("k2048n2048", 2048, 2048),
            ("k4096n1024", 4096, 1024),
            ("k2048n4096", 2048, 4096),
            ("k4096n2048", 4096, 2048),
            ("k4096n4096", 4096, 4096),
            ("k4096n8192", 4096, 8192),
            ("k4096n11008", 4096, 11008),
            ("k896n151936", 896, 151936),
        ],
        // Rows immediately below, at and above **both** ends of the
        // `HALF_PREFILL_GEBP_MIN_WEIGHT`/`..._MAX_WEIGHT` band, at three
        // different `k` so a `k`-dependent effect cannot hide inside a
        // `k * n` gate: 0.79M, 1.05M, 2.1M, 3.1M, 4.2M, 6.3M, 8.4M.
        Ok("band") => vec![
            ("w0.79M", 1024, 768),
            ("w1.05M_k1024", 1024, 1024),
            ("w2.1M_k1024", 1024, 2048),
            ("w2.1M_k2048", 2048, 1024),
            ("w3.1M_k1024", 1024, 3072),
            ("w3.1M_k2048", 2048, 1536),
            ("w4.2M_k1024", 1024, 4096),
            ("w4.2M_k2048", 2048, 2048),
            ("w4.2M_k4096", 4096, 1024),
            ("w6.3M_k2048", 2048, 3072),
            ("w8.4M_k2048", 2048, 4096),
        ],
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
    let trans_b = std::env::var("PROBE_OP").as_deref() == Ok("gemm_transb");
    let fused_bias = std::env::var("PROBE_OP").as_deref() == Ok("fused_matmul_bias");
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::Bf16] {
        for &(label, k, n) in &shapes {
            if trans_b {
                run_transposed(dtype, label, m, k, n);
                continue;
            }
            if fused_bias {
                run_fused_bias(dtype, label, m, k, n);
                continue;
            }
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

    // Last, so the second reading covers everything above it.
    common::report_host_lock(host_lock);
}

/// One transposed-`Gemm` row, timed exactly like the `[K, N]` arm above.
///
/// The weight is generated in `[K, N]` and transposed into `[N, K]` so the
/// values -- and therefore the reference and the digest -- are identical to the
/// untransposed row of the same shape. A route that silently reinterpreted the
/// bit patterns would move `max_rel`, not just the timing.
fn run_transposed(dtype: FloatDType, label: &str, m: usize, k: usize, n: usize) {
    let values = floats(k * n, 0.3);
    let mut transposed = vec![0.0f32; n * k];
    for p in 0..k {
        for j in 0..n {
            transposed[j * k + p] = values[p * n + j];
        }
    }
    let b = Tensor::floats(dtype, &[n, k], &transposed);
    let a = Tensor::floats(dtype, &[m, k], &floats(m * k, 1.1));
    let mut out = Tensor::zeros(dtype, &[m, n]);
    let ins = vec![a.view(), b.view()];

    let cold = median(
        (0..3)
            .map(|_| {
                let kernel = build_gemm_transb(m, k, n);
                let start = Instant::now();
                kernel
                    .execute(&ins, &mut [out.view_mut()])
                    .expect("execute");
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    );

    let kernel = build_gemm_transb(m, k, n);
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

    let reference_b = Tensor::floats(dtype, &[k, n], &values);
    let weight_bytes = (k * n * dtype.size_of()) as f64;
    let got = out.to_f32();
    println!(
        "{:>6} {label:>8} {k:>6} {n:>7} {cold:>10.3} {steady:>10.3} {:>9.1} \
         {:>8.2} {:>10.2e} {:>18}",
        dtype.name(),
        weight_bytes / (steady * 1e6),
        (2.0 * m as f64 * k as f64 * n as f64) / (steady * 1e6),
        max_relative_deviation(&got, &reference(&a, &reference_b, m, k, n)),
        digest(&got)
    );
}

/// One `FusedMatMulBias` row, timed exactly like the `[K, N]` `MatMul` arm.
///
/// The weight, the activation and the shape are byte-identical to that arm, so
/// the only difference between the two rows is the operator — which is the
/// whole point. `max_rel` is measured against the *same* f32 `MatMul`
/// reference plus the bias, so a route that changed accumulation order or
/// applied the bias in the wrong place moves the column rather than passing
/// quietly.
///
/// The bias is deliberately non-zero and non-uniform. A zero bias would make
/// this row indistinguishable from `MatMul` even if the epilogue were dropped
/// entirely, which is the kind of control that cannot detect the thing it
/// exists to detect (ledger §25).
fn run_fused_bias(dtype: FloatDType, label: &str, m: usize, k: usize, n: usize) {
    let b_values = floats(k * n, 0.3);
    let bias_values = floats(n, 2.7);
    let b = Tensor::floats(dtype, &[k, n], &b_values);
    let a = Tensor::floats(dtype, &[m, k], &floats(m * k, 1.1));
    let bias = Tensor::floats(dtype, &[n], &bias_values);
    let mut out = Tensor::zeros(dtype, &[m, n]);
    let ins = vec![a.view(), b.view(), bias.view()];

    let cold = median(
        (0..3)
            .map(|_| {
                let kernel = build_fused_matmul_bias(m, k, n);
                let start = Instant::now();
                kernel
                    .execute(&ins, &mut [out.view_mut()])
                    .expect("execute");
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    );

    let kernel = build_fused_matmul_bias(m, k, n);
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

    // The bias is narrowed to `dtype` and widened back, exactly as the kernel
    // sees it, so the reference does not credit the arm with precision the
    // stored bias does not have.
    let mut want = reference(&a, &b, m, k, n);
    let bias_seen = bias.to_f32();
    for (index, value) in want.iter_mut().enumerate() {
        *value += bias_seen[index % n];
    }
    let weight_bytes = (k * n * dtype.size_of()) as f64;
    let got = out.to_f32();
    println!(
        "{:>6} {label:>8} {k:>6} {n:>7} {cold:>10.3} {steady:>10.3} {:>9.1} \
         {:>8.2} {:>10.2e} {:>18}",
        dtype.name(),
        weight_bytes / (steady * 1e6),
        (2.0 * m as f64 * k as f64 * n as f64) / (steady * 1e6),
        max_relative_deviation(&got, &want),
        digest(&got)
    );
}
