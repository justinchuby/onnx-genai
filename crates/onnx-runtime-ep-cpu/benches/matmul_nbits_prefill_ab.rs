//! Production-path A/B harness for the `MatMulNBits` dense-fallback prefill
//! route (#959, #1176).
//!
//! Drives the **real** CPU EP kernel through `ExecutionProvider::get_kernel` +
//! `Kernel::execute` — the same entry the executor uses — rather than calling a
//! GEMM directly, so what is timed is the path production takes, including the
//! constant-weight prepack decision and the dequant layout choice.
//!
//! Two phases are reported per shape, because the change moves cost between
//! them:
//!
//! * **cold** — a fresh kernel per repetition, so the first `execute` pays the
//!   one-time weight dequant. This is the time-to-first-token term #959
//!   measured (the `dequant-kn` strided scatter).
//! * **steady** — the same kernel re-executed after warmup, so the weight is
//!   cached and only the GEMM is timed.
//!
//! The harness is deliberately build-agnostic: it uses no symbol introduced by
//! the NT change, so the identical file runs on `main` and on the branch and the
//! two runs are a true A/B. It also prints a bit-level digest of the output so
//! the two runs can be compared for bit-identity, not just speed.
//!
//! Run with:
//! ```text
//! cargo bench -p onnx-runtime-ep-cpu --bench matmul_nbits_prefill_ab
//! ```
//! `ONNX_GENAI_PROFILE_MM=1` additionally prints the per-phase prepack lines
//! (`dequant-kn` vs `dequant-nk`) the route change is visible in.

mod common;

use std::time::Instant;

use common::Tensor;
use onnx_runtime_ep_api::{ExecutionProvider, Kernel};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, Node, NodeId};

/// Deterministic pseudo-random bytes for the packed weight.
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

/// XOR-fold of the raw output bits: identical across two builds iff every
/// output element is bit-identical.
fn digest(values: &[f32]) -> u64 {
    let mut acc = 0xcbf2_9ce4_8422_2325u64;
    for (i, v) in values.iter().enumerate() {
        acc ^= (v.to_bits() as u64).rotate_left((i % 61) as u32);
        acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
    }
    acc
}

struct Case {
    label: &'static str,
    bits: i64,
    k: usize,
    n: usize,
    block_size: usize,
    with_g_idx: bool,
}

/// Zero points are packed two-per-byte for 4-bit, one-per-byte for 8-bit.
fn zp_cols(bits: i64, blocks: usize) -> usize {
    if bits == 4 {
        blocks.div_ceil(2)
    } else {
        blocks
    }
}

fn build_kernel(case: &Case, m: usize) -> Box<dyn Kernel> {
    let blocks = case.k.div_ceil(case.block_size);
    let blob = if case.bits == 4 {
        case.block_size / 2
    } else {
        case.block_size
    };
    let mut shapes = vec![
        vec![m, case.k],
        vec![case.n, blocks, blob],
        vec![case.n, blocks],
    ];
    if case.with_g_idx {
        // zero_points then g_idx: the g_idx input is index 4, so a zero_points
        // input has to be present for the shapes to line up.
        shapes.push(vec![case.n, zp_cols(case.bits, blocks)]);
        shapes.push(vec![case.k]);
    }
    let mut node = Node::new(NodeId(0), "MatMulNBits", vec![], vec![]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("K", Attribute::Int(case.k as i64)),
        ("N", Attribute::Int(case.n as i64)),
        ("bits", Attribute::Int(case.bits)),
        ("block_size", Attribute::Int(case.block_size as i64)),
        ("accuracy_level", Attribute::Int(0)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &shapes, 1)
        .expect("CPU EP must register MatMulNBits");
    // Weight, scales (and zero points / g_idx) are graph initializers in a real
    // model: that is what enables the one-time prepack the cold phase measures.
    let mut constant = vec![false; shapes.len()];
    for flag in constant.iter_mut().skip(1) {
        *flag = true;
    }
    kernel.set_constant_inputs(&constant);
    kernel
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let cases = [
        Case {
            label: "int8 dense-fallback",
            bits: 8,
            k: 2048,
            n: 2048,
            block_size: 32,
            with_g_idx: false,
        },
        Case {
            label: "int8 dense-fallback",
            bits: 8,
            k: 4096,
            n: 11008,
            block_size: 32,
            with_g_idx: false,
        },
        Case {
            label: "int4 g_idx dense-fallback",
            bits: 4,
            k: 2048,
            n: 2048,
            block_size: 32,
            with_g_idx: true,
        },
    ];

    println!(
        "{:<26} {:>6} {:>6} {:>5} {:>12} {:>12} {:>18}",
        "case", "k", "n", "m", "cold_ms", "steady_ms", "digest"
    );

    for case in &cases {
        let blocks = case.k.div_ceil(case.block_size);
        let blob = if case.bits == 4 {
            case.block_size / 2
        } else {
            case.block_size
        };
        let packed = packed_bytes(case.n * blocks * blob, 7);
        let scales = floats(case.n * blocks, 0.3)
            .into_iter()
            .map(|v| v.abs().max(0.01) * 0.02)
            .collect::<Vec<_>>();
        let b = Tensor::u8(&[case.n, blocks, blob], &packed);
        let scales_t = Tensor::floats(common::FloatDType::F32, &[case.n, blocks], &scales);
        let zp_cols = zp_cols(case.bits, blocks);
        let zp = Tensor::u8(&[case.n, zp_cols], &vec![0x88u8; case.n * zp_cols]);
        let g_idx = Tensor::i32(
            &[case.k],
            &(0..case.k)
                .map(|i| (i / case.block_size) as i32)
                .collect::<Vec<_>>(),
        );

        for &m in &[8usize, 64] {
            let a = Tensor::floats(
                common::FloatDType::F32,
                &[m, case.k],
                &floats(m * case.k, 1.1),
            );
            let mut out = Tensor::zeros(common::FloatDType::F32, &[m, case.n]);

            let inputs = |with_g_idx: bool| {
                let mut v = vec![a.view(), b.view(), scales_t.view()];
                if with_g_idx {
                    v.push(zp.view());
                    v.push(g_idx.view());
                }
                v
            };

            // Cold: a fresh kernel each rep, so every measured execute pays the
            // one-time dequant of the constant weight.
            let cold = median(
                (0..3)
                    .map(|_| {
                        let kernel = build_kernel(case, m);
                        let ins = inputs(case.with_g_idx);
                        let start = Instant::now();
                        kernel
                            .execute(&ins, &mut [out.view_mut()])
                            .expect("MatMulNBits execute");
                        start.elapsed().as_secs_f64() * 1e3
                    })
                    .collect(),
            );

            // Steady: one kernel, warmed, so the weight cache is already built.
            let kernel = build_kernel(case, m);
            let ins = inputs(case.with_g_idx);
            for _ in 0..2 {
                kernel
                    .execute(&ins, &mut [out.view_mut()])
                    .expect("MatMulNBits warmup execute");
            }
            let steady = median(
                (0..7)
                    .map(|_| {
                        let start = Instant::now();
                        kernel
                            .execute(&ins, &mut [out.view_mut()])
                            .expect("MatMulNBits execute");
                        start.elapsed().as_secs_f64() * 1e3
                    })
                    .collect(),
            );

            println!(
                "{:<26} {:>6} {:>6} {:>5} {:>12.3} {:>12.3} {:>18x}",
                case.label,
                case.k,
                case.n,
                m,
                cold,
                steady,
                digest(&out.to_f32())
            );
        }
    }
}
