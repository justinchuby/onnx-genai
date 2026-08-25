//! Production-path A/B harness for the symmetric int4 **prefill** routes
//! (#1117).
//!
//! Drives the real CPU EP kernel through `ExecutionProvider::get_kernel` +
//! `Kernel::execute`, so what is timed is the path production takes — the
//! dispatch decision included — rather than a kernel function called directly.
//!
//! Three arms, selected by environment so a single build measures all of them
//! (no cross-build comparison, no rebuild between arms):
//!
//! | arm | how | route |
//! |---|---|---|
//! | fused GEBP | default | dequant fused into the packed-panel GEMM (#1117) |
//! | row-serial | `ONNX_GENAI_CPU_MM_INT4_GEBP=0` | the previous borrowed path |
//! | dense f32 | `PROBE_ACC=1` | dequantize to a resident f32 weight, then SGEMM |
//!
//! The dense arm is the *ceiling* reference, not a candidate: it reaches GEMM
//! FLOPs by materializing an f32 weight at 8x the packed size, which is exactly
//! the residency #979 removed. The point of the fused arm is to reach the same
//! FLOPs with the weight still borrowed.
//!
//! Both a cold phase (fresh kernel per rep — what time-to-first-token pays) and
//! a steady phase (warmed kernel) are reported, because a route that caches
//! moves cost between them; the fused route caches nothing, so its two columns
//! should agree.
//!
//! Run with:
//! ```text
//! cargo bench -p onnx-runtime-ep-cpu --bench int4_prefill_route_ab
//! ```
//! `PROBE_BITS=4|8` picks the weight width; `PROBE_SHAPE=small|big` picks one
//! shape; `PROBE_BLOCK=<n>` picks the quantization block size (32 by default;
//! use 16 to reach the `INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED` branch, where
//! the route below the crossover is the generic per-block dot); `PROBE_MS=prefill|cross` picks a row
//! sweep.
//!
//! Every row carries two output fingerprints, `sum` and `fnv`. `fnv` is the
//! one that matters: an FNV-1a fold over the raw output bytes, so it moves for
//! any bit that moves, at the position it moved. It exists because a
//! source-level A/B has to rebuild between arms, and a null taken that way is
//! uninterpretable without it -- a change that never executed and a change that
//! executed and cost nothing produce the same table, and no amount of timing
//! separates them. See `int4_modulo_matrix.py --route-proof`.

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

/// Route proof, not decoration.
///
/// A source-level A/B has to rebuild between arms, so "the arm I built is the
/// arm that ran" is an assumption rather than an observation, and a null is
/// uninterpretable without it: a change that never executed and a change that
/// executed and cost nothing produce the same table. The `fnv` column is an
/// FNV-1a fold over the raw output bytes, so it moves for *any* bit that
/// moves, at the position it moved -- unlike the float sum, which a
/// permutation can leave alone. A deliberately poisoned build is expected to
/// move it on exactly the rows whose route reaches the modified line, and to
/// leave every other row bit-identical. Those untouched rows are the control:
/// they show the poison is not simply changing the whole binary's behaviour.
///
/// The float sum is kept alongside it because it is comparable with the
/// decode-loop harness's `checksum=`, which is a plain sum.
fn checksum(out: &Tensor) -> (f64, u64) {
    let view = out.view();
    let len: usize = view.shape.iter().product();
    // `from_raw_parts` below trusts that `len` elements are contiguous behind
    // the pointer. That holds for a freshly allocated kernel output and fails
    // for a strided or sliced view, and the failure is a read past the
    // allocation rather than a wrong number, so it is checked rather than
    // assumed. Row-major contiguity is exactly `strides[i] == product of the
    // dimensions after i`.
    let mut want = 1i64;
    for (dim, stride) in view.shape.iter().zip(view.strides.iter()).rev() {
        assert_eq!(
            *stride, want,
            "checksum needs a contiguous f32 output; shape {:?} strides {:?} are not row-major",
            view.shape, view.strides
        );
        want *= *dim as i64;
    }
    assert_eq!(
        view.byte_offset, 0,
        "checksum does not handle a byte offset"
    );
    let values = unsafe { std::slice::from_raw_parts(view.data_ptr::<f32>(), len) };
    let sum = values.iter().map(|v| f64::from(*v)).sum();
    let mut fnv = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for byte in v.to_bits().to_le_bytes() {
            fnv ^= u64::from(byte);
            fnv = fnv.wrapping_mul(0x1_0000_01b3);
        }
    }
    (sum, fnv)
}

fn build_kernel(
    k: usize,
    n: usize,
    block_size: usize,
    m: usize,
    acc: i64,
    bits: i64,
) -> Box<dyn Kernel> {
    let blocks = k.div_ceil(block_size);
    let blob = block_size / (8 / bits as usize);
    let shapes = vec![vec![m, k], vec![n, blocks, blob], vec![n, blocks]];
    let mut node = Node::new(NodeId(0), "MatMulNBits", vec![], vec![]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("K", Attribute::Int(k as i64)),
        ("N", Attribute::Int(n as i64)),
        ("bits", Attribute::Int(bits)),
        ("block_size", Attribute::Int(block_size as i64)),
        ("accuracy_level", Attribute::Int(acc)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &shapes, 1)
        .expect("CPU EP must register MatMulNBits");
    kernel.set_constant_inputs(&[false, true, true]);
    kernel
}

fn main() {
    // Match the decode thread topology a served session runs in (#1749).
    common::init_decode_topology();
    // Opened before anything else runs, so the window covers warmup too: a
    // warmup that shared cores with somebody else's run leaves caches and
    // frequency in a state the timed region inherits.
    let host_lock = common::open_host_lock_window();

    // `PROBE_BLOCK` exists because `int4_prefill_gebp_min_rows` returns a
    // *different* threshold for weights whose block size the column-blocked
    // kernels cannot take (`INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED`), and with
    // the block size pinned to 32 this bench could not reach that branch at
    // all. Below 32 the competitor is not the vectorized column-blocked
    // kernel, it is the generic per-block dot, so the crossover is a genuinely
    // different measurement rather than the same one at another size.
    let block_size: usize = std::env::var("PROBE_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let shapes: Vec<(usize, usize)> = match std::env::var("PROBE_SHAPE").as_deref() {
        Ok("big") => vec![(4096, 11008)],
        Ok("small") => vec![(2048, 2048)],
        _ => vec![(2048, 2048), (4096, 11008)],
    };
    let acc: i64 = std::env::var("PROBE_ACC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // `PROBE_BITS=8` drives the same three routes over an 8-bit weight. The
    // 8-bit and 4-bit packs differ only in their unpack cost -- same panel,
    // same store pattern, same microkernel -- so running both is what
    // separates "the pack was unpack bound" from "the pack was store bound".
    let bits: i64 = match std::env::var("PROBE_BITS").as_deref() {
        Ok("8") => 8,
        _ => 4,
    };
    println!("acc_level={acc} bits={bits} block_size={block_size}");
    println!(
        "{:>6} {:>6} {:>5} {:>12} {:>12} {:>12} {:>16} {:>16}",
        "k", "n", "m", "cold_ms", "steady_ms", "gflops", "sum", "fnv"
    );
    for &(k, n) in shapes.iter() {
        let blocks = k.div_ceil(block_size);
        let blob = block_size / (8 / bits as usize);
        let packed = packed_bytes(n * blocks * blob, 7);
        let scales = floats(n * blocks, 0.3)
            .into_iter()
            .map(|v| v.abs().max(0.01) * 0.02)
            .collect::<Vec<_>>();
        let b = Tensor::u8(&[n, blocks, blob], &packed);
        let scales_t = Tensor::floats(common::FloatDType::F32, &[n, blocks], &scales);
        let ms: Vec<usize> = match std::env::var("PROBE_MS").as_deref() {
            Ok("prefill") => vec![8, 64, 256],
            Ok("cross") => vec![1, 2, 4, 8, 16, 32],
            _ => match std::env::var("PROBE_M_LIST") {
                // An explicit list, so the row thresholds in
                // `int4_prefill_gebp_min_rows` can be re-derived at whatever
                // `m` the crossover has moved to rather than only at the
                // powers of two the fixed sweeps happen to cover.
                Ok(list) => list
                    .split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect(),
                Err(_) => vec![1, 8, 64, 256, 512],
            },
        };
        for &m in &ms {
            let a = Tensor::floats(common::FloatDType::F32, &[m, k], &floats(m * k, 1.1));
            let mut out = Tensor::zeros(common::FloatDType::F32, &[m, n]);
            let ins = vec![a.view(), b.view(), scales_t.view()];

            let cold = median(
                (0..3)
                    .map(|_| {
                        let kernel = build_kernel(k, n, block_size, m, acc, bits);
                        let start = Instant::now();
                        kernel
                            .execute(&ins, &mut [out.view_mut()])
                            .expect("execute");
                        start.elapsed().as_secs_f64() * 1e3
                    })
                    .collect(),
            );

            let kernel = build_kernel(k, n, block_size, m, acc, bits);
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
            let gflops = (2.0 * m as f64 * k as f64 * n as f64) / (steady * 1e6);
            let (sum, fnv) = checksum(&out);
            println!(
                "{k:>6} {n:>6} {m:>5} {cold:>12.3} {steady:>12.3} {gflops:>12.2} {sum:>16.6} {fnv:016x}"
            );
        }
    }

    // Last, so the second reading covers everything above it.
    common::report_host_lock(host_lock);
}
