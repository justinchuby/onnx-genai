//! Steady-state **decode-loop** A/B for the int4 `m = 1` route (#1565).
//!
//! `int4_prefill_route_ab` times one op in isolation. That is the wrong shape
//! of measurement for a decode question, and #1563 said so rather than acting
//! on its `m = 1` numbers: `quant_prefill_gebp` returns *before*
//! `with_decode_pool` is installed and drives the **global** pool, so at decode
//! it would fork the whole machine once per op per token. A single-op bench
//! with one session never pays for that -- there is nothing to contend with and
//! the fork cost is amortized over one measurement rather than over a token.
//!
//! So this drives a decode step's worth of projections back to back, for many
//! tokens, from `PROBE_SESSIONS` concurrent sessions:
//!
//! | arm | how |
//! |---|---|
//! | `m = 1` through the fused GEBP | default env, built with the row gates forced to 1 |
//! | today's decode routes | `ONNX_GENAI_CPU_MM_INT4_GEBP=0` |
//!
//! Both arms come from one binary. The second reproduces today's behaviour
//! exactly, because today no int4 prefill route is gated below `m = 2`.
//!
//! Env:
//! - `PROBE_BLOCK` -- quantization block size (default 32). 16 routes below the
//!   gate to `borrowed_affine_int4_matmul`; 32/64/128 route to
//!   `borrowed_affine_int4_matmul_nblock`, a different and much stronger
//!   competitor.
//! - `PROBE_SESSIONS` -- concurrent decode loops (default 1).
//! - `PROBE_TOKENS` -- measured tokens per session (default 64).
//! - `PROBE_LAYERS` -- projection chains per token (default 1).

mod common;

use std::time::Instant;

use common::Tensor;
use onnx_runtime_ep_api::{ExecutionProvider, Kernel};
use onnx_runtime_ep_cpu::{CpuExecutionProvider, with_decode_pool_scope};
use onnx_runtime_ir::{Attribute, Node, NodeId};

/// One decode step's projections for a llama3-8B-shaped model, as `(k, n)`.
/// A decode token pays all of these back to back, which is what makes the
/// per-op fork/join cost a per-token cost rather than a one-off.
const PROJECTIONS: &[(usize, usize, &str)] = &[
    (4096, 6144, "qkv"),
    (4096, 4096, "o"),
    (4096, 14336, "gate"),
    (4096, 14336, "up"),
    (14336, 4096, "down"),
];

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

fn build_kernel(k: usize, n: usize, block_size: usize) -> Box<dyn Kernel> {
    let blocks = k.div_ceil(block_size);
    let blob = block_size / 2;
    let shapes = vec![vec![1, k], vec![n, blocks, blob], vec![n, blocks]];
    let mut node = Node::new(NodeId(0), "MatMulNBits", vec![], vec![]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("K", Attribute::Int(k as i64)),
        ("N", Attribute::Int(n as i64)),
        ("bits", Attribute::Int(4)),
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

/// A weight, shared across sessions the way a served model's weights are.
struct Weight {
    b: Tensor,
    scales: Tensor,
    k: usize,
    n: usize,
}

fn main() {
    let block_size: usize = std::env::var("PROBE_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let sessions: usize = std::env::var("PROBE_SESSIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let tokens: usize = std::env::var("PROBE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let layers: usize = std::env::var("PROBE_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    // `native_decode/cpu.rs` passes the model's `uses_decode_pool`. A
    // block-quantized decoder sets it, so 1 is the default here; 0 measures the
    // dense-pool path the same code takes for other models.
    let spmd: bool = std::env::var("PROBE_SPMD")
        .map(|v| v != "0")
        .unwrap_or(true);

    let weights: Vec<Weight> = PROJECTIONS
        .iter()
        .enumerate()
        .map(|(index, &(k, n, _))| {
            let blocks = k.div_ceil(block_size);
            let blob = block_size / 2;
            let packed = packed_bytes(n * blocks * blob, 7 + index as u64);
            let scales = floats(n * blocks, 0.3)
                .into_iter()
                .map(|v| v.abs().max(0.01) * 0.02)
                .collect::<Vec<_>>();
            Weight {
                b: Tensor::u8(&[n, blocks, blob], &packed),
                scales: Tensor::floats(common::FloatDType::F32, &[n, blocks], &scales),
                k,
                n,
            }
        })
        .collect();

    println!(
        "block_size={block_size} sessions={sessions} tokens={tokens} layers={layers} spmd={spmd}"
    );
    println!(
        "{:>10} {:>12} {:>12} {:>14}",
        "phase", "ms_token", "ms_token_p90", "tokens_s_total"
    );

    // Each session owns its kernels and its activations, and shares the
    // weights, which is how a served model is actually laid out.
    let run_session = |warm: bool| -> Vec<f64> {
        let mut kernels: Vec<Box<dyn Kernel>> = Vec::new();
        let mut activations: Vec<Tensor> = Vec::new();
        let mut outputs: Vec<Tensor> = Vec::new();
        for _ in 0..layers {
            for weight in &weights {
                kernels.push(build_kernel(weight.k, weight.n, block_size));
                activations.push(Tensor::floats(
                    common::FloatDType::F32,
                    &[1, weight.k],
                    &floats(weight.k, 1.1),
                ));
                outputs.push(Tensor::zeros(common::FloatDType::F32, &[1, weight.n]));
            }
        }
        // `dyn Kernel` is not `Sync`, and `with_decode_pool_scope` may run its
        // closure on another thread, so the kernels are moved in and back out
        // each pass rather than borrowed. That is two `Vec` moves per token,
        // which is nothing beside a 100 MB weight read.
        let mut state = (kernels, outputs);
        let step = |(kernels, mut outputs): (Vec<Box<dyn Kernel>>, Vec<Tensor>),
                    activations: &[Tensor],
                    weights: &[Weight]|
         -> (Vec<Box<dyn Kernel>>, Vec<Tensor>) {
            for (index, kernel) in kernels.iter().enumerate() {
                let weight = &weights[index % weights.len()];
                let ins = vec![
                    activations[index].view(),
                    weight.b.view(),
                    weight.scales.view(),
                ];
                kernel
                    .execute(&ins, &mut [outputs[index].view_mut()])
                    .expect("execute");
            }
            (kernels, outputs)
        };
        if warm {
            for _ in 0..3 {
                state = step(state, &activations, &weights);
            }
        }
        // One sample per token: the whole projection chain, inside one
        // `with_decode_pool_scope`, which is exactly how
        // `native_decode/cpu.rs` drives a single-token forward. Getting this
        // wrong changes the answer rather than the precision: outside the scope
        // the GEBP arm forks the 32-wide global pool, inside it the decode pool
        // is already resident and the fan-out it partitions is that one.
        let mut samples = Vec::with_capacity(tokens);
        for _ in 0..tokens {
            let (acts, ws) = (&activations, &weights);
            let moved = state;
            let start = Instant::now();
            let (returned, elapsed) = with_decode_pool_scope(spmd, move || {
                let returned = step(moved, acts, ws);
                (returned, start.elapsed().as_secs_f64() * 1e3)
            });
            state = returned;
            samples.push(elapsed);
        }
        samples
    };

    for (phase, warm) in [("cold", false), ("steady", true)] {
        let started = Instant::now();
        let per_session: Vec<Vec<f64>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..sessions)
                .map(|_| scope.spawn(|| run_session(warm)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let wall = started.elapsed().as_secs_f64();

        let mut all: Vec<f64> = per_session.into_iter().flatten().collect();
        let ms = median(all.clone());
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90 = all[(all.len() * 9 / 10).min(all.len() - 1)];
        // Aggregate throughput is the number the pool question is really about:
        // a wider fork can cut one session's latency and still lose once
        // sessions have to share the machine.
        let total = (sessions * tokens) as f64 / wall;
        println!("{phase:>10} {ms:>12.3} {p90:>12.3} {total:>14.1}");
    }
}
