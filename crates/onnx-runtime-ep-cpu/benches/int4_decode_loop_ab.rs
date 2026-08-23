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
//! - `PROBE_ACCURACY` -- `accuracy_level` (default 0). **4 is the only value
//!   that reaches the packed-nibble kernel**, so without this axis that route
//!   had no decode-loop row at all and only ever appeared in single-op benches.
//! - `PROBE_SESSIONS` -- concurrent decode loops (default 1).
//! - `PROBE_TOKENS` -- measured tokens per session (default 64).
//! - `PROBE_LAYERS` -- projection chains per token (default 1).
//! - `PROBE_ZERO_POINTS` -- `1` supplies a fourth (asymmetric) zero-points
//!   input; the default `0` leaves it absent, which is the **symmetric**
//!   route. This axis matters more than it looks: symmetric int4 takes the
//!   implicit midpoint 8 and never touches a zero-points byte, so any
//!   measurement of zero-point unpacking taken with the default is measuring
//!   a branch that is never entered.
//!
//! To vary the decode pool width, set **`ONNX_GENAI_CPU_DECODE_THREADS`**.
//! `RAYON_NUM_THREADS` does *not* size this pool -- `configured_decode_threads`
//! reads `available_parallelism` and `ONNX_GENAI_CPU_DECODE_THREADS` only. A
//! sweep of `RAYON_NUM_THREADS` therefore holds the width fixed while appearing
//! to vary it, and reports a flat line that reads exactly like "this kernel
//! does not scale". It does scale: measured here at block 32, accuracy 4,
//! 8 threads -> 5.90 ms/token and 16 -> 3.32 ms/token (1.77x, +-0.7% over
//! three interleaved repetitions). The default width already resolves to 16 on
//! a 32-vCPU host, for both the persistent and the flat pool.

//! # What `tokens_s_total` means (#1712)
//!
//! Stated explicitly because it was previously **not the same quantity** as the
//! ORT baseline it was being divided by, and the mismatch was large enough to
//! invent a result. The definition here, and in
//! `ort_matmulnbits_baseline.py`, is now identical in all four respects:
//!
//! | | definition |
//! |---|---|
//! | numerator | `sessions * tokens` -- every measured token from every session |
//! | denominator | wall-clock seconds from the **barrier release** to the last session's join |
//! | warmup | 3 steps per session, completed *before* the barrier, never inside the clock |
//! | over repetitions | **median**, never min or max |
//!
//! Each of those four was previously wrong on at least one side:
//!
//! * The native denominator used to include thread spawn and the three warmup
//!   steps. At `tokens = 24` that charged 27 steps of work against 24 counted
//!   tokens -- a flat ~11% penalty the ORT arm never paid.
//! * There was no barrier, so sessions started staggered and `wall` absorbed
//!   the ramp. ORT has used a `threading.Barrier` throughout.
//! * ORT reported `min` (single-session) or `max` (concurrent) over
//!   repetitions -- the luckiest run -- against a single native shot.
//! * Worst: ORT used **two different statistics either side of `sessions = 1`**.
//!   At `sessions = 1` it reported `1000 / median_ms_per_token`, which excludes
//!   every straggler; at `sessions >= 2` it reported wall-clock aggregate,
//!   which includes them. A baseline that switches from a best-case to a
//!   realistic statistic at `sessions = 2` will always make its opponent look
//!   worst at `sessions = 1`, which is exactly the shape the "the gap is
//!   concurrency-dependent" reading was built on.
//!
//! `spread_%` is `(max - min) / median` across repetitions and is printed so a
//! cell whose noise exceeds its effect cannot be quoted without that being
//! visible in the same row.

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

/// Qwen2.5-7B's decode projections. Not a cosmetic second model: its GQA head
/// layout makes `qkv` **narrow** (n = 4608 against a k of 3584) and its MLP
/// **much wider** relative to the hidden size (18944 vs llama's 14336 on a
/// larger k). Both differences move the `n`-loop trip count, which is exactly
/// the axis the N-blocked kernel's four-column grouping divides. A conclusion
/// drawn only from llama shapes has not been tested against a different
/// n/k ratio at all, and the tail behaviour at `n % 4` is invisible in a set
/// where every `n` is a multiple of 4.
const PROJECTIONS_QWEN: &[(usize, usize, &str)] = &[
    (3584, 4608, "qkv"),
    (3584, 3584, "o"),
    (3584, 18944, "gate"),
    (3584, 18944, "up"),
    (18944, 3584, "down"),
];

fn projections() -> &'static [(usize, usize, &'static str)] {
    match std::env::var("PROBE_MODEL")
        .unwrap_or_else(|_| "llama".into())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "qwen" => PROJECTIONS_QWEN,
        "llama" => PROJECTIONS,
        other => panic!("PROBE_MODEL must be llama or qwen, got {other:?}"),
    }
}

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

fn build_kernel(
    k: usize,
    n: usize,
    block_size: usize,
    accuracy: i64,
    zero_points: bool,
) -> Box<dyn Kernel> {
    let blocks = k.div_ceil(block_size);
    let blob = block_size / 2;
    let mut shapes = vec![vec![1, k], vec![n, blocks, blob], vec![n, blocks]];
    if zero_points {
        shapes.push(vec![n, blocks.div_ceil(2)]);
    }
    let mut node = Node::new(NodeId(0), "MatMulNBits", vec![], vec![]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("K", Attribute::Int(k as i64)),
        ("N", Attribute::Int(n as i64)),
        ("bits", Attribute::Int(4)),
        ("block_size", Attribute::Int(block_size as i64)),
        ("accuracy_level", Attribute::Int(accuracy)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &shapes, 1)
        .expect("CPU EP must register MatMulNBits");
    if zero_points {
        kernel.set_constant_inputs(&[false, true, true, true]);
    } else {
        kernel.set_constant_inputs(&[false, true, true]);
    }
    kernel
}

/// Packed asymmetric zero points, two int4 nibbles per byte, one per block.
///
/// Values sit near the symmetric midpoint 8 the way a real round-to-nearest
/// quantizer's do; the exact values do not change the instruction count, only
/// the arithmetic they feed.
fn packed_zero_points(blocks: usize, n: usize) -> Vec<u8> {
    let per_row = blocks.div_ceil(2);
    let mut bytes = vec![0u8; n * per_row];
    for row in 0..n {
        for block in 0..blocks {
            let value = 7u8 + ((row + block) % 3) as u8;
            let byte = &mut bytes[row * per_row + block / 2];
            *byte |= value << ((block % 2) * 4);
        }
    }
    bytes
}

/// A weight, shared across sessions the way a served model's weights are.
struct Weight {
    b: Tensor,
    scales: Tensor,
    zero_points: Option<Tensor>,
    k: usize,
    n: usize,
}

/// Set by every session; read once after the measured phase. See the checksum
/// comment in `run_session`.
static CHECKSUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn main() {
    // Match the decode thread topology a served session runs in (#1749).
    common::init_decode_topology();

    let block_size: usize = std::env::var("PROBE_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let accuracy: i64 = std::env::var("PROBE_ACCURACY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let sessions: usize = std::env::var("PROBE_SESSIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let tokens: usize = std::env::var("PROBE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let asymmetric: bool = std::env::var("PROBE_ZERO_POINTS")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);
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
    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    let weights: Vec<Weight> = projections()
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
                zero_points: asymmetric
                    .then(|| Tensor::u8(&[n, blocks.div_ceil(2)], &packed_zero_points(blocks, n))),
                k,
                n,
            }
        })
        .collect();

    println!(
        "model={} block_size={block_size} accuracy={accuracy} sessions={sessions} tokens={tokens} layers={layers} spmd={spmd} zero_points={}",
        std::env::var("PROBE_MODEL").unwrap_or_else(|_| "llama".into()),
        if asymmetric {
            "asymmetric"
        } else {
            "symmetric"
        }
    );
    println!(
        "{:>10} {:>12} {:>12} {:>14} {:>9}",
        "phase", "ms_token", "ms_token_p90", "tokens_s_total", "spread_%"
    );

    // Each session owns its kernels and its activations, and shares the
    // weights, which is how a served model is actually laid out.
    let run_session = |warm: bool, barrier: &std::sync::Barrier| -> Vec<f64> {
        let mut kernels: Vec<Box<dyn Kernel>> = Vec::new();
        let mut activations: Vec<Tensor> = Vec::new();
        let mut outputs: Vec<Tensor> = Vec::new();
        for _ in 0..layers {
            for weight in &weights {
                kernels.push(build_kernel(
                    weight.k, weight.n, block_size, accuracy, asymmetric,
                ));
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
                let mut ins = vec![
                    activations[index].view(),
                    weight.b.view(),
                    weight.scales.view(),
                ];
                if let Some(zero_points) = weight.zero_points.as_ref() {
                    ins.push(zero_points.view());
                }
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
        // The barrier is released only after every session has finished its
        // warmup, so `wall` on the driving thread covers the measured tokens
        // and nothing else. Without it, `wall` also contained thread spawn and
        // three warmup steps -- with `tokens = 24` that is 27 steps of work
        // charged against 24 counted tokens, a flat ~11% penalty that the ORT
        // arm never paid because its warmup runs before its clock starts.
        barrier.wait();
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
        // Route proof, not decoration. A benchmark arm that supplies a fourth
        // input the kernel silently ignores would time the *symmetric* route
        // while claiming to time the asymmetric one, and every number taken
        // from it would be attributed to a branch never entered. The checksum
        // is the cheapest evidence the zero points were consumed: symmetric
        // uses the implicit midpoint 8, asymmetric uses 7/8/9, so the two
        // arms cannot agree unless the input was dropped.
        let checksum: f64 = state
            .1
            .iter()
            .map(|out| {
                let view = out.view();
                let len: usize = view.shape.iter().product();
                let values = unsafe { std::slice::from_raw_parts(view.data_ptr::<f32>(), len) };
                values.iter().map(|v| *v as f64).sum::<f64>()
            })
            .sum();
        CHECKSUM.store(checksum.to_bits(), std::sync::atomic::Ordering::Relaxed);
        samples
    };

    // One repetition of one phase. Returns `(median_ms_token, p90, tokens_s_total)`.
    let measure = |warm: bool| -> (f64, f64, f64) {
        let barrier = std::sync::Barrier::new(sessions + 1);
        let (per_session, wall) = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..sessions)
                .map(|_| {
                    let barrier = &barrier;
                    scope.spawn(move || run_session(warm, barrier))
                })
                .collect();
            // Releases every session at once, then starts the clock. All
            // warmup and allocation is already behind us at this point.
            barrier.wait();
            let start = Instant::now();
            let out: Vec<Vec<f64>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            (out, start.elapsed().as_secs_f64())
        });

        let mut all: Vec<f64> = per_session.into_iter().flatten().collect();
        let ms = median(all.clone());
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90 = all[(all.len() * 9 / 10).min(all.len() - 1)];
        // Aggregate throughput is the number the pool question is really about:
        // a wider fork can cut one session's latency and still lose once
        // sessions have to share the machine.
        (ms, p90, (sessions * tokens) as f64 / wall)
    };

    for (phase, warm) in [("cold", false), ("steady", true)] {
        // `cold` is inherently single-shot: repeating it would measure a warm
        // run. Only `steady` is repeated.
        let n = if warm { reps } else { 1 };
        let mut rows: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
        for _ in 0..n {
            rows.push(measure(warm));
        }
        let ms = median(rows.iter().map(|r| r.0).collect());
        let p90 = median(rows.iter().map(|r| r.1).collect());
        // MEDIAN over repetitions, deliberately not min/max. Reporting the
        // best repetition makes every arm look like its luckiest run and makes
        // the spread invisible, which is how a 2x measurement artifact can
        // survive review (see the ratio-definition note in the module docs).
        let mut tps: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let total = median(tps.clone());
        tps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let spread = if total > 0.0 {
            (tps[tps.len() - 1] - tps[0]) / total * 100.0
        } else {
            0.0
        };
        println!("{phase:>10} {ms:>12.3} {p90:>12.3} {total:>14.1} {spread:>9.1}");
    }

    println!(
        "checksum={:.6}",
        f64::from_bits(CHECKSUM.load(std::sync::atomic::Ordering::Relaxed))
    );

    // After the phases, never before: the pool is built at first decode, so this
    // is the earliest point the realized width exists to be read.
    common::report_decode_width();
}
