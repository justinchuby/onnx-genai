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
//! Both arms come from one binary.
//!
//! **Stale claim, corrected 2026-08-23 (#1783).** This used to say the second
//! arm "reproduces today's behaviour exactly, because today no int4 prefill
//! route is gated below `m = 2`". That is no longer true:
//! `INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED` is **1**, so for any block size that
//! is not a multiple of 32 the GEBP gate `m >= 1` admits a *decode* row. At
//! those block sizes the default arm is GEBP and the two arms differ.
//!
//! Env:
//! - `PROBE_BLOCK` -- quantization block size (default 32). 32/64/128 route to
//!   `borrowed_affine_int4_matmul_nblock`, the N-blocked decode kernel.
//!
//!   **`16` does not measure a decode kernel at all, and this line used to say
//!   it routed to `borrowed_affine_int4_matmul`.** It does not.
//!   `int4_prefill_gebp_min_rows` returns
//!   `INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED == 1` for any block size that is
//!   not a multiple of 32, so at `m = 1` the GEBP gate is already satisfied
//!   and block 16 runs the *fused prefill* kernel `quant_prefill_gebp`.
//!   Falsifier: `ONNX_GENAI_CPU_MM_INT4_GEBP=0` changes the block-16 checksum
//!   (844.536810 -> 844.551163) and its time, and leaves block 32 untouched
//!   (978.949310 either way). Any block-16 row taken here is a GEBP row.
//!   These constants are numerics-sensitive and drift whenever a kernel
//!   reassociates its reduction (#1667, #1783 both moved them in the fourth
//!   decimal place); it is the *pattern* -- block 16 moves, block 32 does not
//!   -- that is the route evidence, so re-derive them rather than reading a
//!   mismatch as a route change.
//!   Set `ONNX_GENAI_CPU_MM_INT4_GEBP=0` to reach the decode kernel at 16.
//! - `PROBE_ACCURACY` -- `accuracy_level` (default 0). **4 is the only value
//!   that reaches the packed-nibble kernel**, so without this axis that route
//!   had no decode-loop row at all and only ever appeared in single-op benches.
//! - `PROBE_SESSIONS` -- concurrent decode loops (default 1).
//! - `PROBE_TOKENS` -- measured tokens per session (default 64).
//! - `PROBE_LAYERS` -- projection chains per token (default 1).
//!
//! To vary the decode pool width, set **`ONNX_GENAI_CPU_DECODE_THREADS`**.
//! `RAYON_NUM_THREADS` does *not* size this pool -- `configured_decode_threads`
//! reads `available_parallelism` and `ONNX_GENAI_CPU_DECODE_THREADS` only. A
//! sweep of `RAYON_NUM_THREADS` therefore holds the width fixed while appearing
//! to vary it, and reports a flat line that reads exactly like "this kernel
//! does not scale". It does scale. The width curve once quoted here
//! (`8 threads -> 5.90 ms/token and 16 -> 3.32 ms/token, 1.77x +-0.7% over
//! three interleaved repetitions`) has been **withdrawn (2026-08-23)**: on a
//! shared host `w=16` occupies every physical core and is bimodal, so no
//! narrow interval is supportable there. Measure at `w=8`, which has headroom
//! and held a 9.8% spread over six independent launches against `w=16`'s 514%.
//! See `docs/benchmarks/2026-08-23-acc4-decode-width-remeasurement.md`. The
//! default width already resolves to 16 on a 32-vCPU host, for both the
//! persistent and the flat pool.

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
use common::decode_workload::{Weight, asymmetric_zero_points, build_kernel, floats, weights};
use onnx_runtime_ep_api::Kernel;
use onnx_runtime_ep_cpu::with_decode_pool_scope;

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Set by every session; read once after the measured phases. See the checksum
/// comment in the session loop.
static CHECKSUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One repetition of one phase.
struct Rep {
    ms: f64,
    p90: f64,
    tps: f64,
    wall: f64,
    cpu: Option<common::CpuTime>,
}

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

    let asymmetric = asymmetric_zero_points();
    let weights: Vec<Weight> = weights(block_size, asymmetric);

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
                let ins = weight.inputs(&activations[index]);
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
        // Written by every session, last writer wins. That is well defined
        // rather than racy in substance: the sessions share one deterministic
        // weight set and one deterministic activation, so every session
        // computes the same sum and any of them is the right answer. It is an
        // atomic store read after the scope joins, so there is no UB either.
        //
        // Route proof, not decoration. An arm that supplies a fourth input the
        // kernel silently ignored would time the *symmetric* route while
        // claiming the asymmetric one, and every number taken from it would be
        // attributed to a branch never entered. The checksum is the cheapest
        // evidence the zero points were consumed: symmetric uses the implicit
        // midpoint 8 and asymmetric uses 7/8/9, so the two arms cannot agree
        // unless the input was dropped.
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

    // One repetition of one phase.
    let measure = |warm: bool| -> Rep {
        let barrier = std::sync::Barrier::new(sessions + 1);
        let (per_session, wall, cpu) = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..sessions)
                .map(|_| {
                    let barrier = &barrier;
                    scope.spawn(move || run_session(warm, barrier))
                })
                .collect();
            // Releases every session at once, then starts the clock. All
            // warmup and allocation is already behind us at this point.
            barrier.wait();
            // Bracketed by exactly the same two points as `wall`, so
            // `cpu / (wall * width)` is a busy fraction over the measured
            // window and nothing else. Read after the barrier so pool
            // construction and the three warmup steps are outside it.
            let cpu0 = common::process_cpu_time();
            let start = Instant::now();
            let out: Vec<Vec<f64>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            let elapsed = start.elapsed().as_secs_f64();
            let cpu = match (cpu0, common::process_cpu_time()) {
                (Some(a), Some(b)) => Some(b.since(a)),
                _ => None,
            };
            (out, elapsed, cpu)
        });

        let mut all: Vec<f64> = per_session.into_iter().flatten().collect();
        let ms = median(all.clone());
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90 = all[(all.len() * 9 / 10).min(all.len() - 1)];
        // Aggregate throughput is the number the pool question is really about:
        // a wider fork can cut one session's latency and still lose once
        // sessions have to share the machine.
        Rep {
            ms,
            p90,
            tps: (sessions * tokens) as f64 / wall,
            wall,
            cpu,
        }
    };

    for (phase, warm) in [("cold", false), ("steady", true)] {
        // `cold` is inherently single-shot: repeating it would measure a warm
        // run. Only `steady` is repeated.
        let n = if warm { reps } else { 1 };
        let mut rows: Vec<Rep> = Vec::with_capacity(n);
        for _ in 0..n {
            rows.push(measure(warm));
        }
        let ms = median(rows.iter().map(|r| r.ms).collect());
        let p90 = median(rows.iter().map(|r| r.p90).collect());
        // MEDIAN over repetitions, deliberately not min/max. Reporting the
        // best repetition makes every arm look like its luckiest run and makes
        // the spread invisible, which is how a 2x measurement artifact can
        // survive review (see the ratio-definition note in the module docs).
        let mut tps: Vec<f64> = rows.iter().map(|r| r.tps).collect();
        let total = median(tps.clone());
        tps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let spread = if total > 0.0 {
            (tps[tps.len() - 1] - tps[0]) / total * 100.0
        } else {
            0.0
        };
        println!("{phase:>10} {ms:>12.3} {p90:>12.3} {total:>14.1} {spread:>9.1}");

        // A separate line, and deliberately not prefixed with the phase name:
        // the existing matrix scripts key the throughput row on a leading
        // `steady`, and a second line starting the same way would be parsed as
        // one and fail on the first field.
        //
        // Every field comes from ONE repetition -- the one whose throughput is
        // the median, which is exactly the repetition the `tps` column above
        // reports, since `median` here returns `sorted[len / 2]` rather than
        // interpolating. Taking an independent median per quantity looks
        // equivalent and is not: `tps = tokens / wall`, so sorting by `tps`
        // ascending and sorting by `wall` ascending are *reversed* orders, and
        // at an even repetition count the two medians land on different
        // repetitions. The resulting row describes no run that ever happened,
        // and `cpu / (wall * width)` inherits the full rep-to-rep spread as
        // bias -- measured at 4.4% to 29.7% against an identity that is 0.00%
        // when the row is self-consistent.
        //
        // `user` and `sys` are reported separately rather than summed because
        // they distinguish work from waiting: `sched_yield` is charged to `sys`.
        if rows.iter().all(|r| r.cpu.is_some()) {
            let mut by_tps: Vec<usize> = (0..rows.len()).collect();
            by_tps.sort_by(|&a, &b| rows[a].tps.partial_cmp(&rows[b].tps).unwrap());
            let rep = &rows[by_tps[by_tps.len() / 2]];
            let cpu = rep.cpu.unwrap();
            let counted = (sessions * tokens) as f64;
            let cpu_s = cpu.total_s();
            println!(
                "cpu phase={phase} user_s={:.4} sys_s={:.4} cpu_s={cpu_s:.4} \
                 wall_s={:.4} tokens={counted:.0} tps_rep={:.4} \
                 cpu_s_per_token={:.6} sys_frac={:.4}",
                cpu.user_s,
                cpu.sys_s,
                rep.wall,
                rep.tps,
                cpu_s / counted,
                if cpu_s > 0.0 { cpu.sys_s / cpu_s } else { 0.0 },
            );
        } else {
            println!("cpu phase={phase} unavailable");
        }
    }

    println!(
        "checksum={:.6}",
        f64::from_bits(CHECKSUM.load(std::sync::atomic::Ordering::Relaxed))
    );

    // After the phases, never before: the pool is built at first decode, so this
    // is the earliest point the realized width exists to be read.
    common::report_decode_width();
}
