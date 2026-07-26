//! Cross-kernel fork/join occupancy probe for the persistent SPMD decode pool.
//!
//! Isolates the per-fork/join RAMP cost from in-kernel memory-bound compute by
//! streaming a FIXED total number of weight bytes through the persistent pool
//! while varying how many separate barrier dispatches (fork/joins) that same
//! total work is split across. With the total streamed bytes held constant, any
//! effective-bandwidth loss as the fork/join count rises is pure barrier /
//! wake / ramp overhead (cores idle at the barrier), not memory physics.
//!
//! It also reports per-worker BUSY occupancy: each worker accumulates the
//! wall-clock nanoseconds it spends inside its compute shard, so
//! `sum(busy) / (dispatch_wall * workers)` is the fraction of core-time doing
//! useful streaming vs spinning at the barrier.
//!
//! Run (forced pool, pinned to one socket):
//! ```text
//! taskset -c 0-31 ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1 \
//!   cargo run --release -p onnx-runtime-ep-cpu --example occupancy_probe
//! ```
//! Env knobs: `PROBE_THREADS` (workers), `PROBE_TOTAL_MB` (total streamed
//! bytes), `PROBE_COL_BYTES` (bytes streamed per output column = int4 K/2),
//! `PROBE_FORKS` (comma list of fork/join counts to sweep), `PROBE_ITERS`
//! (timed iterations per config, median reported).

use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use onnx_runtime_ep_cpu::decode_spmd::{build_from_env, SpmdDecodePools};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

/// Stream `col_bytes` of packed int4 weight for one output column, reducing it
/// with a tight, vectorizable accumulation so the loop is bounded by streaming
/// `weight` from memory (the M=1 GEMV memory roofline), not by ALU throughput.
/// `activation` is unused for the reduction value but kept in the signature so
/// callers pass the (cache-resident) activation row; the reduction is a plain
/// widening byte sum, which the autovectorizer turns into wide loads.
#[inline(always)]
fn stream_column(weight: &[u8], _activation: &[f32]) -> f32 {
    // Eight independent lanes so the reduction is latency-free and the loop is
    // purely load-bound (memory streaming), matching the int4 GEMV roofline.
    let mut lanes = [0u64; 8];
    let chunks = weight.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        for (lane, &byte) in lanes.iter_mut().zip(chunk) {
            *lane = lane.wrapping_add(byte as u64);
        }
    }
    let mut acc = lanes.iter().fold(0u64, |a, &l| a.wrapping_add(l));
    for &byte in remainder {
        acc = acc.wrapping_add(byte as u64);
    }
    acc as f32
}

/// One dispatch over `[col_lo, col_hi)` output columns; each worker streams its
/// contiguous shard of columns and records its busy nanoseconds.
fn run_fork(
    pool: &SpmdDecodePools,
    result: &mut [f32],
    weight: &[u8],
    activation: &[f32],
    col_bytes: usize,
    busy_ns: &[AtomicU64],
) {
    pool.dispatch_output_rows_indexed(result, 1, &|global_index, start, outputs| {
        let worker_start = Instant::now();
        for (offset, out) in outputs.iter_mut().enumerate() {
            let col = start + offset;
            let base = col * col_bytes;
            let slice = &weight[base..base + col_bytes];
            *out = stream_column(slice, activation);
        }
        busy_ns[global_index].fetch_add(worker_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    });
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn main() {
    // Force the persistent pool for this probe regardless of the ambient mode.
    unsafe {
        std::env::set_var("ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL", "1");
    }

    let threads = env_usize("PROBE_THREADS", 31);
    let total_mb = env_usize("PROBE_TOTAL_MB", 350);
    let col_bytes = env_usize("PROBE_COL_BYTES", 512); // int4, K=1024
    let iters = env_usize("PROBE_ITERS", 5);
    let forks: Vec<usize> = std::env::var("PROBE_FORKS")
        .unwrap_or_else(|_| "1,4,7,16,28,64,141,400".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let Some(pool) = build_from_env(Some(threads)) else {
        eprintln!("occupancy_probe: persistent pool did not build (need >1 allowed CPU); exiting");
        return;
    };
    let workers = pool.total_workers();

    let total_bytes = total_mb * 1024 * 1024;
    let n_cols = total_bytes / col_bytes;
    let total_bytes = n_cols * col_bytes; // exact
    let weight: Vec<u8> = (0..total_bytes).map(|i| (i * 131 + 7) as u8).collect();
    // A small activation row kept resident in cache (reused across columns).
    let activation: Vec<f32> = (0..4096).map(|i| (i as f32).sin()).collect();
    let mut result = vec![0.0f32; n_cols];

    println!(
        "occupancy_probe: workers={workers} nodes={} total={total_mb}MB col_bytes={col_bytes} \
         n_cols={n_cols} iters={iters}",
        pool.node_count()
    );

    // Hard global warmup: hammer the pool for ~1.5 s so every core has ramped to
    // its steady AVX turbo frequency and `weight` is fully faulted-in and warm in
    // the caches/TLB before ANY config is timed. Without this the first few
    // configs measure the frequency-ramp transient, not fork/join cost.
    {
        let busy_ns: Vec<AtomicU64> = (0..workers).map(|_| AtomicU64::new(0)).collect();
        let warm_start = Instant::now();
        while warm_start.elapsed().as_secs_f64() < 1.5 {
            run_fork(&pool, &mut result, &weight, &activation, col_bytes, &busy_ns);
        }
        black_box(&result);
    }

    println!(
        "{:>6}  {:>10}  {:>10}  {:>12}  {:>12}",
        "forks", "median_ms", "GB/s", "busy_occ_%", "extra_us/fj"
    );

    let mut base_ms = 0.0f64;
    for (idx, &f) in forks.iter().enumerate() {
        // Column boundaries splitting n_cols into f contiguous groups.
        let mut bounds = Vec::with_capacity(f + 1);
        for g in 0..=f {
            bounds.push(n_cols * g / f);
        }
        let busy_ns: Vec<AtomicU64> = (0..workers).map(|_| AtomicU64::new(0)).collect();

        // Per-config warmup pass (already globally warm; this only touches this
        // config's boundary layout).
        for w in 0..f {
            let (lo, hi) = (bounds[w], bounds[w + 1]);
            if hi > lo {
                run_fork(&pool, &mut result[lo..hi], &weight[lo * col_bytes..hi * col_bytes], &activation, col_bytes, &busy_ns);
            }
        }
        for b in &busy_ns {
            b.store(0, Ordering::Relaxed);
        }

        let mut samples = Vec::with_capacity(iters);
        let mut total_wall_s = 0.0f64;
        for _ in 0..iters {
            let start = Instant::now();
            for w in 0..f {
                let (lo, hi) = (bounds[w], bounds[w + 1]);
                if hi > lo {
                    run_fork(
                        &pool,
                        &mut result[lo..hi],
                        &weight[lo * col_bytes..hi * col_bytes],
                        &activation,
                        col_bytes,
                        &busy_ns,
                    );
                }
            }
            let iter_s = start.elapsed().as_secs_f64();
            samples.push(iter_s);
            total_wall_s += iter_s;
        }
        black_box(&result);

        let med_s = median(samples);
        let med_ms = med_s * 1000.0;
        let gbps = total_bytes as f64 / med_s / 1e9;
        // Busy occupancy accumulated across ALL timed iterations: fraction of
        // core-time spent inside compute shards vs spinning at the barrier.
        let total_busy_ns: u64 = busy_ns.iter().map(|b| b.load(Ordering::Relaxed)).sum();
        let busy_occ = total_busy_ns as f64 / (total_wall_s * 1e9 * workers as f64) * 100.0;
        if idx == 0 {
            base_ms = med_ms;
        }
        // Marginal wall-time cost of each extra fork/join over the sweep baseline
        // (first entry), at fixed total streamed bytes.
        let extra_us_per_fj = if f > forks[0] {
            (med_ms - base_ms) * 1000.0 / (f - forks[0]) as f64
        } else {
            0.0
        };
        println!(
            "{f:>6}  {med_ms:>10.3}  {gbps:>10.1}  {busy_occ:>12.1}  {extra_us_per_fj:>12.3}"
        );
    }

    pool.shutdown();
}
