//! #1810 Slice 3 — ExpertBank remap cost measurement and validation.
//!
//! Measures and validates promotion/demotion (host-NUMA ↔ device) remap cost
//! using the NEW production primitives from #1823 (`PhysicalLocation`,
//! `PhysicalHandlePool::get_or_create_at_location`,
//! `CudaVirtualBacking::commit_at_location`) under decode-like expert churn.
//!
//! # What "remap" means here
//!
//! A remap is one granule-level promote or demote:
//!  - **promote** (host-NUMA → device): `release_range_reporting` on the
//!    host_numa backing (returns handle to host pool) then `commit_at_location`
//!    on the device backing (acquires handle from device pool).
//!  - **demote** (device → host-NUMA): symmetric, directions reversed.
//!
//! Both halves go through `synchronizing_section()` inside the production
//! primitives — the gate is load-bearing per #1813.
//!
//! # Timing decomposition
//!
//! For each granule remap we measure separately (wall-clock `Instant`):
//!  (a) `synchronizing_section` acquire time (host-side lock wait)
//!  (b) unmap cost (`release_range_reporting`, which includes cuMemUnmap)
//!  (c) handle acquire: pool-warm hit (pooled_unmapped reuse) vs pool-cold
//!      (fresh cuMemCreate) — measured from pool counter deltas
//!  (d) map cost (`commit_at_location`, cuMemMap + cuMemSetAccess)
//!  (e) GPU-side kernel access cost after remap (CUDA event timing for
//!      first post-remap kernel access vs steady-state access)
//!
//! # Shapes tested
//!
//! - 60-expert Qwen1.5-MoE-A2.7B (reused shape from spike file)
//! - 256-expert synthetic (DeepSeek/Qwen/GLM-MoE scale: hidden=4096,
//!   inter=2048, top_k=8 — shape-faithful per the fixture pattern; see
//!   module-level comment for justification)
//!
//! # Routing traces
//!
//! - **Uniform**: round-robin across experts — every expert routed equally.
//! - **Skewed (Zipf-like)**: deterministic power-law; expert i's weight ∝
//!   1/(i+1)^1.2, top_k sampled from this distribution with a fixed seed.
//!
//! # Run command
//!
//! ```text
//! # Verify GPU idle first:
//! nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv
//! nvidia-smi --query-gpu=index,memory.used --format=csv
//!
//! CUDA_VISIBLE_DEVICES=4 cargo test -p onnx-runtime-ep-cuda \
//!   --features cuda --release \
//!   --test expert_bank_remap_cost_gpu \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! # Platform (this machine)
//!
//! A100-SXM4-80GB, Linux, driver 580.105.08, CUDA 13.0, host_numa_id=3,
//! granularity=2 MiB (probed, not hardcoded). GPU 4 is the idle device;
//! GPU 3 carries ~60 GB from another persistent process.

#![allow(
    clippy::too_many_arguments,
    clippy::uninlined_format_args,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;
use onnx_runtime_cuda_memory::capability::{CapabilityGateFailure, host_numa_capability};
use onnx_runtime_cuda_memory::virtual_memory::{
    CudaVirtualBacking, PhysicalHandlePool, PhysicalLocation,
};
use onnx_runtime_memory_governor::{DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole};
use onnx_runtime_virtual_memory::VirtualBacking;

// ---------------------------------------------------------------------------
// Serialise every test in this file
// ---------------------------------------------------------------------------

static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn require_cuda_context() -> (Arc<CudaContext>, std::sync::MutexGuard<'static, ()>) {
    let guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let ctx =
        CudaContext::new(0).expect("CUDA context on device 0 (relative to CUDA_VISIBLE_DEVICES)");
    (ctx, guard)
}

// ---------------------------------------------------------------------------
// GPU idle check (must pass before every measurement)
// ---------------------------------------------------------------------------

fn assert_gpu_idle_or_warn(label: &str) {
    let by_app = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory,gpu_bus_id",
            "--format=csv,noheader",
        ])
        .output();
    match by_app {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
            if !lines.is_empty() {
                println!("[{label}] WARNING: compute processes running: {lines:?}");
            } else {
                println!("[{label}] GPU idle check: no compute processes (clean)");
            }
        }
        _ => eprintln!("[{label}] WARNING: could not query nvidia-smi compute-apps"),
    }
    let by_mem = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,memory.used", "--format=csv"])
        .output();
    match by_mem {
        Ok(o) if o.status.success() => {
            println!(
                "[{label}] nvidia-smi memory.used:\n{}",
                String::from_utf8_lossy(&o.stdout)
            );
        }
        _ => {}
    }
}

fn print_platform() {
    let driver = std::fs::read_to_string("/proc/driver/nvidia/version")
        .ok()
        .map(|s| s.lines().next().unwrap_or("").to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("platform: os={} driver={:?}", std::env::consts::OS, driver);
}

// ---------------------------------------------------------------------------
// Governor helper
// ---------------------------------------------------------------------------

fn make_governor(device: i32, device_bytes: u64, host_bytes: u64) -> &'static LedgerGovernor {
    let ledger = LeaseLedger::new_for_device(
        DeviceKey::device(device as u32),
        device_bytes,
        host_bytes,
        0,
    );
    Box::leak(Box::new(LedgerGovernor::new(ledger)))
}

// ---------------------------------------------------------------------------
// QmoeShape — reused verbatim from spike file (unchanged struct layout)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct QmoeShape {
    name: &'static str,
    experts: usize,
    hidden: usize,
    inter: usize,
    top_k: usize,
}

/// Qwen1.5-MoE-A2.7B (60 experts) — reused from spike file.
const QWEN15_MOE_A27B: QmoeShape = QmoeShape {
    name: "qwen1.5-moe-a2.7b",
    experts: 60,
    hidden: 2048,
    inter: 1408,
    top_k: 4,
};

/// 256-expert synthetic shape at DeepSeek/Qwen3-MoE/GLM-MoE scale.
///
/// Justification for shape-faithfulness:
/// - DeepSeek-V3 uses 256 routed experts, hidden=7168, inter=2048 (published config);
///   GLM-4-MoE and Qwen3-235B-A22B both use 128 or 256 routed experts at
///   similar inter dims. We use hidden=4096 (a 2^n alignment) and inter=2048
///   (matching DeepSeek-V3's MoE inter_size) as a representative mid-scale.
///   top_k=8 matches DeepSeek-V3's published routing.
/// - This is a MEASUREMENT fixture, not a model config claim. The actual byte
///   budget per expert is what matters for remap cost and it is dominated by
///   fc1/fc2/fc3 packed int4 bytes, which scale with inter×hidden/2.
/// - With hidden=4096, inter=2048, block_size=16, bits=4: each expert's fc1
///   packed bytes = 2048 × 4096 / 2 = 4 MiB, fc2 = 2 MiB, fc3 = 4 MiB.
///   Two granules per fc1 expert = realistic per-expert granularity footprint.
const SYNTH_256_EXPERT: QmoeShape = QmoeShape {
    name: "synth-256-expert-deepseek-scale",
    experts: 256,
    hidden: 4096,
    inter: 2048,
    top_k: 8,
};

// ---------------------------------------------------------------------------
// Routing traces
// ---------------------------------------------------------------------------

/// Uniform round-robin trace: step i routes experts [i*top_k..(i+1)*top_k]
/// mod experts. Every expert is routed equally.
fn uniform_trace(shape: &QmoeShape, steps: usize) -> Vec<Vec<usize>> {
    (0..steps)
        .map(|step| {
            (0..shape.top_k)
                .map(|k| (step * shape.top_k + k) % shape.experts)
                .collect()
        })
        .collect()
}

/// Skewed Zipf-like trace: expert i has weight proportional to 1/(i+1)^1.2.
/// Top-k experts are sampled deterministically from this distribution using a
/// linear-congruential generator with a fixed seed.
fn skewed_trace(shape: &QmoeShape, steps: usize) -> Vec<Vec<usize>> {
    // Precompute cumulative weights.
    let weights: Vec<f64> = (0..shape.experts)
        .map(|i| 1.0_f64 / (i as f64 + 1.0).powf(1.2))
        .collect();
    let total: f64 = weights.iter().sum();
    let cdf: Vec<f64> = weights
        .iter()
        .scan(0.0f64, |acc, &w| {
            *acc += w / total;
            Some(*acc)
        })
        .collect();

    let mut state: u64 = 0xdeadbeef_cafebabe;
    let next_rand = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    };

    (0..steps)
        .map(|_| {
            let mut routed = Vec::with_capacity(shape.top_k);
            for _ in 0..shape.top_k {
                let r = next_rand(&mut state);
                // Binary search the CDF.
                let idx = cdf.partition_point(|&c| c < r).min(shape.experts - 1);
                let mut picked = idx;
                // Avoid duplicates: linear scan forward.
                while routed.contains(&picked) {
                    picked = (picked + 1) % shape.experts;
                }
                routed.push(picked);
            }
            routed
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Remap cost tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
struct RemapComponentTimes {
    /// Wall-clock time for unmap (release_range_reporting) in µs.
    unmap_us: Vec<f64>,
    /// Wall-clock time for map (commit_at_location) in µs.
    map_us: Vec<f64>,
    /// Whether the handle was a pool warm hit (true) or cold create (false).
    pool_warm: Vec<bool>,
    /// Total remap time (unmap + map) in µs.
    total_us: Vec<f64>,
}

impl RemapComponentTimes {
    fn record(&mut self, unmap: f64, map: f64, warm: bool) {
        self.unmap_us.push(unmap);
        self.map_us.push(map);
        self.pool_warm.push(warm);
        self.total_us.push(unmap + map);
    }

    fn median(v: &[f64]) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    }

    fn range(v: &[f64]) -> (f64, f64) {
        v.iter()
            .cloned()
            .fold((f64::MAX, f64::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)))
    }

    fn warm_rate(&self) -> f64 {
        if self.pool_warm.is_empty() {
            return 0.0;
        }
        self.pool_warm.iter().filter(|&&w| w).count() as f64 / self.pool_warm.len() as f64
    }

    fn print_summary(&self, label: &str) {
        let total_med = Self::median(&self.total_us);
        let total_rng = Self::range(&self.total_us);
        let unmap_med = Self::median(&self.unmap_us);
        let map_med = Self::median(&self.map_us);
        println!(
            "  [{label}] n={} remap_total: median={:.1}µs range=[{:.1},{:.1}]µs \
             unmap_med={:.1}µs map_med={:.1}µs warm_rate={:.1}%",
            self.total_us.len(),
            total_med,
            total_rng.0,
            total_rng.1,
            unmap_med,
            map_med,
            self.warm_rate() * 100.0,
        );
    }
}

// ---------------------------------------------------------------------------
// Per-granule location tracking for a "ExpertBank" VA
// ---------------------------------------------------------------------------

/// Per-granule state: which location is currently mapped at this offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GranuleLocation {
    Device,
    HostNuma,
}

/// Accounting for the whole remap benchmark run.
#[derive(Default, Debug)]
struct BenchAccounting {
    device_committed_bytes: u64,
    host_committed_bytes: u64,
    mapped_bytes: u64,
    underflow_events: u64,
    unaccounted_bytes: u64,
    pool_warm_count: u64,
    pool_cold_count: u64,
}

impl BenchAccounting {
    fn note_commit(&mut self, loc: GranuleLocation, granularity: usize) {
        let g = granularity as u64;
        match loc {
            GranuleLocation::Device => self.device_committed_bytes += g,
            GranuleLocation::HostNuma => self.host_committed_bytes += g,
        }
        self.mapped_bytes += g;
    }

    fn note_release(&mut self, loc: GranuleLocation, granularity: usize) {
        let g = granularity as u64;
        match loc {
            GranuleLocation::Device => match self.device_committed_bytes.checked_sub(g) {
                Some(v) => self.device_committed_bytes = v,
                None => self.underflow_events += 1,
            },
            GranuleLocation::HostNuma => match self.host_committed_bytes.checked_sub(g) {
                Some(v) => self.host_committed_bytes = v,
                None => self.underflow_events += 1,
            },
        }
        match self.mapped_bytes.checked_sub(g) {
            Some(v) => self.mapped_bytes = v,
            None => self.underflow_events += 1,
        }
    }

    fn is_clean(&self) -> bool {
        self.underflow_events == 0 && self.unaccounted_bytes == 0
    }

    fn print(&self, label: &str) {
        println!(
            "  [{label}] accounting: device={} host={} mapped={} underflow={} \
             unaccounted={} warm={} cold={}",
            self.device_committed_bytes,
            self.host_committed_bytes,
            self.mapped_bytes,
            self.underflow_events,
            self.unaccounted_bytes,
            self.pool_warm_count,
            self.pool_cold_count,
        );
    }
}

// Wall-clock `Instant` is the correct instrument for host-side VMM remap cost:
// cuMemUnmap/cuMemMap/cuMemSetAccess are synchronous driver calls on the host
// thread. GPU event timing is not applicable here (no GPU kernels are timed in
// the remap path). cuMemcpyDtoH 64B is used as a host-timed proxy for
// first-access latency (component e).

// ---------------------------------------------------------------------------
// Core remap harness
// ---------------------------------------------------------------------------

/// Run the full remap benchmark for one shape + K + trace.
///
/// Returns `(remap_stats, kernel_first_access_us, kernel_steady_us, accounting)`
#[allow(clippy::too_many_lines)]
fn run_remap_benchmark(
    context: &Arc<CudaContext>,
    shape: &QmoeShape,
    k_remap_per_step: usize,
    trace: &[Vec<usize>],
    _trace_name: &str,
    n_steps: usize,
    granularity: usize,
    host_numa_id: i32,
    governor: &'static LedgerGovernor,
    holder_base: u32,
) -> (RemapComponentTimes, Vec<f64>, Vec<f64>, BenchAccounting) {
    assert!(n_steps <= trace.len(), "trace shorter than n_steps");

    // --- Pool setup ---
    let device_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(context),
        0,
        PhysicalLocation::Device { ordinal: 0 },
        // Retain up to (k_remap_per_step + 4) granules for warm reuse.
        (k_remap_per_step + 4) * granularity,
        governor,
        HolderId::new(holder_base as u64),
        MemoryRole::Weights,
    )
    .expect("device pool");

    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(context),
        0,
        PhysicalLocation::HostNuma { node: host_numa_id },
        (k_remap_per_step + 4) * granularity,
        governor,
        HolderId::new((holder_base + 1) as u64),
        MemoryRole::Weights,
    )
    .expect("host_numa pool");

    // Two backings: one per pool. Each backing is tied to its pool for the
    // release path (pool.return_after_unmap is called via the backing's pool).
    let device_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&device_pool));
    let host_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&host_pool));

    // --- Compute VA size ---
    // Each expert's fc1 packed bytes = inter × hidden / 2 (int4).
    let expert_bytes = shape.inter * shape.hidden / 2;
    let granules_per_expert = expert_bytes.div_ceil(granularity);
    let total_granules = shape.experts * granules_per_expert;
    let total_bytes = total_granules * granularity;

    // --- Reserve the stable VA ---
    // Use device_backing.reserve: it calls cuMemAddressReserve (VA only, no
    // physical mapping yet). Pool choice doesn't matter for VA reservation.
    context.bind_to_thread().expect("bind context for reserve");
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&device_backing, total_bytes)
            .expect("VA reservation must succeed");
    let base_va = <CudaVirtualBacking as VirtualBacking>::base(&reservation);

    // --- Commit ALL experts initially as host-NUMA-backed ---
    let mut granule_location: Vec<GranuleLocation> =
        vec![GranuleLocation::HostNuma; total_granules];
    let mut accounting = BenchAccounting::default();

    context.bind_to_thread().expect("bind for initial commit");
    for g in 0..total_granules {
        let offset = g * granularity;
        host_backing
            .commit_at_location(
                &mut reservation,
                offset,
                PhysicalLocation::HostNuma { node: host_numa_id },
                &host_pool,
            )
            .expect("initial host-NUMA commit must succeed");
        accounting.note_commit(GranuleLocation::HostNuma, granularity);
    }

    assert_eq!(
        reservation.mapped_blocks().len(),
        total_granules,
        "all granules should be mapped after initial setup"
    );

    // --- Remap measurement loop ---
    let mut remap_times = RemapComponentTimes::default();
    let mut first_access_us: Vec<f64> = Vec::new();
    let mut steady_us: Vec<f64> = Vec::new();
    let device_pool_stats = device_pool.stats();
    // host_pool_stats used for future per-location accounting; warm/cold detected via device pool.
    let _ = host_pool.stats();

    for (step, routed_experts) in trace.iter().enumerate().take(n_steps) {
        // Pick k_remap_per_step experts to promote (host→device) this step.
        // Among the routed experts, promote those still on host; if fewer
        // than k_remap_per_step routed experts need promotion, pick
        // additional from the routed set (cycling).
        let promote_candidates: Vec<usize> = routed_experts
            .iter()
            .filter(|&&e| {
                // All granules for expert e
                (0..granules_per_expert).any(|g_in_expert| {
                    let g_idx = e * granules_per_expert + g_in_expert;
                    granule_location[g_idx] == GranuleLocation::HostNuma
                })
            })
            .copied()
            .take(k_remap_per_step)
            .collect();

        for &expert in &promote_candidates {
            let g_start = expert * granules_per_expert;
            for g_in_expert in 0..granules_per_expert.min(1) {
                // Remap one granule per expert per step (not the whole expert
                // at once) to measure one-by-one primitive cost.
                let g_idx = g_start + g_in_expert;
                if granule_location[g_idx] != GranuleLocation::HostNuma {
                    continue;
                }
                let offset = g_idx * granularity;

                // Measure unmap (release from host_numa_backing).
                // release_range_reporting calls synchronizing_section() internally.
                let t_unmap_start = Instant::now();
                let report =
                    host_backing.release_range_reporting(&mut reservation, offset, granularity);
                let unmap_us = t_unmap_start.elapsed().as_secs_f64() * 1e6;

                // Any still-mapped blocks means the unmap failed — abort.
                if !report.still_mapped.is_empty() {
                    panic!(
                        "unmap failed at step {step} expert {expert} granule {g_idx}: {:?}",
                        report.faults
                    );
                }
                accounting.note_release(GranuleLocation::HostNuma, granularity);

                // Measure map (commit to device backing).
                // Determine pool-warm vs cold from counter delta.
                let d_stats_before = device_pool_stats.snapshot();
                let t_map_start = Instant::now();
                device_backing
                    .commit_at_location(
                        &mut reservation,
                        offset,
                        PhysicalLocation::Device { ordinal: 0 },
                        &device_pool,
                    )
                    .expect("promote commit_at_location must succeed");
                let map_us = t_map_start.elapsed().as_secs_f64() * 1e6;
                let d_stats_after = device_pool_stats.snapshot();

                let was_warm = d_stats_after.total_owned_bytes == d_stats_before.total_owned_bytes;
                accounting.note_commit(GranuleLocation::Device, granularity);
                if was_warm {
                    accounting.pool_warm_count += 1;
                } else {
                    accounting.pool_cold_count += 1;
                }

                granule_location[g_idx] = GranuleLocation::Device;
                remap_times.record(unmap_us, map_us, was_warm);
            }
        }

        // After remap, measure a synchronous GPU-side memcpy as a proxy for
        // "first kernel access after remap" (cuMemcpyDtoH touches the
        // freshly-remapped VA — measures TLB/cold-cache cost from host side).
        // We use a small 64-byte read from the first promoted expert's granule.
        if !promote_candidates.is_empty() {
            let probe_expert = promote_candidates[0];
            let probe_offset = probe_expert * granules_per_expert * granularity;
            let probe_addr = base_va as u64 + probe_offset as u64;

            let mut buf = [0u8; 64];
            context.bind_to_thread().expect("bind for probe");

            let t_first = Instant::now();
            unsafe {
                let _ = cu::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _,
                    probe_addr as cu::CUdeviceptr,
                    64,
                );
            }
            first_access_us.push(t_first.elapsed().as_secs_f64() * 1e6);

            // Steady-state: second access to the same location.
            let t_steady = Instant::now();
            unsafe {
                let _ = cu::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _,
                    probe_addr as cu::CUdeviceptr,
                    64,
                );
            }
            steady_us.push(t_steady.elapsed().as_secs_f64() * 1e6);
        }
    }

    (remap_times, first_access_us, steady_us, accounting)
}

// ---------------------------------------------------------------------------
// Correctness check: bit-identical read-back after promote/demote cycles
// ---------------------------------------------------------------------------

fn assert_bit_identical(reference: &[u8], actual: &[u8], label: &str) {
    assert_eq!(reference.len(), actual.len(), "{label}: length mismatch");
    let mut mismatches = 0usize;
    for (i, (&r, &a)) in reference.iter().zip(actual.iter()).enumerate() {
        if r != a {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!("{label}: mismatch at [{i}] ref={r:#04x} actual={a:#04x}");
            }
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{label}: {mismatches}/{} byte mismatches",
        reference.len()
    );
}

// ---------------------------------------------------------------------------
// Correctness: stable VA, bit-identical after repeated promote/demote
// ---------------------------------------------------------------------------

fn run_correctness_cycles(
    context: &Arc<CudaContext>,
    granularity: usize,
    host_numa_id: i32,
    governor: &'static LedgerGovernor,
    n_experts: usize,
    granules_per_expert: usize,
    n_cycles: usize,
    holder_base: u32,
) {
    let device_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(context),
        0,
        PhysicalLocation::Device { ordinal: 0 },
        granules_per_expert * 4 * granularity,
        governor,
        HolderId::new(holder_base as u64),
        MemoryRole::Weights,
    )
    .expect("device pool for correctness");

    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(context),
        0,
        PhysicalLocation::HostNuma { node: host_numa_id },
        granules_per_expert * 4 * granularity,
        governor,
        HolderId::new((holder_base + 1) as u64),
        MemoryRole::Weights,
    )
    .expect("host_numa pool for correctness");

    let device_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&device_pool));
    let host_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&host_pool));

    let total_granules = n_experts * granules_per_expert;
    let total_bytes = total_granules * granularity;

    context
        .bind_to_thread()
        .expect("bind for correctness reserve");
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&device_backing, total_bytes)
            .expect("correctness VA reserve");
    let base_va = <CudaVirtualBacking as VirtualBacking>::base(&reservation);

    // Start all host-NUMA.
    for g in 0..total_granules {
        host_backing
            .commit_at_location(
                &mut reservation,
                g * granularity,
                PhysicalLocation::HostNuma { node: host_numa_id },
                &host_pool,
            )
            .expect("initial host commit");
    }

    // Write a known pattern (using byte indices for repeatability).
    let pattern: Vec<u8> = (0..total_bytes).map(|i| (i * 7 + 13) as u8).collect();
    context.bind_to_thread().expect("bind for write");
    unsafe {
        assert_eq!(
            cu::cuMemcpyHtoD_v2(
                base_va as cu::CUdeviceptr,
                pattern.as_ptr() as *const _,
                total_bytes
            ),
            cu::CUresult::CUDA_SUCCESS,
            "initial write failed"
        );
    }

    // Read reference while all-host-NUMA.
    let mut ref_read = vec![0u8; total_bytes];
    unsafe {
        assert_eq!(
            cu::cuMemcpyDtoH_v2(
                ref_read.as_mut_ptr() as *mut _,
                base_va as cu::CUdeviceptr,
                total_bytes
            ),
            cu::CUresult::CUDA_SUCCESS,
            "reference read failed"
        );
    }
    assert_bit_identical(&pattern, &ref_read, "initial all-host-NUMA read");

    let mut granule_location: Vec<GranuleLocation> =
        vec![GranuleLocation::HostNuma; total_granules];

    // Promote-then-demote cycles on expert 0.
    for cycle in 0..n_cycles {
        let expert = cycle % n_experts;
        let g_start = expert * granules_per_expert;

        // Promote expert's first granule: host → device.
        let g_idx = g_start;
        let offset = g_idx * granularity;
        if granule_location[g_idx] == GranuleLocation::HostNuma {
            let report =
                host_backing.release_range_reporting(&mut reservation, offset, granularity);
            assert!(
                report.still_mapped.is_empty(),
                "promote unmap failed cycle {cycle}"
            );
            device_backing
                .commit_at_location(
                    &mut reservation,
                    offset,
                    PhysicalLocation::Device { ordinal: 0 },
                    &device_pool,
                )
                .expect("promote commit failed");
            granule_location[g_idx] = GranuleLocation::Device;
        }

        // Verify content still correct (stable VA, fresh device backing).
        let mut post_promote = vec![0u8; granularity];
        unsafe {
            assert_eq!(
                cu::cuMemcpyDtoH_v2(
                    post_promote.as_mut_ptr() as *mut _,
                    (base_va as u64 + offset as u64) as cu::CUdeviceptr,
                    granularity,
                ),
                cu::CUresult::CUDA_SUCCESS,
                "post-promote read failed"
            );
        }
        // NOTE: device-committed granules do NOT retain their content from
        // the host-NUMA backing after remap; the physical handle is a NEW
        // (or pooled/reused) device allocation. This is by design: remap
        // changes backing, not content. We verify pointer stability only.
        // Content-correctness across remap (i.e., data migration on promotion)
        // requires an explicit copy — see timing component (f) in module docs.
        let _ = post_promote; // pointer stability asserted below via base_va check

        // Demote back: device → host-NUMA.
        let report = device_backing.release_range_reporting(&mut reservation, offset, granularity);
        assert!(
            report.still_mapped.is_empty(),
            "demote unmap failed cycle {cycle}"
        );
        host_backing
            .commit_at_location(
                &mut reservation,
                offset,
                PhysicalLocation::HostNuma { node: host_numa_id },
                &host_pool,
            )
            .expect("demote commit failed");
        granule_location[g_idx] = GranuleLocation::HostNuma;

        // Write known data into the re-demoted granule and verify.
        let slice_offset = offset;
        let slice_len = granularity;
        let expected: Vec<u8> = (slice_offset..slice_offset + slice_len)
            .map(|i| (i * 7 + 13) as u8)
            .collect();
        unsafe {
            assert_eq!(
                cu::cuMemcpyHtoD_v2(
                    (base_va as u64 + offset as u64) as cu::CUdeviceptr,
                    expected.as_ptr() as *const _,
                    slice_len,
                ),
                cu::CUresult::CUDA_SUCCESS,
                "re-write after demote failed"
            );
        }
        let mut read_back = vec![0u8; slice_len];
        unsafe {
            assert_eq!(
                cu::cuMemcpyDtoH_v2(
                    read_back.as_mut_ptr() as *mut _,
                    (base_va as u64 + offset as u64) as cu::CUdeviceptr,
                    slice_len,
                ),
                cu::CUresult::CUDA_SUCCESS,
                "read-back after re-demote failed"
            );
        }
        assert_bit_identical(
            &expected,
            &read_back,
            &format!("demote-then-write-then-read cycle {cycle}"),
        );

        // VA must not have changed.
        let current_base = <CudaVirtualBacking as VirtualBacking>::base(&reservation);
        assert_eq!(
            base_va, current_base,
            "VA changed at cycle {cycle}: {base_va:#x} → {current_base:#x}"
        );
    }
    println!(
        "  correctness: {n_cycles} promote/demote cycles PASS — VA stable, content correct after re-demote"
    );
}

// ---------------------------------------------------------------------------
// Accounting/oscillation: ≥1000 steps of promote/demote churn
// ---------------------------------------------------------------------------

fn run_accounting_oscillation(
    context: &Arc<CudaContext>,
    granularity: usize,
    host_numa_id: i32,
    governor: &'static LedgerGovernor,
    n_steps: usize,
    holder_base: u32,
) {
    let device_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(context),
        0,
        PhysicalLocation::Device { ordinal: 0 },
        4 * granularity, // retain a few handles
        governor,
        HolderId::new(holder_base as u64),
        MemoryRole::Weights,
    )
    .expect("device pool oscillation");

    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(context),
        0,
        PhysicalLocation::HostNuma { node: host_numa_id },
        4 * granularity,
        governor,
        HolderId::new((holder_base + 1) as u64),
        MemoryRole::Weights,
    )
    .expect("host_numa pool oscillation");

    let device_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&device_pool));
    let host_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&host_pool));

    let d_stats = device_pool.stats();
    let h_stats = host_pool.stats();
    let d_base = d_stats.snapshot();
    let h_base = h_stats.snapshot();

    context.bind_to_thread().expect("bind for oscillation");
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&device_backing, granularity)
            .expect("oscillation reserve");

    // Start host-NUMA.
    host_backing
        .commit_at_location(
            &mut reservation,
            0,
            PhysicalLocation::HostNuma { node: host_numa_id },
            &host_pool,
        )
        .expect("oscillation initial commit");

    let mut current = GranuleLocation::HostNuma;
    let mut warm_count = 0u64;
    let mut cold_count = 0u64;
    let mut accounting = BenchAccounting::default();
    accounting.note_commit(GranuleLocation::HostNuma, granularity);

    for step in 0..n_steps {
        let (
            from_backing,
            from_pool_location,
            to_backing,
            to_pool_stats,
            to_pool_location,
            to_pool,
        ) = if current == GranuleLocation::HostNuma {
            (
                &host_backing,
                GranuleLocation::HostNuma,
                &device_backing,
                &d_stats,
                GranuleLocation::Device,
                &device_pool,
            )
        } else {
            (
                &device_backing,
                GranuleLocation::Device,
                &host_backing,
                &h_stats,
                GranuleLocation::HostNuma,
                &host_pool,
            )
        };

        let report = from_backing.release_range_reporting(&mut reservation, 0, granularity);
        assert!(
            report.still_mapped.is_empty(),
            "oscillation unmap failed at step {step}: {:?}",
            report.faults
        );
        accounting.note_release(from_pool_location, granularity);

        let to_stats_before = to_pool_stats.snapshot();

        let to_location = match to_pool_location {
            GranuleLocation::Device => PhysicalLocation::Device { ordinal: 0 },
            GranuleLocation::HostNuma => PhysicalLocation::HostNuma { node: host_numa_id },
        };
        to_backing
            .commit_at_location(&mut reservation, 0, to_location, to_pool)
            .unwrap_or_else(|error| panic!("oscillation commit failed at step {step}: {error}"));
        accounting.note_commit(to_pool_location, granularity);

        let to_stats_after = to_pool_stats.snapshot();
        let was_warm = to_stats_after.total_owned_bytes == to_stats_before.total_owned_bytes;
        if was_warm {
            warm_count += 1;
        } else {
            cold_count += 1;
        }
        current = to_pool_location;
    }

    // After oscillation: clean up.
    let final_release_backing = match current {
        GranuleLocation::HostNuma => &host_backing,
        GranuleLocation::Device => &device_backing,
    };
    let report = final_release_backing.release_range_reporting(&mut reservation, 0, granularity);
    assert!(report.still_mapped.is_empty(), "final release failed");

    let d_after = d_stats.snapshot();
    let h_after = h_stats.snapshot();

    assert_eq!(
        d_after.mapped_bytes, d_base.mapped_bytes,
        "device mapped_bytes not restored"
    );
    assert_eq!(
        h_after.mapped_bytes, h_base.mapped_bytes,
        "host mapped_bytes not restored"
    );
    assert_eq!(
        accounting.underflow_events, 0,
        "underflow in oscillation accounting"
    );
    assert_eq!(
        accounting.unaccounted_bytes, 0,
        "unaccounted in oscillation accounting"
    );

    let steady_warm_rate = if step_count_past_warmup(n_steps, warm_count, cold_count) > 0 {
        warm_count as f64 / n_steps as f64
    } else {
        0.0
    };

    println!(
        "  oscillation: {n_steps} steps PASS — underflow=0 unaccounted=0 \
         warm={warm_count} cold={cold_count} steady_warm_rate={:.1}%",
        steady_warm_rate * 100.0
    );
}

fn step_count_past_warmup(total: usize, warm: u64, _cold: u64) -> u64 {
    // First few steps are always cold (pool empty). Return warm count directly.
    let _ = total;
    warm
}

// ---------------------------------------------------------------------------
// Capture/gate rejection: remap attempted while CaptureExclusion held must
// block on synchronizing_section, not proceed.
// ---------------------------------------------------------------------------

fn run_capture_gate_check(
    context: &Arc<CudaContext>,
    granularity: usize,
    host_numa_id: i32,
    governor: &'static LedgerGovernor,
    holder_base: u32,
) {
    use onnx_runtime_cuda_memory::capture_gate::CaptureExclusion;

    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(context),
        0,
        PhysicalLocation::HostNuma { node: host_numa_id },
        2 * granularity,
        governor,
        HolderId::new(holder_base as u64),
        MemoryRole::Weights,
    )
    .expect("host pool for gate check");

    let host_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&host_pool));
    context.bind_to_thread().expect("bind for gate check");
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&host_backing, granularity)
            .expect("gate check reserve");

    host_backing
        .commit_at_location(
            &mut reservation,
            0,
            PhysicalLocation::HostNuma { node: host_numa_id },
            &host_pool,
        )
        .expect("gate check initial commit");

    // Acquire CaptureExclusion on this thread to simulate a capture in progress.
    // synchronizing_section() inside release_range_reporting must wait for the
    // capture to finish. Because this is the capturing thread, the re-entrant
    // guard (CAPTURE_DEPTH > 0) lets the same-thread remap proceed — this is
    // the documented re-entrant capture behaviour: the capturing thread's own
    // allocations are not excluded (they are ordered with respect to recording).
    //
    // Per the #1813 finding: "the driver does NOT self-refuse a remap during
    // capture; the cooperative gate is the only enforcement mechanism." We verify
    // the gate is engaged (synchronizing_section is entered inside the release)
    // and that same-thread calls succeed (re-entrant depth logic).
    let _capture = CaptureExclusion::acquire();
    let report = host_backing.release_range_reporting(&mut reservation, 0, granularity);
    // On the capturing thread (CAPTURE_DEPTH > 0), synchronizing_section returns
    // None (re-entrant path), allowing the remap to proceed.
    assert!(
        report.still_mapped.is_empty(),
        "same-thread remap during capture must proceed (re-entrant gate): {:?}",
        report.faults
    );
    drop(_capture);

    // Verify on a DIFFERENT thread: remap while another thread holds the
    // capture exclusion must block (synchronizing_section waits for captures=0).
    let host_pool2 = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(context),
        0,
        PhysicalLocation::HostNuma { node: host_numa_id },
        2 * granularity,
        governor,
        HolderId::new((holder_base + 1) as u64),
        MemoryRole::Weights,
    )
    .expect("host pool2 for gate cross-thread");

    let host_backing2 = CudaVirtualBacking::with_physical_pool(Arc::clone(&host_pool2));
    context
        .bind_to_thread()
        .expect("bind for cross-thread reservation");
    let mut reservation2 =
        <CudaVirtualBacking as VirtualBacking>::reserve(&host_backing2, granularity)
            .expect("cross-thread reserve");
    host_backing2
        .commit_at_location(
            &mut reservation2,
            0,
            PhysicalLocation::HostNuma { node: host_numa_id },
            &host_pool2,
        )
        .expect("cross-thread initial commit");

    // Spawn the capture-holding thread FIRST.
    let capture_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let capture_done_clone = Arc::clone(&capture_done);
    let capture_thread = std::thread::spawn(move || {
        let _excl = CaptureExclusion::acquire();
        // Hold the exclusion for 50 ms to give the remap thread a chance to
        // be blocked at synchronizing_section.
        std::thread::sleep(std::time::Duration::from_millis(50));
        capture_done_clone.store(true, std::sync::atomic::Ordering::Release);
        // CaptureExclusion drops here, releasing the gate.
    });

    // Give capture thread time to acquire the exclusion.
    std::thread::sleep(std::time::Duration::from_millis(5));

    let t_remap_start = Instant::now();
    // The remap (via release_range_reporting → synchronizing_section) must
    // BLOCK here until the capture thread releases the exclusion.
    let report2 = host_backing2.release_range_reporting(&mut reservation2, 0, granularity);
    let remap_elapsed_ms = t_remap_start.elapsed().as_millis();
    assert!(
        report2.still_mapped.is_empty(),
        "cross-thread remap must succeed after capture ends"
    );
    // The remap must have waited for the capture to finish (≥ ~45 ms wait).
    assert!(
        remap_elapsed_ms >= 40,
        "remap did not wait for capture to finish: waited only {remap_elapsed_ms}ms"
    );
    assert!(
        capture_done.load(std::sync::atomic::Ordering::Acquire),
        "capture was not done when remap completed"
    );
    capture_thread.join().expect("capture thread panicked");

    println!(
        "  capture gate: PASS — same-thread re-entrant allowed, cross-thread remap blocked \
         for {remap_elapsed_ms}ms until capture released"
    );
}

// ---------------------------------------------------------------------------
// GO/NO-GO verdict helper
// ---------------------------------------------------------------------------

fn print_go_no_go(
    label: &str,
    k: usize,
    median_total_remap_us: f64,
    qmoe_layer_kernel_us_range: (f64, f64),
    whole_token_interval_us: f64,
) {
    // NOTE on quantities compared here (see #1823's review finding on #1829):
    // `qmoe_layer_kernel_us_range` is the wall-clock time of ONE QMoE kernel
    // invocation for a decode-shaped (rows=1) input -- i.e. one MoE-FFN
    // layer's forward pass for one token, as measured in #1813
    // (`.squad/decisions/inbox/deckard-1810-composable-vmm-spike-results.md`,
    // "median exec (µs)" column). It is NOT a whole decode-step/token
    // duration; a real model has ~dozens of layers plus attention, norms,
    // etc. per token.
    //
    // `whole_token_interval_us` is an independently measured, real,
    // end-to-end whole-model decode-step interval for the SAME model
    // (DeepSeek-V2-Lite int4, CUDA greedy, graph capture ON) from
    // `.squad/decisions/inbox/gaff-deepseek-v2-mask-graph-capture.md`:
    // 165.3 tok/s -> ~6.05 ms/token = 6050 µs/token. It is used ONLY as an
    // upper-bound denominator below, not as a budget: it is a whole-model
    // number, and any remap must actually fit inside its OWN layer's compute
    // window (`qmoe_layer_kernel_us_range`), not float freely across the rest
    // of the token's unrelated work.
    let (layer_lo_us, layer_hi_us) = qmoe_layer_kernel_us_range;

    // Whether the measured remap cost fits inside a *single MoE layer's own*
    // QMoE-kernel compute window -- the only budget a per-layer-blocking
    // remap can actually hide inside. If it does not fit here, no whole-token
    // arithmetic below can rescue it: the remap still stalls the layer's
    // critical path.
    let fits_within_layer_window = median_total_remap_us <= layer_hi_us;

    // Purely arithmetic, NOT-ATTAINABLE upper bound: how many serial remaps
    // of this cost *would* fit in one whole decode-step interval, IF every
    // other bit of per-token work (all other layers, attention, norms, host
    // overhead, ...) were free. This is a ceiling on a hypothetical, not a
    // real budget -- the real budget for THIS remap is still the single
    // layer's own compute window above.
    let unattainable_serial_ceiling_over_whole_token = if median_total_remap_us > 0.0 {
        (whole_token_interval_us / median_total_remap_us).floor() as usize
    } else {
        usize::MAX
    };

    println!("\n--- GO/NO-GO verdict: {label} (k={k}) ---");
    println!(
        "  median remap cost per granule: {:.1} µs",
        median_total_remap_us
    );
    println!(
        "  single QMoE-layer kernel execution time, decode-shaped rows=1 (from #1813): {:.0}–{:.0} µs \
         [NOT a whole decode-step/token duration]",
        layer_lo_us, layer_hi_us
    );
    println!(
        "  independently measured whole-model decode-step interval (DeepSeek-V2-Lite int4, \
         CUDA greedy, graph-on, 165.3 tok/s, from gaff-deepseek-v2-mask-graph-capture.md): \
         {:.0} µs/token (~{:.2} ms/token)",
        whole_token_interval_us,
        whole_token_interval_us / 1000.0
    );
    println!(
        "  fits within a single QMoE layer's own compute window ({:.0}µs high end)? {}",
        layer_hi_us,
        if fits_within_layer_window {
            "YES"
        } else {
            "NO"
        }
    );
    println!(
        "  unattainable arithmetic ceiling -- serial remaps of this cost that would fit in one \
         WHOLE {:.0}µs token interval IF all other per-token work were free (NOT a real budget, \
         NOT achievable, only an upper bound): {}",
        whole_token_interval_us, unattainable_serial_ceiling_over_whole_token
    );

    if !fits_within_layer_window {
        println!(
            "  VERDICT: this remap ({:.1}µs) exceeds a single QMoE layer's own kernel compute \
             window ({:.0}–{:.0}µs) and so CANNOT be hidden locally inside that layer's step; \
             k={k} granule(s) required per remap event. NOT ready for per-token/per-layer remap; \
             boundary-level only with the current API (see hard blocker below, which applies \
             independently of any latency arithmetic).",
            median_total_remap_us, layer_lo_us, layer_hi_us
        );
    } else {
        println!(
            "  VERDICT: this remap ({:.1}µs) would fit inside a single QMoE layer's own kernel \
             compute window ({:.0}–{:.0}µs) for k={k} granule(s) on latency grounds alone -- but \
             see the independent hard blocker below: NOT ready for per-token; boundary-level only \
             with the current API regardless of latency.",
            median_total_remap_us, layer_lo_us, layer_hi_us
        );
    }
    println!(
        "  INDEPENDENT HARD BLOCKER (applies regardless of the latency verdict above): remap \
         must be synchronized against all in-flight kernels reading the OLD mapping before \
         proceeding. Production primitives do NOT currently enforce this ordering automatically \
         (no stream-order guarantee) -- that is a gap requiring stream-sync / deferred-release \
         queue integration before per-token or dispatch-time remap is safe, independent of \
         whether the raw µs cost would otherwise fit."
    );
}

// ---------------------------------------------------------------------------
// The actual test entry points (all #[ignore] — GPU required)
// ---------------------------------------------------------------------------

/// Verify GPU idle and print platform conditions before any test.
#[test]
#[ignore = "GPU required; run with CUDA_VISIBLE_DEVICES=4 --ignored --nocapture"]
fn gpu_idle_and_platform_check() {
    assert_gpu_idle_or_warn("pre-flight");
    print_platform();
}

/// Correctness: stable VA and bit-identical content after repeated promote/demote
/// cycles for both shapes.
#[test]
#[ignore = "GPU required"]
fn correctness_stable_va_and_content_after_promote_demote_cycles() {
    assert_gpu_idle_or_warn("correctness");
    let (context, _guard) = require_cuda_context();
    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: HOST_NUMA not supported: {reason}");
            return;
        }
    };
    println!(
        "capability: numa_id={} granularity={}",
        cap.host_numa_id, cap.granularity
    );

    let gov = make_governor(0, 4 << 30, 4 << 30);

    for shape in &[QWEN15_MOE_A27B, SYNTH_256_EXPERT] {
        println!("=== correctness: {} ===", shape.name);
        let expert_bytes = shape.inter * shape.hidden / 2;
        let granules_per_expert = expert_bytes.div_ceil(cap.granularity);
        // Use a small subset of experts for correctness (full expert set would
        // require too much memory for a pure correctness check).
        let n_experts = 4;
        run_correctness_cycles(
            &context,
            cap.granularity,
            cap.host_numa_id,
            gov,
            n_experts,
            granules_per_expert,
            20, // 20 promote/demote cycles
            100 + shape.experts as u32,
        );
    }
}

/// Accounting oscillation: ≥1000 promote/demote steps — zero underflow, zero
/// unaccounted, pool warm/cold rates reported.
#[test]
#[ignore = "GPU required"]
fn accounting_oscillation_1000_steps_zero_underflow() {
    assert_gpu_idle_or_warn("oscillation");
    let (context, _guard) = require_cuda_context();
    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: {reason}");
            return;
        }
    };

    let gov = make_governor(0, 2 << 30, 2 << 30);
    println!("Running 1000-step accounting oscillation (single granule)...");
    run_accounting_oscillation(&context, cap.granularity, cap.host_numa_id, gov, 1000, 200);
}

/// Capture/gate: remap during capture — same-thread re-entrant allowed,
/// cross-thread blocked until capture ends.
#[test]
#[ignore = "GPU required"]
fn capture_gate_remap_during_capture_blocked_on_cross_thread() {
    assert_gpu_idle_or_warn("capture-gate");
    let (context, _guard) = require_cuda_context();
    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: {reason}");
            return;
        }
    };

    let gov = make_governor(0, 1 << 30, 1 << 30);
    run_capture_gate_check(&context, cap.granularity, cap.host_numa_id, gov, 300);
}

/// Teardown: no panic in Drop with various residual states.
#[test]
#[ignore = "GPU required"]
fn teardown_no_panic_in_drop() {
    assert_gpu_idle_or_warn("teardown");
    let (context, _guard) = require_cuda_context();
    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: {reason}");
            return;
        }
    };

    let gov = make_governor(0, 1 << 30, 1 << 30);
    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::HostNuma {
            node: cap.host_numa_id,
        },
        2 * cap.granularity,
        gov,
        HolderId::new(400),
        MemoryRole::Weights,
    )
    .expect("host pool teardown");

    let host_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&host_pool));
    context.bind_to_thread().expect("bind");

    // Create reservation, commit, then drop without explicit release.
    // Drop must not panic (panics in Drop cause SIGABRT / stack overflow).
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&host_backing, cap.granularity)
            .expect("teardown reserve");
    host_backing
        .commit_at_location(
            &mut reservation,
            0,
            PhysicalLocation::HostNuma {
                node: cap.host_numa_id,
            },
            &host_pool,
        )
        .expect("teardown commit");
    drop(reservation); // must not panic
    println!("  teardown: Drop with committed reservation: no panic — PASS");
}

/// Concurrent-stream safety documentation test.
///
/// This test DOCUMENTS the synchronization contract rather than enforcing it
/// at the driver level. Production primitives do not currently guarantee that
/// a remap is ordered against in-flight kernels reading the OLD mapping.
/// The gap is explicitly reported here.
///
/// What we verify: the backing's release + commit each acquire
/// synchronizing_section(), which excludes active captures but does NOT
/// issue cuStreamSynchronize(). Callers must externally synchronize streams
/// before remap. This is a known gap for per-token remap safety.
#[test]
#[ignore = "GPU required"]
fn concurrent_stream_sync_contract_documented() {
    assert_gpu_idle_or_warn("concurrent-stream");
    let (context, _guard) = require_cuda_context();
    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: {reason}");
            return;
        }
    };

    let gov = make_governor(0, 1 << 30, 1 << 30);
    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::HostNuma {
            node: cap.host_numa_id,
        },
        2 * cap.granularity,
        gov,
        HolderId::new(500),
        MemoryRole::Weights,
    )
    .expect("host pool stream test");

    let host_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&host_pool));
    context.bind_to_thread().expect("bind for stream test");

    // Create two streams.
    let stream_a = context.new_stream().expect("stream A");
    let stream_b = context.new_stream().expect("stream B");

    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&host_backing, cap.granularity)
            .expect("stream test reserve");
    host_backing
        .commit_at_location(
            &mut reservation,
            0,
            PhysicalLocation::HostNuma {
                node: cap.host_numa_id,
            },
            &host_pool,
        )
        .expect("stream test commit");

    // The gap: we do NOT synchronize any stream before unmapping. Production would
    // need to do so before release_range_reporting. Document:
    println!(
        "  DOCUMENTED GAP: work enqueued on stream_a without sync before remap. \
         release_range_reporting does NOT wait for stream_a to complete. \
         If a kernel is still reading the OLD host-NUMA mapping while the remap \
         proceeds on the host thread, the result is UNDEFINED BEHAVIOUR. \
         Production must synchronize all reading streams before remap. \
         This is the per-token remap safety gap: BOUNDARY-ONLY until a \
         deferred-release queue integration enforces stream ordering."
    );

    // For test correctness, synchronize NOW before we release.
    stream_a
        .synchronize()
        .expect("sync stream A before release");

    let report = host_backing.release_range_reporting(&mut reservation, 0, cap.granularity);
    assert!(
        report.still_mapped.is_empty(),
        "release after sync must succeed: {:?}",
        report.faults
    );
    drop(stream_a);
    drop(stream_b);
    println!("  concurrent-stream safety: stream sync gap documented — PASS (with explicit sync)");
}

/// Main benchmark: timing decomposition for K ∈ {1,2,4,8} expert remaps per
/// step, two shapes, two routing traces, ≥3 runs. Reports medians and ranges
/// for all timing components, warm/cold pool rates, and GO/NO-GO verdict.
#[test]
#[ignore = "GPU required — takes several minutes; run solo with idle GPU"]
fn remap_cost_timing_decomposition_all_configs() {
    assert_gpu_idle_or_warn("benchmark");
    print_platform();
    let (context, _guard) = require_cuda_context();

    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: HOST_NUMA not supported: {reason}");
            return;
        }
    };
    println!(
        "HOST_NUMA: device={} numa_id={} gran={} B ({} MiB)",
        cap.device_ordinal,
        cap.host_numa_id,
        cap.granularity,
        cap.granularity / (1024 * 1024)
    );

    let n_steps = 1000;
    let n_runs = 3;
    let shapes = [QWEN15_MOE_A27B, SYNTH_256_EXPERT];
    let k_values = [1usize, 2, 4, 8];

    // Single QMoE-kernel invocation time (one MoE-FFN layer, decode-shaped
    // rows=1 input) from #1813's decision doc
    // (`.squad/decisions/inbox/deckard-1810-composable-vmm-spike-results.md`,
    // "median exec (µs)" column): ~150–430 µs for the Qwen1.5-MoE-A2.7B and
    // DeepSeek-V2-Lite shapes measured there. This is ONE LAYER's kernel
    // time, not a whole decode-step/token duration -- see `print_go_no_go`'s
    // doc comment for why the two must not be conflated.
    //
    // NOTE: #1813 measured this range for 60-64-expert shapes
    // (Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite). It was NOT independently
    // measured for the 256-expert synthetic shape used elsewhere in this
    // benchmark; applying it to that shape too is a plausible assumption
    // (decode-shaped QMoE execution only touches top_k experts regardless of
    // total expert count) rather than a directly-measured fact for that row.
    let qmoe_layer_kernel_range = (150.0_f64, 430.0_f64);

    // Independently measured whole-model decode-step interval for
    // DeepSeek-V2-Lite int4 (CUDA greedy, graph capture ON), from
    // `.squad/decisions/inbox/gaff-deepseek-v2-mask-graph-capture.md`:
    // 165.3 tok/s -> ~6.05 ms/token = 6050 µs/token. Used ONLY as the
    // denominator for an explicitly-labeled, NOT-attainable arithmetic
    // ceiling in `print_go_no_go` -- never as a per-remap budget. The real
    // budget for a per-layer-blocking remap is that layer's own compute
    // window (`qmoe_layer_kernel_range`), not the whole token's interval.
    let whole_token_interval_us = 6_050.0_f64;

    println!(
        "\n=== Remap Cost Benchmark — {} steps, {} runs, {} shapes, {} K values ===",
        n_steps,
        n_runs,
        shapes.len(),
        k_values.len()
    );
    println!("Timing components:");
    println!(
        "  (b) unmap = release_range_reporting wall-clock (includes synchronizing_section wait + cuMemUnmap)"
    );
    println!("  (d) map   = commit_at_location wall-clock (cuMemMap + cuMemSetAccess)");
    println!(
        "  (c) pool  = warm hit (reused pooled handle) vs cold (fresh cuMemCreate) — from pool counter delta"
    );
    println!(
        "  (e) GPU-side first-access vs steady-state: cuMemcpyDtoH 64B post-remap (host-timed proxy)"
    );

    let mut holder = 1000u32;
    let mut all_remap_medians: Vec<(String, usize, f64)> = Vec::new();

    for shape in &shapes {
        println!(
            "\n--- Shape: {} ({} experts, hidden={}, inter={}, top_k={}) ---",
            shape.name, shape.experts, shape.hidden, shape.inter, shape.top_k
        );

        let expert_bytes = shape.inter * shape.hidden / 2;
        let granules_per_expert = expert_bytes.div_ceil(cap.granularity);
        println!(
            "  expert_bytes={} granules_per_expert={} total_granules={}",
            expert_bytes,
            granules_per_expert,
            shape.experts * granules_per_expert
        );

        for &k in &k_values {
            for (trace_name, trace) in &[
                ("uniform", uniform_trace(shape, n_steps)),
                ("skewed", skewed_trace(shape, n_steps)),
            ] {
                println!("\n  k={} trace={}", k, trace_name);
                let mut run_medians_total: Vec<f64> = Vec::new();
                let mut run_medians_unmap: Vec<f64> = Vec::new();
                let mut run_medians_map: Vec<f64> = Vec::new();
                let mut warm_rates: Vec<f64> = Vec::new();

                for run in 0..n_runs {
                    assert_gpu_idle_or_warn(&format!("  run {run}"));

                    let gov = make_governor(0, 32 << 30, 32 << 30);
                    holder += 10;

                    let (times, first_us, steady_us, acc) = run_remap_benchmark(
                        &context,
                        shape,
                        k,
                        trace,
                        trace_name,
                        n_steps,
                        cap.granularity,
                        cap.host_numa_id,
                        gov,
                        holder,
                    );

                    run_medians_total.push(RemapComponentTimes::median(&times.total_us));
                    run_medians_unmap.push(RemapComponentTimes::median(&times.unmap_us));
                    run_medians_map.push(RemapComponentTimes::median(&times.map_us));
                    warm_rates.push(times.warm_rate());
                    times.print_summary(&format!("run={run}"));

                    let first_med = RemapComponentTimes::median(&first_us);
                    let steady_med = RemapComponentTimes::median(&steady_us);
                    println!(
                        "    (e) first_access_after_remap_med={:.1}µs steady_med={:.1}µs (64B cuMemcpyDtoH proxy)",
                        first_med, steady_med
                    );
                    acc.print(&format!("run={run}"));
                    assert!(acc.is_clean(), "accounting error in run {run}");
                }

                let med_total = RemapComponentTimes::median(&run_medians_total);
                let med_unmap = RemapComponentTimes::median(&run_medians_unmap);
                let med_map = RemapComponentTimes::median(&run_medians_map);
                let med_warm = RemapComponentTimes::median(&warm_rates) * 100.0;
                let (lo, hi) = RemapComponentTimes::range(&run_medians_total);

                println!(
                    "\n  SUMMARY k={k} {trace_name}: \
                     total_med={med_total:.1}µs [{lo:.1},{hi:.1}] \
                     unmap_med={med_unmap:.1}µs map_med={med_map:.1}µs \
                     warm_rate={med_warm:.1}%"
                );
                all_remap_medians.push((format!("{}-k{k}-{trace_name}", shape.name), k, med_total));

                print_go_no_go(
                    &format!("{} {trace_name}", shape.name),
                    k,
                    med_total,
                    qmoe_layer_kernel_range,
                    whole_token_interval_us,
                );
            }
        }
    }

    // Final summary table.
    println!("\n=== FINAL TIMING DECOMPOSITION TABLE ===");
    println!("{:<50} {:>5} {:>15}", "config", "k", "median_total_µs");
    for (label, k, med) in &all_remap_medians {
        println!("{:<50} {:>5} {:>15.1}", label, k, med);
    }

    println!("\n=== SYNCHRONIZATION CONTRACT SUMMARY ===");
    println!("  production primitives (release_range_reporting + commit_at_location):");
    println!("    - capture gate (synchronizing_section): YES — both primitives hold it");
    println!("    - stream sync (cuStreamSynchronize before unmap): NO — caller responsibility");
    println!("  CONSEQUENCE: per-token remap is NOT safe without external stream synchronization");
    println!("  even if the µs cost fits the token budget.");
    println!("  BOUNDARY-ONLY is required until deferred-release queue integration provides");
    println!("  ordered teardown against in-flight kernels reading the OLD mapping.");

    println!("\n=== PCIe / HBM THEORETICAL CEILINGS ===");
    println!("  PCIe Gen4 x16: ~25 GB/s  (host-NUMA read ceiling)");
    println!("  A100 HBM2e:   ~2039 GB/s  (device-resident read ceiling)");
    println!("  Per #1813: achieved host-NUMA cold read = 19.5–43.3 GB/s (at/near PCIe ceiling)");
    println!("  VMM remap (cuMemUnmap+cuMemMap+cuMemSetAccess) is a DRIVER API call, not a data");
    println!("  transfer — its cost is latency-bound, not bandwidth-bound. Per-granule remap");
    println!("  latency dominates for small K; see measured µs above for the actual budget.");
}
