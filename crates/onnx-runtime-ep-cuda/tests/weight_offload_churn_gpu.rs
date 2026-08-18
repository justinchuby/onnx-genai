#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::uninlined_format_args,
    clippy::cast_precision_loss
)]
//! GPU measurement: per-expert VMM residency **churn** for MoE expert paging.
//!
//! Motivation (WEIGHT_OFFLOAD / MEMORY_MANAGEMENT_MODEL_DESIGN): today
//! `bind_block_quantized_moe` pages the whole expert bank as ONE key, so a real
//! QMoE run reports whole-bank `reads_per_step ≈ 1.0` — dense-like — and the
//! measured router skew (granite-3.0-1b-a400m: top-8/32 experts carry ~45% of
//! read volume, Gini 0.334) is *invisible* through the paging layer. Making the
//! bank page **per-expert** would expose that skew, but at the cost of more,
//! smaller VMM regions — and VMM `cuMemMap`/`cuMemUnmap` churn is a known
//! binding limiter on this box. This benchmark measures that cost directly,
//! before any residency policy is attached, so the plumbing lands on its own
//! merits (correct + neutral) with the churn visible rather than hidden.
//!
//! Arms, all through the real `CudaWeightResidency` VMM path:
//!   A. whole-bank, one key, budget = full bank      (today's behaviour)
//!   B. per-expert keys, budget = full bank          (keying overhead, no churn)
//!   C. per-expert keys, budget = top-k experts      (real per-expert paging)
//!      · C-uniform : routed top-k picked uniformly (worst-case, no reuse)
//!      · C-skewed  : routed top-k with a measured hot core (real reuse)
//!
//! The routed sequences bracket reality: uniform is the churn ceiling; skewed
//! uses a hot core sized so the per-step hot-read share (~0.45-0.50) matches the
//! measured granite top-8 share, so its page-in reduction is the residency win
//! a policy could bank. Swept over expert byte sizes spanning the 2 MiB device
//! granule (#776): sub-granule experts cannot be mapped individually, which is
//! itself a finding for small MoE (granite int4 experts are ~0.75 MiB).
//!
//! Gated on a real CUDA device. Run pinned to a free GPU:
//!   cargo test -p onnx-runtime-ep-cuda --features gpu-tests \
//!     --test weight_offload_churn_gpu -- --nocapture

use onnx_runtime_ep_api::{
    ExternalMmapRegion, LazyWeight, MmapRegionSource, ResidentWeight, WeightHandleError,
};
use onnx_runtime_ep_cuda::{
    CudaExecutionProvider, global_offload_stats, reset_global_offload_stats,
};
use onnx_runtime_ir::DataType;
use std::time::Instant;

const GRANULE: u64 = 2 * 1024 * 1024; // 2 MiB VMM device granule (#776)

/// A host buffer standing in for an ONNX external-data mmap.
struct HostMmap {
    mapping_id: usize,
    bytes: Vec<u8>,
}

impl MmapRegionSource for HostMmap {
    fn region_bytes(&self, region: &ExternalMmapRegion) -> Result<&[u8], WeightHandleError> {
        if region.mapping_id != self.mapping_id {
            return Err(WeightHandleError::DeviceBinding(format!(
                "unknown mapping {}",
                region.mapping_id
            )));
        }
        let end = region
            .offset
            .checked_add(region.len)
            .ok_or_else(|| WeightHandleError::DeviceBinding("region overflow".into()))?;
        self.bytes
            .get(region.offset..end)
            .ok_or_else(|| WeightHandleError::DeviceBinding("region out of bounds".into()))
    }
}

/// Round a byte size up to the 2 MiB VMM granule.
fn round_granule(bytes: u64) -> u64 {
    bytes.div_ceil(GRANULE) * GRANULE
}

/// Build `num_experts` distinct expert weights, each `expert_bytes` large,
/// packed back-to-back in one mmap after a padding prefix, plus a lazy weight
/// per expert. All share one mapping so a single residency can page any of them.
fn expert_bank(num_experts: usize, expert_bytes: usize) -> (HostMmap, Vec<LazyWeight>) {
    let mapping_id = 7;
    let prefix = 256usize;
    let cols = 1024usize;
    let elems = expert_bytes / 4; // f32
    let rows = elems / cols;
    let real_bytes = rows * cols * 4;
    let mut backing = vec![0xABu8; prefix];
    let mut lazies = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let offset = backing.len();
        // Distinct byte pattern per expert (value proves correct region binding
        // if reused for parity; here it only needs to be real device traffic).
        let seed = (e as u32).wrapping_mul(2654435761);
        let mut bytes = Vec::with_capacity(real_bytes);
        let mut x = seed | 1;
        for _ in 0..real_bytes {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            bytes.push((x & 0xff) as u8);
        }
        backing.extend_from_slice(&bytes);
        let region = ExternalMmapRegion {
            mapping_id,
            offset,
            len: real_bytes,
        };
        let shape = vec![rows, cols];
        let resident_bytes = bytes.clone();
        let lazy =
            LazyWeight::block_quantized_moe(DataType::Float32, shape.clone(), vec![region], {
                let shape = shape.clone();
                move || {
                    ResidentWeight::new(DataType::Float32, shape.clone(), resident_bytes.clone())
                }
            })
            .unwrap();
        lazies.push(lazy);
    }
    (
        HostMmap {
            mapping_id,
            bytes: backing,
        },
        lazies,
    )
}

/// Deterministic xorshift for reproducible routed sequences.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// top-k distinct experts chosen uniformly (worst case: no cross-step reuse
/// beyond chance).
fn routed_uniform(rng: &mut Rng, num_experts: usize, top_k: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(top_k);
    while out.len() < top_k {
        let e = rng.below(num_experts) as u64;
        if !out.contains(&e) {
            out.push(e);
        }
    }
    out
}

/// top-k with a fixed hot core always selected, the rest drawn from the cold
/// tail. `hot` sized so per-step hot share = hot/top_k ≈ measured granite top-8
/// share (~0.45-0.50). Models the measured skew's real reuse.
fn routed_skewed(rng: &mut Rng, num_experts: usize, top_k: usize, hot: usize) -> Vec<u64> {
    let mut out: Vec<u64> = (0..hot as u64).collect();
    while out.len() < top_k {
        let e = (hot + rng.below(num_experts - hot)) as u64;
        if !out.contains(&e) {
            out.push(e);
        }
    }
    out
}

struct ArmResult {
    label: &'static str,
    page_ins: u64,
    hits: u64,
    evictions: u64,
    htod_mib: f64,
    htod_ms: f64,
    materialize_ms: f64,
    wall_ms: f64,
    peak_resident_mib: f64,
}

fn run_arm<S: MmapRegionSource>(
    label: &'static str,
    ep: &CudaExecutionProvider,
    budget_bytes: u64,
    lazies: &[LazyWeight],
    host: &S,
    routed: &[Vec<u64>],
) -> ArmResult {
    let residency = ep.weight_residency(budget_bytes);
    // Warm-up step (excluded from timing): first-touch page-ins are unavoidable
    // and not what churn measures; churn is steady-state per-step turnover.
    for &e in &routed[0] {
        let _ = residency.resident(e, &lazies[e as usize], host).unwrap();
    }
    reset_global_offload_stats();
    let base = residency.stats();
    let start = Instant::now();
    for step in routed.iter() {
        let mut held = Vec::with_capacity(step.len());
        for &e in step {
            held.push(residency.resident(e, &lazies[e as usize], host).unwrap());
        }
        // Drop all handles at end of step so the next step's misses may evict.
        drop(held);
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
    let g = global_offload_stats();
    let s = residency.stats();
    ArmResult {
        label,
        page_ins: s.page_ins - base.page_ins,
        hits: s.hits - base.hits,
        evictions: s.evictions - base.evictions,
        htod_mib: g.htod_bytes as f64 / (1024.0 * 1024.0),
        htod_ms: g.htod_ns as f64 / 1e6,
        materialize_ms: g.materialize_ns as f64 / 1e6,
        wall_ms,
        peak_resident_mib: s.peak_resident_bytes as f64 / (1024.0 * 1024.0),
    }
}

fn print_arm(steps: usize, r: &ArmResult) {
    let overhead_ms = (r.wall_ms - r.htod_ms - r.materialize_ms).max(0.0);
    println!(
        "  {:<26} page_ins={:>5} unmaps(evict)={:>5} hits={:>6} | \
         htod={:>7.1} MiB {:>7.2} ms | mat={:>7.2} ms | map/unmap+ovh={:>7.2} ms | \
         wall={:>8.2} ms ({:>6.3} ms/step) | peak_res={:>6.1} MiB",
        r.label,
        r.page_ins,
        r.evictions,
        r.hits,
        r.htod_mib,
        r.htod_ms,
        r.materialize_ms,
        overhead_ms,
        r.wall_ms,
        r.wall_ms / steps as f64,
        r.peak_resident_mib,
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn per_expert_paging_churn_measurement() {
    let ep = match CudaExecutionProvider::new_default() {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("skip: no CUDA GPU available ({e})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    const NUM_EXPERTS: usize = 32; // granite-3.0-1b-a400m routed_expert_count
    const TOP_K: usize = 8; // experts_per_token
    const HOT: usize = 4; // hot core -> per-step hot share 4/8 = 0.50 ~ measured
    const STEPS: usize = 200;

    // Expert byte sizes spanning the 2 MiB granule. Named regimes:
    //   0.75 MiB  granite int4 expert (sub-granule: NOT individually mappable)
    //   3.0  MiB  granite f16 expert  (this fixture; just over one granule)
    //   16.0 MiB  GLM-5.2 / DeepSeek-V4-class expert (multi-granule)
    let sizes_mib: [f64; 4] = [0.75, 2.0, 3.0, 16.0];

    // Deterministic routed sequences, shared across arms for a fair comparison.
    let mut rng_u = Rng(0x1234_5678_9abc_def0);
    let mut rng_s = Rng(0x0fed_cba9_8765_4321);
    let routed_u: Vec<Vec<u64>> = (0..STEPS)
        .map(|_| routed_uniform(&mut rng_u, NUM_EXPERTS, TOP_K))
        .collect();
    let routed_s: Vec<Vec<u64>> = (0..STEPS)
        .map(|_| routed_skewed(&mut rng_s, NUM_EXPERTS, TOP_K, HOT))
        .collect();

    println!("\n=== per-expert MoE paging churn (RTX 4060 8 GB, VMM 2 MiB granule) ===");
    println!(
        "  experts={} top_k={} hot_core={} steps={}  (granite-3.0-1b-a400m routing shape)",
        NUM_EXPERTS, TOP_K, HOT, STEPS
    );

    for &mib in &sizes_mib {
        let expert_bytes = (mib * 1024.0 * 1024.0) as usize;
        let granule_rounded = round_granule(expert_bytes as u64);
        let sub_granule = (expert_bytes as u64) < GRANULE;
        let (host, lazies) = expert_bank(NUM_EXPERTS, expert_bytes);

        println!(
            "\n-- expert = {:.2} MiB (rounds to {:.2} MiB / {} granule{}){} --",
            mib,
            granule_rounded as f64 / (1024.0 * 1024.0),
            granule_rounded / GRANULE,
            if granule_rounded / GRANULE == 1 { "" } else { "s" },
            if sub_granule {
                "  [SUB-GRANULE: cannot be mapped individually at VMM granularity]"
            } else {
                ""
            }
        );

        let full_budget = NUM_EXPERTS as u64 * granule_rounded;
        let topk_budget = TOP_K as u64 * granule_rounded;

        // A. whole-bank, one key, budget = full bank (today's behaviour). Each
        //    step touches the single bank key -> all hits after warm-up.
        let bank_bytes = NUM_EXPERTS * expert_bytes;
        let (bank_host, bank_lazy) = {
            let (h, mut ls) = expert_bank(1, bank_bytes);
            (h, ls.remove(0))
        };
        let bank_routed: Vec<Vec<u64>> = (0..STEPS).map(|_| vec![0u64]).collect();
        let a = run_arm(
            "A whole-bank (1 key)",
            &ep,
            round_granule(bank_bytes as u64),
            std::slice::from_ref(&bank_lazy),
            &bank_host,
            &bank_routed,
        );
        print_arm(STEPS, &a);

        // B. per-expert keys, budget = full bank (no eviction; keying overhead).
        let b = run_arm(
            "B per-expert, full budget",
            &ep,
            full_budget,
            &lazies,
            &host,
            &routed_s,
        );
        print_arm(STEPS, &b);

        // C-uniform. per-expert keys, budget = top-k, uniform routing (ceiling).
        let cu = run_arm(
            "C per-expert@topk uniform",
            &ep,
            topk_budget,
            &lazies,
            &host,
            &routed_u,
        );
        print_arm(STEPS, &cu);

        // C-skewed. per-expert keys, budget = top-k, measured hot core (real reuse).
        let cs = run_arm(
            "C per-expert@topk skewed",
            &ep,
            topk_budget,
            &lazies,
            &host,
            &routed_s,
        );
        print_arm(STEPS, &cs);

        // Derived skew win: page-in reduction skewed vs uniform.
        if cu.page_ins > 0 {
            let saved = 1.0 - (cs.page_ins as f64 / cu.page_ins as f64);
            println!(
                "  -> skew reduces page-ins by {:.1}% ({} -> {} over {} steps); \
                 hot core stays resident",
                saved * 100.0,
                cu.page_ins,
                cs.page_ins,
                STEPS
            );
        }
    }

    println!("\n=== end churn measurement ===\n");
}
