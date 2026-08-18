// Oversubscription-cliff characterization for the VMM map/unmap path (#1295).
//
// The streaming-regime batch-N ceiling (N_max ~4-5 on 8 GB) is reported to be
// VMM map/unmap churn: `vram_free_ms` explodes 9.7 s -> 90 s once
// `mapped_physical` (8.39 GB) exceeds physical VRAM (8.19 GB). This binary
// isolates, on the real driver API, WHICH op costs the time and WHETHER the
// explosion is a function of oversubscription (driver/WDDM eviction) or of our
// own bookkeeping / span count.
//
// It times cuMemCreate / cuMemMap / cuMemSetAccess / touch(cuMemsetD8) /
// cuMemUnmap / cuMemRelease separately, as a function of how full VRAM is, and
// runs a steady per-step churn loop that mirrors decode (hold a resident set,
// map+touch a working chunk, unmap+release it) at fill levels below and above
// physical VRAM.
//
// Run from bench-seqmajor/ with the CUDA runtime on PATH:
//   cargo run --release --bin churn_cliff

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys as cu;
use cudarc::driver::CudaContext;

unsafe fn check(r: cu::CUresult, what: &str) {
    if r != cu::CUresult::CUDA_SUCCESS {
        panic!("{what} failed: {r:?}");
    }
}

fn make_prop(dev: i32) -> cu::CUmemAllocationProp {
    // SAFETY: zero-init a POD prop, then set the fields the driver requires.
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop
}

fn granule(prop: &cu::CUmemAllocationProp) -> usize {
    let mut g = 0usize;
    unsafe {
        check(
            cu::cuMemGetAllocationGranularity(
                &mut g,
                prop,
                cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "granularity",
        );
    }
    g
}

fn access_desc(dev: i32) -> cu::CUmemAccessDesc {
    cu::CUmemAccessDesc {
        location: cu::CUmemLocation {
            type_: cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: dev,
        },
        flags: cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    }
}

fn free_vram_mib() -> usize {
    let mut free = 0usize;
    let mut total = 0usize;
    unsafe {
        let _ = cu::cuMemGetInfo_v2(&mut free, &mut total);
    }
    free >> 20
}

/// One committed granule: its handle and the VA it is mapped at.
struct Granule {
    handle: cu::CUmemGenericAllocationHandle,
    ptr: cu::CUdeviceptr,
}

/// Create + map + set-access + optionally touch one granule at `ptr`.
/// Returns (create_us, map_us, setaccess_us, touch_us).
fn commit_one(
    ptr: cu::CUdeviceptr,
    prop: &cu::CUmemAllocationProp,
    g: usize,
    desc: &cu::CUmemAccessDesc,
    touch: bool,
) -> Option<(Granule, f64, f64, f64, f64)> {
    unsafe {
        let mut h: cu::CUmemGenericAllocationHandle = 0;
        let t0 = Instant::now();
        let cr = cu::cuMemCreate(&mut h, g, prop, 0);
        if cr != cu::CUresult::CUDA_SUCCESS {
            eprintln!("cuMemCreate refused at this fill level: {cr:?}");
            return None;
        }
        let t1 = Instant::now();
        check(cu::cuMemMap(ptr, g, 0, h, 0), "cuMemMap");
        let t2 = Instant::now();
        check(cu::cuMemSetAccess(ptr, g, desc, 1), "cuMemSetAccess");
        let t3 = Instant::now();
        if touch {
            check(cu::cuMemsetD8_v2(ptr, 0xA5, g), "cuMemsetD8");
            check(cu::cuCtxSynchronize(), "cuCtxSynchronize");
        }
        let t4 = Instant::now();
        Some((
            Granule { handle: h, ptr },
            (t1 - t0).as_secs_f64() * 1e6,
            (t2 - t1).as_secs_f64() * 1e6,
            (t3 - t2).as_secs_f64() * 1e6,
            (t4 - t3).as_secs_f64() * 1e6,
        ))
    }
}

/// Unmap + release one granule. Returns (unmap_us, release_us).
fn free_one(gr: &Granule, g: usize) -> (f64, f64) {
    unsafe {
        let t0 = Instant::now();
        check(cu::cuMemUnmap(gr.ptr, g), "cuMemUnmap");
        let t1 = Instant::now();
        check(cu::cuMemRelease(gr.handle), "cuMemRelease");
        let t2 = Instant::now();
        (
            (t1 - t0).as_secs_f64() * 1e6,
            (t2 - t1).as_secs_f64() * 1e6,
        )
    }
}

struct Bucket {
    create: f64,
    map: f64,
    set: f64,
    touch: f64,
    unmap: f64,
    release: f64,
    n: usize,
}
impl Bucket {
    fn new() -> Self {
        Bucket { create: 0.0, map: 0.0, set: 0.0, touch: 0.0, unmap: 0.0, release: 0.0, n: 0 }
    }
}

fn main() {
    let _ctx: Arc<CudaContext> = CudaContext::new(0).expect("ctx");
    let dev = 0i32;
    let prop = make_prop(dev);
    let desc = access_desc(dev);
    let g = granule(&prop);
    let (mut free0, mut total0) = (0usize, 0usize);
    unsafe {
        let _ = cu::cuMemGetInfo_v2(&mut free0, &mut total0);
    }
    let total_mib = total0 >> 20;
    let gran_per_gib = (1usize << 30) / g;
    println!(
        "granule={} MiB  VRAM total={} MiB free={} MiB  granules/GiB={}",
        g >> 20,
        total_mib,
        free0 >> 20,
        gran_per_gib
    );

    // Reserve ~11 GiB of VA (physical is only ~8 GiB) so we can oversubscribe.
    let n_max = (11usize << 30) / g; // 5632 granules
    let reserve = n_max * g;
    let mut base: cu::CUdeviceptr = 0;
    unsafe {
        check(
            cu::cuMemAddressReserve(&mut base, reserve, 0, 0, 0),
            "cuMemAddressReserve",
        );
    }
    println!("reserved {} MiB VA ({} granules)\n", reserve >> 20, n_max);

    // ---- Phase 1+2: commit up to ~1.03x physical VRAM, touching every page,
    // bucketing per-op cost by VRAM-fill decile. Then free all, same bucketing.
    // The bucket key is "how full was VRAM when this granule was committed".
    let target_mib = (total_mib as f64 * 1.03) as usize;
    let target_granules = (target_mib << 20) / g;
    println!(
        "PHASE 1 (commit + touch to {} MiB = {} granules, ~1.03x VRAM):",
        target_granules * (g >> 20),
        target_granules
    );
    println!(
        "  fill%    create_us  map_us  setacc_us  touch_us   free_MiB(after)"
    );
    let mut granules: Vec<Granule> = Vec::with_capacity(target_granules);
    // 10 deciles of physical VRAM plus an over-subscription bucket.
    let mut commit_buckets: Vec<Bucket> = (0..12).map(|_| Bucket::new()).collect();
    for i in 0..target_granules {
        let ptr = base + (i * g) as u64;
        let Some((gr, c, m, s, t)) = commit_one(ptr, &prop, g, &desc, true) else {
            println!("  -> commit stopped at {} MiB committed (driver refused)", i * (g >> 20));
            break;
        };
        let committed_mib = (i + 1) * (g >> 20);
        let fill = committed_mib as f64 / total_mib as f64;
        let b = ((fill * 10.0) as usize).min(11);
        commit_buckets[b].create += c;
        commit_buckets[b].map += m;
        commit_buckets[b].set += s;
        commit_buckets[b].touch += t;
        commit_buckets[b].n += 1;
        granules.push(gr);
        // Print a sampled line roughly every ~256 MiB and near the cliff.
        if (i + 1) % 128 == 0 || fill > 0.98 {
            println!(
                "  {:>5.2}    {:>8.1}  {:>6.1}  {:>8.1}  {:>8.1}   {:>8}",
                fill * 100.0, c, m, s, t, free_vram_mib()
            );
        }
    }
    println!("\n  per-op AVERAGE by VRAM-fill decile (us/granule):");
    println!("  fill-decile   n   create   map    setacc   touch");
    for (idx, b) in commit_buckets.iter().enumerate() {
        if b.n == 0 {
            continue;
        }
        let n = b.n as f64;
        println!(
            "   {:>3}-{:>3}%  {:>4}  {:>6.1}  {:>6.1}  {:>6.1}  {:>7.1}",
            idx * 10,
            idx * 10 + 10,
            b.n,
            b.create / n,
            b.map / n,
            b.set / n,
            b.touch / n
        );
    }

    // ---- Phase 2: free all, bucket the free cost by the fill level at the
    // moment of freeing (freeing from the top down, so fill decreases).
    println!("\nPHASE 2 (unmap + release all, top-down):");
    let mut free_buckets: Vec<Bucket> = (0..12).map(|_| Bucket::new()).collect();
    let count = granules.len();
    for (k, gr) in granules.iter().enumerate().rev() {
        // fill level BEFORE this free
        let remaining_mib = (k + 1) * (g >> 20);
        let fill = remaining_mib as f64 / total_mib as f64;
        let b = ((fill * 10.0) as usize).min(11);
        let (u, r) = free_one(gr, g);
        free_buckets[b].unmap += u;
        free_buckets[b].release += r;
        free_buckets[b].n += 1;
        let _ = count;
    }
    granules.clear();
    println!("  per-op AVERAGE by VRAM-fill decile at time of free (us/granule):");
    println!("  fill-decile   n   unmap   release");
    for (idx, b) in free_buckets.iter().enumerate() {
        if b.n == 0 {
            continue;
        }
        let n = b.n as f64;
        println!(
            "   {:>3}-{:>3}%  {:>4}  {:>6.1}  {:>7.1}",
            idx * 10,
            idx * 10 + 10,
            b.n,
            b.unmap / n,
            b.release / n
        );
    }

    // ---- Phase 3: steady per-step churn (mirrors decode). Hold `resident`
    // granules touched-resident, then repeatedly map+touch+unmap+release a
    // `work` chunk. Do it at three fill levels: comfortably under, near, and
    // over physical VRAM. This is the money experiment: does per-cycle churn
    // explode only when resident+work exceeds VRAM?
    println!("\nPHASE 3 (steady churn: hold R resident, churn W per cycle):");
    println!("  scenario                 R_MiB  W_MiB  peak/VRAM  cyc  map+touch_ms  unmap+rel_ms");
    let work = 96usize; // ~= qwen14b seq-major KV granules crossing together (192 MiB)
    let phys_gran = (total_mib << 20) / g;
    let scenarios: &[(&str, f64)] = &[
        ("under (0.80x VRAM peak)", 0.80),
        ("near  (0.97x VRAM peak)", 0.97),
        ("over  (1.03x VRAM peak)", 1.03),
    ];
    for (name, peak_frac) in scenarios {
        let peak_gran = ((phys_gran as f64) * peak_frac) as usize;
        let resident = peak_gran.saturating_sub(work);
        // Build the resident set (touched).
        let mut res: Vec<Granule> = Vec::with_capacity(resident);
        for i in 0..resident {
            let ptr = base + (i * g) as u64;
            let Some((gr, _, _, _, _)) = commit_one(ptr, &prop, g, &desc, true) else {
                break;
            };
            res.push(gr);
        }
        // Churn `work` granules just above the resident set, for several cycles.
        let cycles = 8usize;
        let mut map_touch_ms = Vec::new();
        let mut unmap_rel_ms = Vec::new();
        for _ in 0..cycles {
            let t0 = Instant::now();
            let mut chunk: Vec<Granule> = Vec::with_capacity(work);
            for j in 0..work {
                let ptr = base + ((resident + j) * g) as u64;
                let Some((gr, _, _, _, _)) = commit_one(ptr, &prop, g, &desc, true) else {
                    break;
                };
                chunk.push(gr);
            }
            let t1 = Instant::now();
            for gr in &chunk {
                let _ = free_one(gr, g);
            }
            let t2 = Instant::now();
            map_touch_ms.push((t1 - t0).as_secs_f64() * 1e3);
            unmap_rel_ms.push((t2 - t1).as_secs_f64() * 1e3);
        }
        map_touch_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unmap_rel_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = |v: &Vec<f64>| v[v.len() / 2];
        println!(
            "  {:<24} {:>5}  {:>5}  {:>7.2}x  {:>3}  {:>11.2}  {:>11.2}",
            name,
            resident * (g >> 20),
            work * (g >> 20),
            *peak_frac,
            cycles,
            med(&map_touch_ms),
            med(&unmap_rel_ms)
        );
        // Tear down the resident set before the next scenario.
        for gr in &res {
            let _ = free_one(gr, g);
        }
        res.clear();
    }

    unsafe {
        let _ = cu::cuMemAddressFree(base, reserve);
    }
    println!("\nDone.");
}
