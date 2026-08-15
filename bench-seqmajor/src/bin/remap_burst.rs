// Remap-burst microbenchmark (real CUDA VMM path, driver API).
//
// Answers the owner's item 2: remap frequency is amortized and negligible, but
// all KV buffers cross their granule boundary on the SAME decode step (they grow
// in lockstep with sequence length), so the cost arrives as a single-token
// latency SPIKE. This measures the real wall-clock cost of committing N granules
// in one burst on this hardware (create + map + set-access), which is exactly the
// extra time added to the decode step on which a granule boundary is crossed.
//
// N is the granule-floor unit count that crosses together:
//   qwen14b  head-major = layers*2*kv_heads = 768   seq-major = layers*2 = 96
//   qwen0.5b head-major = 96                          seq-major = 48
//
// Run from inside bench-seqmajor/ with the CUDA PATH prepended:
//   cargo run --release --bin remap_burst

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

fn granule(prop: &cu::CUmemAllocationProp, flag: cu::CUmemAllocationGranularity_flags) -> usize {
    let mut g = 0usize;
    unsafe {
        check(
            cu::cuMemGetAllocationGranularity(&mut g, prop, flag),
            "granularity",
        );
    }
    g
}

/// Commit `n` fresh granules into consecutive slots of the reserved range,
/// timing the whole burst. Returns (elapsed_ms, handles). Caller must release.
fn commit_burst(
    base: cu::CUdeviceptr,
    prop: &cu::CUmemAllocationProp,
    g: usize,
    dev: i32,
    n: usize,
) -> (f64, Vec<cu::CUmemGenericAllocationHandle>) {
    let mut handles: Vec<cu::CUmemGenericAllocationHandle> = Vec::with_capacity(n);
    let desc = cu::CUmemAccessDesc {
        location: cu::CUmemLocation {
            type_: cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: dev,
        },
        flags: cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    let t0 = Instant::now();
    unsafe {
        for i in 0..n {
            let mut h: cu::CUmemGenericAllocationHandle = 0;
            check(cu::cuMemCreate(&mut h, g, prop, 0), "cuMemCreate");
            let ptr = base + (i * g) as u64;
            check(cu::cuMemMap(ptr, g, 0, h, 0), "cuMemMap");
            check(cu::cuMemSetAccess(ptr, g, &desc, 1), "cuMemSetAccess");
            handles.push(h);
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    (ms, handles)
}

fn release(base: cu::CUdeviceptr, g: usize, handles: &[cu::CUmemGenericAllocationHandle]) {
    unsafe {
        for (i, h) in handles.iter().enumerate() {
            let ptr = base + (i * g) as u64;
            let _ = cu::cuMemUnmap(ptr, g);
            let _ = cu::cuMemRelease(*h);
        }
    }
}

fn main() {
    let _ctx: Arc<CudaContext> = CudaContext::new(0).expect("ctx");
    let dev = 0i32;
    let prop = make_prop(dev);
    let g = granule(
        &prop,
        cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
    );
    let gr = granule(
        &prop,
        cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    );
    println!(
        "granule: MINIMUM={} MiB  RECOMMENDED={} MiB (equal={})",
        g >> 20,
        gr >> 20,
        g == gr
    );

    let n_max = 768usize;
    let reserve = n_max * g;
    let mut base: cu::CUdeviceptr = 0;
    unsafe {
        check(
            cu::cuMemAddressReserve(&mut base, reserve, 0, 0, 0),
            "cuMemAddressReserve",
        );
    }
    println!(
        "reserved {} MiB VA for up to {} granules\n",
        reserve >> 20,
        n_max
    );

    // Per-granule baseline: median of committing 1 granule, many trials.
    let mut singles = Vec::new();
    for _ in 0..32 {
        let (ms, h) = commit_burst(base, &prop, g, dev, 1);
        singles.push(ms);
        release(base, g, &h);
    }
    singles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let per = singles[singles.len() / 2];
    println!("per-granule commit (median of 32): {:.1} us\n", per * 1e3);

    // Burst = all buffers crossing a boundary on the same decode step.
    println!(
        "{:<28} {:>6}  {:>10}  {:>12}",
        "scenario", "N", "burst(ms)", "us/granule"
    );
    let scenarios: &[(&str, usize)] = &[
        ("qwen0.5b seq-major", 48),
        ("qwen0.5b head-major", 96),
        ("qwen14b seq-major", 96),
        ("qwen14b head-major", 768),
    ];
    for (name, n) in scenarios {
        let mut times = Vec::new();
        for _ in 0..5 {
            let (ms, h) = commit_burst(base, &prop, g, dev, *n);
            times.push(ms);
            release(base, g, &h);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = times[times.len() / 2];
        println!(
            "{:<28} {:>6}  {:>10.2}  {:>12.1}",
            name,
            n,
            ms,
            ms * 1e3 / *n as f64
        );
    }

    unsafe {
        let _ = cu::cuMemAddressFree(base, reserve);
    }
    println!("\nInterpretation: burst = extra wall-clock added to the single decode");
    println!("step on which all KV buffers cross a granule boundary in lockstep.");
    println!("Amortized/token it is ~per_granule/crossover (<0.2% of a ~12 ms decode");
    println!("step); it lands as one spike unless commit is scheduled ahead of the");
    println!("write frontier.");
}
