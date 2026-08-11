//! Is the 2 MiB granule forced, and if a finer granule is available does the
//! #759 dummy-page pattern still work there — and what does the finer granule
//! cost? The crossover between fixed-stride+dummy and bucket growth is
//! `granule / (head_dim x elem_bytes)` (see
//! `dummy_fill_and_crossover::fixed_stride_plus_dummy_crossover_vs_bucket_growth`),
//! so it scales *linearly with the granule*. The granule is therefore the
//! single highest-value lever in the whole design: at the 2 MiB RECOMMENDED
//! granule the crossover is 8K-16K tokens (fixed-stride+dummy loses in most
//! serving windows); at a 64 KiB MINIMUM granule it would collapse to a few
//! hundred tokens (fixed-stride+dummy wins almost always) — *if* allocating and
//! mapping at the minimum granule actually works for our access pattern and its
//! 32x-more-mappings cost is bearable.
//!
//! This binary answers that, driver-level, with no engine or KV integration.
//! **The headline measured result on this device: MINIMUM == RECOMMENDED ==
//! 2 MiB — the finer granule this round hoped to exploit is not exposed here, so
//! the crossover lever cannot be pulled on this box.** The tests still exercise
//! and price whatever granule pair the driver reports, so the same binary
//! yields a real trade-off on a device that does expose a finer minimum:
//!
//! * [`report_both_allocation_granularities`] — queries and prints BOTH
//!   `CU_MEM_ALLOC_GRANULARITY_MINIMUM` and `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED`
//!   for the exact allocation properties the production allocator uses. The
//!   2 MiB figure measured so far is the RECOMMENDED value; this shows MINIMUM
//!   is *also* 2 MiB here, rather than assuming a finer granule is available.
//! * [`minimum_granularity_dummy_pattern_works_including_captured_replay`] —
//!   builds the #759 primitive at the *minimum* granule (one dummy handle
//!   aliased across many fine tail granules) and confirms a full-padded read
//!   and a captured-graph device-to-device copy that reads into the dummy tail
//!   both succeed and read back correctly. If the fine granule could not be
//!   mapped or replayed, the crossover lever would be unavailable.
//! * [`granularity_cost_minimum_vs_recommended`] — measures the price of the
//!   finer granule: per-mapping cost, total cost to commit a fixed prefill
//!   region, per-growth-step cost, and captured-copy read throughput over an
//!   identically-sized region backed at each granule (to expose TLB pressure).
//!   32x more mappings means 32x more `cuMemMap`/`cuMemUnmap` calls; this is the
//!   trade that decides whether the minimum granule is usable.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

fn require_cuda() -> Arc<CudaContext> {
    match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "CUDA VMM test requires a CUDA driver; CPU-only runs must leave this test ignored: {error}"
        ),
    }
}

fn check(call: &'static str, result: cu::CUresult) {
    assert_eq!(result, cu::CUresult::CUDA_SUCCESS, "{call}: {result:?}");
}

fn allocation_prop(device_ordinal: i32) -> cu::CUmemAllocationProp {
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_ordinal;
    prop
}

fn query_granularity(device_ordinal: i32, flag: cu::CUmemAllocationGranularity_flags) -> usize {
    let prop = allocation_prop(device_ordinal);
    let mut granularity = 0usize;
    let result = unsafe { cu::cuMemGetAllocationGranularity(&mut granularity, &prop, flag) };
    check("cuMemGetAllocationGranularity", result);
    assert_ne!(granularity, 0, "CUDA reported zero VMM granularity");
    granularity
}

fn minimum_granularity(device_ordinal: i32) -> usize {
    query_granularity(
        device_ordinal,
        cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
    )
}

fn recommended_granularity(device_ordinal: i32) -> usize {
    query_granularity(
        device_ordinal,
        cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    )
}

fn create_stream() -> cu::CUstream {
    let mut stream = std::ptr::null_mut();
    let result = unsafe {
        cu::cuStreamCreate(
            &mut stream,
            cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
        )
    };
    check("cuStreamCreate", result);
    stream
}

fn destroy_stream(stream: cu::CUstream) {
    if !stream.is_null() {
        let _ = unsafe { cu::cuStreamDestroy_v2(stream) };
    }
}

fn reserve(size: usize) -> cu::CUdeviceptr {
    let mut base = 0;
    let result = unsafe { cu::cuMemAddressReserve(&mut base, size, 0, 0, 0) };
    check("cuMemAddressReserve", result);
    base
}

fn free_reservation(base: cu::CUdeviceptr, size: usize) {
    if base != 0 {
        let _ = unsafe { cu::cuMemAddressFree(base, size) };
    }
}

fn create_handle(device_ordinal: i32, size: usize) -> cu::CUmemGenericAllocationHandle {
    let prop = allocation_prop(device_ordinal);
    let mut handle = 0;
    let result = unsafe { cu::cuMemCreate(&mut handle, size, &prop, 0) };
    check("cuMemCreate", result);
    handle
}

fn release_handle(handle: cu::CUmemGenericAllocationHandle) {
    let _ = unsafe { cu::cuMemRelease(handle) };
}

fn set_access(device_ordinal: i32, address: cu::CUdeviceptr, size: usize) {
    let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_ordinal;
    access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    let result = unsafe { cu::cuMemSetAccess(address, size, &access, 1) };
    check("cuMemSetAccess", result);
}

fn map_handle(
    device_ordinal: i32,
    address: cu::CUdeviceptr,
    size: usize,
    handle: cu::CUmemGenericAllocationHandle,
) {
    let result = unsafe { cu::cuMemMap(address, size, 0, handle, 0) };
    check("cuMemMap", result);
    set_access(device_ordinal, address, size);
}

fn unmap(address: cu::CUdeviceptr, size: usize) {
    let _ = unsafe { cu::cuMemUnmap(address, size) };
}

fn write_host(address: cu::CUdeviceptr, value: u8, len: usize) {
    let bytes = vec![value; len];
    let result = unsafe { cu::cuMemcpyHtoD_v2(address, bytes.as_ptr().cast(), bytes.len()) };
    check("cuMemcpyHtoD_v2", result);
}

fn read_host(address: cu::CUdeviceptr, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    let result = unsafe { cu::cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(), address, bytes.len()) };
    check("cuMemcpyDtoH_v2", result);
    bytes
}

/// A captured device-to-device copy of `len` bytes — the decode read hot path
/// in miniature, replayed to measure steady-state read throughput.
struct CapturedCopy {
    graph: cu::CUgraph,
    exec: cu::CUgraphExec,
}

impl CapturedCopy {
    fn capture(
        stream: cu::CUstream,
        dst: cu::CUdeviceptr,
        src: cu::CUdeviceptr,
        len: usize,
    ) -> Self {
        let mut graph = std::ptr::null_mut();
        let mut exec = std::ptr::null_mut();
        let begin = unsafe {
            cu::cuStreamBeginCapture_v2(
                stream,
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        };
        check("cuStreamBeginCapture_v2", begin);
        let record = unsafe { cu::cuMemcpyDtoDAsync_v2(dst, src, len, stream) };
        let end = unsafe { cu::cuStreamEndCapture(stream, &mut graph) };
        check("cuMemcpyDtoDAsync_v2 during capture", record);
        check("cuStreamEndCapture", end);
        assert!(!graph.is_null(), "cuStreamEndCapture returned a null graph");
        let inst = unsafe { cu::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        check("cuGraphInstantiateWithFlags", inst);
        assert!(!exec.is_null(), "null graph exec");
        Self { graph, exec }
    }

    fn launch(&self, stream: cu::CUstream) -> cu::CUresult {
        unsafe { cu::cuGraphLaunch(self.exec, stream) }
    }
}

impl Drop for CapturedCopy {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            let _ = unsafe { cu::cuGraphExecDestroy(self.exec) };
        }
        if !self.graph.is_null() {
            let _ = unsafe { cu::cuGraphDestroy(self.graph) };
        }
    }
}

/// Report BOTH granularities for the exact properties the production allocator
/// uses. Do not assume the 2 MiB seen so far is forced — MINIMUM may be far
/// smaller, and the crossover scales linearly with whichever granule the design
/// actually maps at.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn report_both_allocation_granularities() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;

    let minimum = minimum_granularity(device);
    let recommended = recommended_granularity(device);

    // Crossover in tokens for the two head_dim/dtype combinations the probe
    // cares about, at each granule: crossover = granule / (head_dim x elem).
    let crossover = |granule: usize, head_dim: usize, elem: usize| granule / (head_dim * elem);

    eprintln!(
        "Granularity on this device (PINNED / DEVICE): MINIMUM = {} KiB, RECOMMENDED = {} KiB \
         (ratio {}x). Crossover = granule/(head_dim x elem):\n  \
         head_dim 64 fp16:  MIN {} tok, REC {} tok\n  \
         head_dim 128 fp16: MIN {} tok, REC {} tok",
        minimum / 1024,
        recommended / 1024,
        recommended / minimum.max(1),
        crossover(minimum, 64, 2),
        crossover(recommended, 64, 2),
        crossover(minimum, 128, 2),
        crossover(recommended, 128, 2),
    );

    assert!(
        minimum <= recommended,
        "MINIMUM granularity ({minimum}) must not exceed RECOMMENDED ({recommended})"
    );
    assert!(
        minimum.is_power_of_two(),
        "granule should be a power of two"
    );
}

/// The #759 dummy pattern at the MINIMUM granule: one dummy handle aliased
/// across many fine tail granules, a full-padded read, and a captured-graph
/// read into the dummy tail. If this fails, the fine granule cannot carry the
/// design and the crossover lever is unavailable; if it passes, the crossover
/// can be pulled down to the minimum granule's value.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn minimum_granularity_dummy_pattern_works_including_captured_replay() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = minimum_granularity(device);
    let stream = create_stream();

    // Enough fine granules that aliasing and a captured read genuinely cross
    // many mappings (at 64 KiB this is 4 MiB across 64 tail granules).
    const TOTAL_GRANULES: usize = 96;
    const LIVE_GRANULES: usize = 32;
    let total = granule * TOTAL_GRANULES;
    let live = granule * LIVE_GRANULES;
    const MARKER: u8 = 0x5a;

    let base = reserve(total);
    let sink = reserve(total);

    let mut live_handles = Vec::new();
    for g in 0..LIVE_GRANULES {
        let handle = create_handle(device, granule);
        map_handle(device, base + (g * granule) as u64, granule, handle);
        live_handles.push(handle);
    }
    let dummy = create_handle(device, granule);
    for g in LIVE_GRANULES..TOTAL_GRANULES {
        map_handle(device, base + (g * granule) as u64, granule, dummy);
    }
    let mut sink_handles = Vec::new();
    for g in 0..TOTAL_GRANULES {
        let handle = create_handle(device, granule);
        map_handle(device, sink + (g * granule) as u64, granule, handle);
        sink_handles.push(handle);
    }

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        write_host(base, 0x11, live);
        // One write through one fine tail granule must be visible at every
        // aliased tail granule.
        write_host(base + live as u64, MARKER, granule);

        let mut host = vec![0u8; total];
        let padded = unsafe { cu::cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), base, total) };
        assert_eq!(
            padded,
            cu::CUresult::CUDA_SUCCESS,
            "minimum-granularity dummy tail must make the full padded read succeed; got {padded:?}"
        );
        assert!(
            host[..live].iter().all(|&b| b == 0x11),
            "live prefix must survive minimum-granularity mapping"
        );
        assert!(
            host[live..].iter().all(|&b| b == MARKER),
            "every fine dummy-backed tail byte must alias the one physical granule"
        );

        let copy = CapturedCopy::capture(stream, sink, base, total);
        let launch = copy.launch(stream);
        check("cuGraphLaunch minimum-granularity read", launch);
        let sync = unsafe { cu::cuStreamSynchronize(stream) };
        assert_eq!(
            sync,
            cu::CUresult::CUDA_SUCCESS,
            "a captured read across many fine dummy-backed granules must replay; got {sync:?}"
        );
        let copied = read_host(sink, total);
        assert!(
            copied[..live].iter().all(|&b| b == 0x11)
                && copied[live..].iter().all(|&b| b == MARKER),
            "captured copy across the fine dummy tail must reproduce prefix and aliased tail"
        );
    }));

    for g in 0..TOTAL_GRANULES {
        unmap(sink + (g * granule) as u64, granule);
    }
    for handle in sink_handles {
        release_handle(handle);
    }
    for g in LIVE_GRANULES..TOTAL_GRANULES {
        unmap(base + (g * granule) as u64, granule);
    }
    release_handle(dummy);
    for (g, handle) in live_handles.into_iter().enumerate() {
        unmap(base + (g * granule) as u64, granule);
        release_handle(handle);
    }
    free_reservation(sink, total);
    free_reservation(base, total);
    destroy_stream(stream);
    result.unwrap();
}

/// Fully back `region_bytes` of a reservation with real handles at `granule`,
/// timing the whole commit; return the base, the handles, and the elapsed time.
/// This is the prefill-commit cost for one granule choice.
fn commit_region(
    device: i32,
    granule: usize,
    region_bytes: usize,
) -> (
    cu::CUdeviceptr,
    Vec<cu::CUmemGenericAllocationHandle>,
    std::time::Duration,
) {
    let count = region_bytes / granule;
    let base = reserve(region_bytes);
    let mut handles = Vec::with_capacity(count);
    let start = Instant::now();
    for g in 0..count {
        let handle = create_handle(device, granule);
        map_handle(device, base + (g * granule) as u64, granule, handle);
        handles.push(handle);
    }
    let elapsed = start.elapsed();
    (base, handles, elapsed)
}

fn tear_down_region(
    base: cu::CUdeviceptr,
    granule: usize,
    handles: Vec<cu::CUmemGenericAllocationHandle>,
) {
    let region = granule * handles.len();
    for (g, handle) in handles.into_iter().enumerate() {
        unmap(base + (g * granule) as u64, granule);
        release_handle(handle);
    }
    free_reservation(base, region);
}

/// Time `replays` captured device-to-device copies of the whole region and
/// return bytes/second, a read-throughput proxy sensitive to TLB pressure from
/// many small mappings.
fn measure_read_throughput(
    stream: cu::CUstream,
    dst: cu::CUdeviceptr,
    src: cu::CUdeviceptr,
    region_bytes: usize,
    replays: usize,
) -> f64 {
    let copy = CapturedCopy::capture(stream, dst, src, region_bytes);
    // Warm up.
    check("cuGraphLaunch warmup", copy.launch(stream));
    check("cuStreamSynchronize warmup", unsafe {
        cu::cuStreamSynchronize(stream)
    });
    let start = Instant::now();
    for _ in 0..replays {
        check("cuGraphLaunch throughput", copy.launch(stream));
    }
    check("cuStreamSynchronize throughput", unsafe {
        cu::cuStreamSynchronize(stream)
    });
    let elapsed = start.elapsed().as_secs_f64();
    (region_bytes as f64 * replays as f64) / elapsed
}

/// The cost of the finer granule: per-mapping cost, prefill-commit cost,
/// per-growth-step cost, and read throughput — measured at MINIMUM and
/// RECOMMENDED over an identically-sized region. This is the trade that decides
/// whether pulling the crossover down with the minimum granule is worth it.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn granularity_cost_minimum_vs_recommended() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let minimum = minimum_granularity(device);
    let recommended = recommended_granularity(device);
    let stream = create_stream();

    // One fixed region size backed at each granule so the comparison is
    // apples-to-apples; 32 MiB is a realistic small-prefill KV slice and yields
    // many fine mappings (512 at 64 KiB) versus few coarse ones (16 at 2 MiB).
    let region = 32 * 1024 * 1024usize;
    const REPLAYS: usize = 200;

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Warm the allocator (first cuMemCreate on a context is disproportionate).
        {
            let (b, h, _) = commit_region(device, recommended, recommended);
            tear_down_region(b, recommended, h);
        }

        let (rec_base, rec_handles, rec_commit) = commit_region(device, recommended, region);
        let rec_maps = rec_handles.len();
        let rec_sink = reserve(region);
        let mut rec_sink_handles = Vec::new();
        for g in 0..rec_maps {
            let handle = create_handle(device, recommended);
            map_handle(
                device,
                rec_sink + (g * recommended) as u64,
                recommended,
                handle,
            );
            rec_sink_handles.push(handle);
        }
        write_host(rec_base, 0x24, region);
        let rec_bw = measure_read_throughput(stream, rec_sink, rec_base, region, REPLAYS);

        let (min_base, min_handles, min_commit) = commit_region(device, minimum, region);
        let min_maps = min_handles.len();
        let min_sink = reserve(region);
        let mut min_sink_handles = Vec::new();
        for g in 0..min_maps {
            let handle = create_handle(device, minimum);
            map_handle(device, min_sink + (g * minimum) as u64, minimum, handle);
            min_sink_handles.push(handle);
        }
        write_host(min_base, 0x24, region);
        let min_bw = measure_read_throughput(stream, min_sink, min_base, region, REPLAYS);

        let rec_per_map = rec_commit.as_secs_f64() * 1e6 / rec_maps as f64;
        let min_per_map = min_commit.as_secs_f64() * 1e6 / min_maps as f64;

        // Per-growth-step cost under a fixed full-context stride: 96 head-stripe
        // objects, each needing one map at the step's active granule. At the
        // minimum granule the same 96 objects still need 96 maps per step (the
        // stride is per-object, not per-byte), so per-step cost tracks
        // per-map x 96 at whichever granule the design commits.
        const OBJECTS: usize = 96;
        let rec_step_us = rec_per_map * OBJECTS as f64;
        let min_step_us = min_per_map * OBJECTS as f64;

        eprintln!(
            "Granularity cost over a {} MiB region:\n  \
             RECOMMENDED {} KiB: {} maps, commit {:.1} us total = {:.2} us/map; \
             read {:.1} GiB/s; {OBJECTS}-object growth step ~= {:.1} us.\n  \
             MINIMUM     {} KiB: {} maps, commit {:.1} us total = {:.2} us/map; \
             read {:.1} GiB/s; {OBJECTS}-object growth step ~= {:.1} us.\n  \
             Finer granule = {}x more mappings; commit {:.1}x, throughput {:.2}x \
             (>1 means MIN is slower/faster to read).",
            region / (1024 * 1024),
            recommended / 1024,
            rec_maps,
            rec_commit.as_secs_f64() * 1e6,
            rec_per_map,
            rec_bw / (1024.0 * 1024.0 * 1024.0),
            rec_step_us,
            minimum / 1024,
            min_maps,
            min_commit.as_secs_f64() * 1e6,
            min_per_map,
            min_bw / (1024.0 * 1024.0 * 1024.0),
            min_step_us,
            (recommended / minimum.max(1)).max(1),
            min_commit.as_secs_f64() / rec_commit.as_secs_f64().max(1e-9),
            rec_bw / min_bw.max(1.0),
        );

        assert!(
            rec_bw > 0.0 && min_bw > 0.0,
            "throughput must be measurable"
        );
        assert!(
            rec_commit.as_nanos() > 0 && min_commit.as_nanos() > 0,
            "commit cost must be measurable"
        );

        for g in 0..rec_maps {
            unmap(rec_sink + (g * recommended) as u64, recommended);
        }
        for handle in rec_sink_handles {
            release_handle(handle);
        }
        free_reservation(rec_sink, region);
        for g in 0..min_maps {
            unmap(min_sink + (g * minimum) as u64, minimum);
        }
        for handle in min_sink_handles {
            release_handle(handle);
        }
        free_reservation(min_sink, region);
        tear_down_region(rec_base, recommended, rec_handles);
        tear_down_region(min_base, minimum, min_handles);
    }));

    destroy_stream(stream);
    result.unwrap();
}
