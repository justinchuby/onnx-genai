//! Does a shared "dummy" physical page make the uncommitted tail of a VMM KV
//! reservation *fault-safe* — and what does it cost? — the self-contained
//! driver-level probe for #759, with no engine and no KV integration.
//!
//! #759 proposes mapping every uncommitted virtual page of a KV reservation to
//! one shared physical "dummy" page so a kernel that speculatively reads past
//! the live sequence length hits backed memory instead of faulting. Before any
//! KV code is written around that premise, this probe establishes — or kills —
//! the primitive at the CUDA driver level, extending the proven patterns in
//! `vmm_graph_remap_gpu.rs` (#727: a captured graph replays across
//! unmap/create/map at a stable VA, and one physical handle mapped at two VAs
//! works) and `vmm_kv_contiguous_tail_gpu.rs` (#772: a *padded* read one byte
//! into the uncommitted tail faults `CUDA_ERROR_INVALID_VALUE`).
//!
//! This file answers two of the five #759 questions with driver-level tests;
//! the rest live in their own binaries so a poisoned context or a
//! process-global counter cannot contaminate them
//! (`vmm_dummy_write_protect_gpu.rs` for the write-protection stickiness
//! question Q3, `vmm_dummy_page_ledger_gpu.rs` for the real allocator/ledger
//! charge-once question Q4), and the fill-choice rule (Q2) and crossover
//! analysis live in `dummy_fill_and_crossover.rs` because they are IEEE-754
//! arithmetic and geometry, not CUDA calls:
//!
//! * [`dummy_backed_tail_removes_the_padded_read_fault`] (Q1) — the positive
//!   counterpart of #772's faulting padded read: with one dummy handle mapped
//!   across the tail granules, a full-padded read *and a captured-graph replay
//!   that reads into the tail* both succeed instead of faulting.
//! * [`dummy_to_real_growth_remap_cost`] (Q5) — measures the
//!   `cuMemUnmap`/`cuMemCreate`/`cuMemMap`/`cuMemSetAccess` cost of a
//!   dummy->real growth step, including the fixed-full-context-stride case
//!   where ~96 objects each need a remap per token.
//!
//! ## The fill value is NOT chosen here — and it is not NaN
//!
//! An earlier draft filled the dummy with a NaN "sentinel" for detectability.
//! That is wrong: #721 stage 4's decode kernel reads the padded shape and masks
//! (`score -> -inf` past the live length), and NaN defeats additive masking
//! (`NaN + (-inf) = NaN`, `exp(NaN) = NaN`), poisoning even the correctly-masked
//! output. The fill must be chosen from the measured masking rule in
//! `dummy_fill_and_crossover::masking_determines_the_safe_dummy_fill`: **zeros
//! when the kernel's tail masking is verified in the EP, never NaN**. These
//! driver-level tests only need the read to *succeed and be consistent*, so
//! they write an arbitrary [`READBACK_MARKER`] to prove the mapping and its
//! aliasing — that byte is a test probe, not the production fill.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

/// An arbitrary non-zero byte used only to prove a dummy read returns what was
/// written and that every tail VA aliases the one physical page. It is **not**
/// the production fill — that is decided by the masking rule (zeros, never NaN;
/// see `dummy_fill_and_crossover`).
const READBACK_MARKER: u8 = 0x5a;

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

fn granularity(device_ordinal: i32) -> usize {
    let prop = allocation_prop(device_ordinal);
    let mut granularity = 0usize;
    let result = unsafe {
        cu::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    };
    check("cuMemGetAllocationGranularity", result);
    assert_ne!(granularity, 0, "CUDA reported zero VMM granularity");
    granularity
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

/// Map `handle` at `address` and grant read/write access — the shape of both a
/// live commit and a dummy-page mapping; they differ only in whether `handle`
/// is unique to this VA or shared across the whole tail.
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

/// A captured graph that copies `len` bytes device-to-device — the decode hot
/// path in miniature, reading a stable virtual address (whose tail is dummy
/// backed) and replayed each step.
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

    fn replay(&self, stream: cu::CUstream) -> cu::CUresult {
        let launch = unsafe { cu::cuGraphLaunch(self.exec, stream) };
        if launch != cu::CUresult::CUDA_SUCCESS {
            return launch;
        }
        unsafe { cu::cuStreamSynchronize(stream) }
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

/// Q1 — the dummy page removes the fault. #772 proved a padded read one byte
/// into the uncommitted tail faults `CUDA_ERROR_INVALID_VALUE`. Here one dummy
/// physical handle mapped across every tail granule makes the same padded read
/// — *and a captured-graph device-to-device copy that reads the whole padded
/// range, the decode hot path* — succeed. This is the primitive doing its one
/// job: turning a fault into a (detectably poisoned) read.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn dummy_backed_tail_removes_the_padded_read_fault() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);
    let stream = create_stream();

    const TOTAL_GRANULES: usize = 8;
    const LIVE_GRANULES: usize = 2;
    let total = granule * TOTAL_GRANULES;
    let live = granule * LIVE_GRANULES;

    let base = reserve(total);
    // Reserved, fully committed sink for the captured copy to write into, so
    // the copy's *read* of the dummy tail is the only thing under test.
    let sink = reserve(total);

    let mut live_handles = Vec::new();
    for g in 0..LIVE_GRANULES {
        let handle = create_handle(device, granule);
        map_handle(device, base + (g * granule) as u64, granule, handle);
        live_handles.push(handle);
    }
    // One dummy handle backs every tail granule at once — the #759 primitive.
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
        // Fill the live prefix with ordinary data and, through one tail VA,
        // write an arbitrary readback marker into the shared dummy page. Because
        // every tail granule aliases the one dummy handle, this single write is
        // visible at all of them. The marker only proves the read works and
        // aliases; the production fill is decided by the masking rule (zeros,
        // never NaN), not here.
        write_host(base, 0x11, live);
        write_host(base + live as u64, READBACK_MARKER, granule);

        // The #772 fault, now backed: a full-padded host read across the whole
        // reservation (live prefix + dummy tail) succeeds where the unbacked
        // tail returned CUDA_ERROR_INVALID_VALUE.
        let mut host = vec![0u8; total];
        let padded = unsafe { cu::cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), base, total) };
        assert_eq!(
            padded,
            cu::CUresult::CUDA_SUCCESS,
            "with a dummy handle backing the tail, the padded read that faulted in #772 must \
             succeed; got {padded:?}"
        );
        assert!(
            host[..live].iter().all(|&b| b == 0x11),
            "live prefix must survive the dummy mapping"
        );
        assert!(
            host[live..].iter().all(|&b| b == READBACK_MARKER),
            "every dummy-backed tail byte must read back the marker written through one alias"
        );

        // The decode hot path: a captured graph that reads the entire padded
        // range (including the dummy tail) and replays without faulting.
        let copy = CapturedCopy::capture(stream, sink, base, total);
        let replay = copy.replay(stream);
        assert_eq!(
            replay,
            cu::CUresult::CUDA_SUCCESS,
            "a captured graph reading into the dummy-backed tail must replay without faulting; \
             got {replay:?}"
        );
        let copied = read_host(sink, total);
        assert!(
            copied[..live].iter().all(|&b| b == 0x11)
                && copied[live..].iter().all(|&b| b == READBACK_MARKER),
            "the captured copy must reproduce the live prefix and the dummy-backed tail"
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

/// Growth step for one object: drop the dummy mapping at `address` and map a
/// fresh real handle there. This is exactly what a token that extends into a
/// new granule must do — the dummy tail shrinks by one granule, the live prefix
/// grows by one. Returns the real handle so the caller can time and tear down.
///
/// Ordering rule this probe relies on (documented per the #727 constraint that
/// unmap-under-replay is *not* proven safe): the remap happens with no graph
/// replay in flight. In production this must be sequenced before the step's
/// captured launch, never concurrently with it.
fn grow_one_granule(
    device: i32,
    address: cu::CUdeviceptr,
    granule: usize,
) -> cu::CUmemGenericAllocationHandle {
    unmap(address, granule);
    let real = create_handle(device, granule);
    let map = unsafe { cu::cuMemMap(address, granule, 0, real, 0) };
    check("cuMemMap real over dummy", map);
    set_access(device, address, granule);
    real
}

/// Q5 — what does growth cost? Growing the live length means unmapping the
/// dummy at a VA and mapping a real handle there. This measures that remap for
/// (a) a single granule and (b) the fixed-full-context-stride case where ~96
/// head-stripe objects may each need a remap in one token step. The numbers go
/// on the record so #759's "negligible" cost claim can be judged against the
/// previously measured 9.0%-of-step `vram_free` cost.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn dummy_to_real_growth_remap_cost() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    // qwen2.5-0.5b fixed-full-context head-major geometry: 24 layers x (key +
    // value) x 2 kv heads = 96 head-stripe objects, each of which needs its own
    // remap when the token step crosses a granule boundary.
    const OBJECTS: usize = 96;
    const SINGLE_ITERS: usize = 64;

    let dummy = create_handle(device, granule);

    // (a) Single-granule remap cost, averaged. Each iteration grows one dummy
    // granule to real then returns it to dummy, so the steady state is measured
    // rather than a cold first touch.
    let single_base = reserve(granule);
    map_handle(device, single_base, granule, dummy);

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Warm up (first cuMemCreate on a context is disproportionately slow).
        {
            let h = grow_one_granule(device, single_base, granule);
            unmap(single_base, granule);
            map_handle(device, single_base, granule, dummy);
            release_handle(h);
        }

        let start = Instant::now();
        for _ in 0..SINGLE_ITERS {
            let real = grow_one_granule(device, single_base, granule);
            // Return to dummy so the next iteration starts from the dummy state.
            unmap(single_base, granule);
            let map = unsafe { cu::cuMemMap(single_base, granule, 0, dummy, 0) };
            check("cuMemMap dummy back", map);
            set_access(device, single_base, granule);
            release_handle(real);
        }
        let per_remap = start.elapsed() / SINGLE_ITERS as u32;

        // (b) One realistic token growth step under a fixed full-context
        // stride: 96 independent stripes, each a dummy->real remap. Reserve a
        // granule per object, all dummy-backed, then grow every one and time
        // the whole step.
        let mut bases = Vec::with_capacity(OBJECTS);
        for _ in 0..OBJECTS {
            let b = reserve(granule);
            map_handle(device, b, granule, dummy);
            bases.push(b);
        }
        let step_start = Instant::now();
        let mut reals = Vec::with_capacity(OBJECTS);
        for &b in &bases {
            reals.push(grow_one_granule(device, b, granule));
        }
        let step = step_start.elapsed();
        let per_object = step / OBJECTS as u32;

        eprintln!(
            "Q5 dummy->real growth cost (granule {} MiB): single remap {:.1} us; \
             {OBJECTS}-object fixed-stride token step {:.1} us total = {:.1} us/object. \
             A step that remaps {OBJECTS} granules per token is NOT free -- vram_free alone was \
             previously 9.0% of step time.",
            granule / (1024 * 1024),
            per_remap.as_secs_f64() * 1e6,
            step.as_secs_f64() * 1e6,
            per_object.as_secs_f64() * 1e6,
        );

        assert!(
            per_remap.as_nanos() > 0,
            "a remap must take measurable time"
        );
        assert_eq!(reals.len(), OBJECTS, "every object must be grown");

        for (b, real) in bases.into_iter().zip(reals) {
            unmap(b, granule);
            release_handle(real);
            free_reservation(b, granule);
        }
    }));

    unmap(single_base, granule);
    free_reservation(single_base, granule);
    release_handle(dummy);
    result.unwrap();
}
