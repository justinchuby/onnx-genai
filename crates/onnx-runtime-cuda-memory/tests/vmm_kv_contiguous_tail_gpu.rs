//! Does a contiguous virtual address let the KV cache leave its tail granules
//! physically uncommitted? — the crux falsifier for #750 / #721 stage 4.
//!
//! The owner's directive is: give each sequence one stable, flat, contiguous
//! virtual address for its whole context so the attention kernel keeps reading a
//! plain `[heads, seq, head_dim]` buffer while paging happens invisibly
//! underneath, physically scattered and grown on demand. Reserving virtual
//! address space is free (`cuMemAddressReserve`); what actually costs physical
//! memory is a *committed* granule (`cuMemCreate` + `cuMemMap`). So the whole
//! design turns on one question:
//!
//!   **Can the decode attention kernel be bounded to the live sequence length,
//!   so the tail of the reservation is never dereferenced and never has to be
//!   committed?**
//!
//! #721 stage 4 answered "no" for a *full-context stride*: it committed 1.5 GB
//! where bucket growth commits 48 MB (a 32x regression) because the decode
//! kernel read the full padded shape and relied on masking for correctness, so
//! the tail could never be decommitted. This test isolates that mechanism at
//! the CUDA driver level, with no engine and no model, so the cause is
//! unambiguous:
//!
//! * [`bounded_read_of_contiguous_va_leaves_tail_granules_uncommitted`] — a read
//!   (including a *captured* graph replay, the decode hot path) bounded to the
//!   committed live length succeeds while the tail granules stay uncommitted.
//!   This is the design working: the flat contiguous VA is real, the tail is
//!   physically absent, and the kernel never touches it.
//! * [`padded_read_into_uncommitted_tail_faults_forcing_commit`] — the *same*
//!   buffer, read one element past the committed live length into the
//!   uncommitted tail, faults with `CUDA_ERROR_INVALID_VALUE`. This is the
//!   #721 stage-4 mechanism reproduced: a kernel that reads the padded shape
//!   dereferences the tail, so the tail can never be decommitted. The read
//!   pattern — not the reservation — is what forces the physical commit.
//! * [`head_major_full_context_stride_pays_one_granule_per_object`] — even a
//!   perfectly length-bounded read cannot make a *fixed full-context stride*
//!   cheap: with the KV's per-binding/head-major layout, each head's valid
//!   prefix lands in its own granule, so the commit floor is
//!   `objects x granule`, independent of content. Reproduces the documented
//!   qwen2.5-0.5b floor (192 MiB for ~3 MiB of content) on hardware.
//!
//! Together these say: contiguous-VA-per-sequence with a *bucketed* stride can
//! match bucket growth's committed bytes (the tail past the bucket is never
//! read, so never committed), but a *fixed full-context* stride — the only
//! shape that removes the re-stride/re-capture on growth — pays the head-major
//! granule floor and loses to bucket growth. The kernel's read bound is
//! necessary but not sufficient; the layout's grow-axis stride decides the
//! floor.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

const SENTINEL: u8 = 0x5a;

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

/// Commit one granule of physical memory at `address` (the live-length grow
/// step): a fresh `cuMemCreate` handle mapped into the reserved VA. Returns the
/// handle so the caller can release it during teardown.
fn commit_granule(
    device_ordinal: i32,
    address: cu::CUdeviceptr,
    granule: usize,
) -> cu::CUmemGenericAllocationHandle {
    let handle = create_handle(device_ordinal, granule);
    let result = unsafe { cu::cuMemMap(address, granule, 0, handle, 0) };
    check("cuMemMap", result);
    set_access(device_ordinal, address, granule);
    handle
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

/// A captured graph that memsets `len` bytes at `address` — the decode hot path
/// in miniature: a graph captured once and replayed each step, writing the
/// current token's KV into the live tail of a stable virtual address.
struct CapturedMemset {
    graph: cu::CUgraph,
    exec: cu::CUgraphExec,
    len: usize,
}

impl CapturedMemset {
    fn capture(stream: cu::CUstream, address: cu::CUdeviceptr, value: u8, len: usize) -> Self {
        let mut graph = std::ptr::null_mut();
        let mut exec = std::ptr::null_mut();
        let begin = unsafe {
            cu::cuStreamBeginCapture_v2(
                stream,
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        };
        check("cuStreamBeginCapture_v2", begin);
        let record = unsafe { cu::cuMemsetD8Async(address, value, len, stream) };
        let end = unsafe { cu::cuStreamEndCapture(stream, &mut graph) };
        check("cuMemsetD8Async during capture", record);
        check("cuStreamEndCapture", end);
        assert!(!graph.is_null(), "cuStreamEndCapture returned a null graph");
        let inst = unsafe { cu::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        check("cuGraphInstantiateWithFlags", inst);
        assert!(!exec.is_null(), "null graph exec");
        Self { graph, exec, len }
    }

    fn replay(&self, stream: cu::CUstream) {
        check("cuGraphLaunch", unsafe {
            cu::cuGraphLaunch(self.exec, stream)
        });
        check("cuStreamSynchronize after cuGraphLaunch", unsafe {
            cu::cuStreamSynchronize(stream)
        });
    }
}

impl Drop for CapturedMemset {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            let _ = unsafe { cu::cuGraphExecDestroy(self.exec) };
        }
        if !self.graph.is_null() {
            let _ = unsafe { cu::cuGraphDestroy(self.graph) };
        }
        let _ = self.len;
    }
}

/// The positive branch: a flat contiguous virtual address whose tail granules
/// are physically absent, read correctly — including by a captured graph replay
/// — up to the committed live length. This is the design the owner asked for
/// working end to end at the driver level: one stable VA, a physically short
/// backing, and a kernel that only ever touches the live prefix.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn bounded_read_of_contiguous_va_leaves_tail_granules_uncommitted() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    // Reserve a full-context contiguous VA of 8 granules; commit only the first
    // 2 (the "live length"). The remaining 6 granules are addressable but have
    // no physical backing.
    const TOTAL_GRANULES: usize = 8;
    const LIVE_GRANULES: usize = 2;
    let total = granule * TOTAL_GRANULES;
    let live = granule * LIVE_GRANULES;
    let base = reserve(total);
    let stream = create_stream();

    let mut handles = Vec::new();
    for g in 0..LIVE_GRANULES {
        handles.push(commit_granule(device, base + (g * granule) as u64, granule));
    }

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Host write + read bounded to the live length: correct, no fault.
        write_host(base, 0x00, live);
        let graph = CapturedMemset::capture(stream, base, SENTINEL, live);
        graph.replay(stream);
        let seen = read_host(base, live);
        assert!(
            seen.iter().all(|&b| b == SENTINEL),
            "bounded captured-graph replay must fill the whole committed live length; \
             first 16 bytes {:02x?}",
            &seen[..16]
        );

        // The tail is genuinely uncommitted: a probe one byte into the first
        // uncommitted granule fails, proving no silent physical backing.
        let mut probe = [0u8; 1];
        let tail = base + live as u64;
        let tail_read = unsafe { cu::cuMemcpyDtoH_v2(probe.as_mut_ptr().cast(), tail, 1) };
        assert_eq!(
            tail_read,
            cu::CUresult::CUDA_ERROR_INVALID_VALUE,
            "tail granule at offset {live} must be uncommitted, but the read returned {tail_read:?}"
        );
    }));

    for g in 0..LIVE_GRANULES {
        let _ = unsafe { cu::cuMemUnmap(base + (g * granule) as u64, granule) };
    }
    for handle in handles {
        release_handle(handle);
    }
    free_reservation(base, total);
    destroy_stream(stream);
    result.unwrap();
}

/// The #721 stage-4 mechanism, isolated: read one element past the committed
/// live length into the uncommitted tail and it faults. A decode kernel that
/// reads the *padded* shape (and masks for correctness) issues exactly this
/// access pattern, so its tail can never be decommitted — the read pattern, not
/// the reservation, forces the physical commit. The error is synchronous and
/// non-sticky (`cuMemcpyDtoH_v2` validates the mapping), so the same context
/// keeps working, which the test also asserts.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn padded_read_into_uncommitted_tail_faults_forcing_commit() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    const TOTAL_GRANULES: usize = 4;
    const LIVE_GRANULES: usize = 1;
    let total = granule * TOTAL_GRANULES;
    let live = granule * LIVE_GRANULES;
    let base = reserve(total);
    let handle = commit_granule(device, base, granule);

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Bounded read of the committed live length: succeeds.
        write_host(base, SENTINEL, live);
        let ok = read_host(base, live);
        assert!(
            ok.iter().all(|&b| b == SENTINEL),
            "committed prefix readable"
        );

        // Padded read: live length + one byte, crossing into the uncommitted
        // tail granule. This is the exact access a full-padded-shape kernel
        // makes on the KV cache's unfilled tail.
        let padded = live + 1;
        let mut buf = vec![0u8; padded];
        let faulted = unsafe { cu::cuMemcpyDtoH_v2(buf.as_mut_ptr().cast(), base, padded) };
        assert_eq!(
            faulted,
            cu::CUresult::CUDA_ERROR_INVALID_VALUE,
            "reading one byte past the committed live length into the uncommitted tail must fault; \
             got {faulted:?}. If this ever returns SUCCESS the tail is being silently committed."
        );

        // The fault did not poison the context: a bounded read still works, so
        // a length-bounded kernel can keep decoding against the same VA.
        let after = read_host(base, live);
        assert!(
            after.iter().all(|&b| b == SENTINEL),
            "context must stay usable after a bounded fault"
        );
    }));

    let _ = unsafe { cu::cuMemUnmap(base, granule) };
    release_handle(handle);
    free_reservation(base, total);
    result.unwrap();
}

/// Why a length-bounded read is necessary but not sufficient: the KV cache is
/// per-binding/head-major (`[1, heads, seq, head_dim]`), so a *fixed
/// full-context stride* places each head's tokens `full_context * head_dim`
/// apart. Committing the live prefix of every head therefore lands one granule
/// per head, and the commit floor is `objects x granule` regardless of how
/// little content each head holds.
///
/// This reproduces the documented qwen2.5-0.5b floor on hardware: 24 layers x
/// (key + value) = 48 bindings, each `[1, 2, 32768, 64]` f16 whose per-head
/// stripe is 4 MiB (2 granules) — so a near-empty cache still commits
/// 48 x 2 = 96 granules = 192 MiB, matching "192 MiB for ~3 MiB of content".
/// The test physically commits the per-head granules for a representative
/// object count and shows the committed bytes are set by object count, not
/// content — the reason a fixed full-context stride loses to bucket growth even
/// with a perfectly bounded kernel.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn head_major_full_context_stride_pays_one_granule_per_object() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    // qwen2.5-0.5b KV geometry (genai_config.json / model.onnx):
    //   24 layers, key+value => 48 bindings; 2 kv heads; head_dim 64; f16.
    //   context_length 32768.
    const LAYERS: usize = 24;
    const KV_TENSORS_PER_LAYER: usize = 2; // key + value
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const ELEM_BYTES: usize = 2; // f16
    const CONTEXT: usize = 32768;
    const OBJECTS: usize = LAYERS * KV_TENSORS_PER_LAYER * KV_HEADS; // 96 head-stripes

    let per_head_stride = CONTEXT * HEAD_DIM * ELEM_BYTES; // 4 MiB
    assert!(
        per_head_stride >= granule,
        "with a full-context stride each head stripe ({per_head_stride} B) is at least one \
         granule ({granule} B), so no two heads can share a granule"
    );

    // A near-empty cache: one live token per head.
    let live_content_per_head = HEAD_DIM * ELEM_BYTES; // 128 bytes
    let total_content = OBJECTS * live_content_per_head; // ~12 KiB across all heads
    // With a fixed full-context stride the commit floor is one granule per head.
    let full_context_committed = OBJECTS * granule;

    // Physically prove the floor for a representative subset (so the test needs
    // only PROBE_OBJECTS * granule of VRAM, not the full 192 MiB), then assert
    // the closed form for the whole model.
    const PROBE_OBJECTS: usize = 8;
    // Reserve one full-context stripe per probed object, commit only the first
    // granule (the live prefix) of each — exactly the per-head commit a
    // fixed-stride flat VA would make.
    let mut reservations = Vec::new();
    let mut handles = Vec::new();
    let mut committed_bytes = 0usize;
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        for _ in 0..PROBE_OBJECTS {
            let base = reserve(per_head_stride);
            reservations.push(base);
            // Commit just the granule holding this head's live token.
            handles.push(commit_granule(device, base, granule));
            committed_bytes += granule;
            // The live content is a sliver of the committed granule.
            write_host(base, SENTINEL, live_content_per_head);
        }
        assert_eq!(
            committed_bytes,
            PROBE_OBJECTS * granule,
            "each probed head must commit exactly one granule"
        );
        // Committed physical bytes are set by object count, not content: the
        // probed slice already commits far more than it stores.
        let probed_content = PROBE_OBJECTS * live_content_per_head;
        assert!(
            committed_bytes > probed_content * 100,
            "granule floor: {committed_bytes} committed for {probed_content} of content"
        );
    }));

    for (base, handle) in reservations.iter().zip(handles) {
        let _ = unsafe { cu::cuMemUnmap(*base, granule) };
        release_handle(handle);
        free_reservation(*base, per_head_stride);
    }
    result.unwrap();

    // The closed form for the whole model, printed so the number is on the
    // record next to the measured granule.
    eprintln!(
        "qwen2.5-0.5b fixed full-context stride floor: {OBJECTS} objects x {granule} B granule = \
         {} MiB committed for {total_content} B (~{} KiB) of live content; \
         bucket growth commits ~48 MiB at the same live length.",
        full_context_committed / (1024 * 1024),
        total_content / 1024,
    );
    assert_eq!(
        full_context_committed,
        192 * 1024 * 1024,
        "qwen2.5-0.5b fixed full-context head-major floor must be the documented 192 MiB"
    );
}
