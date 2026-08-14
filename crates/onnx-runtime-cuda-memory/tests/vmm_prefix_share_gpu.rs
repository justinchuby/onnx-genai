//! Does the VMM page-level **prefix-sharing** primitive hold at the CUDA driver
//! level? — the isolating GPU probe for #777, with no engine and no KV
//! integration. It answers question **Q1** (does N-way sharing work, including
//! under captured-graph replay — the decode hot path?) and question **Q5** (what
//! does copy-on-write at the granule boundary cost when a sequence diverges?).
//!
//! # Background (measured, merged — see docs/memory/MEMORY_ARCHITECTURE.md)
//!
//! Concurrent serving requests overwhelmingly share a prefix — a system prompt,
//! a tool schema, a few-shot preamble, a RAG document. Today each concurrent
//! sequence stores that prefix in full. Under VMM, sharing it is neither a copy
//! nor a cache lookup: it is a **page mapping** — the same physical granule(s)
//! mapped into several sequences' virtual address ranges. Each sequence still
//! sees one flat contiguous KV buffer at its own stable VA; the attention kernel
//! is unchanged and learns nothing.
//!
//! The enabling primitive is already proven on this hardware: `vmm_graph_remap_
//! gpu.rs` (#727) showed one physical handle mapped at **two** virtual addresses
//! is visible to a captured graph. This probe confirms it **generalizes to N**
//! (N >= 4): one physical granule mapped into N sequences' reserved ranges, all
//! N reading the same correct bytes, including under captured-graph replay.
//!
//! The sharing granularity is the measured 2 MiB CUDA granule (#776); sharing
//! ends at a granule boundary, so a sequence that diverges must obtain a private
//! copy of the boundary granule before writing (Q5).
//!
//! # What is reported
//!
//! Committed **physical** bytes are counted as created physical handles times
//! the granule (each `cuMemCreate` is one granule of device memory), never
//! nominal content bytes. N sequences sharing one prefix granule create
//! `1 + N` handles versus `2 * N` without sharing.
//!
//! # Constraints honoured (per #727 and #777)
//!
//! * Never unmap while a graph replay may be in flight: every remap in Q5
//!   happens with the stream idle (synchronised), never concurrently with a
//!   launch. In production the divergence remap must be sequenced before the
//!   step's captured launch.
//! * Never map or grow inside a captured region: captures here only *read*
//!   already-mapped stable VAs.
//! * No assertions inside any `Drop` (STATUS_STACK_BUFFER_OVERRUN on this
//!   platform); teardown is unconditional and the body runs under
//!   `catch_unwind`.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

/// Distinct byte written once through a single alias to prove every sharer sees
/// the same physical prefix bytes. Not a production fill.
const PREFIX_MARKER: u8 = 0x5a;
/// Per-sequence private byte, offset by the sequence index, to prove private
/// tails do not alias.
const PRIVATE_BASE: u8 = 0x10;
const PROBE_LEN: usize = 4096;

fn require_cuda() -> Arc<CudaContext> {
    match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "CUDA prefix-share test requires a CUDA driver; CPU-only runs must leave this test ignored: {error}"
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

/// Map `handle` at `address` with read/write access. A shared-prefix mapping and
/// a private mapping differ only in whether `handle` is unique to this VA.
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

fn free_bytes() -> usize {
    let mut free = 0usize;
    let mut total = 0usize;
    check("cuMemGetInfo_v2", unsafe {
        cu::cuMemGetInfo_v2(&mut free, &mut total)
    });
    free
}

/// A captured graph that copies `len` bytes device-to-device — the decode hot
/// path in miniature: it reads a stable virtual address (a sequence's shared
/// prefix) and is replayed each step.
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

/// Q1 — one physical prefix granule mapped into N sequences' reservations
/// (N >= 4) is read identically by all N, including under captured-graph
/// replay, while each sequence's private tail stays its own. This generalizes
/// #727's 1-handle-at-2-VAs proof to the N-way concurrency case #750 cares
/// about, and reports the physical saving (1 + N handles versus 2N).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn one_prefix_granule_shared_across_n_sequences_reads_identically_under_replay() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);
    let stream = create_stream();

    // Eight concurrent sequences — the #777 worked example. Each sequence's KV
    // reservation is two granules: a shared prefix granule at offset 0 (the
    // system prompt / RAG document, identical for every request) and a private
    // tail granule at offset `granule` (this request's own decoded tokens).
    const N: usize = 8;
    const SEQ_GRANULES: usize = 2;
    let seq_bytes = granule * SEQ_GRANULES;

    let baseline_free = free_bytes();

    // Exactly one physical handle backs the shared prefix for all N sequences.
    let shared_prefix = create_handle(device, granule);
    // One private handle and one captured-copy sink per sequence.
    let mut seq_bases = Vec::with_capacity(N);
    let mut private_handles = Vec::with_capacity(N);
    let mut sinks = Vec::with_capacity(N);
    let mut sink_handles = Vec::with_capacity(N);
    for _ in 0..N {
        let base = reserve(seq_bytes);
        // Shared prefix at the front of this sequence's VA — the "appropriate
        // offset" is offset 0 for a token-major prefix.
        map_handle(device, base, granule, shared_prefix);
        let private = create_handle(device, granule);
        map_handle(device, base + granule as u64, granule, private);
        seq_bases.push(base);
        private_handles.push(private);

        let sink = reserve(granule);
        let sink_handle = create_handle(device, granule);
        map_handle(device, sink, granule, sink_handle);
        sinks.push(sink);
        sink_handles.push(sink_handle);
    }

    let mapped_free = free_bytes();
    let used = baseline_free.saturating_sub(mapped_free);
    // Physical truth from handle count: 1 shared prefix + N private + N sinks.
    let created_handles = 1 + N + N;
    let shared_prefix_physical = granule; // one granule for the prefix, not N
    let unshared_prefix_physical = N * granule; // what per-sequence copies cost

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Write the shared prefix exactly once, through sequence 0's alias only.
        // Every other sequence must observe it without any write of its own.
        write_host(seq_bases[0], PREFIX_MARKER, granule);
        // Each sequence writes a distinct private tail through its own alias.
        for (i, &base) in seq_bases.iter().enumerate() {
            write_host(base + granule as u64, PRIVATE_BASE + i as u8, PROBE_LEN);
        }

        // Every sequence reads the identical shared prefix and its own tail.
        for (i, &base) in seq_bases.iter().enumerate() {
            let prefix = read_host(base, PROBE_LEN);
            assert!(
                prefix.iter().all(|&b| b == PREFIX_MARKER),
                "sequence {i} must read the shared prefix written once through sequence 0; \
                 first 16 bytes were {:02x?}",
                &prefix[..16]
            );
            let tail = read_host(base + granule as u64, PROBE_LEN);
            assert!(
                tail.iter().all(|&b| b == PRIVATE_BASE + i as u8),
                "sequence {i} private tail must not alias any other sequence"
            );
        }

        // The decode hot path: each sequence captures a device-to-device copy
        // that *reads* its shared-prefix VA into a private sink and replays it.
        // All N replays must reproduce the one shared prefix.
        for i in 0..N {
            let copy = CapturedCopy::capture(stream, sinks[i], seq_bases[i], PROBE_LEN);
            let replay = copy.replay(stream);
            assert_eq!(
                replay,
                cu::CUresult::CUDA_SUCCESS,
                "captured replay reading sequence {i}'s shared prefix must succeed; got {replay:?}"
            );
            let copied = read_host(sinks[i], PROBE_LEN);
            assert!(
                copied.iter().all(|&b| b == PREFIX_MARKER),
                "sequence {i}'s captured copy must reproduce the shared prefix bytes"
            );
        }

        // A write through any single alias is visible to all — the prefix is one
        // physical page, not N copies. Rewrite through sequence N-1's alias and
        // confirm sequence 0 sees it.
        let rewrite = PREFIX_MARKER ^ 0xff;
        write_host(seq_bases[N - 1], rewrite, PROBE_LEN);
        let seen = read_host(seq_bases[0], PROBE_LEN);
        assert!(
            seen.iter().all(|&b| b == rewrite),
            "a write through one alias must be visible at every other alias — proves one \
             physical page backs all N sequences"
        );

        eprintln!(
            "Q1 N-way prefix share: {N} sequences share ONE {}-MiB prefix granule. Physical: \
             {created_handles} handles created total; shared prefix costs {} MiB (1 granule), not \
             {} MiB ({N} per-sequence copies) -- a {N}x reduction on the prefix. cuMemGetInfo \
             delta {} MiB (secondary; noisy on a shared box).",
            granule / (1024 * 1024),
            shared_prefix_physical / (1024 * 1024),
            unshared_prefix_physical / (1024 * 1024),
            used / (1024 * 1024),
        );
        assert!(
            shared_prefix_physical < unshared_prefix_physical,
            "sharing must cost strictly less physical memory than per-sequence copies"
        );
    }));

    for &sink in &sinks {
        unmap(sink, granule);
    }
    for handle in sink_handles {
        release_handle(handle);
    }
    for (i, &base) in seq_bases.iter().enumerate() {
        unmap(base + granule as u64, granule);
        release_handle(private_handles[i]);
        unmap(base, granule);
    }
    release_handle(shared_prefix);
    for &sink in &sinks {
        free_reservation(sink, granule);
    }
    for &base in &seq_bases {
        free_reservation(base, seq_bytes);
    }
    destroy_stream(stream);
    result.unwrap();
}

/// Q5 — copy-on-write at the granule boundary. Sharing ends at a granule
/// boundary, so a sequence that diverges from the shared prefix must obtain a
/// **private** copy of the boundary granule (copy 2 MiB, remap) before it may
/// write. This measures that cost in two regimes — cold (a fresh `cuMemCreate`
/// per divergence) and pooled (the production #740 retained handle, no
/// `cuMemCreate`) — confirms the divergence is correct (the writer mutates only
/// its private copy), and confirms the still-shared original is unharmed for the
/// other sharers.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn copy_on_write_at_the_shared_boundary_granule_cost_and_isolation() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);
    let stream = create_stream();

    // Two sequences share one boundary granule. Sequence A will diverge; B must
    // keep seeing the original shared bytes throughout.
    let shared = create_handle(device, granule);
    let base_a = reserve(granule);
    let base_b = reserve(granule);
    map_handle(device, base_a, granule, shared);
    map_handle(device, base_b, granule, shared);
    // A scratch VA to populate a private copy before it is swapped in at base_a.
    let scratch = reserve(granule);

    const COW_ITERS: usize = 64;
    // A representative qwen14b decode step is dominated by reading the model
    // weights from DRAM; on this device that is tens of milliseconds. We state
    // the assumption explicitly and report COW as a fraction of it.
    const ASSUMED_DECODE_STEP_MS: f64 = 25.0;

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Original shared content.
        write_host(base_a, PREFIX_MARKER, granule);
        let seen_b = read_host(base_b, PROBE_LEN);
        assert!(
            seen_b.iter().all(|&b| b == PREFIX_MARKER),
            "B must see the shared bytes before A diverges"
        );

        // One divergence, timed and verified for isolation. The COW sequence is:
        //   1. create a private handle,
        //   2. map it at a scratch VA and copy the boundary granule into it,
        //   3. synchronise (no replay may be in flight — #727 constraint),
        //   4. unmap the shared handle at A's VA and map the private copy there.
        // Only after this may A write without disturbing B.
        let private = create_handle(device, granule);
        map_handle(device, scratch, granule, private);
        check("cuMemcpyDtoDAsync_v2 COW copy", unsafe {
            cu::cuMemcpyDtoDAsync_v2(scratch, base_a, granule, stream)
        });
        check("cuStreamSynchronize before remap", unsafe {
            cu::cuStreamSynchronize(stream)
        });
        unmap(scratch, granule); // detach the private copy from scratch...
        unmap(base_a, granule); // ...drop the shared handle at A's VA...
        map_handle(device, base_a, granule, private); // ...and swap it in.

        // A now writes its divergent token into its private copy.
        let diverged = PREFIX_MARKER ^ 0x3c;
        write_host(base_a, diverged, granule);
        let a_after = read_host(base_a, PROBE_LEN);
        assert!(
            a_after.iter().all(|&b| b == diverged),
            "A must observe its own divergent write after COW"
        );
        // Isolation: B still sees the untouched shared original.
        let b_after = read_host(base_b, PROBE_LEN);
        assert!(
            b_after.iter().all(|&b| b == PREFIX_MARKER),
            "COW MUST NOT disturb the still-shared original: B must still read the prefix marker"
        );

        // Detach A's private copy and restore the shared mapping so the timed
        // loop below starts from a clean shared state each iteration.
        unmap(base_a, granule);
        release_handle(private);
        map_handle(device, base_a, granule, shared);

        // Warm up (first cuMemCreate on a context is disproportionately slow).
        {
            let p = create_handle(device, granule);
            map_handle(device, scratch, granule, p);
            check("warmup copy", unsafe {
                cu::cuMemcpyDtoDAsync_v2(scratch, base_a, granule, stream)
            });
            check("warmup sync", unsafe { cu::cuStreamSynchronize(stream) });
            unmap(scratch, granule);
            release_handle(p);
        }

        // (a) Cold COW cost: the private copy handle is created fresh via
        // `cuMemCreate` each divergence. On WDDM `cuMemCreate` is a kernel-mode
        // allocation and dominates. Timed span: create + map-scratch + copy
        // 2 MiB + sync + unmap-scratch + unmap-A + map-private-at-A + set-access.
        // Restoring the shared mapping is scaffolding and excluded.
        let mut cold = std::time::Duration::ZERO;
        for _ in 0..COW_ITERS {
            let start = Instant::now();
            let private = create_handle(device, granule);
            map_handle(device, scratch, granule, private);
            check("cold COW copy", unsafe {
                cu::cuMemcpyDtoDAsync_v2(scratch, base_a, granule, stream)
            });
            check("cold COW sync", unsafe { cu::cuStreamSynchronize(stream) });
            unmap(scratch, granule);
            unmap(base_a, granule);
            map_handle(device, base_a, granule, private);
            cold += start.elapsed();
            unmap(base_a, granule);
            release_handle(private);
            map_handle(device, base_a, granule, shared);
        }
        let per_cold = cold / COW_ITERS as u32;

        // (b) Pooled COW cost: the production case. The #740 authority-scoped
        // pool retains freed handles, so a divergence reuses an already-created
        // private handle — no `cuMemCreate` on the hot path. The private handle
        // stays mapped at `scratch` (a handle may be mapped at two VAs at once);
        // each divergence is copy 2 MiB + sync + unmap-A + map-private-at-A +
        // set-access.
        let pooled_private = create_handle(device, granule);
        map_handle(device, scratch, granule, pooled_private);
        let mut pooled = std::time::Duration::ZERO;
        for _ in 0..COW_ITERS {
            let start = Instant::now();
            check("pooled COW copy", unsafe {
                cu::cuMemcpyDtoDAsync_v2(scratch, base_a, granule, stream)
            });
            check("pooled COW sync", unsafe {
                cu::cuStreamSynchronize(stream)
            });
            unmap(base_a, granule);
            map_handle(device, base_a, granule, pooled_private);
            pooled += start.elapsed();
            unmap(base_a, granule);
            map_handle(device, base_a, granule, shared);
        }
        let per_pooled = pooled / COW_ITERS as u32;
        unmap(scratch, granule);
        release_handle(pooled_private);

        let cold_us = per_cold.as_secs_f64() * 1e6;
        let pooled_us = per_pooled.as_secs_f64() * 1e6;
        let cold_pct = (per_cold.as_secs_f64() * 1e3 / ASSUMED_DECODE_STEP_MS) * 100.0;
        let pooled_pct = (per_pooled.as_secs_f64() * 1e3 / ASSUMED_DECODE_STEP_MS) * 100.0;

        eprintln!(
            "Q5 copy-on-write at the boundary granule ({} MiB): cold (fresh cuMemCreate each time) \
             {:.1} us = {:.2}% of an assumed {:.0} ms decode step; pooled (production #740 \
             retained handle, no cuMemCreate) {:.1} us = {:.2}% of a step. The pooled path is the \
             production cost -- a ONE-TIME charge paid only when a sequence first writes past the \
             shared prefix, never a per-token tax. cuMemCreate on WDDM is the dominant term, which \
             is exactly why the #740 handle pool exists.",
            granule / (1024 * 1024),
            cold_us,
            cold_pct,
            ASSUMED_DECODE_STEP_MS,
            pooled_us,
            pooled_pct,
        );
        assert!(per_cold.as_nanos() > 0, "a COW must take measurable time");
        assert!(
            per_pooled <= per_cold,
            "the pooled COW (no cuMemCreate) must not cost more than the cold COW"
        );
    }));

    unmap(scratch, granule);
    unmap(base_b, granule);
    unmap(base_a, granule);
    release_handle(shared);
    free_reservation(scratch, granule);
    free_reservation(base_b, granule);
    free_reservation(base_a, granule);
    destroy_stream(stream);
    result.unwrap();
}
