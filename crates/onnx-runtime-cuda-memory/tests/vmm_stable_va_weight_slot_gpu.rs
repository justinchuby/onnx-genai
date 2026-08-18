//! Does a captured CUDA graph that *reads* from a stable virtual address still
//! read the correct bytes after the physical granules under that address are
//! swapped for different ones — through the production [`CudaVmmAllocator`]
//! commit/decommit path, not raw `cuMemMap`?
//!
//! This is the step-1 falsifier for #716. #727 proved that a captured graph
//! which *writes* a constant survives `cuMemUnmap`/`cuMemCreate`/`cuMemMap` at
//! the same VA (`vmm_graph_remap_gpu.rs`). Weight paging is the read side and
//! goes through the #740 authority-scoped physical-handle pool rather than raw
//! driver calls: a paged weight is reserved once at a stable VA, its physical
//! granules are committed on page-in and released on page-out, and the same VA
//! is re-committed with a *different* pooled handle when the weight pages back
//! in. If a decode graph captured while the weight was resident cannot read the
//! re-paged bytes correctly, the whole #716 premise (offload + capture together)
//! is unsound and everything downstream is moot.
//!
//! The test models exactly that lifecycle:
//!   1. reserve a stable-VA weight slot (`allocate_committed(.., &[])` — VA only)
//!   2. commit physical granules (`try_commit_span`), copy weight pattern A in
//!   3. capture a graph that copies slot → output, replay, assert output == A
//!   4. page the weight out (`decommit_allocation_range`) — VA stays reserved
//!   5. force a *different* physical granule to back the slot on re-page-in
//!   6. re-commit the same VA, copy a *different* weight pattern B in
//!   7. replay the SAME captured graph, assert output == B
//!
//! Step 7 is the falsifier. A graph that had baked the physical page, cached
//! the value, or read stale memory would still report A; reading B proves the
//! captured graph dereferences whatever physical is currently mapped under the
//! stable VA.
//!
//! Skips loudly without the `gpu-tests` feature, like the sibling `*_gpu`
//! tests — a skip that reads like a pass is how #636 lost 44 tests.

use std::panic::AssertUnwindSafe;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;
use onnx_runtime_memory_governor::VirtualBacking as _;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole,
};

const HOLDER: HolderId = HolderId::new(716);
const GRANULE: usize = 2 << 20;
const PROBE_LEN: usize = 4096;
const PATTERN_A: u8 = 0x5a;
const PATTERN_B: u8 = 0x3c;

fn check(call: &'static str, result: cu::CUresult) {
    assert_eq!(result, cu::CUresult::CUDA_SUCCESS, "{call}: {result:?}");
}

fn create_stream() -> cu::CUstream {
    let mut stream = std::ptr::null_mut();
    check("cuStreamCreate", unsafe {
        cu::cuStreamCreate(
            &mut stream,
            cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
        )
    });
    stream
}

fn destroy_stream(stream: cu::CUstream) {
    if !stream.is_null() {
        let _ = unsafe { cu::cuStreamDestroy_v2(stream) };
    }
}

fn write_device(address: cu::CUdeviceptr, value: u8, len: usize) {
    let bytes = vec![value; len];
    check("cuMemcpyHtoD_v2", unsafe {
        cu::cuMemcpyHtoD_v2(address, bytes.as_ptr().cast(), bytes.len())
    });
}

fn read_device(address: cu::CUdeviceptr, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    check("cuMemcpyDtoH_v2", unsafe {
        cu::cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(), address, bytes.len())
    });
    bytes
}

fn assert_device_bytes(address: cu::CUdeviceptr, expected: u8, label: &str) {
    let bytes = read_device(address, PROBE_LEN);
    assert!(
        bytes.iter().all(|&byte| byte == expected),
        "{label}: expected all bytes 0x{expected:02x}, first 16 were {:02x?}",
        &bytes[..16]
    );
}

/// A captured graph that copies `PROBE_LEN` bytes from `src` to `dst`. The
/// source and destination addresses are baked into the recorded node; only the
/// physical memory under them may legally change between replays.
struct CapturedCopy {
    graph: cu::CUgraph,
    exec: cu::CUgraphExec,
}

impl CapturedCopy {
    fn capture(stream: cu::CUstream, dst: cu::CUdeviceptr, src: cu::CUdeviceptr) -> Self {
        let mut graph = std::ptr::null_mut();
        let mut exec = std::ptr::null_mut();
        check("cuStreamBeginCapture_v2", unsafe {
            cu::cuStreamBeginCapture_v2(
                stream,
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        });
        let record = unsafe { cu::cuMemcpyDtoDAsync_v2(dst, src, PROBE_LEN, stream) };
        let end = unsafe { cu::cuStreamEndCapture(stream, &mut graph) };
        check("cuMemcpyDtoDAsync_v2 during capture", record);
        check("cuStreamEndCapture", end);
        assert!(!graph.is_null(), "cuStreamEndCapture returned a null graph");
        check("cuGraphInstantiateWithFlags", unsafe {
            cu::cuGraphInstantiateWithFlags(&mut exec, graph, 0)
        });
        assert!(!exec.is_null(), "null exec from instantiate");
        Self { graph, exec }
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

fn device_ptr(ptr: std::ptr::NonNull<u8>) -> cu::CUdeviceptr {
    ptr.as_ptr() as usize as cu::CUdeviceptr
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn captured_read_from_stable_va_tracks_repaged_physical_granules() {
    // SAFETY: this integration test owns its process and sets the production
    // pool option before constructing any allocator.
    unsafe {
        std::env::set_var(
            CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            (64usize << 20).to_string(),
        );
    }
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "stable-VA weight-slot test requires a CUDA driver; CPU-only runs must leave it \
             ignored: {error}"
        ),
    };
    context.bind_to_thread().expect("bind CUDA context");
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = CudaVmmAllocator::new(
        context,
        DeviceKey::device(0),
        0,
        64 << 20,
        &governor,
        HOLDER,
        MemoryRole::Weights,
    )
    .expect("reserve VMM weight arena");
    assert!(
        allocator.physical_pool_stats().is_some(),
        "the production physical-handle pool must be installed for this proof"
    );

    let stream = create_stream();

    // A stable-VA weight slot: reserve the address once, commit nothing yet.
    let slot = allocator
        .allocate_committed(GRANULE, 256, &[])
        .expect("reserve stable-VA weight slot");
    let slot_va = device_ptr(slot);
    // A separate committed output the captured graph writes into.
    let output = allocator.allocate(GRANULE, 256).expect("output allocation");
    let output_va = device_ptr(output);

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // --- page-in #1: commit physical, copy weight pattern A ---
        allocator
            .try_commit_span(slot, GRANULE, 0, GRANULE, GRANULE as u64, GRANULE as u64)
            .expect("commit slot granule for page-in A");
        assert_eq!(
            allocator.allocation_committed_bytes(slot, GRANULE, 256),
            GRANULE,
            "the slot must be fully backed after page-in A"
        );
        write_device(slot_va, PATTERN_A, GRANULE);
        write_device(output_va, 0x00, PROBE_LEN);

        // Capture a graph that reads the weight slot at its stable VA.
        let graph = CapturedCopy::capture(stream, output_va, slot_va);
        graph.replay(stream);
        assert_device_bytes(output_va, PATTERN_A, "initial replay reads weight A");

        // --- page-out: release the physical granule; keep the VA reserved ---
        check("cuStreamSynchronize before decommit", unsafe {
            cu::cuStreamSynchronize(stream)
        });
        let unmapped = allocator
            .decommit_allocation_range(slot, GRANULE, 256, 0, GRANULE)
            .expect("decommit slot granule (page-out)");
        assert_eq!(
            unmapped, GRANULE as u64,
            "page-out must unmap exactly the one granule"
        );
        assert_eq!(
            allocator.allocation_committed_bytes(slot, GRANULE, 256),
            0,
            "the slot holds no physical bytes while paged out"
        );
        assert_eq!(
            device_ptr(slot),
            slot_va,
            "the weight slot VA must not move across page-out"
        );

        // Force a *different* physical granule to back the slot next time: a
        // filler allocation grabs the handle the decommit just returned to the
        // pool, so page-in B must map a fresh granule under the same VA.
        let filler = allocator
            .allocate(GRANULE, 256)
            .expect("filler grabs pooled handle");

        // --- page-in #2: re-commit the SAME VA, copy a *different* weight B ---
        allocator
            .try_commit_span(slot, GRANULE, 0, GRANULE, GRANULE as u64, GRANULE as u64)
            .expect("commit slot granule for page-in B");
        assert_eq!(
            device_ptr(slot),
            slot_va,
            "the weight slot VA must not move across re-page-in"
        );
        write_device(slot_va, PATTERN_B, GRANULE);

        // The falsifier: the captured graph, unchanged, must now read B.
        graph.replay(stream);
        assert_device_bytes(
            output_va,
            PATTERN_B,
            "replay after re-paging reads the NEW weight B through the stable VA",
        );

        // SAFETY: `filler` came from this allocator and is unused by CUDA work.
        unsafe { allocator.deallocate(filler, GRANULE, 256) };
    }));

    // SAFETY: both came from this allocator; all copies were synchronous.
    unsafe { allocator.deallocate(slot, GRANULE, 256) };
    unsafe { allocator.deallocate(output, GRANULE, 256) };
    destroy_stream(stream);
    let _ = &governor;
    result.unwrap();
}
