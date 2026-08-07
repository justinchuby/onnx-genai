//! CUDA graph behavior when VMM mappings change under a stable virtual address.
//!
//! This is a deliberately small falsifier for #721: the captured graph records
//! a memset to a reserved CUDA VMM address, then the physical backing changes
//! without changing that virtual address. If replay still writes the expected
//! bytes, graph capture can survive VMM-backed KV growth; if it cannot, stage 4
//! should stop before the KV cache is refactored around that premise.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

const PROBE_LEN: usize = 4096;
const GRAPH_VALUE: u8 = 0x5a;
const ALIAS_VALUE: u8 = 0x3c;

fn require_cuda() -> Arc<CudaContext> {
    match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "CUDA graph/VMM test requires a CUDA driver; CPU-only runs must leave this test ignored: {error}"
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
    let result = unsafe { cu::cuMemUnmap(address, size) };
    check("cuMemUnmap", result);
}

fn unmap_if_mapped(address: cu::CUdeviceptr, size: usize, mapped: bool) {
    if mapped {
        let _ = unsafe { cu::cuMemUnmap(address, size) };
    }
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

fn assert_device_bytes(address: cu::CUdeviceptr, expected: u8, label: &str) {
    let bytes = read_host(address, PROBE_LEN);
    assert!(
        bytes.iter().all(|&byte| byte == expected),
        "{label}: expected all bytes to be 0x{expected:02x}, first 16 bytes were {:02x?}",
        &bytes[..16]
    );
}

struct CapturedMemset {
    graph: cu::CUgraph,
    exec: cu::CUgraphExec,
}

impl CapturedMemset {
    fn capture(stream: cu::CUstream, address: cu::CUdeviceptr, value: u8) -> Self {
        let mut graph = std::ptr::null_mut();
        let mut exec = std::ptr::null_mut();

        let result = unsafe {
            cu::cuStreamBeginCapture_v2(
                stream,
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        };
        check("cuStreamBeginCapture_v2", result);

        let record = unsafe { cu::cuMemsetD8Async(address, value, PROBE_LEN, stream) };
        let end = unsafe { cu::cuStreamEndCapture(stream, &mut graph) };
        check("cuMemsetD8Async during capture", record);
        check("cuStreamEndCapture", end);
        assert!(!graph.is_null(), "cuStreamEndCapture returned a null graph");

        let result = unsafe { cu::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        check("cuGraphInstantiateWithFlags", result);
        assert!(
            !exec.is_null(),
            "cuGraphInstantiateWithFlags returned a null exec"
        );

        Self { graph, exec }
    }

    fn replay(&self, stream: cu::CUstream) {
        let result = unsafe { cu::cuGraphLaunch(self.exec, stream) };
        check("cuGraphLaunch", result);
        let result = unsafe { cu::cuStreamSynchronize(stream) };
        check("cuStreamSynchronize after cuGraphLaunch", result);
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
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn captured_graph_replay_writes_new_physical_memory_after_full_remap() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);
    let stream = create_stream();
    let base = reserve(granule);
    let mut handle = Some(create_handle(device, granule));
    map_handle(device, base, granule, handle.expect("handle is present"));
    let mut mapped = true;

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        write_host(base, 0x00, PROBE_LEN);
        let graph = CapturedMemset::capture(stream, base, GRAPH_VALUE);
        graph.replay(stream);
        assert_device_bytes(base, GRAPH_VALUE, "initial replay");

        let sync = unsafe { cu::cuStreamSynchronize(stream) };
        check("cuStreamSynchronize before cuMemUnmap", sync);
        unmap(base, granule);
        mapped = false;
        release_handle(handle.take().expect("handle is present"));

        handle = Some(create_handle(device, granule));
        map_handle(device, base, granule, handle.expect("handle is present"));
        mapped = true;
        let fresh = read_host(base, PROBE_LEN);
        eprintln!(
            "fresh full-remap mapping before sentinel: first byte 0x{:02x}; all 0x5a = {}; all zero = {}",
            fresh[0],
            fresh.iter().all(|&byte| byte == GRAPH_VALUE),
            fresh.iter().all(|&byte| byte == 0),
        );
        write_host(base, 0x77, PROBE_LEN);
        assert_device_bytes(base, 0x77, "fresh mapping before replay");

        graph.replay(stream);
        assert_device_bytes(base, GRAPH_VALUE, "replay after full remap");
    }));

    unmap_if_mapped(base, granule, mapped);
    if let Some(handle) = handle {
        release_handle(handle);
    }
    free_reservation(base, granule);
    destroy_stream(stream);
    result.unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn captured_graph_replay_survives_growth_shaped_additional_mapping() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);
    let stream = create_stream();
    let base = reserve(granule * 2);
    let first = create_handle(device, granule);
    let second = create_handle(device, granule);
    map_handle(device, base, granule, first);
    let mut second_mapped = false;

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        write_host(base, 0x00, PROBE_LEN);
        let graph = CapturedMemset::capture(stream, base, GRAPH_VALUE);
        graph.replay(stream);
        assert_device_bytes(base, GRAPH_VALUE, "initial replay");

        let sync = unsafe { cu::cuStreamSynchronize(stream) };
        check("cuStreamSynchronize before growth map", sync);
        map_handle(device, base + granule as u64, granule, second);
        second_mapped = true;
        write_host(base, 0x00, PROBE_LEN);
        write_host(base + granule as u64, 0x77, PROBE_LEN);

        graph.replay(stream);
        assert_device_bytes(base, GRAPH_VALUE, "replay after growth-shaped map");
        assert_device_bytes(
            base + granule as u64,
            0x77,
            "additional granule not referenced by captured graph",
        );
    }));

    unmap_if_mapped(base + granule as u64, granule, second_mapped);
    release_handle(second);
    unmap(base, granule);
    release_handle(first);
    free_reservation(base, granule * 2);
    destroy_stream(stream);
    result.unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cu_mem_map_during_thread_local_capture_returns_success_on_this_driver() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);
    let stream = create_stream();
    let base = reserve(granule);
    let handle = create_handle(device, granule);
    let mut graph = std::ptr::null_mut();
    let mut mapped = false;

    let begin = unsafe {
        cu::cuStreamBeginCapture_v2(
            stream,
            cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
    };
    check("cuStreamBeginCapture_v2", begin);

    let map = unsafe { cu::cuMemMap(base, granule, 0, handle, 0) };
    if map == cu::CUresult::CUDA_SUCCESS {
        set_access(device, base, granule);
        mapped = true;
    }
    let end = unsafe { cu::cuStreamEndCapture(stream, &mut graph) };

    eprintln!(
        "cuMemMap during thread-local capture: {map:?}; cuStreamEndCapture after cuMemMap: {end:?}"
    );
    assert_eq!(
        map,
        cu::CUresult::CUDA_SUCCESS,
        "cuMemMap during thread-local capture returned {map:?}"
    );
    assert_eq!(
        end,
        cu::CUresult::CUDA_SUCCESS,
        "cuStreamEndCapture after cuMemMap returned {end:?}"
    );

    if !graph.is_null() {
        let _ = unsafe { cu::cuGraphDestroy(graph) };
    }
    unmap_if_mapped(base, granule, mapped);
    release_handle(handle);
    free_reservation(base, granule);
    destroy_stream(stream);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn one_physical_handle_mapped_at_two_virtual_addresses_is_visible_to_captured_graph() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);
    let stream = create_stream();
    let alias_a = reserve(granule);
    let alias_b = reserve(granule);
    let output = reserve(granule);
    let shared = create_handle(device, granule);
    let output_handle = create_handle(device, granule);
    map_handle(device, alias_a, granule, shared);
    map_handle(device, alias_b, granule, shared);
    map_handle(device, output, granule, output_handle);

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        write_host(alias_b, ALIAS_VALUE, PROBE_LEN);
        write_host(output, 0x00, PROBE_LEN);

        let mut graph = std::ptr::null_mut();
        let mut exec = std::ptr::null_mut();
        let begin = unsafe {
            cu::cuStreamBeginCapture_v2(
                stream,
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        };
        check("cuStreamBeginCapture_v2", begin);
        let copy = unsafe { cu::cuMemcpyDtoDAsync_v2(output, alias_a, PROBE_LEN, stream) };
        let end = unsafe { cu::cuStreamEndCapture(stream, &mut graph) };
        check("cuMemcpyDtoDAsync_v2 during capture", copy);
        check("cuStreamEndCapture", end);
        check("cuGraphInstantiateWithFlags", unsafe {
            cu::cuGraphInstantiateWithFlags(&mut exec, graph, 0)
        });
        check("cuGraphLaunch", unsafe { cu::cuGraphLaunch(exec, stream) });
        check("cuStreamSynchronize after alias graph launch", unsafe {
            cu::cuStreamSynchronize(stream)
        });
        assert_device_bytes(
            output,
            ALIAS_VALUE,
            "captured copy from alias A after write to alias B",
        );

        let _ = unsafe { cu::cuGraphExecDestroy(exec) };
        let _ = unsafe { cu::cuGraphDestroy(graph) };
    }));

    unmap(output, granule);
    release_handle(output_handle);
    unmap(alias_b, granule);
    unmap(alias_a, granule);
    release_handle(shared);
    free_reservation(output, granule);
    free_reservation(alias_b, granule);
    free_reservation(alias_a, granule);
    destroy_stream(stream);
    result.unwrap();
}
