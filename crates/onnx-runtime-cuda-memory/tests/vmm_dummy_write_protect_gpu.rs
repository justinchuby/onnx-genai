//! Q3 for #759 — is the dummy page's write-protection fault *non-sticky*? In
//! its own test binary so that if a write to a read-only mapping poisons the
//! CUDA context, only this probe is affected and the finding is unambiguous.
//!
//! The #759 dummy page is *shared*: one physical page backs the entire tail of
//! every KV reservation. A stray *write* into the tail would therefore corrupt
//! the dummy page seen by every other alias. The defence is to map the dummy
//! `PROT_READ` so writes fault. But a fault is only an acceptable defence if it
//! is *non-sticky* — if it merely rejects the offending operation and leaves
//! the context usable. A write fault that poisons the whole context is worse
//! than the uncommitted-tail fault it was meant to replace: it converts a
//! single bad step into a dead process. That would be a **kill finding** for
//! this mechanism, and this probe reports it rather than working around it.
//!
//! We exercise two write paths into the read-only dummy — a synchronous
//! copy-engine `cuMemcpyHtoD_v2` and an asynchronous `cuMemsetD8Async` whose
//! fault surfaces at the stream sync (closest to a kernel-issued store) — and
//! after each we probe context health with `cuCtxSynchronize` plus a fresh
//! independent allocate/write/read. The test *passes* if the context survives
//! (non-sticky) and *fails loudly* if it is poisoned.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

/// Arbitrary non-zero byte written to confirm reads work; not the production
/// fill (which the masking rule decides — zeros, never NaN).
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

fn set_access_flags(
    device_ordinal: i32,
    address: cu::CUdeviceptr,
    size: usize,
    flags: cu::CUmemAccess_flags,
) {
    let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_ordinal;
    access.flags = flags;
    let result = unsafe { cu::cuMemSetAccess(address, size, &access, 1) };
    check("cuMemSetAccess", result);
}

fn map(
    device_ordinal: i32,
    address: cu::CUdeviceptr,
    size: usize,
    handle: cu::CUmemGenericAllocationHandle,
    flags: cu::CUmemAccess_flags,
) {
    let result = unsafe { cu::cuMemMap(address, size, 0, handle, 0) };
    check("cuMemMap", result);
    set_access_flags(device_ordinal, address, size, flags);
}

fn unmap(address: cu::CUdeviceptr, size: usize) {
    let _ = unsafe { cu::cuMemUnmap(address, size) };
}

fn write_host(address: cu::CUdeviceptr, value: u8, len: usize) {
    let bytes = vec![value; len];
    let result = unsafe { cu::cuMemcpyHtoD_v2(address, bytes.as_ptr().cast(), bytes.len()) };
    check("cuMemcpyHtoD_v2", result);
}

/// Does the context still work? A fresh reservation, commit, write, and read on
/// a brand-new VA — nothing to do with the read-only dummy — must all succeed.
/// If the earlier write poisoned the context, these fail with a sticky error.
fn context_is_healthy(device: i32, granule: usize) -> Result<(), cu::CUresult> {
    let sync = unsafe { cu::cuCtxSynchronize() };
    if sync != cu::CUresult::CUDA_SUCCESS {
        return Err(sync);
    }
    let base = reserve(granule);
    let handle = create_handle(device, granule);
    let map_result = unsafe { cu::cuMemMap(base, granule, 0, handle, 0) };
    let outcome = (|| {
        if map_result != cu::CUresult::CUDA_SUCCESS {
            return Err(map_result);
        }
        set_access_flags(
            device,
            base,
            granule,
            cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        );
        let probe = vec![0x11u8; 4096];
        let w = unsafe { cu::cuMemcpyHtoD_v2(base, probe.as_ptr().cast(), probe.len()) };
        if w != cu::CUresult::CUDA_SUCCESS {
            return Err(w);
        }
        let mut back = vec![0u8; 4096];
        let r = unsafe { cu::cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), base, back.len()) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(r);
        }
        if back.iter().any(|&b| b != 0x11) {
            return Err(cu::CUresult::CUDA_ERROR_UNKNOWN);
        }
        Ok(())
    })();
    unmap(base, granule);
    release_handle(handle);
    free_reservation(base, granule);
    outcome
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn read_only_dummy_write_fault_is_non_sticky() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    let base = reserve(granule);
    let dummy = create_handle(device, granule);
    // Fill the dummy with a readback marker while it is still writable, then
    // downgrade the mapping to read-only — the production posture for a shared
    // dummy page.
    map(
        device,
        base,
        granule,
        dummy,
        cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    );

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        write_host(base, READBACK_MARKER, granule);
        set_access_flags(
            device,
            base,
            granule,
            cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
        );

        // A read still works after the downgrade.
        let mut back = vec![0u8; 4096];
        let read = unsafe { cu::cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), base, back.len()) };
        assert_eq!(
            read,
            cu::CUresult::CUDA_SUCCESS,
            "a read of the read-only dummy must still succeed; got {read:?}"
        );
        assert!(
            back.iter().all(|&b| b == READBACK_MARKER),
            "read-only dummy keeps its contents"
        );

        // Path 1: synchronous copy-engine write into the read-only mapping.
        let payload = vec![0x00u8; 4096];
        let sync_write =
            unsafe { cu::cuMemcpyHtoD_v2(base, payload.as_ptr().cast(), payload.len()) };
        eprintln!("Q3 synchronous cuMemcpyHtoD_v2 into PROT_READ dummy: {sync_write:?}");
        assert_ne!(
            sync_write,
            cu::CUresult::CUDA_SUCCESS,
            "writing a read-only dummy page must fail, or the shared page is unprotected"
        );
        match context_is_healthy(device, granule) {
            Ok(()) => eprintln!("Q3 context healthy after synchronous write fault (non-sticky)"),
            Err(err) => panic!(
                "KILL FINDING: synchronous write to the read-only dummy POISONED the CUDA context \
                 (recovery probe returned {err:?}). The write-protection fault is sticky, so this \
                 'safety' mechanism is worse than the uncommitted-tail fault it replaces."
            ),
        }

        // Path 2: asynchronous memset whose fault surfaces at the stream sync,
        // the closest copy-free proxy for a kernel-issued store.
        let mut stream = std::ptr::null_mut();
        check("cuStreamCreate", unsafe {
            cu::cuStreamCreate(
                &mut stream,
                cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            )
        });
        let async_write = unsafe { cu::cuMemsetD8Async(base, 0x00, granule, stream) };
        let async_sync = unsafe { cu::cuStreamSynchronize(stream) };
        let _ = unsafe { cu::cuStreamDestroy_v2(stream) };
        eprintln!(
            "Q3 asynchronous cuMemsetD8Async into PROT_READ dummy: enqueue {async_write:?}, \
             sync {async_sync:?}"
        );
        assert!(
            async_write != cu::CUresult::CUDA_SUCCESS || async_sync != cu::CUresult::CUDA_SUCCESS,
            "an asynchronous write to the read-only dummy must surface a fault at enqueue or sync"
        );
        match context_is_healthy(device, granule) {
            Ok(()) => eprintln!("Q3 context healthy after asynchronous write fault (non-sticky)"),
            Err(err) => panic!(
                "KILL FINDING: asynchronous write to the read-only dummy POISONED the CUDA context \
                 (recovery probe returned {err:?}). A stream-level write fault into the shared \
                 dummy is sticky and kills the process; write-protection is not a viable defence."
            ),
        }

        eprintln!(
            "Q3 result: read-only dummy rejects both synchronous and asynchronous writes and the \
             context survives both -- write-protection is a non-sticky, viable defence for the \
             shared dummy page on this hardware."
        );
    }));

    unmap(base, granule);
    release_handle(dummy);
    free_reservation(base, granule);
    result.unwrap();
}
