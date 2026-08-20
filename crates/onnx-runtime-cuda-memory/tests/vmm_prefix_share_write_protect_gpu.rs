//! Q3 for #777 — is write protection on a **shared prefix** granule *sound* and
//! *non-sticky*? This isolated binary records the hardware answer without
//! contaminating any other test's CUDA context.
//!
//! A shared prefix granule is read-only by construction: it is committed prefix
//! that every concurrent sequence maps and reads; only the sequence extending
//! past it writes, and it writes into its *own private* tail, never the prefix.
//! The defence against a mis-targeted store is to map the shared granule
//! `PROT_READ` so a write faults instead of silently corrupting the KV that
//! *every other request* is reading through the same physical page.
//!
//! A fault is only an acceptable defence if it is **non-sticky** — if it rejects
//! the offending operation and leaves the context usable. A write fault that
//! poisons the whole context is worse than the corruption it prevents: it
//! converts one mis-targeted store into a dead process serving many requests.
//! **That would be a kill finding for this mechanism, and this probe reports it
//! rather than working around it.**
//!
//! Copy-engine and memset writes are rejected non-stickily and leave the other
//! sharer uncorrupted. A real kernel `st.global`, however, faults with
//! `CUDA_ERROR_ILLEGAL_ADDRESS` and poisons the context on A100. The allocator
//! therefore must not advertise shared mapping as a production capability.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

mod support;
use support::{read_through_device, write_through_device};

/// Arbitrary non-zero byte written to the prefix while writable, to confirm
/// reads work and the shared page is intact. Not a production fill.
const PREFIX_MARKER: u8 = 0x5a;

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

fn map(address: cu::CUdeviceptr, size: usize, handle: cu::CUmemGenericAllocationHandle) {
    let result = unsafe { cu::cuMemMap(address, size, 0, handle, 0) };
    check("cuMemMap", result);
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
/// a brand-new VA — nothing to do with the shared prefix — must all succeed. If
/// the earlier write poisoned the context, these fail with a sticky error.
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

/// Q3 — copy-engine writes are non-sticky, but a production-shaped kernel store
/// proves that read-only shared-prefix protection poisons the CUDA context.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn shared_prefix_kernel_write_fault_is_sticky() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    // One physical prefix handle mapped into two sequences: `victim` is the
    // read-only prefix that receives the stray write; `other` is a second sharer
    // whose read-back proves the rejected write did not corrupt the shared page.
    let shared = create_handle(device, granule);
    let victim = reserve(granule);
    let other = reserve(granule);
    map(victim, granule, shared);
    map(other, granule, shared);

    // Fill the prefix while `victim` is still writable, then downgrade it to
    // read-only — the production posture for a committed shared prefix.
    set_access_flags(
        device,
        victim,
        granule,
        cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    );
    set_access_flags(
        device,
        other,
        granule,
        cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
    );

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        write_host(victim, PREFIX_MARKER, granule);
        check("cuCtxSynchronize(initial fill)", unsafe {
            cu::cuCtxSynchronize()
        });
        set_access_flags(
            device,
            victim,
            granule,
            cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
        );

        // A read of the read-only shared prefix still works, through both VAs.
        let via_victim = read_through_device(victim, 4096);
        assert!(
            via_victim.iter().all(|&b| b == PREFIX_MARKER),
            "a read of the read-only shared prefix must still succeed and return its contents"
        );
        let via_other = read_through_device(other, 4096);
        assert!(
            via_other.iter().all(|&b| b == PREFIX_MARKER),
            "the other sharer must read the same shared prefix bytes"
        );

        // Path 1: synchronous copy-engine write into the read-only prefix.
        let payload = vec![0x00u8; 4096];
        let sync_write =
            unsafe { cu::cuMemcpyHtoD_v2(victim, payload.as_ptr().cast(), payload.len()) };
        eprintln!("Q3 synchronous cuMemcpyHtoD_v2 into PROT_READ shared prefix: {sync_write:?}");
        assert_ne!(
            sync_write,
            cu::CUresult::CUDA_SUCCESS,
            "writing a read-only shared prefix must fail, or the shared page is unprotected"
        );
        match context_is_healthy(device, granule) {
            Ok(()) => eprintln!("Q3 context healthy after synchronous write fault (non-sticky)"),
            Err(err) => panic!(
                "KILL FINDING: a synchronous write to the read-only shared prefix POISONED the \
                 CUDA context (recovery probe returned {err:?}). The write-protection fault is \
                 sticky, so this defence is worse than the corruption it prevents."
            ),
        }
        // Corruption check: the other sharer still reads the original prefix.
        let other_after_sync = read_through_device(other, 4096);
        assert!(
            other_after_sync.iter().all(|&b| b == PREFIX_MARKER),
            "the rejected synchronous write must NOT have corrupted the shared page seen by the \
             other sharer"
        );

        // Path 2: asynchronous memset whose fault surfaces at the stream sync,
        // the closest copy-free proxy for a kernel-issued store.
        let mut stream = std::ptr::null_mut();
        check("cuStreamCreate", unsafe {
            cu::cuStreamCreate(
                &mut stream,
                cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            )
        });
        let async_write = unsafe { cu::cuMemsetD8Async(victim, 0x00, granule, stream) };
        let async_sync = unsafe { cu::cuStreamSynchronize(stream) };
        let _ = unsafe { cu::cuStreamDestroy_v2(stream) };
        eprintln!(
            "Q3 asynchronous cuMemsetD8Async into PROT_READ shared prefix: enqueue \
             {async_write:?}, sync {async_sync:?}"
        );
        assert!(
            async_write != cu::CUresult::CUDA_SUCCESS || async_sync != cu::CUresult::CUDA_SUCCESS,
            "an asynchronous write to the read-only shared prefix must surface a fault at enqueue \
             or sync"
        );
        match context_is_healthy(device, granule) {
            Ok(()) => eprintln!("Q3 context healthy after asynchronous write fault (non-sticky)"),
            Err(err) => panic!(
                "KILL FINDING: an asynchronous write to the read-only shared prefix POISONED the \
                 CUDA context (recovery probe returned {err:?}). A stream-level write fault into \
                 the shared prefix is sticky and kills the process."
            ),
        }

        let (kernel_write, kernel_sync) = write_through_device(victim, 0x00);
        eprintln!(
            "Q3 kernel store into PROT_READ shared prefix: launch {kernel_write:?}, \
             sync {kernel_sync:?}"
        );
        assert!(
            kernel_write != cu::CUresult::CUDA_SUCCESS || kernel_sync != cu::CUresult::CUDA_SUCCESS,
            "a kernel store to the read-only shared prefix must surface a fault"
        );
        let context_error = context_is_healthy(device, granule)
            .expect_err("a kernel protection fault is expected to poison the CUDA context");
        eprintln!(
            "Q3 KILL FINDING: kernel write fault is sticky; recovery returned {context_error:?}"
        );

        eprintln!(
            "Q3 result: copy-engine writes are rejected non-stickily, but a production-shaped \
             kernel store poisons the context -- CUDA shared mapping must remain unadvertised."
        );
    }));

    unmap(other, granule);
    unmap(victim, granule);
    release_handle(shared);
    free_reservation(other, granule);
    free_reservation(victim, granule);
    result.unwrap();
}
