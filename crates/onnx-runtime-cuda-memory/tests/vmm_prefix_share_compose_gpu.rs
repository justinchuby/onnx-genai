//! Q4 for #777 — does prefix sharing **compose with the #759 dummy-page
//! mapping** in one reservation? Both want `cuMemSetAccess` semantics on the
//! same address space: the shared prefix granules are read-only in the committed
//! *head*, the dummy page is read-only in the uncommitted *tail*, and the
//! sequence's own live region between them is read/write. In its own binary
//! because it deliberately provokes write faults on the read-only regions.
//!
//! # The layout under test (token-major, one reservation per sequence)
//!
//! ```text
//! | shared prefix (RO) | private live (RW) |        dummy tail (RO)          |
//! [==== granule 0 ====][==== granule 1 ===][= g2 =][= g3 =] ... [=== gK ===]
//!   one shared handle    private handle       one dummy handle at every tail VA
//! ```
//!
//! * The shared prefix aliases one physical handle shared with other sequences
//!   (the #727/#777 primitive), mapped `PROT_READ`.
//! * The private live granule is this sequence's own handle, `PROT_READWRITE`.
//! * The dummy tail aliases one dummy handle across every uncommitted granule
//!   (the #759 primitive), mapped `PROT_READ` so a speculative read past the
//!   live length hits backed memory instead of faulting.
//!
//! The test confirms all three postures coexist in one reservation: reads
//! succeed everywhere (prefix, live, dummy tail), writes succeed only in the
//! private live region, and writes to either read-only region fault **loudly and
//! non-stickily** without corrupting the shared prefix or the shared dummy page.
//! If they cannot coexist, the conflict is reported rather than worked around.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

const PREFIX_MARKER: u8 = 0x5a;
const LIVE_MARKER: u8 = 0x11;
const DUMMY_MARKER: u8 = 0x77;

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

fn read_host(address: cu::CUdeviceptr, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    let result = unsafe { cu::cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(), address, bytes.len()) };
    check("cuMemcpyDtoH_v2", result);
    bytes
}

/// A fresh independent allocate/write/read on a new VA — proves the context is
/// not poisoned by an earlier write fault.
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
        let probe = vec![0x22u8; 4096];
        let w = unsafe { cu::cuMemcpyHtoD_v2(base, probe.as_ptr().cast(), probe.len()) };
        if w != cu::CUresult::CUDA_SUCCESS {
            return Err(w);
        }
        let mut back = vec![0u8; 4096];
        let r = unsafe { cu::cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), base, back.len()) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(r);
        }
        if back.iter().any(|&b| b != 0x22) {
            return Err(cu::CUresult::CUDA_ERROR_UNKNOWN);
        }
        Ok(())
    })();
    unmap(base, granule);
    release_handle(handle);
    free_reservation(base, granule);
    outcome
}

/// Q4 — one reservation carrying a read-only shared prefix head, a read/write
/// private live granule, and a read-only shared dummy tail, all with distinct
/// `cuMemSetAccess` postures. Reads succeed everywhere; writes succeed only in
/// the private region; writes to either read-only region fault non-stickily and
/// leave both shared pages intact.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn shared_prefix_and_dummy_tail_coexist_in_one_reservation() {
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    const TOTAL_GRANULES: usize = 6;
    const PREFIX_GRANULES: usize = 1; // shared committed head
    const LIVE_GRANULES: usize = 1; // this sequence's private live region
    let total = granule * TOTAL_GRANULES;

    // A second reservation aliasing the same shared prefix and dummy handles, so
    // the corruption checks have another sharer to read back.
    let base = reserve(total);
    let other_prefix = reserve(granule);

    let shared_prefix = create_handle(device, granule);
    let private = create_handle(device, granule);
    let dummy = create_handle(device, granule);

    // Head: shared prefix, read-only.
    map(base, granule, shared_prefix);
    map(other_prefix, granule, shared_prefix);
    // Middle: private live region, read/write.
    let live_off = (PREFIX_GRANULES * granule) as u64;
    map(base + live_off, granule, private);
    // Tail: one dummy handle at every uncommitted granule, read-only.
    for g in (PREFIX_GRANULES + LIVE_GRANULES)..TOTAL_GRANULES {
        map(base + (g * granule) as u64, granule, dummy);
    }

    // Fill the shared regions while writable, then apply the production access
    // postures: prefix RO, live RW, dummy tail RO.
    set_access_flags(
        device,
        base,
        granule,
        cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    );
    set_access_flags(
        device,
        base + live_off,
        granule,
        cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    );
    // Write the dummy through its first tail VA while writable.
    let first_tail = base + ((PREFIX_GRANULES + LIVE_GRANULES) * granule) as u64;
    set_access_flags(
        device,
        first_tail,
        granule,
        cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    );

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        write_host(base, PREFIX_MARKER, granule);
        write_host(base + live_off, LIVE_MARKER, granule);
        write_host(first_tail, DUMMY_MARKER, granule);

        // Apply the read-only postures to the shared prefix and the whole dummy
        // tail; leave the private live region read/write.
        set_access_flags(
            device,
            base,
            granule,
            cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
        );
        set_access_flags(
            device,
            other_prefix,
            granule,
            cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
        );
        for g in (PREFIX_GRANULES + LIVE_GRANULES)..TOTAL_GRANULES {
            set_access_flags(
                device,
                base + (g * granule) as u64,
                granule,
                cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
            );
        }

        // Reads succeed across the whole reservation: prefix, live, and every
        // dummy-backed tail granule (a full-padded read that would fault on an
        // unbacked tail, per #772).
        let host = read_host(base, total);
        assert!(
            host[..granule].iter().all(|&b| b == PREFIX_MARKER),
            "the read-only shared prefix must read back its contents"
        );
        assert!(
            host[granule..2 * granule].iter().all(|&b| b == LIVE_MARKER),
            "the private live region must read back its contents"
        );
        assert!(
            host[2 * granule..].iter().all(|&b| b == DUMMY_MARKER),
            "every dummy-backed tail granule must alias the one dummy page and read the marker"
        );

        // The private live region is writable — the sequence extends its KV
        // here without any fault, while the prefix and tail stay protected.
        write_host(base + live_off, LIVE_MARKER ^ 0xff, 4096);
        let live_back = read_host(base + live_off, 4096);
        assert!(
            live_back.iter().all(|&b| b == LIVE_MARKER ^ 0xff),
            "the private live region must accept writes while the shared regions are protected"
        );

        // A write into the read-only shared prefix faults, non-stickily, and
        // does not corrupt the other sharer.
        let payload = vec![0x00u8; 4096];
        let prefix_write =
            unsafe { cu::cuMemcpyHtoD_v2(base, payload.as_ptr().cast(), payload.len()) };
        eprintln!("Q4 write into read-only shared prefix: {prefix_write:?}");
        assert_ne!(
            prefix_write,
            cu::CUresult::CUDA_SUCCESS,
            "a write into the read-only shared prefix must fault"
        );
        match context_is_healthy(device, granule) {
            Ok(()) => eprintln!("Q4 context healthy after prefix write fault (non-sticky)"),
            Err(err) => panic!(
                "KILL FINDING: writing the read-only shared prefix POISONED the context ({err:?}); \
                 prefix sharing cannot compose with dummy-page protection on this hardware."
            ),
        }
        let other_after = read_host(other_prefix, 4096);
        assert!(
            other_after.iter().all(|&b| b == PREFIX_MARKER),
            "the rejected prefix write must not corrupt the other sharer's view"
        );

        // A write into the read-only dummy tail faults, non-stickily, and does
        // not corrupt the shared dummy page seen at other tail VAs.
        let tail_write =
            unsafe { cu::cuMemcpyHtoD_v2(first_tail, payload.as_ptr().cast(), payload.len()) };
        eprintln!("Q4 write into read-only dummy tail: {tail_write:?}");
        assert_ne!(
            tail_write,
            cu::CUresult::CUDA_SUCCESS,
            "a write into the read-only dummy tail must fault"
        );
        match context_is_healthy(device, granule) {
            Ok(()) => eprintln!("Q4 context healthy after dummy-tail write fault (non-sticky)"),
            Err(err) => panic!(
                "KILL FINDING: writing the read-only dummy tail POISONED the context ({err:?})."
            ),
        }
        let last_tail = base + ((TOTAL_GRANULES - 1) * granule) as u64;
        let tail_after = read_host(last_tail, 4096);
        assert!(
            tail_after.iter().all(|&b| b == DUMMY_MARKER),
            "the rejected tail write must not corrupt the shared dummy page at other tail VAs"
        );

        eprintln!(
            "Q4 result: a read-only shared prefix head, a read/write private live granule, and a \
             read-only shared dummy tail COEXIST in one reservation. Reads succeed everywhere, only \
             the private region accepts writes, and both shared pages survive rejected writes. \
             Prefix sharing (#727/#777) composes with the dummy page (#759) on this hardware."
        );
    }));

    for g in (PREFIX_GRANULES + LIVE_GRANULES)..TOTAL_GRANULES {
        unmap(base + (g * granule) as u64, granule);
    }
    unmap(base + live_off, granule);
    unmap(other_prefix, granule);
    unmap(base, granule);
    release_handle(dummy);
    release_handle(private);
    release_handle(shared_prefix);
    free_reservation(other_prefix, granule);
    free_reservation(base, total);
    result.unwrap();
}
