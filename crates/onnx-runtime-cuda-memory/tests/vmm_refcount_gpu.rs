//! Do the VMM arena's granule reference counts balance?
//!
//! Its own test binary because the counters are process-global: any other
//! test in the same binary that allocates moves them, so a test reading
//! absolute values must not share a process with one. (The first draft of this
//! lived alongside the counter test and the two immediately corrupted each
//! other's readings, which is the failure this separation prevents.)

use cudarc::driver::CudaContext;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CudaVmmAllocator, global_vmm_stats, reset_global_vmm_stats,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
    Tier,
};

const HOLDER: HolderId = HolderId::new(23);

fn allocator(capacity: usize) -> (CudaVmmAllocator, LedgerGovernor) {
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "VMM refcount test requires a CUDA driver; CPU-only runs must leave this test ignored: {error}"
        ),
    };
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = CudaVmmAllocator::new(
        context,
        DeviceKey::device(0),
        0,
        capacity,
        &governor,
        HOLDER,
        MemoryRole::Workspace { step_scoped: false },
    )
    .expect("reserving device address space");
    (allocator, governor)
}
/// Granule reference counts balance across many overlapping allocations.
///
/// # Why this exists
///
/// A code review of #682 found a commit path that mapped a granule without
/// taking a reference for it. The release path then decremented a count that
/// was never incremented, hit zero early, and unmapped memory a *different*
/// live allocation was using -- so the model read foreign data rather than
/// crashing. The release path's response to an already-zero count is to skip,
/// which is the safe action but a silent one, so nothing distinguished that
/// from correct operation.
///
/// Small allocations that share granules are what makes the imbalance
/// reachable: spans are carved byte-wise, so a boundary granule routinely
/// backs two live allocations at once. This exercises exactly that, in an
/// interleaved order, and reads the counter that the skip now records.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn granule_reference_counts_balance_across_shared_allocations() {
    reset_global_vmm_stats();

    let (allocator, governor) = allocator(64 << 20);

    // Sized so that many land inside one 2 MiB granule, and some straddle a
    // boundary -- a granule backing two live allocations is the case the
    // refcount exists for.
    let mut live: Vec<_> = (0..64)
        .map(|_| allocator.allocate(48 << 10, 256).expect("48 KiB fits"))
        .collect();

    // Free every other one first, so releases interleave with granules that
    // are still held. Freeing in allocation order would let each granule's
    // count reach zero only after its last tenant left, which is the easy
    // case and would not provoke an imbalance.
    let mut kept = Vec::new();
    for (index, pointer) in live.drain(..).enumerate() {
        if index % 2 == 0 {
            // SAFETY: the pointer came from this allocator and is still live.
            unsafe { allocator.deallocate(pointer, 48 << 10, 256) };
        } else {
            kept.push(pointer);
        }
    }
    assert_eq!(
        global_vmm_stats().ref_underflows,
        0,
        "freeing half of a set of granule-sharing allocations must not \
         over-release: the granules still have tenants"
    );

    for pointer in kept {
        // SAFETY: the pointer came from this allocator and is still live.
        unsafe { allocator.deallocate(pointer, 48 << 10, 256) };
    }

    let stats = global_vmm_stats();
    assert_eq!(
        stats.ref_underflows, 0,
        "every granule must be released exactly as many times as it was \
         referenced"
    );
    assert_eq!(
        stats.committed_bytes, 0,
        "the last tenant of each granule must unmap it"
    );
    assert_eq!(
        stats.byte_underflows, 0,
        "every byte subtracted from the counters must have been added first; a \
         clamp here means some commit path mapped memory without counting it, \
         and the committed figure in --profile is a lower bound rather than a \
         measurement"
    );
    assert_eq!(
        stats.releases, stats.commits,
        "commits and releases must balance once every allocation is gone"
    );
    assert_eq!(
        governor.used(Tier::Device),
        0,
        "released granules must come back to the tier; a lease still held \
         after every allocation is freed is a leak the driver would not report \
         until a later allocation failed"
    );
}
