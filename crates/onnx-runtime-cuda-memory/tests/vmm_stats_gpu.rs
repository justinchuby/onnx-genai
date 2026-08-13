//! Do the process-global VMM counters report what the arena actually did?
//!
//! The arena was once installed, logged that it was installed, and committed
//! **zero bytes** for an entire generation (#659). The log line was true and
//! useless: nothing could tell it apart from a hook that never fired. Counters
//! can — but only if they are wired to the commit and release paths, which is
//! exactly the kind of wiring that rots silently.
//!
//! # Why this is its own test binary
//!
//! The counters are process-global. Any other test in the same binary that
//! allocates from an arena moves them, so a test that reads absolute values
//! must not share a process with one. Cargo gives each integration test file
//! its own binary, so this file is the isolation.
//!
//! Skips loudly without a GPU, like the other `*_gpu` tests (#636).

use cudarc::driver::CudaContext;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CudaVmmAllocator, global_vmm_stats, reset_global_vmm_stats,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole,
};

const HOLDER: HolderId = HolderId::new(22);

fn allocator(capacity: usize) -> (CudaVmmAllocator, LedgerGovernor) {
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "VMM stats test requires a CUDA driver; CPU-only runs must leave this test ignored: {error}"
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

/// A full allocate/free cycle moves every counter, and leaves the levels at
/// zero.
///
/// One test rather than several because the counters are global: two tests
/// asserting absolute values cannot run concurrently, and splitting them would
/// buy nothing but a `--test-threads=1` requirement.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn the_counters_follow_a_full_allocate_and_free_cycle() {
    reset_global_vmm_stats();

    let before = global_vmm_stats();
    assert_eq!(
        before,
        Default::default(),
        "the reset must clear every counter, or later assertions read stale state"
    );

    let (allocator, _governor) = allocator(64 << 20);

    let reserved = global_vmm_stats();
    assert_eq!(
        reserved.reserved_bytes,
        64 << 20,
        "reserving address space must be visible; an arena that reserved and \
         never committed is the #659 failure and must be distinguishable from \
         no arena at all"
    );
    assert_eq!(
        reserved.commits, 0,
        "reservation must not commit -- that is the whole premise"
    );
    assert_eq!(reserved.committed_bytes, 0);

    let allocation_bytes = 8 << 20;
    let pointer = allocator
        .allocate(allocation_bytes, 256)
        .expect("8 MiB fits");

    let live = global_vmm_stats();
    assert_eq!(live.allocations, 1, "the span handed out must be counted");
    assert!(
        live.commits >= 1,
        "backing an 8 MiB request must map at least one granule"
    );
    let (committed, _) = allocator.committed_and_reserved();
    assert_eq!(
        live.committed_bytes, committed as u64,
        "the counter and the arena must agree; a counter that drifts from the \
         arena is worse than no counter"
    );
    assert_eq!(
        live.peak_committed_bytes, live.committed_bytes,
        "the first commit is also the peak"
    );

    // SAFETY: the pointer came from this allocator and is still live.
    unsafe { allocator.deallocate(pointer, allocation_bytes, 256) };

    let freed = global_vmm_stats();
    assert_eq!(
        freed.committed_bytes, 0,
        "releasing the last allocation in a granule must unmap it and say so"
    );
    assert_eq!(
        freed.releases, 1,
        "one contiguous allocation must be retired with one cuMemUnmap run, \
         not one driver call per physical granule"
    );
    assert_eq!(
        freed.peak_committed_bytes, live.committed_bytes,
        "the peak is a high-water mark and must survive the free"
    );
    assert_eq!(
        freed.reserved_bytes,
        64 << 20,
        "address space is held until the arena is dropped"
    );

    drop(allocator);

    let gone = global_vmm_stats();
    assert_eq!(
        gone.reserved_bytes, 0,
        "a dropped arena must take its reservation off the books, or a second \
         run in the same process reads the first run's memory as still held"
    );
    assert_eq!(gone.committed_bytes, 0);
    assert_eq!(
        gone.ref_underflows, 0,
        "a correct allocate/free cycle must never release a granule whose \
         reference count is already zero; a non-zero reading means some \
         allocation committed a granule without taking a reference for it, and \
         the arena has unmapped memory another allocation believes it owns"
    );
}
