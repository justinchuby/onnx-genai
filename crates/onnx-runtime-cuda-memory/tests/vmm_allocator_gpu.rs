//! Does the VMM-backed device allocator actually hand out usable memory, and
//! does the ledger see exactly what it committed?
//!
//! [`CudaVmmAllocator`] replaces `cuMemAlloc` with reserve-then-commit, which
//! buys two things that are worth testing separately:
//!
//! * **suballocation** — many allocations share a 2 MiB granule, so the
//!   fragmentation bound is one partial granule per arena rather than per
//!   allocation. If this regressed, small tensors would silently start costing
//!   2 MiB each and the only symptom would be running out of memory sooner.
//! * **complete accounting** — every physical byte is leased before it is
//!   mapped. This is what makes the ledger's device tier true rather than a
//!   lower bound (#652), and it is affordable only because commits happen per
//!   granule rather than per allocation.
//!
//! Skips loudly when no GPU is present, in the same way the other `*_gpu`
//! tests do — a skip that reads like a pass is how 44 tests went unnoticed
//! (#636).

use cudarc::driver::CudaContext;
use onnx_runtime_cuda_memory::vmm_allocator::CudaVmmAllocator;
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
    Tier,
};

const HOLDER: HolderId = HolderId::new(21);

/// An allocator over `capacity` bytes of address space, and the ledger behind
/// it, or `None` on a machine with no driver.
fn allocator(capacity: usize, device_bytes: u64) -> (CudaVmmAllocator, LedgerGovernor) {
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "VMM allocator test requires a CUDA driver; CPU-only runs must leave this test ignored: {error}"
        ),
    };
    let governor = LedgerGovernor::new(LeaseLedger::new(device_bytes, 0, 0));
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

/// Reserving address space commits nothing, so the ledger stays empty until
/// something is actually allocated.
///
/// This is the whole premise: address space is free, physical memory is not.
/// If reservation charged the ledger, a large arena would refuse to start on a
/// small card and the approach would be pointless.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn reserving_an_arena_charges_nothing() {
    let (allocator, governor) = allocator(256 << 20, 8 << 30);

    let (committed, reserved) = allocator.committed_and_reserved();
    assert_eq!(committed, 0, "nothing is mapped yet");
    assert!(
        reserved >= 256 << 20,
        "the arena reserved {reserved} bytes of address space"
    );
    assert_eq!(
        governor.used(Tier::Device),
        0,
        "reserved address space is not memory and must not be charged"
    );
}

/// A single allocation commits one granule and the ledger is told about it.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn an_allocation_commits_a_granule_and_the_ledger_sees_it() {
    let (allocator, governor) = allocator(64 << 20, 8 << 30);

    let pointer = allocator.allocate(4096, 256).expect("4 KiB fits");

    let (committed, _) = allocator.committed_and_reserved();
    assert!(committed > 0, "a live allocation must be backed");
    assert_eq!(
        governor.used(Tier::Device) as usize,
        committed,
        "the ledger must hold exactly the committed bytes, not the request size \
         and not the reservation"
    );

    // SAFETY: the pointer came from this allocator and is still live.
    unsafe { allocator.deallocate(pointer, 4096, 256) };

    assert_eq!(
        allocator.committed_and_reserved().0,
        0,
        "the last allocation out of a granule releases it"
    );
    assert_eq!(
        governor.used(Tier::Device),
        0,
        "released granules must come back to the tier"
    );
}

/// Many small allocations share granules rather than each taking one.
///
/// The property that makes 2 MiB granularity affordable for a runtime that
/// does not patch the CUDA driver. 512 allocations of 4 KiB is 2 MiB of
/// demand; without sharing it would commit 1 GiB.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn small_allocations_share_granules_instead_of_each_taking_one() {
    let (allocator, governor) = allocator(64 << 20, 8 << 30);

    let count = 512;
    let size = 4096;
    let pointers: Vec<_> = (0..count)
        .map(|index| {
            allocator
                .allocate(size, 256)
                .unwrap_or_else(|error| panic!("allocation {index} of {size} B: {error}"))
        })
        .collect();

    let (committed, _) = allocator.committed_and_reserved();
    let demand = count * size;
    assert!(
        committed < demand * 4,
        "{count} x {size} B committed {committed} B; granules are not being \
         shared, so every small tensor is costing a whole granule"
    );
    assert_eq!(governor.used(Tier::Device) as usize, committed);
    eprintln!("{count} x {size} B demand = {demand} B, committed {committed} B");

    for pointer in pointers {
        // SAFETY: each pointer came from this allocator and is live exactly once.
        unsafe { allocator.deallocate(pointer, size, 256) };
    }
    assert_eq!(allocator.committed_and_reserved().0, 0);
    assert_eq!(governor.used(Tier::Device), 0);
}

/// Repeated allocate/free cycles leave nothing behind.
///
/// The property a leak breaks quietly. A granule that stays mapped after its
/// last user leaves, or a lease that is not shrunk when one is released, drifts
/// by a little on every cycle -- invisible in a test that runs one, fatal in a
/// session that runs millions.
///
/// Checks both the physical side (granules unmapped) and the accounting side
/// (the ledger returned to zero), because they can fail independently: a
/// granule can be unmapped while the ledger still believes it is held, which
/// reads as exhausted memory that `nvidia-smi` says is free.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn repeated_allocation_cycles_leak_neither_granules_nor_ledger_bytes() {
    let (allocator, governor) = allocator(64 << 20, 8 << 30);

    for round in 0..200usize {
        // Sizes that straddle the granule boundary in both directions, so
        // rounds share granules, split them, and span several.
        let size = match round % 4 {
            0 => 64,
            1 => 4096,
            2 => (2 << 20) + 7,
            _ => (2 << 20) - 7,
        };
        let pointer = allocator
            .allocate(size, 256)
            .unwrap_or_else(|error| panic!("round {round} of {size} B: {error}"));
        // SAFETY: from this allocator, live, freed exactly once.
        unsafe { allocator.deallocate(pointer, size, 256) };

        assert_eq!(
            allocator.committed_and_reserved().0,
            0,
            "round {round}: a granule stayed mapped after its last user left"
        );
        assert_eq!(
            governor.used(Tier::Device),
            0,
            "round {round}: the ledger still holds bytes nothing is using"
        );
    }
}

/// Freeing in an order that interleaves still returns everything.
///
/// Allocations rarely die in the order they were born. Granule reference counts
/// have to survive that, or a granule shared by two spans is released when the
/// first leaves and the second is left reading unmapped memory -- or is never
/// released at all.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn out_of_order_frees_return_every_granule() {
    let (allocator, governor) = allocator(64 << 20, 8 << 30);

    let sizes = [4096usize, 1 << 20, 64, (3 << 20) + 11, 512, 2 << 20];
    let mut live: Vec<_> = sizes
        .iter()
        .map(|&size| (allocator.allocate(size, 256).expect("fits"), size))
        .collect();

    assert!(
        allocator.committed_and_reserved().0 > 0,
        "the fixture should have committed something to test the release of"
    );

    // Free middle-out rather than in order.
    for index in [3usize, 0, 5, 1, 4, 2] {
        let (pointer, size) = live[index];
        // SAFETY: from this allocator, live, freed exactly once -- each index
        // appears once in the order above.
        unsafe { allocator.deallocate(pointer, size, 256) };
    }
    live.clear();

    assert_eq!(
        allocator.committed_and_reserved().0,
        0,
        "every granule should be unmapped once its last span is gone"
    );
    assert_eq!(governor.used(Tier::Device), 0);
}

/// Distinct live allocations never overlap.
///
/// The `DeviceAllocator` contract's central safety promise, and the one whose
/// violation would be silent: two sessions would write over each other's
/// tensors and produce wrong numbers rather than a crash.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn live_allocations_do_not_overlap() {
    let (allocator, _governor) = allocator(64 << 20, 8 << 30);

    let sizes = [64usize, 4096, 100_000, 8, 1 << 20, 300];
    let mut live: Vec<(usize, usize)> = Vec::new();
    let mut pointers = Vec::new();
    for size in sizes {
        let pointer = allocator.allocate(size, 256).expect("fits");
        let start = pointer.as_ptr() as usize;
        for &(other, other_len) in &live {
            assert!(
                start + size <= other || other + other_len <= start,
                "a {size} B allocation at {start:#x} overlaps a live {other_len} B \
                 allocation at {other:#x}"
            );
        }
        live.push((start, size));
        pointers.push((pointer, size));
    }

    for (pointer, size) in pointers {
        // SAFETY: from this allocator, live, freed once.
        unsafe { allocator.deallocate(pointer, size, 256) };
    }
}

/// Adoption records already-committed private-ledger bytes even over limit.
///
/// This is the #694 wiring test: the arena has already mapped the granules, so
/// `adopt_governor` must record the accomplished fact instead of asking whether
/// the bytes may be taken. If it calls `reserve`, the real governor refuses and
/// the ledger stays at the pre-existing lease.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn adoption_records_committed_bytes_even_when_the_tier_is_over_limit() {
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "VMM allocator test requires a CUDA driver; CPU-only runs must leave this test ignored: {error}"
        ),
    };
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 20, 0, 0));
    let existing = governor
        .reserve(
            Tier::Device,
            6 << 20,
            MemoryRole::Weights,
            HolderId::new(22),
        )
        .expect("fixture lease leaves less room than the arena will commit");
    let allocator = CudaVmmAllocator::detached(
        context,
        DeviceKey::device(0),
        0,
        16 << 20,
        HOLDER,
        MemoryRole::Workspace { step_scoped: false },
    )
    .expect("detached arena reserves address space");

    let pointer = allocator
        .allocate(3 << 20, 256)
        .expect("private startup ledger does not refuse the commit");
    let (committed, _) = allocator.committed_and_reserved();
    let remainder = (8 << 20) - (6 << 20);
    assert!(
        committed as u64 > remainder,
        "fixture must commit more than the real governor's remaining {remainder} bytes"
    );

    let adoption = allocator.adopt_governor(&governor, HOLDER);
    assert_eq!(
        adoption.unaccounted_bytes, 0,
        "the reference governor records committed bytes rather than refusing them"
    );
    assert_eq!(adoption.recorded_bytes, committed as u64);
    assert_eq!(
        governor.used(Tier::Device),
        (6 << 20) + committed as u64,
        "adoption must leave the ledger reporting the true committed total"
    );
    assert_eq!(
        governor.oversubscribed_bytes(Tier::Device),
        (6 << 20) + committed as u64 - (8 << 20),
        "over-subscription must be observable after adoption"
    );

    // SAFETY: the pointer came from this allocator and is still live.
    unsafe { allocator.deallocate(pointer, 3 << 20, 256) };
    drop(existing);
}

/// The ledger can refuse, and a refusal leaves nothing mapped.
///
/// A budget that can be exceeded is not a budget (G1). The important half is
/// the second: an allocation that was refused must not have committed memory
/// on its way to failing, or the tier drifts upward on every refusal.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_refused_allocation_commits_nothing() {
    // Four granules of address space, but only ~2 granules of device budget.
    let (allocator, governor) = allocator(8 << 20, 4 << 20);

    let mut pointers = Vec::new();
    let mut refusals = 0;
    for _ in 0..8 {
        match allocator.allocate(1 << 20, 256) {
            Ok(pointer) => pointers.push(pointer),
            Err(_) => refusals += 1,
        }
    }

    assert!(
        refusals > 0,
        "a 4 MiB budget must refuse some of 8 MiB of demand"
    );
    let (committed, _) = allocator.committed_and_reserved();
    assert!(
        committed as u64 <= 4 << 20,
        "committed {committed} B exceeds the 4 MiB tier limit"
    );
    assert_eq!(
        governor.used(Tier::Device) as usize,
        committed,
        "the ledger and the arena must agree after a refusal"
    );

    for pointer in pointers {
        // SAFETY: from this allocator, live, freed once.
        unsafe { allocator.deallocate(pointer, 1 << 20, 256) };
    }
    assert_eq!(governor.used(Tier::Device), 0);
}
