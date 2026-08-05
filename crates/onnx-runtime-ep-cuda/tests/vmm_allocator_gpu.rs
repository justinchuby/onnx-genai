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

use std::sync::Arc;

use cudarc::driver::CudaContext;
use onnx_runtime_ep_cuda::vmm_allocator::CudaVmmAllocator;
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
    Tier,
};

const HOLDER: HolderId = HolderId::new(21);

/// An allocator over `capacity` bytes of address space, and the ledger behind
/// it, or `None` on a machine with no driver.
fn allocator(capacity: usize, device_bytes: u64) -> Option<(CudaVmmAllocator, LedgerGovernor)> {
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => {
            eprintln!(
                "SKIPPED (no CUDA driver): {error}. These tests verify the VMM device \
                 allocator and did NOT run."
            );
            return None;
        }
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
    Some((allocator, governor))
}

/// Reserving address space commits nothing, so the ledger stays empty until
/// something is actually allocated.
///
/// This is the whole premise: address space is free, physical memory is not.
/// If reservation charged the ledger, a large arena would refuse to start on a
/// small card and the approach would be pointless.
#[test]
fn reserving_an_arena_charges_nothing() {
    let Some((allocator, governor)) = allocator(256 << 20, 8 << 30) else {
        return;
    };

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
#[test]
fn an_allocation_commits_a_granule_and_the_ledger_sees_it() {
    let Some((allocator, governor)) = allocator(64 << 20, 8 << 30) else {
        return;
    };

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
#[test]
fn small_allocations_share_granules_instead_of_each_taking_one() {
    let Some((allocator, governor)) = allocator(64 << 20, 8 << 30) else {
        return;
    };

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

/// Distinct live allocations never overlap.
///
/// The `DeviceAllocator` contract's central safety promise, and the one whose
/// violation would be silent: two sessions would write over each other's
/// tensors and produce wrong numbers rather than a crash.
#[test]
fn live_allocations_do_not_overlap() {
    let Some((allocator, _governor)) = allocator(64 << 20, 8 << 30) else {
        return;
    };

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

/// The ledger can refuse, and a refusal leaves nothing mapped.
///
/// A budget that can be exceeded is not a budget (G1). The important half is
/// the second: an allocation that was refused must not have committed memory
/// on its way to failing, or the tier drifts upward on every refusal.
#[test]
fn a_refused_allocation_commits_nothing() {
    // Four granules of address space, but only ~2 granules of device budget.
    let Some((allocator, governor)) = allocator(8 << 20, 4 << 20) else {
        return;
    };

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
