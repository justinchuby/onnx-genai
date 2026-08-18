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
    AllocationCommitRange, DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor,
    MemoryGovernor, MemoryRole, Tier, VirtualBacking,
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn grant_capacity_commits_with_exact_headroom_and_no_pool() {
    let granule = 2_u64 << 20;
    let (allocator, governor) = allocator(64 << 20, granule);
    let requester = governor
        .reserve_mapped_allowance(
            Tier::Device,
            granule,
            MemoryRole::Workspace { step_scoped: false },
            HolderId::new(22),
        )
        .expect("requester allowance");
    let mut grant = governor
        .prepare_mapped_growth(&requester, granule)
        .expect("reserve exact physical headroom");
    assert!(
        governor
            .reserve(Tier::Device, 1, MemoryRole::Activation, HolderId::new(23))
            .is_err(),
        "ordinary claims remain blocked by the live grant"
    );
    let full = 0..4096;
    let allocation = allocator
        .allocate_committed_with_capacity(
            4096,
            256,
            std::slice::from_ref(&full),
            grant.physical_capacity(),
        )
        .expect("allocator consumes grant-bound capacity");
    let pointer = allocation.allocation;
    assert_eq!(allocation.newly_mapped_bytes, granule);
    assert_eq!(grant.physical_capacity().remaining_bytes(), 0);
    grant.commit_bytes(granule).expect("mapped attribution");
    assert_eq!(governor.used(Tier::Device), granule);
    let unmapped = allocator.deallocate_span(pointer);
    assert_eq!(unmapped, granule, "a nonshared allocation unmaps itself");
    assert_eq!(requester.unmap(unmapped), unmapped);
    assert_eq!(requester.mapped_bytes(), 0);
    assert_eq!(governor.used(Tier::Device), 0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn mapped_allocation_rejects_a_different_arena_zone() {
    let granule = 2_u64 << 20;
    let (allocator, governor) = allocator(64 << 20, granule * 2);
    let arena_zone = governor
        .reserve_mapped_allowance(
            Tier::Device,
            granule,
            MemoryRole::Workspace { step_scoped: false },
            HolderId::new(26),
        )
        .expect("arena-zone allowance");
    let mut first_grant = governor
        .prepare_mapped_growth(&arena_zone, granule)
        .expect("arena-zone grant");
    let full = 0..4096;
    let first = allocator
        .allocate_committed_with_capacity(
            4096,
            256,
            std::slice::from_ref(&full),
            first_grant.physical_capacity(),
        )
        .expect("bind arena mapped owner");
    first_grant
        .commit_bytes(first.newly_mapped_bytes)
        .expect("commit arena-zone mapping");

    let wrong_zone = governor
        .reserve_mapped_allowance(
            Tier::Device,
            granule,
            MemoryRole::Workspace { step_scoped: false },
            HolderId::new(27),
        )
        .expect("different-zone allowance");
    let mut grant = governor
        .prepare_mapped_growth(&wrong_zone, granule)
        .expect("different-zone grant");
    let error = allocator
        .allocate_committed_with_capacity(
            4096,
            256,
            std::slice::from_ref(&full),
            grant.physical_capacity(),
        )
        .expect_err("one arena cannot accept a different mapped owner");
    assert!(error.to_string().contains("different allowance"), "{error}");
    assert_eq!(allocator.committed_and_reserved().0, granule as usize);
    let unmapped = allocator.deallocate_span(first.allocation);
    arena_zone.unmap(unmapped);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn adjacent_shared_granule_consumes_mapped_growth_once() {
    fn run(reverse: bool) {
        let granule = 2_u64 << 20;
        let (allocator, governor) = allocator(64 << 20, granule * 2);
        let requester = governor
            .reserve_mapped_allowance(
                Tier::Device,
                granule * 2,
                MemoryRole::Workspace { step_scoped: false },
                HolderId::new(if reverse { 25 } else { 24 }),
            )
            .expect("requester allowance");
        let full = 0..4096;
        let mut first_grant = governor
            .prepare_mapped_growth(&requester, granule)
            .expect("first granule grant");
        let first = allocator
            .allocate_committed_with_capacity(
                4096,
                256,
                std::slice::from_ref(&full),
                first_grant.physical_capacity(),
            )
            .expect("first workspace allocation");
        assert_eq!(first.newly_mapped_bytes, granule);
        first_grant
            .commit_bytes(first.newly_mapped_bytes)
            .expect("attribute first granule");

        let mut second_grant = governor
            .prepare_mapped_growth(&requester, granule)
            .expect("workspace reserves its rounded upper bound");
        let second = allocator
            .allocate_committed_with_capacity(
                4096,
                256,
                std::slice::from_ref(&full),
                second_grant.physical_capacity(),
            )
            .expect("adjacent packed workspace allocation");
        assert_eq!(second.newly_mapped_bytes, 0);
        second_grant
            .commit_bytes(second.newly_mapped_bytes)
            .expect("attribute no additional bytes");
        assert_eq!(requester.mapped_bytes(), granule);

        let (early, last) = if reverse {
            (second.allocation, first.allocation)
        } else {
            (first.allocation, second.allocation)
        };
        let early_unmapped = allocator.deallocate_span(early);
        assert_eq!(early_unmapped, 0, "the surviving workspace retains mapping");
        requester.unmap(early_unmapped);
        assert_eq!(requester.mapped_bytes(), granule);
        assert_eq!(allocator.committed_and_reserved().0, granule as usize);

        let last_unmapped = allocator.deallocate_span(last);
        assert_eq!(last_unmapped, granule, "last reference owns the unmap");
        requester.unmap(last_unmapped);
        assert_eq!(requester.mapped_bytes(), 0);
        assert_eq!(allocator.committed_and_reserved().0, 0);
        assert_eq!(governor.used(Tier::Device), 0);
    }

    run(false);
    run(true);
}

/// A partial commit is a real allocation claim, not just a map operation.
///
/// KV reserves a large span and commits pieces later. If those late commits do
/// not take a per-allocation reference, a neighboring workspace that shares the
/// boundary granule can free first and unmap live KV. This test reproduces that
/// boundary sharing and asserts the committed granule survives until the KV-like
/// span itself is freed.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn late_committed_ranges_hold_their_own_granule_reference() {
    let (allocator, governor) = allocator(64 << 20, 8 << 30);

    let probe = allocator.allocate(4096, 256).expect("probe allocation");
    let granularity = allocator.committed_and_reserved().0;
    unsafe { allocator.deallocate(probe, 4096, 256) };
    assert!(granularity >= 4096, "probe reveals a real granule");

    let kv_len = granularity + 4096;
    let kv = allocator
        .allocate_committed(kv_len, 256, &[])
        .expect("reserve KV-like span without committing");
    let workspace = allocator
        .allocate(4096, 256)
        .expect("workspace shares KV's tail granule");
    assert_eq!(allocator.committed_and_reserved().0, granularity);

    allocator
        .commit_allocation_range(kv, kv_len, 256, granularity, 4096)
        .expect("KV commits into the shared tail granule");
    unsafe { allocator.deallocate(workspace, 4096, 256) };
    assert_eq!(
        allocator.committed_and_reserved().0,
        granularity,
        "freeing the workspace must not unmap the KV-owned tail granule"
    );

    unsafe { allocator.deallocate(kv, kv_len, 256) };
    assert_eq!(allocator.committed_and_reserved().0, 0);
    assert_eq!(
        governor.used(Tier::Device),
        0,
        "partial committed ranges must release their ledger bytes on free"
    );
}

/// A partially committed allocation must release only the granules it claimed.
///
/// KV reserves a span much larger than the tokens it has reached. Releasing
/// every granule the virtual span overlaps would decrement a neighbor's
/// reference in an uncommitted tail bucket and unmap live workspace memory.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn partially_committed_free_does_not_release_neighbor_granules() {
    let (allocator, governor) = allocator(64 << 20, 8 << 30);

    let probe = allocator.allocate(4096, 256).expect("probe allocation");
    let granularity = allocator.committed_and_reserved().0;
    unsafe { allocator.deallocate(probe, 4096, 256) };
    assert!(granularity >= 4096, "probe reveals a real granule");

    let kv_len = granularity + 4096;
    let first_bucket = 0..4096;
    let kv = allocator
        .allocate_committed(kv_len, 256, std::slice::from_ref(&first_bucket))
        .expect("KV-like span commits only its first bucket");
    let workspace = allocator
        .allocate(4096, 256)
        .expect("workspace lands in KV's uncommitted tail granule");
    assert_eq!(allocator.committed_and_reserved().0, granularity * 2);

    unsafe { allocator.deallocate(kv, kv_len, 256) };
    assert_eq!(
        allocator.committed_and_reserved().0,
        granularity,
        "freeing KV must leave the neighbor-owned tail granule mapped"
    );
    assert_eq!(governor.used(Tier::Device) as usize, granularity);

    unsafe { allocator.deallocate(workspace, 4096, 256) };
    assert_eq!(allocator.committed_and_reserved().0, 0);
    assert_eq!(governor.used(Tier::Device), 0);
}

/// Growth rollback must be able to return late commits without freeing the
/// allocation's original prefix.
///
/// Native CUDA KV commits every binding's next bucket before repacking live
/// data. If a later binding refuses the growth, the earlier successful commits
/// are rolled back with `decommit_allocation_range`; this test proves that the
/// rollback releases only the newly claimed granules and leaves the old bucket
/// mapped.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn decommit_range_rolls_back_late_commits_without_releasing_prefix() {
    let (allocator, governor) = allocator(64 << 20, 8 << 30);

    let probe = allocator.allocate(4096, 256).expect("probe allocation");
    let granularity = allocator.committed_and_reserved().0;
    unsafe { allocator.deallocate(probe, 4096, 256) };

    let len = granularity * 3;
    let prefix = 0..granularity;
    let pointer = allocator
        .allocate_committed(len, 256, std::slice::from_ref(&prefix))
        .expect("prefix commit");
    assert_eq!(allocator.committed_and_reserved().0, granularity);

    allocator
        .commit_allocation_range(pointer, len, 256, 0, granularity * 2)
        .expect("late growth commit");
    assert_eq!(allocator.committed_and_reserved().0, granularity * 2);

    assert_eq!(
        allocator
            .decommit_allocation_range(pointer, len, 256, granularity, granularity)
            .expect("rollback late growth commit"),
        granularity as u64
    );
    assert_eq!(
        allocator.committed_and_reserved().0,
        granularity,
        "rollback should release the late bucket but keep the original prefix"
    );
    assert_eq!(governor.used(Tier::Device) as usize, granularity);

    unsafe { allocator.deallocate(pointer, len, 256) };
    assert_eq!(allocator.committed_and_reserved().0, 0);
    assert_eq!(governor.used(Tier::Device), 0);
}

/// Rollback starts at the old logical bucket size, which is rarely aligned to
/// CUDA's physical granule. The decommit path must still release granules that
/// were first claimed by the failed growth while preserving the old prefix.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn decommit_misaligned_growth_tail_releases_new_granules_only() {
    let (allocator, governor) = allocator(64 << 20, 8 << 30);

    let probe = allocator.allocate(4096, 256).expect("probe allocation");
    let granularity = allocator.committed_and_reserved().0;
    unsafe { allocator.deallocate(probe, 4096, 256) };

    let old_bytes = 4096;
    let len = granularity * 3;
    let prefix = 0..old_bytes;
    let pointer = allocator
        .allocate_committed(len, 256, std::slice::from_ref(&prefix))
        .expect("misaligned prefix commit");
    assert_eq!(allocator.committed_and_reserved().0, granularity);

    allocator
        .commit_allocation_range(pointer, len, 256, 0, granularity + old_bytes)
        .expect("growth claims a second granule");
    assert_eq!(allocator.committed_and_reserved().0, granularity * 2);

    assert_eq!(
        allocator
            .decommit_allocation_range(pointer, len, 256, old_bytes, granularity)
            .expect("rollback starts at a non-granule-aligned old bucket"),
        granularity as u64
    );
    assert_eq!(
        allocator.committed_and_reserved().0,
        granularity,
        "misaligned rollback should release the newly claimed granule only"
    );
    assert_eq!(governor.used(Tier::Device) as usize, granularity);

    unsafe { allocator.deallocate(pointer, len, 256) };
    assert_eq!(allocator.committed_and_reserved().0, 0);
    assert_eq!(governor.used(Tier::Device), 0);
}

/// A large virtual allocation can stay mostly uncommitted while disjoint live
/// stripes are mapped underneath it.
///
/// This is the KV-cache failure mode from #656 in allocator form: a binding may
/// reserve its full context address range, but a short sequence must charge only
/// the token stripes it can actually touch. An assertion on the committed byte
/// count catches the vacuous implementation that routes through eager
/// allocation and still "works" while mapping the whole context.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn committed_ranges_do_not_map_the_whole_virtual_allocation() {
    let (allocator, governor) = allocator(256 << 20, 8 << 30);
    let reserved = 128 << 20;
    let pointer = allocator
        .allocate_committed(reserved, 256, &[])
        .expect("reserve a large KV-like allocation");
    assert_eq!(
        allocator.committed_and_reserved().0,
        0,
        "reserving the binding must not map physical memory"
    );

    allocator
        .commit_allocation_range(pointer, reserved, 256, 0, 4096)
        .expect("commit first token stripe");
    allocator
        .commit_allocation_range(pointer, reserved, 256, 64 << 20, 4096)
        .expect("commit second head's token stripe");
    let committed_short = allocator.committed_and_reserved().0;
    assert!(
        committed_short > 0 && committed_short < reserved / 8,
        "two small stripes committed {committed_short} bytes out of {reserved}; \
         this must stay far below the full virtual allocation"
    );
    assert_eq!(governor.used(Tier::Device) as usize, committed_short);

    allocator
        .commit_allocation_range(pointer, reserved, 256, 0, reserved)
        .expect("commit the full context");
    let committed_full = allocator.committed_and_reserved().0;
    assert!(
        committed_full > committed_short * 8,
        "full-context commit {committed_full} must be much larger than short \
         sequence commit {committed_short}"
    );

    // SAFETY: the pointer came from this allocator and is still live.
    unsafe { allocator.deallocate(pointer, reserved, 256) };
    assert_eq!(allocator.committed_and_reserved().0, 0);
    assert_eq!(governor.used(Tier::Device), 0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn disjoint_tiny_ranges_report_and_commit_granule_rounded_bytes() {
    let (allocator, _governor) = allocator(8 << 20, 8 << 20);
    let selected: &dyn DeviceAllocator = &allocator;
    let backing = selected
        .as_virtual_backing()
        .expect("VMM capability is discovered from the selected allocator");

    let probe = selected.allocate(4096, 256).expect("probe allocation");
    let granularity = allocator.committed_and_reserved().0;
    // SAFETY: exact live allocation returned above.
    unsafe { selected.deallocate(probe, 4096, 256) };

    let allocation_bytes = granularity * 2;
    let ptr = backing
        .allocate_committed(allocation_bytes, 256, &[])
        .expect("reserve without committing");
    let ranges = [
        AllocationCommitRange {
            ptr,
            allocation_bytes,
            align: 256,
            offset: 0,
            bytes: 64,
        },
        AllocationCommitRange {
            ptr,
            allocation_bytes,
            align: 256,
            offset: granularity + 16,
            bytes: 32,
        },
    ];
    let mapped = backing
        .mapped_bytes_for_allocation_ranges(&ranges)
        .expect("granule-aware estimate");
    assert_eq!(mapped, (2 * granularity) as u64);
    assert!(mapped > 96, "raw requested-byte sums would undercharge");

    backing
        .commit_allocation_ranges(&ranges)
        .expect("commit both granules");
    assert_eq!(allocator.committed_and_reserved().0 as u64, mapped);
    // SAFETY: whole-allocation release remains on the selected allocator.
    unsafe { selected.deallocate(ptr, allocation_bytes, 256) };
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
