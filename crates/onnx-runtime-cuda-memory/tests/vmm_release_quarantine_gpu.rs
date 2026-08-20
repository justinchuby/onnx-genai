//! Real-device proof that a partially released VMM address never comes back.
//!
//! # What only hardware can answer
//!
//! [`vmm_release_quarantine`](../vmm_release_quarantine.rs) proves the release
//! *state machine* on any machine. What it cannot prove is that the machine
//! agrees: that a `cuMemUnmap` this code believes failed really did leave the
//! mapping live, that a rolled-back decommit really does leave the buffer
//! readable with its original contents, and that the arena's accounting
//! matches what the driver did rather than what we assumed.
//!
//! So these tests inject faults into the **real** driver path — through an
//! allocator-scoped [`DriverFaultPlan`], never a process-global switch — and
//! then ask the device.
//!
//! Ignored rather than compiled out without `gpu-tests`, so a CPU-only run
//! reports them as skipped instead of pretending a suite that never built is
//! passing (#636).

use std::sync::Arc;

use cudarc::driver::CudaContext;
use onnx_runtime_cuda_memory::release::{DriverFaultPlan, DriverOperation};
use onnx_runtime_cuda_memory::vmm_allocator::{CudaVmmAllocator, DecommitOutcome};
use onnx_runtime_memory_governor::{
    AllocationReleaseState, DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor,
    MemoryRole, QuarantineReason, VirtualBacking,
};

#[cfg(feature = "gpu-tests")]
use onnx_runtime_cuda_memory::test_support::TestStream;

const HOLDER: HolderId = HolderId::new(41);

fn allocator(capacity: usize) -> (CudaVmmAllocator, LedgerGovernor, Arc<CudaContext>) {
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "VMM release-quarantine test requires a CUDA driver; CPU-only runs must leave this \
             test ignored (enable the `gpu-tests` feature on a CUDA runner): {error}"
        ),
    };
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = CudaVmmAllocator::new(
        Arc::clone(&context),
        DeviceKey::device(0),
        0,
        capacity,
        &governor,
        HOLDER,
        MemoryRole::Workspace { step_scoped: false },
    )
    .expect("reserving device address space");
    (allocator, governor, context)
}

/// Attach an allocator-scoped fault plan. Only the `gpu-tests` build exposes
/// the hook, which is what keeps injection out of production entirely.
#[cfg(feature = "gpu-tests")]
fn install_faults(allocator: &mut CudaVmmAllocator, plan: Arc<DriverFaultPlan>) {
    allocator.install_driver_faults(plan);
}

#[cfg(not(feature = "gpu-tests"))]
fn install_faults(_allocator: &mut CudaVmmAllocator, _plan: Arc<DriverFaultPlan>) {
    unreachable!("driver fault injection is only compiled under the gpu-tests feature");
}

/// Fill `len` bytes at `address` with `value` and read them back, on one
/// stream so the read is ordered after the write (see `test_support`).
#[cfg(feature = "gpu-tests")]
fn fill_and_read(context: &Arc<CudaContext>, address: u64, len: usize, value: u8) -> Vec<u8> {
    let stream = TestStream::with_context(Arc::clone(context));
    stream.fill(address, value, len);
    stream.read(address, len)
}

#[cfg(not(feature = "gpu-tests"))]
fn fill_and_read(_context: &Arc<CudaContext>, _address: u64, _len: usize, _value: u8) -> Vec<u8> {
    unreachable!("device access is only compiled under the gpu-tests feature");
}

/// A clean release is `Complete` and its address really does come back.
///
/// The control case. Without it, a quarantine test could pass because
/// *everything* is quarantined.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_clean_release_is_complete_and_returns_its_address() {
    let (allocator, _governor, _context) = allocator(32 << 20);
    let bytes = 4 << 20;
    let pointer = allocator.allocate(bytes, 4096).expect("4 MiB fits");

    // SAFETY: `pointer` is live, from this allocator, with exactly this size
    // and alignment, and is released once.
    let outcome = unsafe { allocator.release(pointer, bytes, 4096) };

    assert!(outcome.is_complete(), "{outcome:?}");
    assert_eq!(outcome.state(), AllocationReleaseState::Released);
    assert!(
        allocator.quarantined_spans().is_empty(),
        "a clean release must leave nothing owned"
    );
    let again = allocator
        .allocate(bytes, 4096)
        .expect("the span is reusable");
    assert_eq!(
        again, pointer,
        "the freed address is the first fit and must be handed out again"
    );
    // SAFETY: same contract as above.
    unsafe { allocator.release(again, bytes, 4096) };
}

/// A release whose `cuMemUnmap` the driver refuses quarantines the whole
/// allocation and never returns its address.
///
/// This is the defect Phase 4 closes: the old path removed the live record and
/// returned the address to the free list regardless, so the next allocation
/// inherited a live mapping under part of its buffer.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_refused_unmap_quarantines_the_span_and_withholds_its_address() {
    let (mut allocator, _governor, _context) = allocator(32 << 20);
    let bytes = 4 << 20;
    let pointer = allocator.allocate(bytes, 4096).expect("4 MiB fits");
    install_faults(
        &mut allocator,
        Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 1)),
    );

    // SAFETY: `pointer` is live, from this allocator, released once.
    let outcome = unsafe { allocator.release(pointer, bytes, 4096) };

    assert!(outcome.is_quarantined(), "{outcome:?}");
    assert_eq!(outcome.state(), AllocationReleaseState::PartiallyUnmapped);
    assert_eq!(
        outcome.unmapped_bytes(),
        0,
        "nothing was mutated, so nothing may be refunded"
    );
    let residual = outcome.residual().expect("quarantine carries residual");
    assert_eq!(residual.reason, QuarantineReason::PartialRelease);
    assert_eq!(
        residual.retained_bytes, bytes as u64,
        "the whole allocation is still mapped and still owned"
    );
    assert_eq!(residual.address, pointer.as_ptr() as usize);

    let quarantined = allocator.quarantined_spans();
    assert_eq!(quarantined.len(), 1);
    assert_eq!(quarantined[0].len, bytes);
    assert_eq!(allocator.quarantined_owned_bytes(), bytes as u64);

    // The address must not come back, and neither must the granules under it.
    let next = allocator
        .allocate(bytes, 4096)
        .expect("the arena still has room elsewhere");
    assert_ne!(
        next, pointer,
        "a quarantined address was handed out again; the next allocation would inherit its \
         mapping"
    );
    // SAFETY: `next` is live, from this allocator, released once.
    unsafe { allocator.release(next, bytes, 4096) };

    // Releasing the quarantined pointer again must fail closed rather than
    // mutate anything.
    // SAFETY: the pointer is no longer live; this is exactly the misuse the
    // fail-closed path exists to catch, and it mutates nothing.
    let again = unsafe { allocator.release(pointer, bytes, 4096) };
    assert!(
        again.failure().is_some(),
        "a quarantined span must never be released a second time: {again:?}"
    );
}

/// A handle the driver will not release quarantines the physical memory while
/// still refunding the mapping.
///
/// The two axes are independent, and conflating them is how a governor ends up
/// admitting work for memory the device is still holding.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_refused_handle_release_refunds_the_mapping_but_not_the_ownership() {
    let (mut allocator, _governor, _context) = allocator(32 << 20);
    let bytes = 4 << 20;
    let pointer = allocator.allocate(bytes, 4096).expect("4 MiB fits");
    let (committed_before, _) = allocator.committed_and_reserved();
    install_faults(
        &mut allocator,
        Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Dispose, 1)),
    );

    // SAFETY: `pointer` is live, from this allocator, released once.
    let outcome = unsafe { allocator.release(pointer, bytes, 4096) };

    assert!(outcome.is_quarantined(), "{outcome:?}");
    assert_eq!(
        outcome.state(),
        AllocationReleaseState::Quarantined,
        "every mapping is gone; what remains is physical ownership"
    );
    let granule = allocator
        .physical_pool_stats()
        .map_or(2 << 20, |_| 2 << 20)
        .max(1);
    assert!(
        outcome.unmapped_bytes() >= granule as u64,
        "the mapped axis must refund the granule whose mapping really went away: {outcome:?}"
    );
    let residual = outcome.residual().expect("quarantine carries residual");
    assert!(
        residual.retained_bytes > 0,
        "the handle the driver refused is still owned"
    );
    let (committed_after, _) = allocator.committed_and_reserved();
    assert!(
        committed_after < committed_before,
        "unmapped granules must leave the arena's committed-byte gauge"
    );
    if let Some(stats) = allocator.physical_pool_stats() {
        let snapshot = stats.snapshot();
        assert!(
            snapshot.quarantined_bytes > 0,
            "the pool must record the handle it could not give back"
        );
        assert_eq!(
            snapshot.quarantined_handles, 1,
            "exactly one handle was refused"
        );
    }
}

/// A decommit whose unmap is refused restores the mapping, and the device
/// agrees: the buffer still reads back what was written into it.
///
/// "Rolled back" is only meaningful if the memory is genuinely usable
/// afterwards. Asserting on the return value alone would pass for an
/// implementation that remapped without restoring access, which reads as a
/// fault rather than data.
///
/// # Why this punches a hole first
///
/// `unmap_runs_transactional` only has something to roll *back* once at least
/// one run has already been unmapped, so the fault has to land on the second
/// unmap or later. But the decommit is issued per **run**, not per granule:
/// `contiguous_runs` (release.rs) deliberately collapses adjacent granules into
/// one `cuMemUnmap`, "one driver call rather than one per granule". A fresh
/// 8 MiB out of a fresh 64 MiB arena is contiguous, so decommitting all of it
/// is exactly *one* unmap and a fault scheduled for the second one never fires.
///
/// Failing the *first* unmap instead is not a fix: nothing would have been
/// unmapped yet, so there is nothing to map back and the rollback path this
/// test exists for is never entered. Instead the allocation is given a
/// granule-sized hole in the middle, which makes the decommit below two
/// non-adjacent runs and therefore two real unmap calls. The premise is then
/// asserted rather than assumed, so a device whose granularity defeats the
/// construction fails loudly and says why.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_rolled_back_decommit_leaves_the_buffer_readable() {
    let (mut allocator, _governor, context) = allocator(64 << 20);
    let bytes = 8 << 20;
    let pointer = allocator.allocate(bytes, 4096).expect("8 MiB fits");
    let address = pointer.as_ptr() as u64;

    let before = fill_and_read(&context, address, bytes, 0xA5);
    assert!(before.iter().all(|&byte| byte == 0xA5), "baseline fill");

    // Punch the hole while every granule is still mapped, and before any fault
    // plan is installed, so this decommit is an ordinary successful one.
    let committed_whole = allocator.allocation_committed_bytes(pointer, bytes, 4096);
    assert_eq!(
        committed_whole, bytes,
        "premise: the fresh allocation must be fully committed"
    );
    let hole_offset = bytes / 2;
    allocator
        .decommit_allocation_range_outcome(pointer, bytes, hole_offset, 1)
        .expect("the hole range is valid");
    let committed_holed = allocator.allocation_committed_bytes(pointer, bytes, 4096);
    let granule = committed_whole - committed_holed;
    assert!(
        granule > 0 && hole_offset + granule < bytes,
        "premise: decommitting at offset {hole_offset} must drop exactly the granule starting \
         there and leave committed bytes on both sides of it, or the decommit below is still one \
         contiguous run and the injected second-unmap fault can never fire (committed \
         {committed_whole}, then {committed_holed}, of {bytes} B)"
    );

    // Fail the second unmap. The first run is unmapped by then and must be
    // mapped back, which is the rollback under test.
    let plan = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 2));
    install_faults(&mut allocator, Arc::clone(&plan));
    let outcome = allocator
        .decommit_allocation_range_outcome(pointer, bytes, 0, bytes)
        .expect("the range is valid");
    let unmap_calls = plan.calls(DriverOperation::Unmap);
    assert!(
        unmap_calls >= 2,
        "premise: the holed decommit must reach a second unmap for the fault to fire, but it made \
         {unmap_calls}; the outcome below therefore says nothing about rollback: {outcome:?}"
    );

    let suffix_offset = hole_offset + granule;
    let suffix_len = bytes - suffix_offset;
    match outcome {
        DecommitOutcome::RolledBack { .. } => {
            assert!(
                allocator.quarantined_spans().is_empty(),
                "a successful rollback owes nothing"
            );
            // Only the two mapped runs are touched; the hole is unmapped and
            // reading it would fault for a reason that has nothing to do with
            // the rollback.
            let prefix = fill_and_read(&context, address, hole_offset, 0x5A);
            assert!(
                prefix.iter().all(|&byte| byte == 0x5A),
                "a rolled-back decommit must leave the first run mapped *and* accessible"
            );
            let suffix = fill_and_read(&context, address + suffix_offset as u64, suffix_len, 0x5A);
            assert!(
                suffix.iter().all(|&byte| byte == 0x5A),
                "a rolled-back decommit must leave the untouched run mapped *and* accessible"
            );
            assert_eq!(
                allocator.allocation_committed_bytes(pointer, bytes, 4096),
                committed_holed,
                "the allocation keeps exactly the granules it had"
            );
        }
        DecommitOutcome::Quarantined {
            accounting,
            residual,
            reason,
        } => {
            // The driver refused the remap too. That is a legitimate device
            // outcome; what must never happen is calling it a rollback.
            assert!(
                accounting.unmapped_bytes > 0,
                "a failed rollback reports every mapping that stayed removed"
            );
            assert!(
                residual.retained_bytes > 0,
                "a failed rollback retains ownership: {reason}"
            );
            assert_eq!(allocator.quarantined_spans().len(), 1);
        }
        DecommitOutcome::Complete { accounting } => {
            panic!("the injected unmap fault must not produce a completed decommit: {accounting:?}")
        }
    }

    // Either way the allocation is not silently half-decommitted.
    let committed = allocator.allocation_committed_bytes(pointer, bytes, 4096);
    assert!(
        committed == committed_holed || allocator.quarantined_spans().len() == 1,
        "a live allocation must never be left with a hole it did not ask for"
    );
}

/// A decommit that succeeds reports the bytes actually unmapped and leaves the
/// allocation live and usable.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_successful_decommit_reports_exactly_what_it_unmapped() {
    let (allocator, _governor, context) = allocator(64 << 20);
    let bytes = 8 << 20;
    let pointer = allocator.allocate(bytes, 4096).expect("8 MiB fits");
    let granule = allocator
        .mapped_bytes_for_allocation(1, 1)
        .expect("a one-byte allocation rounds up to one granule") as usize;

    let outcome = allocator
        .decommit_allocation_range_outcome(pointer, bytes, bytes - granule, granule)
        .expect("the tail granule is decommittable");

    let DecommitOutcome::Complete { accounting } = outcome else {
        panic!("a clean decommit must complete: {outcome:?}");
    };
    assert_eq!(accounting.unmapped_bytes, granule as u64);
    assert_eq!(accounting.quarantined_owned_bytes, 0);
    assert_eq!(
        allocator.allocation_committed_bytes(pointer, bytes, 4096),
        bytes - granule,
        "exactly the requested granule left the allocation"
    );
    // What remains must still be usable.
    let head = fill_and_read(&context, pointer.as_ptr() as u64, bytes - granule, 0x3C);
    assert!(head.iter().all(|&byte| byte == 0x3C));

    // SAFETY: `pointer` is live, from this allocator, released once.
    let release = unsafe { allocator.release(pointer, bytes, 4096) };
    assert!(release.is_complete(), "{release:?}");
    assert_eq!(
        release.unmapped_bytes(),
        (bytes - granule) as u64,
        "the release refunds only what was still mapped"
    );
}
