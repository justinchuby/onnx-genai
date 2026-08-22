//! What is still owned after a CUDA VMM release goes wrong — proven without a
//! GPU.
//!
//! # Why these run on the CPU
//!
//! The rule that decides whether a partially released device address becomes
//! reusable is a *state machine*, not a driver call. Leaving it inside
//! `*_gpu.rs` would mean it is exercised only where CUDA hardware runs, which
//! #636 measured as "nowhere by default": 44 tests were silently skipped for
//! months. So the whole rule lives in
//! [`onnx_runtime_cuda_memory::release`], which has no CUDA symbol in it, and
//! is driven here by [`ScriptedDriver`] — a driver that keeps the mapping and
//! ownership facts a real one would, and fails the Nth `cuMemUnmap`,
//! `cuMemMap` or `cuMemRelease` on demand.
//!
//! # What must hold
//!
//! * A failed unmap mutated nothing: the run stays mapped and no handle is
//!   given back. Refunding it would tell a governor there is memory available
//!   that the device is still holding.
//! * A successful unmap whose handle cannot be given back refunds the **mapped**
//!   axis and not the **owned** one, and the handle never becomes reusable.
//! * A decommit either happens or is undone. A rollback that could not remap
//!   is reported as such and never as success.
//! * A shared granule outlives every sharer but the last.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use onnx_runtime_cuda_memory::release::{
    BlockAccess, DriverFaultPlan, DriverOperation, HandleDisposition, MappedBlock, ReleaseDriver,
    ScriptedDriver, SpanReleaseReport, TransactionalUnmap, block_bytes, contiguous_runs,
    dispose_released_blocks, release_runs, unmap_runs_transactional,
};
use onnx_runtime_memory_governor::{
    AllocationReleaseOutcome, AllocationReleaseState, QuarantineReason,
};

const GRANULE: usize = 2 << 20;
const ALLOCATION: usize = 8 * GRANULE;
const ADDRESS: usize = 0x7f00_0000;

fn granules(count: usize) -> Vec<MappedBlock> {
    (0..count)
        .map(|index| MappedBlock::new(index * GRANULE, GRANULE, 0x1000 + index as u64))
        .collect()
}

/// Adjacent granules form one run, which is how a multi-granule weight page
/// costs one `cuMemUnmap` rather than one per 2 MiB.
fn one_run(blocks: &[MappedBlock]) -> Vec<Vec<MappedBlock>> {
    contiguous_runs(blocks.to_vec())
}

/// Each granule released on its own, as scattered KV pages are.
fn separate_runs(blocks: &[MappedBlock]) -> Vec<Vec<MappedBlock>> {
    blocks.iter().map(|&block| vec![block]).collect()
}

fn plan(faults: &[(DriverOperation, usize)]) -> Arc<DriverFaultPlan> {
    let plan = DriverFaultPlan::new();
    for &(operation, nth) in faults {
        plan.schedule(operation, nth);
    }
    Arc::new(plan)
}

/// A failed unmap leaves the *whole* run mapped and owned, whichever run it is.
///
/// Failing the first, a middle and the last run separately is the point: an
/// implementation that unwound by position rather than by fact would pass one
/// of these and quietly lose the others.
#[test]
fn any_single_run_can_fail_its_unmap_without_disturbing_the_others() {
    for failing in 1..=4usize {
        let blocks = granules(4);
        let driver =
            ScriptedDriver::new(blocks.clone(), plan(&[(DriverOperation::Unmap, failing)]));

        let report = release_runs(&driver, &separate_runs(&blocks));

        let retained = blocks[failing - 1];
        assert_eq!(
            report.still_mapped,
            vec![retained],
            "run {failing} must be the only one retained"
        );
        assert_eq!(driver.mapped(), vec![retained]);
        assert_eq!(
            report.unmapped_bytes(),
            3 * GRANULE as u64,
            "run {failing}: only the three that really unmapped refund"
        );
        assert_eq!(report.retained_owned_bytes(), GRANULE as u64);
        assert!(
            !driver.settled_handles().contains(&retained.handle),
            "run {failing}: a handle whose mapping is still live must not be given back"
        );
        assert_eq!(
            report.outcome(ALLOCATION as u64, ADDRESS, 256).state(),
            AllocationReleaseState::PartiallyUnmapped,
            "run {failing}: a mutation followed by a failure is never reported as unchanged"
        );
    }
}

/// A whole run's unmap failing before any mutation is *not* an ambiguous zero:
/// it is a quarantine that names every retained block.
#[test]
fn a_run_that_never_unmapped_reports_quarantine_rather_than_a_bare_failure() {
    let blocks = granules(3);
    let driver = ScriptedDriver::new(blocks.clone(), plan(&[(DriverOperation::Unmap, 1)]));

    let report = release_runs(&driver, &one_run(&blocks));
    let outcome = report.outcome(ALLOCATION as u64, ADDRESS, 256);

    assert_eq!(
        report.still_mapped, blocks,
        "one unmap covers the whole run"
    );
    assert_eq!(outcome.unmapped_bytes(), 0);
    assert!(
        outcome.is_quarantined(),
        "zero unmapped bytes after a driver refusal is quarantine, not Complete: {outcome:?}"
    );
    let residual = outcome
        .residual()
        .expect("quarantined outcomes carry residual");
    assert_eq!(residual.retained_bytes, 3 * GRANULE as u64);
    assert_eq!(residual.address, ADDRESS);
    assert_eq!(residual.align, 256);
    assert_eq!(residual.reason, QuarantineReason::PartialRelease);
}

/// Unmapping everything and then failing to give one handle back refunds the
/// mapped axis in full and the owned axis not at all.
#[test]
fn a_quarantined_handle_refunds_the_mapped_axis_only() {
    let blocks = granules(3);
    let driver = ScriptedDriver::new(blocks.clone(), plan(&[(DriverOperation::Dispose, 2)]));

    let report = release_runs(&driver, &one_run(&blocks));

    assert!(report.still_mapped.is_empty(), "every mapping is gone");
    assert_eq!(
        report.unmapped_bytes(),
        3 * GRANULE as u64,
        "the mapped axis refunds every unmapped granule, quarantined handle included"
    );
    assert_eq!(
        report.retained_owned_bytes(),
        GRANULE as u64,
        "only the handle the driver would not take stays charged"
    );
    assert_eq!(driver.quarantined_handles(), vec![blocks[1].handle]);
    assert_eq!(
        driver.settled_handles(),
        vec![blocks[0].handle, blocks[2].handle],
        "a quarantined handle is never offered for reuse"
    );
    let outcome = report.outcome(ALLOCATION as u64, ADDRESS, 256);
    assert_eq!(outcome.state(), AllocationReleaseState::Quarantined);
    assert_eq!(outcome.unmapped_bytes(), 3 * GRANULE as u64);
}

/// Releasing an allocation that committed nothing is Complete with zero
/// unmapped bytes — the case that must never be confused with failure.
#[test]
fn a_release_that_unmaps_nothing_is_still_complete() {
    let driver = ScriptedDriver::new(Vec::new(), plan(&[]));

    let report = release_runs(&driver, &[]);
    let outcome = report.outcome(ALLOCATION as u64, ADDRESS, 256);

    assert!(report.is_complete());
    assert_eq!(
        outcome,
        AllocationReleaseOutcome::complete(onnx_runtime_memory_governor::ReleaseAccounting::new(
            ALLOCATION as u64,
            0
        ))
    );
    assert_eq!(outcome.state(), AllocationReleaseState::Released);
}

/// A decommit whose unmap is refused puts back exactly what it took, including
/// the protection each block had.
#[test]
fn a_rolled_back_decommit_restores_every_block_with_its_original_protection() {
    let mut blocks = granules(4);
    blocks[2] = MappedBlock::read_only(2 * GRANULE, GRANULE, 0x2002);
    let driver = ScriptedDriver::new(blocks.clone(), plan(&[(DriverOperation::Unmap, 4)]));

    let outcome = unmap_runs_transactional(&driver, &separate_runs(&blocks));

    assert!(
        matches!(outcome, TransactionalUnmap::RolledBack { .. }),
        "{outcome:?}"
    );
    assert_eq!(driver.mapped(), blocks, "the topology is exactly restored");
    assert_eq!(
        driver.mapped()[2].access,
        BlockAccess::ReadOnly,
        "a shared prefix granule must not be silently upgraded to read/write"
    );
    assert!(
        driver.settled_handles().is_empty() && driver.quarantined_handles().is_empty(),
        "no handle may be disposed while a rollback is still possible"
    );
}

/// When the rollback itself fails, the result is not "rolled back": every
/// residual is named, and no handle is disposed.
#[test]
fn a_failed_rollback_is_reported_as_such_and_never_as_restored() {
    let blocks = granules(4);
    let driver = ScriptedDriver::new(
        blocks.clone(),
        plan(&[(DriverOperation::Unmap, 4), (DriverOperation::Remap, 2)]),
    );

    let outcome = unmap_runs_transactional(&driver, &separate_runs(&blocks));

    let TransactionalUnmap::RollbackFailed {
        still_mapped,
        unmapped_handle_owned,
        faults,
    } = outcome
    else {
        panic!("a remap that failed must not be reported as a rollback");
    };
    // Remap walks back in reverse, so blocks 2 then 1 then 0 are attempted and
    // the second attempt (block 1) is the one that fails.
    assert_eq!(unmapped_handle_owned, vec![blocks[1]]);
    assert_eq!(
        still_mapped,
        vec![blocks[0], blocks[2], blocks[3]],
        "restored blocks and the run that was never attempted"
    );
    assert_eq!(faults.len(), 2);
    assert_eq!(driver.mapped(), vec![blocks[0], blocks[2], blocks[3]]);
    assert!(
        driver.settled_handles().is_empty(),
        "the block that could not be remapped keeps its handle"
    );

    let report = SpanReleaseReport {
        still_mapped,
        unmapped_handle_owned,
        ..SpanReleaseReport::default()
    };
    assert_eq!(
        report.outcome(ALLOCATION as u64, ADDRESS, 256).state(),
        AllocationReleaseState::PartiallyUnmapped,
        "a live allocation with a hole in it is not usable and must not be called released"
    );
    assert_eq!(report.retained_owned_bytes(), 4 * GRANULE as u64);
}

/// The unmap phase never disposes, so a rollback always has handles to work
/// with; disposal is a separate second phase.
#[test]
fn the_unmap_phase_retains_every_handle_for_the_disposal_phase() {
    let blocks = granules(3);
    let driver = ScriptedDriver::new(blocks.clone(), plan(&[]));

    let TransactionalUnmap::Unmapped { blocks: unmapped } =
        unmap_runs_transactional(&driver, &one_run(&blocks))
    else {
        panic!("the unmap phase succeeded");
    };
    assert_eq!(unmapped, blocks);
    assert!(driver.mapped().is_empty());
    assert!(
        driver.settled_handles().is_empty(),
        "phase one must not dispose anything"
    );

    let report = dispose_released_blocks(&driver, &unmapped);
    assert!(report.is_complete());
    assert_eq!(report.unmapped_bytes(), 3 * GRANULE as u64);
    assert_eq!(driver.settled_handles().len(), 3);
}

/// A commit that fails partway and cannot unmap what it already mapped reports
/// the residual instead of claiming the allocation never happened.
#[test]
fn a_commit_rollback_that_cannot_unmap_reports_what_stays_mapped() {
    let blocks = granules(3);
    // Rolling a commit back is a release of the blocks it managed to map.
    let driver = ScriptedDriver::new(blocks.clone(), plan(&[(DriverOperation::Unmap, 2)]));

    let report = release_runs(&driver, &separate_runs(&blocks));

    assert_eq!(report.still_mapped, vec![blocks[1]]);
    assert_eq!(driver.mapped(), vec![blocks[1]]);
    assert!(
        !report.is_complete(),
        "an allocation that failed to be born still owns memory and must say so"
    );
    assert_eq!(report.retained_owned_bytes(), GRANULE as u64);
    assert_eq!(
        report
            .residual(ADDRESS, 4096)
            .map(|residual| residual.state),
        Some(AllocationReleaseState::PartiallyUnmapped)
    );
}

/// A shared physical granule is given back only when its last mapping leaves,
/// and a quarantine elsewhere does not shorten that lifetime.
///
/// This is the property a shared prefix depends on: one physical granule is
/// mapped into the owner's writable window and every sharer's read-only one at
/// the same time, so its lifetime is the **union** of all of them. Releasing it
/// when the first sharer finishes would pull memory out from under requests
/// still reading it.
#[test]
fn a_shared_granule_outlives_every_sharer_but_the_last() {
    let shared = MappedBlock::read_only(0, GRANULE, 0x9000);
    let private = MappedBlock::new(GRANULE, GRANULE, 0x9001);
    // The first handle the driver is actually asked to give back is the
    // private one, and it refuses — so this also proves a quarantine elsewhere
    // leaves the shared granule alone.
    let driver = ScriptedDriver::new(
        vec![shared, private],
        plan(&[(DriverOperation::Dispose, 1)]),
    );
    driver.share(shared.handle, 3);

    for sharer in 0..2 {
        let report = release_runs(&driver, &[vec![shared]]);
        assert!(report.is_complete(), "sharer {sharer}: {report:?}");
        assert!(
            driver.settled_handles().is_empty(),
            "sharer {sharer} released a granule another sharer is still reading"
        );
        assert_eq!(
            driver.shared_references(shared.handle),
            2 - sharer,
            "sharer {sharer} must drop exactly one reference"
        );
        // The next sharer's window maps the same physical granule.
        driver.remap(shared).expect("map the shared granule again");
    }

    let private_report = release_runs(&driver, &[vec![private]]);
    assert_eq!(private_report.unmapped_handle_owned, vec![private]);
    assert_eq!(driver.quarantined_handles(), vec![private.handle]);
    assert_eq!(
        driver.shared_references(shared.handle),
        1,
        "quarantining an unrelated handle must not touch the shared refcount"
    );

    let report = release_runs(&driver, &[vec![shared]]);
    assert!(report.is_complete(), "{report:?}");
    assert_eq!(
        driver.settled_handles(),
        vec![shared.handle],
        "the last mapping to leave is the one that gives the granule back"
    );
    assert_eq!(driver.shared_references(shared.handle), 0);
}

/// Concurrent releases of disjoint spans reach the same total state whichever
/// order they interleave in, and the injected fault lands on exactly one span.
///
/// Deterministic in what it asserts rather than in scheduling: the *totals*
/// are order-independent, so this is a real concurrency check that cannot
/// flake on timing.
#[test]
fn concurrent_releases_of_disjoint_spans_conserve_every_byte() {
    const SPANS: usize = 8;
    const PER_SPAN: usize = 4;

    let blocks = granules(SPANS * PER_SPAN);
    // One unmap and one dispose fail somewhere in the middle of the run.
    let faults = plan(&[
        (DriverOperation::Unmap, SPANS / 2),
        (DriverOperation::Dispose, SPANS),
    ]);
    let driver = Arc::new(ScriptedDriver::new(blocks.clone(), Arc::clone(&faults)));
    let unmapped_total = Arc::new(AtomicUsize::new(0));
    let owned_total = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        for span in 0..SPANS {
            let driver = Arc::clone(&driver);
            let unmapped_total = Arc::clone(&unmapped_total);
            let owned_total = Arc::clone(&owned_total);
            let span_blocks = blocks[span * PER_SPAN..(span + 1) * PER_SPAN].to_vec();
            scope.spawn(move || {
                let report = release_runs(driver.as_ref(), &contiguous_runs(span_blocks));
                unmapped_total.fetch_add(report.unmapped_bytes() as usize, Ordering::Relaxed);
                owned_total.fetch_add(report.retained_owned_bytes() as usize, Ordering::Relaxed);
                // Whatever happened, the three buckets partition the span.
                assert_eq!(
                    report.settled.len()
                        + report.still_mapped.len()
                        + report.unmapped_handle_owned.len(),
                    PER_SPAN,
                    "every block must be accounted for exactly once"
                );
            });
        }
    });

    let unmapped = unmapped_total.load(Ordering::Relaxed);
    let owned = owned_total.load(Ordering::Relaxed);
    let still_mapped = driver.mapped();
    let settled = driver.settled_handles();
    let quarantined = driver.quarantined_handles();

    assert_eq!(
        still_mapped.len(),
        PER_SPAN,
        "exactly one span's run failed its unmap"
    );
    assert_eq!(quarantined.len(), 1, "exactly one handle was refused");
    assert_eq!(
        settled.len() + quarantined.len() + still_mapped.len(),
        SPANS * PER_SPAN,
        "no block was lost or double-counted across threads"
    );
    assert_eq!(
        unmapped,
        (SPANS * PER_SPAN - PER_SPAN) * GRANULE,
        "the mapped axis refunds every block whose mapping is gone, once"
    );
    assert_eq!(
        owned,
        (PER_SPAN + 1) * GRANULE,
        "the owned axis retains the still-mapped run plus the quarantined handle"
    );
    for handle in &quarantined {
        assert!(
            !settled.contains(handle),
            "handle {handle:#x} was both quarantined and made reusable"
        );
    }
    for block in &still_mapped {
        assert!(
            !settled.contains(&block.handle),
            "a still-mapped granule's handle was given back"
        );
    }
    assert_eq!(
        block_bytes(&still_mapped),
        (PER_SPAN * GRANULE) as u64,
        "byte totals agree with block counts"
    );
}

/// A fault plan belongs to the object it is attached to, so two releases in
/// the same process cannot see each other's injected faults.
#[test]
fn fault_plans_do_not_leak_between_instances() {
    let blocks = granules(2);
    let faulted = ScriptedDriver::new(blocks.clone(), plan(&[(DriverOperation::Unmap, 1)]));
    let clean = ScriptedDriver::new(blocks.clone(), plan(&[]));

    assert!(!release_runs(&faulted, &one_run(&blocks)).is_complete());
    assert!(
        release_runs(&clean, &one_run(&blocks)).is_complete(),
        "an unrelated driver must be unaffected by another's fault plan"
    );
}

/// Disposal reports settlement and quarantine as distinct answers, and the
/// quarantined one carries the driver's reason.
#[test]
fn disposal_distinguishes_settled_from_quarantined() {
    let block = granules(1)[0];
    let driver = ScriptedDriver::new(vec![block], plan(&[(DriverOperation::Dispose, 1)]));

    let disposition = driver.dispose(block);

    let HandleDisposition::Quarantined(fault) = disposition else {
        panic!("a refused release is not a settlement");
    };
    assert_eq!(fault.operation, DriverOperation::Dispose.name());
    assert!(!fault.reason.is_empty(), "the reason must name something");
    assert_eq!(driver.quarantined_handles(), vec![block.handle]);
}
