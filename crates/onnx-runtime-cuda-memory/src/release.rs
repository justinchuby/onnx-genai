//! What is still true after a CUDA VMM release only partly succeeded.
//!
//! # Why this module exists
//!
//! Releasing one VMM allocation is not one driver call. It is a *sequence*:
//! `cuMemUnmap` a run of adjacent granules, then give each physical handle
//! back — to the retained pool, or to `cuMemRelease`. Every step can fail
//! independently, and the failures mean different things:
//!
//! * `cuMemUnmap` fails → **nothing was mutated for that run**. The granules
//!   are still mapped, still readable by whatever was using them, and still
//!   owned. Forgetting them would hand a stale mapping to the next allocation
//!   that carves the same address.
//! * `cuMemUnmap` succeeds and the handle cannot be given back → the *mapping*
//!   is gone (so the mapped-byte axis must be refunded exactly once) but the
//!   *physical* memory is still owned (so the owned axis must not be). The
//!   handle must never re-enter a reuse pool: a `cuMemRelease` that failed
//!   leaves ownership genuinely uncertain.
//!
//! Collapsing those two into "the release errored" is what made a partially
//! released address reusable. This module keeps them apart, as data, with no
//! CUDA symbol anywhere in it — which is also what makes the state machine
//! testable on a machine with no GPU (#636 found that "GPU-only" state
//! machines are effectively untested).
//!
//! # The shape of the contract
//!
//! [`ReleaseDriver`] is the three mutations a release performs — `unmap`,
//! `remap`, `dispose` — and nothing else. [`release_runs`] and
//! [`unmap_runs_transactional`] drive them and return *exact* residual facts.
//! `virtual_memory.rs` implements the trait with the driver; tests implement it
//! with [`ScriptedDriver`], which fails the Nth call of a chosen operation
//! deterministically and is instance-scoped rather than process-global.
//!
//! # Lock order
//!
//! Every function here runs with the caller's arena lock held and **no** pool
//! lock held; see the lock-order table in [`crate::virtual_memory`]. Nothing
//! here waits on a device.

use std::fmt::Debug;

use onnx_runtime_memory_governor::{
    AllocationReleaseOutcome, AllocationReleaseState, QuarantineReason, ReleaseAccounting,
    ResidualOwnership,
};

/// A CUDA physical allocation handle, as an opaque integer.
///
/// Declared here rather than imported so this module has no CUDA dependency
/// and can be exercised without a driver.
pub type PhysicalHandle = u64;

/// How a block was mapped, so a rollback can restore exactly that.
///
/// A shared prefix granule is mapped read-only on purpose: a mis-targeted
/// store into memory another request is reading must fault rather than corrupt
/// it. Remapping such a block read/write during a rollback would quietly
/// remove that protection, so the protection travels with the block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlockAccess {
    #[default]
    ReadWrite,
    ReadOnly,
}

/// One mapped granule run element: where it lives and what backs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MappedBlock {
    /// Byte offset inside the reservation.
    pub offset: usize,
    /// Length in bytes. One allocation granule for every pooled mapping.
    pub len: usize,
    /// The physical handle mapped at `offset`.
    pub handle: PhysicalHandle,
    /// The protection the mapping was granted.
    pub access: BlockAccess,
}

impl MappedBlock {
    pub const fn new(offset: usize, len: usize, handle: PhysicalHandle) -> Self {
        Self {
            offset,
            len,
            handle,
            access: BlockAccess::ReadWrite,
        }
    }

    /// A block mapped read-only, as every shared prefix granule is.
    pub const fn read_only(offset: usize, len: usize, handle: PhysicalHandle) -> Self {
        Self {
            offset,
            len,
            handle,
            access: BlockAccess::ReadOnly,
        }
    }
}

/// Total bytes across `blocks`.
pub fn block_bytes(blocks: &[MappedBlock]) -> u64 {
    blocks.iter().map(|block| block.len as u64).sum()
}

/// One failed driver mutation, kept as text because it is only ever reported.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DriverFault {
    /// The driver entry point that failed, e.g. `cuMemUnmap`.
    pub operation: &'static str,
    /// What the driver said.
    pub reason: String,
}

impl DriverFault {
    pub fn new(operation: &'static str, reason: impl Into<String>) -> Self {
        Self {
            operation,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for DriverFault {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} failed: {}", self.operation, self.reason)
    }
}

/// What happened to one physical handle after its mapping was removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandleDisposition {
    /// Terminal. The handle was retained for reuse, released to the driver, or
    /// is still held by another live mapping of the same shared granule. In
    /// every case this allocation owns nothing further.
    Settled,
    /// The handle could not be made terminal and is now held in quarantine.
    ///
    /// Its physical bytes stay charged on the owned axis and it is never handed
    /// out again. Only the *mapped* axis was refunded.
    Quarantined(DriverFault),
}

impl HandleDisposition {
    pub const fn is_settled(&self) -> bool {
        matches!(self, Self::Settled)
    }
}

/// The mutations a release performs, and nothing else.
///
/// # Contract
///
/// * `unmap` must be atomic for the run it is given: on `Err` the whole run is
///   still mapped, on `Ok` none of it is. CUDA's `cuMemUnmap` over a run of
///   adjacent mappings satisfies this.
/// * `remap` restores one block that `unmap` removed, using the handle the
///   caller retained. It must also restore access, or the restored topology
///   would be present but unreadable.
/// * `dispose` consumes ownership of one handle. It must never make a handle
///   whose release failed reusable; it reports
///   [`HandleDisposition::Quarantined`] instead.
///
/// The mapped-byte axis is refunded by `dispose` (the mapping is already
/// gone), so implementations must refund it for both dispositions and exactly
/// once.
pub trait ReleaseDriver {
    /// Remove the mapping over `blocks`, which are adjacent and ascending.
    fn unmap(&self, blocks: &[MappedBlock]) -> Result<(), DriverFault>;

    /// Map `block`'s retained handle back where it was, with access restored.
    fn remap(&self, block: MappedBlock) -> Result<(), DriverFault>;

    /// Give up ownership of an unmapped block's handle.
    fn dispose(&self, block: MappedBlock) -> HandleDisposition;
}

/// Exactly what a release left behind, split by axis.
///
/// The three vectors are disjoint and together cover every block handed in, so
/// a caller can reconcile without inference.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpanReleaseReport {
    /// Unmapped, and the handle reached a terminal state. Nothing is owed.
    pub settled: Vec<MappedBlock>,
    /// Still mapped and still owned: the unmap failed before mutating
    /// anything. These must stay recorded as mapped.
    pub still_mapped: Vec<MappedBlock>,
    /// Unmapped, but the handle could not be given back. The mapped axis is
    /// refunded; the owned axis is not.
    pub unmapped_handle_owned: Vec<MappedBlock>,
    /// Every driver fault, in the order they happened.
    pub faults: Vec<DriverFault>,
}

impl SpanReleaseReport {
    /// Whether every block reached a terminal, nothing-owed state.
    pub fn is_complete(&self) -> bool {
        self.still_mapped.is_empty() && self.unmapped_handle_owned.is_empty()
    }

    /// Bytes whose mapping is gone: the mapped-axis refund, counted once.
    pub fn unmapped_bytes(&self) -> u64 {
        block_bytes(&self.settled) + block_bytes(&self.unmapped_handle_owned)
    }

    /// Physical bytes still owned after this release.
    pub fn retained_owned_bytes(&self) -> u64 {
        block_bytes(&self.still_mapped) + block_bytes(&self.unmapped_handle_owned)
    }

    /// Fold another report into this one, keeping fault order.
    pub fn merge(&mut self, other: Self) {
        self.settled.extend(other.settled);
        self.still_mapped.extend(other.still_mapped);
        self.unmapped_handle_owned
            .extend(other.unmapped_handle_owned);
        self.faults.extend(other.faults);
    }

    /// The lifecycle state this residual leaves the allocation in.
    ///
    /// `None` when nothing is owed. Otherwise:
    ///
    /// * [`AllocationReleaseState::PartiallyUnmapped`] while any mapping
    ///   survives, because part of the allocation is still readable;
    /// * [`AllocationReleaseState::Quarantined`] when every mapping is gone
    ///   and what remains is physical ownership.
    ///
    /// Never `Live`: a report only exists after at least one unmap was
    /// attempted, and "nothing was mutated" is expressed by
    /// [`AllocationReleaseOutcome::Failed`] before a report is ever built.
    pub fn residual_state(&self) -> Option<AllocationReleaseState> {
        if self.is_complete() {
            return None;
        }
        Some(if self.still_mapped.is_empty() {
            AllocationReleaseState::Quarantined
        } else {
            AllocationReleaseState::PartiallyUnmapped
        })
    }

    /// What the runtime still owns, for a residual at `address`.
    pub fn residual(&self, address: usize, align: usize) -> Option<ResidualOwnership> {
        Some(ResidualOwnership {
            state: self.residual_state()?,
            reason: QuarantineReason::PartialRelease,
            retained_bytes: self.retained_owned_bytes(),
            address,
            align,
        })
    }

    /// The structured whole-allocation outcome this report implies.
    ///
    /// Complete when nothing is owed — including when zero bytes were
    /// unmapped, which is what an allocation with no committed granules
    /// legitimately reports. Quarantined otherwise: a report exists only after
    /// the driver was asked to mutate something, so "failed and unchanged" is
    /// never one of the answers here.
    pub fn outcome(
        &self,
        allocation_bytes: u64,
        address: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        let accounting = ReleaseAccounting::new(allocation_bytes, self.unmapped_bytes());
        match self.residual(address, align) {
            None => AllocationReleaseOutcome::complete(accounting),
            Some(residual) => AllocationReleaseOutcome::quarantined(accounting, residual),
        }
    }

    /// One line naming every fault, for an error message a human can act on.
    pub fn fault_summary(&self) -> String {
        if self.faults.is_empty() {
            return String::from("no driver fault was reported");
        }
        self.faults
            .iter()
            .map(DriverFault::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Release every run, keeping exact per-block state.
///
/// `runs` are contiguous ascending groups; each is unmapped with one call so a
/// multi-granule weight page costs one driver round-trip rather than one per
/// granule. A run whose unmap fails is reported whole as still mapped and its
/// handles are **not** disposed — the memory is still live behind them.
pub fn release_runs<D: ReleaseDriver + ?Sized>(
    driver: &D,
    runs: &[Vec<MappedBlock>],
) -> SpanReleaseReport {
    let mut report = SpanReleaseReport::default();
    for run in runs {
        if run.is_empty() {
            continue;
        }
        if let Err(fault) = driver.unmap(run) {
            report.faults.push(fault);
            report.still_mapped.extend(run.iter().copied());
            continue;
        }
        dispose_blocks(driver, run, &mut report);
    }
    report
}

/// Phase two of a transactional release: hand back the handles of blocks that
/// are already unmapped.
///
/// Separated from the unmap phase because rollback is only possible while the
/// handles are still ours; once a handle is disposed there is nothing left to
/// remap with.
pub fn dispose_released_blocks<D: ReleaseDriver + ?Sized>(
    driver: &D,
    blocks: &[MappedBlock],
) -> SpanReleaseReport {
    let mut report = SpanReleaseReport::default();
    dispose_blocks(driver, blocks, &mut report);
    report
}

fn dispose_blocks<D: ReleaseDriver + ?Sized>(
    driver: &D,
    blocks: &[MappedBlock],
    report: &mut SpanReleaseReport,
) {
    for &block in blocks {
        match driver.dispose(block) {
            HandleDisposition::Settled => report.settled.push(block),
            HandleDisposition::Quarantined(fault) => {
                report.faults.push(fault);
                report.unmapped_handle_owned.push(block);
            }
        }
    }
}

/// The outcome of the unmap phase of a *transactional* decommit.
///
/// A decommit of a still-live allocation may not leave a hole: either the
/// requested mapping is gone, or the topology the caller had is back. This
/// enum is the only place that distinction is decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransactionalUnmap {
    /// Every run was unmapped. The handles are retained and must now be
    /// disposed with [`dispose_released_blocks`].
    Unmapped { blocks: Vec<MappedBlock> },
    /// An unmap failed and every block already unmapped was mapped back with
    /// its retained handle. The allocation is exactly as it was.
    RolledBack { fault: DriverFault },
    /// An unmap failed *and* the rollback failed. Nothing may be reused.
    RollbackFailed {
        /// Blocks that were mapped back successfully, plus the runs never
        /// attempted: all still mapped and owned.
        still_mapped: Vec<MappedBlock>,
        /// Blocks that are unmapped and whose handles are still held, because
        /// remapping them failed.
        unmapped_handle_owned: Vec<MappedBlock>,
        faults: Vec<DriverFault>,
    },
}

/// Unmap every run, or put back what was unmapped.
///
/// Handles are retained throughout: nothing is disposed here, because a
/// disposed handle cannot be remapped and the rollback would be a lie.
pub fn unmap_runs_transactional<D: ReleaseDriver + ?Sized>(
    driver: &D,
    runs: &[Vec<MappedBlock>],
) -> TransactionalUnmap {
    let mut unmapped: Vec<MappedBlock> = Vec::new();
    for (index, run) in runs.iter().enumerate() {
        if run.is_empty() {
            continue;
        }
        let Err(fault) = driver.unmap(run) else {
            unmapped.extend(run.iter().copied());
            continue;
        };
        // Everything from this run on is untouched and still mapped.
        let mut still_mapped = runs[index..]
            .iter()
            .flat_map(|run| run.iter().copied())
            .collect::<Vec<_>>();
        let mut unmapped_handle_owned = Vec::new();
        let mut faults = vec![fault.clone()];
        for block in unmapped.iter().copied().rev() {
            match driver.remap(block) {
                Ok(()) => still_mapped.push(block),
                Err(remap_fault) => {
                    faults.push(remap_fault);
                    unmapped_handle_owned.push(block);
                }
            }
        }
        if unmapped_handle_owned.is_empty() {
            return TransactionalUnmap::RolledBack { fault };
        }
        still_mapped.sort_unstable_by_key(|block| block.offset);
        unmapped_handle_owned.sort_unstable_by_key(|block| block.offset);
        return TransactionalUnmap::RollbackFailed {
            still_mapped,
            unmapped_handle_owned,
            faults,
        };
    }
    TransactionalUnmap::Unmapped { blocks: unmapped }
}

/// Group ascending blocks into runs of adjacent mappings.
///
/// Adjacency is what lets one `cuMemUnmap` cover several physical handles, so
/// a contiguous weight page is one driver call rather than one per granule.
pub fn contiguous_runs(mut blocks: Vec<MappedBlock>) -> Vec<Vec<MappedBlock>> {
    blocks.sort_unstable_by_key(|block| block.offset);
    let mut runs: Vec<Vec<MappedBlock>> = Vec::new();
    for block in blocks {
        match runs.last_mut() {
            Some(run)
                if run
                    .last()
                    .is_some_and(|last| last.offset + last.len == block.offset) =>
            {
                run.push(block);
            }
            _ => runs.push(vec![block]),
        }
    }
    runs
}

/// Which driver operation a scripted fault targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DriverOperation {
    BindContext,
    Unmap,
    Remap,
    Dispose,
    /// `cuMemSetAccess`. Only meaningful to the granule-transition primitive's
    /// Phase 8, which is the only caller that grants access as a step
    /// distinct from mapping.
    SetAccess,
}

impl DriverOperation {
    pub const fn name(self) -> &'static str {
        match self {
            Self::BindContext => "cuCtxSetCurrent",
            Self::Unmap => "cuMemUnmap",
            Self::Remap => "cuMemMap",
            Self::Dispose => "cuMemRelease",
            Self::SetAccess => "cuMemSetAccess",
        }
    }
}

/// A deterministic "fail the Nth call" plan, scoped to one instance.
///
/// Instance-scoped on purpose: a process-global switch would make two tests in
/// the same binary observe each other's faults, which is exactly the kind of
/// order-dependent flake #797 spent a week finding. Attach one of these to the
/// object under test and nothing else can see it.
///
/// Counting is 1-based: `fail_nth(DriverOperation::Unmap, 2)` fails the second
/// unmap and no other.
#[derive(Debug, Default)]
pub struct DriverFaultPlan {
    inner: std::sync::Mutex<FaultPlanState>,
}

#[derive(Debug, Default)]
struct FaultPlanState {
    scheduled: Vec<(DriverOperation, usize)>,
    counts: Vec<(DriverOperation, usize)>,
}

impl DriverFaultPlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fail the `nth` (1-based) call of `operation`. Call repeatedly to fail
    /// several.
    #[must_use]
    pub fn fail_nth(self, operation: DriverOperation, nth: usize) -> Self {
        self.schedule(operation, nth);
        self
    }

    /// Add a fault to an already-shared plan.
    pub fn schedule(&self, operation: DriverOperation, nth: usize) {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.scheduled.push((operation, nth));
    }

    /// Record one call of `operation` and report whether it must fail.
    pub fn should_fail(&self, operation: DriverOperation) -> bool {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let count = match state.counts.iter_mut().find(|(op, _)| *op == operation) {
            Some((_, count)) => {
                *count += 1;
                *count
            }
            None => {
                state.counts.push((operation, 1));
                1
            }
        };
        state
            .scheduled
            .iter()
            .any(|&(op, nth)| op == operation && nth == count)
    }

    /// How many times `operation` has been called through this plan.
    pub fn calls(&self, operation: DriverOperation) -> usize {
        let state = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state
            .counts
            .iter()
            .find(|(op, _)| *op == operation)
            .map_or(0, |&(_, count)| count)
    }

    /// The fault this plan produces for `operation`.
    pub fn fault(operation: DriverOperation) -> DriverFault {
        DriverFault::new(operation.name(), "injected fault (test plan)")
    }
}

/// A [`ReleaseDriver`] that mutates a recorded mapping table instead of a
/// device, so the state machine can be proven without CUDA.
///
/// It is not a mock in the "assert on calls" sense: it maintains the mapping
/// and ownership facts a driver would, so a test can ask what is *mapped* and
/// what is *owned* afterwards rather than what was called.
#[derive(Debug)]
pub struct ScriptedDriver {
    plan: std::sync::Arc<DriverFaultPlan>,
    state: std::sync::Mutex<ScriptedState>,
}

#[derive(Debug, Default)]
struct ScriptedState {
    mapped: Vec<MappedBlock>,
    /// Handles given back for reuse or released. Terminal.
    settled: Vec<PhysicalHandle>,
    /// Handles retained because disposal failed. Never reusable.
    quarantined: Vec<PhysicalHandle>,
    /// Remaining live mappings per shared handle; a handle with references
    /// left settles without being given back, exactly as the pool's shared
    /// prefix refcount does.
    shared: Vec<(PhysicalHandle, u32)>,
}

impl ScriptedDriver {
    /// A driver over `mapped`, failing according to `plan`.
    pub fn new(mapped: Vec<MappedBlock>, plan: std::sync::Arc<DriverFaultPlan>) -> Self {
        Self {
            plan,
            state: std::sync::Mutex::new(ScriptedState {
                mapped,
                ..ScriptedState::default()
            }),
        }
    }

    /// Declare `handle` mapped `references` times, so only the last dispose
    /// gives the physical memory back.
    pub fn share(&self, handle: PhysicalHandle, references: u32) {
        let mut state = self.lock();
        state.shared.push((handle, references));
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ScriptedState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Blocks still mapped.
    pub fn mapped(&self) -> Vec<MappedBlock> {
        let mut mapped = self.lock().mapped.clone();
        mapped.sort_unstable_by_key(|block| block.offset);
        mapped
    }

    /// Handles that reached a terminal state.
    pub fn settled_handles(&self) -> Vec<PhysicalHandle> {
        self.lock().settled.clone()
    }

    /// Handles held in quarantine. Reusing any of these is the bug this whole
    /// module exists to prevent.
    pub fn quarantined_handles(&self) -> Vec<PhysicalHandle> {
        self.lock().quarantined.clone()
    }

    /// Remaining live mappings of a shared handle.
    pub fn shared_references(&self, handle: PhysicalHandle) -> u32 {
        self.lock()
            .shared
            .iter()
            .find(|(shared, _)| *shared == handle)
            .map_or(0, |&(_, count)| count)
    }
}

impl ReleaseDriver for ScriptedDriver {
    fn unmap(&self, blocks: &[MappedBlock]) -> Result<(), DriverFault> {
        if self.plan.should_fail(DriverOperation::Unmap) {
            return Err(DriverFaultPlan::fault(DriverOperation::Unmap));
        }
        let mut state = self.lock();
        for block in blocks {
            assert!(
                state.mapped.iter().any(|mapped| mapped == block),
                "scripted driver asked to unmap {block:?}, which is not mapped"
            );
            state.mapped.retain(|mapped| mapped != block);
        }
        Ok(())
    }

    fn remap(&self, block: MappedBlock) -> Result<(), DriverFault> {
        if self.plan.should_fail(DriverOperation::Remap) {
            return Err(DriverFaultPlan::fault(DriverOperation::Remap));
        }
        let mut state = self.lock();
        assert!(
            !state
                .mapped
                .iter()
                .any(|mapped| mapped.offset == block.offset),
            "scripted driver asked to remap over a live mapping at {}",
            block.offset
        );
        state.mapped.push(block);
        Ok(())
    }

    fn dispose(&self, block: MappedBlock) -> HandleDisposition {
        {
            let mut state = self.lock();
            if let Some(entry) = state
                .shared
                .iter_mut()
                .find(|(handle, _)| *handle == block.handle)
            {
                entry.1 = entry.1.saturating_sub(1);
                if entry.1 > 0 {
                    // Another reservation still maps this granule: the last
                    // owner releases it, not this one.
                    return HandleDisposition::Settled;
                }
            }
        }
        if self.plan.should_fail(DriverOperation::Dispose) {
            let fault = DriverFaultPlan::fault(DriverOperation::Dispose);
            self.lock().quarantined.push(block.handle);
            return HandleDisposition::Quarantined(fault);
        }
        self.lock().settled.push(block.handle);
        HandleDisposition::Settled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn blocks(count: usize) -> Vec<MappedBlock> {
        (0..count)
            .map(|index| MappedBlock::new(index * 4096, 4096, 100 + index as u64))
            .collect()
    }

    fn runs(blocks: Vec<MappedBlock>) -> Vec<Vec<MappedBlock>> {
        blocks.into_iter().map(|block| vec![block]).collect()
    }

    #[test]
    fn adjacent_blocks_collapse_into_one_run() {
        let mut mapped = blocks(3);
        mapped.push(MappedBlock::new(3 * 4096 + 8192, 4096, 200));
        let runs = contiguous_runs(mapped);

        assert_eq!(runs.len(), 2, "one adjacent run and one detached block");
        assert_eq!(runs[0].len(), 3);
        assert_eq!(runs[1].len(), 1);
    }

    #[test]
    fn a_failed_unmap_keeps_the_whole_run_mapped_and_owned() {
        let mapped = blocks(3);
        let plan = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 1));
        let driver = ScriptedDriver::new(mapped.clone(), plan);

        let report = release_runs(&driver, std::slice::from_ref(&mapped));

        assert!(!report.is_complete());
        assert_eq!(report.still_mapped, mapped);
        assert_eq!(report.unmapped_bytes(), 0, "nothing was mutated");
        assert_eq!(report.retained_owned_bytes(), 3 * 4096);
        assert_eq!(driver.mapped(), mapped, "the run is still mapped");
        assert!(
            driver.settled_handles().is_empty(),
            "no handle may be given back while its mapping is live"
        );
    }

    #[test]
    fn the_first_middle_and_last_run_can_each_fail_alone() {
        for failing in 1..=3usize {
            let mapped = blocks(3);
            let plan = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, failing));
            let driver = ScriptedDriver::new(mapped.clone(), plan);

            let report = release_runs(&driver, &runs(mapped.clone()));

            assert_eq!(
                report.still_mapped,
                vec![mapped[failing - 1]],
                "only run {failing} may be retained"
            );
            assert_eq!(report.settled.len(), 2);
            assert_eq!(report.unmapped_bytes(), 2 * 4096);
            assert_eq!(report.retained_owned_bytes(), 4096);
            assert_eq!(driver.mapped(), vec![mapped[failing - 1]]);
        }
    }

    #[test]
    fn a_handle_that_cannot_be_given_back_refunds_mapped_but_not_owned() {
        let mapped = blocks(2);
        let plan = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Dispose, 2));
        let driver = ScriptedDriver::new(mapped.clone(), plan);

        let report = release_runs(&driver, std::slice::from_ref(&mapped));

        assert!(report.still_mapped.is_empty(), "both blocks were unmapped");
        assert_eq!(report.unmapped_handle_owned, vec![mapped[1]]);
        assert_eq!(
            report.unmapped_bytes(),
            2 * 4096,
            "the mapped axis is refunded for every unmapped block, including the quarantined one"
        );
        assert_eq!(
            report.retained_owned_bytes(),
            4096,
            "only the quarantined handle stays charged on the owned axis"
        );
        assert_eq!(driver.quarantined_handles(), vec![mapped[1].handle]);
        assert!(
            !driver.settled_handles().contains(&mapped[1].handle),
            "a handle whose release failed must never become reusable"
        );
    }

    #[test]
    fn a_shared_granule_survives_until_its_last_mapping_leaves() {
        let block = MappedBlock::new(0, 4096, 900);
        let driver = ScriptedDriver::new(vec![block], Arc::new(DriverFaultPlan::new()));
        driver.share(900, 2);

        let report = release_runs(&driver, &[vec![block]]);

        assert!(report.is_complete(), "this sharer's reference is terminal");
        assert_eq!(driver.shared_references(900), 1);
        assert!(
            driver.settled_handles().is_empty(),
            "the physical granule belongs to the remaining sharer"
        );

        let driver = ScriptedDriver::new(vec![block], Arc::new(DriverFaultPlan::new()));
        driver.share(900, 1);
        let report = release_runs(&driver, &[vec![block]]);
        assert!(report.is_complete());
        assert_eq!(
            driver.settled_handles(),
            vec![900],
            "the last mapping to leave gives the granule back"
        );
    }

    #[test]
    fn a_rolled_back_decommit_restores_the_exact_topology() {
        let mapped = blocks(3);
        let plan = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 3));
        let driver = ScriptedDriver::new(mapped.clone(), plan);

        let outcome = unmap_runs_transactional(&driver, &runs(mapped.clone()));

        assert!(matches!(outcome, TransactionalUnmap::RolledBack { .. }));
        assert_eq!(driver.mapped(), mapped, "every block is mapped again");
        assert!(
            driver.settled_handles().is_empty() && driver.quarantined_handles().is_empty(),
            "no handle may be disposed while a rollback is still possible"
        );
    }

    /// The premise the hardware rollback fixture rests on (#1474).
    ///
    /// `a_rolled_back_decommit_leaves_the_buffer_readable`
    /// (`tests/vmm_release_quarantine_gpu.rs`) schedules its fault on the
    /// *second* unmap, because a rollback needs a run that is already unmapped
    /// — failing the first one leaves nothing to map back and exercises a
    /// different path entirely. That only works if the decommit issues two
    /// unmap calls, and [`contiguous_runs`] exists precisely to collapse an
    /// unbroken range into one. On real hardware the fresh 8 MiB allocation was
    /// unbroken, so the second fault never fired and the decommit completed;
    /// the fixture reported that correctly, but its premise was false.
    ///
    /// Both halves are anchored here so the hardware fixture's reason for
    /// punching a hole first is *executed* on every machine rather than argued
    /// from a doc comment on a GPU-gated path.
    #[test]
    fn a_second_unmap_fault_needs_a_hole_because_a_contiguous_range_is_one_call() {
        let contiguous = blocks(4);
        assert_eq!(
            contiguous_runs(contiguous.clone()).len(),
            1,
            "premise: an unbroken range is one run, and therefore one unmap call"
        );
        let plan = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 2));
        let driver = ScriptedDriver::new(contiguous.clone(), Arc::clone(&plan));

        let outcome = unmap_runs_transactional(&driver, &contiguous_runs(contiguous.clone()));

        assert!(
            matches!(outcome, TransactionalUnmap::Unmapped { .. }),
            "a fault scheduled for the second unmap cannot fire when only one is made, so the \
             decommit completes and no rollback is observed"
        );
        assert_eq!(plan.calls(DriverOperation::Unmap), 1);

        // The same range with one granule already decommitted: two runs, two
        // calls, and the fault now lands with a run behind it to restore.
        let holed = contiguous
            .iter()
            .copied()
            .filter(|block| block.offset != 2 * 4096)
            .collect::<Vec<_>>();
        assert_eq!(
            contiguous_runs(holed.clone()).len(),
            2,
            "the hole is what makes a second unmap call exist at all"
        );
        let plan = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 2));
        let driver = ScriptedDriver::new(holed.clone(), Arc::clone(&plan));

        let outcome = unmap_runs_transactional(&driver, &contiguous_runs(holed.clone()));

        assert!(
            matches!(outcome, TransactionalUnmap::RolledBack { .. }),
            "with two runs the second-unmap fault must roll the first run back"
        );
        assert_eq!(plan.calls(DriverOperation::Unmap), 2);
        assert_eq!(
            driver.mapped(),
            holed,
            "the rollback restored every block it had unmapped"
        );
    }

    #[test]
    fn a_failed_rollback_reports_every_residual_and_disposes_nothing() {
        let mapped = blocks(3);
        let plan = Arc::new(
            DriverFaultPlan::new()
                .fail_nth(DriverOperation::Unmap, 3)
                .fail_nth(DriverOperation::Remap, 1),
        );
        let driver = ScriptedDriver::new(mapped.clone(), plan);

        let outcome = unmap_runs_transactional(&driver, &runs(mapped.clone()));

        let TransactionalUnmap::RollbackFailed {
            still_mapped,
            unmapped_handle_owned,
            faults,
        } = outcome
        else {
            panic!("a failed remap must not be reported as a rollback");
        };
        // Remap runs in reverse, so block 1 (the last unmapped) is the one that
        // fails; block 0 is restored and block 2 was never unmapped.
        assert_eq!(unmapped_handle_owned, vec![mapped[1]]);
        assert_eq!(still_mapped, vec![mapped[0], mapped[2]]);
        assert_eq!(faults.len(), 2, "the unmap fault and the remap fault");
        assert_eq!(driver.mapped(), vec![mapped[0], mapped[2]]);
        assert!(
            driver.settled_handles().is_empty(),
            "an unmapped block whose remap failed keeps its handle"
        );
    }

    #[test]
    fn a_transactional_unmap_that_succeeds_retains_every_handle_for_phase_two() {
        let mapped = blocks(2);
        let driver = ScriptedDriver::new(mapped.clone(), Arc::new(DriverFaultPlan::new()));

        let outcome = unmap_runs_transactional(&driver, std::slice::from_ref(&mapped));

        let TransactionalUnmap::Unmapped { blocks } = outcome else {
            panic!("the unmap phase succeeded");
        };
        assert_eq!(blocks, mapped);
        assert!(
            driver.settled_handles().is_empty(),
            "phase one never disposes"
        );

        let report = dispose_released_blocks(&driver, &blocks);
        assert!(report.is_complete());
        assert_eq!(driver.settled_handles().len(), 2);
    }

    #[test]
    fn faults_are_scoped_to_one_plan_instance() {
        let plan = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 1));
        let faulted = ScriptedDriver::new(blocks(1), Arc::clone(&plan));
        let clean = ScriptedDriver::new(blocks(1), Arc::new(DriverFaultPlan::new()));

        assert!(!release_runs(&faulted, &[blocks(1)]).is_complete());
        assert!(
            release_runs(&clean, &[blocks(1)]).is_complete(),
            "a second driver with its own plan must be unaffected"
        );
        assert_eq!(plan.calls(DriverOperation::Unmap), 1);
    }
}
