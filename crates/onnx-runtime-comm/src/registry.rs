//! Backend buffer-ownership lease registry
//! (`docs/DISTRIBUTED_RUNTIME.md` §3.1, refined against
//! `specs/tla/BufferOwnership.tla` and `specs/tla/REFINEMENT.md`).
//!
//! # The invariant
//!
//! **The backend registry is the single owner of every transport-held
//! allocation lease.** A buffer has exactly one owner at a time; a write lease
//! is exclusive (it conflicts with every other read or write lease on that
//! allocation); read leases may alias; there is no use-after-release and no
//! double-free; and ownership transfer (`Detach`) is atomic and never releases
//! a lease. Detaching the user-visible handle does not release either lease.
//!
//! # Discipline (mirrors `onnx-genai-scheduler::pressure`)
//!
//! * A single authoritative [`std::sync::Mutex`] guards all registry state.
//! * Every linearization point emits exactly one [`ProtocolTraceEvent`] at the
//!   true atomic commit point, under the registry lock, while the registry —
//!   not the trace sink — stays authoritative.
//! * The registry never blocks/waits while holding its lock. Submit either
//!   succeeds immediately (buffers available) or fails with a conflict/free
//!   error; higher layers (the 1c collective-ordering sequencer) own any
//!   waiting.
//! * Terminal operations hold no lease and are reaped immediately so
//!   recomputation stays O(live).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use onnx_runtime_protocol_trace::{
    BufferOwnershipEvent, CONTRACT_REVISION, NullTraceSink, OperationId, PhysicalAllocationId,
    ProtocolEvent, ProtocolSourceId, ProtocolTraceEvent, ProtocolTraceSink,
};

/// The complete read/write lease set for one transport operation
/// (`docs/DISTRIBUTED_RUNTIME.md` §3.1). Read/read aliasing is legal; a write
/// lease conflicts with every other read or write lease on the same physical
/// allocation. An in-place operation lists its allocation in both sets and
/// therefore acquires it exclusively.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommLeaseSet {
    /// Allocations the backend may read until local terminal completion.
    pub reads: Vec<PhysicalAllocationId>,
    /// Allocations the backend may write until local terminal completion.
    pub writes: Vec<PhysicalAllocationId>,
}

impl CommLeaseSet {
    /// A read-only lease set (e.g. a `send` source buffer).
    pub fn read_only(reads: impl IntoIterator<Item = PhysicalAllocationId>) -> Self {
        Self {
            reads: reads.into_iter().collect(),
            writes: Vec::new(),
        }
    }

    /// A write-only lease set (e.g. a `recv` destination buffer).
    pub fn write_only(writes: impl IntoIterator<Item = PhysicalAllocationId>) -> Self {
        Self {
            reads: Vec::new(),
            writes: writes.into_iter().collect(),
        }
    }

    /// An empty lease set (e.g. a `barrier`).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Whether this operation holds no leases at all.
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty()
    }
}

/// Why a registry action could not commit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OwnershipError {
    /// A requested allocation has already been freed by the allocator; it can
    /// never be leased again (`Submit` requires `buffer \notin freed`).
    #[error("operation references freed allocation {0}")]
    FreedAllocation(PhysicalAllocationId),
    /// A requested lease conflicts with an active operation's lease on the same
    /// allocation (write-vs-any). The registry rejects rather than corrupt
    /// ownership; the caller must retry after the conflicting op completes.
    #[error("operation lease on {allocation} conflicts with active operation {holder}")]
    LeaseConflict {
        allocation: PhysicalAllocationId,
        holder: OperationId,
    },
    /// A registry action named an operation that is not present or not in a
    /// state where that action is enabled.
    #[error("operation {operation} is not in a state enabling {action}")]
    NotEnabled {
        operation: OperationId,
        action: &'static str,
    },
    /// `free_buffer` was asked to free an allocation still referenced by an
    /// active operation's read or write lease.
    #[error("cannot free allocation {allocation}: still leased by active operation {holder}")]
    BufferStillLeased {
        allocation: PhysicalAllocationId,
        holder: OperationId,
    },
    /// `free_buffer` was asked to free an allocation that is already freed.
    #[error("allocation {0} is already freed")]
    AlreadyFreed(PhysicalAllocationId),
    /// The bounded exact tombstone window is full. The allocator must seal an
    /// epoch before more out-of-order frees can be recorded.
    #[error("freed-allocation tombstone window is full; seal an allocator epoch")]
    FreedTrackingCapacity,
}

/// Backend registry state for one operation. Maps to `BufferOwnership.tla`'s
/// per-operation `operationState`, `handleAttached`, and `registryOwned`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpState {
    /// `Active`: `operationState = "submitted"`.
    Submitted,
    /// `Active`: `operationState = "aborting"`.
    Aborting,
}

#[derive(Debug, Clone)]
struct OperationEntry {
    reads: BTreeSet<PhysicalAllocationId>,
    writes: BTreeSet<PhysicalAllocationId>,
    state: OpState,
    handle_attached: bool,
}

impl OperationEntry {
    fn references(&self, buffer: PhysicalAllocationId) -> bool {
        self.reads.contains(&buffer) || self.writes.contains(&buffer)
    }
}

/// The single authoritative registry state, guarded by a short-held lock.
#[derive(Debug)]
struct RegistryState {
    operations: BTreeMap<OperationId, OperationEntry>,
    /// Exact tombstones at or above `retired_before`.
    freed: BTreeSet<PhysicalAllocationId>,
    /// Allocator-proven floor: identities below this value can never be leased
    /// again, allowing their exact tombstones to be reclaimed.
    retired_before: u64,
    freed_total: u64,
    next_operation: u64,
    source_sequence: u64,
    /// Count of terminally-completed operations, for the invariant gate.
    completed_ok: u64,
    completed_error: u64,
}

impl RegistryState {
    const MAX_FREED_TOMBSTONES: usize = 4096;

    fn is_freed(&self, buffer: PhysicalAllocationId) -> bool {
        buffer.get() < self.retired_before || self.freed.contains(&buffer)
    }

    /// Checks that a candidate lease set is admissible: no freed allocation and
    /// no conflict with any active operation. Mirrors
    /// `BufferOwnership.tla!BuffersAvailable` generalized to lease *sets*.
    fn buffers_available(
        &self,
        reads: &BTreeSet<PhysicalAllocationId>,
        writes: &BTreeSet<PhysicalAllocationId>,
    ) -> Result<(), OwnershipError> {
        for buffer in reads.iter().chain(writes.iter()) {
            if self.is_freed(*buffer) {
                return Err(OwnershipError::FreedAllocation(*buffer));
            }
        }
        for (holder, other) in &self.operations {
            // A new write conflicts with any active lease (read or write) the
            // other op holds on that allocation.
            for w in writes {
                if other.references(*w) {
                    return Err(OwnershipError::LeaseConflict {
                        allocation: *w,
                        holder: *holder,
                    });
                }
            }
            // A new read conflicts only with an active *write* lease.
            for r in reads {
                if other.writes.contains(r) {
                    return Err(OwnershipError::LeaseConflict {
                        allocation: *r,
                        holder: *holder,
                    });
                }
            }
        }
        Ok(())
    }
}

/// An independent recomputation of the registry invariants from authoritative
/// entries (the Invariant gate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnershipSnapshot {
    /// Number of `submitted` operations.
    pub submitted: usize,
    /// Number of `aborting` operations.
    pub aborting: usize,
    /// Number of active (submitted or aborting) operations still owned.
    pub active: usize,
    /// Number of active operations whose user handle has detached.
    pub detached_active: usize,
    /// Distinct allocations currently under an active read or write lease.
    pub leased_allocations: usize,
    /// Cumulative number of allocations freed since this registry started.
    ///
    /// This is a total-ever-freed counter, not the number of currently live
    /// freed-allocation tombstones.
    pub freed_allocations: usize,
    /// Terminal operations that completed successfully so far.
    pub completed_ok: u64,
    /// Terminal operations that quiesced after abort so far.
    pub completed_error: u64,
}

/// The public backend registry handle. Cheap to clone (shared inner state).
#[derive(Clone)]
pub struct OwnershipRegistry {
    inner: Arc<RegistryInner>,
}

struct RegistryInner {
    state: Mutex<RegistryState>,
    sink: Arc<dyn ProtocolTraceSink>,
    source: ProtocolSourceId,
    topology_epoch: u64,
}

impl OwnershipRegistry {
    /// Creates a registry with a no-op trace sink.
    pub fn new(source: ProtocolSourceId, topology_epoch: u64) -> Self {
        Self::with_sink(source, topology_epoch, Arc::new(NullTraceSink))
    }

    /// Creates a registry that emits linearization events to `sink`.
    pub fn with_sink(
        source: ProtocolSourceId,
        topology_epoch: u64,
        sink: Arc<dyn ProtocolTraceSink>,
    ) -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                state: Mutex::new(RegistryState {
                    operations: BTreeMap::new(),
                    freed: BTreeSet::new(),
                    retired_before: 0,
                    freed_total: 0,
                    next_operation: 1,
                    source_sequence: 0,
                    completed_ok: 0,
                    completed_error: 0,
                }),
                sink,
                source,
                topology_epoch,
            }),
        }
    }

    /// `BufferOwnership!Submit`. The backend registry takes ownership of a new
    /// operation and its full read/write lease sets before transport enqueue.
    ///
    /// **Linearization point:** the operation and its leases are inserted under
    /// the registry lock, after the conflict/free check succeeds, and the
    /// `Submit` event is emitted in the same critical section.
    ///
    /// Returns an [`OperationLease`] — the user-visible handle whose drop
    /// detaches (never releases). On a conflict or freed allocation the request
    /// is rejected with no state change.
    pub fn submit(&self, leases: &CommLeaseSet) -> Result<OperationLease, OwnershipError> {
        let reads: BTreeSet<PhysicalAllocationId> = leases.reads.iter().copied().collect();
        let writes: BTreeSet<PhysicalAllocationId> = leases.writes.iter().copied().collect();

        let mut state = self.inner.lock();
        state.buffers_available(&reads, &writes)?;

        let operation = OperationId::new(state.next_operation);
        state.next_operation += 1;

        let read_vec: Vec<PhysicalAllocationId> = reads.iter().copied().collect();
        let write_vec: Vec<PhysicalAllocationId> = writes.iter().copied().collect();

        state.operations.insert(
            operation,
            OperationEntry {
                reads,
                writes,
                state: OpState::Submitted,
                handle_attached: true,
            },
        );

        self.inner.emit_locked(
            &mut state,
            BufferOwnershipEvent::Submit {
                operation,
                reads: read_vec,
                writes: write_vec,
            },
        );

        Ok(OperationLease {
            operation,
            inner: self.inner.clone(),
            detached: false,
        })
    }

    /// `BufferOwnership!Detach`. The user handle detaches without changing
    /// registry ownership; both leases remain owned. A no-op if the operation is
    /// already terminal (reaped) or already detached.
    ///
    /// **Linearization point:** `handle_attached` flips to `false` under the
    /// registry lock and the `Detach` event is emitted in the same section.
    pub fn detach(&self, operation: OperationId) {
        let mut state = self.inner.lock();
        match state.operations.get_mut(&operation) {
            Some(entry) if entry.handle_attached => entry.handle_attached = false,
            // Already detached, or already terminal: nothing to publish.
            _ => return,
        }
        self.inner
            .emit_locked(&mut state, BufferOwnershipEvent::Detach { operation });
    }

    /// `BufferOwnership!BeginAbort`. A submitted operation becomes aborting while
    /// all leases remain owned.
    pub fn begin_abort(&self, operation: OperationId) -> Result<(), OwnershipError> {
        let mut state = self.inner.lock();
        match state.operations.get_mut(&operation) {
            Some(entry) if entry.state == OpState::Submitted => entry.state = OpState::Aborting,
            _ => {
                return Err(OwnershipError::NotEnabled {
                    operation,
                    action: "begin_abort",
                });
            }
        }
        self.inner
            .emit_locked(&mut state, BufferOwnershipEvent::BeginAbort { operation });
        Ok(())
    }

    /// `BufferOwnership!CompleteSuccess`. The backend proves terminal success and
    /// releases every registry lease in one critical section; each written
    /// allocation advances its reuse generation. The terminal entry is reaped.
    ///
    /// **Linearization point:** the lease release and `Submitted -> terminal`
    /// transition commit atomically under the registry lock, and the
    /// `CompleteSuccess` event is emitted in the same section.
    pub fn complete_success(&self, operation: OperationId) -> Result<(), OwnershipError> {
        self.retire(operation, OpState::Submitted, true)
    }

    /// `BufferOwnership!QuiesceAbort`. An aborting operation reaches quiescence,
    /// releasing every registry lease; each written allocation advances its
    /// reuse generation. The terminal entry is reaped.
    pub fn quiesce_abort(&self, operation: OperationId) -> Result<(), OwnershipError> {
        self.retire(operation, OpState::Aborting, false)
    }

    fn retire(
        &self,
        operation: OperationId,
        required: OpState,
        success: bool,
    ) -> Result<(), OwnershipError> {
        let action = if success {
            "complete_success"
        } else {
            "quiesce_abort"
        };
        let mut state = self.inner.lock();
        let entry = match state.operations.get(&operation) {
            Some(entry) if entry.state == required => entry.clone(),
            _ => return Err(OwnershipError::NotEnabled { operation, action }),
        };
        // Release the leases (reap the terminal entry — it owns nothing now).
        state.operations.remove(&operation);
        let writes: Vec<PhysicalAllocationId> = entry.writes.iter().copied().collect();
        let event = if success {
            state.completed_ok += 1;
            BufferOwnershipEvent::CompleteSuccess { operation, writes }
        } else {
            state.completed_error += 1;
            BufferOwnershipEvent::QuiesceAbort { operation, writes }
        };
        self.inner.emit_locked(&mut state, event);
        Ok(())
    }

    /// `BufferOwnership!FreeBuffer`. The allocator commits free/reuse only after
    /// no registry read or write lease references the physical allocation.
    ///
    /// **Linearization point:** the allocation is added to the freed set under
    /// the registry lock and the `FreeBuffer` event is emitted in the same
    /// section. Because [`PhysicalAllocationId`] is never reused, a freed
    /// allocation can never be leased again (the ABA-prevention obligation).
    pub fn free_buffer(&self, buffer: PhysicalAllocationId) -> Result<(), OwnershipError> {
        let mut state = self.inner.lock();
        if state.is_freed(buffer) {
            return Err(OwnershipError::AlreadyFreed(buffer));
        }
        for (holder, entry) in &state.operations {
            if entry.references(buffer) {
                return Err(OwnershipError::BufferStillLeased {
                    allocation: buffer,
                    holder: *holder,
                });
            }
        }
        if state.freed.len() >= RegistryState::MAX_FREED_TOMBSTONES {
            return Err(OwnershipError::FreedTrackingCapacity);
        }
        state.freed.insert(buffer);
        state.freed_total = state.freed_total.saturating_add(1);
        self.inner
            .emit_locked(&mut state, BufferOwnershipEvent::FreeBuffer { buffer });
        Ok(())
    }

    /// Reclaims exact free tombstones below an allocator-proven identity floor.
    ///
    /// The caller must only advance this floor after proving that no live or
    /// future buffer can carry an identity below `next_live_allocation`. Such
    /// identities remain permanently rejected; only their individual set
    /// entries are discarded, so no-reuse-after-free is preserved while memory
    /// stays bounded across allocator epochs.
    pub fn seal_allocator_epoch(
        &self,
        next_live_allocation: PhysicalAllocationId,
    ) -> Result<(), OwnershipError> {
        let mut state = self.inner.lock();
        for (holder, entry) in &state.operations {
            if let Some(allocation) = entry
                .reads
                .iter()
                .chain(&entry.writes)
                .find(|allocation| allocation.get() < next_live_allocation.get())
            {
                return Err(OwnershipError::BufferStillLeased {
                    allocation: *allocation,
                    holder: *holder,
                });
            }
        }
        state.retired_before = state.retired_before.max(next_live_allocation.get());
        let floor = state.retired_before;
        state.freed.retain(|buffer| buffer.get() >= floor);
        Ok(())
    }

    /// Independently recomputes the registry invariants from authoritative
    /// entries (the Invariant gate). Returns an error if any invariant is
    /// violated — e.g. two active operations hold conflicting leases, or a freed
    /// allocation is still leased.
    pub fn snapshot(&self) -> Result<OwnershipSnapshot, OwnershipError> {
        let state = self.inner.lock();

        let mut submitted = 0usize;
        let mut aborting = 0usize;
        let mut detached_active = 0usize;
        let mut leased: BTreeSet<PhysicalAllocationId> = BTreeSet::new();

        for entry in state.operations.values() {
            match entry.state {
                OpState::Submitted => submitted += 1,
                OpState::Aborting => aborting += 1,
            }
            if !entry.handle_attached {
                detached_active += 1;
            }
            leased.extend(entry.reads.iter().copied());
            leased.extend(entry.writes.iter().copied());
        }

        // Invariant: no freed allocation is still leased (FreedHasNoLease).
        for buffer in &leased {
            if state.is_freed(*buffer) {
                return Err(OwnershipError::FreedAllocation(*buffer));
            }
        }
        // Invariant: no two active operations hold conflicting leases
        // (NoConflictingActiveLeases). Recomputed independently of submit.
        let entries: Vec<&OperationEntry> = state.operations.values().collect();
        for (i, a) in entries.iter().enumerate() {
            for b in entries.iter().skip(i + 1) {
                for w in &a.writes {
                    if b.references(*w) {
                        return Err(OwnershipError::LeaseConflict {
                            allocation: *w,
                            holder: OperationId::new(0),
                        });
                    }
                }
                for w in &b.writes {
                    if a.reads.contains(w) {
                        return Err(OwnershipError::LeaseConflict {
                            allocation: *w,
                            holder: OperationId::new(0),
                        });
                    }
                }
            }
        }

        Ok(OwnershipSnapshot {
            submitted,
            aborting,
            active: submitted + aborting,
            detached_active,
            leased_allocations: leased.len(),
            freed_allocations: usize::try_from(state.freed_total).unwrap_or(usize::MAX),
            completed_ok: state.completed_ok,
            completed_error: state.completed_error,
        })
    }
}

impl RegistryInner {
    fn lock(&self) -> std::sync::MutexGuard<'_, RegistryState> {
        self.state.lock().expect("ownership registry poisoned")
    }

    /// Emits one linearization event under the registry lock. The registry stays
    /// authoritative; the sink only records.
    fn emit_locked(&self, state: &mut RegistryState, kind: BufferOwnershipEvent) {
        let sequence = state.source_sequence;
        state.source_sequence += 1;
        self.sink.record(ProtocolTraceEvent {
            contract_revision: CONTRACT_REVISION,
            topology_epoch: self.topology_epoch,
            source: self.source,
            source_sequence: sequence,
            kind: ProtocolEvent::BufferOwnership(kind),
        });
    }
}

/// The user-visible handle for a submitted transport operation.
///
/// Dropping the handle **detaches** the caller (`BufferOwnership!Detach`); it
/// NEVER releases a registry lease or cancels transport progress. The backend
/// registry retains ownership until it proves terminal completion
/// (`complete_success`) or abort quiescence (`quiesce_abort`). This matches the
/// communicator contract: "dropping the Rust handle cannot stop progress or
/// release storage."
pub struct OperationLease {
    operation: OperationId,
    inner: Arc<RegistryInner>,
    detached: bool,
}

impl std::fmt::Debug for OperationLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperationLease")
            .field("operation", &self.operation)
            .field("detached", &self.detached)
            .finish_non_exhaustive()
    }
}

impl OperationLease {
    /// The stable identity of this operation in the backend registry.
    pub fn operation(&self) -> OperationId {
        self.operation
    }

    /// Explicitly detaches the user handle without releasing registry ownership.
    /// Idempotent. Equivalent to dropping the handle, but callable early.
    pub fn detach(mut self) {
        self.detach_in_place();
    }

    fn detach_in_place(&mut self) {
        if self.detached {
            return;
        }
        self.detached = true;
        let registry = OwnershipRegistry {
            inner: self.inner.clone(),
        };
        registry.detach(self.operation);
    }
}

impl Drop for OperationLease {
    fn drop(&mut self) {
        self.detach_in_place();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> OwnershipRegistry {
        OwnershipRegistry::new(ProtocolSourceId::new(1), 0)
    }

    fn alloc(id: u64) -> PhysicalAllocationId {
        PhysicalAllocationId::new(id)
    }

    #[test]
    fn submit_then_complete_success_releases_and_reaps() {
        let reg = registry();
        let lease = reg.submit(&CommLeaseSet::read_only([alloc(10)])).unwrap();
        let snap = reg.snapshot().unwrap();
        assert_eq!(snap.active, 1);
        assert_eq!(snap.leased_allocations, 1);
        let op = lease.operation();
        reg.complete_success(op).unwrap();
        // Detaching the already-terminal handle must be a harmless no-op.
        drop(lease);
        let snap = reg.snapshot().unwrap();
        assert_eq!(snap.active, 0, "terminal entry must be reaped");
        assert_eq!(snap.leased_allocations, 0);
        assert_eq!(snap.completed_ok, 1);
    }

    #[test]
    fn read_read_aliasing_is_allowed() {
        let reg = registry();
        let a = reg.submit(&CommLeaseSet::read_only([alloc(1)])).unwrap();
        let b = reg.submit(&CommLeaseSet::read_only([alloc(1)])).unwrap();
        let snap = reg.snapshot().unwrap();
        assert_eq!(snap.active, 2);
        assert_eq!(snap.leased_allocations, 1);
        reg.complete_success(a.operation()).unwrap();
        reg.complete_success(b.operation()).unwrap();
    }

    #[test]
    fn write_conflicts_with_any_active_lease() {
        let reg = registry();
        let reader = reg.submit(&CommLeaseSet::read_only([alloc(7)])).unwrap();
        // A write on the same allocation must be rejected while a reader is live.
        let err = reg
            .submit(&CommLeaseSet::write_only([alloc(7)]))
            .unwrap_err();
        assert!(matches!(err, OwnershipError::LeaseConflict { .. }));
        // After the reader completes, the write may proceed (serialization).
        reg.complete_success(reader.operation()).unwrap();
        let writer = reg.submit(&CommLeaseSet::write_only([alloc(7)])).unwrap();
        reg.complete_success(writer.operation()).unwrap();
    }

    #[test]
    fn read_conflicts_with_active_write() {
        let reg = registry();
        let writer = reg.submit(&CommLeaseSet::write_only([alloc(3)])).unwrap();
        let err = reg
            .submit(&CommLeaseSet::read_only([alloc(3)]))
            .unwrap_err();
        assert!(matches!(err, OwnershipError::LeaseConflict { .. }));
        reg.complete_success(writer.operation()).unwrap();
    }

    #[test]
    fn detach_does_not_release_lease() {
        let reg = registry();
        let lease = reg.submit(&CommLeaseSet::write_only([alloc(4)])).unwrap();
        let op = lease.operation();
        lease.detach();
        let snap = reg.snapshot().unwrap();
        assert_eq!(snap.active, 1, "detach must not release the lease");
        assert_eq!(snap.detached_active, 1);
        // A conflicting op is still rejected while the detached op owns the lease.
        assert!(reg.submit(&CommLeaseSet::read_only([alloc(4)])).is_err());
        reg.complete_success(op).unwrap();
    }

    #[test]
    fn abort_path_quiesces_and_reaps() {
        let reg = registry();
        let lease = reg.submit(&CommLeaseSet::write_only([alloc(9)])).unwrap();
        let op = lease.operation();
        reg.begin_abort(op).unwrap();
        assert_eq!(reg.snapshot().unwrap().aborting, 1);
        reg.quiesce_abort(op).unwrap();
        let snap = reg.snapshot().unwrap();
        assert_eq!(snap.active, 0);
        assert_eq!(snap.completed_error, 1);
    }

    #[test]
    fn free_of_leased_allocation_is_rejected() {
        let reg = registry();
        let lease = reg.submit(&CommLeaseSet::read_only([alloc(5)])).unwrap();
        let err = reg.free_buffer(alloc(5)).unwrap_err();
        assert!(matches!(err, OwnershipError::BufferStillLeased { .. }));
        reg.complete_success(lease.operation()).unwrap();
        reg.free_buffer(alloc(5)).unwrap();
    }

    #[test]
    fn submit_after_free_is_rejected_and_double_free_errors() {
        let reg = registry();
        reg.free_buffer(alloc(6)).unwrap();
        let err = reg
            .submit(&CommLeaseSet::read_only([alloc(6)]))
            .unwrap_err();
        assert!(matches!(err, OwnershipError::FreedAllocation(_)));
        let err = reg.free_buffer(alloc(6)).unwrap_err();
        assert!(matches!(err, OwnershipError::AlreadyFreed(_)));
    }

    #[test]
    fn allocator_epoch_reclaims_exact_tombstones_without_allowing_reuse() {
        let reg = registry();
        reg.free_buffer(alloc(1)).unwrap();
        reg.free_buffer(alloc(2)).unwrap();
        reg.seal_allocator_epoch(alloc(3)).unwrap();
        assert!(matches!(
            reg.submit(&CommLeaseSet::read_only([alloc(1)])),
            Err(OwnershipError::FreedAllocation(_))
        ));
        assert_eq!(reg.snapshot().unwrap().freed_allocations, 2);
    }

    #[test]
    fn freed_tombstone_window_is_bounded() {
        let reg = registry();
        for id in 1..=RegistryState::MAX_FREED_TOMBSTONES as u64 {
            reg.free_buffer(alloc(id)).unwrap();
        }
        assert_eq!(
            reg.free_buffer(alloc(RegistryState::MAX_FREED_TOMBSTONES as u64 + 1)),
            Err(OwnershipError::FreedTrackingCapacity)
        );
        reg.seal_allocator_epoch(alloc(RegistryState::MAX_FREED_TOMBSTONES as u64 + 1))
            .unwrap();
        reg.free_buffer(alloc(RegistryState::MAX_FREED_TOMBSTONES as u64 + 1))
            .unwrap();
    }

    #[test]
    fn in_place_op_holds_allocation_exclusively() {
        let reg = registry();
        let lease = reg
            .submit(&CommLeaseSet {
                reads: vec![alloc(2)],
                writes: vec![alloc(2)],
            })
            .unwrap();
        // No other lease may touch the in-place allocation.
        assert!(reg.submit(&CommLeaseSet::read_only([alloc(2)])).is_err());
        reg.complete_success(lease.operation()).unwrap();
    }

    #[test]
    fn complete_wrong_state_is_not_enabled() {
        let reg = registry();
        let lease = reg.submit(&CommLeaseSet::empty()).unwrap();
        let op = lease.operation();
        reg.begin_abort(op).unwrap();
        // complete_success requires Submitted, not Aborting.
        let err = reg.complete_success(op).unwrap_err();
        assert!(matches!(err, OwnershipError::NotEnabled { .. }));
        reg.quiesce_abort(op).unwrap();
    }
}
