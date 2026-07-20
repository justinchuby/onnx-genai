//! Independent conformance replay checker + deterministic campaigns for the
//! communicator buffer-ownership lease protocol.
//!
//! This test crate encodes the abstract `specs/tla/BufferOwnership.tla` state
//! machine as an INDEPENDENT reducer ([`BufferOwnershipProtocol`]) that shares
//! only the identity/event *definitions* with the implementation — never its
//! transition code (`specs/tla/REFINEMENT.md` § "Independent Replay Checker").
//! The campaigns drive the real [`OwnershipRegistry`] (and, for cross-rank
//! coverage, the real [`InProcessCommunicator`]), capture a lossless trace, and
//! feed it to the generic [`ReplayChecker`], which must accept it, confirm every
//! invariant after every transition, and reject leftover active operations at a
//! clean end.
//!
//! The abstraction generalizes the model's single `ReadBuffer`/`WriteBuffer` to
//! full read/write lease *sets* (`REFINEMENT.md` § "Buffer ownership"): two
//! active operations may share read leases, but a write lease conflicts with
//! every other read or write lease on that allocation; an in-place operation
//! lists the same allocation in both sets and is therefore exclusive.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use onnx_runtime_comm::{
    CommInstanceId, CommLeaseSet, Communicator, DType, InProcessCommunicator, OwnershipRegistry,
    block_on,
};
use onnx_runtime_protocol_trace::checker::{
    AbstractProtocol, ActionResolution, BoundedTraceCollector, ConformanceFailure, FailureReason,
    ReplayChecker, TraceEnd, TraceIntegrityError, TraceSnapshot,
};
use onnx_runtime_protocol_trace::{
    BufferOwnershipEvent, OperationId, PhysicalAllocationId, ProtocolEvent, ProtocolSourceId,
    ProtocolTraceEvent,
};

// ─────────────────────── independent abstract reducer ──────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbsOpState {
    Submitted,
    Aborting,
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbsOutcome {
    None,
    Ok,
    Error,
}

#[derive(Debug, Clone)]
struct AbsOperation {
    reads: BTreeSet<PhysicalAllocationId>,
    writes: BTreeSet<PhysicalAllocationId>,
    state: AbsOpState,
    handle_attached: bool,
    registry_owned: bool,
    outcome: AbsOutcome,
    observed_read_gen: BTreeMap<PhysicalAllocationId, u64>,
    observed_write_gen: BTreeMap<PhysicalAllocationId, u64>,
}

impl AbsOperation {
    fn is_active(&self) -> bool {
        matches!(self.state, AbsOpState::Submitted | AbsOpState::Aborting)
    }

    fn references(&self, buffer: PhysicalAllocationId) -> bool {
        self.reads.contains(&buffer) || self.writes.contains(&buffer)
    }
}

/// The independent abstract registry (mirrors `BufferOwnership.tla` variables,
/// generalized to lease sets).
#[derive(Debug, Clone, Default)]
struct BufferOwnershipAbstract {
    operations: BTreeMap<OperationId, AbsOperation>,
    freed: BTreeSet<PhysicalAllocationId>,
    buffer_generation: BTreeMap<PhysicalAllocationId, u64>,
}

impl BufferOwnershipAbstract {
    fn generation(&self, buffer: PhysicalAllocationId) -> u64 {
        self.buffer_generation.get(&buffer).copied().unwrap_or(0)
    }

    /// `BuffersAvailable` generalized to sets: no freed allocation, no conflict
    /// with any active operation.
    fn buffers_available(
        &self,
        reads: &BTreeSet<PhysicalAllocationId>,
        writes: &BTreeSet<PhysicalAllocationId>,
    ) -> Result<(), String> {
        for b in reads.iter().chain(writes.iter()) {
            if self.freed.contains(b) {
                return Err(format!("Submit references freed allocation {b}"));
            }
        }
        for (holder, other) in &self.operations {
            if !other.is_active() {
                continue;
            }
            for w in writes {
                if other.references(*w) {
                    return Err(format!(
                        "write lease {w} conflicts with active operation {holder}"
                    ));
                }
            }
            for r in reads {
                if other.writes.contains(r) {
                    return Err(format!(
                        "read lease {r} conflicts with active write of operation {holder}"
                    ));
                }
            }
        }
        Ok(())
    }
}

fn set_of(v: &[PhysicalAllocationId]) -> BTreeSet<PhysicalAllocationId> {
    v.iter().copied().collect()
}

fn sorted(set: &BTreeSet<PhysicalAllocationId>) -> Vec<PhysicalAllocationId> {
    set.iter().copied().collect()
}

/// The independent abstract state machine.
struct BufferOwnershipProtocol;

impl BufferOwnershipProtocol {
    fn ownership_event(kind: &ProtocolEvent) -> Option<&BufferOwnershipEvent> {
        match kind {
            ProtocolEvent::BufferOwnership(event) => Some(event),
            _ => None,
        }
    }
}

impl AbstractProtocol for BufferOwnershipProtocol {
    type State = BufferOwnershipAbstract;
    type Action = BufferOwnershipEvent;

    fn resolve(&self, state: &Self::State, kind: &ProtocolEvent) -> ActionResolution<Self::Action> {
        let event = match Self::ownership_event(kind) {
            Some(event) => event,
            None => {
                return ActionResolution::None("non-ownership event in ownership trace".into());
            }
        };

        let enabled: Result<(), String> = match event {
            BufferOwnershipEvent::Submit {
                operation,
                reads,
                writes,
            } => {
                if state.operations.contains_key(operation) {
                    Err(format!("Submit for existing operation {operation}"))
                } else {
                    state.buffers_available(&set_of(reads), &set_of(writes))
                }
            }
            BufferOwnershipEvent::Detach { operation } => match state.operations.get(operation) {
                Some(op) if op.is_active() && op.handle_attached => Ok(()),
                _ => Err(format!(
                    "Detach for non-active or already-detached operation {operation}"
                )),
            },
            BufferOwnershipEvent::BeginAbort { operation } => match state.operations.get(operation)
            {
                Some(op) if op.state == AbsOpState::Submitted && op.registry_owned => Ok(()),
                _ => Err(format!(
                    "BeginAbort for non-submitted operation {operation}"
                )),
            },
            BufferOwnershipEvent::CompleteSuccess { operation, writes } => {
                match state.operations.get(operation) {
                    Some(op) if op.state == AbsOpState::Submitted && op.registry_owned => {
                        if sorted(&op.writes) != *writes {
                            Err(format!(
                                "CompleteSuccess write set mismatch for {operation}"
                            ))
                        } else {
                            Ok(())
                        }
                    }
                    _ => Err(format!(
                        "CompleteSuccess for non-submitted operation {operation}"
                    )),
                }
            }
            BufferOwnershipEvent::QuiesceAbort { operation, writes } => {
                match state.operations.get(operation) {
                    Some(op) if op.state == AbsOpState::Aborting && op.registry_owned => {
                        if sorted(&op.writes) != *writes {
                            Err(format!("QuiesceAbort write set mismatch for {operation}"))
                        } else {
                            Ok(())
                        }
                    }
                    _ => Err(format!(
                        "QuiesceAbort for non-aborting operation {operation}"
                    )),
                }
            }
            BufferOwnershipEvent::FreeBuffer { buffer } => {
                if state.freed.contains(buffer) {
                    Err(format!("FreeBuffer double-frees allocation {buffer}"))
                } else if state
                    .operations
                    .values()
                    .any(|op| op.is_active() && op.references(*buffer))
                {
                    Err(format!("FreeBuffer on still-leased allocation {buffer}"))
                } else {
                    Ok(())
                }
            }
        };

        match enabled {
            Ok(()) => ActionResolution::Enabled(event.clone()),
            Err(reason) => ActionResolution::None(reason),
        }
    }

    fn apply(&self, state: &mut Self::State, action: &Self::Action) -> Result<(), String> {
        match action {
            BufferOwnershipEvent::Submit {
                operation,
                reads,
                writes,
            } => {
                let reads = set_of(reads);
                let writes = set_of(writes);
                let observed_read_gen = reads.iter().map(|b| (*b, state.generation(*b))).collect();
                let observed_write_gen =
                    writes.iter().map(|b| (*b, state.generation(*b))).collect();
                state.operations.insert(
                    *operation,
                    AbsOperation {
                        reads,
                        writes,
                        state: AbsOpState::Submitted,
                        handle_attached: true,
                        registry_owned: true,
                        outcome: AbsOutcome::None,
                        observed_read_gen,
                        observed_write_gen,
                    },
                );
            }
            BufferOwnershipEvent::Detach { operation } => {
                state
                    .operations
                    .get_mut(operation)
                    .expect("operation")
                    .handle_attached = false;
            }
            BufferOwnershipEvent::BeginAbort { operation } => {
                state
                    .operations
                    .get_mut(operation)
                    .expect("operation")
                    .state = AbsOpState::Aborting;
            }
            BufferOwnershipEvent::CompleteSuccess { operation, .. } => {
                let op = state.operations.get(operation).expect("operation").clone();
                for w in &op.writes {
                    *state.buffer_generation.entry(*w).or_insert(0) += 1;
                }
                let op = state.operations.get_mut(operation).expect("operation");
                op.state = AbsOpState::Terminal;
                op.registry_owned = false;
                op.outcome = AbsOutcome::Ok;
            }
            BufferOwnershipEvent::QuiesceAbort { operation, .. } => {
                let op = state.operations.get(operation).expect("operation").clone();
                for w in &op.writes {
                    *state.buffer_generation.entry(*w).or_insert(0) += 1;
                }
                let op = state.operations.get_mut(operation).expect("operation");
                op.state = AbsOpState::Terminal;
                op.registry_owned = false;
                op.outcome = AbsOutcome::Error;
            }
            BufferOwnershipEvent::FreeBuffer { buffer } => {
                state.freed.insert(*buffer);
            }
        }
        Ok(())
    }

    fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
        let actives: Vec<(&OperationId, &AbsOperation)> = state
            .operations
            .iter()
            .filter(|(_, op)| op.is_active())
            .collect();

        // NoConflictingActiveLeases (generalized to sets).
        for (i, (id_a, a)) in actives.iter().enumerate() {
            for (id_b, b) in actives.iter().skip(i + 1) {
                for w in &a.writes {
                    if b.references(*w) {
                        return Err(format!(
                            "NoConflictingActiveLeases: {id_a} writes {w} also held by {id_b}"
                        ));
                    }
                }
                for w in &b.writes {
                    if a.references(*w) {
                        return Err(format!(
                            "NoConflictingActiveLeases: {id_b} writes {w} also held by {id_a}"
                        ));
                    }
                }
            }
        }

        for (id, op) in &state.operations {
            match op.state {
                AbsOpState::Submitted | AbsOpState::Aborting => {
                    // ActiveIsRegistryOwned + DetachedActiveIsStillOwned.
                    if !op.registry_owned {
                        return Err(format!("ActiveIsRegistryOwned violated for {id}"));
                    }
                    // ActiveGenerationsMatch: neither a reader nor a writer may
                    // observe its buffer generation change while active.
                    for (b, g) in &op.observed_read_gen {
                        if *g != state.generation(*b) {
                            return Err(format!(
                                "ActiveGenerationsMatch (read) violated for {id} on {b}"
                            ));
                        }
                    }
                    for (b, g) in &op.observed_write_gen {
                        if *g != state.generation(*b) {
                            return Err(format!(
                                "ActiveGenerationsMatch (write) violated for {id} on {b}"
                            ));
                        }
                    }
                    // FreedHasNoLease.
                    for b in op.reads.iter().chain(op.writes.iter()) {
                        if state.freed.contains(b) {
                            return Err(format!("FreedHasNoLease violated for {id} on {b}"));
                        }
                    }
                }
                AbsOpState::Terminal => {
                    // TerminalReleased.
                    if op.registry_owned {
                        return Err(format!("TerminalReleased: {id} still registry-owned"));
                    }
                    if op.outcome == AbsOutcome::None {
                        return Err(format!("TerminalReleased: {id} has no outcome"));
                    }
                }
            }
        }
        Ok(())
    }

    fn active_entries(&self, state: &Self::State) -> Vec<String> {
        state
            .operations
            .iter()
            .filter(|(_, op)| op.is_active())
            .map(|(id, op)| format!("{id} in {:?}", op.state))
            .collect()
    }
}

// ─────────────────────────── deterministic PRNG ────────────────────────────

/// A tiny deterministic splitmix64 PRNG so campaigns are reproducible from a
/// fixed seed (`REFINEMENT.md` § "Required Test Campaigns").
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

// ─────────────────────────── shared helpers ────────────────────────────────

fn alloc(id: u64) -> PhysicalAllocationId {
    PhysicalAllocationId::new(id)
}

fn traced_registry(collector: &Arc<BoundedTraceCollector>) -> OwnershipRegistry {
    OwnershipRegistry::with_sink(ProtocolSourceId::new(7), 0, collector.clone())
}

/// Replays a captured trace against a fresh independent abstract state,
/// asserting acceptance under a clean end.
fn assert_conformant(collector: &BoundedTraceCollector) {
    let checker = ReplayChecker::new(BufferOwnershipProtocol);
    let report = checker
        .run(
            BufferOwnershipAbstract::default(),
            &collector.snapshot(),
            TraceEnd::Clean,
        )
        .unwrap_or_else(|failure| {
            panic!("independent replay checker rejected a valid trace: {failure:?}")
        });
    assert!(report.events_checked > 0, "expected non-empty trace");
}

// ─────────────────────────── campaigns ─────────────────────────────────────

/// Acquire → transfer (detach) → release. The registry keeps ownership across
/// the user handle's detach, then proves terminal success before releasing.
#[test]
fn campaign_acquire_transfer_release() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);

    let lease = registry
        .submit(&CommLeaseSet {
            reads: vec![alloc(1)],
            writes: vec![alloc(2)],
        })
        .unwrap();
    let op = lease.operation();
    // Transfer: detach the user handle; ownership stays with the registry.
    registry.detach(op);
    assert_eq!(registry.snapshot().unwrap().detached_active, 1);
    // Release: prove terminal success, releasing both leases.
    registry.complete_success(op).unwrap();
    drop(lease); // detach on drop is now a no-op (already terminal)

    assert_eq!(registry.snapshot().unwrap().active, 0);
    assert_conformant(&collector);
}

/// Abort path: submit → begin_abort → quiesce_abort. Leases stay owned through
/// aborting and are released only at quiescence.
#[test]
fn campaign_abort_quiesce() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);

    let lease = registry
        .submit(&CommLeaseSet::write_only([alloc(5)]))
        .unwrap();
    let op = lease.operation();
    registry.begin_abort(op).unwrap();
    assert_eq!(registry.snapshot().unwrap().aborting, 1);
    // A conflicting submit is rejected while the op is still aborting (owned).
    assert!(
        registry
            .submit(&CommLeaseSet::read_only([alloc(5)]))
            .is_err()
    );
    registry.quiesce_abort(op).unwrap();
    drop(lease);

    assert_eq!(registry.snapshot().unwrap().active, 0);
    assert_conformant(&collector);
}

/// Read/read aliasing is legal: two concurrent readers of the same allocation.
#[test]
fn campaign_read_read_alias() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);

    let r1 = registry
        .submit(&CommLeaseSet::read_only([alloc(9)]))
        .unwrap();
    let r2 = registry
        .submit(&CommLeaseSet::read_only([alloc(9)]))
        .unwrap();
    assert_eq!(registry.snapshot().unwrap().active, 2);
    // A writer conflicts with both active readers.
    assert!(
        registry
            .submit(&CommLeaseSet::write_only([alloc(9)]))
            .is_err()
    );
    registry.complete_success(r1.operation()).unwrap();
    registry.complete_success(r2.operation()).unwrap();
    drop((r1, r2));

    assert_conformant(&collector);
}

/// Read/write conflict then serialization: a writer must wait for the reader to
/// retire, after which it acquires the exclusive lease.
#[test]
fn campaign_read_write_conflict_then_serialize() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);

    let reader = registry
        .submit(&CommLeaseSet::read_only([alloc(3)]))
        .unwrap();
    // Writer conflicts while the reader is active.
    assert!(
        registry
            .submit(&CommLeaseSet::write_only([alloc(3)]))
            .is_err()
    );
    registry.complete_success(reader.operation()).unwrap();
    drop(reader);
    // Now the writer acquires it exclusively.
    let writer = registry
        .submit(&CommLeaseSet::write_only([alloc(3)]))
        .unwrap();
    registry.complete_success(writer.operation()).unwrap();
    drop(writer);

    assert_conformant(&collector);
}

/// In-place operation: the same allocation in both read and write sets is
/// exclusive, so any other reader/writer conflicts.
#[test]
fn campaign_in_place_is_exclusive() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);

    let inplace = registry
        .submit(&CommLeaseSet {
            reads: vec![alloc(4)],
            writes: vec![alloc(4)],
        })
        .unwrap();
    assert!(
        registry
            .submit(&CommLeaseSet::read_only([alloc(4)]))
            .is_err()
    );
    assert!(
        registry
            .submit(&CommLeaseSet::write_only([alloc(4)]))
            .is_err()
    );
    registry.complete_success(inplace.operation()).unwrap();
    drop(inplace);

    assert_conformant(&collector);
}

/// Allocator reuse boundary: free after release, double-free rejected, and a
/// submit referencing a freed allocation is rejected.
#[test]
fn campaign_allocator_free_boundaries() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);

    let lease = registry
        .submit(&CommLeaseSet::write_only([alloc(11)]))
        .unwrap();
    // Cannot free while leased.
    assert!(registry.free_buffer(alloc(11)).is_err());
    registry.complete_success(lease.operation()).unwrap();
    drop(lease);
    // Now free succeeds; double-free rejected; re-lease rejected.
    registry.free_buffer(alloc(11)).unwrap();
    assert!(registry.free_buffer(alloc(11)).is_err());
    assert!(
        registry
            .submit(&CommLeaseSet::read_only([alloc(11)]))
            .is_err()
    );
    // Freeing a never-leased allocation is fine.
    registry.free_buffer(alloc(999)).unwrap();

    assert_conformant(&collector);
}

/// Allocator reuse generation: a written allocation advances its generation at
/// terminal success; a later operation on the same allocation observes the new
/// generation without violating any active operation's `ActiveGenerationsMatch`.
#[test]
fn campaign_generation_advances_on_write() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);

    for _ in 0..3 {
        let lease = registry
            .submit(&CommLeaseSet::write_only([alloc(21)]))
            .unwrap();
        registry.complete_success(lease.operation()).unwrap();
        drop(lease);
    }
    assert_conformant(&collector);
}

/// Cross-rank operations driven through the real `InProcessCommunicator`: rank
/// pairs exchange buffers, and every send/recv lease lifecycle in the shared
/// registry conforms.
#[test]
fn campaign_cross_rank_send_recv() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);
    let world = InProcessCommunicator::world_with_registry(4, registry.clone());

    // rank i sends to rank (i+1)%4; the receiver reads it back.
    block_on(async {
        for i in 0..4usize {
            let sender = &world[i];
            let dest = (i + 1) % 4;
            let payload = sender.allocate_from(vec![i as u8; 8]);
            let instance = CommInstanceId::new(1, i as u32);
            sender
                .send(
                    instance,
                    &payload,
                    8,
                    DType::U8,
                    onnx_runtime_comm::RankId(dest as u32),
                )
                .await
                .unwrap();

            let receiver = &world[dest];
            let mut sink = receiver.allocate_buffer(8);
            receiver
                .recv(
                    instance,
                    &mut sink,
                    8,
                    DType::U8,
                    onnx_runtime_comm::RankId(i as u32),
                )
                .await
                .unwrap();
            assert_eq!(sink.as_slice(), &[i as u8; 8]);
        }
    });

    // Every send/recv completed synchronously: no active operations remain.
    assert_eq!(registry.snapshot().unwrap().active, 0);
    assert_conformant(&collector);
}

// ─────────────── seeded randomized campaign (determinism gate) ──────────────

/// Runs a seeded random campaign of registry actions and returns the recorded
/// trace so two runs can be compared for reproducibility.
fn run_seeded_campaign(seed: u64) -> Vec<ProtocolTraceEvent> {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);
    let mut rng = SplitMix64::new(seed);

    // A small pool of allocations to force aliasing, conflicts, and reuse.
    let pool: Vec<PhysicalAllocationId> = (1..=6).map(alloc).collect();
    // Live operations we can advance: (op, is_aborting).
    let mut live: Vec<(OperationId, bool)> = Vec::new();

    for _ in 0..300 {
        match rng.below(6) {
            0 | 1 => {
                // Submit with a random small read/write set (best-effort; a
                // conflict returns Err and emits no event, keeping the trace
                // lossless and deterministic).
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                let n = 1 + rng.below(2) as usize;
                for _ in 0..n {
                    let a = pool[rng.below(pool.len() as u64) as usize];
                    if rng.below(2) == 0 {
                        reads.push(a);
                    } else {
                        writes.push(a);
                    }
                }
                if let Ok(lease) = registry.submit(&CommLeaseSet { reads, writes }) {
                    let op = lease.operation();
                    live.push((op, false));
                    lease.detach(); // detach immediately; registry keeps ownership
                }
            }
            2 => {
                if !live.is_empty() {
                    let idx = rng.below(live.len() as u64) as usize;
                    let (op, aborting) = live[idx];
                    if !aborting && registry.begin_abort(op).is_ok() {
                        live[idx].1 = true;
                    }
                }
            }
            3 => {
                if !live.is_empty() {
                    let idx = rng.below(live.len() as u64) as usize;
                    let (op, aborting) = live[idx];
                    let done = if aborting {
                        registry.quiesce_abort(op).is_ok()
                    } else {
                        registry.complete_success(op).is_ok()
                    };
                    if done {
                        live.swap_remove(idx);
                    }
                }
            }
            _ => {
                // Free a random allocation if no active op references it.
                let a = pool[rng.below(pool.len() as u64) as usize];
                let _ = registry.free_buffer(a);
            }
        }
        // Snapshot invariant gate on every step.
        registry.snapshot().expect("snapshot invariant gate");
    }

    // Wind down to a clean end: retire every live operation.
    for (op, aborting) in live.drain(..) {
        if aborting {
            registry.quiesce_abort(op).expect("quiesce");
        } else {
            registry.complete_success(op).expect("complete");
        }
    }
    registry.snapshot().expect("final snapshot invariant gate");

    // The independent checker must accept the whole trace at a clean end.
    let checker = ReplayChecker::new(BufferOwnershipProtocol);
    checker
        .run(
            BufferOwnershipAbstract::default(),
            &collector.snapshot(),
            TraceEnd::Clean,
        )
        .expect("seeded campaign trace must be conformant");

    collector.snapshot().events
}

#[test]
fn campaign_seeded_is_reproducible() {
    let run_a = run_seeded_campaign(0xB0FF_1CE5);
    let run_b = run_seeded_campaign(0xB0FF_1CE5);
    assert_eq!(run_a, run_b, "same seed must produce identical traces");
    assert!(!run_a.is_empty());

    let run_c = run_seeded_campaign(0x1357_9BDF);
    let run_c2 = run_seeded_campaign(0x1357_9BDF);
    assert_eq!(run_c, run_c2, "second seed must also be reproducible");
}

// ─────────────────── self-tests: the checker rejects bad traces ─────────────

fn valid_small_trace() -> Vec<ProtocolTraceEvent> {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);
    let lease = registry
        .submit(&CommLeaseSet::write_only([alloc(1)]))
        .unwrap();
    registry.complete_success(lease.operation()).unwrap();
    drop(lease);
    collector.snapshot().events
}

fn snapshot_of(events: Vec<ProtocolTraceEvent>) -> TraceSnapshot {
    TraceSnapshot {
        events,
        integrity: Ok(()),
    }
}

fn run_checker(events: Vec<ProtocolTraceEvent>, end: TraceEnd) -> Result<(), ConformanceFailure> {
    let checker = ReplayChecker::new(BufferOwnershipProtocol);
    checker
        .run(
            BufferOwnershipAbstract::default(),
            &snapshot_of(events),
            end,
        )
        .map(|_| ())
}

#[test]
fn checker_rejects_unknown_contract_revision() {
    let mut events = valid_small_trace();
    events[0].contract_revision = 999;
    let failure = run_checker(events, TraceEnd::Clean).unwrap_err();
    assert!(matches!(
        failure.reason,
        FailureReason::UnknownContractRevision { .. }
    ));
    assert_eq!(failure.prefix_len, 1);
}

#[test]
fn checker_rejects_duplicate_source_sequence() {
    let mut events = valid_small_trace();
    let dup = events[0].source_sequence;
    events[1].source_sequence = dup;
    let failure = run_checker(events, TraceEnd::Clean).unwrap_err();
    assert!(matches!(
        failure.reason,
        FailureReason::DuplicateSourceSequence { .. }
    ));
}

#[test]
fn checker_rejects_reordered_event() {
    // Swap Submit and CompleteSuccess: CompleteSuccess now precedes Submit.
    let mut events = valid_small_trace();
    events.swap(0, 1);
    for (i, e) in events.iter_mut().enumerate() {
        e.source_sequence = i as u64;
    }
    let failure = run_checker(events, TraceEnd::Clean).unwrap_err();
    assert!(matches!(
        failure.reason,
        FailureReason::NoEnabledAction { .. }
    ));
    assert_eq!(
        failure.prefix_len, 1,
        "fails at the smallest offending prefix"
    );
}

#[test]
fn checker_rejects_lossy_trace() {
    let events = valid_small_trace();
    let lossy = TraceSnapshot {
        events,
        integrity: Err(TraceIntegrityError::BufferOverflow { at_index: 1 }),
    };
    let checker = ReplayChecker::new(BufferOwnershipProtocol);
    let failure = checker
        .run(BufferOwnershipAbstract::default(), &lossy, TraceEnd::Clean)
        .unwrap_err();
    assert!(matches!(failure.reason, FailureReason::TraceIntegrity(_)));
}

#[test]
fn checker_rejects_leftover_active_operation() {
    // A Submit with no terminal resolution leaves an active operation.
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let registry = traced_registry(&collector);
    let lease = registry
        .submit(&CommLeaseSet::write_only([alloc(1)]))
        .unwrap();
    std::mem::forget(lease); // no Detach/terminal is emitted
    let events = collector.snapshot().events;

    let failure = run_checker(events.clone(), TraceEnd::Clean).unwrap_err();
    assert!(matches!(
        failure.reason,
        FailureReason::LeftoverActive { .. }
    ));
    // A declared crash boundary tolerates it.
    assert!(run_checker(events, TraceEnd::CrashBoundary).is_ok());
}

#[test]
fn checker_rejects_free_of_leased_allocation() {
    // Hand-craft a trace: Submit(writes=[1]) then FreeBuffer(1) while active.
    let submit = ProtocolTraceEvent {
        contract_revision: onnx_runtime_protocol_trace::CONTRACT_REVISION,
        topology_epoch: 0,
        source: ProtocolSourceId::new(7),
        source_sequence: 0,
        kind: ProtocolEvent::BufferOwnership(BufferOwnershipEvent::Submit {
            operation: OperationId::new(1),
            reads: vec![],
            writes: vec![alloc(1)],
        }),
    };
    let free = ProtocolTraceEvent {
        contract_revision: onnx_runtime_protocol_trace::CONTRACT_REVISION,
        topology_epoch: 0,
        source: ProtocolSourceId::new(7),
        source_sequence: 1,
        kind: ProtocolEvent::BufferOwnership(BufferOwnershipEvent::FreeBuffer { buffer: alloc(1) }),
    };
    let failure = run_checker(vec![submit, free], TraceEnd::CrashBoundary).unwrap_err();
    assert!(matches!(
        failure.reason,
        FailureReason::NoEnabledAction { .. }
    ));
    assert_eq!(failure.prefix_len, 2);
}
