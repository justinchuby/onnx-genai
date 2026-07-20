//! Independent replay checker and seeded campaigns for
//! `specs/tla/CollectiveOrdering.tla`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use onnx_runtime_comm::{
    CommError, CommInstanceId, Communicator, DType, ExecutionId, GroupId, GroupSpec,
    InProcessCommunicator, RankId, ReduceOp, block_on,
};
use onnx_runtime_protocol_trace::checker::{
    AbstractProtocol, ActionResolution, BoundedTraceCollector, FailureReason, ReplayChecker,
    TraceEnd, TraceIntegrityError, TraceSnapshot,
};
use onnx_runtime_protocol_trace::{
    CollectiveDecision, CollectiveKind, CollectiveOrderingEvent, ProtocolEvent, ProtocolSourceId,
    ProtocolTraceEvent,
};

#[derive(Clone, Debug, PartialEq, Eq)]
enum ObservedSlot {
    Submitted {
        instance: (u64, u32),
        collective: CollectiveKind,
        signature: u64,
    },
    Skipped {
        instance: (u64, u32),
    },
}

type TransportEntry = (u64, u32, CollectiveKind, u64);

#[derive(Default)]
struct AbsGroup {
    members: Vec<u32>,
    hash: u64,
    cursors: BTreeMap<u32, Vec<ObservedSlot>>,
    transport: BTreeMap<u32, Vec<TransportEntry>>,
    submitted: BTreeMap<u32, usize>,
    completed: BTreeMap<u32, usize>,
}

#[derive(Default)]
struct CollectiveAbstract {
    decisions: BTreeMap<u64, CollectiveDecision>,
    decided_prefix: u64,
    groups: BTreeMap<u32, AbsGroup>,
    active: BTreeSet<(u32, u32, u64, u32)>,
    aborted: bool,
    log_at_abort: BTreeMap<(u32, u32), Vec<TransportEntry>>,
}

struct CollectiveProtocol;

fn membership_hash(members: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in (members.len() as u64)
        .to_le_bytes()
        .into_iter()
        .chain(members.iter().flat_map(|rank| rank.to_le_bytes()))
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn is_prefix<T: PartialEq>(left: &[T], right: &[T]) -> bool {
    left.len() <= right.len() && left == &right[..left.len()]
}

fn event_group(
    state: &CollectiveAbstract,
    group: u32,
    members: &[u32],
    hash: u64,
    rank: u32,
) -> Result<(), String> {
    if members.is_empty() || !members.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err("group membership is not sorted, unique, and non-empty".into());
    }
    if membership_hash(members) != hash {
        return Err("ordered-membership hash mismatch".into());
    }
    if members.binary_search(&rank).is_err() {
        return Err(format!("non-member rank {rank} touched group {group}"));
    }
    if let Some(existing) = state.groups.get(&group)
        && (existing.members != members || existing.hash != hash)
    {
        return Err(format!("group {group} identity changed"));
    }
    Ok(())
}

impl AbstractProtocol for CollectiveProtocol {
    type State = CollectiveAbstract;
    type Action = CollectiveOrderingEvent;

    fn resolve(
        &self,
        state: &Self::State,
        event: &ProtocolEvent,
    ) -> ActionResolution<Self::Action> {
        let ProtocolEvent::CollectiveOrdering(event) = event else {
            return ActionResolution::None("non-collective event in ordering trace".into());
        };
        let enabled = match event {
            CollectiveOrderingEvent::Decide {
                execution,
                decision: _,
            } => {
                if state.aborted {
                    Err("decision after abort".into())
                } else if *execution != state.decided_prefix + 1 {
                    Err("decision does not extend the monotonic prefix".into())
                } else {
                    Ok(())
                }
            }
            CollectiveOrderingEvent::Submit {
                group,
                members,
                membership_hash,
                rank,
                execution,
                sequence,
                collective,
                signature,
            } => event_group(state, *group, members, *membership_hash, *rank).and_then(|_| {
                if state.aborted {
                    return Err("submission after abort".into());
                }
                if state.decisions.get(execution) != Some(&CollectiveDecision::Admitted) {
                    return Err("submission belongs to a non-admitted execution".into());
                }
                if state
                    .active
                    .contains(&(*group, *rank, *execution, *sequence))
                {
                    return Err("duplicate active submission".into());
                }
                let candidate = ObservedSlot::Submitted {
                    instance: (*execution, *sequence),
                    collective: *collective,
                    signature: *signature,
                };
                if let Some(group_state) = state.groups.get(group) {
                    let cursor = group_state
                        .cursors
                        .get(rank)
                        .map(Vec::len)
                        .unwrap_or_default();
                    for (other_rank, slots) in &group_state.cursors {
                        if *other_rank != *rank
                            && cursor < slots.len()
                            && slots[cursor] != candidate
                        {
                            return Err("cross-rank transport-order divergence".into());
                        }
                    }
                }
                Ok(())
            }),
            CollectiveOrderingEvent::ObserveSkip {
                group,
                members,
                membership_hash,
                rank,
                execution,
                sequence,
            } => event_group(state, *group, members, *membership_hash, *rank).and_then(|_| {
                if state.aborted {
                    return Err("skip observation after abort".into());
                }
                if state.decisions.get(execution) != Some(&CollectiveDecision::Skipped) {
                    return Err("skip observation belongs to non-skipped execution".into());
                }
                let candidate = ObservedSlot::Skipped {
                    instance: (*execution, *sequence),
                };
                if let Some(group_state) = state.groups.get(group) {
                    let cursor = group_state
                        .cursors
                        .get(rank)
                        .map(Vec::len)
                        .unwrap_or_default();
                    for (other_rank, slots) in &group_state.cursors {
                        if *other_rank != *rank
                            && cursor < slots.len()
                            && slots[cursor] != candidate
                        {
                            return Err("cross-rank skipped-slot divergence".into());
                        }
                    }
                }
                Ok(())
            }),
            CollectiveOrderingEvent::CompleteLocal {
                group,
                members,
                membership_hash,
                rank,
                execution,
                sequence,
                ..
            } => event_group(state, *group, members, *membership_hash, *rank).and_then(|_| {
                if state
                    .active
                    .contains(&(*group, *rank, *execution, *sequence))
                {
                    Ok(())
                } else {
                    Err("completion for a non-active local submission".into())
                }
            }),
            CollectiveOrderingEvent::Abort => {
                if state.aborted {
                    Err("duplicate abort".into())
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
            CollectiveOrderingEvent::Decide {
                execution,
                decision,
            } => {
                state.decisions.insert(*execution, decision.clone());
                state.decided_prefix = *execution;
            }
            CollectiveOrderingEvent::Submit {
                group,
                members,
                membership_hash,
                rank,
                execution,
                sequence,
                collective,
                signature,
            } => {
                let group_state = state.groups.entry(*group).or_insert_with(|| AbsGroup {
                    members: members.clone(),
                    hash: *membership_hash,
                    ..AbsGroup::default()
                });
                group_state
                    .cursors
                    .entry(*rank)
                    .or_default()
                    .push(ObservedSlot::Submitted {
                        instance: (*execution, *sequence),
                        collective: *collective,
                        signature: *signature,
                    });
                group_state.transport.entry(*rank).or_default().push((
                    *execution,
                    *sequence,
                    *collective,
                    *signature,
                ));
                *group_state.submitted.entry(*rank).or_default() += 1;
                state.active.insert((*group, *rank, *execution, *sequence));
            }
            CollectiveOrderingEvent::ObserveSkip {
                group,
                members,
                membership_hash,
                rank,
                execution,
                sequence,
            } => {
                state
                    .groups
                    .entry(*group)
                    .or_insert_with(|| AbsGroup {
                        members: members.clone(),
                        hash: *membership_hash,
                        ..AbsGroup::default()
                    })
                    .cursors
                    .entry(*rank)
                    .or_default()
                    .push(ObservedSlot::Skipped {
                        instance: (*execution, *sequence),
                    });
            }
            CollectiveOrderingEvent::CompleteLocal {
                group,
                rank,
                execution,
                sequence,
                ..
            } => {
                state.active.remove(&(*group, *rank, *execution, *sequence));
                *state
                    .groups
                    .get_mut(group)
                    .expect("validated group")
                    .completed
                    .entry(*rank)
                    .or_default() += 1;
            }
            CollectiveOrderingEvent::Abort => {
                state.aborted = true;
                for (group, group_state) in &state.groups {
                    for (rank, log) in &group_state.transport {
                        state.log_at_abort.insert((*group, *rank), log.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
        for execution in 1..=state.decided_prefix {
            if !state.decisions.contains_key(&execution) {
                return Err("decision prefix contains a hole".into());
            }
        }
        for group in state.groups.values() {
            for rank in &group.members {
                let cursor = group.cursors.get(rank).map(Vec::as_slice).unwrap_or(&[]);
                let transport = group.transport.get(rank).map(Vec::as_slice).unwrap_or(&[]);
                if !transport
                    .windows(2)
                    .all(|pair| (pair[0].0, pair[0].1) < (pair[1].0, pair[1].1))
                {
                    return Err("transport log duplicated or reordered a slot".into());
                }
                let submitted = group.submitted.get(rank).copied().unwrap_or_default();
                let completed = group.completed.get(rank).copied().unwrap_or_default();
                if completed > submitted || transport.len() > cursor.len() {
                    return Err("local completion/cursor exceeded submission".into());
                }
            }
            for (left_index, left_rank) in group.members.iter().enumerate() {
                for right_rank in group.members.iter().skip(left_index + 1) {
                    let left = group
                        .cursors
                        .get(left_rank)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]);
                    let right = group
                        .cursors
                        .get(right_rank)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]);
                    if !is_prefix(left, right) && !is_prefix(right, left) {
                        return Err("group rank logs are not prefix-compatible".into());
                    }
                }
            }
        }
        if state.aborted {
            for ((group, rank), at_abort) in &state.log_at_abort {
                let current = state
                    .groups
                    .get(group)
                    .and_then(|group| group.transport.get(rank))
                    .expect("abort snapshot group exists");
                if current != at_abort {
                    return Err("abort did not freeze transport submission".into());
                }
            }
        }
        Ok(())
    }

    fn active_entries(&self, state: &Self::State) -> Vec<String> {
        state
            .active
            .iter()
            .map(|entry| format!("active collective {entry:?}"))
            .collect()
    }
}

fn collective_snapshot(collector: &BoundedTraceCollector) -> TraceSnapshot {
    let mut snapshot = collector.snapshot();
    snapshot
        .events
        .retain(|event| matches!(event.kind, ProtocolEvent::CollectiveOrdering(_)));
    snapshot
}

fn run_checker(snapshot: &TraceSnapshot) {
    ReplayChecker::new(CollectiveProtocol)
        .run(CollectiveAbstract::default(), snapshot, TraceEnd::Clean)
        .unwrap();
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn seeded_yields(seed: u64, rank: usize, sequence: u32) {
    let mut state = seed ^ ((rank as u64) << 32) ^ u64::from(sequence);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    for _ in 0..(state % 5) {
        std::thread::yield_now();
    }
}

fn concurrent_all_reduce_trace() -> TraceSnapshot {
    const SEED: u64 = 0x401c_c011_ec71;
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let world =
        InProcessCommunicator::world_traced(3, ProtocolSourceId::new(100), 7, collector.clone());
    let threads: Vec<_> = world
        .into_iter()
        .enumerate()
        .map(|(rank, comm)| {
            std::thread::spawn(move || {
                for sequence in 0..2 {
                    seeded_yields(SEED, rank, sequence);
                    let mut buffer =
                        comm.allocate_from(f32_bytes(&[rank as f32 + sequence as f32 + 1.0]));
                    let handle = block_on(comm.all_reduce(
                        CommInstanceId::new(1, sequence),
                        &mut buffer,
                        1,
                        DType::F32,
                        ReduceOp::Sum,
                    ))
                    .unwrap();
                    block_on(handle).unwrap();
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    collective_snapshot(&collector)
}

#[test]
fn campaign_concurrent_multi_rank_collectives() {
    let snapshot = concurrent_all_reduce_trace();
    run_checker(&snapshot);
    assert_eq!(snapshot.events.len(), 1 + 3 * 2 * 2);
}

#[test]
fn campaign_mixed_overlapping_groups() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let world =
        InProcessCommunicator::world_traced(3, ProtocolSourceId::new(200), 8, collector.clone());
    let group_a = InProcessCommunicator::create_group(
        &world,
        GroupSpec {
            id: GroupId(1),
            ranks: vec![RankId(0), RankId(1)],
        },
    )
    .unwrap();
    let group_b = InProcessCommunicator::create_group(
        &world,
        GroupSpec {
            id: GroupId(2),
            ranks: vec![RankId(0), RankId(2)],
        },
    )
    .unwrap();
    let mut jobs = Vec::new();
    for comm in group_a {
        jobs.push(std::thread::spawn(move || {
            let mut buffer = comm.allocate_from(f32_bytes(&[comm.rank().0 as f32 + 1.0]));
            block_on(comm.all_reduce(
                CommInstanceId::new(1, 0),
                &mut buffer,
                1,
                DType::F32,
                ReduceOp::Sum,
            ))
            .unwrap();
        }));
    }
    for comm in group_b {
        jobs.push(std::thread::spawn(move || {
            let mut buffer = comm.allocate_from(f32_bytes(&[9.0]));
            block_on(comm.broadcast(
                CommInstanceId::new(1, 0),
                &mut buffer,
                1,
                DType::F32,
                RankId(0),
            ))
            .unwrap();
        }));
    }
    for job in jobs {
        job.join().unwrap();
    }
    run_checker(&collective_snapshot(&collector));
}

#[test]
fn campaign_abort_mid_flight_freezes_and_quiesces() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let world =
        InProcessCommunicator::world_traced(2, ProtocolSourceId::new(300), 9, collector.clone());
    let rank0 = world[0].clone();
    let waiter = std::thread::spawn(move || {
        let mut buffer = rank0.allocate_from(f32_bytes(&[1.0]));
        block_on(rank0.all_reduce(
            CommInstanceId::new(1, 0),
            &mut buffer,
            1,
            DType::F32,
            ReduceOp::Sum,
        ))
    });
    while collector.len() < 3 {
        std::thread::yield_now();
    }
    block_on(world[1].abort(CommError::Aborted("campaign".into()))).unwrap();
    assert!(matches!(waiter.join().unwrap(), Err(CommError::Aborted(_))));
    run_checker(&collective_snapshot(&collector));
}

#[test]
fn campaign_coordinator_skip_advances_without_transport() {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let world =
        InProcessCommunicator::world_traced(2, ProtocolSourceId::new(400), 10, collector.clone());
    world[0].skip_execution(ExecutionId(1)).unwrap();
    world[1].observe_skipped(CommInstanceId::new(1, 0)).unwrap();
    world[0].observe_skipped(CommInstanceId::new(1, 0)).unwrap();
    run_checker(&collective_snapshot(&collector));
}

#[test]
fn checker_rejects_unknown_revision() {
    let mut snapshot = concurrent_all_reduce_trace();
    snapshot.events[0].contract_revision += 1;
    let error = ReplayChecker::new(CollectiveProtocol)
        .run(CollectiveAbstract::default(), &snapshot, TraceEnd::Clean)
        .unwrap_err();
    assert!(matches!(
        error.reason,
        FailureReason::UnknownContractRevision { .. }
    ));
}

#[test]
fn checker_rejects_duplicate_source_sequence() {
    let mut snapshot = concurrent_all_reduce_trace();
    let source = snapshot.events[1].source;
    let previous = snapshot.events[1].source_sequence;
    let same_source = snapshot
        .events
        .iter_mut()
        .skip(2)
        .find(|event| event.source == source)
        .unwrap();
    same_source.source_sequence = previous;
    let error = ReplayChecker::new(CollectiveProtocol)
        .run(CollectiveAbstract::default(), &snapshot, TraceEnd::Clean)
        .unwrap_err();
    assert!(matches!(
        error.reason,
        FailureReason::DuplicateSourceSequence { .. }
    ));
}

#[test]
fn checker_rejects_reordered_rank_events() {
    let mut snapshot = concurrent_all_reduce_trace();
    let source = snapshot.events[1].source;
    let indices: Vec<usize> = snapshot
        .events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| (event.source == source).then_some(index))
        .collect();
    snapshot.events.swap(indices[0], indices[1]);
    let error = ReplayChecker::new(CollectiveProtocol)
        .run(CollectiveAbstract::default(), &snapshot, TraceEnd::Clean)
        .unwrap_err();
    assert!(matches!(
        error.reason,
        FailureReason::NonMonotonicSourceSequence { .. } | FailureReason::NoEnabledAction { .. }
    ));
}

#[test]
fn checker_rejects_lossy_trace() {
    let snapshot = TraceSnapshot {
        events: Vec::new(),
        integrity: Err(TraceIntegrityError::BufferOverflow { at_index: 2 }),
    };
    let error = ReplayChecker::new(CollectiveProtocol)
        .run(CollectiveAbstract::default(), &snapshot, TraceEnd::Clean)
        .unwrap_err();
    assert!(matches!(error.reason, FailureReason::TraceIntegrity(_)));
}

#[test]
fn checker_rejects_leftover_active_collective() {
    let mut snapshot = concurrent_all_reduce_trace();
    snapshot.events.retain(|event| {
        !matches!(
            event.kind,
            ProtocolEvent::CollectiveOrdering(CollectiveOrderingEvent::CompleteLocal { .. })
        )
    });
    let error = ReplayChecker::new(CollectiveProtocol)
        .run(CollectiveAbstract::default(), &snapshot, TraceEnd::Clean)
        .unwrap_err();
    assert!(matches!(error.reason, FailureReason::LeftoverActive { .. }));
}

#[test]
fn checker_rejects_cross_rank_divergence() {
    let mut snapshot = concurrent_all_reduce_trace();
    let submit = snapshot
        .events
        .iter_mut()
        .filter_map(|event| match &mut event.kind {
            ProtocolEvent::CollectiveOrdering(CollectiveOrderingEvent::Submit {
                rank,
                collective,
                ..
            }) if *rank == 1 => Some(collective),
            _ => None,
        })
        .next()
        .unwrap();
    *submit = CollectiveKind::Broadcast;
    let error = ReplayChecker::new(CollectiveProtocol)
        .run(CollectiveAbstract::default(), &snapshot, TraceEnd::Clean)
        .unwrap_err();
    assert!(matches!(
        error.reason,
        FailureReason::NoEnabledAction { .. } | FailureReason::InvariantViolated { .. }
    ));
}

fn deterministic_trace_bytes() -> Vec<u8> {
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let world =
        InProcessCommunicator::world_traced(2, ProtocolSourceId::new(500), 11, collector.clone());
    world[0].skip_execution(ExecutionId(1)).unwrap();
    world[0].observe_skipped(CommInstanceId::new(1, 0)).unwrap();
    world[1].observe_skipped(CommInstanceId::new(1, 0)).unwrap();
    block_on(world[0].abort(CommError::Aborted("determinism".into()))).unwrap();
    let snapshot = collective_snapshot(&collector);
    run_checker(&snapshot);
    let mut bytes = Vec::new();
    for event in snapshot.events {
        bytes.extend_from_slice(&event.contract_revision.to_le_bytes());
        bytes.extend_from_slice(&event.topology_epoch.to_le_bytes());
        bytes.extend_from_slice(&event.source.get().to_le_bytes());
        bytes.extend_from_slice(&event.source_sequence.to_le_bytes());
        bytes.extend_from_slice(format!("{:?}", event.kind).as_bytes());
        bytes.push(0);
    }
    bytes
}

#[test]
fn campaign_seeded_schedule_is_byte_identical() {
    assert_eq!(deterministic_trace_bytes(), deterministic_trace_bytes());
}

#[test]
fn checker_allows_rank_local_completion_skew() {
    let members = vec![0, 1];
    let hash = membership_hash(&members);
    let envelope = |source, sequence, kind| ProtocolTraceEvent {
        contract_revision: onnx_runtime_protocol_trace::CONTRACT_REVISION,
        topology_epoch: 1,
        source: ProtocolSourceId::new(source),
        source_sequence: sequence,
        kind: ProtocolEvent::CollectiveOrdering(kind),
    };
    let submit = |rank| CollectiveOrderingEvent::Submit {
        group: 0,
        members: members.clone(),
        membership_hash: hash,
        rank,
        execution: 1,
        sequence: 0,
        collective: CollectiveKind::AllReduce,
        signature: 7,
    };
    let complete = |rank| CollectiveOrderingEvent::CompleteLocal {
        group: 0,
        members: members.clone(),
        membership_hash: hash,
        rank,
        execution: 1,
        sequence: 0,
        success: true,
    };
    let snapshot = TraceSnapshot {
        events: vec![
            envelope(
                1,
                0,
                CollectiveOrderingEvent::Decide {
                    execution: 1,
                    decision: CollectiveDecision::Admitted,
                },
            ),
            envelope(2, 0, submit(0)),
            envelope(3, 0, submit(1)),
            envelope(3, 1, complete(1)),
            envelope(2, 1, complete(0)),
        ],
        integrity: Ok(()),
    };
    run_checker(&snapshot);
}

#[test]
fn checker_rejects_submission_after_abort() {
    let snapshot = TraceSnapshot {
        events: vec![
            ProtocolTraceEvent {
                contract_revision: onnx_runtime_protocol_trace::CONTRACT_REVISION,
                topology_epoch: 1,
                source: ProtocolSourceId::new(1),
                source_sequence: 0,
                kind: ProtocolEvent::CollectiveOrdering(CollectiveOrderingEvent::Abort),
            },
            ProtocolTraceEvent {
                contract_revision: onnx_runtime_protocol_trace::CONTRACT_REVISION,
                topology_epoch: 1,
                source: ProtocolSourceId::new(2),
                source_sequence: 0,
                kind: ProtocolEvent::CollectiveOrdering(CollectiveOrderingEvent::Submit {
                    group: 0,
                    members: vec![0, 1],
                    membership_hash: membership_hash(&[0, 1]),
                    rank: 0,
                    execution: 1,
                    sequence: 0,
                    collective: CollectiveKind::AllReduce,
                    signature: 1,
                }),
            },
        ],
        integrity: Ok(()),
    };
    let error = ReplayChecker::new(CollectiveProtocol)
        .run(CollectiveAbstract::default(), &snapshot, TraceEnd::Clean)
        .unwrap_err();
    assert!(matches!(
        error.reason,
        FailureReason::NoEnabledAction { .. }
    ));
}
