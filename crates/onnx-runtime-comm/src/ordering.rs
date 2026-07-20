//! Per-group collective submission sequencer refined against
//! `specs/tla/CollectiveOrdering.tla`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use onnx_runtime_protocol_trace::{
    CONTRACT_REVISION, CollectiveDecision, CollectiveKind, CollectiveOrderingEvent, NullTraceSink,
    ProtocolEvent, ProtocolSourceId, ProtocolTraceEvent, ProtocolTraceSink,
};

use crate::{CommError, CommInstanceId, CommResult, ExecutionId, GroupId, RankId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotDisposition {
    Submitted,
    Skipped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    instance: CommInstanceId,
    disposition: SlotDisposition,
    collective: Option<CollectiveKind>,
    signature: u64,
}

#[derive(Debug)]
struct GroupState {
    members: Vec<RankId>,
    membership_hash: u64,
    canonical: Vec<Slot>,
    cursors: BTreeMap<RankId, usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ActiveSlot {
    group: GroupId,
    rank: RankId,
    instance: CommInstanceId,
}

struct SequencerState {
    decisions: BTreeMap<ExecutionId, CollectiveDecision>,
    decided_prefix: u64,
    groups: BTreeMap<GroupId, GroupState>,
    active: BTreeSet<ActiveSlot>,
    aborted: bool,
    source_sequences: BTreeMap<ProtocolSourceId, u64>,
}

struct SequencerInner {
    state: Mutex<SequencerState>,
    topology_epoch: u64,
    coordinator_source: ProtocolSourceId,
    rank_source_base: u64,
    sink: Arc<dyn ProtocolTraceSink>,
}

/// Shared collective-ordering authority for one topology epoch.
#[derive(Clone)]
pub struct CollectiveSequencer {
    inner: Arc<SequencerInner>,
}

impl CollectiveSequencer {
    pub(crate) fn new() -> Self {
        Self::with_sink(ProtocolSourceId::new(10_000), 0, Arc::new(NullTraceSink))
    }

    pub(crate) fn with_sink(
        source: ProtocolSourceId,
        topology_epoch: u64,
        sink: Arc<dyn ProtocolTraceSink>,
    ) -> Self {
        Self {
            inner: Arc::new(SequencerInner {
                state: Mutex::new(SequencerState {
                    decisions: BTreeMap::new(),
                    decided_prefix: 0,
                    groups: BTreeMap::new(),
                    active: BTreeSet::new(),
                    aborted: false,
                    source_sequences: BTreeMap::new(),
                }),
                topology_epoch,
                coordinator_source: ProtocolSourceId::new(source.get().saturating_add(1)),
                rank_source_base: source.get().saturating_add(2),
                sink,
            }),
        }
    }

    pub(crate) fn register_group(&self, group: GroupId, members: &[RankId]) -> CommResult<u64> {
        if members.is_empty() {
            return Err(CommError::InvalidArgument(
                "communicator group must not be empty".into(),
            ));
        }
        if !members.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(CommError::InvalidArgument(
                "communicator group ranks must be sorted and unique".into(),
            ));
        }
        let hash = membership_hash(members);
        let mut state = self.inner.lock();
        match state.groups.get(&group) {
            Some(existing) if existing.members == members && existing.membership_hash == hash => {
                Ok(hash)
            }
            Some(_) => Err(CommError::Ordering(format!(
                "group {:?} was registered with different membership",
                group
            ))),
            None => {
                state.groups.insert(
                    group,
                    GroupState {
                        members: members.to_vec(),
                        membership_hash: hash,
                        canonical: Vec::new(),
                        cursors: members.iter().copied().map(|rank| (rank, 0)).collect(),
                    },
                );
                Ok(hash)
            }
        }
    }

    /// Publishes the next coordinator decision. Decisions are global and
    /// monotonic across every group in the topology epoch.
    pub fn decide(&self, execution: ExecutionId, decision: CollectiveDecision) -> CommResult<()> {
        let mut state = self.inner.lock();
        Self::decide_locked(&self.inner, &mut state, execution, decision)
    }

    pub(crate) fn submit(
        &self,
        group: GroupId,
        rank: RankId,
        instance: CommInstanceId,
        collective: CollectiveKind,
        signature: u64,
    ) -> CommResult<()> {
        let mut state = self.inner.lock();
        if state.aborted {
            return Err(CommError::Aborted("collective sequencer aborted".into()));
        }
        if !state.decisions.contains_key(&instance.execution) {
            Self::decide_locked(
                &self.inner,
                &mut state,
                instance.execution,
                CollectiveDecision::Admitted,
            )?;
        }
        if state.decisions.get(&instance.execution) != Some(&CollectiveDecision::Admitted) {
            return Err(CommError::Ordering(format!(
                "execution {:?} was skipped",
                instance.execution
            )));
        }
        let slot = Slot {
            instance,
            disposition: SlotDisposition::Submitted,
            collective: Some(collective),
            signature,
        };
        let (members, membership_hash) = Self::advance_group_locked(&mut state, group, rank, slot)?;
        let active = ActiveSlot {
            group,
            rank,
            instance,
        };
        if !state.active.insert(active) {
            return Err(CommError::Ordering(format!(
                "rank {:?} submitted duplicate collective {:?}",
                rank, instance
            )));
        }
        self.inner.emit_rank_locked(
            &mut state,
            rank,
            CollectiveOrderingEvent::Submit {
                group: group.0,
                members: members.iter().map(|member| member.0).collect(),
                membership_hash,
                rank: rank.0,
                execution: instance.execution.0,
                sequence: instance.sequence.0,
                collective,
                signature,
            },
        );
        Ok(())
    }

    pub(crate) fn observe_skip(
        &self,
        group: GroupId,
        rank: RankId,
        instance: CommInstanceId,
    ) -> CommResult<()> {
        let mut state = self.inner.lock();
        if state.aborted {
            return Err(CommError::Aborted("collective sequencer aborted".into()));
        }
        if state.decisions.get(&instance.execution) != Some(&CollectiveDecision::Skipped) {
            return Err(CommError::Ordering(format!(
                "execution {:?} was not skipped",
                instance.execution
            )));
        }
        let slot = Slot {
            instance,
            disposition: SlotDisposition::Skipped,
            collective: None,
            signature: 0,
        };
        let (members, membership_hash) = Self::advance_group_locked(&mut state, group, rank, slot)?;
        self.inner.emit_rank_locked(
            &mut state,
            rank,
            CollectiveOrderingEvent::ObserveSkip {
                group: group.0,
                members: members.iter().map(|member| member.0).collect(),
                membership_hash,
                rank: rank.0,
                execution: instance.execution.0,
                sequence: instance.sequence.0,
            },
        );
        Ok(())
    }

    pub(crate) fn complete(
        &self,
        group: GroupId,
        rank: RankId,
        instance: CommInstanceId,
        success: bool,
    ) -> CommResult<()> {
        let mut state = self.inner.lock();
        let active = ActiveSlot {
            group,
            rank,
            instance,
        };
        if !state.active.remove(&active) {
            // Abort completes every active local operation under the sequencer
            // lock. A waking backend caller then observes an already-terminal
            // slot and must not publish a duplicate completion.
            if state.aborted {
                return Ok(());
            }
            return Err(CommError::Ordering(format!(
                "rank {:?} completed non-active collective {:?}",
                rank, instance
            )));
        }
        let (members, membership_hash) = Self::group_identity(&state, group)?;
        self.inner.emit_rank_locked(
            &mut state,
            rank,
            CollectiveOrderingEvent::CompleteLocal {
                group: group.0,
                members: members.iter().map(|member| member.0).collect(),
                membership_hash,
                rank: rank.0,
                execution: instance.execution.0,
                sequence: instance.sequence.0,
                success,
            },
        );
        Ok(())
    }

    /// Closes submission for the topology epoch, then terminally completes every
    /// already-submitted rank-local operation with an error.
    pub(crate) fn abort(&self) {
        let mut state = self.inner.lock();
        if state.aborted {
            return;
        }
        state.aborted = true;
        self.inner
            .emit_coordinator_locked(&mut state, CollectiveOrderingEvent::Abort);
        let active: Vec<ActiveSlot> = state.active.iter().copied().collect();
        for slot in active {
            state.active.remove(&slot);
            let Ok((members, membership_hash)) = Self::group_identity(&state, slot.group) else {
                continue;
            };
            self.inner.emit_rank_locked(
                &mut state,
                slot.rank,
                CollectiveOrderingEvent::CompleteLocal {
                    group: slot.group.0,
                    members: members.iter().map(|member| member.0).collect(),
                    membership_hash,
                    rank: slot.rank.0,
                    execution: slot.instance.execution.0,
                    sequence: slot.instance.sequence.0,
                    success: false,
                },
            );
        }
    }

    pub(crate) fn is_aborted(&self) -> bool {
        self.inner.lock().aborted
    }

    pub(crate) fn is_active(&self, group: GroupId, rank: RankId, instance: CommInstanceId) -> bool {
        self.inner.lock().active.contains(&ActiveSlot {
            group,
            rank,
            instance,
        })
    }

    fn decide_locked(
        inner: &SequencerInner,
        state: &mut SequencerState,
        execution: ExecutionId,
        decision: CollectiveDecision,
    ) -> CommResult<()> {
        if state.aborted {
            return Err(CommError::Aborted("collective sequencer aborted".into()));
        }
        let expected = state
            .decided_prefix
            .checked_add(1)
            .ok_or_else(|| CommError::Ordering("execution decision prefix overflowed".into()))?;
        if execution.0 != expected {
            return Err(CommError::Ordering(format!(
                "execution decision {:?} is not next prefix value {expected}",
                execution
            )));
        }
        state.decisions.insert(execution, decision.clone());
        state.decided_prefix = expected;
        inner.emit_coordinator_locked(
            state,
            CollectiveOrderingEvent::Decide {
                execution: execution.0,
                decision,
            },
        );
        Ok(())
    }

    fn advance_group_locked(
        state: &mut SequencerState,
        group: GroupId,
        rank: RankId,
        slot: Slot,
    ) -> CommResult<(Vec<RankId>, u64)> {
        let group_state = state
            .groups
            .get_mut(&group)
            .ok_or_else(|| CommError::Ordering(format!("unknown group {:?}", group)))?;
        let cursor = group_state
            .cursors
            .get_mut(&rank)
            .ok_or(CommError::UnknownRank(rank))?;
        if *cursor < group_state.canonical.len() {
            let expected = group_state.canonical[*cursor];
            if expected != slot {
                return Err(CommError::Ordering(format!(
                    "rank {:?} slot {} diverged: expected {:?}, got {:?}",
                    rank, *cursor, expected, slot
                )));
            }
        } else {
            if let Some(previous) = group_state.canonical.last()
                && slot.instance <= previous.instance
            {
                return Err(CommError::Ordering(format!(
                    "collective {:?} is not strictly after {:?}",
                    slot.instance, previous.instance
                )));
            }
            group_state.canonical.push(slot);
        }
        *cursor += 1;
        Ok((group_state.members.clone(), group_state.membership_hash))
    }

    fn group_identity(state: &SequencerState, group: GroupId) -> CommResult<(Vec<RankId>, u64)> {
        let group = state
            .groups
            .get(&group)
            .ok_or_else(|| CommError::Ordering("completion named unknown group".into()))?;
        Ok((group.members.clone(), group.membership_hash))
    }
}

impl SequencerInner {
    fn lock(&self) -> std::sync::MutexGuard<'_, SequencerState> {
        self.state.lock().expect("collective sequencer poisoned")
    }

    fn emit_coordinator_locked(&self, state: &mut SequencerState, event: CollectiveOrderingEvent) {
        self.emit_locked(state, self.coordinator_source, event);
    }

    fn emit_rank_locked(
        &self,
        state: &mut SequencerState,
        rank: RankId,
        event: CollectiveOrderingEvent,
    ) {
        let source = ProtocolSourceId::new(self.rank_source_base.saturating_add(u64::from(rank.0)));
        self.emit_locked(state, source, event);
    }

    fn emit_locked(
        &self,
        state: &mut SequencerState,
        source: ProtocolSourceId,
        event: CollectiveOrderingEvent,
    ) {
        let sequence = state.source_sequences.entry(source).or_default();
        let source_sequence = *sequence;
        *sequence = sequence.saturating_add(1);
        self.sink.record(ProtocolTraceEvent {
            contract_revision: CONTRACT_REVISION,
            topology_epoch: self.topology_epoch,
            source,
            source_sequence,
            kind: ProtocolEvent::CollectiveOrdering(event),
        });
    }
}

/// Stable FNV-1a hash of the ordered world-rank vector.
pub fn membership_hash(members: &[RankId]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in (members.len() as u64)
        .to_le_bytes()
        .into_iter()
        .chain(members.iter().flat_map(|rank| rank.0.to_le_bytes()))
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
