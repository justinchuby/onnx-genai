//! Atomic state, effect, and output publication for one admitted workflow turn.
//!
//! The interpreter deliberately keeps its mutable values in a pass-local SSA
//! map.  This module is the only bridge from that working map to durable session
//! state.  It records the complete declared write set before execution, stages
//! every replacement, and swaps complete committed maps only after every
//! fallible operation has completed.

use crate::decode::clone_value;
use anyhow::Context;
use onnx_genai_metadata::{ResolvedStatePlan, StateIdentity};
use onnx_genai_ort::Value;
use std::collections::{BTreeMap, HashMap};

/// Stable identity of an admitted turn.  It is not an output revision number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TurnTransactionId(pub u64);

/// Stable identity of the committed baseline captured for one turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TurnBaselineId(pub u64);

/// A semantic state value before a turn starts.
pub enum TurnStateBaseline {
    Absent,
    Present(Value),
}

impl std::fmt::Debug for TurnStateBaseline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absent => formatter.write_str("Absent"),
            Self::Present(value) => formatter
                .debug_struct("Present")
                .field("shape", &value.shape())
                .finish(),
        }
    }
}

/// Committed cursor and lineage facts for one output stream.
///
/// These are deliberately typed independently of payloads: payload contents,
/// output names, and container traversal order cannot select a rollback target.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OutputPublicationBaseline {
    pub head: u64,
    pub cursor: u64,
    pub lineage: u64,
    pub closed: bool,
}

/// The complete immutable baseline for an admitted turn.
#[derive(Debug)]
pub struct TurnCommittedBaseline {
    pub id: TurnBaselineId,
    pub transaction: TurnTransactionId,
    pub states: BTreeMap<StateIdentity, TurnStateBaseline>,
    pub effects: BTreeMap<String, u64>,
    pub outputs: BTreeMap<String, OutputPublicationBaseline>,
}

/// Why an admitted turn did not commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnAbortReason {
    ExecutionFailure,
    Cancellation,
    CommitFailure,
}

/// The outcome that retracts an admitted provisional turn.
///
/// This is intentionally not an output revision operation.  Its identity
/// points at the transaction and complete baseline that own every provisional
/// state/effect/output advance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnTransactionOutcome {
    Committed {
        transaction: TurnTransactionId,
        baseline: TurnBaselineId,
    },
    AbortToBaseline {
        transaction: TurnTransactionId,
        baseline: TurnBaselineId,
        reason: TurnAbortReason,
    },
}

/// Publication visibility selected at admission.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TurnPublicationMode {
    /// Outputs remain in the transaction's working set until commit.
    #[default]
    CommitOnly,
    /// Reserved for the typed revision protocol.  It must be rejected before
    /// mutation if its sink cannot retract by transaction identity.
    ProvisionalRevisions,
}

/// Typed admission failure, distinct from an exclusive-lease rejection.
#[derive(Debug, thiserror::Error)]
pub enum TurnTransactionAdmissionError {
    #[error(
        "cannot admit an atomic turn: semantic session state '{state}' has no resolved final \
         writer in the canonical state plan; join its writers before executing"
    )]
    MissingFinalWriter { state: String },
    #[error(
        "cannot admit provisional output mode: output '{output}' has no transaction-addressable \
         retraction sink; use commit_only or install a typed revision sink before executing"
    )]
    UnretractableProvisionalOutput { output: String },
    #[error(
        "cannot admit an atomic turn: failed to snapshot semantic session state '{state}': \
         {message}"
    )]
    StateSnapshot { state: String, message: String },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CommittedOutputState {
    pub(crate) head: u64,
    pub(crate) cursor: u64,
    pub(crate) lineage: u64,
    pub(crate) closed: bool,
}

/// The state staged by one turn.  Its fields are private so callers cannot
/// partially commit state, effects, or output publication.
#[derive(Debug)]
pub(crate) struct TurnTransaction {
    baseline: TurnCommittedBaseline,
    session: Option<String>,
    staged_states: BTreeMap<StateIdentity, TurnStateBaseline>,
    staged_effects: BTreeMap<String, u64>,
    staged_outputs: BTreeMap<String, CommittedOutputState>,
    publication_mode: TurnPublicationMode,
}

impl TurnTransaction {
    /// Admit a runtime-owned participant into the same transaction namespace as
    /// canonical workflow state. Its opaque baseline is held by the participant
    /// itself; this authority still owns the sole turn/baseline identity and
    /// outcome protocol.
    pub(crate) fn admit_runtime_participant(id: TurnTransactionId) -> Self {
        Self {
            baseline: TurnCommittedBaseline {
                id: TurnBaselineId(id.0),
                transaction: id,
                states: BTreeMap::new(),
                effects: BTreeMap::new(),
                outputs: BTreeMap::new(),
            },
            session: None,
            staged_states: BTreeMap::new(),
            staged_effects: BTreeMap::new(),
            staged_outputs: BTreeMap::new(),
            publication_mode: TurnPublicationMode::CommitOnly,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit(
        id: TurnTransactionId,
        session: Option<&str>,
        state_plan: &ResolvedStatePlan,
        effect_domains: impl IntoIterator<Item = String>,
        outputs: impl IntoIterator<Item = String>,
        session_state: &HashMap<(String, String), Value>,
        session_effects: &HashMap<(String, String), u64>,
        session_outputs: &HashMap<(String, String), CommittedOutputState>,
        publication_mode: TurnPublicationMode,
    ) -> Result<Self, TurnTransactionAdmissionError> {
        let mut states = BTreeMap::new();
        let mut staged_states = BTreeMap::new();
        if let Some(session) = session {
            for (_, cell) in state_plan
                .session_cells()
                .filter(|(_, cell)| cell.transaction.required)
            {
                if cell.final_writer.is_none() {
                    return Err(TurnTransactionAdmissionError::MissingFinalWriter {
                        state: cell.identity.0.clone(),
                    });
                }
                let baseline =
                    match session_state.get(&(session.to_string(), cell.identity.0.clone())) {
                        Some(value) => {
                            TurnStateBaseline::Present(clone_value(value).map_err(|error| {
                                TurnTransactionAdmissionError::StateSnapshot {
                                    state: cell.identity.0.clone(),
                                    message: format!("{error:#}"),
                                }
                            })?)
                        }
                        None => TurnStateBaseline::Absent,
                    };
                let staged = clone_state_baseline(&baseline).map_err(|error| {
                    TurnTransactionAdmissionError::StateSnapshot {
                        state: cell.identity.0.clone(),
                        message: format!("{error:#}"),
                    }
                })?;
                staged_states.insert(cell.identity.clone(), staged);
                states.insert(cell.identity.clone(), baseline);
            }
        }

        let mut effects = BTreeMap::new();
        let mut staged_effects = BTreeMap::new();
        let mut output_baselines = BTreeMap::new();
        let mut staged_outputs = BTreeMap::new();
        for effect in effect_domains {
            let cursor = session
                .and_then(|session| session_effects.get(&(session.to_string(), effect.clone())))
                .copied()
                .unwrap_or_default();
            staged_effects.insert(effect.clone(), cursor);
            effects.insert(effect, cursor);
        }
        for output in outputs {
            if publication_mode == TurnPublicationMode::ProvisionalRevisions {
                return Err(
                    TurnTransactionAdmissionError::UnretractableProvisionalOutput { output },
                );
            }
            let committed = session
                .and_then(|session| session_outputs.get(&(session.to_string(), output.clone())))
                .copied()
                .unwrap_or_default();
            output_baselines.insert(
                output.clone(),
                OutputPublicationBaseline {
                    head: committed.head,
                    cursor: committed.cursor,
                    lineage: committed.lineage,
                    closed: committed.closed,
                },
            );
            staged_outputs.insert(output, committed);
        }

        Ok(Self {
            baseline: TurnCommittedBaseline {
                id: TurnBaselineId(id.0),
                transaction: id,
                states,
                effects,
                outputs: output_baselines,
            },
            session: session.map(ToOwned::to_owned),
            staged_states,
            staged_effects,
            staged_outputs,
            publication_mode,
        })
    }

    pub(crate) fn baseline_state(&self, state: &str) -> Option<&TurnStateBaseline> {
        self.baseline
            .states
            .iter()
            .find_map(|(identity, value)| (identity.0 == state).then_some(value))
    }

    pub(crate) fn id(&self) -> TurnTransactionId {
        self.baseline.transaction
    }

    pub(crate) fn output_baseline(&self, output: &str) -> OutputPublicationBaseline {
        self.baseline
            .outputs
            .get(output)
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn stage_state(&mut self, state: StateIdentity, value: Value) {
        self.staged_states
            .insert(state, TurnStateBaseline::Present(value));
    }

    /// Stage output advancement from the typed output identity, never from an
    /// emitted payload or the order a map happens to be traversed.
    pub(crate) fn stage_output(&mut self, output: &str, publications: u64) {
        let state = self
            .staged_outputs
            .get_mut(output)
            .expect("output was admitted into this transaction");
        state.head = state.head.saturating_add(publications);
        state.cursor = state.cursor.saturating_add(publications);
        state.lineage = state.lineage.saturating_add(publications);
    }

    /// Closure is staged with the output head and becomes durable only with the
    /// enclosing turn commit. An abort therefore restores an early finalization
    /// to the admission baseline just like every other output fact.
    pub(crate) fn stage_output_closed(&mut self, output: &str) {
        let state = self
            .staged_outputs
            .get_mut(output)
            .expect("output was admitted into this transaction");
        state.closed = true;
    }

    pub(crate) fn stage_effects(&mut self) {
        for cursor in self.staged_effects.values_mut() {
            *cursor = cursor.saturating_add(1);
        }
    }

    /// Prepare participant-scoped writes, reserve every map before mutation,
    /// then apply them. There are no fallible operations after the reservations,
    /// so every externally observable result is old or complete-new without
    /// copying unrelated sessions' potentially large state tensors.
    pub(crate) fn commit(
        &mut self,
        session_state: &mut HashMap<(String, String), Value>,
        session_effects: &mut HashMap<(String, String), u64>,
        session_outputs: &mut HashMap<(String, String), CommittedOutputState>,
    ) -> anyhow::Result<TurnTransactionOutcome> {
        let Some(session) = &self.session else {
            return Ok(TurnTransactionOutcome::Committed {
                transaction: self.baseline.transaction,
                baseline: self.baseline.id,
            });
        };
        let state_writes = self
            .staged_states
            .iter()
            .map(|(state, value)| {
                Ok((
                    (session.clone(), state.0.clone()),
                    clone_state_baseline(value)?,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let effect_writes = self
            .staged_effects
            .iter()
            .map(|(effect, cursor)| ((session.clone(), effect.clone()), *cursor))
            .collect::<Vec<_>>();
        let output_writes = self
            .staged_outputs
            .iter()
            .map(|(output, state)| ((session.clone(), output.clone()), *state))
            .collect::<Vec<_>>();
        session_state
            .try_reserve(state_writes.len())
            .context("failed to reserve the atomic turn state write set")?;
        session_effects
            .try_reserve(effect_writes.len())
            .context("failed to reserve the atomic turn effect write set")?;
        session_outputs
            .try_reserve(output_writes.len())
            .context("failed to reserve the atomic turn output write set")?;
        for (key, value) in state_writes {
            match value {
                TurnStateBaseline::Absent => {
                    session_state.remove(&key);
                }
                TurnStateBaseline::Present(value) => {
                    session_state.insert(key, value);
                }
            }
        }
        for (key, cursor) in effect_writes {
            session_effects.insert(key, cursor);
        }
        for (key, state) in output_writes {
            session_outputs.insert(key, state);
        }
        Ok(TurnTransactionOutcome::Committed {
            transaction: self.baseline.transaction,
            baseline: self.baseline.id,
        })
    }

    pub(crate) fn committed(&self) -> TurnTransactionOutcome {
        TurnTransactionOutcome::Committed {
            transaction: self.baseline.transaction,
            baseline: self.baseline.id,
        }
    }

    pub(crate) fn abort(&self, reason: TurnAbortReason) -> TurnTransactionOutcome {
        let _ = self.publication_mode;
        TurnTransactionOutcome::AbortToBaseline {
            transaction: self.baseline.transaction,
            baseline: self.baseline.id,
            reason,
        }
    }
}

fn clone_state_baseline(value: &TurnStateBaseline) -> anyhow::Result<TurnStateBaseline> {
    match value {
        TurnStateBaseline::Absent => Ok(TurnStateBaseline::Absent),
        TurnStateBaseline::Present(value) => clone_value(value)
            .map(TurnStateBaseline::Present)
            .context("failed to snapshot a semantic state participant"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_advances_complete_effect_and_output_write_set() -> anyhow::Result<()> {
        let mut states = HashMap::new();
        let mut effects = HashMap::from([
            (("session".to_string(), "grammar".to_string()), 4),
            (("other".to_string(), "grammar".to_string()), 9),
        ]);
        let mut outputs = HashMap::from([(
            ("session".to_string(), "tokens".to_string()),
            CommittedOutputState {
                head: 3,
                cursor: 4,
                lineage: 5,
                closed: false,
            },
        )]);
        let mut turn = TurnTransaction::admit(
            TurnTransactionId(7),
            Some("session"),
            &ResolvedStatePlan::default(),
            ["grammar".to_string()],
            ["tokens".to_string()],
            &states,
            &effects,
            &outputs,
            TurnPublicationMode::CommitOnly,
        )?;
        turn.stage_effects();
        turn.stage_output("tokens", 2);
        assert_eq!(
            turn.commit(&mut states, &mut effects, &mut outputs)?,
            TurnTransactionOutcome::Committed {
                transaction: TurnTransactionId(7),
                baseline: TurnBaselineId(7),
            }
        );
        assert_eq!(effects[&("session".to_string(), "grammar".to_string())], 5);
        assert_eq!(effects[&("other".to_string(), "grammar".to_string())], 9);
        assert_eq!(
            outputs[&("session".to_string(), "tokens".to_string())],
            CommittedOutputState {
                head: 5,
                cursor: 6,
                lineage: 7,
                closed: false,
            }
        );
        Ok(())
    }

    #[test]
    fn abort_is_a_transaction_outcome_not_an_output_operation() -> anyhow::Result<()> {
        let turn = TurnTransaction::admit(
            TurnTransactionId(11),
            None,
            &ResolvedStatePlan::default(),
            std::iter::empty::<String>(),
            std::iter::empty::<String>(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            TurnPublicationMode::CommitOnly,
        )?;
        assert_eq!(
            turn.abort(TurnAbortReason::Cancellation),
            TurnTransactionOutcome::AbortToBaseline {
                transaction: TurnTransactionId(11),
                baseline: TurnBaselineId(11),
                reason: TurnAbortReason::Cancellation,
            }
        );
        Ok(())
    }

    #[test]
    fn provisional_mode_is_refused_before_any_mutation() {
        let error = TurnTransaction::admit(
            TurnTransactionId(1),
            Some("session"),
            &ResolvedStatePlan::default(),
            std::iter::empty::<String>(),
            ["tokens".to_string()],
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            TurnPublicationMode::ProvisionalRevisions,
        )
        .expect_err("an unretractable provisional output must fail admission");
        assert!(matches!(
            error,
            TurnTransactionAdmissionError::UnretractableProvisionalOutput { output }
            if output == "tokens"
        ));
    }

    #[test]
    fn early_output_close_is_staged_until_commit_and_abort_keeps_the_baseline() -> anyhow::Result<()>
    {
        let mut states = HashMap::new();
        let mut effects = HashMap::new();
        let mut outputs = HashMap::from([(
            ("session".to_string(), "answer".to_string()),
            CommittedOutputState::default(),
        )]);
        let mut aborted = TurnTransaction::admit(
            TurnTransactionId(31),
            Some("session"),
            &ResolvedStatePlan::default(),
            std::iter::empty::<String>(),
            ["answer".to_string()],
            &states,
            &effects,
            &outputs,
            TurnPublicationMode::CommitOnly,
        )?;
        aborted.stage_output_closed("answer");
        assert_eq!(
            aborted.abort(TurnAbortReason::Cancellation),
            TurnTransactionOutcome::AbortToBaseline {
                transaction: TurnTransactionId(31),
                baseline: TurnBaselineId(31),
                reason: TurnAbortReason::Cancellation,
            }
        );
        assert!(!outputs[&("session".to_string(), "answer".to_string())].closed);

        let mut committed = TurnTransaction::admit(
            TurnTransactionId(32),
            Some("session"),
            &ResolvedStatePlan::default(),
            std::iter::empty::<String>(),
            ["answer".to_string()],
            &states,
            &effects,
            &outputs,
            TurnPublicationMode::CommitOnly,
        )?;
        committed.stage_output_closed("answer");
        committed.commit(&mut states, &mut effects, &mut outputs)?;
        assert!(outputs[&("session".to_string(), "answer".to_string())].closed);
        Ok(())
    }
}
