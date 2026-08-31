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

use super::OutputStreamId;

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
#[derive(Default)]
pub struct OutputPublicationBaseline {
    pub head: u64,
    pub cursor: u64,
    pub lineage: u64,
    pub closed: bool,
    pub payload: Option<Value>,
}

/// The transport-safe portion of an output stream's admission baseline.
///
/// Payload bytes deliberately do not select rollback: a receiver restores its
/// own prior content by this stream identity and these immutable cursors. This
/// keeps map traversal, payload shape, and output ordering out of the
/// transaction reconciliation contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputStreamBaseline {
    pub output: String,
    pub stream: OutputStreamId,
    pub head: u64,
    pub sequence: u64,
    pub lineage: u64,
    pub closed: bool,
}

impl std::fmt::Debug for OutputPublicationBaseline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OutputPublicationBaseline")
            .field("head", &self.head)
            .field("cursor", &self.cursor)
            .field("lineage", &self.lineage)
            .field("closed", &self.closed)
            .field("payload", &self.payload.as_ref().map(Value::shape))
            .finish()
    }
}

/// The complete immutable baseline for an admitted turn.
#[derive(Debug)]
pub struct TurnCommittedBaseline {
    pub id: TurnBaselineId,
    pub transaction: TurnTransactionId,
    pub states: BTreeMap<StateIdentity, TurnStateBaseline>,
    pub effects: BTreeMap<String, u64>,
    pub outputs: BTreeMap<(String, OutputStreamId), OutputPublicationBaseline>,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnTransactionOutcome {
    Committed {
        transaction: TurnTransactionId,
        baseline: TurnBaselineId,
    },
    AbortToBaseline {
        transaction: TurnTransactionId,
        baseline: TurnBaselineId,
        reason: TurnAbortReason,
        /// Every stream that can have been affected by this admitted turn,
        /// including dynamically named streams with the empty baseline.
        streams: Vec<OutputStreamBaseline>,
    },
}

/// The sole terminal decision made for an admitted turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnTransactionResolution {
    Committed,
    Aborted,
}

impl std::fmt::Display for TurnTransactionResolution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Committed => formatter.write_str("committed"),
            Self::Aborted => formatter.write_str("aborted"),
        }
    }
}

/// A caller attempted to mutate or resolve a turn after its terminal decision.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "turn transaction {transaction:?} is already {resolution}; cannot {operation}. \
     Admit a new turn before staging or resolving more work"
)]
pub struct TurnTransactionResolutionError {
    pub transaction: TurnTransactionId,
    pub resolution: TurnTransactionResolution,
    pub operation: &'static str,
}

/// Commit can fail either because the turn was already resolved or while
/// preparing its complete atomic write set.
#[derive(Debug, thiserror::Error)]
pub enum TurnTransactionCommitError {
    #[error(transparent)]
    Resolved(#[from] TurnTransactionResolutionError),
    #[error(transparent)]
    Prepare(#[from] anyhow::Error),
}

/// Publication visibility selected at admission.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TurnPublicationMode {
    /// Outputs remain in the transaction's working set until commit.
    #[default]
    CommitOnly,
    /// Output publication uses the typed revision protocol and a terminal
    /// transaction outcome for deterministic reconciliation.
    ProvisionalRevisions,
}

impl From<onnx_genai_metadata::WorkflowPublicationMode> for TurnPublicationMode {
    fn from(value: onnx_genai_metadata::WorkflowPublicationMode) -> Self {
        match value {
            onnx_genai_metadata::WorkflowPublicationMode::CommitOnly => Self::CommitOnly,
            onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions => {
                Self::ProvisionalRevisions
            }
        }
    }
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
        "cannot admit an atomic turn: failed to snapshot semantic session state '{state}': \
         {message}"
    )]
    StateSnapshot { state: String, message: String },
}

#[derive(Default)]
pub(crate) struct CommittedOutputState {
    pub(crate) head: u64,
    pub(crate) cursor: u64,
    pub(crate) lineage: u64,
    pub(crate) closed: bool,
    pub(crate) payload: Option<Value>,
}

impl std::fmt::Debug for CommittedOutputState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CommittedOutputState")
            .field("head", &self.head)
            .field("cursor", &self.cursor)
            .field("lineage", &self.lineage)
            .field("closed", &self.closed)
            .field("payload", &self.payload.as_ref().map(Value::shape))
            .finish()
    }
}

/// The state staged by one turn.  Its fields are private so callers cannot
/// partially commit state, effects, or output publication.
#[derive(Debug)]
pub(crate) struct TurnTransaction {
    baseline: TurnCommittedBaseline,
    session: Option<String>,
    staged_states: BTreeMap<StateIdentity, TurnStateBaseline>,
    staged_effects: BTreeMap<String, u64>,
    staged_outputs: BTreeMap<(String, OutputStreamId), CommittedOutputState>,
    publication_mode: TurnPublicationMode,
    resolution: Option<TurnTransactionResolution>,
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
            resolution: None,
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
        session_outputs: &HashMap<(String, String, OutputStreamId), CommittedOutputState>,
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
        let outputs = outputs
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(session) = session {
            for ((owner, output, stream), committed) in session_outputs {
                if owner != session || !outputs.contains(output) {
                    continue;
                }
                let committed = clone_committed_output_state(committed).map_err(|error| {
                    TurnTransactionAdmissionError::StateSnapshot {
                        state: format!("output '{output}' stream '{}'", stream.0),
                        message: format!("{error:#}"),
                    }
                })?;
                let baseline = clone_output_state_as_baseline(&committed).map_err(|error| {
                    TurnTransactionAdmissionError::StateSnapshot {
                        state: format!("output '{output}' stream '{}'", stream.0),
                        message: format!("{error:#}"),
                    }
                })?;
                output_baselines.insert((output.clone(), stream.clone()), baseline);
                staged_outputs.insert((output.clone(), stream.clone()), committed);
            }
        }
        for output in outputs {
            let identity = (output.clone(), OutputStreamId(output.clone()));
            if staged_outputs.contains_key(&identity) {
                continue;
            }
            let committed = CommittedOutputState::default();
            output_baselines.insert(
                identity.clone(),
                OutputPublicationBaseline {
                    head: committed.head,
                    cursor: committed.cursor,
                    lineage: committed.lineage,
                    closed: committed.closed,
                    payload: None,
                },
            );
            staged_outputs.insert(identity, committed);
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
            resolution: None,
        })
    }

    fn ensure_open(&self, operation: &'static str) -> Result<(), TurnTransactionResolutionError> {
        match self.resolution {
            None => Ok(()),
            Some(resolution) => Err(TurnTransactionResolutionError {
                transaction: self.baseline.transaction,
                resolution,
                operation,
            }),
        }
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

    pub(crate) fn baseline_id(&self) -> TurnBaselineId {
        self.baseline.id
    }

    pub(crate) fn publication_mode(&self) -> TurnPublicationMode {
        self.publication_mode
    }

    pub(crate) fn output_baselines(
        &self,
    ) -> anyhow::Result<BTreeMap<(String, OutputStreamId), OutputPublicationBaseline>> {
        self.baseline
            .outputs
            .iter()
            .map(|(identity, baseline)| Ok((identity.clone(), clone_output_baseline(baseline)?)))
            .collect()
    }

    pub(crate) fn stage_state(
        &mut self,
        state: StateIdentity,
        value: Value,
    ) -> Result<(), TurnTransactionResolutionError> {
        self.ensure_open("stage semantic state")?;
        self.staged_states
            .insert(state, TurnStateBaseline::Present(value));
        Ok(())
    }

    /// Replace the complete per-stream output write set resolved by the
    /// canonical publication journal. There is no output-level count or
    /// default-stream inference that can collapse independent streams.
    pub(crate) fn stage_outputs(
        &mut self,
        outputs: BTreeMap<(String, OutputStreamId), CommittedOutputState>,
    ) -> Result<(), TurnTransactionResolutionError> {
        self.ensure_open("stage output publication state")?;
        self.staged_outputs = outputs;
        Ok(())
    }

    pub(crate) fn stage_effects(&mut self) -> Result<(), TurnTransactionResolutionError> {
        self.ensure_open("stage effect cursors")?;
        for cursor in self.staged_effects.values_mut() {
            *cursor = cursor.saturating_add(1);
        }
        Ok(())
    }

    /// Resolve a runtime-owned participant whose durable state is committed by
    /// its enclosing owner rather than the workflow maps below.
    pub(crate) fn commit_runtime_participant(
        &mut self,
    ) -> Result<TurnTransactionOutcome, TurnTransactionResolutionError> {
        self.ensure_open("commit runtime participant")?;
        self.resolution = Some(TurnTransactionResolution::Committed);
        Ok(TurnTransactionOutcome::Committed {
            transaction: self.baseline.transaction,
            baseline: self.baseline.id,
        })
    }

    /// Prepare participant-scoped writes, reserve every map before mutation,
    /// then apply them. There are no fallible operations after the reservations,
    /// so every externally observable result is old or complete-new without
    /// copying unrelated sessions' potentially large state tensors.
    pub(crate) fn commit(
        &mut self,
        session_state: &mut HashMap<(String, String), Value>,
        session_effects: &mut HashMap<(String, String), u64>,
        session_outputs: &mut HashMap<(String, String, OutputStreamId), CommittedOutputState>,
    ) -> Result<TurnTransactionOutcome, TurnTransactionCommitError> {
        self.ensure_open("commit durable state")?;
        let Some(session) = &self.session else {
            self.resolution = Some(TurnTransactionResolution::Committed);
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
            .map(|((output, stream), state)| {
                Ok((
                    (session.clone(), output.clone(), stream.clone()),
                    clone_committed_output_state(state)?,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
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
        self.resolution = Some(TurnTransactionResolution::Committed);
        Ok(TurnTransactionOutcome::Committed {
            transaction: self.baseline.transaction,
            baseline: self.baseline.id,
        })
    }

    pub(crate) fn abort(
        &mut self,
        reason: TurnAbortReason,
    ) -> Result<TurnTransactionOutcome, TurnTransactionResolutionError> {
        let streams = self.baseline.outputs.keys().cloned().collect::<Vec<_>>();
        self.abort_for_streams(reason, streams)
    }

    /// Build the one abort outcome for the streams the publication authority
    /// actually touched. A stream first introduced by this turn had no durable
    /// entry at admission, so its exact baseline is the empty cursor state.
    pub(crate) fn abort_for_streams(
        &mut self,
        reason: TurnAbortReason,
        streams: impl IntoIterator<Item = (String, OutputStreamId)>,
    ) -> Result<TurnTransactionOutcome, TurnTransactionResolutionError> {
        self.ensure_open("abort to the admitted baseline")?;
        let mut streams = streams
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|(output, stream)| {
                let baseline = self.baseline.outputs.get(&(output.clone(), stream.clone()));
                OutputStreamBaseline {
                    output,
                    stream,
                    head: baseline.map_or(0, |value| value.head),
                    sequence: baseline.map_or(0, |value| value.cursor),
                    lineage: baseline.map_or(0, |value| value.lineage),
                    closed: baseline.is_some_and(|value| value.closed),
                }
            })
            .collect::<Vec<_>>();
        streams.sort_by(|left, right| {
            (&left.output, &left.stream).cmp(&(&right.output, &right.stream))
        });
        self.resolution = Some(TurnTransactionResolution::Aborted);
        Ok(TurnTransactionOutcome::AbortToBaseline {
            transaction: self.baseline.transaction,
            baseline: self.baseline.id,
            reason,
            streams,
        })
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

fn clone_output_baseline(
    value: &OutputPublicationBaseline,
) -> anyhow::Result<OutputPublicationBaseline> {
    Ok(OutputPublicationBaseline {
        head: value.head,
        cursor: value.cursor,
        lineage: value.lineage,
        closed: value.closed,
        payload: value.payload.as_ref().map(clone_value).transpose()?,
    })
}

fn clone_committed_output_state(
    value: &CommittedOutputState,
) -> anyhow::Result<CommittedOutputState> {
    Ok(CommittedOutputState {
        head: value.head,
        cursor: value.cursor,
        lineage: value.lineage,
        closed: value.closed,
        payload: value.payload.as_ref().map(clone_value).transpose()?,
    })
}

fn clone_output_state_as_baseline(
    value: &CommittedOutputState,
) -> anyhow::Result<OutputPublicationBaseline> {
    Ok(OutputPublicationBaseline {
        head: value.head,
        cursor: value.cursor,
        lineage: value.lineage,
        closed: value.closed,
        payload: value.payload.as_ref().map(clone_value).transpose()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(session: &str, output: &str, stream: &str) -> (String, String, OutputStreamId) {
        (
            session.to_string(),
            output.to_string(),
            OutputStreamId(stream.to_string()),
        )
    }

    fn output_state(head: u64, cursor: u64, lineage: u64, closed: bool) -> CommittedOutputState {
        CommittedOutputState {
            head,
            cursor,
            lineage,
            closed,
            payload: None,
        }
    }

    fn output_state_with_payload(
        head: u64,
        cursor: u64,
        lineage: u64,
        closed: bool,
        payload: i64,
    ) -> anyhow::Result<CommittedOutputState> {
        Ok(CommittedOutputState {
            head,
            cursor,
            lineage,
            closed,
            payload: Some(Value::from_slice_i64(&[payload], &[1])?),
        })
    }

    fn assert_resolution_error(
        error: TurnTransactionResolutionError,
        transaction: u64,
        resolution: TurnTransactionResolution,
        operation: &'static str,
    ) {
        assert_eq!(
            error,
            TurnTransactionResolutionError {
                transaction: TurnTransactionId(transaction),
                resolution,
                operation,
            }
        );
        assert_eq!(
            error.to_string(),
            format!(
                "turn transaction TurnTransactionId({transaction}) is already {resolution}; \
                 cannot {operation}. Admit a new turn before staging or resolving more work"
            )
        );
    }

    #[test]
    fn commit_advances_complete_effect_and_output_write_set() -> anyhow::Result<()> {
        let mut states = HashMap::new();
        let mut effects = HashMap::from([
            (("session".to_string(), "grammar".to_string()), 4),
            (("other".to_string(), "grammar".to_string()), 9),
        ]);
        let mut outputs = HashMap::from([(
            key("session", "tokens", "text"),
            output_state(3, 4, 5, false),
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
        turn.stage_effects()?;
        turn.stage_outputs(BTreeMap::from([(
            ("tokens".to_string(), OutputStreamId("text".to_string())),
            output_state(5, 6, 7, false),
        )]))?;
        assert_eq!(
            turn.commit(&mut states, &mut effects, &mut outputs)?,
            TurnTransactionOutcome::Committed {
                transaction: TurnTransactionId(7),
                baseline: TurnBaselineId(7),
            }
        );
        assert_eq!(effects[&("session".to_string(), "grammar".to_string())], 5);
        assert_eq!(effects[&("other".to_string(), "grammar".to_string())], 9);
        let committed = &outputs[&key("session", "tokens", "text")];
        assert_eq!(
            (
                committed.head,
                committed.cursor,
                committed.lineage,
                committed.closed
            ),
            (5, 6, 7, false)
        );
        Ok(())
    }

    #[test]
    fn abort_is_a_transaction_outcome_not_an_output_operation() -> anyhow::Result<()> {
        let mut turn = TurnTransaction::admit(
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
            turn.abort(TurnAbortReason::Cancellation)?,
            TurnTransactionOutcome::AbortToBaseline {
                transaction: TurnTransactionId(11),
                baseline: TurnBaselineId(11),
                reason: TurnAbortReason::Cancellation,
                streams: Vec::new(),
            }
        );
        Ok(())
    }

    #[test]
    fn provisional_mode_preserves_the_admission_baseline() -> anyhow::Result<()> {
        let mut turn = TurnTransaction::admit(
            TurnTransactionId(1),
            Some("session"),
            &ResolvedStatePlan::default(),
            std::iter::empty::<String>(),
            ["tokens".to_string()],
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            TurnPublicationMode::ProvisionalRevisions,
        )?;
        let TurnTransactionOutcome::AbortToBaseline { streams, .. } = turn.abort_for_streams(
            TurnAbortReason::Cancellation,
            [("tokens".to_string(), OutputStreamId("named".to_string()))],
        )?
        else {
            unreachable!("abort always has a baseline");
        };
        assert_eq!(
            streams,
            vec![OutputStreamBaseline {
                output: "tokens".to_string(),
                stream: OutputStreamId("named".to_string()),
                head: 0,
                sequence: 0,
                lineage: 0,
                closed: false,
            }]
        );
        Ok(())
    }

    #[test]
    fn named_stream_state_is_atomic_across_abort_and_commit() -> anyhow::Result<()> {
        let mut states = HashMap::new();
        let mut effects = HashMap::new();
        let mut outputs = HashMap::from([
            (
                key("session", "answer", "analysis"),
                output_state_with_payload(2, 3, 2, false, 20)?,
            ),
            (
                key("session", "answer", "final"),
                output_state_with_payload(4, 5, 4, true, 40)?,
            ),
        ]);
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
        aborted.stage_outputs(BTreeMap::from([
            (
                ("answer".to_string(), OutputStreamId("analysis".to_string())),
                output_state_with_payload(6, 7, 6, true, 60)?,
            ),
            (
                ("answer".to_string(), OutputStreamId("retry".to_string())),
                output_state_with_payload(1, 2, 1, true, 10)?,
            ),
        ]))?;
        let TurnTransactionOutcome::AbortToBaseline {
            transaction,
            baseline,
            reason,
            streams,
        } = aborted.abort_for_streams(
            TurnAbortReason::Cancellation,
            [
                ("answer".to_string(), OutputStreamId("analysis".to_string())),
                ("answer".to_string(), OutputStreamId("retry".to_string())),
            ],
        )?
        else {
            unreachable!("an aborted turn has a typed baseline");
        };
        assert_eq!(transaction, TurnTransactionId(31));
        assert_eq!(baseline, TurnBaselineId(31));
        assert_eq!(reason, TurnAbortReason::Cancellation);
        assert_eq!(
            streams,
            vec![
                OutputStreamBaseline {
                    output: "answer".to_string(),
                    stream: OutputStreamId("analysis".to_string()),
                    head: 2,
                    sequence: 3,
                    lineage: 2,
                    closed: false,
                },
                OutputStreamBaseline {
                    output: "answer".to_string(),
                    stream: OutputStreamId("retry".to_string()),
                    head: 0,
                    sequence: 0,
                    lineage: 0,
                    closed: false,
                },
            ]
        );
        assert_eq!(
            (
                outputs[&key("session", "answer", "analysis")].head,
                outputs[&key("session", "answer", "analysis")].closed,
                outputs[&key("session", "answer", "analysis")]
                    .payload
                    .as_ref()
                    .expect("analysis payload")
                    .to_vec_i64()?,
                outputs[&key("session", "answer", "final")].head,
                outputs[&key("session", "answer", "final")].closed,
                outputs[&key("session", "answer", "final")]
                    .payload
                    .as_ref()
                    .expect("final payload")
                    .to_vec_i64()?,
            ),
            (2, false, vec![20], 4, true, vec![40])
        );
        assert!(!outputs.contains_key(&key("session", "answer", "retry")));

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
        committed.stage_outputs(BTreeMap::from([
            (
                ("answer".to_string(), OutputStreamId("analysis".to_string())),
                output_state_with_payload(6, 7, 6, true, 60)?,
            ),
            (
                ("answer".to_string(), OutputStreamId("final".to_string())),
                output_state_with_payload(4, 5, 4, true, 40)?,
            ),
            (
                ("answer".to_string(), OutputStreamId("retry".to_string())),
                output_state_with_payload(1, 2, 1, true, 10)?,
            ),
        ]))?;
        committed.commit(&mut states, &mut effects, &mut outputs)?;
        assert_eq!(
            (
                outputs[&key("session", "answer", "analysis")].head,
                outputs[&key("session", "answer", "analysis")].closed,
                outputs[&key("session", "answer", "analysis")]
                    .payload
                    .as_ref()
                    .expect("analysis payload")
                    .to_vec_i64()?,
                outputs[&key("session", "answer", "final")].head,
                outputs[&key("session", "answer", "final")].closed,
                outputs[&key("session", "answer", "retry")].head,
                outputs[&key("session", "answer", "retry")].closed,
                outputs[&key("session", "answer", "retry")]
                    .payload
                    .as_ref()
                    .expect("retry payload")
                    .to_vec_i64()?,
            ),
            (6, true, vec![60], 4, true, 1, true, vec![10])
        );
        Ok(())
    }

    #[test]
    fn abort_resolves_once_and_rejects_every_later_stage_or_terminal_action() -> anyhow::Result<()>
    {
        let mut states = HashMap::new();
        let mut effects = HashMap::from([(("session".to_string(), "grammar".to_string()), 4)]);
        let mut outputs = HashMap::from([(
            key("session", "answer", "answer"),
            output_state_with_payload(2, 2, 2, false, 7)?,
        )]);
        let durable_before = (
            effects.clone(),
            outputs[&key("session", "answer", "answer")].head,
            outputs[&key("session", "answer", "answer")]
                .payload
                .as_ref()
                .expect("baseline payload")
                .to_raw_bytes()?,
        );
        let mut turn = TurnTransaction::admit(
            TurnTransactionId(41),
            Some("session"),
            &ResolvedStatePlan::default(),
            ["grammar".to_string()],
            ["answer".to_string()],
            &states,
            &effects,
            &outputs,
            TurnPublicationMode::CommitOnly,
        )?;

        assert!(matches!(
            turn.abort(TurnAbortReason::Cancellation)?,
            TurnTransactionOutcome::AbortToBaseline { .. }
        ));
        assert_resolution_error(
            turn.stage_state(
                StateIdentity("memory".to_string()),
                Value::from_slice_i64(&[9], &[1])?,
            )
            .expect_err("abort must close state staging"),
            41,
            TurnTransactionResolution::Aborted,
            "stage semantic state",
        );
        assert_resolution_error(
            turn.stage_effects()
                .expect_err("abort must close effect staging"),
            41,
            TurnTransactionResolution::Aborted,
            "stage effect cursors",
        );
        assert_resolution_error(
            turn.stage_outputs(BTreeMap::new())
                .expect_err("abort must close output staging"),
            41,
            TurnTransactionResolution::Aborted,
            "stage output publication state",
        );
        let TurnTransactionCommitError::Resolved(error) = turn
            .commit(&mut states, &mut effects, &mut outputs)
            .expect_err("abort must close commit")
        else {
            panic!("post-abort commit returned a non-resolution error");
        };
        assert_resolution_error(
            error,
            41,
            TurnTransactionResolution::Aborted,
            "commit durable state",
        );
        assert_resolution_error(
            turn.abort(TurnAbortReason::ExecutionFailure)
                .expect_err("repeated abort must fail"),
            41,
            TurnTransactionResolution::Aborted,
            "abort to the admitted baseline",
        );
        assert_eq!(
            (
                effects,
                outputs[&key("session", "answer", "answer")].head,
                outputs[&key("session", "answer", "answer")]
                    .payload
                    .as_ref()
                    .expect("unchanged payload")
                    .to_raw_bytes()?,
            ),
            durable_before,
            "post-abort misuse must leave durable authority byte-for-byte unchanged"
        );
        Ok(())
    }

    #[test]
    fn commit_resolves_once_and_rejects_every_later_stage_or_abort() -> anyhow::Result<()> {
        let mut states = HashMap::new();
        let mut effects = HashMap::from([(("session".to_string(), "grammar".to_string()), 4)]);
        let mut outputs = HashMap::from([(
            key("session", "answer", "answer"),
            output_state_with_payload(2, 2, 2, false, 7)?,
        )]);
        let mut turn = TurnTransaction::admit(
            TurnTransactionId(42),
            Some("session"),
            &ResolvedStatePlan::default(),
            ["grammar".to_string()],
            ["answer".to_string()],
            &states,
            &effects,
            &outputs,
            TurnPublicationMode::CommitOnly,
        )?;
        turn.stage_effects()?;
        turn.stage_outputs(BTreeMap::from([(
            ("answer".to_string(), OutputStreamId("answer".to_string())),
            output_state_with_payload(3, 3, 3, true, 8)?,
        )]))?;
        assert!(matches!(
            turn.commit(&mut states, &mut effects, &mut outputs)?,
            TurnTransactionOutcome::Committed { .. }
        ));
        let durable_after = (
            effects.clone(),
            outputs[&key("session", "answer", "answer")].head,
            outputs[&key("session", "answer", "answer")]
                .payload
                .as_ref()
                .expect("committed payload")
                .to_raw_bytes()?,
        );

        assert_resolution_error(
            turn.stage_effects()
                .expect_err("commit must close effect staging"),
            42,
            TurnTransactionResolution::Committed,
            "stage effect cursors",
        );
        assert_resolution_error(
            turn.stage_outputs(BTreeMap::new())
                .expect_err("commit must close output staging"),
            42,
            TurnTransactionResolution::Committed,
            "stage output publication state",
        );
        assert_resolution_error(
            turn.abort(TurnAbortReason::Cancellation)
                .expect_err("commit must close abort"),
            42,
            TurnTransactionResolution::Committed,
            "abort to the admitted baseline",
        );
        let TurnTransactionCommitError::Resolved(error) = turn
            .commit(&mut states, &mut effects, &mut outputs)
            .expect_err("repeated commit must fail")
        else {
            panic!("repeated commit returned a non-resolution error");
        };
        assert_resolution_error(
            error,
            42,
            TurnTransactionResolution::Committed,
            "commit durable state",
        );
        assert_eq!(
            (
                effects,
                outputs[&key("session", "answer", "answer")].head,
                outputs[&key("session", "answer", "answer")]
                    .payload
                    .as_ref()
                    .expect("unchanged committed payload")
                    .to_raw_bytes()?,
            ),
            durable_after,
            "post-commit misuse must leave durable authority byte-for-byte unchanged"
        );
        Ok(())
    }
}
