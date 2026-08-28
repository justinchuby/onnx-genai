//! Transport-neutral workflow output publication protocols.
//!
//! An executor owns this journal until its enclosing turn commits. HTTP/SSE
//! adapters receive the resulting ordered publications; they never create,
//! reorder, or roll back semantic output state.

use std::collections::BTreeMap;

use anyhow::{Context as _, Result, bail, ensure};
use onnx_genai_metadata::{WorkflowEmitMode, WorkflowOutputFamily, WorkflowSpec};
use onnx_genai_ort::Value;

use crate::decode::clone_value;

use super::{
    OutputPublicationBaseline, OutputStreamBaseline, TurnAbortReason, TurnBaselineId,
    TurnPublicationMode, TurnTransactionId, TurnTransactionOutcome,
    turn_transaction::CommittedOutputState,
};

/// The sole typed-revision envelope protocol this runtime implements.
pub const TYPED_REVISION_PROTOCOL_VERSION: &str = "1";

/// Logical stream identity scoped by its workflow output.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputStreamId(pub String);

/// Deterministic occurrence number within one logical output stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputSequence(pub u64);

/// Monotonic revision operation number within one logical output stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputRevision(pub u64);

/// Named revision lineage. Zero is the committed empty baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputLineage(pub u64);

/// Whether an output publication is still owned by its admitted turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFinality {
    Provisional,
    Final,
}

/// Operation in the versioned typed-revision protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedRevisionOperation {
    Append,
    Replace,
    Retract,
    Finalize,
}

impl TypedRevisionOperation {
    fn carries_payload(self) -> bool {
        matches!(self, Self::Append | Self::Replace)
    }
}

/// A transport-neutral, typed revision publication.
///
/// `base` is the active lineage observed by the producer. `lineage` is a new
/// lineage for append/replace, or the named current lineage for retraction and
/// finalization. The explicit pair makes stale writers fail rather than
/// silently overwriting a stream head.
pub struct TypedRevisionEnvelope {
    pub version: String,
    pub transaction: TurnTransactionId,
    pub output: String,
    pub stream: OutputStreamId,
    pub sequence: OutputSequence,
    pub revision: OutputRevision,
    pub lineage: OutputLineage,
    pub base: OutputLineage,
    pub operation: TypedRevisionOperation,
    pub payload: Option<Value>,
    pub finality: OutputFinality,
}

impl std::fmt::Debug for TypedRevisionEnvelope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TypedRevisionEnvelope")
            .field("version", &self.version)
            .field("transaction", &self.transaction)
            .field("output", &self.output)
            .field("stream", &self.stream)
            .field("sequence", &self.sequence)
            .field("revision", &self.revision)
            .field("lineage", &self.lineage)
            .field("base", &self.base)
            .field("operation", &self.operation)
            .field("payload", &self.payload.as_ref().map(|value| value.shape()))
            .field("finality", &self.finality)
            .finish()
    }
}

/// A publication visible at a workflow output boundary.
pub enum WorkflowOutputPublication {
    Materialized {
        output: String,
        operation: TypedRevisionOperation,
        payload: Value,
        finality: OutputFinality,
    },
    Event {
        output: String,
        stream: OutputStreamId,
        sequence: OutputSequence,
        payload: Value,
        finality: OutputFinality,
    },
    Revision(TypedRevisionEnvelope),
    /// The admitted turn committed atomically. In provisional mode this is the
    /// typed finality authority for every earlier provisional envelope.
    TransactionCommitted {
        transaction: TurnTransactionId,
        baseline: TurnBaselineId,
    },
    /// The admitted turn did not commit. Receivers restore exactly these
    /// admission cursors and must not infer inverse revisions from payloads.
    AbortToBaseline {
        transaction: TurnTransactionId,
        baseline: TurnBaselineId,
        reason: TurnAbortReason,
        streams: Vec<OutputStreamBaseline>,
    },
}

impl std::fmt::Debug for WorkflowOutputPublication {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Materialized {
                output,
                operation,
                payload,
                finality,
            } => formatter
                .debug_struct("Materialized")
                .field("output", output)
                .field("operation", operation)
                .field("payload", &payload.shape())
                .field("finality", finality)
                .finish(),
            Self::Event {
                output,
                stream,
                sequence,
                payload,
                finality,
            } => formatter
                .debug_struct("Event")
                .field("output", output)
                .field("stream", stream)
                .field("sequence", sequence)
                .field("payload", &payload.shape())
                .field("finality", finality)
                .finish(),
            Self::Revision(envelope) => envelope.fmt(formatter),
            Self::TransactionCommitted {
                transaction,
                baseline,
            } => formatter
                .debug_struct("TransactionCommitted")
                .field("transaction", transaction)
                .field("baseline", baseline)
                .finish(),
            Self::AbortToBaseline {
                transaction,
                baseline,
                reason,
                streams,
            } => formatter
                .debug_struct("AbortToBaseline")
                .field("transaction", transaction)
                .field("baseline", baseline)
                .field("reason", reason)
                .field("streams", streams)
                .finish(),
        }
    }
}

impl WorkflowOutputPublication {
    /// Preserve a transaction outcome as a protocol record without inventing
    /// output-level inverse operations.
    pub fn from_transaction_outcome(outcome: &TurnTransactionOutcome) -> Self {
        match outcome {
            TurnTransactionOutcome::Committed {
                transaction,
                baseline,
            } => Self::TransactionCommitted {
                transaction: *transaction,
                baseline: *baseline,
            },
            TurnTransactionOutcome::AbortToBaseline {
                transaction,
                baseline,
                reason,
                streams,
            } => Self::AbortToBaseline {
                transaction: *transaction,
                baseline: *baseline,
                reason: *reason,
                streams: streams.clone(),
            },
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(match self {
            Self::Materialized {
                output,
                operation,
                payload,
                finality,
            } => Self::Materialized {
                output: output.clone(),
                operation: *operation,
                payload: clone_value(payload)?,
                finality: *finality,
            },
            Self::Event {
                output,
                stream,
                sequence,
                payload,
                finality,
            } => Self::Event {
                output: output.clone(),
                stream: stream.clone(),
                sequence: *sequence,
                payload: clone_value(payload)?,
                finality: *finality,
            },
            Self::Revision(envelope) => Self::Revision(TypedRevisionEnvelope {
                version: envelope.version.clone(),
                transaction: envelope.transaction,
                output: envelope.output.clone(),
                stream: envelope.stream.clone(),
                sequence: envelope.sequence,
                revision: envelope.revision,
                lineage: envelope.lineage,
                base: envelope.base,
                operation: envelope.operation,
                payload: envelope.payload.as_ref().map(clone_value).transpose()?,
                finality: envelope.finality,
            }),
            Self::TransactionCommitted {
                transaction,
                baseline,
            } => Self::TransactionCommitted {
                transaction: *transaction,
                baseline: *baseline,
            },
            Self::AbortToBaseline {
                transaction,
                baseline,
                reason,
                streams,
            } => Self::AbortToBaseline {
                transaction: *transaction,
                baseline: *baseline,
                reason: *reason,
                streams: streams.clone(),
            },
        })
    }
}

/// Validation failures report output and stream context before a caller mutates
/// its output head.
#[derive(Debug, thiserror::Error)]
pub enum RevisionEnvelopeValidationError {
    #[error(
        "typed revision envelope for output '{output}' stream '{stream}' declares unknown protocol version '{version}'; this runtime implements version '{TYPED_REVISION_PROTOCOL_VERSION}'"
    )]
    UnknownVersion {
        output: String,
        stream: String,
        version: String,
    },
    #[error("typed revision envelope targets undeclared revision output '{output}'")]
    UnknownOutput { output: String },
    #[error(
        "typed revision envelope targets output '{output}' family {family:?}, not typed revisions"
    )]
    FamilyMismatch {
        output: String,
        family: WorkflowOutputFamily,
    },
    #[error(
        "typed revision envelope for output '{output}' stream '{stream}' has sequence {actual}, expected {expected}"
    )]
    Sequence {
        output: String,
        stream: String,
        actual: u64,
        expected: u64,
    },
    #[error(
        "typed revision envelope for output '{output}' stream '{stream}' has revision {actual}, expected {expected}"
    )]
    Revision {
        output: String,
        stream: String,
        actual: u64,
        expected: u64,
    },
    #[error(
        "typed revision envelope for output '{output}' stream '{stream}' has base lineage {actual}, expected active lineage {expected}"
    )]
    Base {
        output: String,
        stream: String,
        actual: u64,
        expected: u64,
    },
    #[error("typed revision envelope for output '{output}' stream '{stream}' is post-finalize")]
    Closed { output: String, stream: String },
    #[error(
        "typed revision envelope for output '{output}' stream '{stream}' operation {operation:?} has an invalid payload"
    )]
    Payload {
        output: String,
        stream: String,
        operation: TypedRevisionOperation,
    },
    #[error(
        "typed revision envelope for output '{output}' stream '{stream}' has finality {actual:?} before its transaction commits; revision envelopes must remain provisional until a typed commit outcome"
    )]
    Finality {
        output: String,
        stream: String,
        actual: OutputFinality,
    },
    #[error(
        "typed revision envelope for output '{output}' stream '{stream}' operation {operation:?} has lineage {lineage}, expected {expected}"
    )]
    Lineage {
        output: String,
        stream: String,
        operation: TypedRevisionOperation,
        lineage: u64,
        expected: u64,
    },
    #[error("typed revision envelope for output '{output}' has an empty stream identity")]
    EmptyStream { output: String },
    #[error(
        "typed revision envelope for output '{output}' stream '{stream}' cannot advance because its {field} cursor is exhausted"
    )]
    Exhausted {
        output: String,
        stream: String,
        field: &'static str,
    },
}

#[derive(Debug, Clone, Copy, Default)]
struct RevisionStreamState {
    sequence: u64,
    revision: u64,
    lineage: u64,
    closed: bool,
}

/// Validates and advances typed revision envelopes atomically per logical
/// stream. Failed validation leaves every stream cursor unchanged.
pub struct RevisionEnvelopeValidator {
    outputs: BTreeMap<String, WorkflowOutputFamily>,
    streams: BTreeMap<(String, OutputStreamId), RevisionStreamState>,
}

impl RevisionEnvelopeValidator {
    pub fn new(outputs: impl IntoIterator<Item = (String, WorkflowOutputFamily)>) -> Self {
        Self {
            outputs: outputs.into_iter().collect(),
            streams: BTreeMap::new(),
        }
    }

    pub fn validate_and_apply(
        &mut self,
        envelope: &TypedRevisionEnvelope,
    ) -> std::result::Result<(), RevisionEnvelopeValidationError> {
        let output = envelope.output.clone();
        let stream = envelope.stream.0.clone();
        if envelope.version != TYPED_REVISION_PROTOCOL_VERSION {
            return Err(RevisionEnvelopeValidationError::UnknownVersion {
                output,
                stream,
                version: envelope.version.clone(),
            });
        }
        let Some(family) = self.outputs.get(&envelope.output) else {
            return Err(RevisionEnvelopeValidationError::UnknownOutput {
                output: envelope.output.clone(),
            });
        };
        let WorkflowOutputFamily::Revisions { version } = family else {
            return Err(RevisionEnvelopeValidationError::FamilyMismatch {
                output: envelope.output.clone(),
                family: family.clone(),
            });
        };
        if version != TYPED_REVISION_PROTOCOL_VERSION {
            return Err(RevisionEnvelopeValidationError::UnknownVersion {
                output: envelope.output.clone(),
                stream: envelope.stream.0.clone(),
                version: version.clone(),
            });
        }
        if envelope.stream.0.is_empty() {
            return Err(RevisionEnvelopeValidationError::EmptyStream {
                output: envelope.output.clone(),
            });
        }
        if envelope.finality != OutputFinality::Provisional {
            return Err(RevisionEnvelopeValidationError::Finality {
                output,
                stream,
                actual: envelope.finality,
            });
        }
        let state = self
            .streams
            .get(&(envelope.output.clone(), envelope.stream.clone()))
            .copied()
            .unwrap_or_default();
        if state.closed {
            return Err(RevisionEnvelopeValidationError::Closed { output, stream });
        }
        let Some(expected_sequence) = state.sequence.checked_add(1) else {
            return Err(RevisionEnvelopeValidationError::Exhausted {
                output,
                stream,
                field: "sequence",
            });
        };
        if envelope.sequence.0 != expected_sequence {
            return Err(RevisionEnvelopeValidationError::Sequence {
                output,
                stream,
                actual: envelope.sequence.0,
                expected: expected_sequence,
            });
        }
        let Some(expected_revision) = state.revision.checked_add(1) else {
            return Err(RevisionEnvelopeValidationError::Exhausted {
                output,
                stream,
                field: "revision",
            });
        };
        if envelope.revision.0 != expected_revision {
            return Err(RevisionEnvelopeValidationError::Revision {
                output,
                stream,
                actual: envelope.revision.0,
                expected: expected_revision,
            });
        }
        if envelope.base.0 != state.lineage {
            return Err(RevisionEnvelopeValidationError::Base {
                output,
                stream,
                actual: envelope.base.0,
                expected: state.lineage,
            });
        }
        if envelope.operation.carries_payload() != envelope.payload.is_some() {
            return Err(RevisionEnvelopeValidationError::Payload {
                output,
                stream,
                operation: envelope.operation,
            });
        }
        let expected_lineage = match envelope.operation {
            TypedRevisionOperation::Append | TypedRevisionOperation::Replace => envelope.revision.0,
            TypedRevisionOperation::Retract | TypedRevisionOperation::Finalize => state.lineage,
        };
        if envelope.lineage.0 != expected_lineage {
            return Err(RevisionEnvelopeValidationError::Lineage {
                output,
                stream,
                operation: envelope.operation,
                lineage: envelope.lineage.0,
                expected: expected_lineage,
            });
        }

        let mut next = state;
        next.sequence = envelope.sequence.0;
        next.revision = envelope.revision.0;
        match envelope.operation {
            TypedRevisionOperation::Append | TypedRevisionOperation::Replace => {
                next.lineage = envelope.lineage.0;
            }
            TypedRevisionOperation::Retract => {
                next.lineage = 0;
            }
            TypedRevisionOperation::Finalize => {
                next.closed = true;
            }
        }
        self.streams
            .insert((envelope.output.clone(), envelope.stream.clone()), next);
        Ok(())
    }

    fn state(&self, output: &str, stream: &OutputStreamId) -> RevisionStreamState {
        self.streams
            .get(&(output.to_string(), stream.clone()))
            .copied()
            .unwrap_or_default()
    }
}

/// Pass-local journal that produces all three output families in authored
/// execution order. It is intentionally not a network stream.
#[derive(Debug, Clone)]
enum OutputProtocolFamily {
    /// Pre-v1.5 per-site semantics: replace, append, and event were selected by
    /// the emit itself because no output-level family existed yet.
    Legacy,
    Declared(WorkflowOutputFamily),
}

pub(crate) struct OutputPublicationJournal {
    transaction: TurnTransactionId,
    publication_mode: TurnPublicationMode,
    outputs: BTreeMap<String, OutputProtocolFamily>,
    revisions: RevisionEnvelopeValidator,
    revision_payloads: BTreeMap<(String, OutputStreamId), Option<Value>>,
    event_sequences: BTreeMap<(String, OutputStreamId), u64>,
    materialized: BTreeMap<(String, OutputStreamId), CommittedOutputState>,
    publications: Vec<WorkflowOutputPublication>,
    provisional_delivery_cursor: usize,
}

impl OutputPublicationJournal {
    #[cfg(test)]
    pub(crate) fn new(
        transaction: TurnTransactionId,
        workflow: &WorkflowSpec,
        baselines: BTreeMap<(String, OutputStreamId), OutputPublicationBaseline>,
    ) -> Result<Self> {
        Self::new_with_publication_mode(
            transaction,
            workflow,
            baselines,
            TurnPublicationMode::from(workflow.publication_mode),
        )
    }

    pub(crate) fn new_with_publication_mode(
        transaction: TurnTransactionId,
        workflow: &WorkflowSpec,
        baselines: BTreeMap<(String, OutputStreamId), OutputPublicationBaseline>,
        publication_mode: TurnPublicationMode,
    ) -> Result<Self> {
        ensure!(
            publication_mode == TurnPublicationMode::from(workflow.publication_mode),
            "transaction publication mode {:?} diverges from pipeline.workflow.publication_mode \
             {:?}; reject before output mutation rather than creating two authorities",
            publication_mode,
            workflow.publication_mode,
        );
        if publication_mode == TurnPublicationMode::ProvisionalRevisions {
            for (output, declaration) in &workflow.outputs {
                ensure!(
                    matches!(
                        declaration.family,
                        WorkflowOutputFamily::Revisions { ref version }
                            if version == TYPED_REVISION_PROTOCOL_VERSION
                    ),
                    "cannot admit provisional publication mode: output '{output}' has family {:?}; \
                     provisional_revisions requires typed revision family version '{}' for every \
                     output so abort_to_baseline can reconcile the complete turn",
                    declaration.family,
                    TYPED_REVISION_PROTOCOL_VERSION,
                );
            }
        }
        for (output, declaration) in &workflow.outputs {
            if declaration.family_authored
                && let WorkflowOutputFamily::Revisions { version } = &declaration.family
            {
                ensure!(
                    version == TYPED_REVISION_PROTOCOL_VERSION,
                    "output '{output}' declares typed revision protocol version '{version}', but \
                     this runtime implements version '{TYPED_REVISION_PROTOCOL_VERSION}'"
                );
            }
        }
        let outputs = workflow
            .outputs
            .iter()
            .map(|(name, declaration)| {
                (
                    name.clone(),
                    if declaration.family_authored {
                        OutputProtocolFamily::Declared(declaration.family.clone())
                    } else {
                        OutputProtocolFamily::Legacy
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut revisions = RevisionEnvelopeValidator::new(outputs.iter().map(|(name, family)| {
            (
                name.clone(),
                match family {
                    OutputProtocolFamily::Declared(family) => family.clone(),
                    OutputProtocolFamily::Legacy => WorkflowOutputFamily::Materialized,
                },
            )
        }));
        let mut revision_payloads = BTreeMap::new();
        let mut event_sequences = BTreeMap::new();
        let mut materialized = BTreeMap::new();
        for ((output, stream), baseline) in baselines {
            let Some(family) = outputs.get(&output) else {
                continue;
            };
            match family {
                OutputProtocolFamily::Declared(WorkflowOutputFamily::Revisions { .. })
                    if baseline.head != 0
                        || baseline.cursor != 0
                        || baseline.lineage != 0
                        || baseline.closed
                        || baseline.payload.is_some() =>
                {
                    ensure!(
                        baseline.head == baseline.cursor,
                        "output '{output}' stream '{}' has invalid revision admission baseline: \
                         head {} differs from sequence {}; reject before mutation rather than \
                         publishing an unreconcilable lineage",
                        stream.0,
                        baseline.head,
                        baseline.cursor,
                    );
                    ensure!(
                        baseline.lineage <= baseline.head,
                        "output '{output}' stream '{}' has invalid revision admission baseline: \
                         lineage {} exceeds head {}; reject before mutation rather than \
                         publishing an unreconcilable lineage",
                        stream.0,
                        baseline.lineage,
                        baseline.head,
                    );
                    ensure!(
                        (baseline.lineage == 0) == baseline.payload.is_none(),
                        "output '{output}' stream '{}' has invalid revision admission baseline: \
                         lineage {} and payload presence disagree; reject before mutation",
                        stream.0,
                        baseline.lineage,
                    );
                    revisions.streams.insert(
                        (output.clone(), stream.clone()),
                        RevisionStreamState {
                            sequence: baseline.cursor,
                            revision: baseline.head,
                            lineage: baseline.lineage,
                            closed: baseline.closed,
                        },
                    );
                    revision_payloads.insert((output, stream), baseline.payload);
                }
                OutputProtocolFamily::Declared(WorkflowOutputFamily::Events) => {
                    event_sequences.insert((output, stream), baseline.cursor);
                }
                OutputProtocolFamily::Declared(WorkflowOutputFamily::Materialized) => {
                    materialized.insert(
                        (output, stream),
                        CommittedOutputState {
                            head: baseline.head,
                            cursor: baseline.cursor,
                            lineage: baseline.lineage,
                            closed: baseline.closed,
                            payload: baseline.payload,
                        },
                    );
                }
                OutputProtocolFamily::Legacy if baseline.payload.is_some() => {
                    materialized.insert(
                        (output, stream),
                        CommittedOutputState {
                            head: baseline.head,
                            cursor: baseline.cursor,
                            lineage: baseline.lineage,
                            closed: baseline.closed,
                            payload: baseline.payload,
                        },
                    );
                }
                OutputProtocolFamily::Legacy => {
                    event_sequences.insert((output, stream), baseline.cursor);
                }
                OutputProtocolFamily::Declared(WorkflowOutputFamily::Revisions { .. }) => {}
            }
        }
        Ok(Self {
            transaction,
            publication_mode,
            revisions,
            outputs,
            revision_payloads,
            event_sequences,
            materialized,
            publications: Vec::new(),
            provisional_delivery_cursor: 0,
        })
    }

    /// Validate the declared operation before evaluation moves or materializes
    /// its SSA payload. This mirrors metadata validation at the runtime
    /// boundary, where a specialized caller cannot turn a bad envelope into a
    /// partial output mutation.
    pub(crate) fn validate_emit(
        &self,
        output: &str,
        stream: Option<&str>,
        mode: &WorkflowEmitMode,
    ) -> Result<()> {
        let family = self
            .outputs
            .get(output)
            .with_context(|| format!("workflow emit targets undeclared output '{output}'"))?;
        let stream = OutputStreamId(stream.unwrap_or(output).to_string());
        ensure!(
            !stream.0.is_empty(),
            "workflow emit targets output '{output}' with an empty stream identity"
        );
        match (family, mode) {
            (
                OutputProtocolFamily::Legacy
                | OutputProtocolFamily::Declared(WorkflowOutputFamily::Materialized),
                WorkflowEmitMode::Replace | WorkflowEmitMode::Append,
            ) => ensure!(
                stream.0 == output,
                "materialized output '{output}' cannot publish a named stream '{}'",
                stream.0
            ),
            (
                OutputProtocolFamily::Legacy
                | OutputProtocolFamily::Declared(WorkflowOutputFamily::Events),
                WorkflowEmitMode::Event,
            ) => {}
            (
                OutputProtocolFamily::Declared(WorkflowOutputFamily::Revisions { .. }),
                WorkflowEmitMode::Append
                | WorkflowEmitMode::Replace
                | WorkflowEmitMode::Retract
                | WorkflowEmitMode::Finalize,
            ) => {
                let state = self.revisions.state(output, &stream);
                ensure!(
                    !state.closed,
                    "workflow emit {mode:?} targets finalized output '{output}' stream '{}'",
                    stream.0
                );
                if matches!(mode, WorkflowEmitMode::Retract) {
                    ensure!(
                        state.lineage != 0,
                        "workflow emit Retract targets output '{output}' stream '{}' with no active \
                         revision lineage",
                        stream.0
                    );
                }
            }
            _ => bail!(
                "workflow emit {mode:?} targets output '{output}' with family {family:?}; \
                 select an operation legal for that output family"
            ),
        }
        Ok(())
    }

    pub(crate) fn publish(
        &mut self,
        output: &str,
        stream: Option<&str>,
        mode: &WorkflowEmitMode,
        payload: Option<Value>,
    ) -> Result<()> {
        self.validate_emit(output, stream, mode)?;
        let family = self
            .outputs
            .get(output)
            .expect("validate_emit proved this output exists");
        let stream = OutputStreamId(stream.unwrap_or(output).to_string());
        match (family, mode) {
            (
                OutputProtocolFamily::Legacy
                | OutputProtocolFamily::Declared(WorkflowOutputFamily::Materialized),
                WorkflowEmitMode::Replace | WorkflowEmitMode::Append,
            ) => {
                ensure!(
                    stream.0 == output,
                    "materialized output '{output}' cannot publish a named stream '{}'",
                    stream.0
                );
                let payload = payload.with_context(|| {
                    format!(
                        "workflow emit {mode:?} for materialized output '{output}' has no payload"
                    )
                })?;
                let state = self
                    .materialized
                    .entry((output.to_string(), stream.clone()))
                    .or_default();
                state.head = state.head.checked_add(1).with_context(|| {
                    format!("materialized output '{output}' exhausted its head cursor")
                })?;
                state.cursor = state.cursor.checked_add(1).with_context(|| {
                    format!("materialized output '{output}' exhausted its sequence cursor")
                })?;
                state.lineage = state.head;
                state.payload = Some(clone_value(&payload)?);
                self.publications
                    .push(WorkflowOutputPublication::Materialized {
                        output: output.to_string(),
                        operation: match mode {
                            WorkflowEmitMode::Replace => TypedRevisionOperation::Replace,
                            WorkflowEmitMode::Append => TypedRevisionOperation::Append,
                            _ => unreachable!("matched materialized emit operation"),
                        },
                        payload,
                        finality: OutputFinality::Provisional,
                    });
            }
            (
                OutputProtocolFamily::Legacy
                | OutputProtocolFamily::Declared(WorkflowOutputFamily::Events),
                WorkflowEmitMode::Event,
            ) => {
                let sequence = match self
                    .event_sequences
                    .get(&(output.to_string(), stream.clone()))
                    .copied()
                {
                    Some(sequence) => sequence.checked_add(1).with_context(|| {
                        format!(
                            "workflow event output '{output}' stream '{}' exhausted its sequence cursor",
                            stream.0
                        )
                    })?,
                    None => 1,
                };
                let payload = payload.with_context(|| {
                    format!("workflow emit Event for output '{output}' has no payload")
                })?;
                self.event_sequences
                    .insert((output.to_string(), stream.clone()), sequence);
                self.publications.push(WorkflowOutputPublication::Event {
                    output: output.to_string(),
                    stream,
                    sequence: OutputSequence(sequence),
                    payload,
                    finality: OutputFinality::Provisional,
                });
            }
            (
                OutputProtocolFamily::Declared(WorkflowOutputFamily::Revisions { .. }),
                WorkflowEmitMode::Append
                | WorkflowEmitMode::Replace
                | WorkflowEmitMode::Retract
                | WorkflowEmitMode::Finalize,
            ) => {
                let operation = match mode {
                    WorkflowEmitMode::Append => TypedRevisionOperation::Append,
                    WorkflowEmitMode::Replace => TypedRevisionOperation::Replace,
                    WorkflowEmitMode::Retract => TypedRevisionOperation::Retract,
                    WorkflowEmitMode::Finalize => TypedRevisionOperation::Finalize,
                    _ => unreachable!("matched revision emit operation"),
                };
                self.publish_revision(output, stream, operation, payload)?;
            }
            _ => bail!(
                "workflow emit {mode:?} targets output '{output}' with family {family:?}; \
                 select an operation legal for that output family"
            ),
        }
        Ok(())
    }

    fn publish_revision(
        &mut self,
        output: &str,
        stream: OutputStreamId,
        operation: TypedRevisionOperation,
        payload: Option<Value>,
    ) -> Result<()> {
        let state = self.revisions.state(output, &stream);
        let sequence = state.sequence.checked_add(1).with_context(|| {
            format!(
                "workflow revision output '{output}' stream '{}' exhausted its sequence cursor",
                stream.0
            )
        })?;
        let revision = state.revision.checked_add(1).with_context(|| {
            format!(
                "workflow revision output '{output}' stream '{}' exhausted its revision cursor",
                stream.0
            )
        })?;
        let envelope = TypedRevisionEnvelope {
            version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
            transaction: self.transaction,
            output: output.to_string(),
            stream,
            sequence: OutputSequence(sequence),
            revision: OutputRevision(revision),
            lineage: OutputLineage(match operation {
                TypedRevisionOperation::Append | TypedRevisionOperation::Replace => revision,
                TypedRevisionOperation::Retract | TypedRevisionOperation::Finalize => state.lineage,
            }),
            base: OutputLineage(state.lineage),
            operation,
            payload,
            finality: OutputFinality::Provisional,
        };
        self.revisions
            .validate_and_apply(&envelope)
            .map_err(anyhow::Error::from)?;
        match envelope.operation {
            TypedRevisionOperation::Append | TypedRevisionOperation::Replace => {
                self.revision_payloads.insert(
                    (output.to_string(), envelope.stream.clone()),
                    envelope.payload.as_ref().map(clone_value).transpose()?,
                );
            }
            TypedRevisionOperation::Retract => {
                self.revision_payloads
                    .insert((output.to_string(), envelope.stream.clone()), None);
            }
            TypedRevisionOperation::Finalize => {}
        }
        self.publications
            .push(WorkflowOutputPublication::Revision(envelope));
        Ok(())
    }

    /// Turn commit is the only operation that grants finality. Every open
    /// revision stream receives its default close before the caller exposes the
    /// committed journal.
    pub(crate) fn finalize_on_commit(&mut self) -> Result<()> {
        let open = self
            .revisions
            .streams
            .iter()
            .filter(|(_, state)| !state.closed)
            .map(|((output, stream), _)| (output.clone(), stream.clone()))
            .collect::<Vec<_>>();
        for (output, stream) in open {
            self.publish_revision(&output, stream, TypedRevisionOperation::Finalize, None)?;
        }
        if self.publication_mode == TurnPublicationMode::CommitOnly {
            for publication in &mut self.publications {
                match publication {
                    WorkflowOutputPublication::Materialized { finality, .. }
                    | WorkflowOutputPublication::Event { finality, .. } => {
                        *finality = OutputFinality::Final
                    }
                    WorkflowOutputPublication::Revision(envelope) => {
                        envelope.finality = OutputFinality::Final;
                    }
                    WorkflowOutputPublication::TransactionCommitted { .. }
                    | WorkflowOutputPublication::AbortToBaseline { .. } => {}
                }
            }
        }
        Ok(())
    }

    /// Record the committed transaction after its state/effect/output writes
    /// succeed. This record is the finality authority for provisional output.
    pub(crate) fn record_commit(&mut self, outcome: &TurnTransactionOutcome) {
        if self.publication_mode == TurnPublicationMode::ProvisionalRevisions {
            self.publications
                .push(WorkflowOutputPublication::from_transaction_outcome(outcome));
        }
    }

    /// Take the not-yet-delivered provisional publications in authored order.
    /// A commit-only journal never exposes this pre-commit view.
    pub(crate) fn take_pending_provisionals(&mut self) -> Result<Vec<WorkflowOutputPublication>> {
        if self.publication_mode != TurnPublicationMode::ProvisionalRevisions {
            return Ok(Vec::new());
        }
        let pending = self.publications[self.provisional_delivery_cursor..]
            .iter()
            .map(WorkflowOutputPublication::try_clone)
            .collect::<Result<Vec<_>>>()?;
        self.provisional_delivery_cursor = self.publications.len();
        Ok(pending)
    }

    /// Construct the sole exact abort record from the journal's complete
    /// touched-stream set. The transaction remains the authority for baseline
    /// cursors, while the journal knows dynamically named streams.
    pub(crate) fn abort_outcome(
        &self,
        turn: &super::TurnTransaction,
        reason: TurnAbortReason,
    ) -> TurnTransactionOutcome {
        let streams = self
            .revision_payloads
            .keys()
            .chain(self.revisions.streams.keys())
            .chain(self.event_sequences.keys())
            .chain(self.materialized.keys())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        turn.abort_for_streams(reason, streams)
    }

    pub(crate) fn committed_states(
        &self,
    ) -> Result<BTreeMap<(String, OutputStreamId), CommittedOutputState>> {
        let mut states = self
            .materialized
            .iter()
            .map(|(identity, state)| {
                Ok((
                    identity.clone(),
                    CommittedOutputState {
                        head: state.head,
                        cursor: state.cursor,
                        lineage: state.lineage,
                        closed: state.closed,
                        payload: state.payload.as_ref().map(clone_value).transpose()?,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        for (identity, sequence) in &self.event_sequences {
            states.insert(
                identity.clone(),
                CommittedOutputState {
                    head: *sequence,
                    cursor: *sequence,
                    lineage: *sequence,
                    closed: false,
                    payload: None,
                },
            );
        }
        for (identity, stream) in &self.revisions.streams {
            states.insert(
                identity.clone(),
                CommittedOutputState {
                    head: stream.revision,
                    cursor: stream.sequence,
                    lineage: stream.lineage,
                    closed: stream.closed,
                    payload: self
                        .revision_payloads
                        .get(identity)
                        .and_then(Option::as_ref)
                        .map(clone_value)
                        .transpose()?,
                },
            );
        }
        Ok(states)
    }

    pub(crate) fn take(self) -> Vec<WorkflowOutputPublication> {
        self.publications
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::TurnTransaction;
    use proptest::prelude::*;

    fn value(value: i64) -> Value {
        Value::from_slice_i64(&[value], &[1]).expect("test value")
    }

    fn revision_workflow(
        publication_mode: onnx_genai_metadata::WorkflowPublicationMode,
    ) -> WorkflowSpec {
        WorkflowSpec {
            manifest: onnx_genai_metadata::WorkflowManifest {
                adapter_abis: Default::default(),
            },
            publication_mode,
            inputs: Default::default(),
            outputs: ["answer", "tool"]
                .into_iter()
                .map(|name| {
                    (
                        name.to_string(),
                        onnx_genai_metadata::WorkflowOutput {
                            contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")
                                .expect("output contract"),
                            role: onnx_genai_metadata::WorkflowOutputRole::Tensor,
                            family: WorkflowOutputFamily::Revisions {
                                version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
                            },
                            family_authored: true,
                            value_range: None,
                            stage: onnx_genai_metadata::OutputStage::PreAdapter,
                            media: None,
                        },
                    )
                })
                .collect(),
            components: Default::default(),
            state: Default::default(),
            effects: Default::default(),
            serving: None,
            steps: Default::default(),
        }
    }

    fn revisions() -> RevisionEnvelopeValidator {
        RevisionEnvelopeValidator::new([(
            "answer".to_string(),
            WorkflowOutputFamily::Revisions {
                version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
            },
        )])
    }

    type CommittedStateFingerprint = (String, String, u64, u64, u64, bool, Option<Vec<u8>>);

    fn committed_state_fingerprint(
        states: BTreeMap<(String, OutputStreamId), CommittedOutputState>,
    ) -> Vec<CommittedStateFingerprint> {
        states
            .into_iter()
            .map(|((output, stream), state)| {
                (
                    output,
                    stream.0,
                    state.head,
                    state.cursor,
                    state.lineage,
                    state.closed,
                    state
                        .payload
                        .map(|payload| payload.to_raw_bytes().expect("payload bytes")),
                )
            })
            .collect()
    }

    fn envelope(
        sequence: u64,
        revision: u64,
        lineage: u64,
        base: u64,
        operation: TypedRevisionOperation,
        payload: Option<Value>,
    ) -> TypedRevisionEnvelope {
        TypedRevisionEnvelope {
            version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
            transaction: TurnTransactionId(7),
            output: "answer".to_string(),
            stream: OutputStreamId("main".to_string()),
            sequence: OutputSequence(sequence),
            revision: OutputRevision(revision),
            lineage: OutputLineage(lineage),
            base: OutputLineage(base),
            operation,
            payload,
            finality: OutputFinality::Provisional,
        }
    }

    #[test]
    fn revision_validator_rejects_invalid_base_before_cursor_mutates() {
        let mut validator = revisions();
        validator
            .validate_and_apply(&envelope(
                1,
                1,
                1,
                0,
                TypedRevisionOperation::Append,
                Some(value(1)),
            ))
            .expect("first revision");
        let error = validator
            .validate_and_apply(&envelope(
                2,
                2,
                2,
                0,
                TypedRevisionOperation::Replace,
                Some(value(2)),
            ))
            .expect_err("stale base");
        assert!(matches!(
            error,
            RevisionEnvelopeValidationError::Base { .. }
        ));
        validator
            .validate_and_apply(&envelope(
                2,
                2,
                2,
                1,
                TypedRevisionOperation::Replace,
                Some(value(2)),
            ))
            .expect("state did not advance after rejected envelope");
    }

    #[test]
    fn finalize_closes_a_stream_and_rejects_following_publication() {
        let mut validator = revisions();
        validator
            .validate_and_apply(&envelope(
                1,
                1,
                1,
                0,
                TypedRevisionOperation::Append,
                Some(value(1)),
            ))
            .expect("append");
        validator
            .validate_and_apply(&envelope(
                2,
                2,
                1,
                1,
                TypedRevisionOperation::Finalize,
                None,
            ))
            .expect("finalize");
        let error = validator
            .validate_and_apply(&envelope(
                3,
                3,
                3,
                1,
                TypedRevisionOperation::Append,
                Some(value(2)),
            ))
            .expect_err("post-close append");
        assert!(matches!(
            error,
            RevisionEnvelopeValidationError::Closed { .. }
        ));
    }

    #[test]
    fn retract_requires_the_current_lineage_and_restores_the_empty_head() {
        let mut validator = revisions();
        validator
            .validate_and_apply(&envelope(
                1,
                1,
                1,
                0,
                TypedRevisionOperation::Append,
                Some(value(1)),
            ))
            .expect("append");
        validator
            .validate_and_apply(&envelope(2, 2, 1, 1, TypedRevisionOperation::Retract, None))
            .expect("retract current lineage");
        validator
            .validate_and_apply(&envelope(
                3,
                3,
                3,
                0,
                TypedRevisionOperation::Append,
                Some(value(3)),
            ))
            .expect("append after retract starts from baseline");
    }

    #[test]
    fn unknown_protocol_version_cannot_advance_a_stream() {
        let mut validator = revisions();
        let mut unknown = envelope(1, 1, 1, 0, TypedRevisionOperation::Append, Some(value(1)));
        unknown.version = "2".to_string();
        assert!(matches!(
            validator.validate_and_apply(&unknown),
            Err(RevisionEnvelopeValidationError::UnknownVersion { .. })
        ));
        validator
            .validate_and_apply(&envelope(
                1,
                1,
                1,
                0,
                TypedRevisionOperation::Append,
                Some(value(1)),
            ))
            .expect("rejected version did not advance the cursor");
    }

    #[test]
    fn success_shaped_finality_cannot_precede_the_transaction_commit() {
        let mut validator = revisions();
        let mut final_envelope =
            envelope(1, 1, 1, 0, TypedRevisionOperation::Append, Some(value(1)));
        final_envelope.finality = OutputFinality::Final;
        assert!(matches!(
            validator.validate_and_apply(&final_envelope),
            Err(RevisionEnvelopeValidationError::Finality { .. })
        ));
        validator
            .validate_and_apply(&envelope(
                1,
                1,
                1,
                0,
                TypedRevisionOperation::Append,
                Some(value(1)),
            ))
            .expect("rejected finality cannot advance the stream");
    }

    #[test]
    fn invalid_revision_baseline_is_refused_before_a_journal_can_publish() -> Result<()> {
        let workflow =
            revision_workflow(onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions);
        let error = match OutputPublicationJournal::new(
            TurnTransactionId(44),
            &workflow,
            BTreeMap::from([(
                ("answer".to_string(), OutputStreamId("main".to_string())),
                OutputPublicationBaseline {
                    head: 1,
                    cursor: 2,
                    lineage: 1,
                    closed: false,
                    payload: Some(value(1)),
                },
            )]),
        ) {
            Ok(_) => panic!("a malformed admitted baseline must be unreconcilable"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains("output 'answer' stream 'main'")
                && format!("{error:#}").contains("head 1 differs from sequence 2"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn committed_closed_baseline_rejects_post_close_revision_before_publication() -> Result<()> {
        let workflow = WorkflowSpec {
            manifest: onnx_genai_metadata::WorkflowManifest {
                adapter_abis: Default::default(),
            },
            publication_mode: Default::default(),
            inputs: Default::default(),
            outputs: BTreeMap::from([(
                "answer".to_string(),
                onnx_genai_metadata::WorkflowOutput {
                    contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")?,
                    role: onnx_genai_metadata::WorkflowOutputRole::Tensor,
                    family: WorkflowOutputFamily::Revisions {
                        version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
                    },
                    family_authored: true,
                    value_range: None,
                    stage: onnx_genai_metadata::OutputStage::PreAdapter,
                    media: None,
                },
            )]),
            components: Default::default(),
            state: Default::default(),
            effects: Default::default(),
            serving: None,
            steps: Default::default(),
        };
        let mut journal = OutputPublicationJournal::new(
            TurnTransactionId(8),
            &workflow,
            BTreeMap::from([(
                ("answer".to_string(), OutputStreamId("answer".to_string())),
                OutputPublicationBaseline {
                    closed: true,
                    ..Default::default()
                },
            )]),
        )?;
        assert!(
            journal
                .publish("answer", None, &WorkflowEmitMode::Append, Some(value(1)),)
                .is_err()
        );
        assert!(journal.take().is_empty());
        Ok(())
    }

    #[test]
    fn streams_interleave_without_sharing_cursors() -> Result<()> {
        let workflow = WorkflowSpec {
            manifest: onnx_genai_metadata::WorkflowManifest {
                adapter_abis: Default::default(),
            },
            publication_mode: Default::default(),
            inputs: Default::default(),
            outputs: BTreeMap::from([(
                "answer".to_string(),
                onnx_genai_metadata::WorkflowOutput {
                    contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")?,
                    role: onnx_genai_metadata::WorkflowOutputRole::Tensor,
                    family: WorkflowOutputFamily::Revisions {
                        version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
                    },
                    family_authored: true,
                    value_range: None,
                    stage: onnx_genai_metadata::OutputStage::PreAdapter,
                    media: None,
                },
            )]),
            components: Default::default(),
            state: Default::default(),
            effects: Default::default(),
            serving: None,
            steps: Default::default(),
        };
        let mut journal =
            OutputPublicationJournal::new(TurnTransactionId(2), &workflow, BTreeMap::new())?;
        journal.publish(
            "answer",
            Some("left"),
            &WorkflowEmitMode::Append,
            Some(value(1)),
        )?;
        journal.publish(
            "answer",
            Some("right"),
            &WorkflowEmitMode::Append,
            Some(value(2)),
        )?;
        journal.publish(
            "answer",
            Some("left"),
            &WorkflowEmitMode::Replace,
            Some(value(3)),
        )?;
        journal.finalize_on_commit()?;
        let publications = journal.take();
        let headers = publications
            .iter()
            .filter_map(|publication| match publication {
                WorkflowOutputPublication::Revision(envelope) => Some((
                    envelope.stream.0.as_str(),
                    envelope.sequence.0,
                    envelope.finality,
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            headers,
            vec![
                ("left", 1, OutputFinality::Final),
                ("right", 1, OutputFinality::Final),
                ("left", 2, OutputFinality::Final),
                ("left", 3, OutputFinality::Final),
                ("right", 2, OutputFinality::Final),
            ]
        );
        Ok(())
    }

    #[test]
    fn committed_named_streams_restore_exact_heads_payloads_and_closure() -> Result<()> {
        let workflow = WorkflowSpec {
            manifest: onnx_genai_metadata::WorkflowManifest {
                adapter_abis: Default::default(),
            },
            publication_mode: Default::default(),
            inputs: Default::default(),
            outputs: BTreeMap::from([(
                "answer".to_string(),
                onnx_genai_metadata::WorkflowOutput {
                    contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")?,
                    role: onnx_genai_metadata::WorkflowOutputRole::Tensor,
                    family: WorkflowOutputFamily::Revisions {
                        version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
                    },
                    family_authored: true,
                    value_range: None,
                    stage: onnx_genai_metadata::OutputStage::PreAdapter,
                    media: None,
                },
            )]),
            components: Default::default(),
            state: Default::default(),
            effects: Default::default(),
            serving: None,
            steps: Default::default(),
        };
        let mut first =
            OutputPublicationJournal::new(TurnTransactionId(10), &workflow, BTreeMap::new())?;
        first.publish(
            "answer",
            Some("analysis"),
            &WorkflowEmitMode::Append,
            Some(value(1)),
        )?;
        first.publish(
            "answer",
            Some("final"),
            &WorkflowEmitMode::Replace,
            Some(value(2)),
        )?;
        first.publish("answer", Some("analysis"), &WorkflowEmitMode::Retract, None)?;
        first.publish(
            "answer",
            Some("analysis"),
            &WorkflowEmitMode::Append,
            Some(value(3)),
        )?;
        first.publish("answer", Some("final"), &WorkflowEmitMode::Finalize, None)?;
        first.finalize_on_commit()?;
        let committed = first.committed_states()?;
        let analysis = &committed[&("answer".to_string(), OutputStreamId("analysis".to_string()))];
        let final_stream = &committed[&("answer".to_string(), OutputStreamId("final".to_string()))];
        assert_eq!(
            (
                analysis.head,
                analysis.cursor,
                analysis.lineage,
                analysis.closed,
                analysis
                    .payload
                    .as_ref()
                    .expect("analysis payload")
                    .to_vec_i64()?,
            ),
            (4, 4, 3, true, vec![3])
        );
        assert_eq!(
            (
                final_stream.head,
                final_stream.cursor,
                final_stream.lineage,
                final_stream.closed,
                final_stream
                    .payload
                    .as_ref()
                    .expect("final payload")
                    .to_vec_i64()?,
            ),
            (2, 2, 1, true, vec![2])
        );

        let baselines = committed
            .iter()
            .map(|(identity, state)| {
                Ok((
                    identity.clone(),
                    OutputPublicationBaseline {
                        head: state.head,
                        cursor: state.cursor,
                        lineage: state.lineage,
                        closed: state.closed,
                        payload: state.payload.as_ref().map(clone_value).transpose()?,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut second =
            OutputPublicationJournal::new(TurnTransactionId(11), &workflow, baselines)?;
        let error = second
            .publish(
                "answer",
                Some("analysis"),
                &WorkflowEmitMode::Append,
                Some(value(9)),
            )
            .expect_err("a committed closed stream must remain closed next turn");
        assert!(format!("{error:#}").contains("finalized output 'answer' stream 'analysis'"));

        second.publish(
            "answer",
            Some("retry"),
            &WorkflowEmitMode::Replace,
            Some(value(7)),
        )?;
        let aborted = second.committed_states()?;
        assert!(aborted.contains_key(&("answer".to_string(), OutputStreamId("retry".to_string()))));

        let retry_baselines = committed
            .iter()
            .map(|(identity, state)| {
                Ok((
                    identity.clone(),
                    OutputPublicationBaseline {
                        head: state.head,
                        cursor: state.cursor,
                        lineage: state.lineage,
                        closed: state.closed,
                        payload: state.payload.as_ref().map(clone_value).transpose()?,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut retry =
            OutputPublicationJournal::new(TurnTransactionId(12), &workflow, retry_baselines)?;
        retry.publish(
            "answer",
            Some("retry"),
            &WorkflowEmitMode::Replace,
            Some(value(7)),
        )?;
        retry.finalize_on_commit()?;
        let retried = retry.committed_states()?;
        let retry = &retried[&("answer".to_string(), OutputStreamId("retry".to_string()))];
        assert_eq!(
            (
                retry.head,
                retry.cursor,
                retry.lineage,
                retry.closed,
                retry
                    .payload
                    .as_ref()
                    .expect("retry payload")
                    .to_vec_i64()?,
            ),
            (2, 2, 1, true, vec![7])
        );
        Ok(())
    }

    #[test]
    fn materialized_events_and_revisions_are_all_observed_with_commit_finality() -> Result<()> {
        let output = |family| onnx_genai_metadata::WorkflowOutput {
            contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")
                .expect("output contract"),
            role: onnx_genai_metadata::WorkflowOutputRole::Tensor,
            family,
            family_authored: true,
            value_range: None,
            stage: onnx_genai_metadata::OutputStage::PreAdapter,
            media: None,
        };
        let workflow = WorkflowSpec {
            manifest: onnx_genai_metadata::WorkflowManifest {
                adapter_abis: Default::default(),
            },
            publication_mode: Default::default(),
            inputs: Default::default(),
            // Deliberately insert in a different order from publication. The
            // journal follows authored emits, not map traversal.
            outputs: BTreeMap::from([
                (
                    "value".to_string(),
                    output(WorkflowOutputFamily::Materialized),
                ),
                ("event".to_string(), output(WorkflowOutputFamily::Events)),
                (
                    "revision".to_string(),
                    output(WorkflowOutputFamily::Revisions {
                        version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
                    }),
                ),
            ]),
            components: Default::default(),
            state: Default::default(),
            effects: Default::default(),
            serving: None,
            steps: Default::default(),
        };
        let mut journal =
            OutputPublicationJournal::new(TurnTransactionId(3), &workflow, BTreeMap::new())?;
        journal.publish(
            "event",
            Some("updates"),
            &WorkflowEmitMode::Event,
            Some(value(1)),
        )?;
        journal.publish("value", None, &WorkflowEmitMode::Append, Some(value(2)))?;
        journal.publish(
            "revision",
            Some("main"),
            &WorkflowEmitMode::Replace,
            Some(value(3)),
        )?;
        journal.finalize_on_commit()?;
        let publications = journal.take();
        assert_eq!(publications.len(), 4);
        assert!(matches!(
            &publications[0],
            WorkflowOutputPublication::Event {
                output,
                sequence: OutputSequence(1),
                finality: OutputFinality::Final,
                ..
            } if output == "event"
        ));
        assert!(matches!(
            &publications[1],
            WorkflowOutputPublication::Materialized {
                output,
                operation: TypedRevisionOperation::Append,
                finality: OutputFinality::Final,
                ..
            } if output == "value"
        ));
        assert!(matches!(
            &publications[2],
            WorkflowOutputPublication::Revision(TypedRevisionEnvelope {
                output,
                stream: OutputStreamId(stream),
                sequence: OutputSequence(1),
                finality: OutputFinality::Final,
                ..
            }) if output == "revision" && stream == "main"
        ));
        assert!(matches!(
            &publications[3],
            WorkflowOutputPublication::Revision(TypedRevisionEnvelope {
                operation: TypedRevisionOperation::Finalize,
                finality: OutputFinality::Final,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn legacy_implicit_outputs_preserve_per_site_event_and_materialized_execution() -> Result<()> {
        let output = || onnx_genai_metadata::WorkflowOutput {
            contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")
                .expect("output contract"),
            role: onnx_genai_metadata::WorkflowOutputRole::Tensor,
            family: WorkflowOutputFamily::Materialized,
            family_authored: false,
            value_range: None,
            stage: onnx_genai_metadata::OutputStage::PreAdapter,
            media: None,
        };
        let workflow = WorkflowSpec {
            manifest: onnx_genai_metadata::WorkflowManifest {
                adapter_abis: Default::default(),
            },
            publication_mode: Default::default(),
            inputs: Default::default(),
            outputs: BTreeMap::from([
                ("legacy_event".to_string(), output()),
                ("legacy_value".to_string(), output()),
            ]),
            components: Default::default(),
            state: Default::default(),
            effects: Default::default(),
            serving: None,
            steps: Default::default(),
        };
        let mut journal =
            OutputPublicationJournal::new(TurnTransactionId(19), &workflow, BTreeMap::new())?;
        journal.publish(
            "legacy_event",
            None,
            &WorkflowEmitMode::Event,
            Some(value(3)),
        )?;
        journal.publish(
            "legacy_value",
            None,
            &WorkflowEmitMode::Replace,
            Some(value(7)),
        )?;
        journal.finalize_on_commit()?;

        let publications = journal.take();
        assert!(matches!(
            &publications[0],
            WorkflowOutputPublication::Event {
                output,
                finality: OutputFinality::Final,
                ..
            } if output == "legacy_event"
        ));
        assert!(matches!(
            &publications[1],
            WorkflowOutputPublication::Materialized {
                output,
                finality: OutputFinality::Final,
                ..
            } if output == "legacy_value"
        ));
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128,
            .. ProptestConfig::default()
        })]

        /// Random interleavings exercise the one authority that owns stream
        /// cursor, lineage, closure, finality, and the terminal outcome.
        #[test]
        fn revision_transactions_preserve_stream_isolation_and_baselines(
            actions in prop::collection::vec((0u8..2, 0u8..2, 0u8..4, any::<i64>()), 1..48),
        ) {
            for publication_mode in [
                onnx_genai_metadata::WorkflowPublicationMode::CommitOnly,
                onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions,
            ] {
                let workflow = revision_workflow(publication_mode);
                let turn = TurnTransaction::admit(
                    TurnTransactionId(41),
                    None,
                    &onnx_genai_metadata::ResolvedStatePlan::default(),
                    std::iter::empty::<String>(),
                    ["answer".to_string(), "tool".to_string()],
                    &std::collections::HashMap::new(),
                    &std::collections::HashMap::new(),
                    &std::collections::HashMap::new(),
                    match publication_mode {
                        onnx_genai_metadata::WorkflowPublicationMode::CommitOnly => TurnPublicationMode::CommitOnly,
                        onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions => TurnPublicationMode::ProvisionalRevisions,
                    },
                ).expect("revision-only provisional workflow admits");
                let mut journal = OutputPublicationJournal::new(
                    turn.id(),
                    &workflow,
                    turn.output_baselines().expect("snapshot baselines"),
                ).expect("revision protocol is admissible");
                let mut model = std::collections::BTreeMap::new();

                for (output_index, stream_index, operation, payload) in actions.iter().copied() {
                    let output = if output_index == 0 { "answer" } else { "tool" };
                    let stream = if stream_index == 0 { "left" } else { "right" };
                    let state = model
                        .entry((output, stream))
                        .or_insert((0u64, 0u64, 0u64, false));
                    if state.3 {
                        let before = committed_state_fingerprint(
                            journal.committed_states().expect("state snapshots"),
                        );
                        prop_assert!(journal.publish(
                            output,
                            Some(stream),
                            &WorkflowEmitMode::Append,
                            Some(value(payload)),
                        ).is_err());
                        prop_assert_eq!(
                            committed_state_fingerprint(
                                journal.committed_states()
                                    .expect("rejected operation is atomic"),
                            ),
                            before,
                        );
                        continue;
                    }
                    let mode = match operation {
                        0 => WorkflowEmitMode::Append,
                        1 => WorkflowEmitMode::Replace,
                        2 if state.2 != 0 => WorkflowEmitMode::Retract,
                        _ => WorkflowEmitMode::Finalize,
                    };
                    let carries_payload = matches!(mode, WorkflowEmitMode::Append | WorkflowEmitMode::Replace);
                    journal.publish(
                        output,
                        Some(stream),
                        &mode,
                        carries_payload.then(|| value(payload)),
                    ).expect("the generated legal operation publishes");
                    state.0 += 1;
                    state.1 += 1;
                    match mode {
                        WorkflowEmitMode::Append | WorkflowEmitMode::Replace => state.2 = state.0,
                        WorkflowEmitMode::Retract => state.2 = 0,
                        WorkflowEmitMode::Finalize => state.3 = true,
                        WorkflowEmitMode::Event => unreachable!("revision strategy omits events"),
                    }
                }

                let provisional = journal.take_pending_provisionals().expect("copy envelopes");
                match publication_mode {
                    onnx_genai_metadata::WorkflowPublicationMode::CommitOnly => {
                        prop_assert!(provisional.is_empty());
                    }
                    onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions => {
                        for publication in provisional {
                            let WorkflowOutputPublication::Revision(envelope) = publication else {
                                prop_assert!(false, "only revisions publish before commit");
                                continue;
                            };
                            prop_assert_eq!(envelope.finality, OutputFinality::Provisional);
                        }
                    }
                }

                let abort = journal.abort_outcome(
                    &turn,
                    TurnAbortReason::Cancellation,
                );
                let TurnTransactionOutcome::AbortToBaseline { streams, .. } = abort else {
                    prop_assert!(false, "an aborted turn must name its baseline");
                    continue;
                };
                for baseline in streams {
                    prop_assert_eq!(baseline.head, 0);
                    prop_assert_eq!(baseline.sequence, 0);
                    prop_assert_eq!(baseline.lineage, 0);
                    prop_assert!(!baseline.closed);
                }

                journal.finalize_on_commit().expect("commit default closes streams");
                journal.record_commit(&turn.committed());
                let publications = journal.take();
                for publication in &publications {
                    match (publication_mode, publication) {
                        (_, WorkflowOutputPublication::TransactionCommitted { .. }) => {}
                        (onnx_genai_metadata::WorkflowPublicationMode::CommitOnly,
                         WorkflowOutputPublication::Materialized { finality, .. }
                         | WorkflowOutputPublication::Event { finality, .. }) => {
                            prop_assert_eq!(*finality, OutputFinality::Final);
                        }
                        (onnx_genai_metadata::WorkflowPublicationMode::CommitOnly,
                         WorkflowOutputPublication::Revision(envelope)) => {
                            prop_assert_eq!(envelope.finality, OutputFinality::Final);
                        }
                        (onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions,
                         WorkflowOutputPublication::Revision(envelope)) => {
                            prop_assert_eq!(envelope.finality, OutputFinality::Provisional);
                        }
                        (_, WorkflowOutputPublication::AbortToBaseline { .. }) => {
                            prop_assert!(false, "successful commit never emits an abort");
                        }
                        (onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions, _) => {
                            prop_assert!(false, "provisional mode accepts only revision outputs");
                        }
                    }
                }
                if publication_mode == onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions {
                    let last_is_commit = matches!(
                        publications.last(),
                        Some(WorkflowOutputPublication::TransactionCommitted { .. })
                    );
                    prop_assert!(last_is_commit);
                }
            }
        }

        #[test]
        fn duplicate_or_out_of_order_revisions_fail_without_advancing(
            payload in any::<i64>(),
            bad_sequence in 0u64..2,
        ) {
            let mut validator = revisions();
            validator.validate_and_apply(&envelope(
                1,
                1,
                1,
                0,
                TypedRevisionOperation::Append,
                Some(value(payload)),
            )).expect("first revision");
            let duplicate = envelope(
                bad_sequence,
                2,
                2,
                1,
                TypedRevisionOperation::Replace,
                Some(value(payload.saturating_add(1))),
            );
            prop_assert!(validator.validate_and_apply(&duplicate).is_err());
            validator.validate_and_apply(&envelope(
                2,
                2,
                2,
                1,
                TypedRevisionOperation::Replace,
                Some(value(payload.saturating_add(1))),
            )).expect("illegal input cannot mutate the next valid cursor");
        }
    }
}
