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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputPublicationJournalResolution {
    Committed,
    Aborted,
}

impl std::fmt::Display for OutputPublicationJournalResolution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Committed => formatter.write_str("committed"),
            Self::Aborted => formatter.write_str("aborted"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "output publication journal for transaction {transaction:?} is already {resolution}; \
     cannot {operation}. Admit a new turn before publishing or resolving more output"
)]
struct OutputPublicationJournalResolutionError {
    transaction: TurnTransactionId,
    resolution: OutputPublicationJournalResolution,
    operation: &'static str,
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
    resolution: Option<OutputPublicationJournalResolution>,
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
            resolution: None,
        })
    }

    fn ensure_open(
        &self,
        operation: &'static str,
    ) -> std::result::Result<(), OutputPublicationJournalResolutionError> {
        match self.resolution {
            None => Ok(()),
            Some(resolution) => Err(OutputPublicationJournalResolutionError {
                transaction: self.transaction,
                resolution,
                operation,
            }),
        }
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
        self.ensure_open("validate an output emission")?;
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
        self.ensure_open("stage a typed revision")?;
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
        self.ensure_open("stage commit finalization")?;
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
    pub(crate) fn record_commit(&mut self, outcome: &TurnTransactionOutcome) -> Result<()> {
        self.ensure_open("record a committed transaction")?;
        let TurnTransactionOutcome::Committed { transaction, .. } = outcome else {
            bail!(
                "output publication journal for transaction {:?} cannot record a commit from \
                 abort outcome {outcome:?}",
                self.transaction
            );
        };
        ensure!(
            *transaction == self.transaction,
            "output publication journal for transaction {:?} cannot record commit outcome for \
             transaction {transaction:?}",
            self.transaction
        );
        if self.publication_mode == TurnPublicationMode::ProvisionalRevisions {
            self.publications
                .push(WorkflowOutputPublication::from_transaction_outcome(outcome));
        }
        self.resolution = Some(OutputPublicationJournalResolution::Committed);
        Ok(())
    }

    /// Take the not-yet-delivered provisional publications in authored order.
    /// A commit-only journal never exposes this pre-commit view.
    pub(crate) fn take_pending_provisionals(&mut self) -> Result<Vec<WorkflowOutputPublication>> {
        self.ensure_open("deliver provisional publications")?;
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
        &mut self,
        turn: &mut super::TurnTransaction,
        reason: TurnAbortReason,
    ) -> Result<TurnTransactionOutcome> {
        self.ensure_open("abort to the admitted output baseline")?;
        ensure!(
            turn.id() == self.transaction,
            "output publication journal for transaction {:?} cannot abort turn {:?}; pass the \
             transaction that admitted this journal",
            self.transaction,
            turn.id()
        );
        let streams = self
            .revision_payloads
            .keys()
            .chain(self.revisions.streams.keys())
            .chain(self.event_sequences.keys())
            .chain(self.materialized.keys())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let outcome = turn.abort_for_streams(reason, streams)?;
        if self.publication_mode == TurnPublicationMode::ProvisionalRevisions {
            self.publications
                .push(WorkflowOutputPublication::from_transaction_outcome(
                    &outcome,
                ));
        }
        self.resolution = Some(OutputPublicationJournalResolution::Aborted);
        Ok(outcome)
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
            publication_mode_authored: true,
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
            state: BTreeMap::from([(
                "memory".to_string(),
                onnx_genai_metadata::WorkflowStateCell {
                    contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")
                        .expect("state contract"),
                    class: onnx_genai_metadata::WorkflowStateClass::Semantic,
                    scope: onnx_genai_metadata::WorkflowStateScope::Session,
                    initializer: "prompt".to_string(),
                    recurrence: onnx_genai_metadata::ShapeRecurrence::Invariant,
                    management: onnx_genai_metadata::StateManagement::Runtime,
                    release_boundary: Some(onnx_genai_metadata::StateReleaseBoundary::Session),
                    service_group: None,
                    session: Some(onnx_genai_metadata::SessionLeaseContract {
                        policy: onnx_genai_metadata::SessionMutationPolicy::Exclusive,
                        ttl_seconds: None,
                        optimistic_metadata_version: false,
                        continuation: Some(
                            onnx_genai_metadata::SessionContinuation::PromptPrefix {
                                prompt_input: "prompt".to_string(),
                                tokens_output: "answer".to_string(),
                            },
                        ),
                    }),
                },
            )]),
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

    fn assert_journal_resolution_error(
        error: &anyhow::Error,
        resolution: OutputPublicationJournalResolution,
        operation: &'static str,
    ) {
        let error = error
            .downcast_ref::<OutputPublicationJournalResolutionError>()
            .expect("journal misuse returns its typed terminal-resolution error");
        assert_eq!(
            error,
            &OutputPublicationJournalResolutionError {
                transaction: TurnTransactionId(71),
                resolution,
                operation,
            }
        );
        assert!(
            error.to_string().contains("Admit a new turn"),
            "terminal diagnostic must tell the caller how to recover: {error}"
        );
    }

    fn assert_turn_resolution_error(
        error: crate::pipeline::TurnTransactionResolutionError,
        resolution: OutputPublicationJournalResolution,
        operation: &'static str,
    ) {
        let resolution = match resolution {
            OutputPublicationJournalResolution::Committed => {
                crate::pipeline::TurnTransactionResolution::Committed
            }
            OutputPublicationJournalResolution::Aborted => {
                crate::pipeline::TurnTransactionResolution::Aborted
            }
        };
        assert_eq!(
            error,
            crate::pipeline::TurnTransactionResolutionError {
                transaction: TurnTransactionId(71),
                resolution,
                operation,
            }
        );
        assert!(
            error.to_string().contains("Admit a new turn"),
            "terminal diagnostic must tell the caller how to recover: {error}"
        );
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TransactionTestOperation {
        Emit,
        Replace,
        Retract,
        Finalize,
    }

    impl TransactionTestOperation {
        fn mode(self) -> WorkflowEmitMode {
            match self {
                Self::Emit => WorkflowEmitMode::Append,
                Self::Replace => WorkflowEmitMode::Replace,
                Self::Retract => WorkflowEmitMode::Retract,
                Self::Finalize => WorkflowEmitMode::Finalize,
            }
        }

        fn carries_payload(self) -> bool {
            matches!(self, Self::Emit | Self::Replace)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TransactionTestAction {
        output: u8,
        stream: u8,
        operation: TransactionTestOperation,
        payload: i64,
    }

    impl TransactionTestAction {
        fn identity(self) -> (&'static str, &'static str) {
            (
                if self.output == 0 { "answer" } else { "tool" },
                if self.stream == 0 { "left" } else { "right" },
            )
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TransactionTestStep {
        Output(TransactionTestAction),
        Commit,
        Abort,
    }

    fn transaction_output_action_strategy() -> impl Strategy<Value = TransactionTestAction> {
        (
            0u8..2,
            0u8..2,
            prop_oneof![
                Just(TransactionTestOperation::Emit),
                Just(TransactionTestOperation::Replace),
                Just(TransactionTestOperation::Retract),
                Just(TransactionTestOperation::Finalize),
            ],
            any::<i64>(),
        )
            .prop_map(
                |(output, stream, operation, payload)| TransactionTestAction {
                    output,
                    stream,
                    operation,
                    payload,
                },
            )
    }

    fn transaction_action_strategy() -> impl Strategy<Value = TransactionTestStep> {
        prop_oneof![
            8 => transaction_output_action_strategy().prop_map(TransactionTestStep::Output),
            1 => Just(TransactionTestStep::Commit),
            1 => Just(TransactionTestStep::Abort),
        ]
    }

    fn transaction_trace_strategy() -> impl Strategy<Value = Vec<TransactionTestStep>> {
        (
            prop::collection::vec(transaction_output_action_strategy(), 0..32),
            prop_oneof![
                Just(TransactionTestStep::Commit),
                Just(TransactionTestStep::Abort)
            ],
            prop::collection::vec(transaction_action_strategy(), 0..8),
        )
            .prop_map(|(prefix, terminal, suffix)| {
                // The first terminal owns the outcome. Execute a generated
                // suffix, the repeated terminal, the opposite terminal, and an
                // output so shrinking cannot remove any misuse class.
                let mut trace = prefix
                    .into_iter()
                    .map(TransactionTestStep::Output)
                    .collect::<Vec<_>>();
                trace.push(terminal);
                trace.extend(suffix);
                trace.push(terminal);
                trace.push(match terminal {
                    TransactionTestStep::Commit => TransactionTestStep::Abort,
                    TransactionTestStep::Abort => TransactionTestStep::Commit,
                    TransactionTestStep::Output(_) => unreachable!("the generated terminal"),
                });
                trace.push(TransactionTestStep::Output(TransactionTestAction {
                    output: 1,
                    stream: 0,
                    operation: TransactionTestOperation::Emit,
                    payload: 73,
                }));
                trace
            })
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ReferenceStream {
        head: u64,
        cursor: u64,
        lineage: u64,
        closed: bool,
        payload: Option<i64>,
    }

    impl ReferenceStream {
        fn apply(&mut self, operation: TransactionTestOperation, payload: i64) -> bool {
            if self.closed
                || (matches!(operation, TransactionTestOperation::Retract) && self.lineage == 0)
            {
                return false;
            }
            self.head += 1;
            self.cursor += 1;
            match operation {
                TransactionTestOperation::Emit | TransactionTestOperation::Replace => {
                    self.lineage = self.head;
                    self.payload = Some(payload);
                }
                TransactionTestOperation::Retract => {
                    self.lineage = 0;
                    self.payload = None;
                }
                TransactionTestOperation::Finalize => self.closed = true,
            }
            true
        }

        fn finalize(&mut self) {
            if !self.closed {
                self.head += 1;
                self.cursor += 1;
                self.closed = true;
            }
        }
    }

    fn reference_streams() -> BTreeMap<(String, OutputStreamId), ReferenceStream> {
        [
            ("answer", "left", 2, 2, 2, false, Some(11)),
            ("answer", "right", 3, 3, 0, false, None),
            ("tool", "left", 4, 4, 4, true, Some(21)),
            ("tool", "right", 1, 1, 1, false, Some(31)),
        ]
        .into_iter()
        .map(|(output, stream, head, cursor, lineage, closed, payload)| {
            (
                (output.to_string(), OutputStreamId(stream.to_string())),
                ReferenceStream {
                    head,
                    cursor,
                    lineage,
                    closed,
                    payload,
                },
            )
        })
        .collect()
    }

    fn session_output_state(reference: &ReferenceStream) -> CommittedOutputState {
        CommittedOutputState {
            head: reference.head,
            cursor: reference.cursor,
            lineage: reference.lineage,
            closed: reference.closed,
            payload: reference.payload.map(value),
        }
    }

    fn assert_journal_matches(
        journal: &OutputPublicationJournal,
        expected: &BTreeMap<(String, OutputStreamId), ReferenceStream>,
    ) {
        let actual = journal
            .committed_states()
            .expect("journal state is inspectable")
            .into_iter()
            .map(|(identity, state)| {
                let payload = state.payload.map(|payload| {
                    payload
                        .to_vec_i64()
                        .expect("test payload is int64")
                        .into_iter()
                        .next()
                        .expect("test payload is scalar")
                });
                (
                    identity,
                    ReferenceStream {
                        head: state.head,
                        cursor: state.cursor,
                        lineage: state.lineage,
                        closed: state.closed,
                        payload,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(&actual, expected);
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DurableMapFingerprint {
        states: Vec<(String, String, Vec<i64>)>,
        effects: Vec<(String, String, u64)>,
        outputs: Vec<CommittedStateFingerprint>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TerminalSnapshot {
        expected: BTreeMap<(String, OutputStreamId), ReferenceStream>,
        durable: DurableMapFingerprint,
        journal: Vec<CommittedStateFingerprint>,
        publication_count: usize,
        provisional_delivery_cursor: usize,
    }

    fn map_fingerprints(
        states: &std::collections::HashMap<(String, String), Value>,
        effects: &std::collections::HashMap<(String, String), u64>,
        outputs: &std::collections::HashMap<(String, String, OutputStreamId), CommittedOutputState>,
    ) -> DurableMapFingerprint {
        let mut state_fingerprint = states
            .iter()
            .map(|((session, state), value)| {
                (
                    session.clone(),
                    state.clone(),
                    value.to_vec_i64().expect("state payload"),
                )
            })
            .collect::<Vec<_>>();
        state_fingerprint.sort();
        let mut effect_fingerprint = effects
            .iter()
            .map(|((session, effect), cursor)| (session.clone(), effect.clone(), *cursor))
            .collect::<Vec<_>>();
        effect_fingerprint.sort();
        let mut output_fingerprint = outputs
            .iter()
            .map(|((session, output, stream), state)| {
                (
                    format!("{session}:{output}"),
                    stream.0.clone(),
                    state.head,
                    state.cursor,
                    state.lineage,
                    state.closed,
                    state
                        .payload
                        .as_ref()
                        .map(|payload| payload.to_raw_bytes().expect("output payload")),
                )
            })
            .collect::<Vec<_>>();
        output_fingerprint.sort();
        DurableMapFingerprint {
            states: state_fingerprint,
            effects: effect_fingerprint,
            outputs: output_fingerprint,
        }
    }

    fn exercise_transaction_sequence(
        generated: &[TransactionTestStep],
        publication_mode: onnx_genai_metadata::WorkflowPublicationMode,
    ) {
        let workflow = revision_workflow(publication_mode);
        let state_plan = onnx_genai_metadata::resolve_state_plan(&workflow);
        assert!(
            state_plan
                .cell("memory")
                .is_some_and(|cell| cell.transaction.required && cell.final_writer.is_some()),
            "the property must admit a real semantic session-state participant"
        );
        let mut expected = reference_streams();
        let mut session_states = std::collections::HashMap::from([
            (("session".to_string(), "memory".to_string()), value(7)),
            (("other".to_string(), "memory".to_string()), value(99)),
        ]);
        let mut session_effects = std::collections::HashMap::from([
            (("session".to_string(), "grammar".to_string()), 4),
            (("other".to_string(), "grammar".to_string()), 9),
        ]);
        let mut session_outputs = expected
            .iter()
            .map(|((output, stream), state)| {
                (
                    ("session".to_string(), output.clone(), stream.clone()),
                    session_output_state(state),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        session_outputs.insert(
            (
                "other".to_string(),
                "answer".to_string(),
                OutputStreamId("left".to_string()),
            ),
            session_output_state(&ReferenceStream {
                head: 9,
                cursor: 9,
                lineage: 9,
                closed: false,
                payload: Some(99),
            }),
        );
        let durable_baseline =
            map_fingerprints(&session_states, &session_effects, &session_outputs);
        let mode = TurnPublicationMode::from(publication_mode);
        let mut turn = TurnTransaction::admit(
            TurnTransactionId(71),
            Some("session"),
            &state_plan,
            ["grammar".to_string()],
            ["answer".to_string(), "tool".to_string()],
            &session_states,
            &session_effects,
            &session_outputs,
            mode,
        )
        .expect("non-empty transaction admits");
        let Some(crate::pipeline::TurnStateBaseline::Present(baseline_state)) =
            turn.baseline_state("memory")
        else {
            panic!("the admitted transaction must snapshot the non-empty state baseline");
        };
        assert_eq!(
            baseline_state
                .to_vec_i64()
                .expect("baseline state is inspectable"),
            vec![7]
        );
        let mut journal = OutputPublicationJournal::new_with_publication_mode(
            turn.id(),
            &workflow,
            turn.output_baselines().expect("output baselines"),
            mode,
        )
        .expect("revision journal admits");
        assert_journal_matches(&journal, &expected);
        turn.stage_state(
            onnx_genai_metadata::StateIdentity("memory".to_string()),
            value(8),
        )
        .expect("pre-terminal state staging succeeds");
        turn.stage_effects()
            .expect("pre-terminal effect staging succeeds");

        let required_actions = [
            TransactionTestAction {
                output: 0,
                stream: 1,
                operation: TransactionTestOperation::Retract,
                payload: 0,
            },
            TransactionTestAction {
                output: 0,
                stream: 1,
                operation: TransactionTestOperation::Emit,
                payload: 12,
            },
            TransactionTestAction {
                output: 0,
                stream: 1,
                operation: TransactionTestOperation::Retract,
                payload: 0,
            },
            TransactionTestAction {
                output: 1,
                stream: 1,
                operation: TransactionTestOperation::Replace,
                payload: 32,
            },
            TransactionTestAction {
                output: 0,
                stream: 0,
                operation: TransactionTestOperation::Finalize,
                payload: 0,
            },
            TransactionTestAction {
                output: 0,
                stream: 0,
                operation: TransactionTestOperation::Emit,
                payload: 13,
            },
        ];

        let sequence = required_actions
            .iter()
            .copied()
            .map(TransactionTestStep::Output)
            .chain(generated.iter().copied())
            .collect::<Vec<_>>();
        let mut terminal_outcome = None;
        let mut terminal_snapshot: Option<TerminalSnapshot> = None;
        let mut executed_after_terminal = 0usize;
        for step in sequence {
            if let Some(snapshot) = &terminal_snapshot {
                executed_after_terminal += 1;
                let resolution = match terminal_outcome
                    .as_ref()
                    .expect("a terminal snapshot has an outcome")
                {
                    TurnTransactionOutcome::Committed { .. } => {
                        OutputPublicationJournalResolution::Committed
                    }
                    TurnTransactionOutcome::AbortToBaseline { .. } => {
                        OutputPublicationJournalResolution::Aborted
                    }
                };
                match step {
                    TransactionTestStep::Output(action) => {
                        let (output, stream) = action.identity();
                        let error = journal
                            .publish(
                                output,
                                Some(stream),
                                &action.operation.mode(),
                                action
                                    .operation
                                    .carries_payload()
                                    .then(|| value(action.payload)),
                            )
                            .expect_err("post-terminal output publication must fail closed");
                        assert_journal_resolution_error(
                            &error,
                            resolution,
                            "validate an output emission",
                        );
                        assert_turn_resolution_error(
                            turn.stage_state(
                                onnx_genai_metadata::StateIdentity("memory".to_string()),
                                value(action.payload),
                            )
                            .expect_err("post-terminal state staging must fail closed"),
                            resolution,
                            "stage semantic state",
                        );
                        assert_turn_resolution_error(
                            turn.stage_outputs(
                                journal
                                    .committed_states()
                                    .expect("post-terminal journal remains inspectable"),
                            )
                            .expect_err("post-terminal output staging must fail closed"),
                            resolution,
                            "stage output publication state",
                        );
                        assert_turn_resolution_error(
                            turn.stage_effects()
                                .expect_err("post-terminal effect staging must fail closed"),
                            resolution,
                            "stage effect cursors",
                        );
                    }
                    TransactionTestStep::Commit => {
                        let error = journal
                            .finalize_on_commit()
                            .expect_err("post-terminal finalize-on-commit must fail closed");
                        assert_journal_resolution_error(
                            &error,
                            resolution,
                            "stage commit finalization",
                        );
                        let error = turn
                            .commit(
                                &mut session_states,
                                &mut session_effects,
                                &mut session_outputs,
                            )
                            .expect_err("post-terminal commit must fail closed");
                        let crate::pipeline::TurnTransactionCommitError::Resolved(error) = error
                        else {
                            panic!("post-terminal commit returned a non-resolution error");
                        };
                        assert_turn_resolution_error(error, resolution, "commit durable state");
                        let error = journal
                            .record_commit(
                                terminal_outcome
                                    .as_ref()
                                    .expect("a terminal snapshot has an outcome"),
                            )
                            .expect_err("repeated journal commit must fail closed");
                        assert_journal_resolution_error(
                            &error,
                            resolution,
                            "record a committed transaction",
                        );
                    }
                    TransactionTestStep::Abort => {
                        let error = journal
                            .abort_outcome(&mut turn, TurnAbortReason::Cancellation)
                            .expect_err("post-terminal journal abort must fail closed");
                        assert_journal_resolution_error(
                            &error,
                            resolution,
                            "abort to the admitted output baseline",
                        );
                        assert_turn_resolution_error(
                            turn.abort(TurnAbortReason::Cancellation)
                                .expect_err("post-terminal turn abort must fail closed"),
                            resolution,
                            "abort to the admitted baseline",
                        );
                    }
                }
                assert_eq!(
                    &expected, &snapshot.expected,
                    "the independent model must remain unchanged after rejected post-terminal action"
                );
                assert_eq!(
                    map_fingerprints(&session_states, &session_effects, &session_outputs),
                    snapshot.durable,
                    "post-terminal {step:?} mutated durable state"
                );
                assert_eq!(
                    committed_state_fingerprint(
                        journal
                            .committed_states()
                            .expect("post-terminal journal remains inspectable")
                    ),
                    snapshot.journal,
                    "post-terminal {step:?} mutated output state"
                );
                assert_eq!(journal.publications.len(), snapshot.publication_count);
                assert_eq!(
                    journal.provisional_delivery_cursor,
                    snapshot.provisional_delivery_cursor
                );
                continue;
            }

            match step {
                TransactionTestStep::Output(action) => {
                    let (output, stream) = action.identity();
                    let identity = (output.to_string(), OutputStreamId(stream.to_string()));
                    let mut next = expected[&identity].clone();
                    let accepted = next.apply(action.operation, action.payload);
                    let before = committed_state_fingerprint(
                        journal
                            .committed_states()
                            .expect("pre-operation journal state"),
                    );
                    let result = journal.publish(
                        output,
                        Some(stream),
                        &action.operation.mode(),
                        action
                            .operation
                            .carries_payload()
                            .then(|| value(action.payload)),
                    );
                    assert_eq!(
                        result.is_ok(),
                        accepted,
                        "{action:?} disagreed with the independent stream model: {result:?}"
                    );
                    if accepted {
                        expected.insert(identity, next);
                    } else {
                        assert_eq!(
                            committed_state_fingerprint(
                                journal
                                    .committed_states()
                                    .expect("rejected operation leaves state inspectable")
                            ),
                            before,
                            "rejected {action:?} partially advanced a stream"
                        );
                    }
                    assert_journal_matches(&journal, &expected);
                    turn.stage_outputs(
                        journal.committed_states().expect("staged output write set"),
                    )
                    .expect("pre-terminal output staging succeeds");
                    let pending = journal
                        .take_pending_provisionals()
                        .expect("provisional delivery is cloneable");
                    if publication_mode
                        == onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions
                        && accepted
                    {
                        assert_eq!(pending.len(), 1);
                        let WorkflowOutputPublication::Revision(envelope) = &pending[0] else {
                            panic!("pre-commit publication was not a typed revision");
                        };
                        assert_eq!(envelope.finality, OutputFinality::Provisional);
                    } else {
                        assert!(pending.is_empty());
                    }
                    assert_eq!(
                        map_fingerprints(&session_states, &session_effects, &session_outputs),
                        durable_baseline,
                        "staged writes escaped before the transaction outcome"
                    );
                }
                TransactionTestStep::Commit => {
                    let open_streams = expected.values().filter(|stream| !stream.closed).count();
                    journal
                        .finalize_on_commit()
                        .expect("commit finalizes every open stream");
                    for stream in expected.values_mut() {
                        stream.finalize();
                    }
                    assert_journal_matches(&journal, &expected);
                    let pending_finalizes = journal
                        .take_pending_provisionals()
                        .expect("commit finalizes are cloneable");
                    if publication_mode
                        == onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions
                    {
                        assert_eq!(pending_finalizes.len(), open_streams);
                        assert!(pending_finalizes.iter().all(|publication| matches!(
                            publication,
                            WorkflowOutputPublication::Revision(TypedRevisionEnvelope {
                                operation: TypedRevisionOperation::Finalize,
                                finality: OutputFinality::Provisional,
                                ..
                            })
                        )));
                    } else {
                        assert!(pending_finalizes.is_empty());
                    }
                    turn.stage_outputs(
                        journal
                            .committed_states()
                            .expect("committed output write set"),
                    )
                    .expect("pre-terminal output staging succeeds");
                    let outcome = turn
                        .commit(
                            &mut session_states,
                            &mut session_effects,
                            &mut session_outputs,
                        )
                        .expect("real transaction commit succeeds");
                    assert_eq!(
                        outcome,
                        TurnTransactionOutcome::Committed {
                            transaction: TurnTransactionId(71),
                            baseline: TurnBaselineId(71),
                        }
                    );
                    journal
                        .record_commit(&outcome)
                        .expect("the journal records exactly one commit");
                    terminal_outcome = Some(outcome);
                }
                TransactionTestStep::Abort => {
                    let outcome = journal
                        .abort_outcome(&mut turn, TurnAbortReason::Cancellation)
                        .expect("the journal and turn record exactly one abort");
                    assert!(matches!(
                        outcome,
                        TurnTransactionOutcome::AbortToBaseline {
                            transaction: TurnTransactionId(71),
                            baseline: TurnBaselineId(71),
                            reason: TurnAbortReason::Cancellation,
                            ..
                        }
                    ));
                    terminal_outcome = Some(outcome);
                }
            }

            if terminal_outcome.is_some() {
                terminal_snapshot = Some(TerminalSnapshot {
                    expected: expected.clone(),
                    durable: map_fingerprints(&session_states, &session_effects, &session_outputs),
                    journal: committed_state_fingerprint(
                        journal
                            .committed_states()
                            .expect("terminal journal remains inspectable"),
                    ),
                    publication_count: journal.publications.len(),
                    provisional_delivery_cursor: journal.provisional_delivery_cursor,
                });
            }
        }

        assert!(
            executed_after_terminal >= 3,
            "the generated trace must execute rejected output and terminal actions"
        );
        let outcome = terminal_outcome.expect("every generated trace contains a terminal action");
        match outcome {
            TurnTransactionOutcome::Committed {
                transaction,
                baseline,
            } => {
                assert_eq!(transaction, TurnTransactionId(71));
                assert_eq!(baseline, TurnBaselineId(71));
                let publications = journal.take();
                match publication_mode {
                    onnx_genai_metadata::WorkflowPublicationMode::CommitOnly => {
                        assert!(publications.iter().all(|publication| matches!(
                            publication,
                            WorkflowOutputPublication::Revision(TypedRevisionEnvelope {
                                finality: OutputFinality::Final,
                                ..
                            })
                        )));
                    }
                    onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions => {
                        assert!(publications[..publications.len() - 1].iter().all(
                            |publication| matches!(
                                publication,
                                WorkflowOutputPublication::Revision(TypedRevisionEnvelope {
                                    finality: OutputFinality::Provisional,
                                    ..
                                })
                            )
                        ));
                        assert!(matches!(
                            publications.last(),
                            Some(WorkflowOutputPublication::TransactionCommitted {
                                transaction: TurnTransactionId(71),
                                baseline: TurnBaselineId(71),
                            })
                        ));
                        assert_eq!(
                            publications
                                .iter()
                                .filter(|publication| matches!(
                                    publication,
                                    WorkflowOutputPublication::TransactionCommitted { .. }
                                        | WorkflowOutputPublication::AbortToBaseline { .. }
                                ))
                                .count(),
                            1
                        );
                    }
                }
                assert_ne!(
                    map_fingerprints(&session_states, &session_effects, &session_outputs),
                    durable_baseline,
                    "commit must mutate the admitted session's complete write set"
                );
                assert_eq!(
                    session_states[&("session".to_string(), "memory".to_string())]
                        .to_vec_i64()
                        .expect("committed state"),
                    vec![8]
                );
                assert_eq!(
                    session_states[&("other".to_string(), "memory".to_string())]
                        .to_vec_i64()
                        .expect("isolated state"),
                    vec![99]
                );
                assert_eq!(
                    session_effects[&("session".to_string(), "grammar".to_string())],
                    5
                );
                assert_eq!(
                    session_effects[&("other".to_string(), "grammar".to_string())],
                    9
                );
                for ((output, stream), reference) in &expected {
                    let state =
                        &session_outputs[&("session".to_string(), output.clone(), stream.clone())];
                    assert_eq!(
                        (
                            state.head,
                            state.cursor,
                            state.lineage,
                            state.closed,
                            state
                                .payload
                                .as_ref()
                                .map(|payload| payload.to_vec_i64().expect("committed payload")[0]),
                        ),
                        (
                            reference.head,
                            reference.cursor,
                            reference.lineage,
                            reference.closed,
                            reference.payload,
                        )
                    );
                }
                let isolated = &session_outputs[&(
                    "other".to_string(),
                    "answer".to_string(),
                    OutputStreamId("left".to_string()),
                )];
                assert_eq!(
                    (
                        isolated.head,
                        isolated.cursor,
                        isolated.lineage,
                        isolated.closed,
                        isolated
                            .payload
                            .as_ref()
                            .expect("isolated payload")
                            .to_vec_i64()
                            .expect("isolated payload"),
                    ),
                    (9, 9, 9, false, vec![99])
                );
            }
            TurnTransactionOutcome::AbortToBaseline {
                transaction,
                baseline,
                reason,
                streams,
            } => {
                assert_eq!(transaction, TurnTransactionId(71));
                assert_eq!(baseline, TurnBaselineId(71));
                assert_eq!(reason, TurnAbortReason::Cancellation);
                let baseline_by_stream = reference_streams();
                assert_eq!(streams.len(), baseline_by_stream.len());
                for stream in &streams {
                    let expected =
                        &baseline_by_stream[&(stream.output.clone(), stream.stream.clone())];
                    assert_eq!(
                        (stream.head, stream.sequence, stream.lineage, stream.closed),
                        (
                            expected.head,
                            expected.cursor,
                            expected.lineage,
                            expected.closed,
                        )
                    );
                }
                assert!(matches!(
                    WorkflowOutputPublication::from_transaction_outcome(
                        &TurnTransactionOutcome::AbortToBaseline {
                            transaction,
                            baseline,
                            reason,
                            streams,
                        }
                    ),
                    WorkflowOutputPublication::AbortToBaseline {
                        transaction: TurnTransactionId(71),
                        baseline: TurnBaselineId(71),
                        reason: TurnAbortReason::Cancellation,
                        ..
                    }
                ));
                assert_eq!(
                    map_fingerprints(&session_states, &session_effects, &session_outputs),
                    durable_baseline,
                    "real abort must preserve the exact non-empty state/effect/output baseline"
                );
                let publications = journal.take();
                match publication_mode {
                    onnx_genai_metadata::WorkflowPublicationMode::CommitOnly => {
                        assert!(publications.iter().all(|publication| !matches!(
                            publication,
                            WorkflowOutputPublication::TransactionCommitted { .. }
                                | WorkflowOutputPublication::AbortToBaseline { .. }
                        )));
                    }
                    onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions => {
                        assert!(matches!(
                            publications.last(),
                            Some(WorkflowOutputPublication::AbortToBaseline {
                                transaction: TurnTransactionId(71),
                                baseline: TurnBaselineId(71),
                                reason: TurnAbortReason::Cancellation,
                                ..
                            })
                        ));
                        assert_eq!(
                            publications
                                .iter()
                                .filter(|publication| matches!(
                                    publication,
                                    WorkflowOutputPublication::TransactionCommitted { .. }
                                        | WorkflowOutputPublication::AbortToBaseline { .. }
                                ))
                                .count(),
                            1
                        );
                    }
                }
            }
        }
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
            publication_mode_authored: false,
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
            publication_mode_authored: false,
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
            publication_mode_authored: false,
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
            publication_mode_authored: false,
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
            publication_mode_authored: false,
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

    #[test]
    fn committed_journal_rejects_publication_finalization_and_repeated_outcomes() -> Result<()> {
        let workflow =
            revision_workflow(onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions);
        let mut turn = TurnTransaction::admit(
            TurnTransactionId(51),
            None,
            &onnx_genai_metadata::ResolvedStatePlan::default(),
            std::iter::empty::<String>(),
            ["answer".to_string(), "tool".to_string()],
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            TurnPublicationMode::ProvisionalRevisions,
        )?;
        let mut journal = OutputPublicationJournal::new_with_publication_mode(
            turn.id(),
            &workflow,
            turn.output_baselines()?,
            TurnPublicationMode::ProvisionalRevisions,
        )?;
        journal.publish(
            "answer",
            Some("dynamic"),
            &WorkflowEmitMode::Append,
            Some(value(5)),
        )?;
        journal.finalize_on_commit()?;
        turn.stage_outputs(journal.committed_states()?)?;
        let outcome = turn.commit(
            &mut std::collections::HashMap::new(),
            &mut std::collections::HashMap::new(),
            &mut std::collections::HashMap::new(),
        )?;
        journal.record_commit(&outcome)?;
        let snapshot = (
            committed_state_fingerprint(journal.committed_states()?),
            journal.publications.len(),
            journal.provisional_delivery_cursor,
        );

        for error in [
            journal
                .publish(
                    "answer",
                    Some("dynamic"),
                    &WorkflowEmitMode::Replace,
                    Some(value(6)),
                )
                .expect_err("commit must close publication"),
            journal
                .finalize_on_commit()
                .expect_err("commit must close finalization"),
            journal
                .record_commit(&outcome)
                .expect_err("commit must close repeated terminal records"),
            journal
                .take_pending_provisionals()
                .expect_err("commit must close provisional delivery"),
            journal
                .abort_outcome(&mut turn, TurnAbortReason::Cancellation)
                .expect_err("commit must close abort"),
        ] {
            let error = error
                .downcast_ref::<OutputPublicationJournalResolutionError>()
                .expect("journal misuse returns a typed resolution error");
            assert_eq!(error.transaction, TurnTransactionId(51));
            assert_eq!(
                error.resolution,
                OutputPublicationJournalResolution::Committed
            );
            assert!(error.to_string().contains("Admit a new turn"));
        }
        assert_eq!(
            (
                committed_state_fingerprint(journal.committed_states()?),
                journal.publications.len(),
                journal.provisional_delivery_cursor,
            ),
            snapshot,
            "post-commit journal misuse must leave every cursor, payload, and record unchanged"
        );
        assert_eq!(
            journal
                .publications
                .iter()
                .filter(|publication| matches!(
                    publication,
                    WorkflowOutputPublication::TransactionCommitted { .. }
                        | WorkflowOutputPublication::AbortToBaseline { .. }
                ))
                .count(),
            1,
            "a committed journal must contain exactly one terminal outcome"
        );
        Ok(())
    }

    #[test]
    fn aborted_journal_rejects_publication_commit_and_repeated_abort() -> Result<()> {
        let workflow =
            revision_workflow(onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions);
        let mut turn = TurnTransaction::admit(
            TurnTransactionId(52),
            None,
            &onnx_genai_metadata::ResolvedStatePlan::default(),
            std::iter::empty::<String>(),
            ["answer".to_string(), "tool".to_string()],
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            TurnPublicationMode::ProvisionalRevisions,
        )?;
        let mut journal = OutputPublicationJournal::new_with_publication_mode(
            turn.id(),
            &workflow,
            turn.output_baselines()?,
            TurnPublicationMode::ProvisionalRevisions,
        )?;
        journal.publish(
            "answer",
            Some("dynamic"),
            &WorkflowEmitMode::Append,
            Some(value(5)),
        )?;
        let outcome = journal.abort_outcome(&mut turn, TurnAbortReason::ExecutionFailure)?;
        let snapshot = (
            committed_state_fingerprint(journal.committed_states()?),
            journal.publications.len(),
            journal.provisional_delivery_cursor,
        );

        for error in [
            journal
                .publish("answer", Some("dynamic"), &WorkflowEmitMode::Finalize, None)
                .expect_err("abort must close publication"),
            journal
                .finalize_on_commit()
                .expect_err("abort must close commit finalization"),
            journal
                .record_commit(&outcome)
                .expect_err("abort must close commit recording"),
            journal
                .take_pending_provisionals()
                .expect_err("abort must close provisional delivery"),
            journal
                .abort_outcome(&mut turn, TurnAbortReason::Cancellation)
                .expect_err("abort must close repeated abort"),
        ] {
            let error = error
                .downcast_ref::<OutputPublicationJournalResolutionError>()
                .expect("journal misuse returns a typed resolution error");
            assert_eq!(error.transaction, TurnTransactionId(52));
            assert_eq!(
                error.resolution,
                OutputPublicationJournalResolution::Aborted
            );
            assert!(error.to_string().contains("Admit a new turn"));
        }
        assert_eq!(
            (
                committed_state_fingerprint(journal.committed_states()?),
                journal.publications.len(),
                journal.provisional_delivery_cursor,
            ),
            snapshot,
            "post-abort journal misuse must leave every cursor, payload, and record unchanged"
        );
        assert_eq!(
            journal
                .publications
                .iter()
                .filter(|publication| matches!(
                    publication,
                    WorkflowOutputPublication::TransactionCommitted { .. }
                        | WorkflowOutputPublication::AbortToBaseline { .. }
                ))
                .count(),
            1,
            "an aborted journal must contain exactly one terminal outcome"
        );
        Ok(())
    }

    #[test]
    fn journal_abort_reconciles_a_stream_created_after_admission() -> Result<()> {
        let workflow =
            revision_workflow(onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions);
        let mut turn = TurnTransaction::admit(
            TurnTransactionId(91),
            None,
            &onnx_genai_metadata::ResolvedStatePlan::default(),
            std::iter::empty::<String>(),
            ["answer".to_string()],
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            TurnPublicationMode::ProvisionalRevisions,
        )
        .expect("turn admits");
        let mut journal = OutputPublicationJournal::new_with_publication_mode(
            turn.id(),
            &workflow,
            turn.output_baselines()?,
            TurnPublicationMode::ProvisionalRevisions,
        )?;
        journal.publish(
            "answer",
            Some("dynamic"),
            &WorkflowEmitMode::Append,
            Some(value(5)),
        )?;

        let TurnTransactionOutcome::AbortToBaseline { streams, .. } =
            journal.abort_outcome(&mut turn, TurnAbortReason::ExecutionFailure)?
        else {
            unreachable!("journal abort is a typed transaction outcome");
        };
        assert!(
            streams.iter().any(|baseline| baseline
                == &OutputStreamBaseline {
                    output: "answer".to_string(),
                    stream: OutputStreamId("dynamic".to_string()),
                    head: 0,
                    sequence: 0,
                    lineage: 0,
                    closed: false,
                }),
            "a dynamically introduced stream must reconcile to its empty admission baseline"
        );
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
            trace in transaction_trace_strategy(),
        ) {
            for publication_mode in [
                onnx_genai_metadata::WorkflowPublicationMode::CommitOnly,
                onnx_genai_metadata::WorkflowPublicationMode::ProvisionalRevisions,
            ] {
                exercise_transaction_sequence(&trace, publication_mode);
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
