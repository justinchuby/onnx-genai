//! Transport-neutral workflow output publication protocols.
//!
//! An executor owns this journal until its enclosing turn commits. HTTP/SSE
//! adapters receive the resulting ordered publications; they never create,
//! reorder, or roll back semantic output state.

use std::collections::BTreeMap;

use anyhow::{Context as _, Result, bail, ensure};
use onnx_genai_metadata::{WorkflowEmitMode, WorkflowOutputFamily, WorkflowSpec};
use onnx_genai_ort::Value;

use super::{OutputPublicationBaseline, TurnTransactionId};

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
        }
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
pub(crate) struct OutputPublicationJournal {
    transaction: TurnTransactionId,
    outputs: BTreeMap<String, WorkflowOutputFamily>,
    revisions: RevisionEnvelopeValidator,
    event_sequences: BTreeMap<(String, OutputStreamId), u64>,
    publications: Vec<WorkflowOutputPublication>,
}

impl OutputPublicationJournal {
    pub(crate) fn new(
        transaction: TurnTransactionId,
        workflow: &WorkflowSpec,
        baselines: BTreeMap<String, OutputPublicationBaseline>,
    ) -> Result<Self> {
        for (output, declaration) in &workflow.outputs {
            if let WorkflowOutputFamily::Revisions { version } = &declaration.family {
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
            .map(|(name, declaration)| (name.clone(), declaration.family.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut revisions = RevisionEnvelopeValidator::new(outputs.clone());
        let mut event_sequences = BTreeMap::new();
        for (output, family) in &outputs {
            let baseline = baselines.get(output).copied().unwrap_or_default();
            let stream = OutputStreamId(output.clone());
            if matches!(family, WorkflowOutputFamily::Revisions { .. })
                && (baseline.head != 0
                    || baseline.cursor != 0
                    || baseline.lineage != 0
                    || baseline.closed)
            {
                revisions.streams.insert(
                    (output.clone(), stream.clone()),
                    RevisionStreamState {
                        sequence: baseline.cursor,
                        revision: baseline.head,
                        lineage: baseline.lineage,
                        closed: baseline.closed,
                    },
                );
            }
            if matches!(family, WorkflowOutputFamily::Events) {
                event_sequences.insert((output.clone(), stream), baseline.cursor);
            }
        }
        Ok(Self {
            transaction,
            revisions,
            outputs,
            event_sequences,
            publications: Vec::new(),
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
                WorkflowOutputFamily::Materialized,
                WorkflowEmitMode::Replace | WorkflowEmitMode::Append,
            ) => ensure!(
                stream.0 == output,
                "materialized output '{output}' cannot publish a named stream '{}'",
                stream.0
            ),
            (WorkflowOutputFamily::Events, WorkflowEmitMode::Event) => {}
            (
                WorkflowOutputFamily::Revisions { .. },
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
                WorkflowOutputFamily::Materialized,
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
            (WorkflowOutputFamily::Events, WorkflowEmitMode::Event) => {
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
                WorkflowOutputFamily::Revisions { .. },
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
        for publication in &mut self.publications {
            match publication {
                WorkflowOutputPublication::Materialized { finality, .. }
                | WorkflowOutputPublication::Event { finality, .. } => {
                    *finality = OutputFinality::Final
                }
                WorkflowOutputPublication::Revision(envelope) => {
                    envelope.finality = OutputFinality::Final;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn publication_counts(&self) -> BTreeMap<String, u64> {
        let mut counts = BTreeMap::new();
        for publication in &self.publications {
            let output = match publication {
                WorkflowOutputPublication::Materialized { output, .. }
                | WorkflowOutputPublication::Event { output, .. } => output,
                WorkflowOutputPublication::Revision(envelope) => &envelope.output,
            };
            *counts.entry(output.clone()).or_default() += 1;
        }
        counts
    }

    pub(crate) fn take(self) -> Vec<WorkflowOutputPublication> {
        self.publications
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(value: i64) -> Value {
        Value::from_slice_i64(&[value], &[1]).expect("test value")
    }

    fn revisions() -> RevisionEnvelopeValidator {
        RevisionEnvelopeValidator::new([(
            "answer".to_string(),
            WorkflowOutputFamily::Revisions {
                version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
            },
        )])
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
    fn committed_closed_baseline_rejects_post_close_revision_before_publication() -> Result<()> {
        let workflow = WorkflowSpec {
            manifest: onnx_genai_metadata::WorkflowManifest {
                adapter_abis: Default::default(),
                capabilities: Default::default(),
            },
            inputs: Default::default(),
            outputs: BTreeMap::from([(
                "answer".to_string(),
                onnx_genai_metadata::WorkflowOutput {
                    contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")?,
                    role: onnx_genai_metadata::WorkflowOutputRole::Tensor,
                    family: WorkflowOutputFamily::Revisions {
                        version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
                    },
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
                "answer".to_string(),
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
                capabilities: Default::default(),
            },
            inputs: Default::default(),
            outputs: BTreeMap::from([(
                "answer".to_string(),
                onnx_genai_metadata::WorkflowOutput {
                    contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")?,
                    role: onnx_genai_metadata::WorkflowOutputRole::Tensor,
                    family: WorkflowOutputFamily::Revisions {
                        version: TYPED_REVISION_PROTOCOL_VERSION.to_string(),
                    },
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
    fn materialized_events_and_revisions_are_all_observed_with_commit_finality() -> Result<()> {
        let output = |family| onnx_genai_metadata::WorkflowOutput {
            contract: serde_yaml::from_str("{ dtype: int64, shape: [sequence] }")
                .expect("output contract"),
            role: onnx_genai_metadata::WorkflowOutputRole::Tensor,
            family,
            value_range: None,
            stage: onnx_genai_metadata::OutputStage::PreAdapter,
            media: None,
        };
        let workflow = WorkflowSpec {
            manifest: onnx_genai_metadata::WorkflowManifest {
                adapter_abis: Default::default(),
                capabilities: Default::default(),
            },
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
}
