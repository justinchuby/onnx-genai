//! Complete semantic session-fork admission and publication.

use super::*;
use onnx_genai_metadata::{StateManagement, StateSemanticRole};
use std::collections::BTreeSet;

/// Semantic participant reproduced by a session fork.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SessionForkParticipantKind {
    LogicalPosition,
    Kv,
    RecurrentOrConvolution,
    GenericFeatureState,
    TokenContextHistory,
    RandomState,
    GrammarConstraint,
    ContinuationCursor,
    TransactionalEffect,
    OutputPublication,
    SpeculativeCascade,
}

/// Inspectable participant entry in the one admitted fork plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionForkParticipant {
    pub kind: SessionForkParticipantKind,
    pub name: String,
    pub state_type: String,
    pub backend: String,
}

impl SessionForkParticipant {
    fn new(
        kind: SessionForkParticipantKind,
        name: impl Into<String>,
        state_type: impl Into<String>,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            name: name.into(),
            state_type: state_type.into(),
            backend: backend.into(),
        }
    }
}

/// A fully preflighted semantic fork.
///
/// The private payload owns every fallible deep clone. Consuming the plan in
/// [`Engine::fork_session`] leaves only page-reference publication and
/// infallible map insertion after the child identity is created.
pub struct SessionForkPlan {
    source: SessionId,
    position: SessionPosition,
    participants: Vec<SessionForkParticipant>,
    prepared: PreparedSessionFork,
}

impl std::fmt::Debug for SessionForkPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionForkPlan")
            .field("source", &self.source)
            .field("position", &self.position)
            .field("origin_generation", &self.prepared.origin().generation())
            .field("participants", &self.participants)
            .finish_non_exhaustive()
    }
}

impl SessionForkPlan {
    pub fn source(&self) -> SessionId {
        self.source
    }

    pub fn position(&self) -> SessionPosition {
        self.position
    }

    pub fn participants(&self) -> &[SessionForkParticipant] {
        &self.participants
    }
}

/// Actionable semantic-fork admission/publication failure.
#[derive(Debug, thiserror::Error)]
pub enum SessionForkError {
    #[error("session {session} not found")]
    SessionNotFound { session: SessionId },
    #[error(
        "cannot fork session {session} at token {requested}; current committed length is {current}"
    )]
    PositionOutOfBounds {
        session: SessionId,
        requested: usize,
        current: usize,
    },
    #[error(
        "cannot fork session {session} at token {requested}: participant '{participant}' only has \
         a complete committed baseline at token {current}; checkpoint/fork at that boundary or \
         use a backend that retains rollback-bound participant snapshots"
    )]
    PositionNotCommitted {
        session: SessionId,
        requested: usize,
        current: usize,
        participant: String,
    },
    #[error(
        "cannot fork participant '{participant}' state '{state}' type '{state_type}' on backend \
         '{backend}': {reason}"
    )]
    UnsupportedParticipant {
        participant: String,
        state: String,
        state_type: String,
        backend: String,
        reason: String,
    },
    #[error(
        "failed to snapshot fork participant '{participant}' state '{state}' type '{state_type}' \
         on backend '{backend}': {reason}"
    )]
    SnapshotFailed {
        participant: String,
        state: String,
        state_type: String,
        backend: String,
        reason: String,
    },
    #[error(
        "session fork plan for source {session} at token {position} is stale: {reason}; prepare a \
         new plan from the current committed baseline"
    )]
    StalePlan {
        session: SessionId,
        position: usize,
        reason: String,
    },
    #[error(
        "session fork plan for source {session} at token {position} belongs to a different runtime \
         ownership domain (plan generation {plan_generation}, current engine generation \
         {current_generation}); plans cannot cross engine instances, worker/session-routing \
         domains, backend instances, or engine restarts; prepare a new plan on the engine that \
         currently owns the source session"
    )]
    ForeignOrigin {
        session: SessionId,
        position: usize,
        plan_generation: u64,
        current_generation: u64,
    },
    #[error(
        "session fork plan for source {session} at token {position} belongs to stale engine \
         generation {plan_generation}; the current engine generation is {current_generation}; \
         prepare a new plan from the current engine generation"
    )]
    StaleOrigin {
        session: SessionId,
        position: usize,
        plan_generation: u64,
        current_generation: u64,
    },
    #[error("cannot publish semantic session fork: {0}")]
    Publication(String),
}

enum PreparedSessionFork {
    Workflow(PreparedWorkflowFork),
    Ort(Box<PreparedOrtFork>),
}

impl PreparedSessionFork {
    fn origin(&self) -> SessionForkOrigin {
        match self {
            Self::Workflow(prepared) => prepared.origin,
            Self::Ort(prepared) => prepared.origin,
        }
    }
}

struct PreparedWorkflowFork {
    origin: SessionForkOrigin,
    source_version: u64,
    snapshot: crate::pipeline::WorkflowSessionForkSnapshot,
}

#[derive(Debug, Clone, Copy)]
enum PreparedKvFork {
    Empty,
    CopyOnWrite { source: SessionId, position: usize },
}

struct PreparedOrtFork {
    origin: SessionForkOrigin,
    source_tokens: Vec<TokenId>,
    source_kv_token_count: usize,
    source_draft: Option<(SessionId, Vec<TokenId>, usize)>,
    state: EngineSession,
    target_kv: PreparedKvFork,
    draft_kv: Option<PreparedKvFork>,
}

impl Engine {
    /// Preflight and snapshot every semantic participant before a child exists.
    pub fn prepare_session_fork(
        &self,
        source: SessionId,
        position: SessionPosition,
    ) -> Result<SessionForkPlan, SessionForkError> {
        if !self.holds_decode_core() {
            return self.prepare_workflow_session_fork(source, position);
        }
        #[cfg(feature = "native-backend")]
        if self.decode_backend == EngineDecodeBackend::Native {
            return Err(SessionForkError::UnsupportedParticipant {
                participant: "native.session_state".to_string(),
                state: source.to_string(),
                state_type: "native KV plus recurrent/convolution continuation".to_string(),
                backend: "native".to_string(),
                reason: "the runtime stores one mutable native decoder across sessions and has no \
                         per-session snapshot/import participant yet"
                    .to_string(),
            });
        }
        self.prepare_ort_session_fork(source, position)
    }

    /// Publish a child from a fully admitted plan.
    pub fn fork_session(&mut self, plan: SessionForkPlan) -> Result<SessionId, SessionForkError> {
        self.validate_session_fork_origin(plan.source, plan.position, plan.prepared.origin())?;
        let SessionForkPlan {
            source,
            position,
            prepared,
            ..
        } = plan;
        match prepared {
            PreparedSessionFork::Workflow(prepared) => {
                self.publish_workflow_session_fork(source, position, prepared)
            }
            PreparedSessionFork::Ort(prepared) => {
                self.publish_ort_session_fork(source, position, *prepared)
            }
        }
    }

    fn validate_session_fork_origin(
        &self,
        source: SessionId,
        position: SessionPosition,
        origin: SessionForkOrigin,
    ) -> Result<(), SessionForkError> {
        if origin == self.session_fork_origin {
            return Ok(());
        }
        let error = if origin.same_domain(self.session_fork_origin) {
            SessionForkError::StaleOrigin {
                session: source,
                position: position.get(),
                plan_generation: origin.generation(),
                current_generation: self.session_fork_origin.generation(),
            }
        } else {
            SessionForkError::ForeignOrigin {
                session: source,
                position: position.get(),
                plan_generation: origin.generation(),
                current_generation: self.session_fork_origin.generation(),
            }
        };
        Err(error)
    }

    #[cfg(test)]
    pub(crate) fn advance_session_fork_generation_for_test(&mut self) {
        self.session_fork_origin = self.session_fork_origin.next_generation_for_test();
    }

    fn prepare_workflow_session_fork(
        &self,
        source: SessionId,
        position: SessionPosition,
    ) -> Result<SessionForkPlan, SessionForkError> {
        self.workflow_sessions
            .get(&source)
            .copied()
            .ok_or(SessionForkError::SessionNotFound { session: source })?;
        let current = self
            .workflow
            .session_committed_position(&source.to_string());
        validate_requested_position(
            source,
            position.get(),
            current,
            "workflow output/continuation",
        )?;
        let session = source.to_string();
        if self.workflow.session_has_active_turn(&session) {
            return Err(SessionForkError::UnsupportedParticipant {
                participant: "transaction".to_string(),
                state: session,
                state_type: "in-flight workflow turn".to_string(),
                backend: "workflow-interpreter".to_string(),
                reason: "fork is defined only at a committed boundary; wait for the active turn \
                         to commit or abort"
                    .to_string(),
            });
        }

        let participants = workflow_fork_participants(&self.workflow)?;
        let source_version = self.workflow.session_turn_version(&source.to_string());
        let snapshot = self
            .workflow
            .snapshot_session_for_fork(&source.to_string())
            .map_err(|error| SessionForkError::SnapshotFailed {
                participant: "workflow.semantic_state".to_string(),
                state: source.to_string(),
                state_type: "resolved semantic state/effect/output write set".to_string(),
                backend: "workflow-interpreter".to_string(),
                reason: format!("{error:#}"),
            })?;
        Ok(SessionForkPlan {
            source,
            position,
            participants,
            prepared: PreparedSessionFork::Workflow(PreparedWorkflowFork {
                origin: self.session_fork_origin,
                source_version,
                snapshot,
            }),
        })
    }

    fn publish_workflow_session_fork(
        &mut self,
        source: SessionId,
        position: SessionPosition,
        prepared: PreparedWorkflowFork,
    ) -> Result<SessionId, SessionForkError> {
        self.workflow_sessions
            .get(&source)
            .copied()
            .ok_or(SessionForkError::SessionNotFound { session: source })?;
        let current = self
            .workflow
            .session_committed_position(&source.to_string());
        let current_version = self.workflow.session_turn_version(&source.to_string());
        if current != position.get() || current_version != prepared.source_version {
            return Err(SessionForkError::StalePlan {
                session: source,
                position: position.get(),
                reason: format!(
                    "source is now at token {current}, committed version {current_version}, but \
                     the plan captured version {}",
                    prepared.source_version
                ),
            });
        }
        self.workflow_sessions.try_reserve(1).map_err(|error| {
            SessionForkError::Publication(format!(
                "failed to reserve the child workflow-session entry: {error}"
            ))
        })?;
        let child = self.workflow_session_ids.mint();
        self.workflow
            .install_session_fork(&child.to_string(), prepared.snapshot)
            .map_err(|error| SessionForkError::Publication(format!("{error:#}")))?;
        self.workflow_sessions.insert(child, position.get());
        Ok(child)
    }

    fn prepare_ort_session_fork(
        &self,
        source: SessionId,
        position: SessionPosition,
    ) -> Result<SessionForkPlan, SessionForkError> {
        let state = self
            .sessions
            .get(&source)
            .ok_or(SessionForkError::SessionNotFound { session: source })?;
        let current = state.tokens.len();
        if position.get() > current {
            return Err(SessionForkError::PositionOutOfBounds {
                session: source,
                requested: position.get(),
                current,
            });
        }
        let historical = position.get() < current;
        let fixed_state_names = state.decode_state.fixed_state_names();
        if historical && state.decode_state.has_runner() {
            return Err(SessionForkError::PositionNotCommitted {
                session: source,
                requested: position.get(),
                current,
                participant: "runner-owned decoder KV".to_string(),
            });
        }
        if historical && !fixed_state_names.is_empty() {
            return Err(SessionForkError::PositionNotCommitted {
                session: source,
                requested: position.get(),
                current,
                participant: "recurrent/convolution fixed state".to_string(),
            });
        }
        if historical && state.draft.is_some() {
            return Err(SessionForkError::PositionNotCommitted {
                session: source,
                requested: position.get(),
                current,
                participant: "speculative draft cascade".to_string(),
            });
        }
        if historical && position.get() > state.kv_token_count {
            return Err(SessionForkError::PositionNotCommitted {
                session: source,
                requested: position.get(),
                current,
                participant: format!(
                    "target KV materialized only through token {}",
                    state.kv_token_count
                ),
            });
        }

        let mut participants = vec![
            SessionForkParticipant::new(
                SessionForkParticipantKind::LogicalPosition,
                "session.logical_position",
                format!("committed token boundary {}", position.get()),
                "ort",
            ),
            SessionForkParticipant::new(
                SessionForkParticipantKind::ContinuationCursor,
                "target.tokens",
                "ordered token continuation and KV cursor",
                "ort",
            ),
            SessionForkParticipant::new(
                SessionForkParticipantKind::RandomState,
                "turn_rng",
                "empty at committed boundary; next turn constructs an independent stream",
                "runtime",
            ),
            SessionForkParticipant::new(
                SessionForkParticipantKind::GrammarConstraint,
                "turn_constraints",
                "empty at committed boundary; next turn constructs independent processors",
                "runtime",
            ),
            SessionForkParticipant::new(
                SessionForkParticipantKind::OutputPublication,
                "tokens",
                "commit-only output closed at the committed turn boundary",
                "runtime",
            ),
        ];
        if state.decode_state.use_kv {
            participants.push(SessionForkParticipant::new(
                SessionForkParticipantKind::Kv,
                "target.kv",
                format!("materialized through token {}", state.kv_token_count),
                "ort",
            ));
        }
        for name in fixed_state_names {
            participants.push(SessionForkParticipant::new(
                SessionForkParticipantKind::RecurrentOrConvolution,
                format!("target.{name}"),
                "fixed loop-carried replace state",
                "ort",
            ));
        }

        let target_kv_position = if historical {
            position.get()
        } else {
            state.kv_token_count
        };
        let target_kv = if state.decode_state.has_runner() || !state.decode_state.use_kv {
            PreparedKvFork::Empty
        } else {
            self.kv_cache
                .validate_fork(source, target_kv_position)
                .map_err(|error| SessionForkError::UnsupportedParticipant {
                    participant: "target.kv".to_string(),
                    state: source.to_string(),
                    state_type: "paged KV".to_string(),
                    backend: "onnx-genai-kv".to_string(),
                    reason: error.to_string(),
                })?;
            PreparedKvFork::CopyOnWrite {
                source,
                position: target_kv_position,
            }
        };
        let target_decode =
            self.new_target_decode_state()
                .and_then(|mut target| {
                    if historical && state.decode_state.use_kv {
                        let session = self.session.as_deref().context(MISSING_ORT_SESSION)?;
                        let kv_model = self.kv_model.as_ref().context(
                            "historical paged-KV fork requires resolved KV model geometry",
                        )?;
                        let materialized = self
                            .kv_cache
                            .materialize_sequence_to(source, target_kv_position)
                            .map_err(|error| anyhow::anyhow!("{error}"))?;
                        load_materialized_past(session, kv_model, &mut target, &materialized)?;
                        Ok(target)
                    } else if historical {
                        Ok(target)
                    } else {
                        state.decode_state.clone_for_session_fork(
                            target,
                            state.kv_token_count,
                            "target.decoder_state",
                            "ort",
                        )
                    }
                })
                .map_err(|error| SessionForkError::SnapshotFailed {
                    participant: "target.decoder_state".to_string(),
                    state: source.to_string(),
                    state_type: "KV, recurrent/convolution state, and logical cursors".to_string(),
                    backend: "ort".to_string(),
                    reason: format!("{error:#}"),
                })?;

        let (prepared_draft, draft_kv, source_draft) = match (&self.draft, &state.draft) {
            (Some(model), Some(draft)) => {
                participants.push(SessionForkParticipant::new(
                    SessionForkParticipantKind::SpeculativeCascade,
                    "draft",
                    "draft tokens, KV, fixed state, and accepted-prefix cursor",
                    "ort",
                ));
                if draft.decode_state.use_kv {
                    participants.push(SessionForkParticipant::new(
                        SessionForkParticipantKind::Kv,
                        "draft.kv",
                        format!("materialized through token {}", draft.kv_token_count),
                        "ort",
                    ));
                }
                for name in draft.decode_state.fixed_state_names() {
                    participants.push(SessionForkParticipant::new(
                        SessionForkParticipantKind::RecurrentOrConvolution,
                        format!("draft.{name}"),
                        "fixed loop-carried replace state",
                        "ort",
                    ));
                }
                let draft_kv = if draft.decode_state.has_runner() || !draft.decode_state.use_kv {
                    PreparedKvFork::Empty
                } else {
                    model
                        .kv_cache
                        .validate_fork(draft.seq, draft.kv_token_count)
                        .map_err(|error| SessionForkError::UnsupportedParticipant {
                            participant: "draft.kv".to_string(),
                            state: draft.seq.to_string(),
                            state_type: "speculative paged KV".to_string(),
                            backend: "onnx-genai-kv".to_string(),
                            reason: error.to_string(),
                        })?;
                    PreparedKvFork::CopyOnWrite {
                        source: draft.seq,
                        position: draft.kv_token_count,
                    }
                };
                let target = DecodeState::new_for_path_with_io(
                    &model.session,
                    &model.decode_path,
                    model.io.as_ref(),
                )
                .and_then(|target| {
                    draft.decode_state.clone_for_session_fork(
                        target,
                        draft.kv_token_count,
                        "draft.decoder_state",
                        "ort",
                    )
                })
                .map_err(|error| SessionForkError::SnapshotFailed {
                    participant: "draft.decoder_state".to_string(),
                    state: draft.seq.to_string(),
                    state_type: "speculative KV, recurrent/convolution state, and cursor"
                        .to_string(),
                    backend: "ort".to_string(),
                    reason: format!("{error:#}"),
                })?;
                (
                    Some(DraftSession {
                        seq: draft.seq,
                        tokens: draft.tokens.clone(),
                        kv_token_count: draft.kv_token_count,
                        decode_state: target,
                    }),
                    Some(draft_kv),
                    Some((draft.seq, draft.tokens.clone(), draft.kv_token_count)),
                )
            }
            (None, None) => (None, None, None),
            _ => {
                return Err(SessionForkError::UnsupportedParticipant {
                    participant: "draft".to_string(),
                    state: source.to_string(),
                    state_type: "speculative cascade topology".to_string(),
                    backend: "ort".to_string(),
                    reason: "the engine draft model and source draft session do not form a \
                             complete participant pair"
                        .to_string(),
                });
            }
        };

        Ok(SessionForkPlan {
            source,
            position,
            participants,
            prepared: PreparedSessionFork::Ort(Box::new(PreparedOrtFork {
                origin: self.session_fork_origin,
                source_tokens: state.tokens.clone(),
                source_kv_token_count: state.kv_token_count,
                source_draft,
                state: EngineSession {
                    tokens: state.tokens[..position.get()].to_vec(),
                    kv_token_count: target_kv_position,
                    decode_state: target_decode,
                    draft: prepared_draft,
                    sampled_fastpath_failed: state.sampled_fastpath_failed,
                },
                target_kv,
                draft_kv,
            })),
        })
    }

    fn publish_ort_session_fork(
        &mut self,
        source: SessionId,
        position: SessionPosition,
        mut prepared: PreparedOrtFork,
    ) -> Result<SessionId, SessionForkError> {
        let live = self
            .sessions
            .get(&source)
            .ok_or(SessionForkError::SessionNotFound { session: source })?;
        if live.tokens != prepared.source_tokens
            || live.kv_token_count != prepared.source_kv_token_count
        {
            return Err(SessionForkError::StalePlan {
                session: source,
                position: position.get(),
                reason: "target tokens or KV cursor changed after admission".to_string(),
            });
        }
        match (&prepared.source_draft, &live.draft) {
            (Some((seq, tokens, kv_token_count)), Some(draft))
                if *seq == draft.seq
                    && *tokens == draft.tokens
                    && *kv_token_count == draft.kv_token_count => {}
            (None, None) => {}
            _ => {
                return Err(SessionForkError::StalePlan {
                    session: source,
                    position: position.get(),
                    reason: "speculative draft topology or cursor changed after admission"
                        .to_string(),
                });
            }
        }
        self.sessions.try_reserve(1).map_err(|error| {
            SessionForkError::Publication(format!(
                "failed to reserve the child decoder-session entry: {error}"
            ))
        })?;

        let child = publish_kv_fork(&mut self.kv_cache, prepared.target_kv)?;
        let child_draft = match (self.draft.as_mut(), prepared.draft_kv) {
            (Some(model), Some(kv)) => match publish_kv_fork(&mut model.kv_cache, kv) {
                Ok(seq) => Some(seq),
                Err(error) => {
                    let _ = self.kv_cache.remove(child);
                    return Err(error);
                }
            },
            (None, None) | (Some(_), None) => None,
            (None, Some(_)) => {
                let _ = self.kv_cache.remove(child);
                return Err(SessionForkError::StalePlan {
                    session: source,
                    position: position.get(),
                    reason: "draft model disappeared after admission".to_string(),
                });
            }
        };
        if let Some(draft) = prepared.state.draft.as_mut() {
            let Some(child_draft) = child_draft else {
                let _ = self.kv_cache.remove(child);
                return Err(SessionForkError::Publication(
                    "prepared draft state has no child KV sequence".to_string(),
                ));
            };
            draft.seq = child_draft;
        }
        self.sessions.insert(child, prepared.state);
        Ok(child)
    }
}

fn validate_requested_position(
    source: SessionId,
    requested: usize,
    current: usize,
    participant: &str,
) -> Result<(), SessionForkError> {
    if requested > current {
        return Err(SessionForkError::PositionOutOfBounds {
            session: source,
            requested,
            current,
        });
    }
    if requested < current {
        return Err(SessionForkError::PositionNotCommitted {
            session: source,
            requested,
            current,
            participant: participant.to_string(),
        });
    }
    Ok(())
}

fn workflow_fork_participants(
    workflow: &crate::pipeline::WorkflowRuntime,
) -> Result<Vec<SessionForkParticipant>, SessionForkError> {
    let backend = "workflow-interpreter";
    let state_plan = workflow.resolved_state_plan();
    let mut participants = vec![
        SessionForkParticipant::new(
            SessionForkParticipantKind::LogicalPosition,
            "session.logical_position",
            "current committed invocation boundary",
            backend,
        ),
        SessionForkParticipant::new(
            SessionForkParticipantKind::RandomState,
            "turn_rng",
            "empty at committed boundary; next turn constructs an independent stream",
            "runtime",
        ),
        SessionForkParticipant::new(
            SessionForkParticipantKind::GrammarConstraint,
            "turn_constraints",
            "empty at committed boundary unless represented by semantic workflow state",
            "runtime",
        ),
    ];
    let mut validated_services = BTreeSet::new();
    let mut cascade_edges = BTreeSet::new();
    for (_, cell) in state_plan
        .session_cells()
        .filter(|(_, cell)| cell.transaction.required)
    {
        let state_type = format!("{:?}, rank {}", cell.contract.dtype, cell.contract.rank());
        if cell.lifecycle.management == StateManagement::External {
            return Err(SessionForkError::UnsupportedParticipant {
                participant: "workflow.semantic_state".to_string(),
                state: cell.identity.0.clone(),
                state_type,
                backend: backend.to_string(),
                reason: "external storage exposes no typed clone/import participant".to_string(),
            });
        }
        if let Some(service) = &cell.service {
            validate_workflow_service(
                state_plan,
                &service.group,
                &mut validated_services,
                &mut cascade_edges,
                backend,
            )?;
        }
        let kind = match cell.semantic_role {
            StateSemanticRole::AttentionKv => SessionForkParticipantKind::Kv,
            StateSemanticRole::RecurrentOrConvolution => {
                SessionForkParticipantKind::RecurrentOrConvolution
            }
            StateSemanticRole::TokenContextHistory => {
                SessionForkParticipantKind::TokenContextHistory
            }
            StateSemanticRole::GrammarConstraint => SessionForkParticipantKind::GrammarConstraint,
            StateSemanticRole::Continuation => SessionForkParticipantKind::ContinuationCursor,
            StateSemanticRole::GenericFeature => SessionForkParticipantKind::GenericFeatureState,
        };
        participants.push(SessionForkParticipant::new(
            kind,
            cell.identity.0.clone(),
            state_type,
            backend,
        ));
    }
    for (from, to) in cascade_edges {
        participants.push(SessionForkParticipant::new(
            SessionForkParticipantKind::SpeculativeCascade,
            format!("{from}->{to}"),
            "transitive state-service fork cascade",
            backend,
        ));
    }
    for effect in workflow.transaction_effect_domains() {
        participants.push(SessionForkParticipant::new(
            SessionForkParticipantKind::TransactionalEffect,
            effect,
            "committed effect cursor",
            backend,
        ));
    }
    for output in workflow.output_names() {
        participants.push(SessionForkParticipant::new(
            SessionForkParticipantKind::OutputPublication,
            output,
            "committed head, cursor, lineage, and closure",
            backend,
        ));
    }
    participants.sort_by(|left, right| {
        (left.kind, left.name.as_str()).cmp(&(right.kind, right.name.as_str()))
    });
    Ok(participants)
}

fn validate_workflow_service(
    state_plan: &onnx_genai_metadata::ResolvedStatePlan,
    group: &str,
    validated: &mut BTreeSet<String>,
    cascade_edges: &mut BTreeSet<(String, String)>,
    backend: &str,
) -> Result<(), SessionForkError> {
    if !validated.insert(group.to_string()) {
        return Ok(());
    }
    let service =
        state_plan
            .service(group)
            .ok_or_else(|| SessionForkError::UnsupportedParticipant {
                participant: "workflow.state_service".to_string(),
                state: group.to_string(),
                state_type: "unresolved state-service group".to_string(),
                backend: backend.to_string(),
                reason: "the canonical ResolvedStatePlan has no participant for this cascade"
                    .to_string(),
            })?;
    if !service.snapshot || !service.fork {
        return Err(SessionForkError::UnsupportedParticipant {
            participant: "workflow.state_service".to_string(),
            state: group.to_string(),
            state_type: format!(
                "{:?} state service (snapshot={}, fork={})",
                service.kind, service.snapshot, service.fork
            ),
            backend: backend.to_string(),
            reason: "declare both capabilities.snapshot and capabilities.fork only when the \
                     backend can reproduce the complete state independently"
                .to_string(),
        });
    }
    for cascade in &service.cascade {
        cascade_edges.insert((group.to_string(), cascade.clone()));
        validate_workflow_service(state_plan, cascade, validated, cascade_edges, backend)?;
    }
    Ok(())
}

fn publish_kv_fork(
    cache: &mut PagedKvCache,
    prepared: PreparedKvFork,
) -> Result<SessionId, SessionForkError> {
    match prepared {
        PreparedKvFork::Empty => Ok(cache.create_sequence()),
        PreparedKvFork::CopyOnWrite { source, position } => {
            cache.fork(source, position).map_err(|error| {
                SessionForkError::Publication(format!(
                    "failed to publish copy-on-write KV child from sequence {source} at token \
                 {position}: {error}"
                ))
            })
        }
    }
}
