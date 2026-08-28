//! Chained speculative proposal driving, owned by the universal interpreter.
//!
//! A `chained` proposer emits one distribution per invocation, so materializing
//! a proposal block means driving the *same* typed component repeatedly while
//! threading its recurrence forward. That loop used to live in a direct-`Engine`
//! Rust `propose()` bound to ORT `Session`s, which made it a second execution
//! engine that the backend-neutral component seam could not drive. It lives here
//! now: every proposer step runs through
//! [`WorkflowRuntime::invoke_component_values`], so ORT and native execute the
//! identical chain.
//!
//! Everything the loop needs is read from the package's
//! `speculative.proposal_execution` contract — never inferred from a model name,
//! a port spelling, or a tensor shape:
//!
//! * `token_embedding_input` — the proposer port receiving the fused input.
//! * `logits_output` — the port carrying the next-token distribution.
//! * `recurrent[]` — `{state, input, output}` triples the loop threads and
//!   checkpoints, one workflow state cell each.
//! * `folded_carry_output` — the proposer output carrying carry_k (k >= 1).
//! * `folded_carry_seed` — `{component, output}`: the target output read as
//!   carry_0.
//! * `token_embedding` — `{component, table}`: the embedding table the leading
//!   half of the fused input is gathered from.
//!
//! A folded carry re-enters as the *trailing* half of the fused input and owns
//! no state cell, so it is recomputed from committed tokens on rejection rather
//! than restored — which is why it never appears in `rollback_state`.

use std::collections::BTreeMap;

use anyhow::Context as _;
use onnx_genai_metadata::{
    CandidateTreeTopology, DFlashStateCommit, DFlashStructure, SpeculativeContract,
    SpeculativeProposalExecution, SpeculativeRecurrenceBinding, StatePortAccess,
    WorkflowOutputFamily, WorkflowSpec, WorkflowStep,
};
use onnx_genai_ort::{DataType, Value};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};

use super::execution_admission::{
    CandidateTreeExecutionMode, CandidateTreeExecutionPlan, CandidateTreeTopologyInput,
};
use super::workflow::WorkflowLoopHostOutcome;
use super::{PipelineTensors, WorkflowRuntime};
use crate::config::{FinishReason, GenerateOptions, GenerateResult, GenerateTokenCallback};
use crate::speculative::{
    AcceptanceRule, SamplingRandomness, SpecTree, TreeSamplingInputs, verify_tree_sampling,
};

/// One materialized chained proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainedProposal {
    /// The proposal block: the target's guaranteed token followed by the drafts
    /// the proposer chain produced, in order.
    pub tokens: Vec<i64>,
    /// Proposer invocations this proposal cost, including the bootstrap step.
    pub proposer_invocations: usize,
}

impl ChainedProposal {
    /// Draft tokens only — the block minus the target's guaranteed first token.
    pub fn drafts(&self) -> &[i64] {
        &self.tokens[1..]
    }
}

/// How a proposal block compared against the target's own tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalAcceptance {
    /// Number of proposal-block positions the target confirmed, always >= 1:
    /// position 0 is the target's own guaranteed token.
    pub accepted: usize,
    /// Index of the first rejected draft within the proposal block, if any.
    pub rejected_at: Option<usize>,
    /// Tokens to commit for this verification pass.
    pub committed: Vec<i64>,
}

impl ProposalAcceptance {
    /// Whether a draft was rejected and declared rollback state must be undone.
    pub fn requires_rollback(&self) -> bool {
        self.rejected_at.is_some()
    }
}

/// Inputs a chained proposal needs beyond what the package declares.
#[derive(Debug, Clone, Copy)]
pub struct ChainedProposalOptions {
    /// The last committed context token, whose embedding seeds the first fused
    /// input. carry_0 is the target's hidden state for exactly this position.
    pub seed_token: i64,
    /// The target's own next token, taken for free and used to condition every
    /// draft that follows it.
    pub guaranteed_token: i64,
    /// Proposal block width, capped by the contract's `max_proposal_width`.
    pub width: usize,
}

/// One DFlash flat-block proposal.
pub struct DFlashProposal {
    /// Candidate tokens after the verifier-produced anchor.
    pub tokens: Vec<i64>,
    /// Proposal distribution required by rejection sampling.
    pub probabilities: Option<DFlashProposalProbabilities>,
    /// Proposer outputs retained for accepted-prefix draft-state commit.
    proposer_outputs: PipelineTensors,
    conditioning_trace: Vec<DFlashConditioningTrace>,
    verification_seed: u64,
}

impl std::fmt::Debug for DFlashProposal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DFlashProposal")
            .field("tokens", &self.tokens)
            .field("probabilities", &self.probabilities)
            .field(
                "proposer_outputs",
                &self.proposer_outputs.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Exact proposal-distribution representation emitted by a DFlash contract.
#[derive(Debug, Clone, PartialEq)]
pub enum DFlashProposalProbabilities {
    /// One full target-vocabulary distribution per proposed position.
    FullVocabulary {
        values: Vec<f32>,
        proposal: usize,
        vocabulary: usize,
    },
    /// DFlash 2 selector distribution over explicitly emitted candidate ids.
    SparseCandidates {
        candidate_ids: Vec<i64>,
        values: Vec<f32>,
        proposal: usize,
        candidates: usize,
        vocabulary: usize,
    },
}

impl DFlashProposalProbabilities {
    fn proposal_len(&self) -> usize {
        match self {
            Self::FullVocabulary { proposal, .. } | Self::SparseCandidates { proposal, .. } => {
                *proposal
            }
        }
    }

    fn vocabulary(&self) -> usize {
        match self {
            Self::FullVocabulary { vocabulary, .. } | Self::SparseCandidates { vocabulary, .. } => {
                *vocabulary
            }
        }
    }

    fn row(&self, position: usize) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::FullVocabulary {
                values,
                proposal,
                vocabulary,
            } => {
                anyhow::ensure!(
                    position < *proposal,
                    "proposal probability row {position} is outside {proposal} positions"
                );
                let start = position * vocabulary;
                Ok(values[start..start + vocabulary].to_vec())
            }
            Self::SparseCandidates {
                candidate_ids,
                values,
                proposal,
                candidates,
                vocabulary,
            } => {
                anyhow::ensure!(
                    position < *proposal,
                    "proposal probability row {position} is outside {proposal} positions"
                );
                let mut dense = vec![0.0f32; *vocabulary];
                let start = position * candidates;
                for (&token, &probability) in candidate_ids[start..start + candidates]
                    .iter()
                    .zip(&values[start..start + candidates])
                {
                    let token = usize::try_from(token)
                        .ok()
                        .filter(|token| *token < *vocabulary)
                        .with_context(|| {
                            format!(
                                "DFlash sparse candidate token {token} at proposal position \
                                 {position} is outside vocabulary {vocabulary}"
                            )
                        })?;
                    dense[token] += probability;
                }
                Ok(dense)
            }
        }
    }

    fn truncate(&mut self, proposal_len: usize) {
        match self {
            Self::FullVocabulary {
                values,
                proposal,
                vocabulary,
            } => {
                *proposal = proposal_len.min(*proposal);
                values.truncate(*proposal * *vocabulary);
            }
            Self::SparseCandidates {
                candidate_ids,
                values,
                proposal,
                candidates,
                ..
            } => {
                *proposal = proposal_len.min(*proposal);
                candidate_ids.truncate(*proposal * *candidates);
                values.truncate(*proposal * *candidates);
            }
        }
    }
}

/// Runtime-selected DFlash proposal mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DFlashProposalMode {
    Greedy,
    Sampling { seed: u64 },
}

/// Request-local values needed to materialize one DFlash block.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashProposalOptions {
    /// Verifier-produced token at block position zero.
    pub anchor_token: i64,
    /// Number of masked candidate positions to predict.
    pub width: usize,
    /// Absolute position of the first target-hidden context row.
    pub context_start_position: i64,
    /// Candidate selection mode.
    pub mode: DFlashProposalMode,
    /// EOS ids crop the proposal at the first EOS, inclusive.
    pub eos_token_ids: Vec<i64>,
}

/// Result of exact target verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlashAcceptance {
    /// Proposed positions accepted before correction/bonus.
    pub accepted: usize,
    /// First rejected proposal position, absent on full acceptance.
    pub rejected_at: Option<usize>,
    /// Accepted proposal prefix followed by the target correction or bonus.
    pub committed: Vec<i64>,
}

impl DFlashAcceptance {
    pub fn requires_rollback(&self) -> bool {
        self.rejected_at.is_some()
    }
}

/// Sampling parameters for exact DFlash verification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DFlashVerificationMode {
    Greedy,
    Sampling { temperature: f32 },
}

/// Inspectable DFlash admission facts. This reports the contract the package
/// declared; it contains no scheduling, kernel, or proposal-budget policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlashDiagnostic {
    pub version: String,
    pub proposer: String,
    pub target: String,
    pub target_hidden_sources: Vec<String>,
    pub max_proposal_width: usize,
    pub proposal_probabilities: bool,
    pub rollback_participants: Vec<String>,
    pub draft_private_state: Vec<String>,
    pub structure: &'static str,
    pub shared_batching_supported: bool,
}

/// Inspectable facts for the exact candidate-tree executor selected from a
/// canonical `onnx-genai.speculative@1` declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateTreeDiagnostic {
    pub version: String,
    pub proposer: String,
    pub target: String,
    pub topology: &'static str,
    pub max_proposal_width: usize,
    pub distribution_preserving: bool,
    pub proposal_probabilities: bool,
    pub rollback_participants: Vec<String>,
    pub shared_batching_supported: bool,
}

/// Evidence from one committed candidate-tree verification block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateTreeBlockTrace {
    pub candidates: Vec<u32>,
    pub parents: Vec<Option<usize>>,
    pub accepted_nodes: Vec<usize>,
    pub committed_tokens: Vec<u32>,
}

/// Typed cancellation of an admitted candidate-tree turn.
#[derive(Debug, thiserror::Error)]
#[error(
    "candidate-tree generation cancelled {boundary}; the admitted turn aborted to its explicit \
     committed baseline"
)]
pub struct CandidateTreeGenerationCancelled {
    pub boundary: super::GenerationBoundary,
    pub outcome: super::TurnTransactionOutcome,
}

/// One target feature tensor that actually conditioned a committed DFlash
/// proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlashConditioningTrace {
    pub source: String,
    pub shape: Vec<i64>,
}

/// Transport-neutral evidence from one committed DFlash block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlashBlockTrace {
    pub conditioning: Vec<DFlashConditioningTrace>,
    pub proposer_candidates: Vec<i64>,
    pub accepted: usize,
    pub committed_tokens: Vec<i64>,
}

/// Typed cancellation of an admitted DFlash generation.
#[derive(Debug, thiserror::Error)]
#[error(
    "DFlash generation cancelled {boundary}; the admitted turn aborted to its explicit \
     committed baseline"
)]
pub struct DFlashGenerationCancelled {
    pub boundary: super::GenerationBoundary,
    pub outcome: super::TurnTransactionOutcome,
}

/// Output delivery failed after the candidate-tree transaction committed.
#[derive(Debug, thiserror::Error)]
#[error(
    "candidate-tree output delivery failed after semantic commit for {committed_tokens} token(s); \
     committed state remains durable and this partial delivery will not be replayed \
     automatically: {message}"
)]
pub struct CandidateTreeOutputDeliveryError {
    pub committed_tokens: usize,
    pub message: String,
}
/// S3 transaction participant for target, draft-private, recurrent, and
/// token-context state advanced by one DFlash block.
pub struct DFlashStateTransaction {
    turn: super::TurnTransaction,
    baseline: PipelineTensors,
}

impl std::fmt::Debug for DFlashStateTransaction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DFlashStateTransaction")
            .field("states", &self.baseline.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl CandidateTreeExecutionPlan {
    fn state_for_input<'workflow>(
        &self,
        workflow: &'workflow WorkflowSpec,
        component: &str,
        input: &str,
    ) -> Option<&'workflow str> {
        workflow
            .serving
            .as_ref()?
            .state_service
            .groups
            .values()
            .find_map(|group| {
                group
                    .ports
                    .get(component)?
                    .iter()
                    .find_map(|(cell, alias)| {
                        (alias.access == StatePortAccess::ReadWrite && alias.input == input)
                            .then_some(cell.as_str())
                    })
            })
    }

    fn initial_state(
        &self,
        workflow: &WorkflowSpec,
        values: &PipelineTensors,
    ) -> anyhow::Result<PipelineTensors> {
        let mut state = PipelineTensors::new();
        for cell in &self.rollback_state {
            let Some(source) = [&self.proposer, &self.target]
                .into_iter()
                .find_map(|component| {
                    let alias = self.state_alias(workflow, component, cell)?;
                    let bindings = if component == &self.proposer {
                        &self.proposer_bindings
                    } else {
                        &self.target_bindings
                    };
                    bindings.get(&alias.input)
                })
            else {
                continue;
            };
            state.insert(
                cell.clone(),
                crate::decode::clone_value(values.get(source).with_context(|| {
                    format!(
                        "candidate-tree rollback participant '{cell}' starts from unavailable \
                         workflow value '{source}'"
                    )
                })?)?,
            );
        }
        Ok(state)
    }
}

struct RuntimeCandidateTree {
    tree: SpecTree,
    ancestor_mask: Value,
    position_ids: Value,
}

struct PendingCandidateTree {
    context: Vec<i64>,
    tree: RuntimeCandidateTree,
}

struct CandidateTreeWorkflowHost<'a> {
    runtime: &'a WorkflowRuntime,
    plan: CandidateTreeExecutionPlan,
    contract: &'a SpeculativeContract,
    options: &'a GenerateOptions,
    tokenizer: Option<&'a onnx_genai_ort::Tokenizer>,
    control: super::GenerationControl,
    rng: StdRng,
    turn_identity: Option<(super::TurnTransactionId, super::TurnBaselineId)>,
    candidate_state: Option<PipelineTensors>,
    pending: Option<PendingCandidateTree>,
    generated: Vec<u32>,
    traces: Vec<CandidateTreeBlockTrace>,
    blocks: u64,
    finish_reason: FinishReason,
    text: String,
    token_text: Vec<String>,
    commit_started: bool,
    loop_stop: Option<super::GenerationStopReason>,
    staged_output_observer: Option<&'a mut super::ToolCallStagedOutputObserver>,
}

impl CandidateTreeWorkflowHost<'_> {
    fn transaction_outcome(&self, reason: super::TurnAbortReason) -> super::TurnTransactionOutcome {
        let (transaction, baseline) = self
            .turn_identity
            .expect("candidate-tree host observes boundaries only after turn admission");
        super::TurnTransactionOutcome::AbortToBaseline {
            transaction,
            baseline,
            reason,
        }
    }

    fn observe(&self, boundary: super::GenerationBoundary) -> anyhow::Result<()> {
        match self.control.observe(boundary) {
            Ok(false) => Ok(()),
            Ok(true) => Err(anyhow::Error::new(CandidateTreeGenerationCancelled {
                boundary,
                outcome: self.transaction_outcome(super::TurnAbortReason::Cancellation),
            })),
            Err(error) => Err(error.context(format!(
                "candidate-tree generation checkpoint failed {boundary}"
            ))),
        }
    }

    fn context_value(context: &[i64]) -> anyhow::Result<Value> {
        Ok(Value::from_slice_i64(
            context,
            &[
                1,
                i64::try_from(context.len())
                    .context("candidate-tree context length exceeds i64")?,
            ],
        )?)
    }

    fn ensure_candidate_state(&mut self, values: &PipelineTensors) -> anyhow::Result<()> {
        if self.candidate_state.is_none() {
            self.candidate_state = Some(
                self.plan
                    .initial_state(&self.runtime.plan.workflow, values)?,
            );
        }
        Ok(())
    }

    fn invoke_proposer(
        &mut self,
        context: Vec<i64>,
        values: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.pending.is_none(),
            "candidate-tree proposer at {} was entered twice before target {} completed the \
             admitted verification seam",
            self.plan.proposer_path,
            self.plan.target_path
        );
        anyhow::ensure!(
            self.generated.len() < self.options.max_new_tokens,
            "candidate-tree proposer at {} was entered after the request's max_new_tokens budget \
             was exhausted; make the authored loop condition false before re-entering the seam",
            self.plan.proposer_path
        );
        if self
            .options
            .max_context
            .is_some_and(|limit| context.len() >= limit)
        {
            anyhow::bail!(
                "candidate-tree proposer at {} was entered with context length {}, which already \
                 reaches max_context {}; make the authored loop zero-trip at the context bound",
                self.plan.proposer_path,
                context.len(),
                self.options.max_context.unwrap_or_default()
            );
        }
        self.ensure_candidate_state(values)?;
        self.observe(super::GenerationBoundary::BeforeProposer)?;
        let context_value = Self::context_value(&context)?;
        let empty_accepted = Value::from_slice_i64(&[], &[1, 0])?;
        self.runtime.invoke_candidate_tree_component(
            &self.plan,
            &self.contract.proposer,
            &context_value,
            None,
            &empty_accepted,
            self.candidate_state
                .as_ref()
                .expect("candidate-tree state was initialized"),
            values,
        )?;
        self.observe(super::GenerationBoundary::AfterProposer)?;
        let tree = self
            .runtime
            .decode_runtime_candidate_tree(&self.plan, values)?;
        self.pending = Some(PendingCandidateTree { context, tree });
        Ok(())
    }

    fn clone_values(values: &PipelineTensors) -> anyhow::Result<PipelineTensors> {
        values
            .iter()
            .map(|(name, value)| Ok((name.clone(), crate::decode::clone_value(value)?)))
            .collect()
    }

    fn accepted_state_updates(
        &self,
        component: &str,
        values: &PipelineTensors,
    ) -> anyhow::Result<Vec<(String, String, Value)>> {
        let outputs = if component == self.plan.proposer {
            &self.plan.proposer_outputs
        } else {
            &self.plan.target_outputs
        };
        let mut updates = Vec::new();
        for cell in &self.plan.rollback_state {
            let Some(alias) = self
                .plan
                .state_alias(&self.runtime.plan.workflow, component, cell)
            else {
                continue;
            };
            let Some(port) = alias.output.as_deref() else {
                continue;
            };
            let binding = outputs.get(port).with_context(|| {
                format!(
                    "candidate-tree state '{cell}' output port '{component}::{port}' has no \
                     admitted SSA binding"
                )
            })?;
            let value = crate::decode::clone_value(values.get(binding).with_context(|| {
                format!(
                    "candidate-tree accepted-path recomputation did not produce state '{cell}' \
                     binding '{binding}'"
                )
            })?)?;
            updates.push((cell.clone(), binding.clone(), value));
        }
        Ok(updates)
    }

    fn recompute_accepted_state(
        &mut self,
        context: &[i64],
        accepted: &[u32],
        tree: &RuntimeCandidateTree,
        values: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        let context = Self::context_value(context)?;
        let accepted_i64 = accepted.iter().copied().map(i64::from).collect::<Vec<_>>();
        let accepted_value = Value::from_slice_i64(
            &accepted_i64,
            &[
                1,
                i64::try_from(accepted_i64.len())
                    .context("candidate-tree accepted path length exceeds i64")?,
            ],
        )?;
        let state = self
            .candidate_state
            .as_ref()
            .expect("candidate-tree state was initialized");

        let mut target_values = Self::clone_values(values)?;
        self.runtime.invoke_candidate_tree_component(
            &self.plan,
            &self.contract.target,
            &context,
            Some(tree),
            &accepted_value,
            state,
            &mut target_values,
        )?;
        let target_updates = self.accepted_state_updates(&self.contract.target, &target_values)?;

        let mut proposer_values = Self::clone_values(values)?;
        let empty_accepted = Value::from_slice_i64(&[], &[1, 0])?;
        self.runtime.invoke_candidate_tree_component(
            &self.plan,
            &self.contract.proposer,
            &context,
            None,
            &empty_accepted,
            state,
            &mut proposer_values,
        )?;
        let proposer_updates =
            self.accepted_state_updates(&self.contract.proposer, &proposer_values)?;

        let candidate_state = self
            .candidate_state
            .as_mut()
            .expect("candidate-tree state was initialized");
        for (cell, binding, value) in proposer_updates.into_iter().chain(target_updates) {
            candidate_state.insert(cell, crate::decode::clone_value(&value)?);
            values.insert(binding, value);
        }
        values.insert(self.plan.accepted_path_binding.clone(), accepted_value);
        Ok(())
    }

    fn publish_accepted(
        &self,
        accepted: &[u32],
        publication_journal: &mut Option<super::output::OutputPublicationJournal>,
    ) -> anyhow::Result<()> {
        if !self.plan.publish_tokens_at_seam {
            return Ok(());
        }
        let output = self
            .runtime
            .plan
            .workflow
            .outputs
            .get(&self.plan.token_output)
            .expect("candidate-tree admission resolved the token output");
        let mode = match &output.family {
            WorkflowOutputFamily::Materialized | WorkflowOutputFamily::Revisions { .. } => {
                onnx_genai_metadata::WorkflowEmitMode::Append
            }
            WorkflowOutputFamily::Events => onnx_genai_metadata::WorkflowEmitMode::Event,
        };
        let journal = publication_journal.as_mut().context(
            "candidate-tree accepted path has no enclosing S4 output publication journal",
        )?;
        for token in accepted {
            journal.publish(
                &self.plan.token_output,
                None,
                &mode,
                Some(Value::from_slice_i64(&[i64::from(*token)], &[1])?),
            )?;
        }
        Ok(())
    }

    fn complete_target(
        &mut self,
        values: &mut PipelineTensors,
        publication_journal: &mut Option<super::output::OutputPublicationJournal>,
    ) -> anyhow::Result<Vec<i64>> {
        let pending = self.pending.take().with_context(|| {
            format!(
                "candidate-tree target at {} ran without proposer {} at {} on the same authored \
                 control path",
                self.plan.target_path, self.contract.proposer, self.plan.proposer_path
            )
        })?;
        let context_value = Self::context_value(&pending.context)?;
        let empty_accepted = Value::from_slice_i64(&[], &[1, 0])?;
        self.runtime.invoke_candidate_tree_component(
            &self.plan,
            &self.contract.target,
            &context_value,
            Some(&pending.tree),
            &empty_accepted,
            self.candidate_state
                .as_ref()
                .expect("candidate-tree state was initialized"),
            values,
        )?;
        self.observe(super::GenerationBoundary::AfterVerifier)?;
        let outcome = if self.options.selects_greedily() {
            self.runtime
                .verify_candidate_tree_greedy(&self.plan, &pending.tree.tree, values)?
        } else {
            self.runtime.verify_candidate_tree_sampling(
                &self.plan,
                &pending.tree.tree,
                values,
                &mut self.rng,
            )?
        };
        let remaining = self
            .options
            .max_new_tokens
            .saturating_sub(self.generated.len());
        let remaining_context = self
            .options
            .max_context
            .map(|limit| limit.saturating_sub(pending.context.len()))
            .unwrap_or(remaining);
        let mut committed = outcome
            .tokens
            .into_iter()
            .take(remaining.min(remaining_context))
            .collect::<Vec<_>>();
        if let Some(eos) = committed
            .iter()
            .position(|token| self.options.terminates(*token))
        {
            committed.truncate(eos + 1);
            self.finish_reason = FinishReason::EosToken;
        }
        anyhow::ensure!(
            !committed.is_empty(),
            "candidate-tree target at {} produced no token within the request budget/context; \
             the authored control path must not enter the seam after it becomes zero-trip",
            self.plan.target_path
        );
        let mut accepted_context = pending.context.clone();
        accepted_context.extend(committed.iter().copied().map(i64::from));
        self.recompute_accepted_state(&accepted_context, &committed, &pending.tree, values)?;
        self.publish_accepted(&committed, publication_journal)?;
        self.traces.push(CandidateTreeBlockTrace {
            candidates: pending
                .tree
                .tree
                .nodes()
                .iter()
                .map(|node| node.token)
                .collect(),
            parents: pending
                .tree
                .tree
                .nodes()
                .iter()
                .map(|node| node.parent)
                .collect(),
            accepted_nodes: outcome.nodes,
            committed_tokens: committed.clone(),
        });
        self.generated.extend_from_slice(&committed);
        self.blocks = self.blocks.saturating_add(1);
        if let Some(observer) = self.staged_output_observer.as_deref_mut() {
            let tokenizer = self.tokenizer.context(
                "candidate-tree tool-call observation requires the package tokenizer before commit",
            )?;
            observer
                .observe_tokens(tokenizer, &committed)
                .map_err(|error| {
                    anyhow::Error::new(error).context(
                        "candidate-tree staged tool-call observation failed before semantic commit",
                    )
                })?;
        }
        let mut loop_stop = if self.finish_reason == FinishReason::EosToken {
            Some(super::GenerationStopReason::EosCommitted)
        } else if self.generated.len() >= self.options.max_new_tokens {
            Some(super::GenerationStopReason::BudgetExhausted)
        } else if self
            .options
            .max_context
            .is_some_and(|limit| accepted_context.len() >= limit)
        {
            self.finish_reason = FinishReason::Length;
            Some(super::GenerationStopReason::ContextExhausted)
        } else {
            None
        };
        if loop_stop.is_some()
            && let Some(observer) = self.staged_output_observer.as_deref_mut()
            && matches!(
                observer
                    .finish("candidate-tree terminal generation boundary")
                    .map_err(|error| {
                        anyhow::Error::new(error)
                            .context("validate candidate-tree tool protocol before semantic commit")
                    })?,
                super::StagedOutputObservation::TerminalComplete(_)
            )
        {
            loop_stop = observer.stop_reason();
            self.finish_reason = loop_stop
                .as_ref()
                .expect("terminal tool observer supplies a stop reason")
                .finish_reason();
        }
        self.loop_stop = loop_stop;
        Ok(accepted_context)
    }

    fn execute_target(
        &mut self,
        values: &mut PipelineTensors,
        publication_journal: &mut Option<super::output::OutputPublicationJournal>,
    ) -> anyhow::Result<()> {
        let mut context = self.complete_target(values, publication_journal)?;
        if self.plan.execution_mode == CandidateTreeExecutionMode::DrainAtSeam {
            while self.generated.len() < self.options.max_new_tokens
                && self.finish_reason != FinishReason::EosToken
                && !self
                    .options
                    .max_context
                    .is_some_and(|limit| context.len() >= limit)
            {
                self.invoke_proposer(context, values)?;
                context = self.complete_target(values, publication_journal)?;
            }
            if self
                .options
                .max_context
                .is_some_and(|limit| context.len() >= limit)
                && self.generated.len() < self.options.max_new_tokens
            {
                self.finish_reason = FinishReason::Length;
            }
            let all_tokens = self
                .generated
                .iter()
                .copied()
                .map(i64::from)
                .collect::<Vec<_>>();
            values.insert(
                self.plan.accepted_path_binding.clone(),
                Value::from_slice_i64(
                    &all_tokens,
                    &[
                        1,
                        i64::try_from(all_tokens.len())
                            .context("candidate-tree accepted path length exceeds i64")?,
                    ],
                )?,
            );
        }
        Ok(())
    }
}

impl super::workflow::WorkflowNodeHost for CandidateTreeWorkflowHost<'_> {
    fn hosted_contracts(&self) -> &'static [&'static str] {
        &[]
    }

    fn hosts_invocation(
        &self,
        component: &str,
        inputs: &std::collections::BTreeMap<String, String>,
        outputs: &std::collections::BTreeMap<String, String>,
    ) -> bool {
        (component == self.plan.proposer
            && inputs == &self.plan.proposer_bindings
            && outputs == &self.plan.proposer_outputs)
            || (component == self.plan.target
                && inputs == &self.plan.target_bindings
                && outputs == &self.plan.target_outputs)
    }

    fn begin_turn(&mut self, turn: &super::TurnTransaction) -> anyhow::Result<()> {
        let super::TurnTransactionOutcome::Committed {
            transaction,
            baseline,
        } = turn.committed()
        else {
            unreachable!("a newly admitted transaction is represented by its committed identity")
        };
        self.turn_identity = Some((transaction, baseline));
        if let Some(observer) = self.staged_output_observer.as_deref_mut() {
            observer.begin_turn();
        }
        Ok(())
    }

    fn observe_boundary(&mut self, boundary: super::GenerationBoundary) -> anyhow::Result<()> {
        self.observe(boundary)
    }

    fn before_turn_commit(&mut self, turn: &super::TurnTransaction) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.pending.is_none(),
            "candidate-tree proposer at {} executed without its target at {}; the generic \
             workflow cannot commit a half-entered verification seam",
            self.plan.proposer_path,
            self.plan.target_path
        );
        self.observe(super::GenerationBoundary::BeforeAcceptedPathCommit)?;
        self.text = self
            .tokenizer
            .map(|tokenizer| tokenizer.decode(&self.generated))
            .transpose()
            .context("decode committed candidate-tree output before semantic commit")?
            .unwrap_or_default();
        self.token_text = self
            .generated
            .iter()
            .map(|token| {
                self.tokenizer
                    .map(|tokenizer| tokenizer.decode(&[*token]))
                    .transpose()
                    .map(|text| text.unwrap_or_default())
            })
            .collect::<Result<Vec<_>, _>>()
            .context("decode candidate-tree token events before semantic commit")?;
        if let Some(observer) = self.staged_output_observer.as_deref_mut()
            && matches!(
                observer
                    .finish("candidate-tree semantic commit")
                    .map_err(|error| {
                        anyhow::Error::new(error)
                            .context("validate candidate-tree tool protocol before semantic commit")
                    })?,
                super::StagedOutputObservation::TerminalComplete(_)
            )
        {
            let stop = observer
                .stop_reason()
                .expect("terminal tool observer supplies a stop reason");
            self.finish_reason = stop.finish_reason();
            self.loop_stop = Some(stop);
        }
        match self.control.begin_commit() {
            Ok(true) => {
                self.commit_started = true;
                Ok(())
            }
            Ok(false) => Err(anyhow::Error::new(CandidateTreeGenerationCancelled {
                boundary: super::GenerationBoundary::BeforeSemanticCommit,
                outcome: turn.abort(super::TurnAbortReason::Cancellation),
            })),
            Err(error) => {
                Err(error.context("candidate-tree checkpoint failed before semantic commit"))
            }
        }
    }

    fn loop_host_outcome(&self) -> WorkflowLoopHostOutcome {
        self.loop_stop.clone().map_or(
            WorkflowLoopHostOutcome::Continue,
            WorkflowLoopHostOutcome::Stop,
        )
    }

    fn turn_committed(&mut self, _outcome: super::TurnTransactionOutcome) {
        if let Some(observer) = self.staged_output_observer.as_deref_mut() {
            observer.commit_turn();
        }
        if self.commit_started {
            self.control.finish_commit();
        }
    }

    fn turn_aborted(&mut self, _outcome: super::TurnTransactionOutcome) {
        if let Some(observer) = self.staged_output_observer.as_deref_mut() {
            observer.abort_turn();
        }
        self.control.abort_commit();
    }

    fn execute_contract_node(
        &mut self,
        request: super::workflow::WorkflowNodeRequest<'_>,
    ) -> anyhow::Result<bool> {
        if request.component == self.plan.proposer {
            let context_binding = request
                .inputs
                .get(&self.plan.proposer_context_input)
                .expect("candidate-tree plan resolved proposer context binding");
            let context = request
                .values
                .get(context_binding)
                .with_context(|| {
                    format!(
                        "candidate-tree proposer context binding '{context_binding}' is \
                         unavailable at {}",
                        self.plan.proposer_path
                    )
                })?
                .to_vec_i64()?;
            self.invoke_proposer(context, request.values)?;
            return Ok(true);
        }
        if request.component == self.plan.target {
            self.execute_target(request.values, request.publication_journal)?;
            return Ok(true);
        }
        Ok(false)
    }
}

fn probability_or_logits_rows(
    value: &Value,
    expected_rows: usize,
    authority: &str,
    probabilities: bool,
) -> anyhow::Result<Vec<Vec<f32>>> {
    anyhow::ensure!(
        matches!(
            value.dtype(),
            DataType::Float16 | DataType::BFloat16 | DataType::Float32
        ),
        "candidate-tree {authority} have {:?}; expected float16, bfloat16, or float32",
        value.dtype()
    );
    let shape = value.shape();
    anyhow::ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[1] >= 0 && shape[2] > 0,
        "candidate-tree {authority} have shape {shape:?}; expected \
         [1, tree_nodes_plus_anchor, vocabulary]"
    );
    let rows = usize::try_from(shape[1]).context("candidate-tree row extent is negative")?;
    let vocabulary =
        usize::try_from(shape[2]).context("candidate-tree vocabulary extent is negative")?;
    anyhow::ensure!(
        rows == expected_rows,
        "candidate-tree {authority} have {rows} rows, but flattened path ordering requires \
         exactly {expected_rows}: anchor row 0 followed by one row per proposer node"
    );
    let data = value.to_vec_f32_lossy()?;
    let rows = data
        .chunks_exact(vocabulary)
        .enumerate()
        .map(|(row, values)| {
            anyhow::ensure!(
                values.iter().all(|value| value.is_finite()),
                "candidate-tree {authority} row {row} contains a non-finite value"
            );
            if probabilities {
                anyhow::ensure!(
                    values.iter().all(|value| *value >= 0.0),
                    "candidate-tree {authority} row {row} contains a negative probability"
                );
                let total = values.iter().sum::<f32>();
                anyhow::ensure!(
                    (total - 1.0).abs() <= 1e-4,
                    "candidate-tree {authority} row {row} sums to {total}, expected 1"
                );
            }
            Ok(values.to_vec())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(
        rows.len() == expected_rows,
        "candidate-tree {authority} payload length does not match its declared axes"
    );
    Ok(rows)
}

fn validate_candidate_probability_support(
    tree: &SpecTree,
    proposal: &[Vec<f32>],
) -> anyhow::Result<()> {
    for parent in std::iter::once(None).chain((0..tree.len()).map(Some)) {
        let children = parent.map_or_else(|| tree.roots(), |node| tree.children(node));
        if children.is_empty() {
            continue;
        }
        let row = parent.map_or(0, |node| node + 1);
        let expected = children
            .iter()
            .map(|node| tree.nodes()[*node].token as usize)
            .collect::<std::collections::HashSet<_>>();
        for token in &expected {
            anyhow::ensure!(
                *token < proposal[row].len(),
                "candidate-tree proposal frontier token {token} is outside vocabulary {}",
                proposal[row].len()
            );
            anyhow::ensure!(
                proposal[row][*token] > 0.0,
                "candidate-tree proposal row {row} gives declared frontier token {token} zero \
                 probability"
            );
        }
        for (token, probability) in proposal[row].iter().copied().enumerate() {
            anyhow::ensure!(
                probability <= 1e-6 || expected.contains(&token),
                "candidate-tree proposal row {row} assigns probability {probability} to token \
                 {token}, which is not a declared child of parent {parent:?}; the sampled path \
                 could leave the emitted tree"
            );
        }
    }
    Ok(())
}

fn sample_declared_distribution(probabilities: &[f32], random: f32) -> anyhow::Result<usize> {
    anyhow::ensure!(
        random.is_finite() && (0.0..1.0).contains(&random),
        "candidate-tree sampling random variate {random} is outside [0, 1)"
    );
    let mut cumulative = 0.0_f32;
    for (token, probability) in probabilities.iter().copied().enumerate() {
        cumulative += probability;
        if random < cumulative {
            return Ok(token);
        }
    }
    probabilities
        .iter()
        .rposition(|probability| *probability > 0.0)
        .context("candidate-tree sampling distribution has empty support")
}

/// Resolved, contract-derived view of a chained proposal execution.
#[derive(Debug)]
struct ChainedPlan<'a> {
    proposer: &'a str,
    token_embedding_input: &'a str,
    logits_output: &'a str,
    recurrent: &'a [SpeculativeRecurrenceBinding],
    folded_carry_output: Option<&'a str>,
    /// SSA value the target's `folded_carry_seed` output is bound to.
    folded_carry_seed_value: Option<String>,
    /// The declared token-embedding source, when a folded carry is declared.
    token_embedding: Option<&'a onnx_genai_metadata::TokenEmbeddingSource>,
    /// Proposer port -> SSA value, from the workflow's own invocation of it.
    proposer_bindings: BTreeMap<String, String>,
    /// Proposer port -> SSA value for its outputs, from the same invocation.
    proposer_outputs: BTreeMap<String, String>,
    /// Proposer port -> SSA value the state cell it borrows currently holds.
    ///
    /// Overrides `proposer_bindings` for exactly those ports: see
    /// [`borrowed_state_bindings`].
    borrowed_state: BTreeMap<String, String>,
}

#[derive(Debug)]
struct DFlashPlan<'a> {
    contract: &'a SpeculativeContract,
    version: &'a str,
    conditioning: &'a onnx_genai_metadata::DFlashConditioning,
    block: &'a onnx_genai_metadata::DFlashBlockLayout,
    outputs: &'a onnx_genai_metadata::DFlashOutputs,
    shared_weights: &'a onnx_genai_metadata::DFlashSharedWeights,
    accepted_prefix_state: &'a BTreeMap<String, DFlashStateCommit>,
    structure: &'a DFlashStructure,
    proposer_bindings: BTreeMap<String, String>,
    proposer_outputs: BTreeMap<String, String>,
    target_bindings: BTreeMap<String, String>,
    target_outputs: BTreeMap<String, String>,
    target_tokens_input: String,
}

impl<'a> DFlashPlan<'a> {
    fn resolve(contract: &'a SpeculativeContract, workflow: &WorkflowSpec) -> anyhow::Result<Self> {
        let SpeculativeProposalExecution::DflashFlatBlock {
            version,
            conditioning,
            block,
            outputs,
            shared_weights,
            accepted_prefix_state,
            structure,
            ..
        } = &contract.proposal_execution
        else {
            anyhow::bail!(
                "speculative.proposal_execution is not dflash_flat_block; candidate-tree, \
                 chained, and generic block contracts are independent proposal forms"
            );
        };
        anyhow::ensure!(
            matches!(
                (version.as_str(), structure.as_ref()),
                ("1", DFlashStructure::Base)
            ),
            "this runtime executes only DFlash flat-block version 1 with structure base; \
             version {version} / {structure:?} must be refused at admission"
        );
        let (proposer_bindings, proposer_outputs) =
            component_invocation(workflow, &contract.proposer).with_context(|| {
                format!(
                    "DFlash proposer '{}' is never invoked by the workflow, so its explicit \
                     bindings cannot be resolved",
                    contract.proposer
                )
            })?;
        let (target_bindings, target_outputs) = component_invocation(workflow, &contract.target)
            .with_context(|| {
                format!(
                    "DFlash target '{}' is never invoked by the workflow, so hidden and verifier \
                     outputs cannot be resolved",
                    contract.target
                )
            })?;
        let target_token_inputs = target_bindings
            .iter()
            .filter(|(_, value)| {
                matches!(
                    workflow.inputs.get(*value).map(|input| &input.role),
                    Some(onnx_genai_metadata::SemanticInputRole::Runtime { role, .. })
                        if *role == onnx_genai_metadata::RuntimeInputRole::PromptTokens
                )
            })
            .map(|(port, _)| port.clone())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            target_token_inputs.len() == 1,
            "DFlash target '{}' must bind exactly one declared runtime prompt_tokens input; \
             found {:?}. The verifier token port is semantic metadata, never inferred from a \
             port name.",
            contract.target,
            target_token_inputs
        );
        Ok(Self {
            contract,
            version,
            conditioning,
            block,
            outputs,
            shared_weights,
            accepted_prefix_state,
            structure,
            proposer_bindings,
            proposer_outputs,
            target_bindings,
            target_outputs,
            target_tokens_input: target_token_inputs
                .into_iter()
                .next()
                .expect("checked to contain exactly one target token input"),
        })
    }

    fn source_value_name(&self, source: &onnx_genai_metadata::SpeculativeValueRef) -> Option<&str> {
        if source.component == self.contract.proposer {
            self.proposer_outputs
                .get(&source.output)
                .map(String::as_str)
        } else if source.component == self.contract.target {
            self.target_outputs.get(&source.output).map(String::as_str)
        } else {
            None
        }
    }

    fn has_sampling_probabilities(&self) -> bool {
        if self.outputs.proposal_probabilities.is_some() {
            return true;
        }
        matches!(
            self.structure,
            DFlashStructure::SelectorConvolutionV1 { selector, .. }
                if selector.conditional_probabilities_output.is_some()
        )
    }

    fn state_for_component_input<'workflow>(
        &self,
        workflow: &'workflow WorkflowSpec,
        component: &str,
        input: &str,
    ) -> Option<&'workflow str> {
        workflow
            .serving
            .as_ref()?
            .state_service
            .groups
            .iter()
            .find_map(|(_, group)| {
                group
                    .ports
                    .get(component)?
                    .iter()
                    .find_map(|(cell, alias)| {
                        (alias.access == StatePortAccess::ReadWrite && alias.input == input)
                            .then_some(cell.as_str())
                    })
            })
    }

    fn initial_state(
        &self,
        workflow: &WorkflowSpec,
        values: &PipelineTensors,
    ) -> anyhow::Result<PipelineTensors> {
        let mut state = PipelineTensors::new();
        for cell in &self.contract.rollback_state {
            let source = workflow
                .serving
                .as_ref()
                .and_then(|serving| {
                    serving.state_service.groups.values().find_map(|group| {
                        group.ports.iter().find_map(|(component, aliases)| {
                            let alias = aliases.get(cell)?;
                            if alias.access != StatePortAccess::ReadWrite {
                                return None;
                            }
                            let binding = match component.as_str() {
                                component if component == self.contract.target => {
                                    self.target_bindings.get(&alias.input)
                                }
                                component if component == self.contract.proposer => {
                                    self.proposer_bindings.get(&alias.input)
                                }
                                _ => None,
                            }?;
                            Some(binding.as_str())
                        })
                    })
                })
                .with_context(|| {
                    format!(
                        "DFlash rollback participant '{cell}' has no read-write target or \
                         proposer input binding"
                    )
                })?;
            let value = values.get(source).with_context(|| {
                format!(
                    "DFlash rollback participant '{cell}' starts from unavailable workflow \
                     value '{source}'"
                )
            })?;
            state.insert(cell.clone(), clone_value(value)?);
        }
        Ok(state)
    }

    fn install_state_inputs(
        &self,
        workflow: &WorkflowSpec,
        state: &PipelineTensors,
        values: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        for (component, bindings) in [
            (self.contract.target.as_str(), &self.target_bindings),
            (self.contract.proposer.as_str(), &self.proposer_bindings),
        ] {
            for (port, value_name) in bindings {
                let Some(cell) = self.state_for_component_input(workflow, component, port) else {
                    continue;
                };
                let value = state.get(cell).with_context(|| {
                    format!(
                        "DFlash component '{component}' reads rollback state '{cell}', which \
                         has no transaction-local value"
                    )
                })?;
                values.insert(value_name.clone(), clone_value(value)?);
            }
        }
        Ok(())
    }
}

/// Proposer ports that are read-only views of another component's state.
///
/// `serving.state_service.groups[*].ports.<proposer>` declares an alias with
/// `access: read_only`: the port is not an input of the proposer's own, it *is*
/// the cell some other component owns. Binding it from the value the pass
/// started with hands a drafter the cache as it stood *before* the owner ran —
/// for a pass over a fresh context, an empty one — so every token it drafts is
/// conditioned on nothing and the target contradicts all of them. The borrow is
/// silent, too: the proposal is well formed, the tally is plausible, and the
/// only symptom is an acceptance rate of zero.
///
/// The owner's alias names the `output` port its current contents leave by, and
/// the workflow's own invocation binds that port to an SSA value. That value is
/// what the cell holds now, so that is what the borrowing port binds.
fn borrowed_state_bindings(
    workflow: &WorkflowSpec,
    proposer: &str,
    target: &str,
) -> anyhow::Result<BTreeMap<String, String>> {
    let Some(serving) = workflow.serving.as_ref() else {
        return Ok(BTreeMap::new());
    };
    let mut bound = BTreeMap::new();
    for group in serving.state_service.groups.values() {
        let Some(borrowed) = group.ports.get(proposer) else {
            continue;
        };
        for (cell, alias) in borrowed {
            if alias.access != StatePortAccess::ReadOnly {
                continue;
            }
            let port = &alias.input;
            // Only the cell's read-write owner publishes its contents. A
            // read-only alias may name a present output the artifact emits for
            // kernel-ABI reasons, and the schema is explicit that such a value
            // is not a state transition — binding a drafter to another
            // drafter's discarded output would be the same silent wrong answer
            // this function exists to remove.
            let mut owners = group.ports.iter().filter_map(|(component, aliases)| {
                if component == proposer {
                    return None;
                }
                let alias = aliases.get(cell)?;
                if alias.access != StatePortAccess::ReadWrite {
                    return None;
                }
                Some((component.as_str(), alias.output.as_ref()?.as_str()))
            });
            // The speculative target is the owner a proposer borrows from when
            // more than one component advances the cell; picking whichever
            // component sorted first would reintroduce a stale borrow by a
            // different route.
            let owner = owners
                .clone()
                .find(|(component, _)| *component == target)
                .or_else(|| owners.next());
            let Some((owner, output)) = owner else {
                anyhow::bail!(
                    "component '{proposer}' borrows state cell '{cell}' read-only, but no other \
                     component in its service group owns that cell read-write and publishes it; \
                     a borrowed cache with no owner would be read as the seed the pass began \
                     with, which is not what the package declared"
                );
            };
            let (_, outputs) = component_invocation(workflow, owner).with_context(|| {
                format!(
                    "state cell '{cell}' is owned by component '{owner}', which the workflow \
                     never invokes, so '{proposer}' has nothing to borrow"
                )
            })?;
            let value = outputs.get(output).with_context(|| {
                format!(
                    "component '{owner}' owns state cell '{cell}' through output port \
                     '{output}', which its invocation does not bind to a value"
                )
            })?;
            bound.insert(port.clone(), value.clone());
        }
    }
    Ok(bound)
}

impl WorkflowRuntime {
    /// The package's speculative compatibility contract, when it declares one.
    pub fn speculative_contract(&self) -> Option<&SpeculativeContract> {
        self.plan.speculative.as_ref()
    }

    pub fn candidate_tree_diagnostic(&self) -> Option<CandidateTreeDiagnostic> {
        let contract = self.plan.speculative.as_ref()?;
        let plan = self.plan.execution_admission.candidate_tree_plan()?;
        Some(CandidateTreeDiagnostic {
            version: plan.version.clone(),
            proposer: plan.proposer.clone(),
            target: plan.target.clone(),
            topology: match &plan.topology {
                CandidateTreeTopology::ParentIndices { .. } => "parent_indices",
                CandidateTreeTopology::AncestorMask { .. } => "ancestor_mask",
            },
            max_proposal_width: plan.max_proposal_width,
            distribution_preserving: contract.distribution_preserving,
            proposal_probabilities: contract.verification.probabilities.is_some(),
            rollback_participants: plan.rollback_state.iter().cloned().collect(),
            shared_batching_supported: false,
        })
    }

    pub(crate) fn is_candidate_tree(&self) -> bool {
        self.candidate_tree_diagnostic().is_some()
    }

    pub(crate) fn reject_candidate_tree_raw_execution(
        &self,
        operation: &str,
    ) -> anyhow::Result<()> {
        self.require_execution_admitted()?;
        if self.is_candidate_tree() {
            return Err(
                crate::engine::PackageExecutionError::CandidateTreeRawWorkflowApi {
                    operation: operation.to_string(),
                }
                .into(),
            );
        }
        Ok(())
    }

    pub fn take_candidate_tree_block_traces(&mut self) -> Vec<CandidateTreeBlockTrace> {
        std::mem::take(&mut *self.worker.last_candidate_tree_block_traces.borrow_mut())
    }

    pub(crate) fn last_candidate_tree_block_trace_count(&self) -> usize {
        self.worker.last_candidate_tree_block_traces.borrow().len()
    }

    /// Execute canonical candidate-tree generation through one transaction
    /// authority. Component outputs remain provisional until accepted-path
    /// recomputation and the complete state/effect/output write set commit.
    pub(crate) fn run_candidate_tree_generation(
        &self,
        options: &GenerateOptions,
        request: super::PipelineGenerateRequest,
        tokenizer: Option<&onnx_genai_ort::Tokenizer>,
        staged_output_observer: Option<&mut super::ToolCallStagedOutputObserver>,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.require_execution_admitted()?;
        anyhow::ensure!(
            matches!(
                self.plan.decode_backend,
                crate::EngineDecodeBackend::Auto | crate::EngineDecodeBackend::Ort
            ),
            "candidate-tree execution is implemented for the ORT workflow backend; selected \
             backend is {:?}",
            self.plan.decode_backend
        );
        anyhow::ensure!(
            options.max_new_tokens > 0,
            "candidate-tree generation requires max_new_tokens to be positive"
        );
        let contract = self
            .plan
            .speculative
            .as_ref()
            .context("candidate-tree generation was selected without a speculative contract")?;
        let plan = self
            .plan
            .execution_admission
            .candidate_tree_plan()
            .context(
                "candidate-tree generation has no construction-time SSA/dataflow execution plan",
            )?
            .clone();
        let sampling = !options.selects_greedily();
        if sampling {
            contract
                .admit_sampling()
                .map_err(anyhow::Error::msg)
                .context("candidate-tree sampling admission failed before component execution")?;
            anyhow::ensure!(
                options.temperature == 1.0
                    && options.top_p >= 1.0
                    && options.top_k == 0
                    && options.min_p == 0.0
                    && options.top_a == 0.0
                    && options.typical_p >= 1.0
                    && options.repetition_penalty <= 1.0
                    && options.frequency_penalty == 0.0
                    && options.presence_penalty == 0.0
                    && options.dry.is_none()
                    && options.mirostat.is_none()
                    && options.xtc.is_none()
                    && options.constraint.is_none(),
                "candidate-tree sampling currently preserves the exact declared target \
                 distribution only; temperature/logit processors/grammar require matching \
                 proposer-side transforms and are refused before transaction admission"
            );
        }
        let control = request.generation_control.clone().unwrap_or_default();
        if plan.execution_mode == CandidateTreeExecutionMode::DrainAtSeam
            && let Some(limit) = options.max_context
            && match &request.request.prompt {
                crate::config::GeneratePrompt::TokenIds(tokens) => tokens.len() >= limit,
                crate::config::GeneratePrompt::TokenRows(rows) => {
                    rows.first().is_some_and(|tokens| tokens.len() >= limit)
                }
                crate::config::GeneratePrompt::Text(_) => false,
            }
        {
            return Ok(GenerateResult {
                text: String::new(),
                token_ids: Vec::new(),
                finish_reason: FinishReason::Length,
                tool_calls: Vec::new(),
                prefix_cache_hit_len: 0,
                logprobs: None,
                budget_cap: None,
            });
        }
        let mut execution = super::WorkflowExecutionPlan::new_candidate_tree_driver(self, request)?;
        let mut host = CandidateTreeWorkflowHost {
            runtime: self,
            plan,
            contract,
            options,
            tokenizer,
            control: control.clone(),
            rng: StdRng::seed_from_u64(options.seed.unwrap_or(0)),
            turn_identity: None,
            candidate_state: None,
            pending: None,
            generated: Vec::with_capacity(options.max_new_tokens),
            traces: Vec::new(),
            blocks: 0,
            finish_reason: FinishReason::MaxTokens,
            text: String::new(),
            token_text: Vec::new(),
            commit_started: false,
            loop_stop: None,
            staged_output_observer,
        };
        {
            let mut hosted: Option<&mut dyn super::workflow::WorkflowNodeHost> = Some(&mut host);
            execution.execute_retained_with_host(&mut hosted)?;
        }
        if host.finish_reason == FinishReason::MaxTokens
            && self.last_generation_ended_by_predicate()
        {
            host.finish_reason = FinishReason::EosToken;
        }
        for _ in 0..host.blocks {
            self.record_contract_execution(
                onnx_genai_metadata::decoder_workflow::SPECULATIVE_BLOCK_CONTRACT,
            );
        }
        *self.worker.last_candidate_tree_block_traces.borrow_mut() = host.traces;
        let tool_calls = host
            .staged_output_observer
            .as_deref()
            .map(super::ToolCallStagedOutputObserver::committed_calls)
            .unwrap_or_default();

        match control.observe_after_commit(super::GenerationBoundary::BeforeOutputPublication) {
            Ok(true) if callback.is_some() => {
                return Err(anyhow::Error::new(CandidateTreeOutputDeliveryError {
                    committed_tokens: host.generated.len(),
                    message: "delivery was cancelled before the first callback".to_string(),
                }));
            }
            Err(error) => {
                return Err(anyhow::Error::new(CandidateTreeOutputDeliveryError {
                    committed_tokens: host.generated.len(),
                    message: format!("post-commit output checkpoint failed: {error:#}"),
                }));
            }
            Ok(_) => {}
        }
        if let Some(callback) = callback.as_mut() {
            for (index, token) in host.generated.iter().copied().enumerate() {
                if let Err(error) = callback(crate::config::GenerateToken {
                    token_id: token,
                    text: host.token_text[index].clone(),
                    finish_reason: (index + 1 == host.generated.len())
                        .then(|| host.finish_reason.clone()),
                }) {
                    return Err(anyhow::Error::new(CandidateTreeOutputDeliveryError {
                        committed_tokens: host.generated.len(),
                        message: format!("callback failed at token {index}: {error:#}"),
                    }));
                }
            }
        }
        Ok(GenerateResult {
            text: host.text,
            token_ids: host.generated,
            finish_reason: host.finish_reason,
            tool_calls,
            prefix_cache_hit_len: 0,
            logprobs: None,
            budget_cap: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn invoke_candidate_tree_component(
        &self,
        plan: &CandidateTreeExecutionPlan,
        component: &str,
        context: &Value,
        tree: Option<&RuntimeCandidateTree>,
        accepted_tokens: &Value,
        state: &PipelineTensors,
        values: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        let (bindings, outputs, context_port) = if component == plan.proposer {
            (
                &plan.proposer_bindings,
                &plan.proposer_outputs,
                plan.proposer_context_input.as_str(),
            )
        } else if component == plan.target {
            (
                &plan.target_bindings,
                &plan.target_outputs,
                plan.target_context_input.as_str(),
            )
        } else {
            anyhow::bail!(
                "candidate-tree component '{component}' is neither proposer '{}' nor target '{}'",
                plan.proposer,
                plan.target
            );
        };
        let mut owned = Vec::with_capacity(bindings.len());
        for (port, value_name) in bindings {
            let value = if port == context_port {
                crate::decode::clone_value(context)?
            } else if component == plan.target && port.as_str() == plan.target_candidate_input {
                crate::decode::clone_value(values.get(&plan.target_candidate_value).with_context(
                    || {
                        format!(
                            "candidate-tree target input '{port}' lost its admitted proposer SSA \
                             value '{}'",
                            plan.target_candidate_value
                        )
                    },
                )?)?
            } else if component == plan.target && port.as_str() == plan.target_topology_input {
                match &plan.target_topology_value {
                    CandidateTreeTopologyInput::ProposerValue { value } => {
                        crate::decode::clone_value(values.get(value).with_context(|| {
                            format!(
                                "candidate-tree target topology input '{port}' lost its admitted \
                                 proposer SSA value '{value}'"
                            )
                        })?)?
                    }
                    CandidateTreeTopologyInput::DerivedFromParentIndices {
                        placeholder, ..
                    } => {
                        debug_assert_eq!(value_name, placeholder);
                        crate::decode::clone_value(
                            &tree
                                .context(
                                    "candidate-tree target invocation omitted derived ancestor mask",
                                )?
                                .ancestor_mask,
                        )?
                    }
                }
            } else if component == plan.target && port.as_str() == plan.target_position_input {
                debug_assert_eq!(value_name, &plan.target_position_placeholder);
                crate::decode::clone_value(
                    &tree
                        .context("candidate-tree target invocation omitted position ids")?
                        .position_ids,
                )?
            } else if component == plan.target && port.as_str() == plan.target_accepted_input {
                debug_assert_eq!(value_name, &plan.target_accepted_placeholder);
                crate::decode::clone_value(accepted_tokens)?
            } else if let Some(cell) = plan.state_for_input(&self.plan.workflow, component, port) {
                crate::decode::clone_value(state.get(cell).with_context(|| {
                    format!(
                        "candidate-tree component '{component}' reads unavailable transaction \
                         state '{cell}'"
                    )
                })?)?
            } else {
                crate::decode::clone_value(values.get(value_name).with_context(|| {
                    format!(
                        "candidate-tree component '{component}' input '{port}' references \
                         unavailable workflow value '{value_name}'"
                    )
                })?)?
            };
            owned.push((port.clone(), value));
        }
        let refs = owned
            .iter()
            .map(|(port, value)| (port.as_str(), value))
            .collect::<Vec<_>>();
        let produced = self.invoke_component_values(
            component,
            &refs,
            outputs,
            &std::collections::HashMap::new(),
            1,
        )?;
        for (port, value) in produced {
            let binding = outputs.get(&port).with_context(|| {
                format!("candidate-tree component '{component}' produced unbound output '{port}'")
            })?;
            values.insert(binding.clone(), value);
        }
        Ok(())
    }

    fn decode_runtime_candidate_tree(
        &self,
        plan: &CandidateTreeExecutionPlan,
        values: &PipelineTensors,
    ) -> anyhow::Result<RuntimeCandidateTree> {
        let tokens_binding = &plan.target_candidate_value;
        let tokens_value = values.get(tokens_binding).with_context(|| {
            format!(
                "candidate-tree proposer did not produce candidate token value \
                 '{tokens_binding}'"
            )
        })?;
        anyhow::ensure!(
            tokens_value.dtype() == DataType::Int64,
            "candidate-tree tokens have {:?}; expected int64",
            tokens_value.dtype()
        );
        let shape = tokens_value.shape();
        anyhow::ensure!(
            shape.len() == 2 && shape[0] == 1 && shape[1] > 0,
            "candidate-tree tokens have shape {shape:?}; expected isolated [1, candidates] with \
             at least one candidate"
        );
        let candidate_count =
            usize::try_from(shape[1]).context("candidate-tree candidate extent is negative")?;
        anyhow::ensure!(
            candidate_count <= plan.max_proposal_width,
            "candidate-tree proposer emitted {candidate_count} candidates, exceeding declared \
             max_proposal_width {}",
            plan.max_proposal_width
        );
        let tokens = tokens_value
            .to_vec_i64()?
            .into_iter()
            .enumerate()
            .map(|(index, token)| {
                u32::try_from(token).with_context(|| {
                    format!(
                        "candidate-tree token at flattened node {index} is {token}, outside the \
                         non-negative u32 token domain"
                    )
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let topology_binding = match &plan.target_topology_value {
            CandidateTreeTopologyInput::ProposerValue { value } => value,
            CandidateTreeTopologyInput::DerivedFromParentIndices { topology_value, .. } => {
                topology_value
            }
        };
        let topology = values.get(topology_binding).with_context(|| {
            format!("candidate-tree proposer did not produce topology value '{topology_binding}'")
        })?;
        let tree = match plan.topology {
            CandidateTreeTopology::ParentIndices { .. } => {
                anyhow::ensure!(
                    topology.dtype() == DataType::Int64,
                    "candidate-tree parent topology has {:?}; expected int64",
                    topology.dtype()
                );
                anyhow::ensure!(
                    topology.shape() == [1, shape[1]],
                    "candidate-tree parent topology has shape {:?}; expected [1, {candidate_count}]",
                    topology.shape()
                );
                let parents = topology
                    .to_vec_i64()?
                    .into_iter()
                    .enumerate()
                    .map(|(node, parent)| {
                        if parent == -1 {
                            Ok(None)
                        } else {
                            usize::try_from(parent).map(Some).with_context(|| {
                                format!(
                                    "candidate-tree parent at node {node} is {parent}; version 1 \
                                     uses -1 for roots and non-negative preceding indices"
                                )
                            })
                        }
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                SpecTree::from_parent_indices(tokens, parents)?
            }
            CandidateTreeTopology::AncestorMask { .. } => {
                anyhow::ensure!(
                    topology.dtype() == DataType::Bool,
                    "candidate-tree ancestor topology has {:?}; expected bool",
                    topology.dtype()
                );
                anyhow::ensure!(
                    topology.shape() == [1, shape[1], shape[1]],
                    "candidate-tree ancestor topology has shape {:?}; expected \
                     [1, {candidate_count}, {candidate_count}]",
                    topology.shape()
                );
                let flat = topology.to_vec_bool()?;
                let mask = flat
                    .chunks_exact(candidate_count)
                    .map(|row| row.to_vec())
                    .collect::<Vec<_>>();
                SpecTree::from_ancestor_mask(tokens, &mask)?
            }
        };
        let roots = tree.roots();
        anyhow::ensure!(
            !roots.is_empty(),
            "candidate-tree proposer emitted no root attached to the committed anchor"
        );
        for parent in std::iter::once(None).chain((0..tree.len()).map(Some)) {
            let children = match parent {
                None => roots.clone(),
                Some(node) => tree.children(node),
            };
            let mut sibling_tokens = std::collections::HashSet::new();
            for child in &children {
                anyhow::ensure!(
                    sibling_tokens.insert(tree.nodes()[*child].token),
                    "candidate-tree siblings under parent {parent:?} repeat token {}; target \
                     verification could not identify one accepted node",
                    tree.nodes()[*child].token
                );
            }
            anyhow::ensure!(
                children.len() <= plan.max_proposal_width,
                "candidate-tree parent {parent:?} has {} children, exceeding declared rollback \
                 width {}",
                children.len(),
                plan.max_proposal_width
            );
        }
        let depth = tree
            .nodes()
            .iter()
            .map(|node| node.depth + 1)
            .max()
            .unwrap_or_default();
        anyhow::ensure!(
            depth <= plan.max_proposal_width,
            "candidate-tree depth {depth} exceeds declared rollback width {}",
            plan.max_proposal_width
        );
        let mask = crate::speculative::ancestor_attention_mask(&tree);
        let mask_bytes = mask
            .iter()
            .flat_map(|row| row.iter().map(|value| u8::from(*value)))
            .collect::<Vec<_>>();
        let ancestor_mask = Value::from_raw_bytes(
            mask_bytes,
            &[
                1,
                i64::try_from(tree.len()).context("candidate-tree width exceeds i64")?,
                i64::try_from(tree.len()).context("candidate-tree width exceeds i64")?,
            ],
            DataType::Bool,
        )?;
        let positions = crate::speculative::relative_position_ids(&tree)
            .into_iter()
            .map(i64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .context("candidate-tree depth exceeds i64")?;
        let position_ids = Value::from_slice_i64(
            &positions,
            &[
                1,
                i64::try_from(tree.len()).context("candidate-tree width exceeds i64")?,
            ],
        )?;
        Ok(RuntimeCandidateTree {
            tree,
            ancestor_mask,
            position_ids,
        })
    }

    fn target_tree_rows(
        &self,
        plan: &CandidateTreeExecutionPlan,
        tree: &SpecTree,
        values: &PipelineTensors,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        probability_or_logits_rows(
            values.get(&plan.target_logits_value).with_context(|| {
                format!(
                    "candidate-tree target did not produce declared verifier output '{}'",
                    plan.target_logits_value
                )
            })?,
            tree.len() + 1,
            "target logits",
            false,
        )
    }

    fn verify_candidate_tree_greedy(
        &self,
        plan: &CandidateTreeExecutionPlan,
        tree: &SpecTree,
        values: &PipelineTensors,
    ) -> anyhow::Result<crate::speculative::AcceptOutcome> {
        let rows = self.target_tree_rows(plan, tree, values)?;
        for node in tree.nodes() {
            anyhow::ensure!(
                (node.token as usize) < rows[0].len(),
                "candidate-tree token {} is outside target vocabulary {}",
                node.token,
                rows[0].len()
            );
        }
        tree.accept(AcceptanceRule::Greedy, &rows[0], &rows[1..])
    }

    fn verify_candidate_tree_sampling(
        &self,
        plan: &CandidateTreeExecutionPlan,
        tree: &SpecTree,
        values: &PipelineTensors,
        rng: &mut StdRng,
    ) -> anyhow::Result<crate::speculative::AcceptOutcome> {
        let proposal_value = values
            .get(
                plan.proposal_probabilities_value
                    .as_deref()
                    .context("candidate-tree proposal probabilities were not admitted")?,
            )
            .context("candidate-tree proposer omitted declared proposal probabilities")?;
        let target_value = values
            .get(
                plan.target_probabilities_value
                    .as_deref()
                    .context("candidate-tree target probabilities were not admitted")?,
            )
            .context("candidate-tree target omitted declared target probabilities")?;
        let proposal = probability_or_logits_rows(
            proposal_value,
            tree.len() + 1,
            "proposal probabilities",
            true,
        )?;
        let target =
            probability_or_logits_rows(target_value, tree.len() + 1, "target probabilities", true)?;
        let logits = self.target_tree_rows(plan, tree, values)?;
        anyhow::ensure!(
            proposal[0].len() == target[0].len(),
            "candidate-tree proposal vocabulary {} does not match target vocabulary {}",
            proposal[0].len(),
            target[0].len()
        );
        anyhow::ensure!(
            logits[0].len() == target[0].len(),
            "candidate-tree target logits vocabulary {} does not match declared target \
             probability vocabulary {}",
            logits[0].len(),
            target[0].len()
        );
        for (row, (logits, probabilities)) in logits.iter().zip(&target).enumerate() {
            anyhow::ensure!(
                crate::speculative::argmax(logits) == crate::speculative::argmax(probabilities),
                "candidate-tree target logits and probability bindings disagree on row {row}; \
                 bind outputs from the same verifier distribution"
            );
        }
        validate_candidate_probability_support(tree, &proposal)?;
        let mut proposed_path = Vec::new();
        let mut previous = None;
        loop {
            let frontier = previous.map_or_else(|| tree.roots(), |node| tree.children(node));
            if frontier.is_empty() {
                break;
            }
            let row = previous.map_or(0, |node| node + 1);
            let token = sample_declared_distribution(&proposal[row], rng.random())?;
            let node = frontier
                .into_iter()
                .find(|node| tree.nodes()[*node].token as usize == token)
                .context(
                    "candidate-tree proposal distribution sampled outside its declared frontier",
                )?;
            proposed_path.push(node);
            previous = Some(node);
        }
        let randomness = (0..=proposed_path.len())
            .map(|_| SamplingRandomness {
                acceptance: rng.random(),
                correction: rng.random(),
            })
            .collect();
        let verification = verify_tree_sampling(
            tree,
            0,
            &TreeSamplingInputs {
                proposal_probabilities: proposal,
                target_probabilities: target,
                proposed_path,
                randomness,
            },
        )?;
        Ok(verification.outcome)
    }

    pub fn dflash_diagnostic(&self) -> Option<DFlashDiagnostic> {
        let contract = self.plan.speculative.as_ref()?;
        let SpeculativeProposalExecution::DflashFlatBlock {
            version,
            conditioning,
            outputs,
            draft_private_state,
            structure,
            ..
        } = &contract.proposal_execution
        else {
            return None;
        };
        Some(DFlashDiagnostic {
            version: version.clone(),
            proposer: contract.proposer.clone(),
            target: contract.target.clone(),
            target_hidden_sources: conditioning
                .sources
                .iter()
                .map(|source| format!("{}::{}", source.component, source.output))
                .collect(),
            max_proposal_width: contract.max_proposal_width,
            proposal_probabilities: outputs.proposal_probabilities.is_some()
                || matches!(
                    structure.as_ref(),
                    DFlashStructure::SelectorConvolutionV1 { selector, .. }
                        if selector.conditional_probabilities_output.is_some()
                ),
            rollback_participants: contract.rollback_state.iter().cloned().collect(),
            draft_private_state: draft_private_state.iter().cloned().collect(),
            structure: match structure.as_ref() {
                DFlashStructure::Base => "base",
                DFlashStructure::SelectorConvolutionV1 { .. } => "selector_convolution_v1",
            },
            shared_batching_supported: false,
        })
    }

    /// Run the v1 DFlash flat-block algorithm as one runtime-owned generation
    /// transaction.
    ///
    /// The caller supplies a normal workflow request, not a target/proposer
    /// choreography.  Every tensor crossing the component boundary is found
    /// through the declaration: the target's semantic prompt-token binding,
    /// hidden-output provenance, state-service aliases, and immutable shared
    /// weights.  This is deliberately separate from chained, MTP, prompt
    /// lookup, and tree speculation; those proposal forms have different
    /// contracts and cannot be treated as a DFlash block.
    pub(crate) fn run_dflash_generation(
        &self,
        options: &GenerateOptions,
        request: super::PipelineGenerateRequest,
        tokenizer: Option<&onnx_genai_ort::Tokenizer>,
        _staged_output_observer: Option<&mut super::ToolCallStagedOutputObserver>,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        self.require_execution_admitted()?;
        anyhow::ensure!(
            options.max_new_tokens > 0,
            "DFlash generation requires max_new_tokens to be positive"
        );
        let contract =
            self.plan.speculative.as_ref().context(
                "DFlash generation was selected but the package declares no speculation",
            )?;
        let plan = DFlashPlan::resolve(contract, &self.plan.workflow)?;
        let sampling = !options.selects_greedily();
        if sampling {
            anyhow::ensure!(
                plan.has_sampling_probabilities(),
                "DFlash sampling requires a declared proposal probability distribution; refuse \
                 before proposer execution rather than mutating draft state with an \
                 unverifiable proposal"
            );
            anyhow::ensure!(
                options.top_p >= 1.0
                    && options.top_k == 0
                    && options.min_p == 0.0
                    && options.top_a == 0.0
                    && options.typical_p >= 1.0
                    && options.repetition_penalty <= 1.0
                    && options.frequency_penalty == 0.0
                    && options.presence_penalty == 0.0
                    && options.dry.is_none()
                    && options.mirostat.is_none()
                    && options.xtc.is_none()
                    && options.constraint.is_none(),
                "DFlash sampling currently supports the exact unwarped target distribution \
                 only; processors require matching proposal-side transforms and are refused \
                 before transaction admission"
            );
        }
        let control = request.generation_control.clone().unwrap_or_default();

        // Bind once, before any component executes.  `WorkflowExecutionPlan`
        // owns normal request/default/shape admission; taking the values before
        // its generic execute method is what keeps the DFlash component passes
        // inside this driver's transaction rather than committing an unrelated
        // workflow pass first.
        let (mut values, session_id) =
            super::WorkflowExecutionPlan::new_dflash_driver(self, request)?.into_bound_values();
        let mut state = plan.initial_state(&self.plan.workflow, &values)?;
        self.restore_dflash_session_state(&plan, session_id.as_deref(), &mut state)?;
        plan.install_state_inputs(&self.plan.workflow, &state, &mut values)?;

        let initial_tokens = values
            .get(
                plan.target_bindings
                    .get(&plan.target_tokens_input)
                    .expect("DFlash plan resolved the target token binding"),
            )
            .with_context(|| {
                format!(
                    "DFlash target '{}' has no bound prompt token input '{}'",
                    contract.target, plan.target_tokens_input
                )
            })?
            .to_vec_i64()?;
        anyhow::ensure!(
            !initial_tokens.is_empty(),
            "DFlash generation requires at least one prompt token"
        );
        let initial_shape = values
            .get(
                plan.target_bindings
                    .get(&plan.target_tokens_input)
                    .expect("DFlash plan resolved the target token binding"),
            )
            .expect("target token value was checked above")
            .shape()
            .to_vec();
        anyhow::ensure!(
            initial_shape.len() == 2 && initial_shape[0] == 1,
            "DFlash executes one request row in isolation; target prompt tokens have shape \
             {initial_shape:?}, expected [1, sequence]"
        );
        if options
            .max_context
            .is_some_and(|limit| initial_tokens.len() >= limit)
        {
            return Ok(GenerateResult {
                text: String::new(),
                token_ids: Vec::new(),
                finish_reason: FinishReason::Length,
                tool_calls: Vec::new(),
                prefix_cache_hit_len: 0,
                logprobs: None,
                budget_cap: None,
            });
        }
        let turn = super::TurnTransaction::admit_runtime_participant(
            self.worker.next_turn_transaction_id(),
        );
        let mut context_start = i64::try_from(initial_tokens.len())
            .context("DFlash prompt length exceeds the absolute position type")?;
        let mut generated = Vec::with_capacity(options.max_new_tokens);
        let mut random = StdRng::seed_from_u64(options.seed.unwrap_or(0));
        let mut finish_reason = FinishReason::MaxTokens;
        let mut staged_block_traces = Vec::new();
        let mut staged_contract_executions = 0_u64;

        // The initial target invocation produces the hidden state that
        // conditions the first flat proposal.  It is real component execution,
        // not a caller-injected replacement for hidden features or logits.
        self.invoke_dflash_target(&plan, &mut values, &state, None)?;

        while generated.len() < options.max_new_tokens {
            if options
                .max_context
                .is_some_and(|limit| initial_tokens.len().saturating_add(generated.len()) >= limit)
            {
                finish_reason = FinishReason::Length;
                break;
            }
            let anchor = self.dflash_anchor_token(&plan, &values, options, &mut random)?;
            let remaining = options.max_new_tokens - generated.len();
            let remaining_context = options
                .max_context
                .map(|limit| limit.saturating_sub(initial_tokens.len() + generated.len()))
                .unwrap_or(remaining);
            let width = remaining
                .min(contract.max_proposal_width)
                .min(remaining_context);
            if width == 0 {
                finish_reason = FinishReason::Length;
                break;
            }
            let mode = if sampling {
                DFlashProposalMode::Sampling {
                    seed: random.random(),
                }
            } else {
                DFlashProposalMode::Greedy
            };
            let proposal_options = DFlashProposalOptions {
                anchor_token: anchor,
                width,
                context_start_position: context_start,
                mode,
                eos_token_ids: options
                    .eos_token_ids
                    .iter()
                    .map(|id| i64::from(*id))
                    .collect(),
            };

            // Snapshot every target, draft-private, recurrent, and token
            // context participant *before* the proposer or verifier can
            // advance one.  All error exits below resolve this same
            // transaction back to that complete baseline.
            let transaction = self.observe_dflash_block_checkpoint(
                &control,
                &turn,
                super::GenerationBoundary::BeforeProposer,
                self.begin_dflash_state_transaction(&state)?,
                &mut state,
            )?;
            let proposal = match self.propose_dflash(&values, proposal_options) {
                Ok(proposal) => proposal,
                Err(error) => {
                    let _ = self.abort_dflash_state_transaction(
                        transaction,
                        &mut state,
                        super::TurnAbortReason::ExecutionFailure,
                    );
                    let _ = turn.abort(super::TurnAbortReason::ExecutionFailure);
                    return Err(error.context("DFlash proposer execution failed before commit"));
                }
            };
            let transaction = self.observe_dflash_block_checkpoint(
                &control,
                &turn,
                super::GenerationBoundary::AfterProposer,
                transaction,
                &mut state,
            )?;
            let mut block = Vec::with_capacity(proposal.tokens.len() + 1);
            block.push(anchor);
            block.extend_from_slice(&proposal.tokens);
            if let Err(error) = self.invoke_dflash_target(&plan, &mut values, &state, Some(&block))
            {
                let _ = self.abort_dflash_state_transaction(
                    transaction,
                    &mut state,
                    super::TurnAbortReason::ExecutionFailure,
                );
                let _ = turn.abort(super::TurnAbortReason::ExecutionFailure);
                return Err(error.context("DFlash verifier execution failed before commit"));
            }
            let transaction = self.observe_dflash_block_checkpoint(
                &control,
                &turn,
                super::GenerationBoundary::AfterVerifier,
                transaction,
                &mut state,
            )?;
            // `invoke_dflash_target` updated the driver's SSA map in place.
            // Borrowing it through a separate binding makes the commit's
            // target-output provenance explicit and avoids any caller-provided
            // verifier logits or acceptance decision.
            let verification = if sampling {
                DFlashVerificationMode::Sampling {
                    temperature: options.temperature,
                }
            } else {
                DFlashVerificationMode::Greedy
            };
            let acceptance = match self.verify_dflash(&values, &proposal, verification) {
                Ok(acceptance) => acceptance,
                Err(error) => {
                    let _ = self.abort_dflash_state_transaction(
                        transaction,
                        &mut state,
                        super::TurnAbortReason::ExecutionFailure,
                    );
                    let _ = turn.abort(super::TurnAbortReason::ExecutionFailure);
                    return Err(error.context("DFlash acceptance failed before commit"));
                }
            };
            let mut committed_path = Vec::with_capacity(acceptance.committed.len() + 1);
            committed_path.push(anchor);
            committed_path.extend_from_slice(&acceptance.committed);
            committed_path.truncate(remaining);
            if let Some(eos) = committed_path
                .iter()
                .position(|token| options.terminates(u32::try_from(*token).unwrap_or(u32::MAX)))
            {
                committed_path.truncate(eos + 1);
            }
            let committed_drafts = acceptance
                .accepted
                .min(committed_path.len().saturating_sub(1));
            let committed_acceptance = DFlashAcceptance {
                accepted: committed_drafts,
                rejected_at: acceptance.rejected_at,
                committed: committed_path[1..].to_vec(),
            };

            // The verifier executed the entire candidate suffix. Re-run the
            // target from the admitted baseline on only the path that will
            // commit, including the anchor and correction/bonus. This makes
            // the next proposal consume hidden/state produced by committed
            // tokens, never by a rejected candidate row.
            if let Err(error) = self.invoke_dflash_target(
                &plan,
                &mut values,
                &transaction.baseline,
                Some(&committed_path),
            ) {
                let _ = self.abort_dflash_state_transaction(
                    transaction,
                    &mut state,
                    super::TurnAbortReason::ExecutionFailure,
                );
                let _ = turn.abort(super::TurnAbortReason::ExecutionFailure);
                return Err(error
                    .context("DFlash accepted-path target reconditioning failed before commit"));
            }
            let transaction = self.observe_dflash_block_checkpoint(
                &control,
                &turn,
                super::GenerationBoundary::BeforeAcceptedPrefixCommit,
                transaction,
                &mut state,
            )?;
            if let Err(error) = self.commit_dflash_state_transaction(
                transaction,
                &mut state,
                &proposal,
                &values,
                &committed_acceptance,
                committed_path.len(),
            ) {
                let _ = turn.abort(super::TurnAbortReason::CommitFailure);
                return Err(error.context("DFlash accepted-prefix state commit failed"));
            }
            staged_block_traces.push(DFlashBlockTrace {
                conditioning: proposal.conditioning_trace.clone(),
                proposer_candidates: proposal.tokens.clone(),
                accepted: committed_acceptance.accepted,
                committed_tokens: committed_path.clone(),
            });

            for (port, value) in &proposal.proposer_outputs {
                if let Some(value_name) = plan.proposer_outputs.get(port) {
                    values.insert(value_name.clone(), clone_value(value)?);
                }
            }
            plan.install_state_inputs(&self.plan.workflow, &state, &mut values)?;
            context_start = context_start
                .checked_add(i64::try_from(committed_path.len()).context(
                    "DFlash committed path length does not fit the absolute position type",
                )?)
                .context("DFlash absolute position overflow")?;
            for token in committed_path {
                let token = u32::try_from(token)
                    .context("DFlash verifier emitted a token outside the u32 token domain")?;
                generated.push(token);
                if options.terminates(token) {
                    finish_reason = FinishReason::EosToken;
                    break;
                }
                if generated.len() == options.max_new_tokens {
                    break;
                }
            }
            staged_contract_executions = staged_contract_executions.saturating_add(1);
            if finish_reason == FinishReason::EosToken {
                break;
            }
        }

        let text = tokenizer
            .map(|tokenizer| tokenizer.decode(&generated))
            .transpose()?
            .unwrap_or_default();
        match control.begin_commit() {
            Ok(true) => {}
            Ok(false) => {
                return Err(anyhow::Error::new(DFlashGenerationCancelled {
                    boundary: super::GenerationBoundary::BeforeSemanticCommit,
                    outcome: turn.abort(super::TurnAbortReason::Cancellation),
                }));
            }
            Err(error) => {
                let _ = turn.abort(super::TurnAbortReason::ExecutionFailure);
                return Err(
                    error.context("DFlash generation checkpoint failed before semantic commit")
                );
            }
        }
        if let Err(error) = self.commit_dflash_session_state(&plan, session_id.as_deref(), &state) {
            control.abort_commit();
            let _ = turn.abort(super::TurnAbortReason::CommitFailure);
            return Err(error.context("DFlash semantic state commit failed"));
        }
        for _ in 0..staged_contract_executions {
            self.record_contract_execution(
                onnx_genai_metadata::decoder_workflow::SPECULATIVE_BLOCK_CONTRACT,
            );
        }
        *self.worker.last_dflash_block_traces.borrow_mut() = staged_block_traces;
        let _ = turn.committed();
        control.finish_commit();
        if control
            .observe_after_commit(super::GenerationBoundary::BeforeOutputPublication)
            .context("DFlash output checkpoint failed after semantic commit")?
            && callback.is_some()
        {
            anyhow::bail!(
                "DFlash output delivery was cancelled after semantic commit; committed state \
                 remains durable and output will not be replayed automatically"
            );
        }
        // The entire generated token stream is staged in `generated` until all
        // model execution and accepted-prefix commits have succeeded.  Callback
        // failure is therefore delivery-only: state is already committed and is
        // never semantically rolled back by an external receiver.
        if let Some(callback) = callback.as_mut() {
            for (index, token) in generated.iter().copied().enumerate() {
                callback(crate::config::GenerateToken {
                    token_id: token,
                    text: tokenizer
                        .map(|tokenizer| tokenizer.decode(&[token]))
                        .transpose()?
                        .unwrap_or_default(),
                    finish_reason: (index + 1 == generated.len()).then(|| finish_reason.clone()),
                })
                .with_context(|| {
                    format!(
                        "DFlash output callback failed after semantic commit at token {index}; \
                         committed state remains durable and this partial delivery will not be \
                         replayed automatically"
                    )
                })?;
            }
        }
        Ok(GenerateResult {
            text,
            token_ids: generated,
            finish_reason,
            tool_calls: Vec::new(),
            prefix_cache_hit_len: 0,
            logprobs: None,
            budget_cap: None,
        })
    }

    fn observe_dflash_block_checkpoint(
        &self,
        control: &super::GenerationControl,
        turn: &super::TurnTransaction,
        boundary: super::GenerationBoundary,
        transaction: DFlashStateTransaction,
        state: &mut PipelineTensors,
    ) -> anyhow::Result<DFlashStateTransaction> {
        match control.observe(boundary) {
            Ok(false) => Ok(transaction),
            Ok(true) => {
                let _ = self.abort_dflash_state_transaction(
                    transaction,
                    state,
                    super::TurnAbortReason::Cancellation,
                );
                Err(anyhow::Error::new(DFlashGenerationCancelled {
                    boundary,
                    outcome: turn.abort(super::TurnAbortReason::Cancellation),
                }))
            }
            Err(error) => {
                let _ = self.abort_dflash_state_transaction(
                    transaction,
                    state,
                    super::TurnAbortReason::ExecutionFailure,
                );
                let _ = turn.abort(super::TurnAbortReason::ExecutionFailure);
                Err(error.context(format!("DFlash generation checkpoint failed {boundary}")))
            }
        }
    }

    /// Seed transaction-local DFlash participants from the committed workflow
    /// session.  Invocation-scoped state deliberately keeps its declared
    /// request initializer; a session cell replaces that seed only after a
    /// complete value has been captured under the exact `(session, cell)`
    /// identity.
    fn restore_dflash_session_state(
        &self,
        plan: &DFlashPlan<'_>,
        session_id: Option<&str>,
        state: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        let Some(session_id) = session_id else {
            return Ok(());
        };
        let committed = self.worker.session_state.borrow();
        for cell in &plan.contract.rollback_state {
            if self.plan.workflow.state.get(cell).is_some_and(|state| {
                state.scope == onnx_genai_metadata::WorkflowStateScope::Session
            }) && let Some(value) = committed.get(&(session_id.to_string(), cell.clone()))
            {
                state.insert(cell.clone(), clone_value(value)?);
            }
        }
        Ok(())
    }

    /// Publish every session-scoped DFlash participant as one all-or-nothing
    /// map update after the complete turn has succeeded.  All cloning and map
    /// reservation happens first, leaving the final inserts infallible; an
    /// execution/cancellation error earlier in the drive therefore leaves the
    /// exact S3 baseline untouched.
    fn commit_dflash_session_state(
        &self,
        plan: &DFlashPlan<'_>,
        session_id: Option<&str>,
        state: &PipelineTensors,
    ) -> anyhow::Result<()> {
        let Some(session_id) = session_id else {
            return Ok(());
        };
        let writes = plan
            .contract
            .rollback_state
            .iter()
            .filter(|cell| {
                self.plan.workflow.state.get(*cell).is_some_and(|state| {
                    state.scope == onnx_genai_metadata::WorkflowStateScope::Session
                })
            })
            .map(|cell| {
                Ok((
                    (session_id.to_string(), cell.clone()),
                    clone_value(state.get(cell).with_context(|| {
                        format!("DFlash commit has no transaction-local state '{cell}'")
                    })?)?,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if writes.is_empty() {
            return Ok(());
        }
        let mut committed = self.worker.session_state.borrow_mut();
        committed
            .try_reserve(writes.len())
            .context("failed to reserve the DFlash session-state commit write set")?;
        for (identity, value) in writes {
            committed.insert(identity, value);
        }
        drop(committed);
        let mut versions = self.worker.session_turn_versions.borrow_mut();
        let version = versions.entry(session_id.to_string()).or_default();
        *version = version.saturating_add(1);
        Ok(())
    }

    /// Invoke the declared target graph for the current context or a verifier
    /// block and install every declared output under its workflow SSA binding.
    fn invoke_dflash_target(
        &self,
        plan: &DFlashPlan<'_>,
        values: &mut PipelineTensors,
        state: &PipelineTensors,
        block: Option<&[i64]>,
    ) -> anyhow::Result<()> {
        let mut owned = Vec::new();
        for (port, value_name) in &plan.target_bindings {
            let value =
                if port == &plan.target_tokens_input {
                    match block {
                    Some(tokens) => Value::from_slice_i64(
                        tokens,
                        &[1, i64::try_from(tokens.len()).context(
                            "DFlash verifier block length does not fit the target tensor shape",
                        )?],
                    )?,
                    None => clone_value(values.get(value_name).with_context(|| {
                        format!(
                            "DFlash target input '{port}' references unavailable workflow value \
                             '{value_name}'"
                        )
                    })?)?,
                }
                } else if let Some(cell) =
                    plan.state_for_component_input(&self.plan.workflow, &plan.contract.target, port)
                {
                    clone_value(state.get(cell).with_context(|| {
                        format!(
                            "DFlash target input '{port}' reads rollback state '{cell}', which is \
                         unavailable"
                        )
                    })?)?
                } else {
                    clone_value(values.get(value_name).with_context(|| {
                        format!(
                            "DFlash target input '{port}' references unavailable workflow value \
                         '{value_name}'"
                        )
                    })?)?
                };
            owned.push((port.clone(), value));
        }
        let input_refs = owned
            .iter()
            .map(|(port, value)| (port.as_str(), value))
            .collect::<Vec<_>>();
        let produced = self
            .invoke_component_values(
                &plan.contract.target,
                &input_refs,
                &plan.target_outputs,
                &std::collections::HashMap::new(),
                1,
            )
            .with_context(|| format!("invoke DFlash target '{}'", plan.contract.target))?;
        for (port, value) in produced {
            let value_name = plan.target_outputs.get(&port).with_context(|| {
                format!(
                    "DFlash target '{}' produced undeclared output '{port}'",
                    plan.contract.target
                )
            })?;
            values.insert(value_name.clone(), value);
        }
        Ok(())
    }

    fn dflash_anchor_token(
        &self,
        plan: &DFlashPlan<'_>,
        values: &PipelineTensors,
        options: &GenerateOptions,
        rng: &mut StdRng,
    ) -> anyhow::Result<i64> {
        let value_name = plan
            .target_outputs
            .get(&plan.outputs.verifier_logits.output)
            .with_context(|| {
                format!(
                    "DFlash target logits {}::{} have no workflow binding",
                    plan.outputs.verifier_logits.component, plan.outputs.verifier_logits.output
                )
            })?;
        let logits = values.get(value_name).with_context(|| {
            format!("DFlash target did not produce declared logits value '{value_name}'")
        })?;
        let shape = logits.shape();
        anyhow::ensure!(
            shape.len() == 3 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0,
            "DFlash target logits have shape {shape:?}; expected [1, sequence, vocabulary]"
        );
        let vocabulary =
            usize::try_from(shape[2]).context("DFlash target vocabulary extent is negative")?;
        let data = logits.to_vec_f32_lossy()?;
        let start = data
            .len()
            .checked_sub(vocabulary)
            .context("DFlash target logits omit their final token distribution")?;
        if options.selects_greedily() {
            Ok(crate::sampling::sample_greedy(&data[start..]) as i64)
        } else {
            Ok(
                sample_probability_row(&softmax(&data[start..], options.temperature), rng.random())?
                    as i64,
            )
        }
    }

    /// Materialize one target-hidden-conditioned DFlash proposal block.
    ///
    /// Shared batching is intentionally not claimed: this path admits exactly
    /// one request row and tells a batching caller to execute rows in isolation
    /// before the proposer mutates draft-private state. The same contract and
    /// component dispatch are used for every isolated row.
    pub(crate) fn propose_dflash(
        &self,
        run: &PipelineTensors,
        options: DFlashProposalOptions,
    ) -> anyhow::Result<DFlashProposal> {
        self.require_execution_admitted()?;
        let contract = self
            .plan
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        let plan = DFlashPlan::resolve(contract, &self.plan.workflow)?;
        anyhow::ensure!(
            options.width >= 1,
            "DFlash proposal width must be at least 1"
        );
        anyhow::ensure!(
            options.width <= contract.max_proposal_width,
            "requested DFlash proposal width {} exceeds max_proposal_width {}",
            options.width,
            contract.max_proposal_width
        );
        if matches!(options.mode, DFlashProposalMode::Sampling { .. }) {
            anyhow::ensure!(
                plan.has_sampling_probabilities(),
                "DFlash sampling requires declared proposal probabilities, but contract \
                 version {} exposes neither full-vocabulary probabilities nor a DFlash 2 \
                 selector distribution; use greedy execution or re-export the proposer",
                plan.version
            );
        }

        let proposer = self
            .plan
            .workflow
            .components
            .get(&contract.proposer)
            .expect("metadata validation admitted the proposer");
        let residency = self.component_execution_residency()?;
        let ops = super::device_ops::tensor_ops_for_residency(residency)?;

        let mut hidden_sources = Vec::with_capacity(plan.conditioning.sources.len());
        let mut conditioning_trace = Vec::with_capacity(plan.conditioning.sources.len());
        let mut batch = None;
        let mut context = None;
        let mut hidden_total = 0usize;
        let mut hidden_dtype = None;
        for source in &plan.conditioning.sources {
            let value_name = plan.target_outputs.get(&source.output).with_context(|| {
                format!(
                    "DFlash conditioning source {}::{} has no workflow output binding",
                    source.component, source.output
                )
            })?;
            let value = run.get(value_name).with_context(|| {
                format!(
                    "DFlash conditioning source {}::{} expects workflow value '{value_name}', \
                     which the target pass did not produce",
                    source.component, source.output
                )
            })?;
            let shape = value.shape();
            anyhow::ensure!(
                shape.len() == 3,
                "DFlash conditioning source {}::{} has shape {shape:?}; expected \
                 [batch, sequence, hidden]",
                source.component,
                source.output
            );
            let source_batch = usize::try_from(shape[0]).context("negative DFlash batch extent")?;
            let source_context =
                usize::try_from(shape[1]).context("negative DFlash context extent")?;
            let source_hidden =
                usize::try_from(shape[2]).context("negative DFlash hidden extent")?;
            if let Some(expected) = batch {
                anyhow::ensure!(
                    source_batch == expected,
                    "DFlash conditioning source {}::{} has batch {source_batch}, expected \
                     {expected}",
                    source.component,
                    source.output
                );
            } else {
                batch = Some(source_batch);
            }
            if let Some(expected) = context {
                anyhow::ensure!(
                    source_context == expected,
                    "DFlash conditioning source {}::{} has sequence {source_context}, expected \
                     {expected}",
                    source.component,
                    source.output
                );
            } else {
                context = Some(source_context);
            }
            if let Some(expected) = hidden_dtype {
                anyhow::ensure!(
                    value.dtype() == expected,
                    "DFlash conditioning source {}::{} has {:?}, expected {expected:?}; \
                     concatenate inputs must share one tensor currency",
                    source.component,
                    source.output,
                    value.dtype()
                );
            } else {
                hidden_dtype = Some(value.dtype());
            }
            hidden_total = hidden_total
                .checked_add(source_hidden)
                .context("DFlash conditioning hidden width overflow")?;
            conditioning_trace.push(DFlashConditioningTrace {
                source: format!("{}::{}", source.component, source.output),
                shape: shape.to_vec(),
            });
            hidden_sources.push(ops.adopt(value).with_context(|| {
                format!(
                    "bring DFlash conditioning source {}::{} onto {residency}",
                    source.component, source.output
                )
            })?);
        }
        let batch = batch.context("DFlash declares no target hidden conditioning sources")?;
        let context = context.context("DFlash target hidden context is unavailable")?;
        anyhow::ensure!(
            batch == 1,
            "shared DFlash proposal batching is not supported for batch {batch}; decline the \
             shared optimization before mutation and execute each request row in isolation"
        );
        anyhow::ensure!(
            context > 0,
            "DFlash conditioning requires at least one accepted target-hidden position"
        );
        let hidden_dtype = hidden_dtype.expect("a non-empty source list has a dtype");
        let conditioning = ops.zeros(
            &[
                i64::try_from(batch).context("DFlash batch exceeds i64")?,
                i64::try_from(context).context("DFlash context exceeds i64")?,
                i64::try_from(hidden_total).context("DFlash hidden width exceeds i64")?,
            ],
            hidden_dtype,
        )?;
        let mut hidden_offset = 0usize;
        for source in &hidden_sources {
            ops.scatter_into_last_axis(&conditioning, hidden_offset, source)?;
            hidden_offset += usize::try_from(
                *source
                    .shape()
                    .last()
                    .context("DFlash hidden source has rank zero")?,
            )
            .context("negative DFlash hidden width")?;
        }

        let block_len = proposer
            .ports
            .inputs
            .get(&plan.block.noise_embeddings_input)
            .and_then(|contract| contract.shape.get(1))
            .and_then(|dimension| match dimension {
                onnx_genai_metadata::TensorDimension::Fixed(extent) => {
                    usize::try_from(*extent).ok()
                }
                onnx_genai_metadata::TensorDimension::Symbol(_)
                | onnx_genai_metadata::TensorDimension::Any => None,
            })
            .with_context(|| {
                format!(
                    "DFlash proposer '{}' noise input '{}' must declare a fixed block extent; \
                     proposal width is runtime policy within that structural capacity",
                    contract.proposer, plan.block.noise_embeddings_input
                )
            })?;
        anyhow::ensure!(
            plan.block.first_candidate_position + options.width <= block_len,
            "requested DFlash width {} does not fit declared block extent {block_len} with first \
             candidate position {}",
            options.width,
            plan.block.first_candidate_position
        );
        let embedding =
            self.embedding_table_resident(&plan.shared_weights.input_embedding, residency)?;
        anyhow::ensure!(
            options.anchor_token >= 0 && (options.anchor_token as usize) < embedding.vocab_size(),
            "DFlash anchor token {} is outside shared target vocabulary {}",
            options.anchor_token,
            embedding.vocab_size()
        );
        anyhow::ensure!(
            (plan.block.mask_token_id as usize) < embedding.vocab_size(),
            "DFlash mask token {} is outside shared target vocabulary {}",
            plan.block.mask_token_id,
            embedding.vocab_size()
        );
        let mut noise_ids = vec![i64::from(plan.block.mask_token_id); block_len];
        noise_ids[plan.block.anchor_position] = options.anchor_token;
        let gathered = ops.gather_rows(embedding.value(), &noise_ids)?;
        let noise_embeddings = alias_with_shape(
            gathered,
            &[
                1,
                i64::try_from(block_len).context("DFlash block length exceeds i64")?,
                i64::try_from(embedding.hidden_size())
                    .context("DFlash embedding width exceeds i64")?,
            ],
        )
        .map_err(|error| anyhow::anyhow!("shape DFlash noise embeddings: {error}"))?;
        let masked_positions = Value::from_raw_bytes(
            (0..block_len)
                .map(|position| u8::from(position >= plan.block.first_candidate_position))
                .collect(),
            &[1, block_len as i64],
            DataType::Bool,
        )?;
        let total_positions = context
            .checked_add(block_len)
            .context("DFlash position length overflow")?;
        let position_ids = Value::from_slice_i64(
            &(0..total_positions)
                .map(|offset| {
                    options
                        .context_start_position
                        .checked_add(offset as i64)
                        .context("DFlash absolute position overflow")
                })
                .collect::<anyhow::Result<Vec<_>>>()?,
            &[1, total_positions as i64],
        )?;
        let attention_contract = proposer
            .ports
            .inputs
            .get(&plan.block.attention_mask_input)
            .expect("metadata validation admitted the attention-mask input");
        let attention_mask = match attention_contract.dtype.as_str() {
            "bool" => Value::from_raw_bytes(
                vec![1; total_positions],
                &[1, total_positions as i64],
                DataType::Bool,
            )?,
            "int64" => {
                Value::from_slice_i64(&vec![1; total_positions], &[1, total_positions as i64])?
            }
            dtype => anyhow::bail!(
                "DFlash attention mask port '{}' has unsupported dtype {dtype}; metadata \
                 admission should have required bool or int64",
                plan.block.attention_mask_input
            ),
        };
        let output_projection = self.shared_initializer(&plan.shared_weights.output_projection)?;
        let projection_shape = output_projection.shape();
        let expected_projection = match plan.shared_weights.output_projection.layout {
            onnx_genai_metadata::DFlashProjectionLayout::HiddenVocabulary => {
                [embedding.hidden_size(), embedding.vocab_size()]
            }
            onnx_genai_metadata::DFlashProjectionLayout::VocabularyHidden => {
                [embedding.vocab_size(), embedding.hidden_size()]
            }
        };
        anyhow::ensure!(
            projection_shape == [expected_projection[0] as i64, expected_projection[1] as i64],
            "DFlash shared output projection '{}' has shape {projection_shape:?}, but layout \
             {:?} and embedding [{}, {}] require [{}, {}]",
            plan.shared_weights.output_projection.initializer,
            plan.shared_weights.output_projection.layout,
            embedding.vocab_size(),
            embedding.hidden_size(),
            expected_projection[0],
            expected_projection[1]
        );

        let overridden = [
            plan.conditioning.proposer_input.as_str(),
            plan.block.noise_embeddings_input.as_str(),
            plan.block.masked_positions_input.as_str(),
            plan.block.position_ids_input.as_str(),
            plan.block.attention_mask_input.as_str(),
            plan.shared_weights
                .output_projection
                .proposer_input
                .as_str(),
        ]
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
        let mut owned_inputs = Vec::new();
        for (port, value_name) in &plan.proposer_bindings {
            if overridden.contains(port.as_str()) {
                continue;
            }
            let value = run.get(value_name).with_context(|| {
                format!(
                    "DFlash proposer '{}' input '{port}' references unavailable workflow value \
                     '{value_name}'",
                    contract.proposer
                )
            })?;
            owned_inputs.push((port.clone(), clone_value(value)?));
        }
        owned_inputs.extend([
            (plan.conditioning.proposer_input.clone(), conditioning),
            (plan.block.noise_embeddings_input.clone(), noise_embeddings),
            (plan.block.masked_positions_input.clone(), masked_positions),
            (plan.block.position_ids_input.clone(), position_ids),
            (plan.block.attention_mask_input.clone(), attention_mask),
            (
                plan.shared_weights.output_projection.proposer_input.clone(),
                clone_value(&output_projection)?,
            ),
        ]);
        let input_refs = owned_inputs
            .iter()
            .map(|(port, value)| (port.as_str(), value))
            .collect::<Vec<_>>();
        let produced = self.invoke_component_values(
            &contract.proposer,
            &input_refs,
            &plan.proposer_outputs,
            &std::collections::HashMap::new(),
            1,
        )?;
        let mut proposer_outputs = PipelineTensors::new();
        for (port, value) in produced {
            proposer_outputs.insert(port, value);
        }

        let candidate_value = proposer_outputs
            .get(&plan.outputs.candidate_tokens)
            .with_context(|| {
                format!(
                    "DFlash proposer '{}' did not produce candidate_tokens output '{}'",
                    contract.proposer, plan.outputs.candidate_tokens
                )
            })?;
        let candidate_shape = candidate_value.shape();
        anyhow::ensure!(
            candidate_shape.len() == 2 && candidate_shape[0] == 1,
            "DFlash candidate_tokens has shape {candidate_shape:?}; isolated execution requires \
             [1, proposal]"
        );
        let proposal_extent =
            usize::try_from(candidate_shape[1]).context("negative DFlash proposal extent")?;
        anyhow::ensure!(
            proposal_extent >= options.width,
            "DFlash proposer produced {proposal_extent} candidates for requested width {}",
            options.width
        );
        let mut tokens = candidate_value.to_vec_i64()?[..options.width].to_vec();
        for (position, token) in tokens.iter().copied().enumerate() {
            anyhow::ensure!(
                token >= 0 && (token as usize) < embedding.vocab_size(),
                "DFlash proposer emitted candidate token {token} at position {position}, outside \
                 shared target vocabulary {}",
                embedding.vocab_size()
            );
        }

        let mut probabilities =
            self.dflash_proposal_probabilities(&plan, &proposer_outputs, embedding.vocab_size())?;
        if let Some(probabilities) = &probabilities {
            anyhow::ensure!(
                probabilities.proposal_len() >= options.width,
                "DFlash proposer probabilities cover {} positions, requested {}",
                probabilities.proposal_len(),
                options.width
            );
            anyhow::ensure!(
                probabilities.vocabulary() == embedding.vocab_size(),
                "DFlash proposer probabilities use vocabulary {}, but the shared target \
                 embedding declares {}",
                probabilities.vocabulary(),
                embedding.vocab_size()
            );
        }

        let verification_seed = match options.mode {
            DFlashProposalMode::Greedy => 0,
            DFlashProposalMode::Sampling { seed } => {
                let probabilities = probabilities
                    .as_ref()
                    .context("DFlash sampling passed admission without a proposal distribution")?;
                let mut rng = StdRng::seed_from_u64(seed);
                for (position, token) in tokens.iter_mut().enumerate() {
                    let row = probabilities.row(position)?;
                    *token = sample_probability_row(&row, rng.random())? as i64;
                }
                rng.random()
            }
        };

        if let Some(position) = tokens
            .iter()
            .position(|token| options.eos_token_ids.contains(token))
        {
            tokens.truncate(position + 1);
            if let Some(probabilities) = &mut probabilities {
                probabilities.truncate(position + 1);
            }
        } else if let Some(probabilities) = &mut probabilities {
            probabilities.truncate(options.width);
        }

        Ok(DFlashProposal {
            tokens,
            probabilities,
            proposer_outputs,
            conditioning_trace,
            verification_seed,
        })
    }

    fn dflash_proposal_probabilities(
        &self,
        plan: &DFlashPlan<'_>,
        outputs: &PipelineTensors,
        vocabulary: usize,
    ) -> anyhow::Result<Option<DFlashProposalProbabilities>> {
        if let DFlashStructure::SelectorConvolutionV1 { selector, .. } = plan.structure {
            let Some(probability_port) = &selector.conditional_probabilities_output else {
                return Ok(None);
            };
            let ids = outputs
                .get(&selector.candidate_ids_output)
                .with_context(|| {
                    format!(
                        "DFlash 2 proposer did not produce selector candidate ids '{}'",
                        selector.candidate_ids_output
                    )
                })?;
            let probabilities = outputs.get(probability_port).with_context(|| {
                format!(
                    "DFlash 2 proposer did not produce selector conditional probabilities \
                     '{probability_port}'"
                )
            })?;
            let ids_shape = ids.shape();
            let probability_shape = probabilities.shape();
            anyhow::ensure!(
                ids_shape == probability_shape
                    && ids_shape.len() == 3
                    && ids_shape[0] == 1
                    && ids_shape[2] == selector.top_k as i64,
                "DFlash 2 selector ids/probabilities must both be [1, proposal, top_k={}], got \
                 {ids_shape:?} and {probability_shape:?}",
                selector.top_k
            );
            let proposal =
                usize::try_from(ids_shape[1]).context("negative DFlash 2 proposal extent")?;
            let candidate_ids = ids.to_vec_i64()?;
            let values = probabilities.to_vec_f32()?;
            validate_probability_rows(&values, proposal, selector.top_k, probability_port)?;
            return Ok(Some(DFlashProposalProbabilities::SparseCandidates {
                candidate_ids,
                values,
                proposal,
                candidates: selector.top_k,
                vocabulary,
            }));
        }
        if let Some(port) = &plan.outputs.proposal_probabilities {
            let value = outputs.get(port).with_context(|| {
                format!("DFlash proposer did not produce declared proposal probabilities '{port}'")
            })?;
            let shape = value.shape();
            anyhow::ensure!(
                shape.len() == 3 && shape[0] == 1,
                "DFlash proposal probabilities '{port}' have shape {shape:?}; expected \
                 [1, proposal, vocabulary]"
            );
            let proposal =
                usize::try_from(shape[1]).context("negative DFlash proposal probability extent")?;
            let declared_vocabulary = usize::try_from(shape[2])
                .context("negative DFlash probability vocabulary extent")?;
            anyhow::ensure!(
                declared_vocabulary == vocabulary,
                "DFlash proposal probabilities '{port}' declare vocabulary \
                 {declared_vocabulary}, but shared target weights declare {vocabulary}"
            );
            let values = value.to_vec_f32()?;
            validate_probability_rows(&values, proposal, vocabulary, port)?;
            return Ok(Some(DFlashProposalProbabilities::FullVocabulary {
                values,
                proposal,
                vocabulary,
            }));
        }
        Ok(None)
    }

    /// Verify a DFlash proposal against the target's exact logits.
    pub(crate) fn verify_dflash(
        &self,
        verified: &PipelineTensors,
        proposal: &DFlashProposal,
        mode: DFlashVerificationMode,
    ) -> anyhow::Result<DFlashAcceptance> {
        self.require_execution_admitted()?;
        let contract = self
            .plan
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        let plan = DFlashPlan::resolve(contract, &self.plan.workflow)?;
        if matches!(mode, DFlashVerificationMode::Sampling { .. }) {
            anyhow::ensure!(
                proposal.probabilities.is_some(),
                "DFlash sampling verification requires proposal probabilities; this proposal \
                 carries none, so rejection sampling cannot preserve the target distribution"
            );
        }
        let value_name = plan
            .target_outputs
            .get(&plan.outputs.verifier_logits.output)
            .with_context(|| {
                format!(
                    "DFlash verifier logits output '{}::{}' has no workflow binding",
                    plan.outputs.verifier_logits.component, plan.outputs.verifier_logits.output
                )
            })?;
        let logits = verified.get(value_name).with_context(|| {
            format!("DFlash verification pass did not produce target logits value '{value_name}'")
        })?;
        let shape = logits.shape();
        anyhow::ensure!(
            shape.len() == 3 && shape[0] == 1,
            "DFlash verifier logits have shape {shape:?}; isolated verification requires \
             [1, proposal_plus_bonus, vocabulary]"
        );
        let rows = usize::try_from(shape[1]).context("negative DFlash verifier row extent")?;
        let vocabulary =
            usize::try_from(shape[2]).context("negative DFlash verifier vocabulary extent")?;
        anyhow::ensure!(
            rows == proposal.tokens.len() + 1,
            "verifying a {}-token DFlash proposal requires {} target distributions (one bonus), \
             found {rows}",
            proposal.tokens.len(),
            proposal.tokens.len() + 1
        );
        let target_logits = logits.to_vec_f32()?;

        let (accepted, correction) = match mode {
            DFlashVerificationMode::Greedy => {
                let mut accepted = 0usize;
                for (position, token) in proposal.tokens.iter().enumerate() {
                    let row = &target_logits[position * vocabulary..(position + 1) * vocabulary];
                    let target = crate::sampling::sample_greedy(row) as i64;
                    if target != *token {
                        break;
                    }
                    accepted += 1;
                }
                let row = &target_logits[accepted * vocabulary..(accepted + 1) * vocabulary];
                (accepted, crate::sampling::sample_greedy(row) as i64)
            }
            DFlashVerificationMode::Sampling { temperature } => {
                anyhow::ensure!(
                    temperature.is_finite() && temperature > 0.0,
                    "DFlash sampling temperature must be finite and positive, got {temperature}"
                );
                let probabilities = proposal
                    .probabilities
                    .as_ref()
                    .expect("sampling checked probabilities above");
                anyhow::ensure!(
                    probabilities.vocabulary() == vocabulary,
                    "DFlash proposal vocabulary {} does not match verifier vocabulary \
                     {vocabulary}",
                    probabilities.vocabulary()
                );
                let target_probabilities = target_logits
                    .chunks_exact(vocabulary)
                    .flat_map(|row| softmax(row, temperature))
                    .collect::<Vec<_>>();
                let mut rng = StdRng::seed_from_u64(proposal.verification_seed);
                let mut accepted = 0usize;
                for (position, token) in proposal.tokens.iter().enumerate() {
                    let token = usize::try_from(*token)
                        .ok()
                        .filter(|token| *token < vocabulary)
                        .with_context(|| {
                            format!(
                                "DFlash proposed token {} at position {position} outside \
                                 verifier vocabulary {vocabulary}",
                                proposal.tokens[position]
                            )
                        })?;
                    let q = probabilities.row(position)?;
                    let p_token = target_probabilities[position * vocabulary + token];
                    let q_token = q[token];
                    anyhow::ensure!(
                        q_token.is_finite() && q_token > 0.0,
                        "DFlash proposal probability q({token}) at position {position} is \
                         {q_token}; the emitted token must have positive finite probability"
                    );
                    if rng.random::<f32>() * q_token >= p_token {
                        let p = &target_probabilities
                            [position * vocabulary..(position + 1) * vocabulary];
                        let residual = p
                            .iter()
                            .zip(q)
                            .map(|(p, q)| (p - q).max(0.0))
                            .collect::<Vec<_>>();
                        let total: f32 = residual.iter().sum();
                        let distribution = if total > 0.0 && total.is_finite() {
                            residual
                                .iter()
                                .map(|probability| probability / total)
                                .collect::<Vec<_>>()
                        } else {
                            p.to_vec()
                        };
                        let correction =
                            sample_probability_row(&distribution, rng.random())? as i64;
                        return Ok(DFlashAcceptance {
                            accepted,
                            rejected_at: Some(position),
                            committed: proposal.tokens[..accepted]
                                .iter()
                                .copied()
                                .chain(std::iter::once(correction))
                                .collect(),
                        });
                    }
                    accepted += 1;
                }
                let bonus =
                    &target_probabilities[accepted * vocabulary..(accepted + 1) * vocabulary];
                (
                    accepted,
                    sample_probability_row(bonus, rng.random())? as i64,
                )
            }
        };
        Ok(DFlashAcceptance {
            accepted,
            rejected_at: (accepted < proposal.tokens.len()).then_some(accepted),
            committed: proposal.tokens[..accepted]
                .iter()
                .copied()
                .chain(std::iter::once(correction))
                .collect(),
        })
    }

    /// Admit every declared accepted-prefix participant under the S3
    /// transaction identity before proposer or verifier execution.
    pub(crate) fn begin_dflash_state_transaction(
        &self,
        current: &PipelineTensors,
    ) -> anyhow::Result<DFlashStateTransaction> {
        self.require_execution_admitted()?;
        let contract = self
            .plan
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        DFlashPlan::resolve(contract, &self.plan.workflow)?;
        let mut baseline = PipelineTensors::new();
        for cell in &contract.rollback_state {
            let value = current.get(cell).with_context(|| {
                format!(
                    "cannot admit DFlash transaction: rollback participant state '{cell}' has no \
                     current value"
                )
            })?;
            baseline.insert(cell.clone(), clone_value(value)?);
        }
        Ok(DFlashStateTransaction {
            turn: super::TurnTransaction::admit_runtime_participant(
                self.worker.next_turn_transaction_id(),
            ),
            baseline,
        })
    }

    /// Atomically replace every DFlash participant with the state for exactly
    /// the accepted proposal prefix.
    ///
    /// `acceptance.committed` remains commit-only output until this returns a
    /// committed outcome. A caller may deliver it to a non-retractable callback
    /// only afterwards; a callback error is then a post-commit delivery failure
    /// and never rewinds already committed target/draft state, matching the S3
    /// decoder baseline.
    pub(crate) fn commit_dflash_state_transaction(
        &self,
        transaction: DFlashStateTransaction,
        current: &mut PipelineTensors,
        proposal: &DFlashProposal,
        verified: &PipelineTensors,
        acceptance: &DFlashAcceptance,
        committed_target_tokens: usize,
    ) -> anyhow::Result<super::TurnTransactionOutcome> {
        self.require_execution_admitted()?;
        let contract = self
            .plan
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        let plan = DFlashPlan::resolve(contract, &self.plan.workflow)?;
        anyhow::ensure!(
            acceptance.accepted <= proposal.tokens.len(),
            "DFlash acceptance length {} exceeds proposal length {}",
            acceptance.accepted,
            proposal.tokens.len()
        );
        let mut committed = clone_pipeline_tensors(current)?;
        for (cell, commit) in plan.accepted_prefix_state {
            let state = self
                .plan
                .workflow
                .state
                .get(cell)
                .with_context(|| format!("DFlash state '{cell}' is undeclared"))?;
            let group_name = state
                .service_group
                .as_deref()
                .with_context(|| format!("DFlash state '{cell}' has no state-service group"))?;
            let group = self
                .plan
                .workflow
                .serving
                .as_ref()
                .and_then(|serving| serving.state_service.groups.get(group_name))
                .with_context(|| {
                    format!("DFlash state '{cell}' group '{group_name}' is undeclared")
                })?;
            let source = match commit {
                DFlashStateCommit::Sequence { source }
                | DFlashStateCommit::PrefixSnapshots { source, .. } => source,
            };
            let value = if source.component == contract.proposer {
                proposal.proposer_outputs.get(&source.output)
            } else if source.component == contract.target {
                let value_name = plan.source_value_name(source).with_context(|| {
                    format!(
                        "DFlash state '{cell}' source {}::{} has no workflow binding",
                        source.component, source.output
                    )
                })?;
                verified.get(value_name)
            } else {
                None
            }
            .with_context(|| {
                format!(
                    "DFlash state '{cell}' source {}::{} was not produced",
                    source.component, source.output
                )
            })?;
            let source_is_target = source.component == contract.target;
            let committed_prefix = if source_is_target {
                committed_target_tokens
            } else {
                acceptance.accepted
            };
            let accepted_value = match commit {
                DFlashStateCommit::Sequence { .. } => {
                    let axis = group.sequence_axis.with_context(|| {
                        format!("DFlash sequence state '{cell}' has no sequence axis")
                    })?;
                    let baseline = transaction.baseline.get(cell).with_context(|| {
                        format!("DFlash transaction has no baseline for state '{cell}'")
                    })?;
                    let baseline_len =
                        usize::try_from(*baseline.shape().get(axis).with_context(|| {
                            format!(
                                "DFlash state '{cell}' sequence axis {axis} is outside baseline \
                                 shape {:?}",
                                baseline.shape()
                            )
                        })?)
                        .context("negative DFlash baseline sequence extent")?;
                    let length = baseline_len
                        .checked_add(committed_prefix)
                        .context("DFlash accepted state length overflow")?;
                    super::device_ops::tensor_ops_for(value)?
                        .truncate_axis(value, axis, length)
                        .with_context(|| {
                            format!("truncate DFlash state '{cell}' to accepted prefix {length}")
                        })?
                }
                DFlashStateCommit::PrefixSnapshots { axis, .. } => {
                    let snapshot = if source_is_target {
                        committed_prefix.checked_sub(1).with_context(|| {
                            format!(
                                "DFlash target state '{cell}' cannot select a snapshot for an \
                                 empty committed path"
                            )
                        })?
                    } else {
                        committed_prefix
                    };
                    let slice = super::device_ops::tensor_ops_for(value)?
                        .slice_axis(value, *axis, snapshot, 1)
                        .with_context(|| {
                            format!("select DFlash state '{cell}' prefix snapshot {snapshot}")
                        })?;
                    let mut shape = slice.shape().to_vec();
                    shape.remove(*axis);
                    alias_with_shape(slice, &shape).map_err(|error| {
                        anyhow::anyhow!("remove DFlash state '{cell}' prefix axis {axis}: {error}")
                    })?
                }
            };
            committed.insert(cell.clone(), accepted_value);
        }
        *current = committed;
        Ok(transaction.turn.committed())
    }

    /// Abort a DFlash block to its complete admitted participant baseline.
    pub(crate) fn abort_dflash_state_transaction(
        &self,
        transaction: DFlashStateTransaction,
        current: &mut PipelineTensors,
        reason: super::TurnAbortReason,
    ) -> anyhow::Result<super::TurnTransactionOutcome> {
        self.require_execution_admitted()?;
        for (cell, value) in transaction.baseline {
            current.insert(cell, value);
        }
        Ok(transaction.turn.abort(reason))
    }

    /// Drive a chained speculative proposal through the interpreter's own
    /// component seam.
    ///
    /// `run` holds the SSA values of the completed verification pass — the same
    /// map [`WorkflowRuntime::run_workflow`] produces. Every proposer port is
    /// bound from the workflow's own invocation of that component, so the
    /// borrowed read-only shared KV, position ids, and masks are exactly the
    /// tensors the package declared; only `token_embedding_input` is overridden,
    /// because the fused `concat(embed(last_token), carry)` is what the chain
    /// recomputes each step.
    pub fn propose_chained(
        &self,
        run: &PipelineTensors,
        options: ChainedProposalOptions,
    ) -> anyhow::Result<ChainedProposal> {
        let contract = self
            .plan
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        let plan = ChainedPlan::resolve(contract, &self.plan.workflow)?;
        anyhow::ensure!(
            options.width >= 1,
            "chained proposal width must be at least 1"
        );
        anyhow::ensure!(
            options.width <= contract.max_proposal_width,
            "requested proposal width {} exceeds the package's max_proposal_width {}",
            options.width,
            contract.max_proposal_width
        );

        let declaration = self
            .plan
            .workflow
            .components
            .get(plan.proposer)
            .with_context(|| format!("workflow component '{}' is undeclared", plan.proposer))?;
        // A chained step advances exactly one position, so every proposer port
        // that shares the fused input's position symbol narrows to that step's
        // single position. The symbol comes from the declared port contracts, so
        // a package whose position axis sits elsewhere — or whose mask is keyed
        // on a separate `kv_sequence` symbol, as a borrowed-KV drafter's is —
        // narrows exactly the ports it declared and leaves the rest alone.
        let position_symbol = position_symbol(declaration, plan.token_embedding_input);

        // Fixed proposer bindings: everything the workflow binds except the
        // fused input the chain rebuilds each step.
        let mut fixed: Vec<(String, Value)> = Vec::new();
        for (port, value_name) in &plan.proposer_bindings {
            if port == plan.token_embedding_input {
                continue;
            }
            if plan.recurrent.iter().any(|binding| &binding.input == port) {
                continue;
            }
            // A port declared as a read-only view of another component's state
            // binds what that cell holds *now*, not the seed the pass began
            // with.
            let value_name = plan.borrowed_state.get(port).unwrap_or(value_name);
            let value = run.get(value_name).with_context(|| {
                format!(
                    "chained proposer '{}' input '{port}' references unavailable workflow value \
                     '{value_name}'",
                    plan.proposer
                )
            })?;
            // Narrow first, and where the value already is. Copying the wide
            // tensor down to keep one position of it was the largest transfer
            // in this loop, and every byte of it was discarded on the next
            // line.
            let narrowed = match position_symbol
                .as_deref()
                .and_then(|symbol| symbol_axis(declaration, port, symbol))
            {
                Some(axis) => super::device_ops::tensor_ops_for(value)?
                    .last_along_axis(value, axis)
                    .with_context(|| {
                        format!(
                            "failed to narrow chained proposer '{}' input '{port}' to one \
                             position",
                            plan.proposer
                        )
                    })?,
                // A port with no position symbol binds exactly what the
                // workflow's own invocation bound it, so it is passed through
                // as it stands. Aliasing keeps a device-resident borrowed KV
                // where it is; materializing it here would copy the largest
                // tensor in the loop to learn nothing about it.
                None => clone_value(value)?,
            };
            fixed.push((port.clone(), narrowed));
        }

        // Loop-carried recurrent state: seeded from the declared workflow state
        // cell, threaded input -> output every step.
        let mut recurrent_state: BTreeMap<&str, Value> = BTreeMap::new();
        for binding in plan.recurrent {
            let seed = self
                .workflow_session_state_value(&binding.state)
                .or_else(|| run.get(&binding.state).and_then(|v| clone_value(v).ok()))
                .with_context(|| {
                    format!(
                        "chained proposer recurrence '{}' has no value for state cell '{}'",
                        binding.input, binding.state
                    )
                })?;
            recurrent_state.insert(binding.input.as_str(), seed);
        }

        // Folded carry: carry_0 is the target output the contract names, taken
        // at its most recent position — and left where that output already is.
        // The chain's residency follows from it: the carry is one half of every
        // fused input the loop builds, so whatever device produced it is the
        // device the fused input is assembled on.
        let carry_seed = match &plan.folded_carry_seed_value {
            Some(value_name) => Some(run.get(value_name).with_context(|| {
                format!(
                    "folded_carry_seed names target output value '{value_name}', which the \
                     verification pass did not produce"
                )
            })?),
            None => None,
        };

        // The fused input's declared shape tells the loop how wide each half is
        // and what batch/query extents to build; take it from the port contract
        // the workflow already bound for the single-pass invocation.
        let fused_template = run
            .get(
                plan.proposer_bindings
                    .get(plan.token_embedding_input)
                    .with_context(|| {
                        format!(
                            "chained proposer '{}' does not bind its declared \
                             token_embedding_input '{}'",
                            plan.proposer, plan.token_embedding_input
                        )
                    })?,
            )
            .context("the workflow's fused proposer input value is unavailable")?;
        // The chain's arithmetic currency is whatever the package declared for
        // its fused proposer input: an fp16 export's embeddings, carry and
        // fused buffer are fp16, and widening them here would both cost a
        // conversion per draft token and feed the proposer a tensor its own
        // port contract does not describe.
        let fused_dtype = fused_template.dtype();
        // Where the chain does its tensor algebra: where the proposer executes.
        // That is configured, not discovered, so the fused input is built in the
        // proposer's own memory from the first step rather than assembled on the
        // host and uploaded once per draft token.
        let residency = self.component_execution_residency()?;
        let ops = super::device_ops::tensor_ops_for_residency(residency)?;

        let mut carry = match carry_seed {
            Some(seed) => {
                // carry_0 comes from the target's verification pass, which may
                // have left it wherever that pass emitted it. Narrow it first —
                // one position out of the whole prompt — and adopt only that,
                // once per proposal. Every carry after this one is produced by
                // the proposer itself and is already here.
                let narrowed =
                    last_position_of_carry(super::device_ops::tensor_ops_for(seed)?.as_ref(), seed)
                        .context("folded_carry_seed must name a per-position hidden output")?;
                Some(self.adopt_into(ops.as_ref(), &narrowed).with_context(|| {
                    format!("failed to bring the folded carry seed onto {residency}")
                })?)
            }
            None => None,
        };

        let embedding = match plan.token_embedding {
            Some(source) => {
                let (component, table) = (source.component.as_str(), source.table.as_str());
                let loaded = self.embedding_table_resident(source, residency)?;
                // Both halves are written into one buffer, so the table the
                // package names must speak the fused input's element type.
                // Naming both sides is what makes a mismatched export
                // actionable instead of a wrong-looking draft.
                anyhow::ensure!(
                    loaded.dtype() == fused_dtype,
                    "token_embedding.table '{table}' of component '{component}' is {:?}, but the \
                     proposer's declared fused input '{}' is {fused_dtype:?}; the gathered \
                     embedding is written into that buffer, so the two must agree",
                    loaded.dtype(),
                    plan.token_embedding_input
                );
                Some(loaded)
            }
            None => None,
        };

        // One proposal step advances exactly one position, so the fused input a
        // step binds keeps the workflow's batch extent and its declared width
        // but holds a single position, whatever the single-pass binding's
        // sequence extent was.
        let fused_shape = single_position_shape(fused_template.shape())
            .context("the workflow's fused proposer input has an unusable shape")?;
        let fused_width = *fused_shape
            .last()
            .context("fused proposer input has rank 0")? as usize;
        let batch_rows = fused_shape[..fused_shape.len() - 1]
            .iter()
            .try_fold(1usize, |total, dimension| {
                usize::try_from(*dimension)
                    .ok()
                    .and_then(|dimension| total.checked_mul(dimension))
            })
            .context("fused proposer input has an unusable shape")?;

        // The split between the two halves is loop-invariant — the embedding
        // table's width and the carry's are both fixed by the package — so it
        // is checked once here rather than rediscovered every step.
        let leading = embedding
            .as_ref()
            .map(|table| table.hidden_size())
            .unwrap_or(0);
        match &carry {
            Some(carry) => {
                let (carry_rows, carry_width) = trailing_rows_and_width(carry, "folded carry")?;
                anyhow::ensure!(
                    carry_rows == batch_rows,
                    "folded carry covers {carry_rows} batch rows, but the fused proposer input \
                     has {batch_rows}"
                );
                anyhow::ensure!(
                    leading + carry_width == fused_width,
                    "fused proposer input is {fused_width} wide, but embed({leading}) + \
                     carry({carry_width}) is {}",
                    leading + carry_width
                );
            }
            None => anyhow::ensure!(
                leading == fused_width,
                "fused proposer input is {fused_width} wide, but the gathered embedding is \
                 {leading}"
            ),
        }

        // One buffer for the whole chain, in the residency the halves live in.
        // Rebuilding it per step would allocate — and on a device, upload — the
        // concatenation the scatters below write in place.
        let fused = ops
            .zeros(&fused_shape, fused_dtype)
            .context("failed to allocate the fused proposer input")?;

        let outputs = self.proposer_output_bindings(&plan);
        // Symbols the proposer's *outputs* declare that none of its inputs
        // carry — a vocabulary, above all. The workflow already invoked this
        // component in this very run, so its own bound output values prove the
        // extents; without them the run cannot size a device buffer for those
        // outputs and hands each one back through host memory.
        let output_symbols = proposer_output_symbols(declaration, &plan, run);
        let mut tokens = Vec::with_capacity(options.width);
        tokens.push(options.guaranteed_token);
        let mut last_token = options.seed_token;
        let mut invocations = 0usize;

        for step in 0..options.width {
            if let Some(table) = embedding.as_ref() {
                // The same token embeds into every batch row, so the gather is
                // asked for that row once per row rather than broadcast — the
                // scatter then needs no broadcasting rule to get wrong.
                let embedded = ops
                    .gather_rows(table.value(), &vec![last_token; batch_rows])
                    .with_context(|| {
                        format!("failed to gather the embedding of token {last_token}")
                    })?;
                ops.scatter_into_last_axis(&fused, 0, &embedded)
                    .with_context(|| {
                        format!("failed to place the embedding half of step {step}'s fused input")
                    })?;
            }
            if let Some(carry) = carry.as_ref() {
                ops.scatter_into_last_axis(&fused, leading, carry)
                    .with_context(|| {
                        format!("failed to place the carry half of step {step}'s fused input")
                    })?;
            }
            let mut bound: Vec<(&str, &Value)> = fixed
                .iter()
                .map(|(port, value)| (port.as_str(), value))
                .collect();
            bound.push((plan.token_embedding_input, &fused));
            for (port, value) in &recurrent_state {
                bound.push((port, value));
            }

            let produced = self
                .invoke_component_values(
                    plan.proposer,
                    &bound,
                    &outputs,
                    &output_symbols,
                    batch_rows,
                )
                .with_context(|| {
                    format!("chained proposer '{}' step {step} failed", plan.proposer)
                })?;
            invocations += 1;

            let mut logits = None;
            let mut next_carry = None;
            let mut next_recurrent: BTreeMap<&str, Value> = BTreeMap::new();
            for (port, value) in produced {
                if port == plan.logits_output {
                    logits = Some(value);
                } else if Some(port.as_str()) == plan.folded_carry_output {
                    next_carry = Some(value);
                } else if let Some(binding) =
                    plan.recurrent.iter().find(|binding| binding.output == port)
                {
                    next_recurrent.insert(binding.input.as_str(), value);
                }
            }
            let logits = logits.with_context(|| {
                format!(
                    "chained proposer '{}' produced no '{}' output",
                    plan.proposer, plan.logits_output
                )
            })?;
            // Four bytes back, not a vocabulary: the winning id is all this
            // step needs, and a device that produced the row can find it.
            let drafted = i64::from(self.argmax_token(&logits)?);

            if let Some(value) = next_carry {
                // Narrow where the proposer left it, then adopt — the same two
                // steps the seed takes. Deriving the operations from the value
                // rather than from the chain's residency is what keeps this
                // correct when a backend publishes an output somewhere other
                // than where the chain assembles; on the common path the
                // residencies already agree and the adoption is an O(1) alias.
                let narrowed = last_position_of_carry(
                    super::device_ops::tensor_ops_for(&value)?.as_ref(),
                    &value,
                )
                .with_context(|| {
                    format!(
                        "chained proposer '{}' folded carry output '{}' is not a per-position \
                         hidden state",
                        plan.proposer,
                        plan.folded_carry_output.unwrap_or_default()
                    )
                })?;
                carry = Some(self.adopt_into(ops.as_ref(), &narrowed).with_context(|| {
                    format!("failed to bring the folded carry onto {residency}")
                })?);
            } else if plan.folded_carry_output.is_some() {
                anyhow::bail!(
                    "chained proposer '{}' declares folded_carry_output '{}' but produced none",
                    plan.proposer,
                    plan.folded_carry_output.unwrap_or_default()
                );
            }
            for binding in plan.recurrent {
                let value = next_recurrent.remove(binding.input.as_str()).with_context(|| {
                    format!(
                        "chained proposer '{}' declares recurrence output '{}' but produced none",
                        plan.proposer, binding.output
                    )
                })?;
                recurrent_state.insert(binding.input.as_str(), value);
            }

            if step == 0 {
                // carry_0 is the target's hidden state for the last *context*
                // token, so this step reproduces the target's own next-token
                // prediction — which the target already computed and handed over
                // as `guaranteed_token`. Its draft is therefore redundant and
                // discarded; the step is still run because the chain's carry
                // must advance past the guaranteed token before the first real
                // draft conditions on it. This is a property of the folded-carry
                // contract, not of any particular model.
                last_token = options.guaranteed_token;
            } else {
                tokens.push(drafted);
                last_token = drafted;
            }
        }

        Ok(ChainedProposal {
            tokens,
            proposer_invocations: invocations,
        })
    }

    /// Longest proposal prefix the target confirms.
    ///
    /// `target_tokens[i]` is the target's own token for block position `i`, and
    /// one entry beyond the block is required: verification consumed the whole
    /// block, so the target already produced the token that follows it. That
    /// entry is what makes a fully-accepted block commit `width + 1` tokens and
    /// a rejected block commit the target's correction instead of the draft.
    ///
    /// Position 0 matches by construction — it *is* the target's token — so a
    /// proposal always commits at least one token, and the result equals plain
    /// greedy decoding for a `distribution_preserving` package.
    pub fn accept_chained_proposal(
        &self,
        proposal: &ChainedProposal,
        target_tokens: &[i64],
    ) -> anyhow::Result<ProposalAcceptance> {
        anyhow::ensure!(
            target_tokens.len() == proposal.tokens.len() + 1,
            "verifying a {}-token proposal block needs {} target tokens (one beyond the block), \
             found {}",
            proposal.tokens.len(),
            proposal.tokens.len() + 1,
            target_tokens.len()
        );
        let mut accepted = 0;
        let mut rejected_at = None;
        for (position, proposed) in proposal.tokens.iter().enumerate() {
            if target_tokens[position] == *proposed {
                accepted += 1;
            } else {
                rejected_at = Some(position);
                break;
            }
        }
        anyhow::ensure!(
            accepted >= 1,
            "block position 0 must be the target's own guaranteed token, but the target says \
             {} and the block says {}",
            target_tokens[0],
            proposal.tokens[0]
        );
        let mut committed = proposal.tokens[..accepted].to_vec();
        committed.push(target_tokens[accepted]);
        Ok(ProposalAcceptance {
            accepted,
            rejected_at,
            committed,
        })
    }

    /// Collect the declared `rollback_state` cells out of a completed pass.
    ///
    /// Each cell is resolved through the serving state contract to the target
    /// output the pass wrote it from, so the caller never has to know which port
    /// backs which cell. Values are aliased where the backend allows it, keeping
    /// a device-resident cell on the device instead of forcing a host round-trip
    /// just to be checkpointed.
    pub fn speculative_rollback_state(
        &self,
        run: &PipelineTensors,
    ) -> anyhow::Result<PipelineTensors> {
        let contract = self
            .plan
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        let mut state = PipelineTensors::new();
        for cell in &contract.rollback_state {
            let port = target_state_output(&self.plan.workflow, &contract.target, cell)
                .with_context(|| {
                    format!(
                        "rollback_state cell '{cell}' names no output port on the speculative \
                         target '{}'",
                        contract.target
                    )
                })?;
            let value_name = component_invocation(&self.plan.workflow, &contract.target)
                .and_then(|(_, outputs)| outputs.get(&port).cloned())
                .with_context(|| {
                    format!("the workflow binds no value to target output '{port}'")
                })?;
            let value = run.get(&value_name).with_context(|| {
                format!("rollback_state cell '{cell}' has no value '{value_name}' in this pass")
            })?;
            state.insert(cell.clone(), clone_value(value)?);
        }
        Ok(state)
    }

    /// Truncate every declared `rollback_state` cell to `length` positions.
    ///
    /// The sequence axis comes from the serving state-group the cell belongs to,
    /// so a package whose KV is not `[batch, heads, sequence, head_dim]` rolls
    /// back on its own declared axis. Cells outside `rollback_state` — a folded
    /// carry above all — are untouched by design: they are recomputed from the
    /// committed tokens.
    pub fn rollback_speculative_state(
        &self,
        state: &mut PipelineTensors,
        length: usize,
    ) -> anyhow::Result<()> {
        let contract = self
            .plan
            .speculative
            .as_ref()
            .context("package declares no speculative contract")?;
        for cell in &contract.rollback_state {
            let axis = self.state_sequence_axis(cell).with_context(|| {
                format!("rollback_state cell '{cell}' declares no sequence axis")
            })?;
            let value = state
                .get(cell)
                .with_context(|| format!("rollback_state cell '{cell}' has no value to undo"))?;
            // Narrowing happens where the cell already is. A rejection used to
            // copy the entire KV cache to the host to drop a few positions and
            // upload it again on the next bind — the single largest transfer in
            // the workflow, paid per rejection.
            let truncated = super::device_ops::tensor_ops_for(value)?
                .truncate_axis(value, axis, length)
                .with_context(|| format!("failed to roll back state cell '{cell}'"))?;
            state.insert(cell.clone(), truncated);
        }
        Ok(())
    }

    /// Bring `value` into `ops`' residency, counting anything it brings back.
    ///
    /// Adoption is the seam's one sanctioned crossing, so this is the one place
    /// besides an argmax read-back where a device byte can reach the host — and
    /// therefore the one place besides [`Self::argmax_token`] that has to say
    /// so. A value already in the right residency is aliased and counts
    /// nothing.
    fn adopt_into(
        &self,
        ops: &dyn super::device_ops::ResidentTensorOps,
        value: &Value,
    ) -> anyhow::Result<Value> {
        let source = super::device_ops::residency_of(value)?;
        if source != ops.residency() && ops.residency() == super::device_ops::Residency::Host {
            let bytes = (value.numel() * value.dtype().size_of()) as u64;
            self.worker.counters.device_readback_bytes.set(
                self.worker
                    .counters
                    .device_readback_bytes
                    .get()
                    .saturating_add(bytes),
            );
        }
        ops.adopt(value)
    }

    /// The winning token id of a logits row, read where the row already is.
    ///
    /// Four bytes per row, and on a device-resident chain it is the only
    /// per-step transfer there is. Counting it here — together with
    /// [`Self::adopt_into`], the seam's one other crossing, and the
    /// interpreter's own materialization, which counts itself in
    /// `host_staging_count` — accounts for every byte a proposal brings back.
    fn argmax_token(&self, logits: &Value) -> anyhow::Result<u32> {
        let ops = super::device_ops::tensor_ops_for(logits)?;
        let ids = ops.argmax_rows(logits, 1)?;
        let id = *ids.first().context("argmax returned no row")?;
        if ops.residency() != super::device_ops::Residency::Host {
            let bytes = (ids.len() * std::mem::size_of::<u32>()) as u64;
            self.worker.counters.device_readback_bytes.set(
                self.worker
                    .counters
                    .device_readback_bytes
                    .get()
                    .saturating_add(bytes),
            );
        }
        Ok(id)
    }

    /// Sequence axis a state cell rolls back along, from its serving group.
    fn state_sequence_axis(&self, cell: &str) -> Option<usize> {
        let serving = self.plan.workflow.serving.as_ref()?;
        serving
            .state_service
            .groups
            .values()
            .find(|group| {
                group
                    .ports
                    .values()
                    .any(|component| component.contains_key(cell))
            })
            .and_then(|group| group.sequence_axis)
    }

    fn workflow_session_state_value(&self, cell: &str) -> Option<Value> {
        self.worker
            .session_state
            .borrow()
            .iter()
            .find(|((_, name), _)| name == cell)
            .and_then(|(_, value)| clone_value(value).ok())
    }

    fn proposer_output_bindings(&self, plan: &ChainedPlan<'_>) -> BTreeMap<String, String> {
        let mut outputs = plan.proposer_outputs.clone();
        // The chain reads its own recurrence and carry even when the single-pass
        // step list ignores them.
        outputs
            .entry(plan.logits_output.to_string())
            .or_insert_with(|| format!("{}.{}", plan.proposer, plan.logits_output));
        if let Some(port) = plan.folded_carry_output {
            outputs
                .entry(port.to_string())
                .or_insert_with(|| format!("{}.{port}", plan.proposer));
        }
        for binding in plan.recurrent {
            outputs
                .entry(binding.output.clone())
                .or_insert_with(|| format!("{}.{}", plan.proposer, binding.output));
        }
        outputs
    }

    fn shared_initializer(
        &self,
        source: &onnx_genai_metadata::DFlashOutputProjection,
    ) -> anyhow::Result<Value> {
        let key = (source.component.clone(), source.initializer.clone());
        if let Some(cached) = self.worker.shared_initializers.borrow().get(&key) {
            return clone_value(cached);
        }
        let path = self
            .backend
            .models
            .directory
            .model_paths
            .get(&source.component)
            .with_context(|| {
                format!(
                    "DFlash output projection names component '{}', which has no ONNX artifact",
                    source.component
                )
            })?
            .to_path_buf();
        let (graph, weights) =
            onnx_runtime_loader::load_model_with_weights(&path).with_context(|| {
                format!(
                    "failed to read DFlash shared output projection '{}' from component '{}'",
                    source.initializer, source.component
                )
            })?;
        let (value_id, info) = graph
            .values
            .iter()
            .find(|(_, value)| value.name.as_deref() == Some(source.initializer.as_str()))
            .with_context(|| {
                format!(
                    "DFlash output projection initializer '{}' is not a value in component '{}' \
                     ({})",
                    source.initializer,
                    source.component,
                    path.display()
                )
            })?;
        let weight = graph.initializers.get(&value_id).with_context(|| {
            format!(
                "DFlash output projection '{}' in component '{}' is not an immutable initializer",
                source.initializer, source.component
            )
        })?;
        let shape = onnx_runtime_ir::as_static_shape(&info.shape).with_context(|| {
            format!(
                "DFlash output projection '{}' has no static matrix shape",
                source.initializer
            )
        })?;
        anyhow::ensure!(
            shape.len() == 2,
            "DFlash output projection '{}' must be a rank-2 matrix, found shape {shape:?}",
            source.initializer
        );
        let dtype = super::device_ops::value_dtype_from_ir(info.dtype).with_context(|| {
            format!(
                "DFlash output projection '{}' has unsupported element type {:?}",
                source.initializer, info.dtype
            )
        })?;
        anyhow::ensure!(
            matches!(
                dtype,
                DataType::Float16 | DataType::BFloat16 | DataType::Float32
            ),
            "DFlash output projection '{}' must be floating, got {dtype:?}",
            source.initializer
        );
        let bytes = weights.bytes(weight).with_context(|| {
            format!(
                "DFlash output projection '{}' has no initializer bytes",
                source.initializer
            )
        })?;
        anyhow::ensure!(
            bytes.len() == shape[0] * shape[1] * dtype.size_of(),
            "DFlash output projection '{}' holds {} bytes for shape {shape:?} {dtype:?}, which \
             needs {}",
            source.initializer,
            bytes.len(),
            shape[0] * shape[1] * dtype.size_of()
        );
        let value =
            Value::from_raw_bytes(bytes.to_vec(), &[shape[0] as i64, shape[1] as i64], dtype)
                .map_err(|error| {
                    anyhow::anyhow!(
                        "materialize DFlash output projection '{}': {error}",
                        source.initializer
                    )
                })?;
        self.worker
            .shared_initializers
            .borrow_mut()
            .insert(key, std::rc::Rc::new(clone_value(&value)?));
        Ok(value)
    }

    /// Read the declared `[vocab, hidden]` embedding table out of a component's
    /// ONNX artifact.
    ///
    /// The contract names both the component and the initializer, so this never
    /// guesses which weight is the embedding: a missing or non-2D initializer is
    /// an error naming the contract field that pointed at it.
    /// The declared embedding table, resident where the chain will gather it.
    ///
    /// Read from the artifact once and — for a device residency — uploaded
    /// once, for the runtime's life. A loaded package's artifact cannot change
    /// under it, so re-reading a `[vocab, hidden]` initializer per proposal
    /// buys nothing; re-uploading one per *draft token* would cost more than
    /// the proposal it feeds. The cache is keyed by residency as well as by
    /// name, because a host table and a device mirror are different tensors
    /// answering the same question.
    pub fn embedding_table_resident(
        &self,
        source: &onnx_genai_metadata::TokenEmbeddingSource,
        residency: super::device_ops::Residency,
    ) -> anyhow::Result<std::rc::Rc<EmbeddingTable>> {
        let (component, table) = (source.component.as_str(), source.table.as_str());
        // The declared normalizer is part of what the cached rows *are*, so it
        // is part of what identifies them. Keyed on the bits rather than the
        // value because a cache key has to be hashable and total.
        let key = (
            component.to_string(),
            table.to_string(),
            residency.cache_key(),
            source.scale.map(f32::to_bits),
        );
        if let Some(cached) = self.worker.embedding_tables.borrow().get(&key) {
            return Ok(std::rc::Rc::clone(cached));
        }
        let loaded = std::rc::Rc::new(self.embedding_table(source)?.into_residency(
            residency,
        ).with_context(|| {
            format!("failed to make embedding table '{table}' of component '{component}' resident on {residency}")
        })?);
        self.worker
            .embedding_tables
            .borrow_mut()
            .insert(key, std::rc::Rc::clone(&loaded));
        Ok(loaded)
    }

    /// How many times an embedding table was read out of an artifact.
    ///
    /// The cache above is a performance contract, not an implementation detail:
    /// a multi-round speculative decode that re-read the table would be correct
    /// and unusably slow, which is the class of regression a throughput number
    /// does not attribute. This is what a test holds to one read per table.
    pub fn embedding_table_loads(&self) -> u64 {
        self.worker.counters.embedding_table_loads.get()
    }

    pub fn embedding_table(
        &self,
        source: &onnx_genai_metadata::TokenEmbeddingSource,
    ) -> anyhow::Result<EmbeddingTable> {
        let (component, table) = (source.component.as_str(), source.table.as_str());
        let path = self
            .backend
            .models
            .directory
            .model_paths
            .get(component)
            .with_context(|| {
                format!("token_embedding names component '{component}', which has no ONNX artifact")
            })?
            .to_path_buf();
        let (graph, weights) = onnx_runtime_loader::load_model_with_weights(&path)
            .with_context(|| format!("failed to read '{component}' to gather '{table}'"))?;
        let (value_id, info) = graph
            .values
            .iter()
            .find(|(_, value)| value.name.as_deref() == Some(table))
            .with_context(|| {
                format!(
                    "token_embedding.table '{table}' is not a value in component '{component}' \
                     ({})",
                    path.display()
                )
            })?;
        let weight = graph.initializers.get(&value_id).with_context(|| {
            format!("token_embedding.table '{table}' in '{component}' is not an initializer")
        })?;
        let shape = onnx_runtime_ir::as_static_shape(&info.shape).with_context(|| {
            format!("token_embedding.table '{table}' has no static [vocab, hidden] shape")
        })?;
        anyhow::ensure!(
            shape.len() == 2,
            "token_embedding.table '{table}' must be a [vocab, hidden] matrix, found rank {}",
            shape.len()
        );
        let (vocab, hidden) = (shape[0], shape[1]);
        let dtype = super::device_ops::value_dtype_from_ir(info.dtype).with_context(|| {
            format!(
                "token_embedding.table '{table}' of component '{component}' has element type {:?} \
                 ({}), which the workflow value currency does not carry",
                info.dtype,
                path.display()
            )
        })?;
        let bytes = weights
            .bytes(weight)
            .with_context(|| format!("token_embedding.table '{table}' has no weight bytes"))?;
        anyhow::ensure!(
            bytes.len() == vocab * hidden * dtype.size_of(),
            "token_embedding.table '{table}' holds {} bytes for a [{vocab}, {hidden}] {dtype:?} \
             matrix, which needs {}",
            bytes.len(),
            vocab * hidden * dtype.size_of()
        );
        // The declared normalizer is applied once, to the table, rather than
        // per gathered row: the proposer consumes the target's *scaled*
        // embedding at every step, and a real vocabulary would otherwise pay
        // the conversion per draft token.
        let rows = match source.scale {
            None => bytes.to_vec(),
            Some(scale) => scaled_rows(bytes, dtype, scale).with_context(|| {
                format!(
                    "failed to apply the declared token_embedding.scale {scale} to \
                     '{table}' of component '{component}'"
                )
            })?,
        };
        self.worker.counters.embedding_table_loads.set(
            self.worker
                .counters
                .embedding_table_loads
                .get()
                .saturating_add(1),
        );
        Ok(EmbeddingTable {
            vocab,
            hidden,
            value: Value::from_raw_bytes(
                rows,
                &[
                    i64::try_from(vocab).context("embedding vocabulary exceeds i64")?,
                    i64::try_from(hidden).context("embedding width exceeds i64")?,
                ],
                dtype,
            )
            .map_err(|error| {
                anyhow::anyhow!("failed to materialize embedding table '{table}': {error}")
            })?,
        })
    }
}

/// `bytes`, reinterpreted as `dtype` elements and multiplied by `scale`.
///
/// The multiply is done in `f32` and rounded back, which is what a graph
/// constant folded into a half-precision model computes. Only the float types
/// a table can hold are supported: scaling an integer table would quantize the
/// factor away silently, so it is refused with a diagnostic instead.
fn scaled_rows(bytes: &[u8], dtype: DataType, scale: f32) -> anyhow::Result<Vec<u8>> {
    Ok(match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .flat_map(|chunk| {
                (f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) * scale).to_le_bytes()
            })
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .flat_map(|chunk| {
                half::f16::from_f32(half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32() * scale)
                    .to_le_bytes()
            })
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .flat_map(|chunk| {
                half::bf16::from_f32(
                    half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32() * scale,
                )
                .to_le_bytes()
            })
            .collect(),
        other => anyhow::bail!(
            "token_embedding.scale is declared for a {other:?} table; a normalizer is only \
             meaningful on a float table, so either drop the scale or export the table scaled"
        ),
    })
}

fn clone_pipeline_tensors(values: &PipelineTensors) -> anyhow::Result<PipelineTensors> {
    values
        .iter()
        .map(|(name, value)| Ok((name.clone(), clone_value(value)?)))
        .collect()
}

#[allow(clippy::arc_with_non_send_sync)]
fn alias_with_shape(value: Value, shape: &[i64]) -> onnx_genai_ort::Result<Value> {
    Value::alias_with_shape(std::sync::Arc::new(value), shape)
}

fn validate_probability_rows(
    values: &[f32],
    rows: usize,
    width: usize,
    name: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        values.len() == rows * width,
        "DFlash probability output '{name}' has {} elements, expected {rows} * {width}",
        values.len()
    );
    for (row_index, row) in values.chunks_exact(width).enumerate() {
        anyhow::ensure!(
            row.iter()
                .all(|probability| probability.is_finite() && *probability >= 0.0),
            "DFlash probability output '{name}' row {row_index} contains a negative or non-finite \
             value"
        );
        let total: f32 = row.iter().sum();
        anyhow::ensure!(
            total.is_finite() && (total - 1.0).abs() <= 1e-4,
            "DFlash probability output '{name}' row {row_index} sums to {total}, expected 1; \
             rejection sampling requires normalized proposal probabilities"
        );
    }
    Ok(())
}

fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| value / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let mut probabilities = vec![0.0; logits.len()];
        if !probabilities.is_empty() {
            probabilities[0] = 1.0;
        }
        return probabilities;
    }
    let mut probabilities = logits
        .iter()
        .map(|value| {
            if value.is_finite() {
                (value / temperature - max).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let total: f32 = probabilities.iter().sum();
    if total > 0.0 && total.is_finite() {
        for probability in &mut probabilities {
            *probability /= total;
        }
    }
    probabilities
}

fn sample_probability_row(probabilities: &[f32], draw: f32) -> anyhow::Result<usize> {
    anyhow::ensure!(
        !probabilities.is_empty(),
        "cannot sample an empty DFlash probability row"
    );
    anyhow::ensure!(
        probabilities
            .iter()
            .all(|probability| probability.is_finite() && *probability >= 0.0),
        "DFlash sampling distribution contains a negative or non-finite probability"
    );
    let total: f32 = probabilities.iter().sum();
    anyhow::ensure!(
        total.is_finite() && total > 0.0,
        "DFlash sampling distribution has total probability {total}"
    );
    let target = draw.clamp(0.0, 1.0 - f32::EPSILON) * total;
    let mut cumulative = 0.0f32;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if target < cumulative {
            return Ok(index);
        }
    }
    Ok(probabilities.len() - 1)
}

/// A dense `[vocab, hidden]` row-major embedding table read from a component,
/// held in one residency.
///
/// The table is a *tensor*, not a host array: the gather that reads it happens
/// wherever the proposal chain runs, and a table that only ever exists on the
/// host forces every gathered row across the bus. Holding it as a `Value` is
/// what lets the same type describe the host copy and the device mirror.
pub struct EmbeddingTable {
    vocab: usize,
    hidden: usize,
    value: Value,
}

impl std::fmt::Debug for EmbeddingTable {
    /// A table's contents are a `[vocab, hidden]` matrix that may not even be
    /// host-readable, so the shape is the whole of what is printable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingTable")
            .field("vocab", &self.vocab)
            .field("hidden", &self.hidden)
            .field("shape", &self.value.shape())
            .finish()
    }
}

impl EmbeddingTable {
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// The element type this table's rows are held in.
    ///
    /// A table is read out of the artifact in the type the export wrote, so
    /// this is the package's answer, not a runtime preference. The chain
    /// checks it against the fused proposer input it is written into.
    pub fn dtype(&self) -> DataType {
        self.value.dtype()
    }

    /// The table as a `[vocab, hidden]` tensor, for a residency-preserving
    /// gather.
    pub(crate) fn value(&self) -> &Value {
        &self.value
    }

    /// This table mirrored into `residency`, uploading once if it must.
    ///
    /// Takes the table by value because a mirror *replaces* it: the host copy a
    /// device chain was built from has no further reader, and at a real
    /// vocabulary keeping it alive alongside the mirror would double the
    /// footprint of the largest tensor in the package.
    pub(crate) fn into_residency(
        self,
        residency: super::device_ops::Residency,
    ) -> anyhow::Result<Self> {
        if super::device_ops::residency_of(&self.value)? == residency {
            return Ok(self);
        }
        anyhow::ensure!(
            residency != super::device_ops::Residency::Host,
            "an embedding table already resident on a device is not brought back to the host; \
             gather it where it is, or run this package on the host backend"
        );
        // Adoption is the seam's own crossing, so this upload is ordered against
        // the kernels that will read the table by the same fence every other
        // device write goes through — rather than by an invariant this function
        // would otherwise have to state and the next caller remember.
        let mirror = super::device_ops::tensor_ops_for_residency(residency)?
            .adopt(&self.value)
            .with_context(|| {
                format!(
                    "failed to make a [{}, {}] embedding table resident on {residency}",
                    self.vocab, self.hidden
                )
            })?;
        Ok(Self {
            vocab: self.vocab,
            hidden: self.hidden,
            value: mirror,
        })
    }

    /// The embedding row for `token`, widened to `f32`.
    ///
    /// An out-of-range id is an error rather than a clamp: a proposer that
    /// drafted an id the table cannot embed has left the declared vocabulary,
    /// and silently embedding row 0 would hide that. A device-resident table
    /// has no host rows to borrow and says so.
    ///
    /// One row is widened, never the table: at a real vocabulary a whole-table
    /// conversion is gigabytes to answer a question about 1536 numbers. The
    /// gather the chain actually runs stays in the table's own element type.
    pub fn row(&self, token: i64) -> anyhow::Result<Vec<f32>> {
        let index = usize::try_from(token)
            .ok()
            .filter(|index| *index < self.vocab)
            .with_context(|| {
                format!(
                    "token id {token} has no embedding row in a [{}, {}] table",
                    self.vocab, self.hidden
                )
            })?;
        let dtype = self.value.dtype();
        let element = dtype.size_of();
        let bytes = self.value.as_raw_bytes().map_err(|error| {
            anyhow::anyhow!("this embedding table's rows cannot be read from the host: {error}")
        })?;
        let row = &bytes[index * self.hidden * element..(index + 1) * self.hidden * element];
        match dtype {
            DataType::Float32 => Ok(row
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            DataType::Float16 => Ok(row
                .chunks_exact(2)
                .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                .collect()),
            DataType::BFloat16 => Ok(row
                .chunks_exact(2)
                .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                .collect()),
            other => anyhow::bail!(
                "an embedding row of a {other:?} table has no lossless f32 widening; gather it \
                 in its own element type instead of reading rows"
            ),
        }
    }
}

impl<'a> ChainedPlan<'a> {
    fn resolve(
        contract: &'a SpeculativeContract,
        workflow: &'a WorkflowSpec,
    ) -> anyhow::Result<Self> {
        let SpeculativeProposalExecution::Chained {
            token_embedding_input,
            logits_output,
            recurrent,
            folded_carry_output,
            folded_carry_seed,
            token_embedding,
        } = &contract.proposal_execution
        else {
            anyhow::bail!(
                "speculative.proposal_execution is not `chained`; block proposers emit their \
                 whole block in one invocation and need no proposal chain"
            );
        };
        anyhow::ensure!(
            !recurrent.is_empty() || folded_carry_output.is_some(),
            "a chained proposer must declare at least one of `recurrent` or \
             `folded_carry_output`"
        );
        let (proposer_bindings, proposer_outputs) =
            component_invocation(workflow, &contract.proposer).with_context(|| {
                format!(
                    "speculative proposer '{}' is never invoked by the workflow, so its ports \
                     cannot be bound",
                    contract.proposer
                )
            })?;
        let folded_carry_seed_value = match folded_carry_seed {
            Some(seed) => {
                anyhow::ensure!(
                    seed.component == contract.target,
                    "folded_carry_seed.component '{}' must be the speculative target '{}'",
                    seed.component,
                    contract.target
                );
                let (_, target_outputs) = component_invocation(workflow, &contract.target)
                    .with_context(|| {
                        format!(
                            "speculative target '{}' is never invoked by the workflow",
                            contract.target
                        )
                    })?;
                Some(
                    target_outputs
                        .get(&seed.output)
                        .with_context(|| {
                            format!(
                                "folded_carry_seed names target output '{}', which the workflow \
                                 does not bind to a value",
                                seed.output
                            )
                        })?
                        .clone(),
                )
            }
            None => None,
        };
        if folded_carry_output.is_some() {
            anyhow::ensure!(
                folded_carry_seed_value.is_some(),
                "a folded-carry proposer must declare `folded_carry_seed`"
            );
            anyhow::ensure!(
                token_embedding.is_some(),
                "a folded-carry proposer must declare `token_embedding`"
            );
            let destination = contract
                .port_bindings
                .get("target_hidden_context")
                .map(String::as_str);
            anyhow::ensure!(
                destination == Some(token_embedding_input.as_str()),
                "a folded carry lands in the fused `token_embedding_input` ('{token_embedding_input}'), \
                 but port_bindings.target_hidden_context names {destination:?}"
            );
        }
        Ok(Self {
            proposer: &contract.proposer,
            token_embedding_input,
            logits_output,
            recurrent,
            folded_carry_output: folded_carry_output.as_deref(),
            folded_carry_seed_value,
            token_embedding: token_embedding.as_ref(),
            proposer_bindings,
            proposer_outputs,
            borrowed_state: borrowed_state_bindings(
                workflow,
                &contract.proposer,
                &contract.target,
            )?,
        })
    }
}

/// Values an interpreter-level speculative construct consumes after a pass.
///
/// Island fusion would otherwise elide them: nothing in the step list reads the
/// target's folded-carry seed or the proposer's own bindings a second time. This
/// is the set that keeps them live, computed from the declared contract so a
/// package without a chained proposer contributes nothing.
pub(super) fn externally_used_values(
    contract: Option<&SpeculativeContract>,
    workflow: &WorkflowSpec,
) -> std::collections::HashSet<String> {
    let mut live = std::collections::HashSet::new();
    let Some(contract) = contract else {
        return live;
    };
    if matches!(
        &contract.proposal_execution,
        SpeculativeProposalExecution::CandidateTree { .. }
    ) {
        for component in [&contract.proposer, &contract.target] {
            if let Some((inputs, outputs)) = component_invocation(workflow, component) {
                live.extend(inputs.into_values());
                live.extend(outputs.into_values());
            }
        }
        return live;
    }
    if let SpeculativeProposalExecution::DflashFlatBlock {
        conditioning,
        outputs,
        accepted_prefix_state,
        ..
    } = &contract.proposal_execution
    {
        if let Some((inputs, outputs)) = component_invocation(workflow, &contract.proposer) {
            live.extend(inputs.into_values());
            live.extend(outputs.into_values());
        }
        if let Some((_, target_outputs)) = component_invocation(workflow, &contract.target) {
            for source in &conditioning.sources {
                if let Some(value) = target_outputs.get(&source.output) {
                    live.insert(value.clone());
                }
            }
            if let Some(value) = target_outputs.get(&outputs.verifier_logits.output) {
                live.insert(value.clone());
            }
            for commit in accepted_prefix_state.values() {
                let source = match commit {
                    DFlashStateCommit::Sequence { source }
                    | DFlashStateCommit::PrefixSnapshots { source, .. } => source,
                };
                if source.component == contract.target
                    && let Some(value) = target_outputs.get(&source.output)
                {
                    live.insert(value.clone());
                }
            }
        }
        return live;
    }
    let SpeculativeProposalExecution::Chained {
        folded_carry_seed, ..
    } = &contract.proposal_execution
    else {
        return live;
    };
    if let Some((inputs, outputs)) = component_invocation(workflow, &contract.proposer) {
        live.extend(inputs.into_values());
        live.extend(outputs.into_values());
    }
    if let Some(seed) = folded_carry_seed
        && let Some((_, outputs)) = component_invocation(workflow, &seed.component)
        && let Some(value) = outputs.get(&seed.output)
    {
        live.insert(value.clone());
    }
    // A read-only borrow reads the value its owner published, so that value has
    // a consumer no graph can see. Without this a package whose borrowed cell is
    // not also a rollback cell would have it elided and the chain would fail
    // looking for it.
    if let Ok(borrowed) = borrowed_state_bindings(workflow, &contract.proposer, &contract.target) {
        live.extend(borrowed.into_values());
    }
    // A rejected proposal restores the declared rollback cells, so the values a
    // pass leaves in them must survive fusion too.
    if let Some((_, outputs)) = component_invocation(workflow, &contract.target) {
        for cell in &contract.rollback_state {
            if let Some(port) = target_state_output(workflow, &contract.target, cell)
                && let Some(value) = outputs.get(&port)
            {
                live.insert(value.clone());
            }
        }
    }
    live
}

/// The component output port a serving state group writes `cell` from.
fn target_state_output(workflow: &WorkflowSpec, component: &str, cell: &str) -> Option<String> {
    let service = &workflow.serving.as_ref()?.state_service;
    service.groups.values().find_map(|group| {
        group
            .ports
            .get(component)
            .and_then(|ports| ports.get(cell))
            .and_then(|port| port.output.clone())
    })
}

/// One component's `invoke` bindings: input ports then output ports, each
/// mapping a declared port name to the SSA value the workflow bound it to.
type ComponentBindings = (BTreeMap<String, String>, BTreeMap<String, String>);

/// The workflow's own `invoke` bindings for a component, port -> SSA value.
fn component_invocation(workflow: &WorkflowSpec, component: &str) -> Option<ComponentBindings> {
    fn walk(steps: &[WorkflowStep], component: &str, found: &mut Option<ComponentBindings>) {
        for step in steps {
            match step {
                WorkflowStep::Invoke {
                    component: name,
                    inputs,
                    outputs,
                } if name == component => {
                    if found.is_none() {
                        *found = Some((inputs.clone(), outputs.clone()));
                    }
                }
                WorkflowStep::Sequence { steps } => walk(steps, component, found),
                WorkflowStep::Loop { setup, steps, .. } => {
                    walk(setup, component, found);
                    walk(steps, component, found);
                }
                WorkflowStep::Branch { cases, default, .. } => {
                    for case in cases.values() {
                        walk(std::slice::from_ref(case), component, found);
                    }
                    if let Some(default) = default {
                        walk(std::slice::from_ref(default), component, found);
                    }
                }
                _ => {}
            }
        }
    }
    let mut found = None;
    walk(&workflow.steps, component, &mut found);
    found
}

/// The shape symbol on the fused proposer input's position axis, if it has one.
fn position_symbol(
    declaration: &onnx_genai_metadata::WorkflowComponent,
    port: &str,
) -> Option<String> {
    let shape = &declaration.ports.inputs.get(port)?.shape;
    if shape.len() < 3 {
        return None;
    }
    match &shape[shape.len() - 2] {
        onnx_genai_metadata::TensorDimension::Symbol(symbol) => Some(symbol.clone()),
        onnx_genai_metadata::TensorDimension::Fixed(_)
        | onnx_genai_metadata::TensorDimension::Any => None,
    }
}

/// Shape symbols a proposer's outputs declare that none of its inputs carry.
///
/// A chain step binds one position of each input, so every symbol an input
/// declares is re-bound per step and must never be hinted. What is left is the
/// symbols only the *outputs* mention — a vocabulary, a projected hidden width
/// — whose extents the workflow's own invocation of this same component already
/// proved: the SSA values it bound those outputs to are in this run, with
/// concrete shapes.
///
/// Without those extents an output has no resolvable shape, so no device buffer
/// can be sized for it and the run returns it through host memory — a download
/// of the full logits row, per draft token, which is the transfer the whole
/// chain exists to avoid.
///
/// A symbol two outputs disagree about is dropped rather than guessed: the
/// consequence of omitting a hint is the host fallback that was there before,
/// while the consequence of a wrong one would be a wrongly sized buffer.
fn proposer_output_symbols(
    declaration: &onnx_genai_metadata::WorkflowComponent,
    plan: &ChainedPlan<'_>,
    run: &PipelineTensors,
) -> std::collections::HashMap<String, i64> {
    let input_symbols = declaration
        .ports
        .inputs
        .values()
        .flat_map(|contract| &contract.shape)
        .filter_map(|dimension| match dimension {
            onnx_genai_metadata::TensorDimension::Symbol(symbol) => Some(symbol.as_str()),
            onnx_genai_metadata::TensorDimension::Fixed(_)
            | onnx_genai_metadata::TensorDimension::Any => None,
        })
        .collect::<std::collections::HashSet<_>>();

    let mut hints: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let mut conflicted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (port, value_name) in &plan.proposer_outputs {
        let Some(contract) = declaration.ports.outputs.get(port) else {
            continue;
        };
        let shape = &contract.shape;
        let Some(produced) = run.get(value_name) else {
            continue;
        };
        if produced.shape().len() != shape.len() {
            continue;
        }
        for (dimension, extent) in shape.iter().zip(produced.shape()) {
            let onnx_genai_metadata::TensorDimension::Symbol(symbol) = dimension else {
                continue;
            };
            if input_symbols.contains(symbol.as_str()) || *extent < 0 {
                continue;
            }
            match hints.get(symbol.as_str()) {
                Some(known) if known != extent => {
                    conflicted.insert(symbol.clone());
                }
                _ => {
                    hints.insert(symbol.clone(), *extent);
                }
            }
        }
    }
    for symbol in conflicted {
        hints.remove(&symbol);
    }
    hints
}

/// The axis of `port` carrying `symbol`, if the port declares it.
fn symbol_axis(
    declaration: &onnx_genai_metadata::WorkflowComponent,
    port: &str,
    symbol: &str,
) -> Option<usize> {
    let shape = &declaration.ports.inputs.get(port)?.shape;
    shape.iter().position(|dimension| {
        matches!(dimension, onnx_genai_metadata::TensorDimension::Symbol(name) if name == symbol)
    })
}

/// The shape one proposal step binds for the fused proposer input.
///
/// A chained step advances exactly one position, so the position axis — the one
/// before the feature axis — collapses to 1. Rank-1 and rank-2 fused inputs have
/// no separate position axis and are passed through unchanged.
fn single_position_shape(shape: &[i64]) -> Option<Vec<i64>> {
    if shape.is_empty() {
        return None;
    }
    let mut resolved = shape.to_vec();
    if resolved.len() >= 3 {
        let position = resolved.len() - 2;
        resolved[position] = 1;
    }
    Some(resolved)
}

/// The most recent position of a `[.., positions, features]` per-position
/// value, left where it already is.
///
/// A folded carry is a per-position hidden state: the seed covers every prompt
/// position, and a proposer step covers one. Both reduce the same way — take the
/// final position — so the loop never has to know which of the two it holds.
///
/// The position axis is the one before the feature axis, which is the same
/// layout [`single_position_shape`] collapses when it builds one step's fused
/// input: a carry and the fused input it lands in cannot disagree about where
/// positions are. A rank-2 or smaller carry has no separate position axis and
/// already holds exactly one position.
fn last_position_of_carry(
    ops: &dyn super::device_ops::ResidentTensorOps,
    value: &Value,
) -> anyhow::Result<Value> {
    let shape = value.shape();
    anyhow::ensure!(
        !shape.is_empty(),
        "a folded carry has rank 0, so it has no feature axis"
    );
    match shape.len() >= 3 {
        true => ops.last_along_axis(value, shape.len() - 2),
        false => clone_value(value),
    }
}

/// The `(rows, width)` of a `[.., width]` value, for the fused-input split.
fn trailing_rows_and_width(value: &Value, role: &str) -> anyhow::Result<(usize, usize)> {
    let shape = value.shape();
    let width = usize::try_from(
        *shape
            .last()
            .with_context(|| format!("the {role} has rank 0, so it has no feature axis"))?,
    )
    .with_context(|| format!("the {role} has a negative width"))?;
    anyhow::ensure!(
        width > 0,
        "the {role} has a zero-width feature axis (shape {shape:?})"
    );
    Ok((value.numel() / width, width))
}

/// Clone a workflow value without changing where it lives.
///
/// A value backed by an allocation it owns — a native session's device buffer,
/// above all — is aliased in O(1) so it stays device-resident. Only an
/// owned-host-backing value is deep-copied.
fn clone_value(value: &Value) -> anyhow::Result<Value> {
    if let Some(aliased) = value.try_alias_clone() {
        return aliased
            .map_err(|error| anyhow::anyhow!("failed to alias a workflow value: {error}"));
    }
    value
        .clone_owned()
        .map_err(|error| anyhow::anyhow!("failed to clone a workflow value: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORKFLOW: &str = r#"
manifest:
  capabilities: [workflow_ssa, typed_emit]
inputs: {}
outputs: {}
components:
  target:
    implementation: {kind: onnx, artifact: target/model.onnx}
    ports:
      inputs: {}
      outputs:
        logits: {dtype: float32, shape: [batch, sequence, vocab]}
        hidden_states.0: {dtype: float32, shape: [batch, sequence, hidden]}
  assistant:
    implementation: {kind: onnx, artifact: assistant/model.onnx}
    ports:
      inputs:
        inputs_embeds: {dtype: float32, shape: [batch, sequence, fused]}
      outputs:
        logits: {dtype: float32, shape: [batch, sequence, vocab]}
steps:
- kind: invoke
  component: assistant
  inputs: {inputs_embeds: request.inputs_embeds}
  outputs: {logits: draft.logits}
- kind: invoke
  component: target
  inputs: {}
  outputs: {logits: target.logits, hidden_states.0: target.hidden_states.0}
"#;

    fn workflow() -> WorkflowSpec {
        serde_yaml::from_str(WORKFLOW).expect("workflow spec")
    }

    fn contract(execution: SpeculativeProposalExecution) -> SpeculativeContract {
        SpeculativeContract {
            identity: "onnx-genai.speculative".to_string(),
            version: "1".to_string(),
            proposer: "assistant".to_string(),
            target: "target".to_string(),
            proposal_execution: execution,
            port_bindings: Default::default(),
            target_port_bindings: Default::default(),
            shared_state: Default::default(),
            shared_weights: Default::default(),
            vocabulary: onnx_genai_metadata::SpeculativeVocabulary::Identical,
            max_proposal_width: 4,
            distribution_preserving: true,
            verification: onnx_genai_metadata::SpeculativeVerification {
                target_output: onnx_genai_metadata::SpeculativeValueRef {
                    component: "target".to_string(),
                    output: "logits".to_string(),
                },
                accepted_path: onnx_genai_metadata::SpeculativeAcceptedPath::Runtime {
                    binding: "accepted_prefix".to_string(),
                },
                probabilities: None,
            },
            rollback_state: Default::default(),
        }
    }

    /// Keeping speculative values live is what stops island fusion from
    /// swallowing a proposal's inputs. It must be scoped to packages that
    /// actually declare a chained proposer: a package without one, or with a
    /// `block` proposer that needs no chain, must contribute nothing, or every
    /// workflow in the repo would silently lose fusion.
    #[test]
    fn only_a_chained_contract_keeps_values_live() {
        assert!(externally_used_values(None, &workflow()).is_empty());
        assert!(
            externally_used_values(
                Some(&contract(SpeculativeProposalExecution::Block)),
                &workflow()
            )
            .is_empty()
        );
    }

    /// A chained proposer's own bindings and its folded-carry seed are the
    /// values the driver reads after the pass, so those exactly are the ones
    /// fusion must not elide.
    #[test]
    fn a_chained_contract_keeps_its_bindings_and_carry_seed_live() {
        let contract = contract(SpeculativeProposalExecution::Chained {
            token_embedding_input: "inputs_embeds".to_string(),
            logits_output: "logits".to_string(),
            recurrent: Vec::new(),
            folded_carry_output: Some("projected_state".to_string()),
            folded_carry_seed: Some(onnx_genai_metadata::SpeculativeValueRef {
                component: "target".to_string(),
                output: "hidden_states.0".to_string(),
            }),
            token_embedding: Some(onnx_genai_metadata::TokenEmbeddingSource {
                component: "target".to_string(),
                table: "hidden_table".to_string(),
                scale: None,
            }),
        });
        let live = externally_used_values(Some(&contract), &workflow());
        assert!(live.contains("request.inputs_embeds"), "{live:?}");
        assert!(live.contains("draft.logits"), "{live:?}");
        assert!(live.contains("target.hidden_states.0"), "{live:?}");
        // The target's own logits are emitted by the workflow, not read by the
        // proposal chain, so they are not force-kept here.
        assert!(!live.contains("target.logits"), "{live:?}");
    }

    /// A folded carry re-enters through the fused input and owns no state cell,
    /// so a package that declares one without saying where carry_0 comes from,
    /// or without naming the embedding table, is rejected rather than
    /// convention-inferred.
    #[test]
    fn a_folded_carry_must_declare_its_seed_and_embedding_table() {
        let mut incomplete = contract(SpeculativeProposalExecution::Chained {
            token_embedding_input: "inputs_embeds".to_string(),
            logits_output: "logits".to_string(),
            recurrent: Vec::new(),
            folded_carry_output: Some("projected_state".to_string()),
            folded_carry_seed: None,
            token_embedding: None,
        });
        let error = ChainedPlan::resolve(&incomplete, &workflow())
            .expect_err("a folded carry without a seed must not resolve");
        assert!(
            format!("{error:#}").contains("folded_carry_seed"),
            "{error:#}"
        );

        // With a seed but no embedding table it is still under-declared.
        if let SpeculativeProposalExecution::Chained {
            folded_carry_seed, ..
        } = &mut incomplete.proposal_execution
        {
            *folded_carry_seed = Some(onnx_genai_metadata::SpeculativeValueRef {
                component: "target".to_string(),
                output: "hidden_states.0".to_string(),
            });
        }
        let error = ChainedPlan::resolve(&incomplete, &workflow())
            .expect_err("a folded carry without an embedding table must not resolve");
        assert!(
            format!("{error:#}").contains("token_embedding"),
            "{error:#}"
        );
    }

    /// A chain that declares neither a recurrence nor a folded carry has nothing
    /// to thread, so repeating the proposer would just re-emit one distribution.
    #[test]
    fn a_chained_proposer_must_declare_something_to_thread() {
        let empty = contract(SpeculativeProposalExecution::Chained {
            token_embedding_input: "inputs_embeds".to_string(),
            logits_output: "logits".to_string(),
            recurrent: Vec::new(),
            folded_carry_output: None,
            folded_carry_seed: None,
            token_embedding: None,
        });
        let error = ChainedPlan::resolve(&empty, &workflow())
            .expect_err("a chain with nothing to thread must not resolve");
        assert!(format!("{error:#}").contains("recurrent"), "{error:#}");
    }
}

#[cfg(test)]
mod rollback_residency_tests {
    use super::*;
    use crate::pipeline::device_ops::{HostTensorOps, ResidentTensorOps};

    /// A batch-1 seq-major cache truncates as a view, not a copy.
    ///
    /// This is the shape the native backend declares, and it is the case that
    /// used to copy the whole KV cache to the host and back on every rejection.
    #[test]
    fn a_contiguous_prefix_truncates_to_the_kept_prefix() {
        // [batch=1, sequence=4, hidden=2]
        let value = Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 4, 2])
            .expect("value");
        let truncated = HostTensorOps
            .truncate_axis(&value, 1, 2)
            .expect("truncation succeeds");
        assert_eq!(truncated.shape(), &[1, 2, 2]);
        assert_eq!(
            truncated.to_vec_f32().expect("host read"),
            vec![0.0, 1.0, 2.0, 3.0],
            "the truncation must be the kept prefix, in order"
        );
    }

    /// A strided truncation keeps the declared elements rather than a prefix of
    /// the buffer.
    ///
    /// `[batch, heads, sequence, head_dim]` keeps a prefix of axis 2, which is
    /// interleaved across heads — not a prefix of the buffer. Handing back a
    /// contiguous view here would silently return the wrong elements, so the
    /// strided case copies within its own residency instead.
    #[test]
    fn a_strided_truncation_keeps_the_declared_elements() {
        // [batch=1, heads=2, sequence=2, head_dim=1]
        let value = Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[1, 2, 2, 1]).expect("value");
        let truncated = HostTensorOps
            .truncate_axis(&value, 2, 1)
            .expect("a head-major cache still truncates");
        assert_eq!(truncated.shape(), &[1, 2, 1, 1]);
        assert_eq!(
            truncated.to_vec_f32().expect("host read"),
            vec![0.0, 2.0],
            "each head keeps its own first position, not the buffer's first two elements"
        );
    }

    /// An unchanged length is a no-op rollback, and still produces the cell.
    #[test]
    fn an_unchanged_length_keeps_every_element() {
        let value = Value::from_slice_f32(&[0.0, 1.0], &[1, 2]).expect("value");
        let kept = HostTensorOps
            .truncate_axis(&value, 1, 2)
            .expect("an unchanged length must succeed");
        assert_eq!(kept.shape(), &[1, 2]);
        assert_eq!(kept.to_vec_f32().expect("host read"), vec![0.0, 1.0]);
    }

    /// A length beyond the cell is refused, naming both extents, rather than
    /// aliasing past the buffer.
    #[test]
    fn a_length_beyond_the_cell_is_refused() {
        let value = Value::from_slice_f32(&[0.0, 1.0], &[1, 2]).expect("value");
        let Err(error) = HostTensorOps.truncate_axis(&value, 1, 5) else {
            panic!("a length past the end must be refused");
        };
        let message = format!("{error:#}");
        assert!(message.contains('5') && message.contains('2'), "{message}");
    }

    /// Truncation is a window of the one narrowing primitive, so the fast
    /// contiguous path cannot disagree with the general one.
    #[test]
    fn truncation_agrees_with_the_slice_primitive() {
        let value =
            Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 3, 2]).expect("value");
        for length in 0..=3 {
            let truncated = HostTensorOps
                .truncate_axis(&value, 1, length)
                .expect("truncation")
                .to_vec_f32()
                .expect("host read");
            let sliced = HostTensorOps
                .slice_axis(&value, 1, 0, length)
                .expect("slice")
                .to_vec_f32()
                .expect("host read");
            assert_eq!(truncated, sliced, "length {length}");
        }
    }

    /// A folded carry is narrowed to its final position wherever it lives, and
    /// a carry that already holds one position is left alone.
    #[test]
    fn a_carry_reduces_to_its_final_position() {
        // [batch=1, positions=3, features=2]
        let seeded =
            Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 3, 2]).expect("value");
        let reduced = last_position_of_carry(&HostTensorOps, &seeded).expect("carry narrows");
        assert_eq!(reduced.shape(), &[1, 1, 2]);
        assert_eq!(reduced.to_vec_f32().expect("host read"), vec![4.0, 5.0]);

        // Rank 2 has no separate position axis: it is already one position.
        let flat = Value::from_slice_f32(&[7.0, 8.0], &[1, 2]).expect("value");
        let kept = last_position_of_carry(&HostTensorOps, &flat).expect("carry passes through");
        assert_eq!(kept.shape(), &[1, 2]);
        assert_eq!(kept.to_vec_f32().expect("host read"), vec![7.0, 8.0]);
    }
}
