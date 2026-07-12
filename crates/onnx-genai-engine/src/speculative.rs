//! Speculative decoding engine.

//! Phase 3 currently wires the draft-model/greedy acceptance path directly into
//! [`crate::engine::Engine`] so the public `generate` API remains stable. When
//! `EngineConfig::draft_model` is set, greedy requests propose `K`
//! autoregressive draft tokens, verify them with one target pass, accept the
//! longest target-greedy prefix, and take the target token at the first
//! mismatch. Target paged KV is rewound before committing the target token, and
//! draft KV is rewound to the shared prefix so the correction/bonus token seeds
//! the next draft round. Rejected draft tokens never leak into subsequent
//! decoding state.

use crate::TokenId;
use crate::decode::{
    extract_logits_sequence, next_session_token_logits, propose_draft_tokens,
    run_decode_session_logits, run_decode_step,
};
use crate::decode_loop::{DecodeLoopState, commit_selected_token, reached_context_limit};
use crate::engine::Engine;
use crate::kv_bridge::{
    common_prefix_len, mirror_present_kv_to_pages, rewind_draft_state_to_len,
    rewind_target_state_to_len, trim_overmaterialized_target_kv,
};
use crate::logits::{ProcessorChain, ProcessorContext};
use crate::processors::{ensure_constrained_finish, select_next_token};
use crate::session::{DraftModel, DraftSession, EngineSession};
use crate::{FinishReason, GenerateOptions, GenerateResult, GenerateTokenCallback, SessionId};
use anyhow::Context;
use onnx_genai_kv::KvCacheOps;

/// Speculative acceptance rule implemented by the Phase 3 engine path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptanceRule {
    /// Accept a draft token iff it matches the target model's greedy argmax.
    Greedy,
}

/// Result of a single greedy speculative verification step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreedyStep {
    /// Number of proposed draft tokens accepted before the first mismatch.
    pub accepted_prefix_len: usize,
    /// Whether every proposed draft token was accepted.
    pub fully_accepted: bool,
}

/// Inputs a proposer needs to draft speculative candidates for one verify pass.
pub struct SpeculativeProposerContext<'a> {
    pub width: usize,
    pub generated_tokens: &'a [TokenId],
    pub generated_text: &'a str,
    pub first_step: usize,
    pub options: &'a GenerateOptions,
    pub chain: &'a ProcessorChain,
}

/// Candidate tokens proposed for a target-model verification pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeProposal {
    pub tokens: Vec<TokenId>,
    pub positions: Option<Vec<usize>>,
    pub tree: Option<Vec<Vec<usize>>>,
}

impl SpeculativeProposal {
    pub fn linear(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens,
            positions: None,
            tree: None,
        }
    }
}

/// Outcome reported back to the proposer after verification and commit.
pub struct SpeculativeAcceptContext<'a> {
    pub accepted_prefix_len: usize,
    pub committed_tokens: &'a [TokenId],
    pub target_tokens: &'a [TokenId],
}

/// Source of speculative draft tokens.
pub trait SpeculativeProposer {
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal>;

    fn accept(&mut self, _context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
        Ok(())
    }

    fn rewind(&mut self, _target_tokens: &[TokenId]) -> anyhow::Result<()> {
        Ok(())
    }

    fn name(&self) -> &str;
}

pub(crate) struct DraftModelProposer<'a> {
    draft_model: &'a mut DraftModel,
    draft_state: &'a mut DraftSession,
}

impl<'a> DraftModelProposer<'a> {
    fn new(draft_model: &'a mut DraftModel, draft_state: &'a mut DraftSession) -> Self {
        Self {
            draft_model,
            draft_state,
        }
    }

    fn align_to_target_prefix(
        &mut self,
        target_tokens: &[TokenId],
        prefix_len: usize,
    ) -> anyhow::Result<()> {
        self.draft_state.tokens = target_tokens[..prefix_len].to_vec();
        if self.draft_state.kv_token_count > prefix_len {
            rewind_draft_state_to_len(self.draft_model, self.draft_state, prefix_len)?;
        }
        Ok(())
    }
}

impl SpeculativeProposer for DraftModelProposer<'_> {
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let tokens = propose_draft_tokens(
            self.draft_model,
            self.draft_state,
            context.width,
            context.generated_tokens,
            context.generated_text,
            context.first_step,
            context.options,
            context.chain,
        )?;
        Ok(SpeculativeProposal::linear(tokens))
    }

    fn rewind(&mut self, target_tokens: &[TokenId]) -> anyhow::Result<()> {
        let common_len = common_prefix_len(&self.draft_state.tokens, target_tokens);
        if self.draft_state.kv_token_count > common_len {
            rewind_draft_state_to_len(self.draft_model, self.draft_state, common_len)?;
        }
        self.draft_state.tokens = target_tokens.to_vec();
        Ok(())
    }

    fn name(&self) -> &str {
        "draft_model"
    }
}

impl Engine {
    pub(crate) fn should_use_speculative(&self, options: &GenerateOptions) -> bool {
        self.draft.is_some()
            // Grammar processors carry per-request parser state; draft/verify
            // would need separate parser branches for speculative candidates.
            && options.constraint.is_none()
            && (options.greedy || options.temperature == 0.0)
            && options
                .num_speculative_tokens
                .unwrap_or(self.num_speculative_tokens)
                > 0
            && self.kv_model.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_speculative_loop(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        options: &GenerateOptions,
        chain: &ProcessorChain,
        max_context: Option<usize>,
        prefix_cache_hit_len: usize,
        generated_tokens: &mut Vec<TokenId>,
        generated_text: &mut String,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        let draft_width = options
            .num_speculative_tokens
            .unwrap_or(self.num_speculative_tokens)
            .max(1);
        let mut step = 0;

        loop {
            if generated_tokens.len() >= options.max_new_tokens {
                ensure_constrained_finish(options, generated_text, FinishReason::MaxTokens)?;
                return self.finish_result(
                    generated_tokens,
                    FinishReason::MaxTokens,
                    prefix_cache_hit_len,
                );
            }
            if reached_context_limit(state.tokens.len(), max_context) {
                ensure_constrained_finish(options, generated_text, FinishReason::Length)?;
                return self.finish_result(
                    generated_tokens,
                    FinishReason::Length,
                    prefix_cache_hit_len,
                );
            }

            let remaining_tokens = options.max_new_tokens - generated_tokens.len();
            let remaining_context = max_context
                .map(|limit| limit.saturating_sub(state.tokens.len()))
                .unwrap_or(remaining_tokens);
            let width = draft_width
                .min(remaining_tokens)
                .min(remaining_context)
                .max(1);

            let base_len = state.tokens.len();
            let base_generated_len = generated_tokens.len();
            let mut base_logits = next_session_token_logits(
                &self.session,
                self.kv_model.as_ref(),
                &mut self.kv_cache,
                session_id,
                state,
            )?;

            let draft_tokens = {
                let draft_model = self
                    .draft
                    .as_mut()
                    .context("speculative decoding requested without a draft model")?;
                let draft_state = state
                    .draft
                    .as_mut()
                    .context("speculative session missing draft state")?;
                let mut proposer = DraftModelProposer::new(draft_model, draft_state);
                proposer.align_to_target_prefix(&state.tokens, base_len)?;
                proposer
                    .propose(&SpeculativeProposerContext {
                        width,
                        generated_tokens,
                        generated_text,
                        first_step: step,
                        options,
                        chain,
                    })?
                    .tokens
            };

            state.tokens.extend_from_slice(&draft_tokens);
            let verified_logits = if state.decode_state.has_runner() {
                let logits =
                    run_decode_session_logits(&mut state.decode_state, &draft_tokens, base_len)?;
                self.kv_cache
                    .append(session_id, draft_tokens.len())
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to advance KV sequence {session_id}: {}", e)
                    })?;
                state.kv_token_count += draft_tokens.len();
                logits
            } else {
                let outputs = run_decode_step(
                    &self.session,
                    &mut state.decode_state,
                    &draft_tokens,
                    base_len,
                )?;
                if state.decode_state.use_kv {
                    if let Some(kv_model) = &self.kv_model {
                        mirror_present_kv_to_pages(
                            &self.session,
                            kv_model,
                            &mut self.kv_cache,
                            session_id,
                            &outputs,
                            base_len,
                            draft_tokens.len(),
                        )?;
                    } else {
                        self.kv_cache
                            .append(session_id, draft_tokens.len())
                            .map_err(|e| {
                                anyhow::anyhow!("Failed to advance KV sequence {session_id}: {}", e)
                            })?;
                    }
                    state.kv_token_count += draft_tokens.len();
                }
                extract_logits_sequence(&self.session, outputs)?
            };

            let mut target_logits = Vec::with_capacity(draft_tokens.len() + 1);
            target_logits.push(std::mem::take(&mut base_logits));
            target_logits.extend(verified_logits);

            let mut accepted = 0;
            let mut replacement = None;
            for idx in 0..draft_tokens.len() {
                let mut context = ProcessorContext {
                    prompt_tokens: state.tokens[..base_len].to_vec(),
                    generated_tokens: generated_tokens
                        .iter()
                        .copied()
                        .chain(draft_tokens[..idx].iter().copied())
                        .collect(),
                    generated_text: self
                        .tokenizer
                        .decode(
                            &generated_tokens
                                .iter()
                                .copied()
                                .chain(draft_tokens[..idx].iter().copied())
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to detokenize speculative context: {}", e)
                        })?,
                    step: step + idx,
                };
                let target_token =
                    select_next_token(&mut target_logits[idx], &context, options, chain, 0.0);
                if target_token == draft_tokens[idx] {
                    accepted += 1;
                } else {
                    replacement = Some(target_token);
                    context.generated_tokens.push(target_token);
                    break;
                }
            }

            let mut commit_tokens = draft_tokens[..accepted].to_vec();
            let rewind_len = base_len + accepted;
            rewind_target_state_to_len(
                &self.session,
                self.kv_model.as_ref(),
                &mut self.kv_cache,
                session_id,
                state,
                rewind_len,
            )?;

            if let Some(token) = replacement {
                commit_tokens.push(token);
            } else if generated_tokens.len() + commit_tokens.len() < options.max_new_tokens
                && !reached_context_limit(base_len + commit_tokens.len(), max_context)
            {
                let mut context = ProcessorContext {
                    prompt_tokens: state.tokens[..base_len].to_vec(),
                    generated_tokens: generated_tokens
                        .iter()
                        .copied()
                        .chain(draft_tokens.iter().copied())
                        .collect(),
                    generated_text: self
                        .tokenizer
                        .decode(
                            &generated_tokens
                                .iter()
                                .copied()
                                .chain(draft_tokens.iter().copied())
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to detokenize speculative context: {}", e)
                        })?,
                    step: step + draft_tokens.len(),
                };
                let token = select_next_token(
                    target_logits
                        .last_mut()
                        .context("target verification did not produce next-token logits")?,
                    &context,
                    options,
                    chain,
                    0.0,
                );
                context.generated_tokens.push(token);
                commit_tokens.push(token);
            }

            self.notify_draft_acceptance(state, accepted, &commit_tokens)?;

            for (commit_idx, token_id) in commit_tokens.into_iter().enumerate() {
                if generated_tokens.len() >= options.max_new_tokens
                    || (commit_idx >= accepted
                        && reached_context_limit(state.tokens.len(), max_context))
                {
                    break;
                }
                if commit_idx >= accepted {
                    state.tokens.push(token_id);
                }
                self.scheduler.advance(session_id);
                let prompt_tokens = state.tokens[..base_len.min(state.tokens.len())].to_vec();
                let mut commit_state = DecodeLoopState {
                    generated_tokens: std::mem::take(generated_tokens),
                    generated_text: std::mem::take(generated_text),
                    step,
                    prefix_cache_hit_len,
                };
                let finish_reason = commit_selected_token(
                    &mut commit_state,
                    prompt_tokens,
                    token_id,
                    options,
                    chain,
                    &self.tokenizer,
                    callback.as_deref_mut(),
                )?;
                *generated_tokens = commit_state.generated_tokens;
                *generated_text = commit_state.generated_text;
                step = commit_state.step;
                if let Some(finish_reason) = finish_reason {
                    trim_overmaterialized_target_kv(
                        &self.session,
                        self.kv_model.as_ref(),
                        &mut self.kv_cache,
                        session_id,
                        state,
                    )?;
                    self.sync_draft_to_target(state)?;
                    return self.finish_result(
                        generated_tokens,
                        finish_reason,
                        prefix_cache_hit_len,
                    );
                }
            }

            self.sync_draft_to_target(state)?;

            if generated_tokens.len() == base_generated_len {
                anyhow::bail!("speculative decoding made no progress");
            }
        }
    }

    pub(crate) fn sync_draft_to_target(&mut self, state: &mut EngineSession) -> anyhow::Result<()> {
        if let (Some(draft_model), Some(draft_state)) = (&mut self.draft, &mut state.draft) {
            DraftModelProposer::new(draft_model, draft_state).rewind(&state.tokens)?;
        }
        Ok(())
    }

    fn notify_draft_acceptance(
        &mut self,
        state: &mut EngineSession,
        accepted_prefix_len: usize,
        committed_tokens: &[TokenId],
    ) -> anyhow::Result<()> {
        if let (Some(draft_model), Some(draft_state)) = (&mut self.draft, &mut state.draft) {
            DraftModelProposer::new(draft_model, draft_state).accept(
                &SpeculativeAcceptContext {
                    accepted_prefix_len,
                    committed_tokens,
                    target_tokens: &state.tokens,
                },
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubProposer {
        tokens: Vec<TokenId>,
        accepted: Option<usize>,
        rewound_to: Option<Vec<TokenId>>,
    }

    impl SpeculativeProposer for StubProposer {
        fn propose(
            &mut self,
            _context: &SpeculativeProposerContext<'_>,
        ) -> anyhow::Result<SpeculativeProposal> {
            Ok(SpeculativeProposal::linear(self.tokens.clone()))
        }

        fn accept(&mut self, context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
            self.accepted = Some(context.accepted_prefix_len);
            Ok(())
        }

        fn rewind(&mut self, target_tokens: &[TokenId]) -> anyhow::Result<()> {
            self.rewound_to = Some(target_tokens.to_vec());
            Ok(())
        }

        fn name(&self) -> &str {
            "stub"
        }
    }

    #[test]
    fn speculative_proposer_trait_supports_non_draft_sources() -> anyhow::Result<()> {
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let mut proposer = StubProposer {
            tokens: vec![3, 5],
            accepted: None,
            rewound_to: None,
        };

        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 2,
            generated_tokens: &[1],
            generated_text: "a",
            first_step: 0,
            options: &options,
            chain: &chain,
        })?;
        proposer.accept(&SpeculativeAcceptContext {
            accepted_prefix_len: 1,
            committed_tokens: &[3, 4],
            target_tokens: &[1, 3, 4],
        })?;
        proposer.rewind(&[1, 3, 4])?;

        assert_eq!(proposal.tokens, vec![3, 5]);
        assert_eq!(proposer.accepted, Some(1));
        assert_eq!(proposer.rewound_to, Some(vec![1, 3, 4]));
        Ok(())
    }
}
