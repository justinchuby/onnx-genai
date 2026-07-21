//! Native speculative-decoding driver (WP2).
//!
//! This is the outer token loop used ONLY when a native-backend request opts
//! into an *implemented* speculative mode (prompt-lookup / n-gram, greedy). The
//! plain M=1 captured-graph greedy path
//! ([`NativeDecodeSession::generate_with_callback`] → `run_decode_loop` →
//! `NativeLoopAdapter`) is never reached from here, so the 762 tok/s
//! non-regression guarantee holds structurally: speculation-off control never
//! enters this file.
//!
//! Per outer step (design §3):
//!   1. `past = session.current_len()`; fold the trailing committed token(s) into
//!      the KV via `decode` and read `base_logits` (the target's next-token
//!      distribution for the first uncommitted position).
//!   2. Propose `K` tokens host-side with the [`NgramProposer`]. An empty
//!      proposal early-exits to a single plain greedy step (worst case: no
//!      regression, never a slowdown).
//!   3. `rows = session.decode_verify(&draft, base)` — one target row per draft
//!      position (eager M=K, [K, vocab] host logits).
//!   4. Accept via HOST argmax: the longest draft prefix whose tokens match the
//!      target's greedy pick, plus the free bonus token at the first mismatch.
//!   5. `session.rewind(base + accepted)` — the accepted draft columns stay
//!      resident, unaccepted columns are dropped, and the bonus token trails in
//!      `pending` (fed on the next step), exactly like the plain loop trails the
//!      KV by one committed token.
//!   6. Commit the accepted tokens and the bonus through the shared
//!      [`commit_selected_token`], reusing the plain loop's EOS / stop-sequence /
//!      `max_new_tokens` / `max_context` / streaming semantics.

use crate::config::{FinishReason, GenerateOptions, GenerateResult, GenerateTokenCallback};
use crate::decode_loop::{
    DecodeLoopState, commit_selected_token, finish_result, reached_context_limit,
};
use crate::logits::{ProcessorChain, TokenId};
use crate::native_decode::NativeDecodeSession;
use crate::processors::ensure_constrained_finish;
use crate::sampling::sample_greedy;
use crate::speculative::{
    NgramProposer, SpeculativeProposer, SpeculativeProposerContext, SpeculativeStats,
};
use anyhow::Context;
use onnx_genai_ort::Tokenizer;

/// Outer speculative token loop bound to a single [`NativeDecodeSession`].
///
/// Peer to the plain [`NativeDecodeSession::generate_with_callback`] loop; it
/// owns the token loop itself because it cannot use `run_decode_loop`, whose
/// backend contract is one token per step.
pub(crate) struct NativeSpeculativeDriver<'a> {
    session: &'a mut NativeDecodeSession,
    proposer: NgramProposer,
    /// Maximum draft width proposed per verify pass.
    draft_width: usize,
}

impl<'a> NativeSpeculativeDriver<'a> {
    /// Build a prompt-lookup driver over `session`.
    pub(crate) fn new_prompt_lookup(
        session: &'a mut NativeDecodeSession,
        ngram: usize,
        max_tokens: usize,
        draft_width: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            session,
            proposer: NgramProposer::new(ngram, max_tokens)?,
            draft_width: draft_width.max(1),
        })
    }

    /// Drive greedy speculative generation, streaming committed tokens to
    /// `callback` and accumulating verification diagnostics into `stats`.
    ///
    /// The caller guarantees a greedy request with no processor chain and no
    /// logprobs (see `native_speculation_plan` in `engine.rs`); that is the only
    /// regime in which host-argmax acceptance reproduces greedy selection.
    pub(crate) fn generate(
        &mut self,
        prompt_tokens: &[TokenId],
        options: &GenerateOptions,
        chain: &ProcessorChain,
        tokenizer: &Tokenizer,
        stats: &mut SpeculativeStats,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        if prompt_tokens.is_empty() {
            anyhow::bail!("native speculative generation requires at least one prompt token");
        }
        self.session.reset()?;

        let prompt_len = prompt_tokens.len();
        let mut state = DecodeLoopState::new(0, options.seed, options.top_logprobs);
        // Committed tokens not yet folded into the device KV cache. Mirrors
        // `NativeLoopAdapter::pending_tokens`: the plain loop also trails the KV
        // by one committed token.
        let mut pending: Vec<TokenId> = prompt_tokens.to_vec();

        loop {
            if state.generated_tokens.len() >= options.max_new_tokens {
                ensure_constrained_finish(options, &state.generated_text, FinishReason::MaxTokens)?;
                return finish_result(
                    tokenizer,
                    &state.generated_tokens,
                    FinishReason::MaxTokens,
                    0,
                    state.logprobs.as_deref(),
                );
            }
            let context_len = prompt_len + state.generated_tokens.len();
            if reached_context_limit(context_len, options.max_context) {
                ensure_constrained_finish(options, &state.generated_text, FinishReason::Length)?;
                return finish_result(
                    tokenizer,
                    &state.generated_tokens,
                    FinishReason::Length,
                    0,
                    state.logprobs.as_deref(),
                );
            }

            // Fold the trailing committed token(s) into the KV and read the
            // target's next-token distribution for the first uncommitted position.
            let past = self.session.current_len();
            let base_logits = self
                .session
                .decode(&pending, past)?
                .pop()
                .context("native speculative decode produced no base logits")?;
            pending.clear();
            let base = self.session.current_len();
            debug_assert_eq!(base, context_len);

            let remaining_tokens = options.max_new_tokens - state.generated_tokens.len();
            let remaining_context = options
                .max_context
                .map(|limit| limit.saturating_sub(context_len))
                .unwrap_or(remaining_tokens);
            let width = self
                .draft_width
                .min(remaining_tokens)
                .min(remaining_context)
                .max(1);

            let context_tokens: Vec<TokenId> = prompt_tokens
                .iter()
                .copied()
                .chain(state.generated_tokens.iter().copied())
                .collect();
            let proposer_context = SpeculativeProposerContext {
                width,
                context_tokens: &context_tokens,
                generated_tokens: &state.generated_tokens,
                generated_text: &state.generated_text,
                first_step: state.step,
                options,
                chain,
                target_hidden: None,
                target_hidden_layers: None,
                guaranteed_token: None,
                shared_kv_slices: None,
            };
            let mut draft = self.proposer.propose(&proposer_context)?.tokens;
            draft.truncate(width);

            if draft.is_empty() {
                // No proposal: fall back to a single plain greedy step. Worst case
                // is "no regression", never a slowdown (design §10).
                let token = sample_greedy(&base_logits);
                if let Some(reason) = commit_selected_token(
                    &mut state,
                    prompt_tokens.to_vec(),
                    token,
                    options,
                    chain,
                    tokenizer,
                    callback.as_deref_mut(),
                )? {
                    return finish_result(
                        tokenizer,
                        &state.generated_tokens,
                        reason,
                        0,
                        state.logprobs.as_deref(),
                    );
                }
                pending.push(token);
                continue;
            }

            stats.verification_steps += 1;
            stats.proposed_tokens += draft.len();

            // Eager M=K verify pass: one target row per draft position (predicts
            // the token AFTER each draft token). current_len advances to base + K.
            let rows = self.session.decode_verify(&draft, base)?;

            // ==== WP3 device-accept seam ====
            // Host argmax over the [K+1, vocab] rows. `target_tokens[idx]` is the
            // target's greedy token for output position `base + idx`:
            //   idx == 0 -> base_logits (committed prefix -> next token)
            //   idx  > 0 -> rows[idx - 1] (draft[idx-1] -> next token)
            // WP3 replaces this block with a single device `argmax_rows` launch
            // over the [K+1, vocab] device logits, returning these K+1 ids without
            // copying host logits. The accept / rewind / commit logic below is
            // unchanged and does not need to know which side produced the ids.
            let mut target_tokens = Vec::with_capacity(rows.len() + 1);
            target_tokens.push(sample_greedy(&base_logits));
            for row in &rows {
                target_tokens.push(sample_greedy(row));
            }

            let mut accepted = 0usize;
            while accepted < draft.len() && target_tokens[accepted] == draft[accepted] {
                accepted += 1;
            }
            // The free bonus token: the target's own pick at the first mismatch
            // (or, when every draft token is accepted, the extra token verify
            // yields at position base + K).
            let bonus = target_tokens[accepted];

            stats.accepted_tokens += accepted;
            if accepted >= 2 {
                stats.multi_token_accepts += 1;
            }

            // Roll the device KV back to the committed length: accepted draft
            // columns stay resident, unaccepted columns are dropped, and the bonus
            // token trails in `pending` (fed on the next step).
            self.session.rewind(base + accepted)?;

            // Commit accepted draft tokens followed by the bonus, honoring the
            // same per-token `max_new_tokens` / context-limit / EOS / stop
            // semantics as the plain loop. A mid-run stop returns immediately and
            // never emits past the stopping token.
            let mut commit_iter = draft[..accepted]
                .iter()
                .copied()
                .chain(std::iter::once(bonus))
                .enumerate();
            for (idx, token) in commit_iter.by_ref() {
                let is_bonus = idx == accepted;
                if state.generated_tokens.len() >= options.max_new_tokens {
                    // Token budget reached mid-run: stop here. The outer loop's
                    // top-of-iteration check emits `MaxTokens`; `pending` is empty
                    // so no zero-length decode occurs.
                    break;
                }
                if is_bonus {
                    // The accepted draft width was pre-capped by `remaining_context`,
                    // so only the bonus can reach the context limit.
                    let context_now = prompt_len + state.generated_tokens.len();
                    if reached_context_limit(context_now, options.max_context) {
                        break;
                    }
                }
                if let Some(reason) = commit_selected_token(
                    &mut state,
                    prompt_tokens.to_vec(),
                    token,
                    options,
                    chain,
                    tokenizer,
                    callback.as_deref_mut(),
                )? {
                    return finish_result(
                        tokenizer,
                        &state.generated_tokens,
                        reason,
                        0,
                        state.logprobs.as_deref(),
                    );
                }
                if is_bonus {
                    pending.push(token);
                }
            }
        }
    }
}
