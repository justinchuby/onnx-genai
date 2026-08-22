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
    MtpEmbedder, MtpLmHead, MtpProposer, NgramProposer, SpeculativeProposer,
    SpeculativeProposerContext, SpeculativeStats, argmax,
};
use anyhow::Context;
use onnx_genai_ort::Tokenizer;

pub(crate) fn verification_width(
    draft_width: usize,
    remaining_tokens: usize,
    remaining_context: usize,
) -> usize {
    draft_width.min(remaining_tokens).min(remaining_context)
}

/// Outer speculative token loop bound to a single [`NativeDecodeSession`].
///
/// Peer to the plain [`NativeDecodeSession::generate_with_callback`] loop; it
/// owns the token loop itself because it cannot use `run_decode_loop`, whose
/// backend contract is one token per step.
pub(crate) struct NativeSpeculativeDriver<'a> {
    session: &'a mut NativeDecodeSession,
    proposer: NativeProposer,
    /// Maximum draft width proposed per verify pass.
    draft_width: usize,
}

enum NativeProposer {
    PromptLookup(NgramProposer),
    /// MTP self-speculation: the generic [`MtpProposer`] (ORT MTP head +
    /// shared target embedding / LM head) proposes `guaranteed + K` tokens from
    /// the native target's last hidden state. `hidden_size` is the expected
    /// seed width (`hc_mult * hidden`) used to fail fast on a metadata/model
    /// mismatch before the proposer runs.
    Mtp {
        proposer: Box<MtpProposer<'static, MtpEmbedder, MtpLmHead>>,
        hidden_size: usize,
    },
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
            proposer: NativeProposer::PromptLookup(NgramProposer::new(ngram, max_tokens)?),
            draft_width: draft_width.max(1),
        })
    }

    /// Build an MTP self-speculative driver over `session`.
    ///
    /// The generic [`MtpProposer`] owns the ORT MTP-head session plus the shared
    /// target embedding / LM head; it is seeded each step from the native
    /// target's declared hidden output (`last_hidden()`), which the Gap-2
    /// executor fix makes available on the CUDA path.
    pub(crate) fn new_mtp(
        session: &'a mut NativeDecodeSession,
        proposer: MtpProposer<'static, MtpEmbedder, MtpLmHead>,
        hidden_size: usize,
        draft_width: usize,
    ) -> anyhow::Result<Self> {
        if hidden_size == 0 {
            anyhow::bail!("native MTP proposer requires a positive target hidden width");
        }
        Ok(Self {
            session,
            proposer: NativeProposer::Mtp {
                proposer: Box::new(proposer),
                hidden_size,
            },
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

        // Option-c native MTP verify capture: install the fixed-M (=k+1) padded
        // verify bindings + Verify graph slot ONCE per generation so the M=K
        // verify forward is captured and replayed instead of recaptured every
        // step (the pre-#1650 replays=0 pin). Idempotent and a no-op unless this
        // is an MTP proposer over a graph-enabled hybrid recurrent CUDA session
        // (`configure_verify_capture` self-guards on all of those). Greedy /
        // prompt-lookup / pure-attention / CPU paths are entirely unaffected.
        if matches!(self.proposer, NativeProposer::Mtp { .. }) {
            self.session.configure_verify_capture(self.draft_width)?;
        }

        // Env-gated per-phase timing to ground the base-decode-fusion analysis.
        // ONNX_GENAI_PROFILE_SPEC_PHASES=1 prints cumulative ms for the base
        // decode / propose / verify / commit phases at the end of generation.
        // Inert (no Instant calls) unless the env var is set.
        let phase_profile = std::env::var("ONNX_GENAI_PROFILE_SPEC_PHASES").is_ok();
        let mut t_base = 0f64;
        let mut t_propose = 0f64;
        let mut t_verify = 0f64;
        let mut t_commit = 0f64;
        let mut n_base = 0usize;
        let mut n_verify = 0usize;

        let prompt_len = prompt_tokens.len();
        let mut state = DecodeLoopState::new(0, options.seed, options.top_logprobs);
        // Committed tokens not yet folded into the device KV cache. Mirrors
        // `NativeLoopAdapter::pending_tokens`: the plain loop also trails the KV
        // by one committed token.
        let mut pending: Vec<TokenId> = prompt_tokens.to_vec();

        loop {
            if state.generated_tokens.len() >= options.max_new_tokens {
                if phase_profile {
                    eprintln!(
                        "spec_phases: base={t_base:.1}ms/{n_base} propose={t_propose:.1}ms verify={t_verify:.1}ms/{n_verify} commit={t_commit:.1}ms | per_base={:.3}ms per_verify={:.3}ms",
                        t_base / n_base.max(1) as f64,
                        t_verify / n_verify.max(1) as f64,
                    );
                }
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
            let t0 = phase_profile.then(std::time::Instant::now);
            let base_logits = self
                .session
                .decode(&pending, past)?
                .pop()
                .context("native speculative decode produced no base logits")?;
            if let Some(t0) = t0 {
                t_base += t0.elapsed().as_secs_f64() * 1e3;
                n_base += 1;
            }
            pending.clear();
            let base = self.session.current_len();
            debug_assert_eq!(base, context_len);

            let remaining_tokens = options.max_new_tokens - state.generated_tokens.len();
            let remaining_context = options
                .max_context
                .map(|limit| limit.saturating_sub(context_len))
                .unwrap_or(remaining_tokens);
            let width = verification_width(self.draft_width, remaining_tokens, remaining_context);
            debug_assert!(width > 0);

            let context_tokens: Vec<TokenId> = prompt_tokens
                .iter()
                .copied()
                .chain(state.generated_tokens.iter().copied())
                .collect();
            let tp = phase_profile.then(std::time::Instant::now);
            let mut draft = match &mut self.proposer {
                NativeProposer::PromptLookup(proposer) => {
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
                    };
                    proposer.propose(&proposer_context)?.tokens
                }
                NativeProposer::Mtp {
                    proposer,
                    hidden_size,
                } => {
                    let target_hidden = self.session.last_hidden().with_context(|| {
                        "native MTP proposer requires the target decoder's declared hidden output; the target forward produced no hidden state"
                    })?;
                    if target_hidden.len() != *hidden_size {
                        anyhow::bail!(
                            "native target hidden output has width {}, but MTP metadata declares hc_mult * hidden = {}; fix speculative.target_hidden_size / hc_mult",
                            target_hidden.len(),
                            hidden_size
                        );
                    }
                    let guaranteed = TokenId::try_from(
                        argmax(&base_logits).context("native target logits were empty")?,
                    )
                    .context("native target token id exceeds u32 range")?;
                    let proposer_context = SpeculativeProposerContext {
                        width,
                        context_tokens: &context_tokens,
                        generated_tokens: &state.generated_tokens,
                        generated_text: &state.generated_text,
                        first_step: state.step,
                        options,
                        chain,
                        target_hidden: Some(target_hidden),
                        target_hidden_layers: None,
                        guaranteed_token: Some(guaranteed),
                    };
                    proposer.propose(&proposer_context)?.tokens
                }
            };
            draft.truncate(width);
            if let Some(tp) = tp {
                t_propose += tp.elapsed().as_secs_f64() * 1e3;
            }

            if draft.is_empty() {
                // No proposal: fall back to a single plain greedy step. Worst case
                // is "no regression", never a slowdown (design §10).
                let token = sample_greedy(&base_logits);
                if let Some(reason) = commit_selected_token(
                    &mut state,
                    prompt_tokens,
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

            // A hybrid decoder's Gated-DeltaNet recurrent (SSM) + conv1d state is
            // a destructive rolling cache with no per-step history to prefix-slice,
            // so `rewind` alone cannot roll it back after a rejected draft. Snapshot
            // it at the committed boundary (`base`) BEFORE the verify window
            // destructively advances it by K, so the accept path can rebuild the
            // committed state from exactly the accepted prefix. Inert (returns
            // `None`) for every pure-attention decoder — those keep the plain
            // `rewind`.
            let recurrent_snapshot = if self.session.has_recurrent_state() {
                Some(self.session.snapshot_recurrent_state()?)
            } else {
                None
            };

            // Eager M=K verify pass: one target row per draft position (predicts
            // the token AFTER each draft token). current_len advances to base + K.
            let tv = phase_profile.then(std::time::Instant::now);
            let rows = self.session.decode_verify(&draft, base)?;
            if let Some(tv) = tv {
                t_verify += tv.elapsed().as_secs_f64() * 1e3;
                n_verify += 1;
            }

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
            // token trails in `pending` (fed on the next step). For a hybrid
            // recurrent decoder the KV prefix-slice alone would leave the
            // destructive GDN/conv state stranded at `base + K`; commit it to
            // exactly the accepted prefix instead (snapshot restore + accepted-
            // token re-advance), which also performs the KV rewind. See #1598.
            //
            // Approach-B fast path (full accept): when EVERY draft token is
            // accepted, the eager verify forward already advanced BOTH the KV
            // (`base + K`) and the destructive GDN/conv recurrent state by exactly
            // those `K == accepted` tokens, and the committed length equals the
            // current length — so there is nothing to roll back. Skip the KV
            // rewind, the snapshot restore, AND the accepted-token re-advance
            // entirely: the verify's own post-state IS the committed state. This
            // removes the redundant re-advance target forwards on the majority of
            // steps (every multi-token accept). The snapshot is still taken before
            // the verify because acceptance is only known after it; it is simply
            // unused on this path. Only a PARTIAL accept (`accepted < draft.len()`)
            // needs the snapshot→restore→re-advance rebuild to a shorter prefix.
            let tc = phase_profile.then(std::time::Instant::now);
            if accepted == draft.len() {
                debug_assert_eq!(
                    self.session.current_len(),
                    base + accepted,
                    "full-accept commit: verify must leave KV + state at base + accepted"
                );
            } else {
                match recurrent_snapshot.as_ref() {
                    Some(snapshot) => self.session.commit_recurrent_state_to_accepted(
                        snapshot,
                        base,
                        &draft[..accepted],
                    )?,
                    None => self.session.rewind(base + accepted)?,
                }
            }
            if let Some(tc) = tc {
                t_commit += tc.elapsed().as_secs_f64() * 1e3;
            }

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
                    prompt_tokens,
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
