//! Native speculative-decoding driver (WP2).
//!
//! This is the outer token loop used ONLY when a native-backend request opts
//! into an *implemented* speculative mode (prompt-lookup / n-gram, greedy). The
//! plain M=1 captured-graph greedy path
//! ([`NativeDecodeSession::generate_with_callback`] -> the interpreter's
//! declared generation loop ->
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
/// owns the token loop itself because it cannot use the single-token declared
/// loop, whose
/// backend contract is one token per step.
pub(crate) struct NativeSpeculativeDriver<'a> {
    session: &'a mut NativeDecodeSession,
    proposer: NativeProposer,
    /// Maximum draft width proposed per verify pass.
    draft_width: usize,
    /// Cumulative per-phase timings, accumulated across blocks and reported
    /// once when `ONNX_GENAI_PROFILE_SPEC_PHASES` is set. They live on the
    /// driver rather than on one block because the figure a reader wants is
    /// the generation's, and a block does not know it is the last.
    t_base: f64,
    t_propose: f64,
    t_verify: f64,
    t_commit: f64,
    n_base: usize,
    n_verify: usize,
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
            t_base: 0.0,
            t_propose: 0.0,
            t_verify: 0.0,
            t_commit: 0.0,
            n_base: 0,
            n_verify: 0,
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
            t_base: 0.0,
            t_propose: 0.0,
            t_verify: 0.0,
            t_commit: 0.0,
            n_base: 0,
            n_verify: 0,
        })
    }

    /// Drive greedy speculative generation, streaming committed tokens to
    /// `callback` and accumulating verification diagnostics into `stats`.
    ///
    /// The caller guarantees a greedy request with no processor chain and no
    /// logprobs (see `native_speculation_plan` in `engine.rs`); that is the only
    /// regime in which host-argmax acceptance reproduces greedy selection.
    /// Drive greedy speculative generation through the package's loop
    /// re-authored as a proposal block.
    ///
    /// The iteration comes from `runtime`: the interpreter reads the bound, the
    /// liveness predicate and the emit from the authored document, and the
    /// body's node declares `onnx-genai.speculative-block`. This driver
    /// registers an executor for that contract, so the native propose-verify
    /// step is reached through the same seam the ORT draft-model path uses --
    /// the two are different *implementations* of one declared step, not two
    /// loops.
    ///
    /// The caller guarantees a greedy request with no processor chain and no
    /// logprobs (see `native_speculation_plan` in `engine.rs`); that is the only
    /// regime in which host-argmax acceptance reproduces greedy selection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate(
        &mut self,
        prompt_tokens: &[TokenId],
        options: &GenerateOptions,
        chain: &ProcessorChain,
        tokenizer: &Tokenizer,
        runtime: &crate::pipeline::WorkflowRuntime,
        stats: &mut SpeculativeStats,
        callback: Option<&mut GenerateTokenCallback<'_>>,
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

        let mut block = NativeSpeculativeBlock {
            prompt_tokens,
            options,
            chain,
            tokenizer,
            prompt_len: prompt_tokens.len(),
            phase_profile,
            state: DecodeLoopState::new(0, options.seed, options.top_logprobs),
            // Committed tokens not yet folded into the device KV cache. Mirrors
            // `NativeLoopAdapter::pending_tokens`: the plain loop also trails the
            // KV by one committed token.
            pending: prompt_tokens.to_vec(),
            stats,
            callback,
        };
        let request = crate::pipeline::PipelineGenerateRequest::new(crate::GenerateRequest {
            prompt: crate::GeneratePrompt::TokenIds(vec![0]),
            options: options.clone(),
        });
        let mut host = NativeSpeculativeBlockHost {
            driver: self,
            block: &mut block,
            finished: None,
        };
        let mut cursor = {
            let mut node: Option<&mut dyn crate::pipeline::WorkflowNodeHost> = Some(&mut host);
            crate::pipeline::WorkflowGenerationCursor::start(
                runtime,
                request,
                crate::pipeline::generation::SPECULATIVE_BLOCK_CONTRACTS,
                &mut node,
            )?
        };
        while {
            let mut node: Option<&mut dyn crate::pipeline::WorkflowNodeHost> = Some(&mut host);
            cursor.advance(runtime, &mut node)?
        } {}
        let finished = host.finished.take();
        // Reported from the drive rather than from a block, because a block does
        // not know it is the last one: the declared loop may end on its own
        // bound, and a summary printed only by the token-budget branch would go
        // missing for exactly the runs a profiler was started for.
        if phase_profile {
            self.report_phase_profile();
        }
        crate::pipeline::generation::verify_emitted_tokens(
            runtime,
            &cursor,
            &block.state.generated_tokens,
        )?;
        match finished {
            Some(result) => Ok(result),
            None => {
                ensure_constrained_finish(
                    options,
                    &block.state.generated_text,
                    FinishReason::MaxTokens,
                )?;
                finish_result(
                    tokenizer,
                    &block.state.generated_tokens,
                    FinishReason::MaxTokens,
                    0,
                    block.state.logprobs.as_deref(),
                )
            }
        }
    }

    /// Cumulative per-phase timings for this generation.
    fn report_phase_profile(&self) {
        eprintln!(
            "spec_phases: base={:.1}ms/{} propose={:.1}ms verify={:.1}ms/{} commit={:.1}ms | \
             per_base={:.3}ms per_verify={:.3}ms",
            self.t_base,
            self.n_base,
            self.t_propose,
            self.t_verify,
            self.n_verify,
            self.t_commit,
            self.t_base / self.n_base.max(1) as f64,
            self.t_verify / self.n_verify.max(1) as f64,
        );
    }

    /// Run one propose-verify-accept block against the native session.
    ///
    /// Exactly one iteration and nothing about the loop around it. `Ok(None)`
    /// means the block committed and generation may continue; `Ok(Some(result))`
    /// means this block reached a stop.
    fn advance_block(
        &mut self,
        block: &mut NativeSpeculativeBlock<'_, '_>,
    ) -> anyhow::Result<Option<GenerateResult>> {
        let NativeSpeculativeBlock {
            prompt_tokens,
            options,
            chain,
            tokenizer,
            prompt_len,
            phase_profile,
            ..
        } = *block;
        let state = &mut block.state;
        let pending = &mut block.pending;
        let stats = &mut *block.stats;
        let mut callback = block.callback.as_deref_mut();
        if state.generated_tokens.len() >= options.max_new_tokens {
            ensure_constrained_finish(options, &state.generated_text, FinishReason::MaxTokens)?;
            return Ok(Some(finish_result(
                tokenizer,
                &state.generated_tokens,
                FinishReason::MaxTokens,
                0,
                state.logprobs.as_deref(),
            )?));
        }
        let context_len = prompt_len + state.generated_tokens.len();
        if reached_context_limit(context_len, options.max_context) {
            ensure_constrained_finish(options, &state.generated_text, FinishReason::Length)?;
            return Ok(Some(finish_result(
                tokenizer,
                &state.generated_tokens,
                FinishReason::Length,
                0,
                state.logprobs.as_deref(),
            )?));
        }

        // Fold the trailing committed token(s) into the KV and read the
        // target's next-token distribution for the first uncommitted position.
        let past = self.session.current_len();
        let t0 = phase_profile.then(std::time::Instant::now);
        let base_logits = self
            .session
            .decode(pending, past)?
            .pop()
            .context("native speculative decode produced no base logits")?;
        if let Some(t0) = t0 {
            self.t_base += t0.elapsed().as_secs_f64() * 1e3;
            self.n_base += 1;
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
            self.t_propose += tp.elapsed().as_secs_f64() * 1e3;
        }

        if draft.is_empty() {
            // No proposal: fall back to a single plain greedy step. Worst case
            // is "no regression", never a slowdown (design §10).
            let token = sample_greedy(&base_logits);
            if let Some(reason) = commit_selected_token(
                state,
                prompt_tokens,
                token,
                options,
                chain,
                tokenizer,
                callback.as_deref_mut(),
            )? {
                return Ok(Some(finish_result(
                    tokenizer,
                    &state.generated_tokens,
                    reason,
                    0,
                    state.logprobs.as_deref(),
                )?));
            }
            pending.push(token);
            return Ok(None);
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
            self.t_verify += tv.elapsed().as_secs_f64() * 1e3;
            self.n_verify += 1;
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
            self.t_commit += tc.elapsed().as_secs_f64() * 1e3;
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
                state,
                prompt_tokens,
                token,
                options,
                chain,
                tokenizer,
                callback.as_deref_mut(),
            )? {
                return Ok(Some(finish_result(
                    tokenizer,
                    &state.generated_tokens,
                    reason,
                    0,
                    state.logprobs.as_deref(),
                )?));
            }
            if is_bonus {
                pending.push(token);
            }
        }
        Ok(None)
    }
}

/// Everything one native proposal block needs, carried between iterations.
struct NativeSpeculativeBlock<'a, 'callback> {
    prompt_tokens: &'a [TokenId],
    options: &'a GenerateOptions,
    chain: &'a ProcessorChain,
    tokenizer: &'a Tokenizer,
    prompt_len: usize,
    phase_profile: bool,
    state: DecodeLoopState,
    pending: Vec<TokenId>,
    stats: &'a mut SpeculativeStats,
    callback: Option<&'a mut GenerateTokenCallback<'callback>>,
}

/// The native propose-verify executor, as the interpreter reaches it.
struct NativeSpeculativeBlockHost<'driver, 'session, 'a, 'callback> {
    driver: &'driver mut NativeSpeculativeDriver<'session>,
    block: &'driver mut NativeSpeculativeBlock<'a, 'callback>,
    finished: Option<GenerateResult>,
}

impl crate::pipeline::WorkflowNodeHost for NativeSpeculativeBlockHost<'_, '_, '_, '_> {
    fn hosted_contracts(&self) -> &'static [&'static str] {
        crate::pipeline::generation::SPECULATIVE_BLOCK_CONTRACTS
    }

    fn execute_contract_node(
        &mut self,
        mut request: crate::pipeline::WorkflowNodeRequest<'_>,
    ) -> anyhow::Result<bool> {
        if request.contract != onnx_genai_metadata::decoder_workflow::SPECULATIVE_BLOCK_CONTRACT {
            return Ok(false);
        }
        // Defensive, not load-bearing: the loop reads its predicate before the
        // body and writes the carry after it, and the liveness read is never
        // skipped while a host implements a node in this body, so a stopped
        // block is not re-entered. Publishing the stop again rather than
        // decoding past it keeps that a local property instead of a claim about
        // the interpreter.
        if self.finished.is_some() {
            return publish_block(&mut request, &[], false, 0);
        }
        let before = self.block.state.generated_tokens.len();
        let finished = self.driver.advance_block(self.block)?;
        let committed = self.block.state.generated_tokens[before..].to_vec();
        let context = self.block.prompt_len + self.block.state.generated_tokens.len();
        self.finished = finished;
        publish_block(&mut request, &committed, self.finished.is_none(), context)
    }
}

/// Define the SSA values a block node declares.
///
/// `token` carries the accepted block and `accepted_len` says how much of it is
/// real, so one iteration publishes a variable number of tokens through the
/// same emit a single-token body uses. A rejected suffix never appears: it was
/// rolled back before the commit that produced these tokens.
fn publish_block(
    request: &mut crate::pipeline::WorkflowNodeRequest<'_>,
    committed: &[TokenId],
    active: bool,
    context_len: usize,
) -> anyhow::Result<bool> {
    let width = committed.len().max(1) as i64;
    for (port, value) in request.outputs.iter() {
        let tensor = match port.as_str() {
            "token" => {
                let mut tokens = committed
                    .iter()
                    .map(|id| i64::from(*id))
                    .collect::<Vec<_>>();
                tokens.resize(width as usize, 0);
                onnx_genai_ort::Value::from_slice_i64(&tokens, &[1, width])?
            }
            "active" => block_flag(active)?,
            "done" => block_flag(!active)?,
            "accepted_len" => {
                onnx_genai_ort::Value::from_slice_i64(&[committed.len() as i64], &[1])?
            }
            "cache_lengths" | "lengths" => onnx_genai_ort::Value::from_slice_i64(
                &[i64::try_from(context_len).context("cache length exceeds int64")?],
                &[1],
            )?,
            // State ports the block also declares are the session's buffers.
            _ => continue,
        };
        request.values.insert(value.clone(), tensor);
    }
    Ok(true)
}

fn block_flag(value: bool) -> anyhow::Result<onnx_genai_ort::Value> {
    onnx_genai_ort::Value::from_raw_bytes(
        vec![u8::from(value)],
        &[1],
        onnx_genai_ort::DataType::Bool,
    )
    .map_err(Into::into)
}
