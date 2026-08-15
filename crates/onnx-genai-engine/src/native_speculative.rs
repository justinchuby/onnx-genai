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
use crate::native_decode::{NativeDecodeSession, NativeProposerSession};
use crate::processors::ensure_constrained_finish;
use crate::sampling::sample_greedy;
use crate::speculative::{
    LinearEmbedder, NgramProposer, SpeculativeProposer, SpeculativeProposerContext,
    SpeculativeStats, TokenEmbedder, argmax,
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

/// Outcome of greedy speculative acceptance over one verify pass.
pub(crate) struct AcceptOutcome {
    /// Number of leading draft tokens accepted (each equals the target argmax,
    /// or a near-tie co-winner when tie-tolerant acceptance is enabled).
    pub(crate) accepted: usize,
    /// The free bonus token: the target's own argmax at the first mismatch (or
    /// the trailing verify row when every draft token is accepted).
    pub(crate) bonus: TokenId,
    /// Draft positions rejected where the draft token was a near-tie co-winner
    /// of the row argmax (its logit within `tie_eps` of the row maximum).
    pub(crate) near_tie_rejections: usize,
}

/// Numerical near-tie guard for greedy speculative acceptance.
///
/// `eps` is the absolute logit margin below the row maximum within which a
/// draft token is treated as a *co-winner* of the argmax. `tolerant` decides
/// what happens on a near-tie mismatch:
///   - `tolerant == false` (default): near-ties are only *counted* (diagnostic);
///     the committed token stays the strict argmax, so the stream is
///     byte-identical to plain greedy. This is the safe default and the regime
///     the `spec == greedy` correctness gate asserts.
///   - `tolerant == true`: a near-tie draft token is *accepted* (committed as-is),
///     trading exact byte-identity — only at genuine numerical ties, where
///     greedy is itself ill-defined and nondeterministic run-to-run — for higher
///     acceptance on models with tie-prone logits (e.g. block-32 qwen).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TieGuard {
    pub(crate) eps: f32,
    pub(crate) tolerant: bool,
}

impl TieGuard {
    /// Strict argmax acceptance (byte-identical to plain greedy). `eps == 0`
    /// disables the near-tie probe entirely.
    pub(crate) const STRICT: TieGuard = TieGuard {
        eps: 0.0,
        tolerant: false,
    };

    /// Resolve the guard from the environment (`ONNX_GENAI_SPEC_TIE_EPS`,
    /// `ONNX_GENAI_SPEC_TIE_TOLERANT`). Absent/zero eps ⇒ strict, so the default
    /// decode path is unchanged and byte-identical.
    pub(crate) fn from_env() -> TieGuard {
        let eps = std::env::var("ONNX_GENAI_SPEC_TIE_EPS")
            .ok()
            .and_then(|value| value.trim().parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(0.0);
        if eps == 0.0 {
            // No margin configured ⇒ strict argmax, byte-identical to plain greedy.
            return TieGuard::STRICT;
        }
        let tolerant = matches!(
            std::env::var("ONNX_GENAI_SPEC_TIE_TOLERANT")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("yes")
        );
        TieGuard { eps, tolerant }
    }
}

/// Whether the fused CUDA-graph-captured verify path is enabled
/// (`ONNX_GENAI_SPEC_CAPTURED_VERIFY=1`). Off by default so the tested eager
/// verify path stays the default; when on, the driver still degrades to eager on
/// non-CUDA or non-capturable sessions, so this never affects correctness.
fn captured_verify_from_env() -> bool {
    matches!(
        std::env::var("ONNX_GENAI_SPEC_CAPTURED_VERIFY")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Resolve the row-0 near-tie margin for the captured-verify base-token contract
/// guard (`ONNX_GENAI_SPEC_ROW0_TIE_EPS`).
///
/// Row 0 (the base next-token distribution) is produced by the M=W verify
/// forward, whereas plain greedy — the byte-identity reference — selects the
/// base token from the fp32-accumulate M=1 GEMV. When the M=W verify uses the
/// Marlin fp16-accumulate (or portable tiled) GEMM their argmax can flip, so
/// this guard falls the base token back to a fresh M=1 GEMV decode whenever row
/// 0 is a near-tie within `eps`. (Leon: this is unnecessary under the opt-in
/// per-row M=1 GEMV verify — `ONNX_GENAI_SPEC_PERROW_VERIFY=1` — which makes row
/// 0 byte-identical to the M=1 GEMV by construction, but it is kept for the
/// default Marlin/tiled verify path.)
fn row0_tie_eps_from_env() -> f32 {
    const DEFAULT_ROW0_TIE_EPS: f32 = 1.0;
    std::env::var("ONNX_GENAI_SPEC_ROW0_TIE_EPS")
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(DEFAULT_ROW0_TIE_EPS)
}

/// Whether the base row's greedy choice is a numerical near-tie: the gap between
/// the largest and second-largest logit is within `eps`. Returns `false` for
/// `eps == 0` (guard disabled) or degenerate rows (<2 logits).
fn row0_is_near_tie(logits: &[f32], eps: f32) -> bool {
    if eps <= 0.0 || logits.len() < 2 {
        return false;
    }
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for &v in logits {
        if v > top1 {
            top2 = top1;
            top1 = v;
        } else if v > top2 {
            top2 = v;
        }
    }
    top1 - top2 <= eps
}

/// Adaptive hit-density gate for the fused captured-verify path.
///
/// The prompt-lookup proposal is a cheap CPU-side n-gram search, but the width-W
/// captured verify forward only pays off when the warmed device graph is
/// *replayed* across consecutive steps. An isolated lookup hit surrounded by
/// misses re-warms the graph (a fresh capture, not a replay) and commits only
/// B tokens for ~W× the M=1 cost — a net loss. That is the generic-prose
/// regression: sparse hits never amortize their warmup. The gate only engages
/// the width-W verify once recent would-hit density predicts the graph will
/// replay; below the threshold the driver runs a plain M=1 decode step (never
/// worse than baseline) and re-engages the moment hits cluster.
///
/// Both branches — captured verify and plain M=1 decode — are independently
/// byte-identical to plain greedy, so gating between them cannot change output;
/// it only trades speculative throughput for a guaranteed non-regression floor.
struct HitDensityGate {
    /// Trailing would-hit bits (bit 0 = most recent), masked to `window` bits.
    bits: u32,
    /// Number of trailing steps to consider (1..=32).
    window: u32,
    /// Minimum would-hits within the window required to engage the width-W path.
    min_hits: u32,
    /// When false the gate is disabled: engage on every hit (A/B baseline).
    enabled: bool,
}

impl HitDensityGate {
    /// Resolve the gate from the environment. Enabled by default so the fused
    /// captured path never regresses below plain decode on low-acceptance
    /// prompts. `ONNX_GENAI_SPEC_GATE=0` restores the always-engage behavior
    /// (for A/B), `ONNX_GENAI_SPEC_GATE_WINDOW` / `_MIN_HITS` tune the threshold.
    fn from_env() -> HitDensityGate {
        let enabled = !matches!(
            std::env::var("ONNX_GENAI_SPEC_GATE").ok().as_deref(),
            Some("0") | Some("false") | Some("no")
        );
        let window = std::env::var("ONNX_GENAI_SPEC_GATE_WINDOW")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|value| (1..=32).contains(value))
            .unwrap_or(16);
        let min_hits = std::env::var("ONNX_GENAI_SPEC_GATE_MIN_HITS")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|value| *value >= 1)
            .unwrap_or(window.div_ceil(2))
            .min(window);
        HitDensityGate {
            bits: 0,
            window,
            min_hits,
            enabled,
        }
    }

    #[inline]
    fn window_mask(&self) -> u32 {
        if self.window >= 32 {
            u32::MAX
        } else {
            (1u32 << self.window) - 1
        }
    }

    /// Record whether the current step produced a non-empty draft (a would-hit).
    fn record(&mut self, would_hit: bool) {
        self.bits = ((self.bits << 1) | u32::from(would_hit)) & self.window_mask();
    }

    /// Would-hits within the trailing window (after `record`).
    fn density(&self) -> u32 {
        self.bits.count_ones()
    }

    /// Whether to engage the width-W captured verify this step. `would_hit` must
    /// be the current step's proposal outcome, already passed to `record`.
    fn should_engage(&self, would_hit: bool) -> bool {
        if !would_hit {
            return false;
        }
        if !self.enabled {
            return true;
        }
        self.density() >= self.min_hits
    }
}

/// Greedy speculative acceptance with an optional numerical near-tie guard.
///
/// `base_logits` is the target distribution for the first uncommitted position
/// (`base + 0`); `verify_rows[i]` is the distribution for `base + i + 1`. The
/// longest draft prefix whose tokens equal the target's strict argmax is
/// accepted, plus one free bonus token (the target's argmax at the stopping
/// row). The tie guard only affects the *mismatch* boundary and never changes
/// the bonus, so with [`TieGuard::STRICT`] this reproduces plain greedy exactly.
pub(crate) fn greedy_accept(
    base_logits: &[f32],
    verify_rows: &[Vec<f32>],
    draft: &[TokenId],
    guard: TieGuard,
) -> AcceptOutcome {
    let row = |idx: usize| -> &[f32] {
        if idx == 0 {
            base_logits
        } else {
            &verify_rows[idx - 1]
        }
    };
    let mut accepted = 0usize;
    let mut near_tie_rejections = 0usize;
    while accepted < draft.len() {
        let logits = row(accepted);
        if sample_greedy(logits) == draft[accepted] {
            accepted += 1;
            continue;
        }
        // Strict mismatch. Probe whether the draft token is a near-tie co-winner
        // (its logit within `eps` of the row maximum) before rejecting.
        let near_tie = guard.eps > 0.0 && {
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            logits
                .get(draft[accepted] as usize)
                .map(|&draft_logit| {
                    draft_logit.is_finite() && (max_logit - draft_logit) <= guard.eps
                })
                .unwrap_or(false)
        };
        if near_tie {
            near_tie_rejections += 1;
            if guard.tolerant {
                accepted += 1;
                continue;
            }
        }
        break;
    }
    let bonus = sample_greedy(row(accepted));
    AcceptOutcome {
        accepted,
        bonus,
        near_tie_rejections,
    }
}

/// Outer speculative token loop bound to a single [`NativeDecodeSession`].
///
/// Peer to the plain [`NativeDecodeSession::generate_with_callback`] loop; it
/// owns the token loop itself because it cannot use `run_decode_loop`, whose
/// backend contract is one token per step.
pub(crate) struct NativeSpeculativeDriver<'a> {
    session: &'a mut NativeDecodeSession,
    proposer: NativeProposer<'a>,
    /// Maximum draft width proposed per verify pass.
    draft_width: usize,
    /// Numerical near-tie acceptance guard (strict by default; byte-identical).
    tie_guard: TieGuard,
    /// WP4: run the M=width verify forward through the CUDA-graph-captured fused
    /// path (base + draft in one replayed graph) instead of the eager verify.
    /// Env-gated (`ONNX_GENAI_SPEC_CAPTURED_VERIFY=1`) and prompt-lookup only;
    /// falls back to eager transparently on non-CUDA / non-capturable sessions.
    captured_verify: bool,
    /// Adaptive gate that only engages the width-W captured verify when recent
    /// would-hit density predicts the warmed graph will replay, so sparse-hit
    /// prompts degrade to plain decode instead of paying repeated graph warmup.
    hit_gate: HitDensityGate,
    /// Row-0 (base-token) near-tie margin for the captured-verify contract guard.
    /// The fused path's row 0 comes from the Marlin M=W kernel; plain greedy uses
    /// the M=1 GEMV. When the base row's top-1/top-2 margin is within this eps we
    /// recompute the base token from the M=1 GEMV so the committed base token is
    /// byte-identical to plain greedy (Chew's binding contract). `0` disables it
    /// (A/B only).
    row0_tie_eps: f32,
}

enum NativeProposer<'a> {
    PromptLookup(NgramProposer),
    SharedKv {
        session: &'a mut NativeProposerSession,
        embedder: &'a LinearEmbedder,
        groups: &'a [onnx_genai_metadata::SharedKvGroup],
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
            tie_guard: TieGuard::from_env(),
            captured_verify: captured_verify_from_env(),
            hit_gate: HitDensityGate::from_env(),
            row0_tie_eps: row0_tie_eps_from_env(),
        })
    }

    pub(crate) fn new_shared_kv(
        session: &'a mut NativeDecodeSession,
        proposer_session: &'a mut NativeProposerSession,
        embedder: &'a LinearEmbedder,
        groups: &'a [onnx_genai_metadata::SharedKvGroup],
        hidden_size: usize,
        draft_width: usize,
    ) -> anyhow::Result<Self> {
        if hidden_size == 0 || embedder.hidden_size() != hidden_size {
            anyhow::bail!(
                "native shared-KV proposer hidden size {hidden_size} does not match embedding width {}",
                embedder.hidden_size()
            );
        }
        proposer_session.reset();
        Ok(Self {
            session,
            proposer: NativeProposer::SharedKv {
                session: proposer_session,
                embedder,
                groups,
                hidden_size,
            },
            draft_width: draft_width.max(1),
            tie_guard: TieGuard::from_env(),
            captured_verify: captured_verify_from_env(),
            hit_gate: HitDensityGate::from_env(),
            row0_tie_eps: row0_tie_eps_from_env(),
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

        // BUG1 (#984 re-review, Gaff): the M=1 base-decode graph and the M=width
        // captured-verify graph share the session's single device-graph slot.
        // Track which one was installed last so we can drop it on a mode switch
        // (engage↔miss, or a row-0 tie fallback) BEFORE the incoming path runs —
        // otherwise an M=1 decode can replay a stale width-W verify graph
        // ("invalidated graph replay" / `CUDA_ERROR_ILLEGAL_ADDRESS`). Starts
        // `false`: nothing is installed until the first forward warms a graph.
        let mut prev_engaged = false;

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

            // Fold the trailing committed token(s) into the KV. In the eager
            // path this is a separate base forward; in the fused captured path
            // the base distribution comes from row 0 of the verify forward.
            let past = self.session.current_len();

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

            // WP4 fused captured verify: prompt-lookup-only and steady-state-only
            // (exactly one trailing committed token). It proposes BEFORE the
            // forward so the base distribution (row 0) and the draft verify rows
            // fuse into ONE replayed CUDA graph. Every other configuration
            // (shared-KV, the multi-token prefill step, non-CUDA, capture
            // disabled) takes the unchanged eager path.
            let fused_eligible = self.captured_verify
                && self.session.is_cuda()
                && pending.len() == 1
                && matches!(self.proposer, NativeProposer::PromptLookup(_));

            let (base_logits, base, draft, rows) = if fused_eligible {
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
                let mut draft = match &mut self.proposer {
                    NativeProposer::PromptLookup(proposer) => {
                        proposer.propose(&proposer_context)?.tokens
                    }
                    _ => unreachable!("fused verify path is prompt-lookup only"),
                };
                draft.truncate(width);
                let bonus_token = pending[0];
                // Cheap CPU-side proposal outcome. Engage the width-W captured
                // verify only when recent would-hit density predicts the warmed
                // graph will REPLAY; otherwise degrade to a plain M=1 decode step
                // so sparse-hit prompts never pay repeated graph warmup (the
                // generic-prose regression). Both branches are byte-identical to
                // plain greedy, so this only trades throughput, never output.
                let would_hit = !draft.is_empty();
                self.hit_gate.record(would_hit);
                let engage = self.hit_gate.should_engage(would_hit);
                // Keep the retained verify graph alive across the per-step rewind
                // (sticky for both arms below) so a lookup hit REPLAYS the width-W
                // graph instead of re-capturing it.
                self.session.set_retain_graph_on_rewind(true);
                // BUG1 fix: the M=1 base decode and the M=width verify share ONE
                // device-graph slot. On an engage↔miss transition drop the
                // installed graph so the incoming path re-warms its own graph
                // instead of replaying a foreign/stale one (the miss-path M=1
                // replay of a still-installed verify graph is the illegal-address
                // hazard). We invalidate on the TRANSITION only — not every miss —
                // so consecutive same-mode steps still replay their captured graph
                // at full speed (the whole point of the miss-path M=1 capture).
                if engage != prev_engaged {
                    self.session.invalidate_graph_for_mode_switch()?;
                }
                if !engage {
                    // Lookup miss OR gated-out (sparse would-hit density): run the
                    // normal M=1 CAPTURED decode (device-graph replay = full plain
                    // baseline speed), NOT an eager forward. The mode-switch
                    // invalidation above already dropped any installed width-W
                    // verify graph, so this replays/rewarms only the M=1 slot.
                    // Degrades to exactly a plain greedy step (accepted=0) at
                    // baseline throughput — the guard that keeps sparse-hit prompts
                    // from regressing below plain decode.
                    let base_logits = self
                        .session
                        .decode(&pending, past)?
                        .pop()
                        .context("native gated decode produced no base logits")?;
                    pending.clear();
                    let base = self.session.current_len();
                    debug_assert_eq!(base, context_len);
                    prev_engaged = false;
                    (base_logits, base, Vec::new(), Vec::new())
                } else {
                    // Run ONE fused forward over `[bonus ⊕ draft]` padded to the
                    // fixed capture width. Row 0 is the base next-token
                    // distribution; rows 1..=k are the verify rows — identical
                    // numbers to the eager base + verify forwards. The fixed width
                    // keeps a single constant-signature graph in the slot so it
                    // replays across steps.
                    let fused_tokens: Vec<TokenId> = std::iter::once(bonus_token)
                        .chain(draft.iter().copied())
                        .collect();
                    let fixed_width = 1 + self.draft_width;
                    let mut fused =
                        self.session
                            .decode_verify_captured(&fused_tokens, past, fixed_width)?;
                    debug_assert_eq!(fused.len(), draft.len() + 1);
                    let rows = fused.split_off(1);
                    let base_logits = fused
                        .pop()
                        .context("native fused verify produced no base logits")?;

                    // CONTRACT (Chew, binding): speculative output MUST equal plain
                    // greedy, which sources the base token from the M=1 GEMV — not
                    // "greedy-under-Marlin". Row 0 here is the Marlin M=W kernel; at
                    // a genuine logit near-tie its argmax can flip vs the M=1 GEMV
                    // (the qwen one-token #984 divergence). If the base row is a
                    // near-tie, undo the fused fold and recompute the base token
                    // from a fresh M=1 GEMV decode so the committed base token is
                    // byte-identical to plain greedy. Rare on confident prompts, so
                    // negligible cost; on degenerate near-tie loops it simply
                    // parks on the (correct) plain-decode floor.
                    if row0_is_near_tie(&base_logits, self.row0_tie_eps) {
                        // Undo the width-W KV fold, switch the graph slot back to
                        // the M=1 base, and recompute row 0 from the M=1 GEMV.
                        self.session.rewind(past)?;
                        self.session.invalidate_graph_for_mode_switch()?;
                        let base_logits = self
                            .session
                            .decode(&[bonus_token], past)?
                            .pop()
                            .context("native row-0 tie fallback produced no base logits")?;
                        pending.clear();
                        let base = self.session.current_len();
                        debug_assert_eq!(base, context_len);
                        prev_engaged = false;
                        (base_logits, base, Vec::new(), Vec::new())
                    } else {
                        stats.verification_steps += 1;
                        stats.proposed_tokens += draft.len();
                        pending.clear();
                        let base = past + 1;
                        debug_assert_eq!(base, context_len);
                        prev_engaged = true;
                        (base_logits, base, draft, rows)
                    }
                }
            } else {
                // Eager / non-fused path (multi-token prefill step, shared-KV, or
                // capture disabled). This path invalidates the device graph
                // internally (uncaptured verify), so it leaves the slot in the
                // base/empty state — reset the mode tracker so the next fused
                // engage re-warms the width-W graph rather than assuming a stale
                // verify graph is still installed.
                prev_engaged = false;
                let base_logits = self
                    .session
                    .decode(&pending, past)?
                    .pop()
                    .context("native speculative decode produced no base logits")?;
                pending.clear();
                let base = self.session.current_len();
                debug_assert_eq!(base, context_len);
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
                            shared_kv_slices: None,
                        };
                        proposer.propose(&proposer_context)?.tokens
                    }
                    NativeProposer::SharedKv {
                        session: proposer,
                        embedder,
                        groups,
                        hidden_size,
                    } => {
                        let target_hidden = self.session.last_hidden().with_context(|| {
                        "native shared-KV proposer requires the target decoder's declared io.hidden_output; the target forward produced no hidden state"
                    })?;
                        if target_hidden.len() != *hidden_size {
                            anyhow::bail!(
                                "native target hidden output has width {}, but shared-KV metadata declares backbone_hidden_size {}; fix model.io.hidden_output or speculative.backbone_hidden_size",
                                target_hidden.len(),
                                hidden_size
                            );
                        }
                        let guaranteed = TokenId::try_from(
                            argmax(&base_logits).context("native target logits were empty")?,
                        )
                        .context("native target token id exceeds u32 range")?;
                        let shared_inputs = self.session.shared_kv_inputs(groups)?;
                        let seed = *context_tokens.last().context(
                            "native shared-KV proposer requires at least one context token",
                        )?;
                        let mut hidden = target_hidden.to_vec();
                        let mut token = seed;
                        let mut embeddings = vec![0.0; hidden_size.saturating_mul(2)];
                        let mut tokens = Vec::with_capacity(width);
                        tokens.push(guaranteed);
                        let position = context_tokens.len().saturating_sub(1);
                        for step in 0..width {
                            embedder.embed(token, &mut embeddings[..*hidden_size])?;
                            embeddings[*hidden_size..].copy_from_slice(&hidden);
                            let output = proposer.step_inputs_embeds(
                                &embeddings,
                                position,
                                &shared_inputs,
                            )?;
                            hidden = output.projected_state.with_context(|| {
                            "native shared-KV proposer metadata must assign io.hidden_output to its projected recurrent state"
                        })?;
                            let logits = output.logits.with_context(
                            || "native shared-KV proposer metadata must assign io.logits_output",
                        )?;
                            let drafted =
                                TokenId::try_from(
                                    argmax(logits.last().context(
                                        "native shared-KV proposer emitted no logits rows",
                                    )?)
                                    .context("native shared-KV proposer logits row was empty")?,
                                )
                                .context("native shared-KV proposer token id exceeds u32 range")?;
                            if step == 0 {
                                token = guaranteed;
                            } else {
                                tokens.push(drafted);
                                token = drafted;
                            }
                        }
                        tokens
                    }
                };
                draft.truncate(width);

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

                // Eager M=K verify pass: one target row per draft position (predicts
                // the token AFTER each draft token). current_len advances to base + K.
                let rows = self.session.decode_verify(&draft, base)?;
                (base_logits, base, draft, rows)
            };

            // Greedy acceptance with the numerical near-tie guard. `base_logits`
            // is the target distribution for position `base`; `rows[i]` predicts
            // `base + i + 1`. The longest strict-argmax draft prefix is accepted
            // plus one free bonus token. With `TieGuard::STRICT` (default) the
            // committed stream is byte-identical to plain greedy; a non-zero
            // `near_tie_rejections` flags acceptance (not correctness) lost to
            // tie-prone logits. WP3 will replace the host argmax inside
            // `greedy_accept` with a device `argmax_rows` launch; the accept /
            // rewind / commit logic here is agnostic to which side produced the ids.
            let AcceptOutcome {
                accepted,
                bonus,
                near_tie_rejections,
            } = greedy_accept(&base_logits, &rows, &draft, self.tie_guard);

            stats.accepted_tokens += accepted;
            stats.near_tie_rejections += near_tie_rejections;
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

#[cfg(test)]
mod accept_tests {
    use super::{TieGuard, greedy_accept};
    use crate::logits::TokenId;

    /// One-hot logits row whose argmax is `token`.
    fn onehot(vocab: usize, token: usize, peak: f32) -> Vec<f32> {
        let mut row = vec![0.0f32; vocab];
        row[token] = peak;
        row
    }

    #[test]
    fn strict_accepts_full_matching_prefix_plus_bonus() {
        // base -> 5, then rows predict 6, 7, 8. Draft [5, 6, 7] all match.
        let base = onehot(16, 5, 10.0);
        let rows = vec![
            onehot(16, 6, 10.0),
            onehot(16, 7, 10.0),
            onehot(16, 8, 10.0),
        ];
        let draft: Vec<TokenId> = vec![5, 6, 7];
        let out = greedy_accept(&base, &rows, &draft, TieGuard::STRICT);
        assert_eq!(out.accepted, 3);
        assert_eq!(out.bonus, 8); // trailing verify row after the full accept
        assert_eq!(out.near_tie_rejections, 0);
    }

    #[test]
    fn strict_stops_at_first_mismatch_and_bonus_is_target_argmax() {
        // base -> 5 (draft ok), row0 predicts 6 but draft says 9 -> mismatch at 1.
        let base = onehot(16, 5, 10.0);
        let rows = vec![onehot(16, 6, 10.0), onehot(16, 7, 10.0)];
        let draft: Vec<TokenId> = vec![5, 9, 7];
        let out = greedy_accept(&base, &rows, &draft, TieGuard::STRICT);
        assert_eq!(out.accepted, 1);
        assert_eq!(out.bonus, 6); // the target's own pick at the mismatch row
        assert_eq!(out.near_tie_rejections, 0);
    }

    #[test]
    fn near_tie_is_counted_but_not_accepted_under_strict_default() {
        // Row0 argmax is 6 (10.0) but draft token 9 is within eps (9.95).
        let base = onehot(16, 5, 10.0);
        let mut row0 = onehot(16, 6, 10.0);
        row0[9] = 9.95;
        let rows = vec![row0, onehot(16, 7, 10.0)];
        let draft: Vec<TokenId> = vec![5, 9, 7];
        let guard = TieGuard {
            eps: 0.1,
            tolerant: false,
        };
        let out = greedy_accept(&base, &rows, &draft, guard);
        // Strict: rejected at the near-tie, but the near-tie is DIAGNOSED.
        assert_eq!(out.accepted, 1);
        assert_eq!(out.bonus, 6); // still the strict argmax => byte-identical
        assert_eq!(out.near_tie_rejections, 1);
    }

    #[test]
    fn tolerant_accepts_the_near_tie_draft_token() {
        let base = onehot(16, 5, 10.0);
        let mut row0 = onehot(16, 6, 10.0);
        row0[9] = 9.95;
        // rows.len() == draft.len() is the real verify invariant (one row per draft token).
        let rows = vec![row0, onehot(16, 7, 10.0), onehot(16, 11, 10.0)];
        let draft: Vec<TokenId> = vec![5, 9, 7];
        let guard = TieGuard {
            eps: 0.1,
            tolerant: true,
        };
        let out = greedy_accept(&base, &rows, &draft, guard);
        // Tolerant: the near-tie draft (9) is accepted, then row1 predicts 7 == draft.
        assert_eq!(out.accepted, 3);
        assert_eq!(out.near_tie_rejections, 1);
        assert_eq!(out.bonus, 11); // trailing verify row after the full accept
    }

    #[test]
    fn a_real_mismatch_outside_eps_is_never_a_near_tie() {
        let base = onehot(16, 5, 10.0);
        let mut row0 = onehot(16, 6, 10.0);
        row0[9] = 2.0; // far below max -> genuine rejection
        let rows = vec![row0];
        let draft: Vec<TokenId> = vec![5, 9];
        let guard = TieGuard {
            eps: 0.5,
            tolerant: true,
        };
        let out = greedy_accept(&base, &rows, &draft, guard);
        assert_eq!(out.accepted, 1);
        assert_eq!(out.near_tie_rejections, 0);
        assert_eq!(out.bonus, 6);
    }

    #[test]
    fn empty_draft_yields_base_argmax_bonus() {
        let base = onehot(16, 5, 10.0);
        let rows: Vec<Vec<f32>> = Vec::new();
        let draft: Vec<TokenId> = Vec::new();
        let out = greedy_accept(&base, &rows, &draft, TieGuard::STRICT);
        assert_eq!(out.accepted, 0);
        assert_eq!(out.bonus, 5);
        assert_eq!(out.near_tie_rejections, 0);
    }
}

#[cfg(test)]
mod row0_tie_tests {
    use super::row0_is_near_tie;

    #[test]
    fn confident_row_is_not_a_tie() {
        // top1=10 at index 3, top2=1 elsewhere: margin 9 ≫ eps ⇒ not a tie.
        let mut logits = vec![1.0f32; 32];
        logits[3] = 10.0;
        assert!(!row0_is_near_tie(&logits, 1.0));
    }

    #[test]
    fn near_tie_within_eps_is_detected() {
        // Two co-leaders 5.0 / 4.6: margin 0.4 ≤ eps 1.0 ⇒ tie ⇒ fall back to M=1.
        let mut logits = vec![0.0f32; 32];
        logits[7] = 5.0;
        logits[11] = 4.6;
        assert!(row0_is_near_tie(&logits, 1.0));
        // A tighter eps than the margin classifies it as confident.
        assert!(!row0_is_near_tie(&logits, 0.2));
    }

    #[test]
    fn margin_exactly_eps_counts_as_tie() {
        let mut logits = vec![0.0f32; 8];
        logits[0] = 2.0;
        logits[1] = 1.0;
        // margin == eps ⇒ inclusive ⇒ tie (conservative: prefer the M=1 fallback).
        assert!(row0_is_near_tie(&logits, 1.0));
    }

    #[test]
    fn zero_eps_disables_the_guard() {
        let mut logits = vec![0.0f32; 8];
        logits[0] = 1.0;
        logits[1] = 1.0; // genuine exact tie
        assert!(!row0_is_near_tie(&logits, 0.0));
    }

    #[test]
    fn degenerate_rows_are_never_ties() {
        assert!(!row0_is_near_tie(&[], 1.0));
        assert!(!row0_is_near_tie(&[3.0], 1.0));
    }

    #[test]
    fn duplicate_max_is_a_tie() {
        // Two identical maxima ⇒ margin 0 ⇒ tie for any positive eps.
        let logits = vec![4.0f32, 4.0, 1.0, 0.0];
        assert!(row0_is_near_tie(&logits, 0.001));
    }
}

#[cfg(test)]
mod hit_gate_tests {
    use super::HitDensityGate;

    fn gate(window: u32, min_hits: u32) -> HitDensityGate {
        HitDensityGate {
            bits: 0,
            window,
            min_hits,
            enabled: true,
        }
    }

    #[test]
    fn never_engages_without_a_would_hit() {
        // Even with a saturated window, a miss (empty draft) cannot engage:
        // there is nothing to verify.
        let mut g = gate(4, 1);
        for _ in 0..8 {
            g.record(true);
        }
        assert!(g.should_engage(true));
        assert!(!g.should_engage(false));
    }

    #[test]
    fn engages_only_after_hits_cluster() {
        // window=8, min_hits=4: a fresh gate must see 4 would-hits accumulate in
        // the window before it engages the width-W path.
        let mut g = gate(8, 4);
        let mut engaged_at = None;
        for step in 1..=6 {
            g.record(true);
            if g.should_engage(true) {
                engaged_at = Some(step);
                break;
            }
        }
        assert_eq!(engaged_at, Some(4), "should engage exactly at the 4th hit");
    }

    #[test]
    fn isolated_hits_below_threshold_do_not_engage() {
        // Hits separated by misses keep density under the threshold, so the gate
        // stays on the plain-decode floor (the generic-prose non-regression).
        let mut g = gate(8, 5);
        for _ in 0..8 {
            g.record(true);
            assert!(!g.should_engage(true), "isolated hit must not engage");
            g.record(false);
        }
    }

    #[test]
    fn window_slides_and_disengages_when_hits_age_out() {
        // Once the window fills with hits it engages; a run of misses slides the
        // hits out and it disengages again (returns to the non-regression floor).
        let mut g = gate(4, 3);
        for _ in 0..4 {
            g.record(true);
        }
        assert!(g.should_engage(true));
        // Four misses evict every hit from the 4-wide window.
        for _ in 0..4 {
            g.record(false);
        }
        assert_eq!(g.density(), 0);
        g.record(true);
        assert!(
            !g.should_engage(true),
            "a lone hit after the window drained must not re-engage"
        );
    }

    #[test]
    fn disabled_gate_engages_on_every_hit() {
        let mut g = gate(16, 8);
        g.enabled = false;
        // No recorded history at all, yet a single would-hit engages: this is the
        // always-engage A/B baseline (ONNX_GENAI_SPEC_GATE=0).
        g.record(true);
        assert!(g.should_engage(true));
        assert!(!g.should_engage(false), "still needs a draft to verify");
    }

    #[test]
    fn window_mask_saturates_at_32_bits() {
        // window=32 must not overflow the 1<<window shift; the mask is all-ones.
        let mut g = gate(32, 32);
        for _ in 0..64 {
            g.record(true);
        }
        assert_eq!(g.density(), 32);
        assert!(g.should_engage(true));
    }
}
