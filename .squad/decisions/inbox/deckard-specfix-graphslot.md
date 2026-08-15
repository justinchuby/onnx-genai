# Decision drop — Deckard: speculative-decode capture-stability fix (#984 re-review)

**Author:** Deckard (Systems Dev, CUDA/decode-perf)
**Branch:** `squad/spec-decode-graphslot` (based on #984 head `82cde423`)
**Supersedes review verdicts:** gaff-pr984-quality.md, chew-pr984-correctness.md
**Feature:** opt-in (`ONNX_GENAI_SPEC_CAPTURED_VERIFY=1`), default-OFF; ⚠️ needs re-review.

## What was wrong (both reviewers, both confirmed)
- **BUG 1 (Gaff) — miss-path graph-slot hazard.** The M=1 base-decode graph and
  the M=width captured-verify graph share the session's *single* device-graph
  slot. The old miss branch set `retain_graph_on_rewind` then ran an M=1 decode,
  and `reset_captured_verify()` ran *after* it — so an M=1 decode could replay a
  stale width-W verify graph → glm "invalidated graph replay" + qwen
  `CUDA_ERROR_ILLEGAL_ADDRESS`.
- **BUG 2 (Chew) — GQA workspace under-prep.** The persistent GroupQueryAttention
  workspace is sized by query width and prepared once at prefill for
  `q_seq = prompt_len`. On a short degenerate-repetition prompt
  (`prompt_len < W`, e.g. qwen 哈×20 → 8 tokens, W=9) the width-W verify entered
  with an undersized workspace → executor rejected: *"GroupQueryAttention
  workspace invariant mismatch: requires 184320, prepared 163840."*
- **CONTRACT (Chew, binding).** Spec output MUST equal *plain greedy* (base token
  from the **M=1 GEMV**), not "greedy-under-Marlin." Fusion sourced row 0 from
  Marlin M=W, so at a genuine logit near-tie the base argmax could flip vs the
  M=1 GEMV (the qwen one-token divergence).

## Approach chosen: invalidate-both (mode-transition) — NOT the second graph slot
I implemented the coordinator-sanctioned **fallback**: a correct mode-transition
invalidation of the shared slot, plus the workspace re-prep and the row-0 M=1
contract guard. I deliberately did **not** land a dedicated second device-graph
slot in this increment.

**Why (scoped rationale):** a real second slot requires changing the shared
`ExecutionProvider` graph trait (`begin/end/abort/replay/reset_device_graph` take
no slot param and are implemented by the CPU EP too), the runtime's single
`CudaGraphLifecycle` field, CUDA graph-exec-handle multiplexing, and per-slot
`device_graph_signature`/`capture_schedule` in the session executor — deep,
cross-cutting, and capture-stability-risky. The reviewers required correctness
(no crash, byte-identical), which the mode-transition fix delivers with a far
smaller, auditable blast radius. **Second slot is documented as the scoped
follow-on** (removes the miss↔engage re-warm tax; theoretical ceiling
~111×W/B*). It is NOT needed for correctness and is a perf-only optimization.

## The fix (three parts, all in the opt-in captured path)
1. **BUG 1 — mode-transition slot invalidation** (`native_speculative.rs`,
   `native_decode/mod.rs`). New `NativeDecodeSession::invalidate_graph_for_mode_switch()`
   drops the EP slot and resets *both* the M=1 `graph_phase`/`inline_graph_phase`
   and the retained `verify_capture.phase` to `NeedsWarmup` (verify bindings
   kept). The driver tracks `prev_engaged` and calls it **only on an
   engage↔miss transition, before** the incoming forward — so consecutive
   same-mode steps still replay their captured graph at full speed (the miss-path
   M=1 replay is preserved), but no path can ever replay a foreign/stale graph.
   Invariant restored: `NeedsWarmup ⇔ slot empty`. The eager branch resets
   `prev_engaged=false` (it invalidates internally).
2. **BUG 2 — width-sized workspace reservation** (`native_decode/cuda.rs`,
   `run_verify_captured_cuda`). Before warming/capturing the verify graph (phase
   == `NeedsWarmup`), reserve the persistent GQA workspace for `q_seq = W` via
   `prepare_cuda_prefill_workspace_with_step_inputs`, so the (possibly larger)
   reservation is baked into the captured graph and every replay is valid.
   Idempotent + only on (re)warm; a replay carries the reservation.
3. **CONTRACT — row-0 near-tie → M=1 GEMV fallback** (`native_speculative.rs`).
   After the fused forward, if the base row's top-1/top-2 margin ≤
   `ONNX_GENAI_SPEC_ROW0_TIE_EPS` (default **1.0** logit), undo the fused KV fold
   (`rewind(past)`), switch the slot to the base graph, and recompute the base
   token from a fresh **M=1 GEMV** `decode` — so the committed base token is
   byte-identical to plain greedy. Confident prompts sit far above eps (no
   fallback, full speculative win); degenerate near-tie loops simply park on the
   correct plain-decode floor.

## Validation (GPU7, H200, verified idle before each run)
- **f64 oracle** `matmul_nbits_marlin_numerics`: **7/7** (Marlin untouched).
- **BUG 2 crash-fix:** qwen2.5-14b degenerate prompt (哈×20), captured verify +
  Marlin ON, **ran to completion 160 tokens, 79 verify steps — no illegal
  address, no workspace mismatch.**
- **Byte-identical spec == plain greedy** (`generated_token_ids` diff):
  - glm-4-9b generic prose: PASS (gate-off force-engage AND gate-on adaptive)
  - glm-4-9b repetitive: PASS
  - qwen2.5-14b degenerate (the previously-diverging near-tie): **PASS** —
    identical stream (the row-0 M=1 fallback closes the flip).
- **Favorable win:** glm repetitive **326 tok/s** (2.88× plain 113.5; past ORT
  250), 97% acceptance, byte-identical.
- **Generic no-regression:** gate correctly disengages (0 verify steps),
  byte-identical, no crash/re-warm; 106 vs 112 plain (~5% CPU proposer tax,
  inherent to running prompt-lookup — no longer a warmup crash).
- **Default-OFF:** eager path (feature env unset) byte-identical to plain @160.
- **Regression tests (both reviewer-required):**
  (a) `native_captured_verify_engage_miss_reengage_no_stale_replay_cuda` —
      engaged 2 verify steps, byte-identical, no illegal-address. PASS.
  (b) `native_captured_verify_short_prompt_grows_gqa_workspace_cuda` — qwen-14b
      short prompt (prompt_len < W), 160 tokens, 79 verify steps, no workspace
      mismatch. PASS.
  Plus 6 new fast unit tests for the row-0 near-tie classifier.
- **fmt** clean; **clippy** `-p onnx-genai-engine --features cuda,native-backend
  --all-targets -D warnings` clean except the 2 pre-existing `platform_capacity.rs`
  u64 casts (untouched, as instructed).

## New env knobs
- `ONNX_GENAI_SPEC_ROW0_TIE_EPS` (default 1.0) — row-0 M=1-fallback margin; 0
  disables (A/B only; default MUST stay on for the contract).

## Follow-on (not in this PR)
- Dedicated **second device-graph slot** (M=1 base + M=W verify both installed,
  replay independently): eliminates the miss↔engage re-warm tax → generic prompts
  also win, ceiling ~111×W/B*. Structural EP/session change; perf-only.
