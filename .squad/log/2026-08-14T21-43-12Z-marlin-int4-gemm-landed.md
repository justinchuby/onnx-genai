# Session log — Marlin int4 GEMM (Lever A) LANDED; spec-capture CONDITIONAL-GO now GO in practice

**Timestamp:** 2026-08-14T21:43:12Z
**State backend:** local
**Requested by:** Justin (@justinchuby)
**Scribe batch:** marlin-int4-gemm-landed

## Milestone

**The Marlin fp16×int4 tensor-core GEMM program (Lever A) landed across 3 merged PRs.** Lever A was the
single funded condition of the #957 speculative-capture CONDITIONAL-GO — it is now **delivered**. On the
glm-4-9b canonical gate the M=8 speculative-verify graph went from **41 fragmented segments / capture
B\*=8.76× (hard NO-GO)** to a **SINGLE whole-graph capture with ZERO KernelCaptureUnsupported nodes at
B\*=2.16×**, byte-identical greedy tokens throughout, prefill ~2×.

## What landed

- **#960 (Deckard, 7774ec5b):** from-scratch SM80 `mma.sync.m16n8k16` fused fp16×int4 GEMM + repack, wired
  into `MatMulNBits` M>1 (plain + rmsnorm-prologue + gate_up SwiGLU fused) + split-K + GQA/SkipLN M>1
  capture-safety valves + lm_head dense-GEMM capture plan. Opt-in `ONNX_GENAI_MARLIN_M_GT_1` (default OFF,
  SM80 guard, byte-identical tiled fallback — Rule 11); split-K default-ON.
- **#961 (Pris, 401af46f):** reusable f64 dequant→GEMM numerics oracle gate with a justified tolerance
  envelope; tiled baseline sits ~1 fp16 ULP inside.
- **#962 (Sebastian, df6d3afb):** `marlin_bench` perf/capture harness + the BEFORE/AFTER B\* tables that
  produced the capture arc.

## Reviews

- **Chew (numerics) 🟡 APPROVE-WITH-NOTES:** 11/11 GPU parity tests pass on H200, no correctness bug; notes:
  keep the soft argmax-stability token guarantee opt-in (N1), log/count the silent tiled-fallback (N2),
  nibble-int-zp only (N3).
- **Gaff (quality) 🟢 APPROVE:** zero blocking defects; Rule 11 portability PASS, env-var honesty PASS,
  capture-safety valve sound, fmt/clippy clean; one trivial `cfg(test)` `clippy::unusual_byte_groupings` note.

## Per-model honesty

- **glm-4-9b (block-128):** the clean practical GO at **B\*=2.16×**. CORRECTED attribution (Deckard update-10
  supersedes update-8): the fused gate_up split-K commit is a NO-OP for glm (fused node needs block-32);
  glm's 2.63→2.16× was ENTIRELY the general small-M split-K retune.
- **qwen2.5-14b (block-32):** capture fully solved (whole-graph, zero seams, byte-identical eager) but
  **B\*≈4.7×** — an honest denominator effect (fast tuned block-32 M=1 inflates the ratio), a drafting-depth
  follow-up, NOT a kernel bug.

## Housekeeping

- Merged all 5 inbox drops (deckard-marlin-kernel, pris-marlin-numerics, sebastian-marlin-bench,
  chew-marlin-numerics-review, gaff-marlin-quality-review) into a single consolidated Marlin program entry in
  decisions.md; deleted all 5 drops (README.md kept).
- HARD-GATE size: decisions.md was **45,991 B** and this consolidation would have pushed it over 51,200 B →
  archived the three verbose "Last consolidated" chronicle lines + the detailed #957/#948/#949 spec-capture &
  Lever B sub-entry bodies (now RESOLVED by this landing) to `decisions-archive/2026-08.md` (live keeps
  compact pointers) → **42,104 B** after, under the 50 KB gate.
- HISTORY chronicle gate: **sebastian** history (12,308 B, ~13 dated entries) was a chronicle over the
  >8-entry threshold → compacted (08-12/13 decode arc moved to `history-archive.md`, live keeps role +
  durable lessons + a one-paragraph context + the 08-14 entries). Appended the Marlin outcome to deckard,
  sebastian, pris, chew, gaff; the other four stayed under both the chronicle and 15,360 B gates.
- Wrote orchestration logs for deckard, sebastian, pris, chew, gaff.
- Charter NOTE: the spawn prompt asked to archive by AGE (older-than-30-days); per Scribe charter I archived
  by SIZE and say so. Committed on `chore/scribe-marlin` (main is protected — the coordinator opens/admin-
  merges the PR).

## Standing directive (updated)

Marlin (Lever A) is DELIVERED and the spec-capture condition is MET for glm at B\*=2.16×; the #957
Increment-0 re-probe is DONE (this landing IS the post-Marlin measurement). Now unblocked (per #957 Stage 4):
fund the actual speculative build (verify sub-graph + capture-stable selective KV-commit + exact-greedy
near-tie guard #935 + draft sources) — glm is a practical GO; deeper/stronger drafting is the lever for
qwen's B\*. Marlin stays opt-in default-OFF; close Chew N1/N2 before any default flip.
