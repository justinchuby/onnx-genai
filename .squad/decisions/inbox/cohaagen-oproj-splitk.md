# Decision: o_proj split-K grid-widening — NEGATIVE result (do not merge)

**Author:** Cohaagen (perf)
**Date:** 2026-07-27
**Scope:** 7B native CUDA decode optimization attempt. Reverted — negative result.
**Branch/PR:** `squad/oproj-splitk-gemv` (draft, docs-only after revert).

## Context

Follow-up to the 7B bottleneck localization (#437): the square o_proj int4 GEMV
`gemv_f16_general` (K=N=3584) is 19.5% of the 7B native decode and is
grid-starved. Hypothesis: route it through the existing symmetric split-K entry
(`matmul_nbits_gemv_f16_scales_f16_splitk`, PR #203) by widening the dispatch
gate from `n < SM*16` (~2 CTA/SM) to a device-driven `< 1 wave/SM` occupancy
check (H200 = 1056 CTAs/wave). No new kernel; reuse existing K_SPLIT=2 machinery.

## What was measured (device 1, H200, steady tok/s, alternating BEFORE/AFTER)

BEFORE = origin/main; AFTER = split-K gate.

| model | BEFORE | AFTER | Δ | verdict |
|---|---|---|---|---|
| 7B   | 309.05 | 307.23 | **−0.59%** | repeatable regression (5/5 trials) |
| 1.5B | 725.05 | 723.65 | −0.19% | parity (noise) |
| 0.5B | ~993   | ~996   | +0.3%  | parity (noise) |

- 7B greedy token IDs **byte-identical** before/after (no repetition/garbage).
- GPU parity test at K=N=3584 matched f64 reference within int4 tolerance.
- 0.5B/1.5B already split-K under both gates → unchanged by construction.

## Verdict: REVERTED

The existing split-K is 2-way (K_SPLIT=2): it only doubles the grid, lifting
o_proj from ~0.42 → ~0.85 wave (still sub-wave) while adding a shared-memory
reduction to every column. At this occupancy the reduction tax outweighs the
grid-fill gain → small, repeatable 7B regression. Matches #203's deliberate
~2 CTA/SM cap: moderately-starved shapes don't benefit from a 2-way split.

## Top-2 optimization candidates (for a future reviewed PR, NOT this lever)

1. **Larger split factor for o_proj** — a K_SPLIT>2 (e.g. 3–4) grid-widened
   `general` variant that fills ≥1 wave. This is a *new kernel* (out of scope
   for the "reuse existing machinery" guardrail) and needs its own A/B. This is
   the most direct route to the 19.5% o_proj slice.
2. **GQA decode split-K flash (#1, 33.1%)** — already split-K-tuned; lower
   marginal headroom but the single largest op. Profile with Nsight Compute
   before touching.

**Guardrail recorded:** do NOT re-try the 2-way split-K lever on o_proj — no win.
(Companion to the register-prefetch-on-symmetric-gate/up negative memory.)
