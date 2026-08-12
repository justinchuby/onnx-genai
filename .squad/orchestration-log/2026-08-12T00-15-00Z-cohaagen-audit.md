# Orchestration Log — Cohaagen (upstream audit)

**Timestamp:** 2026-08-12T00:15:00Z
**Agent:** Cohaagen (Opus)
**Wave:** CUDA MatMulNBits upstream workstream

## Task

Upstream audit: confirm whether upstream ORT hardcodes `kColsPerThreadBlock = 8` and whether any colliding work exists.

## Outcome

- Confirmed `matmul_4bits_common.cuh:15`: `constexpr int kColsPerThreadBlock = 8;`
- Confirmed `matmul_4bits_m1_impl.cuh:135`: `dim3 blocks((n+7)/8, 1)` — no SM-count adaptation.
- PR #29469 tunes M-crossover (which kernel), not grid geometry — no overlap.
- No colliding in-flight work found across 30+ PRs.

## Verdict

**Genuine uncovered gap.** Change is ~25 LOC, bit-identical, upstream-idiomatic.
Caveat: +2.08% claim has no provenance; benchmarks required before un-drafting.
