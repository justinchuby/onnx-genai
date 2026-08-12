# Session log — 2026-08-12T02:00:00Z — apple-framework-policy

**Branch:** `squad/ep-plugin-parity-cuda`
**Requested by:** @justinchuby

## Summary

Two parallel workstreams completed:

**Apple framework infrastructure (PR #32001):**
- Policy corrected: Accelerate/BNNS/vDSP ARE upstream-eligible when opt-in gated with portable fallback.
- Luba opened draft PR #32001 (23 lines, 2 files). Luv reviewed; three substantive findings (S1/S2/S3).
- Isidore revised under lockout (Luba + Luv both barred): FATAL_ERROR → warn+disable, build.py plumbed, dangling define removed. Head: `d16a108252`.
- PR A (#32001) remains draft. PRs B/C/D (Accelerate/BNNS/vDSP kernels) prepared but not started — require Apple hardware benchmarks.

**TensorRT blocker on PR #31988:**
- Root cause: host `.cc` test included `.cuh` which pulled CUB via `<cuda_bf16.h>`. ~40 `blockIdx` errors.
- Leon extracted host-only header; Deckard's earlier "inherited" assumption disproved by cross-PR comparison.
- Head: `34fe91e8dd`. Blocker cleared.

## Inbox merged (6 drops)
coordinator-apple-framework-policy.md, deckard-31988-build-fix.md, isidore-32001-fixes.md,
isidore-objc-static-analysis.md, leon-31988-tensorrt.md, luba-apple-framework-option.md,
luv-review-32001.md
