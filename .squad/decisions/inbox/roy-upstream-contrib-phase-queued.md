# Decision: Upstream ORT Contribution Phase — Queued

**By:** Roy (Lead)  
**Date:** 2026-08-11  
**Status:** QUEUED (planning only — no implementation until entry gate passes)

## Entry Gate

Phase does not start until:
1. PR #762 merged or approved with no blocking reviews
2. Native nxrt ABI and CUDA EP open items closed or explicitly deferred
3. Resch/Batty/Sebastian inventory docs landed on `main`
4. Justin gives explicit go

## Routing

| Class | Owner(s) |
|-------|----------|
| CPU kernels (Intel/AVX) | Resch |
| CPU kernels (Apple/NEON) | Iran |
| CPU kernels (ARM/SVE) | Luba |
| CUDA kernels | Deckard, Leon, Batty |
| Numerics gate | Chew |
| Benchmark validity | Sebastian |
| Reachability / "was it on?" | Challenger |
| Security/provenance | Holden |
| Tests | Pris |

## Key Constraint

Upstream work is C++/CUDA in a fork of `microsoft/onnxruntime`. Completely separate from this Rust repo and PR #762.

## Reference

Full plan: `docs/UPSTREAM_ORT_CONTRIB_PLAN.md`
