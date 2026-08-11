# Decision: Upstream ORT Contribution Validation Methodology

**Author:** Sebastian
**Date:** 2026-08-11
**Status:** Proposed

## Summary

Defined the validation protocol (`docs/UPSTREAM_ORT_CONTRIB_METHODOLOGY.md`) that any upstream kernel contribution to `microsoft/onnxruntime` must follow.

## Acceptance criteria (brief)

1. **Reachability:** Dispatch test that fails on silent fallback; positive proof the optimized kernel runs.
2. **Numeric parity:** fp64 reference, per-dtype ULP/relative bounds, edge-case coverage, regression-locked tolerance.
3. **Model-level benchmarks:** p50/p95 latency + throughput + memory, 30+ reps, warmup, variance disclosed, Amdahl sanity.
4. **Same-artifact methodology:** Identical model/quant/build/hardware/threads/clocks; 12-item red-flag rejection list.

## Review ownership

- Numerics → Chew
- Benchmark validity → Sebastian
- Reachability → Challenger
