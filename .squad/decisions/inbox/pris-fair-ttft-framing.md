# Decision: report time-to-first-token from process start

**Date:** 2026-07-28  
**Author:** Pris  
**Status:** proposed

## Context

TTFT alone favours runtimes that front-load work into model load. ORT
pre-packs weights during model load (~5× slower than native) which makes its
TTFT look better. The honest cold-start metric is model_load + TTFT.

## Decision

1. `compare.rs` now reports **process start → first token ms** (model_load +
   TTFT) as a derived column in both the Markdown table and the JSON output,
   plus a ratio row.
2. `examples/profiles/README.md` explains the mechanism and presents both
   framings with independently verified numbers.
3. The CI benchmark comment (PR #306) is NOT changed. It runs kernel
   micro-benchmarks (criterion), not model-level load+TTFT measurements.
   Adding model-level cold-start would require downloading models in CI,
   significantly increasing workflow cost and duration. The `compare` binary
   remains the right tool for model-level comparison.

## Verified figures (TinyStories-33M, M1 Max, load 3–5)

| Metric | native | ORT |
|---|---:|---:|
| model load (median) | 29.0–30.4 ms | 146.9–161.9 ms |
| TTFT (median) | 26.2–27.1 ms | 3.4–3.6 ms |
| process start → first token | 55.2–57.5 ms | 150.3–165.5 ms |

Native is 2.6–2.7× faster to first token from process start.
Sebastian's reported 54.6 / 150.5 ms corroborated within 5%.
