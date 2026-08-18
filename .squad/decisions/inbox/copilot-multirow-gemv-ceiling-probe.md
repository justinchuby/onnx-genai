# Decision: multi-row decode GEMV not worth building (resident-model prize ≤1 ms/step)

**Author:** Copilot (multi-request batching slice)
**Date:** 2026-08-18
**Status:** proposed (docs-only; no code change)

## Context

Following the merged looped single-row decode GEMV (PR #1312, which removed the
M≥2 tiled-GEMM cliff), the standing question was whether to build a **true
multi-row GEMV** (one weight read → M outputs), gated by the owner behind a
cheap roofline check: build only if decode at M=2..8 is weight-bandwidth-bound.

## Decision

**Do not build the multi-row GEMV** for the currently-measurable target. A
control-arm ceiling probe on the resident `qwen05b-q4` (RTX 4060 8 GB, CUDA
13.1) — capping the looped GEMV to one weight read per node, a bound the
multi-row change cannot beat — measures the entire prize at **≤1 ms/step: ~7% at
M=2, ~4% at M=4, ~0% (noise floor) at M=8**. The ~14 ms M≥2 decode step is
dominated by **fixed non-matmul batch overhead**, not redundant weight reads; the
0.5B weight matrices (~2 MB) are too small for the read to bind. A large
multi-kernel CUDA change returning ≤7% fails 简洁高效.

## Consequences / handoffs

- Full method + tables: `docs/benchmarks/2026-08-18-multirow-gemv-ceiling-probe.md`.
- The multi-row prize **may** be larger on datacentre GPUs where a large model is
  resident (10–30 MB weight tiles, bandwidth-bound) — but that is **untestable on
  this 8 GB box** and must be sized with the same ceiling-probe method before any
  kernel is written. Do not rebuild it expecting a large win without that data.
- **Adjacent lead (routing):** the real small-batch limiter here is the ~14 ms
  fixed batch-decode overhead (5.4× M=1→M=2 jump in the non-matmul path —
  attention/KV/scheduler), a different slice from multi-request batching.
- The looped GEMV (#1312) stands as the shipped, measured win (6.6×/7.7×/7.0×
  aggregate tok/s at N=2/4/8 on qwen05b-q4; neutral on qwen14b streaming).
