# Decision: Thin-M GEMM bypass for f32 prefill on Apple Silicon

**Date**: 2025-07-22
**Author**: Iran (Mac CPU Optimization)
**Status**: Proposed

## Context

For f32 models with small prompt lengths (M=1..16), `cblas_sgemm` achieves only
~25 GB/s effective bandwidth on shapes like [7,768]×[768,50257] — roughly 10×
below the achievable memory bandwidth. This is because Accelerate's tiling and
thread dispatch are designed for large square matrices; the fixed overhead dominates
when M is small relative to N and K.

## Decision

Add a NEON column-parallel thin-M GEMM path that bypasses `cblas_sgemm` when:
- M ∈ [2, 16] (M=1 already has its own GEMV path)
- K × N > 4,000,000 (B exceeds SLC, streaming is better than panel tiling)
- B is a constant (model weight) with pre-transposed B_T available
- f32 dtype (f16 uses BNNS, unaffected)

The kernel processes strips of 4 B_T columns at a time, computing all M rows per
strip while data is L1-hot. Rayon parallelism distributes strips across cores.

## Threshold Portability

- **Mechanism (general)**: When B exceeds SLC capacity, streaming B_T once with
  NEON dot products is faster than cblas's panel tiling for thin M. This holds
  across all Apple Silicon (SLC ranges 8-48 MB; threshold 4M elements = 16 MB
  conservatively covers all).
- **Coefficient (fitted)**: M crossover at 16 was measured on M1 Max. Bracket
  [16, 24] observed; 16 chosen conservatively. Labeled `THIN_M_MAX` in source
  with measurement annotation.

## Precomputation

f32 weight transposes are now precomputed during model load (matching the existing
f16 precompute pattern). This prevents a TTFT spike on first inference after
Engine creation — the transpose cost is amortized into model load.

## Results (TinyStories-33M, "Once upon a time", M1 Max)

| Metric | Baseline (cblas) | Thin-M GEMM | Change |
|--------|-----------------|-------------|--------|
| TTFT   | 17.7 ms [17.4, 22.3] | 13.3 ms [13.1, 18.8] | **-25%** |
| Model load | 46.9 ms | 44.5 ms | neutral |

No regression on qwen2.5-0.5b-f16 (BNNS path): 56.3 ms TTFT unaffected.

## Counter & Manifest

- Counter: `THIN_M_GEMM_TEST_HITS`
- Manifest: `MatMul/f32_thin_m/aarch64-apple-darwin/tier2`
