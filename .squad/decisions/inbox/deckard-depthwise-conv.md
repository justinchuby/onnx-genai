# Decision: Depthwise/Grouped Conv dispatches to im2col+GEMM (tier 2)

**Date:** 2026-07-28
**Author:** Deckard (Systems Dev)
**Status:** Proposed (pending Chew numerics gate)

## Context

Depthwise convolution (groups == in_channels == out_channels) is the backbone
of MobileNet, EfficientNet, and most edge-oriented vision models. After #317
and #324 brought standard Conv from 643× slower than ORT to 1.43× faster,
depthwise still fell to the scalar reference because the dispatch guard
`is_group1 = self.group == 1` excluded all grouped convolutions from the
BNNS and im2col+GEMM tiers.

## Measurement

**MobileNetV2 (1.0, 224×224) — native vs ORT CPU:**
- Native (with this PR): 53 ms
- ORT: 4.4–6.3 ms
- Gap: ~8–12× slower than ORT

**ResNet-18 regression check:**
- Native: 8.7 ms, ORT: 13.0 ms → 1.49× faster than ORT ✓ (was 1.43×)

The remaining MobileNetV2 gap is expected: ORT uses MLAS's fully vectorized
depthwise kernel (NEON-optimized, register-tiled) while our tier 2 calls
cblas_sgemm per-group for M=1 GEMV operations.

## Investigation: BNNS grouped convolution

The `BNNSLayerParametersConvolution` struct has a `groups` field, suggesting
BNNS supports grouped convolution. **Empirically, this is false:**

`BNNSFilterCreateLayerConvolution` returns NULL for ANY groups > 1,
regardless of:
- oc_per_group value (tested 1, 2)
- ic_per_group value (tested 1, 2)
- spatial dimensions (tested 2×2 through 14×14)
- kernel size (tested 1×1, 3×3)

The `groups` field appears vestigial in the legacy BNNS API. The newer
BNNSGraph API (macOS 14+) may support it, but requires a fundamentally
different integration pattern.

## Decision

**Tier 2: Grouped im2col + cblas_sgemm** for all grouped convolutions.

The implementation:
1. Iterates over groups
2. For each group: im2col on that group's input channels, then GEMM with
   that group's weight slice
3. Uses the existing `gemm_with_backend` which routes to Accelerate cblas

This satisfies the "one implementation, no arch forks" constraint (works on
Intel, ARM, and Apple Silicon via their respective BLAS libraries).

### Why not a direct NEON depthwise kernel?

For depthwise conv, each group's GEMM is M=1, K=9, N=oH×oW — essentially a
dot-product-broadcast. A direct loop with NEON fmla would avoid im2col's
memory duplication and likely 2–3× im2col+GEMV. However:
- Violates "one implementation, no arch forks" (Resch/Intel can't use it)
- The portable im2col+GEMM path is correct, tested, and moves depthwise from
  tier 3 to tier 2
- A platform-specific tier between BNNS and im2col can be added later as
  agreed by Iran/Resch/Luba

### Why not use the MLAS path?

The `conv.rs` (MLAS-backed) already handles depthwise via NCHWc, but:
- mlas-sys doesn't compile on arm64-apple (avx2 kernels)
- The `conv_ref.rs` path is what the benchmarks (#317, #324) measured
- This PR fixes `conv_ref.rs` specifically

## Implications

- Depthwise conv moves from tier 3 (scalar, ~100s× slower) to tier 2
  (im2col+cblas, ~8–12× slower than ORT on MobileNetV2)
- ResNet-18 is unaffected (group=1 path unchanged)
- LLM path is unaffected (no Conv ops in transformer decode)
- The manifest exclusion for depthwise is replaced with a claim

## Future work

To close the remaining 8–12× gap to ORT on depthwise-heavy models:
1. BNNSGraph API integration (macOS 14+) — may support grouped conv natively
2. Platform-specific depthwise kernel (requires pod consensus)
3. Port MLAS NCHWc blocked path to arm64 build
