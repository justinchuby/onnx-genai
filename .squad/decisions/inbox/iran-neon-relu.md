# Decision: NEON fast path for Relu (MLAS gate audit item #2)

**Date:** 2026-07-28  
**Author:** Iran  
**PR:** (to be filled after push)  
**Unblocks:** #360 (Resch's MLAS gate audit lint)

## Context

Resch's audit (PR #360) identified Relu as the second HIGH-priority violation of the
same pattern that made Clip 76.8% of MobileNetV2 runtime: the only fast path sat behind
`cfg(feature = "mlas")` — unreachable on macOS by construction. Every call allocated a
`Vec`, widened to f32, clamped, and wrote back. For ResNet-18: 8 surviving Relu nodes ×
up to 784 KB spatial tensors (after Conv+BN+Relu fusion eliminates the other 9).

## Attribution

**ResNet-18, batch=1, M1 Max, load 7.18:**

| Metric | Before | After |
|--------|--------|-------|
| Relu total | 0.43 ms (5.17%) | 0.094 ms (1.17%) |
| Model total | 8.29 ms | 7.94 ms |
| Per-op speedup | — | 4.6× |
| Model speedup | — | 1.044× |

**Amdahl projection:** 1/(1 − 0.0517) = 1.054× ceiling. Measured 1.044× — within
projection, confirming Relu is not a bottleneck-dominator on ResNet-18. The fix is
justified as a correctness-of-dispatch matter (eliminating unnecessary allocation) and
as the requirement for #360's lint to pass.

**Load context:** Measurements at load 7.18 (1m) and corroborated at load 17.94 (1m).
Both show Relu dropping from ~5% to ~1.2% of inference.

## Semantics decision: NaN and signed-zero behaviour

**Choice:** Match MLAS — `vmaxq_f32(x, 0)` propagates NaN.

ARM NEON `vmaxq_f32` (FMAX instruction): when either operand is NaN, returns the first
operand (the NaN). This means `max(NaN, 0) = NaN`, matching the existing `relu_in_place`
scalar path and ONNX/numpy `maximum(0, NaN) = NaN` contract.

This does NOT match IEEE 754-2008 `maxNum` (which would return 0 for `maxNum(NaN, 0)`).
The choice is deliberate and documented in the module docstring.

For signed zero: `vmaxq_f32(-0.0, +0.0) = +0.0` (ARM FMAX returns +0 when operands are
±0). The scalar path agrees: `(-0.0f32).max(0.0) = 0.0`. Both paths produce identical
bits.

## Manifest row

Added `[[claim]]` for `(Relu, contiguous_f32, all, tier2)` with counter
`RELU_F32_FAST_TEST_HITS`. This is what Resch's lint in #360 will verify.

## Structure

Follows Deckard's Clip pattern exactly:
1. Contiguity + dtype + shape guard → bail to generic path if non-contiguous
2. Overlap/aliasing guard → bail if input/output alias
3. Zero-allocation NEON SIMD (4×4 bulk + 4-lane tail + scalar tail)
4. Dispatch counter increment
5. Scalar fallback on non-aarch64

## Tests added

- `relu_f32_fast_path_fires_on_contiguous_input` — reachability proof (counter increment)
- `relu_f32_fast_path_matches_scalar_reference` — numerics parity for lengths 1/15/16/17/1023
- `relu_f32_fast_path_nan_semantics` — NaN propagation and signed-zero through fast path
