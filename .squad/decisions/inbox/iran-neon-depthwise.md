# Decision: Direct NEON depthwise convolution kernel

**Author:** Iran (Mac CPU Optimization)
**Date:** 2026-07-28
**Status:** Proposed (PR pending Chew numerics review)

## Context

PR #334 routed depthwise/grouped Conv to im2col+GEMM, bringing MobileNetV2 from ~100× to ~12× ORT. But depthwise convolution is structurally memory-bound — each group has M=1, K=kernel_size, so im2col expands memory traffic by ~kernel_size× for almost no arithmetic density gain.

BNNS genuinely rejects `groups > 1` (confirmed by FFI probe), so tier 1 is unavailable for depthwise.

## Decision

Add a direct NEON depthwise convolution kernel as tier 2a, between BNNS and im2col+GEMM.

- **Depthwise proper** (`groups == in_channels == out_channels`): routes to the new NEON direct kernel on aarch64, avoiding the im2col buffer entirely.
- **Grouped-but-not-depthwise** (`1 < groups < in_channels`): stays on im2col+GEMM — these cases have higher arithmetic density per group and im2col remains cost-effective.
- **Channel multiplier** (`groups == in_channels, out_channels != in_channels`): stays on im2col+GEMM — the multiplier means each group's GEMM has M > 1.

### Kernel specialization

- **3×3, stride 1, undilated**: Split output row into scalar left boundary + NEON-vectorized interior (4-wide FMA with raw pointer loads, guaranteed in-bounds) + scalar right boundary. This covers the majority of MobileNet/EfficientNet depthwise layers.
- **3×3, stride 2, undilated**: Scalar loop (stride-2 prevents contiguous NEON loads) but eliminates im2col buffer allocation.
- **General fallback**: Handles arbitrary kernel size, stride, and dilation. Direct computation, no im2col.

### Architecture

One implementation behind `#[cfg(target_arch = "aarch64")]`, no `target_os` gate — shared by macOS (Iran), Linux ARM (Luba), and Windows ARM (Resch).

## Measured result

MobileNetV2, interleaved A/B:
- **Before (im2col, PR #334):** ~53 ms native, ~12× ORT
- **After (NEON direct):** ~47 ms native, ~7.2× ORT

Chew's 4-8× estimate was for the depthwise layers in isolation. The overall model improvement is ~1.7× because depthwise Conv is ~15-20% of MobileNetV2's total runtime — the remaining gap is pointwise Conv, Add, Clip, and GlobalAveragePool layers.

## Trade-offs

- The 3×3 stride-2 path uses scalar (no NEON vectorization) because stride-2 prevents contiguous loads. This is still 2× better than im2col for that case due to eliminating the buffer.
- The general fallback is scalar — could add NEON vectorization for 5×5 kernels later if EfficientNet profiling shows it matters.
