# MobileNetV2 remaining gap: Clip dispatch-miss (defect #14)

**Date:** 2026-07-28T13:50:00Z  
**Author:** Deckard  
**PR:** #359  
**Branch:** squad/mobilenet-remaining

## Attribution

Per-op profiling (`ONNX_GENAI_PROFILE_OPS=1`) on MobileNetV2-12 (opset 12, 105 nodes:
52 Conv, 35 Clip, 10 Add, 1 GlobalAveragePool, 1 Gemm, misc):

| op_type | time (ms) | share |
|---------|-----------|-------|
| Clip | 34.0 | 76.8% |
| Conv | 9.6 | 21.5% |
| Gemm | 0.74 | 1.7% |
| Add | 0.08 | 0.2% |

**Clip dominates.** Not Conv, not pooling, not layout.

## Root cause

Dispatch-miss defect instance #14. The `Clip` kernel in `selection.rs` has a fast
contiguous path only behind `#[cfg(feature = "mlas")]`. Without MLAS the code falls
through to `clip_typed<T>()` → `to_dense::<T>()` → per-element strided read into a
newly allocated Vec → element-wise clamp → `write_dense()` copy back to output. Two
full copies of every ~600KB activation tensor, 35 times = ~42MB of wasted bandwidth.

## Fix

Added `clip_contiguous_f32_fast()` — a MLAS-independent path that:
- Zero-copy reads input via pointer (contiguous f32 guaranteed)
- Writes directly to output with NEON `vmaxq_f32`/`vminq_f32` (4×4 unroll = 16 elem/iter)
- Scalar fallback on non-aarch64

## Projection and measurement

- Clip per-op: 34ms → 0.86ms (39.5x)
- Amdahl: 1/(1-0.768 + 0.768/39.5) = **3.93x projected**
- Measured: **3.75–3.91x** (corroborated, load 7–11, M1 Max)

## Remaining gap

After: MobileNetV2 ≈ 11.5ms. ORT ≈ 6.3ms (inferred from original 0.14x ratio).
New ratio: ~0.55x (1.8x behind). The residual is Conv-dominated (85% of post-fix
runtime) — standard im2col+cblas vs MLAS NCHWc gap, no single lever remaining.

## Standing lessons reinforced

1. Attribution before implementation (correct again — would have guessed Conv wrong)
2. Dispatch-miss defect class: adequate path existed behind a feature gate
3. Amdahl projection matched measurement within noise
