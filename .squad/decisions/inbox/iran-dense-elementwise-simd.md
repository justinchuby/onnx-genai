# Decision: Dense-elementwise SIMD kernel replaces contiguity guards

**Date:** 2026-07-28
**Author:** Iran
**PR:** #366
**Status:** Pending Chew review

## Context

Justin identified that per-element ops (Relu, Clip, SiLU) were guarding
their SIMD fast paths with `is_contiguous()` — strict row-major check.
This is over-conservative: a per-element op only needs elements packed
without holes (dense), regardless of axis permutation. The contiguity
guard is an instance of "fast path exists but never executes" for any
dense-but-permuted tensor (e.g. NHWC layout in an NCHW model).

## Decision

1. **Shared `dense_elementwise.rs` module** with `ElementwiseOp` trait and
   `try_dense_elementwise()` dispatcher. Guard condition: same shape, same
   strides, both `is_dense()`. This is the correct minimal condition.

2. **`is_dense()` added to `onnx-runtime-ir::layout`** as a first-class
   layout predicate. Strictly weaker than `is_contiguous()`.

3. **Dtype coverage:** f32 (NEON SIMD), f16 (NEON vcvt widen/narrow — all
   aarch64), bf16 (scalar widen/narrow). Native `fp16` target_feature NOT
   used because not universal across supported Apple Silicon parts.

4. **Audit applied to:** Relu, Clip, SiLU, binary elementwise (Add/Mul/Sub/Div).
   All relaxed from `is_contiguous` to `is_dense` + stride match.

## Performance

- Layout relaxation: **latent-only** (0.0x today). No current model hits the
  relaxed path. Value: prevents future slow-path regression.
- f16/bf16: **infrastructure** (0.0x today). No current model exercises
  standalone f16 Relu/Clip on CPU EP. Value: readiness for fp16 models.

## Standing constraints

- NaN **must propagate** on all paths. Use `vmaxq_f32` (FMAX), never
  `vmaxnmq_f32` (FMAXNM). Never use `f32::max`/`f32::min`.
- Signed zero divergence (-0→+0 on NEON) is accepted per ONNX spec.
- Every dispatch path needs a counter and a manifest row.
