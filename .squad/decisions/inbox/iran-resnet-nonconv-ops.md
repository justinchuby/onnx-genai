# Decision: ResNet-18 non-Conv op acceleration via Accelerate framework

**Author:** Iran (Mac CPU Optimization Engineer)
**Date:** 2026-07-27
**Status:** Proposed (PR open, awaiting Chew numerics review)
**Context:** Ninth instance of the "scalar path on Apple Silicon" defect pattern

## Problem

After PR #317 brought Conv from 8792 ms → 93 ms (BNNS dispatch), ResNet-18
remained at 0.15× ORT (93 ms native vs 12.9 ms ORT). Per-op profiling revealed:

| Op         | Before (ms) | % of runtime | Root cause                        |
|------------|-------------|--------------|-----------------------------------|
| MaxPool    | 75–160      | 77–88%       | Scalar N-D reference (no MLAS on ARM) |
| Add        | 9–31        | 5–22%        | Scalar broadcast walk (no MLAS on ARM) |
| BatchNorm  | 0           | 0%           | Already fused by ConvBatchNormActivationFusion |
| Conv       | 9–16        | 5–17%        | BNNS (fast, correct)              |
| Relu       | 0.4–1       | 0.5–5%       | Scalar loop (NaN-aware)           |

Additionally, the BatchNormalization kernel was registered at opset 15 only,
blocking models at opset 7–14 (e.g. standard ResNet-18 at opset 8).

## Solution

Three changes:

### 1. BNNS 2D Pooling (MaxPool → 0.13 ms, ~580× faster)

`BNNSFilterCreateLayerPooling` dispatches spatial pooling to Apple's vectorized
implementation. Gated on:
- rank-4 input (2D spatial)
- f32 dtype
- no dilations
- no indices output
- MaxPool or AveragePool without padding (avg padding semantics differ)

Falls back to scalar reference for all other cases.

### 2. vDSP_vadd for Add (→ 0.28 ms, ~35× faster)

Contiguous f32 same-shape elementwise addition via Accelerate's vDSP_vadd.
Guards: same shape, contiguous, no aliasing. Broadcasting falls through to
the generic scalar path.

### 3. BatchNormalization opset 7+ registration

The kernel's inference logic (5 inputs → 1 output) is semantically unchanged
since opset 7. Lowered `since_version` from 15 to 7 so the optimizer's
Conv+BN fusion and the standalone BN kernel both work on older models.

## Result

| Metric         | Before | After  | Δ       |
|----------------|--------|--------|---------|
| Native (ms)    | 93     | 9.4    | 9.9×    |
| ORT (ms)       | 12.9   | 13.4   | ~same   |
| native/ORT     | 7.2×   | 0.70×  | **native is 30% faster** |

## Non-negotiables satisfied

- [x] `_TEST_HITS` reachability tests for every new dispatch branch
- [x] Numerics parity (PASS, max_abs=1.1e-5, max_rel=5.6e-4)
- [x] One implementation, no arch forks (all Apple paths cfg-gated)
- [x] Cross-platform compilation (non-Apple paths compile-gated out)
- [x] No regression to LLM path (changes are vision-model only)

## Architecture

Per Justin's directive: **fusion decisions live in the EP, shared mechanism
lives in the optimizer.** The Conv+BN fusion is an EP optimizer pass in
`onnx-runtime-ep-cpu/src/optimizer.rs`. The Pool/Add fast paths are pure
kernel-level dispatch in the EP — no new optimizer passes needed.
