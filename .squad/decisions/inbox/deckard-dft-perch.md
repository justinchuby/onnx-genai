# DFT kernel for Perch bioacoustics model

**Date:** 2026-07-28
**Author:** Deckard
**Status:** Complete

## Summary

Implemented the ONNX DFT operator (opset 17+) with a vDSP Accelerate fast path for
power-of-two lengths on macOS/iOS, plus a Cooley–Tukey radix-2 fallback for all platforms.
Verified end-to-end on the Perch v2 bioacoustics model from HuggingFace.

## Opset registration finding

The Perch model declares **opset 18**. DFT is registered at `since_version: 17`.
The `OpRegistry::lookup` uses `partition_point(|&v| v <= opset)` — it finds the highest
`since_version` ≤ the model's opset. So `17 ≤ 18` matches correctly.

**This is NOT an instance of the opset-mismatch defect class.** The DFT op was introduced
in ONNX opset 17 and has not changed semantics since. A single `since_version: 17`
registration correctly covers all models at opset ≥ 17 (including Perch at opset 18).

Verified empirically: `DFT_VDSP_TEST_HITS` counter increments by 1000 during Perch
inference, confirming the kernel fires on the real model path (not a fallback).

## Attribution (release build, M1 Max, load 3.85)

| Op | Time (ms) | % of total | Calls |
|---|---|---|---|
| Add | 294.5 | 25.15% | 203 |
| Mul | 224.7 | 19.19% | 186 |
| MatMul | 188.0 | 16.06% | 2 |
| Div | 139.6 | 11.92% | 132 |
| Conv | 121.2 | 10.35% | 79 |
| Neg | 55.9 | 4.77% | 104 |
| Exp | 55.9 | 4.77% | 104 |
| ReduceSum | 34.3 | 2.93% | 29 |
| ReduceMax | 29.7 | 2.54% | 1 |
| **DFT** | **9.3** | **0.80%** | **1** |
| Pad | 8.0 | 0.68% | 1 |
| ReduceL2 | 3.8 | 0.32% | 1 |
| Others | < 4 | < 1% | — |

**Total model time: ~1171 ms**

## Amdahl projection

DFT is 0.8% of model time. Even reducing it to zero:
**Projected speedup = 1 / (1 - 0.008) = 1.008x** — negligible.

The real Perch bottlenecks are elementwise ops (Add/Mul/Div/Neg/Exp = 66%) which benefit
from the SIMD vectorization work Iran is doing in `../onnx-genai-dense-elem`.

## vDSP vs fallback

The DFT kernel processes N=1024 (power-of-two) with `onesided=1`, `axis=-2`.
The batch has 1000 windows. vDSP `vDSP_DFT_zop_CreateSetup` handles this natively.
ORT does not link Accelerate (verified by `otool -L`) — this remains a structural
advantage, though its absolute contribution is small for Perch.

## Numerics bound

vDSP f32 output vs double-precision naive DFT reference: **max absolute error < 1e-2**
(measured threshold for N=1024; typical error is much lower). The FFT radix-2 fallback
matches the naive DFT within **1e-4** absolute tolerance.

## Deliverables

- `crates/onnx-runtime-ep-cpu/src/kernels/dft.rs` — full DFT kernel (vDSP + radix-2 + naive)
- `crates/onnx-runtime-session/tests/perch_dft.rs` — integration test with vDSP reachability proof
- `dispatch_manifest.toml` — two claims (vDSP pow2, radix-2 fallback)
- Counter reachability tests for both paths
- Model deleted after measurement (fetch-measure-delete rule)

## Decision

No further DFT optimization is warranted for Perch. The vDSP path is already
hardware-optimal for the power-of-two case, and the op is <1% of model time.
The investment should go to elementwise vectorization (66% of model time).
