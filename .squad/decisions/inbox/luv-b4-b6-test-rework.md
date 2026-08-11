# B4 + B6: BF16 LayerNorm Test Rework

**Date**: 2026-08-11
**Author**: Luv (Code Reviewer)
**PR**: microsoft/onnxruntime#31974
**Branch**: `nxrt/mlas-bf16-layernorm` in `/workspace/upstream/ort-bf16`

## B4 — Deleted `test_layernorm_bf16.cpp` (1037 lines)

**Decision**: Delete entirely.

**Rationale**: The file lived in `test/mlas/unittest/` and used the MLAS test
harness (`MlasTestBase`, `MlasTestFixture`) but called **zero MLAS APIs**.
It tested:
- BFloat16 round-to-nearest-even rounding rules
- fp64 oracle consistency (Welford vs two-pass)
- Representation error floor measurements
- Adversarial cases (catastrophic cancellation, denormals, near-max)
- BF16 vs FP16 precision comparison

None of this validates code touched by PR #31974 (CPU EP kernel registration).
The file registered 45 tests under `BF16Rounding` and `BF16LayerNormPrecision`
test suites — these were the "45 MLAS kernel tests" claim, which was false.

No bf16 rounding/conversion helpers were relocated because the PR does not
introduce any such helpers — the existing `BFloat16` class in
`core/common/float16.h` is tested elsewhere.

## B6 — New Coverage in `layer_norm_bf16_cpu_test.cc`

The file was rewritten (10 → 17 tests) to cover all 5 BF16 registrations:

| Registration | Tests |
|---|---|
| Core `LayerNormalization` opset 17 | 4 (Small, NoBias, NonMultiple, Larger) |
| Core opset 17 + Mean/InvStdDev stats | 2 (Small, Larger) |
| Contrib `LayerNormalization` opset 1–16 | 2 (Small, Larger) |
| Contrib `SimplifiedLayerNormalization` | 3 (Small, NonMultiple, Larger) |
| Contrib `SkipLayerNormalization` | 3 (Basic, NoBeta, Larger) |
| Contrib `SkipSimplifiedLayerNormalization` | 3 (Basic, NonMultiple, Larger) |

### New: SkipLayerNormalization (non-simplified)
3 tests covering the `SkipLayerNormalization` contrib op with BFloat16,
which was registered but untested.

### New: Contrib LayerNormalization opset 1–16
2 tests covering the versioned contrib registration (distinct from opset 17).
These also assert Mean and InvStdDev as float outputs.

### New: Mean/InvStdDev float stat outputs (B5 regression test)
2 tests assert Mean and InvStdDev at **f32-grade tolerance (1e-5)**, not
bf16-grade. These are the regression tests for B5 (stat-narrowing bug).

The pre-B5 code at `layer_norm_impl.cc:291-295` does:
```cpp
mean_data[task_idx] = BFloat16(mean);       // ← loses ~0.4% at unit scale
inv_std_dev_data[task_idx] = BFloat16(1.0f / std_dev);
```
With kF32StatTolerance = 1e-5, this WILL FAIL because bf16 round-trip error
at unit scale is ~0.0078, far exceeding 1e-5. The test currently passes
because Iran's B5 fix (WriteStat) already landed in this branch.

## Tolerance Policy

| Output type | Tolerance | Rationale |
|---|---|---|
| BFloat16 Y | 0.016 abs (2 bf16 ULP at unit scale) | 7-bit mantissa; 0.5 ULP rep floor + ≤1 ULP accumulation |
| float Mean | 1e-5 abs | U=float per ONNX spec; catches bf16 round-trip (~0.4%) |
| float InvStdDev | 1e-5 abs | Same as Mean |

No blanket tolerance — bf16-typed outputs get bf16-grade, float-typed stats get f32-grade.

## Persona Comments Removed

- Line 22: `"based on Chew's empirical measurements, 45/45 kernel tests"` → replaced with the measured values and rationale
- Line 28: `"the accumulation error Chew measured"` → replaced with description of the error source
- Header comment: `"Resch's implementation"` reference removed (was in the deleted MLAS file)

## Test Counts

| Suite | Before | After |
|---|---|---|
| `LayerNormBFloat16CpuTest` | 10 | **17** |
| Full `*LayerNorm*` suite | 89 | **96** |
| MLAS bf16 tests (deleted) | 45 | **0** |

**Honest public claim**: 17 CPU EP operator tests covering all 5 BF16
LayerNorm registrations, with f32-grade stat accuracy verification.
