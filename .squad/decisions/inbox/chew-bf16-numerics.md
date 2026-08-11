# Chew — BF16 LayerNorm/RMSNorm Numerics Report

**Date:** 2026-08-11  
**Branch:** `nxrt/mlas-bf16-layernorm` (worktree `/workspace/upstream/ort-bf16`)  
**File:** `onnxruntime/test/mlas/unittest/test_layernorm_bf16.cpp`

## Summary

45 tests pass covering BF16 rounding validation, fp64 oracle consistency, representation-error floor measurement, adversarial cases, and widen-accumulate-narrow simulation. Resch's kernel has not landed yet, so kernel-vs-oracle comparison is deferred — the oracle and test infrastructure are ready.

## 1. Rounding Rule Verdict: **PASS — Round-to-Nearest-Even confirmed**

```
BF16 rounding: tie-to-even 336/336 correct, directional 672/672 correct
ORT bf16 vs truncation: 4966/10000 differ, ORT closer/equal in 4966/4966 differing cases
```

ORT's `BFloat16` constructor implements RNE. The mechanism: `U32 += (upper_bits & 1) + 0x7FFF` adds 0x7FFF plus the LSB of the result, which is the standard RNE bias for bf16. Truncation differs in ~50% of cases; ORT is always at least as close. No rounding-rule inconsistency with the rest of ORT — the same `BFloat16Impl::ToUint16Impl` is used everywhere.

## 2. BF16 Representation-Error Floor

The unavoidable error from quantizing fp64 oracle outputs to bf16:

| N      | max_abs    | max_rel    | max_ulp | rms        |
|--------|------------|------------|---------|------------|
| 64     | 3.792e-03  | 3.629e-03  | 0       | 1.458e-03  |
| 256    | 3.878e-03  | 3.791e-03  | 0       | 1.553e-03  |
| 1024   | 3.894e-03  | 3.878e-03  | 0       | 1.648e-03  |
| 4096   | 3.885e-03  | 3.699e-03  | 0       | 1.613e-03  |
| 16384  | 3.874e-03  | 3.836e-03  | 0       | 1.590e-03  |
| 65536  | 3.904e-03  | 3.801e-03  | 0       | 1.584e-03  |

**Key finding:** max_abs ≈ 3.9e-3 ≈ 0.5 × bf16 ULP for values near 1.0 (bf16 ULP at 1.0 = 2^-7 ≈ 7.8e-3). The 0 max_ulp means the fp64→bf16 quantization never exceeds 0.5 ULP — exactly as expected for RNE. This is the floor; no kernel can do better.

## 3. Widen-Accumulate-Narrow Simulation

Simulating f32 two-pass arithmetic on bf16 inputs:

| N      | simplified | kernel max_ulp | err_above_floor |
|--------|-----------|----------------|-----------------|
| 64     | 0         | 0              | 0               |
| 256    | 0         | 0              | 0               |
| 1024   | 0         | 0              | 0               |
| 4096   | 0         | 0              | 0               |
| 4096   | 1         | 1              | 1               |
| 16384  | 0         | 1              | 1               |
| 65536  | 0         | 1              | 1               |
| 65536  | 1         | 1              | 1               |

**Verdict:** f32 two-pass accumulation on bf16 inputs produces at most **1 bf16 ULP** of error above the representation floor, even at N=65536. This is excellent. The error appears only for large N where f32 accumulation order matters.

**Recommended tolerance for Resch's kernel: ≤2 bf16 ULP** (allows 1 ULP for the kernel's accumulation + 1 ULP margin for SIMD reordering). In absolute terms, this is ≈1.6e-2 for values near 1.0.

## 4. Adversarial Cases

### Catastrophic cancellation (large mean, tiny variance)
```
N=64-4096: mean=1002.0, variance=4.0, inv_std=0.5, rep_max_ulp=0
```
No issues — bf16's limited mantissa means the "perturbations" at 1 ULP around 1000.0 are actually ±4.0 (the bf16 step at that magnitude), so variance is not near-zero. This is fundamentally different from fp16/fp32 catastrophic cancellation. **The catastrophic cancellation scenario that broke two-pass on #31973 does not apply to bf16 with the same severity** because the coarse quantization makes the variance relatively large.

### Near-zero variance (identical inputs)
```
N=64-1024: mean=42.0, variance=0.0, output ≈ 0.0
```
Epsilon protects against division by zero. All outputs finite.

### Denormal inputs
```
N=64: mean=2.985e-39, var=2.878e-78, all outputs finite
```

### Near-max values
```
N=64: base=9.984e+03, mean=9.982e+03, var=8.060e+03, all finite
```

### High dynamic range (1e-3 to 1e3)
```
N=64-1024: rep_max_ulp=0, all finite
```

## 5. BF16 vs FP16 Cross-Check

```
BF16 max input err: 1.561e-02, FP16 max input err: 1.951e-03, Ratio: 8.0×
```

Exactly as expected: bf16 has 8 mantissa bits vs fp16's 11 bits, so 2^3 = 8× coarser quantization.

## 6. Tolerance Recommendation

| Parameter | Recommended tolerance |
|-----------|----------------------|
| Output elements | ≤2 bf16 ULP (≈1.6e-2 at unit scale) |
| Mean | ≤1 bf16 ULP of the mean value |
| Inv_std | ≤2 bf16 ULP of the inv_std value |

**Do NOT use the f32 test's 1e-4 tolerance** — that is meaningless for bf16 where 1 ULP ≈ 7.8e-3.

## 7. Verdict on Approach

**ACCEPT** — the widen→f32-accumulate→narrow approach is sound for bf16 LayerNorm/RMSNorm. The f32 arithmetic provides more than enough precision; the bottleneck is entirely the bf16 I/O quantization (≤0.5 ULP floor), with kernel accumulation adding at most 1 ULP even at N=65536. The rounding rule is correct (RNE) and consistent across ORT.

**Pending:** Resch's actual kernel landing — at that point, the oracle tests will activate kernel-vs-oracle comparison. The test file is structured to add `MlasLayerNormBF16()` calls when the API exists.

## 8. CI Fix — Dead Functions Removed (2026-08-11)

**Root cause of 46 CI failures:** Two unused static functions in `test_layernorm_bf16.cpp` triggered `-Werror=unused-function`:

1. **`BF16Ulp` (was line 78)** — computed the float-valued ULP magnitude of a bf16 value. All test tolerances use integer ULP distance via `BF16UlpDistance` instead. Genuinely leftover scaffolding; removed.
2. **`ReportErrors` (was line 445)** — private static method on `BF16LayerNormPrecisionTest` that formatted error decomposition. Was intended for kernel-vs-oracle comparison when Resch's kernel lands, but no test method calls it yet. Removed now; will be re-added when the kernel hook activates.

**Sweep of all PR-touched files:** No other `-Werror` issues found. Implementation files (`layer_norm_impl.cc`, `layer_norm.cc`, `skip_layer_norm.cc`, `cpu_contrib_kernels.cc`, `cpu_execution_provider.cc`) properly use `ORT_UNUSED_PARAMETER` for conditionally-unused parameters. No sign-compare, shadowing, or unused-variable warnings.

**Minimal-build consideration:** The test file has no conditional compilation (`#ifdef`), so no code becomes dead under minimal build flags (exceptions disabled, reduced types). The implementation files gate unused parameters with `ORT_UNUSED_PARAMETER`.

**Validation:** Built without `--compile_no_warning_as_error` — clean. All 45 MLAS bf16 tests pass.
