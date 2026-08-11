# Holden — Independent Re-Review of PR #31973 (AVX2 LayerNorm/RMSNorm)

**Date**: 2026-08-11
**Reviewer**: Holden (Security Engineer, adversarial re-reviewer)
**Status**: No prior involvement with this PR
**Commit reviewed**: `95fd1cdf82` (tip of `nxrt/mlas-avx2-layernorm`)

---

## Verdict: **READY FOR REVIEW** (no blockers)

### SUBSTANTIVE (2)

**S1. Main sweep tolerance has only 12% headroom** — `test_layernorm.cpp:1171`
- `Fp64ParitySweep` worst measured error: 2.23e-02 vs threshold 2.5e-02.
- On a different CPU microarchitecture (different FMA rounding, different
  instruction scheduling), this margin could flip. The B1 check at 5e-2 has
  52% headroom and is fine; the main sweep is fragile.
- **Recommendation**: bump `kMaxRelError` to 3e-2 or 3.5e-2. The 2.23e-02
  worst case occurs at cond≈1e5 (base=1e5, spread=1) where fp32 second-pass
  subtraction inherently loses ~5 digits — this is a precision limit, not a
  kernel bug. A 3% threshold still catches any broken algorithm (>>10%) while
  giving 35%+ headroom.
- **Owner**: Pris

**S2. DISABLED_AdversarialPrecisionReport fails when enabled** — `test_layernorm.cpp:928`
- Scenario 6 (near-fp32-max) produces 100% relative error because
  sum-of-squared-deviations overflows fp32. The test header says "measurement
  tool, not a gate" but the final `EXPECT_LT(worst_avx2, 0.005)` assertion
  contradicts that — it WILL fail if someone enables it.
- **Recommendation**: either (a) remove the assertion and leave only the
  printf table, or (b) exclude Scenario 6 from the assertion. A test that
  fails when you run it, even if DISABLED, creates confusion.
- **Owner**: Pris

### NITS (2)

**N1. Comment says "Welford" in several test names/prints** — `test_layernorm.cpp` passim
- The kernel no longer uses Welford. Variable names like `out_welford`,
  scenario labels "Welford SIMD AVX2" etc. are stale from the prior
  iteration. Not incorrect (the scalar baseline IS Welford) but confusing
  when reading "avx2_welford" which refers to the centered two-pass kernel.

**N2. RMSNorm + MeanOut path accumulates sum in double unnecessarily** — `layernorm_kernel_avx2.cpp:80-91`
- The double sum is only needed for full LayerNorm's centered second pass.
  For RMSNorm with MeanOut, the mean is a pure output — its precision doesn't
  affect normalization. This adds ~1 extra `vcvtps2pd + vaddpd` per 8
  elements to the RMSNorm hot path. Minor perf cost but strictly unnecessary.
- **Owner**: Iran

---

## Detailed Analysis

### Is the double-precision first pass genuine?

**YES.** Independently verified:
- `layernorm_kernel_avx2.cpp:145-150`: `_mm_loadu_ps` → `_mm256_cvtps_pd` →
  `_mm256_add_pd` into `__m256d vsumd`. Each float is widened to double
  *before* addition. This is NOT the B1 shape (fp32 lane accumulation poured
  into a double variable after rounding).
- Scalar tail: `dsum += static_cast<double>(Input[i])` — double throughout.
- Mean: `static_cast<float>(dsum / static_cast<double>(n))` — division in
  double, narrow only at the end.

### Has B1 been eliminated?

**YES.** B1 was caused by per-lane fp32 mean accumulation in Welford's
`mean += delta / count` update. The new kernel has no per-lane fp32
accumulation of any statistic. The mean is computed in double (pass 1);
the variance is computed in fp32 but centered on the accurate mean (pass 2).

The second-pass variance accumulation in fp32 does NOT have the B1 weakness
because it accumulates `(x − mean)²` where `x − mean` is small (centered).
Overflow only occurs when individual deviations are near fp32 max — which is
the near-fp32-max scenario, not a realistic input distribution.

### Accuracy numbers — independently reproduced

| Scenario | Claimed | Measured | Match? |
|---|---|---|---|
| B1 (cond=1e7) | 3.30e-02 | 3.30e-02 | ✅ |
| Sweep worst | ~2.2e-02 | 2.23e-02 | ✅ |
| Realistic LLM | < 1e-4 | < 3.74e-07 | ✅ |
| Old Welford B1 | 2.49e-01 | N/A (not tested) | Taken on trust |

### Test suite integrity

**Would this suite catch B1?** YES — `Fp64ParitySweep`'s explicit B1 check
asserts < 5e-2; old kernel error was 0.249. Clear separation.

**Would it catch a new, different regression?** PARTIALLY.
- Regressions on realistic inputs (LLM activations, benign data): caught by
  `RealisticLLMPrecision` (tol 1e-4) and `LargeNBenignPrecision` (tol 1e-3).
- Regressions at high condition numbers (cond 1e5-1e6): caught by main sweep
  but with thin margin (S1 above).
- Regressions at cond > 1e6: only the B1 regression check at one specific
  point. A new failure at base=1e6, spread=1e-4 would be missed.
- Near-fp32-max overflow: not caught (DISABLED test only, and it fails).

The `cond < 1e6` gate is defensible: at cond ≥ 1e6, `x − mean` in fp32
genuinely loses all signal. The excluded region is NOT "merely inconvenient"
— it is a fundamental fp32 precision boundary. No fp32 kernel can pass there.

**Did the relaxations hollow out the suite?** NO. The vector-normalized error
metric is the correct choice for LayerNorm output (approximately standard
normal, so individual near-zero elements are expected). The condition gate
correctly delineates fp32's capability boundary. The 2.5% tolerance is
appropriate for the included region but could use more headroom (S1).

### Cross-platform safety

**RVV unaffected**: `NormSize < 8` gate is inside `#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)` in `layernorm.cpp:47-53`. RVV kernel registration in `platform.cpp:314` is unconditional within the RISC-V block. ✅

**Silent skip impossible on AVX2 hardware**: `LayerNormF32Kernel` is set to
`&MlasLayerNormKernelAvx2` unconditionally in the AVX2 feature block
(`platform.cpp:515`). If AVX2 is detected, the kernel is always registered.
The only skip path is `NormSize < 8`, which tests explicitly verify. ✅

### Kernel correctness

- **No UB**: no signed integer overflow, no out-of-bounds access.
- **Alignment**: all loads use `loadu` (unaligned-safe). ✅
- **Tail**: `i + 8 <= n` / `i + 4 <= n` guards prevent overread. ✅
- **NormSize boundaries**: N=8 → exactly 1 vector iteration (pass 2), 2
  iterations (pass 1 at stride 4). N=9 → 1 vector + 1 scalar. Verified
  via tests. ✅
- **Null MeanOut/InvStdDevOut**: guarded. ✅
- **Simplified mode**: correctly skips mean subtraction in output loop,
  uses sum-of-squares for denominator. `assert(!Simplified || Bias == nullptr)`
  is debug-only precondition. ✅

### MSVC `/arch:AVX2`

`cmake/onnxruntime_mlas.cmake:242-243` adds `/arch:AVX2` to
`layernorm_kernel_avx2.cpp` on Windows. The file is also added to the
non-Windows AVX2 source list at line 863. ✅

---

## What I verified vs took on trust

| Item | Verified | Method |
|---|---|---|
| 41 tests pass | ✅ | Ran `onnxruntime_mlas_test --gtest_filter="*LayerNorm*"` |
| B1 error = 3.30e-02 | ✅ | Observed in test output |
| Sweep worst = 2.23e-02 | ✅ | Observed in test output |
| Double accumulation genuine | ✅ | Read intrinsics line by line |
| RVV not regressed | ✅ | Read `layernorm.cpp` and `platform.cpp` |
| DISABLED test fails | ✅ | Ran with `--gtest_also_run_disabled_tests` |
| Old Welford B1 = 0.249 | Trust | Not independently measured (old code not present) |
| Speedup claims (14.3×) | Trust | Did not run benchmarks |
| Windows MSVC build | Trust | No Windows environment available |
