# Pris — LayerNorm test blocker fixes (PR #31973)

**Date**: 2026-08-11
**Author**: Pris (Tester)
**PR**: microsoft/onnxruntime#31973

## Summary

Addressed reviewer blockers B3, B4, N5/N6 in `test_layernorm.cpp`.

### B3 — Platform-gated dispatch assertions

- Added `HasLayerNormKernel()` helper that checks `GetMlasPlatform().LayerNormF32Kernel != nullptr`
- All tests that exercise the SIMD kernel now `GTEST_SKIP()` when no kernel is registered
- On platforms WITH a kernel, the reachability guard survives: `NormSize >= 8` still asserts `used == true`
- Pattern follows existing MLAS convention (`GetMlasPlatform().Avx2Supported_` + `GTEST_SKIP()`)

### B4 — RVV zero-variance accommodation

- RVV kernel uses `E[x²] - mean²` (two-pass) in fp32, not Welford
- For constant input `c`, this computes `c² - c²` which is exactly 0 when `c²` is representable
- For large `c`, fp32 accumulation rounding could produce a tiny residual (positive or negative)
- Test now documents this cross-implementation difference and uses tolerances that accommodate both
- Assertions check: (1) all outputs finite, (2) match fp64 ref within 0.5% rel / 2e-4 abs floor

### N5/N6 — fp64 parity sweep

Grid: `base ∈ {1e3, 1e4, 1e5, 1e6}`, `spread ∈ {1, 1e-1, 1e-2, 1e-3}`, `eps ∈ {1e-5, 1e-6, 1e-12}`, `NormSize ∈ {9, 15, 33, 127, 255, 256, 512, 1024, 2048, 4096}` — 480 cases total, including non-multiples-of-8.

Tolerance: 1e-3 (0.1%) max relative error vs fp64.

**B1 reproduction confirmed**: `base=1e5, spread=1e-2, N=1024, eps=1e-6` → err=1.8151e+01, correctly CAUGHT. The sweep would have detected the ~1000× regression.

### Current status

- 40/41 tests pass. Fp64ParitySweep correctly **fails** (374/480 cases, worst=86.2) against the current kernel, which is expected — Iran's centered two-pass fix hasn't landed yet.
- Once Iran's fix lands, the sweep should go green for any implementation within 0.1% of fp64.
