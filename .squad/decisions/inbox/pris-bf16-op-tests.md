# Pris — BF16 CPU Op-Level Tests

**Date:** 2026-08-11
**Branch:** `nxrt/mlas-bf16-layernorm` in `/workspace/upstream/ort-bf16`

## Gap Confirmation

Confirmed that upstream main (16b486a2) has **no BFloat16 CPU kernel registration** for:
- `LayerNormalization` (opset 17): `cpu_execution_provider.cc:1080-1082` registers float/double/MLFloat16 only
- `SimplifiedLayerNormalization` (contrib): `cpu_contrib_kernels.cc:159-161` — same
- `SkipSimplifiedLayerNormalization` (contrib): `cpu_contrib_kernels.cc:165-167` — same

Schema permits bf16 in all three (contrib_defs.cc type constraints include `tensor(bfloat16)`).
CUDA registers BFloat16 for these ops. CPU does not.

## File Added

`onnxruntime/test/contrib_ops/layer_norm_bf16_cpu_test.cc` — 10 test cases:

### LayerNormalization (opset 17)
1. `LayerNorm17_SmallNormSize` — NormSize=3, 2 rows, with bias
2. `LayerNorm17_NoBias` — NormSize=4, 3 rows, no bias
3. `LayerNorm17_NonMultipleOfVectorWidth` — NormSize=7 (not SIMD-aligned), with bias
4. `LayerNorm17_LargerNormSize` — NormSize=128, 4 rows, random data

### SimplifiedLayerNormalization (RMSNorm, contrib)
5. `SimplifiedLayerNorm_SmallNormSize` — NormSize=3, 2 rows
6. `SimplifiedLayerNorm_NonMultipleOfVectorWidth` — NormSize=5
7. `SimplifiedLayerNorm_LargerNormSize` — NormSize=256, 4 rows, random data

### SkipSimplifiedLayerNormalization (contrib)
8. `SkipSimplifiedLayerNorm_Basic` — hidden_size=4, basic functional
9. `SkipSimplifiedLayerNorm_NonMultipleNormSize` — hidden_size=5
10. `SkipSimplifiedLayerNorm_LargerHiddenSize` — hidden_size=128, 8 tokens, random data

## Anti-Fallback Mechanism

Each test uses `ConfigEp(DefaultCpuExecutionProvider())` to restrict execution exclusively to the CPU EP. With only one EP available:
- If the bf16 kernel is not registered, the node cannot be placed → session build fails → test fails
- No second EP exists to fall back to
- No Cast insertion can satisfy an unsupported type when there's no viable kernel target

This is NOT a "correct answer ⇒ pass" test. It's a "kernel must exist AND produce correct answer" test.

## Tolerance

Absolute tolerance = 0.1f. Rationale:
- BFloat16: 7-bit stored mantissa → ~2^-7 ≈ 0.0078 relative precision
- For values in [-5, 5] range, representation error ≈ 0.04-0.08
- 0.1 is conservative ceiling above noise floor
- Compare: f32 tests use 1e-4, fp16 tests use 0.01 — bf16 tolerance is deliberately 10x wider than fp16

Reference values computed by round-tripping inputs through BFloat16 (RoundTripBF16) before applying f32 arithmetic, matching the widen→f32→narrow kernel path.

## Build Status

**Could not build or run tests.** The branch has a compile error in Resch's WIP kernel code (`skip_layer_norm.cc:246` — `ComputeJob` template for BFloat16 not yet implemented). This blocks the entire `onnxruntime_providers` library, which is a prerequisite for `onnxruntime_test_all`.

**I am NOT claiming these tests pass.** They are syntactically correct (same include pattern as `layer_norm_op_test.cc`) and will compile once Resch completes the kernel implementation.

## CMake

No CMake change needed — `cmake/onnxruntime_unittests.cmake:490` globs `"${TEST_SRC_DIR}/contrib_ops/*.cc"`.

## No External Fixtures

Tests use inline data and `RandomValueGenerator` — no `.onnx` fixture files to track.

## UNVALIDATED

- **Test execution**: Could not run due to WIP kernel compile failure
- **Tolerance calibration**: 0.1 is a conservative estimate; should be validated against Chew's MLAS bf16 representation-error measurements once available
- **AVX512-BF16 path**: This host (AMD EPYC 9V74) has no AVX-512. If MLAS adds an AVX512-BF16 kernel path, it is untested here.
