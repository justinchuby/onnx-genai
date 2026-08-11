# Decision: CUDA Candidate CPU-Reference Test Harness

**Author:** Pris (Tester)
**Date:** 2026-08-11
**Status:** Implemented — pending Batty's candidate selection and CMake wiring

## What was built

File: `onnxruntime/test/contrib_ops/cuda_kernels/cuda_candidate_cpu_reference_test.cc`
Location: `/workspace/upstream/ort-cuda` worktree, branch `nxrt/cuda-kernel-audit`

A 718-line GTest harness covering **both** CUDA kernel candidates with CPU-only validation:

### MatMulNBits int4 (Section 2) — 11 tests
- **DequantRoundTrip** (6 tests): Independent int4 pack/unpack/dequant verified against scalar reference. Covers block-128 standard, no-zero-point, non-multiple-of-block (K=200), small block (32), single-element, exact-boundary.
- **ExtremeScales** (1 test): Tiny (1e-30) and huge (1e+20) scale edge cases.
- **MatMul reference** (4 tests): Full GEMV (M=1) and small-batch matmul against independently-dequantized weights. Covers block-128 with/without zero points, non-multiple K, minimal (1×1×1).

### QMoE parallel routing (Section 3) — 7 tests
- Serial vs parallel-style top-k comparison (3 tests): Verifies serial repeated-argmax matches `partial_sort`-based selection for top-1, top-2, and top-2 without normalization.
- **Adversarial tie-breaking** (3 tests): All-equal logits, partial ties (experts 2/4/6 tied), boundary ties (4 experts at 1.0, top-3). All verify lower-index-wins convention.
- **Weight normalization** (1 test): Asserts selected weights sum to 1.0.
- **Stress** (1 test): 64 experts, 16 tokens, top-4.

### GPU honesty boundary (Section 4) — 3 tests
All gated with `SKIP_IF_NO_GPU()` macro that emits:
```
[UNVALIDATED] No CUDA device detected. This test requires real GPU hardware
and CANNOT be considered validated without it. See onnx-genai#768...
```
These are **intentional FAIL()** stubs — they will never silently pass.

### Reachability stubs (Section 5) — 2 tests (USE_CUDA only)
Skeleton registration checks compiled only when CUDA EP is linked.

## Upstream conventions observed

1. **CUDA gating**: `HasCudaEnvironment(min_arch)` from `test/common/cuda_op_test_utils.h`. Returns false when `DefaultCudaExecutionProvider()` is null. Tests `GTEST_SKIP()` explicitly.
2. **CPU reference pattern**: `matmul_4bits_test.cc` lines 42-73: `QuantizeDequantize()` uses MLAS to quantize, then builds expected output with a triple-loop matmul. MoE test has parallel CPU/CUDA `OpTester` paths (lines 40-93 vs 96-128).
3. **SM90 validation test** (`matmul_nbits_sm90_validation_test.cc`): Pure logic test with no GPU — our model. Compiled into `onnxruntime_providers_cuda_ut` module.
4. **Build target**: `onnxruntime_providers_cuda_ut` (files under `contrib_ops/cuda_kernels/`), requires `onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS`.

## CMake change needed (Batty to add)

In `cmake/onnxruntime_unittests.cmake`, add to the `onnxruntime_providers_cuda_ut` source list:
```cmake
${TEST_SRC_DIR}/contrib_ops/cuda_kernels/cuda_candidate_cpu_reference_test.cc
```

## What could NOT be validated

- **No gtest on this host** — cannot compile standalone. File follows exact patterns of `matmul_nbits_sm90_validation_test.cc` which compiles in ORT's cmake build.
- **No GPU** — all 3 GPU-required tests skip with `[UNVALIDATED]`.
- **No CUDA toolkit** — registration tests (`USE_CUDA` guard) are compiled out.
- **Candidate not yet selected** — both candidates covered; unused section can be trimmed after Batty reports.

## What remains

1. Batty selects candidate → trim unused candidate's tests
2. CMake entry added → compile in ORT build
3. GPU CI run → GPU-required tests execute and fill the coverage gap
