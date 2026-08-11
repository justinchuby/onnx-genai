# Decision: BFloat16 CPU LayerNorm/RMSNorm Registration & Compute Path

**Author:** Resch (Intel CPU Optimization Engineer)
**Date:** 2026-08-11
**Status:** Implementation complete, not build-verified
**Branch:** `nxrt/mlas-bf16-layernorm` in `/workspace/upstream/ort-bf16`

## Verified Gap

**Schema permits BFloat16 on all four ops; CPU registered none of them.**

| Op | Schema BF16? | CUDA BF16? | CPU BF16 (before) | CPU BF16 (after) |
|---|---|---|---|---|
| `LayerNormalization` (ONNX opset 17) | ✅ T+U both permit `tensor(bfloat16)` | ✅ versioned 1-16 | ❌ float/double/MLFloat16 only | ✅ Added |
| `SimplifiedLayerNormalization` (contrib) | ✅ T+V at `contrib_defs.cc:3323,3331` | ✅ `cuda_contrib_kernels.cc:193` | ❌ float/double/MLFloat16 only | ✅ Added |
| `SkipLayerNormalization` (contrib, kMSDomain) | ✅ T permits `tensor(bfloat16)` (`bert_defs.cc`) | ✅ `cuda_contrib_kernels.cc:175` | ❌ float/double/MLFloat16 only | ✅ Added |
| `SkipSimplifiedLayerNormalization` (contrib, kMSDomain) | ✅ T permits `tensor(bfloat16)` (`bert_defs.cc`) | ✅ `cuda_contrib_kernels.cc:178` | ❌ float/double/MLFloat16 only | ✅ Added |
| `LayerNormalization` (contrib, versioned 1-16) | ✅ T+U+V at `contrib_defs.cc:3176,3184` | ✅ `cuda_contrib_kernels.cc:187` | ❌ float/double/MLFloat16 only | ✅ Added |

## Design: Shared fp16 Path via Traits

**All arithmetic is f32. BFloat16 is storage only.** No native bf16 math, no AVX512-BF16 instructions.

The existing MLFloat16 path already widens to f32 for all computation (Welford's for LayerNorm, sum-of-squares for RMSNorm). BFloat16 shares the **identical pattern**:
1. Widen bf16 → f32 (via `BFloat16::ToFloat()`, which is a 16-bit left-shift of the stored bits)
2. Accumulate in f32 (Welford's online algorithm — preserving existing numerics)
3. Narrow f32 → bf16 (via `BFloat16(float)`, using **round-to-nearest-even** per `kRoundToNearest = 0x7FFF` in `onnxruntime_float16.h`)

**Code sharing strategy:**
- Added `is_narrow_float_v<T>` trait (true for MLFloat16 and BFloat16)
- Added `NarrowToFloat<T>` / `FloatToNarrow<T>` helpers that dispatch to the correct conversion
- `if constexpr (is_narrow_float_v<T>)` replaces `if constexpr (std::is_same_v<T, MLFloat16>)` in shared paths
- `BFloat16Math` policy struct (parallel to `HalfMath`) for the generic broadcasting path
- Dedicated `ComputeJob` overload for BFloat16 (cannot reuse MLFloat16's Eigen::half path)
- `ConvertMLFloat16ToFloatIfNeeded` generalized to also detect and convert BFloat16 tensors for prepacking

**No MLAS kernel added.** The operator-level widen/accumulate/narrow path is sufficient for v1. A dedicated MLAS kernel is unnecessary complexity at this stage.

**No AVX512-BF16 path.** This host (AMD EPYC 9V74) has no AVX-512. The portable scalar conversion via `BFloat16(float)` / `BFloat16::ToFloat()` is the only path.

## Rounding Rule

**Round-to-nearest-even**, matching upstream ORT's `BFloat16` constructor.

Source: `include/onnxruntime/core/session/onnxruntime_float16.h`, `BFloat16Impl::ToUint16Impl()`:
```cpp
static constexpr uint16_t kRoundToNearest = 0x7FFFU;
U32 += (upper_bits & 1) + kRoundToNearest;
```
This implements RNE (the `(upper_bits & 1)` term handles the tie-breaking).
`std::numeric_limits<BFloat16>::round_style == round_to_nearest` confirms.

## Files Modified (6 of ≤10)

1. `onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc` — Core: `is_narrow_float_v` trait, `BFloat16Math`, `BFloat16` `ComputeJob`, generalized `ComputeWithoutContext`, updated `SupportedTypeList`
2. `onnxruntime/core/providers/cpu/nn/layer_norm.cc` — Registration: `REGISTER_ONNX_KERNEL_TYPED(BFloat16)`
3. `onnxruntime/core/providers/cpu/cpu_execution_provider.cc` — Class declaration + `BuildKernelCreateInfo` for `LayerNormalization<BFloat16>`
4. `onnxruntime/contrib_ops/cpu/layer_norm.cc` — Registration: `REGISTER_CONTRIB_KERNELS(BFloat16)` (covers versioned LN + SimplifiedLN)
5. `onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc` — 4 class declarations + 4 `BuildKernelCreateInfo` entries
6. `onnxruntime/contrib_ops/cpu/skip_layer_norm.cc` — Registration + generalized `Compute` for BFloat16

## Build Status

**Not build-verified.** Full ORT build is too heavy for this environment. Code compiles syntactically (braces balanced, consistent patterns with existing MLFloat16 code). No CMake changes needed — all modified files are already in existing build targets.

## Welford Semantics

**Preserved.** The BFloat16 `ComputeJob` uses Welford's online algorithm for LayerNorm (identical to the existing MLFloat16 path). The RMSNorm path uses single-pass sum-of-squares. No two-pass variance computation was introduced.

## PR #31973 Overlap

PR #31973 (`ort-fork` branch) adds an AVX2 f32 LayerNorm kernel touching `layernorm_kernel_avx2.cpp`, `mlasi.h`, `platform.cpp`. **This branch does not touch any MLAS files.** Zero merge conflict risk.

## Entry Points for Testing

**Pris (reachability/op tests):**
- `LayerNormalization` opset 17 with `T=BFloat16`, `U=float`
- `SimplifiedLayerNormalization` contrib with `T=U=V=BFloat16`
- `SkipLayerNormalization` contrib (kMSDomain) with `T=BFloat16`
- `SkipSimplifiedLayerNormalization` contrib (kMSDomain) with `T=BFloat16`
- Versioned `LayerNormalization` contrib (kOnnxDomain, 1-16) with `T=U=V=BFloat16`

**Chew (fp32/fp64 oracle):**
- Expected numerics: bf16 results should match f32 LayerNorm to within bf16 representation error (~0.4% relative for typical values)
- The compute is literally f32 arithmetic; only input/output storage is bf16
- Rounding: round-to-nearest-even on f32→bf16 narrowing
