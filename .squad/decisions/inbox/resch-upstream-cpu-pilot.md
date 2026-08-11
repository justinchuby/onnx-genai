# Resch Upstream CPU Pilot — Decision Record

**Date:** 2026-08-11
**Author:** Resch (Intel CPU Optimization)
**PR Context:** #763 ranked plan, CPU upstream pilot

## Phase 1 — What x86 Does Today for fp16 MatMul

On x86, `MlasFp16AccelerationSupported()` returns `false` (no `MLAS_F16VEC_INTRINSICS_SUPPORTED` defined for x86 — only ARM64/RISC-V).
`MlasHalfGemmNativePackBSize()` returns `0` on x86 (KleidiAI override is ARM64-only).

Therefore `has_accelerated_half_gemm` is `false` at:
- `matmul.cc:414` → falls through to `math::MatMul<MLFloat16>` call at line 417
- `math_cpu.cc:200` → same check, falls into **Eigen-based fp32-accumulate GEMM**:
  ```cpp
  C_mat.noalias() =
      (B.cast<float>() * A.cast<float>()).cast<Eigen::half>();
  ```

This is numerically sound (fp32 accumulation) and leverages Eigen's vectorised SGEMM.

**Hardware reality:** This host has AVX2 only (`lscpu` shows `avx avx2`, no `avx512*`).
AVX2 provides F16C (half↔float conversion) but NO native fp16 arithmetic.
Native fp16 compute requires AVX512-FP16 (Sapphire Rapids+). Any AVX2 "half GEMM"
would convert to fp32, compute, convert back — identical to what Eigen already does.

**Evidence files:**
- `core/mlas/lib/halfgemm.cpp:322-353` — `MlasHalfGemmBatch` gated by `#ifdef MLAS_TARGET_ARM64`
- `core/mlas/lib/halfgemm.cpp:384-475` — `MLAS_HALF_GEMM_KERNEL_DEFAULT` is scalar fp16↔fp32 element-by-element
- `core/mlas/lib/halfgemm.cpp:48-60` — `MlasFp16AccelerationSupported()` returns false without `MLAS_F16VEC_INTRINSICS_SUPPORTED`
- `core/util/math_cpu.cc:194-228` — Eigen fp32 accumulate fallback for `math::MatMul<MLFloat16>`
- `core/providers/cpu/math/matmul.cc:413-422` — dispatch decision

## Correction: GatherBlockQuantized CPU

The ranked shortlist's candidate #2 (GatherBlockQuantized CPU kernel gap) was **incorrect**.
A CPU kernel exists at `onnxruntime/contrib_ops/cpu/quantization/gather_block_quantized.cc`.
The prior search covered `core/providers/cpu/` and missed `contrib_ops/cpu/`. **Plan doc should be corrected.**

## Phase 2 — Verdict

**(b) fp16 GEMM on x86 is NOT viable.** Adding an AVX2 half GEMM would replicate what Eigen
already does (convert→compute in fp32→convert back). No measurable improvement possible
without AVX512-FP16 hardware.

**Chosen alternative: AVX2 LayerNorm/RMSNorm (SimplifiedLayerNormalization) kernel**

### Why this is a genuine gap

1. `MlasLayerNormF32` exists with dispatch infrastructure (`mlasi.h:741`, `layernorm.cpp`)
2. Only one platform kernel exists: RVV (`riscv64/layernorm_kernel_rvv.cpp`)
3. **No x86 kernel** — `platform.cpp` never sets `LayerNormF32Kernel` for x86
4. The float path falls back to scalar Welford + scalar normalize (`layer_norm_impl.cc:48-100`)
5. The MLFloat16 path (`layer_norm_impl.cc:112-193`) is purely scalar: element-by-element
   `static_cast<float>(input_vec[i])` → scalar variance → scalar normalize → `gsl::narrow_cast<Eigen::half>`
6. AVX2+FMA3 can vectorize both passes: 8-wide `_mm256_loadu_ps`, `_mm256_fmadd_ps` for
   sum-of-squares reduction, `_mm256_mul_ps` + `_mm256_fmadd_ps` for normalize+scale+bias

### File scope (5 files, ≤10 limit)

| File | Change |
|------|--------|
| `onnxruntime/core/mlas/lib/layernorm_kernel_avx2.cpp` | **NEW** — AVX2 kernel implementation |
| `onnxruntime/core/mlas/lib/mlasi.h` | Declaration of `MlasLayerNormKernelAvx2` |
| `onnxruntime/core/mlas/lib/platform.cpp` | Dispatch wiring in AVX2 block |
| `cmake/onnxruntime_mlas.cmake` | Add to Windows + Linux AVX2 source lists |
| `.squad/decisions/inbox/resch-upstream-cpu-pilot.md` | This decision record |

## Phase 3 — What Was Implemented

An AVX2 `MlasLayerNormKernelAvx2` kernel matching the `MLAS_LAYERNORM_F32_KERNEL` signature,
following the structure of `layernorm_kernel_rvv.cpp`:

- **Pass 1:** Single-pass vectorized sum + sum-of-squares using 256-bit registers (8 floats/iter),
  horizontal reduction via `_mm256_extractf128_ps` + `_mm_add_ps` cascade, scalar tail
- **Pass 2:** Vectorized normalize with three branches (Simplified/no-bias/with-bias),
  using `_mm256_fmadd_ps` for fused multiply-add in the bias case, scalar tail
- **Fail-closed:** Guarded by `#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)`.
  Dispatch only activates when AVX2 is detected by CPUID in `platform.cpp`.

## Phase 4 — Build Verification

**Successfully compiled the full `onnxruntime_mlas` static library** via:
```
cmake -S cmake -B build_mlas -Donnxruntime_BUILD_UNIT_TESTS=OFF -Donnxruntime_ENABLE_CPUINFO=OFF
cmake --build build_mlas --target onnxruntime_mlas -j$(nproc)
```
Build output confirms `layernorm_kernel_avx2.cpp.o` was compiled with `-mavx2 -mfma -mf16c`
and linked into `libonnxruntime_mlas.a`. **Zero warnings, zero errors.**

No runtime benchmark was performed — this host has the right ISA (AVX2) but no ORT test
binary was built. I cannot claim any speedup without measurement.

## Entry Points for Pris

Pris should drive these entry points in her test/benchmark harness:

1. **MLAS unit test:** Call `MlasLayerNormF32()` (declared in `mlas.h:1703`) with various
   `NormSize` values (1, 7, 8, 15, 16, 127, 128, 1024), both `Simplified=true` (RMSNorm)
   and `Simplified=false` (LayerNorm), with and without Bias.
   Compare against a scalar reference within `1e-5` tolerance.

2. **Operator-level test:** `SimplifiedLayerNormalization` and `LayerNormalization` with
   float inputs through the CPU EP. The operator calls `MlasLayerNormF32` at
   `layer_norm_impl.cc:48`.

3. **Function signature:**
   ```cpp
   void MLASCALL MlasLayerNormKernelAvx2(
       const float* Input, const float* Scale, const float* Bias,
       float* Output, float* MeanOut, float* InvStdDevOut,
       size_t NormSize, float Epsilon, bool Simplified);
   ```

## Blockers

None for the current scope. Future work:
- An fp16 LayerNorm MLAS kernel (accepting MLFloat16 I/O with internal fp32) would
  further improve the MLFloat16 path in `layer_norm_impl.cc:112-193`, which is still
  purely scalar. This would require a new `MLAS_LAYERNORM_F16_KERNEL` typedef.
