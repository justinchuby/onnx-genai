# Upstream ORT ARM Inventory

**Date**: 2026-08-11
**Author**: Luba (ARM CPU / QNN EP)
**Requested by**: @justinchuby
**Upstream ref**: `microsoft/onnxruntime` main @ `16b486a2` (read-only worktree at `/workspace/upstream/ort-cuda`)

---

## Executive Summary

**ARM is upstream ORT's best-supported SIMD target.** After thorough inspection, upstream MLAS carries 50+ ARM-specific source files (NEON kernels, SVE elementwise, KleidiAI integration, 29 aarch64 assembly files, 18 Windows arm64 assembly files), covering GEMM, convolution, quantization, attention, activation, softmax, rotary embedding, and more — in both fp32 and fp16. Our project has substantial Rust NEON intrinsics (~8 kernel files), but these are **not directly upstreamable** to ORT's C++ codebase. There are a few narrow x86→ARM asymmetric gaps, but none survive the combined filter of impact, novelty, testability, and acceptance likelihood.

**Recommendation: DECLINE.** No ARM upstream candidate survives scrutiny. This is the honest outcome.

---

## 1. Our ARM-Specific Work (onnx-genai)

We have **genuine, non-trivial NEON intrinsics** in our Rust CPU EP — this is not merely "portable code that compiles on ARM." Key files (all under `crates/onnx-runtime-ep-cpu/src/kernels/`):

| File | ARM content | NEON intrinsic use |
|------|-------------|-------------------|
| `dense_elementwise.rs` | f16 widen/narrow via `vcvt_f32_f16`/`vcvt_f16_f32`, `relu_f32_neon`, `clip_f32_neon` | `vmaxq_f32`, `vminq_f32`, `vcvt_f32_f16`, `vcvt_f16_f32` |
| `activations.rs` | `silu_f32_neon` — 4-wide SiLU with polynomial exp approximation | `vdupq_n_f32`, `vmulq_f32`, `vaddq_f32`, `vnegq_f32`, etc. |
| `accelerate_gemm.rs` | Extensive f16↔f32 NEON conversion helpers, GEMV accumulation | `float32x4_t` loads, `vcvt` intrinsics, `vfmaq_f32` |
| `half_gemm.rs` | NEON-accelerated fp16 GEMM tiling paths | `vfmaq_f32`, `vcvt_f32_f16`, accumulation intrinsics |
| `sdpa.rs` | NEON-accelerated scaled dot-product attention | Extensive `aarch64::*` intrinsics (22 cfg-gated blocks) |
| `matmul_nbits.rs` | **Most substantial**: NEON int4/int8 quantized matmul dequant+accumulate | `int8x16_t`, `int32x4_t`, `vdotq_s32` (dotprod), 60+ aarch64 references |
| `conv_ref.rs` | NEON convolution inner loops | `vfmaq_f32`, `vaddq_f32` |
| `relu.rs` | NEON ReLU | `vmaxq_f32` |
| `selection.rs` | NEON selection/gather | 5 aarch64 references |

Supporting infrastructure:
- `crates/onnx-runtime-cpuinfo/src/lib.rs`: NEON/SVE/SVE2/dotprod/fp16_arith detection via cpuinfo (lines 115–200)
- `crates/onnx-genai-ort/src/session/env_config.rs`: Windows ARM64 concurrency tuning (line 130+)
- `crates/mlas-sys/build.rs`: ARM64 cross-compilation support

**However**: All of this is **Rust code using `std::arch::aarch64`**, not C/C++. Upstream ORT's MLAS is C++ with hand-written NEON intrinsics and assembly. Our Rust implementations are **not directly portable** — upstreaming would require a complete rewrite in C++, which is not "upstreaming our work" but "writing new C++ kernels inspired by our Rust ones."

---

## 2. Upstream ORT ARM Coverage (Verified)

### 2.1 NEON Kernels (C++ intrinsics)

| Category | Files | Notes |
|----------|-------|-------|
| **GEMM** | `halfgemm_kernel_neon.cpp`, `halfgemm_kernel_neon_fp16.cpp`, `hgemm_kernel_neon.cpp`, `sbgemm_kernel_neon.cpp` | fp16, fp32, bf16 GEMM |
| **Quantized GEMM** | `qgemm_kernel_neon.cpp`, `qnbitgemm_kernel_neon.cpp`, `sqnbitgemm_kernel_neon_fp32.cpp`, `sqnbitgemm_kernel_neon_int8.cpp`, `sqnbitgemm_kernel_neon_int8_2bit.cpp`, `sqnbitgemm_kernel_neon_int8_i8mm.cpp` | int4/int8 NBits, i8mm |
| **HalfPrec NBits** | `hqnbitgemm_kernel_neon_fp16.cpp`, `hqnbitgemm_kernel_neon_fp16_8bit.cpp` | fp16 output quantized matmul |
| **Convolution** | `sbconv_kernel_neon.cpp`, `sconv_nchwc_kernel_neon.cpp` | bf16 and fp32 conv |
| **Attention** | `softmax_kernel_neon.cpp`, `softmax_kernel_neon_fp16.cpp` | NEON softmax |
| **Activations** | `activate_fp16.cpp`, `erf_neon_fp16.cpp`, `gelu_neon_fp16.cpp` | fp16 activations |
| **Elementwise** | `eltwise_kernel_neon.cpp` (Add_Fp16 only), `cast_kernel_neon.cpp` | Thin; most elementwise is SVE |
| **Other** | `rotary_embedding_kernel_neon.cpp`, `rotary_embedding_kernel_neon_fp16.cpp`, `qkv_quant_kernel_neon.cpp`, `spool_nchwc_kernel_neon.cpp` | RoPE, QKV quant, pooling |

### 2.2 Assembly (aarch64 .S — 29 files)

`SgemmKernelNeon.S`, `SgemvKernelNeon.S`, `SbgemmKernelNeon.S`, `HalfGemmKernelNeon.S`, `QgemmU8X8KernelNeon.S`, `QgemmS8S8KernelNeon.S`, `QgemmS8S8KernelSdot.S`, `QgemmU8X8KernelUdot.S`, `SymQgemmS8KernelNeon.S`, `SymQgemmS8KernelSdot.S`, `ConvSymS8KernelNeon.S`, `ConvSymU8KernelNeon.S`, `ConvSymU8KernelDot.S`, `DepthwiseQConvKernelSize9Neon.S`, `DepthwiseQConvSymS8KernelNeon.S`, `DepthwiseQConvSymU8KernelNeon.S`, `SconvKernelNeon.S`, `SconvKernelNeonBf16.S`, `SconvDepthwiseKernelNeon.S`, `SconvDepthwiseKernelNeonBf16.S`, `SconvPointwiseKernelNeon.S`, `SconvPointwiseKernelNeonBf16.S`, `SconvNchwcKernelNeon.S`, plus Windows arm64 .asm variants.

### 2.3 SVE (Scalable Vector Extension)

`onnxruntime/core/mlas/lib/sve/` — verified dispatch covers:
- **Erf**, **Logistic**, **Tanh**, **Exp**, **SumExp**, **ReduceMaximum**, **ReduceMinimumMaximum**, **Softmax**, **LogSoftmax** (all fp32)
- **Tanh FP16**, **Erf FP16**, **Gelu FP16** (TanhArg + Scale + Combine)
- Plus `elementwise_sve_asm.S` portable machine-code variant

### 2.4 KleidiAI Integration

`onnxruntime/core/mlas/lib/kleidiai/` — 6 files:
- `sgemm_kleidiai.cpp`, `sbgemm_kleidiai.cpp`, `halfgemm_kleidiai.cpp`, `qgemm_kleidiai.cpp`, `convolve_kleidiai.cpp`, `halfconv_kleidiai.cpp`

With extensive micro-kernel assembly under `kleidiai/kai/ukernels/matmul/` covering f32, f16, bf16, int4, int8 with NEON dotprod, i8mm, and SVE variants.

---

## 3. Gap Analysis: x86 Kernels Without ARM Equivalent

| Op / Kernel | x86 Path | ARM Path | Verdict | Notes |
|------------|----------|----------|---------|-------|
| **FlashAttention / GQA** | `flashattn.cpp` (AMD64, ~1650 LOC) | Falls to standard attention path | **PARTIAL** | Issue #29613 (closed) fixed L2 cache detection that silently disabled flash attention on Linux/aarch64. Flash attention now works on ARM via the CPU EP's C++ attention kernels, just without MLAS-level SIMD flashattn. Not a gap in functionality, only in optimization level. |
| **SiLU (fused)** | `silu.cpp` → `GetMlasPlatform().SiluKernelRoutine` (AMD64/RISCV64) | Falls to two-pass: `MlasComputeLogistic` + `MlasEltwiseMul` | **PARTIAL** | Issue #29076 (open): "CPU EP does not fuse float16 Swish/SiLU to QuickGelu (slow on ARM)". Known gap, already tracked upstream. |
| **Logistic (NEON)** | SSE2 intrinsics in `logistic.cpp` | SVE path exists; NEON-only falls to scalar | **PARTIAL** | SVE covers Graviton3+/newer Cortex. NEON-only (Apple Silicon, Snapdragon) uses scalar fallback. |
| **Tanh (NEON)** | SSE2 intrinsics in `compute.cpp` | SVE path exists; NEON-only falls to scalar | **PARTIAL** | Same situation as logistic. |
| **Q4GEMM** | `q4gemm_avx512.cpp` | No ARM equivalent | **GAP** | Q4 block-quantized GEMM only has AVX-512. However, `qnbitgemm_kernel_neon.cpp` and `sqnbitgemm_kernel_neon_*.cpp` cover the newer NxBit quantized GEMM format on NEON, which is the actively maintained path. Q4GEMM is legacy. |
| **Saturation check** | x86-specific | None | **GAP (trivial)** | Debugging utility, not performance-critical. |

---

## 4. In-Flight Upstream ARM Work

| PR | Title | Status | Impact |
|----|-------|--------|--------|
| [#31146](https://github.com/microsoft/onnxruntime/pull/31146) | SVE i8mm QGEMM kernels | **Open** | Extends SVE quantized GEMM coverage |
| [#31143](https://github.com/microsoft/onnxruntime/pull/31143) | Route fp32 SGEMM to KleidiAI SVE | **Open** | SVE SGEMM acceleration |
| [#26027](https://github.com/microsoft/onnxruntime/pull/26027) | SVE support for Sgemm kernel | **Open** | Original SVE SGEMM effort |
| [#31145](https://github.com/microsoft/onnxruntime/pull/31145) | SVE elementwise + FEXPA exp | **Closed** | Already merged or superseded |
| [#29076](https://github.com/microsoft/onnxruntime/issues/29076) | FP16 Swish/SiLU QuickGelu fusion | **Open issue** | Known gap, tracked |

---

## 5. Candidate Assessment

### 5a. Potential Candidates (all fail on closer inspection)

**NEON Logistic/Tanh/SiLU kernels** — The most promising shape of gap: x86 has SIMD (SSE2/AVX), ARM NEON-only falls to scalar, SVE covers only newer chips.
- **Impact**: Moderate (activation functions are <5% of LLM inference time; GEMM dominates)
- **Testability**: ❌ **Fatal** — no ARM hardware on this host (AMD EPYC 9V74). Cannot build, run, or benchmark.
- **Novelty**: Our Rust `silu_f32_neon` exists but would need complete C++ rewrite, not a port.
- **Acceptance**: Uncertain — upstream may prefer extending SVE coverage or KleidiAI over adding NEON scalar-math approximations.
- **Verdict**: **Does not survive.** Low impact + untestable + requires C++ rewrite.

**NEON FlashAttention** — Writing NEON-optimized flash attention for ARM.
- **Impact**: High (attention is significant in LLM inference)
- **Testability**: ❌ **Fatal** — same untestability problem.
- **Complexity**: Very high (~1650 LOC in x86 version, memory-tiling algorithm)
- **In-flight**: Issue #29613 was fixed; flash attention works on ARM, just not MLAS-SIMD optimized.
- **Verdict**: **Does not survive.** High complexity + untestable + enormous scope.

### 5b. Explicit Non-Candidates

| Item | Reason |
|------|--------|
| Any NEON/SVE GEMM kernel | Upstream has comprehensive NEON assembly + KleidiAI + in-flight SVE PRs |
| Quantized MatMul (NBits) | Upstream has `qnbitgemm_kernel_neon.cpp`, `sqnbitgemm_kernel_neon_*.cpp`, `hqnbitgemm_kernel_neon_fp16*.cpp` — exhaustive coverage |
| Softmax | `softmax_kernel_neon.cpp` + fp16 variant exist upstream |
| Rotary embedding | `rotary_embedding_kernel_neon.cpp` + fp16 variant exist upstream |
| Erf/Gelu fp16 | `erf_neon_fp16.cpp`, `gelu_neon_fp16.cpp` exist upstream |
| Convolution | Extensive NEON + bf16 assembly + KleidiAI conv exist upstream |
| Our Rust NEON code as-is | Written in Rust with `std::arch::aarch64`, not portable to ORT's C++ MLAS |
| QNN EP work | QNN is a separate execution provider, not MLAS kernel work |

---

## 6. Testability Constraint

**This host is x86-64 (AMD EPYC 9V74). There is no ARM hardware available.** This is a hard constraint:
- Cannot cross-compile and run ARM NEON code
- Cannot benchmark any ARM optimization
- Cross-compilation might type-check at best (Rust cross-compile with `aarch64-unknown-linux-gnu` target)
- C++ MLAS cross-compilation requires ARM toolchain setup not present here
- This programme has repeatedly refused to ship code that cannot be exercised on available hardware

Any ARM upstream candidate would require access to ARM hardware (Apple Silicon Mac, Graviton EC2, Windows ARM device) for development and testing.

---

## 7. Recommendation

**DECLINE — no ARM upstream candidate survives scrutiny.**

Rationale:
1. **Upstream ARM coverage is already comprehensive.** MLAS has 50+ ARM-specific files covering GEMM, quantized GEMM, convolution, attention, activations, and more — in NEON, SVE, and KleidiAI. ARM is arguably ORT's best-supported SIMD target.
2. **Our ARM work is Rust, not C++.** We have genuine NEON intrinsics in ~8 kernel files, but upstreaming to ORT requires C++ MLAS code. This would be "writing new C++ kernels" not "upstreaming our work."
3. **The remaining gaps are narrow.** NEON logistic/tanh/silu are partial gaps (SVE covers them on newer chips), and flash attention works on ARM (just not MLAS-SIMD optimized). Both are low-priority given activation functions are <5% of LLM time.
4. **No ARM hardware for testing.** Cannot build, run, or benchmark anything ARM on this host.
5. **In-flight work is actively closing remaining gaps.** PRs #31143, #31146, #26027 are expanding SVE coverage.

This is an honest "ARM is already well covered upstream" conclusion, and is a genuinely valuable outcome for the programme.

---

## 8. Open Questions for @justinchuby

1. **Is there ARM hardware available elsewhere** (e.g., Apple Silicon CI, Graviton runners) that could change the testability constraint?
2. **Should we focus energy on QNN EP work** instead? QNN (Qualcomm Neural Network) for Snapdragon/Windows-on-ARM is a different story — it's an execution provider, not MLAS kernel work, and might have different upstream dynamics.
3. **Are there specific ARM performance regressions** you've observed that might point to a gap not visible from code inspection alone?
4. **Should we consider contributing to the in-flight SVE PRs** (#31143, #31146) rather than opening a new PR? Reviewing/testing existing ARM PRs might be higher leverage than new work.

---

## Appendix: Methodology

- **Verified**: All file references checked against actual files in upstream worktree (`/workspace/upstream/ort-cuda`) and our repo (`/workspace/dev/onnx-genai`). SIMD macro/intrinsic presence confirmed with grep.
- **Verified**: In-flight PRs checked via GitHub search against `microsoft/onnxruntime`.
- **Verified**: Issue #29076 (SiLU/Swish) and #29613 (flash attention ARM) confirmed via GitHub search.
- **Unverified**: Whether Q4GEMM is truly legacy vs. actively used — inferred from the existence of the newer NxBit GEMM infrastructure.
- **Unverified**: Exact performance impact of scalar fallback for logistic/tanh on NEON-only ARM — would require benchmarking on ARM hardware.
