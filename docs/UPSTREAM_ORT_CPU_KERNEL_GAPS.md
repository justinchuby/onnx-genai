# Upstream ORT CPU Kernel Gap Analysis

**Author:** Resch (Intel CPU Optimization Engineer)
**Date:** 2026-08-11
**Status:** ANALYSIS ONLY — no code changes, no upstream PRs

---

## 1. Inventory of Our Meaningfully Optimized CPU Kernels

| # | Op | Domain | File | Optimization | ISA | Delegates to MLAS? |
|---|---|---|---|---|---|---|
| 1 | MatMulNBits | com.microsoft | `matmul_nbits.rs` | VNNI int4-direct M=1 GEMV (block-32 sym), AVX-512 VNNI vpdpbusd dot, per-N-shard decode pool, int8 activation quantization per K-block | AVX2VNNI / AVX512VNNI | **Hybrid**: M≥threshold → MLAS SQNBit; M=1 decode → our hand-written VNNI path |
| 2 | MatMulNBits (activation quant) | com.microsoft | `simd_quant.rs` | AVX-512 vectorized per-block int8 activation quantizer (feeds VNNI dot), bit-identical to scalar | AVX-512F | No — our own; feeds our int8 dot and MLAS CompInt8 |
| 3 | Half GEMM | internal | `half_gemm.rs` | Blocked register-tiled f16/bf16→f32 GEMM with F16C widening, AVX2+FMA microkernel, rayon-parallel | AVX2+FMA / F16C | No — entirely ours |
| 4 | BlockQuantizedMatMul | pkg.nxrt | `block_quantized_matmul.rs` | GGUF block format decode (MXFP4, IQ family), AVX-512 E2M1 decoder, cached dense expansion | AVX-512F | No — our own decode + cached dequant→dense GEMM via MLAS/ours |
| 5 | QMoE | com.microsoft | `qmoe.rs` | Route-first expert offload, per-active-expert dequant with mmap, block-cooperative parallel routing | Scalar (routing) | Reuses `dequantize_nbits_row` from matmul_nbits |
| 6 | GroupQueryAttention | com.microsoft | `group_query_attention.rs` | Flash-decoding split-KV for long context, attended-window-only scoring, shared SDPA decode core | Via SDPA | No |
| 7 | SDPA core | internal | `sdpa.rs` | AVX2+FMA dot/AXPY for Q·K scoring and P·V accumulation, flash-decoding partial combine | AVX2+FMA | No |
| 8 | RMSNorm / SkipSimplifiedLayerNorm | ai.onnx / com.microsoft | `rmsnorm.rs`, `skip_simplified_layernorm.rs`, `simd_normalize.rs` | AVX-512 and AVX2 vectorized normalize-and-scale | AVX-512F / AVX2 | No |
| 9 | RotaryEmbedding | ai.onnx + com.microsoft | `rotary_embedding.rs` | Supports both standard (opset 23) and contrib input orderings; f16/bf16 widened compute | Scalar (no hand SIMD) | No |
| 10 | GatherBlockQuantized | com.microsoft | `gather_block_quantized.rs` | On-the-fly dequant during gather (2/4/8-bit), avoids full embedding expansion | Scalar | No |
| 11 | LinearAttention (Gated DeltaNet) | com.microsoft | `linear_attention.rs` | Full gated_delta recurrence CPU port, GQA support both directions | Scalar | No |
| 12 | CausalConvWithState | com.microsoft | `causal_conv.rs` | Faithful CPU port with SiLU fusion, f16/bf16 dtype parameterization | Scalar | SiLU via MLAS |
| 13 | x86 SGEMM | internal | `x86_sgemm.rs` | AVX2 register-tiled blocked SGEMM | AVX2 | Alternative to MLAS SGEMM |

---

## 2. Upstream Comparison (per candidate)

### 2.1 MatMulNBits — VNNI int4 direct decode & AVX-512 activation quantizer

**Upstream:** `onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc` exists. MLAS provides `SQNBitGemm` with AVX2, AVX2VNNI, AVX512, AVX512VNNI kernels (`sqnbitgemm_kernel_avx512vnni.cpp`, etc.). The MLAS `sqnbitgemm_q8_block.h` performs activation quantization.

**Gap analysis:** Upstream's MLAS already has AVX-512 VNNI SQNBit kernels. However:
- Our **AVX-512 activation quantizer** (`simd_quant.rs`) is a standalone vectorized block quantizer with proven bit-identity guarantees and NaN safety. Upstream's `sqnbitgemm_q8_block.h` performs the same function but its vectorization level is **unverified** from this analysis — it may be scalar or SSE-class.
- Our **bounded decode pool** (capping at 8 threads, topology-aware) is a threading policy, not a kernel — not upstreamable as a kernel.
- Our **per-N-shard dispatch** into the decode pool is a scheduling innovation but tightly coupled to our threadpool model.

**Verdict: PARTIAL** — The VNNI dot kernel itself is already in MLAS. The activation quantizer vectorization *may* be a gap but requires verification of MLAS's existing `sqnbitgemm_q8_block.h` implementation.

**Citation:** `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512vnni.cpp`, `sqnbitgemm_q8_block.h`

---

### 2.2 Half-precision (f16/bf16) GEMM on x86

**Upstream:** MLAS has `halfgemm.cpp` with NEON kernel (`halfgemm_kernel_neon.cpp`) and RISC-V (`halfgemm_kernel_rvv.cpp`). **No x86 AVX2/F16C kernel exists** for the half GEMM path. Open issues confirm: #22467 ("FP16 support for MatMul and GEMM on CPU EP", open), #20630 ("bf16 kernel for MatMul in CPU EP", open).

**Verdict: GAP** — Upstream lacks x86 f16/bf16 GEMM entirely. Our `half_gemm.rs` (AVX2+FMA, F16C widening, blocked register-tiled) fills this gap.

**Citation:** No `halfgemm_kernel_avx*.cpp` in MLAS. Issues microsoft/onnxruntime#22467, #20630 (both open).

---

### 2.3 BlockQuantizedMatMul (GGUF/MXFP4/IQ formats)

**Upstream:** Only QNN-provider `lpbqmatmul_fusion` references `BlockQuantizedMatMul`. No CPU kernel exists for GGUF native block formats (MXFP4, IQ1-4 family). The op is our `pkg.nxrt` domain.

**Verdict: NOT UPSTREAMABLE AS-IS** — The op is in our custom domain (`pkg.nxrt`) and depends on our GGUF loader metadata. However, the **MXFP4 decode logic** (OCP MX E2M1/E8M0 with AVX-512 SIMD) could be contributed as a standalone MLAS utility or as a new ORT contrib op if ORT adopts GGUF or OCP MX formats.

---

### 2.4 QMoE — Route-first expert offload with block-cooperative parallel routing

**Upstream:** `onnxruntime/contrib_ops/cpu/moe/moe_quantization_cpu.cc` exists. Also `moe_helper.h`. Upstream has QMoE for WebGPU and CUDA.

**Gap analysis:** Upstream has a CPU `QMoE` / `moe_quantization_cpu` implementation. Our innovations are:
1. **Route-first offload** — compute routes first, dequantize only selected expert slices from mmap. This is tightly coupled to our `WeightOffloadHostCache` and mmap infrastructure.
2. **Block-cooperative parallel routing** — the router parallelization from PR #684 that raised decode from serial to ~62 tok/s.

**Verdict: PARTIAL** — The route-parallel optimization is genuinely novel and portable (it's just a better top-k with parallel reduction). The offload path is not portable without upstream adopting a similar weight-paging model.

**Citation:** `onnxruntime/contrib_ops/cpu/moe/moe_quantization_cpu.cc`, `moe_helper.h`

---

### 2.5 GroupQueryAttention — Flash-decoding split-KV on CPU

**Upstream:** `onnxruntime/contrib_ops/cpu/bert/group_query_attention.cc` exists.

**Gap analysis:** Upstream has a CPU GQA kernel. Our additions:
1. **Flash-decoding (split-KV)**: splits the KV dimension across workers when window ≥ 1536, combines partials. This is a standard technique on GPU but rare on CPU — **unverified** whether upstream's CPU GQA implements it.
2. **AVX2+FMA SDPA core**: vectorized dot/AXPY for scoring and accumulation.
3. **Attended-window-only scoring**: avoids allocating a full-length scratch for long contexts with local windows.

**Verdict: PARTIAL (likely GAP for split-KV and SIMD SDPA)** — Upstream has GQA CPU, but flash-decoding split-KV on CPU is unlikely to be present (it was recently added for GPU). Requires reading `group_query_attention.cc` to confirm.

**Citation:** `onnxruntime/contrib_ops/cpu/bert/group_query_attention.cc`

---

### 2.6 RMSNorm / SkipSimplifiedLayerNorm — AVX-512/AVX2 vectorized

**Upstream:** `onnxruntime/contrib_ops/cpu/skip_layer_norm.cc` exists. No `RMSNormalization` found under `providers/cpu`. Standard ONNX opset-23 `RMSNormalization` is new.

**Gap analysis:** Our `simd_normalize.rs` provides explicit AVX-512 and AVX2 normalize-and-scale loops. Upstream's `SkipLayerNorm` CPU kernel likely uses MLAS or compiler autovectorization. ORT may not yet have an opset-23 `RMSNormalization` CPU kernel at all.

**Verdict: PARTIAL (possible GAP for opset-23 RMSNorm CPU)** — Need to verify whether ORT's CPU provider registers `ai.onnx::RMSNormalization`. The SIMD normalize is straightforward to port.

**Citation:** `onnxruntime/contrib_ops/cpu/skip_layer_norm.cc`

---

### 2.7 RotaryEmbedding

**Upstream:** `onnxruntime/contrib_ops/cpu/bert/rotary_embedding.cc` AND new `onnxruntime/core/providers/cpu/llm/rotary_embedding.cc` exist. MLAS also has `rotary_embedding.cpp`.

**Verdict: ALREADY-COVERED** — Upstream has both contrib and standard CPU RotaryEmbedding, plus MLAS support. Our implementation has no hand-SIMD advantage.

---

### 2.8 GatherBlockQuantized (CPU)

**Upstream:** Only found for CUDA, JS, and WebGPU (`contrib_ops/cuda/quantization/gather_block_quantized.cc`, etc.). **No CPU implementation found.**

**Verdict: GAP** — Upstream lacks a CPU `GatherBlockQuantized` kernel. Our implementation handles 2/4/8-bit on-the-fly dequant and is a faithful port of ORT's contrib spec.

**Citation:** No file at `onnxruntime/contrib_ops/cpu/quantization/gather_block_quantized.cc`.

---

### 2.9 LinearAttention (Gated DeltaNet)

**Upstream:** `onnxruntime/contrib_ops/cpu/bert/linear_attention.h` exists.

**Verdict: ALREADY-COVERED** — Upstream has a CPU LinearAttention kernel. Our version is a faithful port verified against ORT 1.26. No unique optimization.

---

### 2.10 CausalConvWithState

**Upstream:** `onnxruntime/contrib_ops/cpu/bert/causal_conv_with_state.cc` exists.

**Verdict: ALREADY-COVERED** — Upstream has the same kernel.

---

## 3. Ranked Shortlist of Upstreamable Candidates

| Rank | Candidate | Expected Model Impact | Hardware Breadth | Portability | Impl Cost | Acceptance Likelihood | Score |
|---:|---|---|---|---|---|---|---|
| 1 | **f16/bf16 x86 GEMM** (`half_gemm.rs`) | HIGH — enables f16 inference on x86 CPU without full dequant | All AVX2 x86 (vast majority) | HIGH — no nxrt deps, standard MLAS pattern | MEDIUM — needs C++ port into MLAS panel framework | HIGH — open issues requesting it | ★★★★★ |
| 2 | **GatherBlockQuantized CPU** | MEDIUM — embedding table lookup for quantized models | Universal (scalar, no ISA dep) | HIGH — already matches ORT contrib spec | LOW — straightforward C++ | HIGH — fills obvious gap in their own op | ★★★★☆ |
| 3 | **AVX-512 activation quantizer** (`simd_quant.rs`) | MEDIUM — feeds VNNI decode; ~15-25% of M=1 activation quant time on SPR | AVX-512 hosts (server, recent desktop) | HIGH — pure numeric, no runtime deps | LOW-MEDIUM — C++ intrinsics, well-isolated | MEDIUM — may already be vectorized in MLAS (unverified) | ★★★☆☆ |
| 4 | **Flash-decoding split-KV for CPU GQA** | MEDIUM-HIGH — long-context decode latency (>2K tokens) | Universal (threading, no ISA dep) | MEDIUM — needs adaptation to ORT threadpool | MEDIUM | MEDIUM — novel for CPU, may require benchmarking | ★★★☆☆ |
| 5 | **SIMD normalize-and-scale (RMSNorm)** | LOW-MEDIUM — norm is small % of decode step | AVX2/AVX-512 | HIGH — trivial, no deps | LOW | MEDIUM-HIGH — if opset-23 RMSNorm CPU is missing | ★★☆☆☆ |

---

## 4. Per-Candidate Porting Notes

### 4.1 f16/bf16 x86 GEMM → MLAS `halfgemm`

**What must change:**
- Rewrite from Rust to C++ using MLAS's panel-packing framework (`MlasGemmPackedA/B` pattern)
- Replace `rayon::par_iter` with MLAS's `MlasThreadPool` parallel-for
- Register as a new `MLAS_HALF_GEMM_DISPATCH` architecture path alongside NEON/RVV
- MR/NR/KC tuning parameters may differ for MLAS's expectations

**nxrt dependencies to sever:** None meaningful — the kernel is self-contained with runtime `is_x86_feature_detected!` (becomes MLAS cpuid dispatch).

**Tests ORT would expect:** Correctness against f32 reference (tolerance ~n·u), perf regression gate for existing NEON path, CI must work without AVX2 (scalar fallback).

### 4.2 GatherBlockQuantized CPU

**What must change:**
- Port from Rust to C++ following `contrib_ops/cpu/` pattern
- Register under `com.microsoft` domain (same as CUDA/WebGPU versions)
- Use ORT's `OpKernelContext` tensor accessors instead of our `TensorView`

**nxrt dependencies to sever:** None — already follows ORT's spec exactly.

**Tests ORT would expect:** Parity with CUDA kernel for small tensors, coverage of 2/4/8-bit paths with and without zero points.

### 4.3 AVX-512 Activation Quantizer

**What must change:**
- Port intrinsics to C++ (already uses `_mm512_*` intrinsics, direct mapping)
- Integrate into MLAS `sqnbitgemm_q8_block.h` as an AVX-512 specialization
- Preserve the NaN-safety fallback (upstream may or may not care about this edge)

**nxrt dependencies to sever:** None — pure numeric function.

**Tests ORT would expect:** Bit-identity with scalar path, NaN/inf edge cases.

### 4.4 Flash-Decoding Split-KV for CPU GQA

**What must change:**
- Port partial-combine pattern into ORT's existing `group_query_attention.cc`
- Replace our rayon decode pool with ORT's `concurrency::ThreadPool`
- Tune `SPLIT_MIN_KV` threshold for ORT's threading model

**nxrt dependencies to sever:** Our SDPA core (would need to use ORT's existing attention math or port the AVX2 dot/AXPY).

**Tests ORT would expect:** Numerical parity with non-split path, perf improvement evidence at >2K context.

### 4.5 SIMD Normalize-and-Scale

**What must change:**
- Trivial intrinsics port to C++ (15-20 lines of AVX2, 15 of AVX-512)
- Integrate into ORT's `skip_layer_norm.cc` or a new `rms_normalization.cc`

**nxrt dependencies to sever:** None.

---

## 5. Explicit Non-Candidates

| Kernel | Reason NOT upstreamable |
|---|---|
| **BlockQuantizedMatMul** (GGUF/IQ) | `pkg.nxrt` domain; depends on our GGUF loader metadata, immutable-constant-slot caching, and mmap-based weight system. ORT has no GGUF format support. Would require ORT to adopt GGUF or OCP-MX as a first-class format. |
| **QMoE route-first offload** | Tightly coupled to our `WeightOffloadHostCache`, mmap region catalog, and `ExternalMmapRegion` abstraction. ORT's weight management is fundamentally different. |
| **Bounded decode threadpool** (8-thread cap) | A runtime threading policy, not a kernel. ORT has its own threadpool with different design goals (work-stealing, not bounded fork-join). |
| **Per-N-shard MLAS dispatch** | Scheduling innovation over MLAS's existing API. It exploits our decode pool topology; upstream would need to rethink their threading model. Also introduces ~1 ULP non-determinism across N-partition boundaries. |
| **x86 SGEMM** (`x86_sgemm.rs`) | MLAS already has a highly optimized x86 SGEMM (the industry standard). Our version is an alternative that is NOT faster — it exists for `no-mlas` builds. |
| **QMoE block-cooperative routing** | While the parallel top-k algorithm is portable, the model-level benefit (62 tok/s on 35B-A3B) comes only with our specific MoE configuration. For upstream's general case, the routing is not the bottleneck — the GEMM is. Marginal benefit. |
| **LinearAttention / CausalConvWithState** | Already upstream. Our versions are faithful ports, not improvements. |
| **RotaryEmbedding** | Already upstream with MLAS support. No advantage. |

---

## 6. Open Questions for Justin

1. **Half GEMM scope:** Should we propose the full panel-pack GEMM to MLAS, or just the F16C widening micro-kernel as a leaf? The former is a larger contribution but more impactful; the latter is easier to get accepted.

2. **GatherBlockQuantized — is it genuinely missing or just unfound?** I could not find it under `contrib_ops/cpu/` via code search. If it exists under a different name or was recently added, this drops from the shortlist. Should we file an upstream issue to confirm?

3. **AVX-512 activation quantizer — already vectorized in MLAS?** The file `sqnbitgemm_q8_block.h` handles activation quantization but I cannot confirm its vectorization level without reading the source. If MLAS already has an AVX-512 path here, our contribution is redundant. Should we clone upstream briefly to verify? (Currently forbidden by scope.)

4. **Flash-decoding on CPU — appetite?** This is a novel contribution (GPU technique adapted to CPU). ORT's team may not have considered it. Should we open a discussion issue first, or prepare a full PR?

5. **MXFP4/OCP-MX support:** If ORT ever adopts OCP MX formats, our AVX-512 E2M1 decoder would be immediately relevant. Should we track this as a "conditional candidate" contingent on upstream format adoption?

6. **Priority:** Given limited bandwidth, should we focus on the single highest-impact item (f16/bf16 GEMM, which has explicit upstream demand via open issues) or pursue multiple smaller items in parallel?

---

## Methodology Notes

- **Verified** (code search confirmed): Upstream file existence/absence, open issue numbers, MLAS kernel file inventory.
- **Inferred** (not confirmed by reading source): Whether upstream's `group_query_attention.cc` uses flash-decoding, whether `sqnbitgemm_q8_block.h` has AVX-512 vectorization, whether ORT has opset-23 `RMSNormalization` registered on CPU.
- **Benchmark evidence** (from our own `docs/BENCH_MLAS_INT4_E2E.md`): MLAS SQNBit tie with our hand-path at M=1 on SPR; decode is DRAM-bandwidth-bound so kernel choice is a wash — the advantage is avoiding int8 activation rounding, not throughput.
- **Decisions.md evidence**: QMoE route parallelization measured at 62 tok/s on 35B-A3B; MLAS vs hand-path tie confirmed on SPR; our half-GEMM is the operational path for f16/bf16 models on x86.
