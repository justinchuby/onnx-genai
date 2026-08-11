# Resch — Upstream CPU Kernel Gap Analysis

**Date:** 2026-08-11
**Status:** Analysis complete; no code written, no upstream action taken.

## Ranked Shortlist (upstreamable)

1. **f16/bf16 x86 GEMM** — upstream has zero x86 half-GEMM support (open issues #22467, #20630). Our `half_gemm.rs` (AVX2+FMA, F16C) fills this directly.
2. **GatherBlockQuantized CPU** — upstream has CUDA/WebGPU/JS but no CPU kernel. Ours matches their spec exactly.
3. **AVX-512 activation quantizer** — may fill a vectorization gap in MLAS `sqnbitgemm_q8_block.h` (unverified).
4. **Flash-decoding split-KV for CPU GQA** — novel CPU technique, likely absent upstream.
5. **SIMD RMSNorm normalize-and-scale** — trivial, contingent on whether ORT has opset-23 RMSNorm CPU.

## Key Non-Candidates

- **BlockQuantizedMatMul** (GGUF/IQ): pkg.nxrt domain, depends on our loader/mmap.
- **QMoE offload**: coupled to our WeightOffloadHostCache.
- **Per-N-shard dispatch / bounded decode pool**: runtime policy, not kernel.
- **x86 SGEMM / RotaryEmbedding / LinearAttention / CausalConv**: already upstream or not faster.

## Open Questions for Justin

1. Half GEMM scope: full panel-pack GEMM or just micro-kernel?
2. Confirm GatherBlockQuantized CPU is truly absent upstream.
3. Verify MLAS activation quantizer vectorization level before proposing ours.
4. Flash-decoding on CPU: open discussion issue first?
5. Priority: single big item (f16 GEMM) or multiple smaller ones?

Full analysis: `docs/UPSTREAM_ORT_CPU_KERNEL_GAPS.md`
