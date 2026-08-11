# Upstream ORT CUDA Kernel Gaps — Ranked Shortlist

**Author:** Batty
**Date:** 2026-08-11
**Ref:** `docs/UPSTREAM_ORT_CUDA_KERNEL_GAPS.md`

## Kernel-level vs Runtime-level Split

**Upstreamable (kernel-level):**
1. MatMulNBits int4 block-128 GEMV specialization
2. MatMulNBits SM-fill grid tuning (#148)
3. GQA decode split-K (fp16/f32)
4. QMoE parallel routing kernel
5. MatMulNBits accuracy_level=4 blockwise GEMV

**Not upstreamable (runtime-level):**
- CUDA graph capture system (per-op eligibility, VMM remap, symbolic-dim pinning)
- VMM weight paging / granule pooling / eviction
- Paged/tiered KV cache (growth counters, tier migration)
- CompressedSparseAttention (DeepSeek MLA, pkg.nxrt domain)
- BlockQuantizedMoE GGUF formats (custom domain + memory governor)

## Ranked Shortlist

| Rank | Candidate | Impact | Acceptance Likelihood |
|---:|---|---|---|
| 1 | MatMulNBits int4 block-128 GEMV | +36–60% tok/s on 0.5B–1.5B | HIGH (addresses #23004) |
| 2 | SM-fill grid tuning | +2–10% on H200 | HIGH (simple, non-controversial) |
| 3 | GQA decode split-K fp16 | Flat decode latency at long ctx | MEDIUM (cuDNN path may compete) |
| 4 | QMoE parallel routing | +30% MoE decode | HIGH (addresses #28987) |
| 5 | accuracy_level=4 blockwise GEMV | Enables int8-activation GEMV | MEDIUM (niche) |

## Decision

This is analysis only. No implementation until Justin green-lights after EP-compatibility milestone is stable.
