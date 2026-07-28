# Whisper-tiny Encoder Attribution — Root Causes and Fixes

**Date:** 2026-07-28  
**Author:** Iran (Mac CPU Optimization Engineer)  
**Status:** Measured, fix implemented, PR pending review

## Headline

Whisper-tiny encoder: **3780 ms → 193 ms** (19.6× speedup), from **29.5× ORT → 1.52× ORT**.

## Root Causes (in measured order of impact)

### 1. Conv 1D on scalar reference: 2726 ms (72%)

**The "MatMul-bound, not Conv-bound" claim was wrong.** The previous Conv optimization (#317) targeted rank-4 (2D) convolutions only. Whisper's frontend has two 1D convolutions:

- Conv1: [1, 80, 3000] → [384, 80, 3] (80 channels → 384, kernel=3)
- Conv2: [1, 384, 3000] → [384, 384, 3] (stride=2, produces 1500 frames)

These are **rank-3** tensors. The dispatch at line 273 (`let is_rank4 = self.x_shape.len() == 4`) gated both BNNS (Tier 1) and im2col+GEMM (Tier 2) behind rank-4, sending 1D convolutions directly to the O(N²) scalar triple-loop (Tier 3).

**Fix:** Promote rank-3 shapes to rank-4 by inserting H=1: `[N, C, W] → [N, C, 1, W]`. This is a logical reshape with no memory movement. The promoted shape flows through the existing BNNS and im2col+GEMM infrastructure unchanged.

**Result:** Conv: 2726 ms → 3 ms (908× faster).

### 2. SDPA on single-threaded NEON dot/axpy: 933 ms (25%)

The Attention op runs scaled dot-product attention with:
- 6 heads, seq_len = 1500, head_dim = 64, 4 layers
- Total QKᵀ + P·V compute: ~13.8 GFLOP

On macOS without the `mlas` feature, the SDPA dispatcher falls to `sdpa_f32_neon` — a single-threaded serial loop using NEON `dot_f32` and `axpy_f32`. At M=1 decode this is fine (small kv_seq), but at seq=1500 the O(seq²) attention dominates.

**Fix:** Added `sdpa_f32_accelerate` — mirrors the existing MLAS fast path but uses Apple's `cblas_sgemm` (which reaches AMX). Parallelized across `(batch, head)` tiles via Rayon. The two GEMMs per tile — `logits = alpha · Q · Kᵀ` and `context = probs · V` — run on AMX instead of scalar NEON.

**Result:** Attention: 933 ms → 68 ms (13.7× faster).

### 3. MatMul (projections): Working correctly

The 24 f32 MatMul operations ([1500,384]×[384,384] etc.) correctly reach `accelerate_gemm::sgemm → cblas_sgemm` at 23 ms total. No issue here — the "our MatMul path is fast" observation was correct; it just wasn't what dominated.

## Why the prior attribution was wrong

The Conv BNNS fix in #317 was measured on Whisper and "left it completely unchanged at 3808 ms." This was correctly observed! But the conclusion "therefore it's MatMul-bound" was a non-sequitur: #317 only fixed rank-4 Conv, and Whisper's Convs are rank-3. The fix never executed.

## Dtype separation

Whisper encoder is **entirely fp32** (dtype=1, all initializers f32). The fp16 BNNS paths, fp16 GEMV, and half_gemm infrastructure are irrelevant. The 0.92× Qwen fp32 gap is a separate issue (different shapes and no attention dominance).

## Remaining gap: native = 1.52× ORT

After fixes, the residual profile is:
- Attention: 68 ms (37%) — now AMX-backed, within ~2× of ORT's fused implementation
- Gelu: 45 ms (25%) — scalar implementation, could use vDSP/vForce
- FusedMatMulBias: 23 ms (12%) — already fast
- Mul: 20 ms (11%) — elementwise, could use vDSP
- LayerNorm: 16 ms (9%) — scalar
- Add: 7 ms (4%) — already uses vDSP from #324

Reaching ORT parity would require vectorizing Gelu + Mul + LayerNorm (81 ms combined). These are new kernel families — scheduled separately.

## Test coverage

- 981/981 ep-cpu lib tests pass (0 failures)
- Numerics parity vs ORT: max_abs = 1.19e-7 (PASS)
- Conv tests: 40/40 pass including updated 1D dispatch test
- SDPA tests: 13/13 pass including dispatch reachability
- Cross-compile check: PASS (FFI-free subset)

## Dispatch manifest updates

- `Conv/standard` description updated to note 1D promotion
- Added `SDPA/accelerate` claim (tier1, `SDPA_ACCELERATE_TEST_HITS`)
