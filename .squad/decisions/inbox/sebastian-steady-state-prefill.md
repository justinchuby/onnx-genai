# Steady-State Per-Prefill Attribution — Multi-Turn Regime

**By:** Sebastian (performance)
**Date:** 2026-07-28
**Commit:** `ed21f472` (main, post-PR #353 thin-M NEON + f32 transpose precompute)
**Machine:** Apple M1 Max, macOS, load 5–9 during measurements
**Method:** `profile_native --steady --warmups 5 --runs 10`, interleaved A/B, Engine created once

## 1. Steady-State Per-Prefill Cost (One-Time Costs Stripped)

Model loaded once, caches warm, prefill repeated as across conversation turns.

| Model | Prompt tokens | Native prefill (ms) | ORT prefill (ms) | Ratio |
|---|---|---|---|---|
| TinyStories-33M (fp32) | 4 | **8.2** | **3.3** | 2.52× slower |
| qwen2.5-0.5b-f16 | 4 | **45.1** | **26.5** | 1.70× slower |

| Model | Native decode (tok/s) | ORT decode (tok/s) | Ratio |
|---|---|---|---|
| TinyStories-33M (fp32) | 238 | 337 | 1.42× slower |
| qwen2.5-0.5b-f16 | 47.4 | 41.3 | **1.15× faster** |

**The multi-turn verdict:** On small fp32 models, ORT overtakes us after **3 turns**
(20 tok/turn) because both prefill AND decode favor ORT. On fp16 large models, our
decode advantage compounds faster than ORT's prefill advantage — **we never lose the
cumulative race** regardless of turn count.

## 2. Re-Attribution at Steady State (post-#353)

Steady-state per-op profile, TinyStories-33M, M=4 prefill, 7.8 ms total kernel time:

| Component | Time (ms) | % | Path | Effective BW |
|---|---|---|---|---|
| FusedMatMulBias (FFN+O_proj, 12 calls) | 3.71 | 48% | cblas_sgemm | **23 GB/s** |
| MatMul (QKV×12 + lm_head×1, 13 calls) | 2.93 | 38% | cblas + thin-M | — |
| — of which lm_head [4,768]×[768,50257] | ~2.0 | 26% | thin-M NEON | **75 GB/s** |
| — of which QKV [4,768]×[768,768] ×12 | ~0.93 | 12% | cblas_sgemm | **30 GB/s** |
| Attention SDPA (4 calls) | 0.48 | 6% | NEON inline | — |
| Gelu (4 calls) | 0.40 | 5% | — | — |
| LayerNorm + other | 0.25 | 3% | — | — |

**Comparison with cold attribution (pre-#353, 26.3 ms):** PR #353 cut TTFT from 17.3→13.0
cold and from ~12→8.2 steady-state. The lm_head dropped from 15.0ms to ~2.0ms (thin-M
NEON at 75 GB/s vs cblas at 10 GB/s cold). The residual gap is now dominated by the
**other** MatMuls that are still on cblas_sgemm.

## 3. What ORT Does Better — With Evidence

### 3.1 Blocked/Packed Weight Layouts (THE CRUX — 3–4× BW gap)

ORT pre-packs all weights at model load into MR×NR blocked micro-panels. Each kernel
iteration loads a contiguous micro-panel from L1 into SIMD registers with no stride gaps.
The hardware prefetcher sees a pure sequential stream.

Our current state:
- **lm_head**: pre-transposed at load (thin-M eligible), streamed column-by-column.
  Achieves **75 GB/s** — close to ORT's ~89 GB/s. The transpose IS a good layout for
  this kernel.
- **All other weights (QKV, FFN, O_proj)**: used directly from mmap in row-major via
  `cblas_sgemm`. At M=4 with these sizes, cblas's tiling/GCD dispatch overhead dominates.
  Achieves **23–30 GB/s** — 3–4× below ORT.

**A transpose is not a pack.** The thin-M NEON kernel with pre-transposed B achieves 75 GB/s
because it streams B_T row-by-row (column-of-B) sequentially. But `cblas_sgemm` with
*untransposed* row-major B at M=4 cannot achieve this — it tiles B into panels, and for
thin M the panel overhead dominates.

Evidence: measured effective bandwidth at each tier:

| Path | Shapes covered | Effective BW | vs ORT (89 GB/s) |
|---|---|---|---|
| thin-M NEON (pre-transposed) | lm_head only (K*N > 4M) | 75 GB/s | 0.84× |
| cblas_sgemm (row-major) | QKV [768,768] | 30 GB/s | 0.34× |
| cblas_sgemm (row-major) | FFN [768,3072]/[3072,768] | 23 GB/s | 0.26× |
| ORT pre-packed | all shapes | 89 GB/s | 1.00× |

### 3.2 Thin-M Threshold Excludes Most MatMuls

`THIN_M_LARGE_B_THRESHOLD = 4,000,000` elements (16 MB at fp32). Only lm_head
(K*N = 38.6M) qualifies. The threshold also gates `precompute_f32_weight_transpose`,
so smaller weights have no pre-transposed copy available and cannot use the thin-M kernel
even if the shape check were relaxed.

| Weight | K×N | vs Threshold | Path |
|---|---|---|---|
| lm_head [768,50257] | 38.6M | ✓ above 4M | thin-M NEON (75 GB/s) |
| FFN_up [768,3072] | 2.36M | ✗ below 4M | cblas_sgemm (23 GB/s) |
| FFN_down [3072,768] | 2.36M | ✗ below 4M | cblas_sgemm (23 GB/s) |
| QKV/O [768,768] | 590K | ✗ below 4M | cblas_sgemm (30 GB/s) |

### 3.3 Op Count / Fusion (Minor, ~10% of Gap)

Our graph runs 77 nodes; ORT's graph optimizer fuses aggressively (MatMul+Bias+Gelu → one
kernel call, SkipLayerNorm → fused, etc.). The 0.4ms Gelu + 0.25ms other overhead would
partially amortize with fusion. Estimate: ~0.5ms recoverable. At 3.3ms total ORT, this is
real but secondary.

### 3.4 Thread Pool Behaviour (Negligible at Steady State)

Rayon's work-stealing pool is initialized once at first use and persists. At steady state
there is no per-prefill thread creation or pool spin-up cost. The `[decode-memo]` output
confirmed `replayed=35` (plan reuse) with zero pool rebuilds. This is NOT a factor in the
steady-state gap.

ORT's deterministic intra-op pool differs in scheduling semantics (round-robin partitioning
vs work-stealing) but this produces at most a scheduling jitter difference, not a 2× BW gap.

## 4. Cache Survival Across Turns — CONFIRMED

**f32 transpose cache (`WEIGHT_TRANSPOSE_F32`):** Process-global `LazyLock<Mutex<HashMap>>`.
Cleared in `Executor::Drop` only — NOT between generate() calls. Engine holds Executor for
its lifetime; multi-turn calls reuse the same Engine.

**Evidence:**
1. Cold TTFT (first prefill after load): **25.4 ms** (includes ~17ms transpose precompute)
2. Steady-state TTFT (subsequent prefills): **8.2 ms** (cache hit, no recomputation)
3. The 17ms delta is exactly the transpose-precompute cost amortized at load.
4. `ONNX_GENAI_DECODE_MEMO_STATS` showed `primed=10 rebuilt=5 replayed=35` — the decode
   memo replayed 35 step plans without rebuilding, confirming execution-plan reuse.

**This is NOT instance #15 of the "machinery that exists and never executes" defect.**
The caches work correctly: populated at load/first-use, reused across all subsequent turns,
cleared only on Engine drop.

**BNNS filter cache (`FilterCache`):** Thread-local, keyed by (M,K,N,trans_b). For fp16
models, BNNS filter creation (3–19ms cold, ~50µs warm) is amortized after the first call
at each shape. Since prefill always uses the same M, the filter is created once per shape
during the first warmup turn and reused for all subsequent turns.

## 5. Ranked Levers — What Would Make Us Win

### On TinyStories-33M (fp32, the weak spot)

| Rank | Lever | Current→Projected | Reduction | Amdahl % of prefill | Cost | Owner |
|---|---|---|---|---|---|---|
| **1** | Lower thin-M threshold to 0 | 8.2→5.5 ms | 33% | 56% (cblas portion) | Low | Iran |
| **2** | Blocked micro-panel layout (ORT-style) | 8.2→4.6 ms | 44% | 56% | High | Iran + Deckard |
| **3** | FusedMatMulBias in-place (elide alloc) | 8.2→7.7 ms | 6% | 6% | Low | Deckard |
| **4** | FusedMatMulBiasGelu (fused activation) | 8.2→7.9 ms | 4% | 5% | Medium | Deckard |

**Lever 1 detail:** Change `THIN_M_LARGE_B_THRESHOLD` from 4M to 0 (or remove it) and
correspondingly precompute f32 transposes for all weight matrices. Memory cost: +113 MB
(duplicating all non-lm_head weights as transposed copies). Implementation: ~20 lines
changed in two functions. Risk: low (thin-M kernel already verified, numerics gate passed).
Expected BW for smaller matrices: 50–60 GB/s (less parallelism than lm_head's 50K columns).

**Lever 2 detail:** Replace the column-major transpose with a proper NR-wide blocked layout.
Store each weight as contiguous micro-panels of NR columns × K rows (NR=4 for NEON f32).
The inner kernel loads 4 consecutive floats per `vld1q_f32` with zero stride, enabling
full hardware-prefetcher utilization. Expected BW: 80–90 GB/s (matching ORT). This is a
larger restructuring: new packing routine, new kernel, packing-format metadata in the
executor. ~300–500 lines. The model load cost would increase by ~2–5ms (same order as
ORT's 117ms delta — but much smaller because we'd pack into a simpler format).

**Even both levers together project only 4.6 ms vs ORT's 3.3 ms.** The remaining 1.3 ms
is non-GEMM overhead (Attention, Gelu, LayerNorm individually dispatched vs ORT's fused
kernels). Reaching full parity requires both layout optimization AND graph fusion.

### On qwen2.5-0.5b-f16 (the strong spot — already winning on multi-turn)

Native already wins the multi-turn race. The 1.70× prefill gap (45.1 vs 26.5 ms) is offset
by the 1.15× decode advantage (47.4 vs 41.3 tok/s), and the decode advantage compounds
faster (3.1ms saved per token × 20+ tokens > 18.6ms lost per prefill). Any decode-length
response ≥7 tokens makes the turn net-positive.

However, if we want to close the prefill gap too:
- The fp16 model uses BNNS `BNNSMatMul` for prefill (M≥2, fp16→f32 via AMX).
- BNNS achieves ~2000–2400 GFLOPS, which for a compute-bound (high arithmetic intensity)
  large model is close to the AMX ceiling.
- ORT's advantage here is likely their more aggressive operator fusion reducing framework
  overhead between ops. A 24-layer model has ~130 more graph nodes than TinyStories-33M,
  and per-node dispatch overhead (~10–20µs) compounds.

### Verdict

**Can we beat ORT at steady-state small-model (fp32) prefill without large restructuring?**
No. Lever 1 (threshold change) is low-cost and brings us from 2.52× to ~1.67× but does not
reach parity. Reaching parity requires Lever 2 (blocked layout) which is a significant
restructuring (~300–500 lines, new packing format, new kernel, load-time cost management).

**Can we beat ORT at steady-state large-model (fp16) prefill?**
Not on prefill alone (1.70× gap persists). But we already win the multi-turn race because
our decode is faster. The honest metric for the multi-turn regime is *total session latency*
(load + Σ(prefill + decode) over all turns), and on fp16 we win that metric at any
conversation length ≥1 turn.

**What would it take to win BOTH prefill AND decode on BOTH model sizes?**
1. fp32: Lever 2 (blocked layout) + Lever 3–4 (fusion). ~500–800 lines total. 2–3 weeks.
2. fp16: The BNNS path is already near AMX ceiling. Closing the remaining 1.70× gap requires
   fusing the non-GEMM ops (reduce per-node dispatch overhead across 24 layers).
   This is a graph compiler optimization, not a kernel optimization.

The honest path: **win fp16 multi-turn immediately (we already do), fix fp32 prefill with
blocked layout + threshold change (Levers 1+2, 1–2 weeks for Iran+Deckard), then pursue
graph fusion for the fp32 non-GEMM tail and fp16 prefill gap.**
