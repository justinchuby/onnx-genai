# Prefill Fusion Instrumentation — Findings

**Author:** Iran (Mac CPU Optimization Engineer)
**Date:** 2026-07-27
**Status:** Investigation complete; no new fusion ships by default

## Summary

Instrumented all optimizer passes on `models/qwen2.5-0.5b-f16` (Qwen2.5 0.5B, f16,
24 layers, GQA 14Q/2KV heads) and measured which fusions fire, total dispatch count,
and the effect of a prototype sibling-projection merge.

## Key Findings

### 1. SDPA Fusion Is Correctly Not Firing

The model arrives from the exporter with **pre-fused contrib ops**:
- 24 `com.microsoft.Attention` (SDPA already fused)
- 49 `com.microsoft.RMSNormalization` (LayerNorm equiv)
- 48 `com.microsoft.RotaryEmbedding`

The `try_match_attention()` pattern matcher anchors on Softmax nodes — since no
decomposed MatMul→Scale→Softmax→MatMul pattern exists in this graph, SDPA has
nothing to match. **This is correct behavior, not a bug.**

### 2. Which Fusions Fire

| Pass | Matches | Effect |
|------|---------|--------|
| MatMul+Bias→FusedMatMulBias | 120 | Fuses Add into GEMM epilogue |
| CpuSiluFusion (Sigmoid+Mul→Swish) | 24 | Fuses gate activation |
| SDPA (try_match_attention) | 0 | Nothing to match (correct) |
| LayerNorm | 0 | Model uses RMSNormalization (correct) |
| GELU | 0 | Model uses SiLU, not Erf-GELU (correct) |

### 3. Actual Op Count

- Raw graph: **590 ops**
- After session-level passes (constant fold + dead node elimination): ~420
- After EP-level passes (SiLU fusion + OpFusion): **350 ops**

This is the actual dispatch count during prefill — lower than the estimated 336
from the task spec (that figure likely referred to a subset of attention-related ops).

### 4. Sibling Projection Merge — Implemented, Measured, Gated Off

Built a `SiblingProjectionMerge` pass that:
- Detects 2+ sibling FusedMatMulBias/MatMul sharing the same activation input
- Concatenates weights along N axis, emits one merged GEMM + Split
- Reduces ops: 350 → 326 (saves 24 dispatches)

**However, measurement shows the wider merged GEMMs are slower on BNNS/Apple Silicon:**

| Metric | Without fusion | With fusion | Delta | Conditions |
|--------|---------------|-------------|-------|------------|
| TTFT | 91 ms | 153 ms | **+68% regression** | load 6–8 |
| Decode | 70 tok/s | 69 tok/s | ~neutral | load 6–8 |
| Model load | 248 ms | 280 ms | +13% (weight concat) | load 6–8 |

⚠️ **Load caveat:** These numbers were measured at system load 2.5–8 (M1 Max,
nearly uncontested). At load ~12 (typical developer workstation), TTFT baseline
is ~160–180 ms — matching Justin's independent measurement of 159.7 ms. The
**relative regression (+68%)** is valid (both arms measured at same load), but
the **absolute 91 ms** is not representative of normal workstation conditions.
At load 12, the regression ratio would still hold: ~160→~260 ms.

**Root cause:** BNNS internal tiling prefers the individual smaller GEMMs
([40,896]×[896,896] + [896,128] + [896,128]) over one wider GEMM
([40,896]×[896,1152]). The dispatch overhead saved (24 fewer op dispatches) is
dwarfed by the GEMM kernel efficiency loss.

**Decision:** Pass is correct and fully tested (numeric parity + dispatch
reachability), but gated behind `ONNX_RT_SIBLING_MERGE=1`. Not active by default.

**Would the merge help on `half_gemm.rs`?** Likely yes. The `half_gemm.rs` path
uses a hand-written tile-based GEMM (MR=4, NR=8) with rayon parallelism over
N-tiles. Unlike BNNS:
- Each function call has rayon thread-pool dispatch overhead
- 3 calls × rayon-sync is more expensive than 1 call with more tiles
- N=1152 divides evenly by NR=8 (144 tiles), giving better load balance
- No AMX-specific tiling concerns

This path is used on: x86_64 (no BNNS), non-Apple aarch64 (Android, Linux ARM),
and Apple Silicon when BNNS is bypassed (non-contiguous weights). The merge should
be re-evaluated when targeting those platforms.

### 5. Current Performance Baseline

**Load-dependent — numbers must always state system load.**

| Load | TTFT | Decode | End-to-end | Notes |
|------|------|--------|------------|-------|
| 2.5–8 | ~91 ms | ~70 tok/s | ~55 tok/s | Low contention, optimistic |
| ~12 | ~160–182 ms | ~70 tok/s | ~46 tok/s | Normal workstation load |

Measured with `compare --model models/qwen2.5-0.5b-f16 --runs 5 --warmups 2
--max-tokens 20 --direct-backend native`. 40 prompt tokens, Apple M1 Max.

The 91 ms figure is NOT directly comparable to the 159.7 ms / 168 ms measured at
load 12. BNNS uses multiple threads internally; at high load those threads contend
for CPU time, scaling TTFT roughly linearly with effective available bandwidth
(122 GB/s at load 3 vs 73 GB/s at load 12).

## Architectural Implications

1. **Dispatch count is not the bottleneck at current TTFT.** At 90 ms for 350 ops,
   average per-op time is 0.26 ms — dominated by compute, not dispatch overhead.
2. **The next prefill lever is NOT op-count reduction** — it's improving GEMM
   utilization within the existing 350 dispatches (or ensuring we're hitting the
   fast BNNS path for all of them).
3. The `SiblingProjectionMerge` pass is the correct architecture for backends where
   dispatch overhead dominates (quantized models with many tiny ops, or backends
   without BNNS-level thread pooling). Keep it gated for that future use.

## Files Changed

- `crates/onnx-runtime-ep-cpu/src/optimizer.rs` — `SiblingProjectionMerge` pass + env gate
- `crates/onnx-runtime-ep-cpu/src/lib.rs` — export
- `crates/onnx-runtime-session/tests/prefill_fusion_audit.rs` — instrumentation test
- `crates/onnx-runtime-session/tests/sibling_projection_merge_parity.rs` — parity/reachability tests
