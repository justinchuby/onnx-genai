# Decision: 7B native decode bottleneck localization (Foundry, H200)

**Author:** Cohaagen (perf)
**Date:** 2026-07-27
**Scope:** Measurement / localization only — no kernel changes.

## Context

Follow-up to the Foundry native-vs-ORT baseline (native wins all four models;
7B was the thinnest lead at 1.13×). Captured a real native CUDA-graph decode
trace of qwen2.5-7b-instruct on H200 (device 1) via the CLI merged timeline
(`onnx-genai --profile-trace`, `ONNX_GENAI_EP=cuda --backend native`). Aggregated
one steady decode step's on-device kernel spans across all 28 layers.

## Headline: where the 7B native decode spends kernel time

| # | op | kernel_variant | % of kernel time |
| --- | --- | --- | ---: |
| 1 | GroupQueryAttention | `attention_gqa_decode_fp16_splitk` | 33.1% |
| 2 | MatMulNBits | `gemv_f16_general` (o_proj) | 19.5% |
| 3 | MatMulNBits | `gemv_f16_down_projection` | 16.0% |
| 4 | MatMulNBits | `gate_up_swiglu_rmsnorm_fused` | 15.6% |
| 5 | MatMulNBits | `gemv_f16_scales_f16_rmsnorm` (qkv) | 15.3% |

Family rollup: **int4 GEMV (MatMulNBits) 66.9%**, **GQA 33.1%**. All GEMVs are
symmetric int4 (`zero_points=false`), block_size=32, fp16 scales. Everything is
CUDA-graph captured; GQA decode prep is already fused at steady state.

## Top-2 optimization candidates (for a SEPARATE reviewed PR)

**Candidate A (higher leverage, guardrail-safe): o_proj `gemv_f16_general`
split-K / grid-widening.**
The square o_proj (K=N=3584) fails the tall-skinny gate and falls on the
single-warp `general` GEMV — the prime grid-starvation suspect on H200's 132 SMs.
Route grid-starved `general` GEMVs through the split-K / column-widening treatment
already used by `gemv_f16_down_projection` and `matmul_nbits_gemv_f16_scales_f16_splitk`,
gated on a measured `<1 wave/SM` check. File: `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs`.
- Guardrail (prior perf memory): **register-prefetch on the symmetric gate/up GEMV
  REGRESSES** — do NOT touch `gate_up_swiglu_rmsnorm_fused` with prefetch.
- Guardrail: **grid-starved (<1 wave/SM) int GEMV wants split-K, not prefetch.**

**Candidate B (lower leverage): GQA decode partial+merge launch fusion.**
`attention_gqa_decode_fp16_splitk` is already tuned (on-device split count fills
the SMs, prep fused). The only structural lever is folding the separate
softmax-merge reduction launch into the partial epilogue (removes one launch/layer).
File: `crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs` (~L2233/L2212).

## Next step

Verify o_proj occupancy (waves/SM) with Nsight Compute on device 1, then implement
Candidate A behind a grid-starvation gate in a reviewed PR. Do not merge kernel
changes from this scoping pass.

Full report: `docs/benchmarks/2026-07-27-foundry-native-vs-ort-cuda.md`
(section "7B bottleneck localization"), branch `squad/perf-7b-bottleneck`.
