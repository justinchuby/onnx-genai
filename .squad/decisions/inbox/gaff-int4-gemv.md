# Decision: int4 MatMulNBits GEMV latency — occupancy-gated pipe vs plain

**Author:** Gaff (CUDA-kernel specialist)
**Date:** 2026-08-19
**Branch:** `squad/int4-gemv-latency`
**Scope:** `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs`

## Context

Deckard's attribution (#4): native int4 MatMulNBits GEMV was 1135µs/step vs
ORT 942µs (+193µs). Native's kernel ncu'd as DRAM 20% / SM 39% / warps 63% →
latency/issue-bound (Long-Scoreboard on global weight loads), not DRAM-roofline,
so headroom was expected.

## What was tried and rejected

**Deep prefetch (PF>2) — NO-GO.** Generalized the pipe kernel's prefetch depth to
a template param (PF=4) and gated it for grid-starved shapes. ncu showed NO
improvement (4.3µs unchanged, Long-Scoreboard stalls unchanged, regs 40→49).
Root cause: the grid-starved int4 projections here are all K=1024 → only 4
weight loads/warp — too little per-warp work to hide latency with more loads in
flight. Occupancy is warp-capped (one warp per output column ⇒ total warps = N),
so it can't rise without split-K, and split-K reorders the fp32 partial reduction
→ NOT byte-identical (barred). Reverted entirely.

**Honest floor:** the grid-starved int4 projections are at their practical
byte-identical latency floor. The existing pipe kernel already runs ~4.3–4.7µs/
call, already beating ORT's cited 4733ns/call; there is no byte-identical lever
left on those shapes.

## What landed

**Occupancy-gated pipe-vs-plain entry selection.** The pipe (prefetch) entry uses
more registers to hide weight-load latency — a win on grid-starved projections
but a LOSS on launches that already fill the SMs many waves over, where its lower
occupancy dominates. The wide LM-head GEMV is the only well-occupied block-32 int4
GEMV in this model.

Measured (H200, 132 SMs, isolated ncu, `--graph-profiling node`):
- LM head N=248320, grid=31040 (~235 CTAs/SM): **plain 85.0µs vs pipe 98.8µs (−14%)**
- Projection N=4096, grid=512 (~3.9 CTAs/SM): pipe 4.66µs vs plain 4.79µs (pipe kept)

Gate: route to the plain entry when `ceil(N / columns_per_block) >= mp_count*32`
CTAs; keep pipe otherwise. Both entries are **byte-identical** (same lane→nibble
mapping, same fp16 accumulation order) — a pure occupancy/register trade.

## Results

- **End-to-end (paired A/B, interleaved, env-toggle on same binary):** gate-ON beat
  old all-pipe by **+1.3 to +1.6 tok/s every round** (~242 → ~243.5 tok/s). Paired
  delta cancels shared-box noise (absolute tok/s drifted 239–244).
- **Byte-identity:** greedy tokens identical gate-on vs gate-off (string-equal).
- **Golden lock:** `qwen35_0_8b_text_decode_lock` **PASSES** (`ok`).
- **Unit tests:** `scales_f16_pipeline_is_bit_identical_to_scalar`,
  `fp16_gemv_variant_selection_is_structural`, `fp16_gemv_matches_dequant_reference`
  (+ new `scales_f16_pipe_well_occupied_routes_lm_head_to_plain`) all pass.

## Properties

- **General:** gate keys only on N, launch width, and live SM count — no hardcoded
  shapes/head-dim/layer-count; works for arbitrary MatMulNBits dims and both
  fp16/fp32 accumulation paths. Not special-cased to qwen3.5.
- **Capture-safe:** launch-time constant, stable across CUDA-graph replays; no host
  syncs or per-step alloc.
- **Opt-out:** `ONNX_GENAI_GEMV_PIPE_WELLOCC=0` forces the pre-gate all-pipe path;
  `=1` forces well-occupied-everywhere. `ONNX_GENAI_GEMV_PIPELINE=0` still forces
  the plain entry everywhere (pre-existing A/B knob).

## Verdict

One clean, free, byte-identical win (de-pipeline the well-occupied LM head, −14% on
that kernel, ~+0.6% end-to-end). The remaining +193µs int4 gap on grid-starved
projections is at its byte-identical floor — closing it further would require
split-K (reorders fp32 reduction → not byte-identical), which the requirements bar.
