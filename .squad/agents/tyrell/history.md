# tyrell — History

## 2026-07-15T01:52:00Z — Session update

- Delivered KV insertion architecture revisions: Mobius controls its export contract; Phase 1 drops `past_present_share_buffer` for functional GQA, while paged attention is M=1-gated.

## 2026-07-19T18:05Z — GLM-5.2 fp32 + int4 E2E

- Proved tiny synthetic `glm_moe_dsa` runs prefill plus eight decode steps in fp32 (`bd908bf`) and int4 (`daa3518`).
- Fixed Mobius indexer RoPE to rotate the full `index_head_dim` (`1198522`); quantized exporter helper landed as `c5740c4`.
- Verified 34 asymmetric block-32 `MatMulNBits` nodes cover the full graph, including all MoE experts; fused QMoE/BlockQuantizedMoE export remains open.

- 2026-07-19: Added opt-in fused `com.microsoft::QMoE` emission for GLM/DeepSeek experts (onnx-genai fe3e342; mobius 93cbcf7). Gated synthetic E2E passes through ORT's contrib QMoE kernel, not native Rust. Grouped-routing regression was repaired by Deckard before Chew approval.

## 2026-07-20T05:20:00Z — MLAS feature passthrough

- Plumbed the opt-in `mlas` feature through session, engine, server, and bench (`294d795`), making CPU MLAS reachable with `--features mlas` and `NXRT_CPU_GEMM_BACKEND=mlas`; coordinator-verified builds and `cargo tree` propagation.


## 2026-07-20T13:35:00Z — Multistream performance and issue #40

- Landed issue #40 Phase-1 slice 1a (`0d1d265`: shared protocol trace + ticketed pressure) and 1b (`e4d2883`: Communicator + BufferOwnership); slice 1c collectives/order remains in progress.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.
## 2026-07-22T12:00:00Z — 7B CUDA-graph progress entry
- Added the `PROGRESS.md` entry for Gaff's Qwen2.5-7B CUDA-graph A/B result; merged to `main` as `e1eeae4`.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Progress docs should now treat WP-B as fully landed after WP-B3 `3d84b9b` plus clippy `6f217a4`.
## 2026-07-24T05:11:20+0000 — Whole-step DeepSeek CUDA-graph capture

- DeepSeek-V2-Lite int4 decode reached one captured segment and **0 eager seams** (727→0) on main after Leon's Reshape fold (`661618b8`) and Rachael's mask-island closure (`3dc0843b`).
- Integration retained deterministic coherent output (` Paris.\nThe currency of France is the Euro.`); CUDA library gate: 205/0.
