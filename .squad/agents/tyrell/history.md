# Tyrell — History (compacted 2026-07-29)

**Role:** Mobius exporter contract, KV-insertion/attention architecture, multistream performance, and PROGRESS/roadmap doc keeper. Preserve Mobius's ownership of its export contract, opt-in fused emission gated through ORT contrib kernels, and SM-general CUDA performance.

## Durable lessons
- Mobius controls its export contract: Phase 1 drops `past_present_share_buffer` for functional GQA, while paged attention is M=1-gated. Mobius indexer RoPE must rotate the full `index_head_dim` (`1198522`).
- Opt-in fused `com.microsoft::QMoE` emission (onnx-genai `fe3e342`, mobius `93cbcf7`) runs through ORT's contrib QMoE kernel, not native Rust; grouped-routing regressions are a known hazard (Deckard repaired one pre-approval).
- The opt-in `mlas` CPU GEMM feature is reachable via `--features mlas` and `NXRT_CPU_GEMM_BACKEND=mlas` (`294d795`), plumbed through session/engine/server/bench.
- Cross-OS decode affinity with safe multi-NUMA auto-enable landed as `122b31a`; Windows multigroup validation stayed non-blocking. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only `sm_90`.

## Recent work (current wave, ~2026-07-28)
## 2026-07-27T02:00:00Z — Roadmap wave update
- Reviewed PR #304 / #62 and approved bit-identical Tier A/Tier B paged GQA plus zero-present-allocation invariants.

## 2026-07-28T17:40:00+0000
Approved the final #362 control-flow inference regression-fix cycle.

Full pre-compaction history in `history-archive.md`.