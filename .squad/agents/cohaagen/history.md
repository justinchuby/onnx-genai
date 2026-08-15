# cohaagen — History

## Role
CUDA performance and kernel specialist. Owns CUDA EP op-coverage (#67), weight-offload (#63/#87), kernel tuning (o_proj split-K, CausalConvWithState, LinearAttention), and decode profiling.

_Entries before 2026-08-11 archived to `history-archive.md` (Scribe rounds 1–9 + 2026-08-12 compaction). Archived: PR #380 re-review; 7B perf; #63/#87 weight pager; #480/#484/#525 kernels; #529; #535/#544/#552; GQA capture; 27B roofline; fused LinearAttention; Foundry sweep; DeepSeek/GLM; Thread-3 hetero; 35B-A3B full unblock chain through #618/#625/#676/#684/#700/#708._

## 2026-08-11T21:10:00Z — Upstream audit: SM-count columns-per-CTA for M=1 MatMulNBits

- Confirmed upstream hardcodes `kColsPerThreadBlock = 8` with `grid.y = 1` in `matmul_4bits_m1_impl.cuh:135`.
- No SM-count adaptation exists in the M=1 or batched GEMV paths.
- PR #29469 (online tuning of small-M cap) is orthogonal — tunes *which* kernel, not grid geometry.
- No colliding in-flight work found across 30+ PRs and recent commits.
- **Verdict: genuine uncovered gap.** Contribution is small (~25 LOC), bit-identical, upstream-idiomatic.
- Caveat: +2.08% claim has no provenance; fresh benchmarks on ≥2 GPU generations required before PR.
- Wrote `.squad/decisions/inbox/cohaagen-matmulnbits-upstream-audit.md`.

## 2026-08-11 — MatMulNBits SM-adaptive grid PR shipped

- Implemented `SelectColsPerBlock(n, sm_count)` in upstream ORT.
- Templated M=1 kernel on `cols_per_block` (8/4/2).
- 10 files changed, ~225 insertions, 62 deletions.
- Opened draft PR: microsoft/onnxruntime#31988.
- No performance numbers published; benchmark methodology documented in PR body.
- Leak check passed (no persona names, no squad files in committed content).

## 2026-08-12T00:15:00Z — MatMulNBits upstream workstream recap

- Upstream audit confirmed genuine gap: `kColsPerThreadBlock = 8` hardcoded, no SM adaptation anywhere in M=1 or batched GEMV paths.
- Implementation shipped as draft PR #31988: `SelectColsPerBlock(n, sm_count)` → 8/4/2, templated kernel, multiProcessorCount threaded through 3 layers.
- No performance claims published; benchmark methodology documented in PR.
- Routing guard added by Chew under lockout: accepted-shape set preserved exactly (n%8==0 required).
- PR stays draft until GPU benchmarks on ≥2 GPU generations.
- CPU AMX QNBit prefill: no PR (host has no AMX/VNNI).
- Split-K excluded: 2-way K_SPLIT=2 regressed 7B o_proj GEMV by −0.59%.
## Historical context
Older detailed dated entries through 2026-08-04T00:40:00Z — PR #625 native loader and 35B QMoE follow-through were moved to `history-archive.md` during Scribe compaction on 2026-08-11T03:25:00Z. Keep this live file focused on current routing-relevant context; full chronology is preserved in the archive.

## 2026-08-06T00:00:00Z — 35B-A3B native sparse QMoE shipped

- Cohaagen-34 fixed native CUDA QMoE `router_probs` rank handling for 3-D tensors, measured Config A at 31.13 ms/tok / 32.12 tok/s, and opened #676.
- Cohaagen-35 measured Config C (ORT-GenAI 0.14.1 / ORT 1.27 full stack, dense-fallback Q4_K_M) at 461.23 ms/tok / 2.17 tok/s.
- Cohaagen-36 used a full-fp32 oracle to adjudicate token-119: QMoE token 33803 matches oracle, dense int4 token 5342 is the low-precision outlier; regression test landed in #676.
- Coordinator merged #625 and #676; 35B-A3B native sparse QMoE is shipped at roughly 12.5–14.8× over the ORT dense-fallback stack.

## 2026-08-06T00:00:00Z — PR #700 hybrid Mamba cache correctness

- Fixed #695 by disabling native host/device KV-mirror prefix reuse whenever `has_recurrent_state()` is true, forcing full recompute for hybrid Mamba/attention decoders.
- Kept single-shot byte identity and added always-on gate coverage plus an env-gated GPU continuation regression where reused argmax matches the fresh oracle token `33803`.
- PR #700 merged and closed #695; ORT paged-reuse residual tracked separately as #701.

## 2026-08-06T12:30:27Z — PR #684 QMoE router parallelization

- Cohaagen-37 profiled 35B-A3B QMoE decode and proved `qmoe_route` was the roofline limiter: 65.3% GPU time, rows=1 row-parallel top-k, GPU effectively idle.
- Authored merged PR #684: block-cooperative byte-exact top-k router, 27/27 qmoe GPU tests, decode improved 30.99 → 16.14 ms/tok (1.92×, ~62 tok/s).
- Updated issue #610 scorecard to ~24× over dense and ~28.6× over the ORT-GenAI dense-fallback ceiling; next levers are CUDA-graph capture repair and norm/pointwise fusion.

## 2026-08-06T19:40:00Z — 35B-A3B CUDA-graph capture C3 shipped

- Shipped PR #708/C3 making GatedDeltaNet Split capture-safe (resolved output-shape sizes, no host-read/sync): 13.415 → 12.132 ms/tok, 184 → 154 segments, token@119 `33803`; rejected unsafe C2 sync elision and moved on to pinned-vs-growing symbol classification after strict-C1 proved a no-op.

- 2026-08-07: C1 build-time growing-symbol classifier mechanically collapsed capture segments (154→34, +4.3% tok/s) but was shelved behind #722 due to a 35B QMoE fp16 captured-vs-eager near-tie; revive after fp32-teacher-forcing oracle re-anchor.

## 2026-08-11T03:25:00Z — GLM/DeepSeek native load gaps fixed

- Fixed GLM-4-9B native CUDA load by honoring runtime CUDA KV capacity instead of reserving metadata `max_sequence_length=131072`; PR #770 auto-merge pending CI.
- Fixed DeepSeek-V2-Lite QMoE by accepting Cast-backed scale inputs as fp32 scale sources rather than requiring direct initializers; PR #771 auto-merge pending CI.
- Validation: GLM native greedy lock passes at 93.6 tok/s, DeepSeek lock passes at 52.5 tok/s, CPU/CUDA first-token top-40 sets match; no model-name gates, and root cause was not partial rotary or MLA.

## 2026-08-11T16:03:10Z — 27B hybrid GDN native CUDA shipped

- Enabled Qwen3.5/3.6-27B hybrid GDN native CUDA: the 27B artifact's thin `inference_metadata.yaml` carries no `io` port contract, so `resolve_kv_layers` returned None ("per-layer KV page geometry unknown").
- Fix `maybe_fill_hybrid_io_from_graph` in `engine/load.rs` auto-derives the decoder io contract from the ONNX graph port inventory, gated on non-empty `state_pairs`; DRY, no model-name gate — unblocks the whole hybrid GDN family.
- Byte-exact: native argmax 11751 " Paris" == fp32 oracle, top-1 margin 2.549 nats; locked by `qwen35_27b_hybrid_native_cuda_e2e.rs`. PR #779 (auto-merge, awaiting CI).

- **2026-08-14 (#914, MERGED):** DeepSeek-V2 tiny Attention+QMoE golden decode-lock — committed fixture (RotaryEmbedding + standard ONNX Attention + int4 QMoE), prompt `[3]` locks greedy `[11]×8` on native CPU and CUDA. Standing fact: DeepSeek-V2-Lite native path = standard Attention + QMoE, NOT a custom MLA op.
