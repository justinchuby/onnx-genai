# cohaagen — History

## Role
CUDA performance and kernel specialist. Owns CUDA EP op-coverage (#67), weight-offload (#63/#87), kernel tuning (o_proj split-K, CausalConvWithState, LinearAttention), and decode profiling.

_Entries before 2026-07-31T03:03:15Z archived to `history-archive.md` (Scribe round 9). Archived: PR #380 re-review; 7B perf (o_proj split-K revert); #63 inc2 (#444 weight pager); #480 (CausalConvWithState+GBQ); #484+#525 (LinearAttention+RoPE-contrib/NonZero); #529 (qwen3.5 100% CUDA placement)._

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
