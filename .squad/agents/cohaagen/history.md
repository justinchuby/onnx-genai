# cohaagen — History

## Role
CUDA performance and kernel specialist. Owns CUDA EP op-coverage (#67), weight-offload (#63/#87), kernel tuning (o_proj split-K, CausalConvWithState, LinearAttention), and decode profiling.

_Entries before 2026-07-31T03:03:15Z archived to `history-archive.md` (Scribe round 9). Archived: PR #380 re-review; 7B perf (o_proj split-K revert); #63 inc2 (#444 weight pager); #480 (CausalConvWithState+GBQ); #484+#525 (LinearAttention+RoPE-contrib/NonZero); #529 (qwen3.5 100% CUDA placement)._

## 2026-07-31T03:03:15Z — PR #535 merged: text-only decode pipeline synthesis (hybrid loader unblock)

- PR #535 merged: unblocks the split VLM package whose image preprocessing is unrepresentable (`smart_resize`). New `GenAiConfigError::UnrepresentablePreprocessing` → `to_strict_text_only_pipeline_metadata` synthesizes an embedding→decoder AR pipeline with NO vision component. Modality-driven, NOT a model-name case. Flips the `qwen35_0_8b_hybrid_native_cuda_e2e` parity harness active.
- Weight-offload chain (#63): #444 first increment merged — `page_lazy_weight` dispatch-layer seam + `CudaWeightResidency` LRU, gated behind `ONNX_GENAI_WEIGHT_OFFLOAD=1`. NEXT: #87 async page-in (PLAN-ONLY, awaiting Justin green-light).
- Guardrail proven: o_proj 2-way split-K (K_SPLIT=2) REGRESSES the 7B o_proj GEMV (−0.59%) — **do NOT re-try**. A K_SPLIT>2 new kernel is the future candidate.

## 2026-07-31T08:48:28Z — PR #544 async page-in MERGED; #552 observability MERGED; GQA capture validated

- PR #544 merged (Harry test fix; Melina RC→APPROVE): async `cuMemcpyHtoDAsync`+`compute_wait_fence` page-in; eviction re-serializes (WAR); ~1% win at 2 MiB only; byte-identical all budgets. `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN` default ON.
- PR #552 merged (Lori APPROVE): `NativeDecodeSession::load_with_resolved_io` — genai_config decoders now emit `cuda_graph captures/fallbacks/fallback_report` + `--trace` capture-reject reasons. `profile_native` simple path now fully observable.
- GQA capture default-on validated (verification-only, no code change): Qwen2.5-0.5B int4 GQA — ON 2.14× eager, byte-identical, zero declines. Standing question answered. Gemma-4-E2B unmeasurable: stale HF export + GAP-3. Capture default-on for multi-component deferred until mary's native pipeline + re-export.

## 2026-07-31T10:24:07Z — 27B decode roofline profile (profile-only, no code changed)

- Profiled Qwen3.6-27B int4 native-CUDA resident decode on H200. Steady-state: 168 ms/tok (~35× off roofline).
- Root cause: `Scan` (48 LinearAttention blocks/step) = **56.5%** of decode step — structurally un-capturable control-flow. MatMulNBits already at roofline (4.4 ms, 2%) — **do NOT touch for this**.
- Capture engages (fallbacks=0); Transpose per-call sync 9.2% is next lever in Cohaagen's lane (separate from Scan fix).
- Fix is Justin's / pipeline-control-flow area. This was a profile+identify task only; no worktree remains.
