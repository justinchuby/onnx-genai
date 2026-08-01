# mary — History

## Role
Lead engineer, large-model memory and multi-component pipeline decode. Owns the native pipeline seam (#384 increments), weight-offload A/B on 27B+ models, and 35B-A3B bring-up.

_Entries before 2026-07-30T21:15 archived to `history-archive.md` (Scribe round 8). Archived: CUDA reduction/claim-gate, DeepSeek/GLM bring-up, 27B persistent-state, Conv/Silu blockers, #445 TopK, Inc1/Inc2a/Inc2b pipeline._

## 2026-07-30T21:15:00Z — Native pipeline Inc3a + Inc3b merged (#485, #487)

- PR #485 merged: Inc3a — CUDA native decoder via `inputs_embeds`; on-GPU token parity at positions [0,5,6,7].
- PR #487 merged (Lori APPROVED): Inc3b — generic routed CUDA ports; `decode_cuda_eager_step_inputs`/`prepare_cuda_owned_step_inputs` metadata-driven; removed `load.rs` CUDA Routed refusal; captured fast path byte-identical; KV device-resident. mask/ReduceSum finding = ARTIFACT not blocker, proven by real qwen3-0.6b native-CUDA e2e locking 32 tokens to ORT-CUDA on a mask-consuming decoder.
- MILESTONE: native multi-component pipeline CUDA decode path (Inc2a→Inc3b) fully on main; real qwen3-0.6b native-CUDA matches ORT-CUDA for 32 tokens.
- In flight: Inc3c (perf).

## 2026-07-31T00:25:00Z — Native pipeline Inc3c merged (#533) — native CUDA decode BEATS ORT

- PR #533 merged (Lori APPROVED): Inc3c — THE landmark. Real qwen3-0.6b, device 4, real ORT-CUDA: captured ceiling 612 tok/s, eager pipeline 220, ORT-CUDA 443. Default-off `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` writes a persistent `[1,1,width]` device binding per routed port each step, reusing captured `run_one_token` (mask frozen, KV device-resident) ⇒ 1.38x ORT WIN. Metadata-driven from `session.inputs()` (load.rs:481-508); generalizes to 35B-A3B GQA. Engagement proven non-tautologically via counter `NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES` (OFF=0/ON=3, tokens byte-identical).
- In flight (mary-2): real-model capture-engagement validation (does real qwen3-0.6b ENGAGE capture & beat ORT, or DECLINE via Concat-KV) + default-on recommendation.

## 2026-07-31T03:03:15Z — LANDMARK: rank-3 mrope native positions (#543) + capture-flag class finding (#541)

- PR #543 merged (Melina APPROVED, e2e 1 passed/0 failed, 309 conformance cases): rank-3 mrope native positions — native-CUDA hybrid decode == ORT token-for-token (16 identical tokens) on real qwen3.5-0.8b, the first real-weights split-package `inputs_embeds` native == ORT proof. rank-2 byte-identical.
- DRY pattern (reuse this): shared `decode::position_ids_from_starts(starts, input_len)` factored from ORT `build_position_step`, called by BOTH ORT and native drivers. Coordinate rank comes from the declared `position_ids` shape via `declared_position_rank` (rank 2 → 1 legacy `[1,S]`; rank 3 → static leading dim; symbolic → loud error) — NO hardcode-to-3, NO model-name gate; stored once on `NativeDecodeSession`+`DecodeCudaState`.
- Execution gap fixed: ep-cuda `range.rs` rejected `[1]`-shaped single-element scalars (the mrope `k_mrope/range/Range` gap). Lesson: 100% CUDA placement is NOT execution — a covered op can still reject a real graph's tensor shape (#529 placement != #543 execution).
- PR #541 merged (validation-only): capture-step-inputs flag only engages for multi-component `inputs_embeds` decoders; qwen3-0.6b is the wrong class (single-component, `Engine::from_dir`, counter=0 DECLINE proven GREEN via `qwen3_0_6b_capture_step_inputs_decline`). Its 614/206/433 tok/s beats-ORT-1.42× is the token-id CUDA-graph lever, not the capture flag. Keep default-off until a real-weights `inputs_embeds` model runs it e2e.

## 2026-07-31T08:48:28Z — 27B offload A/B proven; session-reuse fix #554; pipeline keystone test

- **27B offload A/B (H200):** Qwen3.6-27B int4 (497 MatMulNBits). 2 GiB budget → 6.2 GiB peak VRAM (2.9× vs 17.7 GiB resident), byte-exact at all budgets. Cliff: ≤12 GiB → 0.11 tok/s (bandwidth-bound, whole working set evicted/token). cuda_graph auto-off under offload. Action item: add `model.io` block to canonical int4-cuda package.
- **#553 bug found (#554 MERGED, Harry APPROVED):** `NativeDecodeSession` 2nd+ generation returns garbage — conv_state/recurrent_state not zeroed between generates. Not graph-capture related. Fixed.
- **Native pipeline keystone test:** `native_full_pipeline_parity` drives `tiny-gemma4-vlm` with native embedding + native decoder simultaneously → `[0,5,6,7] == ORT`. Closes composite-native safety gap ahead of 35B-A3B wiring.

## 2026-07-31T10:24:07Z — Scan-capture increment assessed: STOP+report; right-sized workstream proposed

- Reproduced 176.7 ms/tok baseline (matches Cohaagen ±5%). Confirmed `--trace` Scan seam.
- Single-trip Scan inline is NOT an increment: (1) shared prefill+decode plan — static inline corrupts prefill; (2) control-flow structurally declined at `provider.rs:458`; (3) no child-body-fold machinery.
- **Right-sized: Approach 1 (runtime dual-path)** — 1a correctness behind flag, 1b body enters capture. Blast radius: #443/#543 core. Reference decode tokens locked. **Awaiting Justin go-ahead.**
- No code changed; worktree `wt-mary-scan` removed.
