# Leon — History (compacted 2026-08-12)

**Role:** Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry from `inference_metadata.yaml`. Preserve device-buffer ownership, past/present aliasing, exact real-model comparison, reviewer lockouts.

**Historical summary through 2026-08-12:** Generalized shared KV, attention-sink SWA, connectors, prefix payload materialization. Hardened loaders/fusion (unsupported dtypes fail-closed, LayerNorm operand-order guarded, opset validation recursive, `nxrt_*` C ABI). CUDA graph/capture correctness. PR #291 rewind policy split. Unified native CUDA/ORT KV capacity policy. EP plugin compute hardening (BL2/BL3 slot fidelity wave 1). Clippy dead_code cleanup. NEW-1 fix + f16/bf16 marshaling. Stream EP memory leak fix. Device data-transfer contract. TensorRT build fix (#31988). Apple Accelerate arm64 detection (#32001). BF16 LayerNorm PrePack counter + MLFloat16 stats coverage (#31974 — introduced regression, fixed by Coco).

Older detailed work archived in `history-archive.md`.

## 2026-08-12 — PR #31974 final cleanup: PrePack counter, MLFloat16 stats, centralised trait

- Threaded `number_of_pre_packed_weights_counter` through `RunBF16CpuOnly`; PrePack A/B tests now assert counter=0 (non-initializer) and counter=2 (initializer).
- Added `LayerNorm17_MLFloat16_MeanInvStdDev_FloatPrecision` test for fp16 stat precision.
- Moved `is_narrow_float_v` to `narrow_float_utils.h`.
- Verified counter non-vacuity by breaking PrePack and observing test failure.
- Test counts: 21 BF16 (was 20), 107 LayerNorm suite (was 106).
- Head SHA: 59b84aca7a
- ⚠️ This commit introduced a regression — see entry below.

## 2026-08-12 — PR #31974 regression: is_packed default flip caused float LayerNorm breakage

Commit `59b84aca7a` (Leon) introduced a regression: flipped `is_packed` default from `false` to `true` in `LayerNormImpl::PrePack`. `ConvertMLFloat16ToFloatIfNeeded` only sets `is_packed` inside narrow-float branches; for float inputs it is a no-op, so float dispatch incorrectly believed Scale/Bias were prepacked and failed with "Missing Input: Scale". Nine float `LayerNormTest` cases broke. Coco root-caused and fixed in `e036e53d31` (one-line restore of `false` default). Full-suite results: BF16 21/21, LayerNorm 107/107, SkipLayerNorm 26/26. The `narrow_float_utils.h` centralisation was sound and kept.

**Lesson reinforced:** A flag set only on some code paths must default to the conservative value. Set it explicitly where the work happens.

## 2026-08-12 — CUDA capture arc COMPLETE (shared: 11.4 → 23.13 tok/s)
Blocker 3 (PIN) landed as **#852** (`70a5971d`): engine-gated pin of the GQA
fixed-capacity KV seq symbols to constant so the capture classifier stops
force-declining GQA (disqualifying set 53 → 0), keeping the kernel `capture_support()`
gate as an independent backstop; growing/paged KV stays vetoed. Prerequisite #3 of
the 5-blocker chain (#848 → #850 → #852 → #855 → #854). My pin exposed the bf16 GQA
kernel gap I flagged to Sebastian (#855). Team result: native decode **11.4 → 23.13
tok/s**, capture fully engaged (1 segment / 0 seams).

- **2026-08-14 (#921, MERGED):** textproto fixture sweep — converted 29 committed inline-weight ONNX fixtures to `model.onnx.textproto` and established the convention (keep binary only for external-data sidecars or real-ORT/ORT-GenAI package loaders). Added the in-memory textproto→binary ORT shim `tests/common/ort_session.rs`; each conversion round-trip verified and suites re-run green.
## 2026-08-17T22:20Z — Dispatched on DeepSeek-V2-Lite MoE workspace-planner blocker

- Dispatched as `leon-3` to own Gap 1 from Luv's GLM/DeepSeek scope: DeepSeek-V2-Lite MoE cannot run E2E because prepare-only workspace planning cannot resolve runtime-dependent Attention KV shape `v_model.Unsqueeze_18`.
- Target area: session executor / bindings workspace planning around `crates/onnx-runtime-session/src/executor/bindings.rs` near the unresolved Attention input.
- Goal: unblock real-model MoE E2E so Luv can profile and validate QMoE expert-GEMV Gap 2.
## 2026-08-18T00:35Z — DeepSeek-V2-Lite planner fix landed via PR #1150

- Workspace-planner shape-resolution fix for DeepSeek-V2-Lite MoE landed on `main` as part of PR #1150 squash `e075a715`, combined with Luv's oracle/f64 numerics artifact.
- Rachael's initial 🔴 on the silent golden move triggered reviewer lockout; Leon did not revise the rejected artifact. Lockout cleared on merge.
- Final outcome: V2-Lite MoE E2E is unblocked; correctness is gated by a native-CUDA/f64-justified golden rather than CPU bit-identity.


## 2026-08-18T03:15Z — V2-Lite `_d1` planner fix landed; Engine long-context follow-up

- PR #1181 landed as `c9c7f64c`, fixing the V2-Lite additive-mask query-axis workspace-planner under-resolution by deriving exact `_d1` shape through deterministic mask-cone producers.
- Rachael approved the planner path as exact, bounded, and fail-closed; Wallace's final real-model A/B showed the combined classifier+planner unlock is byte-identical over 320 tokens and 1.79× faster under capture.
- Follow-up assigned: long-context `Engine::generate` Attention workspace under-plan on node 38 (requires 33288 bytes vs prepared 16904 around ~320 tokens), reproducing in eager and capture and therefore not capture-specific.
## 2026-08-18T04:15Z — Engine long-context Attention workspace fix merged (#1189)

- PR #1189 landed on `main` as `b416a3e0`, fixing the Engine/native CUDA decode path to re-run governed workspace preparation whenever KV/mask capacity grows.
- Real V2-Lite validation: 340 generated tokens in eager and capture were token-identical; eager 47.32 tok/s, capture 89.69 tok/s, captures=2, replays=336, fallbacks=0.
- Durable lesson: admission/prefill workspace preparation is not enough when Engine decode later grows physical KV capacity; reprepare against persistent decode bindings before eager/capture execution resumes.
