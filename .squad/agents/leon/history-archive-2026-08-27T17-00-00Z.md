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

## 2026-08-24 — CUDA DFT via cuFFT (PR #2080)

- Added f32 real/complex CUDA DFT: forward/inverse, full/onesided, arbitrary axis, and truncating/zero-padding dft_length.
- Added wheel-aware dynamic cuFFT loading, RAII stream-bound plans, a 16-entry LRU, and EP-governed packed/work workspace; capture fails closed.
- RTX 4060 CUDA 13.1 matrix: 7 passed, 0 failed, 0 ignored. Sign, inverse-scale, and N/2+1 mutations each failed their dedicated test.
- Verified `nvidia-cufft==12.1.0.78` + `nvidia-nvjitlink==13.1.115`; binaries remain external NVIDIA-licensed dependencies.
- Broader conformance coverage guard has four pre-existing missing profile entries (PagedAttention, Mish, Celu, TensorScatter); DFT is correctly profiled.

## 2026-08-25 — Batched CUDA STFT (PR #2091)

- Added f32 CUDA STFT v17 using #2080's shared cuFFT RAII plan/cache/workspace and #2083's CPU/shape semantics.
- One fused kernel extracts/windows all complete frames, one PlanMany call transforms batch×frames, and one kernel unpacks output; capture and strided inputs fail closed.
- RTX 4060 CUDA 13.1 suite: 9 passed, 0 failed, 0 ignored; DFT regression 7/7. Workload [2,16,1], N=5, step=3 used 4 frames/signal, FFT batch 8, 2 explicit NVRTC launches + 1 cuFFT call, 512 B governed workspace.
- Window-ignore, non-overlap-step, dropped-final-frame, and missing-Nyquist mutations each failed their dedicated tests.
- No throughput claim; Linux/H100/H200 and cuFFT internal launch count remain unmeasured.

## 2026-08-25 — CPU/plugin NonMaxSuppression KernelSizedOutput (PR #2112)

- Migrated NMS to #2101 host-only KernelSizedOutput; selection runs exactly once and returns owned Int64 `[selected,3]` bytes shared by native/plugin paths.
- Plugin now claims NMS via exact census; ORT E2E assigned NMS to cpu_ep and produced exact dynamic `[4,3]` rows with one 96-byte materialization copy.
- CPU targeted NMS: 8 passed; plugin census: 5; generic kernel-sized: 4; real NMS E2E: 1. Clippy passed.
- IoU, score-threshold, center-box, double-selection, skipped-copy, removed-census, and empty-census mutations were all caught.
- CUDA NMS remains deferred until the device-sized-output policy is established by CUDA Unique work.

## 2026-08-25 — Bounded CUDA NonMaxSuppression (PR #2130)

- Reused #2113 DeviceWorkspace and #2112 CPU NMS semantics; f32 static contiguous boxes/scores, <=256 boxes and <=256 batch×class groups.
- Parallel per-group score filter/bitonic sort plus bounded deterministic suppression; only 8-byte count D2H, then one device materialization. No full-input D2H or double selection.
- RTX 4060 GPU suite: 6 passed, 0 ignored; real CUDA plugin E2E: 1 passed; Unique policy regressions: 2 passed; exact CUDA/plugin Clippy passed.
- Representative B=1/C=2/N=32: 1 prepare, 1 count, 1 materialize launch; 16 rows, 272 B governed workspace, 8 B D2H.
- Seven requested mutations were caught. No throughput claim; capture and out-of-bound geometry fail closed.


## 2026-08-26 — #1896 synchronous H2D contract retained

Established that synchronous pageable H2D must complete device residency before return and order against the EP nonblocking compute stream; async overlap remains in `htod_async` with explicit fences. The initial revision was rejected because its schedule was not formal and Drop could panic; teardown paths must remain non-panicking.
<!-- Full pre-compaction hot-history snapshot archived by Scribe on 2026-08-27; original hot history above is preserved subject to checkout line-ending normalization. -->
