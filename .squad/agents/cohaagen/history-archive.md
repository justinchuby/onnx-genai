# cohaagen — History Archive

_Archived by Scribe round 9, 2026-07-31T10:24:07Z._

## 2026-07-29T00:45:00+0000 — PR #380 re-review
- Re-reviewed Melina's encoder-decoder fixture correction for issue #377 and approved PR #380, merged as `47c3331d`.
- Ran the CLI ORT E2E gate (23/23); metadata/I/O-detection reviews require that gate alongside engine/native unit tests.

## 2026-07-30T09:16:00Z — 7B native CUDA perf findings

- The Foundry baseline reports native CUDA ahead of ORT; 7B tracing localized o_proj at 19.5% of kernel time.
- Reverted the two-way o_proj split-K gate after a repeatable 0.59% 7B regression; do not retry that lever without a new higher-split kernel experiment.

## 2026-07-30T13:36:00Z — Issue #63 increment 2 delivered; #87 assessment complete

- PR #444 merged: wired GPU weight pager into live decode hot-path with `ONNX_GENAI_WEIGHT_OFFLOAD=1`. Added `CudaWeightResidency` LRU device-page cache, env-gated device policy, extended lazy boundary to `com.microsoft::QMoE` + `MatMulNBits`.
- Test: 0.6B native CUDA on 2MiB budget: 32-token decode 2.79 → 2.30 tok/s (1.21× slowdown); tokens identical; paging counters confirm page-ins/evictions.
- Issue #87: assessed async H2D prefetch infrastructure — all async/copy-stream machinery exists and is GPU-tested. Missing only live wiring (residency currently synchronous). Inc1 = switch to async, Inc2 = double-buffered look-ahead. Unblocked, awaiting signal.

## 2026-07-30T15:20:00Z — PR #480 merged (CUDA CausalConvWithState + GBQ coverage)

- PR #480 merged (Melina APPROVED): NVRTC `CausalConvWithState` kernel (fp32/fp16/bf16, f32-accum to match ORT/CPU) + `GatherBlockQuantized` coverage declaration (#67). Oracle chain CUDA->CPU EP->ORT; advertised ops 161 -> 163. Closed the registered-but-undeclared GBQ coverage-of-coverage hole.
- Empirical #67 audit: classic transformer decode is 100% covered on CUDA; control-flow (If/Loop/Scan) is executor-handled and must NOT be added to the EP; remaining real gaps are the Qwen3.5 hybrid family (CausalConvWithState landed, LinearAttention next).
- Follow-up (safe-to-defer, fail-closed): GBQ bits=4 odd-blocks-per-row. In flight: LinearAttention (rank-2 gap), PR pending.

## 2026-07-30T21:15:00Z — PR #484 + #525 merged (CUDA LinearAttention; #67 coverage-polish)

- PR #484 merged: CUDA LinearAttention (Gated DeltaNet) kernel — per-thread f32-register-column state; 4/4 parity; qwen3.5 hybrid node placement 0→18/18/24. Coordinator resolved a merge conflict with #480 in kernels/mod.rs/provider.rs/docs (union of ops).
- PR #525 merged (Melina APPROVED): #67 coverage-polish — RotaryEmbedding com.microsoft + fixed dtype-check bug (Int64 position_ids compared vs float); Bool NonZero on both EPs via `to_dense_bool`; GatherBlockQuantized odd-blocks-per-row LOUD fail-closed gate + honest doc softening. Op counts unchanged (no new op names).
- Qwen3.5 hybrid recurrent op set now fully CUDA-covered (CausalConvWithState #480 + LinearAttention #484 + RoPE-contrib/NonZero #525).

## 2026-07-31T00:25:00Z — PR #529 merged (qwen3.5-0.8b hybrid 100% CUDA placement)

- PR #529 merged: qwen3.5-0.8b hybrid places 100% on CUDA — split package embedding.onnx 24 nodes + text.onnx 1265 = 1289 nodes, 0 declines (after #480/#484/#525). Regression lock `qwen35_0_8b_placement_lock`.
- E2e decode still BLOCKED on the loader: `Engine::from_dir` rejects the 3-onnx split; `from_pipeline_dir` refuses during vision `smart_resize` admission. Parity harness `qwen35_0_8b_hybrid_native_cuda_e2e` graceful-skips until fixed.
- In flight (cohaagen-4): loader-unblock — admit the text-only split hybrid for decode, flip the e2e parity harness active.

## Archived from live history (Scribe compaction 2026-08-12T00:15:00Z)

Entries 2026-07-31T03:03:15Z through 2026-08-07 moved here. Summary:
- PR #535 text-only pipeline synthesis; #544 async page-in; #552 observability; GQA capture.
- 27B roofline profile: Scan 56.5%; MatMulNBits already at roofline. o_proj split-K (K_SPLIT=2) regressed −0.59% — do NOT retry.
- #594/#592 fused LinearAttention merged; 27B 40.8 ms/tok.
- Foundry sweep: native ahead on all dense models. DeepSeek/GLM sweep corrected stale metadata.
- Thread-3 hetero legalization #602 (locked out, Deckard revised) and #606 scaffold merged.
- 35B-A3B: PackedMHA root cause (mobius export bug); fairness vet; fp16 TopK PR #612.
- PR #616 35B-A3B native GPU unblock: cuDNN reduce fix, 2726 ms/tok corrected.
- PR #618 Lever A reduce capture: 2726 → 405 ms/tok.
- PR #625 + #676: 35B-A3B native sparse QMoE shipped ~12.5–14.8× over ORT dense stack.
- PR #700 hybrid Mamba cache correctness.
- PR #684 QMoE router parallelization: 30.99 → 16.14 ms/tok, ~62 tok/s.
- PR #708/C3 CUDA-graph capture: 13.415 → 12.132 ms/tok; C1 classifier shelved behind #722.
## 2026-08-11T03:25:00Z: Scribe compaction from live history

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
- The Foundry baseline reports native CUDA ahead of ORT; 7B tracing localized o_proj at 19.5% of kernel time.
- Reverted the two-way o_proj split-K gate after a repeatable 0.59% 7B regression; do not retry that lever without a new higher-split kernel experiment.

## 2026-08-02T10:05:00+0000 — Fused LinearAttention + Inc-1b default wave

- Authored #594 (EP-driven keep-as-op and dual-domain fused LinearAttention) and #592 (removed `ONNX_GENAI_DECODE_INLINE_SCAN`; automatic graph-property gate). Both merged to main.
- Reported 27B fused LinearAttention decode around 41.3 ms/tok with 48 fused ops / 0 Scans; composition oracle on final main was in flight at Scribe time.

## 2026-08-02T11:40:00+0000 — Final fused LinearAttention validation

- Ran the final-main #592 + #594 composition oracle: 27B byte-exact greedy PASS (CUDA == CPU == expected ids) and structural 48 fused `LinearAttention` / 0 Scans PASS.
- Confirmed steady decode on merged main at 40.8 ms/tok (24.5 tok/s); fused-LinearAttention lane complete.

## 2026-08-02T15:45:00+0000 — Foundry native-vs-ORT sweep and 7B scope

- Ran dense Foundry native-CUDA vs ORT-CUDA steady sweep: native beat ORT on every dense model (0.5b 1.75×, 1.5b 1.66×, Phi-4-mini 1.35×, 7b 1.14×); no native loss found.
- Scoped qwen2.5-7b GQA and found no bounded win: Nsight showed MatMulNBits/HBM-bound decode (~88% GPU kernel time); tested GQA tweak regressed and was reverted.
- Adjudicated qwen2.5-1.5b divergence as native-correct and authored merged PR #597 regression lock.

## 2026-08-02T16:45:00+0000 — DeepSeek/GLM post-#434 metadata validation

- Ran the DeepSeek/GLM sweep and confirmed initial broad failures were stale pre-#434 artifact metadata, not product bugs.
- Regenerated sidecars with current Mobius #434 metadata: DeepSeek-V2 tiny native+ORT admit, DeepSeek-V2-Lite real int4 native/ORT tokens match with native ~250× faster, and GLM-4-9B native matches golden greedy lock.
- GLM-5.2 tiny/q4/qmoe remain blocked on unmerged Mobius #404; add a real DeepSeek-V2-Lite non-tiny-vocab lock before treating it as full correctness coverage.

## 2026-08-02T19:00:00+0000 — Thread-3 hetero legalization design + #602

- Authored the Thread-3 pipeline map and relocation design; key finding: the multi-EP kept-function correctness hole is latent because `hetero.rs` is not wired into the default single-EP session path.
- Implemented bounded Phase 0+1 in #602: post-assignment legalization fixpoint in `hetero::plan`, fail-closed ambiguous `(domain, op_type)`, and CPU-only fake-provider tests. Locked out after Harry's round-1 review; Deckard took the revision.

## 2026-08-03T02:40:00+0000 — PR #606 Thread-3 Phase 3 scaffold

- Authored and merged #606 after finding cross-EP tensor movement is only in standalone `hetero::execute`, not the stateful session `Executor`; did not fake per-op hetero execution.
- Shipped opt-in `ONNX_GENAI_HETERO` fail-closed placement guard, `SessionError::HeterogeneousExecutionUnsupported`, C API mapping, and 8 new CPU-fake hetero tests; single-EP flag-off path remains byte-identical.
- Coordinator resolved the stale decode-inline option flip and pivoted from unused integrated Phase-3 execution to the Qwen3.6-35B-A3B `PackedMultiHeadAttention` admission bug.

## 2026-08-03T03:10:00+0000 — 35B-A3B PackedMHA root cause

- Root-caused the Qwen3.6/Qwen3.5 35B-A3B `vision_encoder` admission failure as a mobius export bug, not an onnx-genai loader bug.
- Confirmed ORT `PackedMultiHeadAttention` requires the optional `bias` positional slot before `token_offset`/`cumulative_sequence_length`; mobius call sites emitted 6 inputs but the fallback function declared only 5 formals, so onnx-genai correctly rejected `6 > 5`.

## 2026-08-03T06:40:00+0000 — 35B-A3B validation and fairness vet

- Validated Qwen3.6-35B-A3B under strict apples-to-apples methodology: ORT-CUDA crashes, while native is blocked by fp16 MoE-gate `TopK`, rank-3 mRoPE positions, and unimplemented native pipeline decode, so 35B is a capability gap on both engines today.
- Fairness-vetted native-vs-ORT claims for issue #610: report only same-artifact steady-state oracle-correct comparisons as throughput multipliers; report crashes, graph rejects, and CPU/different-kernel fallbacks as capability gaps.
- Started fp16 TopK CUDA enablement on `squad/cuda-fp16-topk` to unblock dense_fallback MoE routers.

## 2026-08-03T07:40:00+0000 — PR #612 fp16 TopK conformance and GAP-3 kickoff

- Authored merged PR #612 after finding fp16/bf16 CUDA TopK was already on main via #445; kept the work test-only and added router-shape fp16 GPU==CPU, non-final-axis, and EP-claim coverage.
- Validated 15/15 GPU tests, CUDA clippy `-D warnings`, and fmt; flagged latent CPU EP non-final-axis TopK ordering as a separate follow-up.
- Began GAP-3 native pipeline decode scoping for Qwen3.6-35B-A3B.

## 2026-08-03T09:00:00Z — 35B-A3B origin/main revalidation correction

- Cohaagen-24 found GAP-3 native pipeline decode was already landed on `origin/main`; PR #613 merged as docs truth-up/design note instead of duplicating implementation.
- Cohaagen-25 revalidated 35B-A3B on fresh `origin/main` @ `0a5ac3c5`: full native pipeline decode is output-correct vs native-CPU oracle, and fp16 TopK/Softmax execute on CUDA EP.
- Current native GPU blocker is now a cuDNN f16/bf16 `ReduceSum` compute-type bug plus decoder device-wiring; ORT-CUDA still hard-crashes. Cohaagen-26 is in flight to fix and measure.

## 2026-08-03T10:00:00Z — PR #616 35B-A3B native GPU unblock

- Authored merged PR #616 fixing cuDNN f16/bf16 reduce compute-type behavior and native pipeline decoder device wiring.
- Measured Qwen3.6-35B-A3B native GPU decode on H200 at 2726 ms/tok / 0.37 tok/s, correct vs native-CPU oracle and GPU-confirmed; ORT-CUDA still crashes, so this is a native capability number.
- Follow-up perf work is host/sync overhead and low GPU utilization, not correctness/unblock.

## 2026-08-03T12:30:00Z — 35B-A3B Lever A reduce capture

- Profiled 35B native CUDA decode and found capture shredded by 10240 fp16 `ReduceSum` seams from dense_fallback MoE, causing host/sync-bound 2725 ms/tok decode.
- Authored merged PR #618 making cuDNN float ReduceSum/Mean capture-eligible via descriptor/workspace caching, sync gating, warmed axes, and shape-stable capture-safe marking.
- Measured 35B decode improvement to 405 ms/tok / 2.47 tok/s (~6.7×), byte-exact vs CPU oracle; Lever B now targets ReduceSumSquare/Split/LinearAttention seams.

## 2026-08-04T00:40:00Z — PR #625 native loader and 35B QMoE follow-through

- Delivered the #625 native loader fix with GraphIo/GraphIoMetadata and rebased onto origin/main; locked out after Harry found the initializer-input leak.
- Confirmed ORT 1.28 still rejects fp16-activation/fp32-scale QMoE, making config-B a capability gap rather than a loader issue.
- Left GPU watcher PID 1060559 for config-A measurement while external vLLM occupies the GPU.
