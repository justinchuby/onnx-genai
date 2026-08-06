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
