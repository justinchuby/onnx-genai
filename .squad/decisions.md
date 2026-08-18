# Decisions — live standing directives

Last consolidated: 2026-08-18T01:35Z (Scribe merged V2-Lite graph-capture/QMoE-occupancy arc; processed 15 inbox drops; no archive gate because live ledger stayed below 20KB.)

Standing governance rules and active directives. Full narrative is archived; keep this file to current decisions plus durable rules.

## Ledger health rule

Archive by SIZE, not age. Age-only archiving can silently no-op during high-volume campaigns because most entries are recent. When the live ledger crosses the spawn-budget gate, preserve full history in an archive and keep `decisions.md` to standing directives, active decisions, and pointers. Assemble from inbox drops, dedupe, then delete merged drops; leave `decisions/inbox/README.md`.

## Archived decode-vs-ORT program pointer

The detailed 2026-08-14/15 glm-4-9b-int4 decode-vs-ORT narrative was archived by Scribe at 2026-08-17T19:05Z because the live ledger crossed the size gate while merging the gate/up cp.async NO-GO. See `.squad/decisions/archive/2026-08-17T19-05Z-decode-program-archive.md`. Current active directives remain below.

## Current decode campaign standing

Landed byte-identical wins: #1134 GEMV PF=2 prefetch default-ON, #1137 gate/up SwiGLU symmetric zero-point bias-fold default-ON, and #1139 gate/up RMS-fused occupancy raise default-ON. Detailed review/perf narratives for these and the subsequent no-go probes are archived in `.squad/decisions/archive/2026-08-17T21-40Z-decode-program-archive.md`.

Active standing: native int4 decode is ahead of ORT-CUDA on the three measured models, but claims must continue to carry the CUDA-graph asymmetry until Wallace finishes the ORT CUDA-graph fairness follow-up.

## 2026-08-17 — Kernel-fusion scope: batch-1 vein MINED OUT (NO-GO)

Luv's profiling-only fusion survey closed the batch-1 byte-identical decode campaign. The decisive finding: **QKV fusion already exists in-repo** as `CudaQkvProjectionFusion` (`crates/onnx-runtime-ep-cuda/src/optimizer.rs:2103`), gated by default-OFF `ONNX_GENAI_CUDA_ENABLE_QKV_FUSION`. It is byte-exact and cuts captured GEMV launches/token by −104, but earlier Muse-Glimmer-30B int4 measurement was flat-to-worse (**47.33→47.26 tok/s**), and Luv independently reproduced a hard regression on qwen2.5-14b: median **167.73→149.43 tok/s (−10.8%)**, base winning **3/3** rounds.

Root cause is now evidenced five ways — preperm, cp.async, down occupancy, q/o occupancy, and QKV fusion: batch-1 decode is **weight-DRAM-bandwidth-bound**. Every GEMV streams disjoint int4 weights once/token, so fusion cannot cut the dominant bytes; CUDA graphs already amortize launch overhead, leaving only ~1.1us GPU bubbles and ~10KB activation round-trips (<0.3% of traffic). QKV fusion is worse on 14b because the fused wide-N node drops the tuned `_pipe` / `_down_c2` dispatch and adds a `Split` copy, forfeiting #1134/#1139 wins. Input-RMSNorm→QKV and down→next-RMS folds would likewise save launches/tiny activation traffic, not weight bytes, and are RMS-order delicate; gate/up and lm_head fusion are already done.

Decision: **NO-GO / do not spend another implementation cycle on batch-1 byte-identical fusion or single-kernel micro-opts.** Native already leads ORT on all three measured int4 models, but keep the graph-capture fairness caveat until Wallace verifies ORT under equal CUDA-graph conditions. Coordinator pivot: Wallace owns ORT CUDA-graph fairness follow-up; Luv pivots to GLM/DeepSeek CUDA-kernel support scope. The only remaining structural decode lever is raising arithmetic intensity with M>1 batched verify/spec decode (additive-only/shelved) or lower-bit quantization (oracle-gated, not byte-identical).

## Native-vs-ORT fairness rule

Native-vs-ORT claims must compare the same artifact, quantization, accuracy level, and steady-state methodology with oracle-correct output. If one engine crashes, rejects the graph, runs CPU, disables CUDA graphs, or uses a different weight file/config, report a capability/config gap rather than a throughput multiplier. For ORT-genai decode, verify CUDA provider and share-buffer/cuda-graph fast path are active before quoting tok/s.

## Benchmark and profiling discipline

Separate measured, estimated, and projected. Same-run PR-vs-base deltas beat absolute numbers under shared-host load. For CUDA-graph decode, `ONNX_GENAI_PROFILE_OPS=1` is a host/eager dispatch view and can mis-rank kernels; use `nsys --cuda-graph-trace=node` for kernel mix and `ncu --graph-profiling node --set full` for stall mechanism. A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder.

## Numerics and portability discipline

Default-on CUDA decode optimizations must be portable or explicitly arch-gated with byte-identical fallback. Token byte-identity is an argmax stability claim, not a numeric invariant; numeric changes need oracle/tolerance justification. Preserve Rule 11: unsupported devices must fall back without behavior loss. Env knobs used for A/B must be documented, deterministic under capture, and not hide default regressions.

## Testing and CI standing directives

- `cargo test --workspace` silently truncates on failure; use `--no-fail-fast` for full-suite evidence.
- Run new tests in isolation before trusting full-suite green. Assert on what code did, not summaries.
- An agent self-report is not evidence; verify with code, command output, and tests.
- Reviewer lockout is enforced: authors do not revise their own rejected artifacts.
- CI is asynchronous; required local targeted tests/builds/hardware probes remain blocking, but do not idle solely waiting for CI.
- Never commit `.squad/` files to external repos; if that happens, purge history rather than only deleting in a follow-up commit.

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and `.squad/decisions/archive/`. The complete live ledger immediately before the 2026-08-15T03:10 decode-arc compaction is `.squad/decisions/archive/2026-08-15T03-10-00Z-decisions-pre-decode-arc.md`.

## 2026-08-17 — Native-vs-ORT fairness: the lead is architectural, not per-kernel

Wallace's CUDA-graph fairness study resolved the standing native-vs-ORT caveat. Stock ORT CUDA EP cannot capture a replayable CUDA graph on the current onnxruntime-genai int4 decode models: dynamic KV growth and CPU-assigned shape-massaging nodes make `enable_cuda_graph=1` no-op, fail, or corrupt throughput. Native can capture because it owns the static-shaped KV decode step and device sampler; that capture capability is the moat.

Equalizable eager-vs-eager medians show the mechanism clearly: qwen2.5-14b ORT eager 83.39 tok/s vs native eager+on-GPU-argmax 106.54 (1.28×), but native eager+host-argmax 67.54 (0.81×); qwen2.5-7b ORT 273.61 vs native eager 206.95 (0.76×); Phi-4-mini ORT 230.06 vs native eager 198.03 (0.86×). The production native headline remains real for users — graph+device-argmax delivers 167.08/312.27/313.10 tok/s, or 2.00×/1.14×/1.36× ORT eager — but the lead is **graph-capture + on-GPU argmax integration**, not intrinsically faster per-kernel math. This corroborates the batch-1 kernel NO-GO: do not spend more cycles chasing byte-identical per-kernel decode micro-opts to beat ORT; pursue structural capabilities or higher arithmetic intensity instead.

## 2026-08-17 — GLM/DeepSeek support scope

Luv's GLM/DeepSeek scope pass found support is already mature. Dense GLM-4-9B int4 runs native E2E at 98.2 tok/s, and DeepSeek-R1-distill-Qwen-1.5B int4 runs native E2E at 690 tok/s. There is no GLM/DeepSeek-specific dense-kernel gap: these ride the already-tuned MatMulNBits GEMV family.

MLA is **not** a custom-op gap for the available DeepSeek-V2-Lite export; it lowers to standard RotaryEmbedding + Attention. QMoE router/expert-GEMV, CompressedSparseAttention, IndexShare/DSA, sparse KV gather, and related fixtures already exist and are CPU-oracle validated.

Two active gaps remain. **Gap 1**: DeepSeek-V2-Lite MoE E2E is blocked in the workspace planner because Attention KV input `v_model.Unsqueeze_18` has runtime-dependent shape with no exact graph-metadata bound; owner Leon / Engine-KV-buffers. **Gap 2**: QMoE expert-GEMV is under-optimized relative to the dense-GEMV campaign (no prefetch/occupancy/vectorized-load treatment) and is Luv's CUDA-kernel target under default-off flag `ONNX_GENAI_QMOE_VEC`, with real-model profiling gated by Gap 1.

## 2026-08-18 — DeepSeek-V2-Lite MoE support/correctness merged (#1150)

PR #1150 landed on `main` as squash `e075a715`, combining Leon's DeepSeek-V2-Lite MoE workspace-planner shape-resolution fix with Luv's oracle correction. The branch `squad/qmoe-divergence-fix` supplied the final artifact; `squad/leon-deepseek-v2-planner` was folded in and both branches were deleted after merge. Outcome: DeepSeek-V2-Lite MoE is supported and the native-CUDA decode lock is now explicitly f64/oracle-justified.

Rachael first rejected the planner artifact because it silently moved the golden from the native-CPU stream to the CUDA stream after the planner unblocked E2E execution. That lockout was honored: Leon did not revise the rejected artifact. Luv authored the separate oracle rebase and numerics artifact, then rebased it onto current `main` before merge; Rachael re-gated it green.

Final decision: the apparent "CUDA numerics bug" was a wrong-oracle premise, not a kernel/workspace bug. CPU-vs-CUDA divergence is benign f32 accumulation-order drift. The repo already treats f64, not native CPU, as truth for int4 reductions (dense `matmul_nbits_gpu.rs` precedent uses tolerance and f64-reference tests). The token-5 expert-set swap is a below-fp32-resolution near-tie: CPU selected expert 61 and CUDA selected expert 1 for the sixth top-k slot after identical first five experts; router-logit delta was about `5e-5`, below measured reassociation drift (`4.7e-5`–`6.4e-5`). Rachael ruled that adding an epsilon tie-break would be arbitrary and could pin the model to the less accurate stream.

Evidence: CUDA was closer to f64 truth in every relevant measurement — QMoE GEMV K=512 top_k=1 CUDA/f64 `4.62e-6` vs CPU/f64 `6.98e-6`, top_k=2 `2.93e-6` vs `5.30e-6`, and dense int8 block32 `1.26e-6` vs `3.60e-4` (~285x closer). The new `qmoe_int4_identity_expert_gemv_within_f64_roundoff` test bounds CPU and CUDA separately against f64 instead of asserting CPU==CUDA. The DeepSeek decode lock names the golden as native-CUDA/f64-justified, not CPU-golden. The env-gated router probe `ONNX_GENAI_QMOE_ROUTE_DUMP` remains default-OFF.

Validation before merge: V2-Lite CUDA lock passed with CUDA graphs OFF and ON; all 30 `qmoe_gpu` tests passed; all five dense `matmul_nbits_gpu` f64-reference tests passed; the planner regression `prepare_workspace_resolves_reshape_flatten_chain_exactly` passed; Qwen3 dense native-CUDA lock passed. Luv confirmed the rebased golden held after #1129's MoE-claiming change.

Policy precedent: for int4 GEMV/QMoE reductions, CPU bit-identity is not an oracle when accumulation order differs. Correctness is bounded agreement with an independent higher-precision reference plus deterministic backend output and explicit golden rationale.

## 2026-08-18 — DeepSeek-V2-Lite MoE performance baseline and ORT capability gap

Wallace measured the corrected post-#1150 DeepSeek-V2-Lite int4 MoE artifact on H200: native CUDA decode is **~55.6 tok/s median** in eager mode. CUDA graph ON and OFF are equivalent today because native declines capture on the MLA/MoE attention-mask path (`attention_mask_consumers_are_capacity_aware`); on-GPU argmax is not material for this smaller vocab.

Stock ORT CUDA EP cannot serve this int4 QMoE model on GPU: all 26 `QMoE` nodes lack CUDA kernels, require EP fallback to load, insert 104 CPU/GPU boundary copies, and run at **~0.20 tok/s** with CPU experts. Report this as a capability gap — native is the only measured GPU engine for int4 QMoE here — not as a meaningful native-vs-ORT throughput multiplier.

## 2026-08-18 — QMoE gate/up occupancy lever: GO, default-OFF

Luv rebuilt the QMoE occupancy lever on the corrected #1150 baseline as `ONNX_GENAI_QMOE_OCC` (default-OFF, symmetric int4 gate/up only). The live lever is fused gate/up SwiGLU, not down-proj: `__launch_bounds__(256,6)` cuts registers **54→40**, stays spill-free, raises achieved occupancy **43.3%→63.8%**, and reduces the isolated kernel **42.3→37.8us (−10.6%)**; the `(256,8)` variant spills and is rejected.

Byte-identity is green (`qmoe_occ_is_bit_identical`, full QMoE suite, E2E tokens equal to golden). End-to-end V2-Lite decode converts modestly but consistently, about **56.6→57.2 tok/s (~+1%)**, because gate/up is a small slice of MoE decode. Decision: review/ship as default-OFF modest win; no remaining QMoE occupancy lever after down/grouped paths proved saturated or different-lane.

## 2026-08-18 — V2-Lite MoE CUDA graph-capture unlock: GO scope

Wallace scoped the V2-Lite graph-capture blocker and found the current attention-mask decline is an **over-conservative capacity-policy bail**, not a hard correctness wall. V2-Lite's mask cone (`CumSum`/`Unsqueeze`/`GreaterOrEqual`/`And`/`Where`/`Cast`) terminates in capacity-form `Attention` input 3; padded columns are forced to fp16 −65504, unlike GLM-5.2's logical-width indexer `Add` case.

Decision: implement a topology-gated padded-capacity classification for the additive-mask-builder-to-`Attention[3]` pattern, while keeping the general `CumSum`/`Unsqueeze` denylist and GLM-5.2 regression guard. No kernel edits are expected; tests must pin the V2-Lite positive case and GLM negative case. Gate the feature on empirical byte-identity after implementation (Wallace: frozen-mask capture vs eager, ≥300 tokens, 0.000% divergence). Projected upside is conservative **54.3→~82–86 tok/s** (+50% class), larger than remaining per-kernel micro-opts; Rachael review is mandatory.

## 2026-08-18 — Greedy argmax tie-break canonical remains lowest-index

Wallace confirmed the canonical greedy tie-break is **lowest token id**: host sampler, native tensor argmax, ONNX/ORT ArgMax default, and historical design all use first/lowest. The post-#1112 highest-index device_argmax was a default native-CUDA parity drift, not an opt-in benchmark-only choice. Decision: reconcile device_argmax back to lowest-index and protect with deliberate tie tests; this was later resolved by #1119.

## 2026-08-18 — Device-resident token-feedback loop remains opt-in small win

Holden's `ONNX_GENAI_DEVICE_TOKEN_LOOP=k` keeps sampled tokens on-device across captured replays and is byte-identical on GLM and qwen, with fallbacks=0. It genuinely shrinks the nsys cross-replay gap, but unprofiled wall-clock gain is only **~1%** because the earlier 17.4% host-feedback gap was profiler-inflated. Decision: keep default-OFF/opt-in; require hardware parity tests before any default promotion.

## 2026-08-18 — Older Deckard decode drops deduped to archived/current decode standing

Processed Deckard's pending 2026-08-15 drops for GQA aux merge, flash-decoding split retune, GQA core-warps, lever-2, group-reuse NO-GO, cp.async NO-GO, GEMV v2, fp16 GEMV, lm_head cuBLASLt, and default-on lm_head cuBLASLt. These are already covered by the archived decode-program narrative and the live standing rule that batch-1 byte-identical single-kernel work is mined out; do not duplicate their per-PR tables in the live ledger.
