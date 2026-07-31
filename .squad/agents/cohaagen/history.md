# cohaagen — History

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
- In flight: hybrid-e2e.

## 2026-07-31T00:25:00Z — PR #529 merged (qwen3.5-0.8b hybrid 100% CUDA placement)

- PR #529 merged: qwen3.5-0.8b hybrid places 100% on CUDA — split package embedding.onnx 24 nodes + text.onnx 1265 = 1289 nodes, 0 declines (after #480/#484/#525). Regression lock `qwen35_0_8b_placement_lock`.
- E2e decode still BLOCKED on the loader: `Engine::from_dir` rejects the 3-onnx split; `from_pipeline_dir` refuses during vision `smart_resize` admission. Parity harness `qwen35_0_8b_hybrid_native_cuda_e2e` graceful-skips until fixed.
- In flight (cohaagen-4): loader-unblock — admit the text-only split hybrid for decode, flip the e2e parity harness active.
