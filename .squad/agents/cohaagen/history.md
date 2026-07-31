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
