# Cohaagen finding: QMoE GEMV one-task launch for decode

Date: 2026-08-10
Branch: `squad/moe-profile-nextlever`

## Change shipped

QMoE decode uses `qmoe_linear_impl` GEMV when each selected expert has count=1 (< `gemm_min_tokens=2`). The previous launch capped blocks at `SM * 16`, so batch=1 decode grid-strided multiple output features through each CTA and paid the trailing `__syncthreads()` after every output to protect shared `warp_sums` reuse.

The shipped order-preserving lever launches one CTA per output task for small-route GEMV (`routes <= 16`, i.e. batch-1/2 decode). The fp32 K-reduction order inside each output is unchanged: same 256-thread strided partials, same `block_sum`. Because a CTA handles only one task in this regime, the trailing reuse barrier is skipped; larger/prefill cases retain the capped grid-stride path.

## Evidence

Baseline clean steady decode on H200 GPU1: **11.563 ms/tok** median (3 runs, 128 tokens, warmups=1, decode_skip=8).
Final steady decode: **11.511 ms/tok** median in the final run; an earlier same-code run measured **11.382 ms/tok**. Treat as a small ~0.5–1.6% win under shared-host variance.

Nsight Compute on `qmoe_linear_f32` improved duration from ~31.8 us to ~27.6 us. No-eligible scheduler cycles dropped from ~46% to ~23%; CTA-barrier stall share dropped from ~48% to ~37%. Nsight Systems showed `qmoe_linear_f32` total dropping from 317.6 ms to 283.6 ms over the profiled decode window; `qmoe_linear_f16` slightly regressed, but end-to-end stayed positive/no-regression within variance.

Correctness gates passed:
- `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests qmoe_ -- --nocapture` (27 QMoE GPU tests passed).
- `qwen36_35b_a3b_qmoe_native_cuda_matches_fp32_oracle` on GPU1 with CUDA graph enabled: teacher-forced argmax stayed **33803**; autoregressive token@119 was the known benign #722 token **46283**.

## f32/f16 asymmetry

`qmoe_linear_f32` is the down projection from the f32 activated SwiGLU scratch (`activated`) into hidden. `qmoe_linear_f16` is the gate/up projection from fp16 hidden activations. The f32 input is therefore not incidental; casting it to f16 would change activation/down-projection math and was not shipped.
