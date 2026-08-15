# Decision drop — optional cuBLASLt fp16 lm_head decode path

**Author:** Deckard
**Branch:** `squad/int4-gemv-wideload-v2`
**Date:** 2026-08-15
**Scope:** dense fp16 M=1 `MatMul` used by glm lm_head / final vocab projection. GEMV kernels in `matmul_nbits.rs` were not touched.

## Baseline profile

Nsight Compute on `matmul_dense_gemv_f16` (glm-4-9b-int4, GPU6, `--set full --graph-profiling node`, 3 launches):

| metric | native fp16 GEMV |
|---|---:|
| lm_head kernel time | **421.536 us** |
| DRAM bytes/s | 2.951 TB/s |
| DRAM read pct of peak | 61.21% |
| L1/TEX throughput | 26.76% |
| SM throughput | 27.13% |
| active warps | 55.63% |
| registers/thread | 32 |
| long-scoreboard stall | 35.20 warps/issue |

Limiter: the custom one-thread-per-vocab-column kernel is DRAM/long-scoreboard bound. It streams the fp16 vocab weights efficiently (32 B/sector) but does scalar FMA per output column and takes ~421 us/token for glm's K=4096, N≈151k lm_head.

## Change

Added opt-in env gate `ONNX_GENAI_LMHEAD_CUBLASLT=1` in `crates/onnx-runtime-ep-cuda/src/kernels/matmul.rs`. When enabled, fp16 dense M=1 `MatMul` reuses the existing cached cuBLASLt `DenseGemmPlan` path (`CUBLAS_COMPUTE_32F`, persistent workspace, capture-safe after warmup) instead of the native fp16 GEMV. If cuBLASLt plan selection/launch is unavailable outside capture, it falls back to the native GEMV; during capture it still errors on a cold/invalid plan rather than silently changing graph semantics. Default remains the safe native path.

## cuBLASLt profile

Nsight Systems identified cuBLASLt kernel `nvjet_sm90_hsh_384x8_64x4_2x1_v_bz_NNT`. Nsight Compute (same GPU/session):

| metric | cuBLASLt fp16 M=1 |
|---|---:|
| lm_head kernel time | **290.880 us** |
| DRAM bytes/s | 4.415 TB/s |
| DRAM read pct of peak | 91.53% |
| L1/TEX throughput | 16.43% |
| SM throughput | 5.43% |
| active warps | 14.61% |
| registers/thread | 168 |
| long-scoreboard stall | 39.90 warps/issue |

This matches ORT's ~278 us lm_head class: cuBLASLt drives HBM near peak and cuts the final projection by **130.7 us/token (1.45x kernel speedup)**.

## End-to-end glm decode A/B

Command: `profile_native --model /home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda --ep cuda --backend native --steady --tokens 160 --decode-skip 40 --warmups 1 --runs 3` on GPU6.

| path | median tok/s | token status |
|---|---:|---|
| native lm_head (default, with Deckard-3 multicol GEMV commit on branch) | **207.78 tok/s** | baseline |
| `ONNX_GENAI_LMHEAD_CUBLASLT=1` + multicol GEMV | **213.44 tok/s** | **byte-identical** to default (160-token diff clean) |

qwen regression smoke (`/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx`, 64 tokens, decode-skip 16): final same-session default **70.16 tok/s**, cuBLASLt **86.77 tok/s** under a contended host, generated token stream byte-identical. Earlier quiet same-session run before the multicol commit measured default **152.40 tok/s**, cuBLASLt **154.67 tok/s**; no token regression in either run.

Fresh ORT comparison available from this harness on the ORT-fair artifact (`/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda-ortfair`, graph off due the known graph-on `ort_value` failure): **194.84 tok/s**, byte-identical to native for the measured prompt. The combined branch now measures **213.44 tok/s** native with multicol GEMV + opt-in cuBLASLt lm_head, but the standing certified CUDA-graph ORT comparator remains ~250 tok/s.

## Validation

- `cargo build --release -p onnx-genai-bench --features bench-native,bench-ort,cuda,cuda-ort --bin profile_native --quiet` — pass.
- `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test matmul_gpu --quiet` — 5/5 pass.
- `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test matmul_nbits_marlin_numerics --quiet` — 7/7 pass.
- `cargo fmt --all -- --check` — pass.
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings` — pass.
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda --all-targets -- -D warnings` — fails only in pre-existing unrelated test/lib-test lints outside this change (`matmul_nbits.rs` literal grouping, `gqa_shared_prefix_parity_gpu.rs` type complexity, `index_share_gpu.rs` needless lifetime, `standard_attention.rs` test/doc lints, `optimizer.rs` approx constant). I did not touch those unrelated files.
