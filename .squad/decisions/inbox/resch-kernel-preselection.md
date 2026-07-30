# Resch — MatMulNBits MLAS QNBit kernel preselection audit

Timestamp: 2026-07-29T18:10:00-07:00
Branch: qwen3-perf-followups / PR #398

## Audit

The aarch64 MLAS QNBit decode path was already caching the expensive packed-B representation in `mlas_shards`/`mlas_packed`, but the hot `execute` path still did route checks before reaching the cached shard and called the wrapper that created a fresh `SQNBitGemmWorkspace` per invocation. The old code also selected the native `DotKernel` before trying MLAS, so MLAS-handled calls still paid a runtime feature-selection check they did not use.

## Change

`MatMulNBits` now stores MLAS packed weights in `MlasPreparedPacked`, which owns the `SQNBitPackedB` plus a reusable `SQNBitGemmWorkspace` pre-reserved for M=1 decode. Once a constant MLAS shard/full-width pack is initialized, later calls fast-path directly to the cached object and skip the MLAS-route selection gates. `selected_dot_kernel()` is moved after the MLAS attempt, so MLAS decode calls do not select the native fallback kernel.

## Remaining mlas-sys boundary

This removes per-call workspace allocation/repacking in `matmul_nbits.rs`, but `mlas_sys::sqnbit_gemm_with_workspace` still calls `SQNBitGemmWorkspace::reserve_for`, which re-queries MLAS workspace size every call. Eliminating that last query requires an mlas-sys API that accepts a precomputed workspace size or exposes a fixed prepared workspace runner. Also, MLAS/KleidiAI still chooses the ukernel inside `MlasQNBitGemmBatch`; the vendored registry confirms M=1 on an I8MM-capable Oryon selects the dotprod GEMV (`qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod`), while M>1 uses the i8mm GEMM.

## Measurement

Host was heavily contended. Before baseline from request: MatMulNBits ~8.6 ms/token bucket (197 calls). After (`ONNX_GENAI_PROFILE_OPS=1`, native, qwen3-0.6b, runs=3/tokens=96): decode MatMulNBits best 7.300 ms, median 9.766 ms across 375 decode forwards. No clean median win; best case shows allocation/selection shaving is below host noise.

Best-case 12-run steady decode (max throughput, tokens=96, decode_skip=8): native best 77.84 tok/s, median 52.12 tok/s; ORT best 97.99 tok/s, median 55.44 tok/s. These are load-corrupted absolute numbers; ORT had a clean early window while native did not.

## Correctness

`cargo check -p onnx-runtime-ep-cpu --features mlas` passed. `cargo test -p onnx-runtime-ep-cpu --features mlas matmulnbits -- --nocapture --test-threads=1` passed (43 passed, 2 ignored). Parallel test mode showed an order/env-sensitive pre-existing KAI-vs-MLAS test failure; the same test passed alone and in serialized mode.
