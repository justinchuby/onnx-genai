# Resch: decode residual fusion follow-up

## Context
After the ARM64 MLAS QNBit SPMD-shard fix, MatMulNBits is at parity with ORT. The remaining qwen3-0.6b decode gap is non-MatMulNBits work.

## Profile
Native MLAS decode (`ONNX_GENAI_PROFILE_OPS=1`) now shows steady-token non-MatMulNBits dominated by:
- `GroupQueryAttention`: ~5-7% of node time after seq=1 layout/output fast paths.
- `SimplifiedLayerNormalization` + `SkipSimplifiedLayerNormalization`: ~4.5-5% after output-only contiguous fast paths.
- `Reshape`: ~2% (~56 calls/token), likely remaining executor/view/copy/barrier overhead.
- `FusedSiluMul`: ~1.5-1.7%, replacing separate `Swish` + `Mul` (~2.1-2.4% before).

## Changes
- `GroupQueryAttention`: skip seq=1 BSH→BHSD transpose, split packed seq=1 Q/K/V as contiguous head blocks, and write decode output directly.
- `SimplifiedLayerNormalization`: output-only contiguous f32 fast path writes directly to the output buffer.
- `SkipSimplifiedLayerNormalization`: output-only contiguous f32 fast path uses one row scratch instead of materializing full sum/output vectors.
- Added `com.microsoft::FusedSiluMul` plus `CpuSiluMulFusion` to fold `Swish/Silu(x) * rhs` into one kernel.

## Benchmark (same-window, qwen3-0.6b, 128 tokens, runs=5)
- native MLAS: median 9.581 ms/token = 104.37 tok/s
- KAI fallback: median 9.948 ms/token = 100.53 tok/s
- ORT: median 9.413 ms/token = 106.24 tok/s

The fusion work improved residuals but did not honestly pass ORT in this noisy window; native remains ~1.8% behind ORT median. Single native runs reached 105.7 tok/s vs ORT median 106.2, but the honest median is still below ORT.

## Residual / next lever
GQA phase profiling shows the attention math dominates GQA once copy/widen/output overheads are reduced; remaining local-kernel wins are small. The next likely lever is executor-owned: eliminate the remaining ~56 `Reshape` dispatches/views/copies and/or batch/fuse per-layer decode barriers. If staying kernel-local, the next target is a deeper GQA attention-core optimization rather than more layout copying.
