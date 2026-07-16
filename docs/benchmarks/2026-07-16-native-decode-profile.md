# Native CPU int4 decode profile — 2026-07-16

## Setup

- Model: `/home/justinchu/qwen2.5-0.5b-int4-onnx`
- CPU path: `MatMulNBits` int8/VNNI (`accuracy_level=4`)
- Prompt: `The capital of France is`
- Benchmark: release build, 3 generated tokens, 1 warmup
- Profiling: `ONNX_GENAI_PROFILE_OPS=1`

The table is a representative steady-state decode step before the optimization.
Times include executor setup and kernel execution for each node.

| op_type | ms/step | step % | calls |
|---|---:|---:|---:|
| Gather | 688.004 | 88.37% | 2 |
| MatMulNBits | 78.283 | 10.06% | 121 |
| Mul | 4.592 | 0.59% | 48 |
| GroupQueryAttention | 2.329 | 0.30% | 24 |
| SkipSimplifiedLayerNormalization | 2.182 | 0.28% | 48 |

## Bottleneck and change

`Gather` materialized its entire input with `to_dense_bytes` before selecting
rows. For token embedding lookup this copied the full vocabulary embedding
table twice per decode step, although only one row was needed.

The CPU `Gather` kernel now detects the common contiguous `axis=0` case and
copies only selected rows directly from the input view into the output view.
The existing generic strided path remains unchanged. This is dtype-agnostic,
and it does not affect `MatMulNBits` numerics or the default
`accuracy_level=0` path.

## Before/after

Unprofiled release benchmark, 3 tokens × 3 runs:

| version | tok/s | ms/step |
|---|---:|---:|
| Before | 1.07 | 934.809 |
| After | 4.71 | 212.295 |

Generated token IDs remained `[12095, 13, 1084]` (coherent continuation:
`Paris`).

After the change, a representative steady-state step attributed 87.36% to
`MatMulNBits` (87.223 ms, 121 calls), 4.52% to `Mul` (4.515 ms, 48 calls),
2.46% to `SkipSimplifiedLayerNormalization` (2.456 ms, 48 calls), and 0.01%
to `Gather` (0.008 ms, 2 calls).

## Next bottleneck

`MatMulNBits` is now dominant. A future pass should reduce its 121 per-step
dispatches and GEMV overhead, starting with activation-quantization reuse
across projections that consume the same layer input and measuring Rayon task
overhead for small output dimensions.
