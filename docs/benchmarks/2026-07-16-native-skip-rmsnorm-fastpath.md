# Native decode: allocation-free fused RMSNorm

## Setup

- Model: `/home/justinchu/qwen2.5-0.5b-int4-onnx`
- Host: dual-socket 2×48-core Xeon
- `RAYON_NUM_THREADS=24`
- Release build: `onnx-genai-bench --features bench-native`
- Profiler: `profile_native --tokens 8 --warmups 1 --runs 3`
- The one-time first-step weight-prepack sample was excluded from op medians.

## Baseline profile

| op | median ms/step | median share |
|---|---:|---:|
| MatMulNBits | 14.883 | 81.93% |
| SkipSimplifiedLayerNormalization | 1.113 | 6.13% |
| GroupQueryAttention | 0.912 | 5.01% |
| Silu | 0.633 | 3.47% |
| Add | 0.411 | 2.25% |
| Mul | 0.119 | 0.65% |

MatMulNBits remains dominant. The fused residual RMSNorm was the largest
low-risk non-matmul target: every decode step runs 48 instances over 896
elements, with contiguous same-shaped input/skip tensors and a live residual-sum
output.

## Change

`SkipSimplifiedLayerNormalization` now borrows contiguous f32 inputs and writes
the residual sum and normalized result directly into their output buffers. This
removes input, skip, gamma, sum, and normalized-output temporary allocations
from the model's common path. Broadcasted, strided, statistics-output, and
output-only cases retain the existing scalar fallback.

The fused op median fell from **1.113 ms to 0.742 ms (-33.3%)**, and its median
profile share fell from 6.13% to 4.06%.

## End-to-end result

Five alternating before/after process pairs used 24-token requests, two measured
runs per process, at 24 Rayon workers. Absolute process medians were
**44.20 → 46.45 tok/s (+5.1%)**; the median of paired speedups was **+9.1%**
(5/5 pairs won). Short native decode runs remain noisy on this shared
dual-socket host, so the op-local reduction is the more stable measurement.

Greedy output stayed identical. The first four decode tokens remained
`[11576, 42740, 11, 358]`; all 24 checked tokens matched before and after.
`cargo test -p onnx-runtime-ep-cpu` passed 413 tests.
