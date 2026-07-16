# Native INT4 decode profile 2 — 2026-07-16

## Setup

- Model: `/home/justinchu/qwen2.5-0.5b-int4-onnx`
- Host: dual-socket Intel Xeon Platinum 8480C, 96 physical cores
- Build: `cargo build --release -p onnx-genai-bench --features bench-native`
- Runtime: `RAYON_NUM_THREADS=24`, CPU EP, 4 generated tokens
- Throughput sample: 3 warmups, 10 measured runs, nine interleaved
  baseline/change processes
- Op sample: `ONNX_GENAI_PROFILE_OPS=1`, 60 one-token executor invocations per
  version

Twenty-four Rayon workers remain the useful operating point for this small
model. Expanding each projection across both sockets costs more than it saves.
Greedy token IDs were `[11576, 42740, 11, 358]` in every run.

## Fresh baseline profile

| rank | op/category | median ms/step | node time |
|---:|---|---:|---:|
| 1 | MatMulNBits (121 calls) | 19.849 | 74.67% |
| 2 | Mul (48 calls) | 3.119 | 11.73% |
| 3 | Sigmoid (24 calls) | 1.354 | 5.09% |
| 4 | SkipSimplifiedLayerNormalization (48 calls) | 1.145 | 4.31% |
| 5 | GroupQueryAttention (24 calls) | 0.692 | 2.61% |
| 6 | Add (24 calls) | 0.442 | 1.66% |
| 7 | final SimplifiedLayerNormalization | 0.018 | 0.07% |
| 8 | Gather (2 calls) | 0.007 | 0.03% |

There is no standalone RoPE node in this graph, so it has no separate profiler
row; its attention work is represented by `GroupQueryAttention`. Remaining
shape/cast/reduction/constant/subtract work is below 0.1% in aggregate.

`MatMulNBits` is still the largest architectural lever. Its best next changes
(projection fusion, cross-node activation-quantization reuse, or a direct-int4
kernel) require graph/executor or substantial kernel work. The best clean,
self-contained win in this pass was instead the newly exposed 11.7% `Mul`
hotspot.

## Change: allocation-free contiguous f32 Mul

The generic binary elementwise path supports every dtype, striding, and
broadcast form by materializing both inputs, an accumulator, and the output.
Qwen decode's 48 `Mul` nodes are same-shape contiguous f32 tensors. They now
take a checked direct path that multiplies input slices into the output buffer,
without four temporary allocations/copies per node. Broadcast, strided,
non-f32, and aliased-buffer cases retain the generic path.

## Effect

Unprofiled throughput is the median of nine interleaved processes:

| version | tok/s | ms/step | change |
|---|---:|---:|---:|
| baseline | 40.50 | 24.692 | — |
| contiguous f32 Mul | **44.22** | **22.616** | **+9.2% tok/s** |

Three final rebuilt runs measured 45.29, 46.24, and 43.34 tok/s. The profiler
shows the intended local effect:

| op/category | before ms (%) | after ms (%) |
|---|---:|---:|
| MatMulNBits | 19.849 (74.67%) | 17.369 (81.31%) |
| Mul | 3.119 (11.73%) | **0.249 (1.17%)** |
| Sigmoid | 1.354 (5.09%) | 1.357 (6.35%) |
| all RMSNorm | 1.163 (4.38%) | 1.166 (5.46%) |
| GroupQueryAttention | 0.692 (2.61%) | 0.661 (3.09%) |
| Add | 0.442 (1.66%) | 0.461 (2.16%) |
| Gather | 0.007 (0.03%) | 0.006 (0.03%) |

Absolute timings vary with host load, so throughput uses interleaved unprofiled
medians. The robust result is Mul's 92% local reduction and the corresponding
9.2% end-to-end gain. MatMulNBits rises in share because the denominator fell.

## Ranked remaining levers

1. **MatMulNBits projection fusion / quantization reuse (81%)** — fuse QKV and
   gate/up projections that share an activation, quantize that activation once,
   and issue fewer, larger parallel kernels. This is the highest-ROI larger
   design, but needs graph-level grouping and fused-output handling.
2. **MatMulNBits kernel work** — evaluate direct-int4 dot products and
   cache/K-blocking against the current int8/VNNI path; retain accuracy level 4
   tolerances.
3. **Fuse Sigmoid + Mul into SiLU (about 7.5% combined)** — a graph rewrite or
   fused kernel would remove one materialization and one dispatch per MLP.
4. **RMSNorm (5.5%)** — vectorize/fuse residual normalization while preserving
   accumulation accuracy.
5. **GroupQueryAttention including RoPE work (3.1%)** — optimize only after the
   projection and SiLU opportunities.

## Follow-up: fuse the verified SiLU decomposition

Graph inspection found 24 `Sigmoid` nodes, one per transformer layer. Every
Sigmoid consumes a gate-projection output, has exactly one consumer, and feeds a
`Mul` whose other input is that same gate-projection value:

```text
gate ───────────────┐
  └─ Sigmoid(gate) ─┴─ Mul → SiLU(gate)
```

The executor now lowers only that exact single-consumer pattern to
`com.microsoft::Silu`. The CPU kernel writes the contiguous f32 result directly
to the executor-owned output buffer after equal-shape, contiguity, dtype, and
non-alias checks. Strided and other non-fast-path inputs retain the dense
fallback. This removes the intermediate Sigmoid tensor and one dispatch per
layer; unrelated Sigmoid and Mul nodes are unchanged.

Fresh five-process samples used the same `RAYON_NUM_THREADS=24`, four-token,
three-warmup, ten-run harness:

| version | median tok/s | median ms/step | change |
|---|---:|---:|---:|
| contiguous f32 Mul baseline | 44.53 | 22.455 | — |
| fused SiLU | **45.69** | **21.887** | **+2.6% tok/s** |

Host load was visibly noisy (individual fused runs ranged from 43.29 to
49.35 tok/s), so the per-op profile is the stronger local signal. Medians from
60 one-token invocations were:

| op/category | before median ms (%) | after median ms (%) |
|---|---:|---:|
| Sigmoid | 1.486 (6.55%) | **removed (0%)** |
| fused Silu | — | **0.658 (3.40%)** |
| Mul | 0.225 (0.99%, 48 calls) | 0.127 (0.65%, 24 calls) |

The fused kernel replaces the 24 Sigmoid calls and the 24 self-multiply calls;
the remaining 24 Mul calls are the SwiGLU product with `up_proj`. Compared with
the former Sigmoid plus half of Mul time, the fused local path is about 59%
faster. Greedy output remained `[11576, 42740, 11, 358]` in every throughput
run.
