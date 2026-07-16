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

## Follow-up: MLAS-style direct-int4 VNNI GEMV

### MatMulNBits phase breakdown

Fine-grained, env-gated timers were added temporarily and removed after
measurement. Across steady-state decode steps, the 121 MatMulNBits calls spent:

| phase | median ms/step | MatMulNBits time |
|---|---:|---:|
| input densification + output allocation | 0.648 | 3.1% |
| activation quantization | 0.658 | 3.2% |
| threaded VNNI GEMV | 19.520 | 93.7% |
| int8 weight prepack | 0.000 steady-state | 0% |

The first process invocation spent about 2.04 seconds unpacking all constant
weights to int8. Every later invocation hit the `OnceLock`; prepack is
session-start amortized, not a decode-step cost.

At one worker the GEMV phase took 73.076 ms versus 19.520 ms at 24 workers,
only 3.74x scaling. The old steady-state stream was at least 493.96 MB of
expanded int8 weights, 61.75 MB of scales, and 61.75 MB of block sums per token.
That is about 8.45 GB/s at one worker and 31.6 GB/s at 24 workers, far below
host DRAM capability. Its arithmetic intensity was about 1.6 integer
multiply-add operations per byte. The evidence points to the expanded stream,
121 repeated Rayon barriers/shape-sized tasks, and inefficient block-level SIMD
reduction rather than DRAM saturation or activation quantization.

ORT MLAS's block-32 M=1 kernel keeps B packed. It SIMD-unpacks nibbles, applies
the symmetric zero point, feeds signed values to VNNI, accumulates scaled float
lanes across K blocks, and folds the lanes only at the end. It also tiles
multiple K blocks and output columns. The previous Rust path instead expanded
the whole model to int8, streamed a separate block-sum array, and horizontally
reduced a VNNI accumulator after every 32-element block.

### Change

The symmetric, no-`g_idx`, block-32, M=1 `accuracy_level=4` path now caches the
original packed bytes and scales. Runtime AVX2 + AVX-VNNI or
AVX512-VNNI/AVX512VL detection selects a fused kernel that:

1. loads 16 packed bytes for each 32-weight block;
2. interleaves low/high nibbles and subtracts zero point 8 in SIMD;
3. forms a signed int8 dot product with VNNI;
4. multiplies lane sums by the block scale and accumulates them as f32; and
5. folds once per output instead of once per block.

Unsupported CPUs and other block/zero-point/group-index shapes retain the
existing int8 or fp32 paths. The steady-state minimum weight stream falls from
617.45 MB to 308.73 MB per token, doubling nominal arithmetic intensity to
about 3.2 integer operations/byte. One-time first-invocation MatMulNBits time
also fell from a paired median 1.941 seconds to 1.234 seconds because prepack no
longer expands B or constructs block sums.

### Results

The main comparison alternated seven baseline/change processes at 24 workers.
Each process generated four tokens, used three warmups and ten measured runs,
and reproduced `[11576, 42740, 11, 358]`.

| version | median tok/s | median ms/step | change |
|---|---:|---:|---:|
| int8-prepacked VNNI | 46.72 | 21.404 | — |
| direct-int4 VNNI | **49.15** | **20.346** | **+5.2% tok/s** |

The median of the seven within-pair speedups was +6.5%; host load made
individual pairs range from -3.3% to +25.3%. The paired op profiler gives the
cleaner local result:

| version | MatMulNBits ms/step | node-time share |
|---|---:|---:|
| int8-prepacked VNNI | 16.454 | 84.58% |
| direct-int4 VNNI | **14.154** | **82.53%** |

That is a 14.0% local MatMulNBits reduction and a 2.05-point share reduction.
The earlier independent 60-step profiles measured 19.339 to 15.840 ms
(-18.1%), consistent in direction despite machine noise.

The same paired binaries were also sampled at larger pools:

| Rayon workers | baseline tok/s | direct-int4 tok/s | change |
|---:|---:|---:|---:|
| 48 | 35.64 | 38.34 | +7.6% |
| 96 | 15.31 | 29.53 | +92.9% |

Twenty-four workers remain best in absolute throughput. The unusually large
96-worker relative gain comes from the existing policy leaving medium
projections serial on the two-socket pool: cutting their resident weight stream
in half matters much more there, but does not make 96 workers the preferred
configuration.

### Remaining MatMulNBits levers

1. Tile four output columns and two/four K blocks as MLAS does, reusing
   activation loads and scale broadcasts while increasing instruction-level
   parallelism.
2. Quantize activations per block and share the quantized activation across
   projections with the same input; this requires a numerics review and likely
   executor/graph support.
3. Replace the fixed 48-worker topology cliff with NUMA-aware scheduling and
   weight placement.
4. Fuse QKV and gate/up projection dispatch at graph level after the standalone
   kernel is exhausted.

## Follow-up: multi-column direct-int4 tiling did not improve decode

The direct-int4 M=1 kernel was tested with four and eight output columns per
inner K-block pass. Each candidate kept independent SIMD accumulators, loaded
the quantized activation block once per column tile, preserved the existing
contiguous N partition, and used the existing single-column path for N tails.
Both widths covered a partial K block and an N tail in experimental tests.

The primary 24-worker comparison interleaved seven processes per version. Each
process generated four tokens with three warmups and ten measured runs:

| column tile | median tok/s | median ms/step | vs. baseline |
|---:|---:|---:|---:|
| 1 (current kernel) | **54.91** | **18.211** | — |
| 4 | 52.67 | 18.987 | -4.1% |
| 8 | 52.65 | 18.993 | -4.1% |

Five additional interleaved profiler processes per version produced 25
steady-state one-token samples after excluding each process's prepack warmup:

| column tile | MatMulNBits median ms | mean ms | median node-time share |
|---:|---:|---:|---:|
| 1 | 15.222 | **15.281** | 83.39% |
| 4 | 15.087 | 16.637 | 83.34% |
| 8 | **13.626** | 17.258 | **82.60%** |

Eight columns reduced the median inner-kernel time, but introduced enough slow
steps to make mean MatMulNBits latency 12.9% worse and end-to-end throughput
4.1% worse. Four columns was effectively flat at the median and 8.9% worse by
the mean. The largest sampled MatMulNBits step was 17.368 ms for the current
kernel, 46.922 ms for four columns, and 44.249 ms for eight columns.

Exploratory two-process samples at larger pools were not interleaved and were
more host-load-sensitive, but the per-op direction consistently disfavored
tiling:

| workers | tile 1 tok/s, MatMulNBits ms (%) | tile 4 tok/s, MatMulNBits ms (%) | tile 8 tok/s, MatMulNBits ms (%) |
|---:|---:|---:|---:|
| 24 | 54.91, 15.222 (83.39%) | 52.67, 15.087 (83.34%) | 52.65, 13.626 (82.60%) |
| 48 | 34.94, 19.900 (87.46%) | 40.75, 25.661 (88.07%) | 35.06, 34.088 (90.09%) |
| 96 | 30.72, 28.841 (90.63%) | 29.33, 30.320 (91.30%) | 28.04, 32.103 (91.65%) |

The 48-worker throughput anomaly conflicts with its much slower local profile
and is attributed to the non-interleaved host noise; 24-worker interleaving is
the acceptance result. Greedy output remained `[11576, 42740, 11, 358]` in
every four-token run.

The likely limitation is the row-major packed-weight layout: adjacent output
columns are separated by a full packed K row, so a wider tile creates more
independent weight streams and register pressure while saving loads from an
activation vector that already fits in cache. The lower median for width eight
shows that activation reuse can accelerate an undisturbed inner loop, but the
long-tail and larger-pool regressions make it unsuitable as the production
kernel. The tiling code was therefore reverted; the simpler one-column direct
kernel remains.

The three targeted direct-int4 tests passed for both candidates. After the
revert, `cargo test -p onnx-runtime-ep-cpu` passed all 411 unit tests (plus one
ignored doctest), and the release native benchmark build passed.

### Ranked remaining levers

1. **Cross-projection activation reuse** — quantize an input once for Q/K/V or
   gate/up projections and share it across calls; this removes repeated work
   without multiplying non-contiguous weight streams inside one kernel.
2. **NUMA-aware scheduling and weight placement** — keep projection tasks and
   their packed weights on one socket, replacing the sharp degradation above
   48 workers.
3. **Projection fusion** — group QKV and gate/up dispatch so shared activation
   work, Rayon barriers, and executor overhead are amortized at graph level.
