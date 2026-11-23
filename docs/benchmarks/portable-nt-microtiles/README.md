# Portable native NT micro-tiles

Measured on Apple M1 Max / ARM64 / macOS 27, Rust 1.98.0 (LLVM 22.1.8).
This is a synthetic CPU-EP operator experiment, not model token throughput.
The default binaries contain no MLAS symbols; the research positive control
contains 37. The default normal dependency tree contains no `mlas-sys`.

## Mechanism and numerical contract

The portable transposed-B kernel computes up to 4x4 independent outputs while
retaining each output's original left-to-right f32 K chain. It does not use
fast-math, K reassociation, a new scheduler or heap packing in the micro-kernel.
Compressed BlockQuantizedMatMul/MoE decode up to four rows per worker, reuse the
tile across routed tokens and write the existing column-major buffer directly.
The AVX2 and MLAS routes retain their original one-row operand geometry;
NBits/QMoE retain one decoded row and their existing arithmetic contract.

A sixteen-accumulator K-reassociated candidate was rejected: the public
MatMul case M=16/K=1024/N=2048 changed one argmax. Two logits originally at
`0x3fd96455` and `0x3fd96457` both became `0x3fd9645c`. The minimized regression
pins the two original results and the winning logit. Independent scalar-chain
bit comparisons cover full/edge tiles, empty/zero K, tails and K=4097.
The rejected version and all its measurements are retained locally and marked
rejected; its faster microbenchmarks are not evidence for this implementation.

## Measurement and controls

Both source arms were rebuilt in the same private worktree/manifest path,
cleaning governor/API/CPU release artifacts on each switch. The candidate is
restored from a persistent hash-checked backup on every exit path. All heavy
work holds `/tmp/onnx-genai-hostlock-darwin.flock`, excluding the llama.cpp chat.

The [exact measured operator source](measured-operators.rs) constructs real
public CPU-EP kernels with a 64 MiB governed Host derived-weight limit.
Shapes are K=1024 and N=2048, contiguous f32 activations and seeded MXFP4
weights. MoE has two experts, top-1 normalized routing, FC1/FC2 and SiLU.
Zero retention proves zero dense expansions/retained decoded bytes; warm controls
prove cache hits, exact 8/16 MiB retained payloads and zero usage after teardown.
This limit is not a complete process-memory cap.

Each operator warms for 50 ms process CPU time. Common repetitions per shape
are calibrated to target at least 20 ms CPU per batch (15 ms floor gate).
Report the median of nine batches per process, then three isolated process
medians/arm; order is before/after, after/before, before/after. Relu has a
1,000-call warmup and nine 30,000-call batches. Thread width is fixed identically
per pair; CPU time includes all threads. RSS is the absolute process high-water
mark at the end of the operator phase, includes setup, and is never differenced.
All 96 operator runs have deterministic, byte-identical before/after outputs
and identical row argmaxes. [Full ranges, hashes and controls](summary.json)
are part of the report.

Earlier short/cross-worktree-path sweeps failed controls and are not used for
claims. In this final sweep, the M=16 streamed MoE Relu ratio was 0.875 and its
timing is excluded. The single-row warm dense MatMul estimate was unstable;
a focused five-process-per-label baseline/null/candidate run uses the exact
same binary for baseline and null. Their median difference is about 1%; the
candidate differs by 2.1% (1.6 us), below the 30.5 us within-arm spread. There
is no measurable dense-path change from that control, not a speedup claim.

## Default native observations

Process CPU milliseconds per call, medians as defined above. These are changed
operator paths whose Relu controls passed; the excluded MoE row is not shown.
Unchanged warm dense MatMul measurements are controls, not claimed benefits.

| Operator | Mode | Rows | Threads | Before ms | After ms | Ratio |
|---|---|---:|---:|---:|---:|---:|
| matmul | streamed | 1 | 1 | 5.1036 | 3.7405 | 1.36x |
| matmul | streamed | 4 | 1 | 10.2735 | 3.7993 | 2.70x |
| matmul | streamed | 16 | 1 | 31.3260 | 5.5178 | 5.68x |
| moe | streamed | 1 | 1 | 10.3300 | 7.4873 | 1.38x |
| moe | streamed | 4 | 1 | 20.9785 | 7.6380 | 2.75x |
| moe | warm | 1 | 1 | 3.5885 | 1.1114 | 3.23x |
| moe | warm | 4 | 1 | 14.3549 | 1.3805 | 10.40x |
| moe | warm | 16 | 1 | 57.4165 | 5.4128 | 10.61x |
| matmul | streamed | 4 | 4 | 10.8133 | 4.1065 | 2.63x |
| moe | streamed | 4 | 4 | 22.1407 | 8.4887 | 2.61x |
| moe | warm | 4 | 4 | 15.1785 | 4.8382 | 3.14x |

Absolute operator-phase peak RSS ranges remain broadly overlapping; this is
not a demonstrated RSS reduction. Decoded storage can grow from one to four
rows per worker: at most `min(workers,N) * (FEATURE_TILE*K + 256)` f32 values,
shrunk to each shard's actual feature count. Output/grouped activation buffers
are separate. This remains call-owned scratch, not public funded workspace;
no new cache, issuer callback or per-column allocation protocol was added.

## Reference gap and validation

A separate MLAS-enabled binary compares the actual serial entry with caller-
owned reusable output and a route witness. The diagnostic Generic tag means
the built-in NT dispatcher, including runtime ISA detection; it is not a
scalar-only ISA override. Native/reference CPU microseconds are respectively
3.850/9.026 for 1x4096x1, 64.02/50.53 for 1x4096x64, 19.49/11.57 for
4x1024x64 and 39.90/13.92 for 16x512x64. Larger-row native work still trails
MLAS. No whole-family graduation or reference-removal claim follows.

Native CPU: 2,001 unit + 65 integration tests, session 286. Optional MLAS:
2,043 unit + 66 integration, session 288. The integration totals include two
new layout tests for partial tiles, per-shard decoder reuse, bias and rejected
incomplete shards, run in both configurations. Strict all-target Clippy passes for
CPU/session/engine in both configurations. Release scalar-chain/argmax tests
and explicit native/reference route tests each select and pass two tests.
An extra strict public-rustdoc probe has the same 63 baseline diagnostics;
it is not a passing gate. No CUDA device execution or x86 execution was done
on this ARM host; existing runtime ISA dispatch remains intact.

For reproduction, copy `measured-operators.rs` to the CPU crate's `examples/`
as `nxrt_serial_operators.rs` (its common-module path is relative to that
location), build with `cargo build --release -p onnx-runtime-ep-cpu --example
nxrt_serial_operators`, without the MLAS feature, then invoke:

```sh
nxrt_serial_operators <matmul|moe> <streamed|warm> <rows> 1024 2048 <repetitions> <output.f32>
```

Set `RAYON_NUM_THREADS` and `ONNX_GENAI_CPU_DECODE_THREADS` to the row's width,
`MLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`,
`ONNX_GENAI_HOST_ALLOC_CACHE_FLOOR_BYTES=1048576`, and clear the backend override.
Set `ONNX_GENAI_CPU_DENSE_WEIGHT_CACHE_BYTES=0` for streamed runs; for warm
runs use 8388608 bytes for MatMul or 16777216 for MoE. The source also declares
these retention ceilings explicitly. Exact per-shape counts, binary/source
hashes, raw `.f32` outputs and host-specific drivers remain under `/tmp/nxrt-serial-*`. Use the same source
path for both arms and the shared stable host lock for the complete sweep.
