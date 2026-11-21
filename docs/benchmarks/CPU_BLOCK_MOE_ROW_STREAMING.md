# CPU block-MoE row streaming

The complete f32 projection set of one selected expert must fit
`ONNX_GENAI_CPU_BLOCK_QUANT_CACHE_BYTES` to use dense execution. Otherwise the
CPU EP decodes one row per worker shard and reuses it for all grouped tokens.
The default dense/cache path remains available for fitting experts. The change
also removes duplicate CPU weight-contract exports; EP API owns those contracts.

## Measurement scope

Measured on Apple M1 Max, 10 cores, 32 GiB RAM, four Rayon threads, default Rust
CPU kernels without MLAS. This synthetic MXFP4 operator benchmark is not LLM
tokens/second, GPU execution, or an all-format performance claim. Correctness
tests separately cover all 15 block formats and mixed-format MoE projections.

Baseline revision: `267408c9df76c6859147fc9f38b07bc31c17bd44`. The new benchmark
source was copied unchanged into a detached baseline worktree; both executables
were built in release mode. [Raw records](CPU_BLOCK_MOE_ROW_STREAMING.results.json)
contain all 45 process samples, baseline revision and measured source hashes.
Those hashes identify the measured snapshot before subsequent documentation
and test-fixture cleanup; the measured runtime behavior is unchanged.

Geometry: hidden 1024, intermediate 4096, two experts, top-k one, SiLU, no third
projection. One expert is selected per row. Each fresh process warms once, then
measures four calls. Three repetitions interleave five arms, rotating order:
baseline cold, new streamed, baseline warm, new warm, new cold. `cold` marks
weights mutable and expands both projections on every call; `warm` seals them
and fills the cache before timing. Streamed execution has zero cache budget.
The real expansion counter requires eight timed expansions for cold arms and
zero for warm/streamed arms.

Every arm sets `RAYON_NUM_THREADS=4` and
`ONNX_GENAI_HOST_ALLOC_CACHE_FLOOR_BYTES=1048576`. The latter bypasses the existing
startup floor calibration, which otherwise touches roughly 512 MiB and obscures
the weight-memory comparison in process peak RSS. It does not represent a
startup-memory fix. The dense-cache budget is zero for streamed and 268435456
bytes for cold/warm. No GPU or MLAS backend was enabled.

## Results

CPU time is process user+system time per call, including worker threads; it is
not wall time. RSS is process peak, including input tensors and initialization.
Medians of three fresh processes:

| Rows | Baseline cold CPU ms | Streamed CPU ms | Baseline warm CPU ms | New warm CPU ms | New cold CPU ms | Baseline cold RSS MiB | Streamed RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75.924 | 14.474 | 7.819 | 7.815 | 75.904 | 68.125 | 20.156 |
| 16 | 192.549 | 129.190 | 122.420 | 122.679 | 190.830 | 68.609 | 20.906 |
| 48 | 431.603 | 373.179 | 363.217 | 364.361 | 431.802 | 69.891 | 22.547 |

The single-row cold case reduces measured CPU time by about 5.25x and peak RSS
by about 70%. Warm cached execution stays within the measurement spread. Larger
groups amortize dense expansion and show smaller differences. The unchanged
Relu control has arm medians approximately 0.86–0.99 ms, so the modest 48-row CPU
difference is not evidence for a broad throughput claim. All full output arrays
from all 45 samples were compared at `abs(a-b) <= 1e-5 + 1e-4*abs(b)`, with the
first baseline cold output as reference for each row count. Checksums alone were
not used for correctness.

## Reproduction

Build each source tree with the same benchmark source and Cargo bench entry:

```sh
cargo build --release -p onnx-runtime-ep-cpu --bench bounded_expert_bench
```

Select the resulting executable in `target/release/deps/` (not its `.d` file):

```sh
RAYON_NUM_THREADS=4 \
ONNX_GENAI_HOST_ALLOC_CACHE_FLOOR_BYTES=1048576 \
ONNX_GENAI_CPU_BLOCK_QUANT_CACHE_BYTES=0 \
/path/to/bounded_expert_bench streamed 1 1024 4096 4 /tmp/streamed.f32
```

For cold/warm use their mode name and cache budget 268435456; run them on both
revisions. Repeat with rows 16 and 48. Interleave three repetitions and compare
the complete little-endian f32 files with the tolerance above. Keep host settings
and the floor override identical. The benchmark reports phase peaks, expansion
counts, CPU time, wall time and an unchanged 1000-call Relu control. Measurements
on other CPUs, MLAS builds, formats or geometries require their own evidence.

## Memory and execution limits

Decoded weight scratch for one projection is bounded by
`min(Rayon workers, output features) * (input features + 256) * sizeof(f32)`.
There is also an output-sized column-major buffer and the final projection
output, grouped inputs, and activation buffers. Contiguous compressed weights
are borrowed; strided views copy the selected expert's compressed bytes. The
existing cache budget controls dense eligibility, not global process memory:
concurrent callers may still hold evicted dense Arcs.

This uses canonical decoders followed by f32 GEMM, with runtime ISA dispatch and
a portable fallback. It is not an integer-domain dot product. The serial inner
GEMM avoids nested Rayon scheduling; the optional MLAS build uses its existing
SGEMM entry point with its explicit inner-threadpool flag disabled. A separate
MLAS integration test checks that large serial calls do not enter the pool while
the unchanged parallel control does. Shared routing, activation, gating, output dtype and stride
validation remain in force. No CUDA performance or overlap claim is made.


## Validation

Default CPU unit suite: 1,854 passed, 16 existing ignored tests. Public API
doctests pass, including both removed import surfaces failing to compile.
The MLAS serial/parallel integration test passes with the pool counters proving
that serial calls stay out of the backend.

A complete macOS MLAS-feature run also exposed 11 failures outside this change:
backend-selection assertions, vDSP/BNNS counters, KAI dispatch, activation ULP
parity and NCHWc layout tests. The identical 11 test names fail on the unchanged
`267408c9d` baseline (1,855 passed, 11 failed, 26 ignored), so this optional
configuration is not reported as a green full suite. Default CPU validation
and the new streamed-format/gating cases are separate evidence.

Self-review caught MLAS's default SGEMM pool dispatch in an outer-sharded call.
The SGEMM contract now makes that choice explicit. It also caught a stale native
archive when sharing Cargo targets between worktrees: the MLAS build script
captured its source root at compile time. It now resolves `CARGO_MANIFEST_DIR`
when Cargo runs the script; after cleaning only MLAS artifacts, the real native
serial/parallel counter test passes. No test assertion was weakened.


## Shared projection use

The bounded row helper also serves standalone `BlockQuantizedMatMul` when its
complete dense matrix exceeds the same cache budget, including budget zero.
Fitting matrices keep their dense GEMM/cache path. This reuses the decoder and
serial inner GEMM; it adds no independent batching or memory authority. The
measurements above remain MoE measurements and do not establish standalone
MatMul performance. Separate operator tests compare all 15 formats and partial
blocks to dense and f64 math, and cover f32/f16/bf16 storage, fresh/strided packed
inputs, bias and empty leading dimensions. Neither operator's local cache limit
is a global working-memory grant.
