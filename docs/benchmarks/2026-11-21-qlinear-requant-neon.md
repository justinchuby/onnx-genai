# Native ARM64 QLinearMatMul requantization

This change absorbs the vector rounding/clipping mechanism into native Rust
NEON code. It does not call MLAS and adds no retained weight or scratch buffer.
It is one part of QLinearMatMul absorption; it does not graduate the packed,
requantized MLAS GEMM or claim an end-to-end inference improvement.

## Contract

For shared per-tensor A scales, the existing per-column combined scales retain
the association `a_scale * b_scale / y_scale`. Eight output lanes use i32-to-f32
conversion, multiplication, explicit NaN-to-zero selection, clipping at integral
bounds relative to the output zero point, ties-to-even conversion and narrowing.
Integral bounds make clipping before rounding equivalent to the previous
round-then-clip result. A valid i8/u8 zero point makes conversion and its addition
safe in i32. Scalar tails, per-row A scales, other CPU architectures and the
out-of-dtype internal zero-point fallback preserve the existing scalar contract,
including saturating i64 addition. No fast-math or fused multiply-add is added.

Tests compare exact bytes against the previous arithmetic for widths 0..33,
65 and 257, offset slices, signed/unsigned range endpoints, invalid internal zero
points, i32 extremes, signed zeros, denormals, huge scales, infinity and NaN.
Independent hand-written tie fixtures guard even rounding. A dispatch counter
and manifest claim prove the NEON route runs. The row entry point also checks
batch-specific/per-row scales and a three-thread case above the existing
parallel threshold. Existing full-kernel integer-oracle fixtures remain intact.

## Measurement method and scope

Apple M1 Max (10 logical CPUs, 32 GiB), macOS 27.0.1 arm64, Rust 1.98.0. The release-only ignored diagnostic
`kernels::qlinear_matmul::tests::bench_requantization_neon_ab` runs the portable
previous-expression loop, selected native row kernel and repeated-native A/A
control in rotating order. Production selects portable versus vector row traversal
once per batch, before its serial/parallel row loop. The diagnostic similarly
selects the row function before each timed sample; widths below eight select the
identical portable function in all arms. The compiler may vectorize the portable loop.
All arms use the same prepared accumulator/scale slices and caller-owned output,
check exact output before and after each sample, and black-box the inputs and
output. Each of 21 samples processes about one million elements. CPU time uses
`getrusage(RUSAGE_SELF)`; elapsed time is reported separately. Native test builds
include one route-counter atomic increment per row, which production omits.

Widths cover the scalar-only case (7), minimum vector block (8), tails (17, 257)
and wider rows (128, 2048, 5120), for both output dtypes. This is a hot-cache row
epilogue diagnostic: GEMM, zero-point correction, combined-scale preparation,
allocation, batch-level dispatch selection, parallel scheduling and session
overhead are outside its interval.
It is not a streaming-memory, whole-QLinearMatMul or MLAS A/B benchmark.

The repository hostlock script does not support Darwin. The local outer harness
uses the stable `/tmp/onnx-genai-hostlock-darwin.flock` inode with a nonblocking
kernel `flock` for each complete sweep, and records CPU/elapsed time, process
maximum RSS and before/after load. It never reaps or unlinks the lock. This
coordinates participating local runs; it does not establish CPU affinity or
exclude foreign work outside the protocol. Whole-command maximum RSS includes
build/harness activity and is not an arm-level memory measurement.

Run after enumerating the exact test and confirming one match:

```sh
cargo test --offline --locked -p onnx-runtime-ep-cpu --lib --release \
  kernels::qlinear_matmul::tests::bench_requantization_neon_ab \
  -- --exact --ignored --nocapture
```

## Local diagnostic results

Three complete, separately locked sweeps; each cell below uses the median of
the three per-sweep CPU medians. The ratio range includes both dtypes in all
three sweeps. Raw CPU/elapsed medians, within-sweep sample ranges, provenance
and the exact Rust source hash are in the
[machine-readable results](2026-11-21-qlinear-requant-neon.results.json).

| Columns | Portable CPU ns/element (i8 / u8) | Native CPU ns/element (i8 / u8) | Portable/native range |
|---:|---:|---:|---:|
| 7 | 0.888 / 0.831 | 0.886 / 0.831 | 0.998–1.003× |
| 8 | 0.733 / 0.719 | 0.581 / 0.581 | 1.235–1.262× |
| 17 | 0.520 / 0.499 | 0.456 / 0.438 | 1.138–1.140× |
| 128 | 0.382 / 0.381 | 0.209 / 0.209 | 1.823–1.837× |
| 257 | 0.381 / 0.379 | 0.202 / 0.200 | 1.886–1.895× |
| 2048 | 0.379 / 0.378 | 0.192 / 0.191 | 1.969–1.979× |
| 5120 | 0.379 / 0.378 | 0.190 / 0.190 | 1.979–1.995× |

The repeated-native CPU median ratio ranged 0.970–1.005
across all cells, including one wider-row control about 3% slower. Sub-vector
rows select the identical portable kernel in every arm; their ratios are a null
control, not a speedup. The vector-row diagnostic is locally favourable across
these widths; it does not establish the supported-ISA/shape or whole-operator
graduation bar. No end-to-end gain or memory reduction is inferred.

Final validation: native QLinearMatMul 19 passed / 1 manual diagnostic ignored;
optional MLAS QLinearMatMul 34 passed / 3 manual diagnostics ignored. The three
new correctness tests also pass in release (one manual diagnostic ignored).
All filtered inventories were checked. Both all-target strict Clippy builds,
dispatch reachability/manifest, feature-gate coverage, platform naming,
formatting and whitespace checks pass.

