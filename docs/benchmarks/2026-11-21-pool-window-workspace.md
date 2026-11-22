# Native Pool window workspace reuse

The generic native Pool walker now reuses four rank-sized coordinate buffers
instead of allocating fresh vectors for every output window and kernel sample.
A counting allocator around actual kernel execution observes **8 allocations per
call** for all four measured geometries, cold and repeated, in debug and optimized
native builds and in the optimized MLAS-feature build. The old walker takes
261–146885 allocations in the same probe. These are allocation counts, **not
latency, peak RSS, CPU savings, or a claim of parity with MLAS**.

## Mechanism and ownership

The old walk decoded each flattened output/kernel position into a new vector,
allocated window starts per output, and allocated input coordinates per sample.
The [vendored MLAS 1-D kernel](../../crates/mlas-sys/vendor/mlas/onnxruntime/core/mlas/lib/pooling.cpp)
walks window positions directly. The absorbed principle is to keep traversal
state and advance it, rather than repeatedly materialize each coordinate.

`PoolKernel::pool_dense` now owns call-local output, start, kernel, and input
coordinate buffers, each sized by the runtime spatial rank. The shared
`strided::next_index` advances output/kernel coordinates in row-major order;
padding-only samples still advance the cursor. Workspace is **4 × spatial rank**
integer elements (128 logical payload bytes for rank 4 on this 64-bit host),
independent of output count and kernel volume. It is not a retained cache.
No input-size or kernel-size offset table is introduced. Sum order, NaN and
signed-zero behavior, max ties, dilations, both index storage orders, and the
N/C index base remain unchanged.

The index output buffer is created only when MaxPool indices are requested.
Value output staging remains: strided writes, dtype narrowing, and alias-safe
materialization still use the existing boundary. Input widening/materialization,
shape metadata, actual output values, and requested indices can still scale with
tensor size; the bounded claim applies to coordinate workspace, not whole-call
memory. BNNS and optional MLAS dispatch eligibility are unchanged. The existing
GlobalPool feature exemption only moves to its new source line.

## Actual execution allocation proof

Frozen implementation/test source: `f967e8c1589327f605f61b0211a60e6615bda549`.
The integration test uses the public factory and tensor-kernel boundary, with
caller-owned input/output and factory construction outside the observation.
Every allocation, zeroed allocation, and reallocation on the calling thread is
counted during the complete synchronous `execute` call. The generic walker
is serial. Other libtest threads are excluded by constant, allocation-free TLS.
LpPool with p=1 deliberately cannot route to MLAS/BNNS, so both feature builds
exercise the same native window walker rather than accidentally measure an
accelerated backend. Output values are checked, making an empty/no-op execution
fail, and an intentional allocation proves the counter is armed.

| Input length | Kernel width | Output elements | Before (debug and release) | After (cold and repeated) |
|---:|---:|---:|---:|---:|
| 64 | 1 | 64 | 261 | 8 |
| 64 | 17 | 48 | 1733 | 8 |
| 4096 | 1 | 4096 | 16389 | 8 |
| 4096 | 17 | 4080 | 146885 | 8 |

The new test fails by its exact named allocation assertion against the previous
production file at baseline `fb434c1edb6329eeaa9e0d1c87109af4fe5ba959`, in both
debug and optimized builds. The release negative control swaps only that source
file, records its source/binary hashes, verifies the named failure and four
nonconstant counts, and restores the candidate file in a `finally` block.
The new implementation passes cold/repeated equality and geometry-invariant
counts in native debug, native release, and MLAS-feature release configurations.

This measures allocator operations, not allocated bytes, retained bytes, or RSS.
The eight calls include geometry/output allocation; it is not an allocation-free
whole-kernel claim. No timing ratio or family graduation follows from the counts.
The optional MLAS Pool/NCHWc references remain.

## Numerical and boundary validation

A separate test-only historical walker retains the old flattened-coordinate
algorithm. **1920 non-vacuous cases** compare output bits and indices with the
new walker across spatial ranks 1–5, geometry variants, all four auto-pad modes,
ceil on/off, finite cancellation/tie patterns and NaN/Inf/subnormals, AveragePool
with and without padding in the divisor, LpPool p=1/2, and MaxPool with both
storage orders. Omitting MaxPool indices must keep value bits and return no
index staging. Additional cases cover zero input extents, zero batch, and
padding-only windows. Existing dtype, stride, dilation, selected-backend and
alias-boundary tests remain.

The complete native debug library suite passed **1941 tests (18 ignored)**.
That run preceded removal of two needless borrows in the test-only historical
oracle; the production walker did not change. The final frozen native Pooling
module passes **16 tests**, and the complete optimized MLAS-feature library
suite passes **1962 tests (30 ignored)**, including both new numerical proofs.

Strict all-target Clippy passes in both feature configurations. Format,
whitespace, dispatch claims/counter reachability, platform naming, and feature
gate checks pass. No compatibility API, new runtime knob, vendor dependency,
or model-specific shape gate is introduced.

## Scope and reproduction

Apple M1 Max, 10 physical/logical CPUs, 32 GiB, macOS 27.0.1, rustc 1.98.0 /
LLVM 22.1.8. Full matrices and release controls hold the stable Darwin advisory
lock. These deterministic allocation results do not require a wall-time A/A
interpretation. Wrapper resource timings include compilation/tests and are not
per-kernel measurements. Other CPUs/ISAs, CUDA, model throughput, and MLAS
performance parity were not measured.

[Metadata and source/binary hashes](2026-11-21-pool-window-workspace/metadata.json)
and compressed raw red/green/test/lint logs are retained alongside the
[release negative-control receipt](2026-11-21-pool-window-workspace/release-red-receipt.json).

```sh
cargo test -p onnx-runtime-ep-cpu --test pool_window_allocation -- \
  --exact pool_window_allocations_do_not_scale_with_output_or_kernel_elements --nocapture
cargo test -p onnx-runtime-ep-cpu --release --test pool_window_allocation -- \
  --exact pool_window_allocations_do_not_scale_with_output_or_kernel_elements --nocapture
cargo test -p onnx-runtime-ep-cpu --release --features mlas --test pool_window_allocation -- \
  --exact pool_window_allocations_do_not_scale_with_output_or_kernel_elements --nocapture
cargo test -p onnx-runtime-ep-cpu --lib kernels::pooling::tests::
```

To reproduce the negative control in an isolated checkout, retain the new probe
and replace only `pooling.rs` with its baseline file, then run the exact probe
above. Restore the candidate before normal development. Read the selected test's
named pass/failure and counts: an exit code or a zero-test listing alone does
not prove the allocation invariant was exercised.
