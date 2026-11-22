# Generic f32 GEMM: borrowed column panels

The Generic backend's thin-M column-parallel route previously gathered every
worker's B columns into a dense vector, computed a second output strip, collected
all strips and copied them into the caller's output. Every call allocated
`sizeof(f32) * (K*N + M*N)` bytes of matrix elements, in addition to descriptors
and scheduler allocations. The copied B scaled with the entire weight, including
when the caller already supplied a resident, contiguous f32 initializer.

The replacement borrows B with its original row stride. Each output row is split
into non-overlapping mutable slices; workers own their column slices directly.
Only row-reference metadata is allocated. `GemmOutputRows` gives dense row blocks
and borrowed strips the same row access contract, so both retain the original
4x4 register tile, K=256 panel boundaries, multiply/add order and row reuse.
All production code in this route remains safe Rust, with no pointer aliasing
claims or copied-output writeback. Empty K initializes the caller output to +0
before any column window is formed.

For `S <= min(N, threads)` strips, the requested descriptor storage is bounded by
`S * size_of(Vec<&mut [f32]>) + M*S * size_of(&mut [f32])`. Allocator rounding and
Rayon metadata are not modeled by that expression. This is not zero allocation
or a claim of lower total memory for every tiny matrix; the removed storage is
specifically the B/C matrix-element copies. No new persistent cache, backend,
ISA choice, memory cap, caller-output allocation or compatibility API is added.
The native x86 SIMD, Apple Accelerate and optional MLAS GEMM routes are unchanged.

## Numerical and allocation evidence

The independent scalar oracle retains the old K-panel addition boundaries.
Tests check exact finite output bits and output sentinels across 1/2/4/8-thread
pools, partial row/column tiles, empty K and multiple K panels. Explicit zero,
subnormal, infinity and NaN fixtures check their distinct outcomes; no old
numerical tolerance or test is weakened. A counter/manifest entry checks the
actual Generic column route. The original M=1 reference test remains intact.

The separate `generic_gemm_allocation` integration binary uses the public
`MatMulKernel` with B explicitly marked as a constant initializer and a
child-process `generic` backend override. A global allocator
observes only the dispatching thread and all four explicitly registered Rayon
workers, excluding libtest's own bookkeeping. A deliberate 1,053,700-byte worker
allocation proves the observer is live. At M=3, K=1025, N=257, the first call and
four warmed calls allocate no individual region at or above 65,536 bytes. Small
row-reference/shape metadata is intentionally outside that claim. Exact output
checks also preserve the original accumulator's +0 sign.

Restoring the old B-copy algorithm makes the named guard fail on its first call
with 1,053,700 large allocated bytes. The production source is restored in a
`finally` block and its SHA-256 checked, then the same probe passes again. An
initial warmed-only version also rejected the old path with 4,214,800 bytes over
four calls; the final probe additionally rejects hiding a weight copy in a cache
on the first call.

## Performance diagnostic boundary

The ignored `bench_generic_borrowed_columns_ab` compares a test-only transcription
of the pre-change Generic implementation with current Generic and a repeated-current A/A control,
in the same explicit 1/4/8-thread pools. The test-only reference retains the old
actual matrix copies, original tile implementation and scheduling. Source:
`cef77f9f25b2b1379ee8d9339878b26eabe057cd`. It shares the unchanged tile constants
with the candidate to hold arithmetic width and blocking fixed. Every warm-up/sample checks exact output
against the independent panel-order oracle.

Each of 21 samples rotates arm order. One-call CPU-time windows exposed excessive
A/A noise for short parallel calls, so the final diagnostic batches a deterministic
amount of work per sample: `ceil(2^24 / (M*K*N))`, clamped to 1..128 calls, then
normalizes CPU/elapsed time to one complete GEMM call. Matrix/allocator operations
inside Generic are included. Tensor preparation, the outer MatMul boundary and
whole-session overhead are outside the interval. CPU time sums process threads;
Rayon spinning is included. Non-Unix builds explicitly report unavailable CPU
time rather than substituting elapsed time.

Local host: Apple M1 Max, 10 logical CPUs, 32 GiB, macOS 27.0.1, Rust 1.98.0.
The repository hostlock does not support Darwin. Each whole sweep uses the stable
`/tmp/onnx-genai-hostlock-darwin.flock` inode with a nonblocking kernel flock and
logs before/after load plus whole-command CPU/elapsed/maximum RSS. This coordinates
participating runs, without establishing affinity or excluding foreign activity.
Whole-command maximum RSS is not per-arm memory evidence. The allocation proof
above, rather than RSS or timing, establishes the eliminated large copies.

Enumerate the exact test and confirm one match before running, under the host
lock:

```sh
cargo test --offline --locked -p onnx-runtime-ep-cpu --lib --release \
  kernels::matmul::tests::bench_generic_borrowed_columns_ab \
  -- --exact --ignored --nocapture
```

## Local diagnostic results

Three separately locked batched sweeps, with 21 samples per arm in each. Ranges
below retain every per-sweep CPU median ratio, rather than selecting a favourable
run. [Per-arm summaries and provenance](2026-11-21-generic-gemm-borrowed-columns.results.json)
include CPU/elapsed medians, within-sweep CPU ranges, call counts, source hashes
and the allocation observer's controls.

| Pool | M×K×N | Copy/borrow CPU range | Borrow/repeated-borrow CPU range | Route |
|---:|---:|---:|---:|---|
| 1 | 1×256×512 | 0.990–1.003× | 1.000–1.008× | row control |
| 1 | 1×1025×257 | 0.999–1.000× | 0.999–1.000× | row control |
| 1 | 1×2048×2048 | 0.995–1.032× | 0.962–1.005× | row control |
| 1 | 3×1025×513 | 1.000–1.000× | 1.000–1.001× | row control |
| 1 | 7×1025×513 | 1.000–1.000× | 1.000–1.000× | row control |
| 1 | 15×257×131 | 0.997–1.001× | 1.001–1.002× | row control |
| 4 | 1×256×512 | 1.461–1.478× | 0.990–1.004× | column borrow |
| 4 | 1×1025×257 | 1.377–1.435× | 1.005–1.041× | column borrow |
| 4 | 1×2048×2048 | 1.475–1.502× | 0.998–1.010× | column borrow |
| 4 | 3×1025×513 | 1.202–1.209× | 1.002–1.003× | column borrow |
| 4 | 7×1025×513 | 0.999–1.000× | 1.000–1.002× | row control |
| 4 | 15×257×131 | 0.997–1.002× | 1.002–1.015× | row control |
| 8 | 1×256×512 | 1.321–1.349× | 0.986–1.015× | column borrow |
| 8 | 1×1025×257 | 1.334–1.369× | 0.990–1.003× | column borrow |
| 8 | 1×2048×2048 | 1.609–1.618× | 1.017–1.047× | column borrow |
| 8 | 3×1025×513 | 1.133–1.176× | 1.003–1.019× | column borrow |
| 8 | 7×1025×513 | 1.081–1.103× | 0.971–1.011× | column borrow |
| 8 | 15×257×131 | 0.991–1.015× | 0.981–1.020× | row control |

Column-route CPU ratios ranged 1.081–1.618× locally.
The unchanged row-route controls ranged 0.990–1.032×;
repeated-borrow ratios across all cells ranged 0.962–1.047×.
The controls do not establish a gain or regression for the row route. This is
one host and a scoped Generic-kernel diagnostic, not MLAS graduation, a universal
ISA/NUMA result or an end-to-end inference claim.

For the M=1, K=N=2048 column case, old matrix-element allocations totaled
16,785,408 requested bytes per call (16 MiB B plus 8 KiB output strips). The
replacement's row-reference descriptor request is 160 bytes at four workers or
320 bytes at eight workers on this 64-bit host, excluding allocator rounding
and scheduler/outer-kernel metadata. These are allocation-layout facts, not a
claim of an equally large reduction in peak RSS.

Validation on the measurement base: full native CPU library 1,919 passed /
17 ignored (including the new manual diagnostic); optional MLAS MatMul module
72 passed / 2 manual diagnostics ignored, selected from a 74-test inventory.
The new two numerical/dispatch guards pass in release (one manual diagnostic
ignored), and the public cold/warm allocation guard passes in debug and release.
The final source also passes native/MLAS all-target strict Clippy, dispatch
reachability/manifest, feature-gate coverage, platform naming, formatting and
whitespace checks. No existing correctness test was newly ignored.


After integrating merged #2397 (`d265b33bed7cdc50a6f793b8ce8c51569b2c03c3`), the
full native library passes 1,922 tests / 18 ignored, optional MLAS MatMul again
passes 72 / 2 ignored, and the public allocation guard again passes. Both strict
all-target Clippy configurations and the combined dispatch guards pass (42
counters, 31 claims, 3 unchanged exclusions, 1 unchanged feature exemption).
The manifest's append-only conflict retained both new claims. The GEMM source
remained byte-identical across rebase. The final allocation-probe hash includes
the explicit constant-B fixture, which also passes debug/release and rejects the
restored old copy path after integration.
