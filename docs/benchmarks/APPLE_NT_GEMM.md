# Apple transposed-B GEMM and dense NBits prefill

Apple's native backend previously declined the NT GEMM capability. Dense NBits
prefill therefore rebuilt a KN expansion, even when the granted NK cache was
admitted. The new backend reads NK through row-major transposed-B Accelerate
SGEMM. The existing NT predicate and cache/grant decisions remain canonical;
no new scheduling or memory knob is added, and no MLAS reference is retired.
The application materializes no extra KN buffer; framework-internal packing or
scratch is outside these ownership observations.

## Numerical contract

This is a numerical backend extension. Matrix equality does not imply identical
f32 reduction order: the initial C probe found 5,046 different NN/NT bit patterns
among 134,400 outputs in 210 matrix cells on this M1 Max. Its maximum NT absolute
error versus f64 was about 3.50e-6. That probe is preserved as exploration, not
pooled with the Rust validation or treated as throughput evidence.

The production Rust tests use seeded finite inputs over M=1,2,3,7,17,33,65;
K=1,3,16,33,127,257; N=1,3,9,33,129. For both NN and NT each output is checked
against a separate f64 dot, with absolute bound
`(2*K+4) * f32::EPSILON * sum(abs(a*b))`. All 210 cells and 134,400 outputs are
checked, including non-square shapes, tails and f32-aligned offset slices.
Guard elements surround inputs/output. NN/NT bit differences and maximum NT
error are reported rather than asserting equality. A separate K=1 smoke test
checks NaN and signed infinities; it makes no NaN-payload preservation claim.

The prior universal bit-identity comments are corrected for every NT backend.
No model token parity, global numerical equivalence or other-ISA execution is
inferred from this finite corpus. The optional MLAS route remains a reference.

## Cached ownership and FFI boundaries

The native-entry regression fails before the change: Apple prefill has no
resident NK entry. Afterward three warm prefill calls build the granted NK
weight once and retain the same 396 Host ledger bytes until teardown.

A real executor uses symbolic row counts M=3,1,5,3 with grouped int4 weights.
Every output byte equals the exact binary fixture result 0.5, distinct kernels
exist, and prefill/decode share one 33*3 f32 expansion (396 ledger bytes).
A separate session owns another expansion; each teardown refunds its own.
These are ledger ownership and build-count observations, not RSS, allocator
byte counts, latency, model throughput or CUDA measurements.

A common validated operand type owns the NN/NT/GEMV slice borrows. Extent checks
and checked multiplication precede use; signed 32-bit dimensions are checked
only when selecting BLAS. The native large-GEMV branch keeps its usize extent
domain. Zero axes avoid invalid BLAS leading dimensions; K=0 writes zero output.
Both NN and NT reject short/long A/B/C buffers and oversized ABI dimensions with
an actionable error before writing. Release validation checks these guards are
not debug assertions. No user-facing compatibility forwarding API is retained.

NN and NT retain lazy ledger observation. The static guard now expects both
sites and continues to reject every eager recorder. The initial count-guard
failure is preserved separately from final results.

## Final validation

| Build | CPU library passed / ignored | Session library passed / ignored | Strict all-target Clippy |
|---|---|---|---|
| native | 1966 / 19 | 274 / 0 | both crates pass |
| mlas | 1994 / 33 | 275 / 0 | both crates pass |

Optimized SGEMM tests: four passed (including ABI guards and finite/nonfinite
cases); optimized native prefill reuse: one passed. Final Rust numerical records:

| Build | Cells / outputs | NN/NT different bits | Maximum NT absolute error vs f64 |
|---|---|---|---|
| debug | 210 / 134400 | 5048 | 3.497700199e-06 |
| release | 210 / 134400 | 5048 | 3.497700199e-06 |

Both outputs satisfy the stated f64 bound in every finite cell. This corpus
does not establish bit identity or model token equivalence.

## Reproduction and evidence

Final source fingerprints, complete tests/Clippy, optimized numerical/boundary
checks, route red, raw numerical output and exploratory C probe are preserved in
[`apple-nt-gemm`](apple-nt-gemm/).

```sh
cargo test --locked -p onnx-runtime-ep-cpu --lib \
  kernels::matmul::accelerate_gemm::tests::sgemm -- --list
cargo test --locked -p onnx-runtime-ep-cpu --release --lib \
  kernels::matmul::accelerate_gemm::tests::sgemm -- --nocapture
cargo test --locked -p onnx-runtime-session --lib \
  executor::tests::apple_nbits_prefill_and_decode_share_one_dense_expansion \
  -- --exact --nocapture
# Complete --lib suites and strict --all-targets Clippy are also run for CPU
# and session crates, both default and --features mlas configurations.
```

Host validation uses ARM64 macOS, Rust 1.98, the cached ORT prebuilt and the
shared Darwin lock. Wrapper lifetime CPU/RSS readings are command diagnostics,
not GEMM performance measurements. No affinity or isolation claim is made.
