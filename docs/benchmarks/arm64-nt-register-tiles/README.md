# ARM64 NT register tiles, preserving each K chain

On Apple M1 Max / macOS 27.0.1, one thread, the native NT kernel lowers CPU
time in the measured 4-row and 16-row shapes by about 19% and 18%. A warm
resident CPU-EP MoE fixture improves 16.8%; streamed block MatMul has no
demonstrated endpoint gain. These are kernel/operator observations, not model
throughput, CUDA results, or GEMM-family graduation. MLAS remains faster on
several measured shapes.

## Implementation and numerical contract

The new private ARM64 kernel uses 4x4 output tiles and transposes four contiguous
B loads in registers. Each SIMD lane is an independent output column. Consecutive
K lanes are consumed in their original order with separate multiply/add
rounding; there is no K reduction or FMA. Column tails reuse each weight across
four rows; bottom rows use the portable kernel. Runtime NEON detection and
rows/columns >=4 gate dispatch. Small shapes retain the existing fallback.
There is no heap scratch, packed-weight copy, new resource request, worker pool,
or public API. The measured checkpoint also fixed an optional x86 dot-row
feature gate; that same fix is now inherited from main.

Finite oracle tests compare output bits across full/partial tiles and K tails,
including empty shapes. FMA- and reassociation-sensitive fixtures and an actual
public-dispatch two-ULP argmax fixture pin the arithmetic contract. Nonfinite
classes, signed zero, subnormals and guarded output extents are covered; NaN
payload identity is not promised. Disassembly of the measured candidate has
44 `fmul`, 92 `fadd`, no `fmla`, `fmadd` or `faddp`, plus register permutations.

## Default-build serial controls

CPU microseconds/call, median of three isolated process medians; each process
has nine batches. Shapes are M x K x N. Generic selects the built-in dispatcher,
not a forced scalar ISA. Both default binaries have zero MLAS symbols.

| M x K x N | Before us | After us | After/before | Null/before |
|---|---:|---:|---:|---:|
| 1 x 4096 x 1 | 3.697461 | 3.767762 | 1.0190 | 0.9996 |
| 1 x 4096 x 64 | 61.855314 | 61.899488 | 1.0007 | 1.0002 |
| 4 x 1024 x 64 | 19.495134 | 15.795251 | 0.8102 | 0.9997 |
| 16 x 512 x 64 | 38.656238 | 31.598687 | 0.8174 | 0.9999 |
| 5 x 35 x 7 | 0.178916 | 0.173967 | 0.9723 | 1.0080 |
| 4 x 4096 x 4 | 4.401521 | 3.872454 | 0.8798 | 0.9938 |
| 4 x 1 x 4 | 0.011021 | 0.008069 | 0.7322 | 1.0000 |

The two unchanged unit-row cases have overlapping process-median ranges;
the first point estimate is +1.9%, so this is not a universal zero-overhead
claim. `summary.json` retains process ranges, wall time and control ratios.
All 63 default processes produce equal full output hashes and row argmaxes
within each shape. An independent f64 oracle checks error <=1e-4+1e-5*abs(oracle).

## Corrected same-binary reference

Optional `mlas` build; Generic and explicit Mlas alternate order in nine paired
batches, three isolated processes per shape. Native/reference CPU ratios below
are medians of within-process paired batch-ratio medians, not ratios of the
marginal absolute medians. Absolute times are marginal medians in microseconds.

| M x K x N | Native us | MLAS us | Paired native/MLAS |
|---|---:|---:|---:|
| 1 x 4096 x 1 | 3.6968 | 8.7874 | 0.4208 |
| 1 x 4096 x 64 | 61.8567 | 48.5705 | 1.2757 |
| 4 x 1024 x 64 | 15.8094 | 11.2845 | 1.4011 |
| 16 x 512 x 64 | 31.3254 | 13.5877 | 2.2967 |

Both routes satisfy the same f64 error bound; bit identity versus MLAS is not
required. The larger-row native gaps remain about 1.4x and 2.3x. This increment
does not justify removing the research reference or graduating the family.

## Actual public CPU-EP operators

`get_kernel` then `kernel.execute`, one thread, default builds without MLAS.
This is not an InferenceSession or a full model. CPU milliseconds/call:

| Fixture | Before ms | After ms | After/before |
|---|---:|---:|---:|
| matmul / streamed / 4 / 1024 / 2048 | 3.735423 | 3.763769 | 1.0076 |
| moe / warm / 16 / 512 / 512 | 0.654301 | 0.544147 | 0.8316 |

Streamed BlockQuantizedMatMul uses MXFP4/f32, no retained dense expansion.
Its process-median ranges overlap: before 3.726–3.859 ms, after 3.704–3.774 ms.
No endpoint gain or diagnosed bottleneck is claimed from that result.

Warm BlockQuantizedMoE uses H=I=512, two experts, top-k=1, SiLU, M=16.
It retains 2 MiB of derived weights under a 64 MiB derived-weight authority.
Two initial expansions are followed by zero timed expansions and two cache
hits/call. CPU time falls 16.8%, with disjoint process-median ranges
(0.646–0.663 versus 0.541–0.550 ms). All 18 operator processes preserve full
outputs and row argmaxes. Ledger use matches retained derived bytes and returns
to zero on kernel drop. This cap covers derived weights, not all process memory
or the separately proposed funded workspace. These counters do not establish
zero-byte issuer-call counts; the new math introduces no resource operations.

## Protocol, provenance and rejected evidence

Measured source: `eab5878c5273e8486ee2c6b728eb813f03f20e3f`.
Baseline: `bb1b14cf3f564e7b41ac6b5cc30df19a77b0105c`.
Rust 1.98.0 / LLVM 22.1.8, aarch64-apple-darwin, release profile.
Exact production-source, probe and executable SHA-256 identities are in both
`provenance.json` files. Serial baseline reuse was verified against the earlier
immutable executable, compiler and probe identities; it was not rebuilt merely
to generate another identical artifact.

All compilation and measurement held the shared stable flock
`/tmp/onnx-genai-hostlock-darwin.flock`, coordinated with the llama.cpp chat.
The probe excludes a 100 ms CPU warm-up and adaptive batched pilots, selects
common repetitions targeting 100 ms for the faster arm, then rotates baseline,
candidate and an identical-baseline null across three process repetitions.
Each process has nine batches plus an unchanged ReLU control (100,000 calls).
All final default/operator duration guards (>=20 ms) and control-median drift
checks (within 5%) pass. The raw reference CPU/wall medians are essentially 1.0;
a <=1.2 guard rejects unexpected concurrent CPU work.

Thread environment: RAYON_NUM_THREADS=1, MLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, and crucially
ONNX_GENAI_MLAS_THREADPOOL_THREADS=1. Absolute process peak RSS is retained in
raw runs; no RSS difference or savings is claimed.

The first prototype (`74f9602b6e5487d36d5b61d23e3d40adebefe932`) regressed
5x35x7 by about 23% because one-row column-tail calls lost four-row weight reuse.
The final source fixes that regression; the final independent sweep above is
the evidence used for acceptance. Initial optional-reference timings were
rejected: MLAS_NUM_THREADS alone did not constrain this project's global pool;
CPU/wall around 3.66 exposed background worker activity. They are not used in
any table. Initial harness corrections also preceded formal measurement:
backend argument order, a symbol audit based on required symbols rather than
an arbitrary 1,000-symbol threshold (a valid binary had 978), and batched pilots
to avoid zero-resolution one-call CPU times. Empty/only-main negative controls
check the final symbol audit.

Raw final logs/output buffers/symbols/disassembly are retained at
`/tmp/nxrt-neon-nt-refined-perf` and `/tmp/nxrt-neon-nt-operators`; rejected
prototype and invalid reference records remain at `/tmp/nxrt-neon-nt-perf`.
Consumed executables may be removed after hash verification; records remain.
This directory commits the raw rounds, summaries, provenance, exact Rust probes,
executed Python drivers and final native disassembly. Drivers intentionally
retain their original absolute environment paths. Reproduction must use an
isolated checkout at the measured source, cached ORT, and the shared lock; they
clean/rebuild private packages and temporarily restore the baseline source.
Do not run them in another chat's working tree.

## Validation and remaining work

Final source: native CPU 2,005 passed / 20 ignored, session 286 passed;
optional MLAS CPU 2,047 passed / 35 ignored, session 288 passed. Strict all-target
CPU/session Clippy passes in both modes. Exact validation commands are in
`validation.py`; final logs remain `/tmp/nxrt-neon-nt-final-check.log` and
`/tmp/nxrt-neon-nt-{native,mlas,clippy,mlas-clippy}.log`.
No CUDA device, x86 execution, Linux/Windows ARM64 execution or whole-model
throughput was validated here. Further native capability work must close the
measured reference gap while retaining the scalar-chain/argmax contract.

## Integration after upstream merges

After measurement, #2438 and #2427 merged. This branch was rebased onto main
`f667643f2`; the NT implementation and all ARM64 math are unchanged. The only
MatMul-file difference versus the measured source is an upstream duplicate
non-MLAS cfg annotation on an x86-only helper. Original hashes and measured
revisions above remain authoritative for timing. In particular the historical
operator probe uses the pre-migration direct-execute contract; use the recorded
source checkpoint to reproduce it, rather than running it against today's
workspace-required operator API. No performance numbers are asserted for the
rebased endpoint or reinterpreted as a new baseline comparison.

Rebased local unit validation passes: native CPU 2,013 / 20 ignored and session
292; optional MLAS CPU 2,055 / 35 ignored and session 294. Integration source
identities and the separate validation driver are recorded alongside the original
measurement artifacts. Logs use `/tmp/nxrt-neon-nt-integrated-*.log`.
Strict all-target CPU/session Clippy passes in both configurations.
