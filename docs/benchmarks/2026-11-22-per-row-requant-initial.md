# Initial ARM64 per-row requantization candidate

This initial candidate is retained separately from the [final recording](2026-11-22-per-row-requant.md).
It predates reconciliation with the Atomic-only dispatch ledger checks; its
thread-local route witness and measurements remain valid for this source. The
final source restores ledger counters at batch dispatch while keeping local
row witnesses, and is re-recorded rather than reusing these timings.

Source: `012573cb34e522d86a5629f6bcd6976e6d8728e8`, default CPU build,
Rust 1.98, Apple M1 Max / ARM64 macOS. Three independent process sweeps use the
same source and binary hashes, recorded in each run's metadata. Each cell has
21 rotating portable/native-A/native-B samples, with exact output checks
before and after every timed block and a separate calling-thread vector witness.

The prior per-row activation-scale epilogue was scalar. Native NEON now forms
`(a_scale * b_scale) / y_scale` in four-lane registers, then uses the same
NaN/clamp/ties-to-even/pack core as shared scales. No reciprocal, reassociation,
FMA or scale-buffer allocation is introduced. Widths below eight keep the
portable route; ISA/extent/zero-range failures and tails remain portable.
Two positive finite counterexamples distinguish the required scale association:
alternative reciprocal, accumulator-first or reassociated expressions produce
64 where the required output is 63. The [saved inputs](2026-11-22-per-row-requant-initial/association-counterexamples.json)
are also pinned in the source tests.

## Measurement scope

The clocks bracket repeated **synchronous serial row-epilogue calls**, including
loop/black-box overhead. Inputs/output allocation, warm validation, checksums
and printing are outside the timed interval. Process CPU uses `getrusage`;
wall time uses `Instant`. There is no worker pool or GEMM in this diagnostic.
Native arms include test-only TLS witness updates; shipped builds omit them.
These are portable/native row comparisons, not model speedups or CPU savings.

The Darwin shared flock is held across recording; metadata retains its path
and wrapper load readings. No core affinity, foreign-core isolation, clock
frequency or other hardware-tier equivalence is claimed.

## Results

All 2,646 samples / 42 cells are retained. Each metric qualifies a cell only
when `max(native-A/native-B, native-B/native-A) <= 1.05` for that metric's medians.
41/42 CPU and wall cells pass. Sweep 3, width 8, Int8 is **unqualified**:
CPU A/A 1.0591603 and wall A/A 1.0596928. It remains in the raw CSV and JSON;
its timing is excluded from the qualified ranges below. No retry was used to
replace that cell. Width 7 selects the identical portable function in all arms.

The table shows the range of qualified sweep medians, using the mean of both
native-arm medians as denominator; a larger portable/native ratio is better.
For widths 65–2048, qualified CPU ratios are 1.569–1.854 and wall ratios
1.569–1.854. Small widths offer less benefit and this does not graduate the
family or remove its optional MLAS reference. The current MLAS reference route
declines per-row A quantization, so it is not one of the diagnostic arms.

| Columns | Output | CPU portable/native | Wall portable/native | Qualified sweeps |
|---|---|---|---|---|
| 7 | Int8 | 1.000–1.008 | 1.000–1.008 | 3/3 |
| 7 | Uint8 | 1.001–1.007 | 1.001–1.007 | 3/3 |
| 8 | Int8 | 1.135–1.135 | 1.135–1.135 | 2/3 |
| 8 | Uint8 | 1.100–1.122 | 1.099–1.123 | 3/3 |
| 9 | Int8 | 1.048–1.058 | 1.047–1.058 | 3/3 |
| 9 | Uint8 | 1.015–1.016 | 1.016–1.017 | 3/3 |
| 17 | Int8 | 1.157–1.158 | 1.157–1.157 | 3/3 |
| 17 | Uint8 | 1.135–1.137 | 1.135–1.138 | 3/3 |
| 65 | Int8 | 1.574–1.575 | 1.575–1.578 | 3/3 |
| 65 | Uint8 | 1.569–1.572 | 1.569–1.572 | 3/3 |
| 257 | Int8 | 1.798–1.806 | 1.798–1.808 | 3/3 |
| 257 | Uint8 | 1.794–1.800 | 1.796–1.803 | 3/3 |
| 2048 | Int8 | 1.846–1.854 | 1.849–1.854 | 3/3 |
| 2048 | Uint8 | 1.846–1.852 | 1.846–1.852 | 3/3 |

## Correctness and reproducibility

The new native-entry witness fails before implementation and passes now.
Default CPU library: 1,955 passed, 19 ignored. The optimized QLinearMatMul
module also passes 22 tests (2 manual diagnostics ignored), including the
per-row and shared-scale numerical cases. MLAS-feature CPU library:
1,983 passed, 33 ignored (the pressure child is exercised by its bounded parent).
Tests cover misaligned slices, lengths/tails, signed/unsigned saturation,
NaN/Inf/subnormals, invalid-zero fallback, rounding-association counterexamples,
multiple batches and parallel rows. Default-release and MLAS-debug actual
QLinearMatMul allocation regressions still observe 24 execute allocations in
all six cells; the new path introduces no retained or temporary scale array.
No end-to-end model/token-ID, CUDA or other-ISA execution is claimed.

Both feature configurations pass strict all-target Clippy. The reader has 16
controls, run normally and under Python `-O`, including empty/incomplete and
duplicate corpora, protocol/cell/arm/iteration/rotation mistakes, duration/checksum/
native-witness failures, and retention of a synthetic bad-A/A cell.

```sh
cargo test --locked -p onnx-runtime-ep-cpu --release --lib \
  kernels::qlinear_matmul::tests::bench_per_row_requantization_neon_ab \
  -- --exact --ignored --list
# Hold the platform's shared host lock for recording, then run the same command
# with --nocapture instead of --list. Preserve every QRE_ record from one run.
python3 -O scripts/bench_qlinear_per_row.py \
  docs/benchmarks/2026-11-22-per-row-requant-initial/run-1.csv
python3 -m unittest discover -s scripts -p test_bench_qlinear_per_row.py -v
python3 -O -m unittest discover -s scripts -p test_bench_qlinear_per_row.py -v
```

[Run 1](2026-11-22-per-row-requant-initial/run-1.csv),
[run 2](2026-11-22-per-row-requant-initial/run-2.csv),
[run 3](2026-11-22-per-row-requant-initial/run-3.csv),
[aggregate](2026-11-22-per-row-requant-initial/aggregate.json).
Raw transcripts, metadata, regenerated summaries, full validation logs, the
route red and reader control logs are adjacent. Recording used
`CARGO_INCREMENTAL=0` and the existing cached ORT prebuilt; Cargo/lock/source/
binary hashes and compiler versions are in run metadata. Full-test/wrapper
CPU/RSS readings concern complete command lifetimes and are **not** epilogue
performance evidence.
