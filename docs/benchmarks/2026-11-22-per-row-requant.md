# ARM64 per-row QLinearMatMul requantization

This final recording follows the [initial candidate](2026-11-22-per-row-requant-initial.md).
The final source retains ledger-compatible Atomic dispatch counters once per
batch and adds separate per-thread, per-mode row witnesses. No static verifier
was relaxed. Counter reconciliation changed the compiled source, so all three
sweeps and final validations below were recorded afresh. The initial candidate’s
failed A/A cell remains archived separately; it is not pooled into this corpus.

Source: `6ff2fec10`, default CPU build,
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
64 where the required output is 63. The [saved inputs](2026-11-22-per-row-requant/association-counterexamples.json)
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
when `max(native-A/native-B, native-B/native-A) <= 1.05` for that metric’s medians.
CPU: 42/42 cells qualify. No failed A/A cells.
WALL: 42/42 cells qualify. No failed A/A cells.
No retry replaced a cell. Width 7 selects the identical portable function in all arms.

The table gives ranges over qualified independent sweeps, with the mean of both
native-arm medians as denominator; a larger portable/native ratio is better.
Widths 65–2048: CPU ratios 1.549–1.859.
Widths 65–2048: WALL ratios 1.549–1.860.
Small widths offer less benefit. This does not graduate the family or remove
its optional MLAS reference. That reference declines per-row A quantization, so
it is not one of these diagnostic arms.

| Columns | Output | CPU portable/native | Wall portable/native | Qualified sweeps CPU/wall |
|---|---|---|---|---|
| 7 | Int8 | 0.997–1.003 | 0.996–1.003 | 3/3; 3/3 |
| 7 | Uint8 | 0.999–1.005 | 0.999–1.006 | 3/3; 3/3 |
| 8 | Int8 | 1.134–1.150 | 1.134–1.150 | 3/3; 3/3 |
| 8 | Uint8 | 1.075–1.091 | 1.074–1.091 | 3/3; 3/3 |
| 9 | Int8 | 1.024–1.025 | 1.024–1.024 | 3/3; 3/3 |
| 9 | Uint8 | 1.000–1.004 | 1.001–1.004 | 3/3; 3/3 |
| 17 | Int8 | 1.142–1.143 | 1.143–1.143 | 3/3; 3/3 |
| 17 | Uint8 | 1.118–1.121 | 1.117–1.120 | 3/3; 3/3 |
| 65 | Int8 | 1.562–1.569 | 1.563–1.569 | 3/3; 3/3 |
| 65 | Uint8 | 1.549–1.555 | 1.549–1.556 | 3/3; 3/3 |
| 257 | Int8 | 1.793–1.800 | 1.794–1.800 | 3/3; 3/3 |
| 257 | Uint8 | 1.793–1.798 | 1.793–1.799 | 3/3; 3/3 |
| 2048 | Int8 | 1.853–1.859 | 1.853–1.860 | 3/3; 3/3 |
| 2048 | Uint8 | 1.839–1.848 | 1.844–1.850 | 3/3; 3/3 |

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
  docs/benchmarks/2026-11-22-per-row-requant/run-1.csv
python3 -m unittest discover -s scripts -p test_bench_qlinear_per_row.py -v
python3 -O -m unittest discover -s scripts -p test_bench_qlinear_per_row.py -v
```

[Run 1](2026-11-22-per-row-requant/run-1.csv),
[run 2](2026-11-22-per-row-requant/run-2.csv),
[run 3](2026-11-22-per-row-requant/run-3.csv),
[aggregate](2026-11-22-per-row-requant/aggregate.json).
Raw transcripts, metadata, regenerated summaries, full validation logs, the
route red and reader control logs are adjacent. Recording used
`CARGO_INCREMENTAL=0` and the existing cached ORT prebuilt; Cargo/lock/source/
binary hashes and compiler versions are in run metadata. Full-test/wrapper
CPU/RSS readings concern complete command lifetimes and are **not** epilogue
performance evidence.
