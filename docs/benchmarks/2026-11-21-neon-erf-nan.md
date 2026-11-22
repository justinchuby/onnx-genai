# ARM64 Erf/GELU: remove redundant NaN repair

The native NEON Erf and exact GELU paths now use the polynomial's NaN
propagation instead of explicitly masking and repairing every vector. On this
Apple M1 Max, qualified cells with at least 257 elements show about **7–12% less
wall time** than the previous NEON body, with bitwise identical outputs.
**179 of 180 cells pass native A/A; one cell is unqualified. MLAS still wins
some qualified comparisons, so this does not graduate the activation family
or remove the optional reference dependency.**

## Mechanism and correctness

Production instantiates `erf_lanes::<false>` / `gelu_lanes::<false>`. In ordinary
IEEE floating-point mode, NEON FMIN propagates and quiets NaNs. The small
polynomial preserves the payload; the ordered split comparison selects that
branch for NaNs, and the final sign OR restores the original sign. Inactive
large-branch lanes are still zeroed **before** exp range reduction and integer
conversion. Coefficients, evaluation order, numerical bounds, signed zeros,
infinities, scalar dispatch seam, scheduling, and partial-tail mapping remain.
The old explicit repair is a compile-time, test-only control, with identical
per-call test-hit counter bookkeeping. There is no production FPCR read or new
runtime setting. Generic scheduler instantiations remain in
`simd_activations.rs`.

The proof tests cover:

- All **16777214 distinct f32 NaN payload/sign combinations**, for both Erf
  and GELU and both implementations, expecting exactly `input_bits | 0x00400000`.
  Bounded 4096-element chunks avoid a retained giant allocation.
- **1400014** additional finite/random/special bit patterns compared bitwise
  before/after, plus lengths 0–67 at four offsets with output sentinels.
- The retained scalar-oracle bounds: maximum Erf absolute error
  `5.960464477539063e-8`; GELU error divided by `max(1, |x|)`
  `1.1920391808974291e-7`. Native dispatch witnesses, actual native scheduling,
  nested-Rayon split suppression, and BiasGelu consistency remain tested.

Proof and benchmark tests read FPCR and require ordinary rounding, subnormal,
and NaN controls. All nine measurement children recorded `[0, 0]` before/after.
These are output-value guarantees in the ordinary Rust floating-point
environment; they do not assert identical FPSR sticky exception flags or
coverage of nonstandard FPCR modes. BiasGelu latency was not measured.

## Frozen measurement and controls

Measured source: `93e80ae91f1a1708ea348efd469809ca4002fe0d`, tree
`325fe3cb52d28dca0e91cc2ef64fd9bbf869d0fd`. Apple M1 Max, 10 physical/logical
CPUs, 32 GiB, macOS 27.0.1, rustc 1.98.0 / LLVM 22.1.8. The optimized MLAS
lib-test artifact has opt level 3, debug assertions and overflow checks off.
The runner binds its executable to the exact Cargo JSON artifact and refuses
a dirty/untracked checkout before measurement. [Metadata](2026-11-21-neon-erf-nan/metadata.json)
records source, executable, runner and [Cargo receipt](2026-11-21-neon-erf-nan/cargo-build.jsonl.gz)
hashes, compiler details, exact nonempty selector, and child resource/load receipts.

Five explicit arms run in the same binary: scalar-before, native, native A/A,
MLAS, and guarded-NEON-before. Three sweeps rotate widths 1/4/8 and case/arm
order; 21 repetitions cover serial and normal-runtime modes, both operations,
and lengths 32/257/4096/65537/4194307. Each call batch is
`clamp(1048576 / n, 1, 8192)`. Medians use wall time divided by batch count.
Full validation, input preparation, logging, and output-token reads are outside
timing windows. Test counters remain enabled: these are instrumented kernel-call
latencies, not exact production latency or model throughput.

**18900 timed samples, 900 full-buffer validations, and 180 cells are retained.**
Full digests are stable across widths/modes; guarded-before/native output hashes
and tokens agree. Normal-runtime calls originate outside Rayon, with actual
worker participation required for eligible large cases. The stable Darwin host
lock spans all nine children (all exit zero; total wall 153.338 s). This remains
a shared desktop, without affinity or foreign-core attribution. The lock and
load receipts do not prove exclusive CPU use. Only wall durations support these
comparisons; child CPU/RSS receipts do not establish per-kernel CPU savings.
Other ARM64 CPUs, x86, CUDA, ORT host pools, and end-to-end models were not measured.

The predeclared native/native-A/A band is 0.95–1.05. **179 cells pass**, with
ratios 0.98111–1.02769. The failed cell is sweep 2, width 8, runtime GELU,
4194307 elements: native 0.854208 ms versus identical native-A/A 1.130667 ms,
ratio **0.75549**. Its numerical checks pass, but its comparative performance
is unqualified. No timing rows were removed and no rerun was used to replace it.
The cause is not attributed to a particular background task or processor core.

## Qualified results

Ranges include all three sweeps, both modes, and all budgets, excluding only
the explicitly unqualified cell from performance conclusions. Each row has 18
qualified cells except the final GELU row (17). Ratios above one favor native.
The complete [raw samples](2026-11-21-neon-erf-nan/samples.csv.gz) and
[per-cell results](2026-11-21-neon-erf-nan/results.json) include the failed cell.

| Operation | Elements | Qualified cells | Guarded NEON before / native | MLAS / native |
|---|---:|---:|---:|---:|
| Erf | 32 | 18 | 1.0356–1.0632 | 0.9795–1.0114 |
| Erf | 257 | 18 | 1.0764–1.1094 | 0.9574–0.9924 |
| Erf | 4096 | 18 | 1.1009–1.1147 | 0.9787–0.9849 |
| Erf | 65537 | 18 | 1.0769–1.1120 | 0.9560–0.9984 |
| Erf | 4194307 | 18 | 1.0735–1.1218 | 0.9401–0.9950 |
| Exact GELU | 32 | 18 | 1.0323–1.0709 | 1.1840–1.3567 |
| Exact GELU | 257 | 18 | 1.0817–1.0969 | 0.9499–0.9616 |
| Exact GELU | 4096 | 18 | 1.0858–1.1210 | 0.9165–0.9393 |
| Exact GELU | 65537 | 18 | 1.0957–1.1341 | 0.9348–0.9676 |
| Exact GELU | 4194307 | 17 | 1.0832–1.1199 | 0.9672–1.0729 |

For n >= 257 the qualified guarded-before/native ratios are 1.07349–1.13410,
corresponding to 6.85–11.82% less wall time. Some smallest-case changes lie
inside the 5% repeatability band; this is not a universal small-case win claim.
The remaining MLAS gap warrants separate loop scheduling/unrolling experiments;
that is a hypothesis, not an implemented or measured improvement here.

## Validation and reproduction

At the measured source, the complete native debug library suite passed
**1939 tests (18 ignored)** and the optimized MLAS library suite passed
**1960 (30 ignored)**. Strict all-target Clippy passed in both configurations.
The source also fixes a test-only QMoE metrics race exposed by the complete
suite: a mutex guard now retires thread-local cache owners before releasing the
metrics lock. Previously a late TLS destructor could subtract residency after
the next test reset its gauges. The exact regression failed with 4 versus 0
bytes before the fix and passes after it; production accounting is unchanged.

After measuring, main's squash-merged baseline was integrated through an
ancestry-only merge. Main at `02aeb1e9060953979a5dd65f2c8f68d0123b0ace` and the
original baseline had identical complete trees
(`ccc469085ec7437c3517f2c64f2c5158c9efd497`); the merge retained the measured
source tree exactly. Documentation and evidence were then added separately.

To reproduce, check out the measured source in a clean checkout, build before
measuring, and select the executable from Cargo's exact lib-test artifact:

```sh
cargo test -p onnx-runtime-ep-cpu --release --features mlas --lib \
  --no-run --message-format=json-render-diagnostics > /tmp/neon-erf-nan-cargo.jsonl
python scripts/bench_neon_erf.py run /tmp/fresh-neon-erf-nan \
  --binary /absolute/path/to/release/lib-test-executable \
  --cargo-receipt /tmp/neon-erf-nan-cargo.jsonl
python -O scripts/bench_neon_erf.py analyze docs/benchmarks/2026-11-21-neon-erf-nan
python -O scripts/bench_neon_erf.py analyze docs/benchmarks/2026-11-21-neon-erf
```

The current reader explicitly accepts both historical four-arm schema 1 and
five-arm schema 2; it never infers arm meaning from row count. Ten
[negative reader controls](2026-11-21-neon-erf-nan/reader-controls.json) reject
corrupt bytes, duplicate identities, wrong width/batch/arm labels, nonstandard
FPCR mode, missing full validation, absent worker participation, changed guarded
hashes, and changed saved summaries. Reanalysis passes under Python `-O`.
