# Native ARM64 NEON Softmax

Contiguous ARM64 native Softmax rows of at least eight elements now evaluate
max/exp/sum/normalize with NEON, without linking MLAS. On the shared M1 Max
experiment, qualified large cases run **2.25–2.77× as fast as the
former scalar body**. Short rows retain their original scalar arithmetic.
**107/108 final cells pass native A/A; the failed cell is preserved. MLAS still
wins the large comparisons, so this is capability absorption, not graduation
or removal of the optional reference.**

## Arithmetic and dispatch

The four-lane exponential uses the existing native AVX2 degree-six fit and
Cody-Waite range reduction. Coefficients now have one shared definition; their
values and AVX2 arithmetic are unchanged. Runtime NEON detection and a minimum
of two complete vectors select the ARM64 route. Smaller rows and unsupported
ISAs retain the scalar implementation. Incomplete vector tails retain libm exp.
Only contiguous row reductions change; interior-axis handling and opset
coerce-to-2D/per-axis semantics remain.

NEON preserves ARM64's gradual underflow. For exponents below -126 it first
constructs normal `2^(k+126)`, multiplies by the polynomial, then scales by
`2^-126`. This retains representable tiny probabilities instead of adopting the
AVX2 path's subnormal flush policy. Arguments at/below -104, including -Inf,
become zero; NaNs are masked before integer range reduction and quieted with
sign/payload preserved. The row maximum uses ordered compare/select, ignoring
both quiet and signalling NaNs as the scalar maximum did. NaNs poison the row
sum; +Inf and fully -Inf rows still normalize to NaN. No numerical guard or
existing tolerance is relaxed.

The scale factor is selected **before** multiplication: ordinary lanes multiply
by one, rather than compute and discard a subnormal result. The initial
implementation's disassembly confirms an unconditional `MIN_POSITIVE` multiply
before its result select; the [final disassembly](2026-11-21-neon-softmax/native-out.disasm.gz)
selects the factor first. [Extraction provenance](2026-11-21-neon-softmax/codegen.json)
binds it to the exact measured executable. This establishes the generated-code
change, not a hardware-counter attribution of its cycle cost.

Original and improved exponential arithmetic remain compile-time controls only
in tests; production instantiates the improved form exclusively. There is no
runtime switch. In-place and disjoint-output calls share one row core, which
loads each complete vector before storing its matching output. No allocation,
intermediate tensor copy, new cache, or scheduler policy is introduced. The
existing production row scheduler is shared with the explicit diagnostic bodies
through a monomorphized helper.

## Numerical and dispatch proof

The direct exponential test performs **6315476 lane checks**: strided coverage
of the nonpositive normal domain and every f32 argument in `[-104, -87]`,
including normal/subnormal and round-to-zero transitions. Observed maximum
error against f64-exp rounded to f32 is **1 ULP**, within the retained 2-ULP bar.
The two exponential implementations agree bitwise on every finite check.
Separate per-lane probes pin infinities, signed zero, signalling/quiet NaNs and
unaffected neighboring lanes. Tests require ordinary rounding/subnormal/NaN
controls; there is no production FPCR read.

Native-entry tests cover widths 1–137 and 257/1024/4099, all four input offsets,
output sentinels, f64 row error <=1e-6, and exact in-place/out-of-place equality.
A numerical witness distinct from scalar, plus the dispatch counter, proves the
caller's NEON route rather than trusting a concurrently incremented counter.
A huge logit is placed at every position for widths 1–65: missing a vector or
tail in the maximum scan must fail. Nonfinite rows and positive tiny
probabilities at -90/-100 retain their contracts. Existing scheduling, masks,
alias-boundary, dtype, empty-extent and destination-poison tests remain.

At frozen final source `cd4dc683d95fca1a8a3f14123eca1a2550258bbb`, the complete
native debug library suite passes **1945 tests (18 ignored)**; the optimized
MLAS-feature suite passes **1966 (31 ignored)**. Both named Softmax integration
differentials pass, and strict all-target Clippy passes in both configurations.
The final direct-exp test is also retained with its named pass and observed
ULP/count output. [Validation receipts](2026-11-21-neon-softmax/validation.json)
bind the compressed raw logs. Formatting, whitespace, dispatch manifest and
reachability, platform naming, and feature-gate checks pass.

## Final experiment and qualified results

Apple M1 Max, 10 physical/logical CPUs, 32 GiB, macOS 27.0.1, rustc 1.98.0 /
LLVM 22.1.8. The [metadata](2026-11-21-neon-softmax/metadata.json) binds the
optimized MLAS lib-test executable to its exact
[Cargo artifact receipt](2026-11-21-neon-softmax/cargo-build.jsonl.gz), source
commit/tree/file hashes, runner hash, and nonempty exact ignored-test selector.
The complete checkout was frozen before building/measuring.

Five same-process arms are scalar-before, native, native A/A, MLAS, and initial
NEON arithmetic. Scalar/native per-row test counters are matched where NEON
runs. The initial arithmetic control uses the same row driver and bookkeeping on
active vector rows. It still evaluates its vector body for the short cases,
where native now uses scalar. It is an arithmetic control, not the old binary's
complete dispatch entry.
The diagnostic selects a function pointer per chunk, and counters remain
compiled in: these are controlled kernel-call latencies, not exact shipping
operator latency.

Three sweeps rotate widths 1/4/8 and case/mode/arm order. Each cell has 21
repetitions; batches contain `clamp(262144 / (rows*cols), 1, 2048)` calls.
Medians divide wall duration by batch count. Input preparation, full f64 oracle
validation/digests, logging, and three-position output tokens are outside each
timing window. Inputs are an ordinary finite `[-16,0]` grid; masked/special-row
latency is not measured. Validation requires actual worker-task participation
for eligible runtime cases, not just the configured budget. Per-arm output
hashes are stable across widths/modes; native A/A agree. Active vector rows
agree bitwise with the initial NEON arithmetic, and short rows agree bitwise
with scalar-before. Caller FPCR before/after is `[0,0]` in all nine children.

**11340 timed samples, 540 full-buffer validations, and 108 cells are retained.**
Native A/A uses the predeclared 0.95–1.05 band. **107 pass**, range
0.97090–1.03007. The failed cell is sweep 2, width 8, runtime 32x1024:
native 22.750 us versus identical native A/A 17.28125 us, ratio **1.31646**.
Its numeric checks pass, but its comparative performance is unqualified.
No timing rows were deleted or substituted. Its cause is not attributed to a
particular processor core or background task.

Ranges below span all three sweeps/modes/budgets. Each shape contributes 18
qualified cells except 32x1024 (17). Ratios above one favor native.
The [raw samples](2026-11-21-neon-softmax/samples.csv.gz) and
[per-cell results](2026-11-21-neon-softmax/results.json) include the failed cell.

| Rows | Columns | Qualified cells | Scalar-before / native | MLAS / native |
|---:|---:|---:|---:|---:|
| 1 | 4 | 18 | 0.998364–1.001639 | 1.654667–1.669932 |
| 1 | 7 | 18 | 0.981614–0.987642 | 1.263508–1.284085 |
| 1 | 257 | 18 | 2.578045–2.607466 | 0.487552–0.492220 |
| 32 | 1024 | 17 | 2.246626–2.593361 | 0.409424–0.456437 |
| 128 | 4099 | 18 | 2.515819–2.586131 | 0.423597–0.445851 |
| 16 | 65539 | 18 | 2.606244–2.771529 | 0.432489–0.470024 |

Small-case differences lie inside the 5% band; their arithmetic is the original
scalar body, not a claimed vector speedup. MLAS remains about 2.0–2.4x faster
on the qualified large cases. Native A/A repeatability does not prove every
comparator is stationary or the host isolated. The shared stable Darwin flock
spans the complete nine-child sweep (all zero exits; summed wall 6.207 s).
This is a shared desktop without affinity or foreign-core attribution. Only
wall durations support these comparisons; wait4 CPU/RSS receipts are provenance,
not per-kernel CPU/memory evidence. No CUDA, other CPU/ISA, ORT host-pool,
end-to-end model throughput, or token-ID claim is made.

## Preserved earlier mechanisms

All three frozen implementations and **31752 total timing samples** remain
archived; they are not pooled into the final comparisons.

- Initial four-arm version, source `0569b131866041d882e5d22d4f9ce146ac500123`:
  [9072 samples / 432 validations](2026-11-21-neon-softmax-initial/metadata.json),
  106/108 A/A pass. Failed cells are sweep1/width8/runtime/16x65539
  (1.11560) and sweep2/width8/runtime/32x1024 (1.07439). Its
  [source patch](2026-11-21-neon-softmax-initial/source.patch.gz) is based on
  `a584e1544f6a7d7ec1f10f6dc54d40c90f174918`; initial native-entry disassemblies
  are retained beside it.
- Factor-preselection version, source `21ecd7d425d46c057ab52c233cc993c53f05af5b`:
  [11340 samples / 540 validations](2026-11-21-neon-softmax-neutral/metadata.json),
  all 108 A/A pass. The initial-arithmetic/new-native ratio is not uniformly a
  win, especially for tiny rows and one runtime shape. The scalar comparison
  exposes no stable small-row benefit, motivating the final two-vector seam.
- Final version changes that seam and matched short-row bookkeeping. It keeps
  all earlier failed controls instead of rerunning an identical candidate
  until its numbers improve. No universal percentage gain is attributed solely
  to factor preselection.

## Reproduction and reader controls

Build the frozen source first and choose the exact lib-test executable from
Cargo's compiler-artifact record, then use a fresh output directory:

```sh
cargo test -p onnx-runtime-ep-cpu --release --features mlas --lib \
  --no-run --message-format=json-render-diagnostics > /tmp/neon-softmax-cargo.jsonl
python scripts/bench_neon_softmax.py run /tmp/fresh-neon-softmax \
  --binary /absolute/path/to/release/lib-test-executable \
  --cargo-receipt /tmp/neon-softmax-cargo.jsonl
```

Reanalyze the archived results from the current checkout:

```sh
python -O scripts/bench_neon_softmax.py analyze docs/benchmarks/2026-11-21-neon-softmax
python -O scripts/bench_neon_softmax.py analyze docs/benchmarks/2026-11-21-neon-softmax-initial
python -O scripts/bench_neon_softmax.py analyze docs/benchmarks/2026-11-21-neon-softmax-neutral
```

Schema 1 explicitly identifies four arms; schema 2 adds the initial NEON
arithmetic control; schema 3 also declares the native row seam and verifies
short-row scalar identity. These are evidence formats, not runtime compatibility
APIs. The current producer emits schema 3; reproduction of an older producer
requires its frozen source. Reanalysis of all three archives passes under `-O`.

Sixteen [negative reader controls](2026-11-21-neon-softmax/reader-controls.json)
reject corrupt raw bytes, duplicated/wrong shape identities, wrong batches/arm
labels, nonstandard subnormal mode, missing validation or worker participation,
altered A/A and initial-NEON full hashes/tokens, a wrong row seam, altered
short-scalar full hashes/tokens, and changed saved medians. A separate synthetic
poor A/A cell must be retained and marked unqualified. Control-driver assertions
are enabled; its saved-summary control invokes the optimized Python CLI.
[Control provenance](2026-11-21-neon-softmax/reader-control-provenance.json)
records the final reader used to check all historical formats.
