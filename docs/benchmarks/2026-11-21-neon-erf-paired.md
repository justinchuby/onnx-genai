# Rejected ARM64 Erf/GELU paired-loop experiment

Processing two independent four-lane vectors per loop does **not** improve the
native Erf/GELU implementation on this M1 Max experiment. All 180 cells pass
native A/A, but the paired loop regresses small Erf calls by about 17% and
small GELU calls by up to 23%; larger cases do not show a repeatable improvement.
**The candidate was withdrawn. Production retains the single-vector loop from
the [NaN-propagation optimization](2026-11-21-neon-erf-nan.md).**

## Hypothesis and implementation

The vendored `onnxruntime/core/mlas/lib/erf.cpp` also processes four elements
per iteration; it has no explicit paired-vector loop to port. Inspection of
our optimized single-vector GELU shows coefficient reloads inside the loop.
The hypothesis was that exposing two independent polynomial evaluations would
allow useful instruction overlap despite this pressure.

The candidate changes the mapper, not the polynomial. It loads two four-lane
vectors, evaluates the unchanged kernel independently on each, and stores both.
Any remaining complete vector and partial tail use the previous mapping.
No coefficient, arithmetic order within an element, dispatch seam, scheduler,
NaN repair policy, or numerical bound changes. BiasGelu stays on its original
loop. The exact former single-vector body remains as a test-only control, with
the same dispatch wrapper and per-call test-hit bookkeeping as the candidate.
The existing guarded-NaN controls remain for correctness tests.

An extra direct test checks before/after bits and output sentinels at lengths
0–257, 4095/4096/4097/65537, four input offsets, and both operations, including
random f32 patterns and explicit zeros, infinities, NaNs, and subnormals. The
complete optimized MLAS-feature library suite passes **1961 tests (30 ignored)**,
including the exhaustive NaN payload/sign proof and the existing numerical,
dispatch, scheduling, nesting, and bias tests. Correctness alone did not justify
keeping the candidate.

## Frozen experiment

Source `c438c593b115f3f2c9f591c059af1a1423c45fa5` is the complete measured
candidate, based on `97a28bedd6c3569c05ab0ac7f4a4f4ae438dbbbb`. Both source files
and runner hashes, source tree, executable hash, exact nonempty diagnostic
selector, compiler metadata, and child load/resource receipts are in
[metadata](2026-11-21-neon-erf-paired/metadata.json). The compressed
[Cargo JSON receipt](2026-11-21-neon-erf-paired/cargo-build.jsonl.gz) binds the
optimized MLAS lib-test executable (opt level 3, assertions/overflow checks off).
The entire checkout was clean before measurement and remained frozen throughout.

Hardware/software: Apple M1 Max, 10 physical/logical CPUs, 32 GiB, macOS 27.0.1,
rustc 1.98.0 / LLVM 22.1.8. Five explicitly labeled schema-3 arms are scalar-before,
paired native, paired native A/A, MLAS, and single-vector NEON before. The last
arm differs from schema 2's guarded-NaN control; the reader never guesses arm
identity from its position or count.

The same-binary protocol is unchanged: three sweeps, widths 1/4/8, serial and
normal-runtime modes, both operations, lengths 32/257/4096/65537/4194307, and
21 repetitions with rotating case/arm order. Batches contain
`clamp(1048576 / n, 1, 8192)` calls; medians use wall duration divided by batch
count. Full-buffer oracle/digest checks, input preparation, logging, and
three-position output tokens are outside timing windows. Native and before
full hashes/tokens agree across sweeps, widths, and modes. Eligible runtime
cases require actual worker participation, not merely a configured budget.
FPCR before/after is `[0, 0]` in every child.

**18900 samples, 900 full validations, and 180 cells are retained; none was
removed.** Every native A/A median ratio is within the predeclared 0.95–1.05
band, observed range **0.97825–1.02822**. All nine children exit zero; summed
wall time is **152.019 s**. The stable Darwin advisory host lock spans the
whole sweep. This is a shared desktop, without affinity or foreign-core
attribution; load receipts and the lock do not prove exclusive CPU use.
These are instrumented kernel-call wall latencies with test counters enabled,
not production latency, per-kernel CPU cost, model throughput, or cross-hardware
coverage. No CUDA, other ARM64, x86, ORT host-pool, or end-to-end model run is implied.

## Results

Each row spans 18 cells (all three sweeps, both modes, all three budgets).
Ratios above one favor the paired candidate. The complete
[raw samples](2026-11-21-neon-erf-paired/samples.csv.gz) and
[per-cell results](2026-11-21-neon-erf-paired/results.json) are preserved.

| Operation | Elements | Single-vector before / paired candidate | MLAS / paired candidate |
|---|---:|---:|---:|
| Erf | 32 | 0.851440–0.857321 | 0.884797–0.897438 |
| Erf | 257 | 0.927723–0.944800 | 0.901384–0.912255 |
| Erf | 4096 | 0.957736–0.982084 | 0.936076–0.957637 |
| Erf | 65537 | 0.954407–0.963132 | 0.939652–0.945336 |
| Erf | 4194307 | 0.957160–0.988639 | 0.920860–0.953632 |
| Exact GELU | 32 | 0.812444–0.959283 | 1.097683–1.161611 |
| Exact GELU | 257 | 0.956075–0.986051 | 0.917704–0.948203 |
| Exact GELU | 4096 | 0.988073–1.000534 | 0.924667–0.934434 |
| Exact GELU | 65537 | 0.994990–0.999028 | 0.942952–0.963940 |
| Exact GELU | 4194307 | 0.996223–1.006837 | 0.998940–1.068171 |

At length 32, paired Erf takes 16.64–17.45% longer, and paired GELU
4.24–23.09% longer. Apparent large-GELU improvements peak below 0.7%, within
the 5% repeatability band. There is no case for adding a production size gate
or a second loop strategy on this evidence. This is not activation graduation.

## Code generation and decision

The exact measured executable's single-vector GELU reserves `0x100` (256) stack
bytes. The paired function saves `0x70` and additionally reserves `0x220`, for
`0x290` (656) bytes. Its main loop has more coefficient reloads and temporary
spills. The [paired disassembly](2026-11-21-neon-erf-paired/gelu.disasm.gz),
[single-vector disassembly](2026-11-21-neon-erf-paired/gelu_single_before.disasm.gz),
and [extraction provenance](2026-11-21-neon-erf-paired/codegen.json) retain
exact symbol boundaries, commands, tool version, binary binding, and checksums.
This confirms added register pressure in the generated candidate; it does not
quantify which instructions caused each measured regression. No hardware-counter
or cycle attribution was performed.

The paired production mapper and its temporary controls/test were removed after
measurement. The Rust activation files are byte-identical to the tested
NaN-propagation baseline. Only the reader's explicit historical schema-3 support
and experiment evidence remain. The current producer still emits schema 2 for
the retained implementation; it does not silently emit paired-loop labels.
Do not retry simple loop unrolling as an assumed win. A later candidate needs a
mechanism that reduces live coefficient/temporary pressure and must retain the
same numerical and repeatability controls.

## Reproduction and reader controls

To reproduce the rejected candidate, check out its frozen source above, or apply
the [compressed candidate patch](2026-11-21-neon-erf-paired/candidate.patch.gz)
to baseline `97a28bedd6c3569c05ab0ac7f4a4f4ae438dbbbb` in an isolated checkout.
Commit the patched tree before running: the producer requires a frozen checkout.
The patch includes the candidate, its exact old-loop controls/test, and schema-3
producer; the [patch proof](2026-11-21-neon-erf-paired/patch-proof.json) verifies
the resulting complete tree against the measured commit.

```sh
cargo test -p onnx-runtime-ep-cpu --release --features mlas --lib \
  --no-run --message-format=json-render-diagnostics > /tmp/neon-erf-paired-cargo.jsonl
python scripts/bench_neon_erf.py run /tmp/fresh-neon-erf-paired \
  --binary /absolute/path/to/release/lib-test-executable \
  --cargo-receipt /tmp/neon-erf-paired-cargo.jsonl
```

Reanalyze the archived results from the current checkout:

```sh
python -O scripts/bench_neon_erf.py analyze docs/benchmarks/2026-11-21-neon-erf-paired
```

Use Cargo's exact lib-test executable record, not a cached filename.
Eleven [negative reader controls](2026-11-21-neon-erf-paired/reader-controls.json)
reject raw corruption, duplicate identity, wrong width/batch/arm label,
nonstandard FPCR, missing validation, absent worker participation, altered
single-before full digest/token, and changed saved summaries. Reanalysis under
Python `-O` passes for all three archived schemas.
