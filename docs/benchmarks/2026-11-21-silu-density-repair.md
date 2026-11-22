# Native SiLU: repair sets without input rescans

This Apple M1 Max diagnostic evaluates the final native exp-based SiLU repair
strategy. It supersedes the ordinary-vector deficit reported in the
[historical baseline](2026-11-21-silu-native.md); the rejected MLAS rational's
accuracy counterexample remains valid. The exp polynomial and stable scalar
formula are unchanged, and the MLAS research route remains.

## Implementation and numerical proof

Record one exception bit per four-lane vector in each of four u32 lane masks.
Each bounded group contains at most `u32::BITS` vectors (128 elements), derived
from mask capacity rather than model dimensions. Ordered `abs(x) <= 18` rejects
NaN, infinity and out-of-band values. Decode `first + 4*bit + lane` only for
set bits; every recorded bit belongs to a live vector in the group, so its
index is below the group end. The sparse store's unchecked accesses carry
this local bounds proof and the existing equal-length caller invariant.

There are exactly three repair-set cases:

- Empty: no repair and no reread of input or output.
- Full: every mask equals its live-vector bit mask, so repair contiguous
  input/output slices without reclassifying or decoding individual positions.
- Partial: materialize four mask words on the stack and repair only set bits.

A complete bitmap occupies 16 bytes; compiler locals/spills and total stack
frame size are not measured by that fact. No heap scratch, persistent cache,
weight copy or new pool is introduced. The scalar-only case returns before
vector setup and preserves the output-length guard. Scalar tails retain their
existing staged f64-exp/f32 arithmetic. Existing block scheduling, host-pool
preference and nesting policy are unchanged.

An output fingerprint was captured from unchanged `393ca8430` before the
optimization: `2259da8105d9ccf1`. It covers 180,314 outputs from dense/mixed raw
IEEE inputs at 11 lengths, with NaNs canonicalized. It passes in debug and
release on the measured Apple ARM64 tier. The independent f64 error guard,
IEEE/tail tests and thread/host-pool invariance guards remain. Dedicated
observations verify empty/partial/full repair sets, all bitmap positions,
partial groups, scalar tails and output sentinels. Timing and production
instantiate observation as const-false; the observer is erased from them.
Disabling repair makes the named exceptional-lane guard fail, and the exact
source was restored before full regression tests.

## Measurement

The same binary rotates four arms: MLAS, the previous unconditional repair
scan, native, and repeated-native A/A. The before arm uses the same exp
arithmetic with precise tracking disabled at compile time; that instantiation
is reachable only from test code, not a production fallback or compatibility
API. Inputs and outputs are validated before timing, including bitwise
before/after equality. Both arms share the scheduler and requested budget.

Three sweeps rotate fresh 1/4/8-thread processes. There are serial/runtime
modes, three input populations and five lengths. Each arm has 25 samples per
cell, batching `ceil(2^20/n)` calls clamped to 1..16384. Four-arm order rotates;
with 25 samples, positions occur six or seven times, so small differences do
not establish a gain. Raw order and all process CPU/wall samples are retained.
The shared Darwin fcntl lock covers the entire sweep, not individual cells.
No affinity or exclusive-machine claim is made.

27,000 samples and 270 cells were retained from nine exact-selector diagnostic
children, each with one pass and no ignored test. Source commit `f71a780b0`
matches the recorded activation-file SHA-256. Full native regression: 1,927
passed / 18 existing ignored; optional MLAS: 1,948 passed / 29 ignored. Both
all-target strict Clippy configurations and five release contract tests pass.

Ratios below exceed one when native is faster. Ranges cover all widths,
modes and three sweeps; small gains are not a uniform speedup claim.

| Population | n | Before/native wall | MLAS/native wall |
|---|---:|---:|---:|
| ordinary | 3 | 1.000–1.011 | 1.110–1.131 |
| ordinary | 17 | 1.145–1.171 | 1.109–1.147 |
| ordinary | 4,096 | 1.565–1.589 | 1.112–1.117 |
| ordinary | 65,537 | 1.564–1.611 | 1.101–1.144 |
| ordinary | 1,048,581 | 1.557–1.572 | 1.136–1.399 |
| sparse repairs | 3 | 1.005–1.008 | 1.363–1.382 |
| sparse repairs | 17 | 1.082–1.089 | 1.342–1.351 |
| sparse repairs | 4,096 | 1.540–1.571 | 1.766–1.858 |
| sparse repairs | 65,537 | 1.516–1.571 | 1.736–1.817 |
| sparse repairs | 1,048,581 | 1.513–1.555 | 1.729–1.833 |
| dense negative tails | 3 | 0.999–1.011 | 1.693–1.709 |
| dense negative tails | 17 | 1.019–1.026 | 1.126–1.146 |
| dense negative tails | 4,096 | 1.048–1.057 | 1.063–1.087 |
| dense negative tails | 65,537 | 1.023–1.057 | 1.063–1.071 |
| dense negative tails | 1,048,581 | 1.048–1.054 | 1.066–1.072 |

Wall A/A ratios span about 0.978–1.014 across all cells. Large ordinary and
sparse before/after gains are well outside that control spread. Dense/tiny
before/after effects are smaller and do not establish broad gains.

**CPU qualification:** the eight-thread largest runtime cells have unreliable
per-cell CPU windows: A/A ratios span 0.931–1.914 overall. In one dense cell,
native and repeated-native CPU medians are about 2.036 ms and 1.064 ms for
the same function. That control failure prevents a CPU-cost conclusion; it
is not evidence of either CPU savings or a production CPU regression. Raw
CPU values remain visible. Whole-child usage covers several seconds and no
child was discarded, but that aggregate mixes all arms/cells and cannot repair
per-cell attribution. Validate a process CPU clock or lengthen batching before
resource/graduation claims. The independent wall clock and wall A/A control
support the latency diagnostic; no global MLAS graduation is declared.

[Per-cell metadata](2026-11-21-silu-density-repair.results.json) ·
[Raw samples](2026-11-21-silu-density-repair.samples.csv.gz)

This is a reused-buffer, warmed kernel diagnostic. It does not establish
whole-model speedup, RSS reduction or a ceiling without a model profile.
Other ISA/OS tiers and CUDA were not measured. The remaining MLAS capabilities
and global graduation requirements are still outstanding.

## Retained prototypes

The [block-mask prototype](2026-11-21-silu-block-repair.results.json) sped up
ordinary vectors but retained a full rescan when one exception appeared,
regressing sparse repairs. The [position-mask prototype](2026-11-21-silu-precise-repair.results.json)
removed sparse rescans but decoded dense exception sets one bit at a time.
Their complete samples are retained; their source commits are recorded in the
metadata. The final empty/full/partial classification addresses the actual
repair-set property without a workload-specific density threshold.

## Reproduce

```sh
python3 scripts/analyze_silu_native.py docs/benchmarks/2026-11-21-silu-density-repair.results.json
```

The metadata explicitly declares arms, repetitions, matrix axes and batching.
The reader checks their identities, cardinalities, archive checksum, unique
repetitions and every median/ratio. It also validates the previous datasets
with explicit declarations, with no format-guessing fallback. Corrupt archive,
wrong-width identity at unchanged cardinality, and duplicate-summary controls
are rejected, including with Python -O.

Build the recorded measured source, then hold the shared host lock for the
whole width/sweep loop. On the measured Darwin host the lock is the stable
fcntl inode `/tmp/onnx-genai-hostlock-darwin.flock`; repository hostlock.sh is
unsupported there. A single explicit diagnostic invocation is:

```sh
ONNX_GENAI_CPU_TASK_THREADS=4 RAYON_NUM_THREADS=1 \
  cargo test --release --locked -p onnx-runtime-ep-cpu --features mlas --lib \
  kernels::activations::silu_bench::silu_native_vs_mlas_rotating \
  -- --exact --ignored --nocapture --test-threads=1
```

The retained child receipts use the already-built executable, excluding Cargo
and build usage. In-test wall timers span synchronous calls, black-box
barriers and scheduling; process CPU includes all threads and timer
bookkeeping. Allocation, validation, warmup and printing are outside the loop.
