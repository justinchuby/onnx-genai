# Native ARM64 Softmax loop reuse

The native NEON row core unrolls maximum reduction across four independent
vectors, exp evaluation across two vectors, and normalization across four.
The vector sum still adds exponentials in the original order. A full 16-element
tile is required before initializing and merging the four maximum chains;
this avoids the small-row cost exposed by the initial candidate.

The existing scalar seam, NEON capability check, exp coefficients, gradual
underflow and nonfinite rules remain. No allocation, retained payload, quota
operation, scheduler policy or public EP interface is added. `UNROLL=false` is
a compile-time historical test control; production instantiates the new body.

## What the evidence establishes

Apple M1 Max, 10 logical CPUs, 32 GiB, macOS 27, rustc 1.98.0 / LLVM 22.1.8.
The optimized reference build contains both native implementations and MLAS.
The same stable Darwin flock covers construction and every process in a sweep.
External per-core CPU monitoring is unavailable here; the report does not claim
an otherwise idle host. Native unit-test row counters remain compiled in and
are matched between native arms; MLAS has no such counter.

The accepted measurement is **whole-process CPU for a fixed workload**, from
parent `wait4` after each child exits. It includes process startup, allocation,
four warm calls, complete f64 validation, the repeated row workload and output
verification. It is not a microsecond CPU snapshot, pure kernel latency or LLM
throughput. The native path is explicitly selected in the research binary;
these are not exact shipping-operator or model measurements.

Each process uses one arm. Arms are interleaved from the same executable:

| Arm | Entry |
|---|---|
| 0 | Old arithmetic, matched Result ABI, row validation and runtime ISA seam |
| 1 | Public native row entry with unrolled body |
| 2 | Exact duplicate of arm 1, process A/A control |
| 3 | MLAS softmax, called under the same outer row scheduler |
| 4 | Old arithmetic with only the row function forced out of line |

This changes the original same-address-space A/B condition because neither
in-process CPU clock survived the physical-capacity check below. Inputs,
executable, configuration, output validation and work quantities remain matched.

The main matrix has 528 child processes: three repetitions of setup-only and
fixed work for five arms, eight shapes and budgets 1/8, plus doubled-work arms
0/1 on the four larger shapes. A normal process performs
`floor(134217728 / (rows * columns))` calls; setup-only performs zero repeated
calls. Doubled work is exactly twice the normal call count. All processes
execute the same validation and warmup. The parent records exit status, CPU,
wall duration, peak RSS, full-output digest and actual worker/counter witnesses.
No process CPU is divided by calls and presented as kernel latency.

Predeclared quality bars are process A/A within 5%, setup CPU no more than 25%
of whole-process CPU for arms 0/1, and setup-adjusted work scaling within 15%
of 2x. The adjusted quantity is used only as a dose-response check; reported
performance ratios use unadjusted complete-process CPU. All samples also obey
a physical CPU/wall bound. Full native hashes match across old/new, outlined,
A/A, budget, setup/work and doubled-work variants. MLAS receives the same f64
error bound without requiring its distinct arithmetic to be bit-identical.

### Qualified results

Ratios are medians of the three paired **old/new whole-process CPU** ratios,
not ratios of two independently selected medians. Ranges include every pair.
Small controls are retained; their small differences are not a tiny-tensor
speed claim.

| Budget | Shape | Old / new CPU | Pair range | CPU reduction |
|---:|---|---:|---|---:|
| 1 | 1 x 4 | 1.032 | 1.026–1.037 | 3.1% |
| 1 | 1 x 7 | 1.022 | 0.989–1.038 | 2.1% |
| 1 | 1 x 8 | 1.020 | 0.975–1.023 | 2.0% |
| 1 | 1 x 16 | 1.098 | 1.082–1.136 | 8.9% |
| 1 | 1 x 257 | 1.274 | 1.261–1.277 | 21.5% |
| 1 | 32 x 1024 | 1.349 | 1.344–1.350 | 25.9% |
| 1 | 128 x 4099 | 1.289 | 1.288–1.334 | 22.4% |
| 1 | 16 x 65539 | 1.255 | 1.235–1.272 | 20.3% |
| 8 | 32 x 1024 † | 1.339 | 1.317–1.412 | 25.3% |
| 8 | 128 x 4099 | 1.233 | 1.195–1.246 | 18.9% |
| 8 | 16 x 65539 | 1.238 | 1.232–1.243 | 19.3% |

† The original budget-8 `32 x 1024` cell fails A/A (1.092) and scaling
(1.454 for the candidate); its three ratios span 0.909–1.319. It establishes
neither a benefit nor a regression and is **not replaced or deleted** in the
original matrix. One predeclared independent focused repeat uses the same
binary, workload and five arms plus setup/double controls (36 processes).
It passes A/A (0.998) and scaling (candidate 2.001). The table labels that
separate result; the original matrix remains 15/16 qualified, not 16/16.

Qualified larger shapes show about 19–26% less complete-process CPU. Tiny
controls establish no observed regression in these fixtures, not a universal
zero-overhead promise. In the single-budget larger cases, MLAS still uses
roughly 31–38% less CPU than the new native path. Softmax is **not graduated**,
and this does not justify deleting its optional reference or generalizing to
other CPUs, other dtypes, CUDA, full models or different scheduling regimes.
Peak RSS includes the test executable, input/output and validation buffers;
it is recorded per process, not treated as a kernel-memory improvement.

## Why earlier numbers are rejected

The rejected folders preserve the hypothesis sequence and all raw data:

- `study`: old row-body control, public candidate entry. It lacks matched
  public-entry validation, so short-row comparisons are not operator evidence.
- `matched`: matched public-entry Result ABI, validation and ISA seam.
- `outline-control`: adds old arithmetic with an independent row function.
- `final`: applies the full-16-element precondition to maximum unrolling.
- `clock`: tries `CLOCK_PROCESS_CPUTIME_ID` with a physical-bound hard guard.

All four `getrusage` short-window matrices are rejected for CPU claims. They
contain respectively 29, 28, 51 and 57 windows exceeding 11 CPU-seconds per
wall second on this 10-logical-CPU host; some exceed hundreds. Many A/A cells
still pass. `CLOCK_PROCESS_CPUTIME_ID` also fails the hard guard: 8,595,000 CPU
ns in 242,791 wall ns. The failed test stops collection. These observations
invalidate those instruments for this particular multithreaded short-window
protocol; they do not establish a general OS clock bug or identify its cause.
Their CSV/validation hashes and disassembly remain useful provenance, **not
performance conclusions**. The accepted whole-process protocol does not use
those deltas and does not substitute wall time for CPU time.

A portability gate changed while the failed `clock` build was active; its
source-freeze mismatch is also recorded. The accepted process runs verify
unchanged source and executable hashes across their complete matrices.
Excluded pre-timing failures include Darwin timeval integer types, an FPCR
parser expecting a second printed record when the test asserted the end mode,
and a missing parent-runner import. The incomplete FPCR pilot is retained with
its rejection reason. None is silently counted as a completed experiment.

The reference mechanism was inspected in the vendored
[`MlasComputeSumExpF32Kernel` and maximum/output loops](../../../crates/mlas-sys/vendor/mlas/onnxruntime/core/mlas/lib/compute.cpp).
Binary-bound disassembly exposed a separate effect: the new public entry calls
an independent vector row while the old control embeds it. The outlining-only
arm was added before adoption to distinguish that from the unrolled body.
The accepted process data retain both controls; separation alone does not
explain the larger-row result. This is generated-code/control evidence, not
hardware-counter attribution of individual instruction costs.

## Numerical and validation evidence

The direct old/new test covers widths 1–137 and 257/1024/4099, all four output
offsets, in-place/disjoint calls, sentinels, NaN/Inf rows and signed zeros. It
requires exact output bits. Existing maximum-at-every-position and tiny-positive
probability witnesses remain. The unchanged exp proof checks 6,315,476 lanes
against rounded f64 exp, observes max 1 ULP and retains its 2-ULP bar.

Final full-suite and lint receipts are under `validation/`; their concrete
counts are summarized in `validation.json`. Both named Softmax differentials
must actually run. The parser controls use synthetic data explicitly labeled
as such: complete matrices are accepted, while missing/duplicate processes,
zero work mislabeled as work, absent row-hit witnesses, impossible physical
CPU, unstable digests, missing validation and missing workers are rejected.

Full optimized testing also exposed an existing MoE fixture whose constant
scalar oracle took a different zero fold from the runtime row expression.
Its runtime kernel is unchanged. The fixture uses runtime tensor/oracle inputs,
still compares complete row bits across all layouts, and directly pins the
bias-state zero rounding before the min/clamp-containing activation chain.
The failing exact selector, corrected exact pass and final suites are recorded.
This is a test-oracle correction, not a Softmax-dependent MoE runtime fix.

## Reproduction and integrity

`manifest.json` hashes every retained payload. The process/focused folders
contain terminal resource receipts, metadata, archived stdout/stderr, the
measurement runner and reader, synthetic reader controls, Cargo artifact
receipts, frozen source snapshots and a reconstruction patch. Disassembly
records bind their exact measured executable hashes; large binaries remain
local and are not committed. The frozen measurement source precedes an
irrelevant redundant-cast lint cleanup in the diagnostic. The shipped NEON
primitive source hash is unchanged by that cleanup or the MoE test correction.

The runners preserve the actual host paths and hardware read receipt. To repeat,
use a clean checkout at the recorded parent, decompress and apply `candidate.patch.gz`, point
`root`, target/ORT paths and output directory at that checkout, and refresh the
hardware receipt from the real machine. Build with the recorded optimized MLAS
feature configuration; enumerate the exact ignored selector before running it.
Use the shared stable Darwin flock over the **whole** build/matrix; the runner's
busy-lock control checks that an outer wrapper already holds it. Never unlink
that inode. Process-end `wait4` and RSS units here are Darwin-specific; another
platform needs its own validated resource collector. Run the readers against
unchanged raw receipts and retain failed quality cells. `qualification.py`
records the final tests/lints; documentation paths are the only late additions.

No inference performance, memory-tier placement or resource-accounting policy
is selected by these benchmark arguments. The experiment supplies no reason to
change workspace admission granularity or add an EP protocol.
