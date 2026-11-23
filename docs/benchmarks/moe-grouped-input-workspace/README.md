# MoE grouped input in the execution workspace

BlockQuantizedMoE previously allocated its gathered input separately for every
expert group. It now lends one gathered-input region from the same execution
workspace as projection scratch. Serial experts reuse that region; no new pool,
per-expert lease or public EP API is introduced. This expands actual accounted
scratch coverage, not a complete process-memory cap or an RSS-saving claim.

## Ownership and initialization

Peak bytes are the maximum complete serial FC1/FC2/optional-FC3 projection layout,
plus one `rows * hidden * sizeof(f32)` gathered input. They are added because
both are live during projection; neither is multiplied by expert count/top-k.
Total input rows bound one expert group because routing selects each expert at
most once per row. Dense admission still covers the complete streamed fallback.

One `WorkspaceLayout` binds the whole capacity/address range and disjoint regions.
The private f32 owner validates every region's size/alignment before creating
MaybeUninit references. Projected scratch is initialized lazily once per dispatch.
Gathering writes each actual row before exposing only its initialized prefix;
unused capacity is untouched and no full gather clear is added. Returned slices
have short exclusive borrows; CPU worker projections join before the next group.
The original scalar accumulation and route ordering are unchanged.

Full metadata validation precedes an empty-final-output fast path. Zero rows or
zero hidden width need neither scratch nor decoding. Previously zero hidden
width passed shape checks but could fail in dense weight decoding; an actual
empty-output regression now requires execution without a workspace and metadata
requirement zero. Invalid metadata is still rejected before the early return.

Original input/router conversions, task tables, biases, FC intermediates and final
output vectors remain outside this migration. It removes one specific per-group
allocation; whole inference is not allocation-free.

## Actual session comparison

Apple M1 Max, macOS 27.0.1, Rust 1.98.0 / LLVM 22.1.8, default native release
builds with zero MLAS symbols. Four experts/top-k two/SiLU, seeded MXFP4/f32,
the same two experts selected for every input row. One or four explicitly
configured Rayon workers. Measurements execute public InferenceSession strict
prepare -> run, including output destruction. They are not whole-model/token
throughput or CUDA results.

Times below are medians of three isolated process medians, CPU microseconds per
call. `dense=0` forces streamed projection; `dense=1` retains the selected dense
expert projections. A control pass qualifies the timing cell, not a claim of
zero cost. Wall-time medians/ranges and every numeric batch remain in JSON.

| M / H / I / threads / dense | Before us | After us | CPU ratio | Wall ratio | Null CPU ratio | Control pass |
|---|---:|---:|---:|---:|---:|---|
| 1 / 32 / 32 / 1 / 0 | 7.9360 | 8.0393 | 1.0130 | 1.0130 | 1.0024 | yes |
| 1 / 256 / 512 / 1 / 0 | 938.5566 | 928.4623 | 0.9892 | 0.9892 | 0.9903 | yes |
| 4 / 256 / 512 / 1 / 0 | 974.8932 | 969.5825 | 0.9946 | 0.9942 | 0.9978 | yes |
| 4 / 256 / 512 / 4 / 0 | 1312.4634 | 1236.4146 | 0.9421 | 0.9613 | 0.9411 | NO: excluded |
| 1 / 32 / 32 / 1 / 1 | 5.8684 | 5.8507 | 0.9970 | 0.9970 | 0.9934 | yes |
| 4 / 256 / 512 / 1 / 1 | 176.7696 | 175.6108 | 0.9934 | 0.9934 | 1.0027 | yes |

Interpretation must retain the process ranges, null arms and control qualifications
above. Small timing changes do not establish a universal speedup or zero-overhead
guarantee. No family-graduation or MLAS-removal conclusion follows from scratch
ownership changes.

The qualified original single-thread cells range from -1.08% to +1.30% CPU
point estimates. The tiny streamed case is approximately +0.103 us, with
overlapping process-median ranges: a small cost remains possible. Controls also
shift, so no isolated speedup is claimed for the small negative point estimates.

The original four-thread cell fails because its identical-baseline null control
drifts -6.58%; its apparent improvement is excluded. A single focused independent
repeat reuses the same verified binaries and common repetitions, with three
rotated processes per label and nine batches. Both control checks and duration
checks pass. CPU median is 1.230341 -> 1.229549 ms;
wall median is 0.418078 -> 0.416421 ms. Process-median ranges
overlap broadly. This supports no distinguishable change in that focused
fixture, not a universal zero-overhead guarantee. All nine outputs/argmaxes agree.
Its separate raw records and driver are in `focused-four-thread/`; the failed
original cell remains above. No further repeats were selected to obtain a win.

## Accounting and protocol

All 54 full output byte sequences and row argmaxes match within each case across
baseline/candidate/null. The probe checks finite results. Every final process
verifies manager-funded workspace equals its declared requirement, warm funding
is stable and Host ledger usage returns to zero after teardown/trim. A 64 MiB
Host authority funds workspace plus retained derived weights; it is not a cap on
all process allocations. Dense cases retain exactly `16 * H * I` bytes (two
selected experts, two f32 projections each), recorded independently from scratch.

| M / H / I / threads / dense | Before funded scratch B | After funded scratch B |
|---|---:|---:|
| 1 / 32 / 32 / 1 / 0 | 512 | 640 |
| 1 / 256 / 512 / 1 / 0 | 8192 | 9216 |
| 4 / 256 / 512 / 1 / 0 | 12288 | 16384 |
| 4 / 256 / 512 / 4 / 0 | 36864 | 40960 |
| 1 / 32 / 32 / 1 / 1 | 512 | 640 |
| 4 / 256 / 512 / 1 / 1 | 12288 | 16384 |

Each increase equals precisely `4*M*H`; it accounts for previously call-owned
storage rather than measuring RSS savings. Existing recording-governor tests
exercise 4,096 warm borrow/compute/return cycles and require zero issuer
callbacks, including zero-byte calls. The timing probe's used-byte counters
alone are not proof of issuer callback counts.

All build/pilot/measurement work holds the shared stable host flock
`/tmp/onnx-genai-hostlock-darwin.flock`, coordinated with the llama.cpp chat.
Both arms use the same source path/manifest, temporarily adding the same libc
dev dependency and restoring manifest/lock/source before timing. Baseline,
candidate and identical-baseline null rotate order across three processes; each
process excludes a 100 ms CPU warm-up and collects nine timed batches. Excluded
256-call pilots choose common repetitions targeting 100 ms per inference batch.
An unchanged public ReLU control uses nine batches of 100,000 calls. Duration
threshold is >=20 ms; control-median drift threshold is within 5% for both
candidate and null. A failed control cell is excluded, not corrected into a win.
Absolute process peak RSS is retained; no RSS delta is inferred.

Required entry/session symbols, zero MLAS symbols and negative symbol-audit
controls are checked. The candidate's actual funding delta is a load-bearing
witness that the changed kernel was linked. Final baseline reuse verifies the
same base revision, probe SHA, compiler and executable SHA before copying the
immutable original binary; it does not rebuild an unchanged baseline merely to
collect another artifact.

Measured base: `f667643f20f1bdfeea80c6b5c4e25070d26891bc`.
Final measured source: `24843e592e9818e891e27b875fac80e837f362b8`.
`provenance.json` records compiler and complete source/probe/executable hashes.
The first prototype `b934ee56b0d424ae4063e8de6d338a1d510f8e53` is preserved in
`prototype/`: 54 exact outputs, small positive tiny CPU point estimates with
overlapping ranges, and one wide streamed control failure. Source review found
three repeated layout validations; the final private composite binder removes
that duplication while retaining all geometry checks. The final independent
sweep supersedes prototype timings for acceptance; noise prevents attributing
every timing difference solely to validation consolidation.

Raw logs, outputs, symbol audits and source backups remain
`/tmp/nxrt-moe-gather-final-perf` and `/tmp/nxrt-moe-gather-perf`. Consumed binaries
may be removed after hash verification. Exact executed probes/drivers are retained
here. Reproduction requires an isolated checkout at the measured revision and
its recorded absolute environment paths/cached ORT; never run source-switching
drivers in another chat's checkout. The shared lock inode must remain stable.

## Validation

Final source: native CPU 2,012 / 20 ignored and session 292; optional MLAS CPU
2,054 / 35 ignored and session 294 pass. Strict all-target CPU/session Clippy
passes in both configurations. Tests cover region isolation, odd byte extents,
misalignment, gather ordering/reuse/untouched tail, empty gather, invalid source
indices/overflow/capacity, partial-failure initialization, dense scratch laziness,
empty outputs, complete peak accounting and warm issuer callbacks. The authorized
resource-design review found the empty-dimension boundary, confirmed its fix and
reviewed the consolidated binder without another blocker. No CUDA execution or
GPU completion/reclamation behavior was validated by this Host-only change.

Validation commands are in `validation.py`; final logs remain
`/tmp/nxrt-moe-gather-final-{native,mlas,clippy,mlas-clippy}.log`.

## Integration on current main

The branch was rebased onto main `8a2a3c21e` after #2439/#2440 merged. Both
measured gather source files retain identical SHA-256 identities. Upstream ARM64
math and the lazy-input interface changed, so the timing tables remain scoped
to their recorded source/base, not a new combined-endpoint measurement.

Integrated validation passes: native CPU 2,016 / 20 ignored and session 295;
optional MLAS CPU 2,058 / 35 ignored and session 297; all 48 public executor
integration tests. Strict all-target CPU/session Clippy passes in both modes.
`integration-source.json` records this separate checkpoint and dependencies;
logs remain `/tmp/nxrt-moe-gather-integrated-*.log`.

`measured-source.patch` reconstructs both exact measured source files on the
recorded base, verified against their SHA-256 hashes even if local measured
commit ids are absent from rebased history. Commit that source-only patch in an
isolated checkout before running a reproduction; the original absolute paths
and optional baseline cache must be handled deliberately. The exact stable-flock
wrapper is retained as `measured-host-lock.py`. These are historical executed
artifacts, not a turnkey public benchmark command for an arbitrary checkout.
