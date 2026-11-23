# Float MoE projection workspace

Float `com.microsoft::MoE` now borrows its FC1, optional FC3, activated and
expert-output matrices from the same execution lane as gathered inputs. There
are no per-expert claims/refunds or private pools. Final accumulation/result,
routing, input/weight widening and GEMM implementation scratch remain outside
this migration. The scoped cap is not a complete process-memory cap.

## Lifetime and capacity

Let H be hidden width, I intermediate width, F the validated FC1 width, and G
be I for a separate FC3 projection, otherwise zero. The existing fixed work
threshold determines whether grouped execution is possible. The logical row
bound R is rows when grouped is impossible, otherwise rows*top_k. It never
uses the preparing pool width to exclude a route, since execution may change
pools. Neither route multiplies scratch by expert count.

| Route | Bank A width | Bank B width | Phase reuse |
| --- | --- | --- | --- |
| Non-gated | H | I | gather A → FC1 B → in-place activation B → FC2 A → scatter A |
| Gated, including SiLU+FC3 | max(H,I) | max(F+G,H) | gather A → FC1/FC3 B → activation A → FC2 B → scatter B |

One checked `WorkspaceLayout<2>` requires `R*(A+B)*sizeof(f32)` with disjoint
borrowed regions. The pool may lend a larger retained lane; its charge tracks
that actual capacity, not just this logical requirement. Banks are reused only
after the previous phase's last read.
The graph still prepares one shared workspace peak for sequential nodes;
independent concurrent executions need separate lanes. Overflow, short and
misaligned storage are rejected before output writes. Empty/zero-hidden
executions need no projection workspace. Single-row execution now uses a lane
for FC results even though its gather can still borrow the original row.

These are conservative fixed banks, not a minimal interval allocator. A gated
H much larger than I can reserve nearly twice the ideal phase peak; balanced
routing can also use less than the metadata worst case. The bound is explicit
and admission must fit it. Increasing attributed workspace reflects previously
unaccounted FC storage plus this bound; it does not prove an RSS reduction.

`zeroed_prefix` initializes only the actual projection destination, never
certifies initialization of untouched bank capacity, and performs no admission.
Both GEMM paths retain their zero-initialized destination convention. Per-expert
activation stays serial; grouped activation/gather retain the old threshold and
worker budget. Applied/Pending bias modes preserve the old bias and absent-bias
`+0` order, including their distinction from already biased projections. The
existing `activate`/`swiglu` math is shared; both Float workspace routes share
fused-pair classification. QMoE/BlockQuantizedMoE keep their active owned-result
seam and its original bulk-copy split-half path.

## Correctness and ownership evidence

- Native CPU: 2,019 passed / 20 ignored. Optional MLAS: 2,061 / 35 ignored.
  Session library: 296 passed. MoE focus: 22 passed / 1 ignored.
- A real governed Host pool covers all three SwiGLU layouts, SiLU+FC3,
  non-gated activation, zero rows/hidden/intermediate, both width orderings,
  threshold boundaries, skew and different preparation/execution pools.
  Independent scalar-reference outputs agree. In these exact-capacity tests,
  declared bytes equal funded backing, warm workspace issuer callbacks are zero (including zero
  bytes), and idle trim reconciles both books to zero.
- The renamed cross-pool resource test also passes with MLAS budget 1.
- Shared projection-layout target: 11 passed. Prefix initialization uses fresh
  MaybeUninit backing and proves that partial initialization does not certify
  the whole bank; capacity failure and subsequent reuse are covered.
- The named prefix test passes local nightly Miri with default settings and no
  MIRIFLAGS override. This validates that primitive, not every MoE execution.
- Strict CPU/session all-target and MLAS CPU all-target Clippy pass.
  Logs/drivers are retained here. No CUDA device was used or validated.

## Measured sources and protocol

Baseline is `4ccc0ce2d36f2d87197e4ef41e9e0b669789a9d1`: the previous gather-only
workspace with the linked-file identity fix. `candidate.patch` reconstructs the
measured candidate. After rebasing onto main #2447 (`28fad980b`), the same three
source files are preserved at `e165460c0c6d4f1367f536810e6b8b23fd02254d`.
Subsequent review corrects the scalar-oracle Rustdoc and restores the original
bulk-copy split-half path in the owned projection helper, outside the timed
Float workspace path. `source-after-measurement.patch` and
`published-source.json` record these differences and byte-identical hashes for
the entire timed Float prefix and shared GEMM primitive. Float expressions,
layouts and guards are unchanged. These timings must not be
relabeled as a comparison against a later main commit.

Apple M1 Max / ARM64 / macOS 27.0.1, Rust 1.98 release, native CPU backend.
Independent baseline/candidate builds at one source/manifest path clean the
changed release packages. Symbol audits verify main/InferenceSession and no
MLAS symbols; provenance retains binary/probe/source hashes. Rayon width is
reported, not a measured backend/core placement. The graph has four experts,
top-2 normalized routing, deterministic distinct float weight banks and
alternating expert pairs. `identity`, fused interleaved/split SwiGLU, unfused
SwiGLU and SiLU+FC3 are distinct probe variants.

Public InferenceSession strict prepare → run includes output-owner destruction.
The 64 MiB Host source funds workspace only; original weights, outputs, routing,
widening and unrelated allocator/library storage are outside it. Every process
checks declared/funded bytes, finite full outputs, argmax and final zero usage.
Builds and each entire sweep hold the stable host flock shared with llama.cpp.
Supplementary process snapshots show OS/app background activity: this is not an
idle-host or continuously observed per-core scheduling claim.

## Original eight-cell study

Three fresh processes per arm/cell, nine batches per process, with rotated
baseline/candidate/byte-identical-baseline-null order. Excluded pilots choose
common repetitions targeting 100 ms CPU per batch. Every formal inference and
Relu-control batch exceeds 20 ms CPU. All 72 full outputs/argmax agree within
shape. Seven control cells pass the ±5% median gate; the forced split-SwiGLU
cell fails because its null Relu median shifts -5.64%. Its timing is retained
and is insufficient performance evidence.

| Rows / H / I / Rayon | Variant / gate | CPU us before → after | CPU change / null | Wall us before → after | Wall change | Funded bytes before → after | Controls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 / 64 / 64 / 1 | identity / default | 6.895 → 6.743 | -2.22% / -0.19% | 6.896 → 6.743 | -2.22% | 0 → 512 | pass |
| 8 / 256 / 256 / 1 | identity / default | 135.989 → 131.561 | -3.26% / +1.11% | 135.992 → 131.565 | -3.26% | 8192 → 16384 | pass |
| 512 / 256 / 256 / 4 | identity / 0 | 7255.923 → 7693.231 | +6.03% / +3.95% | 1964.667 → 2058.660 | +4.78% | 1048576 → 2097152 | pass |
| 512 / 1024 / 1024 / 4 | identity / default | 110548.000 → 110377.000 | -0.15% / +0.17% | 28167.250 → 28136.375 | -0.11% | 4194304 → 8388608 | pass |
| 1 / 64 / 128 / 1 | swiglu1 / default | 15.255 → 14.647 | -3.98% / -1.13% | 15.255 → 14.648 | -3.98% | 0 → 1536 | pass |
| 8 / 256 / 32 / 1 | swiglu0 / default | 31.022 → 31.058 | +0.12% / -0.60% | 31.022 → 31.060 | +0.12% | 8192 → 16384 | pass |
| 512 / 256 / 256 / 4 | swiglu2 / 0 | 12694.875 → 13219.500 | +4.13% / +2.51% | 3474.964 → 3650.917 | +5.06% | 1048576 → 3145728 | FAIL |
| 8 / 64 / 128 / 4 | silu3 / 0 | 195.165 → 188.106 | -3.62% / +9.72% | 75.264 → 78.706 | +4.57% | 4096 → 24576 | pass |

The forced parallel cells are noisy: identity's +6.03% CPU point estimate comes
with a +3.95% null estimate and overlapping ranges. The small SiLU+FC3 cell's
CPU decrease accompanies a wall increase in this study. These facts preclude
claiming uniformly improved latency from the original sweep.

## Fixed diagnostic of the three parallel cells

The diagnostic uses the same sources/binaries (hashes verified), doubles the
existing repetitions to target 200 ms CPU, and executes each of six arm orders
once per cell: six fresh processes per arm, nine batches each. No new pilots,
rebuilt arms or result-driven repeated attempts replace the original study.
All 54 full outputs/argmax agree with the original data. All three duration and
Relu-control median gates pass. `diagnostic/protocol.json` records the fixed
protocol. The original summarizer assumed 9 rather than 18 records per cell;
after all 54 processes completed, only summary generation was corrected with
`summarize.py`; no measurements were rerun for that correction.

| Rows / H / I / Rayon | Variant / gate | CPU us before → after | CPU change / null | Wall us before → after | Wall change | Funded bytes before → after | Controls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 / 256 / 256 / 4 | identity / 0 | 7312.808 → 7450.865 | +1.89% / +1.38% | 2038.344 → 2021.725 | -0.82% | 1048576 → 2097152 | pass |
| 512 / 256 / 256 / 4 | swiglu2 / 0 | 12757.406 → 12840.812 | +0.65% / -0.70% | 3457.359 → 3489.337 | +0.92% | 1048576 → 3145728 | pass |
| 8 / 64 / 128 / 4 | silu3 / 0 | 216.605 → 178.688 | -17.50% / -0.65% | 75.563 → 73.838 | -2.28% | 4096 → 24576 | pass |

Identity and split-SwiGLU CPU changes are +1.89% and +0.65%, with overlapping
process ranges and null changes +1.38% / -0.70%. The large original differences
did not reproduce stably, but a small cost cannot be ruled out. SiLU+FC3 uses
less process CPU in the diagnostic while its wall change is much smaller;
these are different metrics. Full ranges and raw batches remain in both
summary/runs files. Neither study establishes universal zero overhead or a
full-model speedup. The structural guarantee is zero warm workspace issuer
callbacks; ordinary validation, initialization and lane operations still cost
work. Adoption is for bounded, reusable ownership of the actual intermediates.

## Artifacts and cleanup

`driver.py`, `probe.rs`, external Cargo manifest/lock, exact patch, build logs,
provenance, both studies and output verification retain the measurement.
Raw `.f32` outputs, process snapshots and individual logs remain under
`/tmp/nxrt-float-moe-fc-perf`; `host-context.json` records their hashes and limits.
All 126 formal outputs agree within geometry/variant across both studies.

After verification, five probe copies/aliases and the private Miri target were
removed under the shared lock: 902,526,372 logical bytes, without a measured
physical-disk saving. `cleanup.json` records verified executable hashes.
Sources, raw results, shared dependencies and the stable lock inode remain.

Final review restored the pre-existing bulk-copy split-half loop in the owned
projection helper; the timed Float prefix and shared GEMM primitive remain
byte-identical to the measured checkpoint. Complete native CPU 2,019, MLAS CPU
2,061, session 296, layout 11 and strict native/MLAS all-target Clippy passed
again. `post-review-validation.json` records counts and final raw-log hashes;
those logs remain in `/tmp/nxrt-float-moe-fc-validation`. No timings were rerun
or substituted for this owned-path preservation.
