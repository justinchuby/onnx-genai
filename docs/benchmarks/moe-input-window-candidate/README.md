# Contiguous MoE input windows: candidate not adopted

Decision: retain the single funded gather path from #2441. This candidate is
preserved only as an experiment; its production code is not landed by this report.
It passed correctness tests and reduced single-row workspace by 128–1,024 bytes
in these fixtures, but did not establish a repeatable latency benefit. Initial
qualified regressions were not reproduced at their original magnitudes, so they
are not presented as established universal losses either. Adding a hot-path
branch for an uncertain benefit does not meet the current performance/simplicity
tradeoff. This is not a permanent rejection of contiguous input borrowing.

## Candidate and correctness

The candidate checks actual routed row order and borrows a checked contiguous
subrange of the already-converted dense input. Sparse, repeated or reordered
indices retain the existing gather path. This does not eliminate the original
TensorView-to-dense conversion. Single-row batches need no gather reservation
because any expert receives that unique row at most once; multi-row batches
retain the entire coarse rows*hidden bound regardless of actual routing.
No new allocator, quota source or public EP API was proposed.

Native CPU 2,018 / 20 ignored and session 295; optional MLAS CPU 2,060 /
35 ignored and session 297 pass. Both strict all-target CPU/session Clippy modes
pass. A subsequent test-only assertion of the exact one-row projection-only
bound passes targeted execution in both feature configurations. New tests prove
original-pointer subrange borrowing and rejection of reordered/repeated/invalid
indices, plus real dense and streamed kernels mixing borrowed and gathered
expert groups without clobbering unused capacity.

`bound-validation.py` also executes the public Host pool and CPU ownership-carrier
4,096-iteration tests. The MoE-specific issuer regression has **17 iterations per
activation mode: one cold iteration then 16 warm iterations**, not 4,096 MoE
executions. Pool/carrier checks and operator checks are distinct evidence scopes.
The warm operator assertion observes no additional issuer callbacks, including
zero-byte accounting. Used-byte metrics alone are not callback-count evidence.

## Original session sweep

Apple M1 Max / macOS 27.0.1, Rust 1.98.0 / LLVM 22.1.8, default native release
builds with zero MLAS symbols. Synthetic public InferenceSession strict prepare
-> run, including output destruction. Four experts, top-k two, SiLU, seeded
MXFP4/f32, one or four configured Rayon workers. These are operator/session
fixtures, not model throughput, MLAS reference timings or CUDA results.

`partial=0` routes every row to the same two experts, forcing contiguous groups.
`partial=1` alternates two expert pairs across rows, forcing discontiguous groups
and exercising the gather fallback; dense cases then retain four experts instead
of two. `dense=0` disables retention; `dense=1` retains the selected f32 projections.
Times are CPU microseconds/call, median of three isolated process medians.

| M / H / I / threads / dense / partial | Before us | Candidate us | CPU ratio | Null ratio | Control pass |
|---|---:|---:|---:|---:|---|
| 1 / 32 / 32 / 1 / 0 / 0 | 7.9496 | 7.9562 | 1.0008 | 0.9971 | yes |
| 1 / 256 / 512 / 1 / 0 / 0 | 919.6697 | 955.0367 | 1.0385 | 0.9994 | yes |
| 4 / 256 / 512 / 1 / 0 / 0 | 967.9439 | 951.9533 | 0.9835 | 0.9751 | NO: excluded |
| 4 / 256 / 512 / 4 / 0 / 0 | 1198.3553 | 1300.5526 | 1.0853 | 0.9971 | yes |
| 1 / 32 / 32 / 1 / 1 / 0 | 5.8728 | 5.9341 | 1.0104 | 1.0027 | yes |
| 4 / 256 / 512 / 1 / 1 / 0 | 146.3571 | 145.5233 | 0.9943 | 1.0014 | yes |
| 4 / 256 / 512 / 1 / 0 / 1 | 2160.8750 | 2108.0208 | 0.9755 | 0.9709 | yes |
| 4 / 256 / 512 / 1 / 1 / 1 | 507.8177 | 507.7552 | 0.9999 | 0.9987 | yes |

The one-thread wide streamed and four-thread streamed cells initially rise
3.85% and 8.53%, with disjoint process-median CPU ranges and passing control
medians. The small cached case rises 1.04%, also with disjoint ranges. These
observations triggered one focused independent repeat. The multi-row one-thread
contiguous streamed cell fails the original control guard and is excluded.
No apparent gain in that excluded cell is accepted.

## One independent focused repeat

The same immutable before/candidate/null executables are verified by SHA-256;
common repetitions from the original pilots are reused. Three rotated isolated
processes per label, nine batches per process, no source/build changes.
All focused cells pass both duration and control guards.

| M / H / I / threads / dense | Before us | Candidate us | CPU ratio | Null ratio |
|---|---:|---:|---:|---:|
| 1 / 256 / 512 / 1 / 0 | 933.7156 | 928.0459 | 0.9939 | 0.9842 |
| 4 / 256 / 512 / 4 / 0 | 1194.0395 | 1204.1184 | 1.0084 | 1.0062 |
| 1 / 32 / 32 / 1 / 1 | 5.9411 | 5.8983 | 0.9928 | 0.9897 |

The large original regressions do not repeat at their initial magnitudes.
The focused wide single-row result changes -0.61% with overlapping ranges;
its identical-binary null changes -1.58%. The four-thread candidate changes
+0.84%, while its identical-binary null changes +0.62% and also has a slow outlier.
The small cached result changes -0.72% with overlapping ranges; its null changes
-1.03%. This is insufficient to establish either a reliable speedup or a universal
regression. Do not average studies to erase the original concerns or choose only
the favorable repeat. No causal explanation is assigned without profiling.

## Accounting and measurement protocol

All 72 original and 27 focused full outputs and argmaxes agree within their
case, including across studies. The probe requires finite outputs, stable warm
funding, exact equality of declared and manager-funded scratch, and zero Host
usage after teardown/trim. The 64 MiB Host authority covers workspace and retained
derived weights, not task tables, original conversions, FC intermediates,
outputs or all process memory. Single-row scratch changes 640->512 B or
9,216->8,192 B; multi-row bounds stay unchanged. Actual larger reused slots may
remain funded until normal idle reclaim; no per-token shrinking/refund is added.
Absolute peak RSS is recorded, not interpreted as savings.

Both builds use the same source path and manifest, the same temporary libc dev
dependency and cached ORT 1.32. Source/manifest/lock are restored before timing.
A shared stable flock `/tmp/onnx-genai-hostlock-darwin.flock` protects every build,
pilot and sweep, coordinated with the llama.cpp chat. The original eight cases
rotate baseline/candidate/identical-baseline-null order over three processes.
Each process excludes a 100 ms CPU warm-up and collects nine inference batches;
excluded 256-call pilots choose common repetitions targeting 100 ms. An unchanged
public ReLU control uses nine 100,000-call batches. The thresholds are >=20 ms
per inference/control batch and control-median drift within 5% for both candidate
and null. Raw ranges, wall times, CPU times, controls and flags are preserved.

Required main/session symbols, zero MLAS symbols and negative symbol-audit
controls are checked. The actual single-row funding reduction witnesses linking
the changed implementation. No hardware-tier generalization, end-to-end model
claim or CUDA/x86 execution claim is made from this ARM64 host.

## Preserved source and reproduction

Measured baseline: `c3697260b27b36129cddfb6b67a87dd42c3299cb` (#2441 head).
Candidate: `b439e633b8f1eddd008cc2a2c060514f63b3df0e`.
Candidate code is held on an unmerged experiment branch; this report changes no
runtime source. `candidate-source.patch` reconstructs the exact candidate file
on the recorded baseline and is verified against the source SHA-256 in
`provenance.json`. Compiler, source/probe and executable identities are recorded.

Exact probes, drivers and the stable-flock wrapper are committed alongside raw
rounds and summaries. They retain original absolute paths and temporarily switch
source, so reproduction requires an isolated source-only checkout and deliberate
environment preparation; do not run a historical driver in another chat's tree.
Raw outputs/logs/symbol audits and source backups remain `/tmp/nxrt-moe-window-perf`
and `/tmp/nxrt-moe-window-focused-perf`. Consumed binaries can be removed after
hash verification; evidence and the lock inode remain. Validation logs are
`/tmp/nxrt-moe-window-{native,mlas,clippy,mlas-clippy}.log` plus the targeted logs
named by `bound-validation.py`.

Revisit only with a profiled mechanism and repeatable benefit, keeping complete
input/fallback bounds and row-order proofs. Do not add benchmark-shape or model
identity exceptions to force a favorable result.
