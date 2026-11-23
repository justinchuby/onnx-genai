# Initialize projection scratch once per streamed dispatch

This follows the governed scratch migration in
[the earlier MoE report](../moe-session-workspace/README.md). A private owner
binds one required f32 prefix, initializes it lazily on the first streamed
projection, and lends short exclusive typed slices to later projections.
Dense resident hits do not initialize scratch. The proof ends at dispatch
return; it is never cached across pool borrowers. Projection results stay
owned, and scratch funding, shard count and arithmetic order are unchanged.
This removes repeated clearing and raw workspace reconstruction at each FC.

Measured baseline: `9886e7a9a201c71b6e7870645c45278429e18086`.
Measured candidate: `1a7ea336a6483fe9d55eebeb8c6203f5e9409c81`.
The implementation is retained as the parent of this report-only commit.
Both source versions were built at the same worktree/manifest path, cleaning
the changed release packages on every arm switch. Source restoration and all
changed-source hashes were verified. Exact executed sources, compiler/source/
binary hashes and raw numeric batches are in the `moe/` and `matmul/` folders.

## Conditions and protocol

Apple M1 Max, macOS 27.0.1, Rust 1.98.0, default native builds with zero MLAS
symbols and a load-bearing `_main` symbol audit. Seeded MXFP4/f32 projections,
zero dense retention, explicit governed Host pool, 64 MiB workspace/derived cap.
This scoped cap does not account for the full process working set. The MoE
fixture has four experts, top-k two, SiLU, and separate FC1/FC2 banks; all rows
select the same two experts. K/N below mean hidden/intermediate for MoE.

Actual public `InferenceSession` strict prepare→run includes output destruction.
Each cell uses three isolated processes per baseline/candidate/identical-
baseline-null label, nine timed batches per process, with label order rotated
across processes. Excluded pilots choose common repetition counts targeting
100 ms of process CPU time per batch. The unchanged Relu control has nine
100,000-call batches. There is one measured sweep per operator, with no
result-driven repeats. Build/pilot/sweep phases hold the stable shared
`/tmp/onnx-genai-hostlock-darwin.flock`, excluding the cooperating llama.cpp task.
Supplementary snapshots record process names/CPU use, not command lines;
Darwin provides no per-core foreign-utilization trace in this protocol.

All six cells pass the prespecified minimum-duration and ±5% control-median
guards. Shortest inference/control CPU batches: MoE 87.662/77.661 ms;
MatMul 86.834/74.526 ms. Three process medians are too few to
establish small effects independently of observed control/null drift.

## Results and limits

Medians of process medians; ranges and every raw batch remain in JSON.

| Operator / M / K / N / threads | CPU µs/call before → after | CPU change | Wall µs/call before → after | Funded B, both arms |
| --- | --- | --- | --- | --- |
| MoE / 1 / 32 / 32 / 1 | 8.055 → 8.138 | +1.03% | 8.055 → 8.139 | 512 |
| MoE / 1 / 256 / 512 / 1 | 931.787 → 916.426 | -1.65% | 931.785 → 916.431 | 8192 |
| MoE / 4 / 256 / 512 / 1 | 968.413 → 975.510 | +0.73% | 968.413 → 975.746 | 12288 |
| MoE / 4 / 256 / 512 / 4 | 1239.312 → 1219.425 | -1.60% | 422.569 → 413.945 | 36864 |
| MatMul / 1 / 32 / 32 / 1 | 3.566 → 3.594 | +0.79% | 3.566 → 3.594 | 512 |
| MatMul / 4 / 256 / 512 / 4 | 282.743 → 285.545 | +0.99% | 89.683 → 89.803 | 24576 |

MoE's larger single-token and four-thread CPU ranges lie below baseline, but
controls drift as well (the four-thread control drops 3.50%). Tiny MoE shifts
+1.03% with overlapping ranges and a +0.83% identical-binary null shift. The
four-row/single-thread ranges also overlap. These observations support no
universal speedup or zero-overhead claim.

MatMul's four-thread CPU median rises 0.99%, with a narrow gap between process-
median ranges, while its unchanged control rises 1.02%; wall time rises 0.13%
with overlapping ranges. A small CPU cost is not ruled out. Tiny MatMul ranges
overlap, and its identical-binary null moves −1.49%. Keep these limits rather
than normalizing away the controls or declaring a regression absent.

This is a separate matched experiment against the already-governed
implementation. Do not multiply timing ratios across reports or treat the
earlier migration's latency caveat as closed by this experiment. Neither
optional MLAS timing, x86 execution, CUDA correctness/performance, nor full-model
token throughput is established here.

All 54 measured full outputs and row argmaxes agree exactly within their
operator/shape. Declared bytes equal actual funded pool bytes and are unchanged
between arms. Every process asserts zero ledger usage after teardown/trim.
Absolute RSS is retained without differencing; no RSS-saving claim follows.
The f32 proof/clearing behavior is independently pinned by regressions covering
lazy dense dispatch, repeated initialization, pool-slot suffix preservation,
new-dispatch proof reset, invalid bindings, poisoned scratch, K=0 and empty
outputs. Existing warmed scratch tests still prove zero issuer claim/release
callbacks, including zero-byte callbacks. Owned intermediate/output allocations
and dense resident-set admission remain outside this scratch-only change.

Native CPU 2,009 / session 292 tests, optional MLAS CPU 2,051 / session 294 tests,
and all 48 public executor integration tests pass. Strict all-target native
and MLAS Clippy pass. These are correctness checks, separate from timing.

Each executed driver resolves the recorded candidate's `HEAD^` as baseline.
To reproduce, adapt the absolute source, ORT-cache, target and probe paths,
retain fresh provenance, and hold the shared lock throughout the phase. Raw
logs, full symbol audits, source backups, `.f32` outputs and process snapshots
remain under `/tmp/nxrt-initialized-scratch-perf` and
`/tmp/nxrt-initialized-scratch-matmul-perf`. Consumed executables may be removed
after hash verification; exact sources, raw evidence and the stable lock remain.

All 12 required real-ORT shared-EP end-to-end tests also pass with cached
ORT 1.32 and the current rebuilt plugin. Missing ORT is a failure in this run.
