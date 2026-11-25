# Block projection execution-context adoption record

Adopt the execution-context correction for CPU BlockQuantizedMatMul and
BlockQuantizedMoE: construction must not freeze the logical Rayon shard count.
Preparation selects the actual execution pool, and launch consumes that selection
once for both workspace layout and compute. Ready cache hits retain their bound
without another metadata query. One coarse governed workspace remains the owner;
there is no quota request/refund per row, tile, shard or expert.

This is a correctness and parallelism-preservation decision, not a speedup or
zero-overhead claim. No CUDA execution was available.

## Evidence and limits

The [initial 240-process study](../block-projection-execution-context/README.md)
measured both CPU and wall time. It included qualified regressions (about 4.4%
CPU for singleton MoE and 8.1% CPU for MatMul construction4/execution1), noisy
cells and greater CPU work when restoring construction1/execution4 parallelism.
Its rejection language describes that first candidate and remains unchanged.
The [90-process shared-image controls and final functional validation](../block-projection-context-controls/README.md)
retain those results; the MatMul4→1 temporal A/A control failed there, so its
ratio is inconclusive. Those historical status statements predate this decision.
No sample from either study has been removed or relabeled.

The new diagnostic isolates warm **wall time** with one executable, shared input
Tensor owners, and exactly one preadmitted 16 KiB workspace allocation. It uses
the final Ready-cache candidate. Three processes contain 72 windows, with fresh
Sessions constructed and warmed five times outside each timer. Each window runs
8192 strict prepare/run/output-drop calls (16384 for doubled-work controls),
using MXFP4 MatMul M=8, K=128, N=256. Normal order is ABBA BAAB, reversed in odd
repetitions; A is constructor-frozen and B selects the actual execution pool.
Weights are shared graph inputs, not duplicated initializers.

| Construction → execution pool | Median paired candidate/baseline wall | Baseline / candidate temporal A/A | Adjusted doubled work | Qualified |
| --- | --- | --- | --- | --- |
| 1 → 1 | 0.992296 | 0.955718 / 1.008713 | 1.930639 / 1.965972 | yes |
| 4 → 1 | 0.998231 | 1.005929 / 1.011148 | 2.052530 / 1.942127 | yes |

Both cells pass the predeclared ±5% temporal A/A, ≥50ms work, ≤1% empty overhead,
and [1.7,2.3] adjusted doubled-work controls. These samples do not reproduce a
stable warm wall regression in the two selected scenarios. They do **not**
attribute the earlier CPU differences, measure per-arm CPU, establish statistical
equivalence, cover MoE or other shapes, measure end-to-end throughput, or qualify
construction1/execution4 latency. Process CPU receipts include construction,
warmup and teardown and cannot be assigned to either arm. All three final output
buffers are finite, nonzero and byte-identical to the original MatMul fixture.

The probe actively asserts workspace allocation identity and charge before/after
every window and Session drop, unchanged factory counters during timed work,
output byte equality, and zero retained charge after final trim. Its untimed
trace assertion witnesses the streamed variant; this study does not archive raw
per-window traces. Earlier studies do archive their raw traces. The cold factory
selector is thread-local and exists only in the temporary diagnostic source;
it is absent from the production tree. No production API was added.

The first reader incorrectly required even zero-call empty timers to be positive;
some were below one clock tick. Only empty windows now allow zero. Negative empty
and zero timed-work values remain rejected. `reader-correction.json` records the
correction; the benchmark ran once, with no sample rerun. Eleven malformed-input
controls are rejected and both intentionally failed A/A controls remain unqualified.

## Functional validation and provenance

The source seal matches the previously completed final-candidate validation:
native CPU 2068 passed (21 ignored), MLAS CPU 2110 (37 ignored), decoded layout15,
Session376, plugin workspace cache19, host block oracles2 and QMoE oracle1.
Strict all-target CPU/CPU-plugin/Session Clippy passed in native and MLAS modes.
The source remained unchanged during this study; the temporary instrumentation
was restored byte-for-byte. Tests cover pool changes, exact bounds, Ready query
elision, missing/consumed/failed preparation, original shape/format guards and
workspace issuer-count behavior. Existing GPU evidence is compile/list only.

`metadata.json` records executable hashes before and after the run, the original
source hash and probe hash. The executable is not committed. `instrumented.rs.gz`
is diagnostic-only (including the old factory copy), not an API or second
production implementation. Build logs, complete stdout/stderr, final buffers,
raw receipts, protocol, readers and rejected-reader controls are retained here.
Drivers retain original local paths for provenance and require path adaptation
and rebuilding the diagnostic executable to reproduce timings. Do not run them
concurrently with other host performance work; use the shared host flock.

Run `python3 -B docs/benchmarks/block-projection-paired-owner/verify.py .` from the
repository root to verify the sealed evidence and current production sources.
