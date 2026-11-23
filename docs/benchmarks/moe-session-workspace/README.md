# Coarse workspace for streamed CPU MoE

Synthetic operator/session validation on Apple M1 Max, macOS 27.0.1, Rust 1.98.0.
Default native builds contain zero MLAS symbols. This is not a full-model,
CUDA, x86, or optional-MLAS timing claim.

The seeded MXFP4/f32 graph has four experts, top-k two, SiLU, and distinct
FC1/FC2 banks; every row routes to the same two experts. K/N below mean
hidden/intermediate widths. Actual public `InferenceSession` strict
prepare→run includes output destruction. There is zero dense-weight retention,
an explicit governed Host workspace pool, and a 64 MiB workspace/derived cap.
The cap does not account for the complete process working set.

Baseline: `46f1321d7e7524703eaba5f8898ec29f5c741417`. Measured candidate: `56542d7d03fa07bffa5e883d16daad4e42cd5172`.
Both arms were compiled at the same source/manifest path, cleaning changed
release packages at each source switch. Source restoration was checked before
measurement. Immutable candidate sources and compiler/binary/source hashes are
retained in the raw directory. The candidate is retained as the parent of this
report-only commit. No production arithmetic changed during reporting.

Three isolated processes per label, nine batches per process, baseline/after/
identical-baseline-null order rotated across processes. Excluded pilots selected
common iteration counts targeting 100 ms of process CPU time per batch. The
shortest measured inference/control batches were 97.504/77.976 ms.
The unchanged Relu control used nine batches of 100,000 calls. All cells passed
the prespecified minimum-duration and ±5% control-median guards.

The shared stable `/tmp/onnx-genai-hostlock-darwin.flock` covered the complete
build phase and, separately, the pilot/sweep phase. The initial driver stopped
before any pilot because the sandbox refused its supplementary `ps` snapshot.
`measured-resume.py` verified the existing binary hashes and clean source before
running the first pilot and the one measured sweep, with permission for the
snapshot. No result-driven reruns or rebuilt arms were substituted. Snapshots
contain process names/CPU usage, not full command lines. Darwin does not supply
a per-core foreign-utilization record here; the flock, process CPU/wall times
and controls limit the inference that can be drawn from a shared host.

| M / K / N / threads | CPU µs/call before → after | CPU change | Wall µs/call before → after | Funded scratch B |
| --- | --- | --- | --- | --- |
| 1 / 32 / 32 / 1 | 8.163 → 8.161 | -0.03% | 8.164 → 8.163 | 512 |
| 1 / 256 / 512 / 1 | 946.264 → 948.840 | +0.27% | 946.284 → 948.858 | 8192 |
| 4 / 256 / 512 / 1 | 986.922 → 992.569 | +0.57% | 986.960 → 992.635 | 12288 |
| 4 / 256 / 512 / 4 | 1248.679 → 1246.037 | -0.21% | 413.475 → 422.904 | 36864 |

These are medians of process medians; full ranges and raw batches are in the
JSON files. CPU medians differ by −0.21% to +0.57%. The M=4/one-thread candidate
CPU range is above baseline, but the identical-baseline null also shifts by
+0.66% and overlaps the candidate. The four-thread wall median rises by 2.28%
with nonoverlapping process-median ranges, while its unchanged control rises
3.01%; this does not rule out a small latency cost. These results do not justify
a universal zero-overhead or speedup claim. They show similar process CPU cost
on the measured shapes, with the latency caveat preserved.

All 36 measured full outputs and per-row argmaxes agree exactly within their
shape. Candidate declared bytes equal actual funded pool bytes. Every process
asserts zero usage after session teardown and idle-pool trim. Baseline scratch
was call-owned and unaccounted by this workspace ledger; zero baseline funding
is not zero physical scratch. Absolute RSS includes setup/inference and is
retained without differencing; no RSS-saving claim follows.

Separate recording-issuer regressions prove warm scratch checkout, computation,
and return invoke neither claim nor release, including zero-byte callbacks.
They cover top-k skew, FC3/fused gates and a larger execution pool than the fixed
logical shard plan. This is a structural quota-path guarantee, independent of
the timing results. Routing, grouped input, activation and owned intermediate
results remain outside this scratch migration; dense resident-set admission is
still its existing separate cache operation.

The exact executed probe/build driver/resume sources are beside this report.
`measured-driver.py` resolves the recorded candidate's `HEAD^` as baseline;
for a new experiment adapt the absolute checkout/target/ORT paths and retain new
provenance. Raw logs, full symbol audits, source backups, `.f32` outputs and
supplementary snapshots remain in `/tmp/nxrt-moe-workspace-perf`. Consumed
executables may be removed after hash verification.
