# Session workspace comparison on Apple M1 Max

The new resource path funds streamed BlockQuantizedMatMul scratch through the
existing Host workspace pool and plans it through the selected EP declaration.
This experiment checks the complete public `InferenceSession` strict
prepare→run path, including output destruction, against the immediately prior
implementation. It does not measure a complete language model.

All 36 measured process runs produced identical full output bytes and row
argmaxes. All four cells meet the prespecified minimum-duration and Relu-control
median guards. Their before/after CPU ranges overlap: this measurement does not
distinguish a regression. It establishes neither a speedup nor universal zero
inference overhead. The separate recording-governor regression proves warm
workspace borrow/compute/return makes zero issuer callbacks, including zero-byte
ones; the timing experiment measures what that mechanism alone cannot prove.

## Conditions and controls

- Apple M1 Max, macOS 27.0.1, ARM64; Rust 1.98.0 / LLVM 22.1.8.
- Baseline `bb1b14cf3f564e7b41ac6b5cc30df19a77b0105c`; measured candidate
  `a9c1cbe03351750b9739ca721592edd4fbc0e7da`. Both default native builds;
  each audited binary has more than 21,000 symbols, `_main` is present, and
  neither has MLAS symbols. The null label runs the **identical baseline binary**.
- Both builds use the same manifest/source path. Changed API/CPU/session release
  packages are cleaned at every source switch. Candidate files are backed up
  persistently and restored in `finally`; the final source tree was clean.
  The only temporary manifest addition is the same `libc=0.2` dev dependency
  for `getrusage` in both arms; it and the lockfile are restored afterward.
- Explicit CPU provider, governed Host workspace pool, zero dense/expert cache
  retention, 64 MiB shared workspace/derived-weight limit. This is **not a
  complete process-memory limit**; final output tensors and other ordinary
  allocations remain separate. After session teardown and pool trim, funded
  Host usage is zero in every run.
- Seeded finite MXFP4 weights and f32 activations; optimization disabled. Each
  process fixes the actual caller Rayon pool to one or four threads, warms for
  100 ms process CPU time, then runs nine batches. Common repetitions per cell
  are chosen by excluded baseline pilots targeting 100 ms CPU per batch.
- Three isolated processes per label per cell. Label order rotates as
  baseline/after/null, after/null/baseline, null/baseline/after. The unchanged
  direct Relu kernel is warmed and measured in nine batches of 100,000 calls.
- The complete build/pilot/sweep holds the same stable
  `/tmp/onnx-genai-hostlock-darwin.flock` used by the llama.cpp chat. Per-run
  process snapshots are supplementary context, not an idle-host assertion.
  Darwin per-core foreign utilization is not available here; process CPU/wall
  samples, the Relu control and identical-binary null are retained.
- Prespecified gates: each inference and control batch must have at least
  20 ms process CPU time, and after/baseline plus null/baseline control medians
  must lie in [0.95, 1.05]. All cells pass. Minimum measured inference batch:
  96.328 ms; minimum control batch: 75.931 ms. Pilots are excluded.

## Results

Entries are median CPU time/call across three process medians, followed by the
full range of those process medians. CPU sums all threads; it is not elapsed
latency. Complete wall-time ranges are in `summary.json`.

| M / K / N / threads | Unit | Before [range] | After [range] | Same-binary null [range] | After funded scratch bytes |
| --- | --- | --- | --- | --- | --- |
| 1 / 32 / 32 / 1 | µs | 3.4997 [3.4951, 3.6403] | 3.5317 [3.4887, 3.5488] | 3.4905 [3.4849, 3.5343] | 512 |
| 1 / 1024 / 2048 / 1 | ms | 3.7144 [3.6599, 3.8205] | 3.7148 [3.6564, 3.7250] | 3.7097 [3.6633, 3.7158] | 16,384 |
| 4 / 1024 / 2048 / 1 | ms | 3.7827 [3.7270, 3.7866] | 3.7633 [3.7227, 3.7889] | 3.7811 [3.7674, 3.7934] | 49,152 |
| 4 / 1024 / 2048 / 4 | ms | 4.1243 [4.0476, 4.2225] | 4.0486 [4.0206, 4.0778] | 4.1132 [4.0720, 4.1300] | 98,304 |

The tiny cell's point estimate changes by about +0.9% (about 32 ns); the larger
cells change by approximately +0.01%, −0.5%, and −1.8%. Each is within the
observed ranges, so none is promoted to a performance claim. The four-thread
cell's elapsed medians are 1.062→1.045 ms; the same qualification applies.
Control median ratios span 0.977–1.037, and identical-binary inference null
ratios span 0.9973–0.9996.

Before and null report zero governed workspace, because the old scratch is
call-owned. Candidate requirements and actual funded backing agree: 512,
16,384, 49,152 and 98,304 bytes. This demonstrates newly accounted scratch,
not an RSS saving. Absolute `getrusage` high-water RSS includes setup and the
inference phase, excludes later output verification and control work, and is
never differenced; raw values and ranges remain in the JSON artifacts.

## Inspect and reproduce

`measured-session.rs` and `measured-driver.py` are the exact executed sources,
not a rewritten benchmark. The driver was run from a private worktree at the
measured candidate revision and copies both binaries before restoring source.
It contains this host's absolute checkout, ORT-cache and target paths; adapt
those and the shared-lock wrapper for a different host, then retain new
provenance rather than mixing its results with this report. At the recorded
candidate revision the driver resolves `HEAD^` as the baseline. Both source
revisions are retained in this PR's history.

`provenance.json` records compiler identity, production-source and probe hashes,
source path, binary hashes, and symbol-audit counts. `runs.json` preserves every
measured batch, output hash, row argmax and funding observation; `summary.json`
contains the unrounded summaries and guard results. Raw stdout/stderr, full
symbol audits, host snapshots and `.f32` outputs remain under
`/tmp/nxrt-workspace-perf` on the measurement host. Build logs and excluded pilots
are also retained. Generated probe binaries may be removed after their hashes
and symbol audits are recorded.

This CPU-only validation does not establish CUDA device correctness, x86 timing,
full-model token throughput, or the complete lifetime support of opaque foreign
plugins. In particular, the ORT adapter continues to decline SessionPersistent
scratch rather than supply a short-lived buffer.
