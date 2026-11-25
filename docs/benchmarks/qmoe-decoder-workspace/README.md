# QMoE decode scratch shares the prepared expert lane

Integer-affine QMoE now puts decoded row tiles and column-major projection
staging in a fourth region of its existing StepScoped FC/contribution workspace.
Serial FC1, FC2 and effective separate FC3 reuse the maximum **complete**
projection layout, rather than summing independently maximal components. Tile1,
serial GEMM arithmetic, route order and publication-after-success are preserved.
Dense paths do not initialize this region; the first streamed projection clears
it once per dispatch. No per-row/tile/expert workspace quota operation is added.

Capacity depends on the selected Rayon width. The provider's allocation-free
context reader supplies the cache key; native complete-preparation memo and both
ORT metadata caches compare it. ORT captures the reader cold and retains its
provider, without acquiring a Shared-provider mutex per Compute. External
callers use a stable lazy-global class; workers use their actual width. Equal
widths have equal capacity. The execution refiner independently selects and stores
the actual plan before admission, and launch consumes it. Failed refinement
clears a prior plan; missing/consumed plans fail before publishing output.

A constructor-frozen plan was rejected because construction1→execution4 can
cap parallelism at one shard. Its functional results and source patch remain
under diagnostic/; it was never benchmarked and has no latency claim.

The admitted retained footprint grows because decode buffers were previously
allocated outside this lane. This is canonical accounting and ownership reuse,
not a lower-RSS or full-process-cap claim. Routing, widening, caller tensors and
transient weight admission remain outside the workspace bound. Poisoned-backing
tests cover dense untouched regions, streamed reuse, short spans, guards, all
integer bit widths, zero axes, aliases and IEEE route order. A preadmitted peak
survives both cross-pool directions with one owner and zero warm workspace
loan/return issuer calls. Positive transient weight requests still reach the
same authority, decline when full and follow their original protocol.

## Measurement

Apple M1 Max,10 logical CPUs,32GiB, optimized native CPU builds without MLAS
symbols. Baseline is merged #2655 (`3e8c8e776686a3c2489b4290c98a906e4be7e77e`).
Both executables were freshly built with the same new public-session probe and
Cargo.lock. The construction pool builds the session and prepares once; the
execution pool prepares, warms and runs. Two new cases deliberately cross1→4
and4→1. Initial logical demand, execution logical demand and actual retained
physical peak are separate receipts. Replaced idle backing is trimmed before
setting the streamed case's shared limit to its own binary's physical peak;
dense cases allow64MiB. Actual final traces must witness the requested path.

One stable Darwin flock covers both builds and the whole matrix, coordinated
with the other chat. No per-core foreign-work monitor is available and no idle
host is claimed. Parent terminal wait4 records complete child user+system CPU,
wall and maximum RSS, including setup, five warm calls, repeated strict
prepare/run/output drop, final traced dispatch, validation/file IO and retirement.
This is whole-process cost, not isolated kernel latency or model throughput.
There is no short-window CPU clock and no GPU execution.

192 processes:3 repetitions*8 cases*(baseline/candidate/null setup+work and
baseline/candidate double work). Normal calls=floor(268435456/(R*H*I*2*P)),
where P=2 or3 projections. Setup has zero repeated calls; double has exactly2x.
Order rotates/reverses. Predeclared CPU **and wall** qualification independently
requires baseline/null A/A within5%, setup<=25%, setup-adjusted double in[1.7,2.3]
and work>=50ms; CPU must also obey wall*11. Independently derived capacities,
retained funding, trim0, observed path and finite nonzero complete outputs are
checked. Reader controls accept a complete synthetic fixture, reject13 malformed
variants and keep intentionally noisy CPU/wall A/A cells unqualified. Every
sample is retained; none was replaced.

## Results

Ratios below are medians of three paired candidate/baseline whole-process ratios.
All192 complete outputs and per-row argmax values are bit-identical across arms;
matching streamed shapes also agree across the selected pool widths.

| R / H / I | Construction→execution | Path / gate | CPU ratio | Wall ratio | CPU / wall qualified | Logical workspace baseline→candidate |
|---|---|---|---:|---:|---|---|
|1 /32 /64|1→1|streamed /none|0.9943|0.9951|yes /yes|640→896|
|1 /64 /128|1→1|dense /none|1.0045|1.0043|yes /yes|1280→1792|
|1 /64 /128|1→1|streamed /separate|1.0003|0.9999|yes /yes|2048→2560|
|8 /128 /256|1→1|streamed /none|0.9880|0.9878|yes /yes|20480→29184|
|8 /128 /256|4→4|streamed /separate|0.9917|0.9950|yes /yes|32768→43008|
|8 /128 /256|4→4|dense /separate|0.9910*|0.9979|**no** /yes|32768→43008|
|8 /128 /256|1→4|streamed /separate|0.9893|0.9707|yes /yes|32768→43008|
|8 /128 /256|4→1|streamed /separate|1.0058|1.0058|yes /yes|32768→41472|

For4→1 the candidate actually retains43008 bytes from the wider initial
preparation;41472 is the narrower logical demand. The streamed cap uses43008.

*Dense4→4 CPU A/A=0.9337 exceeds the5% bar. Its CPU ratio is descriptive only;
CPU attribution for that cell is **inconclusive**. Wall A/A=0.9581 qualifies.
Thus CPU qualifies7/8 and wall8/8. Both cross-pool directions qualify both
metrics. All paired ranges remain in results.json; aggregate qualification does
not make every repetition quiet. For example streamed4→4 has one wall pair1.0710,
and dense4→4 has one CPU pair0.9334. These observations establish no stable
speedup, universal zero overhead, CUDA behavior or full-model result; a small
cost remains possible. No persistent regression is established in qualified
cells.

## Validation and provenance

Native CPU2066 passed/21 ignored; MLAS2108/37; ep-api121; decoded-layout15;
Session374; plugin399; strict all-target Clippy for API/CPU/plugin/Session in
native and MLAS configurations; host-only CUDA-package CPU oracle1 passed.
The production source prefix was unchanged by two final test-only corrections:
remove a needless borrow, and count actual declared TLS resolutions rather than
mistaking a cold provider `.with` for TLS. A focused lifecycle test and both
Clippy configurations cover those final sources. Their prefix hashes, initial
failure logs and subsequent success logs are retained. A source-seal diagnostic
caught an architecture-document edit during checks; all Rust hashes matched,
and the seal was corrected to explicitly cover runtime crates without rerunning
or discarding a test. No GPU tests were executed.

The manifest hashes all payloads. study/ contains the exact source snapshots,
patch, probe, manifest/lock, build logs, binary hashes,192 terminal receipts,
complete output bytes, stdout/stderr and dispatch events. validation/ keeps
commands, logs and source receipts; diagnostic/ preserves the rejected design.
Executable bytes stay outside the repository. Historical absolute runner paths
are provenance, not a portable build recipe. Run `python3 verify.py` here;
optionally pass the measured source checkout root to check source identity.
