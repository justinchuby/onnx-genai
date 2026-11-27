# Native Session workspace metadata memo: controlled comparison

All six timing cells are **diagnostic**. None cleared the frozen whole-host
foreign-CPU gate, so this cohort publishes no latency ratio, speedup, slowdown
or zero-overhead claim for #2810. Every other frozen control passed. The
foreign-CPU gate is conservative whole-machine activity, not proof of contention
on the child's particular core; the sampler cannot identify that core.

## What ran

- Baseline runtime `0b63647d5ad874d5fa2b7157c600d76ec660bafd` versus
  metadata-memo runtime `91752cf4639139d947cc73ee0c5f514f40a0dda8`.
- Apple M1 Max, 10 exposed logical CPUs, one explicit Rayon/decode thread;
  optimized default full CPU EP, no MLAS. The nested probe repeats the production
  CPU package's single-codegen-unit override, uses resolver 2 and dependency
  versions from the pinned runtime lock.
- Both binaries were built at the same manifest, source and target paths.
  Only the five production files changed by #2810 were switched. All other
  changed runtime paths are `cfg(test)` files and are excluded from these
  non-test release builds. Candidate source is backed up and restored in
  `finally`; restoration and every runtime-file hash are recorded.
- Baseline executable SHA256
  `c23c8b81c8f33f8aa66b30b7a7ecdf93896201061d481ab1da6495f45a0dc448`;
  candidate
  `aadd2232d11e58e059a791e21a0cd08ec07a3360bde76564c1ea88efae1698b4`.
  The null label executes the **identical baseline binary**. Cargo artifacts,
  symbol audit and actual dynamic dependencies are archived with the build.
- Public `InferenceSession::run`, or public prepare plus run, including
  destruction of all returned outputs. Optimization is disabled; the observed
  prebound-hit delta proves each timed request dispatches every declared node.
  This is a synthetic Session cost study, not a model throughput measurement.
- Q8_0 diagonal weights have scale one. Inputs keep a 127 anchor while each
  non-anchor lane gains one per node. All activation blocks remain exactly
  quantizable through twelve nodes. A coordinate integer oracle checks full
  outputs, independently recomputed in Python. Relu uses the same seeded input.
- Timed outputs are black-boxed and dropped inside the window. Each window
  performs an **additional untimed request** for complete-output validation;
  this is not a claim that the destroyed timed output itself was inspected.

## Frozen protocol and outcome

The [protocol](protocol.json), probe and controller were committed before
measurement. Twelve excluded calibration processes choose each case's common
call count from the maximum baseline/candidate 150-ms estimates. A fixed seeded
Latin label schedule runs three fresh processes per label per case: 54 formal
processes, 972 windows. Each process has nine single-call and nine double-call
windows, alternating pair order. All samples and failures are retained; no
measured request was rerun based on its result.

All 54 processes exceeded the predeclared maximum of 0.5 other busy cores,
measured as complete-process average over the whole exposed machine. Observed
values span **1.056–3.309 cores**. The minimum timed CPU window was 144.262 ms.
All CPU/wall, within-process A/A, double-call, identical-binary null, process
spread and whole-process setup-fraction controls passed. Failing the foreign
activity gate still makes every comparison diagnostic; those other controls do
not retrospectively waive it.

| Workload | Nodes | Rows | API | Other busy cores, range | Managed workspace bytes | Status |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Relu | 12 | 1 | run | 1.684–2.776 | 0 | diagnostic |
| Q8 projection | 1 | 1 | run | 1.150–2.531 | 164 | diagnostic |
| Q8 projection | 3 | 1 | run | 1.056–2.771 | 164 | diagnostic |
| Q8 projection | 12 | 1 | run | 1.239–3.309 | 164 | diagnostic |
| Q8 projection | 3 | 8 | run | 1.220–1.803 | 2,320 | diagnostic |
| Q8 projection | 3 | 1 | prepare + run | 1.200–2.538 | 164 | diagnostic |

The sampler reuses the audited uncached Mach `host_processor_info` reader.
It records raw tick vectors and subtracts each complete child's `wait4` CPU.
The stable shared `/tmp/onnx-genai-hostlock-darwin.flock` covers the complete
cohort and is inherited by children. There is no CPU affinity and no
window-level foreign-CPU bound. A sandbox denied `sysctl` before any cohort
or measured child existed; sampling then used host-counter read permission.
That preflight was not a discarded measurement.

## Deterministic evidence and limits

All 1,080 formal complete-output validation requests passed. Timed windows
contained 34,675,614 public Session calls and 126,815,868 observed node dispatches.
These are **experiment-wide totals**, not one inference's workspace calls and
not counts of quota requests. Every warm window had zero issuer callbacks,
including zero-byte callbacks; an explicit zero-byte positive control verifies
the observer. Each Q8 process retained the same actual workspace identity,
then settled to zero after teardown and trim. Relu had no workspace allocation.

Q8 rows=1 requires 164 bytes for one, three and twelve serial projections;
rows=8 requires 2,320 bytes. The 64 MiB shared workspace/derived authority is
not a process-memory limit: ordinary tensors and metadata remain outside this
scope. Absolute process high-water RSS is retained, never differenced into a
memory-saving claim. Both arms use the same pool policy; this study does not
isolate nanoseconds per pool lock or measure multi-session contention.

The functional #2810 regression separately proves three cold metadata queries
and zero repeated queries across 64 warm three-node runs, while preserving 192
live refinements. Avoided queries and zero quota callbacks do not establish a
latency improvement. This cohort cannot resolve that remaining question.

## Verify and reproduce

The [raw archive](results/raw-results.tar.gz) contains all build receipts,
compressed runtime source projections, the exact measured probe/controllers,
calibrations, schedule, process outputs, raw host counters and unmodified summary.
See [facts](results/facts.json), [summary](results/summary.json) and
[verification](results/verification.json). The verifier reconstructs outputs,
dispatch counts, host arithmetic, schedule and every gate. Three checksum- and
summary-repaired semantic corruptions are rejected. `audit_sources.py` also
compares archived production bytes with the immutable runtime commits.

Unpack the archive in a fresh directory and run:

```sh
python3 docs/benchmarks/native-workspace-memo/verify.py /tmp/unpacked-results
python3 docs/benchmarks/native-workspace-memo/audit_sources.py /tmp/unpacked-results
python3 docs/benchmarks/native-workspace-memo/mutations.py /tmp/unpacked-results
```

The measured build/controller snapshots are under `source-snapshot/`; a build
commit and a later environment-sanitization commit are separately identified in
the receipts. Reproduction requires the pinned candidate runtime checkout with
this study directory copied into it, the frozen dependency lock and the same
compiler configuration. The pinned repository supplies its ORT API headers;
this native probe needs no ORT installation or runtime download. `build.py`
rejects unrelated runtime edits. Use fresh
output directories:

```sh
python3 docs/benchmarks/native-workspace-memo/build.py /tmp/new-memo-target /tmp/new-memo-build
python3 docs/benchmarks/native-workspace-memo/run.py /tmp/new-memo-build/build.json /tmp/new-memo-results
```

On a restricted sandbox, sampling needs permission to read `sysctl` and Mach
host counters. The stable lock file must already exist; neither script replaces
it. Absolute recorded paths may differ on another checkout, and new binaries
need not be byte-identical across paths/toolchains. Every new run must qualify
its own host conditions. No CUDA execution or quality extrapolation follows.
