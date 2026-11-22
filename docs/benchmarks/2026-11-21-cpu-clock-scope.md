# CPU accounting boundaries on Apple M1 Max

This standalone diagnostic investigates the unresolved CPU A/A variation in the
[SiLU repair experiment](https://github.com/justinchuby/onnx-genai/blob/658e6b165b5851091e4a3a08388b1390169af829/docs/benchmarks/2026-11-21-silu-density-repair.md).
It does not benchmark inference, change runtime instrumentation, or establish the
cause of every SiLU sample. Switching from `getrusage(RUSAGE_SELF)` to
`clock_gettime(CLOCK_PROCESS_CPUTIME_ID)` alone is not supported as a fix: both
process APIs exhibit almost identical A/A excursions here. Moving the resource
boundary through worker quiescence substantially reduces the largest excursions,
but does not make every cell repeatable within 5%.

## Scope and protocol

Apple M1 Max, 10 physical/logical CPUs, 32 GiB RAM, macOS 27.0.1 (26A434), Apple
clang 21.0.0, `-std=c11 -O3 -Wall -Wextra -Werror -pthread`. Source at
`5d270bbc5c5307ab219a4e11d5d2e792d309db7c`; source, runner, and executable SHA-256
are recorded in [metadata](2026-11-21-cpu-clock-scope/metadata.json). The subsequent
runner edit retains the host-lock descriptor in children across parent death and
improves malformed-input errors; the measured C source is unchanged.

The runner holds the stable Darwin advisory lock
`/tmp/onnx-genai-hostlock-darwin.flock` over the whole sweep. It records each
child's PID, exit status, `wait4` CPU/wall/RSS, load averages, and clock metadata.
All nine children exited zero; their total elapsed time was 77.432 s and CPU time
401.100 s. This is a shared desktop without affinity or foreign-CPU attribution;
the lock/load receipts do not prove exclusive use. Linux `hostlock.sh` admission,
Linux host monitoring, CUDA, other CPUs, and inference workloads were not tested.

The persistent pthread pool uses 1/4/8 **workers plus an actively waiting
coordinator**. Worker count therefore is not total process concurrency. Each
worker performs deterministic xorshift work for 16384/131072/1048576 iterations.
The matrix varies batch 1/16, immediate park or a 1 ms active tail, and optional
per-thread CPU observation. The tail can be interrupted by another job in a
batch. Thread-clock observations perturb execution and are explicitly ablated.
The observed worker sum brackets useful work and its clock bookkeeping; main
thread CPU separately brackets coordination. Neither is a nonintrusive oracle
for total process CPU.

Each cell contains two identical A/A arms, 21 samples per arm, alternating order
by repetition. Three sweeps rotate worker-count order. There are **9072 samples,
216 A/A cells, and nine idle controls**, with no discarded rows. A generation
check rejects stale completion; checksums must agree across arms, spin modes,
and observation modes for the same work. A mutex/condition barrier verifies all
workers have entered their wait before the next sample starts.

Two boundaries are retained:

- `wall_ns`: submission through result readiness. `getrusage_ns` and `process_ns`
  enclose this latency window, with the process clock outermost.
- `*_drained_ns`: the same start through all workers parking, including the
  post-result active tail. These are resource-accounting windows, not request
  latency. Waiting for quiescence adds real work and can also change what a CPU
  snapshot reports; this experiment does not isolate those two contributions.

The process clock reports **1000 ns resolution**, thread clock 42 ns; the
completion atomic reports lock-free. Nanosecond units do not imply nanosecond
process-clock resolution or accuracy. `getrusage` uses microsecond fields.

## Results

Ratios below are **median(A0) / median(A1)** per cell, not paired-sample ratios.
The 5% band is a necessary repeatability check, not a certificate of clock
accuracy. Each row spans 72 cells, including every spin/observation/batch case.

| Workers | Result-ready process CPU A/A range | Cells outside 0.95–1.05 | Drained process CPU A/A range | Cells outside 0.95–1.05 |
|---|---:|---:|---:|---:|
| 1 | 0.931–1.079 | 5/72 | 0.931–1.077 | 4/72 |
| 4 | 0.977–1.035 | 0/72 | 0.9504–1.021 | 0/72 |
| 8 | 0.435–4.073 | 9/72 | 0.918–1.077 | 3/72 |

`getrusage` has the same per-row failure counts, with closely matching ranges.
The clearest boundary example is sweep 3, workers 8, spin enabled, observation
disabled, 1048576 iterations, batch 1:

| Median | A0 | A1 | A0/A1 |
|---|---:|---:|---:|
| Result latency | 3.126 ms | 3.083 ms | 1.014 |
| Result-ready `getrusage` | 24.657 ms | 6.053 ms | 4.074 |
| Result-ready process clock | 24.658 ms | 6.054 ms | 4.073 |
| Drained process clock | 28.809 ms | 28.739 ms | 1.0024 |

A different cell demonstrates remaining real coordination variation: sweep 3,
workers 8, spin enabled, observation enabled, 16384 iterations, batch 16. Worker
CPU medians are 4.444/4.453 ms (ratio 0.998), while coordinator CPU is
2.829/2.279 ms (1.241), result-ready process CPU 18.448/15.655 ms (1.178), and
drained process CPU 25.501/23.668 ms (1.077). Useful-work repeatability alone does
not establish total-process repeatability. Larger batches are not a general fix.

Across individual eight-worker samples, median `getrusage`/process-clock agreement
is 0.9997 parked and 0.99993 spinning at result readiness. Individual ratios reach
0.755 during spinning: the two API calls have nested, nonidentical intervals, so
this is not an accuracy comparison against an external reference.

For the 4536 observed samples, process CPU never falls below worker CPU plus
coordinator CPU. Immediate `getrusage` falls below that sum twice, by only
83/167 ns, less than its microsecond field resolution. Neither drained counter
has such a shortfall. These checks are diagnostics, not assertions that thread
clocks certify process accounting. Idle controls slept 26.991–30.049 ms and
charged 12–33 microseconds of process CPU; no idle samples were removed.

## Consequence for MLAS retirement

Keep the earlier SiLU wall-latency evidence separate from CPU-resource claims.
Do not swap CPU APIs and call the measurement problem solved. Before using CPU
cost to graduate another native kernel, define whether its contract measures
request latency, useful worker computation, or pool cost through quiescence;
retain both result-ready and drained observations when investigating active
pool tails, and require workload-specific A/A controls. Repeating this diagnostic
with a different API does not certify an inference benchmark.

## Reproduction and validation

```sh
python scripts/bench_cpu_clocks.py run /tmp/fresh-clock-run
python -O scripts/bench_cpu_clocks.py analyze /tmp/fresh-clock-run
python -O scripts/bench_cpu_clocks.py analyze docs/benchmarks/2026-11-21-cpu-clock-scope
```

`run` requires Darwin and a fresh directory; lock contention fails admission.
It compiles before measuring and saves all child CSVs/stderr. The committed
[compressed samples](2026-11-21-cpu-clock-scope/samples.csv.gz), metadata, and
[per-cell summaries](2026-11-21-cpu-clock-scope/results.json) suffice for independent
reanalysis. The reader verifies the raw checksum, complete case identities,
repetition/arm counts, work checksum consistency, observation ablation, nonempty
measurements, idle controls, and every saved median/ratio, including under `-O`.

Validation: strict Clang build, ThreadSanitizer at eight workers over 48 samples
across all 24 single-repetition cells, Ruff, and runner smoke at all three worker widths.
Reader controls reject corrupted CSV bytes, duplicate identities, an invalid
width with unchanged row count, changed work checksum, missing idle control, and
invalid arm. No Rust runtime code changed, so the full runtime test matrix was
not repeated. All raw anomalous samples remain available for investigation.
