# Q8 projection loop-order study

Production keeps its existing row-first recipe. This study compares one test-only
candidate against the actual shared Q8 projection, with activation conversion,
fresh checked test scratch binding, integer dots, and transpose inside each
synchronous timing window. Routing, operator preparation/final publication,
governed checkout/return, model execution, and CUDA are outside this boundary.
Scratch backing and output vectors are allocated once before timing; the
operator's separate final-output-bank initialization is excluded. This study
does not measure or claim that an entire projection/operator is allocation-free.

The complete measured source snapshot is
[`362580b`](https://github.com/justinchuby/onnx-genai/commit/362580bc8f627e1f5b0aa1b12070dd302c72aa59).
Use that snapshot and the recorded source hashes for measurement reproduction.

The production base is
[`bb8ec2e`](https://github.com/justinchuby/onnx-genai/commit/bb8ec2e887ba7f7e13296236c698e329cba94f7a).
The only change to the production module is a `cfg(test)` study declaration.
The [study module](../../../crates/onnx-runtime-ep-cpu/src/kernels/packed_projection/loop_order_study.rs)
contains the candidate. Its recipe matches the production one except that it
visits feature/block/row instead of feature/row/block, explicitly clears each
feature's accumulators, and adds bias once after all blocks. Each row retains
its original ascending block accumulation order and f32 expression; no FMA or
cross-block reduction is introduced. Weight scale conversion/checking can be
shared in the source loop, at the cost of repeated column accumulator accesses.
Neither source operation count establishes machine instruction count or speed.

The ordinary correctness test covers Q4_0/Q8_0, R=0/1/3/8/32,
K=0/1/31/32/33/1003/1024, N=0/1/7, absent/negative-zero/dyadic bias,
dirty scratch reuse and non-finite weight scales. Its independent element-wise
f64 oracle uses small signed codes and power-of-two scales with exactly
representable f32 prefix sums. Complete output bits must match for these
fixtures; arbitrary half-scale mantissas need separate rounding stress before
adoption. Failures may dirty private staging as before.
The Darwin-only ignored measurement entry cannot be selected in shipped builds.

## Frozen experiment

`run.py` acquires the existing stable
`/tmp/onnx-genai-hostlock-darwin.flock` without recreating it and holds the same
open file description across the entire sweep. Every child inherits its FD.
Both lock endpoint inodes are retained in each process receipt.

The native CPU release build uses the repository's one-codegen-unit policy,
without MLAS. A fresh one-worker Rayon pool is established and asserted by each
process. No CPU affinity is set. Cases are Q4/Q8 x R=1/3/8/32 x K=32/1024,
N=256. Both arms keep explicit Q8 activation arithmetic; Q4/Q8 names select
weight formats. Inputs and weights are warmed; these small fixed working sets do not
represent a model's cold weights or larger cache/memory tiers.

One calibration per case chooses a shared call count for approximately 0.2 s
from the row-first arm. Counts, source/binary hashes and a seed-20261126 balanced
schedule freeze before confirmatory measurements. Four independent processes
per case alternate ABBA and BAAB. Each process runs four paired windows,
two row A/A windows, two candidate A/A windows, one common checked-binding-only
setup window, and one double-work window for each arm. Each doubled projection
binds a fresh lane, including its normal initialization. Output hashing and
oracles are outside the timers.

A stale-executable schema smoke must observe all eleven window labels before
calibration. A named correctness test must also execute and pass. There is no
resume or favorable-cell retry: an interrupted experiment remains incomplete.

The frozen gates require every process/window in a cell to pass:

- Whole-process foreign CPU average at most 0.5 cores.
- Non-setup process CPU/wall in [0.85, 1.15].
- Each arm's A/A wall and CPU ratios in [0.95, 1.05].
- Each arm's double-work wall and CPU ratios in [1.8, 2.2].
- Common checked-binding setup wall fraction at most 0.1 for each arm.

The [previous audited Mach sampler](../packed-activation-study/run.py) is reused
rather than copied. It reads uncached per-processor ticks over each complete
child process, subtracts complete child CPU, and aggregates all logical CPUs.
That average includes fixture construction, warmup and checks; it does **not**
bound contention in each timing window or identify the worker's particular core.
A/A and efficiency checks supplement this coarse check. A qualified cell means
these stated controls passed on this shared Apple CPU, not that solo execution
or inference latency was proved.

Any failed gate makes the complete cell diagnostic. Its summary exposes no
candidate speed ratio. The analyzer preserves every failure and reports median,
range and n only for qualified cells. `verify.py` independently reconstructs all
sixteen fixture digests in Python, checks raw stdout/receipts, schedule and
cardinality, reconstructs host-counter arithmetic, and requires byte-identical
summary regeneration.

## Results

The single completed sweep ran on 2026-11-26 on Apple M1 Max (10 logical CPUs),
using native CPU release and one Rayon worker. It retained 16 calibrations, 64 confirmatory processes
and 704 windows. All output checks passed. Two of sixteen cells passed every
frozen coarse control; the other fourteen failed the foreign process-average
CPU gate and expose no speed ratios. All A/A, double-work, binding-setup and
own CPU/wall gates passed. No failed cell was rerun.

For the two accepted cells, candidate time divided by original row-first time
is below. Each has four independent process ratios; ranges retain every ratio.

| Weights | R | K | N | Wall ratio: median [min, max] | CPU ratio: median [min, max] |
| --- | ---: | ---: | ---: | ---: | ---: |
| q4 | 1 | 32 | 256 | 1.8788 [1.8755, 1.8875] | 1.8786 [1.8755, 1.8870] |
| q8 | 32 | 1024 | 256 | 0.9728 [0.9693, 0.9785] | 0.9728 [0.9694, 0.9784] |

The candidate took about 88% more time in the accepted Q4 single-row/single-block
cell, while time fell only about 2.7% in the accepted Q8 R=32,K=1024 cell.
These are conditional observations on the stated shared CPU controls, not a
solo-run or model-level regression/speedup claim. They do not support an
unconditional loop exchange. Keep the existing production recipe, without
adding a shape threshold from this sparse accepted subset. A new tiled candidate
would require its own frozen experiment, rather than selective retries here.

[Complete cell classifications](results/summary.json),
[independent verification](results/verification.json), and
[three verifier falsification controls](results/negative-controls.json) are
retained. The latter reject a forged summary, a missing process even after
repairing checksums, and consistent forged output digests after repairing raw
stdout/receipt checksums. Python reconstructs the true fixture digest to reject
the last forgery. The [manifest](results/manifest.json) covers every original raw
file inside the deterministic [archive](results/raw-results.tar.gz), including
calibrations, schedule, all process outputs and complete timing windows.

Production remains unchanged. Any future adoption needs actual
operator/publication/governed-resource measurements, larger working sets,
multiworker checks, arbitrary-scale rounding stress, and the existing
numerical/behavioral gates.

## Reproduction

From the repository root on Darwin, with the stable shared flock already
established by the cooperating sessions:

```sh
python3 docs/benchmarks/q8-loop-order/build.py /tmp/nxrt-q8-loop-study-target
```

The build prints the release test executable and writes
`/tmp/nxrt-q8-loop-study-target/loop-study-build.json`, binding it to source,
Cargo profile/features, compiler and binary SHA-256. Use that exact executable:

```sh
python3 docs/benchmarks/q8-loop-order/run.py /path/to/onnx_runtime_ep_cpu-TESTHASH \
  /tmp/new-q8-loop-results \
  --build-receipt /tmp/nxrt-q8-loop-study-target/loop-study-build.json
python3 docs/benchmarks/q8-loop-order/analyze.py /tmp/new-q8-loop-results \
  > /tmp/new-q8-loop-results/summary.json
python3 docs/benchmarks/q8-loop-order/verify.py /tmp/new-q8-loop-results
```

To verify the archived sweep without rerunning measurements:

```sh
mkdir /tmp/q8-loop-archived
tar -xzf docs/benchmarks/q8-loop-order/results/raw-results.tar.gz -C /tmp/q8-loop-archived
python3 docs/benchmarks/q8-loop-order/verify.py /tmp/q8-loop-archived
python3 docs/benchmarks/q8-loop-order/self_check.py /tmp/q8-loop-archived
```

`pack_results.py` reproduces the archive and copied views from an extracted
results directory. Every member is a flat regular synthetic data file.

`build.py` and `run.py` each acquire the shared flock themselves. Use the frozen
source revision and receipt hashes when reproducing archived results. This is
an experiment in the opt-in Q8 arithmetic; it neither changes that default nor
establishes ggml parity, model quality, token equivalence or a GPU result.
