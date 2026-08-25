# Session concurrency and ORT worker sharding

Date: 2026-08-25

## Scope

This benchmark measures:

- one session serialized through its exclusive lease;
- overlapping requests to the same session, which must return the typed conflict;
- concurrent requests to distinct sessions;
- stateless least-loaded routing; and
- direct decode, workflow, and server-driver overhead.

The complete W=1,2,4 by C=1,2,4,8 results are machine-readable:

- [`2026-08-25-session-concurrency-synthetic.json`](2026-08-25-session-concurrency-synthetic.json)
- [`2026-08-25-session-concurrency-ort-cpu.json`](2026-08-25-session-concurrency-ort-cpu.json)

The JSON files contain every matrix row and summary percentile. Add `--raw-samples` to the
commands when individual observations are needed; it was omitted here to keep the checked-in
artifacts compact.

## Reproduction

Base: merged N>1 main `66ca26a5d`.

Synthetic evidence was measured from clean commit `65a6902cf2ab30c1d943d80e19aa83fd111627a1`:

```bash
CARGO_TARGET_DIR=/datadisks/disk1/justinchu/relocated/cargo-target \
taskset -c 1-16 cargo run --release -p onnx-genai-bench \
  --no-default-features --features session-concurrency \
  --bin session_concurrency -- \
  --mode synthetic --workers 1,2,4 --concurrency 1,2,4,8 \
  --warmups 3 --iterations 100 --work-units 5000000 \
  --output docs/benchmarks/2026-08-25-session-concurrency-synthetic.json
```

CPU ORT evidence was measured from clean commit
`875d35133a8ccd344b3b0b83d41feb71d47f8763`:

```bash
CARGO_TARGET_DIR=/datadisks/disk1/justinchu/relocated/cargo-target \
taskset -c 1-16 cargo run --release -p onnx-genai-bench \
  --no-default-features --features session-concurrency \
  --bin session_concurrency -- \
  --mode ort --provider cpu --intra-op-threads 1 \
  --workers 1,2,4 --concurrency 1,2,4,8 \
  --warmups 10 --iterations 10000 --max-new-tokens 4 \
  --output docs/benchmarks/2026-08-25-session-concurrency-ort-cpu.json
```

Warmups are excluded. Percentiles use nearest rank. Prompts and work are deterministic:
seed `20260825`, prompt formula
`[1 + i%29, 1 + (7i+3)%29, 1 + (13i+5)%29]`. Real requests use a three-token prompt and
four generated tokens. The serialized scenario uses one-token turns and one generated token
so repeated turns remain within the fixture's 16-token context; compare speedups only within
a scenario.

## Environment

| Item | Value |
|---|---|
| Build | release, Rust 1.98.0 |
| OS | Linux 6.6.141.1-1.azl3, x86-64 |
| CPU | Intel Xeon Platinum 8480C; process affinity 1-16 |
| Host memory | 1,820.9 GiB |
| ORT | 1.29.0, API 29 |
| Providers reported by linked ORT | `CPUExecutionProvider` only |
| Model | committed `tests/fixtures/tiny-llm/model.onnx.textproto` |
| Model SHA-256 | `d70b079444e021bc6cdea4f6d732788d439deaf9f4d51beb8b84117597592438` |
| GPU snapshot | 8x NVIDIA H200, driver 580.105.08; not used by this run |
| Host lock | unprotected; CPU affinity is not host exclusivity |

The repository host-lock helper was not used, so absolute timings are observations from a
shared host rather than publishable hardware limits. CPU affinity materially reduced scheduler
variance. Each measured ORT cell lasted about 0.9-7.5 seconds, making `/proc/self/stat` CPU
accounting useful but still approximate.

Peak process RSS was 7.6 MiB in synthetic mode and 412.3 MiB in the sequential ORT process.
The ORT high-water mark includes allocator retention and cells that created 10,000 distinct
sessions, so it is not an isolated per-worker delta. The resource governor reported 8, 16, and
32 MiB host budgets for W=1,2,4. Its 0.072 MiB VRAM estimate was unchanged on CPU and is not
an observed CUDA allocation.

## Correctness gates

Both artifacts report all gates as true:

- W=1 output parity with the closest direct path;
- typed same-session conflict;
- distinct-session request overlap;
- exact completion and conflict counts; and
- zero worker, session, turn, and lease counter drift.

At C=8, every conflict row completed 10,000 real owners and rejected exactly 70,000 contenders.
The ORT conflict p99 was 0.304-0.314 microseconds across W=1,2,4. Distinct-session workers were
evenly assigned; stateless W4/C8 completions were 2476/2505/2481/2538.

`max_steady_state_overlap` measures overlapping client-observed first-token-to-completion
windows. It proves overlap but can exceed W briefly because client-side completion receipt
outlives worker compute; it is not a count of simultaneous ORT kernels.

## Overhead attribution

| Mode/path | Total p50 | Increment over prior path |
|---|---:|---:|
| Synthetic direct fixed work | 11.302 ms | - |
| Synthetic W1 driver | 11.343 ms | 0.041 ms (0.36%) |
| Direct ORT `DecodeSession` | 0.081 ms | - |
| Direct `Engine` workflow | 0.186 ms | 0.105 ms |
| W1 stateless server driver | 0.297 ms | 0.111 ms |

The synthetic delta is routing, channel, owner-thread handshake, and accounting around the
same fixed CPU function. Synthetic work executes on blocking request-side tasks while worker
commands enforce queue ownership, so its near-linear result demonstrates dispatcher and lease
scaling, not model-kernel scaling.

For the tiny ORT fixture, workflow interpretation, sampling, scheduling, and session lifecycle
add 0.105 ms; routing, admission, streaming, and queue dispatch add another 0.111 ms. The W1
driver p50 is 3.67x direct decode because the model itself takes only 0.081 ms. These are
attribution bounds measured in separate loops, not additive constants for larger models.

## Synthetic raw summary

The fixed work is five million deterministic iterations per request.

| Scenario | W | C | req/s | speedup vs W1 | CPU | TTFT p50/p95/p99 ms | steady p50/p95/p99 ms | total p50/p95/p99 ms | conflicts | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| serialized | 1 | 8 | 87.30 | 1.000x | 100% | 85.880/86.350/86.406 | 5.687/5.942/5.961 | 91.585/92.041/92.120 | 0 | 1 |
| serialized | 2 | 8 | 87.30 | 1.000x | 100% | 85.799/86.469/86.624 | 5.692/5.932/5.959 | 91.496/92.200/92.333 | 0 | 1 |
| serialized | 4 | 8 | 87.38 | 1.001x | 100% | 85.834/86.059/86.176 | 5.689/5.912/5.946 | 91.527/91.745/91.872 | 0 | 1 |
| conflict | 1 | 8 | 87.10 | 1.000x | 99% | 5.700/5.724/5.821 | 5.689/5.911/5.964 | 11.390/11.615/11.669 | 700 | 1 |
| conflict | 2 | 8 | 86.97 | 0.999x | 98% | 5.708/5.737/5.875 | 5.694/5.943/5.958 | 11.405/11.641/11.665 | 700 | 1 |
| conflict | 4 | 8 | 87.11 | 1.000x | 98% | 5.705/5.844/5.883 | 5.689/5.929/5.950 | 11.396/11.633/11.646 | 700 | 1 |
| distinct | 1 | 8 | 88.10 | 1.000x | 100% | 85.006/85.204/85.257 | 5.693/5.952/5.995 | 90.712/91.040/91.120 | 0 | 1 |
| distinct | 2 | 8 | 176.30 | 2.001x | 203% | 39.639/39.740/39.743 | 5.690/5.902/5.959 | 45.332/45.516/45.613 | 0 | 2 |
| distinct | 4 | 8 | 352.31 | 3.999x | 405% | 16.963/17.183/17.203 | 5.688/5.962/6.060 | 22.665/22.899/23.018 | 0 | 4 |
| stateless | 1 | 8 | 88.13 | 1.000x | 101% | 85.003/85.206/85.243 | 5.693/5.925/5.966 | 90.702/90.928/91.039 | 0 | 1 |
| stateless | 2 | 8 | 176.02 | 1.997x | 201% | 39.656/39.862/39.897 | 5.694/5.993/6.067 | 45.374/45.672/45.745 | 0 | 2 |
| stateless | 4 | 8 | 352.14 | 3.996x | 401% | 16.948/17.017/17.093 | 5.687/5.956/6.096 | 22.636/22.924/23.024 | 0 | 4 |

Distinct and stateless work reach 1.98x at W2/C2 and 3.96x at W4/C4, with approximately
proportional CPU use and balanced workers. At C=8, throughput remains saturated while p50
total latency falls from about 90.7 ms at W1 to 22.6 ms at W4. Serialized and conflict-owner
throughput remains approximately 1x because the one-session lease deliberately admits one
owner.

## CPU ORT raw summary

| Scenario | W | C | req/s | generated tok/s | speedup vs W1 | CPU | TTFT p50/p95/p99 ms | steady p50/p95/p99 ms | total p50/p95/p99 ms | conflicts | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| serialized | 1 | 8 | 5943 | 5943 | 1.000x | 107% | 0.698/1.262/1.409 | 0.059/0.072/0.206 | 0.757/1.322/1.460 | 0 | 1 |
| serialized | 2 | 8 | 6270 | 6270 | 1.055x | 108% | 0.654/1.204/1.345 | 0.056/0.066/0.203 | 0.710/1.262/1.399 | 0 | 1 |
| serialized | 4 | 8 | 6392 | 6392 | 1.076x | 111% | 0.643/1.176/1.308 | 0.054/0.062/0.202 | 0.697/1.232/1.363 | 0 | 1 |
| conflict | 1 | 8 | 2758 | 11031 | 1.000x | 97% | 0.080/0.157/0.242 | 0.203/0.260/0.347 | 0.284/0.369/0.448 | 70000 | 1 |
| conflict | 2 | 8 | 2858 | 11432 | 1.036x | 96% | 0.076/0.166/0.240 | 0.194/0.253/0.345 | 0.271/0.356/0.436 | 70000 | 1 |
| conflict | 4 | 8 | 2913 | 11653 | 1.056x | 96% | 0.075/0.174/0.239 | 0.189/0.249/0.340 | 0.265/0.360/0.430 | 70000 | 1 |
| distinct | 1 | 8 | 1340 | 5358 | 1.000x | 108% | 7.075/9.877/10.056 | 0.238/0.284/0.374 | 7.311/10.127/10.304 | 0 | 1 |
| distinct | 2 | 8 | 6293 | 25170 | 4.698x | 223% | 0.980/2.450/3.299 | 0.207/0.276/0.360 | 1.188/2.678/3.542 | 0 | 3 |
| distinct | 4 | 8 | 10652 | 42610 | 7.952x | 384% | 0.360/1.128/1.439 | 0.270/0.386/0.479 | 0.634/1.421/1.833 | 0 | 5 |
| stateless | 1 | 8 | 3553 | 14214 | 1.000x | 123% | 2.043/2.111/2.204 | 0.212/0.269/0.356 | 2.253/2.319/2.413 | 0 | 1 |
| stateless | 2 | 8 | 6557 | 26229 | 1.845x | 223% | 0.970/1.085/1.186 | 0.225/0.292/0.378 | 1.196/1.326/1.421 | 0 | 3 |
| stateless | 4 | 8 | 10008 | 40034 | 2.817x | 392% | 0.478/0.632/0.701 | 0.300/0.437/0.509 | 0.781/0.964/1.048 | 0 | 5 |

For the cleaner stateless comparison, W2/C8 reaches 1.845x and W4/C8 reaches 2.817x.
Aggregate generated-token throughput rises from 14.2k/s to 26.2k/s and 40.0k/s. Queue
latency falls: total p50/p99 goes from 2.253/2.413 ms at W1 to 0.781/1.048 ms at W4.
Per-request steady-state p50 increases from 0.212 to 0.300 ms, so throughput improves while
individual compute/service time worsens slightly. The corresponding per-active-request
steady-state rate falls from 14.4k to 9.78k token/s; aggregate throughput scales because more
workers execute concurrently, not because each request gets faster. In synthetic C8 rows, that
per-active rate remains approximately 875 million work units/s while aggregate throughput
scales almost exactly with W.

The distinct-session speedups above W are not model-compute speedups. The W1 path degrades as
10,000 live sessions and concurrent state restore/writeback share one worker; sharding also
partitions that session-state and queue cost. The result is evidence that the single-worker
session path is a bottleneck under this fixture, not evidence that four ORT workers make model
compute 7.95x faster.

One-session rows remain overlap=1 and approximately flat across W. No global lease blocks
distinct sessions: their overlap and balanced completion counts increase with W. With
`intra_op_threads=1`, the remaining sublinear stateless scaling is best explained by fixed
workflow/dispatch cost, queueing, shared cache or memory effects, and async support work.
The model is too small for model compute or memory bandwidth to dominate, and these data do
not isolate ORT inter-op threads.

## CUDA status

CUDA hardware was present, but the linked ORT exposed only `CPUExecutionProvider`. A
feature-enabled probe failed rather than silently falling back:

```text
CUDAExecutionProvider was requested, but the linked ONNX Runtime does not report it
(available providers: ["CPUExecutionProvider"]). The CUDA provider library
'libonnxruntime_providers_cuda.so' is missing or could not be loaded.
```

After installing a GPU-enabled ORT and placing its core and provider libraries first in
`LD_LIBRARY_PATH`, run:

```bash
CARGO_TARGET_DIR=/datadisks/disk1/justinchu/relocated/cargo-target \
taskset -c 1-16 cargo run --release -p onnx-genai-bench \
  --no-default-features --features session-concurrency-cuda \
  --bin session_concurrency -- \
  --mode ort --provider cuda --intra-op-threads 1 \
  --workers 1,2,4 --concurrency 1,2,4,8 \
  --warmups 10 --iterations 10000 --max-new-tokens 4 \
  --output docs/benchmarks/2026-08-25-session-concurrency-ort-cuda.json
```

No CUDA result is claimed.

## Recommended gates

Keep correctness gates hard and timing-free in unit tests. On a dedicated performance runner,
use relative medians from at least five runs:

- synthetic distinct and stateless: W2/C2 >=1.7x and W4/C4 >=3.2x W1 throughput;
- synthetic dispatch overhead <=5% of direct fixed-work p50;
- exact conflicts `(C-1) * iterations`, zero errors, exact completions, and no counter drift;
- every worker receives work when C>=W, and distinct overlap is greater than one;
- informational CPU ORT targets: stateless W2/C>=2 >=1.3x and W4/C>=4 >=2.2x; and
- conflict p99 below 5 microseconds only on that dedicated runner.

Do not gate absolute tiny-model latency or the superlinear distinct-session speedup. Before
making the ORT targets blocking, collect five host-locked CPU runs and five CUDA runs with a
representative model, then gate the median and flag p95/p99 regressions separately from
throughput regressions.
