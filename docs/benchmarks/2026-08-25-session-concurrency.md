# Session concurrency and ORT worker sharding

Initial report date: 2026-08-25

Corrected measurement date: 2026-08-26

## Scope and artifacts

This benchmark measures one-session serialization, typed overlapping-request conflicts,
distinct sessions, stateless least-loaded routing, and direct decode/workflow/driver overhead.
The complete W=1,2,4 by C=1,2,4,8 matrices are machine-readable:

- [`2026-08-25-session-concurrency-synthetic.json`](2026-08-25-session-concurrency-synthetic.json)
- [`2026-08-25-session-concurrency-ort-cpu.json`](2026-08-25-session-concurrency-ort-cpu.json)

Schema v3 records prompt tokens submitted and computed, prefix-cache hit requests/tokens/max
hit length, tri-state correctness gates, and resource used/limit/headroom. Individual timing
samples remain opt-in through `--raw-samples`.

## Correction from the original report

The original ORT artifact is superseded. Its three-token prompt generator cycles after 29
requests, while ORT cross-session prefix lookup ignored `GenerateOptions::cold_start`.
Consequently, direct `DecodeSession` recomputed every prompt but later direct `Engine` and
driver requests could reuse prompt KV. The original 0.105 ms workflow-overhead attribution was
not an equivalent-work comparison.

The corrected implementation:

- makes ORT `cold_start` reset persistent session state;
- disables in-process and connector prefix reuse for that cold request;
- records `prefix_cache_hit_len` for every direct Engine and driver completion;
- rejects a run if any measured cold request reports a hit; and
- compares all 10,000 direct decode/workflow samples for equal prompt length, generated work,
  and token output.

Every corrected ORT row reports zero hit requests, zero hit tokens, max hit length zero, and
`prompt_tokens_computed == prompt_tokens_submitted`. The corrected direct Engine p50 is
0.193 ms instead of 0.186 ms (+3.8%); workflow overhead is 0.111 ms instead of 0.105 ms
(+5.7%). W1 driver p50 is 0.301 ms instead of 0.297 ms (+1.3%). The prior numbers must not be
used.

## Reproduction

Base: merged N>1 main `66ca26a5d`.

Synthetic evidence was measured from clean commit
`0b57da3683000f8954af060c1237bb05fff1a6fd`:

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
`1350f1b7d670508cbda416bfdaccf30d90e9de92`:

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

Warmups are excluded; percentiles use nearest rank. Seed `20260825` and prompt formula
`[1 + i%29, 1 + (7i+3)%29, 1 + (13i+5)%29]` are fixed. Cycling is now safe because every
measured ORT request is cold and the zero-hit invariant is enforced. Direct decode, direct
Engine, and stateless driver paths each process three prompt tokens and generate four tokens.
The serialized scenario processes one prompt token and generates one token per cold request;
it measures ownership/queue serialization, not multi-turn KV continuation.

## Environment

| Item | Value |
|---|---|
| Build | release, Rust 1.98.0 |
| OS | Linux 6.6.141.1-1.azl3, x86-64 |
| CPU | Intel Xeon Platinum 8480C; process affinity 1-16 |
| Host memory | 1,820.9 GiB |
| ORT | 1.29.0, API 29 |
| Providers | `CPUExecutionProvider` only |
| Model | committed `tests/fixtures/tiny-llm/model.onnx.textproto` |
| Model SHA-256 | `d70b079444e021bc6cdea4f6d732788d439deaf9f4d51beb8b84117597592438` |
| GPU snapshot | 8x NVIDIA H200, driver 580.105.08; not used |
| Host lock | unprotected; CPU affinity is not host exclusivity |

Absolute timings remain shared-host observations. CPU affinity reduced scheduler variance.
Each corrected ORT cell lasted approximately 0.9-7.6 seconds, so process CPU accounting is
useful but approximate. On hosts without Linux `/proc` CPU accounting, `cpu_time_ms` and
`process_cpu_percent` remain `null`; absence is never reported as zero work.

Peak RSS was 7.6 MiB for synthetic and 411.1 MiB for the sequential ORT process. The ORT
high-water mark includes allocator retention and cells that created 10,000 sessions; it is not
an isolated per-worker delta.

### Governor resource semantics

These are ledger snapshots, not budgets inferred from the `used` value:

| W | Tier | Used | Limit | Headroom |
|---:|---|---:|---:|---:|
| 1 | host | 8 MiB | 455.229 GiB | 466146.403 MiB |
| 2 | host | 16 MiB | 455.229 GiB | 466138.403 MiB |
| 4 | host | 32 MiB | 455.229 GiB | 466122.403 MiB |
| 1/2/4 | device | 0.072 MiB | 455.229 GiB | 466154.332 MiB |
| 1/2/4 | disk | not configured | not configured | not configured |

The device value is the governor's CPU-run accounting estimate, not observed CUDA allocation.
Every JSON row records `used_bytes`, `limit_bytes`, and `headroom_bytes` separately under
`governed_resources.host`, `.disk`, and `.device`.

## Correctness gates

Gate values are now tri-state:

- `true`: a qualifying cell ran and passed;
- `false`: a qualifying cell ran and failed; and
- `null`: the requested matrix did not exercise the gate.

The CLI writes the JSON report before returning a nonzero status for any
`false` gate. This preserves the failed measurement and makes the tri-state
contract observable instead of discarding the report on the first failed
invariant.

Typed conflict and distinct client-stream overlap require a W>1/C>1 cell. A W=1/C=1-only regression test
therefore serializes both as `null`, rather than vacuously reporting success. Both full
schema-v4 artifacts report all five gates `true`: W1 output parity, typed conflict,
distinct client-stream overlap,
exact counts, and zero counter drift.

At C=8, every ORT conflict row completed 10,000 owners and rejected exactly 70,000 contenders.
Conflict p99 was 0.279/0.273/0.286 microseconds for W=1/2/4. Stateless W4/C8 completions were
2488/2512/2491/2509. `max_client_stream_overlap` measures client-observed first-token-to-finish
windows and can briefly exceed W; it is not simultaneous-kernel count.

## Overhead attribution

| Mode/path | Total p50 | Increment over prior path |
|---|---:|---:|
| Synthetic direct fixed work | 11.283 ms | - |
| Synthetic W1 driver | 11.357 ms | 0.074 ms (0.66%) |
| Direct ORT `DecodeSession` | 0.081 ms | - |
| Direct `Engine` workflow | 0.188 ms | 0.107 ms |
| W1 stateless server driver | 0.297 ms | 0.109 ms |

The ORT rows now perform equivalent full-prompt work with zero prefix hits. Workflow
interpretation, sampling, scheduling, and session lifecycle add 0.107 ms; routing, admission,
streaming, and dispatch add 0.109 ms. The tiny fixture remains overhead-dominated: W1 driver
p50 is 3.67x direct decode. These independently measured deltas are attribution bounds, not
constants for larger models.

## Synthetic summary

Five million deterministic work iterations are performed per request.

| Scenario | W | C | req/s | speedup | CPU | TTFT p50/p95/p99 ms | total p50/p95/p99 ms | conflicts | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| serialized | 1 | 8 | 87.39 | 1.000x | 99% | 85.735/86.166/86.816 | 91.410/91.848/92.498 | 0 | 1 |
| serialized | 2 | 8 | 87.44 | 1.001x | 100% | 85.761/86.080/86.350 | 91.434/91.766/92.027 | 0 | 1 |
| serialized | 4 | 8 | 87.31 | 0.999x | 100% | 85.856/86.387/86.598 | 91.536/92.096/92.277 | 0 | 1 |
| conflict | 1 | 8 | 87.10 | 1.000x | 99% | 5.705/5.848/5.865 | 11.389/11.650/11.683 | 700 | 1 |
| conflict | 2 | 8 | 87.22 | 1.001x | 100% | 5.705/5.732/5.855 | 11.384/11.590/11.629 | 700 | 1 |
| conflict | 4 | 8 | 87.12 | 1.000x | 98% | 5.704/5.723/5.856 | 11.384/11.634/11.647 | 700 | 1 |
| distinct | 1 | 8 | 88.22 | 1.000x | 101% | 84.938/85.196/85.227 | 90.626/90.905/91.003 | 0 | 1 |
| distinct | 2 | 8 | 176.24 | 1.998x | 199% | 39.523/40.119/40.149 | 45.226/45.822/45.977 | 0 | 2 |
| distinct | 4 | 8 | 351.41 | 3.983x | 397% | 16.909/17.789/17.843 | 22.583/23.494/23.694 | 0 | 4 |
| stateless | 1 | 8 | 88.15 | 1.000x | 101% | 84.964/85.290/85.378 | 90.686/91.048/91.100 | 0 | 1 |
| stateless | 2 | 8 | 176.15 | 1.998x | 201% | 39.607/39.768/40.027 | 45.295/45.576/45.740 | 0 | 2 |
| stateless | 4 | 8 | 352.96 | 4.004x | 399% | 16.941/17.020/17.099 | 22.618/22.783/22.867 | 0 | 4 |

Distinct and stateless fixed work remain near-linear. Serialized and conflict-owner throughput
remain approximately 1x because one session admits one owner. This is dispatcher/lease evidence,
not model-performance evidence.

## Corrected CPU ORT summary

Every row below has prefix hits `0 requests / 0 tokens`.

| Scenario | W | C | req/s | generated tok/s | speedup | CPU | TTFT p50/p95/p99 ms | steady p50/p95/p99 ms | total p50/p95/p99 ms | conflicts | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| serialized | 1 | 8 | 5581 | 5581 | 1.000x | 109% | 0.742/1.344/1.495 | 0.059/0.071/0.209 | 0.800/1.408/1.556 | 0 | 1 |
| serialized | 2 | 8 | 5865 | 5865 | 1.051x | 108% | 0.707/1.288/1.423 | 0.056/0.066/0.203 | 0.763/1.343/1.476 | 0 | 1 |
| serialized | 4 | 8 | 6097 | 6097 | 1.092x | 109% | 0.676/1.239/1.378 | 0.053/0.059/0.204 | 0.730/1.292/1.438 | 0 | 1 |
| conflict | 1 | 8 | 2761 | 11044 | 1.000x | 97% | 0.082/0.170/0.241 | 0.201/0.256/0.342 | 0.284/0.363/0.442 | 70000 | 1 |
| conflict | 2 | 8 | 2865 | 11461 | 1.038x | 97% | 0.078/0.175/0.242 | 0.192/0.245/0.338 | 0.271/0.349/0.433 | 70000 | 1 |
| conflict | 4 | 8 | 2960 | 11842 | 1.072x | 96% | 0.076/0.171/0.234 | 0.185/0.232/0.331 | 0.262/0.329/0.422 | 70000 | 1 |
| distinct | 1 | 8 | 1332 | 5328 | 1.000x | 109% | 7.171/9.803/9.977 | 0.237/0.286/0.377 | 7.419/10.052/10.224 | 0 | 1 |
| distinct | 2 | 8 | 6007 | 24027 | 4.510x | 220% | 0.984/2.182/3.656 | 0.213/0.289/0.386 | 1.202/2.401/3.877 | 0 | 3 |
| distinct | 4 | 8 | 11450 | 45802 | 8.597x | 401% | 0.395/0.985/1.190 | 0.252/0.368/0.452 | 0.650/1.254/1.461 | 0 | 5 |
| stateless | 1 | 8 | 3518 | 14073 | 1.000x | 122% | 2.037/2.104/2.195 | 0.209/0.266/0.352 | 2.243/2.311/2.397 | 0 | 1 |
| stateless | 2 | 8 | 6493 | 25970 | 1.845x | 223% | 0.987/1.125/1.222 | 0.225/0.304/0.397 | 1.212/1.369/1.469 | 0 | 3 |
| stateless | 4 | 8 | 10372 | 41489 | 2.948x | 397% | 0.464/0.613/0.693 | 0.285/0.395/0.486 | 0.753/0.927/1.026 | 0 | 5 |

Stateless C8 throughput scales 1.85x at W2 and 2.95x at W4. Aggregate token throughput rises
from 14.1k/s to 26.0k/s and 41.5k/s, while total p50/p99 falls from 2.243/2.397 ms to
0.753/1.026 ms. Per-active-request steady rate falls from 14.6k to 10.3k token/s, so aggregate
throughput improves through concurrency while individual service cost rises.

The distinct-session speedups above W are not compute scaling. Ten thousand live sessions and
state restore/writeback degrade the W1 path; sharding also partitions that state and queue cost.
One-session rows remain overlap=1. With `intra_op_threads=1`, stateless scaling is limited by
workflow/dispatch overhead, queueing, shared cache/memory effects, and async support work. This
tiny model is too small for model compute or memory bandwidth to dominate.

The host was not exclusively locked. Earlier schema-v4 attempts exposed isolated multi-second
W2/W4 cells while neighboring cells remained near one second; one attempt also correctly wrote
a failed typed-conflict/count report before exiting nonzero. The committed artifact is a complete
passing run, but single-cell ORT timing remains observational rather than a release gate.

## CUDA status

CUDA hardware was present, but linked ORT exposed only `CPUExecutionProvider`. The
feature-enabled probe failed explicitly because `libonnxruntime_providers_cuda.so` was missing
or unloadable. No CUDA result is claimed.

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

## Recommended gates

Keep timing-free correctness gates in unit tests. On a dedicated runner, use medians from at
least five runs:

- require zero prefix-cache hits and submitted prompt tokens equal computed prompt tokens;
- synthetic distinct/stateless W2/C2 >=1.7x and W4/C4 >=3.2x W1 throughput;
- synthetic dispatch overhead <=5% of direct fixed-work p50;
- exact conflicts `(C-1) * iterations`, zero errors/completion drift/counter drift;
- require qualifying W>1/C>1 cells before conflict or client-stream overlap may report `true`;
- informational ORT stateless W2/C>=2 >=1.3x and W4/C>=4 >=2.2x; and
- conflict p99 below 5 microseconds only on that dedicated runner.

Do not gate absolute tiny-model latency or superlinear distinct-session speedup. Collect five
host-locked CPU and CUDA runs with a representative model before making ORT timing gates
blocking.
