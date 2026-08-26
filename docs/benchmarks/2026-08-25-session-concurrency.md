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
`17954ae2a5719701e942c941f4967e7c8dfa7439`:

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
`e672200289935a9a345e65248e3c6307814d6056`:

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
| Synthetic direct fixed work | 11.291 ms | - |
| Synthetic W1 driver | 11.351 ms | 0.060 ms (0.53%) |
| Direct ORT `DecodeSession` | 0.082 ms | - |
| Direct `Engine` workflow | 0.193 ms | 0.111 ms |
| W1 stateless server driver | 0.301 ms | 0.108 ms |

The ORT rows now perform equivalent full-prompt work with zero prefix hits. Workflow
interpretation, sampling, scheduling, and session lifecycle add 0.111 ms; routing, admission,
streaming, and dispatch add 0.108 ms. The tiny fixture remains overhead-dominated: W1 driver
p50 is 3.67x direct decode. These independently measured deltas are attribution bounds, not
constants for larger models.

## Synthetic summary

Five million deterministic work iterations are performed per request.

| Scenario | W | C | req/s | speedup | CPU | TTFT p50/p95/p99 ms | total p50/p95/p99 ms | conflicts | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| serialized | 1 | 8 | 87.60 | 1.000x | 100% | 85.582/85.880/85.985 | 91.273/91.613/91.756 | 0 | 1 |
| serialized | 2 | 8 | 87.43 | 0.998x | 100% | 85.735/86.108/86.340 | 91.444/91.784/92.020 | 0 | 1 |
| serialized | 4 | 8 | 87.42 | 0.998x | 101% | 85.701/86.202/86.516 | 91.390/91.979/92.221 | 0 | 1 |
| conflict | 1 | 8 | 87.18 | 1.000x | 99% | 5.702/5.728/5.752 | 11.381/11.636/11.678 | 700 | 1 |
| conflict | 2 | 8 | 87.07 | 0.999x | 100% | 5.705/5.746/5.878 | 11.384/11.634/11.678 | 700 | 1 |
| conflict | 4 | 8 | 87.26 | 1.001x | 99% | 5.702/5.734/5.866 | 11.385/11.594/11.707 | 700 | 1 |
| distinct | 1 | 8 | 88.17 | 1.000x | 101% | 84.961/85.201/85.245 | 90.649/90.927/90.971 | 0 | 1 |
| distinct | 2 | 8 | 176.46 | 2.001x | 199% | 39.590/39.757/39.846 | 45.271/45.526/45.695 | 0 | 2 |
| distinct | 4 | 8 | 352.48 | 3.998x | 402% | 16.935/17.093/17.168 | 22.618/22.854/22.954 | 0 | 4 |
| stateless | 1 | 8 | 88.20 | 1.000x | 100% | 84.938/85.178/85.398 | 90.613/91.010/91.166 | 0 | 1 |
| stateless | 2 | 8 | 176.40 | 2.000x | 201% | 39.595/39.738/39.768 | 45.276/45.447/45.597 | 0 | 2 |
| stateless | 4 | 8 | 352.67 | 3.998x | 402% | 16.942/17.039/17.067 | 22.620/22.972/23.006 | 0 | 4 |

Distinct and stateless fixed work remain near-linear. Serialized and conflict-owner throughput
remain approximately 1x because one session admits one owner. This is dispatcher/lease evidence,
not model-performance evidence.

## Corrected CPU ORT summary

Every row below has prefix hits `0 requests / 0 tokens`.

| Scenario | W | C | req/s | generated tok/s | speedup | CPU | TTFT p50/p95/p99 ms | steady p50/p95/p99 ms | total p50/p95/p99 ms | conflicts | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| serialized | 1 | 8 | 5636 | 5636 | 1.000x | 104% | 0.736/1.336/1.465 | 0.059/0.069/0.204 | 0.795/1.393/1.523 | 0 | 1 |
| serialized | 2 | 8 | 5814 | 5814 | 1.032x | 107% | 0.711/1.305/1.447 | 0.057/0.064/0.211 | 0.767/1.361/1.498 | 0 | 1 |
| serialized | 4 | 8 | 6070 | 6070 | 1.077x | 114% | 0.681/1.236/1.369 | 0.054/0.059/0.198 | 0.735/1.293/1.427 | 0 | 1 |
| conflict | 1 | 8 | 2733 | 10930 | 1.000x | 95% | 0.083/0.187/0.249 | 0.202/0.268/0.351 | 0.286/0.376/0.454 | 70000 | 1 |
| conflict | 2 | 8 | 2835 | 11341 | 1.038x | 95% | 0.080/0.186/0.247 | 0.194/0.253/0.341 | 0.274/0.359/0.441 | 70000 | 1 |
| conflict | 4 | 8 | 2919 | 11674 | 1.068x | 95% | 0.078/0.178/0.240 | 0.187/0.239/0.334 | 0.266/0.343/0.428 | 70000 | 1 |
| distinct | 1 | 8 | 1320 | 5280 | 1.000x | 109% | 7.204/9.950/10.205 | 0.238/0.288/0.374 | 7.452/10.202/10.457 | 0 | 1 |
| distinct | 2 | 8 | 5956 | 23825 | 4.512x | 220% | 1.068/2.618/2.861 | 0.214/0.296/0.388 | 1.279/2.852/3.108 | 0 | 3 |
| distinct | 4 | 8 | 11227 | 44909 | 8.505x | 400% | 0.402/0.989/1.268 | 0.259/0.383/0.459 | 0.666/1.249/1.558 | 0 | 5 |
| stateless | 1 | 8 | 3474 | 13897 | 1.000x | 122% | 2.059/2.126/2.211 | 0.211/0.280/0.358 | 2.269/2.345/2.429 | 0 | 1 |
| stateless | 2 | 8 | 6288 | 25152 | 1.810x | 222% | 1.018/1.158/1.252 | 0.233/0.322/0.411 | 1.253/1.412/1.509 | 0 | 3 |
| stateless | 4 | 8 | 10213 | 40852 | 2.940x | 396% | 0.471/0.620/0.698 | 0.289/0.419/0.495 | 0.764/0.939/1.024 | 0 | 5 |

Stateless C8 throughput scales 1.81x at W2 and 2.94x at W4. Aggregate token throughput rises
from 13.9k/s to 25.2k/s and 40.9k/s, while total p50/p99 falls from 2.269/2.429 ms to
0.764/1.024 ms. Per-active-request steady rate falls from 14.4k to 10.2k token/s, so aggregate
throughput improves through concurrency while individual service cost rises.

The distinct-session speedups above W are not compute scaling. Ten thousand live sessions and
state restore/writeback degrade the W1 path; sharding also partitions that state and queue cost.
One-session rows remain overlap=1. With `intra_op_threads=1`, stateless scaling is limited by
workflow/dispatch overhead, queueing, shared cache/memory effects, and async support work. This
tiny model is too small for model compute or memory bandwidth to dominate.

Compared with the invalid artifact, corrected stateless C8 req/s changed by -2.2% at W1,
-4.1% at W2, and +2.0% at W4. These run-to-run and cold-work differences do not change the
qualitative conclusion, but only the corrected zero-hit artifact supports overhead attribution.

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
- require qualifying W>1/C>1 cells before conflict or overlap may report `true`;
- informational ORT stateless W2/C>=2 >=1.3x and W4/C>=4 >=2.2x; and
- conflict p99 below 5 microseconds only on that dedicated runner.

Do not gate absolute tiny-model latency or superlinear distinct-session speedup. Collect five
host-locked CPU and CUDA runs with a representative model before making ORT timing gates
blocking.
