### 2026-07-22: Native 7B CPU decode profile
**By:** Sebastian

## Method

- Host: dual-socket Intel Xeon Platinum 8480C, 96 physical cores, no SMT, two NUMA nodes.
- Model: Foundry Qwen2.5-Coder-7B int4 v4; prompt `Write a function to sort a list.` (8 tokens); greedy 24-token generation.
- Build: `cargo build --release -p onnx-genai-bench --features mlas --bin profile_native`.
- No CPU pinning; runs were sequential on the otherwise shared host.
- Per-node timing used the existing zero-cost-when-disabled `ONNX_GENAI_PROFILE_OPS=1` executor hook. The table is the mean of 23 measured M=1 forwards after the measured prefill.
- `ONNX_GENAI_PROFILE=1` measured host sampling separately. `profile_native` now resets warmup statistics and prints this existing stage profiler; the focused synthetic integration test covers enabled reporting.

## Important correction to the headline latency

The reported approximately 113 ms/generated-token number is **not one M=1 decode step**. `profile_native`'s default throughput timer includes one 8-token prompt prefill per 24 generated tokens.

At 32 decode threads in this run:

| measurement | result |
|---|---:|
| Default 24-token end-to-end benchmark | 116.662 ms/token, 8.57 tok/s |
| Steady M=1 decode (`--steady --decode-skip 8`, combined two runs) | 79.456 ms/token, 12.59 tok/s |
| Prefill/reset amortization in the default benchmark | 37.206 ms/generated token (31.9%) |

Thus only about 68% of the headline 116.7 ms/token is steady M=1 decode. Optimization claims must state which metric they improve.

## M=1 per-stage breakdown

The matched profiled generation measured 83.394 ms per M=1 step (profiling/load overhead makes this about 5% slower than the unprofiled 79.456 ms). Percentages are the robust result:

| stage | ms/M=1 step | share |
|---|---:|---:|
| `MatMulNBits` projections (141 calls) | 64.334 | **77.1%** |
| Elementwise/activation: `Silu` + `Add` + `Mul` | 7.934 | **9.5%** |
| GQA/attention, including RoPE | 5.335 | **6.4%** |
| RMSNorm/LayerNorm | 3.275 | **3.9%** |
| Sampling/host argmax | 0.079 | **0.1%** |
| Executor/native-decode orchestration and remaining tiny nodes | 2.437 | **2.9%** |
| **Total** | **83.394** | **100%** |

The residual is an upper bound because it also contains enabled-profiler bookkeeping. Sampling, token commit, and detokenization together are below 0.1 ms/token and are not material.

## MatMulNBits routing

M=1 does **not** use MLAS SQNBit under the current configuration. `NXRT_SQNBIT_PREFILL_MIN` was unset, so the default threshold is 16; `try_mlas_sqnbit` returns before packing when `m < 16`. The benchmark therefore uses the specialized packed hand int4/VNNI path for M=1. Building with `--features mlas` does not change this routing.

An exploratory `NXRT_SQNBIT_PREFILL_MIN=2` run kept M=1 on the hand path while sending the 8-row prompt to MLAS; it measured 8.43 tok/s versus 8.57 tok/s at the default threshold, so lowering the crossover is not an optimization on this workload.

## Thread scaling

Requested default-harness results (one prefill per 24 generated tokens, two measured runs):

| `ONNX_GENAI_CPU_DECODE_THREADS` | ms/generated token | tok/s | vs. 32 |
|---:|---:|---:|---:|
| 8 | 150.908 | 6.63 | -22.6% |
| 16 | 125.908 | 7.94 | -7.4% |
| **32** | **116.662** | **8.57** | — |
| 48 | 131.342 | 7.61 | -11.2% |

Steady M=1 combined across the two runs:

| threads | ms/M=1 token | tok/s |
|---:|---:|---:|
| 8 | 112.992 | 8.85 |
| 16 | 83.569 | 11.97 |
| **32** | **79.456** | **12.59** |
| 48 | 103.928 | 9.62 |

Thirty-two threads is the clear operating point for this 7B model on this dual-socket host; 48 crosses into synchronization/NUMA regression.

## Ranked optimization targets

1. **MatMulNBits cross-node efficiency (77.1%)** — keep the hand int4/VNNI M=1 backend, but target projection grouping, activation-quantization reuse, direct executor-output writes, and fewer per-projection barriers. A 20% local reduction is a 15.4% M=1 latency reduction; a 30% local reduction is 23.1%.
2. **Fuse projection-adjacent elementwise work (9.5%)** — combine eligible bias/residual and gate/up SiLU work structurally, preserving generic fallbacks. Recovering half this bucket yields about 4.8% lower M=1 latency; the absolute ceiling is 9.5%.
3. **GQA/attention (6.4% here, increasing with context)** — reduce remaining per-layer attention setup/copies and reuse scratch/static KV views. Halving this bucket yields about 3.2% at this short context, with larger upside at long context.

RMSNorm is the next target at 3.9%, preferably as part of residual-plus-normalization fusion. Sampling and generic loop orchestration are not priority work.
