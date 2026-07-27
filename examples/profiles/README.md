# Profile samples

Real `--profile` reports captured from Qwen2.5-0.5B-Instruct, one per execution
provider, so the shape of the output can be read without running anything — and
so the numbers can be compared against a later change.

| File | Backend | Provider | Model |
| --- | --- | --- | --- |
| [`qwen2.5-0.5b-cpu.txt`](qwen2.5-0.5b-cpu.txt) | ONNX Runtime | CPU | FP32 |
| [`qwen2.5-0.5b-f16-cpu.txt`](qwen2.5-0.5b-f16-cpu.txt) | ONNX Runtime | CPU | FP16 |
| [`qwen2.5-0.5b-native.txt`](qwen2.5-0.5b-native.txt) | native | CPU | FP32 |
| [`qwen2.5-0.5b-f16-native.txt`](qwen2.5-0.5b-f16-native.txt) | native | CPU | FP16 |
| [`qwen2.5-0.5b-metal.txt`](qwen2.5-0.5b-metal.txt) | ONNX Runtime | MLX/Metal plugin | FP32 |
| [`qwen2.5-0.5b-native-mlx.txt`](qwen2.5-0.5b-native-mlx.txt) | native | MLX/Metal plugin | FP32 |
| `*.json` | | the same runs via `--profile-json`, for diffing or plotting | |

Captured on an Apple M1 Max (32 GiB, macOS 26.5.2) with a release build. The
machine was not idle, so treat the absolute milliseconds as indicative and the
*ratios* as the point.

## Regenerating

```bash
cargo build --release -p onnx-genai-cli

# ORT CPU FP32
ONNX_GENAI_EP=cpu ./target/release/onnx-genai --profile \
  generate models/qwen2.5-0.5b \
  --prompt "Write a short Rust function that reverses a string." \
  --max-new-tokens 200 --temperature 0

# ORT CPU FP16
ONNX_GENAI_EP=cpu ./target/release/onnx-genai --profile \
  generate models/qwen2.5-0.5b-f16 \
  --prompt "Write a short Rust function that reverses a string." \
  --max-new-tokens 200 --temperature 0

# native CPU FP32 (the SPMD pool is the default; no env var needed)
ONNX_GENAI_BACKEND=native ONNX_GENAI_EP=cpu \
  ./target/release/onnx-genai --profile \
  generate models/qwen2.5-0.5b \
  --prompt "Write a short Rust function that reverses a string." \
  --max-new-tokens 200 --temperature 0

# native CPU FP16
ONNX_GENAI_BACKEND=native ONNX_GENAI_EP=cpu \
  ./target/release/onnx-genai --profile \
  generate models/qwen2.5-0.5b-f16 \
  --prompt "Write a short Rust function that reverses a string." \
  --max-new-tokens 200 --temperature 0
```

For the Metal run, point at the MLX plugin the Python packages ship. It is then
auto-selected on macOS:

```bash
ONNX_GENAI_METAL_EP_LIB=$(python -c 'import onnxruntime_mlx, os;
print(os.path.join(os.path.dirname(onnxruntime_mlx.__file__), "libonnxruntime_mlx_ep.dylib"))') \
./target/release/onnx-genai --profile generate models/qwen2.5-0.5b ...
```

For the native backend, set `ONNX_GENAI_BACKEND=native`. The persistent SPMD
pool is the default; no additional env var is needed. For the FP16 model, point
at `models/qwen2.5-0.5b-f16`.

Add `--profile-trace out.json` for a Perfetto timeline instead of these
aggregates; see [`../traces/`](../traces/).

## One-command native CPU vs ORT CPU comparison

Use the bench crate's `compare` binary when optimizing the native CPU execution
provider against ONNX Runtime's CPU EP. It loads the same model, renders the
same prompt through the model chat template, alternates native and ORT CPU runs,
measures the host's sequential CPU memory-bandwidth roofline, discards warmups,
then reports medians with p10-p95 spread plus JSON for plotting:

```bash
cargo run --release -p onnx-genai-bench --features bench-native --bin compare -- \
  --model models/qwen2.5-0.5b \
  --prompt "Write a short Rust function that reverses a string." \
  --tokens 50 --decode-skip 2 --warmups 1 --runs 5 \
  --profile-json target/pris-cpu-compare.json
```

This follows Sebastian's M1 Max methodology: one discarded full-generation
warmup, at least five measured repetitions, a fixed 40-token prompt, 50 generated
tokens, and the first two generated tokens excluded from the decode-throughput
window. The output includes both absolute decode tok/s and decode roofline
fraction, so a lower-bandwidth M1 Air result remains interpretable instead of
being compared directly to this M1 Max. Keep the machine idle and cool between
final runs.

## What the samples show

```
                      ORT+CPU  ORT+CPU f16    native  native f16  ORT+Metal  native+MLX
model load           2710 ms     1988 ms      134 ms     138 ms     492 ms      216 ms
time to first token   114 ms      119 ms     1023 ms    1366 ms     504 ms      342 ms
decode throughput   45.5 tok/s  40.5 tok/s  33.6 tok/s  43.6 tok/s  69.3 tok/s  62.8 tok/s
end-to-end          44.4 tok/s  39.5 tok/s  28.8 tok/s  33.7 tok/s  40.1 tok/s  44.0 tok/s
```

The CPU FP16 pair is the headline result. The native CPU EP reads FP16 weights
directly from the memory-mapped model file via NEON `fcvtl`, streaming half the
bytes of FP32. ONNX Runtime's CPU EP cannot do this -- it widens FP16 to FP32
before every GEMM, so it pays a conversion cost and gets none of the bandwidth
benefit. The result is that **native FP16 (43.6 tok/s) beats ORT FP16
(40.5 tok/s)** on the same model and dtype, an architectural advantage rather
than a tuning difference. On a quiet host the native FP16 steady-state (p50
17.3 ms = 57.8 tok/s) exceeds ORT's FP32 rate; the profile's mean includes a
~1 s pool-initialisation spike that pulls the average down.

On FP32, ORT still leads: 45.5 vs 33.6 tok/s. The native EP reaches ~74% of
ORT's FP32 decode throughput using multi-threaded NEON GEMV on a persistent
SPMD worker pool. The remaining gap is structural: ORT's MLAS fuses subgraph
operations and has a mature thread pool, while the native EP dispatches 434
individual ops per token.

**Prefill/TTFT remains a weakness.** The native backend's time to first token
is ~10x worse than ORT (1023--1366 ms vs 114--119 ms). Prefill is compute-bound
rather than bandwidth-bound and has not been optimised in this campaign; it is a
separate regime from the decode path that these changes target. This pulls
native's end-to-end throughput well below its decode rate.

The native CPU profiles use the persistent SPMD decode pool, which is the
default. The pool is deterministically selected — no host probing or
load-adaptive calibration runs unless explicitly requested via
`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=auto`. Under heavy co-tenant load the
pool's busy-wait barrier contends with neighbours, so throughput degrades more
than the flat legacy path (`=0`). This is the accepted tradeoff for
predictability: the same prompt at temperature 0 always follows the same
floating-point reduction order, regardless of system load. Users on shared
machines who prefer adaptation can set `=auto` to let a startup calibrator pick
the faster path (pool on quiet hosts, flat under load). The decode throughput
shown above is the mean across all generated tokens including a ~1 s
pool-initialisation spike on the first decode step; the p50 inter-token latency
in the `.txt` files is the better measure of steady-state performance.

The native backend loads an order of magnitude faster than ONNX Runtime (134 ms
vs 2710 ms) because it memory-maps weights instead of building a session graph.

Running the native backend through the MLX plugin (`native+MLX`) recovers full
GPU-accelerated decode while keeping the fast load, which is why it leads on
end-to-end throughput despite ORT+Metal decoding fastest.

Every number above is read out of the committed `.txt` files by
[`../../scripts/check_profile_table.py`](../../scripts/check_profile_table.py),
which CI runs, so the table cannot drift from the samples it describes.

Between the two ONNX Runtime columns, Metal has the faster steady-state decode
(1.5x) and the slower start, so they arrive at the same end-to-end number here.
Which one wins depends entirely on how many tokens are generated: at shorter
sequences CPU wins, at longer sequences Metal does.

### Further headroom (not pursued in this campaign)

- **FP16 is at ~55 GB/s of the ~112 GB/s achievable GEMV roof (~49%).** At
  ORT-level 80% efficiency, ~90 tok/s is theoretically reachable.
- **Gate+up GEMV fusion** would halve activation re-reads and dispatch overhead
  on the FFN block.
- **Graph-level op fusion** would reduce the 434 individual op dispatches
  toward ORT's fused subgraph count.
- **Prefill/TTFT** is a separate compute-bound regime that needs its own
  optimisation pass.
- **Q4** has a theoretical ~450 tok/s ceiling but needs both a real int4
  aarch64 kernel and a compatible model export.

An earlier revision of this sample recorded Metal as *slower* than CPU. That
was real, and the trace said why: the MLX plugin declined every `Attention`
node because it refused a causal mask combined with an explicit mask, so
attention ran on the CPU and the graph was cut into 28 subgraphs. Fixing the
plugin to claim those nodes is what moved it from 36 to 69 tok/s.

## Why a stage's `us/call` can look nothing like the latency line

`loop.step` on the Metal run reports far more per call than the
`inter-token latency` line suggests, and both are right: **the first
`loop.step` carries the prefill.** Back the one-off cost out and the rest is
uniform. Read `inter-token latency` and its percentiles for steady-state
decode, and `us/call` only for stages that are not called once per token.

The native run shows why the percentiles are printed at all: the SPMD pool
initialisation adds a ~1 s spike on the first decode step, producing a large
gap between the mean and median. Read the p50 for steady-state decode
performance and the p99/max for worst-case latency.

## Reading the rest of the report

* **Decode throughput excludes the prefill wait; end-to-end includes it.** A long
  prompt inflates the second without the model decoding any faster.
* **Percentiles sit next to the mean** because the mean can hide tail latency.
  For the native CPU run the first decode step carries a ~1 s SPMD pool
  initialisation cost, so the mean is much higher than the p50 steady-state.
  Always read p50 for typical token latency and p99/max for worst-case.
* **`kv page activity`** is a delta for this run, not lifetime totals. Evictions
  and allocation failures appear only when they happen.
* **`device memory breakdown`** splits the ceiling into weights and KV. The line
  about activations says they are *not yet measured* rather than folding them
  into another number or reporting them as zero.
* **`execution provider`** names what actually resolved, not what was requested:
  on macOS the Metal plugin is auto-selected without appearing in any
  environment variable the caller set.
