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

Captured on an Apple M1 Max (32 GiB, macOS 26.5.2) with a release build at
system load 4–5 (10 cores). All profiles were run back-to-back to match
conditions. The ratios between columns are the point; absolute milliseconds
depend on host load — see the **Load sensitivity** section below.

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
model load           5178 ms     2077 ms      120 ms     319 ms     492 ms      216 ms
time to first token   120 ms      139 ms     1075 ms      97 ms     504 ms      342 ms
decode throughput   45.5 tok/s  39.9 tok/s  32.1 tok/s  53.1 tok/s  69.3 tok/s  62.8 tok/s
end-to-end          44.3 tok/s  38.9 tok/s  27.5 tok/s  52.1 tok/s  40.1 tok/s  44.0 tok/s
```

The CPU FP16 pair is the headline result. **Native FP16 now leads on every
metric**: TTFT (97 vs 139 ms = 0.70×), decode (53.1 vs 39.9 tok/s = 1.33×),
end-to-end (52.1 vs 38.9 tok/s = 1.34×), and model load (319 vs 2077 ms =
6.5× faster). The native CPU EP uses BNNS/AMX for FP16 prefill GEMMs (reaching
1472–2436 GFLOPS via Apple's matrix coprocessor) and NEON GEMV with direct
FP16-weight streaming for decode. ONNX Runtime's CPU EP cannot use AMX — it
widens FP16 to FP32 before every GEMM, so it pays a conversion cost and gets
none of the bandwidth benefit.

On FP32, ORT still leads decode: 45.5 vs 32.1 tok/s. The native EP reaches
~70% of ORT's FP32 decode throughput using multi-threaded NEON GEMV on a
persistent SPMD worker pool. The FP32 native prefill (1075 ms) is still slow
because it uses Accelerate `cblas_sgemm` without the BNNS fp16→f32 AMX path
that makes FP16 prefill fast. FP32 prefill has not been optimized.

The native CPU profiles use the persistent SPMD decode pool, which is the
default. The pool is deterministically selected — no host probing or
load-adaptive calibration runs unless explicitly requested via
`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=auto`.

The native backend loads an order of magnitude faster than ONNX Runtime (120–319 ms
vs 2077–5178 ms) because it memory-maps weights instead of building a session graph.

Running the native backend through the MLX plugin (`native+MLX`) recovers
GPU-accelerated decode while keeping the fast load.

Every number above is read out of the committed `.txt` files by
[`../../scripts/check_profile_table.py`](../../scripts/check_profile_table.py),
which CI runs, so the table cannot drift from the samples it describes.

## Cold-start vs steady-state framing

**TTFT alone is not a fair cold-start metric between these two runtimes.**
ORT pre-packs weights during model load (quantization layout transforms,
graph optimisation, session construction). This makes ORT's model load ~5×
slower than native's, while making its TTFT look better — the work that
native does on first inference, ORT has already done during load.

The honest cold-start metric is **time to first token from process start**:
model load + TTFT. This is what a user waiting for the first word actually
experiences. TTFT alone is the honest metric only for a warm, already-loaded
process serving many requests (e.g. a long-running server).

Measured on TinyStories-33M, Apple M1 Max, load 3–5, three interleaved runs:

| Metric | native | ORT | Ratio |
|---|---:|---:|---|
| model load | 29.0 ms | 146.9 ms | native 5.1× faster |
| TTFT | 26.2 ms | 3.4 ms | native 7.7× slower |
| **time to first token from process start** | **55.2 ms** | **150.3 ms** | **native 2.7× faster** |

Both framings are true; they answer different questions:

- **"How fast does the user see the first word after launching?"** →
  time-to-first-token from process start. Native wins 2.7×.
- **"How fast does an already-loaded server respond?"** →
  TTFT alone. ORT wins ~4–8× on this model because it front-loaded
  the work into model load.

The mechanism is weight pre-packing: ORT's session builder transposes,
pads, and tiles weight matrices into SIMD-friendly layouts at load time.
Native loads weights via memory-map with no transform, so the equivalent
work (if any) is deferred to inference.

The `compare` binary reports all three numbers — model load, TTFT, and
their sum — so readers can choose the framing that matches their
deployment scenario.

## Load sensitivity — an honest limitation

**The native FP16 TTFT advantage is real but more load-sensitive than ORT's.**
The BNNS/AMX prefill path dispatches GEMM work onto macOS Grand Central
Dispatch (GCD) internal threads. Under host contention those threads starve,
and prefill latency degrades more than ORT's own thread pool.

Measured on the same M1 Max:

| Load avg (10 cores) | Native TTFT | ORT TTFT | Ratio |
|---|---:|---:|---:|
| ~3–5 (quiet) | 78–97 ms | 108–139 ms | **0.70–0.83×** native wins |
| ~6–8 (moderate) | 90–103 ms | 113–126 ms | ~0.82× native wins |
| ~12+ (busy) | 160–182 ms | 120–140 ms | **~1.3× native loses** |

At load 12+ the native EP loses TTFT to ORT despite having faster kernels.
This is GCD thread contention inside Apple's BNNS framework — BNNS's work
items queue behind other processes' GCD blocks. ORT avoids this because its
thread pool is process-private.

**Report `uptime` with any benchmark.** The figures in this file were captured
at load 4–5. If your machine is busy, expect the native TTFT to be 50–100%
higher than these numbers; ORT is more stable. This is a genuine robustness
weakness scheduled for investigation (possibly related to the measured 4×
per-thread collapse when Accelerate is called inside a Rayon parallel region).

### Further headroom (not pursued yet)

- **FP16 decode is at ~55 GB/s of the ~112 GB/s achievable GEMV roof (~49%).**
  At ORT-level 80% efficiency, ~90 tok/s is theoretically reachable.
- **Accelerate sgemm for Attention SDPA** would recover ~5–8 ms on the FP16
  prefill path by replacing single-threaded NEON loops with AMX-backed GEMMs.
- **Native fp16 elementwise ops** (Mul, RMSNorm, RotaryEmbedding) would
  eliminate per-op fp16↔f32 widening for compute-trivial ops: ~3–5 ms.
- **Fused SiLU·Mul** would eliminate one widen+narrow round-trip per layer in
  the FFN block: ~3–4 ms.
- **FP32 prefill** has not been optimized (1075 ms); this is not urgent now
  that FP16 is the recommended model format.
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
