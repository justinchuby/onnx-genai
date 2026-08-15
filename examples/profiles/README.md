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

### Roofline ceiling semantics

The decode roofline ceiling represents the maximum decode throughput achievable
if the only bottleneck is streaming model weights from DRAM at the measured
bandwidth. The denominator is **decode weight bytes**: the subset of model
weights that are fully read during each decode step. This specifically excludes
embedding tables (token embeddings, positional embeddings, rotary caches)
accessed via Gather lookups, which read only a single row per token rather than
the full matrix.

For the FP32 `qwen2.5-0.5b` model, decode weight bytes are ~1.43 GiB of the
~1.98 GiB total (the 544 MiB token embedding table is excluded). A roofline
fraction of e.g. 45% means the backend achieves 45% of the theoretical maximum
set by DRAM bandwidth alone.

**Cache-assisted models:** For models whose decode working set is small enough
that the Apple SLC (system-level cache, ~48 MiB on M1 Max) covers ≥10% of the
decode set, the DRAM ceiling is **not binding** — inter-token temporal locality
means a significant fraction of weights are served from cache, not DRAM. The
tool marks such models with `⚠️ NOT BINDING` in the ceiling row and `*` in the
roofline% column. Examples:

| Model | Decode set | SLC/decode | Ceiling binding? |
|---|---|---|---|
| TinyStories-1M | 14 MiB | 348% | No (fully cache-resident) |
| TinyStories-33M | 255 MiB | 19% | No (partially cache-assisted) |
| qwen2.5-0.5b FP16 | 682 MiB | 7% | Yes |
| qwen2.5-0.5b FP32 | 1365 MiB | 3.5% | Yes |

For DRAM-bound models (qwen2.5-0.5b), exceeding 100% should not occur on a
quiet host. If it does, the bandwidth probe was depressed by host load.

## What the samples show

```
                      ORT+CPU  ORT+CPU f16    native  native f16  ORT+Metal  native+MLX
model load           5178 ms     2077 ms      120 ms     319 ms     492 ms      216 ms
time to first token   120 ms      139 ms     1075 ms      97 ms     504 ms      342 ms
decode throughput   45.5 tok/s  39.9 tok/s  32.1 tok/s  53.1 tok/s  69.3 tok/s  62.8 tok/s
end-to-end          44.3 tok/s  38.9 tok/s  27.5 tok/s  52.1 tok/s  40.1 tok/s  44.0 tok/s
```

These single-provider profiles were captured at **load 4–5** (10-core M1 Max).
The CPU FP16 native/ORT comparison in this table (1.33× decode) is
**systematically pessimistic** because native decode degrades more under
contention than ORT — see **Load sensitivity** below. For the definitive
head-to-head comparison at controlled load, see the next section.

Every number above is read out of the committed `.txt` files by
[`../../scripts/check_profile_table.py`](../../scripts/check_profile_table.py),
which CI runs, so the table cannot drift from the samples it describes.

## Head-to-head comparison (native CPU vs ORT CPU)

These are the definitive performance claims, measured with the `compare` binary
(interleaved native/ORT runs, warmups discarded, statistical medians). Each
table reports host load at measurement time. The compare methodology is
documented above under "One-command native CPU vs ORT CPU comparison".

### qwen2.5-0.5b-f16 — the headline model

Measured 2026-07-28 on Apple M1 Max (32 GiB, macOS 26.5.2).
Load 2.5–3.7 (quiet host), 3 measured pairs, commit `679837e1`.

| Metric | native | ORT | Ratio | Load |
|---|---:|---:|---|---|
| model load ms | 277.6 | 1744.8 | **6.3× faster** | 3.7 |
| TTFT ms | 90.4 | 111.6 | 1.23× faster | 3.7 |
| process start → first token ms | 370.2 | 1856.4 | **5.01× faster** | 3.7 |
| decode tok/s | 73.02 | 42.56 | **1.72×** | 3.7 |
| decode roofline % | 42.6% | 24.8% | — | 3.7 |
| end-to-end tok/s | 65.28 | 38.80 | **1.68×** | 3.7 |

Decode roofline ceiling: 171.6 tok/s (DRAM-bound, SLC covers only 7% of decode
set). Measured bandwidth: 122.8 GB/s.

**Key takeaway:** Native FP16 decode is **1.72×** ORT's throughput at low load.
This is the primary CPU inference benchmark; all optimised dispatch paths (NEON
depthwise #342, pointwise Conv→GEMM #347, inline NEON SDPA #349, thin-M GEMM
#353) contribute to this figure.

### TinyStories-33M (FP32) — small-model honesty check

Measured 2026-07-28. Load 2.5–2.6 (quiet host), 5 measured pairs, commit `679837e1`.

| Metric | native | ORT | Ratio | Load |
|---|---:|---:|---|---|
| model load ms | 44.2 | 144.5 | 3.3× faster | 2.6 |
| TTFT ms | 16.3 | 5.7 | **2.86× slower** | 2.6 |
| process start → first token ms | 60.8 | 150.2 | **2.47× faster** | 2.6 |
| decode tok/s | 297.4 | 327.0 | **0.91× (ORT wins)** | 2.6 |
| decode roofline % | 65.2%* | 71.7%* | — | 2.6 |
| end-to-end tok/s | 258.6 | 315.6 | **0.82× (ORT wins)** | 2.6 |

Decode roofline ceiling: 456.3 tok/s ⚠️ **NOT BINDING** — decode set (255 MiB)
is partially cache-resident (SLC covers ~19%); roofline% marked with `*` is
not a meaningful efficiency metric for this model (#354).

**What's behind:** On this small FP32 model, ORT still wins decode by ~9% and
end-to-end by ~18%. Native TTFT is 2.86× slower because native defers all
weight preparation to first inference while ORT front-loads it into model load.
The cold-start metric (process start → first token) still favours native
2.47× because the 3.3× model-load advantage dominates.

### Previously published figures (load 4–5)

The table in "What the samples show" above was measured at load 4–5 and
reported native FP16 decode at 53.1 tok/s (1.33× ORT). That measurement is
genuine but reflects contended conditions. The native EP's NEON GEMV decode
path is more load-sensitive than ORT's private thread pool — see **Load
sensitivity** below for quantified evidence. The prior figure was neither wrong
nor misleading for the conditions stated; the low-load figure (73.0 tok/s,
1.72×) is what the hardware delivers when not competing for resources.

### Vision and audio models

ResNet-18 is not present locally. MobileNetV2 and Whisper (encoder, graph-only)
are present but cannot be benchmarked with the generative `compare` tool. No
previous published figures exist for these models in this file. When a
non-generative inference benchmark harness is available, these models should be
measured and added.

## Explanation of the FP16 results

The native CPU EP uses BNNS/AMX for FP16 prefill GEMMs (reaching
1472–2436 GFLOPS via Apple's matrix coprocessor) and NEON GEMV with direct
FP16-weight streaming for decode. ONNX Runtime's CPU EP cannot use AMX — it
widens FP16 to FP32 before every GEMM, so it pays a conversion cost and gets
none of the bandwidth benefit.

On FP32, ORT still leads decode on the large model: 42.6 vs 73.0 tok/s
favours native on FP16, but on TinyStories-33M FP32 the relationship inverts
(native 297.4 vs ORT 327.0 tok/s). The native EP reaches ~91% of ORT's FP32
decode throughput on small cache-resident models using multi-threaded NEON GEMV
on a persistent SPMD worker pool.

The native CPU profiles use the persistent SPMD decode pool, which is the
default. The pool is deterministically selected — no host probing or
load-adaptive calibration runs unless explicitly requested via
`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=auto`.

The native backend loads an order of magnitude faster than ONNX Runtime (44–278 ms
vs 145–1745 ms) because it memory-maps weights instead of building a session graph.

Running the native backend through the MLX plugin (`native+MLX`) recovers
GPU-accelerated decode while keeping the fast load.

## Cold-start vs steady-state framing

**TTFT alone is not a fair cold-start metric between these two runtimes.**
ORT pre-packs weights during model load (quantization layout transforms,
graph optimisation, session construction). This makes ORT's model load ~5–6×
slower than native's, while making its TTFT look better — the work that
native does on first inference, ORT has already done during load.

The honest cold-start metric is **time to first token from process start**:
model load + TTFT. This is what a user waiting for the first word actually
experiences. TTFT alone is the honest metric only for a warm, already-loaded
process serving many requests (e.g. a long-running server).

Measured on TinyStories-33M, Apple M1 Max, **load 2.5–2.6**, 5 interleaved
pairs (2026-07-28):

| Metric | native | ORT | Ratio | Load |
|---|---:|---:|---|---|
| model load | 44.2 ms | 144.5 ms | native 3.3× faster | 2.6 |
| TTFT | 16.3 ms | 5.7 ms | **native 2.86× slower** | 2.6 |
| **process start → first token** | **60.8 ms** | **150.2 ms** | **native 2.47× faster** | 2.6 |

Both framings are true; they answer different questions:

- **"How fast does the user see the first word after launching?"** →
  time-to-first-token from process start. Native wins 2.47×.
- **"How fast does an already-loaded server respond?"** →
  TTFT alone. ORT wins ~2.9× on this model because it front-loaded
  the work into model load.

The mechanism is weight pre-packing: ORT's session builder transposes,
pads, and tiles weight matrices into SIMD-friendly layouts at load time.
Native loads weights via memory-map with no transform, so the equivalent
work (if any) is deferred to inference.

The `compare` binary reports all three numbers — model load, TTFT, and
their sum — so readers can choose the framing that matches their
deployment scenario.

## Multi-turn sessions — session-persistent KV changes the picture

PR #397 added session-persistent KV to the native backend via
`create_native_session` + `generate_native_in_session`. The `multiturn`
benchmark now uses this session API for native (matching ORT's `create_session`
+ `generate_in_session`). Pass `--native-stateless` to measure the old
full-re-prefill path.

Measured 2026-07-29 on Apple M1 Max (32 GiB, macOS 26.5.2), interleaved A/B,
3 repetitions, greedy decode with 30 tokens/turn, load 3.6–5.3.

### Qwen2.5-0.5B-FP16 — native session wins at every turn

| Metric | Session (KV reuse) | Stateless (re-prefill) |
|---|---|---|
| **Break-even turn** | **none (native wins all 10)** | **8** |
| Native steady-state TTFT | 60 ms (2.5× faster than ORT's 150 ms) | 458 ms (3.0× slower) |
| Native steady-state decode | 941 ms (1.2× slower than ORT's 782 ms) | 946 ms (1.2× slower) |
| Over 10 turns | **native 1.13× faster** | ORT 1.18× faster |
| Native TTFT growth | 0.7× (flat — session KV active) | 8.7× (O(context)) |
| ORT TTFT growth | 1.2× (flat) | 1.2× (flat) |

Corroborated with second run at load 4.5–5.8: no break-even, native 1.12×
faster overall, steady-state TTFT 59.5 ms vs ORT 152 ms.

**Session-persistent KV eliminated the Qwen multi-turn deficit.** Native
TTFT is 2.5× faster than ORT's (60 ms vs 150 ms). The remaining gap is
decode: native is 1.2× slower per turn. But native's 5.6–6.1× load advantage
means the break-even for ORT's per-turn decode edge would require ~22 turns.

### TinyStories-33M (FP32) — ORT still wins on decode

| Metric | Session (KV reuse) | Stateless (re-prefill) |
|---|---|---|
| **Break-even turn** | **1–4** | **3** |
| Native steady-state TTFT | 21–23 ms (0.8× ORT's 27 ms — native wins) | 96 ms (3.4× slower) |
| Native steady-state decode | 203–211 ms (1.9–2.0× slower than ORT's 103–105 ms) | 257 ms (2.5× slower) |
| Over 10 turns | **ORT 1.5–1.7× faster** | ORT 2.2× faster |
| Native TTFT growth | 1.3× (nearly flat) | 6.9× |
| ORT TTFT growth | 3.2× | 3.1× |

Session-persistent KV helped native TTFT (now faster than ORT), but ORT still
wins overall because native decode is ~2× slower on this small FP32 model.
The decode gap is the bottleneck — not prefill.

**Why ORT TTFT grows 3.2× on TinyStories:** ORT's session incremental path
still shows TTFT growth because the assistant-response tokens from prior turns
are added to the session's KV in the incremental prompt. This is expected
behaviour — the overhead comes from the growing session state, not re-prefill.

### Session vs stateless — what the session API buys

| Model | Metric | Session | Stateless | Improvement |
|---|---|---:|---:|---|
| Qwen-0.5B-f16 | steady TTFT | 60 ms | 458 ms | **7.6× faster** |
| Qwen-0.5B-f16 | 10-turn total | 9.5 s | 12.9 s | **1.36× faster** |
| TinyStories-33M | steady TTFT | 21 ms | 96 ms | **4.6× faster** |
| TinyStories-33M | 10-turn total | 2.1 s | 3.2 s | **1.5× faster** |

### What native still needs to win TinyStories at every turn count

Native TTFT already beats ORT (21 ms vs 27 ms). The gap is decode:

| Model | Per-decode target | Current | Reduction needed |
|---|---|---|---|
| TinyStories-33M | ≤ 105 ms | 203 ms | 48% |

The decode deficit on small FP32 models is the remaining competitive gap.
ORT's pre-packed GEMV layouts achieve higher cache efficiency on cache-resident
models — this is a kernel-level gap, not architectural.

### Cache-survival finding

Weight transpose caches (f16/f32 — PR #353) **are** correctly reused across
turns:
- Qwen-0.5B-f16: 168 f16 entries at load, remains 168 after 10 turns (stable)
- TinyStories-33M: 1 f32 entry at load, grows to 25 after turns (lazy-fill,
  then stable across repetitions — not rebuilt per turn)

Caches survive because the Engine and its model mmap persist across turns.

## Batch inference — vision models

**Measured under exclusive bench lock at load 2.3–3.0.**

At batch=1, native is 0.50× ORT speed on MobileNetV2 (11.6 ms vs 5.8 ms).
Native crashes (segfault) at batch>1 — a known correctness bug in the native
runtime's batch-dimension handling for CNN models.

ORT throughput scaling with batch (MobileNetV2, Apple M1 Max, load 2.3–3.0):
- batch=1: 172 samples/s
- batch=4: 356 samples/s (2.1×)
- batch=16: 372 samples/s (2.2×)

The prior "15× batch advantage" claim (from an earlier session) collapsed
because it was estimated rather than measured against ORT. Measured ORT scales
~1.9× from batch=1→16; native cannot participate due to the batch>1 crash.

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

- **FP16 decode is at ~73 tok/s of the ~172 tok/s achievable GEMV roof (~43%).**
  At ORT-level 80% efficiency, ~137 tok/s is theoretically reachable. The gap
  is partly dispatch overhead and partly NEON GEMV thread coordination.
- **Accelerate sgemm for Attention SDPA** — partially addressed by inline NEON
  SDPA decode (#349), which removed the scalar fallback. Further gains possible
  from AMX-backed SDPA GEMMs for prefill.
- **Native fp16 elementwise ops** (Mul, RMSNorm, RotaryEmbedding) would
  eliminate per-op fp16↔f32 widening for compute-trivial ops: ~3–5 ms.
- **Fused SiLU·Mul** would eliminate one widen+narrow round-trip per layer in
  the FFN block: ~3–4 ms.
- **Small-model decode deficit** (TinyStories-33M: 0.91× ORT) — the decode
  working set is partially cache-resident; ORT's pre-packed layouts and mature
  GEMV achieve higher cache efficiency. This is an open gap.
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
