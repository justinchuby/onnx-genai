# Native CUDA vs. ORT GenAI CUDA baseline — 2026-07-23

## Result: ORT 0.14 CUDA baseline unavailable

This is a clean native baseline, but **not an apples-to-apples comparison**:
the requested ORT GenAI 0.14.x CUDA runtime was unavailable on this host.
The user-site `onnxruntime_genai==0.14.1` imports, but lacks
`libonnxruntime-genai-cuda.so`. The `onnx` conda environment instead has
`0.15.0-dev`; it was deliberately excluded rather than presenting a
cross-version result as ORT 0.14. Consequently every ORT, ratio, and
BEATS/TRAILS cell is `N/A`, not zero or an estimate.

| Model | Native @128 tok/s (ms/token) | Native @1024 tok/s (ms/token) | ORT 0.14 @128 | ORT 0.14 @1024 | Native / ORT | Native HBM roofline @128 / @1024 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B int4 | 821.10 (1.218) | 771.87 (1.296) | N/A | N/A | N/A | 6.94% / 6.65% |
| Qwen2.5-1.5B int4 | 481.29 (2.078) | 455.56 (2.195) | N/A | N/A | N/A | 12.63% / 12.13% |
| Qwen2.5-7B int4 | 230.84 (4.332) | 222.95 (4.485) | N/A | N/A | N/A | 27.50% / 26.73% |
| Phi-4-mini int4/int8 | 92.94 (10.759) | 88.82 (11.259) | N/A | N/A | N/A | 8.00% / 7.80% |

No model can honestly be labelled **BEATS** or **TRAILS** without the
version-matched ORT CUDA package. Re-run this report after installing a CUDA
wheel/package in the `onnx` environment specifically for ORT GenAI 0.14.x.

## Method and validity checks

- Source: current `bench/ort-vs-native-cuda` checkout; release build:
  `cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native`.
- Models were the four requested Foundry `Microsoft/.../v4` or `v5` directories.
- GPU: physical GPU 0 of the shared 8x H200 host; load averages ranged from
  `12.21` to `29.00` during native samples. Phi was remeasured after an initially
  noisy set; the table uses the clean final set (individual samples at 128:
  90.30, 94.13, 92.94 tok/s; at 1024: 91.73, 88.82, 88.62 tok/s).
- Native invocation: greedy `The capital of France is`, `--tokens 128|1024
  --steady --warmups 1 --runs 3 --decode-skip 8 --ep cuda`, with
  `ONNX_GENAI_DEVICE_KV=1`, `ONNX_GENAI_CUDA_GRAPH=1`, and
  `ONNX_GENAI_REQUIRE_CUDA=1`.
- The harness checks that the three greedy token streams are identical. All four
  diagnostic 32-token runs were coherent (`" Paris..."`), deterministic, graph
  enabled, and reported one measured capture, 29 replays, zero fallbacks, and
  zero KV H2D/D2H calls and bytes.

The roofline uses the requested 3.35 TB/s and
`3.35e12 / (streamed initializer bytes + average cached-KV bytes)`. It excludes
the input embedding table (one row is read at decode), includes the quantized LM
head, and uses average decoded contexts of 72.5 and 520.5 tokens. This gives
rooflines (tok/s) of 11,834/11,608 (0.5B), 3,811/3,756 (1.5B), 840/834 (7B),
and 1,162/1,139 (Phi).

## Where the native gap is

There is no ORT result from which to designate a worst *trailing* model.
The lowest native roofline utilization is Qwen2.5-0.5B, while Phi has the
largest absolute per-token latency, so Phi was profiled as the practical
investigation target.

`ONNX_GENAI_PROFILE_OPS=1` on Phi (32-token diagnostic) is intrusive and
therefore not used for the throughput table. Its warm operator summaries show
the high-call-count work is `Cast` (257 calls; 7.884 ms in one 44.120-ms
instrumented step), `GroupQueryAttention` (32; 23.472 ms), `MatMulNBits`
(161; 2.118 ms), and `SkipSimplifiedLayerNormalization` (64; 0.936 ms).
The profiler reports host-side operator spans, not per-CUDA-kernel timings or a
kernel/launch split; therefore a numeric non-kernel per-token overhead cannot
be derived honestly from it. The non-instrumented Phi wall time is 10.759 ms
per token at 128, including all launch/dispatch overhead. A GPU-kernel trace
(Nsight or equivalent) is required to split that value.

## Resolving Qwen2.5-0.5B: 815 vs. 459 tok/s

Current main, with the documented graph/device-KV configuration, is
**821.10 tok/s at 128** and **771.87 tok/s at 1024**. The `815` family of
numbers is therefore the correct current graph-replay measurement. The `459`
number in the decisions ledger is an earlier post split-K result and is
consistent with the non-graph/eager regime rather than a current graph-replay
baseline: the immediately preceding graph-off report measured
434.14/427.65 tok/s. CUDA graph replay removes the large host launch/dispatch
cost of this many-kernel decode path; package/model changes do not explain a
near-2x change. The earlier record did not establish the same enabled-graph
measurement contract, while today's diagnostic explicitly proves it.
