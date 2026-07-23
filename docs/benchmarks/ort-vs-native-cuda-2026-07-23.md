# Native CUDA vs. ORT GenAI CUDA — 2026-07-23

## f0af865 baseline

| Model | Native @128 tok/s (ms/token) | ORT 0.14.1 @128 tok/s (ms/token) | Native / ORT @128 | Native @1024 tok/s (ms/token) | ORT 0.14.1 @1024 tok/s (ms/token) | Native / ORT @1024 | Native HBM roofline @128 / @1024 | Smoothness |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B int4 | 821.10 (1.218) | 731.88 (1.366) | **112.19% BEATS +12.19%** | 771.87 (1.296) | 638.76 (1.566) | **120.84% BEATS +20.84%** | 6.94% / 6.65% | coherent + deterministic: yes |
| Qwen2.5-1.5B int4 | 481.29 (2.078) | 483.15 (2.070) | **99.62% TRAILS -0.38%** | 455.56 (2.195) | 472.09 (2.118) | **96.50% TRAILS -3.50%** | 12.63% / 12.13% | coherent + deterministic: yes |
| Qwen2.5-7B int4 | 230.84 (4.332) | 280.21 (3.569) | **82.38% TRAILS -17.62%** | 222.95 (4.485) | 276.68 (3.614) | **80.58% TRAILS -19.42%** | 27.50% / 26.73% | coherent + deterministic: yes |
| Phi-4-mini int4/int8 | 92.94 (10.759) | 236.56 (4.227) | **39.29% TRAILS -60.71%** | 88.82 (11.259) | 210.25 (4.756) | **42.24% TRAILS -57.76%** | 8.00% / 7.80% | coherent + deterministic: yes |

## Post-fusion (main 0672400)

Native post-fusion measurements were initially blocked at `0672400`; that build
defect was repaired by main `64238b5`. The native values below are from the
unmodified repaired main and supersede the prior blocked status.

| Model | Native @128 tok/s (ms/token) | ORT 0.14.1 @128 tok/s (ms/token) | Native / ORT @128 | Native @1024 tok/s (ms/token) | ORT 0.14.1 @1024 tok/s (ms/token) | Native / ORT @1024 | Native delta vs baseline | Smoothness |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen2.5-0.5B int4 | 792.72 (1.261) | 741.83 (1.348) | **106.86% BEATS +6.86%** | not rerun | 643.13 (1.555) | n/a | **−3.46%** vs 821.10; still beats ORT | coherent + deterministic: yes |
| Qwen2.5-1.5B int4 | 514.05 (1.945) | 487.14 (2.053) | **105.52% BEATS +5.52%** | not rerun | 475.13 (2.105) | n/a | **+6.81%** vs 481.29 | coherent + deterministic: yes |
| Qwen2.5-7B int4 | 252.74 (3.957) | 267.23 (3.742) | **94.58% TRAILS −5.42%** | not rerun | 277.07 (3.609) | n/a | **+9.49%** vs 230.84 (confirms claimed ≈+9.5%) | coherent + deterministic: yes |
| Phi-4-mini int4/int8 | 92.18 (10.848) | 229.62 (4.355) | **40.14% TRAILS −59.86%** | not rerun | 206.60 (4.840) | n/a | **−0.82%** vs 92.94 | coherent + deterministic: yes |

ORT was rerun with `onnxruntime-genai-cuda==0.14.1`, CUDA graph enabled for
Qwen (disabled for Phi), greedy decoding, and
`min_length=max_length=prompt_tokens + output_tokens`. Each table value is the
median of three 120-token or 1016-token steady windows after one warmup; all
three sequences per row were identical and started with `" Paris..."`. The
first 0.5B 128-token set was noisy (737.74, 724.22, 644.86 tok/s), so it was
remeasured; the table uses its clean 741.83, 741.40, 741.95 tok/s set. Host
load after the rerun was 5.17/10.54/29.32;
the concurrent CPU work makes this result useful as an apples-to-apples ORT
rerun. Native used greedy decoding with EOS disabled (therefore always exactly
128 output tokens), CUDA graph and device KV enabled, one warmup, and three
120-token steady windows after the eight-token exclusion. Diagnostics for all
four models reported graph enabled, zero fallbacks, zero KV H2D/D2H calls and
bytes, deterministic streams, and coherent `" Paris..."` continuations.
Native ladder load averages were 29.66/27.01/30.40 before and
19.53/24.81/29.56 after the sequential run; concurrent CPU testing was active.

### 7B fusion isolation

At 128 tokens, 7B is **252.74 tok/s** with the fusion enabled and **230.60
tok/s** with `ONNX_GENAI_CUDA_DISABLE_RMSNORM_FUSION=1`: **+9.60%**. The
off result agrees with the 230.84 tok/s pre-fusion baseline, independently
isolating the win from unrelated changes. A 32-token trace has 537 kernel
events with fusion versus 621 without (−84, −13.5%); specifically,
`SkipSimplifiedLayerNormalization` `skip_rmsnorm_f16_warp_half4` events fall
from 168 to 84 and `gemv_f16_general` from 114 to 58, while 28
`gemm_f16_tiled_rmsnorm` fused events appear. The on/off generated token
streams and decoded continuation are identical. The 0.5B −3.46% regression is
slightly larger than the reported −2.7%, but it remains 6.86% ahead of ORT;
the size-floor gate was not present in `64238b5`.

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
- ORT used the separately installed `onnxruntime-genai-cuda==0.14.1` wheel in a
  fresh venv, greedy with `min_length=max_length=prompt_tokens + output_tokens`
  to prevent early EOS, one warmup, three measured generations, and the same
  eight-token steady-window exclusion. It used the CUDA-12 `libcudart`,
  `libcublas`, and `libcufft` pip-library directories plus CUDA-12-compatible
  cuDNN on `LD_LIBRARY_PATH`; the wheel otherwise fails to load
  `libcublasLt.so.12`. ORT CUDA graph capture was enabled for Qwen. Phi has
  control-flow nodes and ORT 0.14.1 rejects graph capture for it, so its
  graph-off result is its best runnable ORT configuration. All ORT greedy
  continuations were coherent and began `" Paris..."`; the three runs per
  model/length were deterministic. Load averages during ORT samples were
  8.72--17.26.

The roofline uses the requested 3.35 TB/s and
`3.35e12 / (streamed initializer bytes + average cached-KV bytes)`. It excludes
the input embedding table (one row is read at decode), includes the quantized LM
head, and uses average decoded contexts of 72.5 and 520.5 tokens. This gives
rooflines (tok/s) of 11,834/11,608 (0.5B), 3,811/3,756 (1.5B), 840/834 (7B),
and 1,162/1,139 (Phi).

## Where the native gap is

On post-fusion main, Phi-4-mini remains the worst trailing model: native is
59.86% behind ORT at 128 tokens. The f0af865 baseline was 60.71% behind at 128
tokens and 57.76% behind at 1024; 1024 native post-fusion was not rerun. It
also has the largest native absolute per-token latency, so it was profiled.

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

## f0af865 baseline: resolving Qwen2.5-0.5B 815 vs. 459 tok/s

The f0af865 baseline, with the documented graph/device-KV configuration, was
**821.10 tok/s at 128** and **771.87 tok/s at 1024**. The `815` family of
numbers is therefore the correct baseline graph-replay measurement. The `459`
number in the decisions ledger is an earlier post split-K result and is
consistent with the non-graph/eager regime rather than a baseline graph-replay
baseline: the immediately preceding graph-off report measured
434.14/427.65 tok/s. CUDA graph replay removes the large host launch/dispatch
cost of this many-kernel decode path; package/model changes do not explain a
near-2x change. The earlier record did not establish the same enabled-graph
measurement contract, while today's diagnostic explicitly proves it.
