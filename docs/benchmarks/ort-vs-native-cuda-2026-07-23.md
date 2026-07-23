# Native CUDA vs. ORT GenAI CUDA — 2026-07-23

## Pre-fusion baseline @ 4372f1b — 2026-07-23

This is the clean pre-SwiGLU-RMS zero-point-fusion baseline from `4372f1b`,
measured on physical GPU 5 (NVIDIA H200). A concurrent CPU-benchmark team may
have added host-side noise.

| Model | Native tok/s (median of 3) | ORT 0.14.1 tok/s | Delta | Coherent? | Segments / fallbacks |
|---|---:|---:|---:|:---:|---:|
| Qwen2.5-0.5B int4 | 821.35 | 741.83 | **+10.72%** | Yes | 1 / 0 |
| Qwen2.5-1.5B int4 | 586.82 | 487.14 | **+20.46%** | Deterministic; repetitive | 1 / 0 |
| Qwen2.5-7B int4 | 288.64 | 267.23 | **+8.01%** | Yes | 1 / 0 |
| Phi-4-mini int4/int8 | 136.49 | 229.62 | **-40.56%** | Yes | 3 / 0 |

Native samples (tok/s) were 823.18/821.35/814.49, 575.13/586.82/586.88,
288.27/288.81/288.64, and 137.19/134.65/136.49 respectively; no sample was
discarded as an obvious outlier. Each diagnostic run reported one measured
capture, 30 replays, and zero fallbacks. Streams were token-identical; the
known 1.5B repetitive-output divergence was present. The ORT values are the
existing authoritative 0.14.1 medians in the post-fusion table below.
Deltas are `(native - ORT) / ORT * 100`.

## Post-fusion @ 2715151 — 2026-07-23

This rerun measures the asymmetric-int4 zero-point SwiGLU-RMS fusion firing on
Phi at `2715151`, again on physical GPU 5 (NVIDIA H200).

| Model | Native tok/s (median of 3) | ORT 0.14.1 tok/s | Delta | Coherent? | Segments / fallbacks |
|---|---:|---:|---:|:---:|---:|
| Qwen2.5-0.5B int4 | 816.15 | 741.83 | **+10.02%** | Yes | 1 / 0 |
| Qwen2.5-1.5B int4 | 535.87 | 487.14 | **+10.00%** | Deterministic; repetitive | 1 / 0 |
| Qwen2.5-7B int4 | 253.00 | 267.23 | **-5.33%** | Yes | 1 / 0 |
| Phi-4-mini int4/int8 | 166.12 | 229.62 | **-27.65%** | Yes | 3 / 0 |

Native samples (tok/s) were 816.15/817.15/799.00,
535.92/535.87/507.35, 252.93/253.00/253.03, and
164.04/168.99/166.12 respectively. The low 1.5B third sample is a likely
host-noise outlier; its exclusion would not change the reported median.
Repeated 7B sets remained near 253 tok/s, so its regression from the
pre-fusion 288.64 tok/s baseline is reproducible rather than an isolated
sample. A concurrent CPU-benchmark team may have added host-side noise.
Diagnostic runs remained token-identical and reported one measured capture,
30 replays, and zero fallbacks. The known Qwen 1.5B native greedy divergence
remains: its stream is deterministic but falls into repetitive prose.

### Phi fusion A/B

On the same `2715151` binary, fusion ON reached **166.12 tok/s** versus
**156.24 tok/s** with `ONNX_GENAI_CUDA_DISABLE_RMSNORM_FUSION=1`, an exact
**+6.32%** fusion gain. Against the `4372f1b` pre-fusion baseline
(136.49 tok/s), the complete post-fusion stack is **+21.71%** faster.

### Nsight Systems captured-decode profile

`nsys profile --cuda-graph-trace=node` over a 64-token Phi run (one warmup,
one measured generation) exposed the graph-node kernels. The captured decode
kernel-time breakdown was:

| Kernel | GPU kernel time |
|---|---:|
| `matmul_nbits_gemv_int8_f16` | **28.0%** |
| `matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm` | **24.7%** |
| `matmul_nbits_gemv_f16_scales_f16` | **19.8%** |
| `skip_rmsnorm_f16_warp_half4` | **15.0%** |
| GQA attention + prep + merge | **11.7%** |

The next largest removable seam is therefore the standalone RMSNorm feeding
the qkv/down int8 GEMVs: int8 GEMV plus standalone RMSNorm account for
**43.0%** of GPU kernel time. This supports prioritizing the in-progress
int8-fused path; GQA is materially smaller.

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

#### Epilogue-only (64238b5)

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

### Post-SwiGLU fusion (main 05e1fd1)

| Model | Native @128 tok/s (ms/token) | ORT 0.14.1 @128 tok/s (ms/token) | Native / ORT @128 | Change vs. epilogue-only | Smoothness |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B int4 | 818.75 (1.221) | 741.83 (1.348) | **110.37% BEATS +10.37%** | +3.28% | coherent + deterministic: yes |
| Qwen2.5-1.5B int4 | 572.20 (1.748) | 487.14 (2.053) | **117.46% BEATS +17.46%** | +11.31% | coherent + deterministic: yes |
| Qwen2.5-7B int4 | 286.90 (3.486) | 267.23 (3.742) | **107.36% BEATS +7.36%** | +13.52% | coherent + deterministic: yes |
| Phi-4-mini int4/int8 | 93.92 (10.648) | 229.62 (4.355) | **40.90% TRAILS −59.10%** | +1.89% | coherent + deterministic: yes |

The 7B A/B isolates both RMSNorm fusions: enabled is **286.90 tok/s** and
`ONNX_GENAI_CUDA_DISABLE_RMSNORM_FUSION=1` is **230.29 tok/s**, a **+24.58%**
gain. This confirms the claimed approximately 285 tok/s result and moves
native 7B ahead of ORT by 7.36%. A 32-token trace contains 453 events enabled,
down from 537 with the prior epilogue-only fusion and 621 with both fusions
disabled. The enabled trace has `gate_up_swiglu_rmsnorm_prefill` (28) and
`gate_up_swiglu_rmsnorm_fused` (56), while the disabled trace has 168
`skip_rmsnorm_f16_warp_half4` events; its absence confirms the SwiGLU-RMS
fusion fires. The on/off token IDs and decoded `" Paris..."` text are
identical.

The 1.5B 514.05→572.20 tok/s gain is from the SwiGLU-RMS fusion now firing:
its hidden size is 1536, above the 1280 fusion floor, while the preceding
epilogue-only measurement predated that fusion. In contrast, 0.5B's hidden
size is 896, below the floor. Its 818.75 tok/s is 3.28% above the preceding
792.72 measurement, which is run-to-run variance from lower host load
(13.42/13.61/14.77 before and 10.07/12.79/14.47 after), not a new fusion
effect. Its 537-event trace retains the unfused `gate_up_swiglu_*` and
`skip_rmsnorm_f16_warp_half4` variants, confirming the size floor kept the new
SwiGLU fusion inert.

### Post-Phi vectorization (main cf65ea7)

Three-run medians on physical GPU 5 refresh the four-model 128-token ladder
(the earlier ladders described in the Method section used physical GPU 0):

| Model | Native @128 tok/s (ms/token) | ORT 0.14.1 @128 tok/s (ms/token) | Native / ORT @128 | Change vs. 05e1fd1 | Smoothness spot check |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B int4 | 823.45 (1.214) | 741.83 (1.348) | **111.00% BEATS +11.00%** | +0.57% | readable, deterministic; long raw continuation eventually repeats |
| Qwen2.5-1.5B int4 | 574.31 (1.741) | 487.14 (2.053) | **117.89% BEATS +17.89%** | +0.37% | readable, deterministic; long raw continuation eventually repeats |
| Qwen2.5-7B int4 | 287.66 (3.476) | 267.23 (3.742) | **107.65% BEATS +7.65%** | +0.26% | readable, deterministic; raw continuation includes a `Human:` turn |
| Phi-4-mini int4/int8 | 131.40 (7.610) | 229.62 (4.355) | **57.22% TRAILS −42.78%** | **+39.90%** | readable, deterministic; raw continuation includes a `# Exercise` turn |

Phi rises from 93.92 to **131.40 tok/s** after the fp32-gamma vectorized
SkipRMSNorm (`8a0814e`) and int8 FP16-GEMV vectorization (`cf65ea7`). This
closes its native/ORT gap from −59.10% to **−42.78%**. All four timed runs
were deterministic; their real-prompt continuations were readable and
non-garbled, although the known raw-prompt template/repetition behavior is
recorded above rather than called smooth. The three per-run Phi samples were
131.40, 129.35, and 132.00 tok/s. Host load was 22.14/23.02/31.56 before and
16.96/21.79/30.97 after the sequential GPU-5 ladder.

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

On the epilogue-only `64238b5` measurement, Phi-4-mini was 59.86% behind ORT
at 128 tokens; current `05e1fd1` is 59.10% behind. The f0af865 baseline was
60.71% behind at 128 tokens and 57.76% behind at 1024; 1024 native
post-fusion was not rerun. It also has the largest native absolute per-token
latency, so it was profiled.

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
