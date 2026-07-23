# Native CUDA vs. ORT GenAI CUDA — 2026-07-23

## Pre-fusion baseline @ 4372f1b — 2026-07-23

This is the clean pre-SwiGLU-RMS zero-point-fusion baseline from `4372f1b`,
measured on physical GPU 5 (NVIDIA H200). A concurrent CPU-benchmark team may
have added host-side noise.

| Model | Native tok/s (median of 3) | ORT 0.14.1 tok/s | Delta | Coherent? | Segments / fallbacks |
|---|---:|---:|---:|:---:|---:|
| Qwen2.5-0.5B int4 | 821.35 | 741.83 | **+10.72%** | Yes | 1 / 0 |
| Qwen2.5-1.5B int4 | 586.82 | 487.14 | **+20.46%** | Yes | 1 / 0 |
| Qwen2.5-7B int4 | 288.64 | 267.23 | **+8.01%** | Yes | 1 / 0 |
| Phi-4-mini int4/int8 | 136.49 | 229.62 | **-40.56%** | Yes | 3 / 0 |

Native samples (tok/s) were 823.18/821.35/814.49, 575.13/586.82/586.88,
288.27/288.81/288.64, and 137.19/134.65/136.49 respectively; no sample was
discarded as an obvious outlier. Each diagnostic run reported one measured
capture, 30 replays, zero fallbacks, and coherent greedy text. The ORT values
are the existing authoritative 0.14.1 medians in the post-fusion table below.
Deltas are `(native - ORT) / ORT * 100`.

## Post-int8-fused @ c34f813 — 2026-07-23

This milestone adds the Phi int8 qkv/down RMSNorm-GEMV fusion on top of the
zero-point-aware SwiGLU-RMS fusion. Measurements used physical GPU 5
(NVIDIA H200), one warmup, three 120-token steady windows after skipping the
first eight emitted tokens, and CUDA graph/device-KV enabled.

| Model | Native tok/s (median of 3) | ORT 0.14.1 tok/s | Delta | Coherent? | Segments / fallbacks |
|---|---:|---:|---:|:---:|---:|
| Qwen2.5-0.5B int4 | 819.42 | 741.83 | **+10.46%** | Yes | 1 / 0 |
| Qwen2.5-1.5B int4 | 584.45 | 487.14 | **+19.98%** | Deterministic; repetitive | 1 / 0 |
| Qwen2.5-7B int4 | 287.03 | 267.23 | **+7.41%** | Yes | 1 / 0 |
| DeepSeek-Coder-1.3B int4 | 728.26 | 646.88 | **+12.58%** | Yes; coherent code | 1 / 0 |
| Phi-4-mini int4/int8 | 171.16 | 229.62 | **-25.46%** | Yes | 3 / 0 |

The DeepSeek-Coder value is the dedicated native/ORT CUDA-graph comparison;
a `c34f813` regression guard measured 730.20/729.41/729.39 tok/s (median
729.41), confirming it remains intact. Native samples for the other rows were
819.39/819.48/819.42, 584.45/584.54/584.07,
287.06/286.47/287.03, and 170.31/174.57/171.16 respectively. No sample was
discarded as an obvious outlier. Diagnostics reported one measured capture,
zero fallbacks, and zero KV transfers for every model; Phi retains three
captured segments around its two control-flow seams. Sampled host load averages
ranged from 1.62 to 2.38, and physical GPU 5 remained exclusive.

### Phi full-fusion A/B

On `c34f813`, Phi fusion ON is **171.16 tok/s** versus **162.78 tok/s** with
`ONNX_GENAI_CUDA_DISABLE_RMSNORM_FUSION=1`, an exact **+5.15%** full-fusion
gain. The complete optimization stack is **+25.40%** versus the clean
`4372f1b` pre-fusion baseline of 136.49 tok/s.

### Nsight Systems captured-decode profile

`nsys profile --cuda-graph-trace=node` over a 64-token Phi run exposed the
captured graph-node kernels. GPU kernel time is now:

| Kernel | GPU kernel time |
|---|---:|
| zero-point fused gate-up/SwiGLU/RMSNorm int4 GEMV | **31.9%** |
| zero-point fused int8 GEMV + RMSNorm | **18.5%** |
| zero-point int4 GEMV | **15.7%** |
| remaining standalone int8 GEMV | **13.3%** |
| GQA attention + prep + merge | **12.7%** |
| zero-point int4 GEMV + RMSNorm | **7.1%** |

The former standalone `skip_rmsnorm_f16_warp_half4` hotspot is absent from
captured decode, confirming the int8 fusion landed. Int4 GEMV variants now
consume **54.7%** in aggregate, led by the 31.9% fused gate-up kernel.
Optimizing its zero-point int4 dequantization/GEMV is therefore the highest
impact next lever; the remaining standalone int8 down-projection is a smaller
13.3% follow-up, approximately tied with aggregate GQA.

## GLM-4-9B + DeepSeek-R1-Distill-1.5B @ ad49bf3 — 2026-07-23

| Model | Native tok/s (median of 3) | ORT 0.14.1 tok/s | Delta | Coherent? | Segments / fallbacks |
|---|---:|---:|---:|:---:|---:|
| DeepSeek-R1-Distill-Qwen-1.5B int4 | 576.41 | 489.19 | **+17.83%** | Yes; repetitive | 1 / 0 |
| GLM-4-9B GPTQ int4 | 110.34 | **Cannot load** | N/A | Yes | 41 / 0 |

DeepSeek-R1 samples were 576.40/576.49/576.41 tok/s natively and
491.86/488.29/489.19 tok/s under ORT GenAI 0.14.1 with CUDA graph enabled.
Its greedy continuation is readable but exhibits the same benign fp16
knife-edge repetition caveat as other 1.5B-family measurements.

GLM-4 native samples were 110.34/109.96/110.36 tok/s. The Mobius output lacks
an ORT `genai_config.json`; after supplying an equivalent scratch
configuration, ORT GenAI 0.14.1 still rejects the model because its bundled
GQA schema does not recognize the required partial-RoPE
`rotary_embedding_dim` attribute.
ORT therefore never reaches CUDA graph capture, so there is no meaningful ORT
throughput or segment count. The native EP runs coherent GLM-4 output where
ORT 0.14.1 cannot load the graph.

Native GLM-4 capture installs **41 segments** around **40 eager fused-MLP
gate/up activation `Split` seams (one per layer)**, with zero fallbacks.
Despite that fragmentation, graph capture raises throughput from **85.51 to
110.34 tok/s** (**+29.04%**) versus forced eager execution. Eliminating the
host-reading, stream-synchronizing `Split` seams is the open GLM-4 performance
lever; Batty is analyzing capture defragmentation.

## GLM-4 static-Split capture @ bd9b3a7 — 2026-07-23

The EP-side static single-input `Split` capture path collapses the existing
GLM-4 export from **41 captured segments / 40 eager seams** to **1 captured
segment / 0 eager seams**, with zero fallbacks. This is a general EP
improvement: any graph whose split sizes and output shapes are static can
benefit without requiring a model-specific rewrite.
The 40 capture-breaking nodes are the fused-MLP gate/up activation Split
(one per layer), `Split(axis=-1, num_outputs=2)` on `gate_up_proj`, named
`model/layers.N/mlp/Split_node_*`; they are not RoPE splits.

| Model | Native tok/s (median of 3) | Prior native | Change | Coherent? | Segments / fallbacks |
|---|---:|---:|---:|:---:|---:|
| GLM-4-9B GPTQ int4 | **118.85** | 110.34 | **+7.71%** | Yes | **1 / 0** |

The measured samples were 118.85/118.41/118.88 tok/s. The third run had a
noisy 131.144-ms prefill, but its 118.88 tok/s steady decode was not an
outlier. ORT GenAI 0.14.1 still cannot load this graph because its bundled GQA
attention schema rejects the required partial-RoPE `rotary_embedding_dim`
attribute. This separate GQA-schema incompatibility is unrelated to the
fused-MLP Split seams.
Relative to forced eager execution at 85.51 tok/s, whole-graph capture is now
**+38.99%**.

The equivalent Mobius-side MLP pre-split in PR #424 (Chew-approved, reported
**+7.1%**) removes the same structural seam at graph-emission time and remains
pending Justin's merge.

Regression guards remained stable: Qwen2.5-7B measured **286.95 tok/s** and
DeepSeek-Coder-1.3B measured **728.53 tok/s**; both retained one captured
segment, coherent deterministic output, and zero fallbacks. Host load samples
were 6.47--6.49, and physical GPU 5 remained exclusive.

## Phi split-K + GQA stacked @ d8dd707 — 2026-07-23

This milestone stacks asymmetric-zero-point split-K int4 GEMV (`caaf85a`) and
occupancy-aware fp16 GQA decode (`d8dd707`) on the complete Phi fusion stack.
Measurements used physical GPU 5 (NVIDIA H200), one warmup, 120-token steady
windows after skipping the first eight emitted tokens, and CUDA graph/device-KV
enabled.

| Model | Native tok/s (median of 3–5) | ORT 0.14.1 tok/s | Delta | Coherent? | Segments / fallbacks |
|---|---:|---:|---:|:---:|---:|
| Qwen2.5-0.5B int4 | 903.05 | 741.83 | **+21.73%** | Yes | 1 / 0 |
| Qwen2.5-1.5B int4 | 622.20 | 487.14 | **+27.73%** | Deterministic; repetitive | 1 / 0 |
| Qwen2.5-7B int4 | 295.32 | 267.23 | **+10.51%** | Yes | 1 / 0 |
| DeepSeek-Coder-1.3B int4 | 792.85 | 646.88 | **+22.57%** | Yes; coherent code | 1 / 0 |
| GLM-4-9B GPTQ int4 | 118.85 | **Cannot load** | N/A | Yes | 1 / 0 |
| Phi-4-mini int4/int8 | **184.27** | 229.62 | **-19.75%** | Yes | 3 / 0 |

Phi samples were 178.23/184.27/184.38/184.00/184.49 tok/s. The first run was
an obvious low warm-state outlier, but the five-run median is unaffected. The
new result is **+7.66%** over the prior 171.16 tok/s milestone and compresses
the ORT gap from -25.46% to **-19.75%**. Phi retains three captured segments
around its `Greater` and `If` seams, with zero fallbacks.

Qwen samples were 903.13/902.71/903.05, 622.32/621.95/622.20, and
295.32/295.47/294.97 tok/s. DeepSeek-Coder samples were
790.55/792.91/792.85 tok/s. The current GLM-4 regression guard measured
120.93/120.94/120.96 tok/s (median 120.94), consistent with its authoritative
118.85 tok/s row. Every guard retained one captured segment and zero
fallbacks. Host load averages ranged from 0.88 to 3.44, and physical GPU 5
remained exclusive.

### Phi full-fusion A/B

Fusion ON is **184.27 tok/s** versus **167.64 tok/s** with
`ONNX_GENAI_CUDA_DISABLE_RMSNORM_FUSION=1`, an exact **+9.92%** current-stack
gain. Both paths produced coherent greedy output.

### Nsight Systems captured-decode profile

`nsys profile --cuda-graph-trace=node` over a 128-token Phi run exposed the
captured graph-node kernels. Percentages below use only kernels with a CUDA
graph node ID, excluding prefill:

| Kernel | Captured-decode GPU kernel time |
|---|---:|
| fused gate-up/SwiGLU/RMSNorm zero-point int4 GEMV | **33.3%** |
| fused int8 GEMV + RMSNorm | **19.4%** |
| remaining standalone int8 GEMV | **13.9%** |
| split-K zero-point int4 GEMV | **12.1%** |
| GQA attention + merge + prep | **13.1%** |
| zero-point int4 GEMV + RMSNorm | **7.4%** |
| miscellaneous captured kernels | **0.7%** |

The split-K kernel averages **8.11 us**. The occupancy-aware GQA attention core
averages **5.89 us**, but merge (**4.63 us**) and prep (**2.65 us**) leave the
aggregate GQA share at 13.1%. The fused gate-up/SwiGLU/RMSNorm kernel is now
the clear single-kernel leader at 33.3%; all int4 variants total **52.8%**.
Because that gate-up grid is not occupancy-starved, the next lever should
target its dequantization/GEMV efficiency rather than applying split-K
mechanically. Fusing or reducing GQA prep+merge is a secondary, bounded lever.

## Comprehensive refresh @ bd05b75 — 2026-07-23

This refresh covers the complete currently-runnable H200 model matrix after
the fused int4-zp gate-up software prefetch (`76e35b2`), graph-capturable QMoE
(`0175a12`), and GLM-5.2 logical-mask repair (`bd05b75`). Native measurements
used physical GPU 5, greedy decoding, one warmup, and three 120-token steady
windows after the eight-token exclusion. The synthetic GLM-5.2 models use five
56-token steady windows.

| Model | Native tok/s | ORT GenAI 0.14.1 tok/s | Delta | Coherent? | Native segments / fallbacks | Native capture |
|---|---:|---:|---:|:---:|---:|:---:|
| Qwen2.5-0.5B int4 | 900.78 | 735.47 | **+22.48%** | Yes | 1 / 0 | ON |
| Qwen2.5-1.5B int4 | 622.60 | 486.26 | **+28.04%** | Deterministic; repetitive | 1 / 0 | ON |
| Qwen2.5-7B int4 | 295.19 | 283.76 | **+4.03%** | Yes | 1 / 0 | ON |
| DeepSeek-Coder-1.3B int4 | 793.01 | 653.93 | **+21.27%** | Yes; coherent code | 1 / 0 | ON |
| DeepSeek-R1-Distill-Qwen-1.5B int4 | 622.69 | 488.17 | **+27.56%** | Yes; repetitive | 1 / 0 | ON |
| GLM-4-9B GPTQ int4 | 120.80 | **Cannot load** — GQA rejects `rotary_embedding_dim` | N/A | Yes | 1 / 0 | ON |
| Phi-4-mini int4/int8 | **188.54** | 229.62 | **-17.89%** | Yes | 3 / 0 | ON |
| GLM-5.2-tiny dense | 70.63 | **Cannot load** — no compatible GLM-5.2/DSA config | N/A | Yes; synthetic token stream | N/A / 0 | OFF (automatic) |
| GLM-5.2-tiny q4 | 148.58 | **Cannot load** — no compatible GLM-5.2/DSA config | N/A | Yes; synthetic token stream | N/A / 0 | OFF (automatic) |
| GLM-5.2-tiny QMoE | **174.41** | **Cannot load** — no compatible GLM-5.2/DSA+QMoE config | N/A | Yes; synthetic token stream | N/A / 0 | OFF (automatic) |

Fresh ORT medians use CUDA graph for Qwen and both DeepSeek-family rows.
The fresh Qwen-7B ORT median is 283.76 tok/s, **+6.19%** above the earlier
267.23 reference, so this section's native margin is intentionally the more
conservative +4.03%.
Phi uses the requested canonical 229.62 tok/s graph-off reference because
ORT rejects CUDA graph capture for its control-flow graph; a same-session
graph-off probe measured 239.43 tok/s and was not substituted into the
canonical comparison. GLM-4 remains an important capability differentiator:
native runs it coherently in one captured segment at 120.80 tok/s, while ORT
0.14.1 cannot initialize the graph.

Phi has progressed from **-59.86%** versus ORT at session start to **-17.89%**,
now reaching 188.54 tok/s. The latest software prefetch adds **+2.32%** over
the prior 184.27 tok/s stacked split-K/GQA milestone.

GLM-5.2 MoE is now a native-CUDA capability milestone: the tiny dense, q4,
and fused-QMoE graphs all execute DSA indexing and decode end-to-end with zero
fallbacks. Their numeric-token output is deterministic and structurally valid;
these synthetic models are not semantic-quality or beat-ORT claims. The QMoE
kernel itself is CUDA-graph-capturable, but model-level capture is
automatically disabled for all three exports because their bindings expose a
growing logical prefix whose launch geometry cannot safely be replayed.

Host load rose from 3.83 to 9.07 during the matrix. Large-model samples were
tightly grouped. GLM-5.2 dense run 4 (48.01 vs 70.63 tok/s median) and q4 run
4 (124.82 vs 148.58 median) were clear contention outliers; neither affects
the five-run median. A separate `profile_native` job later overlapped an
additional Phi check and depressed that check to 173.54 tok/s; after it exited,
the authoritative five-run Phi samples were
186.53/189.75/186.35/190.52/188.54 tok/s (median 188.54). Physical GPU 5
remained exclusive throughout.

*Phi may gain a further increment from an in-flight fused-int8 split-K
(Deckard) — to be appended.*

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
