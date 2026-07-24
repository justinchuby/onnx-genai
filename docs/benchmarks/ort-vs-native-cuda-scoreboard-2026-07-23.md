# Native CUDA vs. ORT GenAI CUDA scoreboard — 2026-07-23

## Scope and method

This is the real-weight, dense-model baseline for the native-faster-than-ORT
mandate.  It intentionally excludes DeepSeek-V2-Lite and Qwen3.5-35B-A3B:
their local HF entries are configuration-only and do not contain runnable
weights.

- Baseline revision: `8793ea9e418eadd7aaaadf0d1d2461dac7354a0b`
  (`origin/main`, includes `719d2fe`).  Phi after revision:
  `97c1a56` (`perf/phi-ondevice-rope`).
- Phi verification GPU: physical NVIDIA H200 **3**, selected at 0%
  utilization and 4 MiB allocated before and after all timed runs;
  `CUDA_VISIBLE_DEVICES=3`.
- Native: `profile_native`, release `bench-native,cuda` build; CUDA, greedy
  bare `Hello` prompt, 2 warmups, 9 runs, 120 output tokens, steady decode
  window (first 8 tokens excluded).  The displayed native figure is the
  median of the nine per-run throughputs.
- ORT: `onnxruntime-genai-cuda==0.14.1`, CUDA, greedy bare `Hello` prompt, 2
  warmups, 9 runs, 120 tokens.  All four models loaded and generated coherent
  text.  Its figure is the aggregate rate over the nine full decode runs,
  because OGA's public Python loop does not expose the native harness's
  callback/skip window.
- Ratio is native / ORT.  Positive deltas beat ORT.

## Current real-weight results

| model | params | native tok/s (median, N=9) | ORT tok/s (N=9) | native / ORT | notes |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B Instruct, INT4 | 0.5B | 901.02 | 553.68 | **162.73% (+62.73%)** | Pre-fix dense-path median; `719d2fe` does not touch this path. Both backends generated coherent output. |
| Qwen2.5-1.5B Instruct, INT4 | 1.5B | 619.95 | 453.29 | **136.77% (+36.77%)** | Pre-fix dense-path median; `719d2fe` does not touch this path. Both backends generated coherent output. |
| Qwen2.5-7B Instruct, INT4 | 7B | 295.96 | 267.06 | **110.82% (+10.82%)** | Pre-fix dense-path median; `719d2fe` does not touch this path. Both backends generated coherent output. |
| Phi-4-mini-instruct, INT4/INT8 | 3.8B | **before: 204.77**; **after: 321.98** (median of 4 × N=9) | 229.62 canonical; 236.48 direct rerun | **140.22% (+40.22%)** canonical; **136.15% (+36.15%)** direct | Verified `perf/phi-ondevice-rope` (`97c1a56`): LongRoPE branch selection de-hosted. After medians: 321.10–322.83 tok/s; finite output. |

**Headline:** Native CUDA now beats ORT on **all real-weight models**: Qwen
0.5B **+62.7%**, 1.5B **+36.8%**, 7B **+10.8%**, and Phi **+40.2%** after
LongRoPE de-hosting.

## Phi LongRoPE verification and contention caveat

On a confirmed-idle GPU 3, `perf/phi-ondevice-rope` measured **321.98 tok/s**
as the median of four independently launched nine-run steady benchmarks
(per-launch medians: **322.40, 321.56, 321.10, 322.83**; range
**321.10–322.83**). This reproduces the claimed ~322 tok/s result. The same
idle window's own `origin/main` baseline was **204.77 tok/s**, the median of
two nine-run launches (198.02 and 211.52), so the measured before/after lift
is **+57.2%**. This baseline is ours, rather than Deckard's 203.50 tok/s
reference. Two additional after launches were rejected because the harness
detected non-byte-identical greedy tokens across their measured runs, before
they emitted a median; the four reported launches passed that check.

Against the canonical 229.62 tok/s ORT result, native is **+40.22%**; against
the direct 236.48 tok/s ORT rerun it is **+36.15%**. Decode was finite on all
accepted runs. The runner does not expose a captured-*region* count, but its
inspection run reported CUDA graphs enabled, 3 capture events, 354 replays,
and **zero fallbacks** (the three events are two warmups plus the measured
generation, not a region count).

The shared host is also used by CPU benchmark jobs and concurrent development
work. GPU 3 was 0% utilized with 4 MiB allocated immediately before and after
the timed runs (and had no `nvidia-smi pmon` process), but CPU scheduling and
system-level contention remain a caveat. Do not interpret the difference from
a reserved-host result as a kernel regression.

## Pre-on-device Phi decode diagnostic

An Nsight Systems node-trace over two 128-token generations confirms **two**
captured graph regions: four `cuStreamBeginCapture` calls and 508
`cuGraphLaunch` calls, or two launches per each of the 254 decode forwards.
It records 60,414 kernel instances, **236.0 kernels/decode-forward**, taking
**2.948 ms/token**. The uninstrumented native run measured **5.150
ms/token**, leaving about **2.20 ms/token** outside GPU kernels (the Nsight
wall time itself is not used because instrumentation increases it).

Before `97c1a56`, the remaining dominant host cost was the still-eager LongRoPE `If`: the native
op trace reports a **1.935 ms** median per decode-forward. The replayed
`Greater` kernel is only about **1.28 us/token**, and GQA is inside the
captured regions (its GPU prep/attention/merge total is about **0.406
ms/token**), so neither an eager Greater read nor GQA dispatch is the current
first target. The 236 kernels are already submitted through two graph
launches, not individually. `97c1a56` implements the next lever: make the
LongRoPE branch select fully on-device.

No model failed to load on either backend.
