# Native CUDA vs. ORT GenAI CUDA scoreboard — 2026-07-23

## Scope and method

This is the real-weight, dense-model baseline for the native-faster-than-ORT
mandate.  It intentionally excludes DeepSeek-V2-Lite and Qwen3.5-35B-A3B:
their local HF entries are configuration-only and do not contain runnable
weights.

- Revision: `8793ea9e418eadd7aaaadf0d1d2461dac7354a0b` (`origin/main`, includes
  `719d2fe`).
- GPU: physical NVIDIA H200 **5**, selected at 0% utilization and 0 MiB
  allocated before the run; `CUDA_VISIBLE_DEVICES=5`.
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
| Phi-4-mini-instruct, INT4/INT8 | 3.8B | **before: 193.89**; **after: 197.31** | 229.62 canonical; 236.48 direct rerun | **85.93% (-14.07%)** canonical; **83.44% (-16.56%)** direct | `719d2fe` removes the LongRoPE `Greater` + invariant `If` capture seam. Decode was finite and byte-identical across all nine runs. |

Native now beats ORT on all Qwen dense models, including the weight-heavy 7B
variant. Phi's canonical gap narrowed from about **-15.6%** to **-14.07%**;
the same-host comparison against the 236.48 tok/s ORT rerun is **-16.56%**.

## Phi mandate reference and contention caveat

The clean pre-fix Phi mandate reference was **193.89 native tok/s** against
the canonical ORT reference of **229.62 tok/s**: **84.44%**, or **-15.56%**.
On the fixed main, the requested nine-run native rerun measured **197.31
tok/s** (samples: 194.44--202.10), a finite, deterministic decode. The
directly rerun ORT rate remains 236.48 tok/s; it is recorded alongside the
canonical reference because the two runs were not reserved-host A/Bs.

`719d2fe` is the capture-seam fix that collapses Phi's LongRoPE
`Greater`/invariant-`If` path from three captured graph regions to two. The
runner's aggregate graph counters do not expose a per-decode region count;
they reported enabled, 3 captures, 354 replays, and zero fallbacks over two
warmups plus one 120-token inspection run. The shared host is also used by
CPU benchmark jobs and concurrent development work. GPU 5 was 0% utilized
with 0 MiB allocated immediately before and after the timed run, but CPU
scheduling and system-level contention remain a caveat. Do not interpret
the difference from a reserved-host result as a kernel regression.

No model failed to load on either backend.
