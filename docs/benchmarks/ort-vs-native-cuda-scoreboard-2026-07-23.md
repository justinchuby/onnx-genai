# Native CUDA vs. ORT GenAI CUDA scoreboard — 2026-07-23

## Scope and method

This is the real-weight, dense-model baseline for the native-faster-than-ORT
mandate.  It intentionally excludes DeepSeek-V2-Lite and Qwen3.5-35B-A3B:
their local HF entries are configuration-only and do not contain runnable
weights.

- Revision: `10734044787a9cb451317805a526fa758744084c` (`origin/main`).
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
| Qwen2.5-0.5B Instruct, INT4 | 0.5B | 901.02 | 553.68 | **162.73% (+62.73%)** | Both backends loaded and generated coherent output. |
| Qwen2.5-1.5B Instruct, INT4 | 1.5B | 619.95 | 453.29 | **136.77% (+36.77%)** | Both backends loaded and generated coherent output. |
| Qwen2.5-7B Instruct, INT4 | 7B | 295.96 | 267.06 | **110.82% (+10.82%)** | Both backends loaded and generated coherent output. |
| Phi-4-mini-instruct, INT4/INT8 | 3.8B | 186.19 | 236.48 | **78.73% (-21.27%)** | Both backends loaded and generated coherent output; native remains the only mandate miss. |

Native already beats ORT for every runnable Qwen model, including the
weight-heavy 7B variant.  Phi is the sole laggard.

## Phi mandate reference and contention caveat

The standing clean Phi mandate reference is **193.89 native tok/s** against
the canonical ORT reference of **229.62 tok/s**: **84.44%**, or
**-15.56%**.  This is the Phi gap Deckard's concurrent capture-seam work in
`executor.rs` is intended to address.

The live nine-run Phi median above was lower (186.19 tok/s; -21.27% versus
the directly rerun 236.48 tok/s ORT rate).  Its native samples ranged from
160.12 to 191.95 tok/s despite GPU 5 being idle before and after the run.
The host is shared with CPU benchmark jobs and concurrent development work,
so CPU scheduling and system-level contention remain a caveat.  The raw
live measurements are retained as the authoritative snapshot for this
revision; do not interpret the 5.71-point Phi difference from the standing
reference as a kernel regression without a reserved-host rerun.

No model failed to load on either backend.
