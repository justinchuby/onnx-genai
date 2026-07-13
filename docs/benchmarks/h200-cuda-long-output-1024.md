# H200 CUDA — long-output (1024-token) decode benchmark — 2026-07-13

Steady-state decode comparison with a **full 1024 output tokens** per request
(the earlier 128-token runs stopped early under greedy, giving small-sample
decode). Longer output isolates true per-token decode throughput.

> **Update (post-fix):** onnx-genai decode improved from **223 → 413 tok/s**
> after enabling the runtime-owned shared KV buffer (O(1)/token) path for this
> model. See [After the shared-buffer fix](#after-the-shared-buffer-fix) below.

## Setup

| item | value |
|---|---|
| GPU | NVIDIA H200 (`sm_90`), driver 580.105.08 |
| Model (onnx-genai + Foundry) | **identical** ONNX `model.onnx`, Qwen2.5-0.5B, fp16 GQA KV (Foundry `qwen2.5-0.5b-instruct-cuda-gpu:4`) |
| Model (Ollama) | Qwen2.5-0.5B-Instruct **GGUF Q4_K_M** (llama.cpp), SHA `74a4da8c…` — different quant |
| onnx-genai | ORT `onnxruntime-gpu 1.27.0` (CUDA 13); `EP=cuda DEVICE_KV=1 CUDA_GRAPH=0` |
| Foundry Local | 0.10.0, ORT-genai CUDA |
| Ollama | 0.31.2, llama.cpp CUDA, all layers on GPU |
| Protocol | streaming; `temperature=0 top_p=1 seed=0`; `max_tokens=1024`; 1 warmup + 3 runs (median); long "write a 1500-word essay" prompt so all runtimes emit the full 1024 tokens |

## Results — before the fix (median of 3, all emitted 1024 tokens)

| runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | out tokens |
|---|---:|---:|---:|---:|
| onnx-genai CUDA | **99.9** | 223.3 | 4681.0 | 1024 |
| Foundry Local | 134.1 | 452.2 | 2393.8 | 1024 |
| Ollama Q4_K_M | 167.7 | **505.1** | **2191.8** | 1024 |

### Findings (before the fix)

- **Steady-state decode: onnx-genai was ~2× slower** than Foundry on the
  **identical** `model.onnx` (223 vs 452 tok/s) and ~2.3× slower than Ollama's
  GGUF-Q4 (505). Over 1024 tokens this dominated total latency.
- **onnx-genai decode degraded with context**: 316 tok/s at 128 output tokens →
  223 tok/s at 1024. Foundry stayed flat (~450). Root cause: onnx-genai fell
  into the growing `ZeroCopyRebind` KV path (per-token cost scales with context)
  because the Foundry model ships no `inference_metadata.yaml`, so the O(1)
  shared-buffer KV path was never selected.

## After the shared-buffer fix

Two changes enable the runtime-owned, max-length shared KV buffer path
(`present.* -> past_key_values.*` aliased in place, O(1)/token) for this model:

1. **Allocator lifetime fix** (`onnx-genai-ort`): the device KV allocator that
   backs the shared buffers is now retained for the lifetime of the KV `Value`s,
   fixing a use-after-free SIGSEGV at session close that made the shared-buffer
   path unusable on CUDA.
2. **`genai_config.json` compat layer** (`onnx-genai-genai-config`): ORT-genai /
   Foundry models carry `search.past_present_share_buffer: true` in
   `genai_config.json` but no `inference_metadata.yaml`. onnx-genai now converts
   that config into native inference metadata, auto-selecting the shared-buffer
   path (KV dtype read from the ONNX graph). Our own `inference_metadata.yaml`
   remains the preferred source when present.

Config: `EP=cuda DEVICE_KV=1 CUDA_GRAPH=0`, **no hand-authored metadata** — the
fast path is auto-detected from the shipped `genai_config.json`.

| runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | out tokens |
|---|---:|---:|---:|---:|
| onnx-genai CUDA | **103.3** | **412.8** | 2581.2 | 1024 |
| Foundry Local | 140.0 | 453.5 | 2396.1 | 1024 |
| Ollama Q4_K_M | 170.8 | 504.4 | **2198.9** | 1024 |

### Findings (after the fix)

- **onnx-genai decode: 223 → 413 tok/s (+85%)** on the identical model. Decode
  no longer degrades with context (the O(1) shared buffer replaces the growing
  rebind path). Total latency for 1024 tokens dropped from 4.68 s → 2.58 s.
- **onnx-genai is now ~91% of Foundry** (413 vs 453) on the same ONNX model and
  the same ORT CUDA kernels, and still wins TTFT (103 vs 140 ms).
- **Remaining ~40 tok/s gap to Foundry** is orchestration overhead. Foundry
  runs with CUDA graphs; onnx-genai still uses `CUDA_GRAPH=0` because the
  growing `attention_mask` breaks graph capture. Making the decode step bind a
  fixed-capacity `attention_mask` (now that KV is static) should let CUDA-graph
  replay close or eliminate that gap — the next optimization.
- Ollama leads decode/total but on a lighter GGUF-Q4 quant (weight-bandwidth
  advantage over ONNX fp16); it is a llama.cpp reference point, not a
  like-for-like runtime win. The fair 1:1 ceiling for this model is Foundry.

## Caveats

- Single-request latency on a quiet machine, warm model residency; not
  concurrent-serving throughput.
- onnx-genai vs Foundry is 1:1 (same ONNX model). Ollama is a different
  format/quant and is labeled as such.

