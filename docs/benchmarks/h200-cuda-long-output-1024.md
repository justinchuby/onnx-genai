# H200 CUDA — long-output (1024-token) decode benchmark — 2026-07-13

Steady-state decode comparison with a **full 1024 output tokens** per request
(the earlier 128-token runs stopped early under greedy, giving small-sample
decode). Longer output isolates true per-token decode throughput.

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

## Results (median of 3, all emitted 1024 tokens)

| runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | out tokens |
|---|---:|---:|---:|---:|
| onnx-genai CUDA | **99.9** | 223.3 | 4681.0 | 1024 |
| Foundry Local | 134.1 | 452.2 | 2393.8 | 1024 |
| Ollama Q4_K_M | 167.7 | **505.1** | **2191.8** | 1024 |

## Findings

- **Steady-state decode: onnx-genai is ~2× slower** than Foundry on the
  **identical** `model.onnx` (223 vs 452 tok/s) and ~2.3× slower than Ollama's
  GGUF-Q4 (505). Over 1024 tokens this dominates total latency (4.68 s vs
  ~2.2–2.4 s).
- **onnx-genai decode also degrades with context**: 316 tok/s at 128 output
  tokens → 223 tok/s at 1024. Foundry stays flat (~450), so the gap is
  onnx-genai's decode path scaling with sequence length, not just the known
  prefill deficit. This is now the top perf target alongside prefill.
- **onnx-genai still wins TTFT** (99.9 ms) even at this larger `max_tokens`.
- Ollama leads decode and total latency, but on a lighter GGUF-Q4 quant
  (weight-bandwidth advantage over ONNX fp16) — a llama.cpp reference point,
  not a like-for-like runtime win.

## Caveats

- Single-request latency on a quiet machine, warm model residency; not
  concurrent-serving throughput.
- onnx-genai vs Foundry is 1:1 (same ONNX model). Ollama is a different
  format/quant and is labeled as such.
