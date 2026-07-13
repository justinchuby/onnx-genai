# Cross-runtime benchmark — 2026-07-13 (aifx-clou00000I)

> Three CUDA runtimes on one **NVIDIA H200**, Qwen2.5-0.5B-Instruct.
>
> - **onnx-genai** and **Foundry Local** load the **identical** ONNX
>   `model.onnx` (fp16 GQA KV) → true 1:1 runtime comparison.
> - **Ollama** runs the same weights but as a **GGUF Q4_K_M** (llama.cpp) —
>   a **different quantization/format**, so it is *not* apples-to-apples on the
>   model; it is included as a llama.cpp reference point. GGUF Q4 is lighter
>   weight-bandwidth than fp16, which flatters its decode throughput.

## Setup

| item | value |
|---|---|
| GPU | NVIDIA H200 (`sm_90`), driver 580.105.08 (8× visible) |
| onnx-genai | ORT `onnxruntime-gpu 1.27.0` (CUDA 13 + cuDNN 9); `EP=cuda DEVICE_KV=1 CUDA_GRAPH=0` |
| Foundry Local | 0.10.0, ORT-genai CUDA; same `model.onnx` as onnx-genai |
| Ollama | 0.31.2, llama.cpp CUDA (all layers on GPU); GGUF Q4_K_M, SHA `74a4da8c…` |

## Findings

- **Decode throughput: Ollama wins big** — 506 tok/s vs 363 (Foundry) vs 316
  (onnx-genai) on the short prompt. This is largely the GGUF-Q4 vs ONNX-fp16
  weight-bandwidth advantage plus llama.cpp's tuned decode kernels, **not** a
  like-for-like runtime win.
- **Short-prompt TTFT: onnx-genai wins** — 38.6 ms vs 137 ms (Foundry) vs
  168 ms (Ollama).
- **Long-prompt prefill: Foundry wins, onnx-genai is the deficit** — TTFT at
  858 tokens: Foundry 139 ms, Ollama 174 ms, onnx-genai 717 ms (no fused
  prefill path; top optimization target).
- **1:1 (onnx-genai vs Foundry, same model.onnx):** onnx-genai is much better at
  short-prompt TTFT, Foundry is much better at long-prompt prefill, decode is
  within ~15% (Foundry ahead).
- *Long-row decode is small-sample (~22–24 output tokens; greedy gave short
  answers) — read the long rows for prefill/TTFT, not decode.*

## Machine and run metadata

| field | value |
|---|---|
| CPU | unknown |
| cores | 96 |
| OS | Linux 6.6.141.1-1.azl3 x86_64 x86_64 |
| rustc | rustc 1.97.0 (2d8144b78 2026-07-07) |
| git commit | cc37ab35b150c305a01e05097d47b533804d7b67 |
| working tree | dirty |
| power | unknown; record power profile manually |
| run timestamp (Unix) | 1783953824 |
| harness | 1 warmup(s), 5 measured run(s), max_tokens=128, greedy |

## Runtime configuration

| runtime | endpoint | model | format / quantization | execution settings | status |
|---|---|---|---|---|---|
| onnx-genai CUDA | `http://127.0.0.1:8093/v1` | `qwen2.5-0.5b-cuda` | ONNX fp16 GQA (Foundry model.onnx) | H200 CUDA EP; DEVICE_KV=1; CUDA_GRAPH=0 | available (qwen2.5-0.5b-cuda) |
| Foundry Local | `http://127.0.0.1:39839/v1` | `qwen2.5-0.5b-instruct-cuda-gpu` | ONNX fp16 GQA (same model.onnx) | H200 CUDA (ORT-genai runtime) | available (qwen2.5-1.5b-instruct-cuda-gpu, Phi-4-mini-instruct-cuda-gpu, qwen2.5-0.5b-instruct-cuda-gpu, qwen2.5-coder-7b-instruct-generic-cpu, qwen3.5-9b-generic-cpu, qwen2.5-7b-instruct-cuda-gpu) |
| Ollama | `http://127.0.0.1:11434/v1` | `qwen05-cuda:latest` | GGUF Q4_K_M (llama.cpp) | H200 CUDA; all layers on GPU; SHA 74a4da8c | available (qwen05-cuda:latest) |

## Methodology

- OpenAI `POST /v1/chat/completions` streaming API; fixed system prompt; `temperature=0`, `top_p=1`, `seed=0`, and `max_tokens=128`.
- 1 warmup run(s) were discarded, followed by 5 measured run(s). Cells show **median / p90**; the bold cell is the median winner for that prompt and metric.
- TTFT is request start to first non-empty streamed content. Total latency ends when the response stream closes after `[DONE]`.
- Decode throughput excludes TTFT: `(generated_tokens - 1) / (total - TTFT)`. Generated tokens use streamed `usage.completion_tokens` when supplied, otherwise one non-empty content event is counted as one token.
- Estimated prefill throughput is `rendered_prompt_tokens / TTFT`; it includes HTTP, queueing, chat-template processing, and first-token decode, so treat it as an API-level estimate rather than kernel-only prefill speed.
- All runtimes receive the same explicit system/user messages. Prompt token counts use the Qwen2.5 chat template and tokenizer.

## Results

| prompt | prompt tokens | runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | estimated prefill tok/s ↑ | output tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| short | 59 | onnx-genai CUDA | **38.6 / 39.4** | 315.90 / 316.14 | 441.4 / 444.4 | **1526.6 / 1568.2** | 128 / 128 |
| short | 59 | Foundry Local | 137.1 / 138.9 | 362.72 / 363.42 | 486.8 / 495.4 | 430.4 / 431.7 | 128 / 128 |
| short | 59 | Ollama | 168.5 / 174.4 | **506.47 / 507.57** | **418.7 / 426.3** | 350.2 / 359.7 | 128 / 128 |
| long | 858 | onnx-genai CUDA | 717.4 / 727.4 | 160.64 / 172.04 | 852.4 / 860.1 | 1196.1 / 1199.9 | 23 / 23 |
| long | 858 | Foundry Local | **138.6 / 139.0** | 123.31 / 123.49 | 309.3 / 310.8 | **6189.9 / 6250.3** | 22 / 22 |
| long | 858 | Ollama | 174.1 / 185.8 | **467.64 / 469.20** | **223.4 / 234.9** | 4928.7 / 5082.2 | 24 / 24 |

## Automatic comparison against onnx-genai

| prompt | competitor | TTFT | decode throughput | total latency | estimated prefill |
|---|---|---:|---:|---:|---:|
| — | — | onnx-genai result unavailable | — | — | — |

## Fairness caveats

- The HTTP layer, model family/size, prompts, and generation policy are common, but ONNX and GGUF quantizations can differ. The runtime table is part of the result and must identify the exact formats; do not compare unlabeled or selectively chosen quants.
- This is single-request latency/decode performance, not concurrent serving throughput. Background load, thermals, power source, and model residency affect results.
- Default runtime threading is intentional because this measures deployment behavior. The single-thread ORT setting used by exact-equality tests is not used here.
