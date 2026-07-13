# Cross-runtime benchmark — 2026-07-13 (aifx-clou00000I)

> **First end-to-end onnx-genai CUDA run on real Hopper hardware (H200).** Prior
> to this, the H200 CUDA path in `docs/benchmarks/H200-CUDA-runbook.md` had only
> been exercised via CPU fallback. This is a **true 1:1 comparison**: onnx-genai
> and Foundry Local load the **identical `model.onnx`** (Foundry's
> `qwen2.5-0.5b-instruct-cuda-gpu:4` package, fp16 GQA, 24 layers), so any
> difference is the inference runtime, not the model or quantization.

## Setup

| item | value |
|---|---|
| GPU | NVIDIA H200 (`sm_90`), driver 580.105.08, 8× visible (used device 0) |
| ONNX Runtime | `onnxruntime-gpu==1.27.0` PyPI wheel (Release, **CUDA 13** + cuDNN 9), CUDA EP |
| onnx-genai build | `cargo build --release -p onnx-genai-server --features cuda`, `ORT_ROOT` = wheel CUDA libs + official 1.27 headers |
| Model (both runtimes) | `~/.foundry/cache/models/Microsoft/qwen2.5-0.5b-instruct-cuda-gpu-4/v4/model.onnx` (fp16 GQA, `past_present_share_buffer`) |
| onnx-genai flags | `ONNX_GENAI_EP=cuda`, `ONNX_GENAI_DEVICE_KV=1`, `ONNX_GENAI_CUDA_GRAPH=0` |

## Coherence gate & flag isolation (runbook §6)

Verified coherent output ("The capital of France is Paris.") and confirmed GPU 0
utilization (39–63%) during generation. Flag isolation on this model:

| `CUDA_GRAPH` | `DEVICE_KV` | result |
|---|---|---|
| 0 | 0 | ✅ coherent (base CUDA EP) |
| 0 | 1 | ✅ coherent, O(1)/token KV, survives 400-token gen — **benchmarked config** |
| 1 | 0 | ❌ `ORT error: the ort_value must contain a constructed tensor` |
| 1 | 1 | ❌ same graph-capture failure |

**CUDA graph capture fails** on this model: the growing `attention_mask`/KV each
step breaks stable-address replay (a documented runbook limitation, not a
regression). Device-resident fp16 KV works and is the fast, correct path.

## Machine and run metadata

| field | value |
|---|---|
| CPU | unknown |
| cores | 96 |
| OS | Linux 6.6.141.1-1.azl3 x86_64 x86_64 |
| rustc | rustc 1.97.0 (2d8144b78 2026-07-07) |
| git commit | 8370d4781af5618589b7b3895f7abfeb352c971e |
| working tree | dirty |
| power | unknown; record power profile manually |
| run timestamp (Unix) | 1783952941 |
| harness | 1 warmup(s), 5 measured run(s), max_tokens=128, greedy |

## Runtime configuration

| runtime | endpoint | model | format / quantization | execution settings | status |
|---|---|---|---|---|---|
| onnx-genai CUDA | `http://127.0.0.1:8093/v1` | `qwen2.5-0.5b-cuda` | ONNX fp16 GQA (Foundry model.onnx) | H200 CUDA EP; ONNX_GENAI_DEVICE_KV=1; CUDA_GRAPH=0 | available (qwen2.5-0.5b-cuda) |
| Foundry Local | `http://127.0.0.1:39839/v1` | `qwen2.5-0.5b-instruct-cuda-gpu` | ONNX fp16 GQA (same model.onnx) | H200 CUDA (ORT-genai runtime) | available (qwen2.5-1.5b-instruct-cuda-gpu, Phi-4-mini-instruct-cuda-gpu, qwen2.5-0.5b-instruct-cuda-gpu, qwen2.5-coder-7b-instruct-generic-cpu, qwen3.5-9b-generic-cpu, qwen2.5-7b-instruct-cuda-gpu) |

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
| short | 59 | onnx-genai CUDA | **40.6 / 40.8** | 310.86 / 311.88 | **449.1 / 450.2** | **1452.9 / 1454.2** | 128 / 128 |
| short | 59 | Foundry Local | 144.2 / 145.3 | **357.03 / 357.54** | 501.5 / 504.6 | 409.1 / 421.8 | 128 / 128 |
| long | 858 | onnx-genai CUDA | 752.0 / 760.4 | **147.50 / 148.67** | 899.6 / 910.4 | 1140.9 / 1148.0 | 23 / 23 |
| long | 858 | Foundry Local | **143.8 / 146.3** | 118.94 / 121.98 | **319.1 / 321.4** | **5966.6 / 6098.2** | 22 / 22 |

### Findings

- **Short-prompt TTFT: onnx-genai wins decisively** — 40.6 ms vs Foundry's
  144.2 ms (~3.5× lower). Its total latency for a 128-token completion is also
  lower (449 ms vs 501 ms).
- **Steady-state decode: Foundry is ~15% faster** on the short prompt (357 vs
  311 tok/s). onnx-genai's device-KV decode is competitive but not ahead.
- **Long-prompt prefill: onnx-genai is the clear deficit** — at 858 prompt
  tokens its TTFT is 752 ms vs Foundry's 144 ms (~5×). This matches the known
  runbook note: onnx-genai has no fused prefill path, so prefill cost scales
  poorly with prompt length. This is the top optimization target.
- **Caveat on the "long" row:** greedy decode produced only ~22–23 output
  tokens (the model gave a short 3-bullet answer), so the long-prompt *decode
  tok/s* is a small-sample estimate; the long row is meaningful for **prefill/
  TTFT**, not decode.

> Note: the auto-comparison table below is empty because it keys on a runtime
> literally named `onnx-genai`; here the baseline is labeled `onnx-genai CUDA`.
> Read the Results table directly.

## Automatic comparison against onnx-genai

| prompt | competitor | TTFT | decode throughput | total latency | estimated prefill |
|---|---|---:|---:|---:|---:|
| — | — | onnx-genai result unavailable | — | — | — |

## Fairness caveats

- The HTTP layer, model family/size, prompts, and generation policy are common, but ONNX and GGUF quantizations can differ. The runtime table is part of the result and must identify the exact formats; do not compare unlabeled or selectively chosen quants.
- This is single-request latency/decode performance, not concurrent serving throughput. Background load, thermals, power source, and model residency affect results.
- Default runtime threading is intentional because this measures deployment behavior. The single-thread ORT setting used by exact-equality tests is not used here.
