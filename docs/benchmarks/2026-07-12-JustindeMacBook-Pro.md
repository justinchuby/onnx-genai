# Cross-runtime benchmark — 2026-07-12 (JustindeMacBook-Pro)

## Machine and run metadata

| field | value |
|---|---|
| CPU | Apple M1 Max |
| cores | 10 |
| OS | Darwin 25.5.0 arm64 arm |
| rustc | rustc 1.97.0 (2d8144b78 2026-07-07) |
| git commit | 6309a819e277a9354e8828a28e631cf9803b428b |
| working tree | dirty |
| power | Battery Power; macOS powermode=0 |
| run timestamp (Unix) | 1783901425 |
| harness | 1 warmup(s), 5 measured run(s), max_tokens=64, greedy |

## Runtime configuration

| runtime | endpoint | model | format / quantization | execution settings | status |
|---|---|---|---|---|---|
| onnx-genai | `http://127.0.0.1:8080/v1` | `qwen2.5-0.5b` | ONNX f32, dynamic KV cache (1.98 GB) | CPU EP; ORT default threads; onnx-genai 0.1.0 | available (qwen2.5-0.5b) |
| Ollama (llama.cpp) | `http://127.0.0.1:11434/v1` | `qwen2.5:0.5b-q8-bench` | GGUF Q8_0 (531.1 MB) | Metal/default threads; Ollama 0.12.6 | available (qwen2.5:0.5b-q8-bench, qwen2.5:0.5b, gemma3:1b, qwen2.5:latest) |
| LM Studio | `http://127.0.0.1:1234/v1` | `qwen2.5-0.5b-q8-bench` | same GGUF Q8_0 (531.1 MB) | llama.cpp Metal 2.24.0; GPU offload=max; context=2048; parallel=1 | available (qwen2.5-0.5b-q8-bench, bartowski/qwen2.5-0.5b-instruct, ollama/qwen2.5-0.5b-instruct, gemma-4-12b-it, text-embedding-nomic-embed-text-v1.5) |

## Methodology

- OpenAI `POST /v1/chat/completions` streaming API; fixed system prompt; `temperature=0`, `top_p=1`, `seed=0`, and `max_tokens=64`.
- 1 warmup run(s) were discarded, followed by 5 measured run(s). Cells show **median / p90**; the bold cell is the median winner for that prompt and metric.
- TTFT is request start to first non-empty streamed content. Total latency ends when the response stream closes after `[DONE]`.
- Decode throughput excludes TTFT: `(generated_tokens - 1) / (total - TTFT)`. Generated tokens use streamed `usage.completion_tokens` when supplied, otherwise one non-empty content event is counted as one token.
- Estimated prefill throughput is `rendered_prompt_tokens / TTFT`; it includes HTTP, queueing, chat-template processing, and first-token decode, so treat it as an API-level estimate rather than kernel-only prefill speed.
- All runtimes receive the same explicit system/user messages. Prompt token counts use the Qwen2.5 chat template and tokenizer.

## Results

| prompt | prompt tokens | runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | estimated prefill tok/s ↑ | output tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| short | 59 | onnx-genai | 167.4 / 174.8 | 44.29 / 44.58 | 1588.3 / 1609.4 | 352.4 / 355.2 | 64 / 64 |
| short | 59 | Ollama (llama.cpp) | 91.7 / 104.2 | 160.72 / 161.11 | 483.2 / 497.7 | 643.5 / 677.2 | 64 / 64 |
| short | 59 | LM Studio | **60.1 / 62.8** | **211.43 / 215.92** | **356.2 / 359.0** | **981.0 / 1004.2** | 64 / 64 |
| long | 858 | onnx-genai | 2441.8 / 2780.6 | 37.29 / 40.87 | 4172.4 / 4706.6 | 351.4 / 370.2 | 64 / 64 |
| long | 858 | Ollama (llama.cpp) | 98.5 / 99.4 | 147.29 / 147.63 | 526.9 / 531.8 | 8708.9 / 9109.7 | 64 / 64 |
| long | 858 | LM Studio | **66.1 / 77.9** | **212.65 / 216.65** | **362.8 / 375.0** | **12971.7 / 13550.4** | 64 / 64 |

## Automatic comparison against onnx-genai

| prompt | competitor | TTFT | decode throughput | total latency | estimated prefill |
|---|---|---:|---:|---:|---:|
| short | Ollama (llama.cpp) | 82.6% higher | 72.4% slower | 228.7% higher | 45.2% slower |
| short | LM Studio | 178.4% higher | 79.0% slower | 346.0% higher | 64.1% slower |
| long | Ollama (llama.cpp) | 2378.4% higher | 74.7% slower | 691.8% higher | 96.0% slower |
| long | LM Studio | 3591.6% higher | 82.5% slower | 1050.2% higher | 97.3% slower |

## Fairness caveats

- The HTTP layer, model family/size, prompts, and generation policy are common, but ONNX and GGUF quantizations can differ. The runtime table is part of the result and must identify the exact formats; do not compare unlabeled or selectively chosen quants.
- This is single-request latency/decode performance, not concurrent serving throughput. Background load, thermals, power source, and model residency affect results.
- Default runtime threading is intentional because this measures deployment behavior. The single-thread ORT setting used by exact-equality tests is not used here.

## Verdict

onnx-genai is **not faster** than either llama.cpp runtime in this baseline, and it wins no
reported median metric. Its short-prompt TTFT is 82.6% higher than Ollama and 178.4% higher
than LM Studio. Decode is 72.4% slower than Ollama and 79.0% slower than LM Studio for the
short prompt; the long-prompt gap is 74.7% and 82.5%. Long-context TTFT is the largest
deficit: 2.44 s for onnx-genai versus 98.5 ms for Ollama and 66.1 ms for LM Studio.

The comparison uses the closest practical formats available today: onnx-genai's 1.98 GB
f32 ONNX model versus one identical 531.1 MB Q8_0 GGUF file imported into both competitors.
The GGUF was downloaded once through LM Studio; Ollama reused that local file when creating
`qwen2.5:0.5b-q8-bench`. This is an honest deployment comparison rather than
numeric-parity testing, and the remaining format difference still favors the quantized
llama.cpp runtimes.

## Highest-priority optimization levers

1. Quantize the ONNX graph to an optimized int8/weight-only format and fuse Qwen attention,
   rotary, RMSNorm, and MatMul paths, then rerun directly against this Q8_0 baseline.
2. Reuse ORT IoBinding, input/output tensors, masks, position IDs, logits views, and KV
   allocations across decode steps; evaluate int8 KV to reduce bandwidth and cache pressure.
3. Profile prefill separately and tune M1 CPU threading plus sequence-length-aware
   dynamic/static-cache buckets. A full 2048-token static-cache graph improved decode in a
   smoke run but imposed excessive fixed prefill cost, so one cache shape is not sufficient.
