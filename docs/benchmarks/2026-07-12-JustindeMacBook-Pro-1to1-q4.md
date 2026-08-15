# Same-source Q4 cross-runtime benchmark — 2026-07-12 (JustindeMacBook-Pro)

## Machine and run metadata

| field | value |
|---|---|
| CPU | Apple M1 Max |
| cores | 10 |
| OS | Darwin 25.5.0 arm64 arm |
| rustc | rustc 1.97.0 (2d8144b78 2026-07-07) |
| git commit | 184275eea8b490b9b7629fca932be31560185381 |
| working tree | clean |
| power | AC Power; macOS powermode=0 |
| run timestamp (Unix) | 1783903147 |
| harness | 1 warmup(s), 5 measured run(s), max_tokens=64, greedy |

## Model identity and conversion

| field | value |
|---|---|
| Hugging Face source | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| source file | `qwen2.5-0.5b-instruct-q4_0.gguf` (428,730,208 bytes) |
| source SHA-256 | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` |
| GGUF tensor types | 169 Q4_0, 121 F32, 1 Q8_0 |
| Mobius commit | `e21e5b2b7fb8831c7bb6e980735e8cf0e5cfef9b` |
| conversion result | 168 `com.microsoft.MatMulNBits` nodes; loaded successfully by onnx-genai with ORT 1.27.0 |
| ONNX SHA-256 | graph `dfd0ab7bde182bd15154d56e0c89270d3a1a9c5d8f0a28bb49f205a34926eea5`; data `dd553f4598bbce08e1ff83cd5489626d06443560568aee33348dc0b1d0e6265d` |

The first-choice Q4_K_M file was not used because the current Mobius conversion failed
honestly with a quantized weight shape mismatch (`model expects [896, 28, 32], got
[896, 896]`). The official Q4_0 file is the verified quantized path for this run.

Mobius printed `Quantized mode: preserving GGUF quantization as MatMulNBits...`.
Projection weights therefore retain the GGUF Q4_0 codes on the ONNX side instead of
becoming fp32. Current Mobius still dequantizes the Q4_0 token embedding and Q8_0 output
head to fp32, so this is a same-source, same-projection-quant comparison rather than
byte-identical storage for every tensor. That remaining importer limitation likely
penalizes ONNX memory traffic and output-head decode performance and must not be hidden.

## Runtime configuration

| runtime | endpoint | model | format / quantization | execution settings | status |
|---|---|---|---|---|---|
| onnx-genai | `http://127.0.0.1:8080/v1` | `qwen2.5-0.5b-q4-1to1` | same-source ONNX; 168 Q4_0 `MatMulNBits` projections | CPU EP; ORT 1.27.0 default threads | available (qwen2.5-0.5b-q4-1to1) |
| Ollama (llama.cpp) | `http://127.0.0.1:11434/v1` | `qwen05-q4-1to1:latest` | exact source GGUF Q4_0 | Metal/default threads; Ollama 0.12.6; imported from same file | available (qwen05-q4-1to1:latest, qwen2.5:0.5b-q8-bench, qwen2.5:0.5b, gemma3:1b, qwen2.5:latest) |
| LM Studio | `http://127.0.0.1:1234/v1` | `qwen05-q4-1to1` | exact source GGUF Q4_0 | llama.cpp Metal 2.24.0; GPU offload=max; context=2048; parallel=1 | available (qwen05-q4-1to1, qwen2.5-0.5b-instruct@q4_0, qwen/qwen2.5-0.5b-instruct@q8_0, bartowski/qwen2.5-0.5b-instruct, ollama/qwen2.5-0.5b-instruct, gemma-4-12b-it, text-embedding-nomic-embed-text-v1.5) |

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
| short | 59 | onnx-genai | 170.1 / 171.1 | 43.78 / 43.84 | 1609.1 / 1624.4 | 346.9 / 348.6 | 64 / 64 |
| short | 59 | Ollama (llama.cpp) | 86.7 / 97.3 | **210.43 / 212.69** | **385.6 / 403.7** | 680.5 / 688.0 | 64 / 64 |
| short | 59 | LM Studio | **62.8 / 69.1** | 191.99 / 200.04 | 396.4 / 430.9 | **939.2 / 1078.2** | 64 / 64 |
| long | 858 | onnx-genai | 2253.9 / 2462.1 | 40.66 / 41.26 | 3803.3 / 4028.5 | 380.7 / 385.4 | 64 / 64 |
| long | 858 | Ollama (llama.cpp) | 110.1 / 131.1 | 187.84 / 188.86 | 445.5 / 479.5 | 7791.1 / 9052.6 | 64 / 64 |
| long | 858 | LM Studio | **74.0 / 79.7** | **203.80 / 219.58** | **385.0 / 422.1** | **11593.3 / 12761.2** | 64 / 64 |

## Automatic comparison against onnx-genai

| prompt | competitor | TTFT | decode throughput | total latency | estimated prefill |
|---|---|---:|---:|---:|---:|
| short | Ollama (llama.cpp) | 96.2% higher | 79.2% slower | 317.4% higher | 49.0% slower |
| short | LM Studio | 170.8% higher | 77.2% slower | 306.0% higher | 63.1% slower |
| long | Ollama (llama.cpp) | 1946.6% higher | 78.4% slower | 753.7% higher | 95.1% slower |
| long | LM Studio | 2945.4% higher | 80.0% slower | 887.9% higher | 96.7% slower |

## Fairness caveats

- Both llama.cpp runtimes read the exact SHA-256-identified GGUF. ONNX was generated from
  those same bytes and preserves all 168 transformer projection matrices as Q4_0
  `MatMulNBits`. Mobius currently expands the quantized embedding and output head to fp32;
  this is the only known weight-format mismatch.
- onnx-genai used ORT CPU EP while both llama.cpp deployments used Apple Metal. The result
  measures the deployable runtime/backend stacks, not an identical CPU-only kernel backend.
- This is single-request latency/decode performance, not concurrent serving throughput. Background load, thermals, power source, and model residency affect results.
- Default runtime threading is intentional because this measures deployment behavior. The single-thread ORT setting used by exact-equality tests is not used here.

## Verdict

onnx-genai loaded and executed the `MatMulNBits` graph, but it is still slower on every
reported median metric. For the short prompt, its 43.78 tok/s decode is 79.2% slower than
Ollama and 77.2% slower than LM Studio; TTFT is 96.2% and 170.8% higher. For the long
prompt, decode is 78.4% and 80.0% slower, while TTFT is 20.5x and 30.5x the competitor
latency. Total latency is 4.17x/4.06x as high for short and 8.54x/9.88x as high for long.

Quantizing the projection matrices did not materially close the earlier gap. The largest
remaining deficit is prefill, followed by per-token decode. This run also shows that the
current Mobius importer and ORT CPU-vs-Metal backend difference prevent a perfectly
storage- and device-identical comparison, even though the dominant transformer matrices
now use the same source Q4_0 codes.

## Highest-priority runtime levers

1. Preserve the GGUF token embedding and Q8_0 output head instead of expanding both to
   fp32, then profile ORT 1.27 `MatMulNBits`/MLAS kernels and available EP choices on Apple
   Silicon. The fp32 vocabulary projection is directly on the decode critical path.
2. Reuse ORT `IoBinding`, KV tensors, masks, position IDs, logits buffers, and allocator
   state across decode steps so each token does not rebuild bindings or copy cache data.
3. Build a separate fused prefill path with GQA/attention rewrites, sequence-length cache
   buckets, and measured intra/inter-op thread settings. Long-prompt TTFT is the clearest
   bottleneck and should be profiled independently from decode.

## Reproduce

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
mkdir -p models/gguf
curl -L --fail -o models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
shasum -a 256 models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf

conda run -n onnx python -m pip install \
  'onnx-shape-inference==0.2.0' 'onnxscript==0.7.1' 'gguf==0.19.0'
cd /Users/justinc/Documents/GitHub/mobius
PYTHONPATH=src conda run -n onnx python -m mobius build-gguf \
  /Users/justinc/Documents/GitHub/onnx-genai/models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  --output /Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-q4-onnx \
  --keep-quantized
cd /Users/justinc/Documents/GitHub/onnx-genai
conda run -n onnx python -c \
  "import onnx; m=onnx.load('models/qwen2.5-0.5b-q4-onnx/model.onnx', load_external_data=False); print(sum(n.domain == 'com.microsoft' and n.op_type == 'MatMulNBits' for n in m.graph.node))"

cp models/qwen2.5-0.5b/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,genai_config.json} \
  models/qwen2.5-0.5b-q4-onnx/
ONNX_GENAI_EP=cpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-q4-onnx \
  --model-id qwen2.5-0.5b-q4-1to1 \
  --addr 127.0.0.1:8080
```

In separate shells, import/load the exact same GGUF:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
printf 'FROM %s\n' "$PWD/models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf" \
  > models/benchmarks/Modelfile.q4-1to1
ollama create qwen05-q4-1to1 -f models/benchmarks/Modelfile.q4-1to1

LM_DIR="$HOME/.cache/lm-studio/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
mkdir -p "$LM_DIR"
ln models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  "$LM_DIR/qwen2.5-0.5b-instruct-q4_0.gguf"
lms server start -p 1234
lms load 'qwen2.5-0.5b-instruct@q4_0' \
  --gpu max --context-length 2048 --parallel 1 \
  --identifier qwen05-q4-1to1 -y
```

Run the harness:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
RUNS=5 WARMUPS=1 MAX_TOKENS=64 \
OUTPUT=docs/benchmarks/2026-07-12-JustindeMacBook-Pro-1to1-q4.md \
ONNX_RUNTIME='onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b-q4-1to1|same-source ONNX MatMulNBits Q4_0|CPU EP; ORT default threads' \
OLLAMA_RUNTIME='Ollama (llama.cpp)|http://127.0.0.1:11434/v1|qwen05-q4-1to1:latest|exact source GGUF Q4_0|Metal/default threads' \
LM_STUDIO_RUNTIME='LM Studio|http://127.0.0.1:1234/v1|qwen05-q4-1to1|exact source GGUF Q4_0|Metal; context=2048; parallel=1' \
scripts/compare_runtimes.sh
```
