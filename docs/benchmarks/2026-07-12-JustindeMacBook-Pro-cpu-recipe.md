# CPU recipe: onnx-genai vs llama.cpp and Foundry Local

## Verdict

The full CPU recipe is correct and closes the llama.cpp decode gap, but it does
**not** establish an across-the-board win over LM Studio:

- Against **LM Studio CPU**, onnx-genai is 2.0% faster on the short prompt
  (158.62 vs 155.56 tok/s), but 28.1% slower on the long prompt
  (115.19 vs 160.17 tok/s). Long-prompt TTFT is also 17.0x slower.
- Against **Ollama CPU**, onnx-genai is 20.0% faster on the short prompt and
  1.5% faster on the long prompt in decode. Its long-prompt total latency is
  still 2.78x higher because llama.cpp prefill is much faster.
- **Foundry Local CPU** is the fastest ONNX/ORT result: 27.8% faster than
  onnx-genai on short decode and 43.9% faster on long decode.

Headline: **we beat Ollama decode and narrowly beat LM Studio short-context
decode, but we do not yet beat llama.cpp consistently. Foundry Local remains
the CPU target to catch.**

## Machine and protocol

| field | value |
|---|---|
| machine | MacBook Pro 18,2; Apple M1 Max |
| CPU | 10 cores (8 performance + 2 efficiency) |
| memory | 32 GB |
| OS | macOS 26.5.1; Darwin 25.5.0; arm64 |
| power | AC power; macOS power mode 0 |
| onnx-genai commit | `c321b3f5b3499b7e6b7e3dad2ab33124536f5308` (working tree already dirty) |
| Rust | 1.97.0 |
| protocol | OpenAI streaming API, greedy, `temperature=0`, `top_p=1`, `seed=0`, `max_tokens=64` |
| repetitions | 1 discarded warmup + 5 measured runs per runtime and prompt |
| statistics | median / interpolated p90 |

TTFT is request start to first non-empty streamed content. Decode throughput is
`(completion_tokens - 1) / (total - TTFT)`. The estimated prefill rate includes
HTTP, chat-template handling, queueing, and first-token decode.

## Models and provenance

The onnx-genai, LM Studio, and Ollama models all derive from the exact same
Qwen2.5-0.5B-Instruct Q4_0 GGUF:

`models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf`

SHA-256:
`7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed`

| runtime | model and quantization | CPU configuration |
|---|---|---|
| onnx-genai | Mobius conversion of the source GGUF; 169 int4/block-32 `MatMulNBits`, all `accuracy_level=4`; one `GatherBlockQuantized`; quantized `lm_head`; 24 `GroupQueryAttention`; zero fp32 `MatMul` | onnx-genai CPU EP; ORT 1.27; default threads |
| LM Studio | exact source GGUF Q4_0, 408.87 MiB loaded | LM Studio 0.4.18+1; llama.cpp runtime 2.24.0; `--gpu off`; context 2048; parallel 1; speculation off |
| Ollama | exact source GGUF Q4_0 imported by SHA-256 | Ollama 0.12.6; `num_gpu 0`; `ollama ps` confirmed 100% CPU |
| Foundry Local | catalog model `qwen2.5-0.5b-instruct-generic-cpu:4`; ONNX int4/block-32; 121 `MatMulNBits`, all `accuracy_level=4`; quantized head; fp32 embedding; fused QKV projection | Foundry Local SDK 1.2.3; ORT 1.26.0; CPU EP |

Foundry Local is not byte-identical to the GGUF-derived model. Its catalog
package is 822 MB and uses a fp32 embedding. It has five quantized matmuls per
layer because Q/K/V are packed into one `qkv_proj`, versus seven per layer in
the Mobius graph. Foundry model hashes:

- `model.onnx`: `997228203ae563c7871e0d78e45e35f9062009822d5a945cf54091f14098cd21`
- `model.onnx.data`: `2b4b7d307030abc296469752eb246e3d7e0080d9ae5aa429cda97635967a35ab`

The Mobius integration branch was `int/cpu-recipe` at `fb02fb8`, combining
quantized embedding/head (`3e23990`), CPU `accuracy_level=4` (`907e5f2`), and
the metadata exporter (`541b8a7`). The generated ONNX hashes were:

- `model.onnx`: `089a37167ac2b0afb3e3044bae4c9a7ea6883eeabe12aee5bda2ae59f11a78fd`
- `model.onnx.data`: `b6b3f5b28a7a3aea9475f64799f12cc1d47baff0d39c4ad6220c8d9f927752d5`

## Correctness gate

Before recording performance, all four services answered:

> What is the capital of France? Answer in one word.

with **Paris**. All four also produced coherent text for the benchmark's short
prompt. Prompt token counts matched across runtimes (59 tokens), confirming the
same Qwen chat rendering for the measured request.

## Four-way CPU results

Cells are **median / p90**.

| prompt | runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | estimated prefill tok/s ↑ | output tokens |
|---|---|---:|---:|---:|---:|---:|
| short (59 tokens) | onnx-genai CPU | 89.2 / 95.8 | 158.62 / 164.42 | 482.6 / 511.6 | 661.1 / 680.3 | 64 / 64 |
| short | LM Studio CPU | **60.5 / 69.9** | 155.56 / 163.31 | 475.8 / 486.8 | **974.8 / 1034.2** | 64 / 64 |
| short | Ollama CPU | 100.5 / 173.4 | 132.20 / 135.43 | 577.1 / 680.6 | 587.0 / 594.2 | 64 / 64 |
| short | Foundry Local CPU | 75.2 / 88.7 | **202.67 / 212.88** | **385.7 / 404.3** | 784.4 / 788.2 | 64 / 64 |
| long (858 tokens) | onnx-genai CPU | 1176.5 / 1371.9 | 115.19 / 119.42 | 1785.1 / 1952.7 | 729.3 / 782.0 | 64 / 64 |
| long | LM Studio CPU | **69.1 / 83.2** | 160.17 / 169.88 | **457.0 / 487.8** | **12412.9 / 13732.0** | 64 / 64 |
| long | Ollama CPU | 90.8 / 111.2 | 113.53 / 123.97 | 642.0 / 713.3 | 9445.4 / 9815.8 | 64 / 64 |
| long | Foundry Local CPU | 1035.9 / 1042.6 | **165.77 / 169.33** | 1253.9 / 1269.4 | 828.2 / 836.4 | 36 / 36 |

Foundry stopped naturally after 36 tokens on the long prompt. Its decode rate
is still comparable, but its long-prompt total latency is not directly
comparable with the three 64-token results.

## Optimization trajectory

The earlier direct decode probes established the model-side optimization:

| configuration | direct decode tok/s | change |
|---|---:|---:|
| Q4 projections, missing `accuracy_level` | 39.3 | baseline |
| add `accuracy_level=4` | 91.8 | 2.34x |
| add quantized embedding/head | 194.7 | 4.95x vs baseline |

The 194.7 tok/s result was a quick direct-engine probe with a different prompt,
not the definitive API workload. Under the fixed two-prompt API protocol, the
full recipe delivered 158.62 tok/s short and 115.19 tok/s long. The long-context
drop and 729 tok/s prefill rate show that CPU prefill/KV work is now the main
remaining llama.cpp gap.

The Foundry comparison identifies another decode opportunity: pack Q/K/V into
one quantized projection, matching its 121-node graph instead of Mobius's 169
separate quantized matmuls.

## Long-context decode regression: root cause and fix (2026-07-12, Sebastian)

The long-context decode drop (158.6 short → 115.2 long, while LM Studio *rose*
156 → 160) was **ours**, and it was a **runtime KV-path selection bug**, not a
model-build defect.

**Root cause — growing KV cache.** The CPU-recipe model is a
GroupQueryAttention export with a *growing* KV contract: `past_key_values.N`
is `[batch, 2, past_sequence_len, 64]` and `present.N` is
`[batch, 2, past_sequence_len + sequence_len, 64]`, fp32 KV, no static-cache
signature and no `past_present_share_buffer` ORT metadata. Our runtime's
share-buffer gate (`shared_kv_buffer_len_from_metadata`) only admitted **GQA +
fp16** KV, so this **GQA + fp32** model fell through to the growing
`DecodeKvMode::ZeroCopyRebind` path. There, every decode step reallocates a
`present` of size `past+1` and concatenates the entire past KV into it before
rebinding it as the next `past` — **O(context) memory traffic per token**, so
per-token cost scales with context.

**Profiler evidence (decode-only, `long_context_bench`, M1 Max, default
threads):**

| context depth | growing KV (before) | shared-buffer (after) |
|---|---:|---:|
| 1–64 | 5.33 ms/tok (187.6 tok/s) | 4.43 ms/tok (226.0 tok/s) |
| 859–922 (benchmark long depth) | **8.94 ms/tok (111.9 tok/s)** | **6.33 ms/tok (158.1 tok/s)** |

RSS grew 312→360 MB under the growing path and stayed flat under shared-buffer,
confirming per-step KV reallocation. The 111.9 tok/s growing-path number
reproduces the 115.2 tok/s measured here.

**Fix.** `crates/onnx-genai-engine/src/decode.rs` now admits **fp16 *or* fp32**
GQA KV to the runtime-owned shared-buffer path (the ORT GQA kernel supports
`past_present_share_buffer` for both). The engine now routes this model to
`PastPresent { shared_buffer: true, max_len: Some(4096) }` (O(1)/token KV,
`present.*` aliased onto one max-length `past_key_values.*` buffer). The greedy
token trace is **bit-identical** to the growing path, and output stays coherent.

**Result:** long-context decode **111.9 → 158.1 tok/s (+41%)**, closing the gap
to LM Studio (160.2) and near Foundry (165.8); short context also improves
slightly with no regression. The remaining CPU gap vs Foundry is now prefill/
TTFT and the 169-vs-121 Q/K/V MatMul packing. See
`.squad/decisions/inbox/sebastian-long-context.md`.

## Reproduce

### Build the CPU recipe

The integration branch's `build-gguf` command still requires
`--keep-quantized`; without it, the graph is dequantized.

```bash
cd /Users/justinc/Documents/GitHub/mobius
git switch -c int/cpu-recipe origin/main 2>/dev/null || git switch int/cpu-recipe
git merge --no-edit \
  feat/gguf-quantized-head \
  perf/matmulnbits-accuracy-level \
  feat/onnx-genai-metadata-export

cat > /Users/justinc/Documents/GitHub/onnx-genai/models/.scratch/build_cpu_recipe.py <<'PY'
from mobius.integrations.gguf import build_from_gguf
from mobius.integrations.onnx_genai import write_inference_metadata

source = "/Users/justinc/Documents/GitHub/onnx-genai/models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf"
output = "/Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-cpu-recipe"
package = build_from_gguf(source, keep_quantized=True, execution_provider="cpu")
package.save(output, external_data="onnx")
write_inference_metadata(package, output)
PY

PYTHONPATH=src conda run -n onnx python \
  /Users/justinc/Documents/GitHub/onnx-genai/models/.scratch/build_cpu_recipe.py
cp -p /Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-q4-head/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt} \
  /Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-cpu-recipe/
```

### Start the runtimes

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
ONNX_GENAI_EP=cpu ./target/release/onnx-genai-server \
  --model models/qwen2.5-0.5b-cpu-recipe \
  --model-id qwen2.5-0.5b-cpu-recipe --addr 127.0.0.1:8080

lms server start -p 1234
lms load 'qwen2.5-0.5b-instruct@q4_0' \
  --gpu off --context-length 2048 --parallel 1 \
  --no-speculative-draft-mtp --identifier qwen05-q4-cpu-bench -y

ollama create qwen2.5:0.5b-q4-cpu-recipe-bench \
  -f models/.scratch/Modelfile.cpu-recipe
```

Foundry Local 1.2.3 uses `foundry_local_sdk` (not the older
`foundry_local` import):

```python
from foundry_local_sdk import Configuration, FoundryLocalManager

config = Configuration(
    app_name="onnx-genai-cpu-benchmark",
    web=Configuration.WebService(urls="http://127.0.0.1:5273"),
)
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance
model = manager.catalog.get_model_variant(
    "qwen2.5-0.5b-instruct-generic-cpu:4"
)
model.download()
model.load()
manager.start_web_service()
```

### Run the harness

```bash
./target/release/compare --runs 5 --warmups 1 --max-tokens 64 \
  --runtime 'onnx-genai CPU|http://127.0.0.1:8080/v1|qwen2.5-0.5b-cpu-recipe|ONNX int4: 169 MatMulNBits acc4 + GatherBlockQuantized|CPU EP; ORT 1.27; default threads' \
  --runtime 'LM Studio CPU|http://127.0.0.1:1234/v1|qwen05-q4-cpu-bench|exact source GGUF Q4_0|llama.cpp 2.24.0; GPU off; context 2048; parallel 1; speculation off' \
  --runtime 'Ollama CPU|http://127.0.0.1:11434/v1|qwen2.5:0.5b-q4-cpu-recipe-bench|exact source GGUF Q4_0|Ollama 0.12.6; num_gpu 0; 100% CPU' \
  --runtime 'Foundry Local CPU|http://127.0.0.1:5273/v1|qwen2.5-0.5b-instruct-generic-cpu|ONNX int4: 121 MatMulNBits acc4; fp32 embedding|Foundry Local SDK 1.2.3; ORT 1.26.0; CPU EP'
```

### Reproduce the long-context KV diagnostic

Decode-only per-context-depth timing (growing KV vs shared-buffer), on the same
`DecodeSession` the engine uses:

```bash
cargo build --release -p onnx-genai-ort --example long_context_bench
# growing KV (pre-fix default for fp32 GQA):
./target/release/examples/long_context_bench \
  --model models/qwen2.5-0.5b-cpu-recipe --mode past-present \
  --max-tokens 950 --buckets 64,858,922
# runtime-owned shared buffer (post-fix path):
./target/release/examples/long_context_bench \
  --model models/qwen2.5-0.5b-cpu-recipe --mode shared \
  --max-tokens 950 --buckets 64,858,922
```

The `token_trace` line is identical across both modes (correctness), while
`avg_ms_per_token` in the `859,922` bucket drops from ~8.9 ms to ~6.3 ms.
