# Q4 + GQA WebGPU vs Q4 Metal — 2026-07-12 (JustindeMacBook-Pro)

## Verdict

Mobius `build-gguf --ep webgpu` **did produce the requested combined graph**:
168 Q4 `MatMulNBits` projections, 24 `GroupQueryAttention` nodes, and zero
`Attention` nodes. ORT placed every quantized projection and all 24 GQA nodes
on WebGPU. The model passed the correctness gate:

```text
The capital of France is Paris.
```

Q4 + GQA raises onnx-genai WebGPU decode from the prior correctness-valid fp16
GQA result of 19.40/19.07 tok/s to **30.52/29.21 tok/s** (short/long), a
57.3%/53.2% improvement. It remains slower than the correctness-valid Q4 CPU
result of 40.17/31.53 tok/s and is still **6.61x/7.59x behind** the matched LM
Studio Metal run.

## Machine and run metadata

| field | value |
|---|---|
| machine | MacBook Pro 2021 (`MacBookPro18,2`) |
| CPU / GPU | Apple M1 Max, 10 CPU cores, 32 GPU cores, Metal 4 |
| memory | 32 GB unified |
| OS | macOS 26.5.1 / Darwin 25.5.0 arm64 |
| rustc | rustc 1.97.0 (2d8144b78 2026-07-07) |
| onnx-genai commit | `2011f20575f6aff874a4c65d2e47ad2d068c66f8` |
| Mobius branch / commit | `int/q4-gqa-bench` / `cc8a59c` |
| working tree before run | dirty only from pre-existing `.squad/agents/batty/history.md` |
| power | AC Power; battery 100%; macOS powermode=0 |
| run timestamps | onnx-genai 20:05:11 PDT; LM Studio 20:05:56 PDT |
| harness | `onnx-genai-bench compare`; 1 warmup, 5 measured runs, max_tokens=64, greedy |
| runtimes | ORT 1.27 WebGPU; LM Studio llama.cpp 2.24.0 |

Each runtime was benchmarked while it was the only benchmark model loaded. LM
Studio state was backed up before the run and restored exactly afterward.

## Model provenance and serving contract

The ONNX model and LM Studio read the same source GGUF bytes:

| artifact | size | SHA-256 |
|---|---:|---|
| source `qwen2.5-0.5b-instruct-q4_0.gguf` | 428,730,208 bytes | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` |
| Q4+GQA `model.onnx` | 352,360 bytes | `aee4115746d877c44a4024d658028d9ce8aa7d2a15640b600226951427b62d29` |
| Q4+GQA `model.onnx.data` | 753,762,304 bytes | `e1e25e6e1e212dbb3eda8663dd7bf40968a28b0f8d7846d8909281d6b4c62190` |

`build-gguf` does not accept `--runtime onnx-genai`, so the model was built with
`--keep-quantized --dtype f16 --ep webgpu`; the Mobius onnx-genai emitter was
then used separately to write `inference_metadata.yaml`. The active contract is:

```yaml
required_capabilities:
  - grouped_query_attention
model:
  attention:
    type: group_query_attention
    num_kv_heads: 2
    num_attention_heads: 14
    head_dim: 64
  max_sequence_length: 4096
  runtime_configurable:
    kv_cache:
      dtype:
        - float16
kv_cache:
  native_dtype: float16
```

There is no active `genai_config.json` in the model directory.

As with the earlier Mobius Q4 import, the 168 transformer projections preserve
Q4_0 codes, while the embedding and output head are expanded to fp16. The ONNX
package is therefore 754 MB rather than the GGUF's 429 MB; it has Q4 decode
matmuls, but not byte-identical quantized storage for every weight.

## Graph and placement proof

### Saved ONNX graph

| node | count |
|---|---:|
| `com.microsoft::MatMulNBits` | **168** |
| `com.microsoft::GroupQueryAttention` | **24** |
| any `Attention` | **0** |
| total nodes | 394 |

The public past/present KV tensors and logits are fp16. KV shape is
`[batch, 2, sequence, 64]`, matching 2 KV heads and head dimension 64.

### ORT-optimized placement

Verbose ORT placement reported 199 WebGPU nodes and 6 CPU shape/sequence-length
nodes:

| placement | optimized node counts |
|---|---|
| WebGPU | 51 `MatMulNBits`, 23 `MatMulNBitsQkv`, 24 `MatMulNBitsMlp`, 24 `GroupQueryAttention`, plus surrounding graph |
| CPU | 1 `ReduceSum`, 1 `Sub`, 2 `Cast`, 1 `Shape`, 1 `Gather` |
| transfer nodes | **1 H2D (`MemcpyFromHost`) / 0 D2H** |

The optimized quantized projection count is still all 168 original projections:
`51 + 23×3 + 24×2 = 168`. Placement explicitly listed Q/K/V
`MatMulNBits`, GQA, and output projections on WebGPU.

The 1-H2D/0-D2H graph boundary does **not** mean KV is device-resident. The
runtime's shared KV buffer is still allocated with the CPU allocator, so ORT
must upload host-backed KV data during decode even though those transfers are
not represented as separate static graph copy nodes.

## Correctness gate

Both runtimes answered the pre-benchmark prompt coherently:

```text
Prompt: What is the capital of France? Answer in one short sentence.
onnx-genai Q4+GQA WebGPU: The capital of France is Paris.
LM Studio Q4 Metal:       The capital of France is Paris.
```

No performance result was recorded before this gate passed.

## Decisive comparison

Cells are median / interpolated p90. TTFT includes HTTP, scheduling, template
work, and first-token decode. Decode throughput excludes TTFT.

| prompt | runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | estimated prefill tok/s ↑ | output tokens |
|---|---|---:|---:|---:|---:|---:|
| short (59) | onnx-genai Q4+GQA WebGPU | 86.0 / 91.4 | 30.52 / 30.70 | 2148.3 / 2155.6 | 685.7 / 727.8 | 64 / 64 |
| short (59) | LM Studio Q4 Metal | **64.7 / 73.5** | **201.60 / 216.85** | **368.5 / 443.4** | **911.6 / 1264.5** | 64 / 64 |
| long (858) | onnx-genai Q4+GQA WebGPU | 407.5 / 430.1 | 29.21 / 31.02 | 1815.5 / 1861.7 | 2105.5 / 2135.8 | 42 / 42 |
| long (858) | LM Studio Q4 Metal | **60.7 / 73.5** | **221.82 / 243.75** | **337.0 / 387.2** | **14130.3 / 16230.8** | 64 / 64 |

The onnx-genai long response reached EOS at 42 tokens. Decode tok/s remains
per-token comparable; long total latency is not directly comparable to LM
Studio's 64-token completion.

### Relative standing

| prompt | TTFT | decode | estimated prefill |
|---|---:|---:|---:|
| short | 32.9% higher | 84.9% slower; **6.61x gap**; 15.1% of LM Studio | 24.8% slower |
| long | 571.3% higher | 86.8% slower; **7.59x gap**; 13.2% of LM Studio | 85.1% slower |

The previous published LM Studio checkpoint was 236.24/235.08 tok/s. Against
that approximately 236 tok/s reference, Q4+GQA WebGPU reaches 12.9%/12.4% and
is 7.74x/8.05x behind. The current same-session rerun measured
201.60/221.82 tok/s; the current rerun, not the historical peak, is used for the
decisive comparison above.

## Trajectory

| correctness-valid checkpoint | short decode | long decode | change to Q4+GQA WebGPU |
|---|---:|---:|---:|
| fp16 GQA WebGPU | 19.40 tok/s | 19.07 tok/s | Q4+GQA is **57.3% / 53.2% faster** |
| Q4 CPU | **40.17 tok/s** | **31.53 tok/s** | Q4+GQA WebGPU is 24.0% / 7.4% slower |
| **Q4+GQA WebGPU** | **30.52 tok/s** | **29.21 tok/s** | current result |
| LM Studio Q4 Metal, current | 201.60 tok/s | 221.82 tok/s | 6.61x / 7.59x faster |
| LM Studio Q4 Metal, prior | 236.24 tok/s | 235.08 tok/s | historical reference |

Quantization plus GQA materially helps WebGPU decode, proving that fp16 weight
bandwidth was a real cost. It does not yet beat the Q4 CPU path and does not
close the dominant Metal gap. The decisive remaining problem is runtime buffer
residency and launch/replay overhead, not whether the graph contains the
requested Q4 and fused attention operators.

## Top remaining levers

1. **Device-resident WebGPU KV allocator and persistent IoBinding.** Allocate
   the shared past/present buffer on WebGPU, alias present onto past there, and
   keep masks, positions, token input, and logits buffers resident.
2. **Enable WebGPU graph capture/replay.** The decode graph and shapes are stable,
   but the runtime currently appends WebGPU with no provider options and does
   not enable graph capture.
3. **Remove remaining decode bandwidth and launch costs.** Preserve or quantize
   the embedding/output head, reuse every per-token allocation, and profile the
   vocabulary projection plus optimized `MatMulNBitsQkv`/`MatMulNBitsMlp`
   kernels after KV residency is fixed.

## Reproduce

### Build and emit metadata

```bash
cd /Users/justinc/Documents/GitHub/mobius
git switch int/q4-gqa-bench
PYTHONPATH=src conda run -n onnx python -m mobius build-gguf \
  /Users/justinc/Documents/GitHub/onnx-genai/models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  --keep-quantized --dtype f16 --ep webgpu \
  --output /Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-q4-gqa-webgpu

cd /Users/justinc/Documents/GitHub/onnx-genai
cp models/qwen2.5-0.5b-q4-onnx-fixed/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt} \
  models/qwen2.5-0.5b-q4-gqa-webgpu/

cd /Users/justinc/Documents/GitHub/mobius
PYTHONPATH=src conda run -n onnx python -c \
  "import dataclasses; from pathlib import Path; from mobius._builder import resolve_dtype; from mobius.integrations.gguf._reader import GGUFModel; from mobius.integrations.gguf._config_mapping import gguf_to_config; from mobius.integrations.onnx_genai.inference_metadata import generate_inference_metadata,_to_yaml; g=GGUFModel('/Users/justinc/Documents/GitHub/onnx-genai/models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf'); c=dataclasses.replace(gguf_to_config(g),dtype=resolve_dtype('f16')); Path('/Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-q4-gqa-webgpu/inference_metadata.yaml').write_text(_to_yaml(generate_inference_metadata(c,max_sequence_length=4096)),encoding='utf-8')"
```

### Verify graph and correctness

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
conda run -n onnx python -c \
  "from collections import Counter; import onnx; m=onnx.load('models/qwen2.5-0.5b-q4-gqa-webgpu/model.onnx',load_external_data=False); c=Counter((n.domain,n.op_type) for n in m.graph.node); print(c[('com.microsoft','MatMulNBits')],c[('com.microsoft','GroupQueryAttention')],sum(v for (d,o),v in c.items() if o=='Attention'))"

ONNX_GENAI_EP=webgpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-q4-gqa-webgpu \
  --model-id qwen2.5-0.5b-q4-gqa-webgpu \
  --addr 127.0.0.1:8080

curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b-q4-gqa-webgpu","messages":[{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],"temperature":0,"top_p":1,"seed":0,"max_tokens":32,"stream":false}'
```

For the placement audit, temporarily change `Environment::new` in
`crates/onnx-genai-ort/src/env.rs` from
`ORT_LOGGING_LEVEL_WARNING` to `ORT_LOGGING_LEVEL_VERBOSE`, capture one server
startup, stop the server, then restore the file:

```bash
ONNX_GENAI_EP=webgpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-q4-gqa-webgpu \
  --model-id qwen2.5-0.5b-q4-gqa-webgpu \
  --addr 127.0.0.1:8080 \
  > models/.scratch/q4-gqa-bench/placement.log 2>&1
git checkout -- crates/onnx-genai-ort/src/env.rs
```

### Benchmark onnx-genai

```bash
cargo run --release -p onnx-genai-bench --bin compare -- \
  --runs 5 --warmups 1 --max-tokens 64 \
  --output models/.scratch/q4-gqa-bench/ours.md \
  --runtime 'onnx-genai WebGPU Q4+GQA|http://127.0.0.1:8080/v1|qwen2.5-0.5b-q4-gqa-webgpu|same-source Q4_0 ONNX; 168 MatMulNBits; 24 GroupQueryAttention; fp16 KV|WebGPU EP; ORT 1.27.0; coherence verified (Paris)'
```

### Benchmark LM Studio and restore state

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
mkdir -p models/.scratch/q4-gqa-bench/lmstudio-backup
cp -pR "$HOME/.cache/lm-studio/.internal" \
  models/.scratch/q4-gqa-bench/lmstudio-backup/internal
cp -p "$HOME/.cache/lm-studio/settings.json" \
  models/.scratch/q4-gqa-bench/lmstudio-backup/settings.json

LM_DIR="$HOME/.cache/lm-studio/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
mkdir -p "$LM_DIR"
ln models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  "$LM_DIR/qwen2.5-0.5b-instruct-q4_0.gguf"
lms server start -p 1234
lms load 'qwen2.5-0.5b-instruct@q4_0' \
  --gpu max --context-length 2048 --parallel 1 \
  --no-speculative-draft-mtp --identifier qwen05-q4-gpu-bench -y

cargo run --release -p onnx-genai-bench --bin compare -- \
  --runs 5 --warmups 1 --max-tokens 64 \
  --output models/.scratch/q4-gqa-bench/lm-gpu.md \
  --runtime 'LM Studio GPU Q4_0|http://127.0.0.1:1234/v1|qwen05-q4-gpu-bench|exact source GGUF Q4_0|llama.cpp Metal 2.24.0; GPU=max; context=2048; parallel=1; speculation=off'

lms unload --all
lms server stop
rm -f "$LM_DIR/qwen2.5-0.5b-instruct-q4_0.gguf"
rm -rf "$HOME/.cache/lm-studio/.internal"
cp -pR models/.scratch/q4-gqa-bench/lmstudio-backup/internal \
  "$HOME/.cache/lm-studio/.internal"
cp -p models/.scratch/q4-gqa-bench/lmstudio-backup/settings.json \
  "$HOME/.cache/lm-studio/settings.json"
```
