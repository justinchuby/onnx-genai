# WebGPU vs LM Studio benchmark — 2026-07-12 (JustindeMacBook-Pro)

## Machine and run metadata

| field | value |
|---|---|
| machine | MacBook Pro 2021 (`MacBookPro18,2`) |
| CPU | Apple M1 Max, 10 cores |
| GPU | Apple M1 Max, 32 cores, Metal 4 |
| memory | 32 GB unified |
| OS | Darwin 25.5.0 arm64 |
| rustc | rustc 1.97.0 (2d8144b78 2026-07-07) |
| git commit | `3c819e96a5ad24058086655249603cfd70c62a41` |
| power | AC Power; macOS powermode=0 |
| run timestamp | 2026-07-12 17:58 PDT (Unix 1783904293) |
| harness | 1 warmup, 5 measured runs, max_tokens=64, greedy |

## MatMulNBits-on-WebGPU verdict

The intended 1:1 Q4 graph is
`models/qwen2.5-0.5b-q4-onnx/`, generated from the exact GGUF used by LM Studio.

| artifact | SHA-256 |
|---|---|
| source GGUF Q4_0 | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` |
| Q4 ONNX graph | `dfd0ab7bde182bd15154d56e0c89270d3a1a9c5d8f0a28bb49f205a34926eea5` |
| Q4 ONNX data | `dd553f4598bbce08e1ff83cd5489626d06443560568aee33348dc0b1d0e6265d` |

ORT 1.27.0 enabled `WebGpuExecutionProvider` and loaded the graph. Verbose node
placement reported 254 nodes on WebGPU and 29 on CPU. The CPU nodes were 24
`Attention` nodes plus shape/cast plumbing. All 168 original quantized
projections were represented on WebGPU:

- 51 `MatMulNBits`
- 23 `MatMulNBitsQkv`, representing 69 original Q/K/V matmuls
- 24 `MatMulNBitsMlp`, representing 48 original gate/up matmuls

**Support verdict:** `MatMulNBits` was assigned natively to WebGPU; it did not
fall back to CPU. The Q4 model nevertheless failed the required correctness
gate. For the fixed 59-token prompt, both CPU EP and WebGPU EP deterministically
returned the same invalid 16-token text:

```text
辱字母`\amaistenoleon ​​attelehem:@"%@",|iornownd.baomidoupuEventData
```

Because the failure reproduces on CPU, the evidence points to the converted Q4
graph/weights rather than a WebGPU-only kernel error. It still makes the Q4
artifact unsuitable for a valid 1:1 benchmark. No Q4 WebGPU performance number
is reported.

## fp16 WebGPU fallback

`DTYPE=f16 scripts/build_qwen.sh` produced a non-quantized model, but the first
load failed exactly with:

```text
Error: present KV output 'present.0.key' must be Float32 rank >= 3,
got Float16 [-1, 2, -1, 64]
```

The benchmark model keeps fp16 weights and all 169 internal `MatMul` nodes, with
fp32 boundary casts for logits and dynamic KV tensors so the current runtime can
consume it.

| artifact | value |
|---|---|
| model | `models/qwen2.5-0.5b-f16-webgpu/` |
| graph SHA-256 | `569302610df1afced141ad37f4afc16b713ee54536a31cca71d0180cacd0ad4f` |
| data SHA-256 | `6a43016e2f1cfe96cb9dc337653ef7a65dc117f661da71e6a7347a2717cb94cf` |
| graph | 169 fp16 `MatMul`; 98 casts including one original cast |
| WebGPU placement | 684 nodes, including all 169 `MatMul` nodes |
| CPU placement | 29 nodes: 24 `Attention`, 3 `Shape`, 1 `Cast`, 1 `Concat` |
| output check | coherent deterministic Qwen response |

This is **fp16 ONNX/WebGPU versus Q4_0 GGUF/Metal**, not quantization parity.

## Runtime configuration

| runtime | endpoint | model / quantization | execution settings |
|---|---|---|---|
| onnx-genai | `http://127.0.0.1:8080/v1` | Qwen2.5-0.5B fp16 ONNX; fp32 logits/KV boundary casts | ORT 1.27.0 WebGPU EP; 169 matmuls on WebGPU; attention on CPU |
| LM Studio | `http://127.0.0.1:1234/v1` | exact 428,730,208-byte source GGUF Q4_0 | llama.cpp Metal 2.24.0; GPU=max; context=2048; parallel=1; speculation=off |

## Methodology

- OpenAI streaming `POST /v1/chat/completions`; identical system/user messages.
- `temperature=0`, `top_p=1`, `seed=0`, `max_tokens=64`.
- One warmup discarded, then five measured runs.
- Cells are **median / p90**. TTFT ends at first non-empty content.
- Decode throughput excludes TTFT. Estimated prefill is prompt tokens / TTFT
  and includes HTTP, scheduling, template work, and first-token decode.
- The long WebGPU response reached EOS at 50 tokens; LM Studio generated 64.
  Decode tok/s remains per-token comparable. Total latency therefore favors
  onnx-genai and is still much worse.

## WebGPU vs LM Studio results

| prompt | prompt tokens | runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | estimated prefill tok/s ↑ | output tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| short | 59 | onnx-genai WebGPU | 159.1 / 192.8 | 9.04 / 9.12 | 7128.0 / 7206.5 | 370.9 / 396.0 | 64 / 64 |
| short | 59 | LM Studio | **66.0 / 72.3** | **184.21 / 203.87** | **407.9 / 427.8** | **894.1 / 922.7** | 64 / 64 |
| long | 858 | onnx-genai WebGPU | 980.6 / 1129.6 | 7.24 / 7.31 | 7778.4 / 7974.8 | 875.0 / 884.5 | 50 / 50 |
| long | 858 | LM Studio | **69.8 / 71.1** | **186.21 / 190.85** | **408.1 / 449.6** | **12294.0 / 16238.7** | 64 / 64 |

| prompt | onnx-genai WebGPU relative to LM Studio median |
|---|---|
| short | TTFT 141.1% higher; decode 95.1% slower; total 1647.5% higher; prefill 58.5% slower |
| long | TTFT 1304.9% higher; decode 96.1% slower; total 1806.0% higher; prefill 92.9% slower |

## WebGPU vs onnx-genai CPU-EP baseline

The prior CPU report used the Q4 graph, so this comparison also changes fp16
versus Q4. The newly discovered invalid Q4 output means its rates are historical
performance measurements, not a correctness-valid model baseline.

| prompt | metric | CPU EP Q4 median | WebGPU EP fp16 median | WebGPU delta |
|---|---|---:|---:|---:|
| short | TTFT | 170.1 ms | 159.1 ms | 6.5% lower |
| short | decode | 43.78 tok/s | 9.04 tok/s | **79.4% lower** |
| short | total | 1609.1 ms | 7128.0 ms | 343.0% higher |
| short | prefill | 346.9 tok/s | 370.9 tok/s | 6.9% higher |
| long | TTFT | 2253.9 ms | 980.6 ms | 56.5% lower |
| long | decode | 40.66 tok/s | 7.24 tok/s | **82.2% lower** |
| long | total | 3803.3 ms | 7778.4 ms | 104.5% higher |
| long | prefill | 380.7 tok/s | 875.0 tok/s | 129.8% higher |

WebGPU improved TTFT/prefill, especially for the long prompt, but decode was
4.84x slower for short and 5.62x slower for long. For this 0.5B model, WebGPU
does not beat the CPU EP on the dominant autoregressive decode path.

## Verdict and top runtime levers

onnx-genai WebGPU is slower than LM Studio on every reported median metric.
The largest gap is decode: 9.04 versus 184.21 tok/s short and 7.24 versus
186.21 tok/s long. WebGPU also loses to the historical CPU EP decode baseline.

Top three concrete levers:

1. Move the 24 attention nodes and dynamic KV update path onto WebGPU, avoiding
   the 121 device-to-host and 74 host-to-device copies seen in fp16 placement.
2. Add native fp16 logits/KV support plus persistent IoBinding/device-resident
   KV buffers so 97 boundary casts and per-token CPU/GPU transfers disappear.
3. Fix and correctness-test the Q4 importer, then optimize the already native
   WebGPU `MatMulNBitsQkv`/`MatMulNBitsMlp` path to benchmark compressed weights
   without the fp16 bandwidth and quantization mismatch.

## Reproduce

Use the existing Q4 artifacts; do not download or reconvert them:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
shasum -a 256 models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  models/qwen2.5-0.5b-q4-onnx/model.onnx \
  models/qwen2.5-0.5b-q4-onnx/model.onnx.data

ONNX_GENAI_EP=webgpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-q4-onnx \
  --model-id qwen2.5-0.5b-q4-webgpu \
  --addr 127.0.0.1:8080
```

After the Q4 correctness failure, build and wrap fp16:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
DTYPE=f16 OUT_DIR="$PWD/models/qwen2.5-0.5b-f16" scripts/build_qwen.sh

rm -rf models/qwen2.5-0.5b-f16-webgpu
mkdir -p models/qwen2.5-0.5b-f16-webgpu
python - <<'PY'
from pathlib import Path
import os, shutil
import onnx
from onnx import TensorProto, helper

src = Path("models/qwen2.5-0.5b-f16")
dst = Path("models/qwen2.5-0.5b-f16-webgpu")
model = onnx.load(src / "model.onnx", load_external_data=False)

for value in model.graph.input:
    tensor = value.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT16:
        continue
    public, internal = value.name, value.name + "__fp16"
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == public:
                node.input[index] = internal
    model.graph.node.insert(
        0, helper.make_node("Cast", [public], [internal],
                            to=TensorProto.FLOAT16,
                            name=public + "/CastToFp16"))
    tensor.elem_type = TensorProto.FLOAT

for value in model.graph.output:
    tensor = value.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT16:
        continue
    public, internal = value.name, value.name + "__fp16"
    for node in model.graph.node:
        for index, name in enumerate(node.output):
            if name == public:
                node.output[index] = internal
    model.graph.node.append(
        helper.make_node("Cast", [internal], [public],
                         to=TensorProto.FLOAT,
                         name=public + "/CastToFp32"))
    tensor.elem_type = TensorProto.FLOAT

onnx.save_model(model, dst / "model.onnx")
os.link(src / "model.onnx.data", dst / "model.onnx.data")
for name in ["tokenizer_config.json", "tokenizer.json", "merges.txt",
             "vocab.json", "genai_config.json"]:
    shutil.copy2(src / name, dst / name)
PY

ONNX_GENAI_EP=webgpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-f16-webgpu \
  --model-id qwen2.5-0.5b-f16-webgpu \
  --addr 127.0.0.1:8080
```

In another shell, back up LM Studio state and load the exact GGUF:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
rm -rf models/.scratch/lmstudio-backup
mkdir -p models/.scratch/lmstudio-backup
cp -pR "$HOME/.cache/lm-studio/.internal" \
  models/.scratch/lmstudio-backup/internal
cp -p "$HOME/.cache/lm-studio/settings.json" \
  models/.scratch/lmstudio-backup/settings.json
LM_DIR="$HOME/.cache/lm-studio/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
mkdir -p "$LM_DIR"
ln models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf \
  "$LM_DIR/qwen2.5-0.5b-instruct-q4_0.gguf"
lms server start -p 1234
lms load 'qwen2.5-0.5b-instruct@q4_0' \
  --gpu max --context-length 2048 --parallel 1 \
  --no-speculative-draft-mtp \
  --identifier qwen05-q4-webgpu-bench -y
```

Run only onnx-genai and LM Studio:

```bash
RUNS=5 WARMUPS=1 MAX_TOKENS=64 \
OUTPUT=docs/benchmarks/2026-07-12-JustindeMacBook-Pro-webgpu.md \
ONNX_RUNTIME='onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b-f16-webgpu|ONNX fp16 weights; fp32 logits/KV API casts|WebGPU EP; ORT 1.27.0; default threads' \
LM_STUDIO_RUNTIME='LM Studio|http://127.0.0.1:1234/v1|qwen05-q4-webgpu-bench|exact source GGUF Q4_0|llama.cpp Metal 2.24.0; GPU=max; context=2048; parallel=1; speculation=off' \
scripts/compare_runtimes.sh
```

Restore LM Studio to its pre-benchmark state:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
lms unload --all
lms server stop
rm -f "$HOME/.cache/lm-studio/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf"
rm -rf "$HOME/.cache/lm-studio/.internal"
cp -pR models/.scratch/lmstudio-backup/internal \
  "$HOME/.cache/lm-studio/.internal"
cp -p models/.scratch/lmstudio-backup/settings.json \
  "$HOME/.cache/lm-studio/settings.json"
rm -rf models/.scratch/lmstudio-backup
```
