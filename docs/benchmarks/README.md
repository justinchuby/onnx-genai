# Cross-runtime benchmarks

These reports compare onnx-genai with LM Studio through their OpenAI-compatible
streaming APIs. The primary periodic comparison is GPU-to-GPU: ONNX Runtime
WebGPU EP versus LM Studio's llama.cpp Metal backend. Ollama is not part of the
primary comparison.

## Current-state comparison

The definitive CPU recipe comparison is
[`2026-07-12-JustindeMacBook-Pro-cpu-recipe.md`](2026-07-12-JustindeMacBook-Pro-cpu-recipe.md).
The combined `accuracy_level=4` plus quantized embedding/head model reaches
158.62 tok/s on the short prompt: 2.0% ahead of LM Studio CPU and 20.0% ahead
of Ollama CPU. It does not win universally: LM Studio leads long-context decode
160.17 to 115.19 tok/s, while Foundry Local's ORT CPU model leads both decode
cases at 202.67 and 165.77 tok/s.

The current **H200 CUDA / GPU** checkpoint is
[`2026-07-13-H200-cuda-onnxgenai-vs-ollama-foundry.md`](2026-07-13-H200-cuda-onnxgenai-vs-ollama-foundry.md).
With CUDA-graph decode capture and a vectorized greedy argmax, onnx-genai CUDA
reaches 481 tok/s decode with 68 ms TTFT on Qwen2.5-0.5B (1024 greedy tokens):
it beats Foundry Local on every axis and ties Ollama Q4_K_M on end-to-end total
latency (2199 vs 2185 ms), winning TTFT (−59%) while trailing steady-state
decode by ~5%. Reproduce via the
[H200 CUDA runbook](H200-CUDA-runbook.md).

The current Mac GPU checkpoint is
[`2026-07-12-JustindeMacBook-Pro-q4-gqa-webgpu.md`](2026-07-12-JustindeMacBook-Pro-q4-gqa-webgpu.md).
It is the first correctness-verified, same-source GPU comparison with both Q4
`MatMulNBits` weights and on-device `GroupQueryAttention`: 30.52/29.21 tok/s
for onnx-genai WebGPU versus 201.60/221.82 tok/s for LM Studio Metal. The prior
[`gqa-fixed`](2026-07-12-JustindeMacBook-Pro-gqa-fixed.md) report contains the
fp16-GQA and Q4-CPU trajectory.

## Methodology

- Model family and size: Qwen2.5-0.5B-Instruct.
- Fixed system prompt plus committed short- and long-context prompts.
- Greedy generation: `temperature=0`, `top_p=1`, `seed=0`, normally 64 tokens.
- One discarded warmup and five measured runs by default.
- Reports show median and interpolated p90 for TTFT, decode throughput, total
  latency, and estimated API-level prefill throughput.
- Run on a quiet machine with stable power and record the GPU, execution
  provider, model format, quantization, hashes, context limit, and runtime
  versions.

## Primary GPU-to-GPU comparison

The intended 1:1 path uses:

- `models/qwen2.5-0.5b-q4-onnx/`: 168 Q4_0 `MatMulNBits` projections.
- `models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf`: the exact source GGUF loaded
  by LM Studio.

The verified GGUF SHA-256 is
`7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed`.
Do not re-download or reconvert when these files are already present.

### Q4 WebGPU correctness gate

Before benchmarking, start the Q4 ONNX model with WebGPU:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
ONNX_GENAI_EP=webgpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-q4-onnx \
  --model-id qwen2.5-0.5b-q4-webgpu \
  --addr 127.0.0.1:8080
```

On 2026-07-12, ORT 1.27 assigned all 168 original quantized projections to
WebGPU, after optimization into 51 `MatMulNBits`, 23 `MatMulNBitsQkv`, and 24
`MatMulNBitsMlp` nodes. It did not fall back those matmuls to CPU. However, the
model produced the same invalid text on CPU and WebGPU, so the Q4 graph failed
the correctness gate and was not performance-benchmarked. See
`2026-07-12-JustindeMacBook-Pro-webgpu.md`.

### fp16 WebGPU fallback

Build a non-quantized fp16 model:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
DTYPE=f16 OUT_DIR="$PWD/models/qwen2.5-0.5b-f16" scripts/build_qwen.sh
```

The runtime currently requires fp32 logits and dynamic KV API tensors. Wrap the
fp16 graph with boundary casts while leaving all weights and internal matmuls
fp16; the exact Python command is recorded in the dated report. Start it with:

```bash
ONNX_GENAI_EP=webgpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-f16-webgpu \
  --model-id qwen2.5-0.5b-f16-webgpu \
  --addr 127.0.0.1:8080
```

This fallback is **fp16 ONNX on WebGPU versus Q4_0 GGUF on Metal**. It is a
GPU-to-GPU runtime comparison, not quantization parity.

## LM Studio

Back up LM Studio state before changing it. Hard-link the already downloaded
GGUF so LM Studio reads the same inode, then load it with full Metal offload:

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

## Run and save a report

`scripts/compare_runtimes.sh` now targets only onnx-genai and LM Studio:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
RUNS=5 WARMUPS=1 MAX_TOKENS=64 \
OUTPUT=docs/benchmarks/2026-07-12-JustindeMacBook-Pro-webgpu.md \
ONNX_RUNTIME='onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b-f16-webgpu|ONNX fp16 weights; fp32 logits/KV API casts|WebGPU EP; ORT 1.27.0; default threads' \
LM_STUDIO_RUNTIME='LM Studio|http://127.0.0.1:1234/v1|qwen05-q4-webgpu-bench|exact source GGUF Q4_0|llama.cpp Metal 2.24.0; GPU=max; context=2048; parallel=1; speculation=off' \
scripts/compare_runtimes.sh
```

After the run, unload the benchmark model, stop the server, remove the temporary
hard link, and restore the saved LM Studio configuration:

```bash
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

The historical CPU-EP Q4 baseline is
`2026-07-12-JustindeMacBook-Pro-1to1-q4.md`. Its decode rates were 43.78 tok/s
short and 40.66 tok/s long. The new WebGPU report compares against those values
directly, while noting that the Q4 graph's output is not correctness-valid.
