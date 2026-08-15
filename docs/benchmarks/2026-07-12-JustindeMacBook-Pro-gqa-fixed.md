# GQA + fixed-Q4 checkpoint — 2026-07-12 (JustindeMacBook-Pro)

## Machine and run metadata

| field | value |
|---|---|
| machine | MacBook Pro 2021 (`MacBookPro18,2`) |
| CPU | Apple M1 Max, 10 cores |
| GPU | Apple M1 Max, 32 cores, Metal 4 |
| memory | 32 GB unified |
| OS | macOS 26.5.1 / Darwin 25.5.0 arm64 |
| rustc | rustc 1.97.0 (2d8144b78 2026-07-07) |
| git commit | `f41f3b13ea5465154c2cc87d6931c5bd9db3d9c6` |
| working tree | dirty before this run (`.squad/agents/batty/history.md`) |
| power | AC Power; battery 100%; macOS powermode=0 |
| run timestamps | 2026-07-12 19:29–19:30 PDT |
| harness | `onnx-genai-bench compare`; 1 warmup, 5 measured runs, max_tokens=64, greedy |
| LM Studio runtime | llama.cpp 2.24.0 |

## Correctness gate and artifact identity

Both onnx-genai models passed the required pre-benchmark prompt:
`What is the capital of France? Answer in one short sentence.`

```text
The capital of France is Paris.
```

| artifact | graph SHA-256 | data/source SHA-256 | correctness |
|---|---|---|---|
| `models/qwen2.5-0.5b-gqa-webgpu/` | `b8db25c82b9e4b63ef6b4da0c3046b7e58b4d57a4558ff11c4d13e2ee55b5975` | `3379129593473b3b1b8e11bf3dcaa7bdb18ec4eb5bca4b3e492772860a21f8a7` | verified coherent |
| `models/qwen2.5-0.5b-q4-onnx-fixed/` | `d6e67c08307294d89546a5798153e9d278b5e0b09aa6573a749a0d3d0084aa8c` | `07b19502d6e6fd8a4c6dd4cfc2addc51862262209f342f8f69bb5b8246945598` | verified coherent; first valid Q4 CPU checkpoint |
| `models/gguf/qwen2.5-0.5b-instruct-q4_0.gguf` | — | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` | known-good coherent source GGUF |

## Runtime configuration

| runtime | model / quantization | execution provider / settings | correctness |
|---|---|---|---|
| onnx-genai WebGPU | fp16 ONNX; `com.microsoft::GroupQueryAttention`; fp16 GQA KV | ORT 1.27 WebGPU; 24 GQA / 0 Attention; placement boundary 1 H2D / 0 D2H | verified coherent |
| onnx-genai CPU | corrected same-source Q4_0 ONNX; 168 `MatMulNBits` projections | ORT 1.27 CPU EP; default threads | verified coherent |
| LM Studio GPU | exact source GGUF Q4_0 | llama.cpp Metal 2.24.0; `--gpu max`; context=2048; parallel=1; speculation off | known-good coherent GGUF |
| LM Studio CPU | exact source GGUF Q4_0 | llama.cpp CPU 2.24.0; `--gpu off`; context=2048; parallel=1; speculation off | known-good coherent GGUF |

The GQA WebGPU and LM Studio GPU rows are a deployable GPU-stack comparison,
not quantization parity. The two CPU rows use the same source Q4_0 projection
codes, but the ONNX artifact still expands the embedding/output head and is
1.32 GB versus the 428.7 MB GGUF.

## Methodology

- OpenAI streaming `POST /v1/chat/completions`; identical committed short and
  long prompts.
- `temperature=0`, `top_p=1`, `seed=0`, `max_tokens=64`.
- One discarded warmup and five measured runs for each runtime and prompt.
- Cells are **median / interpolated p90**. TTFT ends at first non-empty content.
- Decode throughput excludes TTFT. Estimated prefill is prompt tokens / TTFT
  and includes HTTP, scheduling, template work, and first-token decode.
- Each row was measured while it was the only benchmark model loaded. LM Studio
  state was backed up before the run and restored afterward.

## Four-runtime headline

Each metric cell is `short (59 tokens) · long (858 tokens)`, median / p90.

| runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | estimated prefill tok/s ↑ |
|---|---:|---:|---:|---:|
| onnx-genai WebGPU GQA fp16 | 95.1 / 96.6 · 415.3 / 560.0 | 19.40 / 20.44 · 19.07 / 20.29 | 3344.3 / 3363.7 · 3712.0 / 3922.8 | 620.6 / 649.3 · 2065.9 / 2167.5 |
| onnx-genai CPU fixed Q4 | 175.3 / 235.9 · 2932.8 / 3173.2 | 40.17 / 41.05 · 31.53 / 39.29 | 1775.1 / 2060.6 · 4925.9 / 5301.8 | 336.5 / 341.1 · 292.6 / 337.4 |
| LM Studio GPU Q4_0 | **63.3 / 90.2 · 65.2 / 74.0** | **236.24 / 244.67 · 235.08 / 237.81** | **331.1 / 357.7 · 335.5 / 344.4** | **931.4 / 1022.1 · 13155.1 / 13278.1** |
| LM Studio CPU Q4_0 | 72.8 / 134.8 · 81.5 / 119.8 | 140.81 / 162.08 · 100.59 / 157.19 | 511.7 / 813.7 · 714.3 / 847.7 | 809.9 / 884.0 · 10522.8 / 12973.1 |

## Detailed results

| prompt | prompt tokens | runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | estimated prefill tok/s ↑ | output tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| short | 59 | onnx-genai WebGPU GQA | 95.1 / 96.6 | 19.40 / 20.44 | 3344.3 / 3363.7 | 620.6 / 649.3 | 64 / 64 |
| short | 59 | onnx-genai CPU fixed Q4 | 175.3 / 235.9 | 40.17 / 41.05 | 1775.1 / 2060.6 | 336.5 / 341.1 | 64 / 64 |
| short | 59 | LM Studio GPU | **63.3 / 90.2** | **236.24 / 244.67** | **331.1 / 357.7** | **931.4 / 1022.1** | 64 / 64 |
| short | 59 | LM Studio CPU | 72.8 / 134.8 | 140.81 / 162.08 | 511.7 / 813.7 | 809.9 / 884.0 | 64 / 64 |
| long | 858 | onnx-genai WebGPU GQA | 415.3 / 560.0 | 19.07 / 20.29 | 3712.0 / 3922.8 | 2065.9 / 2167.5 | 64 / 64 |
| long | 858 | onnx-genai CPU fixed Q4 | 2932.8 / 3173.2 | 31.53 / 39.29 | 4925.9 / 5301.8 | 292.6 / 337.4 | 64 / 64 |
| long | 858 | LM Studio GPU | **65.2 / 74.0** | **235.08 / 237.81** | **335.5 / 344.4** | **13155.1 / 13278.1** | 64 / 64 |
| long | 858 | LM Studio CPU | 81.5 / 119.8 | 100.59 / 157.19 | 714.3 / 847.7 | 10522.8 / 12973.1 | 64 / 64 |

## Trajectory

| checkpoint | short decode | long decode | interpretation |
|---|---:|---:|---|
| old fp16 WebGPU fallback | 9.04 tok/s | 7.24 tok/s | correctness-valid, but 24 Attention nodes and heavy transfer plumbing |
| fixed GQA quick sanity | ~21 tok/s | — | requested 9 → 21 trajectory |
| fixed GQA measured checkpoint | 19.40 tok/s | 19.07 tok/s | 2.15x / 2.63x the old medians |
| old Q4 CPU | 43.78 tok/s | 40.66 tok/s | invalid output; historical only |
| fixed Q4 CPU | **40.17 tok/s** | **31.53 tok/s** | first correctness-valid Q4 CPU number |

The fixed GQA path is materially closer: short decode more than doubled, long
decode improved 2.63x, short TTFT fell from 159.1 to 95.1 ms, and long TTFT
fell from 980.6 to 415.3 ms. The controlled five-run median is about 19 tok/s;
the approximately 21 tok/s value remains the quick-sanity observation rather
than a substituted benchmark result.

## Honest standing

### WebGPU GQA versus LM Studio GPU

| prompt | TTFT | decode | total | estimated prefill |
|---|---:|---:|---:|---:|
| short | 50.2% higher (1.50x) | 91.8% slower (12.18x gap) | 910.1% higher (10.10x) | 33.4% slower |
| long | 537.0% higher (6.37x) | 91.9% slower (12.33x gap) | 1006.4% higher (11.06x) | 84.3% slower |

We are closer than the old 9 tok/s checkpoint, but still far from Metal on the
autoregressive path. GQA roughly halves the short decode gap from about 20.4x
to 12.2x and the long gap from about 25.7x to 12.3x. WebGPU still loses every
median metric to LM Studio GPU.

### Fixed-Q4 CPU versus LM Studio CPU

| prompt | TTFT | decode | total | estimated prefill |
|---|---:|---:|---:|---:|
| short | 140.8% higher (2.41x) | 71.5% slower (3.51x gap) | 246.9% higher (3.47x) | 58.5% slower |
| long | 3498.5% higher (35.99x) | 68.7% slower (3.19x gap) | 589.6% higher (6.90x) | 97.2% slower |

This is the first valid CPU-vs-CPU Q4 comparison. On decode, onnx-genai/ORT is
about 3.5x behind llama.cpp for the short prompt and 3.2x behind for the long
prompt. Long-prompt prefill is the larger failure: TTFT is 36x higher, making
total latency 6.9x higher despite the smaller decode ratio.

Within onnx-genai, WebGPU GQA has 45.8% lower short TTFT and 85.8% lower long
TTFT than fixed-Q4 CPU, but its decode remains 51.7% slower short and 39.5%
slower long. The better long prefill is enough for WebGPU total latency to beat
our CPU row by 24.6% on the long prompt, but not on the short prompt.

## Top remaining levers

1. **Device-resident WebGPU KV allocation, persistent IoBinding, and graph
   capture.** The shared KV buffer is still CPU-allocated, so the apparent
   1-H2D/0-D2H placement boundary still entails a host-to-device KV upload on
   every decode step. Keep KV, masks, positions, and logits buffers on-device
   and capture/replay the stable token graph.
2. **Fix the CPU prefill path.** Profile GQA/attention and `MatMulNBits` prefill
   separately, add sequence-length buckets/fusions, and tune ORT threading.
   The 2.93-second long-prompt TTFT, versus llama.cpp CPU's 81.5 ms, is now the
   clearest apples-to-apples bottleneck.
3. **Close the remaining quantized-storage and decode-kernel gap.** Preserve
   the GGUF quantized embedding/output head instead of expanding the ONNX
   artifact to 1.32 GB, reuse all per-step allocations, and profile/fuse the
   vocabulary projection and Q4 kernels. After that, test a correctness-gated
   quantized GQA WebGPU graph to remove the current fp16-vs-Q4 GPU mismatch.

## Exact reproduce commands

Build once:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
cargo build --release -p onnx-genai-server -p onnx-genai-bench
```

Start GQA WebGPU and leave it running:

```bash
ONNX_GENAI_EP=webgpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-gqa-webgpu \
  --model-id qwen2.5-0.5b-gqa-webgpu \
  --addr 127.0.0.1:8080
```

In another shell, correctness-check and benchmark it:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b-gqa-webgpu","messages":[{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],"temperature":0,"top_p":1,"seed":0,"max_tokens":32,"stream":false}'
cargo run --release -p onnx-genai-bench --bin compare -- \
  --runs 5 --warmups 1 --max-tokens 64 \
  --output models/.scratch/gqa-fixed-bench/ours-webgpu.md \
  --runtime 'onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b-gqa-webgpu|ONNX fp16; com.microsoft::GroupQueryAttention; fp16 GQA KV|WebGPU EP; ORT 1.27.0; 24 GQA / 0 Attention; 1 H2D / 0 D2H; coherence verified'
```

Stop that server, then start fixed-Q4 CPU and leave it running:

```bash
ONNX_GENAI_EP=cpu cargo run --release -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-q4-onnx-fixed \
  --model-id qwen2.5-0.5b-q4-onnx-fixed \
  --addr 127.0.0.1:8080
```

In another shell, correctness-check and benchmark it:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b-q4-onnx-fixed","messages":[{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],"temperature":0,"top_p":1,"seed":0,"max_tokens":32,"stream":false}'
cargo run --release -p onnx-genai-bench --bin compare -- \
  --runs 5 --warmups 1 --max-tokens 64 \
  --output models/.scratch/gqa-fixed-bench/ours-cpu.md \
  --runtime 'onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b-q4-onnx-fixed|same-source ONNX Q4_0; 168 MatMulNBits projections; corrected Qwen2 bias/permute|CPU EP; ORT 1.27.0; default threads; coherence verified (Paris)'
```

Back up LM Studio state, hard-link the existing GGUF, and run GPU:

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai
rm -rf models/.scratch/gqa-fixed-bench/lmstudio-backup
mkdir -p models/.scratch/gqa-fixed-bench/lmstudio-backup
cp -pR "$HOME/.cache/lm-studio/.internal" \
  models/.scratch/gqa-fixed-bench/lmstudio-backup/internal
cp -p "$HOME/.cache/lm-studio/settings.json" \
  models/.scratch/gqa-fixed-bench/lmstudio-backup/settings.json

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
  --output models/.scratch/gqa-fixed-bench/lm-gpu.md \
  --runtime 'LM Studio GPU|http://127.0.0.1:1234/v1|qwen05-q4-gpu-bench|exact source GGUF Q4_0|llama.cpp Metal 2.24.0; GPU=max; context=2048; parallel=1; speculation=off'
```

Switch the same GGUF to CPU and benchmark:

```bash
lms unload --all
lms load 'qwen2.5-0.5b-instruct@q4_0' \
  --gpu off --context-length 2048 --parallel 1 \
  --no-speculative-draft-mtp --identifier qwen05-q4-cpu-bench -y

cargo run --release -p onnx-genai-bench --bin compare -- \
  --runs 5 --warmups 1 --max-tokens 64 \
  --output models/.scratch/gqa-fixed-bench/lm-cpu.md \
  --runtime 'LM Studio CPU|http://127.0.0.1:1234/v1|qwen05-q4-cpu-bench|exact source GGUF Q4_0|llama.cpp CPU; GPU=off; context=2048; parallel=1; speculation=off'
```

Restore LM Studio:

```bash
lms unload --all
lms server stop
rm -f "$HOME/.cache/lm-studio/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf"
rm -rf "$HOME/.cache/lm-studio/.internal"
cp -pR models/.scratch/gqa-fixed-bench/lmstudio-backup/internal \
  "$HOME/.cache/lm-studio/.internal"
cp -p models/.scratch/gqa-fixed-bench/lmstudio-backup/settings.json \
  "$HOME/.cache/lm-studio/settings.json"
```
