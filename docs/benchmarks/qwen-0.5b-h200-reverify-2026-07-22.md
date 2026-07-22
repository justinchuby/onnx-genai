# Qwen2.5-0.5B int4 H200 re-verification — 2026-07-22

## Result

Native CUDA decode with device-resident KV and whole-step CUDA graph replay
reached:

| Output tokens | Median decode throughput | 886 tok/s roofline | RTX 4060 baseline |
|---:|---:|---:|---:|
| 128 | **810.06 tok/s** | **91.4%** | **2.13x / +113.2%** |
| 1024 | **778.59 tok/s** | **87.9%** | **2.05x / +104.9%** |

The H200 therefore clearly exceeds the approximately 380 tok/s RTX 4060 laptop
result at both lengths. The measurements also reproduce the prior
820.65/781.20 tok/s observation within 1.3% at 128 tokens and 0.4% at 1024
tokens.

## Method

- Source: `origin/main` at `2af64f5`.
- GPU: NVIDIA H200, 143,771 MiB; driver 580.105.08.
- Model:
  `/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-0.5b-instruct-cuda-gpu-4/v4`.
  The earlier path without the `Microsoft/` namespace does not exist.
- Prompt: `The capital of France is`.
- Greedy generation with EOS stopping disabled.
- Two discarded full-generation warmups and three measured runs.
- `--steady --decode-skip 8` excludes prefill, graph capture, and the first
  eight emitted tokens from each decode window.
- `ONNX_GENAI_DEVICE_KV=1`, `ONNX_GENAI_CUDA_GRAPH=1`, and
  `CUDA_VISIBLE_DEVICES=0`.

Per-run throughput:

- 128 tokens: 805.89, 810.13, 810.06 tok/s; median **810.06 tok/s**.
- 1024 tokens: 779.03, 778.56, 778.59 tok/s; median **778.59 tok/s**.

A separate 32-token smoke run confirmed `enabled=true`, zero CUDA graph
fallbacks, and zero measured KV H2D/D2H calls or bytes. It decoded the coherent
prefix `" Paris. It is the largest city in the world ..."`.

## Reproduction

```bash
cargo build --release -p onnx-genai-bench \
  --features bench-native,cuda --bin profile_native

export CUDA_VISIBLE_DEVICES=0
export ONNX_GENAI_DEVICE_KV=1
export ONNX_GENAI_CUDA_GRAPH=1

./target/release/profile_native \
  --model /home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-0.5b-instruct-cuda-gpu-4/v4 \
  --tokens 128 --warmups 2 --runs 3 \
  --steady --decode-skip 8 --ep cuda \
  --prompt "The capital of France is"
```
