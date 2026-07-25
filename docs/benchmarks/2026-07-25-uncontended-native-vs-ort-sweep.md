# Uncontended native CUDA vs. ORT CUDA sweep — 2026-07-25

## Result

On physical H200 GPU 6, native CUDA beat ORT CUDA in every measured row.
The CUDA-targeted artifacts provide the cleanest competitive comparison:

- Qwen2.5-0.5B: **907.87 vs. 583.31 tok/s, 1.556× (+55.6%)**.
- DeepSeek-R1-Distill-Qwen-1.5B: **633.69 vs. 445.92 tok/s,
  1.421× (+42.1%)**.

The fp32-activation accuracy-level-4 artifacts also measured much faster
natively, but their ORT baselines are not representative optimized-ORT
comparisons: ORT inserted 67/57 `Memcpy` nodes and warned about partial CUDA EP
assignment. Their native `model`/ORT ratios were 25.302× for Phi-3.5-mini and
5.357× for Qwen2.5-Coder-7B.

## Measurements

Values are medians of three runs. Parentheses contain min–max tok/s and the
full-range span as a percentage of the median. Each process used two warmups,
generated 128 tokens, and excluded the first eight emitted tokens from the
steady-decode interval.

| Model | Native `model` | Native `fp16` | ORT CUDA | model/ORT | fp16/ORT | fp16/model |
|---|---:|---:|---:|---:|---:|---:|
| Phi-3.5-mini acc-4 | 193.31 (193.11–193.32; 0.11%) | 378.89 (378.67–378.89; 0.06%) | 7.64 (7.56–7.68; 1.57%) | **25.302×** | **49.593×** | **1.960×** |
| Qwen2.5-Coder-7B acc-4 | 159.10 (159.05–159.21; 0.10%) | 300.87 (299.36–302.21; 0.95%) | 29.70 (29.46–29.89; 1.45%) | **5.357×** | **10.130×** | **1.891×** |
| Qwen2.5-0.5B int4 CUDA | 907.87 (882.13–909.10; 2.97%) | 914.89 (914.85–915.35; 0.05%) | 583.31 (582.80–583.68; 0.15%) | **1.556×** | **1.568×** | 1.008× |
| DeepSeek-R1-Distill-Qwen-1.5B int4 CUDA | 633.69 (632.86–633.94; 0.17%) | 635.94 (635.93–635.94; <0.01%) | 445.92 (445.56–446.19; 0.14%) | **1.421×** | **1.426×** | 1.004× |

The opt-in fp16 rewrite adds a real **1.960×** speedup on Phi-3.5-mini and
**1.891×** on Qwen2.5-Coder-7B. Qwen2.5-0.5B and DeepSeek already carry fp16
activation/scales, so `--decode-precision fp16` is a documented no-op; their
0.8% and 0.4% differences are noise, not fp16-mode wins.

## Host state and credibility

- Source: `9f1618bb158273509e4ba1e4d636d38a41290b21`.
- GPU: physical NVIDIA H200 6, selected with
  `CUDA_VISIBLE_DEVICES=6`; CPU affinity was `taskset -c 1`.
- GPU 6 showed 0 MiB allocated and 0% utilization before every configuration.
  After configurations it showed 0 MiB and 0–1% residual utilization. GPUs 0
  and 2–7 were also compute-idle in the snapshots. GPU 1 retained the other
  team's approximately 129.6 GB allocation and was never selected.
- The configurations were launched back-to-back in `native model`, `native
  fp16`, `ORT` order for each model. No selected-GPU contention appeared, so
  these are credible absolute CUDA decode rates.
- The host was not completely CPU-idle: after the sweep, load average was 6.28
  on 96 logical CPUs, and a three-second sample found CPU 1 about 65% idle.
  This is a secondary caveat for host-assisted paths, especially the partially
  assigned generic-CPU ORT artifacts. Tight spreads in all but one Qwen2.5-0.5B
  native run bound the observed effect; that configuration's full spread was
  still 2.97%.

## ORT artifact caveat

Phi-3.5-mini and Qwen2.5-Coder-7B are the on-box `generic-cpu-*` artifacts.
ORT CUDA reported 67 and 57 inserted `Memcpy` nodes, respectively, plus nodes
not assigned to the preferred EP. The measured native wins are real for these
exact artifacts and commands, but **must not be generalized to a claim against
CUDA-targeted, fully assigned ORT exports**. Qwen2.5-0.5B and DeepSeek used
CUDA-targeted artifacts and did not emit the `Memcpy` warning.

## Correctness sanity

No load failures occurred. All three paths were deterministic within their
three measured runs.

- Phi-3.5-mini produced a readable CodeLingo-guide continuation. Its decoded
  text begins with escaped `\u0080\u0099` bytes; this pre-existing detokenization
  blemish is visible in every prior check, not numerical garbage. Native fp16
  and ORT token streams matched; native model precision remained readable.
- **Qwen2.5-Coder-7B output is not fully clean:** all three configurations
  emitted the same intelligible “Hello! How can I assist you today?” line, then
  a long EOS/padding/newline tail, with a leading `ampie` fragment. Because the
  token streams are identical across native model, native fp16, and ORT, this is
  an artifact/prompt formatting issue rather than backend or fp16 divergence.
- Qwen2.5-0.5B produced the same coherent Python file-handling continuation in
  all three configurations.
- DeepSeek-R1-Distill-Qwen-1.5B produced the same coherent Heron's-formula
  continuation in all three configurations.

The merged decode-correctness regression locks remain the correctness
authority; this section is only the requested generated-text eyeball.

## Exact commands

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
cd /home/justinchu/wt-deckard-sweep
cargo build --release -p onnx-genai-bench \
  --features bench-native,bench-ort,cuda --bin profile_native

BIN=./target/release/profile_native
COMMON="--ep cuda --steady --warmups 2 --runs 3 --tokens 128"

PHI=/home/justinchu/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2
CODER=/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4
QWEN05=/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-0.5b-instruct-cuda-gpu-4/v4
DEEPSEEK=/home/justinchu/glm-e2e-artifacts/deepseek-r1-distill-qwen-1.5b-int4-cuda

for MODEL in "$PHI" "$CODER" "$QWEN05" "$DEEPSEEK"; do
  CUDA_VISIBLE_DEVICES=6 taskset -c 1 $BIN --model "$MODEL" $COMMON \
    --backend native --decode-precision model
  CUDA_VISIBLE_DEVICES=6 taskset -c 1 $BIN --model "$MODEL" $COMMON \
    --backend native --decode-precision fp16
  CUDA_VISIBLE_DEVICES=6 taskset -c 1 $BIN --model "$MODEL" $COMMON \
    --backend ort --decode-precision model
done
```
