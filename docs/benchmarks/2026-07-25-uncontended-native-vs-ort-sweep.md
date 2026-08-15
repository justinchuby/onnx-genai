# Uncontended native CUDA vs. ORT CUDA sweep — 2026-07-25

## Result

On physical H200 GPU 6, the only valid native-vs-ORT comparison is on the
CUDA-targeted artifacts, and there native CUDA won:

- Qwen2.5-0.5B: **907.87 vs. 583.31 tok/s, 1.556× (+55.6%)**.
- DeepSeek-R1-Distill-Qwen-1.5B: **633.69 vs. 445.92 tok/s,
  1.421× (+42.1%)**.

The fp32-activation accuracy-level-4 artifacts (Phi-3.5-mini, Qwen2.5-Coder-7B)
measured much faster natively, but **their ORT rows are an invalid,
partial-CUDA-EP baseline and must not be used as a "vs ORT" claim.** ORT
appended the CUDA EP but could not place the `generic-cpu` graph on the GPU: it
inserted 67/57 `Memcpy` nodes and warned about partial CUDA EP assignment, so
much of the graph ran on the CPU with host↔device thrash. The resulting
25.302×/49.593× (Phi-3.5-mini) and 5.357×/10.130× (Qwen2.5-Coder-7B) ratios
therefore compare native CUDA against a broken CPU-fallback baseline, not
against ORT running on the GPU. They are excluded from every headline claim
below.

## Measurements

Values are medians of three runs. Parentheses contain min–max tok/s and the
full-range span as a percentage of the median. Each process used two warmups,
generated 128 tokens, and excluded the first eight emitted tokens from the
steady-decode interval.

| Model | Native `model` | Native `fp16` | ORT CUDA | model/ORT | fp16/ORT | fp16/model |
|---|---:|---:|---:|---:|---:|---:|
| Phi-3.5-mini acc-4 | 193.31 (193.11–193.32; 0.11%) | 378.89 (378.67–378.89; 0.06%) | 7.64 (7.56–7.68; 1.57%) — partial-EP† | 25.302×† | 49.593×† | **1.960×** |
| Qwen2.5-Coder-7B acc-4 | 159.10 (159.05–159.21; 0.10%) | 300.87 (299.36–302.21; 0.95%) | 29.70 (29.46–29.89; 1.45%) — partial-EP† | 5.357×† | 10.130×† | **1.891×** |
| Qwen2.5-0.5B int4 CUDA | 907.87 (882.13–909.10; 2.97%) | 914.89 (914.85–915.35; 0.05%) | 583.31 (582.80–583.68; 0.15%) | **1.556×** | **1.568×** | 1.008× |
| DeepSeek-R1-Distill-Qwen-1.5B int4 CUDA | 633.69 (632.86–633.94; 0.17%) | 635.94 (635.93–635.94; <0.01%) | 445.92 (445.56–446.19; 0.14%) | **1.421×** | **1.426×** | 1.004× |

† Invalid baseline: the `generic-cpu` Phi-3.5-mini and Qwen2.5-Coder-7B ORT runs
could not be placed on the GPU (67/57 inserted `Memcpy` nodes; partial CUDA EP
assignment), so ORT ran largely on the CPU. The `model/ORT` and `fp16/ORT`
ratios for these two rows are **not** valid native-vs-ORT results and are
excluded from every headline claim. Only the `fp16/model` column (same-GPU,
both native) is a valid measurement for these two rows.

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
The bench's `--backend ort --ep cuda` path *does* append the CUDA execution
provider (`SessionOptionsAppendExecutionProvider_CUDA_V2`), but the CUDA EP
claims only the graph nodes it supports; the `generic-cpu` export leaves most of
the graph on the CPU. ORT CUDA reported 67 and 57 inserted `Memcpy` nodes,
respectively, plus nodes not assigned to the preferred EP — i.e., the ORT run
was largely a CPU-fallback baseline with host↔device copies, which is why it
measured 7.64 / 29.70 tok/s. **These ORT numbers are an invalid baseline: the
`model/ORT` and `fp16/ORT` ratios for these two models are not a native-vs-ORT
result and must not be headlined**, either for these artifacts or generalized to
CUDA-targeted, fully assigned ORT exports. Qwen2.5-0.5B (`cuda-gpu`) and
DeepSeek (`int4-cuda`) used CUDA-targeted artifacts, placed fully on the GPU,
did not emit the `Memcpy` warning, and are the only rows whose native-vs-ORT
ratios (1.556× / 1.421×) are valid.

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
