# Native CUDA vs. ORT CUDA full decoder sweep — 2026-07-24

## Summary

Native beat ORT on eight of the nine comparable on-box decoder artifacts.
Qwen3-0.6B is the clear optimization target: the Foundry generic-CPU package
contains a `GatherBlockQuantized` embedding node that the native CUDA EP does not
claim, so the requested native-CUDA run fell back to the CPU and reached only
**0.410x** ORT. Among models that did run on the native CUDA EP, Qwen2.5-7B had
the thinnest lead at **1.169x**. Qwen3.5-2B-text was not comparable because
native fell back to CPU and ORT failed during warmup.

## Method

- Date: 2026-07-24.
- Source: `origin/main` at
  `25fb4fb1bddfc5b7318ac22a6edfc511c216a41e`.
- GPU: physical NVIDIA H200 GPU 0, selected as the non-GPU1 device with the
  most free memory and pinned with `CUDA_VISIBLE_DEVICES=0`. GPU 1 was never
  used; it retained the other team's approximately 129.6-GB allocation.
- CPU affinity: `taskset -c 1`.
- Decode: prompt `The capital of France is`, 128 generated tokens, greedy
  sampling, EOS stopping disabled, two warmups, three measured runs,
  `--steady --decode-skip 8`.
- Values are medians of the three measured steady-decode throughputs. Spread is
  the measured min-max tok/s range. Ratio is native / ORT.
- Native and ORT used the identical model directory for every row.

The host was contended. The initial `nvidia-smi` snapshot showed GPU 0 at 99%
utilization with 34,583 MiB used, while GPU 1 held 129,589 MiB. Contention
changed during the sweep and later GPU 0 snapshots were idle. Initial depressed
or noisy measurements were therefore rerun; the table uses the rerun window for
all comparable rows. Qwen2.5-1.5B retained the largest selected intra-run spread
(374.82-430.18 tok/s, 14.8%), below the 15% rerun threshold. These results are
best interpreted as paired ratios and observed ranges, not uncontended absolute
throughput.

## Results

| Model | Native tok/s (median) | ORT tok/s (median) | Ratio | Native spread | ORT spread | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B Instruct int4 | 858.76 | 551.04 | **1.558x** | 853.11-859.07 | 550.07-551.47 | ✅ native ahead |
| Qwen2.5-1.5B Instruct int4 | 376.56 | 246.49 | **1.528x** | 374.82-430.18 | 241.90-249.73 | ✅ native ahead |
| Qwen2.5-7B Instruct int4 | 301.41 | 257.89 | **1.169x** | 296.67-301.48 | 254.20-258.39 | ✅ native ahead |
| Qwen3-0.6B generic int4 | 10.54 | 25.73 | **0.410x** | 10.53-10.66 | 25.64-25.90 | ❌ native slower (CPU fallback) |
| Phi-4-mini Instruct int4/int8 | 321.35 | 229.16 | **1.402x** | 320.80-321.58 | 226.59-229.51 | ✅ native ahead |
| Phi-3.5-mini Instruct int4 | 124.49 | 5.56 | **22.390x** | 122.45-125.00 | 5.54-6.12 | ✅ native ahead |
| DeepSeek-R1-Distill-Qwen-1.5B int4 | 631.61 | 435.96 | **1.449x** | 629.52-632.52 | 431.82-436.09 | ✅ native ahead |
| DeepSeek-Coder-1.3B int4 | 800.22 | 639.81 | **1.251x** | 799.60-801.48 | 630.13-640.79 | ✅ native ahead |
| Qwen2.5-Coder-7B Instruct int4 | 130.05 | 29.71 | **4.377x** | 130.01-130.12 | 29.71-29.81 | ✅ native ahead |

No comparable model landed in the thin-margin band of 1.000x-1.099x.
Qwen3-0.6B is nevertheless a higher-priority target than the thinnest positive
margin because the Foundry artifact currently misses native CUDA execution
entirely.

## Non-comparable decoder artifact

| Model | Native result | ORT result | Disposition |
|---|---|---|---|
| Qwen3.5-2B-text generic | 3.45 tok/s, 3.44-3.45; requested CUDA but whole-session CPU fallback | Failed during warmup | ⚠️ CUDA support/load target; no ratio |

For Qwen3.5-2B-text, native reported unsupported
`CausalConvWithState`, `LinearAttention`, `GatherBlockQuantized`, and
`RotaryEmbedding` nodes and used the CPU EP. ORT loaded the graph but generation
failed with:

```text
state input 'past_key_values.0.conv_state' has dynamic or invalid shape
[-1, 6144, 3]; zero initialization requires every fixed-state dimension
to be concrete and positive
```

The inventory also contained Qwen3.5-0.8B and Qwen3.5-9B multimodal
embedding/vision/text pipelines and several Whisper/Nemotron speech pipelines;
they are not decoder-only `model.onnx` packages and are outside this decode
harness sweep. The `v4-bs128` Qwen2.5-0.5B directory is a duplicate cache
variant of the same model, not an additional model.

## Commands

```bash
cd /home/justinchu/wt-deckard-bench
source /home/justinchu/onnx-genai/.cudaenv.sh

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

cargo build --release -p onnx-genai-bench \
  --features bench-native,bench-ort,cuda --bin profile_native

for backend in native ort; do
  CUDA_VISIBLE_DEVICES=0 taskset -c 1 target/release/profile_native \
    --model "$MODEL_DIR" \
    --backend "$backend" \
    --ep cuda \
    --prompt "The capital of France is" \
    --tokens 128 \
    --steady \
    --decode-skip 8 \
    --warmups 2 \
    --runs 3
done
```

`MODEL_DIR` was set, in table order, to:

```text
/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-0.5b-instruct-cuda-gpu-4/v4
/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4
/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-7b-instruct-cuda-gpu-4/v4
/home/justinchu/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4
/home/justinchu/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-cuda-gpu-5/v5
/home/justinchu/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2
/home/justinchu/glm-e2e-artifacts/deepseek-r1-distill-qwen-1.5b-int4-cuda
/home/justinchu/glm-e2e-artifacts/deepseek-coder-1.3b-int4-cuda
/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4
/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-2b-text-generic-cpu-1/v1
```
