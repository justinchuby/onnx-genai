# H200 native CUDA decode — 2026-07-22

## Headline

Current `main` (`3d84b9b`) reproduces the strong Qwen2.5-0.5B native CUDA
decode result. With device-resident KV and whole-step CUDA graph replay, the
model reaches **820.65 tok/s at 128 output tokens** and **781.20 tok/s at
1024 output tokens**.

Against the user's approximately **380 tok/s** RTX 4060 laptop reference, those
results are:

- **2.16x / +116.0%** at 128 output tokens; and
- **2.06x / +105.6%** at 1024 output tokens.

Using the supplied approximately **886 tok/s** Qwen0.5B int4 HBM roofline, the
128-token result reaches **92.6% of roofline**. CUDA graph replay is the winning
configuration, improving throughput by **89.0%** at 128 tokens and **82.7%** at
1024 tokens over eager execution.

## Results

Decode throughput is the median of three measured generations after two full
warmup generations. `profile_native --steady --decode-skip 8` excludes prefill,
capture, and the first eight emitted tokens from the decode window.

| Model | Output length | Configuration | Decode tok/s | Qwen roofline | Coherence check | Rough H200 SM utilization |
|---|---:|---|---:|---:|---|---:|
| Qwen2.5-0.5B Instruct int4 | 128 | Native CUDA, device KV, graph **on** | **820.65** | **92.6%** | Starts `" Paris."` | 94% avg active, 97% max |
| Qwen2.5-0.5B Instruct int4 | 1024 | Native CUDA, device KV, graph **on** | **781.20** | **88.2%** | Same deterministic prefix | 94% avg active, 97% max |
| Qwen2.5-0.5B Instruct int4 | 128 | Native CUDA, device KV, graph **off** | 434.14 | 49.0% | Token-identical to graph-on | 67% avg active, 68% max |
| Qwen2.5-0.5B Instruct int4 | 1024 | Native CUDA, device KV, graph **off** | 427.65 | 48.3% | Token-identical to graph-on | 67% avg active, 68% max |
| Phi-4-mini Instruct int4/int8 | 128 | Native CUDA, device KV, graph **on** | **94.50** | n/a | Starts `" Paris."` | 68% avg active, 79% max |
| Phi-4-mini Instruct int4/int8 | 1024 | Native CUDA, device KV, graph **on** | **93.19** | n/a | Same deterministic prefix | 68% avg active, 79% max |

The utilization values are one-second `nvidia-smi dmon` samples from
representative steady-decode runs; they are intentionally approximate.

## Correctness and device-resident execution

A separate 32-token smoke run decoded:

```text
Qwen: " Paris. It is the largest city in the world ..."
Phi:  " Paris. What is the capital of France? Paris. ..."
```

Both models therefore produced the expected answer to `The capital of France
is`. Both smoke runs reported CUDA graph capture enabled, **zero fallbacks**, and
zero measured KV H2D/D2H calls or bytes.

## Environment and commands

- GPU: NVIDIA H200, 143,771 MiB, driver 580.105.08.
- GPU selection: physical GPU 0 via `CUDA_VISIBLE_DEVICES=0`.
- Models:
  - `/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-0.5b-instruct-cuda-gpu-4/v4`
  - `/home/justinchu/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-cuda-gpu-5/v5`
- Build:

  ```bash
  cargo build --release -p onnx-genai-bench \
    --features bench-native,cuda --bin profile_native
  ```

- Runtime controls:

  ```bash
  ONNX_GENAI_DEVICE_KV=1
  ONNX_GENAI_CUDA_GRAPH=1  # or 0 for the eager Qwen rows
  ```

- Representative invocation:

  ```bash
  ./target/release/profile_native \
    --model "$MODEL" --tokens 1024 --warmups 2 --runs 3 \
    --steady --decode-skip 8 --ep cuda \
    --prompt "The capital of France is"
  ```

## ORT GenAI / Foundry comparison

An apples-to-apples ORT result was not included. The in-tree `profile_decode`
probe was easy to build with `--features cuda-ort`, but its CUDA-graph path
charged roughly 10–11 seconds of capture/setup to each measured request
(11.37 tok/s at length 128 and 88.60 tok/s at length 1024), while graph-off
unexpectedly produced only 100–111 tok/s. Those values conflict with established
same-model ORT H200 results and do not isolate steady decode, so presenting them
as a runtime comparison would be misleading. Foundry Local was therefore not
pursued further in this run.

## Conclusion

The RTX 4060 baseline is decisively beaten. Current native CUDA remains near the
stated Qwen0.5B roofline at short decode and retains 88% at 1024 tokens. The
remaining opportunity is the long-sequence decline and the roughly 6.5–11.8%
gap to the supplied roofline; the eager result confirms whole-step graph replay
is essential rather than optional for this launch-heavy decode graph.
