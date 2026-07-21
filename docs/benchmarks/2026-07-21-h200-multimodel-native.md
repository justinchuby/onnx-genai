# H200 native CUDA multi-model decode — 2026-07-21

## Summary

Measurements used the existing `profile_native` harness on commit
`035ad9f85d2a4fd6654d9be0fbc18e578189b9c9`. Qwen is faster than the
same-model ORT GenAI reference. Gemma 4 E2B does not currently place fully on
the native CUDA EP.

| Model | Native CUDA status | Prompt / output tokens | Warmup / samples | Median |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B int4, fp16 activation/KV | **Runs, coherent, zero fallbacks** | 1 / 256 | 5 per process / 3 processes | **771.40 tok/s** |
| Gemma 4 E2B text decoder | **Blocked at CUDA placement** | 1 / 1 requested | 0 / strict probe | no native result |

The Qwen median is:

- **+17.35%** versus the same-model ORT GenAI reference, 657.34 tok/s;
- **2.03x** the cited RTX 4060 result, approximately 380 tok/s;
- **+1.19%** versus the prior 762.31 tok/s H200 native median; and
- inside the existing **750–790 tok/s** practical roofline, 18.6 tok/s
  (2.35%) below its upper edge.

No additional distinct ready model found in the opportunistic probes completed
a native CUDA token. The exact gaps are listed below.

## Environment and method

- GPU: NVIDIA H200, 143,771 MiB, driver 580.105.08.
- Host: Linux 6.6.141.1-1.azl3, x86-64.
- Rust: `rustc 1.97.0`, `cargo 1.97.0`.
- Build:

  ```bash
  cargo build --release -p onnx-genai-bench \
    --features bench-native,cuda --bin profile_native
  ```

- Decode: greedy, temperature 0, EOS stopping disabled by the harness.
- Native CUDA controls:

  ```text
  ONNX_GENAI_CUDA_GRAPH=1
  ONNX_GENAI_DEVICE_KV=1
  ONNX_GENAI_REQUIRE_CUDA=1
  --ep cuda
  ```

- The supplied conda-library glob resolved to CUDA 13 NVRTC. Because CUDA 13
  emitted bad PTX in prior runs, the measured processes prepended the installed
  CUDA 12.6 NVRTC directory. Local loader aliases named `libnvrtc.so` and
  `libnvrtc.so.13` pointed to
  `/usr/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12`.
  `nvrtcVersion` reported 12.6, and `LD_DEBUG=libs` verified the benchmark
  loaded that library plus `libnvrtc-builtins.so.12.6`.
- Each reported sample is an independent process with five untimed warmup
  generations followed by one timed 256-token generation. The table reports
  the median of three process samples.
- `profile_native` does not split prefill and decode timing. The one-token
  prompt minimizes prefill and is required by the current fp16
  `MatMulNBits` path: a five-token prompt failed because fp16 is currently
  supported only by the symmetric block-32 M=1 decode GEMV.

## Qwen2.5-0.5B int4

Fixture:
`/home/justinchu/ana-bench/qwen-oga-cuda-graph-a4`
(24 layers, GQA KV heads 2, head size 64).

Artifact hashes:

```text
model.onnx      0ba0908e0ce8e39fcb18462787f572bfa7ca840f98c206c43919b3bec4e83eea
model.onnx.data ad98abcb190a2085bca70110df04b04128c405fda8e0a526e2b7850d4d36a184
```

Command shape:

```bash
./target/release/profile_native \
  --model /home/justinchu/ana-bench/qwen-oga-cuda-graph-a4 \
  --tokens 256 --warmups 5 --runs 1 \
  --prompt Hello --ep cuda
```

`Hello` tokenized to `[9707]`, so the prompt count was one. Results:

| Process | tok/s | ms/token | Graph captures / replays / fallbacks |
|---:|---:|---:|---:|
| 1 | 766.49 | 1.305 | 6 / 1,524 / 0 |
| 2 | 773.62 | 1.293 | 6 / 1,524 / 0 |
| 3 | 771.40 | 1.296 | 6 / 1,524 / 0 |
| **Median** | **771.40** | **1.296** | **0 fallbacks** |

A full-output smoke and the unfiltered 256-token runs produced the same
coherent greedy continuation beginning:

```text
I am a beginner in Python and I am trying to create a simple program...
```

This confirms today's `main` remains faster than ORT on the established
fixture. The stable in-repo ORT GenAI reference is 657.34 tok/s at 256 output
tokens; it was not rerun in this measurement session.

## Gemma 4 E2B text decoder

Fixture: `/home/justinchu/gemma4-e2b-onnx`.
`model.onnx` SHA-256:
`6a2ab727c2b491b737d15a1bacfc077f4afd10b8a41ba79f2f063f633b82775e`.

Strict native CUDA probe:

```bash
./target/release/profile_native \
  --model /home/justinchu/gemma4-e2b-onnx \
  --tokens 1 --warmups 0 --runs 1 \
  --prompt Hello --ep cuda
```

`Hello` tokenized to `[9259]`. With `ONNX_GENAI_REQUIRE_CUDA=1`, model load
failed before generation:

```text
CUDA execution required by ONNX_GENAI_REQUIRE_CUDA=1, but CPU fallback is
needed: 1299 nodes assigned to CPU ... GPU EP cuda_ep did not claim 90 node(s):
ai.onnx::ConstantOfShape: no handler ... at opset 24 [count=20];
ai.onnx::Gelu: no handler ... at opset 24 [count=70].
Heterogeneous CUDA+CPU placement is unavailable, so the whole session uses
cpu_ep.
```

Without the strict gate, the runtime warned and fell the whole graph back to
CPU; a one-token diagnostic took 20,904 ms (0.05 tok/s). That is not a native
CUDA measurement and is not reported as a benchmark result. The prior real
native Gemma result was likewise CPU-only. The generic follow-up is CUDA
opset-24 support for `ConstantOfShape` and `Gelu`, not a Gemma-specific
architecture branch.

## Opportunistic model probes

These were strict CUDA one-token probes, not benchmark samples:

| Fixture | Gap |
|---|---|
| Foundry Qwen2.5-1.5B int4 | Reached CUDA execution, then GQA rejected scalar `seqlens_k`: it requires a non-negative int32 `[batch_size]` tensor. |
| Foundry Phi-4-mini CUDA GPU | Reached CUDA execution, then `MatMulNBits` rejected 8-bit weights: native CUDA currently supports only `bits=4`. |
| GLM-5.2 tiny / tiny-q4 | Full CUDA placement blocked because `ScatterElements` indices are int32 while the kernel requires int64. |
| DeepSeek-V2 tiny | Full CUDA placement blocked by missing opset-24 `OneHot`. |
| Gemma 4 E2B assistant | Placement blocked by `ConstantOfShape`, `Gelu`, fp16 `ScatterElements`, and fp16 `TopK` coverage gaps. |

The Qwen1.5B gap may be an export/shape-contract follow-up: emit
`seqlens_k` as `[batch_size]`, or generically accept a scalar only for validated
batch-1 semantics. No model file was modified for this report.

## Conclusions and next levers

1. The optimized Qwen path is healthy: **771.40 tok/s**, zero graph fallbacks,
   and a 17.35% lead over ORT GenAI.
2. Qwen is already near the existing practical roofline. A material next speed
   step likely requires persistent/layer-level fusion; another narrow
   single-kernel tweak has limited budget.
3. The immediate blocker to benchmarking more architectures is CUDA operator
   coverage and shape/dtype compatibility, not Qwen decode speed:
   `ConstantOfShape`/`Gelu` for Gemma, batch-1 `seqlens_k` shape handling for
   Qwen1.5B, and 8-bit `MatMulNBits` for Phi-4.
4. Multi-token fp16 prefill remains unsupported by the native Qwen
   `MatMulNBits` kernel. Future comparative runs should either retain an
   explicitly documented one-token prompt or add a generic M>1 fp16 path.
