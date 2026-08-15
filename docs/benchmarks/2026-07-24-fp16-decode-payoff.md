# Opt-in fp16 decode payoff (2026-07-24)

## Summary

The opt-in `DecodePrecision::Fp16` rewrite nearly doubled native CUDA steady-decode
throughput on both available fp32-activation, accuracy-level-4 int4 artifacts:

| Model | Native `model` | Native `fp16` | ORT CUDA | fp16/model | fp16/ORT |
|---|---:|---:|---:|---:|---:|
| Phi-3.5-mini | 193.27 tok/s (193.15–193.33) | 378.35 tok/s (377.75–378.51) | 7.69 tok/s (7.62–7.71) ⚠️ | **1.96×** | 49.20× ⚠️ |
| Qwen2.5-Coder-7B | 159.26 tok/s (159.20–159.26) | 300.70 tok/s (300.25–301.33) | 29.09 tok/s (29.05–29.14) ⚠️ | **1.89×** | 10.34× ⚠️ |

Values are the median of three measured runs; parentheses show the min–max
spread. Each process used two warmups and generated 128 tokens, with the first
eight emitted tokens excluded from steady-decode timing.

## Interpretation and contention caveat

This was a shared 8×H200 host. GPUs 3 and 5–7 were at 99% utilization and GPU 1
held about 129 GB for another team. GPU 4 was selected because the before/after
snapshots showed 0% utilization and approximately 4 MiB allocated, but shared-host
load can change between samples. Absolute tok/s therefore should not be treated
as a quiet-host result; the stable, back-to-back **fp16/model ratios** are the
useful result.

The ORT numbers are **⚠️ inconclusive as competitive baselines**. These were the
only on-box Phi-3.5-mini and Qwen2.5-Coder-7B artifacts, both named
`generic-cpu-*`. ORT CUDA warned that it inserted 67 and 57 `Memcpy` nodes,
respectively, and that some nodes were not assigned to the preferred EP. The
reported fp16/ORT ratios document the requested three-way run, but must not be
read as representative native-versus-optimized-ORT speedups. Re-benchmark ORT
with CUDA-targeted artifacts on a quiet host before making that claim.

## Correctness sanity check

The merged regression locks remain the correctness authority. As a lightweight
sanity check, all measured paths produced readable output:

- Phi-3.5-mini continued a coherent request for a “CodeLingo” programming-language
  guide. Native fp16 and ORT emitted the same measured token stream.
- Qwen2.5-Coder-7B answered, “Hello! How can I assist you today?” before emitting
  configured EOS/padding tokens. Native model precision, native fp16, and ORT
  emitted the same measured token stream.

## Environment and commands

- Source: `9271c881` plus the benchmark CLI flag in this branch.
- GPU pinning: `CUDA_VISIBLE_DEVICES=4 taskset -c 1`
- Build:

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
cargo build --release -p onnx-genai-bench \
  --features bench-native,bench-ort,cuda --bin profile_native
```

The configurations were run back-to-back in `model`, `fp16`, `ort` order for
each model:

```bash
BIN=target/release/profile_native
COMMON="--ep cuda --steady --warmups 2 --runs 3 --tokens 128"

PHI=/home/justinchu/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2
CUDA_VISIBLE_DEVICES=4 taskset -c 1 $BIN --model "$PHI" $COMMON \
  --backend native --decode-precision model
CUDA_VISIBLE_DEVICES=4 taskset -c 1 $BIN --model "$PHI" $COMMON \
  --backend native --decode-precision fp16
CUDA_VISIBLE_DEVICES=4 taskset -c 1 $BIN --model "$PHI" $COMMON \
  --backend ort --decode-precision model

QWEN=/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4
CUDA_VISIBLE_DEVICES=4 taskset -c 1 $BIN --model "$QWEN" $COMMON \
  --backend native --decode-precision model
CUDA_VISIBLE_DEVICES=4 taskset -c 1 $BIN --model "$QWEN" $COMMON \
  --backend native --decode-precision fp16
CUDA_VISIBLE_DEVICES=4 taskset -c 1 $BIN --model "$QWEN" $COMMON \
  --backend ort --decode-precision model
```
