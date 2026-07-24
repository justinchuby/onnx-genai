# Native CUDA vs. ORT CUDA regression confirmation — 2026-07-24

## Method

- Source: `origin/main` at `e203b812`.
- GPU: single NVIDIA H200 GPU 6, pinned with `CUDA_VISIBLE_DEVICES=6`.
- Harness: `profile_native --backend native|ort --steady --warmups 2 --tokens 128`, with `taskset -c 1` and the same GPU pin for both backends.
- Decode: single-stream greedy steady-state decode, 128 tokens, two warmups, and three measured runs. Values are medians of the three runs; ratio is native / ORT.

## Results

| model | native tok/s | ORT tok/s | native / ORT |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 910.52 | 583.87 | **1.559×** |
| Qwen2.5-7B | 302.27 | 272.53 | **1.109×** |
| Phi-4-mini | 322.25 | 237.16 | **1.359×** |

## Verdict

This regression-confirmation run followed several CPU-optimization-team PRs merging into shared main. It confirms no CUDA decode regression: native remains faster than onnxruntime-genai on every on-box model, by **1.11×–1.56×**.

## Caveats

The host CPU was contended by another team, so absolute wall-time and tok/s values can be noisy between runs. The native-vs-ORT ratio under identical same-GPU, same-pin conditions is the meaningful signal. These are single-H200, single-stream greedy-decode results, not multi-GPU or batched throughput claims.

## Command shape

```bash
CUDA_VISIBLE_DEVICES=6 taskset -c 1 target/release/profile_native \
  --model <model-dir> --backend <native|ort> --ep cuda \
  --prompt "The capital of France is" \
  --steady --warmups 2 --runs 3 --tokens 128
```
