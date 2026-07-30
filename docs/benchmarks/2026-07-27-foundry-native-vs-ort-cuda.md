# Foundry Local — Native CUDA EP vs ORT decode throughput (H200)

**Date:** 2026-07-27
**Engineer:** Cohaagen (perf)
**Mission:** Measure the native CUDA EP vs ORT decode-throughput gap on real Foundry
Local int4 deployment targets, and localize bottlenecks where native is slower.

## Hardware / software

| Item | Value |
| --- | --- |
| GPU | NVIDIA H200 (device 0, pinned `CUDA_VISIBLE_DEVICES=0 taskset -c 0`) |
| ONNX Runtime | 1.27.0 (API 27), CUDA EP — `.ort-cuda-1.27/root/lib/libonnxruntime.so.1.27.0` |
| Tool | `onnx-genai-bench` `profile_native` (`--features bench-native,bench-ort,cuda`) |
| Mode | `--steady` steady-state decode, `--tokens 128 --warmups 2 --runs 3` (median of 3 internal runs) |
| Prompt | "Explain the theory of relativity in detail." |
| Sampling | OFF (greedy, byte-identical fast path) |

> Note: the ORT backend requires `ONNX_GENAI_ORT_LIB` to point at the CUDA-enabled
> ORT shared library. `.cudaenv.sh` alone leaves it unset, so `profile_native`
> falls back to the CPU-only prebuilt ORT (`CUDAExecutionProvider` not exposed).
> All ORT runs below were taken with
> `ONNX_GENAI_ORT_LIB=$ORT_ROOT/lib/libonnxruntime.so.1.27.0`.

## Results

Each model was run at least twice per backend; medians below are steady-state
decode tok/s. Spread across repeats was <~2% after discarding first-touch
outliers (see 0.5B note).

| Model (Foundry int4, CUDA) | Native tok/s | ORT tok/s | Ratio (native/ort) | Verdict |
| --- | --- | --- | --- | --- |
| qwen2.5-0.5b-instruct | **995** | 580 | 1.72× | **native faster** |
| qwen2.5-1.5b-instruct | **720** | 438 | 1.64× | **native faster** |
| qwen2.5-7b-instruct   | **308** | 272 | 1.13× | **native faster** |
| Phi-4-mini-instruct   | **315** | 231 | 1.36× | **native faster** |

Per-repeat medians (steady tok/s):

| Model | Native reps | ORT reps |
| --- | --- | --- |
| qwen2.5-0.5b | 1001.9, 995.9, 993.6, 986.5 (rep discarded: 918.6 first-touch blip) | 581.7, 577.6 |
| qwen2.5-1.5b | 720.0, 721.1 | 436.2, 439.6 |
| qwen2.5-7b   | 307.5, 308.1 | 270.3, 274.2 |
| Phi-4-mini   | 314.2, 315.0 | 230.9, 230.7 |

**Headline:** Native CUDA EP is faster than ORT on every Foundry Local target
measured — from +13% on the 7B up to +72% on the 0.5B. The advantage shrinks as
the model grows (0.5B 1.72× → 7B 1.13×), consistent with the decode becoming
increasingly GEMM/memory-bandwidth-bound where both backends converge on the same
underlying int4 matmul kernels; native's per-step scheduling/launch overhead
savings dominate on the small models.

**No model showed native < ort**, so the bottleneck-localization / `--trace`
step (which applies only to native-slower cases) was not triggered.

## Correctness sanity (native vs ort, first ~30 tokens)

No garbage, repetition, or spacing artifacts observed on any model. ✅

### qwen2.5-0.5b — native
> " The theory of relativity is a set of principles that describe the physical
> laws of the universe at the speed of light. The theory of relativity was
> developed by Albert Einstein in the 20th century and has been the foundation…"

### qwen2.5-0.5b — ort
> " The theory of relativity is a set of principles that describe the physical
> laws of the universe at the speed of light. The theory of relativity was
> developed by Albert Einstein in the 20th century and has been the foundation…"

(0.5B native and ort are **token-identical**.)

### qwen2.5-7b — native
> " The theory of relativity is a fundamental concept in modern physics that
> describes the relationship between space and time. It was developed by Albert
> Einstein in the early 20th century and consists of two parts: the special
> theory of relativity and the general theory of relativity…"

### qwen2.5-7b — ort
> " The theory of relativity is a fundamental concept in modern physics that
> describes the relationship between space and time. It was developed by Albert
> Einstein in the early 20th century and consists of two parts: the special
> theory of relativity and the general theory of relativity…"

(7B native and ort are **token-identical**.)

Additional coherence checks: Phi-4-mini native/ort token-identical; qwen2.5-1.5b
native and ort diverge after the first sentence (expected under greedy decode
from tiny int4 logit differences) but both remain fully coherent English with no
repetition or spacing defects — **not** a regression flag.

## Bottleneck notes

None required — native is faster on all four targets. The narrowing margin on
the 7B (1.13×) is the natural place to look next if we want to widen native's
lead on large models, but optimization is explicitly out of scope for this
baseline (separate reviewed PR).

## Reproduction

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export ONNX_GENAI_ORT_LIB="$ORT_ROOT/lib/libonnxruntime.so.1.27.0"
cd /home/justinchu/onnx-genai
cargo build --release -p onnx-genai-bench \
  --features bench-native,bench-ort,cuda --bin profile_native

for backend in native ort; do
  CUDA_VISIBLE_DEVICES=0 taskset -c 0 \
    target/release/profile_native \
      --model <MODEL_DIR> --ep cuda --backend $backend \
      --steady --tokens 128 --warmups 2 --runs 3 \
      --prompt "Explain the theory of relativity in detail."
done
```
