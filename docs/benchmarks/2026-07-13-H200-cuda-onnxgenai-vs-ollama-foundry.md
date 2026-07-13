# H200 CUDA: onnx-genai vs Ollama vs Foundry Local (Qwen2.5-0.5B)

First correctness-verified GPU benchmark of the onnx-genai **CUDA** path against
the two llama.cpp / ONNX-Runtime GPU peers, after the CUDA-graph decode capture
and vectorized-argmax work landed on `fix/cuda-shared-buffer-kv`.

## TL;DR

- **onnx-genai wins TTFT decisively** (68 ms vs Ollama 165 ms / Foundry 132 ms)
  and beats Foundry Local on every axis.
- **End-to-end total latency is a dead heat with Ollama** (2199 vs 2185 ms for
  1024 greedy tokens): our far lower prefill/TTFT cancels Ollama's faster
  steady-state decode.
- On **steady-state decode tok/s** onnx-genai trails Ollama's hand-tuned Q4_K_M
  GEMV by ~5% (481 vs 507). The gap is **not** weight-bandwidth in the
  `MatMulNBits` kernel (a from-source ORT 1.28 build with the TRT-LLM `fpA_intB`
  fast path does not help — see the update below); it is dominated by ORT running
  ~366 small kernels/token that underfill the H200 SMs. Closing it needs operator
  fusion, not a faster int4 kernel.
- This cycle moved onnx-genai HTTP decode **402 → 471 → 481 tok/s**
  (CUDA graph, then vectorized greedy argmax).

## Environment

| | |
|---|---|
| GPU | NVIDIA H200 (143 GB), driver 580.105.08 |
| onnx-genai | commit `46dcfe6` (`fix/cuda-shared-buffer-kv`); CUDA EP; ORT 1.27.0 |
| flags | `ONNX_GENAI_EP=cuda ONNX_GENAI_DEVICE_KV=1 ONNX_GENAI_CUDA_GRAPH=1` |
| onnx-genai model | Foundry `qwen2.5-0.5b-instruct-cuda-gpu-4/v4` (int4 `MatMulNBits`, 24× GQA, fp16 KV) |
| Foundry Local | `qwen2.5-0.5b-instruct-cuda-gpu` (`cuda-gpu-4` variant); onnxruntime-genai CUDA |
| Ollama | `qwen05-cuda:latest`, GGUF **Q4_K_M** (491 MB), all layers on GPU (llama.cpp) |

## Protocol

- OpenAI-compatible **HTTP streaming** to every runtime through the identical
  client (`.bench_long.py`), so server + protocol overhead is charged equally.
- Qwen2.5-0.5B-Instruct, greedy (`temperature=0`, `top_p=1`, `seed=0`),
  `max_tokens=1024`, one long ~1500-word essay prompt (~40 prompt tokens).
- 1 discarded warmup + **5 measured runs**; medians reported.
- Decode tok/s excludes TTFT: `(completion_tokens - 1) / (total - TTFT)`.

## Results (1024 output tokens, greedy, medians of 5)

| runtime | TTFT ms | decode tok/s | total ms | out tok |
|---|--:|--:|--:|--:|
| **onnx-genai CUDA** | **68.4** | 481.2 | 2199.1 | 1024 |
| Foundry Local | 131.8 | 452.6 | 2392.3 | 1024 |
| Ollama Q4_K_M | 165.2 | **507.0** | **2184.7** | 1024 |

- onnx-genai vs **Foundry**: +6.3% decode, −48% TTFT, −8.1% total — a clean win.
- onnx-genai vs **Ollama**: −5.1% decode, **−59% TTFT**, +0.7% total — parity on
  wall-clock latency; we win prefill, they win steady decode.

## Where the onnx-genai per-token time goes (profiler, graph on, 1024 tok)

`ONNX_GENAI_PROFILE=1 profile_decode`, µs/token:

| stage | µs/token | note |
|---|--:|---|
| `ort.session_run` | ~1818 | int4 `MatMulNBits` GEMV + 366 GQA/proj kernels; weight-bandwidth bound |
| `engine.logits_to_vec` | ~79 | fp16→f32 conversion of the 152k-vocab logits |
| `loop.sampling` | ~64 | vectorized greedy argmax |

### CUDA-graph isolation (profiler, 1024 tok)

| | decode tok/s | `session_run` µs/token |
|---|--:|--:|
| graph **off** | 443 | 1941 |
| graph **on** | 475 | 1802 |

Capturing and replaying the static-shape decode step removes ~139 µs/token of
kernel-launch overhead (+7%). The launch overhead is smaller than on latency-
bound hardware because H200 `session_run` is dominated by actual GEMV compute,
not launches — so the remaining decode gap to llama.cpp lives in the kernel.

## Correctness

Generated text is **bit-identical between graph-on and graph-off** across
single- and multi-generation runs (server reuses the session per request); the
multi-generation heap-corruption and frozen-output bugs found while bringing up
graph capture are fixed (unique `gpu_graph_id` per generation; CPU inputs
re-bound every step). See the runbook §8 troubleshooting.

## What would close the last 5% on decode

### Update (2026-07-13): the ORT int4 kernel knobs are exhausted

A from-source ORT **1.28** CUDA build (SM90a, `onnxruntime_USE_FPA_INTB_GEMM=ON`)
was built and benchmarked to test the two candidate kernel levers. Both are
**dead ends** for a 0.5B model on H200:

| lever | result |
|---|---|
| `accuracy_level=4` (int8/SDOT) | **no-op on CUDA** — the CUDA `MatMulNBits` ctor never reads it; compute is always in activation dtype (fp16). CPU-only win. A/B confirmed (495.3 vs 493.1). |
| TRT-LLM `fpA_intB` fast path (`ORT_FPA_INTB_GEMM`) | **no help / crashes.** Built a `block_size=128` int4 model (from Qwen GPTQ-Int4, symmetric, N%64==0) that satisfies the fast-path gate. `=2` (GEMV) is ~5% *slower* (431 vs 454); `=1`/`=4` **crash** at prefill (`fpA_intB_gemv/dispatcher.h:387 unsupported m`) because CUTLASS has no GEMM tile for our tiny N (128/224) and the fallback picks the GEMV kernel for m≥16. |
| ORT 1.27 → 1.28 default kernel | **neutral** — ~486 tok/s on both at 1024 tokens. |

The TRT-LLM `fpA_intB` kernels are tuned for **large-N** LLMs; our N=128/224/896
shapes are too small to benefit.

### The real bottleneck is kernel count, not weight bandwidth

`session_run` ≈ 1800 µs/token, but Qwen-0.5B's int4 weights are only ~0.4 GB →
~83 µs at H200's 4.8 TB/s HBM3e. **We are not weight-bandwidth bound.** The time
is dominated by **~366 tiny kernels/token** that each underfill the 132 SMs and
serialize on one stream. CUDA graph already removes the *launch* overhead
(443 → 475 tok/s, `session_run` 1941 → 1802 µs) but cannot fix per-kernel SM
underutilization. That is precisely llama.cpp's edge: far fewer, fused, larger
kernels. Closing the last ~4% needs **operator fusion / a fused-attention path**,
not a faster `MatMulNBits` — a much larger effort than any env-var or kernel swap.

- A cheaper in-tree option: an fp16-argmax fast path that skips the 79 µs
  `logits_to_vec` conversion for pure-greedy requests (~+15 tok/s), at the cost
  of a greedy-only code path with fp16 NaN-handling care.

### Bottom line

On the metric that matters — end-to-end latency — we are already at parity with
Ollama and win decisively on TTFT. The decode gap is ~4% and only recoverable
via custom fused kernels, not via ORT's existing int4 knobs.
