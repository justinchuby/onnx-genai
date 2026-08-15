# Foundry Local vs onnx-genai: model-vs-runtime isolation (CPU)

**Author:** Sebastian (Performance) · **Date:** 2026-07-13 · **Machine:** Apple M1 Max

## TL;DR

The decisive experiment — running **Foundry Local's exact CPU `model.onnx` through
our runtime** — shows **decode parity**, not an FL advantage:

- OURS-on-FL-model **~215 tok/s** (direct) ≈ FL-on-FL-model **~200-212** ≈
  OURS-on-our-model **~206**. Feeding FL's model into our runtime does not speed us
  up. → The win is **neither model structure nor ORT kernel**.
- Warm HTTP: short decode **211.8 (ours) vs 212.1 (FL)** — dead even; **long decode
  175.0 (ours) vs 159.8 (FL)** — **we now lead** after the fp32-GQA shared-KV fix.
- Foundry Local's C++ SDK sets **zero custom ORT SessionOptions** — it delegates to
  onnxruntime-genai. We already match its posture (ORT_ENABLE_ALL + IO binding +
  shared-KV buffer). **No missing session option.**
- FL's fused QKV (121 vs our 169 `MatMulNBits`, 48 fewer dispatches/token) is
  **decode-neutral** on CPU (bandwidth-bound); the only residual FL edge is
  **TTFT/prefill (~2-4%)**.
- The 2026-07-12 "FL leads 202.7 / 165.8" gap was **pre-KV-fix + thermal/under-warmed
  sampling**, not a reproducible decode deficit.

## Machine and protocol

| field | value |
|---|---|
| machine | MacBook Pro 18,2; Apple M1 Max |
| CPU | 10 cores (8 performance + 2 efficiency) |
| memory | 32 GB |
| OS | macOS 26.5.1 (build 25F80); Darwin arm64 |
| onnx-genai commit | `8370d4781af5618589b7b3895f7abfeb352c971e` (working tree dirty: long-context KV fix in `decode.rs`) |
| Foundry Local | SDK 1.2.3; onnxruntime-genai CPU; ORT 1.26 |
| onnx-genai runtime | CPU EP; ORT 1.27; `optimization_level = ORT_ENABLE_ALL`; ORT-default threads |
| protocol (HTTP) | OpenAI streaming, greedy `temperature=0`, `max_tokens=64`; **2 discarded warmups + 5 measured**; median / p90 |
| protocol (direct) | `profile_decode` engine loop, `temperature=0`, 64 decode tokens, 2 warmups + 5 runs, interleaved |

Decode tok/s excludes TTFT: `(completion_tokens - 1) / (total - TTFT)`.

## Models under test

Both models are Qwen2.5-0.5B-Instruct, int4 block-32 `MatMulNBits`, all `accuracy_level=4`, GroupQueryAttention (14 attn / 2 KV heads, head_dim 64), fp32 KV, growing-KV export.

| model | source | MatMulNBits | nodes | embedding | Q/K/V |
|---|---|---:|---:|---|---|
| **Foundry Local CPU** | `qwen2.5-0.5b-instruct-generic-cpu:4` (822 MB) | **121** | **299** | fp32 `Gather` | **fused `qkv_proj` (N=1152)** |
| **onnx-genai cpu-recipe** | Mobius from Q4_0 GGUF | 169 | 394 | `GatherBlockQuantized` (int4) | three matmuls (896/128/128) |

FL model `model.onnx` SHA-256 `997228203ae563c7871e0d78e45e35f9062009822d5a945cf54091f14098cd21` — byte-identical to the 2026-07-12 benchmark's FL model, so results are directly comparable.

## TASK A — the decisive experiment: run FL's exact model through OUR runtime

I loaded FL's cached CPU `model.onnx` into onnx-genai (CPU EP) with a hand-written
`inference_metadata.yaml` (GQA 14/2, head_dim 64, fp32 KV, max_seq 4096). Output
was coherent ("... capital of France is Paris. It is the largest city in France
..."). FL's model has **no `position_ids` input**; our binder skips absent inputs,
so it ran unmodified.

**Direct-engine decode (`profile_decode`, interleaved to cancel thermal drift):**

| run | OURS on **FL model** | OURS on **our model** |
|---|---:|---:|
| rep 1 | 215.2 tok/s | 200.6 tok/s |
| rep 2 | 186.7 tok/s | 209.1 tok/s |
| rep 3 | 216.8 tok/s | 206.1 tok/s |
| **median** | **~215** | **~206** |

**HTTP decode, warm, interleaved OURS vs FL (median / p90):**

| workload | OURS-http | FL-http |
|---|---:|---:|
| short (59 tok) decode | **211.8 / 214.3** | 212.1 / 214.8 |
| long (858 tok) decode | **175.0 / 176.6** | 159.8 / 165.7 |
| short TTFT ms | 78.2 / 79.6 | **76.4 / 78.4** |
| long TTFT ms | 1075.6 / 1080.5 | **1036.8 / 1041.1** |

### Verdict: PARITY on decode — neither model structure nor ORT kernel is the lever

- **OURS-on-FL-model (~215 direct, ~212 HTTP) ≈ FL-on-FL-model (~200-212) ≈ OURS-on-our-model (~206-212).** Running FL's exact ONNX through our runtime does **not** make us faster — it lands at the same speed as our own model.
- Therefore FL's **model structure gives no decode advantage that our runtime is failing to exploit**, and the decode **kernel is at parity** (both are ORT `MatMulNBits accuracy_level=4`).
- On **long context, onnx-genai now leads FL (175.0 vs 159.8 tok/s)** after the fp32-GQA shared-KV-buffer fix (`decode.rs`). The 2026-07-12 gap (our 115.2 long) was that pre-fix growing-KV bug plus thermal-throttled sampling.
- The residual FL edge is **TTFT/prefill only** (~2-4%), not decode.

### Why the 2026-07-12 "FL leads 202.7 / 165.8" gap did not reproduce

The machine has large run-to-run variance (single unwarmed HTTP samples ranged
85-216 tok/s; the first request after model load is cold — ORT prepacking, arena
allocation, and paging the 800 MB-1.3 GB weights). With **2 discarded warmups**
our server steadies at ~212 short. The prior 158.6 short / 115.2 long was
(a) measured before the long-context KV fix and (b) under-warmed / throttled.
Foundry Local was stable at ~200-212 across both days; only our number moved.

## TASK A.3 — graph-structure diff (real, but decode-neutral here)

Per decoder layer:

| | Foundry Local | onnx-genai cpu-recipe |
|---|---|---|
| Q/K/V projection | **1** `MatMulNBits` (N=1152, fused) | **3** (N=896,128,128) |
| other matmuls/layer | o, gate, up, down (4) | o, gate, up, down (4) |
| **matmuls / layer** | **5** | **7** |
| total `MatMulNBits` | 121 (120 layer + 1 head) | 169 (168 layer + 1 head) |
| total graph nodes | 299 | 394 |
| embedding | fp32 `Gather` | int4 `GatherBlockQuantized` |
| residual `Add` nodes | 24 (1/layer) | 72 (3/layer) |

FL issues **48 fewer `MatMulNBits` dispatches per token** (2/layer × 24) by fusing
Q+K+V into one N=1152 matmul, and ~95 fewer graph nodes overall.

**But this is decode-neutral on CPU at this size.** Fusing Q/K/V does **not**
reduce weight bytes read (Q+K+V weights are identical whether one matmul or three);
at decode (M=1) `MatMulNBits` is **memory-bandwidth bound**, so fusion only saves
kernel-launch / thread-pool-barrier overhead — which is already negligible.
Measurement confirms it: FL's 121-node model (~215) and our 169-node model (~206)
decode at the same speed in the same runtime. Fused QKV is a graph-tidiness /
prefill-throughput nicety, **not** a decode win to chase.

## TASK B — Foundry Local C++ / ORT orchestration

Source: `/Users/justinc/Documents/GitHub/Foundry-Local/sdk_v2/cpp/src/`.

**Finding: Foundry Local sets essentially NO custom ORT SessionOptions.** It does
not construct raw ORT sessions at all — it delegates to **onnxruntime-genai**
(`OgaConfig` / `OgaModel::Create` / `OgaGenerator`) and lets that library own the
inference loop, IO binding, and KV cache.

- No `SetGraphOptimizationLevel`, `SetIntraOpNumThreads`, `SetInterOpNumThreads`,
  `SetExecutionMode`, `Enable/DisableMemPattern`, `Enable/DisableCpuMemArena`, or
  `AddConfigEntry` calls exist in the SDK tree. The raw ORT wrapper
  `predictive/inference_session.cc:85-107` is unimplemented (throws).
- Model path uses ORT-GenAI: `OgaConfig::Create`, `ClearProviders`,
  `AppendProvider`, `OgaModel::Create` at
  `inferencing/generative/genai_model_instance.cc:29-58`. The only provider knob
  is CUDA-only `enable_cuda_graph=0` (`genai_model_instance.cc:45-48`) — irrelevant
  on CPU.
- Decode is `OgaGenerator::GenerateNextToken()` + `GetNextTokens()`
  (`inferencing/generative/chat/onnx_chat_generator.cc:54-112`); prefill is
  `AppendTokenSequences` (`onnx_chat_generator.cc:267-390`). Append-only context
  growth and `RewindTo` reuse generator state without rebuilding the session
  (`onnx_chat_generator.cc:138-166`). **No `Ort::IoBinding` and no
  `past_present_share_buffer` code in FL's SDK** — both live inside
  onnxruntime-genai's C++ core.
- The CPU model's `genai_config.json` (`.../qwen2.5-0.5b-instruct-generic-cpu-4/v4`)
  confirms it: `decoder.session_options` = `{ log_id, provider_options: [] }` — no
  threading, graph-opt, or arena entries. Only `search.past_present_share_buffer:
  true`, which onnxruntime-genai consumes internally.

**We already match FL's ORT posture:** `session.rs:59` sets
`optimization_level = 99` (ORT_ENABLE_ALL) and leaves intra/inter-op at 0
(ORT decides) — identical to FL. Our runtime does its own IO binding
(`bind_input`/`bind_output`) and its own runtime-owned shared-KV-buffer path
(`decode.rs`), i.e. the same techniques onnxruntime-genai uses. **There is no
session option FL sets that we are missing.**

## Prioritized next steps

1. **Correct the record (highest value, no code).** onnx-genai is at CPU decode
   parity with Foundry Local on FL's own model and **ahead on long context**
   (175 vs 160 tok/s). Stop treating FL as a decode target to catch; the
   2026-07-12 gap was pre-KV-fix + thermal. Re-run the four-way benchmark with
   >=2 warmups to publish corrected numbers.
2. **Benchmark warmup discipline (harness).** First-request cold-start swings
   HTTP decode from ~120 to ~212 tok/s. Always discard >=2 warmups; consider a
   server startup priming inference so first real request is warm. *(Owner: Leon /
   Sebastian — server + harness; low-risk.)*
3. **TTFT / prefill is the only residual FL edge (~2-4%).** Chunked / prefill-
   specific path is the lever, not decode. *(Owner: Leon — runtime prefill.)*
4. **Fused QKV in Mobius is LOW priority for CPU decode** (bandwidth-bound;
   measured neutral). Worth doing only for prefill throughput / graph size, not
   decode tok/s. *(Owner: Sapper — Mobius model build.)*

## No runtime code change made

The task allowed implementing a missing low-risk session option. **None exists** —
FL sets no ORT session options we lack, and our runtime already matches/beats it
on decode. A speculative session tweak (e.g. disabling the memory arena) would be
unjustified and could regress. Documented above for follow-up instead.

## Reproduce

```bash
# 1. Download FL's CPU model (goes to ~/.onnx-genai-foundry-analysis/cache/... because app_name sets the cache dir):
conda run -n onnx python models/.scratch/fl_download_cpu.py   # then fl_load.py to materialize + get_path

# 2. Point our runtime at FL's model (symlinks + hand-written metadata):
#    models/foundry-cpu/{model.onnx,model.onnx.data -> FL cache; tokenizer.json; inference_metadata.yaml}

# 3. Decisive direct-engine decode (interleaved):
for M in foundry-cpu qwen2.5-0.5b-cpu-recipe; do
  ONNX_GENAI_EP=cpu ./target/release/profile_decode --model models/$M \
    --tokens 64 --warmups 2 --runs 5 --prompt "The capital of France is"
done

# 4. HTTP parity (start our server + FL service, then compare with >=2 warmups):
ONNX_GENAI_EP=cpu ./target/release/onnx-genai-server --model models/qwen2.5-0.5b-cpu-recipe --model-id recipe --addr 127.0.0.1:8091
conda run -n onnx python models/.scratch/fl_serve.py   # FL on :5273
./target/release/compare --runs 5 --warmups 2 --max-tokens 64 \
  --tokenizer models/qwen2.5-0.5b-cpu-recipe/tokenizer.json \
  --runtime 'ours|http://127.0.0.1:8091/v1|recipe|ONNX int4 169 MatMulNBits acc4|CPU EP' \
  --runtime 'FL|http://127.0.0.1:5273/v1|qwen2.5-0.5b-instruct-generic-cpu|ONNX int4 121 MatMulNBits fused-QKV|FL SDK 1.2.3 ORT-genai'
```
