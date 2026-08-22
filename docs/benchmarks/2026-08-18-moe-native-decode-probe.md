---
title: "First native-CUDA MoE decode: does onnx-genai run a real MoE model at all?"
date: 2026-08-18
hardware: "Intel i7-13800H (14C/20T), RTX 4060 Laptop 8 GB, WDDM, driver 591.55, CUDA 13.1"
model: "granite-3.0-1b-a400m-f16-mobius (granitemoe, 32 experts, top-8, 24 layers, dense_fallback f16)"
binary: "onnx-genai-cli release, --features native-cuda (native CUDA EP), ORT support lib 1.27.0 (CPU package)"
status: measured (end-to-end run, wall-clock, graph op inventory) + inferred (why-slow attribution) — separated
---

# First native-CUDA MoE decode

Every MoE-offload finding this week has been a trace-driven *measurement of a
ceiling*. None of them confirmed the **floor**: can `onnx-genai`'s native CUDA
path decode a real MoE model at all? With `granite-1b-a400m-f16-mobius` (fits in
8 GB, real trained router) that question is finally testable. This is a
**scoping probe**, not an optimisation — report first, build nothing.

**Box:** i7-13800H (14C/20T), **RTX 4060 Laptop 8 GB**, WDDM, driver 591.55,
CUDA 13.1. **Binary:** `onnx-genai` release built `--features native-cuda`.
**Model:** `granite-1b-a400m-f16-mobius`. Greedy, `--raw`. Reproduce below.

---

## Q1 — Does the native CUDA path decode granite? **Yes. (MEASURED)**

```
onnx-genai generate <model> "<prompt>" --backend native --device cuda --greedy --raw --max-new-tokens N
```

It runs end to end and produces **coherent** text (e.g. *"A mixture-of-experts
model is a type of probabilistic model that combines the predictions of multiple
expert models to make a final prediction…"*). Weights are **fully resident** in
VRAM (`total_weight_bytes` ≈ 2.78 GB, `fits_resolved_device_budget = true`,
`weight_offload_enabled = false`, **0 page-ins** — no offload exercised).

**Measured throughput (2 runs, greedy):**

| run | tokens | model load | TTFT | decode tok/s | ITL mean / p50 / p90 / p99 (ms) |
|---|---|---|---|---|---|
| 128-tok | 128 | 5.3 s | 1434 ms | **1.11** | 902 / 826 / 1239 / 1792 |
| 96-tok  | 96  | 11.6 s (cold) | 5332 ms | **1.21** | 826 / 770 / 930 / 2208 |

So the floor **exists** — but it is **~1.1–1.2 tok/s**, roughly 900 ms/token, for
a model with only ~400M active parameters on a GPU that should do far better.
Raw JSON: `data/granite-native-cuda-profile.json`.

### Why so slow (INFERRED — attribution, not a fresh measurement)

Two structural facts from the run log and the graph explain it; neither is an
op-support gap:

1. **CUDA graph capture is DECLINED**, so every step runs **eagerly, per-op**.
   The log: capture predicate `persistent_inputs_have_fixed_logical_shapes`
   declines because the KV inputs *"expose a growing logical prefix instead of
   fixed capacity; decode continues eagerly."* (physical `[1,8,256,64]` vs a
   logical prefix that grows `0,1,2,…`). No graph replay ⇒ full per-op launch
   overhead every token.
2. **`dense_fallback` recomputes all 32 experts every layer.** The exported
   graph is **9 582 nodes** of pure `ai.onnx` (see Q4): 2 425 `MatMul`,
   768 `Equal` (32 experts × 24 layers, one-hot masks), 768 `Sigmoid`,
   768 `ReduceSum`, 3 121 `Mul`. All 32 experts are computed and masked — there
   is no sparse gather — so the eager launch count per token is enormous.

The slowness is therefore a **launch-bound eager-execution + dense-MoE** effect,
**not** a missing kernel. (This is an inference from the log + op inventory; it is
not a profiled per-op breakdown, which would be the next measurement if the owner
wants the fix scoped.)

---

## Q2 — How does it compare to ORT? **No ORT baseline is obtainable on this binary. (MEASURED failure)**

The `native-cuda` CLI bundles the **CPU-only** ORT package: `onnxruntime.dll` +
`onnxruntime_providers_shared.dll` are present, but **no
`onnxruntime_providers_cuda.dll`**. So `--backend ort` (and `auto`, which
resolves to ORT for this model) runs on the **CPU EP**, and it **fails at layer 0**:

```
[E:onnxruntime] Non-zero status code returned while running Attention node.
Name:'model/layers.0/self_attn/Attention_node_29'
attention_helper.h:147 ... attn_mask->Shape()[last] == parameters.total_sequence_length was false.
inconsistent total_sequence_length (between attn_mask and past_key and past_value)
```

ORT 1.27.0's **CPU `Attention` kernel** rejects this export's
attn_mask/total_sequence_length shaping (the growing-KV-prefix decode shape).
Full log: `data/granite-ort-cpu-attention-failure.txt`.

**Consequences, stated plainly:**
- We **cannot** produce a native-vs-ORT tok/s A/B on granite with this binary —
  ORT cannot execute the model here at all.
- This is **not** a "no CUDA" claim: our **native path used the GPU** (VMM arena,
  resident weights). It is a *packaging* fact — this raw binary ships CPU ORT; the
  ORT **CUDA** EP is only wired through the Python `onnxruntime-gpu` wheel path,
  which this binary does not load.
- Net finding: **the native path runs a MoE model that this binary's ORT cannot.**
  That is a real result even though it denies us the intended A/B.

---

## Q3 — If it did not run, why? **N/A — it runs.**

No op is missing on the native path. The only "failure" is the *performance*
diagnosis in Q1 (capture decline + dense-MoE launch count) and the *ORT-side*
failure in Q2 (ORT CPU `Attention` op). No node needs a new native kernel to make
granite decode; making it **fast** is a separate, later question (graph-capture
under growing KV, and/or sparse expert dispatch).

---

## Q4 — Dense vs quantised: which path did this exercise? **Dense only. (MEASURED)**

`granite-1b-a400m-f16-mobius` is `representation: dense_fallback`, **f16**. Its
graph is **pure `ai.onnx` (opset 24) + `com.microsoft:1`**, with **zero**
`pkg.nxrt` / `QMoE` / `BlockQuantizedMoE` / quantized nodes:

```
opset: ai.onnx=24, com.microsoft=1 ; 9582 nodes
  3121 Mul   2425 MatMul   770 Cast   768 Sigmoid   768 Equal   768 ReduceSum
  744 Add   48 RotaryEmbedding   24 Attention   24 TopK   24 Softmax   1 RMSNormalization ...
  native/quant/MoE ops: NONE
```

**This matters for scope:**
- This run exercises **dense MoE via standard ops**. It does **not** touch the
  native **`BlockQuantizedMoE` / `QMoE`** CUDA kernels
  (`crates/onnx-runtime-ep-cuda/src/kernels/{qmoe,block_quantized_moe,qmoe_gemm,qmoe_grouping}.rs`),
  which are a **different code path** with their own GPU tests. **A green granite
  run does not imply the quantised path works.**
- The engine only routes to the native backend automatically when a model
  contains `pkg.nxrt::BlockQuantizedMatMul`
  (`model_proto_requires_native_backend`, `engine/decode_backend.rs`). granite has
  none, so it is ORT-by-default and had to be forced with `--backend native`.
- **The offload work depends on the quantised path**, not this one. File-backed /
  paged experts presuppose the canonical int4 `BlockQuantizedMoE` layout (the only
  file-backable one). That path is **untested by this fixture**: this run had
  `weight_offload_enabled = false` and **0 page-ins**. We have now confirmed the
  *dense* floor; the *quantised + offload* floor remains unconfirmed and needs a
  QMoE fixture that fits 8 GB (or a `VRAM_LIMIT`-constrained oversubscription
  setup) to exercise.

---

## What this establishes / does not
- **Measured:** native CUDA decodes granite end-to-end with coherent output;
  wall-clock TTFT + decode tok/s (1.11–1.21 tok/s) with weights fully resident;
  graph-capture declined; the exact ORT CPU-EP `Attention` failure; the full op
  inventory (dense, no quant/QMoE nodes).
- **Inferred:** the *cause* of the 1.1 tok/s (eager per-op launch × dense 32-expert
  recompute) — reasoned from the log + op count, not a profiled per-op breakdown.
- **Not established:** a native-vs-ORT A/B (ORT cannot run the model on this
  binary); any number for the **quantised `BlockQuantizedMoE`** path or for
  **weight offload/paging** (neither was exercised); and whether graph capture
  under a growing KV prefix would recover the throughput (that is the obvious next
  probe, deliberately not done here — report first).

### Reproduce
```powershell
$nv = "$env:LOCALAPPDATA\anaconda3\Lib\site-packages\nvidia"
$env:PATH = "$nv\cu13\bin\x86_64;$nv\cudnn\bin;$env:PATH"
cargo build -p onnx-genai-cli --bin onnx-genai --release --features native-cuda
$m = "C:\path\to\granite-1b-a400m-f16-mobius"
# native CUDA (runs):
.\target\release\onnx-genai.exe --profile generate $m "..." --backend native --device cuda --greedy --raw --max-new-tokens 128
# ORT (fails on this binary's CPU EP):
.\target\release\onnx-genai.exe generate $m "hi" --backend ort --greedy --raw --max-new-tokens 8
```
