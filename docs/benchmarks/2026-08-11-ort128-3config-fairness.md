# ORT 1.28 three-config decode fairness benchmark (2026-08-11)

Branch: `squad/ort128-3config-fairness` at `4dd5119b` (`origin/main`, includes #765 fused QMoE decode).  Host was shared; report medians and treat small deltas as noisy.  GPU pinned with `CUDA_VISIBLE_DEVICES=2`.

## Runtime confirmation

Rust engine runs used the freshly released ORT 1.28 library, not the stale build-baseline copy:

```text
onnx-genai: selected ONNX Runtime 1.28.0 (API 28) from /home/tlwu/git/onnxruntime/.venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.28.0 (ONNX_GENAI_ORT_LIB)
ort_available_execution_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

ORT-GenAI direct used system Python's `onnxruntime-genai` package and also loaded ORT 1.28:

```text
python_onnxruntime_version: 1.28.0 providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

The ORT 1.28 capi directory was combined with CUDA 12.9 libraries because ORT's CUDA provider depends on `libcublasLt.so.12`:

```text
/home/tlwu/git/onnxruntime/.venv/lib/python3.14/site-packages/onnxruntime/capi
/home/tlwu/cuda12.9/targets/x86_64-linux/lib
/home/tlwu/cudnn9.19_cuda13/lib
```

## Configs and measurement

- **A. Native CUDA EP**: `profile_native --ep cuda --backend native --steady --warmups 1 --runs 3 --tokens 128`, with `ONNX_GENAI_PROFILE_OPS=1`.
- **B. ORT as our backend**: `profile_native --ep cuda --backend ort --steady --warmups 1 --runs 3 --tokens 128`, with `ONNX_GENAI_ORT_LIB` pinned to ORT 1.28.
- **C. ORT-GenAI direct**: Python `onnxruntime_genai` API, CUDA provider, greedy (`do_sample=False`, `temperature=0.0`, `top_p=1.0`, `top_k=0`), same prompt (`Hello`) and `max_length = prompt + 128`.

Throughput is steady-state decode after skipping the first 8 generated tokens, matching `profile_native`'s default steady window.

## Follow-up: ORT-GenAI direct parity lock

Harry's review caught that Config C needed a stronger proof that it was running the same greedy decode contract as Config B.  I reran Config C with pre-tokenized raw prompt IDs (`Hello` -> `[9707]`), no chat template, and explicit greedy/no-penalty search options:

```text
do_sample=False
temperature=0.0
top_k=1
top_p=1.0
repetition_penalty=1.0
num_beams=1
```

I also probed `past_present_share_buffer=False` for DeepSeek; it did not change the sequence.

The ORT-GenAI API confirmed the effective options, e.g.:

```text
search_options: {... 'do_sample': False, 'repetition_penalty': 1.0, 'temperature': 0.0, 'top_k': 1.0, 'top_p': 1.0}
prompt_token_ids: [9707]
```

Parity-locked Config C results:

| Model | A native CUDA EP | B ORT backend | C ORT-GenAI direct, parity-locked | Native vs C | New token parity verdict |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B int4 | 559.03 tok/s (1.789 ms/tok) | 101.43 tok/s (9.859 ms/tok) | 227.32 tok/s (4.399 ms/tok) | **2.46x faster** | B=C exact for 128 generated tokens. A diverges from B/C at token index 68 (`2213` vs `8794`). |
| DeepSeek-R1-Distill-Qwen-1.5B int4 | 593.36 tok/s (1.685 ms/tok) | 226.06 tok/s (4.424 ms/tok) | 485.03 tok/s (2.062 ms/tok) | **1.22x faster** | C still diverges from both A and B at token index 10 (`25` vs `11`), while A and B agree until index 31 (`22756` vs `21495`). This is not sampling or chat-template drift: the run used raw prompt IDs and confirmed greedy/no-penalty options. Probed `past_present_share_buffer=False`, `top_k=0`, and BOS/prompt variants; none matched B. Remaining likely cause is an ORT-GenAI host/runtime path difference for this DeepSeek/Qwen artifact (KV/position/feed semantics or graph/session handling), despite both paths using ORT 1.28. |

## Section 1: apples-to-apples steady-state decode

Local artifact search found usable CUDA/int4 ONNX artifacts for Qwen2.5-0.5B and DeepSeek-R1-Distill-Qwen-1.5B only.  Qwen2.5-1.5B, Phi-4-mini, and Qwen2.5/Qwen 7B were skipped: local search found HF caches and Olive recipes/QNN configs, but no CUDA ONNX int4 artifact directory with `model.onnx` + tokenizer suitable for these three configs.

| Model | Artifact | A native CUDA EP | B ORT backend | C ORT-GenAI direct | Native vs C | Token parity verdict |
|---|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B int4 | `/home/justinchu/qwen2.5-0.5b-int4-onnx` | 557.57 tok/s (1.794 ms/tok) | 97.19 tok/s (10.289 ms/tok) | 203.22 tok/s (4.921 ms/tok) | **2.74x faster** | B=C exact for 128 tokens; A diverges from B/C at generated token index 68 (`2213` vs `8794`). |
| DeepSeek-R1-Distill-Qwen-1.5B int4 | `/home/justinchu/glm-e2e-artifacts/deepseek-r1-distill-qwen-1.5b-int4-cuda` | 599.49 tok/s (1.668 ms/tok) | 222.72 tok/s (4.490 ms/tok) | 487.26 tok/s (2.052 ms/tok) | **1.23x faster** | A=B through token 30, then diverge at token 31 (`22756` vs `21495`). C diverges from both A and B at token 10 (`25` vs `11`). |
| Qwen2.5-1.5B int4 | not found | skipped | skipped | skipped | n/a | No local CUDA ONNX int4 artifact found. |
| Phi-4-mini int4 | not found | skipped | skipped | skipped | n/a | No local CUDA ONNX int4 artifact found. |
| Qwen 7B int4 | not found | skipped | skipped | skipped | n/a | No local CUDA ONNX int4 artifact found. |

### Native per-op attribution snapshots

`ONNX_GENAI_PROFILE_OPS=1` is only emitted by the native executor path.  Last measured run top native op shares:

| Model | Top native op shares |
|---|---|
| Qwen2.5-0.5B | `GroupQueryAttention` 40.93%, `MatMulNBits` 29.58% |
| DeepSeek-R1-Distill-Qwen-1.5B | `GroupQueryAttention` 59.85%, `MatMulNBits` 34.04% |

### Generated token IDs and diffs

Qwen2.5-0.5B (`A` diverges from `B/C` at index 68; `B=C` exact):

```text
A native: [271, 40, 1079, 264, 48948, 304, 13027, 323, 358, 1079, 4460, 311, 1855, 264, 4285, 2025, 429, 15804, 264, 1034, 323, 23473, 700, 279, 8794, 315, 279, 1034, 13, 358, 614, 1012, 2701, 279, 43812, 323, 10295, 3897, 553, 279, 13027, 9705, 323, 14284, 42124, 13, 358, 614, 1083, 1012, 1667, 279, 1565, 2508, 63, 729, 311, 1787, 279, 1034, 323, 1565, 878, 63, 311, 1349, 279, 1034, 2213, 13, 4354, 11, 358, 1079, 537, 2704, 1246, 311, 10277, 3265, 279, 1034, 1283, 358, 614, 8060, 5290, 1181, 8794, 13, 358, 614, 1083, 1012, 1667, 1565, 4197, 63, 5114, 311, 5978, 429, 279, 1034, 374, 10277, 7877, 1283, 279, 1565, 4197, 63, 2504, 374, 51283, 13, 2980, 4325, 4486, 8474, 752, 389, 1246, 311, 10277, 3265, 279, 1034]
B ORT:    [271, 40, 1079, 264, 48948, 304, 13027, 323, 358, 1079, 4460, 311, 1855, 264, 4285, 2025, 429, 15804, 264, 1034, 323, 23473, 700, 279, 8794, 315, 279, 1034, 13, 358, 614, 1012, 2701, 279, 43812, 323, 10295, 3897, 553, 279, 13027, 9705, 323, 14284, 42124, 13, 358, 614, 1083, 1012, 1667, 279, 1565, 2508, 63, 729, 311, 1787, 279, 1034, 323, 1565, 878, 63, 311, 1349, 279, 1034, 8794, 13, 4354, 11, 358, 1079, 537, 2704, 1246, 311, 10277, 3265, 279, 1034, 1283, 358, 614, 8060, 5290, 432, 13, 358, 614, 1083, 1012, 1667, 1565, 4197, 63, 5114, 311, 5978, 429, 279, 1034, 374, 10277, 7877, 1283, 279, 1565, 4197, 63, 2504, 374, 51283, 13, 2160, 1052, 264, 1616, 311, 3265, 279, 1034, 1283, 358, 614, 8060, 5290]
C direct: [271, 40, 1079, 264, 48948, 304, 13027, 323, 358, 1079, 4460, 311, 1855, 264, 4285, 2025, 429, 15804, 264, 1034, 323, 23473, 700, 279, 8794, 315, 279, 1034, 13, 358, 614, 1012, 2701, 279, 43812, 323, 10295, 3897, 553, 279, 13027, 9705, 323, 14284, 42124, 13, 358, 614, 1083, 1012, 1667, 279, 1565, 2508, 63, 729, 311, 1787, 279, 1034, 323, 1565, 878, 63, 311, 1349, 279, 1034, 8794, 13, 4354, 11, 358, 1079, 537, 2704, 1246, 311, 10277, 3265, 279, 1034, 1283, 358, 614, 8060, 5290, 432, 13, 358, 614, 1083, 1012, 1667, 1565, 4197, 63, 5114, 311, 5978, 429, 279, 1034, 374, 10277, 7877, 1283, 279, 1565, 4197, 63, 2504, 374, 51283, 13, 2160, 1052, 264, 1616, 311, 3265, 279, 1034, 1283, 358, 614, 8060, 5290]
```

DeepSeek-R1-Distill-Qwen-1.5B (`A/B` diverge at index 31; `C` diverges from both at index 10):

```text
A native: [323, 9702, 498, 0, 358, 614, 264, 6888, 3491, 1588, 11, 323, 358, 1184, 311, 7071, 700, 1246, 311, 11625, 432, 13, 576, 3491, 374, 911, 9271, 279, 3082, 315, 264, 22756, 13, 576, 22756, 702, 264, 3084, 315, 220, 16, 17, 8153, 323, 264, 2374, 315, 220, 23, 8153, 13, 358, 1184, 311, 1477, 279, 3082, 304, 9334, 8153, 13, 358, 2776, 264, 2699, 21815, 911, 1246, 311, 5486, 419, 11, 714, 358, 1744, 358, 646, 7071, 432, 700, 553, 14719, 432, 1495, 3019, 553, 3019, 13, 6771, 752, 1430, 311, 975, 1526, 432, 382, 5338, 11, 358, 1414, 429, 279, 3082, 315, 264, 22756, 374, 16588, 553, 84192, 1181, 3084, 553, 1181, 2374, 13, 2055, 11, 304, 419, 1142, 11, 358, 1184, 311, 30270, 220, 16]
B ORT:    [323, 9702, 498, 0, 358, 614, 264, 6888, 3491, 1588, 11, 323, 358, 1184, 311, 7071, 700, 1246, 311, 11625, 432, 13, 576, 3491, 374, 911, 9271, 279, 3082, 315, 264, 21495, 979, 2661, 279, 28316, 315, 1181, 2326, 11067, 13, 358, 6099, 2494, 911, 6252, 263, 594, 14806, 504, 847, 17047, 536, 11, 714, 358, 2776, 537, 11368, 2704, 1246, 311, 3796, 432, 13, 6771, 752, 1430, 311, 19091, 323, 975, 1526, 458, 3110, 382, 32313, 11, 773, 279, 3491, 374, 25, 7379, 279, 3082, 315, 264, 21495, 448, 11067, 315, 3084, 220, 20, 11, 220, 21, 11, 323, 220, 22, 8153, 13, 358, 1184, 311, 7071, 700, 1246, 311, 3796, 6252, 263, 594, 14806, 1588, 13, 358, 1744, 6252, 263, 594, 14806, 17601, 37614, 279, 18267]
C direct: [323, 9702, 498, 0, 358, 614, 264, 6888, 3491, 1588, 25, 330, 40, 1184, 311, 1477, 279, 897, 315, 856, 304, 279, 23606, 220, 17, 87, 488, 220, 18, 284, 220, 22, 1189, 358, 2776, 537, 2704, 1246, 311, 5486, 419, 13, 2980, 498, 1492, 752, 30, 4940, 3308, 11, 358, 646, 1492, 498, 448, 429, 13, 6771, 752, 1744, 911, 1246, 311, 11625, 419, 23606, 3019, 553, 3019, 382, 5338, 11, 358, 1184, 311, 42123, 279, 3890, 856, 13, 2014, 653, 429, 11, 358, 1265, 633, 9279, 315, 279, 6783, 4647, 389, 279, 2115, 3108, 315, 279, 23606, 13, 576, 23606, 374, 220, 17, 87, 488, 220, 18, 284, 220, 22, 13, 2055, 11, 279, 6783, 4647, 374, 220, 18, 13, 2014, 21725, 432, 11, 358, 646]
```

## Section 2: ORT 1.28 capability matrix

Capability probes used shorter runs (`--warmups 0 --runs 1 --tokens 16 --decode-skip 4`) to verify load/generate status.  These are capability checks, not fair throughput numbers.

| Model | Artifact | A native CUDA EP | B ORT backend (ORT 1.28) | C ORT-GenAI direct |
|---|---|---|---|---|
| Qwen3.6-35B-A3B QMoE | `/home/justinchu/qwen36-35b-a3b-qmoe-artifacts` (`--pipeline`) | **Runs**: 90.20 tok/s in short probe; generated 16 tokens. | **Crashes at load**: `Type parameter (T) of Optype (QMoE) bound to different types (tensor(float16) and tensor(float) in node (moe.layer0.qmoe)`. | Initially missing `genai_config.json`; generated text-only config also fails at ONNX load with the same QMoE type-binding error (details below). |
| GLM-4-9B int4 | `/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda` | **Unsupported native load**: model needs unsupported native operators. | **Crashes at load**: `Unrecognized attribute: rotary_embedding_dim for operator GroupQueryAttention`. | **Not runnable from this artifact**: missing `genai_config.json`. |
| DeepSeek-V2-Lite QMoE int4 | `/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4` | **Unsupported native load**: model needs unsupported native operators. | **Timed out** after 600 s after ORT session/backend selection; no token generated. | **Not runnable from this artifact**: missing `genai_config.json`. |
| Qwen3.6-27B int4 | `/home/justinchu/mary-models/qwen3.6-27b-int4-cuda` | **Unsupported native load**: model needs unsupported native operators. | **Loads, then generation setup fails**: `cannot resolve model.io.token_input ... 3 ports match: input_ids, attention_mask, position_ids; declare the exact graph port`. | **Not runnable from this artifact**: missing `genai_config.json`. |

### 35B-A3B ORT-GenAI direct follow-up

The root 35B-A3B artifact still has no `genai_config.json`, so I generated a scratch ORT-GenAI config against the available `merged/model.onnx` using the local HF Qwen3.6-35B-A3B config and the ORT-GenAI model type string present in the installed wheel:

- `model.type = qwen3_5_moe` first proved this is a multimodal/package model type and is not valid for the text-only merged ONNX: `The model path is a directory. Loading a model package from a directory path is not supported.`
- `model.type = qwen3_5_moe_text` reached ONNX session load and failed with the same ORT 1.28 QMoE type-binding error as Config B:

```text
RuntimeError: Load model from .bench/og35/model.onnx failed:
Type Error: Type parameter (T) of Optype (QMoE) bound to different types
(tensor(float16) and tensor(float) in node (moe.layer0.qmoe).
```

Verdict: with the available 35B-A3B ONNX artifact, ORT-GenAI direct does **not** bypass the QMoE type-binding failure.  The installed ORT-GenAI package contains Qwen3.5/Qwen3.5-MoE builder support, but running the full Olive export would be a separate long conversion/export job rather than a direct test of the current artifact.  For the current artifact, native CUDA EP remains the only tested runtime path that loads and generates.

## Headline

For the two locally available Section-1 artifacts, native CUDA EP beats ORT-GenAI direct's full-stack ceiling under parity-locked greedy decode:

- Qwen2.5-0.5B: **559.03 vs 227.32 tok/s** (**2.46x faster**) with token caveat: native diverges from ORT after token 68; ORT backend and ORT-GenAI direct match exactly.
- DeepSeek-R1-Distill-Qwen-1.5B: **593.36 vs 485.03 tok/s** (**1.22x faster**) with token caveat: ORT-GenAI direct still diverges from both engine paths at token 10 even with raw prompt IDs and explicit greedy/no-penalty options.

Capability check: ORT 1.28 did **not** make the 35B-A3B QMoE artifact runnable through ORT backend; it still fails QMoE type binding at load.  A generated text-only ORT-GenAI config reaches the same QMoE type-binding failure, so ORT-GenAI direct does not currently run the available 35B-A3B artifact either.
