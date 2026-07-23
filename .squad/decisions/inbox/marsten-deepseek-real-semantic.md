### 2026-07-23: Real DeepSeek-V2-Lite int4 semantic validation is blocked by block-128 MatMulNBits
**By:** Marsten
**What:** ❌ **Broken.** The real-weight int4 artifact cannot complete its first
native CUDA prefill, so it produces no native tokens or decoded text. Strict CUDA
placement succeeds (zero CPU EP fallbacks), but layer 0 `q_proj` fails because all
189 `MatMulNBits` nodes use block size 128 while the native fp16 CUDA
`MatMulNBits` path supports only block size 32.
**Why:** This is an export/runtime layout incompatibility before the first MLA
Attention or QMoE executes, not semantic drift from int4 quantization. Re-export
the dense projections with block size 32 (Mobius/export owner), or separately add
native fp16 block-128 `MatMulNBits` support (CUDA kernel owner). QMoE also declares
block size 128, but it was not reached and is not established as failing.

## Reproduction

- Branch: `bench/ort-vs-native-cuda`
- GPU: physical GPU 0, NVIDIA H200, idle at launch
- Artifact:
  `/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4/`
- Native mode: `CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_REQUIRE_CUDA=1`,
  `profile_native --ep cuda --backend native --tokens 64 --warmups 0 --runs 1`
- Prompt (rendered DeepSeek chat template):

```text
<｜begin▁of▁sentence｜>User: Answer in one concise sentence: What is the capital of France and what landmark is it famous for?

Assistant:
```

The native and HF reference prompt token IDs match:

```text
[100000, 5726, 25, 35829, 279, 634, 46019, 4976, 25, 2461, 317,
 254, 6077, 280, 7239, 285, 856, 44872, 317, 359, 9679, 327, 30,
 185, 185, 77398, 25]
```

## Native CUDA result

- Decoded text: **none — generation failed before token 1**
- Generated tokens: **0/64**
- Native throughput: **N/A**
- Finite-token check: **N/A; no logits/tokens were produced**
- CPU EP fallbacks: **0** (`ONNX_GENAI_REQUIRE_CUDA=1` accepted placement)
- CUDA graph fallback count: **N/A; execution failed during initial prefill**
- QMoE/MLA execution: **not reached**

Actual error:

```text
node 18 ("model/layers.0/self_attn/q_proj/MatMulNBits_node_20",
op 'com.microsoft::MatMulNBits') failed:
MatMulNBits CUDA fp16 activations received block_size=128.
The native fp16 decode and prefill kernels implement the block-32 packed layout.
```

ONNX inspection confirms:

```text
MatMulNBits: 189, all block_size=128
QMoE: 26, declared block_size=128
Attention: 27 (default ONNX domain)
GroupQueryAttention: 0
```

## HF bf16 reference

Transformers 4.39.3, `trust_remote_code=True`, eager attention, bf16, greedy,
64 new tokens:

```text
 The capital of France is Paris. The most famous landmark in Paris is the Eiffel Tower.

User: What is the capital of France and what landmark is it famous for?

Assistant: The capital of France is Paris. The most famous landmark in Paris is the Eiffel Tower.

User:
```

HF reference throughput was 9.03 tok/s (64 tokens in 7.086 s).

## Early token comparison

Native has no token 1, so the divergence point is **before token 1**; semantic
top-1 agreement cannot be measured.

| # | Native ID / text | HF bf16 ID / text |
|---:|---|---|
| 1 | N/A | `429` / `" The"` |
| 2 | N/A | `6077` / `" capital"` |
| 3 | N/A | `280` / `" of"` |
| 4 | N/A | `7239` / `" France"` |
| 5 | N/A | `317` / `" is"` |
| 6 | N/A | `8913` / `" Paris"` |
| 7 | N/A | `13` / `"."` |
| 8 | N/A | `429` / `" The"` |
| 9 | N/A | `1094` / `" most"` |
| 10 | N/A | `9679` / `" famous"` |
| 11 | N/A | `44872` / `" landmark"` |
| 12 | N/A | `279` / `" in"` |
| 13 | N/A | `8913` / `" Paris"` |
| 14 | N/A | `317` / `" is"` |
| 15 | N/A | `254` / `" the"` |

## Verdict

❌ **Broken**, specifically an fp16 dense `MatMulNBits` block-size incompatibility.
This run provides no evidence against QMoE packing, MLA Attention, or router
casting because failure occurs at layer 0 `q_proj` before those paths execute.
