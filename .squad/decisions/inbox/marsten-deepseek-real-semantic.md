### 2026-07-23: Block-32 real DeepSeek-V2-Lite remains semantically broken on native CUDA
**By:** Marsten
**What:** ❌ The block-32 real-weight artifact runs 64 tokens at 23.23 tok/s
with zero CPU EP placement fallbacks, but native output is multilingual garbage
with repetition collapse and diverges from HF at token 1. CUDA graph capture also
falls back at the first MLA `Attention` because its capacity-backed output fails
the kernel's contiguous/dtype/expected-size check.
**Why:** Block-32 removes the earlier `MatMulNBits` hard failure but does not
establish semantic correctness. ORT CUDA on the identical ONNX produces the
HF-matching first eight tokens, proving the artifact and prompt are viable.

## Prompt

Rendered DeepSeek chat template:

```text
<｜begin▁of▁sentence｜>User: Answer in one concise sentence: What is the capital of France and what landmark is it famous for?

Assistant:
```

Both native and HF use prompt IDs:

```text
[100000, 5726, 25, 35829, 279, 634, 46019, 4976, 25, 2461, 317,
 254, 6077, 280, 7239, 285, 856, 44872, 317, 359, 9679, 327, 30,
 185, 185, 77398, 25]
```

## Native CUDA result

- Artifact:
  `/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4-blk32/`
- GPU: physical GPU 0, idle NVIDIA H200
- Strict placement: `ONNX_GENAI_REQUIRE_CUDA=1`
- Generated: 64/64 valid vocabulary token IDs; no NaN/inf error surfaced
- Throughput: **23.23 tok/s**, 43.053 ms/token
- CPU EP placement fallbacks: **0**
- CUDA graph fallbacks: **2 cumulative, 1 measured**
- Semantic divergence: **token 1**

Actual native decoded text:

```text
 interns找工作 fibre间接 tanMNR Hiram contrast centred Hiram直观以外» centred直观 centred Hiram直观 hashtag hashtag hashtagonar hashtag tan hashtag hashtagonar hashtag hashtag hashtag hashtag hashtag Tweet hashtag hashtag Tweet Tweet TweetExpectedExpectedExpected tan tan tan tan tan MST hashtag MST MSTmosphere replic MSTITES MST MSTExpected博客博客 iPod Responsible souvenir sloganiero
```

This is not coherent language output and collapses into repeated `hashtag`,
`Tweet`, `Expected`, `tan`, and `MST` tokens.

## HF bf16 reference

Actual reference text:

```text
 The capital of France is Paris. The most famous landmark in Paris is the Eiffel Tower.
```

## Early token comparison

| # | Native ID / text | HF bf16 ID / text |
|---:|---|---|
| 1 | `60936` / `" interns"` | `429` / `" The"` |
| 2 | `96847` / `"找工作"` | `6077` / `" capital"` |
| 3 | `38130` / `" fibre"` | `280` / `" of"` |
| 4 | `60461` / `"间接"` | `7239` / `" France"` |
| 5 | `12749` / `" tan"` | `317` / `" is"` |
| 6 | `43781` / `"MNR"` | `8913` / `" Paris"` |
| 7 | `90390` / `" Hiram"` | `13` / `"."` |
| 8 | `8659` / `" contrast"` | `429` / `" The"` |
| 9 | `62083` / `" centred"` | `1094` / `" most"` |
| 10 | `90390` / `" Hiram"` | `9679` / `" famous"` |
| 11 | `80333` / `"直观"` | `44872` / `" landmark"` |
| 12 | `35448` / `"以外"` | `279` / `" in"` |
| 13 | `5608` / `"»"` | `8913` / `" Paris"` |
| 14 | `62083` / `" centred"` | `317` / `" is"` |
| 15 | `80333` / `"直观"` | `254` / `" the"` |

No early top-1 agreement exists.

## Root-cause evidence

1. **CUDA graph fallback is precisely MLA `Attention` node 38.**
   `model/layers.0/self_attn/Attention_node_40` has Q/K head width 192 and V
   width 128. Capture rejects its output with:

   ```text
   Attention: output must be contiguous and use the input dtype with the expected shape
   ```

   Graph-disabled native execution has zero graph fallbacks but remains broken,
   so this capture defect is real but is not the sole semantic cause.

2. **The ONNX artifact itself produces faithful quantized output under ORT.**
   ORT CUDA on the same model/prompt starts:

   ```text
   [429, 6077, 280, 7239, 317, 8913, 13, 429, 427, 96575, 25943, ...]
    The capital of France is Paris. The Eiffel Tower ...
   ```

   It matches HF exactly for tokens 1-8 before reasonable int4 divergence.

3. **QMoE export packing is not the demonstrated fault.** Manual dequantization
   of layer-1 expert 0 against the bf16 checkpoint gives cosine similarity
   0.99670 for interleaved gate/up FC1 and 0.99674 for FC2. Forcing native QMoE
   prefill away from grouped GEMM and onto GEMV still begins with the wrong
   newline token, so the failure is not limited to grouped-prefill packing.

The remaining semantic fault is therefore inside native CUDA execution, after
placement and before logits. Current evidence narrows dispatch to the native
fp16 MLA `Attention` path or the general QMoE execution path; it does not justify
claiming one without an intermediate-tensor oracle. The proven, separately
actionable defect is the MLA `Attention` capture output binding above.

## Verdict

❌ **Broken.** Zero CPU EP fallbacks and finite token IDs are insufficient:
native diverges at token 1 and emits systematic garbage/repetition, while ORT on
the same quantized graph is semantically faithful.
