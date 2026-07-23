### 2026-07-23: Retain standard Attention for unequal MLA K/V dimensions
**By:** Batty
**What:** Mobius Attention-to-GQA rewrites now decline when static current or cached K/V last dimensions differ. The DeepSeek-V2-Lite CUDA no-weight contract retains 27 standard Attention nodes with 192-dimensional K and 128-dimensional V, and emits zero GroupQueryAttention nodes.
**Why:** `com.microsoft::GroupQueryAttention` requires equal K/V head dimensions, while standard `ai.onnx::Attention` supports DeepSeek MLA's distinct Q/K and V dimensions. A structural guard prevents invalid fusion without model-name special cases or changing ordinary Qwen/Phi/GLM fusion.

## Implementation

Mobius commit: `00309be` on branch `feat/glm4-gptq-import`
PR (for Justin to review and merge): <https://github.com/onnxruntime/mobius/pull/424>

The shared `_has_unequal_kv_head_dimensions` helper checks both:

- current K/V tensors; and
- past-key/past-value cache tensors.

It declines only when both compared last dimensions are statically known and
unequal. Unknown dimensions preserve the existing rewrite behavior. Both
`RotaryAttentionToGQA` and the universal `AttentionToGQA` fallback use the same
helper.

## Contract and validation

Exact smoke command:

```bash
mobius build \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --no-weights --dtype f16 --ep cuda \
  <output-dir>
```

Before `00309be`:

```text
Attention=0
GroupQueryAttention=27
past key=[batch,16,past_sequence_len,192]
past value=[batch,16,past_sequence_len,128]
```

After `00309be`:

```text
Attention=27
GroupQueryAttention=0
past key=[batch,16,past_sequence_len,192]
past value=[batch,16,past_sequence_len,128]
```

The new `arch_validation` contract test asserts those four conditions for the
real DeepSeek-V2-Lite config. The complete Mobius GQA rewrite test file passes:

```text
23 passed
```

That suite includes non-regression coverage for:

- Qwen3 standard GQA fusion (`Attention` 28 → GQA 28);
- Qwen2 biased packed-QKV fusion, which mirrors Phi-3/4 QKV-bias structure;
- GLM4 interleaved-RoPE fusion and attribute propagation;
- fallback GQA fusion, packed/unpacked paths, softcap, and ORT execution.

Ruff check and format-check pass for both changed files.
