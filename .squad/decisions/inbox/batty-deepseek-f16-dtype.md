### 2026-07-23: DeepSeek f16 QMoE strict-CUDA dtype fix
**By:** Batty

**What:** Mobius PR #404 commit `bd88fa8` makes DeepSeek router MatMul operands consistently float32 and adds low-precision graph contracts for MLA Attention masks and RotaryEmbedding inputs.

**Root cause:** The original smoke diagnosis was incomplete. Mobius already follows the shared low-precision convention:

- `create_attention_bias(..., dtype=config.dtype)` emits an f16/bf16 Attention mask.
- `BaseRope` stores precise f32 caches but casts gathered cos/sin values to `config.dtype`.

The first scratch “f16” artifact mixed f16 weights with a config still declaring f32 activations, producing the reported mask/RoPE mismatches. A correctly configured f16 export proved those auxiliaries already match Q/K/V and RoPE X, but then exposed the real production bug: `DeepSeekMoEGate` cast hidden states to f32 while leaving the gate weight f16, so native CUDA rejected the router MatMul’s mixed operands.

**Fix:** Both `DeepSeekMoEGate.forward()` and `route_for_qmoe()` now share `_router_logits()`, which casts hidden states and the transposed gate weight to f32 before MatMul. Expert selection remains numerically stable in f32 while accepting f16/bf16 checkpoints.

**Regression coverage:**

- f16 and bf16 DeepSeek MLA exports assert both RotaryEmbedding nodes have X/cos/sin in the activation dtype.
- f16 and bf16 exports assert the standard Attention mask matches the query activation dtype.
- Both static and fused-QMoE DeepSeek routing paths assert the router MatMul receives two f32 inputs.
- Relevant suite: 107 passed.
- Ruff 0.15.14: clean.

**Strict native CUDA result:** Re-exported the two-layer, 64-expert/top-6, asymmetric-int4 artifact with f16 activations at `/home/justinchu/ds-e2e-artifacts/deepseek-v2-realshape-qmoe-int4-f16`. On idle H200 GPU 5, `ONNX_GENAI_REQUIRE_CUDA=1` completed a 32-token decode with zero fallback/error warnings:

```
[250,69,69,33,90,161,201,141,176,172,250,155,107,203,72,30,
 141,231,138,150,1,50,172,97,160,208,208,81,81,68,235,44]
```

Token-zero top-40 log probabilities plus the selected token probability were all finite (41 values, range approximately `[-5.14484,-4.22784]`). Steady decode measured 2.534 ms/token (394.64 tokens/s). Profiling confirmed one QMoE and two Attention executions per token.
