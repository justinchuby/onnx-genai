# Batty: expert-major int4 QMoE emitter

**Status:** Implemented on Mobius PR [#424](https://github.com/onnxruntime/mobius/pull/424), commit `7e82326`.

## Design and ABI packing

- Mobius now emits one `com.microsoft::QMoE` for GPTQ/AWQ int4 routed MoE layers whose gate exposes full routing scores (`mobius/components/_moe.py:244-359`).
- FC1 is the checkpoint's chunked `gate_up_proj`, packed expert-major as `[E, 2I, H/2]`; `swiglu_fusion=2` tells the native kernel that gate rows precede up rows.
- FC2 is packed expert-major as `[E, H, I/2]`.
- Scales are float32 `[E, out, in/block_size]`; asymmetric zero-points are uint8 `[E, out, ceil(blocks/2)]`. Mobius preserves scale dtype through model casting because native CUDA requires float32 scales and routing tensors (`qmoe.rs:836,959,1076-1077`).
- GPTQ/AWQ preprocessing now preserves the leading expert axis and flattens MatMulNBits `[blocks, blob]` into QMoE's packed input dimension (`mobius/_weight_utils.py:557-628`).
- The explicit router remains in ONNX. DeepSeek passes selection scores separately from aggregation scores through QMoE input 14; routed scaling is applied after QMoE (`mobius/models/deepseek.py:80-130`).
- DeepSeek's two shared experts remain outside QMoE as an always-on dense gated int4 MLP (`mobius/models/deepseek.py:271-312`).

## Structural result

DeepSeek-V2-Lite with an injected GPTQ int4/block-128 config:

- Before: 5,208 `MatMul` nodes in the weightless float graph, including 192 routed-expert MatMuls per MoE layer.
- After: 26 `QMoE`, 27 standard `Attention`, 0 `GroupQueryAttention`, 27 plain `MatMul` (routers/final projection), and 189 `MatMulNBits`.
- Every one of the 26 MoE layers has exactly one QMoE and three shared-expert `MatMulNBits`; no `.moe.experts.*` per-expert expansion remains (`_group_query_attention_test.py:240-295`).

## Differential and regression validation

- Synthetic 64-expert/top-6 GPTQ pack-and-compute differential matches the existing static loop-over-experts reference at `atol=rtol=1e-5` (`components/_moe_test.py:165-236`).
- Expert-major GPTQ preprocessing shape coverage: 64 experts, including packed zero-points (`_weight_utils_test.py:513-550`).
- Targeted Mobius suite: **150 passed**.
- Full GQA rewrite suite: **24 passed**, preserving Qwen/Phi/GLM behavior.
- Ruff and `git diff --check`: clean.

## Remaining follow-up

Run a full weighted DeepSeek-V2-Lite int4 export and native-CUDA generation smoke. This change validates the graph and packing contract structurally and differentially; it intentionally does not download or generate the production-size artifact.
