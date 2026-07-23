# Batty: expert-major int4 QMoE emitter

**Status:** Implemented on Mobius PR [#404](https://github.com/onnxruntime/mobius/pull/404), commit `751645b`.

## Design and ABI packing

- `FusedQuantizedMoE` emits one `com.microsoft::QMoE` per routed MoE layer.
- The packer accepts fused 3-D GPTQ/AWQ checkpoint tensors and already-repacked per-expert GGUF/MatMulNBits tensors.
- FC1 is expert-major `[E, 2I, packed_H]`; chunked checkpoint gate/up rows are converted to interleaved `[g0,u0,g1,u1,...]` for `swiglu_fusion=1`.
- FC2 is expert-major `[E, H, packed_I]`.
- Scales are `[E, out, blocks]` and cast to float32 at the QMoE boundary.
- Asymmetric quantization is supported: packed FC1/FC2 zero-points are declared and wired through QMoE inputs 11/12. Symmetric quantization uses the implicit midpoint.
- The explicit router and input-14 aggregation weights are preserved. Shared experts remain separate dense quantized gated MLPs.

## GGUF correction

The per-expert direct-repack path now verifies that the repacked tensor's zero-point presence matches the target symmetric/asymmetric graph. A mismatch forces dequantization and requantization, preventing missing or semantically incompatible zero-point initializers.

## Validation

- DeepSeek-V2-Lite weightless int4 graph: 26 `QMoE`, 27 standard `Attention`, 0 `GroupQueryAttention`, 27 plain `MatMul`, and 189 `MatMulNBits`.
- Synthetic 64-expert/top-6 differential matches the static per-expert reference at `atol=rtol=1e-5`.
- Fused GPTQ layout, asymmetric zero-point wiring, shared-expert retention, and GGUF zero-point mismatch tests pass.
- Targeted Mobius suite: **46 passed**; Ruff and `git diff --check` clean.

## Remaining follow-up

Run a full weighted DeepSeek-V2-Lite int4 export and native-CUDA generation smoke.
