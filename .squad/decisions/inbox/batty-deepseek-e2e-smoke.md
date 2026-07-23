### 2026-07-23: DeepSeek-V2 real-shape QMoE native CUDA smoke
**By:** Batty

**What:** A reduced two-layer, random-weight DeepSeek-V2-Lite artifact now exercises the complete Mobius expert-packing to native-CUDA execution path. The artifact is at `/home/justinchu/ds-e2e-artifacts/deepseek-v2-realshape-qmoe-int4`.

**Artifact contract**

- Mobius PR #404 head: `6f2a52a`; native runtime: onnx-genai `1073404`.
- Hidden size 2048, 16 attention heads, two layers, MLA K head size 192 and V head size 128.
- One MoE layer with 64 routed experts, top-6 routing, two always-on shared experts, and routed intermediate size 1408.
- Random asymmetric GPTQ int4 tensors, block size 128, were supplied in raw fused checkpoint form and passed through `DeepSeekForCausalLM.preprocess_weights`; this tests the real emitter/packer rather than directly manufacturing final QMoE initializers.
- Graph counts: one QMoE, two standard `ai.onnx::Attention`, zero GroupQueryAttention, two MatMul, and 16 MatMulNBits. There is no per-expert MatMul expansion.
- QMoE tensors match the native expert-major ABI: FC1 weights `[64,2816,1024]`, scales `[64,2816,16]`, zero points `[64,2816,8]`; FC2 weights `[64,2048,704]`, scales `[64,2048,11]`, zero points `[64,2048,6]`. QMoE inputs 11/12 carry asymmetric zero points and input 14 carries aggregation weights.

**Native CUDA result**

Built `profile_native` with `--features bench-native,cuda` and ran on idle H200 GPU 4:

```
CUDA_VISIBLE_DEVICES=4 \
ONNX_GENAI_REQUIRE_CUDA=1 \
ONNX_GENAI_PROFILE_OPS=1 \
profile_native --model <artifact> --ep cuda --steady --tokens 32 \
  --warmups 1 --runs 1 --prompt "1 2 3 4"
```

The strict run completed, so every graph node was claimed by the CUDA EP; the log contains zero fallback or CPU-placement warnings. It generated 32 finite token IDs:

```
[169,169,111,15,73,206,82,149,149,70,70,55,55,68,68,253,
 253,244,244,177,177,233,165,175,206,96,74,1,142,127,107,6]
```

The random-weight text is consequently not semantically meaningful, but generation is stable and finite. Token-zero top-40 log probabilities were all finite (selected token 169 at `-4.1758413315`; observed range approximately `[-5.2189,-4.1758]`).

Steady performance was 11.615 ms prefill and 3.061 ms/token (326.68 tokens/s) over the 24 measured decode tokens. Per-token profiling shows one QMoE call at approximately 0.338-0.340 ms, two Attention calls at approximately 0.13-0.15 ms total, and 16 MatMulNBits calls. The first QMoE call took approximately 675 ms from cold startup/compilation.

**ORT GenAI comparison**

`onnxruntime-genai==0.14.1` does not support this model family. After generating the standard Mobius `genai_config.json`, `onnxruntime_genai.Model(directory)` fails before session creation with:

```
RuntimeError: Unsupported model_type in config.json: deepseek_v2
```

Therefore no same-artifact ORT decode or logits comparison is available; this is a native-only smoke.

**Important dtype finding and remaining work**

The first f16-activation export was correctly rejected by strict CUDA placement because the graph supplied a float32 attention mask to f16 Q/K/V and float32 RotaryEmbedding cos/sin to f16 X. Re-exporting activations as f32 retained the int4 QMoE/MatMulNBits weights and allowed the structural smoke above to pass.

The critical emitter-to-kernel wiring is proven. The remaining production milestone is a full-weight DeepSeek-V2-Lite export and smoke. Before that, Mobius should make generated Attention masks and RotaryEmbedding cos/sin match the activation dtype so the intended f16 artifact also passes strict CUDA placement.

**Evidence:** `model.onnx`, `native_cuda.log`, `native_logprobs.json`, and `ort_genai_load.log` in the artifact directory; exporter script at `/home/justinchu/mobius-batty-404/.scratch/batty_export_deepseek_qmoe.py`.
