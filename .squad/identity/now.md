# Team Focus — now

**Goal:** (A) run GLM/DeepSeek/Qwen/Gemma4/Phi smoothly on the native stack; (B) native CUDA EP decisively faster than onnxruntime-genai; (C) continuously benchmark foundry-local cuda-gpu models on 8×H200.

**main HEAD:** `0672400` — after CUDA perf + smoothness wave 1.

**Standing vs ORT (onnxruntime-genai-cuda 0.14.1, @128 greedy):** Qwen0.5B +12% (ahead), 1.5B parity, 7B ~−9% (after epilogue fusion, from −18%), Phi-4-mini −58% (top target — ORT graph-captures, native cannot due to control flow).

**Wave 2 in flight (5 worktree agents):** Marsten (post-fusion re-bench + smoothness), Deckard (0.5B fusion size-floor + next epilogue fusion), Batty (Phi-4-mini CUDA-graph capture), Keaton (f16/bf16 IndexShare storage), Irmgard (native_engine MoE test-fixture fix).

**Next levers:** Phi control-flow capture, per-layer megakernel fusion (~227 kernels/token), int8 SDOT (DP4A), GLM/DeepSeek real-weight E2E (blocked on Mobius fused QMoE emitter), Gemma4 VLM native pipeline.
