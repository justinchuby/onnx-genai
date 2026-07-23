# Now

**Updated:** 2026-07-23T04:09Z

Focus: CUDA performance + GLM/DeepSeek native-runtime support.

Just landed to main (13f641f): device-resident CUDA **IndexShare v1** (GLM/DeepSeek selected-token attention, GPU-native, 7 parity tests, capture=false pending device index validation) and **f16 CUDA standard Attention** (was f32-only; fp32 accum retained; bf16 deferred). Both reviewed (Chew/Gaff 🟡→addressed) and re-verified green on merged main.

Next CUDA/GLM-DeepSeek: IndexShare CUDA-graph capture (device-side index validation), f16/bf16 IndexShare storage, bf16 standard Attention, Mobius fused QMoE/BlockQuantizedMoE emitter, MTP state threading, CSA Phase B perf tuning.

Op-note: spawned "Scribe"-flavored agents keep hitting the squad.agent.md canary-refusal loop (even on gpt-5.6-terra); coordinator does the decisions/log merge inline under the local state backend.
