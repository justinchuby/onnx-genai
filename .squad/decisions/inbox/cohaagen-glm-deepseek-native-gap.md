# Native GLM-4-9B and DeepSeek-V2-Lite load gaps

Date: 2026-08-11
Owner: Cohaagen

## Findings

### GLM-4-9B

Model: `/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda`

Initial native load failure was reported through the generic unsupported-native-operator wrapper, but the pinned inner error was KV admission, not an unsupported op:

```text
cannot reserve 5368709120 bytes of KV for one sequence at 131072 tokens of context on Device ...
131072 is the model's declared max_sequence_length
```

The model has 40 `com.microsoft::GroupQueryAttention` nodes with partial rotary attributes:

```text
num_heads=32, kv_num_heads=2, scale=0.0883883461,
do_rotary=1, rotary_interleaved=1, rotary_embedding_dim=64
```

This path is tractable and already handled by the native CUDA GQA execution path: the kernel derives the rotary width from the cos/sin cache, so the non-rotary tail passes through. No new GLM-specific kernel gate is needed.

Fix: native KV reservation now charges the actual runtime CUDA KV capacity (`cuda_kv_debug_stats().hard_max_len`, e.g. `ONNX_GENAI_CUDA_KV_MAX_LEN=4096`) before falling back to metadata `max_sequence_length`. This lets long-context exports load when the runtime deliberately caps KV for decode/profiling.

Validation on GPU 2:

- Native CUDA short greedy run loads and emits token ids `[11, 358, 1079, 264, 220, 18, 6498, 1042]` (`", I am a 3rd year"`), 93.64 tok/s in a one-run smoke.
- Existing ignored lock `glm4_9b_native_cuda_matches_golden_greedy_sequence` passes against the golden greedy stream.
- CPU-vs-CUDA first-token top-logprob probe: selected token 11 in both; top-40 token ids identical; max common top-40 logprob delta ≈ 0.00794; top-1 margin 0.64844 in both.

### DeepSeek-V2-Lite QMoE

Model: `/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4`

Initial native load failure:

```text
QMoE input 3 is not a graph initializer
```

Pinned offending node shape: 26 `com.microsoft::QMoE` nodes. The packed expert weights and zero-points are direct initializers, but scale inputs 3 and 6 are `Cast(to=Float32)` nodes whose sources are fp16 initializers:

```text
input2 fc1_experts_weights: initializer
input3 Cast(fc1_scales fp16 initializer -> fp32)
input5 fc2_experts_weights: initializer
input6 Cast(fc2_scales fp16 initializer -> fp32)
input11/input12 zero_points: initializer
```

This is tractable infrastructure, not an MLA kernel gap: static placement only needs the backing initializer dimensions/regions, while runtime QMoE still receives the fp32 Cast value it expects.

Fix: static QMoE placement accepts a one-hop default-domain `Cast(initializer)` as initializer-backed for required/optional QMoE weight-region classification.

Validation on GPU 2:

- Native CUDA short greedy run loads and emits token ids `[11, 304, 608, 245]` (`", I am a"`), 52.48 tok/s in a one-run smoke.
- Existing ignored lock `deepseek_v2_lite_native_cuda_matches_golden_greedy_sequence` passes against the golden greedy stream.
- CPU-vs-CUDA first-token top-logprob probe: selected token 11 in both; top-40 token set identical with only low-ranked order swaps after rank 30; max common top-40 logprob delta ≈ 0.02848; top-1 margins 1.78125 CPU and 1.77344 CUDA.

## Tractability verdict

- GLM-4-9B: quick win. Not missing partial-RoPE support; the blocker was KV reservation using metadata max context instead of the effective runtime CUDA KV cap.
- DeepSeek-V2-Lite QMoE: quick win. Not a new attention/MLA kernel requirement for load/decode; the blocker was QMoE placement rejecting Cast-backed scale initializers.
- No model-name gates were added. Both fixes are DRY runtime/placement behavior.
