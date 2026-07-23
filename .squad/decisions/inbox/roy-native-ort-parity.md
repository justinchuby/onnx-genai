### 2026-07-23: Real-weight native CUDA decode is faithful to the trusted reference
**By:** Roy
**What:** Ran 128-token greedy decode through the same `profile_native` engine loop for native CUDA and ORT CUDA on Phi-4-mini and Qwen2.5 0.5B/1.5B/7B real weights. The prompt was:

> Explain why deterministic greedy decoding is useful when validating a new transformer inference backend. Include concrete numerical failure modes and how you would distinguish harmless floating-point near-ties from implementation bugs.

GPU 0 was idle (0 MiB, 0% utilization) before the runs. GPU 1 was occupied by VLLM (129,465 MiB); it was avoided. No contention was observed on GPU 0. Qwen runs used `ONNX_GENAI_CUDA_GRAPH=0` for both backends because ORT CUDA graph capture failed on the 0.5B package with `ort_value must contain a constructed tensor`; this only disables capture and does not change the model math. The 0.5B native sequence was also identical with capture enabled.

| Model | Native/ORT common prefix | Same IDs at aligned positions | First divergence (0-based) | First IDs (native / ORT CUDA) | Classification | Evidence |
|---|---:|---:|---:|---|---|---|
| Phi-4-mini | 128/128 | 128/128 | none | — | ✅ exact parity | All 128 generated IDs identical. |
| Qwen2.5-0.5B | 128/128 | 128/128 | none | — | ✅ exact parity | All 128 generated IDs identical. |
| Qwen2.5-1.5B | 22/128 | 23/128 | 22 (token 23) | 81917 / 22406 | ✅ native is reference-faithful; CUDA ORT near-tie | At the split, native top-2 logprobs were `81917=-2.18533`, `22406=-2.20095` (gap 0.015625). ORT CUDA rounded them to an exact tie at `-2.19943` and selected 22406. ORT CPU selected 81917 and reproduced the same 0.015625 margin (`-2.18856` vs `-2.20418`). The remaining mismatch is autoregressive cascade, not a native logits bug. |
| Qwen2.5-7B | 19/128 | 20/128 | 19 (token 20) | 2797 / 12966 | ✅ native is reference-faithful; ORT CUDA differs | Native CUDA top-2 were `2797=-1.15276`, `12966=-1.32464`. ORT CPU selected the same native token with `-1.15226` / `-1.32413`; the entire shown top distribution differs from native only by a constant ~0.000507 log-softmax offset, preserving ranks and the 0.171875 margin. ORT CUDA instead selected 12966 (`-1.15503` vs 2797 `-1.23316`). Thus the early CUDA-ORT split is on the ORT CUDA side; subsequent mismatches are cascade. |

**Why:** Native CUDA is token-exact with ORT CUDA on Phi-4-mini and Qwen2.5-0.5B. For both early Qwen divergences, the independent ORT CPU reference selects native CUDA's token. Qwen1.5B is a true near-tie rounded differently by ORT CUDA; Qwen7B is stronger evidence that native is more accurate than ORT CUDA because native's top distribution matches ORT CPU essentially exactly. No native kernel, dtype, RoPE, attention, or normalization correctness bug was found, so no kernel-owner dispatch is needed.

**Validation:** Built with `cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native`. Native trace through the Qwen7B split exercised fp16/int4 decode (`gemv_f16_general`, `attention_gqa_decode_fp16_splitk`, `skip_rmsnorm_f16_warp_half4`, `gate_up_swiglu_fused`, `gemv_f16_down_projection`); the ORT CPU reference proves their composed result is correct at the divergent step.
