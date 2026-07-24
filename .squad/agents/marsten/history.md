# Marsten — History

- 2026-07-23: Re-benchmarked post-fusion CUDA performance, established the Qwen regression was real code rather than CPU noise, and confirmed `12efc92` restored Qwen 7B to 286.73 tok/s.
- 2026-07-23T15:45:00Z: Nsight/profile analysis found Phi's primary remaining gap is eager Greater and host If capture seams: 2.258 ms/token (43.8% of wall time) is outside 2.899 ms/token GPU work. Fixed-decode control-flow specialization is the priority over further GEMV tuning.
- 2026-07-23T18:30:00Z: Independently verified the landed on-device LongRoPE select (`97c1a56`) at 321.98 tok/s on idle GPU 3, +40.22% versus canonical ORT. Native CUDA now leads ORT on every available real-weight Qwen and Phi benchmark.
- 2026-07-23T20:30:00Z: Real DeepSeek-V2-Lite semantic execution is blocked before token 1: all dense MatMulNBits nodes are block-128, while native fp16 CUDA currently supports block-32. Strict CUDA placement had zero fallbacks; QMoE/MLA were not reached, so no semantic conclusion is warranted.
## 2026-07-23T22-29-16Z — GLM-4-9b native support smoke
- Confirmed real GLM-4-9b int4 native CUDA decode is coherent on both baseline `569507c` and DeepSeek dtod fix `1fe314f`, byte-identical with cuda-graph active (`captures=1`, `replays=61`, `fallbacks=0`). ORT-genai cannot load this export without `genai_config.json`, so evidence is native-only.
- 2026-07-24: Validated real-model coverage after DeepSeek fixes: DeepSeek-V2-Lite MoE coherent, GLM-4-9b dense native coherent, and R1-Distill-Qwen-1.5B native exactly matches ORT-genai for 32 tokens.
