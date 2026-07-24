# Marsten — History

- 2026-07-23: Re-benchmarked post-fusion CUDA performance, established the Qwen regression was real code rather than CPU noise, and confirmed `12efc92` restored Qwen 7B to 286.73 tok/s.
- 2026-07-23T15:45:00Z: Nsight/profile analysis found Phi's primary remaining gap is eager Greater and host If capture seams: 2.258 ms/token (43.8% of wall time) is outside 2.899 ms/token GPU work. Fixed-decode control-flow specialization is the priority over further GEMV tuning.
- 2026-07-23T18:30:00Z: Independently verified the landed on-device LongRoPE select (`97c1a56`) at 321.98 tok/s on idle GPU 3, +40.22% versus canonical ORT. Native CUDA now leads ORT on every available real-weight Qwen and Phi benchmark.
- 2026-07-23T20:30:00Z: Real DeepSeek-V2-Lite semantic execution is blocked before token 1: all dense MatMulNBits nodes are block-128, while native fp16 CUDA currently supports block-32. Strict CUDA placement had zero fallbacks; QMoE/MLA were not reached, so no semantic conclusion is warranted.
## 2026-07-23T22-29-16Z — GLM-4-9b native support smoke
- Confirmed real GLM-4-9b int4 native CUDA decode is coherent on both baseline `569507c` and DeepSeek dtod fix `1fe314f`, byte-identical with cuda-graph active (`captures=1`, `replays=61`, `fallbacks=0`). ORT-genai cannot load this export without `genai_config.json`, so evidence is native-only.
- 2026-07-24: Validated real-model coverage after DeepSeek fixes: DeepSeek-V2-Lite MoE coherent, GLM-4-9b dense native coherent, and R1-Distill-Qwen-1.5B native exactly matches ORT-genai for 32 tokens.

## 2026-07-24T05:48:20+0000 — Definitive foundry-local native/ORT A/B

- On GPU 1, same harness and exact token parity: native whole-step capture measured Qwen2.5-0.5B **902 vs 584 tok/s** for ORT (1.55×) and Phi-4-mini **322 vs 238 tok/s** (1.35×).
- DeepSeek-V2-Lite native capture versus eager native was **79.2 vs 44.5 tok/s** at block-32 (1.78×) and **84.6 vs 46.8 tok/s** at block-128 (1.81×). ORT capture could not be enabled for the two A/B models (`ort_value must contain a constructed tensor`), so the ORT side is explicitly eager.

## 2026-07-24T06:14:01+0000 — Down-GEMV performance confirmation pending

- Assigned independent Qwen-7B end-to-end measurement for the merged, bit-exact int4 down-GEMV shared-staging removal (`720fa032`).
- No performance result is yet available; retain the change's validation evidence but do not claim a throughput delta.

## 2026-07-24T07:25:03+0000 — Down-GEMV performance confirmed

- Independent runs close the pending validation: Qwen2.5-7B improved 296.21→302.34 tok/s (+2.07%) and ratio to ORT 1.08→1.10×. Qwen 1.5B and 0.5B improved +1.79% and +1.24%; all runs retained identical tokens and capture.
