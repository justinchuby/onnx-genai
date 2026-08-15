# Final CPU whole-model benchmark result

Matched-load, steady decode A/B on the same Xeon 8480C confirms native
onnx-genai CPU outperforms onnxruntime-genai 0.14.1 CPU on every tested model:

| Model | Native (tok/s) | ORT GenAI (tok/s) | Speedup |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B f16 | 154.9 | 86.5 | 1.79x |
| Qwen2.5-1.5B f16 | 74.0 | 40.6 | 1.82x |
| Qwen2.5-coder-7B int4 generic-cpu | 32.7 | 21.1 | 1.55x |

Five per-op wins landed: f32 SiLU, f16 SiLU, f16 Mul, SIMD
SkipLayerNorm, and QKV-bias fusion. See PR #105.
