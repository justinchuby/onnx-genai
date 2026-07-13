# Leon — History

## 2026-07-12: Joined
Hired as Engine Dev (KV & runtime buffers) to add capacity alongside Batty as the runtime grew (9 crates, concurrent engine/KV workstreams). Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Key context: runtime owns the KV cache; use our own InferenceMetadata (`inference_metadata.yaml`) not ORT-GenAI `genai_config.json`; static-cache/GQA use device-resident buffers with present→past IoBinding aliasing; WebGPU decode needs GQA op + quantized (Q4 MatMulNBits) weights. Real-model exact-equality tests use `intra_op_threads=1`.
