# Sapper — History

## 2026-07-12: Joined
Hired as Systems Dev to add capacity alongside Deckard on model building and preprocessing. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: `onnx-genai-preprocess` is its own crate (image + audio); Mobius (`../mobius`) builds models — `build-gguf` (Q4 MatMulNBits), `--ep webgpu` (GQA), `--static-cache`; we emit our own `InferenceMetadata` (`inference_metadata.yaml`) not ORT-GenAI genai_config. Python builders use onnxscript/onnx-ir. Mobius PRs must pass `lintrunner` (RUFF + RUFF-FORMAT).
