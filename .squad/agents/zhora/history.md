# Zhora — History

## 2026-07-12: Joined
Hired as Server Dev to add capacity alongside Rachael on the OpenAI-compatible HTTP surface. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: server is modularized (routes/driver/sse/types/state/session/metrics/image_input/audio_input); chat/completions/vision/audio/streaming/sessions/observability shipped; open API work includes `/v1/embeddings` (#7) and logprobs server formatting (#8). Handlers stay thin over the batched engine driver.
