# Zhora — History

## 2026-07-12: Joined
Hired as Server Dev to add capacity alongside Rachael on the OpenAI-compatible HTTP surface. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: server is modularized (routes/driver/sse/types/state/session/metrics/image_input/audio_input); chat/completions/vision/audio/streaming/sessions/observability shipped; open API work includes `/v1/embeddings` (#7) and logprobs server formatting (#8). Handlers stay thin over the batched engine driver.

## 2026-07-13: Landed debug endpoints and queue-depth cap
Added `/v1/debug/config`, `/v1/debug/sessions`, `/v1/debug/kv`, and `/v1/debug/trace`; renamed the server admission boundary to configurable `max_queue_depth` (`--max-queue-depth` / `ONNX_GENAI_MAX_QUEUE_DEPTH`). Landed as commit `afcf094`.
