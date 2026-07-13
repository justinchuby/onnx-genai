# Zhora — History

## 2026-07-12: Joined
Hired as Server Dev to add capacity alongside Rachael on the OpenAI-compatible HTTP surface. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: server is modularized (routes/driver/sse/types/state/session/metrics/image_input/audio_input); chat/completions/vision/audio/streaming/sessions/observability shipped; open API work includes `/v1/embeddings` (#7) and logprobs server formatting (#8). Handlers stay thin over the batched engine driver.

## 2026-07-13: Landed debug endpoints and queue-depth cap
Added `/v1/debug/config`, `/v1/debug/sessions`, `/v1/debug/kv`, and `/v1/debug/trace`; renamed the server admission boundary to configurable `max_queue_depth` (`--max-queue-depth` / `ONNX_GENAI_MAX_QUEUE_DEPTH`). Landed as commit `afcf094`.

## 2026-07-13T20:55:00Z — Model lifecycle M1 + /v1/embeddings wiring
- Implemented issue #9 model lifecycle Milestone 1: extracted ModelHandle + ModelRegistry from AppState (pure refactor). ModelHandle bundles all per-model fields; ModelRegistry wraps HashMap<String, Arc<ModelHandle>> with resolve/insert/ids/default_id. Zero behavior change — single-model fallback preserved. 52 tests green. Commit: 9ab4fa9.
- Wired POST /v1/embeddings through DriverCommand::Embed (oneshot-reply) to engine embed_with_options. Mean pooling via EmbeddingOptions::default(). Pipeline models return clear error. Double tokenization intentional. dimensions truncation deferred.
- M2 (real routing), M3 (load/unload), M4 (status field), LRU eviction deferred.
