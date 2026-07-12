# rachael — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12

## 2026-07-12T09:20:00-07:00 — Generate CLI delivered
- Added `onnx-genai generate` with model path, generation option flags, stop sequences, streaming, and prompt support.
- CLI maps args to `GenerateOptions`/`GenerateRequest` and calls Batty's engine API; tiny-fixture greedy generation now runs end-to-end.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Rachael delivered the OpenAI-compatible HTTP server with `/health`, `/v1/models`, `/v1/chat/completions`, SSE streaming, `X-Session-Id` persistent session addressing, `POST /v1/sessions`, and `DELETE /v1/sessions/{id}` lifecycle support.
