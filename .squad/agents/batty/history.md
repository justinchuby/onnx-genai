# batty — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12



## 2026-07-12T09:13:00-07:00 — Generation API and engine loop shell delivered
- Delivered `GenerateRequest`, `GenerateOptions`, `GenerateResult`, `GenerateToken`, callback support, `FinishReason`, `StopSequence`, and `Engine::generate` / `generate_with_callback`.
- Key contract for next-batch wiring: processor order is repetition penalty, stop-sequence termination, temperature, top-k, top-p; remaining backend stubs are prompt tokenization, token detokenization, and next-token logits.
