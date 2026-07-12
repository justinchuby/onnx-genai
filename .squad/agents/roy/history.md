# roy — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12



## 2026-07-12T09:13:00-07:00 — Phase 1 foundation plan delivered
- Assessed Phase 1 status and identified real ORT CPU execution, model/tokenizer discovery, and minimal greedy generation as the critical path.
- Shared context for next batch: Deckard supplied ORT/tokenizer contracts, Batty supplied the generation API, and Pris supplied deterministic metadata/fixture coverage.
