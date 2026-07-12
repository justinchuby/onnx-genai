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


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Roy's Phase 2 plan was executed successfully: paged KV tensor storage, prefix cache lifecycle/CoW, persistent multi-session engine APIs, HTTP/SSE session surface, and Pris's exit tests are now in place. Shared contracts include `prefix_cache_hit_len`, `X-Session-Id`, and standalone ORT runtime packaging.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Phase 3 plan completed and executed. Team delivered speculative decoding, tiered/quantized KV, priority/preemption, streaming/accounting hardening, and validation; speedup limitation is environment-bound locally.

## 2026-07-12T12:02:00-07:00 — Phase 4 and long-context plans completed
Roy's Phase 4, tool-use/grammar, and long-context plans were executed: pipeline execution, constrained decoding, OpenAI tool use, Qwen/Hermes validation, and O(1)/token static-cache decode are now recorded. Next roadmap follows DESIGN §23-28 plus paged attention.
