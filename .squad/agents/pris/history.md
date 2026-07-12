# pris — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12



## 2026-07-12T09:13:00-07:00 — Metadata tests and tiny LLM fixture delivered
- Delivered metadata parser tests for valid YAML/JSON, malformed/schema-invalid parse errors, and runtime capability validation.
- Added deterministic tiny GPT-2-style fixture at `tests/fixtures/tiny-llm/` for next-batch ORT/tokenizer/generation integration without external model downloads.

## 2026-07-12T09:20:00-07:00 — Tiny fixture enabled Phase 1 E2E
- The deterministic `tests/fixtures/tiny-llm/` model and tokenizer enabled the first end-to-end greedy generation smoke test through the facade CLI, engine, tokenizer, and ORT session.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Pris delivered Phase 2 coverage for interleaved persistent sessions, reset isolation, KV fork CoW independence, same-session prefix hit (`prefix_cache_hit_len > 0`, warm hit observed as 6), and cross-session prefix reuse with matching greedy output.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Delivered Phase 3 validation: real TinyStories coherent CLI/HTTP generation, 12-session KV pressure pass with no OOM, speculative correctness harness, and documented CPU/tiny-model speedup limitation.

## 2026-07-12T12:02:00-07:00 — Qwen, Hermes, VLM, and long-context validation delivered
Validated Qwen2.5-0.5B Mobius builds and coherent generation, HTTP tool use, Hermes/coding-agent tool-loop acceptance, tiny VLM fixture scaffolding, static-cache scatter models, and flat 25-27 ms/token long-context decode.

## 2026-07-12T13:14:00-07:00 — Harness hardening merged
Pris's coding-agent harness sandbox is now in decisions: workspace path confinement, no shell execution, argv allow-list, guarded Python scripts, symlink/traversal rejection, and passing self-test.
