# deckard — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12

- 2026-07-12T08:56:27-07:00 — Updated `.gitignore` with Rust and Python generated-artifact coverage; decision merged by Scribe.


## 2026-07-12T09:13:00-07:00 — ORT session, model-directory, and tokenizer contracts delivered
- Delivered real CPU `Environment`/`Session` load-run APIs, tensor `Value` helpers, graph metadata accessors, optional IoBinding, `ModelDirectory::load`, and `Tokenizer` encode/decode helpers.
- Key contract for next-batch wiring: `Session::run` accepts named `Value` inputs and returns outputs ordered by `output_names()` / `outputs()`; tokenizer decode skips special tokens and exposes optional EOS id.
