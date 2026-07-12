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

## 2026-07-12T09:20:00-07:00 — ORT/tokenizer APIs enabled Phase 1 E2E
- The CPU `Session`, graph I/O metadata, tensor `Value` helpers, `ModelDirectory`, and `Tokenizer` APIs enabled Batty and Rachael to complete end-to-end greedy generation via the CLI tiny fixture.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Deckard delivered paged KV tensor storage (`new_with_tensor_config`, `append_token_kv`, `write_token_kv`, `materialize_sequence`), prefix cache page ownership/refcount lifecycle, CoW-safe writes, and ORT `1.27.0` runtime packaging so the server boots standalone without `DYLD_LIBRARY_PATH`.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Delivered Phase 3 KV work: hot/cold LRU page tiering plus opt-in `KvDType::Int8` symmetric per-page quantized KV that materializes back to f32 through existing cache APIs.

## 2026-07-12T12:02:00-07:00 — ORT chat-template and decode substrate delivered
Delivered pipeline schema/loader support, MiniJinja chat templates, multi-EOS discovery, fp16 Value helpers, zero-copy IoBinding DecodeSession, and StaticCacheDecodeSession with runtime-owned static KV buffers.

## 2026-07-12T13:14:00-07:00 — ORT hardening and batching notes merged
Deckard's GPU EP, batched static-cache decode, and ORT checksum notes are now in decisions. WebGPU/CoreML are selectable but slower than CPU on small Qwen decode; batched static decode matches unbatched but needs compaction.

## 2026-07-12T13:52:00-07:00 — §26 active-row compaction complete
- Deckard's `BatchedStaticCacheDecodeSession` active-row API is now part of the serving contract: `set_active_rows`, `compact`, `admit_row`, `deactivate_row`, `step_active`, and slot diagnostics back Sebastian/Rachael continuous batching.
- Future paged-attention and ORT work should keep logical row ids stable while allowing packed physical execution.

## 2026-07-12T14:28:00-07:00 — ORT comparison suite deterministic
- Deckard made all five ORT real-model comparison tests use `intra_op_threads=1` and a shared test `Environment`.
- The rare ORT FP-tie active-compaction flake was eliminated: 20/20 `onnx-genai-ort` and 5/5 full-workspace runs stayed clean.
- Future exact ORT comparisons should prefer single-threaded intra-op execution unless the assertion is tolerant to reduction-order differences.
