# scribe — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12


- 2026-07-18: Merged 17 decision notes; wrote orchestration/session logs and progress update. No history exceeded summarization threshold.

## 2026-07-18T01:20:34Z — PR #25 and CUDA SparseKvGather archive
- Merged seven decision notes, wrote orchestration/session logs, and updated progress. Ash and Hudson remain in flight.

## 2026-07-26T19:45:52Z — Scribe update

- Merged 14 decision inbox notes, wrote Deckard/Leon orchestration logs, updated session focus, and checked archive/history gates; no history file exceeded 15 KB.
## 2026-07-28T11:35:49Z — Decision-ledger compaction rebase lesson
- Size-compaction of shared append-only files is not rebase-safe: concurrent appends can silently reinflate `.squad/decisions.md` while preserving a compacted header. Re-run compaction against tip immediately before merging a compaction PR.
