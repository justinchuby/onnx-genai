# Session log — CLI sampling/render follow-ups

**Timestamp:** 2026-07-30T12:45:00-07:00
**Branch:** `qwen3-perf-followups`
**Agents:** Batty, Sebastian, Scribe

Merged the Batty/Sebastian decision drops plus all other pending decision inbox drops into the Scribe ledger/archive and deleted the consumed drop files, leaving `decisions/inbox/README.md` in place.

## Key facts recorded
- Default CLI builds lack `native-backend`; `run`/`generate --backend auto` resolve to ORT and load `onnxruntime.dll`. Native requires `--features native-backend`.
- ORT sampling decode for Qwen3-0.6B HF is now 93.97 tok/s versus greedy 98.74 tok/s for 150 tokens with global `--profile`; sampling is within about 5% of greedy and no longer the CLI bottleneck.
- Qwen3's live current turn is not opened by the template's `<think>` branch; `opened_by_template` must be false for the live turn.
- REPL drawing should coalesce frames while preserving all token buffers.
