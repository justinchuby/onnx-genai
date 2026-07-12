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

## 2026-07-12T09:20:00-07:00 — Phase 1 engine wiring completed
- Wired generation to real ORT session execution and HF tokenizer loading.
- Added graph input discovery for `input_ids`, `attention_mask`, `position_ids`, and past/present KV names; threads model-owned KV tensors when present and falls back to full-sequence reruns otherwise.
- Tiny-fixture CLI greedy generation now runs end-to-end; 13 engine tests pass.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Batty delivered persistent engine sessions, stateless `generate` compatibility, minimal FCFS scheduler admission, paged-KV mirroring, same/cross-session prefix reuse, and `GenerateResult::prefix_cache_hit_len` for cache observability.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Delivered Phase 3 engine work: greedy speculative decoding, priority scheduling with swap preemption, context-window guard, and real-draft KV rewind fix; real differing model speculation is target-greedy token-identical.

## 2026-07-12T12:02:00-07:00 — Phase 4 engine and decode migration delivered
Delivered constrained decoding (JSON FSM + llguidance JSON Schema/Regex/Lark), pipeline executor APIs, and engine migration to DecodeSession/StaticCacheDecodeSession for O(1)/token KV movement.
