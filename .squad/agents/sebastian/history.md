# Sebastian — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Phases 1-4 done; tool use/grammar/chat-template; Qwen2.5-0.5B runs; Hermes agent E2E; long-context O(1)/token via static-cache in-place KV. Working on DESIGN §26 batched serving + reviews.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-12

## 2026-07-12T13:14:00-07:00 — Performance review merged
Sebastian's perf review is now in decisions. §26 should prioritize active-row compaction, ORT KV as hot source of truth, fewer per-step allocations, direct/borrowed logits access, and explicit snapshot/import/export for paged KV.

## 2026-07-12T13:52:00-07:00 — §26 Stage A/B complete
- Sebastian delivered `Engine::generate_batched_static` and `ContinuousBatchManager`; fixed batched static-cache generation matches individual runs and measured 6.2x throughput on the tiny fixture.
- Future scheduler/perf work should preserve the `submit`/`step`/`poll` contract and use Deckard's active-row compaction when rows finish or new requests are admitted.
