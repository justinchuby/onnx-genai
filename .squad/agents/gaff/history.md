# Gaff — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Phases 1-4 done; tool use/grammar/chat-template; Qwen2.5-0.5B runs; Hermes agent E2E; long-context O(1)/token via static-cache in-place KV. Working on DESIGN §26 batched serving + reviews.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-12

## 2026-07-12T13:14:00-07:00 — Engine quality review merged
Gaff's review is now in decisions: engine.rs is a ~3,300-line god module. Refactor toward shared decode loop, module split, DecodeBackend, Sampler, proposer/verifier seams, and targeted tests.


## 2026-07-13T18:30:00Z — Review/fix batch
- Reviewed Zhora debug/queue-depth and Sapper token-expansion. Rejected Zhora's unauthenticated `/v1/debug/*` session-ID exposure and flagged Sapper thumbnail ordering; lockout fixes moved to Rachael and Deckard.
