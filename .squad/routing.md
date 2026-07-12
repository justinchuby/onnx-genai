# Routing

Maps work domains to team members. The coordinator uses this to dispatch.

| Domain / Signal | Owner |
|-----------------|-------|
| Architecture, design decisions, metadata standard, scope, code review | Roy (Lead) |
| KV cache, paged memory, prefix trie, tiered storage, ORT sessions, backend, tokenizers, Rust internals | Deckard (Systems) |
| Generation engine, scheduler, continuous batching, speculative decoding, logit chain, sampling | Batty (Engine) |
| HTTP server, OpenAI-compatible API, request/response handling, streaming | Rachael (Server) |
| Tests, benchmarks, correctness, edge cases, fixtures | Pris (Tester) |
| Code review, quality, clarity, maintainability, extensibility verdicts | Gaff (Reviewer) |
| Performance, throughput, latency, batched serving, KV efficiency, benchmarks | Sebastian (Perf) |
| Security, FFI/unsafe audit, server hardening, supply-chain, untrusted-input safety | Holden (Security) |
| Memory, decisions, session logs | Scribe |
| Work queue, backlog, keep-alive | Ralph |
| RAI review, content safety | Rai |
| Claim verification, devil's advocate, pre-mortem | Fact Checker |

## Notes

- Cross-cutting changes (touching multiple crates) → Roy triages, then fans out.
- Metadata parsing/validation lives in `onnx-genai-metadata` → Deckard, with Roy on standard conformance.
- Performance-critical hot paths → the owning dev pairs with Pris for benchmarks.
