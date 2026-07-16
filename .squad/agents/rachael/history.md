# rachael — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12

## 2026-07-12T09:20:00-07:00 — Generate CLI delivered
- Added `onnx-genai generate` with model path, generation option flags, stop sequences, streaming, and prompt support.
- CLI maps args to `GenerateOptions`/`GenerateRequest` and calls Batty's engine API; tiny-fixture greedy generation now runs end-to-end.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Rachael delivered the OpenAI-compatible HTTP server with `/health`, `/v1/models`, `/v1/chat/completions`, SSE streaming, `X-Session-Id` persistent session addressing, `POST /v1/sessions`, and `DELETE /v1/sessions/{id}` lifecycle support.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Delivered Phase 3 server hardening: prompt-token usage accounting, SSE stop-sequence buffering/suppression with terminal stop chunks, `[DONE]`, and client-disconnect cancellation at callback boundaries.

## 2026-07-12T12:02:00-07:00 — Tool-use server integration delivered
Delivered OpenAI response_format JSON constraints, tools/tool_choice/tool-role handling, <tool_call> parsing into tool_calls, and forced tool_choice with generated Lark %json grammar.

## 2026-07-12T13:14:00-07:00 — Server hardening merged
Rachael's server DoS/session hardening is now in decisions: max_output_tokens=4096, max_sessions=256 LRU, 128-bit CSPRNG session ids, context-token caps, and loopback/no-auth deployment notes.

## 2026-07-12T13:52:00-07:00 — §26 Stage C server complete
- Rachael replaced the global generation mutex with a single engine driver thread and channels; concurrent STATIC-CACHE HTTP requests now share `ContinuousBatchManager` batched forward passes.
- Server behavior preserves streaming, tool turns, caps, CSPRNG session ids, and past/present fallback; future tool-pause/resume work should extend the driver protocol rather than reintroduce shared Engine locking.


### 2026-07-12T14:50:00-07:00
Batched-driver DoS hardening is canonical: admission is bounded by `max_pending` with HTTP 429, and driver output delivery is non-blocking so slow or closed clients are dropped without stalling other rows.

## 2026-07-12T16:14:00-07:00 — Issues #2/#4 and OpenAI surface logged
- Server split (#2) and legacy completions/FIM endpoint (#4) are canonical decisions.
- OpenAI surface now includes chat, tools, FIM via `/v1/completions`, image input parts for `/v1/chat/completions`, and streaming.
- Vision routing accepts data URI / bounded HTTP(S) images and routes VLM pipeline requests; real quality depends on Pris delivering a mobius CLIP+decoder VLM package.

## 2026-07-12T17:30:00-07:00 — Vision fidelity and audio endpoints logged
- Metadata-driven vision preprocessing and OpenAI-compatible `input_audio` plus `/v1/audio/transcriptions` routing are now canonical.
- Real audio/vision quality remains gated on production Mobius model packages and complete processor metadata.


## 2026-07-13T18:30:00Z — Review/fix batch
- Owned Zhora's reviewer-lockout security follow-up and landed `2e67806`, gating `/v1/debug/*` default-off and redacting session identifiers.


## 2026-07-20T00:00:00Z — Embeddings empty-model default fix (M2 follow-up)

- Commit: 561ee1a | Issue: #9 follow-up
- Trigger: Chew's 🟡 M2 review (Zhora locked out per reviewer protocol).
- Removed unconditional `if request.model.trim().is_empty() { return Err(...) }` guard from `validate_embedding_request` in `crates/onnx-genai-server/src/routes.rs`.
- Added two tests: `empty_model_field_falls_back_to_default_on_embeddings` and `unknown_model_returns_404_on_embeddings_endpoint`.
- Routing parity now holds across all four inference endpoints.
- §37 / Issue #9 epic fully complete.
- Next: §34 router epic (R1/R2/R3) has kicked off.

- 2026-07-14T19:05:00Z — Revised unsupported-op diagnostics to explicit `OpsetVersion::{Known, Undeclared}` and graceful unnamed nodes. Final loader fail-fast validation makes undeclared opsets unreachable on normal paths.

## 2026-07-15T01:52:00Z — Session update

- Delivered zero-copy mmap initializer borrowing (`3df84d0`). The later producer-aliasing soundness restriction was landed by Zhora.

## 2026-07-15T00:00:00Z — Cross-agent session update

- Made incomplete graph construction crate-private so full validation follows initializer attachment; merged in loader validation.

- 2026-07-16T00:00:01Z — Confirmed all 24 Qwen2.5-0.5B Sigmoids are `Mul(x, Sigmoid(x))` and landed guarded executor lowering plus allocation-free CPU SiLU (`682c93d`). Sigmoid share fell 6.55%→0%; decode improved 44.5→45.7 tok/s. Sebastian 🟢 approved; `d116a96` adds the multi-consumer negative test.

- 2026-07-16T00:00:00Z — Delivered default-on `nxrt.eager` and `nxrt.genai` Python submodules plus `docs/PYTHON.md` (`56a4a90`). The initial genai Engine wrapper was later revised by Sebastian after Holden identified a cross-thread PyO3 panic; the merged, cleared result is `41d8c31`.

## 2026-07-16T14:20:00Z — onnx-rs full-spec serde review
- 🔴 Rejected Zhora's full-spec claim: vendored ONNX v1.16.2/IR10 proto is stale against v1.22.0/IR13, and base64 retained-proto native text is non-authoritative. Zhora is locked out; Batty owns the revision.

## 2026-07-16T15:39:27Z — Scribe session update

- 🟢 Third onnx-rs re-review cleared Deckard’s authoritative native-text residual merge, including adversarial list-cardinality and opaque-string edits.
