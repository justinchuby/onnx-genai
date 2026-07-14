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

## 2026-07-14T02:37:00Z — Reviewed ort2-loader + loader-weights
- **ort2-loader (7e0e367):** 🟡 — ONNX→IR pipeline, protox build, mmap weights, shape inference.
- **loader-weights (dd5297d):** 🟡 — WeightStore re-export, norm_axis fix.

## 2026-07-14T05:04:00Z — ORT2 review: loader const-fold-lite shape inference

- **squad/ort2-loader-shapeinfer** review (🟢 SHIP): No wrong constant found — every fold aborts via `?`/`None` on unknown/non-integer operands; symbolic operands degrade to fresh symbols; bounds enforced at all entry points. `bert_toy_optimized_every_value_resolves` ran on real model (257 KB, not skipped) — passed. All 27/27 tests green. Advisories: A1 `Div` truncation vs floor for negative operands (no positive-dim impact); A2 `Shape` of unresolved input folds to rank-0 (pre-existing).

## 2026-07-14T07:20:00Z — ORT2 shape-inference crate review

- **gaff-6:** Reviewed `onnx-runtime-shape-inference` registry dispatch, topo driver, shape-data side-table, and public API. 🟢 APPROVE. Ran 4 integrity probes: registry opset boundaries correct, driver transactional (no half-annotated state on error), shape-data per-call HashMap (no stale leakage), API minimal and panic-free. IR contract NOT modified — zero lines changed in `onnx-runtime-ir`. Roy not locked out.

## 2026-07-14T10:00:00Z — ORT2 fused-domain dispatch review (gaff-7)

- **Task:** Review Batty's fused-op contrib domain change on `squad/ort2-fused-domain` (`1e894de`). Verify dispatch/registry soundness.
- **Verdict:** 🟢 GREEN — dispatch set correct, normalization symmetric, no phantom kernel registration.
- **Key checks:** Provider gate accept set == registered (op_type, domain) pairs exactly; default-domain == PHASE1_OPS 1:1; len invariant holds. ai.onnx→"" normalization symmetric in both `lookup` and `supports`. Contrib opset u64::MAX → resolves v1, no panic. Dual-domain LayerNorm: distinct OpKey entries, no cross-domain resolution. FusedMatMulBias/FusedGemm: supports()=false in both domains → rejected at placement. Debug+release all green; bert_toy PASS max_abs 1.192e-7; clippy clean.

## 2026-07-14T11:40:00Z — Review: ORT2 fusion-executable FusedMatMulBias (gaff-8)

- **Reviewed:** `squad/ort2-fusion-executable` @ `0f4811e` (author: Batty).
- **Scope:** FusedMatMulBias kernel numerics, shape rule, registry/dispatch, MatMul+Add operand-order generality.
- **Verdict:** 🟡 Approve with required follow-up.
- **Key confirmed:** Kernel numerics correct ✅; `matmul_dense` extraction no regression ✅; shape rule consistent ✅; registry/dispatch consistent ✅; operand-order ROBUST (not baked to bert_toy) ✅.
- **Gap (G1):** MatMul+Add fusion has no shape guard — silent-wrong when bias operand would expand matmul output. Owner: Roy/Deckard (Batty locked out).

## 2026-07-14T14:35:00Z — gaff-10: Reviewed leon-11 (LayerNorm order guard) + roy-17 (EPContext loader)

**gaff-10a — Leon LayerNorm order guard:** 🟢 APPROVE. Guard structural/model-agnostic; non-tautological positive + adversarial coverage; drift and reference bounds separate; 31→33 tests; debug+release+clippy green.

**gaff-10b — Roy EPContext loader LOAD path:** 🟢 APPROVE. Opaque blob preservation byte-for-exact-byte (scoped to `is_ep_context_op && ep_cache_context` only); path-safety rejects before join; mmap unsafe follows weights.rs idiom; 7/7 epcontext + 15/15 loader tests green; clippy clean.

## 2026-07-14T16:20:00Z — onnx-encoder v1 review (gaff-11)
Reviewed Roy's ONNX encoder v1 (`9ffd65c`). 🟢 GREEN — round-trip fidelity and prost encoding correct for Phase-1/2 scope. Real BERT fixture (257 KB) byte-exact. 4 non-blocking advisories: A1 subgraph formal I/O silently omitted (recommend guard), A2 model metadata silently defaulted, A3 STRING byte-exact doc nuance, A4 external re-inlining bloat. Did not identify the §55.6 model-agnostic violation (found by Leon). Not locked out.

## 2026-07-14T16:45:00Z — gaff-12: Review EPContext §55.4 writer v1 (batty-16)

- **Reviewed:** `squad/ort2-epctx-writer` @ `7eb30ff` (author: Batty). Scope: round-trip fidelity + path/security safety.
- **Verdict:** 🟢 GREEN — byte-exact both modes; sidecar sanitizer resists path traversal. No blocking findings.
- **Confirmed**: embed non-UTF-8 byte-exact; external sidecar verbatim write+mmap; hostile inputs (traversal sequences, NUL, `\`) all sanitized to in-directory filenames; node boundary `X→EPContext→Y` preserved.
- **Advisory A**: sidecar collision on duplicate sanitized (source, partition_name) — suggested index disambiguator. **Advisory B**: sanitizer test covers only `/`.
- Reproduced: loader 15+3 ok; session 10 ok.

## 2026-07-14T17:50:00Z — gaff-13: EPContext §55.4 writer v3 (revision owner, deckard-18 named)

- **Task:** Fix deckard-18's 🔴 blocking regression — remove over-broad `(source, partition_name)` duplicate-primary rejection from leon-14 v2. Batty and Leon locked out.
- **Change:** Deleted blanket `HashSet<(&str, &str)>` guard in `dump_ep_context`; updated `# Errors` doc-comment; deleted `duplicate_partition_identity_is_rejected` test.
- **Added:** `duplicate_primary_identity_round_trips_external` — two same-source/same-(empty)-name primaries, distinct non-UTF-8 blobs, external mode → `m_ctx_p0_EpA.bin`/`m_ctx_p1_EpA.bin` distinct; each reloaded byte-exact.
- **Kept intact:** B1 injective `_p{index}_` sidecar names; A1 enable-gating; A2 NodeId seam doc; sanitizer test.
- **Commit:** `0fa025e` (= `6e65e85`). **deckard-19 🟢 APPROVE.** Final merged commit on main.

## 2026-07-14T18:55:00Z — gaff-14: Review EPContext §55.5 capi FFI + e2e round-trip (chew-25)

Non-author review of `squad/ort2-epctx-options` @ `3e8dbde`. Scope: capi FFI memory safety + e2e correctness.

- Audited four new capi entry points: null handling PASS, invalid UTF-8 PASS, ownership/lifetime PASS (borrow-not-consume), panics across FFI PASS (all in `guard`).
- E2e: byte-exact non-UTF-8 round-trip via mock EP confirmed; disabled path writes nothing confirmed.
- **Verdict: 🟢 GREEN** — 2 non-blocking advisories:
  - A1: No negative FFI tests for null/invalid-UTF-8 into `ort2_add_session_config_entry` (coverage gap only).
  - A2: Released-handle reuse unguarded — by design, matches existing opaque-handle contract.
