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
