# deckard — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu. **Team formed:** 2026-07-12.

---

## 2026-07-12 — Phase 1–3 + Infrastructure

- **Phase 1** (2026-07-12T09:13:00): CPU `Environment`/`Session` load-run APIs, tensor `Value` helpers, graph metadata, `ModelDirectory::load`, `Tokenizer` encode/decode helpers.
- **Phase 2** (2026-07-12T09:38:00): Paged KV tensor storage (`new_with_tensor_config`, append/write/materialize), prefix cache page ownership/refcount, CoW-safe writes, ORT 1.27.0 standalone packaging.
- **Phase 3** (2026-07-12T10:10:00): Hot/cold LRU page tiering; opt-in `KvDType::Int8` symmetric per-page quantized KV.
- **ORT substrate** (2026-07-12T12:02:00): Pipeline schema/loader, MiniJinja chat templates, multi-EOS discovery, fp16 Value helpers, zero-copy IoBinding, StaticCacheDecodeSession.
- **Hardening** (2026-07-12T13:14:00): GPU EP, batched static-cache decode, ORT checksum notes.
- **Active-row compaction §26** (2026-07-12T13:52:00): `BatchedStaticCacheDecodeSession` active-row API (`set_active_rows`, `compact`, `admit_row`, etc.).
- **ORT comparison suite** (2026-07-12T14:28:00): All 5 ORT real-model comparison tests use `intra_op_threads=1` + shared `Environment` — flake eliminated.
- **Release path** (2026-07-12T14:50:00): `.github/workflows/publish.yml`, CARGO_REGISTRY_TOKEN, leaves-first order, skip-if-published.
- **Preprocess/Mobius** (2026-07-12T17:30:00): Image tiling, ORT compaction tests, Mobius paged-cache draft PR #395.
- **Reviewer-lockout follow-up** (2026-07-13T18:30:00): Sapper thumbnail token/pixel order fix `8a0cf4b`.
- **Model lifecycle scope** (2026-07-13T20:55:00): §37/#9 M1–M4 milestone plan (Zhora implemented M1).

## 2026-07-13T23:15:17Z — §38 K3 code review
Reviewed Leon's connector_bridge/engine/config (commit 2667b3d). All 7 high-risk items clean (no nested-runtime panic, no refcount aliasing, correct chunk-boundary math, inert Null, honest deferral, model-agnostic, clean lock discipline). 🟢 SHIP. Advisory (K4-materialize): chunk hash was prefix-independent — Zhora fixed (ac12480).

## 2026-07-14T02:37:00Z — ORT2 loader + loader-weights + Perfetto reviews
- ort2-loader 🟡 Gaff, loader-weights 🟡 Gaff. Perfetto #13 🟢 SHIP.

## 2026-07-14T05:04:00Z — ORT2 loader const-fold-lite (reviewed 🟢 Gaff)
`squad/ort2-loader-shapeinfer` (b6f032e): ConstEnv partial evaluator; bert_toy 135→50 unresolved values.

## 2026-07-14T06:06:00Z — H-D1 Three-Layer Overflow Fix (deckard, cherry-picked to main)
Holden 🔴 rejected Batty; Deckard authored fix. Layer A: `checked_storage_bytes` in IR dtype. Layer B: `SessionError::ShapeOverflow` in executor. Layer C: i128 address math in ep-cpu strided. 4 regression tests; Holden re-review 🟡 SHIP. Cherry-picked: dbf2d70, 9dcdc04, f749012.

## 2026-07-14T07:20:00Z — ORT2 shape-inference fix (deckard-10)
Dual 🔴 reject (Chew: FusedMatMul; Holden: DimExpr overflow). Roy locked out. Fixed: (1) dedicated `fused_matmul` handler matching ORT contrib_defs.cc; (2) all DimExpr combiners use `checked_*` with overflow sentinel. Applied all advisories. Commit `09988f3`. Both re-reviews 🟢. Merged: 4d24634 + f9b5caa. Deckard now locked out of shape-inference artifact.

## 2026-07-14T08:40:00Z — ORT2 IR dtype hardening (deckard-11, merged 909f0a0)
Fixed two wrong `DataType` discriminants (Float8E5M2: 18→19; Uint4: 23→21). Added Float8E4M3FNUZ=18, Float8E5M2FNUZ=20, Float4E2M1=23. Hardened unmodeled attrs. Reviews: Chew 🟢, Holden 🟡. Deckard locked out of IR dtype artifact.

## 2026-07-14 — deckard-12, deckard-13: Fusion reviews
- deckard-12: Roy decline-to-fuse guards 🟢 APPROVE.
- deckard-13: Batty DAG-aware LayerNorm matcher 🟢 APPROVE. Advisories A2/A3 noted.

## 2026-07-14T14:35:00Z — deckard-14: EPContext §55 ep-api contract merged
`EpContext`, `EpContextRegistry`, `build_ep_context_registry`. Three EP trait methods with safe defaults. New `EpError` variants. 22 unit + 4 lib tests. Merged `d18a8a3`. ⚠️ Shared-checkout race during task; lesson: parallel commit-producing agents must use separate worktrees.

## 2026-07-14T15:00:00Z — deckard-15: Review EPContext CONSUME path (batty-15)
`squad/ort2-epcontext-session` @ `d59edc5`. Verdict: 🟡 approve with 4 non-blocking advisories (A1 covered_nodes dedup; A2 dup primary diagnostic; A3 placement discarded; A4 session-level traversal test).

## 2026-07-14T16:20:00Z — deckard-16: ONNX encoder v2 revision
Leon 🔴 blocked Roy v1 for §55.6 violation. Deckard revised: `Attribute::String` holds `Vec<u8>` throughout IR — eliminates all EPContext special-cases. All 9+15+7+34+40 tests green. Commit `55c7608`, merged `de7ccce`. Leon 🟢 APPROVE.

## 2026-07-14T16:45:00Z — deckard-17: EPContext writer v1 review (🔴 BLOCK)
B1 data-loss: non-injective sidecar names silently overwrite partition blobs. Named Leon as revision owner; Batty locked out.

## 2026-07-14T17:45:00Z — deckard-18: EPContext writer v2 re-review (🔴 BLOCK)
B1 resolved, but blanket `(source, partition_name)` duplicate-identity rejection over-fires on legitimate distinct primaries. Named Gaff as revision owner; Batty and Leon locked out.

## 2026-07-14T18:00:00Z — deckard-19: EPContext writer v3 re-review (🟢 APPROVE)
Gaff's v3 @ `6e65e85`. Regression closed: no duplicate-identity rejection; `duplicate_primary_identity_round_trips_external` passes; B1/A1/A2/sanitizer intact. All suites + clippy + 8-crate build PASS. Final merged commit on main: `0fa025e`.

## 2026-07-14T18:55:00Z — deckard-20: Review EPContext §55.5 parse_options + model-agnosticism + export seam (chew-25)

Non-author review of `squad/ort2-epctx-options` @ `3e8dbde95effb006b28d117e3f8c5491d464e95f`. Scope: parse_options refactor regression, §21.4 validation, model-agnosticism, export seam.

- parse_options refactor: no regression — all previously recognized keys preserved; optimization behavior identical.
- §21.4 validation: fail-closed for all three keys; all parse-and-reject paths tested.
- Model-agnosticism: grep confirmed no model/vendor/op literals in production `src/`.
- Export seam: `export_ep_context` correct; disabled path side-effect-free; `TODO(compiler)` seam correctly placed; executor accessors immutable `pub(crate)`.
- C API: no divergent option logic; verbatim forwarding only.
- **Verdict: 🟢 APPROVE** — no regressions, no advisories.
