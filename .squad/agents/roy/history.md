# roy — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12



## 2026-07-12T09:13:00-07:00 — Phase 1 foundation plan delivered
- Assessed Phase 1 status and identified real ORT CPU execution, model/tokenizer discovery, and minimal greedy generation as the critical path.
- Shared context for next batch: Deckard supplied ORT/tokenizer contracts, Batty supplied the generation API, and Pris supplied deterministic metadata/fixture coverage.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Roy's Phase 2 plan was executed successfully: paged KV tensor storage, prefix cache lifecycle/CoW, persistent multi-session engine APIs, HTTP/SSE session surface, and Pris's exit tests are now in place. Shared contracts include `prefix_cache_hit_len`, `X-Session-Id`, and standalone ORT runtime packaging.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Phase 3 plan completed and executed. Team delivered speculative decoding, tiered/quantized KV, priority/preemption, streaming/accounting hardening, and validation; speedup limitation is environment-bound locally.

## 2026-07-12T12:02:00-07:00 — Phase 4 and long-context plans completed
Roy's Phase 4, tool-use/grammar, and long-context plans were executed: pipeline execution, constrained decoding, OpenAI tool use, Qwen/Hermes validation, and O(1)/token static-cache decode are now recorded. Next roadmap follows DESIGN §23-28 plus paged attention.

## 2026-07-12T13:14:00-07:00 — Architecture review merged
Roy's workspace review is now in decisions: crate split is sound, but engine.rs must be decomposed and §26 needs an engine loop/channel plus DecodeBackend before true batching. §27/§28 need SpeculativeProposer/verifier seams.

## 2026-07-20T00:00:00Z — §34 Router R2+R3+affinity+hardening landed
- R2 (commit 1f58099): Created `crates/onnx-genai-router/` — pure session-aware routing core. Modules: `config.rs`, `node.rs`, `router.rs`, `session_map.rs`, `prefix_map.rs`. Policies: AffinityThenLoad, PrefixThenLoad, LeastKvUsage, Weighted. FNV-1a 64-bit prefix hash; optional JSON session-map persistence. 36 unit tests, clippy clean.
- R3 (commit ee8e464): Runnable reverse-proxy binary with `node_poller`, `proxy`, `api`, `metrics`, `state`, `main`. hyper-util client for transparent SSE streaming; hand-rolled Prometheus text; draining semantic; lazy rebalance. `/router/status|sessions|metrics|drain|rebalance` endpoints; all else proxied. 67 tests, clippy clean.
- Affinity weight fix (commit 54e5363): `Weighted` policy corrected from binary gate to continuous scoring bonus per §34.5. Formula: `kv_usage × kv_weight + normalized_queue × queue_weight − bonus`, where `bonus = affinity_weight` if affinity node and below overload threshold.
- R3 hardening (commit a36cbbd, post Deckard 🟡 review): (1) concurrent poller via `join_all`; (2) miss-on-unknown-id; (3) 16 MiB response cap on session affinity capture; (4) rebalance overload guard (`least_loaded_node_below_threshold`). 73 tests total.


## 2026-07-14T02:37:00Z — ORT2 Phase 1 foundation merged
- **Commit:** 203161c — 6 crates scaffolded, `onnx-runtime-ir` with 34 passing tests
- IR gaps flagged by Deckard (Track A): `DataType::from_onnx` fp8/int4 numbering vs ONNX spec, no `DataType::Undefined`, no unknown-rank `Shape` sentinel. Roy to address before quantized-model work.

## 2026-07-14T05:04:00Z — ORT2 session executor + dynshape merged (Track D)

- **squad/ort2-session** (24b8129): Sequential CPU executor — topo-order dispatch, shape-keyed kernel cache, `view_bounds` gate on all views before dispatch, Miri-clean borrow strategy. 8/8 tests. Reviewed 🟢 Chew + 🟡 Holden.
- **squad/ort2-session-dynshape** (da8eab3): Runtime symbolic-shape resolution — `bind_symbols`/`resolve_all`/`ensure_buffer` pipeline; buffer reuse on same shape, realloc on change; cache keyed on resolved shapes. 14/14 tests; Miri-clean. Reviewed 🟡 Holden.
- Open advisories (non-blocking): Holden A1 (mid-run error-path buffer leak), H-D1 (unchecked shape-multiply), Chew A2 (gappy-optional input compaction).
- Loader gap flagged to Deckard: shape rules for `Attention`/`EmbedLayerNormalization` needed before full bert_toy run.
- Softmax opset-12 guard (from Chew ep-cpu review) assigned Roy/Deckard (Batty locked).

## 2026-07-14T06:06:00Z — EPContext Design + embed_mode Fix (roy-10, roy-11)

- **roy-10:** Authored docs/ORT2.md §55 (EPContext node design) on branch `squad/ort2-epcontext-design` @ c48f5c4. Covered op schema (all 10 attrs), embed_mode/main_context semantics, session-option keys, model-agnostic dispatch via EpContextRegistry. Merged to main: **96f1ed4**.
- **fact-checker** found one required fix: §21.4 `ep.context_embed_mode` default stated as `1`, ORT runtime default is `0`.
- **roy-11:** Applied correction (§21.4 default 1→0; `EpContextGenOptions.embed_mode` → `ExternalFile`) + TOC update. Merged to main: **cf614e4** (current HEAD).
- Roy-12 in flight: updating docs/ORT2.md §15 (CuTe Kernel Strategy) with CUDA EP stack decision — not yet landed.
- **roy-12:** Updated docs/ORT2.md §15 (CUDA EP Kernel Strategy — cudarc + cuBLASLt + CuTe + cuDNN/FA3 stack; cuTile deferred Phase 3). Doc-only commit `edd2b3a`, merged to main.
- **roy-13:** Authored `crates/onnx-runtime-shape-inference` on `squad/ort2-shape-inference`. Extensible per-op registry, DimExpr symbolic polynomial, shape-DATA propagation, 40+ handlers, bert_toy fully resolves. 56 tests green. Sent to review. Chew (🔴 FusedMatMul) + Holden (🔴 DimExpr overflow) rejected; Gaff (🟢 registry/driver/API) approved. Roy locked out per reviewer-protocol — Deckard assigned fix. Post-fix re-reviews both 🟢. Merged to main: **4d24634** (feat) + **f9b5caa** (fix).
- Roy is currently in flight wiring shape-inference into loader/session — not yet landed.

## 2026-07-14T08:40:00Z — ORT2 shape-inference wiring (roy-14, merged 98a3310)

- **roy-14:** Wired `onnx-runtime-shape-inference` into the loader. Loader now owns static shape inference: `build_from_bytes_with_weights` runs `registry.infer_graph(MergePolicy::Permissive)` after graph build. Deleted `const-fold-lite` `shape_inference.rs` (~1.1k LOC, `KnownVal`/`ConstEnv` pass) — no shim, pre-release. Session JIT (`dynamic_output_shapes`) retained as fallback for data-dependent extents.
- **broadcast_dim fix:** Changed `context.rs` `broadcast_dim` to keep the smaller `SymbolId` representative (not mint fresh) when two symbolic dims meet at a broadcast axis. Fixes Expand-contamination regression (`bert_toy` UnresolvedShape for value `"106"`). Matches ORT symbolic inference ("keep representative"). Added `add_two_distinct_symbols_keeps_named_representative` test.
- **Verification:** `bert_toy` conformance max_abs 1.192e-7 (unchanged). Full ORT2 suite green debug+release.
- **Reviews:** Chew (chew-17) 🟢; Holden (holden-10) 🟢. Merged to main: **98a3310**.

## 2026-07-14T11:00:00Z — Session optimize stage activated (roy-15, merged 5a2d527)

- **roy-15:** Wired `onnx-runtime-optimizer` into `onnx-runtime-session`'s `build()` pipeline as opt-in stage. New `SessionBuilder.option("optimization", "none"|"basic"|"all")` (default `"none"`). Default path byte-identical. Pipeline: load → optimize_graph(level) → opset-import → re-infer(Permissive) → compile → allocate, all gated on level. `basic` (const-fold + DCE) parity 0.0 vs opt-off; conformance 1.192e-7 unchanged. `all` (OpFusion) not yet executable — schema-unaware fusion produces kernel-less `FusedMatMulBias`/`FusedGemm` + wrong-arity 5-input `com.microsoft::LayerNormalization`; fails cleanly before numerics; tripwire test `full_optimization_fusion_path_is_not_yet_executable` guards regression. Follow-up flagged to Batty. 53 tests green debug+release, clippy clean. Chew (chew-19) review: 🟢 APPROVE. Merged to main `5a2d527`.

## 2026-07-14 — roy-16: ORT2 fusion decline-to-fuse guards
Hardened LayerNorm (axis/epsilon/structure guards) and MatMul+Add (trailing-broadcast bias guard) in `fusion.rs` to decline-to-fuse when assumptions unproven. 5 new unit tests. Deckard (deckard-12) 🟢 approved. bert_toy 32× FusedMatMulBias preserved; `"all"` vs off 0.0, vs ref 1.192e-7. Merged main `8f222bd`.

## 2026-07-14T14:35:00Z — roy-17: EPContext §55 loader LOAD path merged

- New `crates/onnx-runtime-loader/src/epcontext.rs`: `EpContextNode<'g>` typed view, `EmbedMode`, `EpContextBlob { Embedded(Vec<u8>), External { path, map: Mmap } }`, `resolve_ep_context`.
- Lossless binary blob: `graph_builder` special-cases `ep_cache_context` → UINT8 tensor storage (avoids `from_utf8_lossy` corruption).
- Path-traversal guard: rejects absolute, `..` parent-dir, and root/prefix components before `join`.
- 7 new tests green debug + release. Loader writer (§55.4) is a later task.
- Merged to main `d18a8a3` (part 1). Gaff (gaff-10) 🟢 APPROVE.
- **Integration note:** session integrator should use `source_key` (not `source`) in `EpError::NoEpForContext` (Deckard's ep-api, thiserror 2.0 constraint).

## 2026-07-14T16:20:00Z — onnx-encoder v1 (roy-18)
Authored ONNX encoder v1: `crates/onnx-runtime-loader/src/encoder.rs` (+518), `tests/encoder.rs` (+488). Model-agnostic inverse of the loader decode path. Byte-exact round-trip on synthetic + real BERT 257 KB fixture. Gaff: 🟢 (4 non-blocking advisories). Leon: 🔴 BLOCK — `is_ep_context_op` + `"ep_cache_context"` literal in generic attribute layer violates §55.6. Locked out of the encoder artifact for this cycle; Deckard assigned as revision owner. v2 implemented by Deckard (commit de7ccce).

## 2026-07-14T13:55:00Z — roy-19: Review — FusedGemm kernel + parity (batty-17)

Reviewed Batty's FusedGemm work (commit `9e302a6`). Verified fusion-guard generalization with a throwaway expanding-bias probe (guard declines correctly; probe reverted). bert_toy unchanged and green. Kernel stage order `Relu(MatMul + bias)` correct; byte-identical to FusedMatMulBias + `relu_in_place`. Shape rule delegates to `matmul_shape`. Synthetic parity test high quality (tight 1e-6 atol, Relu actually clamps negatives). 8-crate build + clippy + all suites green. **Verdict: 🟢 GREEN approve.** Advisory: add permanent 3-node FusedGemm expanding-bias decline test for in-repo regression protection.

## 2026-07-14T14:50:00Z — roy-20: Review — AttentionFusion matcher/guards (batty-18)

Reviewed Batty's `com.microsoft::FusedAttention` matcher in `fusion.rs` (optimizer/matcher half; kernel/shape = Chew). Built and ran 4 adversarial decline tests: classifier-head-bias-softmax, scaled-by-relu-not-qk, ambiguous-both-scaled, interior-value-escape — all correctly declined. Verified all decline guards return None (no silent defaults), both Div/Mul scale forms correct, k_transposed contract agrees with kernel, bert_toy fuse + conformance held, Roy's folded FusedGemm decline test present and asserting correctly. Optimizer 40 + session 18 + ep-cpu 5 fused_attention tests; 8-crate build; clippy clean. **Verdict: 🟢 APPROVE.** Non-blocking follow-up: document/test rank-2 FusedAttention equivalence (semantics-preserving, not a bug).
