# batty — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12

## [Summary 2026-07-12 — 2026-07-13] Engine, KV, and GQA foundations

Phases 1–4 engine wiring (greedy generation, paged-KV, prefix reuse, constrained decoding, `DecodeSession`/`StaticCacheDecodeSession`). §24 sampler processors, §23 FIM APIs, §25 extensibility seams (`SpeculativeProposer`, `Sampler`, `ProcessorChain`), §27 prompt-lookup speculative decoding (`NgramProposer`). CI clippy blocking (`--workspace --all-targets -- -D warnings`). Decode ownership canonical: ORT owns forward exec + KV; engine owns generation policy + logical KV. fp16 GQA WebGPU via `com.microsoft::GroupQueryAttention` landed; GQA KV config moved from genai_config.json to our own `inference_metadata.yaml` (per Justin's correction). Metal-prefill hybrid: KV-handoff seam proven, premise falsified (Metal TTFT 1.5–2× slower than CPU — do not productionize). SWA/sink hardening: debug_assert! guards + rewind_to fix (commit 4e51d59). Vision token-expansion (#14, 79a030a) initially landed; Luv rejected → lockout → Leon fixed. Gemma4 E2B W2: per-layer `LayerTensorConfig` + `MaterializedLayerKv` (commit 9db1a3c, 78 tests; connector KvPayload still uniform — fix before any mixed-geometry model). Advisory A2 pending (Batty): `try_connector_kv_injection` should fall back Ok(None) instead of hard-failing on import_runner_kv error.

## 2026-07-14T02:37:00Z — ORT2 ep-api + ep-cpu merged
- **ep-api (65ec9f6):** DeviceBuffer ownership hardening, DLPack alignment (byte_offset, i64 strides), Cost non_exhaustive. Reviewed 🟡 Holden. 17 tests.
- **ep-cpu (ea30279):** CpuExecutionProvider + 7 Phase-1 pure-Rust kernels (MatMul, Add, Relu, Reshape, Transpose, Gather, LayerNorm). Reviewed 🟡 Chew + 🟡 Holden. 39 tests.
- Track D (session) must call `strided::view_in_bounds` before kernel dispatch; kernels trust caller for storage bounds.

## 2026-07-14T05:04:00Z — ORT2 capi Track E + ep-cpu +17 kernels merged

- **squad/ort2-capi** (8c9c8fc): Phase-1 C ABI — opaque handles, null-guarded, catch_unwind-fenced, atomic `ort2_run` commit, `SessionError→OrtErrorCode` mapping. 12/12 tests; Miri-clean. Closes Phase 1. Reviewed 🟢 Holden.
- **squad/ort2-epcpu-ops** (e485a83): +17 bert_toy kernels — Sub/Mul/Div/Pow/Min, Sqrt/Erf/Tanh, Cast, ReduceMean, Softmax, Shape, Unsqueeze, Expand, Slice, Constant, Gemm. 90/90 tests; no new deps. Reviewed 🟡 Chew.
- Softmax uses opset-13 per-axis semantics (correct for bert_toy last-axis; opset-12 coerce guard advisory assigned Roy/Deckard — Batty locked on this advisory).
- Loader gaps flagged (Slice/Expand/Constant shape inference) → addressed by Deckard b6f032e.

## 2026-07-14T06:06:00Z — Phase-1 Hardening: 6 Advisories Closed + capi Fix

- Closed 6 deferred LOW-severity advisories from Phase-1 reviews (Chew + Holden):
  1. Softmax opset≤12 vs ≥13 dual semantics (coerce_2d + dual registry SoftmaxLegacy@1/Softmax@13; effective_opset plumbed)
  2. Min/Max NaN-propagation (explicit is_nan() guard)
  3. Cast saturate: num_to_int! macro directly to target type
  4. checked_numel + SessionError::ShapeOverflow at both alloc sites (H-D1 preliminary)
  5. Multi-output dynamic_output_shapes guard (OutputShapeCountMismatch)
  6. Slice geometry extracted to shared slice_plan helper
- Fixed capi map_session_error non-exhaustive match (SymbolConflict/RankMismatch/UnresolvedShape/ShapeOverflow/OutputShapeCountMismatch arms added; no catch-all _).
- **Holden review (🔴):** checked_numel closed dims-product overflow but storage_bytes(numel) still unchecked → heap OOB for [2^61]×f64. **Batty locked out of H-D1 storage-sizing artifact**; fix reassigned to Deckard.
- **Chew review (🟢):** All 6 fixes numerics-correct.
- Deckard completed H-D1 fix; Holden re-reviewed → **🟡 SHIP**; merged to main.

## 2026-07-14T10:00:00Z — ORT2 fused-op contrib domain (batty-12)

- **Task:** Move optimizer-emitted fused ops to `com.microsoft` contrib domain; generalize ep-cpu dispatch to key on `(domain, op_type)` via registry.
- **Work:** Added `CONTRIB_DOMAIN = "com.microsoft"` in optimizer/fusion.rs; `apply_fusion` sets domain. Added `OpRegistry::supports(op_type, domain)` + `norm_domain` (ai.onnx↔"") to ep-api/registry.rs (applied in both `lookup` and `supports`). Registered com.microsoft/LayerNorm in ep-cpu (additive; same LayerNormFactory); updated len invariant (PHASE1_OPS+2). Changed provider.rs `supports_op` gate to `registry.supports`. Added com.microsoft LayerNorm shape rule in shape-inference. Left FusedMatMulBias/FusedGemm kernel-less in both domains (none existed).
- **Result:** debug+release green optimizer(27)/ep-cpu(102)/ep-api(17)/shape-inference(70)/session(19). bert_toy conformance PASS max_abs 1.192e-7. clippy clean.
- **Review:** Gaff gaff-7 → 🟢 GREEN. Merged to main (`8cab9d2`).

## 2026-07-14T11:40:00Z — ORT2 fusion-executable (batty-13)

- **Task:** Make `optimization="all"` executable + parity-correct on `bert_toy`.
- **Delivered:** `0f4811e` → merged as `e9bf155` (cherry-pick to main).
- **Changes:** Schema-aware LayerNorm fusion (`fusion.rs`), `FusedMatMulBias` CPU kernel (new file), shared `matmul_dense` extraction, `FusedMatMulBias` shape rule, tripwire → real parity assertion.
- **Parity:** `opt=all` vs opt-off = 0.0 (byte-identical); vs reference = 1.192e-7 (unchanged). Full suite green debug+release.
- **Reviews:** Chew 🟡 (F1: opset-18 axis-as-input; F2: non-f32 epsilon — both decline-to-fuse gaps); Gaff 🟡 (G1: MatMul+Add shape guard for bias-expanding Add).
- **Batty locked out** of follow-ups F1, F2, G1. Owners: Roy/Deckard/Leon.

## 2026-07-14 — batty-14: ORT2 DAG-aware LayerNorm e2e fusion
Diagnosed bert_toy LayerNorm as 10-op split-diff variant. Added `try_match_layernorm` DAG-aware matcher (9-op + 10-op); `layernorm_spec` generalized to 9-or-10 nodes with same-X guard. Now 12× LayerNormalization + 32× FusedMatMulBias. `"all"` vs ref 1.043e-7, vs off 1.416e-7. Chew (chew-21) and Deckard (deckard-13) both 🟢 approved. Merged main `1817890`.

## 2026-07-14T15:00:00Z — batty-15: ORT2 EPContext session CONSUME path (§55.3)

- Implemented `session/src/epcontext.rs`: `load_ep_context_nodes(graph, model_dir, eps) -> Result<EpContextPlacement>`.
- Two-phase dispatch: Phase 1 (main_context=true) claims EPs and calls `ep.load_context`; Phase 2 (main_context=false) resolves references by (source, partition_name), no second blob load. Payload dedup via `HashSet<(source, bytes)>` — shared packed binaries load exactly once.
- Executor bypass: EPContext nodes skipped by `is_ep_context_op` predicate — never reach CPU kernel dispatch.
- Model-dir threading for embed_mode=0 external blobs (§19.2 policy).
- New `SessionError::DanglingEpContext { source_key, partition_name }`.
- 7 new tests (MockCompiledEp): embed/external round-trip, unclaimed/dedup/dangling/dup-source/session-level. All green; clippy clean; no new unsafe.
- Merged to main `46f2861`. Reviews: Deckard (deckard-15) 🟡, Chew (chew-23) 🟡.
- Advisory owners (Batty locked out): A1 covered_nodes dedup gap; A2 duplicate primary detection; A4 session-level traversal test (owner: Pris per Chew).

## 2026-07-14T18:20:00Z — batty-16: ORT2 EPContext §55.4 writer v1

- **Task:** Implement EPContext dump/write path — the inverse of the consume path (batty-15).
- **Delivered:** `206742e` — `dump_ep_context`, `EpContextDumpConfig`, `EpContextPartition`, embed+external sidecar, subgraph replacement, 3 loader + 3 session tests.
- **Model-agnostic**: reuses loader's existing constants; no new op/vendor literals in production code.
- **embed_mode=1**: `Attribute::String(Vec<u8>)` byte-exact; **embed_mode=0**: sidecar beside model, bare relative filename.
- **Reviews**: gaff-12 🟢 (advisories A+B); deckard-17 🔴 B1 data-loss — non-injective sidecar names silently overwrite partition blobs.
- **LOCKED OUT** of this artifact after deckard-17 rejection. Leon owned v2 revision; Gaff owned v3 revision.
- Final commit on main: `0fa025e` (gaff-13 v3).

## 2026-07-14T13:55:00Z — batty-17: FusedGemm CPU kernel + shape rule + synthetic parity (Phase-2 optimizer)

Implemented the last deferred item of the Phase-2 optimizer: `com.microsoft::FusedGemm` CPU kernel (`ep-cpu/src/kernels/fused_gemm.rs`) = `Relu(MatMul(A,B)+bias)` reusing `matmul_dense`, `broadcast_apply`, and new shared `relu::relu_in_place`; registered `("FusedGemm","com.microsoft",1)`. Shape-inference handler in `linalg.rs` (output == MatMul shape; bias broadcasts; Relu shape-neutral). Extended bias-broadcast decline-guard in `fusion.rs` to cover `FusedGemm`. Synthetic parity test `fused_gemm_parity.rs`: max_abs 0.0 fused vs unfused; asserts 1 FusedGemm / 0 stray nodes. Commit `9e302a6` → cherry-picked to main as `4916618`. **Reviewer: roy-19 🟢** (1 advisory: add permanent expanding-bias decline test). **Fusion trio COMPLETE: LayerNorm ✅ / FusedMatMulBias ✅ / FusedGemm ✅.**

## 2026-07-14T14:50:00Z — batty-18: AttentionFusion — SDPA core fused into com.microsoft::FusedAttention (64edd75)

Implemented the final Phase-2 optimizer sub-item. Fused `MatMul(Q,Kᵀ)·scale[+mask]→Softmax(axis=-1)→MatMul(·,V)` into `com.microsoft::FusedAttention` with inputs `[Q, K, V, (mask)]`, attributes `scale` (f32) + `k_transposed` (0/1). Strict decline-to-fuse guards (explicit last-axis axis, concrete scalar scale with MatMul-output check, left-operand Softmax, exactly-one Add scale operand, no interior escape). New files: `ep-cpu/src/kernels/fused_attention.rs` (batched SDPA kernel + 5 unit tests), `session/tests/fused_attention_parity.rs` (masked + unmasked, ATOL 1e-6). `softmax_slices` made `pub(crate)` (pure visibility change). Folded Roy's expanding-bias FusedGemm decline test. bert_toy: 5 FusedAttention / 0 Softmax; LayerNorm=12 / FusedMatMulBias=32 unaffected. Parity: vs reference 1.043e-7 (bound 2e-3); vs opt-off 1.416e-7 (DRIFT_ATOL 1e-5). Commit `39a23c8` → cherry-picked to main as `64edd75`. **Reviewers: roy-20 🟢** (4 adversarial declines) **+ chew-26 🟢** (kernel numerics + shape + parity). **Phase-2 OpFusion + AttentionFusion COMPLETE. Fusion set: ConstantFolding ✅ / DeadNodeElimination ✅ / OpFusion [LayerNorm + FusedMatMulBias + FusedGemm + FusedAttention] ✅ / AttentionFusion ✅.**
