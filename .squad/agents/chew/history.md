# Chew — History

## 2026-07-12: Joined
Hired as a Code Reviewer specializing in numerics/precision as the runtime took on fp16/Q4 quantization, GQA KV, and Mobius model conversion. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: a prior Q4 GGUF→ONNX conversion "loaded but produced garbage" (missing Qwen2 biases + wrong reverse-permute) and a sampling RNG bug returned token 0 — exactly the silent precision defects to catch. Verify against references; require coherent output, not just successful load.


## 2026-07-13T18:30:00Z — Review/fix batch
- Reviewed Leon's DESIGN §40 SWA/attention-sink work and approved it with three optional LOW nits; no rejection lockout needed.

## 2026-07-13T23:15:17Z — §38 K1/K2 review

Reviewed §38 K1 (`crates/onnx-genai-kv/src/connector.rs`) and K2 (`crates/onnx-genai-kv/src/local_tiered.rs`).

- **Top risk verified clean:** `LocalTieredConnector` owns SEPARATE `PageTable`/`PrefixCache` instances from the engine's in-process cache — no refcount aliasing, no double-free risk.
- **Real defect found:** `cpu_load_ms_per_page` declared and defaulted but never read in `locate()`. Load estimate was always `on_cpu * 1.0` (implicit 1 ms/page) regardless of configured rate.
- Verdict: 🟡 **ship-with-recommendations**. Defect remediated by Zhora (commit 30ee870) before K3 landed.

### Shared context for future KV connector reviews
- The engine's `prefix_cache` (refcounted, `lookup_shared`/`release_shared`) and the connector's `PrefixCache`/`PageTable` must remain STRICTLY SEPARATE — any aliasing creates double-free risk.
- `KvTensorRef` is currently a size-only placeholder — no real KV bytes are stored/fetched yet. K4-materialize will require giving it a real device-tensor handle.
- Prefix-dependent hash invariant is now in place (Zhora, commit ac12480): `KvCacheKey` equality ⟹ identical prefix through that chunk.

## 2026-07-13T23:50:16Z — §38 K4 review

Reviewed Leon's K4 real KV byte materialization (commit `786e268`, read-only, Leon locked out). Verdict: 🟡 **SHIP-with-advisories**.

- **Byte-layout symmetry confirmed correct:** extract (`chunk_payload_from_exported`) and inject (`past_kv_from_payloads`) are symmetric; all four layout sites (extract, inject, `materialize_sequence`, past-tensor shape) agree. No transpose/stride mismatch.
- **No false hits:** prefix-dependent cumulative FNV-1a hash ensures `KvCacheKey` equality ⟹ identical prefix through that chunk.
- **Fetch-vs-recompute gate correct, no off-by-one.**
- **All deferred paths safely no-op** (non-runner, non-f32, continuing session).
- **Gold test rigorous** — simulates fresh node, asserts non-trivial fetch + token-identical output vs full recompute.
- **Advisory A1 → Pris:** `tiny-llm` fixture is single-layer; add multi-layer gold fixture to close cross-layer ordering dimension.
- **Advisory A2 → Batty:** `try_connector_kv_injection` should gracefully fall back (`Ok(None)`) on `import_runner_kv` failure instead of hard-failing `generate`.


## 2026-07-14T00-49-37Z — Gemma4 E2B real-run batch (reviews)

**W2+W3 per-layer KV geometry review (commit 9db1a3c)**
- Verdict: 🟡 SHIP-with-advisories
- Confirmed: per-layer byte extraction, `target_layers.last()` OOB guard, per-layer page sizing, shim removal, write/read byte symmetry for hd256 and hd512
- Advisory: connector KvPayload path uniform-only — dead code for E2B but must be fixed before enabling connector on heterogeneous model

**Milestone B engine fixes review (commit 10f82b3)**
- Verdict: 🟢 SHIP
- Confirmed: fp16↔f32 lossless in required directions, SWA decode-path change bounded to SWA models only, updated lib test correct, widen/narrow are true inverses for past KV, `milestone_b_real.rs` CI-hermetic
- Nit: `detect_shared_kv_proposer` reorder (non-gating, more correct behavior)

## 2026-07-14T02:37:00Z — Reviewed ep-cpu + speculative fix
- **ep-cpu (ea30279):** 🟡 numerics review — signed off on naive GEMM correctness, int4/uint4 `storage_bytes`, LayerNorm population variance.
- **gemma4-accept (8089a1f):** 🟡 numerics review — signed off on `inputs_embeds` concat fix, LinearEmbedder scale application, acceptance metrics.

## 2026-07-14T05:04:00Z — ORT2 reviews: session Track D + ep-cpu +17 kernels

- **squad/ort2-session** review (🟢): Verified topo-sort correctness, value dep resolution, view materialization, initializer/input binding, cache key collision-free. Hand-verified test references in Python. Minor advisories: gappy-optional compaction, cache key omits dtypes.
- **squad/ort2-epcpu-ops** review (🟡): 90/90 tests pass. Independently verified softmax stability, broadcast, Erf accuracy, Gemm. No blocking numeric bug for bert_toy. Advisories: Softmax opset-12 guard (assign Roy/Deckard), Min NaN propagation, Cast overflow saturation vs UB (documented). Conformance harness must confirm last-axis Softmax assumption.

## 2026-07-14T07:20:00Z — ORT2 shape-inference crate review cycle

- **chew-15:** Reviewed `onnx-runtime-shape-inference` op-rule correctness. 🔴 REJECT — `com.microsoft::FusedMatMul` reused plain `matmul`, ignoring `transA`/`transB`/`transBatch`. Common `transB=1` case (`[8,64]·[32,64]ᵀ`) produced `[8,64]` instead of `[8,32]`. All other 40+ op handlers HELD correct. Three non-blocking advisories. Fix assigned to Deckard (Roy locked out).
- **chew-16:** Re-reviewed Deckard's FusedMatMul fix (`09988f3`). 🟢 SHIP — handler verified line-for-line against ORT contrib_defs.cc. Cited case correct, 7 new tests pass, all advisories applied. 69/69 tests green. Roy and Deckard both locked out of this artifact.

## 2026-07-14T08:40:00Z — ORT2 shape-inference wiring + IR dtype hardening reviews

- **chew-17:** Reviewed Roy's shape-inference wiring (`f4141b9`). 🟢 GREEN. Broadcast change conformance-safe (ANON_SYMBOL_FLOOR invariant — smaller-id always prefers session-bindable graph symbol). const-fold-lite deletion safe. `bert_toy` 1.192e-7 unchanged. 52 op-rule tests pass. Two non-blocking advisories (doc phrasing; pre-existing merge_shapes both-symbolic arm).
- **chew-18:** Reviewed Deckard's IR dtype hardening (`f965f0b`). 🟢 APPROVE. All 21 discriminants independently verified against ONNX spec. `to_onnx = self as i32` correct. All classifiers correct for new variants (Float8E4M3FNUZ, Float8E5M2FNUZ, Float4E2M1). Round-trip and unknown tests comprehensive. Advisory: vendored proto stale (stops at INT4=22, missing FLOAT4E2M1=23 — no runtime bug since from_onnx reads raw int). Recommended follow-up: bump vendored proto (owner: Roy/Batty/Leon).

## 2026-07-14T11:00:00Z — ORT2 session optimize review (chew-19)

- **chew-19:** Reviewed Roy's session `optimize` stage activation (`c92a2f2`, `git diff 6f2e518...c92a2f2`, +435/-12). Verified: (1) default-off byte-invariant — `optimize_graph()` provably no-op for `None`, no unconditional re-infer, conformance unchanged 1.192e-7; (2) `basic` genuinely inert (0.0 vs opt-off, 1.192e-7 vs reference, output shapes correct); (3) re-inference ordering sound (passes → opset → re-infer → from_parts); (4) `all`-path fails cleanly before numerics, tripwire non-tautological; (5) suite green debug+release, clippy clean, no new unsafe. Non-blocking note: tripwire `Err(other)` arm could be tightened. 🟢 **APPROVE** — no fix owner required.

## 2026-07-14T11:40:00Z — Review: ORT2 fusion-executable LayerNorm (chew-20)

- **Reviewed:** `squad/ort2-fusion-executable` @ `0f4811e` (author: Batty).
- **Scope:** Schema-aware LayerNorm fusion correctness + model-agnosticism.
- **Verdict:** 🟡 Approve with follow-ups.
- **Key confirmed:** Operand disambiguation order-independent/model-agnostic ✅; epsilon extraction robust ✅; parity real (0.0 vs off, 1.192e-7 vs ref) ✅; unit test asserts values not arity ✅.
- **Follow-ups raised:** F1 (opset-18 axis-as-input → decline), F2 (non-f32 epsilon → decline), F3 (hard-error vs decline-match), F4 (nit: assert byte-identity). Owner: Roy/Deckard/Leon.

## 2026-07-14 — chew-21: Review ORT2 DAG-aware LayerNorm (batty-14)
Reviewed matcher internals; authored adversarial probes (different_x, reversed, 9op_reversed). 🟢 APPROVE. A-CHEW-1 (pre-existing): Sub operand order not asserted — reversed Sub over-matches with sign-flip, but reproduced identically on base 9-op matcher. Recommend follow-up (Roy/Deckard/Leon; Batty locked out).

## 2026-07-14T14:35:00Z — chew-22: Reviewed deckard-14 (EPContext ep-api contract)

🟢 APPROVE. Model-agnostic dispatch verified (zero hardcoded vendor names); reject-duplicate semantics correct (non-transactional, callers treat duplicate as fatal); trait defaults object-safe; ep-cpu 105 + session tests green unchanged; `source_key` naming confirmed for thiserror 2.0.18; no new unsafe.

**Non-blocking advisories logged (for session integrator):**
- A-CHEW-1: `register` non-transactional; treat `DuplicateContextSource` as fatal.
- A-CHEW-2: Use `source_key` (not `source`) in `NoEpForContext`; map ONNX `source` attr → `claim`.

## 2026-07-14T15:00:00Z — chew-23: Review EPContext CONSUME path (batty-15)

Non-author review of `squad/ort2-epcontext-session` @ `d59edc5` (author: Batty). Focused on model-agnosticism, placement/execution bypass safety, test quality, unsafe/clippy.

- Model-agnostic confirmed: zero hardcoded vendor names in production code; QNN literal only in unclaimed fixture
- No CPU fall-through: all session construction paths call `load_ep_context_nodes` before `Executor::build`; EPContext nodes skipped by `is_ep_context_op`; no path to CPU kernel dispatch
- Test quality: non-UTF8 byte exactness, CARGO_TARGET_TMPDIR, MockCompiledEp genuinely invoked, error tests on concrete variants/fields
- No new unsafe; git diff --check clean; 34 tests pass; clippy exit 0

**Verdict: 🟡 Yellow — approve with test advisories:**
1. Add positive executor-bypass test with a claimed mock EP
2. Assert full EpContext struct fields, not only ctx.data

## 2026-07-14T16:20:00Z — CAPI DanglingEpContext regression fix (chew-24)
Fixed pre-existing non-exhaustive match in `onnx-runtime-capi`: mapped `SessionError::DanglingEpContext` → `OrtErrorCode::InvalidGraph`. Regression introduced when the EPContext consume-path merge added the new variant on main. Retained explicit exhaustive match for compile-time guard. Found via full cross-crate build gate. Commit d3f0c0a.

## 2026-07-14T18:55:00Z — chew-25: EPContext §21.4/§55.5 options wired end-to-end (author)

Authored `squad/ort2-epctx-options` (off `origin/main` `0fa025e`). Commit `3e8dbde` → cherry-picked to main as `c3d454c`.

- Implemented `SessionBuilder::parse_options` (one validating pass; three `ep.context_*` keys + existing optimization; unknown→`UnknownOption`, bad value→`InvalidOption`).
- Implemented `InferenceSession::export_ep_context` + `pub(crate)` `Executor::graph()`/`weights()` compiler-integration seam.
- Added capi surface: `OrtSessionOptions` opaque handle + `ort2_create/release_session_options` + `ort2_add_session_config_entry` + `ort2_create_session_with_options`.
- All gates green. EPContext §55 now complete end-to-end.
- Reviewers: gaff-14 🟢 (2 non-blocking advisories: A1 negative FFI tests, A2 handle-reuse by design), deckard-20 🟢 (no regressions, no advisories).

## 2026-07-14T14:50:00Z — chew-26: Review — AttentionFusion kernel/numerics/shape (batty-18)

Reviewed Batty's `com.microsoft::FusedAttention` kernel, softmax helper extraction, shape rule, and parity test (kernel/shape/numerics half; optimizer/matcher = Roy). Verified: `softmax_slices` is a pure visibility change (body unchanged); scale-before-mask-before-softmax order correct; both k_transposed branches compute Q·Kᵀ correctly; batched leading dims and mask broadcast correct; hand-derived unmasked pre-transposed test numerics match. k_transposed matcher↔kernel contract consistent. Shape rule mirrors k_transposed swap; 3 shape tests correct. Parity test ATOL=1e-6 (not loosened), non-tautological. Zero new unsafe; `#![forbid(unsafe_code)]` intact. ep-cpu 113 + shape 57 + session all green; clippy -D warnings clean. **Verdict: 🟢 GREEN — approve.** No revision required.

- 2026-07-14T19:05:00Z — Fixed clippy findings and corrected pytest count for the Python binding; changes included in merged commit `878559f`.
