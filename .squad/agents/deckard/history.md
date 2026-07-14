# deckard — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12

- 2026-07-12T08:56:27-07:00 — Updated `.gitignore` with Rust and Python generated-artifact coverage; decision merged by Scribe.


## 2026-07-12T09:13:00-07:00 — ORT session, model-directory, and tokenizer contracts delivered
- Delivered real CPU `Environment`/`Session` load-run APIs, tensor `Value` helpers, graph metadata accessors, optional IoBinding, `ModelDirectory::load`, and `Tokenizer` encode/decode helpers.
- Key contract for next-batch wiring: `Session::run` accepts named `Value` inputs and returns outputs ordered by `output_names()` / `outputs()`; tokenizer decode skips special tokens and exposes optional EOS id.

## 2026-07-12T09:20:00-07:00 — ORT/tokenizer APIs enabled Phase 1 E2E
- The CPU `Session`, graph I/O metadata, tensor `Value` helpers, `ModelDirectory`, and `Tokenizer` APIs enabled Batty and Rachael to complete end-to-end greedy generation via the CLI tiny fixture.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Deckard delivered paged KV tensor storage (`new_with_tensor_config`, `append_token_kv`, `write_token_kv`, `materialize_sequence`), prefix cache page ownership/refcount lifecycle, CoW-safe writes, and ORT `1.27.0` runtime packaging so the server boots standalone without `DYLD_LIBRARY_PATH`.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Delivered Phase 3 KV work: hot/cold LRU page tiering plus opt-in `KvDType::Int8` symmetric per-page quantized KV that materializes back to f32 through existing cache APIs.

## 2026-07-12T12:02:00-07:00 — ORT chat-template and decode substrate delivered
Delivered pipeline schema/loader support, MiniJinja chat templates, multi-EOS discovery, fp16 Value helpers, zero-copy IoBinding DecodeSession, and StaticCacheDecodeSession with runtime-owned static KV buffers.

## 2026-07-12T13:14:00-07:00 — ORT hardening and batching notes merged
Deckard's GPU EP, batched static-cache decode, and ORT checksum notes are now in decisions. WebGPU/CoreML are selectable but slower than CPU on small Qwen decode; batched static decode matches unbatched but needs compaction.

## 2026-07-12T13:52:00-07:00 — §26 active-row compaction complete
- Deckard's `BatchedStaticCacheDecodeSession` active-row API is now part of the serving contract: `set_active_rows`, `compact`, `admit_row`, `deactivate_row`, `step_active`, and slot diagnostics back Sebastian/Rachael continuous batching.
- Future paged-attention and ORT work should keep logical row ids stable while allowing packed physical execution.

## 2026-07-12T14:28:00-07:00 — ORT comparison suite deterministic
- Deckard made all five ORT real-model comparison tests use `intra_op_threads=1` and a shared test `Environment`.
- The rare ORT FP-tie active-compaction flake was eliminated: 20/20 `onnx-genai-ort` and 5/5 full-workspace runs stayed clean.
- Future exact ORT comparisons should prefer single-threaded intra-op execution unless the assertion is tolerant to reduction-order differences.


### 2026-07-12T14:50:00-07:00
Published v0.1.0 release path is canonical: `.github/workflows/publish.yml` uses crates env, CARGO_REGISTRY_TOKEN, leaves-first order, idempotent skip-if-published checks with UA header. CI is live with fmt/build/test blocking and clippy non-blocking. Speculator discovery and MTP ORT execution are recorded.

## 2026-07-12T17:30:00-07:00 — Preprocess, tiling, compaction, and Mobius paged-cache logged
- `onnx-genai-preprocess`, metadata-driven/LLaVA-style image tiling, tolerant serialized ORT compaction tests, and Mobius paged-cache draft PR #395 are now recorded.
- Future onnx-genai paged attention should drive Mobius `key_pool`/`value_pool`, `block_table`, `slot_mapping`, and `nonpad_kv_seqlen` contracts.


## 2026-07-13T18:30:00Z — Review/fix batch
- Owned Sapper's reviewer-lockout follow-up and landed `8a0cf4b`, making thumbnail token/pixel order authoritative from tensor layout.

## 2026-07-13T20:55:00Z — §37/#9 model lifecycle read-only scope
- Scoped issue #9 model lifecycle architecture (read-only). Produced milestone plan M1–M4: M1 ModelHandle/Registry extraction, M2 real routing errors, M3 load/unload (RwLock), M4 status field. Saved to session files. Zhora independently implemented M1 in the same batch.

## 2026-07-13T23:15:17Z — §38 K3 code review

Reviewed `crates/onnx-genai-engine/src/{connector_bridge.rs, engine.rs, config.rs, lib.rs}` (Leon, commit 2667b3d).

Independently verified 7 high-risk items — all clean:
1. No nested-runtime panic: dedicated `std::thread` + private current-thread Tokio; `block_on` not called from existing Tokio context.
2. No refcount aliasing: engine's `prefix_cache` strictly separate from connector's `PrefixCache`/`PageTable`.
3. Correct chunk-boundary math: aligns with K1 `chunk_tokens` contract.
4. Inert Null default: byte-for-byte identical to prior behavior.
5. Honest deferral: `would_extend_tokens` metric only; `prefix_cache_hit_len` not altered; no faked hits.
6. Model-agnostic: no model-name branches.
7. Clean lock discipline: no locks held across `.await`.

Verdict: 🟢 **SHIP**.

**Advisory (K4-materialize):** chunk hash was prefix-INDEPENDENT — K4 materialization would copy wrong KV and silently corrupt outputs. Zhora subsequently fixed this (commit ac12480).

### Shared context for K4-materialize
- `TODO(K3-materialize)` in `connector_bridge.rs`: fetch chunks → copy KV into paged cache → shorten prefill.
- Blocked on `KvTensorRef` needing real device-tensor handle.
- **Prefix-dependent hash invariant now established** (Zhora, ac12480): `KvCacheKey` equality ⟹ identical prefix through that chunk. Safe to trust for copy-on-hit.
- Lock discipline invariant: std guard in `ConnectorBridge` (if any) must NEVER be held across `.await`.


## 2026-07-14T02:37:00Z — ort2-loader + loader-weights + Perfetto review
- **ort2-loader (7e0e367):** ONNX→IR pipeline with protox, graph_builder, mmap weights, shape_inference. 15 tests. Reviewed 🟡 Gaff.
- **loader-weights (dd5297d):** `load_model_with_weights` → `Arc<WeightStore>`, norm_axis off-by-one fix. 18 tests. Reviewed 🟡 Gaff.
- **Perfetto #13 review (8d1bf3d):** 🟢 SHIP — all 6 security/correctness criteria pass (gate parity, no data leak via `&'static str`, refactor safe, honest empty, OTLP deferred, model-agnostic).

## 2026-07-14T05:04:00Z — ORT2 loader const-fold-lite shape inference merged

- **squad/ort2-loader-shapeinfer** (b6f032e): Bounded partial evaluator (`ConstEnv`, `KnownVal`, `IntElem::Const|Sym`). Value-prop ops: Constant/Shape/Gather/Slice/Concat/Reshape/elementwise-int. bert_toy: 135→50 unresolved values; all residuals are scalar Constants; no structural op left shape-less. 27/27 tests (real model test not skipped). Reviewed 🟢 Gaff.
- Open: loader shape rules for `Attention` + `EmbedLayerNormalization` needed before full bert_toy session run (flagged by Roy session dynshape note).
- Gaff advisory A1: `Div` truncates vs floor for negative operands (no current impact).

## 2026-07-14T06:06:00Z — H-D1 Three-Layer Overflow Fix (cherry-picked to main)

- After Holden's 🔴 rejection of Batty's preliminary checked_numel work (Batty locked out of H-D1 artifact), Deckard authored the three-layer fix:
  - **Layer A** (`onnx-runtime-ir/src/dtype.rs`): `DataType::checked_storage_bytes(count) -> Option<usize>` — sub-byte div_ceil + `checked_mul`; `storage_bytes` reimplemented on top with `.expect`.
  - **Layer B** (`onnx-runtime-session/src/executor.rs`): `checked_storage_bytes` helper → `SessionError::ShapeOverflow`; both `ensure_buffer` and JIT alloc routed through it; `.max(1)` after checked multiply.
  - **Layer C** (`onnx-runtime-ep-cpu/src/strided.rs::view_in_bounds`): i128 address math with `checked_mul`/`checked_add`; overflow → `EpError::InvalidTensorView`.
- 4 new regression tests; all crate tests + bert_toy green; clippy clean; no new `unsafe`.
- Holden re-review (holden-7): **🟡 SHIP**. Cherry-picked to main: **dbf2d70**, **9dcdc04**, **f749012**.
