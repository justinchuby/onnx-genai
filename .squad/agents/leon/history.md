# Leon — History

## 2026-07-12: Joined
Hired as Engine Dev (KV & runtime buffers) to add capacity alongside Batty as the runtime grew (9 crates, concurrent engine/KV workstreams). Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Key context: runtime owns the KV cache; use our own InferenceMetadata (`inference_metadata.yaml`) not ORT-GenAI `genai_config.json`; static-cache/GQA use device-resident buffers with present→past IoBinding aliasing; WebGPU decode needs GQA op + quantized (Q4 MatMulNBits) weights. Real-model exact-equality tests use `intra_op_threads=1`.

## 2026-07-13: Landed attention-sink SWA support
Extended sliding-window attention with StreamingLLM-style sink-token retention across metadata, engine decode state, runtime KV buffers, and paged-KV bookkeeping. Landed as commit `2371864`.


## 2026-07-13T18:30:00Z — Review/fix batch
- Owned Batty's reviewer-lockout fix for issue #14 and landed `458fb78`, adding multi-image over-count bailouts and `tokens_per_tile` guards.

## 2026-07-13T20:55:00Z — SharedKv generalization + robustness fix
- Reviewed Luv's 🟡 gap: malformed speculative metadata block aborted all model loading.
- Renamed all runtime identifiers from `Gemma4Assistant*` to `SharedKv*` across metadata, ORT, and engine (ProposalType, module, types, engine field, wire value). Canonical wire: `proposal_type: shared_kv`.
- Dropped back-compat `gemma4_assistant` alias entirely (pre-release): now degrades to `ProposalType::Unknown`.
- Fixed robustness gap: `resolve_shared_kv` degrades to `Unknown` when `shared_kv` is empty or any group has empty `target_layers` — malformed block no longer aborts load.
- Test `legacy_gemma4_assistant_proposal_type_degrades_to_unknown` added. All tests green; integration test token-identical to greedy. Commit: f101377.

## 2026-07-13T23:15:17Z — §38 K3: Wire KvCacheConnector into engine prefix-cache path

**Commit:** 2667b3d

New/modified files in `crates/onnx-genai-engine/src/`:
- `connector_bridge.rs` (new): `ConnectorBridge` — private current-thread Tokio runtime drives async trait from sync engine `std::thread`. Null bridge: no runtime, all methods early-return, fully inert.
- `engine.rs`: `connector` field placed BEFORE drop-order-critical `_environment` field; `build_connector_bridge`; STORE in `insert_cached_prefixes`; metrics LOOKUP in `prepare_session_prefix`; `last_connector_stats()`.
- `config.rs`: `KvConnectorBackend` (Null | LocalTiered) + `KvConnectorConfig` (model_id, chunk_size, store_priority, recompute_ms_per_token). Default is Null.
- `lib.rs`: re-exports.

**LIVE:** STORE after prefill; fetch-vs-recompute LOOKUP (`would_extend_tokens` metric only).
**DEFERRED — `TODO(K3-materialize)`:** fetch chunks → copy KV into paged cache → shorten prefill. Blocked on `KvTensorRef` needing real device-tensor handle. `prefix_cache_hit_len` is NOT modified — outputs stay correct.

Resolved rebase conflict on `_environment` drop-order field. 11 new connector tests; 102 engine lib tests passed; workspace builds + clippy clean.
Reviewed by Deckard: 🟢 SHIP. Advisory: prefix-independent hash = K4-materialize landmine (fixed by Zhora, commit ac12480).

### K4-materialize TODO (shared context for next implementor)
- Seam: `lookup_extension` in `connector_bridge.rs` returns `would_extend_tokens` but does not alter `prefix_cache_hit_len`.
- To complete: give `KvTensorRef`/`FetchedKv` a real device-tensor handle; store real KV bytes in `store`; in `lookup_extension` after a hit, copy fetched KV pages into the engine's paged cache and extend `prefix_cache_hit_len`.
- **Prerequisite invariant now met:** `KvCacheKey` equality ⟹ identical prefix through that chunk (Zhora fixed prefix-dependent hash, commit ac12480).

## 2026-07-13T23:50:16Z — §38 K4: Real KV byte materialization

**Commit:** 786e268

Replaced `KvTensorRef { size_bytes }` placeholder with `KvPayload` carrying real f32 KV bytes in head-major `[num_kv_heads, num_tokens, head_dim]` layout. Wired extract-on-store (`export_runner_kv` → `chunk_payload_from_exported` → `store_prefix_with`) and inject-on-lookup (`fetch_extension` → `past_kv_from_payloads` → `import_runner_kv`) gated by f32 + ZeroCopyRebind + fresh-session. Gold test `local_tiered_connector_fetch_reuse_is_token_identical` proves token-identical output to full recompute. 73 kv + 104 engine tests pass; clippy clean. §38 PROGRESS.md → ✅ Done (coordinator commit bc7ecb6).

Chew reviewed (read-only, 🟡 SHIP-with-advisories): layout correctness confirmed. Advisories routed to Pris (A1: multi-layer fixture) and Batty (A2: graceful recompute fallback).


## 2026-07-14T00-49-37Z — Gemma4 E2B real-run batch (W3 + Milestone B)

**W3 — Engine per-layer KV migration; shim removed** (commit 9db1a3c with Batty)
- `kv_bridge.rs`: `layer_configs_from_key_outputs` builds per-layer configs; `mirror_present_kv_to_pages` uses per-layer config
- `engine.rs`: both paged caches use `new_with_layer_tensor_configs`
- `speculative.rs`: `shared_kv_slices_from_materialized` reads per-group dims from `materialized.layers[target_layers.last()]`
- Deleted `MaterializedKv::num_kv_heads/head_dim` uniform shim; 2 new per-layer unit tests
- `cargo test -p onnx-genai-engine --lib` → 107 passed

**Milestone B — Real Gemma4 E2B shared-KV speculative on CUDA** (commit 10f82b3)
- Token-identical to greedy on real heterogeneous weights (hd256/hd512), H200 CUDA ✅
- Engine fixes: SWA paged windowed path; dtype-agnostic proposer (f16↔f32); `Value::from_f32_slice_as`; past KV injected in graph dtype; `cuda` feature passthrough; `tests/milestone_b_real.rs`
- **Chew review:** 🟢 SHIP
- **Perf note:** 0.53× (acceptance ~25%, `multi_token_accepts==0`) — projected_state hidden-space; correctness unaffected; speedup follow-up deferred

## 2026-07-14T02:37:00Z — Gemma4 speculative acceptance fix merged
- **Commit:** 8089a1f (with Pris) — Reviewed 🟡 Chew
- Root cause: engine fed `concat(prev_hidden, cur_hidden)` instead of `concat(embed(last_token), last_hidden)` to shared-KV proposer's `pre_projection`.
- Fix: `speculative.input_embedding` field + `LinearEmbedder`; proposer rewritten to correct contract.
- Result: acceptance 25% → 70.6%, multi_token_accepts 0 → 12/17, token-identity preserved.
- Speculative still 0.58x; speedup is separate follow-up (lower N_spec or lighter lm_head).
- Opened Sapper action item for durable Mobius export of `input_embedding.f32`.

## 2026-07-14T10:00:00Z — ORT2 dtype fail-close (leon-10)

- **Task:** Close silent-Float32 fallback gap identified by Holden's holden-11 finding. Make all `DataType::from_onnx(raw)->None` decode sites in `onnx-runtime-loader` fail closed.
- **Work:** Added `LoaderError::UnsupportedDataType { raw, context }` + `decode_dtype` helper. All 8 real-dtype sites now fail closed: initializer value, value-info TensorType/SparseTensorType, convert_tensor Constant/attribute, convert_type_proto Tensor/SparseTensor/Map key. Preserved intentional non-dtype defaults. Bumped proto with FLOAT4E2M1=23 (doc only). Added 2 new loader tests.
- **Result:** loader 15 + ir 40 tests green debug+release; bert_toy conformance PASS max_abs 1.192e-7; clippy clean.
- **Review:** Holden holden-12 → 🟢 GREEN. Merged to main (`a822a21` → `06a2423`).

## 2026-07-14T14:35:00Z — leon-11: LayerNorm operand-order guard merged

- Closed A-CHEW-1, A2, A3 from chew-21/deckard-13 reviews of batty-14 (DAG-aware matcher).
- Added operand-order guard to `layernorm_spec`: both centering `Sub(x, mean)` nodes must have input[0]==X, input[1]==mean (exactly-binary; else decline). Closes sign-flip over-match on 9-op and 10-op paths.
- Added `fuses_layernorm_split_chain` (isolated 10-op positive test) and `declines_layernorm_when_numerator_sub_reversed` (adversarial decline test).
- `DRIFT_ATOL = 1e-5` scoped to all-vs-opt-off only; conformance tolerance (2e-3) unchanged.
- Merged to main `a02d46e`. Gaff (gaff-10) 🟢 APPROVE.
