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

