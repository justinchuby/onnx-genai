# Zhora — History

## 2026-07-12: Joined
Hired as Server Dev to add capacity alongside Rachael on the OpenAI-compatible HTTP surface. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: server is modularized (routes/driver/sse/types/state/session/metrics/image_input/audio_input); chat/completions/vision/audio/streaming/sessions/observability shipped; open API work includes `/v1/embeddings` (#7) and logprobs server formatting (#8). Handlers stay thin over the batched engine driver.

## 2026-07-13: Landed debug endpoints and queue-depth cap
Added `/v1/debug/config`, `/v1/debug/sessions`, `/v1/debug/kv`, and `/v1/debug/trace`; renamed the server admission boundary to configurable `max_queue_depth` (`--max-queue-depth` / `ONNX_GENAI_MAX_QUEUE_DEPTH`). Landed as commit `afcf094`.

## 2026-07-13T20:55:00Z — Model lifecycle M1 + /v1/embeddings wiring
- Implemented issue #9 model lifecycle Milestone 1: extracted ModelHandle + ModelRegistry from AppState (pure refactor). ModelHandle bundles all per-model fields; ModelRegistry wraps HashMap<String, Arc<ModelHandle>> with resolve/insert/ids/default_id. Zero behavior change — single-model fallback preserved. 52 tests green. Commit: 9ab4fa9.
- Wired POST /v1/embeddings through DriverCommand::Embed (oneshot-reply) to engine embed_with_options. Mean pooling via EmbeddingOptions::default(). Pipeline models return clear error. Double tokenization intentional. dimensions truncation deferred.
- M2 (real routing), M3 (load/unload), M4 (status field), LRU eviction deferred.


## 2026-07-20T00:00:00Z — Model lifecycle M2 + M3 complete

### M2 — Multi-Model Config, Startup Load, Routing, Deterministic Default
- Commit: b5934c6 | Issue: #9
- Added `src/models_config.rs`: TOML/JSON config parsing + directory scan.
- CLI: `--models-config`, `--models-dir`, `--model` (mutually exclusive via `ArgGroup`).
- `AppState::load_from_specs`: eager-loads all specs at startup.
- `resolve_model` in `routes.rs`: empty `model` → deterministic default; unknown name → 404.
- Removed M1 silent fallback from `ModelRegistry::resolve`.
- Deterministic insertion-order fields (`order`, `default_id`) in `ModelRegistry`.
- 55 lib + 20 HTTP integration tests; clippy clean.
- Review: 🟡 Chew — found embeddings routing inconsistency (fixed by Rachael, commit 561ee1a).

### M3 — Runtime Load/Unload, LRU Eviction, Lazy Load, Admin Endpoints
- Commit: a5106f5 | Issue: #9
- `ModelRegistry` → cloneable shared handle (`Arc<RwLock<RegistryInner>>`).
- Lock discipline: `std::sync::RwLock` only; never held across `spawn_blocking`/`.await`.
- Per-id async load guards prevent double-build of the same lazy model.
- Unified `build_handle` shared by startup and runtime load paths.
- LRU eviction: `max_loaded_models` cap; prefers non-default victims; never drops below 1.
- Admin endpoints (`GET/POST/DELETE /v1/admin/models/*`) gated by `--enable-admin-endpoints`.
- Default model lazily reloaded after unload on next empty-model request.
- 66 lib + 20 HTTP integration tests; all M1/M2 tests pass; clippy clean.
- Review: 🟢 Deckard — all concurrency invariants confirmed.

### Status
§37 / Issue #9 model lifecycle epic: COMPLETE (M1 + M2 + M3).
Locked out of embeddings follow-up per reviewer protocol; Rachael delivered fix.
Next: §34 router epic (R1/R2/R3) has kicked off.

## 2026-07-13T23:15:17Z — §38 KV Connector K1 + K2 + cpu-load fix + prefix-hash fix

### K1 — Pluggable KvCacheConnector abstraction
- Added `KvCacheConnector` async trait + `NullConnector` in `crates/onnx-genai-kv/src/connector.rs`.
- Types: `KvCacheKey`, `KvCacheLocation`, `KvStoreEntry`, `FetchedKv`, `ConnectorCapabilities`, `CachePriority`, `CompressionFormat`, `ConnectorHealth`, `ConnectorError`.
- Chunking: `chunk_tokens`, `TokenChunk`, `hash_tokens` (FNV-1a 64-bit, process-independent). `DEFAULT_CHUNK_SIZE = 256`.
- Deps added: `async-trait` to workspace + kv crate; `tokio` to kv dev-deps.
- 55 tests green; clippy clean.

### K2 — LocalTieredConnector
- Implemented `LocalTieredConnector` in `crates/onnx-genai-kv/src/local_tiered.rs`.
- Bridges existing `PageTable` (hot/cold tiering) + `PrefixCache` (content-addressed index). Single `std::sync::Mutex` lock; no std guard held across `.await`.
- Priority-aware eviction (Opportunistic < Session < SystemPrompt); pinning; Fp8 codec.
- `PrefixCache::remove` primitive added to `src/prefix_cache.rs`.
- 11 new tests (66 total); clippy clean.
- Reviewed by Chew (🟡): `cpu_load_ms_per_page` unscaled defect found.

### cpu_load_ms_per_page fix (commit 30ee870)
- `locate()` now: `estimated_load_ms = pages_needing_upload * cpu_load_ms_per_page`.
- Added `cpu_load_ms_scales_by_configured_rate` test.

### Prefix-dependent chunk hash fix (commit ac12480)
- `chunk_tokens` threads cumulative FNV-1a state across chunk boundaries.
- **Invariant:** equal `KvCacheKey` ⟹ identical token sequence from position 0 through end of chunk.
- Preserves genuine prefix sharing; defuses K4-materialize landmine flagged by Deckard.
- Tests: `chunk_hash_is_prefix_dependent`, `chunk_hash_shared_prefix_still_collides`, hardcoded-value stability guard.

### K4-materialize TODO (shared context)
- `TODO(K3-materialize)` in `connector_bridge.rs`: fetch hit chunks → copy KV into paged cache → shorten prefill.
- Blocked on `KvTensorRef` needing a real device-tensor handle (currently size-only placeholder).
- The prefix-dependent-hash invariant is now in place. K4 implementor can trust `KvCacheKey` equality as proof of identical prefix.


## 2026-07-14T02:37:00Z — Perfetto trace export #13 merged
- **Commit:** 8d1bf3d — Reviewed 🟢 Deckard
- `GET /v1/debug/trace/perfetto` → Chrome Trace Event Format document, gated same as sibling debug routes.
- No data leak (`&'static str` stage names only), honest empty case, OTLP deferred.
- Metrics ENDPOINTS extended to 14 entries.
