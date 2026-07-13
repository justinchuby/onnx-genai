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
