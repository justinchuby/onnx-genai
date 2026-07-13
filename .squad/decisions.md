# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-20T00:00:00Z: Decisions archive rollover
**By:** Scribe
**What:** Archived all 2026-07-12 entries (68 KB) to `decisions/archive/2026-07-20T00-00-00Z-decisions-pre-0713.md`. decisions.md exceeded the 50 KB threshold; entries older than 7 days (relative to 2026-07-20) were moved to archive. Recent 2026-07-13+ entries are retained below.
**Why:** Keep the hot decisions file lean per Scribe charter (>=50KB → archive entries >7 days).

---

### 2026-07-13: Sliding Window Attention — attention-sink (StreamingLLM) support + documented ORT boundary
**By:** Leon
**What:** Extended SWA (DESIGN §40) with attention-sink token retention. Metadata gains `model.attention.sink_tokens: Option<usize>` (§40.9). The paged KV cache gains `PagedKvCache::apply_sliding_window_with_sinks(seq, window, sink_tokens)` — pinning the leading sink pages and evicting only the middle window pages (sink pinning is page-granular; `sink_tokens==0` delegates to the existing contiguous `apply_sliding_window`). The engine threads `sink_tokens` from metadata through `detect_model_decode_path` → `ModelDecodePath::PastPresent` → `DecodeState`, and `apply_window_after_step`/`rewind_windowed` keep `[0, sink) ∪ [window_start, len)` token-exactly in the runtime KV buffer that feeds ORT.
**Why:** Contiguous single-window SWA was already implemented end-to-end; the real §40 gap was attention sinks (§40.4), which are correctness-critical — dropping the first tokens under a naive window corrupts the attention distribution. The runtime KV buffer (exact) and the paged cache (page-granular bookkeeping for rewind/prefix) are decoupled, so buffer sinks can be token-exact while paged sinks stay page-aligned without conflict.
**Boundary deferred to Mobius/ORT:** (1) hybrid per-layer attention patterns (§40.3) need per-layer KV buffers and per-layer graph masks — not expressible with a single shared decode buffer today; (2) feeding **discontinuous** `position_ids` (§40.8) into a contiguous ORT graph after window/sink eviction requires model/EP support (rotating cache or `local_window_size` contract). `detect_model_decode_path` already refuses SWA + static-cache and SWA + shared-buffer combos, and `load_materialized_past` refuses windowed/sink materialize into a contiguous graph, so the runtime never silently produces wrong outputs — it declines the unsupported path instead.

---

### 2026-07-13: Add server debug introspection and queue-depth admission control
**By:** Zhora
**What:** Added `/v1/debug/config`, `/v1/debug/sessions`, `/v1/debug/kv`, and `/v1/debug/trace`; renamed the server's active-plus-queued generation cap to `max_queue_depth` (`--max-queue-depth` / `ONNX_GENAI_MAX_QUEUE_DEPTH`).
**Why:** The debug endpoints expose current server configuration, sessions, existing cache/admission metrics, and tracing status without creating new engine subsystems. The explicit queue-depth setting documents and configures the existing semaphore-based HTTP 429 admission boundary.

---

### 2026-07-13T18:30:00Z: Reviewer rejection lockout and debug hardening
**By:** Scribe
**What:** Reviewer-rejection lockout was applied correctly in the review/fix batch: Zhora was locked out after Gaff rejected unauthenticated `/v1/debug/*`; Sapper was locked out after Gaff flagged thumbnail token/pixel ordering; Batty was locked out after Luv rejected vision token-expansion multi-image accounting and `tokens_per_tile` guards. Follow-up fixes were owned by Rachael (`2e67806`), Deckard (`8a0cf4b`), and Leon (`458fb78`) respectively.
**Security:** `/v1/debug/*` is now hardened default-off and redacts session identifiers, closing the rejected unauthenticated session-ID disclosure path.
**Why:** Rejections must move remediation to a different owner, and debug surfaces must fail closed unless explicitly enabled.

---

### 2026-07-13T20:14:53Z: Runtime must stay model-agnostic — no hardcoded model logic or names
**By:** squad-coordinator (requested by Justin Chu)
**What:** The runtime must not hardcode model-specific logic or model names. Config and metadata must be generic and generalizable — behavior driven by structural/architectural properties (I/O signatures, layer-type patterns, hidden sizes, shared-KV descriptors) read from metadata, NOT named model branches (`if model == "gemma4"`). Test fixtures may retain model-derived filenames; only runtime logic and config keys must be generic.
**Why:** This was recorded when generalizing the Gemma4 assistant proposer and applies permanently to all future runtime development.

---

### 2026-07-13: Gemma4 `*-assistant` shared-KV speculative decoding — runtime vertical slice + wire schema
**By:** Batty
**What:** Added first-class runtime support for the Gemma4 `*-assistant` "shared-KV proposer" (neither MTP nor EAGLE-3). The assistant owns no KV cache; it reads slices of the target model's paged KV cache through `shared_kv.*` inputs, carries its own internal `lm_head` (emits full draft `logits`), and threads a `projected_state` output into the next step's `inputs_embeds`. Delivered compiling + tested across metadata, ORT, config, proposer, KV-slice sharing, engine load, and a synthetic ONNX fixture proving speculative == plain greedy. **Note:** the initial wire `proposal_type` was `gemma4_assistant`; this was subsequently generalized to `shared_kv` by Leon (see decision below). The ONNX graph I/O contract (inputs_embeds, shared_kv.*, logits, projected_state), detection criteria, and KV-slice sharing assumptions are unchanged by that rename.
**Wire schema (canonical names post-rename):** `proposal_type: shared_kv`; `model`, `backbone_hidden_size`, `vocab_size`, `projected_state_output`, `logits_output`, `shared_kv` groups (each with `name` and `target_layers`). The parser degrades a malformed block to `Unknown` rather than hard-failing load.
**Why:** This delivers the speculative proposer for Gemma4-style shared-KV draft architectures. Commit: f6b4f6d (initial), superseded by f101377 (rename).

---

### 2026-07-13: Selectable KV cache storage dtype (design #15)
**By:** Batty
**What:** Threaded a selectable KV-cache storage dtype (`KvDType`) from config to the paged cache mirror, making `fp8_e4m3fn`, `fp8_e5m2`, and `int8` storage reachable at runtime. Knobs: `EngineConfig::kv_cache_dtype: KvDType` (defaults to `KvDType::F32`); server CLI `--kv-cache-dtype <f32|int8|fp8_e4m3fn|fp8_e5m2>` / env `ONNX_GENAI_KV_CACHE_DTYPE`. Draft model KV cache is hardcoded to `KvDType::F32` (ephemeral/tiny — quantisation yields negligible savings). No quantisation logic was added to the engine; `PagedKvCache` handles encode/decode internally via `PageTensorConfig.dtype`. All four dtypes are accepted end-to-end; `cargo test --workspace` and clippy `-D warnings` pass.
**Why:** Enables memory-efficient KV storage for production deployments without changing default behavior.

---

### 2026-07-13: Generalize `gemma4_assistant` proposer to architecture-based `SharedKvProposer`
**By:** Leon
**What:** Per the model-agnostic runtime policy, renamed all runtime identifiers from `Gemma4Assistant*` to `SharedKv*`. Canonical wire value: `proposal_type: shared_kv` (also accepts `shared-kv` kebab alias). Deprecated aliases `gemma4_assistant`/`gemma4-assistant` removed entirely — they now degrade to `ProposalType::Unknown` (not a load failure). Back-compat alias was dropped as pre-release. Rename spans metadata (ProposalType::SharedKv, SharedKvProposerSpec, resolve_shared_kv), ORT (module shared_kv_proposer.rs, SharedKvProposerSession/Signature/StepOutput), and engine (SharedKvProposerConfig, SharedKvProposerModel, SpeculativeMode::SharedKv, SharedKvProposer). Test fixture filenames left as-is (`scripts/build_tiny_gemma4_assistant.py`, `tests/fixtures/tiny-gemma4-assistant/`); runtime type references inside tests updated to new names.
**Robustness fix (from Luv's 🟡 review):** `resolve_shared_kv` now degrades to `SpeculatorProposerStatus::Unknown` when `shared_kv` is empty OR any group has empty `target_layers`. Previously a malformed speculative block resolved as "supported", then hard-errored in validation and aborted the entire model load — breaking even non-speculative generation. Now malformed metadata falls back gracefully to non-speculative.
**Validation:** `cargo build` clean; metadata lib tests green (incl. `legacy_gemma4_assistant_proposal_type_degrades_to_unknown`); `gemma4_assistant_full` integration test token-identical to greedy; clippy `-D warnings` clean. Commit: f101377.
**Why:** Enforces the model-agnostic runtime policy. The proposer is an architecture (shared-KV draft borrowing target KV slices), not a named model.

---

### 2026-07-13: Gemma4 multimodal export is a major runtime effort, not a metadata patch
**By:** Sapper / Roy
**What:** Exporting Gemma4 E2B/12B vision through Mobius and smoke-testing in onnx-genai is deferred as a large architecture item. Requires: (1) multi-tensor rank-3 pre-patchified vision ingestion + `pixel_position_ids` + f16 pixel dtype (server currently forces one Float32 rank-4 `pixel_values`); (2) embedding→decoder orchestration because Gemma4 feeds `inputs_embeds` from a separate embedding model, not token IDs; (3) Mobius PR #398 (`--runtime onnx-genai`) extended to emit pipeline topology, tokenizer copy, and `pipeline.vision`. Concrete Gemma4 values: placeholder id `258880` (`<|image|>`), `tokens_per_tile=280` (E2B). Continue autonomous backlog on self-contained items instead.
**Why:** Prevent a fruitless "just add two vision metadata fields" attempt that would produce a package that cannot load.

---

### 2026-07-13: Gemma4 Mobius exports are not yet consumable by onnx-genai VLM runtime
**By:** Sapper
**What:** Treat Gemma4 E2B/12B end-to-end as blocked on a broader Mobius/runtime adapter, not only the new `pipeline.vision` metadata fields. PR #398 emits decoder-only metadata; Gemma4 vision graphs require rank-3 pre-patchified `pixel_values` plus `pixel_position_ids`, and their separate embedding component is not supported by the current generic pipeline loop. The server currently accepts one Float32 rank-4 `pixel_values` tensor and the engine decoder loop requires token IDs (plus routed extras). Adding image soft-token ID 258880 and 280 soft tokens per image/tile alone cannot make the exported four-model package load or run.
**Why:** Ensures future agents don't attempt a partial Gemma4 VLM wiring that will silently produce wrong outputs.

---

### 2026-07-13: Mobius onnx-genai emitter updated to emit canonical `proposal_type: shared_kv`
**By:** Sapper
**What:** Updated the Mobius onnx-genai emitter to emit `proposal_type: shared_kv` (replacing `gemma4_assistant`) in inference metadata for shared-KV speculative draft models. Tests 17/17 (schema/metadata unit tests) + 41/41 (exporter integration tests) passing. Pushed as commit 498ecf0 on branch `feat/gemma4-assistant-onnx-genai` in the onnxruntime/mobius repo.
**Why:** Aligns the Mobius emitter with Leon's runtime rename (shared_kv canonical); old wire value `gemma4_assistant` now degrades to Unknown in the runtime.

---

### 2026-07-13: Wire /v1/embeddings — server-crate seam
**By:** Zhora
**What:** Wired `POST /v1/embeddings` through the engine driver to the engine's existing `embed_with_options` API. Design choices: (1) `DriverCommand::Embed` follows the oneshot-reply pattern (like `session_token_count`), not the streaming DriverEvent channel. (2) `EmbeddingOptions::default()` (mean pooling, no normalization) — the OpenAI embeddings API does not expose pooling strategy. (3) Pipeline models return a clear error rather than panicking. (4) Double tokenization (validate + execute) is intentional — avoids refactoring validation. (5) Removed `ApiError::not_implemented` (dead after this change). (6) `dimensions` truncation not implemented — field validated (>0) but vector not truncated; add when a model with adjustable-dimension embeddings is supported.
**Why:** Completes the embeddings server surface. The engine already supports `embed_with_options`; this wires the HTTP seam.

---

### 2026-07-13: Model lifecycle M1 — ModelHandle + ModelRegistry (pure refactor)
**By:** Zhora
**Issue:** #9 (model lifecycle), Milestone 1
**What:** Extracted all per-model fields from `AppState` into `ModelHandle` (`id`, `engine`, `tokenizer`, `chat_template`, `model_max_context`, `fim_config`, `pipeline`, `vision_input`, `audio_input`, `last_request_at`). `ModelRegistry` wraps `HashMap<String, Arc<ModelHandle>>` with `insert`, `resolve` (updates `last_request_at`; falls back to `default_id()` for empty/unknown requests — preserving single-model behavior), `ids()`, and `default_id()`. `AppState` now holds `registry: ModelRegistry` + `sessions` + `config` + `started_at`. Zero behavior change: all 52 tests (32 unit + 20 integration) pass. Internal helpers refactored to accept `(state: AppState, handle: Arc<ModelHandle>, …)`.
**Deferred:** M2 (real routing errors for unknown models), M3 (load/unload with RwLock), M4 (status field), LRU eviction (last_request_at tracked but not acted on). Commit: 9ab4fa9.
**Why:** Lays the clean separation needed for multi-model routing without changing behavior.

---

### 2026-07-13: SWA/attention-sink hardening nits — rewind_to sink fix, first-activation asserts, draft rationale
**By:** Batty (nits from Chew's review)
**What:** Three targeted fixes to the SWA/sink implementation: (1) **First-activation `debug_assert!`** — added two debug_assert calls at the moment the sink region first becomes active: `page_count >= sink_pages` (sink boundary does not overlap unallocated storage) and `keep_from >= sink_len_target` (window start does not regress into sink). Release behavior unchanged. (2) **`rewind_to` sink symmetry fix** — was incorrectly rejecting positions in the pinned sink prefix `[0, sink_len)` with `KvError::PositionEvicted`. Guard changed from `position < retained_start` to `position < retained_start && (sink == 0 || position >= sink)`. Post-rewind: if `position < sink`, resets `sink_len = 0` and `retained_start = 0` (plain contiguous prefix, no gap). New test: `rewind_into_sink_discards_window_and_resets_gap_bookkeeping`. (3) **Draft `sink_tokens=0` documented** — added multi-line comment explaining why the draft decode path is constructed with `sink_tokens=0` and `sliding_window=None` (sink is no-op without a window; draft architectures have independent KV constraints; correct fix path is to load draft's own inference_metadata). Commit: 4e51d59.
**Why:** The rewind_to bug made valid rewind targets inside the sink prefix incorrectly fail; the asserts and rationale comment prevent silent regressions.

---

### 2026-07-13: M2 Multi-Model Config, Startup Load, Request Routing, and Deterministic Default
**By:** Zhora (Rust server engineer)
**Issue:** #9 (Milestone 2)
**Commit:** b5934c6
**What:** Added `src/models_config.rs` with TOML/JSON multi-model config (`--models-config`), directory-scan startup (`--models-dir`), and single-model `--model` kept backward-compatible. All three modes are mutually exclusive via `clap::ArgGroup`. `AppState::load_from_specs` iterates specs and eagerly loads all of them (M3 handles true lazy loading). Request routing uses `resolve_model` in all four inference handlers: empty/whitespace `model` → deterministic default; named unknown → 404. `ModelRegistry::resolve` no longer silently falls back to default on unknown names. Registry insertion order fields (`order: Vec<String>`, `default_id: Option<String>`) make `/v1/models` listing and default selection deterministic across ≥2 models. 55 lib + 20 HTTP integration tests pass; clippy clean.
**Why:** M1 could only load one model and had non-deterministic HashMap iteration. M2 enables multi-model servers with a strictly typed routing contract and reproducible default model selection.

---

### 2026-07-13: M3 Runtime Load/Unload, LRU Eviction, Lazy Loading, and Admin Endpoints
**By:** Zhora (Rust server engineer)
**Issue:** #9 (Milestone 3)
**Commit:** a5106f5
**What:** `ModelRegistry` is now a cloneable shared handle (`Arc<RwLock<RegistryInner>>`). Lock discipline: `std::sync::RwLock` only held for short synchronous critical sections; never across `spawn_blocking`/`.await`. Per-id load guards (`load_locks`) prevent concurrent double-builds of the same lazy model. `build_handle` is the single construction path shared by eager startup and runtime lazy load. `available` holds all configured specs; `models` holds currently loaded handles. `resolve_model` is now async: on miss, checks `available` and calls `load(id).await` (lazy load) or returns 404. LRU eviction: `max_loaded_models: Option<usize>` cap; `pick_lru_victim` prefers non-default; never drops below 1 model. Admin endpoints (`GET/POST/DELETE /v1/admin/models/*`) gated by `enable_admin_endpoints` flag (default false). Unload keeps spec in `available` for reload; default model is lazily reloaded on next empty-model request if unloaded. 66 lib + 20 HTTP integration tests pass; clippy clean.
**Why:** M2 loaded everything eagerly and had no runtime model management. M3 enables memory-bounded servers that load on demand, evict stale models, and allow operator-driven load/unload without restart.

---

### 2026-07-20: Remove empty-model reject guard from /v1/embeddings (M2 follow-up)
**By:** Rachael (Rust engineer)
**Commit:** 561ee1a
**What:** Removed the unconditional `if request.model.trim().is_empty() { return Err(...bad_request...) }` guard from `validate_embedding_request` in `routes.rs`. Added two tests: `empty_model_field_falls_back_to_default_on_embeddings` (empty `model` → 200 via registry default) and `unknown_model_returns_404_on_embeddings_endpoint` (unknown name → 404). Zhora was locked out per reviewer protocol (Chew's 🟡 review on M2 identified the inconsistency).
**Why:** The routing contract for all inference endpoints is: empty `model` → deterministic default; unknown named model → 404. The embeddings guard short-circuited after `resolve_model` had already succeeded, making `/v1/embeddings` the only endpoint that rejected a valid empty-model request with a spurious 400. Removing the guard restores parity across all four inference endpoints.

---

### 2026-07-20: §34 Router Epic Kickoff — §37 Model Lifecycle Complete
**By:** Scribe (on behalf of coordinator orchestration batch)
**What:** §37 / Issue #9 model lifecycle (M1+M2+M3) is fully complete: M1 (ModelHandle + ModelRegistry refactor), M2 (multi-model config + routing + deterministic default, commit b5934c6), M3 (runtime load/unload + LRU eviction + lazy load + admin endpoints, commit a5106f5), and embeddings routing fix (commit 561ee1a, Rachael). §34 router epic (R1/R2/R3) has kicked off. Pris and Roy are working R1/R2/R3 in parallel with Zhora/Rachael available for follow-up.
**Why:** Captures the handoff point between §37 completion and §34 commencement for audit purposes.

---

### 2026-07-20: §34.8 Node Status Contract (R1) — GET /v1/status on onnx-genai-server
**By:** Pris (Rust engineer)
**Commit:** 050259f
**What:** Added `GET /v1/status` node-status endpoint to `onnx-genai-server`, returning the §34.8 contract as typed serde structs `NodeStatus` + `SessionStatus`. Replaced the previous ad-hoc `/v1/status` shape with the §34.8 node-status shape (pre-release, no back-compat alias). New `--node-id` CLI arg with env fallback `ONNX_GENAI_NODE_ID`; default resolved by `default_node_id()` → hostname else CSPRNG `node-<hex>`. Removed now-dead `MetricsSnapshot::total_requests` and `AppState::started_at`. Real fields: `node_id`, `healthy`, `queue_depth`, `active_sessions`, `sessions[].id` (redacted). Placeholder zeros: `kv_usage`, `kv_pages_*`, `paused_sessions`, `tokens_per_second`, `batch_utilization`, per-session `priority`/`kv_pages`/`state`, `prefix_hashes` — all documented with `// not yet tracked` comments. Files changed: `crates/onnx-genai-server/src/{routes.rs, state.rs, main.rs, lib.rs, metrics.rs, tests.rs}`.
**Why:** The router (§34) polls `/v1/status` every 1-2s. Placeholders use documented zeros so downstream consumers can distinguish "0" from "unmeasured" once engine exposes KV/throughput getters. `node_id` is decoupled from any model so a multi-model node reports one node identity.
**Follow-ups:** Wire real `kv_pages_*`/`kv_usage` once engine exposes paged-KV stats; add rolling `tokens_per_second` + `batch_utilization`; track per-session priority/state and prefix hashes.

---

### 2026-07-20: §34.8 Node Status — f32 alignment fix (R1 follow-up)
**By:** Pris (Rust engineer)
**Commit:** 74314e8
**What:** Changed `NodeStatus.kv_usage` and `NodeStatus.batch_utilization` from `f64` to `f32` in `crates/onnx-genai-server/src/routes.rs`. The cluster router's mirror struct (`crates/onnx-genai-router/src/node.rs` `NodeStatus`) deserializes both fields as `f32` and uses `f32` for overload-threshold comparisons (`NodeState.kv_usage: f32`, `accepts_affinity(overload_threshold: f32)`). `tokens_per_second` is `f64` in both — left unchanged.
**Why:** The server was serializing `kv_usage`/`batch_utilization` as `f64`, causing a silent serde downcast on the router side. Per the §34.8 contract both sides must agree on canonical width.

---

### 2026-07-20: onnx-genai-router core (R2) — pure session-aware routing
**By:** Roy (Rust engineer)
**Commit:** 1f58099
**What:** Created new standalone crate `crates/onnx-genai-router/` implementing the model-agnostic, session-aware router core from DESIGN.md §34. Added to root `[workspace] members`. Pure logic + config + polling data model + full unit tests; proxy/HTTP server deferred to R3. Modules: `config.rs` (`RouterConfig` YAML: listen, nodes[], routing, health, session_map; `RoutingPolicy` enum); `node.rs` (`NodeId`, `NodeState`, `NodeStatus` deserialize mirror, `NodeStatusFetcher` trait as R3 async seam); `router.rs` (`Router`, `RouteRequest`, `RoutingDecision`, affinity→prefix→least-loaded per §34.5); `session_map.rs` (affinity table + optional JSON persistence + `MigrationEvent`/`MigrationReason`); `prefix_map.rs` (prefix-hash→node map + FNV-1a 64-bit `hash_system_prompt`). Key decisions: (1) `/v1/status` contract mirrored not shared — router must NOT depend on server/engine/ORT crates; (2) `route`/`least_loaded_node` return `Option<NodeId>`, never panic; (3) model-agnostic opaque ids; (4) custom serde for `RoutingPolicy` YAML shape; (5) FNV-1a 64-bit prefix hash for cross-process stability.
**Why:** Generic round-robin LBs are harmful for LLM inference (KV affinity, load asymmetry, prefix sharing — §34.1). This crate provides the smart routing layer kept small, pure, and fully unit-tested so R3 can add transport without touching decision logic.
**Validation:** `cargo test -p onnx-genai-router` → 36 passed; clippy `-D warnings` → clean; workspace build ok.

---

### 2026-07-20: onnx-genai-router R3 — networking/runtime (proxy, API, poller, metrics)
**By:** Roy (Rust engineer)
**Commit:** ee8e464
**What:** Turned `crates/onnx-genai-router/` into a runnable model-agnostic reverse-proxy binary on top of the R2 pure core. Added modules: `node_poller.rs`, `proxy.rs`, `api.rs`, `metrics.rs`, `state.rs`, `main.rs`, `tests/proxy_integration.rs`. Extended `router.rs` additively (draining set + `rebalance()` + `record_session_affinity()`). Key decisions: (1) hyper-util legacy client for transparent SSE streaming, no reqwest; (2) request bodies buffered ≤16 MiB for field extraction, response bodies streamed; (3) model-agnostic extraction (`session_id`/`session` for affinity, first system-role content for prefix hash); (4) `std::sync::Mutex` (not tokio) behind Arc, guard always dropped before `.await`; (5) `draining: HashSet<NodeId>` for §34.7 drain semantic; (6) rebalance reassigns affinity for unhealthy/draining/overloaded sessions only (lazy re-prefill); (7) hand-rolled Prometheus text, no prometheus crate; (8) lean deps: axum, hyper, hyper-util, http-body-util, bytes, clap, anyhow, tracing-subscriber. Endpoints: `GET /router/status`, `GET /router/sessions`, `GET /router/metrics`, `POST /router/drain/{node_id}`, `POST /router/rebalance`; all else proxied. CLI: `--config <router.yaml>` / `ONNX_GENAI_ROUTER_CONFIG`, optional `--listen`.
**Why:** R2 shipped the pure decision core; R3 gives it a transport for actual reverse-proxy inference traffic with node health tracking, operational controls, and metrics — without model-specific behavior.
**Validation:** `cargo test -p onnx-genai-router` → 67 passed; clippy `-D warnings` → clean; manual smoke: binary boots, poller demotes unreachable node, all API endpoints behave correctly.

---

### 2026-07-20: onnx-genai-router — `affinity_weight` as continuous scoring bonus in Weighted policy
**By:** Roy (Rust engineer)
**Commit:** 54e5363
**What:** Implemented `affinity_weight` in the `Weighted` routing policy as a continuous scoring bonus rather than a binary gate. Removed binary `Step::Affinity` gate from `Weighted`'s `policy_steps` (only `Step::Prefix` remains as pre-step). Added `weighted_fallback_node(&self, request)` scoring all healthy, non-draining candidates via `weighted_node_score`: `kv_usage × kv_weight + normalized_queue × queue_weight − bonus`, where `bonus = affinity_weight` if the node is the session's affinity target AND `kv_usage < overload_threshold`, else 0. Removed misleading comment `"affinity_weight is applied in the affinity step, not here"` from `load_score()`. `least_loaded_node()`/`load_score()` unchanged (serve `rebalance()`).
**Why:** DESIGN.md §34.5 defines Weighted as `affinity × 0.5 + kv_usage × 0.3 + queue_depth × 0.2` — affinity is a scored term, not a binary filter. The previous binary gate made Weighted identical to AffinityThenLoad for session-carrying requests. The fix makes Weighted a genuine smooth blend: affinity node wins more often when `affinity_weight` is high but loses to less-loaded nodes when the load delta is large enough.

---

### 2026-07-20: onnx-genai-router R3 — concurrency hardening (4 fixes)
**By:** Roy (Rust engineer)
**Commit:** a36cbbd
**What:** Four hardening items from Deckard's 🟡 concurrency review of R3 (commit 54e5363). All 66 pre-existing + 7 new tests green; clippy clean.
1. **Concurrent poller** [Medium]: `node_poller::poll_once` now snapshots `(id, address)` under one short lock, releases it, drives all `fetch` futures concurrently via `futures_util::future::join_all`, then applies each result under a short lock. Added `futures-util` dep (std+async-await). Mutex never held across `.await`. Chose `join_all` over `JoinSet` because `fetch` futures borrow `&F` (not `'static`).
2. **Miss-on-unknown-id** [Low]: When `update_node` returns `false` (node self-reports id not in config), `apply_poll_result` now calls `record_node_miss` so the node accrues misses and demotes to unhealthy. Previously it stayed in healthy/zero-load state, attracting least-loaded routing.
3. **Response cap** [Low]: `proxy::capture_session_affinity` no longer does uncapped `body.collect()`. If upstream advertises `Content-Length > MAX_REQUEST_BODY` (16 MiB), response is streamed through untouched (affinity capture skipped, request NOT failed). Otherwise body buffered with `axum::body::to_bytes(_, cap)`.
4. **Rebalance overload guard** [Low]: Added `Router::least_loaded_node_below_threshold` (healthy && !draining && `kv_usage < overload_threshold`); used in `rebalance()` instead of `least_loaded_node`. When all healthy non-draining nodes are at/above threshold, rebalance leaves sessions in place — no thrash migration + re-prefill cost.
**Why:** Serial poller degraded health refresh for healthy nodes when cluster was degraded; unknown-id nodes attracted traffic incorrectly; uncapped response buffering was a DoS vector; rebalance could thrash between saturated nodes with no benefit.

---

### 2026-07-20T00:00:00Z: §38 Distributed KV Connector — K1 abstraction foundation
**By:** Zhora
**What:** Added a pluggable external-KV-cache connector abstraction in
`crates/onnx-genai-kv` as new module `src/connector.rs` (re-exported from
`lib.rs`). This is the K1 milestone: pure abstraction + `NullConnector` +
tests. No engine/scheduler wiring (K3) and no concrete backends
(LocalTiered/LMCache, K2).

Surface:
- Async trait `KvCacheConnector: Send + Sync` (`#[async_trait]`): `lookup`,
  `lookup_batch` (default impl loops over `lookup`; overridable for RTT
  amortization), `store`, `fetch`, `prefetch` (sync/non-blocking), `pin`,
  `unpin`, `evict`, `health`, `capabilities`. Object-safe (test asserts
  `Arc<dyn KvCacheConnector>`) so K3 can hold it dynamically.
- Types (§38.4): `KvCacheKey{model_id,layer_range,chunk_hash,chunk_index,
  num_tokens}` (derives Clone/Hash/Eq/PartialEq/Debug); `KvCacheLocation`
  (LocalGpu/LocalCpu/LocalDisk/Remote/NotFound); `KvStoreEntry`; `FetchedKv`;
  `ConnectorCapabilities`; `CachePriority`; `CompressionFormat`;
  `ConnectorHealth` (enum Healthy / Degraded{detail} / Unavailable{detail}).
- Error type `ConnectorError` (NotFound/Backend/Unsupported) +
  `ConnectorResult<T>` alias — kept separate from `KvError` since connector
  failures are transport/IO-dominated.
- Chunking (§38.8): `chunk_tokens(&[TokenId], chunk_size) -> Vec<TokenChunk>`,
  `TokenChunk{index,tokens,hash}` with `to_key(model_id, layer_range)`, and
  `DEFAULT_CHUNK_SIZE = 256`.
- `NullConnector`: lookup/lookup_batch → NotFound; store/pin/unpin/evict → Ok
  no-op; fetch → Err(NotFound); prefetch → no-op; health → Healthy;
  capabilities → {distributed:false, prefetch:false, pinnable:false,
  max_chunk_tokens: usize::MAX, compression:[None]}.
- Reused existing crate types: `PageId`, `Device`, `TokenId`, `thiserror`.

**Stable chunk-hash choice:** `hash_tokens` = **FNV-1a (64-bit)** over the
little-endian bytes of each `u32` token id, with fixed offset-basis/prime
constants inlined. Chosen over Rust's `DefaultHasher` (SipHash is
per-process-randomly-seeded and would break cross-node prefix sharing). Hash
is per-chunk (depends only on that chunk's tokens, never neighbours), enabling
chunk-granular prefix sharing. A test pins hardcoded expected hashes so any
future hasher change is caught.

**KvTensorRef placeholder:** `KvStoreEntry.kv_data`/tensor data uses a minimal
opaque `KvTensorRef{size_bytes}` placeholder (doc-commented) so K1 stays free
of ORT/engine deps. K2 will flesh it into a real device-memory descriptor /
tensor view without changing the trait surface.

**Model-agnostic:** `model_id` is an opaque namespacing string; nothing
branches on model names. `chunk_size`, compression, and capabilities are
params/config, never hardcoded per model.

**Deferred:** engine/scheduler wiring → K3; concrete backends
(LocalTiered/LMCache) → K2.

**Deps:** added `async-trait` to workspace deps and to onnx-genai-kv
`[dependencies]`; added `tokio` (workspace) to onnx-genai-kv
`[dev-dependencies]` for `#[tokio::test]`.

**Validation:** `cargo test -p onnx-genai-kv --lib` → 55 passed (20 new
connector tests, existing kv tests green). `cargo clippy -p onnx-genai-kv
--all-targets -- -D warnings` → clean.

**Why:** Establishes the connector seam so distributed/tiered KV and cross-node
prefix sharing can be added in K2/K3 without reshaping the abstraction, while
keeping the default (single-node, no offload) behaviour a trivial no-op.

---

### 2026-07-20T00:00:00Z: §38 Distributed KV Connector — K2 LocalTieredConnector

**By:** Zhora

**What:** Implemented the default, ships-by-default `LocalTieredConnector`
(DESIGN §38.5.1) plus `LocalTieredConfig` and `DiskTierConfig` in the new module
`crates/onnx-genai-kv/src/local_tiered.rs` (re-exported from `lib.rs`). Added a
small `PrefixCache::remove(tokens) -> Vec<PageId>` primitive to
`src/prefix_cache.rs`. This is the concrete single-node GPU→CPU tiered backend
behind the K1 `KvCacheConnector` trait. No scheduler wiring (that is K3).

Files:
- `src/local_tiered.rs` (new): connector, config, disk-tier config, 11 tests.
- `src/prefix_cache.rs`: added `remove` (detach a specific prefix node, return
  its pages; does NOT touch page-table refcounts — the owner does).
- `src/lib.rs`: `pub mod local_tiered;` + re-export
  `DiskTierConfig, LocalTieredConfig, LocalTieredConnector`.

**How the bridge works (reuse, not reinvent):**
- `PageTable` owns physical pages and hot/cold tiering. `Device::Gpu(0)` = hot,
  `Device::Cpu` = cold. Each stored chunk holds exactly ONE page-table ref,
  released on `evict`. When the hot pool fills, `PageTable::allocate` auto-
  offloads the LRU hot page to CPU; `fetch`/`prefetch` promote back to GPU.
- `PrefixCache` is the content-addressed prefix index: every chunk is registered
  under a deterministic token path derived from its `KvCacheKey`
  (`key_path` = FNV(model_id) ++ layer_range ++ chunk_index ++ chunk_hash ++
  num_tokens). Chunks with identical content resolve to the same pages
  (chunk-granular prefix sharing via `PrefixCache::lookup`/`remove`). I used
  `insert`/`lookup`/`remove` (no refcount side effects) so retention stays owned
  by the connector+PageTable — clean, single-ref accounting.
- `chunks: HashMap<KvCacheKey, ChunkEntry>` is the authoritative O(1) resolver
  used by lookup/fetch; records page_ids, priority, ttl, size_bytes, and path.

**Location / estimate model:** a chunk of `num_tokens` occupies
`ceil(num_tokens / page_size)` pages. `lookup` reports `LocalGpu{page_ids}` when
ALL pages are `Device::Gpu(_)`, else `LocalCpu{estimated_load_ms, size_bytes}`
where `estimated_load_ms = pages_needing_upload * cpu_load_ms_per_page`
(default 1 ms/page, per design "~1ms for typical page") and
`size_bytes = pages * bytes_per_page` (halved for Fp8). Honest linear estimates;
`fetch` reports the *actual* elapsed `transfer_time`. `LocalDisk`/`Remote` are
never fabricated.

**Compression wiring:** `LocalTieredConfig.compression`; `new()` rejects
`CacheGen`/`Zstd` with `ConnectorError::Unsupported` (only `None` and `Fp8`
implemented). Fp8 halves `size_bytes` and the store path exercises the real
codec via `fp8::encode_f32`/`decode_f32` (E4M3Fn). Real tensor-byte compression
lands with the K2+ tensor handle (`KvTensorRef` is still a size-only
placeholder). `capabilities().compression == [None, Fp8]`.

**Priority + pinning:** hard eviction (total `max_cached_pages` budget) is
priority-aware: Opportunistic evicted before Session before SystemPrompt, ties
break on oldest `stored_at`; pinned chunks are skipped. Tier demotion (hot→cold
under `hot_capacity` pressure) is LRU and also skips pinned pages and never
demotes a page of higher priority than the incoming chunk. `pin`/`unpin` toggle
an internal pinned set; `evict` frees pages + drops the mapping (and the
prefix-cache node).

**Disk-tier decision:** NOT implemented this milestone. `disk_backend` defaults
to `None`; `DiskTierConfig{path}` lets a disk tier be *configured*, and `health`
returns `Degraded` when configured-but-unavailable (dir missing), `Healthy`
otherwise. A real mmap/direct-I/O spill + `LocalDisk` locations are a documented
future extension. No fake `LocalDisk` results.

**Prefetch approach:** non-blocking `fn` — `try_lock` the interior mutex; if
acquired, best-effort synchronous promote of the chunk's pages to GPU (respecting
pins/room); if the lock is busy, the hint is silently dropped. No background
thread, so the crate stays runtime-free (no tokio dependency added); a future
revision may queue hints for an offload task.

**Lock discipline:** all interior state (`PageTable`, `PrefixCache`, `chunks`,
`pinned`) lives behind ONE `std::sync::Mutex`. Every critical section is
synchronous and short; the guard is always dropped before the async method
returns — a std guard is NEVER held across `.await` (structurally ensured; there
are no awaits inside the locked regions). Clippy `-D warnings` clean.

**Tests (11 new, all in `#[cfg(test)]`):**
`store_then_lookup_reports_local_gpu_and_unknown_is_not_found`,
`overflowing_hot_capacity_offloads_to_cpu_tier`,
`fetch_promotes_cpu_resident_chunk_and_missing_errors`,
`identical_content_shares_the_same_pages_via_prefix_cache`,
`pin_prevents_eviction_and_unpin_allows_it_and_evict_drops_mapping`,
`opportunistic_is_evicted_before_session_and_system_prompt`,
`compression_none_and_fp8_round_trip`, `unsupported_compression_is_rejected`,
`capabilities_and_health_are_reported`, `lookup_batch_resolves_all_keys`,
`prefetch_is_non_blocking_and_promotes_when_possible`.

**Why:** delivers the concrete default local backend for §38 by bridging the
existing kv-crate facilities (PageTable tiering + PrefixCache prefix sharing +
fp8 codec) behind the K1 trait, model-agnostic and fully config-driven, without
reinventing tiering or prefix matching, and without engine coupling.

**Validation:** `cargo test -p onnx-genai-kv --lib` → 66 passed (55 K1/prior +
11 new); `cargo clippy -p onnx-genai-kv --all-targets -- -D warnings` → clean.

---

### 2026-07-13: Wire KvCacheConnector into the engine's prefix-cache-hit path (K3)

**By:** Leon

**What:**
Extended `onnx-genai-engine` so an optional, model-agnostic `KvCacheConnector`
participates in cross-session prefix reuse, without disturbing the existing
in-process `lookup_shared`/`release_shared` refcount path.

- **Config (generic):** new `EngineConfig.kv_connector: KvConnectorConfig`
  (`config.rs`). It expresses *which backend* (`KvConnectorBackend::Null` |
  `LocalTiered(LocalTieredConfig)`) plus backend-neutral knobs (`model_id`,
  `chunk_size`, `store_priority`, `recompute_ms_per_token`). Default is `Null`,
  which reproduces today's behavior exactly. No per-model branches; `model_id`
  is an opaque key namespace, defaulting to the model directory name.
- **Bridge:** new `connector_bridge.rs` with `ConnectorBridge`. A `null()`
  bridge is fully inert (no runtime, every method early-returns). An active
  bridge owns a private current-thread Tokio runtime and `block_on`s the async
  trait (the shipped backends complete synchronously, never yielding).
- **STORE (LIVE):** `Engine::insert_cached_prefixes` now also calls
  `ConnectorBridge::store_prefix(&state.tokens, kv_token_count)`, chunking the
  resident KV at `chunk_size` and pushing each *complete* chunk with the
  configured `CachePriority`.
- **LOOKUP (LIVE, reporting only):** `Engine::prepare_session_prefix` computes
  the in-process hit as before, then calls
  `ConnectorBridge::lookup_extension(prompt_tokens, in_process_hit)`. That
  chunks the tokens beyond the in-process boundary, builds `KvCacheKey`s via the
  connector's chunk-hash helper, calls `lookup_batch`, and walks the contiguous
  run of resident chunks — counting a chunk only while `estimated_load_ms <=
  num_tokens * recompute_ms_per_token` (fetch-vs-recompute signal, DESIGN §38).
- **Observability:** `Engine::last_connector_stats() -> ConnectorStats`
  (lookups / chunk_hits / would_extend_tokens / fetched_tokens / stores).

**LIVE end-to-end:** generic connector config + construction; store-after-prefill;
connector lookup with contiguous-hit + fetch-vs-recompute decision; stats.
Default `Null` path is byte-for-byte unchanged (verified by test).

**DEFERRED — `TODO(K3-materialize)`:** actually fetching hit chunks and copying
their KV into the engine's paged KV cache so prefill can be *shortened*. Reason:
the K1 `KvTensorRef`/`FetchedKv` carry only a `size_bytes` placeholder plus page
ids in the connector's *own* PageTable — there is no real device-tensor handle
to copy from, and the connector's `store` never received real KV bytes. Wiring
materialization would require giving those types a real tensor handle. Until
then `lookup_extension` returns `would_extend_tokens` for metrics but does NOT
alter `prefix_cache_hit_len`, so generation output stays exactly correct — we
never claim a hit we cannot serve.

**Why:**
- Model-agnostic config was a hard user directive; an enum of backends + neutral
  knobs generalizes to remote/distributed connectors later with no engine change.
- The engine's `prefix_cache` (refcounted shared pages) and the connector's own
  `PrefixCache`/`PageTable` are kept strictly separate — no aliasing, no
  double-free risk, per the K1/K2 foundation notes.
- Honest deferral over faked materialization: silently substituting placeholder
  KV would corrupt outputs. The store + lookup halves are fully real and tested,
  leaving a precise, isolated seam for the fetch→materialize follow-up.

**Validation:** `cargo build -p onnx-genai-engine` clean; `cargo test -p
onnx-genai-engine --lib` = 102 passed / 0 failed / 1 ignored (11 new connector
tests); `cargo clippy -p onnx-genai-engine --lib` and `--tests` clean with
`-D warnings`; `cargo test -p onnx-genai-kv --lib` = 67 passed; full workspace
builds. Commit: 2667b3d.

---

### 2026-07-13: Definition of "pages_needing_upload" in estimated_load_ms

**By:** Zhora

**What:** `pages_needing_upload` in `locate()` is defined as the count of pages in a chunk that are currently NOT on a GPU device (i.e., residing on the CPU tier). These are the pages that the K3 scheduler would need to upload before the chunk is hot. Pages already on GPU are excluded — they are zero-cost to access and need no upload.

**Why:** The module doc (line ~29) states the estimate is `pages_needing_upload * cpu_load_ms_per_page`. The code already counted CPU-resident (non-GPU) pages as `on_cpu`; the bug was that this count was used directly as milliseconds (equivalent to assuming `cpu_load_ms_per_page = 1.0`) rather than being multiplied by the configured rate. The fix multiplies `on_cpu` by `cpu_load_ms_per_page`, which is consistent with the doc's intent and gives correct scaling for any configured value. Commit: 30ee870.

---

### 2026-07-13: KvCacheKey chunk hash is now prefix-dependent (cumulative FNV state)

**By:** Zhora

**What:**
Reworked the §38 chunk-hashing scheme so `KvCacheKey.chunk_hash` encodes the
full preceding token context instead of just that chunk's tokens.

- Chose the **folded-chunk_hash** design (NOT a new `prefix_hash` field).
  `KvCacheKey`'s shape is unchanged; `chunk_hash` now carries a *cumulative*
  hash. Implementation: `chunk_tokens` threads a single FNV-1a state across
  chunk boundaries, snapshotting it at each boundary, so
  `chunks[i].hash == hash_tokens(&all_tokens[0..=end_of_chunk_i])`. Continuing
  the running FNV state is exactly "the cumulative hash of all preceding chunks'
  tokens ++ this chunk's tokens" and is cleaner than re-folding the previous
  digest's bytes (no extra step, and it stays trivially equal to the pure hash
  of the covered prefix, which makes the guard test self-evident).
- Pure `hash_tokens` is untouched (still the position-independent FNV-1a of the
  slice passed in) and is still used by `LocalTieredConnector::key_path` to hash
  the opaque `model_id`. Its hardcoded-value guard is unchanged.
- Added a hardcoded-value guard for the *rolling* chunk hashes
  (`chunk_tokens_cumulative_hash_is_stable_against_hardcoded_values`) to keep
  determinism + process-independence locked, and updated the misleading
  "depends only on that chunk's tokens — never on surrounding chunks" doc on
  `hash_tokens`/`KvCacheKey`/`TokenChunk` to describe prefix-dependence.

**Invariant established:** equal `KvCacheKey` ⟹ identical token sequence from
position 0 through the end of that chunk (for a fixed `model_id` + `layer_range`).

**Why:**
Deckard's K3 review surfaced a latent correctness landmine: a chunk's real KV is
prefix-dependent (causal attention — chunk N depends on tokens 0..N), but the K1
key hashed only the chunk window. Two sequences sharing chunk N's tokens but
differing earlier produced an identical key yet different real KV. Harmless in
K3 (metrics only), but K4 materialization would copy fetched KV into the paged
cache on the strength of this key and **silently corrupt output**. The cumulative
scheme fixes this while preserving genuine cross-node prefix sharing: two
requests with the same prefix through chunk N still get the same key for chunk N
(shared system prompts / common prefixes still dedupe); only genuinely-different
prefixes now correctly diverge. No backward-compat shim (pre-release); no new
deps; trait object-safety unchanged; engine `key_for`/`chunk_tokens` API surface
unchanged so `connector_bridge.rs` picks up prefix-dependence transparently.

Proven by: `chunk_hash_is_prefix_dependent` (shared window + different earlier
tokens ⟹ different key) and `chunk_hash_shared_prefix_still_collides`
(identical prefix ⟹ identical keys through the shared boundary, diverging after).
Commit: ac12480.

