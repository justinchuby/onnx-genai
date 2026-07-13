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

