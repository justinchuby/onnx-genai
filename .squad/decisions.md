# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-12T13:52:00-07:00: Decisions archive rollover
**By:** Scribe
**What:** Archived prior canonical decisions to `decisions/archive/2026-07-12T13-52-00Z-decisions.md` because `decisions.md` reached 156336 bytes.
**Why:** Keep the hot decisions file small while retaining the full historical record under `.squad/decisions/archive/`.

---

### 2026-07-12T13:12:00-07:00: Engine decomposition and DecodeBackend seam
**By:** Batty
**What:** Decomposed `onnx-genai-engine`'s former god `engine.rs` into focused modules: `config.rs` for public request/result/config DTOs, `session.rs` for session/draft state, `decode.rs` for decode-path detection plus the internal `DecodeBackend` seam, `decode_loop.rs` for the shared token commit/generation loop, `kv_bridge.rs` for paged-KV mirroring/materialization/rewind, `processors.rs` for processor-chain/token-selection helpers, and `speculative.rs` for the draft/verify implementation. `engine.rs` is now the orchestrator and re-exports the same public API types as before.
**Why:** This makes DESIGN §26 batched serving and future decode backends additive: `DecodeBackend` exposes `current_len`, `max_context`, `decode`, `rewind`, and `reset`, and is implemented for the existing past/present `DecodeSession` and `StaticCacheDecodeSession` paths. Direct/session/priority/pipeline generation now use `decode_loop::{run_decode_loop, step_decode_loop}`; speculative commit uses the same `commit_selected_token` path after draft verification. Behavior was validated unchanged with `cargo check --workspace`, `cargo test --workspace`, and focused tiny fixtures: `tiny-llm` past/present tokens remain `[22, 22, 20]`, `tiny-llm-scatter` static-cache tokens remain `[23, 15, 28]`, and speculative tiny-llm output remains equal to baseline greedy.

---

### 2026-07-12T13:32:00-07:00: Batched static-cache active-row compaction
**By:** Deckard
**What:** `BatchedStaticCacheDecodeSession` now supports active-row compaction for STATIC-CACHE TensorScatter models. I chose compaction over row-view/gather because ORT IoBinding binds whole OrtValues; after packing live rows into physical slots `0..active_count`, we can bind prefix aliases shaped `[active_count, MAX_LEN, KV_DIM]` for KV inputs/outputs and run the model with a smaller batch. The public surface is additive: `set_active_rows(&[row_ids])`, `deactivate_row(row)`, `compact()`, `assign_row(row)`/`admit_row(row)`, `step_active`, `step_active_select`, plus slot/active diagnostics (`active_rows`, `active_batch_size`, `physical_slot`, `logical_row_for_physical_slot`, `inactive_compute_fraction`). Existing full-B `prefill`/`step`/`step_select` continue to work.
**Why:** Row-view/gather over arbitrary batch rows is not directly exposed by the ORT C IoBinding API we use, while prefix aliases are already compatible with our `Value::alias_with_shape` mechanism. Compaction copies each moved KV row (`MAX_LEN * KV_DIM` per KV tensor) only when active membership/order changes, then avoids per-step model compute for holes. Slot mapping is explicit: stable logical row ids map to physical slots; `compact()` frees inactive logical rows by removing their physical mapping; `admit_row()` assigns a freed slot, zeros that KV region, resets its cursor, and marks it active. Correctness was validated on `tests/fixtures/tiny-llm-scatter/`: B=4, deactivate 2 rows, active-only `[2,1,V]` logits/tokens for the remaining rows matched their pre-deactivation/unbatched traces, and admitting a replacement row into a freed slot matched its single-row trace. With 2 active of B=4, active-only compute is estimated at 50% of full-B for each decode step after the one-time compaction copy; after admitting a third row, estimated compute is 75% of full-B (25% saved).

---

### 2026-07-12T13:44:00-07:00: Server batch driver owns static-cache continuous batching
**By:** Rachael

**What:** `onnx-genai-server` now moves the Engine into one dedicated `onnx-genai-batch-driver` thread and talks to it through bounded Tokio channels instead of sharing generation through `Arc<Mutex<Engine>>`. For STATIC-CACHE models, stateless `/v1/chat/completions` requests submit `GenerateRequest`s to one `ContinuousBatchManager`; the driver drains incoming submissions, queues beyond `max_batch=4`, steps active rows, and routes token/final events back to each request's output channel. Streaming requests turn token events into SSE chunks; non-streaming requests collect the final result and keep the existing OpenAI response/tool parsing path. Past/present models, and X-Session-Id requests until the engine manager accepts SessionId submissions, run on the same driver thread through the existing per-request engine path to preserve persistent KV semantics.

**Why:** A single owner keeps the static-cache batched session's unsafe Send/Sync assumptions sound while letting concurrent HTTP requests share batched forward passes. Bounded command/output channels provide server-side back-pressure, while the manager's FIFO queue holds requests that exceed the active row capacity. Tool calls remain turn-based for OpenAI compatibility: a parsed `<tool_call>` completes the assistant turn with `finish_reason: "tool_calls"`; role=`tool` follow-ups are separate submissions rather than mid-token pauses.

**Validation:** `cargo check --workspace` and `cargo test --workspace` passed. Added `concurrent_static_cache_chat_completions_share_batched_driver`, which fires four concurrent `/v1/chat/completions` requests against `tests/fixtures/tiny-llm-scatter` and verifies each response matches an independent direct static-cache generation.

---

### 2026-07-12T13:24:00-07:00: Stage A batched static-cache decode loop
**By:** Sebastian

**What:** Added `Engine::generate_batched_static(Vec<GenerateRequest>) -> Result<Vec<GenerateResult>>` for fixed-batch STATIC-CACHE models. The path builds one `BatchedStaticCacheDecodeSession`, pre-fills request rows, then advances all active rows with one batched static-cache forward per decode step. Row logits are demuxed and each row applies its own processor chain, sampling options, stop/EOS/context/max-token checks, generated-token state, and result assembly.

**Correctness:** `crates/onnx-genai-engine/tests/batched_static_decode.rs` asserts batched rows on `tests/fixtures/tiny-llm-scatter/` exactly equal running each request alone through the existing single-sequence engine. This preserves Deckard's ORT-level batched==unbatched property through engine processors and stop handling.

**Row activity:** Finished rows call `BatchedStaticCacheDecodeSession::deactivate_row`, so inactive rows are no longer sampled, committed, or returned to the server-facing result path. Physical ORT compute is still fixed-B because the current ORT batch allocates/binds full `[B, MAX_LEN, KV_DIM]` buffers and has no row-view/prefix-batch compaction API. True compaction remains Stage B backend work: add logical-to-physical row mapping plus packed active row views or a cheap cache-row move so `(B-active)/B` compute is reclaimed without replaying live contexts.

**Measurement:** Ignored micro-measurement `batched_static_decode_reports_tiny_scatter_throughput` on `tiny-llm-scatter` with 8 rows × 8 generated tokens reported sequential `19851.12 tok/s` (`3.224ms`) vs batched `123562.14 tok/s` (`517.958µs`), `6.22x` aggregate speedup on this tiny CPU fixture. Both paths excluded model-load time; the fixture is small enough that this is a loop/ORT-call-count signal, not a real-model GPU throughput claim.

**Stage B/server contract:** The server/scheduler can batch compatible static-cache requests by passing a fixed set of arrived requests to this API and receiving results in input order. Continuous batching, mid-batch join/leave, PAUSED-on-tool, non-static past/present batching, and efficient compaction are intentionally deferred; Stage B should drive stable row handles and add packed active-row compaction before treating paused or finished rows as free capacity.

---

### 2026-07-12T13:32:00-07:00: Continuous batching manager for static-cache serving
**By:** Sebastian

**What:** Added an engine-level continuous-batch manager for STATIC-CACHE models. `Engine::run_continuous_batch(requests, max_batch)` is the synchronous Stage B driver; `Engine::continuous_batch_manager(max_batch)` exposes the lower-level `ContinuousBatchManager` with `submit`, `step`, `poll`, queue length, active length, and idle/work checks. The manager keeps FIFO pending requests plus fixed logical row slots, pre-fills an admitted request into a freed row, advances rows with pending logits one token per step, deactivates rows on EOS/stop/max/context, emits `ContinuousBatchEvent::{Token, Finished}`, and returns final driver results in input order.

**Why:** Stage A fixed-batch decode only runs the initial N requests to completion. Stage B needs row join/leave so an agent swarm can keep the batch full as requests finish at different times, while preserving each row's processor chain, sampling options, stops, constraints, context limits, and result correctness.

**Correctness:** `crates/onnx-genai-engine/tests/batched_static_decode.rs` now runs 16 `tiny-llm-scatter` requests through `max_batch=4` and asserts every result exactly equals running the same request individually through `Engine::generate`.

**Throughput:** Real fixture measurement (`cargo test -p onnx-genai-engine --test batched_static_decode -- --ignored --nocapture`) on `tests/fixtures/tiny-llm-scatter/`: sequential 18,232.10 tok/s (128 tokens, 7.020584ms), Stage A static batch16 174,873.42 tok/s (731.958µs), continuous max_batch=4 50,878.63 tok/s (2.515791ms, 2.79x vs sequential), continuous max_batch=8 59,989.44 tok/s (2.133709ms, 3.29x vs sequential). Deckard's active-row compaction API is present; the manager uses `step_active` when all remaining active rows need the next decode, so draining tails can skip inactive physical rows. Mid-admission prefill still uses fixed logical-row `step_select`, so this tiny CPU fixture is dominated by unbatched admission overhead and remains below Stage A's one-shot prefill/full-N batch.

**Stage C contract:** The server/runtime should own one `ContinuousBatchManager` per compatible static-cache model/config, call `submit` as requests arrive, call `step` from the runtime loop while work exists, and drain `poll` for per-token streaming plus final results. PAUSED-on-tool should retain the logical row and generated state but mark it non-runnable; on resume, reactivate/requeue that row before stepping. True async channels, HTTP concurrency, and tool pause/resume state transitions are Stage C, not part of this batch.

---

### 2026-07-12T13:52:00-07:00: §26-COMPLETE multi-agent serving and engine decomposition
**By:** Scribe
**What:** DESIGN §26 Multi-Agent Serving is complete across all three stages, with the engine refactor landed first. Batty decomposed `engine.rs` from 3286 lines to a focused 1275-line orchestrator across `config`, `session`, `decode`, `decode_loop`, `kv_bridge`, `processors`, and `speculative`, introduced the `DecodeBackend` trait, and moved direct/session/priority/pipeline/speculative paths onto the shared decode loop. Sebastian delivered Stage A `Engine::generate_batched_static` fixed-batch STATIC-CACHE decode with batched results matching individual generation and a measured 6.2x tiny-fixture throughput gain, then Stage B `ContinuousBatchManager` with `submit`, `step`, `poll`, FIFO admission, finish/evict, and active-row stepping. Deckard delivered ORT active-row compaction for `BatchedStaticCacheDecodeSession` through `set_active_rows`, `compact`, `admit_row`/`assign_row`, `deactivate_row`, `step_active`, and slot diagnostics, saving about 50% compute at 2 active rows of 4. Rachael delivered Stage C server integration: one engine driver thread owns the Engine and `ContinuousBatchManager`, bounded channels replace the global generation mutex, concurrent STATIC-CACHE HTTP requests share batched forward passes, streaming and non-streaming responses receive token/final events, and tool turns, request caps, CSPRNG sessions, and past/present fallback remain preserved.
**Why:** §26's serving contract is now explicit and complete: compatible static-cache requests batch through `DecodeBackend`/`BatchedStaticCacheDecodeSession` and `ContinuousBatchManager`, the server serializes unsafe runtime ownership in one driver while exposing concurrent HTTP behavior, and `cargo test --workspace` stayed green throughout. Remaining serving-related work moves to §27/§28 advanced speculative paths, paged attention, and extensibility traits rather than baseline batched serving.

---
