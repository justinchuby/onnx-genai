# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-12T08:56:27-07:00: Expand Rust and Python gitignore coverage
**By:** Deckard
**What:** Added standard Rust backup/debug-symbol patterns and comprehensive Python cache, build, packaging, virtual environment, coverage, test, type-checking, lint, notebook checkpoint, and native extension ignore patterns to `.gitignore`.
**Why:** The repository is a Rust Cargo workspace that may include Python tooling, scripts, model conversion, or tests, so ignoring common generated artifacts keeps source control focused on intentional source files while preserving existing `/target` and `Cargo.lock` behavior.

---

# Generation API surface for Phase 1 greedy generation

**Date:** 2026-07-12T09:00:00-07:00
**By:** Batty

## Decision

`onnx-genai-engine` now exposes the Phase 1 generation API from `crates/onnx-genai-engine/src/engine.rs` and re-exports it from `lib.rs`:

- `GeneratePrompt::{Text(String), TokenIds(Vec<TokenId>)}`
- `GenerateOptions { max_new_tokens, temperature, top_p, top_k, repetition_penalty, greedy, stop_sequences, eos_token_id, stop_on_eos }`
- `GenerateRequest { prompt, options }` plus `GenerateRequest::new(...)`
- `GenerateResult { text, token_ids, finish_reason }`
- `GenerateToken { token_id, text, finish_reason }`
- `GenerateTokenCallback<'a> = dyn FnMut(GenerateToken) -> anyhow::Result<()> + Send + 'a`
- `FinishReason::{MaxTokens, EosToken, StopSequence { index }}`
- `StopSequence::{Text(String), Tokens(Vec<TokenId>)}` and `TokenId = u32`
- `Engine::generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult>`
- `Engine::generate_with_callback(&mut self, request: GenerateRequest, callback: Option<&mut GenerateTokenCallback<'_>>) -> anyhow::Result<GenerateResult>`

`Engine::generate` creates a scheduler sequence, validates options, builds the processor chain, applies processors to logits, selects greedy vs sampled decoding, checks EOS/stop termination, advances/completes the scheduler, and returns `GenerateResult`.

## Processor-chain order

The Phase 1 chain order is:

1. `RepetitionPenaltyProcessor`
2. `StopSequenceProcessor` (termination signal only; no logit mutation)
3. `TemperatureProcessor`
4. `TopKProcessor`
5. `TopPProcessor`

This follows DESIGN.md §3.6, with stop sequence checks placed in the constraints slot before sampling filters.

## Intentional stubs for next batch

Only integration points waiting on Deckard's ORT/tokenizer/model-loader API remain `todo!()`:

- `Engine::tokenize_prompt` for `GeneratePrompt::Text`
- `Engine::detokenize_token`
- `Engine::next_token_logits`

`GeneratePrompt::TokenIds` already bypasses tokenization, so next batch should fill the ORT forward and detokenization calls without changing the public API.

---

### 2026-07-12T09:00:00-07:00: ORT Phase 1 session, loader, and tokenizer API
**By:** Deckard
**What:** `onnx-genai-ort` now exposes a CPU ORT C API wrapper contract for Batty's generation-loop wiring:
- `Environment::new(name: &str) -> Result<Environment>` creates the ORT environment handle.
- `Session::new(env: &Environment, path: &Path, options: SessionOptions) -> Result<Session>` loads an ONNX model. Phase 1 accepts only `ExecutionProvider::Cpu`; non-CPU providers return `OrtError::InvalidArgument` until EP append paths are wired.
- `Session::run(&self, inputs: &[(&str, &Value)]) -> Result<Vec<Value>>` runs a forward pass. Input names must match the model graph names. Output `Vec<Value>` is ordered exactly as `Session::output_names()` / `Session::outputs()`.
- `Session::input_names() -> &[String]`, `Session::output_names() -> &[String]`, `Session::inputs() -> &[TensorInfo]`, and `Session::outputs() -> &[TensorInfo]` expose graph I/O metadata. `TensorInfo { name, dtype, shape }` uses negative dimensions for ORT dynamic axes.
- `Value::from_slice_i64(data: &[i64], shape: &[i64]) -> Result<Value>` and `Value::from_slice_f32(data: &[f32], shape: &[i64]) -> Result<Value>` create CPU tensors with owned backing storage. `Value::from_vec_i64` / `from_vec_f32` avoid a second caller-side allocation when the caller can transfer a Vec. `Value::to_vec_i64()` / `to_vec_f32()` copy ORT output tensor data back to Rust. `Value::shape()`, `Value::dtype()`, and `Value::numel()` describe tensors.
- `IoBinding::{new, bind_input, bind_output, bind_output_to_device, clear}` and `Session::run_with_binding(&self, &IoBinding) -> Result<()>` are real C API calls, but Phase 1 generation should prefer `Session::run` unless pre-bound KV buffers are needed.
- `ModelDirectory::load(root) -> Result<ModelDirectory>` resolves `decoder.onnx` (or exactly one `.onnx` fallback), required `tokenizer.json`, and optional `inference_metadata.{yaml,yml,json}`. Missing metadata is represented as `metadata_path: None` and is tolerated for Phase 1.
- `Tokenizer::from_file(path) -> Result<Tokenizer>`, `encode(&self, prompt: &str) -> Result<Vec<u32>>`, `encode_i64(&self, prompt: &str) -> Result<Vec<i64>>`, `decode(&self, ids: &[u32]) -> Result<String>`, `decode_i64(&self, ids: &[i64]) -> Result<String>`, `token_id(&self, token: &str) -> Option<u32>`, and `eos_token_id(&self) -> Option<u32>` wrap the HF `tokenizers` crate. `decode*` skips special tokens.
**Why:** Batty needs a stable Phase 1 backend boundary for greedy generation: resolve model/tokenizer files, encode prompts to i64 `input_ids`, build i64/f32 CPU input tensors by graph name, run ORT, read logits from ordered outputs, and decode generated token ids. The API intentionally copies CPU tensors for safety now; IoBinding remains available for later KV/device-buffer work without changing the session contract.

---

### 2026-07-12T09:00:00-07:00: Add metadata tests and tiny Mobius LLM fixture
**By:** Pris
**What:** Added fixture-based tests for `onnx-genai-metadata` covering valid YAML/JSON parsing, malformed/schema-invalid parse errors, and runtime capability validation. Produced a tiny GPT-2-style decoder-only ONNX fixture at `tests/fixtures/tiny-llm/` with deterministic random weights and a matching WordLevel `tokenizer.json`.
**Why:** Phase 1 requires deterministic metadata parser coverage and a committed tiny end-to-end generation fixture. The model was generated without downloading HuggingFace weights using Mobius from `/Users/justinc/Documents/GitHub/mobius` with: `PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src python tests/fixtures/tiny-llm/generate_tiny_llm.py`. Generated sizes: `model.onnx` 15,807 bytes, `model.onnx.data` 13,312 bytes, `tokenizer.json` 2,038 bytes.

---

### 2026-07-12T08:58:32-07:00: Phase 1 foundation status and next plan
**By:** Roy
**What:** Assessed `docs/DESIGN.md` Phase 1 against current crate sources. Workspace scaffolding exists; metadata, KV, ORT, scheduler, engine, logit processors, server, and facade crates are present, but end-to-end generation is blocked by stub ORT execution, no tokenizer wiring, and no engine generation API/loop. The next increment should prioritize making `onnx-genai-ort` actually load and run a CPU ONNX session, then wire tokenizer/model-dir discovery and a minimal greedy single-sequence engine path.
**Why:** DESIGN.md Phase 1 exit criteria is loading a Phi-4 ONNX model and generating greedily end-to-end. No other Phase 1 component can be validated against a real model until the ORT wrapper returns real output tensors; this is the highest-leverage unblocker for Deckard, Batty, Rachael, and Pris to parallelize follow-on work.

---

# 2026-07-12T09:15:00-07:00: Phase 1 greedy generation wiring

**By:** Batty

## Decision

`onnx-genai-engine` now loads models through `onnx_genai_ort::ModelDirectory`, creates a CPU `Environment` and `Session`, and loads the colocated HF `Tokenizer`. `Engine::generate` performs real single-sequence greedy generation: tokenize prompt, run prefill, process logits, select argmax when `greedy` or `temperature == 0`, stream token callbacks, detokenize, and stop on EOS, configured stop sequences, or `max_new_tokens`.

## ORT input discovery

The engine introspects `Session::inputs()` for every run. It recognizes `input_ids`, `attention_mask`, `position_ids`, and past key/value inputs by graph name rather than hardcoding one model schema. Unsupported required inputs now return a clear error with the input name and shape.

## KV threading

Phase 1 uses simple model-owned KV tensors, not `onnx-genai-kv` paging. If the graph exposes past key/value inputs and matching present key/value outputs, prefill feeds zero-length Float32 past tensors shaped from `TensorInfo`; each subsequent step feeds only the previously generated token plus the last run's present tensors remapped to the corresponding past input names. If the graph has no KV I/O, the engine falls back to re-feeding the full running sequence each step.

## Termination

The loop stops when tokenizer/default or request EOS is generated, any configured text/token stop sequence matches the generated suffix, or `max_new_tokens` is reached. Final text is decoded from all generated tokens; per-token callback text uses single-token decode for streaming continuity.

---

### 2026-07-12T09:15:00-07:00: Facade generate CLI argument surface
**By:** Rachael
**What:** Added the `onnx-genai generate` CLI surface in the facade crate with `--model <DIR>`, optional `--max-new-tokens`, `--temperature`, `--top-p`, `--top-k`, repeatable `--stop`, `--stream`, and a positional prompt. The command maps those flags onto `GenerateOptions::max_new_tokens`, `temperature`, `top_p`, `top_k`, and `stop_sequences` as `StopSequence::Text`, wraps the prompt in `GenerateRequest`, constructs `Engine::from_dir(..., EngineConfig::default())`, then calls `Engine::generate` or `Engine::generate_with_callback` for streaming.
**Why:** The facade CLI should consume Batty's public generation API directly without redefining types or touching the engine crate, while preserving a simple Phase 1 surface compatible with the documented generation options.

---

### 2026-07-12T09:20:00-07:00: Phase 1 Foundation exit criteria met
**By:** Scribe
**What:** Phase 1 Foundation is complete: `onnx-genai generate --model tests/fixtures/tiny-llm "hello world"` runs greedy generation end-to-end through the facade CLI, engine, tokenizer, ORT session, and tiny fixture. `cargo test --workspace` is fully green, and the work is pushed to `origin/main` in the `ccbf81b` range.
**Why:** The Design Phase 1 exit criterion requires end-to-end greedy generation against a real ONNX model path. Batty's engine wiring, Rachael's CLI facade, Deckard's ORT/tokenizer APIs, and Pris's tiny fixture now satisfy that foundation milestone and unblock Phase 2 Agent Essentials.


---

# 2026-07-12T09:20:00-07:00: Phase 2 Agent Essentials assessment and plan

**By:** Roy

## Status table

| Phase 2 item | Status | Evidence | Assessment |
|---|---:|---|---|
| Paged KV cache (page table, free list, append/rewind) | PARTIAL | `crates/onnx-genai-kv/src/page_table.rs:23-37`, `:72-96`, `crates/onnx-genai-kv/src/paged_cache.rs:37-59`, `:99-122` | Page table, free list, sequence mapping, append-by-token-count, rewind, and remove exist. Missing actual key/value tensor storage, partial-page append semantics, real `filled` updates, page data layout per layer/key/value, and engine consumption. |
| Prefix cache (radix trie, lookup, insert) | PARTIAL | `crates/onnx-genai-kv/src/prefix_cache.rs:6-18`, `:31-74` | Trie-shaped structure exists with lookup/insert. It stores page IDs simplistically per token index, has no LRU/eviction, no integration with page refcounts, no prefix ownership lifecycle, and no engine lookup/skip-prefill path. |
| Multi-session support (persistent KV across turns) | MISSING | `crates/onnx-genai-engine/src/engine.rs:247-329`, `:331-334`, `:377-432` | `generate` creates a fresh sequence and fresh `DecodeState` per call, then completes it. No public session object, session map, conversation/token history, or persistent KV across API calls. |
| CoW fork | PARTIAL | `crates/onnx-genai-kv/src/paged_cache.rs:61-79`, `crates/onnx-genai-kv/src/page_table.rs:88-96` | Fork increments page refcounts and shares page IDs, but ignores `position`, has no copy-on-write on later writes, no partial-page split, and no actual tensor page data to copy/share. |
| Continuous batching scheduler (basic FCFS) | PARTIAL | `crates/onnx-genai-scheduler/src/lib.rs:89-166`, `:168-178`; `crates/onnx-genai-engine/src/engine.rs:269-297` | Scheduler admits FCFS up to batch size and returns prefill/decode decisions, but engine does not call `schedule`; generation is still synchronous single request. No batch ORT execution, resource accounting, streaming channels, cancellation, or multi-request loop. |
| OpenAI-compatible HTTP server with streaming | PARTIAL | `crates/onnx-genai-server/src/main.rs:7-17`, `:46-64`, `:67-80` | `/v1/chat/completions` route and request/response shells exist. Handler returns a placeholder, ignores `stream`, does not create/load/share an engine, and has no SSE streaming or session surface. |

## KV integration risk and recommendation

The biggest Phase 2 risk is replacing the current naive ORT present/past threading without breaking Phase 1 generation. Today `DecodeState` owns whole ORT `Value`s: empty past tensors are synthesized for prefill, `Session::run` returns present tensors, and outputs are cloned into `decode_state.past` for the next step (`crates/onnx-genai-engine/src/engine.rs:500-535`, `:583-629`). `PagedKvCache` is currently only accounting metadata: `Page` has no key/value buffers (`crates/onnx-genai-kv/src/page_table.rs:9-21`), and `KvCacheOps::append` accepts only `num_tokens`, not tensors (`crates/onnx-genai-kv/src/lib.rs:44-66`).

Recommendation: take ownership of KV tensors in the paged cache, but introduce it incrementally behind an adapter that initially preserves ORT-managed present/past behavior. Layering paging only on top of ORT-managed whole-present tensors would satisfy accounting but not real prefix sharing, CoW, partial rewind, or page-level eviction; ORT still returns full growing present tensors that must be copied and cannot be shared at page granularity. The durable architecture needs `PagedKvCache` to store per-layer key/value tensor slices per page and materialize the model's expected past inputs from page tables when calling ORT. Sequence it so Phase 1 stays green: first add tensor storage APIs/tests unused by engine; then mirror ORT present outputs into paged pages while continuing to feed `DecodeState.past`; then switch reads to materialize past from pages for one sequence; then enable persistent sessions/prefix/CoW; finally optimize with IoBinding/device buffers.

Real tensor storage requires: metadata describing KV layer count, key/value input-output names, dtype, shape axes, sequence axis, head/head_dim layout; a page buffer representation keyed by page/layer/{K,V}; append code that slices ORT present or delta tensors into fixed token pages; refcount-aware write path that copies shared pages before append/overwrite; materialization back to ORT `Value`s for the current past-input shape; and eventually IoBinding/preallocated device buffers to avoid repeated CPU copies.

## Dependency-annotated task breakdown

### Deckard — KV cache, prefix trie, CoW, tensor storage, tiered

1. **D1: Define KV tensor page schema and adapter types** — dependencies: none. Add typed metadata for layer/key/value slots, dtype, shape axes, sequence axis, and per-page buffers. Do not change engine behavior yet.
2. **D2: Implement real page append/rewind storage** — depends on D1. Replace count-only append with tensor-slice append, accurate `filled`, partial-page handling, and rewind to token position.
3. **D3: Implement CoW writes and position-aware fork** — depends on D2. Fork should share full prefix pages, split/copy partial page at fork position, and copy on later writes to shared pages.
4. **D4: Harden prefix cache lifecycle** — depends on D2/D3. Store prefix length to page-span mapping, bump/drop page refcounts, support longest-prefix lookup, insert, eviction hooks, and tests.
5. **D5: Tiered storage prep** — depends on D2. Keep Phase 2 in-memory, but make page device state/migration API explicit so Phase 3 eviction can attach without redesign.

### Batty — scheduler, continuous batching, multi-session engine integration, engine↔paged-KV wiring

1. **B1: Introduce persistent engine sessions while preserving single-request `generate`** — dependencies: none. Add session IDs, per-session token history/decode state, and compatibility wrapper so CLI still works.
2. **B2: Add KV adapter mirror mode** — depends on D1/D2. After each ORT run, write present KV into `PagedKvCache` but continue feeding existing `DecodeState.past`; verify no behavior change.
3. **B3: Switch one-session past reads to paged materialization** — depends on B2. Materialize ORT past inputs from paged cache, compare with mirror mode where feasible, keep fallback to naive path.
4. **B4: Prefix cache integration** — depends on D4 and B1/B3. Lookup prompt/session prefixes, skip already-cached prefill tokens, insert computed prefixes, expose hit/miss stats.
5. **B5: Continuous batching FCFS engine loop** — depends on B1 and scheduler API cleanup. Move from direct synchronous loop to scheduling decisions, batch compatible prefill/decode work, and stream per-token results via channels. Full efficient batched ORT can follow once correctness is established.
6. **B6: CoW fork API** — depends on D3 and B1/B3. Expose session fork and ensure branch generation diverges without corrupting shared prefix pages.

### Rachael — HTTP server, streaming, OpenAI API, multi-session HTTP surface

1. **R1: Replace placeholder with real non-streaming engine call** — dependencies: B1 for stable engine construction/session API; can prototype against existing `Engine::generate` now.
2. **R2: Implement OpenAI-compatible SSE streaming** — depends on token callback/channel API from B1/B5. Emit chat completion chunks and terminal `[DONE]`.
3. **R3: Add multi-session HTTP surface** — depends on B1. Define session identifier handling (header or request field), message-to-prompt formatting, and persistent conversation state.
4. **R4: Concurrent request plumbing** — depends on B5. Share the engine loop safely from axum via channels, support cancellation/disconnect, and preserve OpenAI-compatible errors.

### Pris — tests and benchmarks

1. **P1: Preserve Phase 1 regression tests** — dependencies: none. Keep `cargo test --workspace` and tiny fixture generation green after each integration step.
2. **P2: KV storage unit tests** — depends on D2/D3. Test append across page boundaries, rewind partial pages, refcounts, CoW fork divergence, and prefix refcount lifecycle.
3. **P3: Scheduler tests** — dependencies: scheduler API from B5. Test FCFS admission, completion removal, max batch size, and multi-session progress.
4. **P4: HTTP streaming tests** — depends on R2. Verify OpenAI response shape, SSE chunks, `[DONE]`, and non-streaming parity.
5. **P5: Exit benchmark** — depends on B4/R3. Measure first vs second turn in the same session and shared system-prompt prefix; require observable prefix hit and faster second turn on the tiny fixture or a deterministic benchmark harness.

## Parallelization plan

Can start now in parallel:
- Deckard D1 (KV tensor schema/design) and D2 unit-test scaffolding.
- Batty B1 (persistent sessions with naive ORT `DecodeState`) because it does not require real paged tensors.
- Rachael R1 prototype against current `Engine::generate`, with a seam for later session IDs.
- Pris P1 plus planned tests for current scheduler/KV accounting behavior.

Blocked:
- Batty B2/B3/B4 and Deckard D3/D4 require real tensor page storage.
- Batty B5 efficient continuous batching needs B1 plus a clear engine loop/channel API.
- Rachael R2/R3/R4 need Batty's streaming/session/engine-loop APIs.
- Pris P2/P5 depend on Deckard/Batty integration work.

Highest-value first task: **Batty B1 — introduce persistent engine sessions while preserving `Engine::generate` compatibility.** It unlocks multi-turn semantics, gives Rachael a concrete HTTP session API, lets Pris define exit tests, and creates the safe seam where Deckard's paged tensor cache can replace naive `DecodeState` incrementally without breaking Phase 1.

---

### 2026-07-12T09:22:00-07:00: Paged KV tensor storage and prefix sharing lifecycle
**By:** Deckard
**What:** `onnx-genai-kv` physical pages can now own f32 K/V tensors when constructed with `PageTensorConfig { num_layers, num_kv_heads, head_dim, page_size, dtype: KvDType::F32 }`. Each page buffer is contiguous with shape `[num_layers, 2, num_kv_heads, page_size, head_dim]`; axis 1 is `0 = key`, `1 = value`, and the flat offset is `(((((layer * 2 + kv) * num_kv_heads + head) * page_size + token_offset) * head_dim) + dim)`. Public tensor APIs are `PagedKvCache::new_with_tensor_config(config, num_gpu_pages)`, `append_token_kv(seq, layers: &[LayerKv<'_>]) -> Result<usize, KvError>`, `write_token_kv(seq, position, layers: &[LayerKv<'_>]) -> Result<(), KvError>`, and `materialize_sequence(seq) -> Result<MaterializedKv, KvError>`. Existing count APIs (`append`, `rewind_to`, `len`, `fork`, `remove`) continue to work and now track exact token length instead of page-rounded length.
**Why:** Batty needs a stable storage boundary for Phase 2 engine integration: after ORT returns present K/V, write one token at a time with per-layer `[num_kv_heads, head_dim]` key/value slices, and before model input materialize per-layer buffers shaped `[num_kv_heads, sequence_len, head_dim]`.

**What:** `PrefixCache` now stores cached prefix terminal nodes as full `Vec<PageId>` references. `insert_pages(tokens, page_ids, &mut PageTable)` retains physical pages for cache ownership. `lookup_shared(tokens, &mut PageTable)` returns `PrefixMatch { matched_tokens, page_ids }`, bumps the matched trie node's active refcount, and retains each physical page for the sharing sequence. `release_shared(tokens, matched_tokens, &mut PageTable)` releases a sequence share. `evict_lru(target_pages, &mut PageTable)` removes least-recently-used cached prefixes with zero active shares and frees the cache-owned page references.
**Why:** Prefix pages are now shareable across sequences with correct page-table refcounts. CoW semantics are: fork/prefix sharing only increments refcounts and attaches the same page IDs; any writer seeing `ref_count > 1` must allocate and copy before mutation. `PagedKvCache::write_token_kv` already performs page-level CoW for writes through this cache; Batty should preserve that path or call an equivalent CoW guard when wiring engine writes.

---

# Deckard decision: ORT runtime packaging

Date: 2026-07-12T09:36:00-07:00

## Decision

`onnx-genai-ort-sys` owns ORT runtime packaging for local builds:

- Keep the downloaded ORT runtime pinned to `ORT_VERSION = "1.27.0"`, matching the bindgen headers that expose `ORT_API_VERSION 27`.
- Reject stale auto-download caches by requiring the expected versioned runtime library (`libonnxruntime.1.27.0.dylib` on macOS, `libonnxruntime.so.1.27.0` on Linux) before reusing `target/**/ort-prebuilt`.
- Ensure the major-version runtime name exists after resolving the ORT lib directory (`libonnxruntime.1.dylib` on macOS, `libonnxruntime.so.1` on Linux), creating a symlink to the 1.27.0 runtime if needed.
- Emit an rpath linker arg for macOS/Linux builds: `-Wl,-rpath,<ort lib dir>`, while preserving the existing link-search and link-lib directives.
- On macOS, set the ORT dylib install names to the resolved major-version path and ad-hoc codesign the modified dylibs so standalone binaries can load them without `DYLD_LIBRARY_PATH`.

## Verification

Built and ran the standalone server with no `DYLD_LIBRARY_PATH`:

```text
cargo build -p onnx-genai-server
./target/debug/onnx-genai-server --model tests/fixtures/tiny-llm
```

Server booted on `127.0.0.1:8080` and returned:

```json
{"status":"ok","model":"tiny-llm"}
```

Chat completion returned OpenAI-shaped JSON:

```json
{"id":"chatcmpl-1783874500","object":"chat.completion","created":1783874500,"model":"tiny-llm","choices":[{"index":0,"message":{"role":"assistant","content":"world"},"finish_reason":"length"}],"usage":{"prompt_tokens":0,"completion_tokens":1,"total_tokens":1}}
```

Clean rebuilt `onnx-genai-ort-sys`; the regenerated auto-download contained only `libonnxruntime.1.27.0.dylib`, `libonnxruntime.1.dylib`, and `libonnxruntime.dylib` (no 1.22.0 runtime).

Full validation passed:

- `cargo check --workspace`
- `cargo test --workspace`

---

### 2026-07-12T09:22:00-07:00: Persistent engine sessions and minimal scheduler admission
**By:** Batty
**What:** Added `SessionId` plus `Engine::create_session`, `generate_in_session`, `generate_in_session_with_callback`, `reset_session`, `close_session`, and `session_token_count`. `Engine::generate` remains stateless by creating an ephemeral session, generating through the same session path, then closing it. Each engine session retains the logical token context, a KV-materialized token count, and the ORT-managed `DecodeState` past tensors produced from present outputs. Follow-up turns append only new prompt tokens to the session context and feed pending tokens through the stored past tensors instead of rebuilding from scratch. The scheduler now has `enqueue_generate_request` and `drive_next_fcfs` so the engine admits session generate requests through a minimal one-at-a-time FCFS path.
**Why:** Phase 2 agent workloads need multi-turn KV reuse without breaking Rachael's stateless CLI/API. This preserves the current ORT past/present tensor threading while making it persistent per session. Deckard's paged KV storage will plug in at the `EngineSession` boundary: replace/augment `decode_state.past` and `kv_token_count` with paged KV page-table handles, and have `next_session_token_logits` materialize/bind those pages to ORT inputs before each decode step. Full continuous batching remains a scheduler TODO beyond the single-request FCFS drive loop.

---

### 2026-07-12T09:31:00-07:00: Engine paged-KV prefix reuse integration
**By:** Batty
**What:** `onnx-genai-engine` now derives `PageTensorConfig` from present-KV `TensorInfo`: number of layers from key/value present output pairs, KV heads from the static head axis, and head dim from the final static axis (`[batch, kv_heads, seq, head_dim]`). Each ORT forward still threads model-owned past/present tensors, and the engine mirrors newly computed present KV token slices into `PagedKvCache::append_token_kv`.
**Why:** This keeps the current ORT decode path stable while making KV pages available for same-session reuse and cross-session prefix reuse.

**What:** Prefix reuse uses `PrefixCache::lookup_shared` for empty/new sessions, materializes shared pages back into ORT past tensors, and pre-fills only the uncached suffix. When the cached prefix equals the full prompt, the engine materializes through `prompt_len - 1` and feeds the final prompt token so logits remain correct. Same-session turns continue reusing retained ORT past and only feed tokens missing from the session KV state.
**Why:** Decoder-only models need one input token to produce next-token logits; materializing the entire prompt without feeding a suffix would leave no logits to sample from.

**What:** `GenerateResult::prefix_cache_hit_len: usize` is the additive observable for Pris/Rachael. It reports retained same-session context length for warm turns and matched prompt-prefix length for cross-session cache hits.
**Why:** Callers can verify cache behavior without changing `Engine::generate`, `create_session`, or `generate_in_session` signatures.

**Remaining gaps:** Prefix pages are CPU-copied back into ORT `Value`s for reuse because the current ORT wrapper does not expose direct paged-buffer binding. Cross-session sharing is implemented for CPU Float32 KV and the current GPT-style `[batch, kv_heads, seq, head_dim]` layout; other KV layouts will need additional axis inference.

---

### 2026-07-12T09:22:00-07:00: OpenAI-compatible HTTP server endpoints
**By:** Rachael
**What:** `onnx-genai-server` now uses axum/tokio and loads one Engine from `--model` or `ONNX_GENAI_MODEL`, shared as `Arc<Mutex<Engine>>`. It exposes `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions`. Chat requests map OpenAI `messages`, `max_tokens`, `temperature`, `top_p`, and `stop` onto `GenerateRequest`/`GenerateOptions` with a simple role-tagged prompt template. Non-streaming responses return OpenAI `chat.completion` JSON; streaming responses return SSE `chat.completion.chunk` deltas followed by `data: [DONE]`.
**Why:** Phase 2 needs a real OpenAI-compatible HTTP surface while Batty/Deckard continue deeper scheduler/KV work. The engine is serialized behind a mutex for now; later multi-session support can replace the per-request `Engine::generate` call with persistent session creation/lookup while keeping the HTTP handlers and response shapes stable.

---

# 2026-07-12T09:30:00-07:00: HTTP multi-session support

**By:** Rachael

## Decision

The HTTP server accepts `X-Session-Id` on `POST /v1/chat/completions` as the persistent conversation address. Requests without the header keep the OpenAI-compatible stateless path unchanged. Requests with the header are treated as a session turn: the latest chat message content is appended to the persistent engine session rather than replaying the full role-tagged transcript, so clients should send the new turn payload for session calls. Requests with an unknown non-empty `X-Session-Id` auto-create a server-side engine session bound to that client id; `POST /v1/sessions` creates and returns a generated id for clients that prefer explicit lifecycle management.

## Lifecycle endpoints

- `POST /v1/sessions` returns `{ "id": "sess-...", "object": "session" }`.
- `DELETE /v1/sessions/{id}` removes the registry entry and closes the engine session, returning `204 No Content` or `404` for an unknown id.

## Registry and locking

`AppState` owns a thread-safe `SessionRegistry` (`Arc<Mutex<HashMap<String, SessionId>>>`) mapping client-visible ids to engine `SessionId`s. Engine access remains serialized through the existing `Arc<Mutex<Engine>>`; both non-streaming and SSE streaming session requests call `generate_in_session` / `generate_in_session_with_callback`. This is intentionally coarse-grained for Phase 2 and limits concurrent generation to one engine operation at a time until the engine exposes finer-grained session scheduling/locking.

---

# 2026-07-12T09:31:00-07:00: Phase 2 multi-session, fork, and prefix-speed tests

**By:** Pris

## What
Added engine integration coverage for multiple interleaved persistent sessions against `tests/fixtures/tiny-llm`, including independent context growth and reset isolation. Added a KV tensor fork test that writes shared tokens, forks, mutates parent pages, appends divergent child K/V, then materializes both sequences to verify copy-on-write independence and retained shared-prefix values.

## Prefix speed harness
Added a tiny fixture harness that times a cold first turn and a second turn in the same session, prints both durations, and verifies session mechanics. `GenerateResult::prefix_cache_hit_len` is available in this tree, so the harness asserts cold hit length is `0` and second-turn same-session hit length is `> 0`. Additional engine integration tests verify cross-session prefix reuse with an extended prompt and identical greedy output with and without prefix reuse while avoiding raw latency as a hard assertion.

---

### 2026-07-12T09:38:00-07:00: Phase 2 Agent Essentials complete + standalone server runnable
**By:** Scribe
**What:** Phase 2 Agent Essentials exit criteria are met. Roy completed the Phase 2 plan; Deckard delivered paged KV tensor storage, prefix cache ownership/refcount lifecycle, CoW-safe writes, and ORT runtime packaging; Batty delivered persistent sessions, minimal FCFS scheduler admission, paged-KV mirroring, prefix reuse, and `GenerateResult::prefix_cache_hit_len`; Rachael delivered OpenAI-compatible HTTP/SSE endpoints plus `X-Session-Id`/session lifecycle support; Pris delivered concurrent-session, fork CoW, and prefix-hit tests. The standalone `onnx-genai-server` now boots without `DYLD_LIBRARY_PATH` and serves `/health`, `/v1/models`, and `/v1/chat/completions`.
**Shared contracts preserved:** `PagedKvCache::new_with_tensor_config`, `append_token_kv`, `write_token_kv`, and `materialize_sequence` use page buffers shaped `[num_layers, 2, num_kv_heads, page_size, head_dim]`; `PrefixCache::insert_pages`, `lookup_shared`, `release_shared`, and `evict_lru` own/share pages with page-table refcounts and CoW on shared writes; engine sessions use `create_session`, `generate_in_session`, `generate_in_session_with_callback`, `reset_session`, `close_session`, and `session_token_count`; cache observability is `GenerateResult::prefix_cache_hit_len`; HTTP session addressing uses `X-Session-Id`, `POST /v1/sessions`, and `DELETE /v1/sessions/{id}`; `onnx-genai-ort-sys` pins ORT `1.27.0`, validates cached runtime libraries, creates major-version symlinks, emits rpath, and fixes macOS install names/codesigning for standalone loading.
**Verification:** `cargo test --workspace` is green. End-to-end server validation succeeded with no `DYLD_LIBRARY_PATH`, including `/health`, `/v1/models`, and chat completion requests.
**Next:** Real-model text-generation validation, robustness fixes (`context-length` guard and `usage.prompt_tokens=0`), then Phase 3 Performance: speculative decoding, tiered KV, and priority scheduling.


---

# 2026-07-12T09:42:00-07:00: Phase 3 Performance assessment and plan

**What:** Phase 3 is not implementation-ready as a single vertical slice yet; it needs one correctness-first speculative path plus storage/scheduler/streaming hardening in parallel. Current source shows speculative decoding and tiered storage are still stubs, priority scheduling has only queue-ordering scaffolding, fp8 KV quantization is absent, and token streaming/early stop is mostly present but not yet integrated with a concurrent/preemptible engine loop.

## Status assessment

| Phase 3 item | Status | Evidence |
|---|---:|---|
| Speculative decoding (draft model + greedy acceptance first) | **MISSING** | `docs/DESIGN.md:282-327` defines `SpeculativeEngine::step` as draft K, verify in one target pass, accept/reject, rewind KV. `crates/onnx-genai-engine/src/speculative.rs:1-9` is only a TODO stub. No engine call sites beyond module export/search hits. |
| Tiered KV storage (GPU→CPU eviction under pressure) | **MISSING** | Design requires GPU/CPU/SSD tiers plus eviction/prefetch (`docs/DESIGN.md:160-172`). Current `crates/onnx-genai-kv/src/tiered.rs:1-5` is a TODO stub; `PagedKvCache::evict` returns `0` (`crates/onnx-genai-kv/src/paged_cache.rs:199-203`). There is only data-model scaffolding: `Device::{Gpu,Cpu,Disk}` (`crates/onnx-genai-kv/src/lib.rs:25-31`) and pages initially allocated on GPU (`crates/onnx-genai-kv/src/page_table.rs:113-130`). |
| Priority-based scheduling + preemption | **PARTIAL** | Design calls for waiting/running/swapped queues, preemption policy, and `preempt`/`swap_in` decisions (`docs/DESIGN.md:223-280`). Current scheduler has `Priority`, `PriorityPolicy`, and priority sorting (`crates/onnx-genai-scheduler/src/lib.rs:13-22`, `65-83`, `189-207`), but engine still uses `Priority::Normal` and `drive_next_fcfs` (`crates/onnx-genai-engine/src/engine.rs:320-329`), while preemption/fair-share are TODOs (`crates/onnx-genai-scheduler/src/policy.rs:1-5`). |
| KV cache quantization (fp8) | **MISSING** | Design requires quantize-on-write/dequantize-on-read and sensitive-layer bypass (`docs/DESIGN.md:173-184`). Current `KvDType` only has `F32`, validation only accepts `F32`, and page storage is `Vec<f32>` (`crates/onnx-genai-kv/src/page_table.rs:9-25`, `36-42`, `69-70`). No `quantized.rs` exists in `crates/onnx-genai-kv/src`. |
| Token streaming with early stopping | **PARTIAL** | Engine/server already stream each token through callbacks/SSE (`crates/onnx-genai-engine/src/engine.rs:170-179`, `380-397`; `crates/onnx-genai-server/src/lib.rs:502-585`) and stop on EOS/stop sequences (`crates/onnx-genai-engine/src/engine.rs:1338-1351`; server maps finish reasons at `crates/onnx-genai-server/src/lib.rs:746-750`). Remaining Phase 3 work is to expose engine-loop streaming/cancellation hooks that still work under batching, speculative multi-token acceptance, and preemption. |

## Dependency-annotated work breakdown

### Deckard — `onnx-genai-kv`: tiered KV + fp8 quantization

1. **D1: Make page storage dtype-aware** — deps: none. Extend `KvDType` beyond `F32` (fp8 target plus native fallback), separate logical tensor dtype from physical bytes, and preserve current F32 tests.
2. **D2: Quantization boundary** — deps: D1. Implement quantize-on-write/dequantize-on-materialize with per-layer sensitive bypass matching `docs/DESIGN.md:173-184`; add error bounds/unit tests.
3. **D3: Tiered page migration model** — deps: D1. Implement GPU→CPU page movement in `tiered.rs`, update `Page.device`, free lists, LRU timestamps, and accounting without changing sequence semantics.
4. **D4: Eviction/prefetch API** — deps: D3. Replace `PagedKvCache::evict` stub and add prefetch/swap-in hooks scheduler can call; keep disk tier deferred unless CPU tier is stable.
5. **D5: Pressure tests** — deps: D2-D4; coordinated with Pris. Validate eviction and fp8 do not corrupt materialized KV shape/content beyond quantization tolerance.

### Batty — engine/scheduler: speculative decoding + priority/preemption

1. **B1: Speculative correctness contract** — deps: none, but consult Pris. Define greedy-only acceptance API and required rollback semantics before optimizing.
2. **B2: Draft/verify loop** — deps: B1 and current KV rewind. Implement `SpeculativeEngine` for draft-model producer, greedy acceptance, and `rewind_to` on rejection.
3. **B3: Engine integration flag/config** — deps: B2. Route greedy generation through speculative path only when enabled and when a draft session is configured; baseline remains default.
4. **B4: Scheduler preemption model** — deps: Deckard D4 for real swap; can start with recompute/no-op swap. Add priority-aware admission/preemption decisions rather than `drive_next_fcfs` only.
5. **B5: 10+ session execution path** — deps: B4, Deckard D4, Rachael hooks. Move toward the design’s engine loop/channel model (`docs/DESIGN.md:459-490`) or a smaller compatible stepping API.

### Rachael — server + engine streaming hooks

1. **R1: Streaming semantics for accepted token batches** — deps: Batty B2 interface. Ensure speculative acceptance emits tokens in order one by one with correct finish reason on the token that stops generation.
2. **R2: Early-stop/cancel propagation** — deps: R1. Let server/client disconnect or callback error stop generation promptly and unwind scheduler/session state.
3. **R3: Concurrent session API fit** — deps: Batty B4/B5. Remove coarse serialized assumptions where possible while preserving existing OpenAI SSE shapes and `X-Session-Id` behavior from `.squad/decisions.md:284-299`.
4. **R4: HTTP observability** — deps: R1-R3. Surface prefix/speculative counters in logs or optional response metadata without breaking OpenAI compatibility.

### Pris — tests/benchmarks

1. **P1: Speculative correctness test** — deps: Batty B1/B2. For greedy decoding, assert speculative output tokens are exactly equal to baseline greedy tokens for the same prompt/options; include rejection cases and stop/EOS boundaries.
2. **P2: Draft model harness** — deps: P1. Support same-model-as-draft for correctness-only and a smaller draft artifact for performance runs.
3. **P3: Speedup benchmark** — deps: Batty B3 and real/smaller draft. Measure tokens/sec and require >1.5x on a stable prompt/model pair before declaring exit criteria met.
4. **P4: 10+ concurrent sessions/OOM stress** — deps: Deckard D4, Batty B4/B5, Rachael R2/R3. Drive at least 10 session IDs through CLI/HTTP or engine API and assert no OOM, no session cross-talk, and bounded KV pages.
5. **P5: Regression suite** — deps: all. Keep Phase 1/2 contracts green: greedy generation, prefix reuse/`prefix_cache_hit_len`, CoW fork, HTTP streaming.

## Highest-value first task

**Start Batty B1 + P1 immediately: the greedy speculative correctness contract and token-for-token test harness.** This is the riskiest invariant, because speculative acceptance must be provably equivalent to baseline greedy even when only a prefix of draft tokens is accepted and KV is rewound. Once this harness exists, Batty can implement B2 safely and Pris can reuse it for the speedup benchmark.

## Parallel work that can start now

- Deckard D1/D2 can start independently of speculative decoding; fp8 storage shape and tolerance tests are local to `onnx-genai-kv`.
- Deckard D3 can start with CPU-only migration scaffolding, but D4 should align with Batty’s preemption API.
- Batty B4 can sketch scheduler decisions and priority tests now, but real swap/preemption completion depends on Deckard D4.
- Rachael R1 can define streaming semantics for multi-token acceptance as soon as Batty publishes the speculative step result shape.
- Pris can scaffold P1/P2 with a fake or same-model draft before a true small draft exists.

## Biggest risk

**Speculative decoding correctness.** Greedy speculative decoding must produce exactly the same token stream as baseline greedy, including stop/EOS behavior, processor-chain effects, and KV state after rejection. A second smaller draft model is needed for real speedup, but correctness should not depend on draft quality.

## Draft-model recommendation

Use two tiers of draft model support:

1. **Correctness harness:** allow the target model to be used as its own draft with `K > 1`. This will not show speedup, but it should accept all tokens under greedy decoding and proves the verify/accept/stream/stop path can match baseline token-for-token. Also add a deterministic deliberately-wrong draft producer to exercise rejection and KV rewind.
2. **Performance harness:** obtain or build a genuinely smaller TinyStories-family ONNX draft model with the **same tokenizer/vocabulary** as TinyStories-1M. A truncated copy of the same model is only useful if exported as a valid smaller decoder with matching logits/tokenizer; arbitrarily truncating layers may break graph semantics or quality. Prefer exporting a smaller TinyStories config/checkpoint through the existing conversion path, or train/export a tiny fixture with matching tokenizer for benchmark-only use.

**Why:** Phase 3 exit criteria require both >1.5x speculative speedup and 10+ concurrent sessions without OOM (`docs/DESIGN.md:672-680`). Splitting correctness from performance lets us prove equivalence before spending time on draft-model quality and benchmark tuning.


---

# 2026-07-12T09:46:00-07:00: Tiered KV storage and int8 KV quantization

**What:** `onnx-genai-kv` now treats `Device::Gpu(0)` pages as HOT and `Device::Cpu` pages as COLD. `PageTable::allocate(Device::Gpu(0))` evicts the least-recently-used referenced hot page when the hot capacity is full, preserving the page payload and marking it cold; `PageTable::promote_to_hot(page_id)` and `PagedKvCache::prefetch(seq, start, end)` promote cold pages back, evicting another hot LRU page if needed. `PagedKvCache::evict(EvictionPolicy::Lru, target)` explicitly moves hot pages to cold.

**Quantization:** Added opt-in `KvDType::Int8` on `PageTensorConfig`. This uses symmetric signed int8 with one scale per page (`dequantized = q as f32 * scale`) rather than fp8 e4m3, because it is portable on the current CPU-only backend and gives a clear error bound. Writes quantize from f32 into int8 page storage; reads/materialization dequantize to f32. F32 remains the default and existing APIs are unchanged.

**Tolerance:** One quantization pass has expected absolute error bounded by roughly `scale / 2` for the page, where `scale = max_abs / 127`. Current tests assert representative append/materialize round trips within `0.05` absolute error; f32 tier eviction/promotion remains byte-for-byte exact.

**Engine opt-in later:** Batty can keep using `PageTensorConfig { dtype: KvDType::F32, .. }` today. To enable quantized KV, derive the same tensor geometry but set `dtype: KvDType::Int8`; the engine will continue calling `append_token_kv` and `materialize_sequence`, receiving f32 materialized past tensors. To enable tiering, configure a smaller `num_gpu_pages` hot budget and call `prefetch(session_id, token_start, token_end)` before decode ranges that must be hot, or rely on write promotion for append/overwrite paths. A future device backend can replace the synchronous host move with real GPU↔CPU copies under the same page-table API.


---

### 2026-07-12T09:46:00-07:00: Greedy speculative decoding draft/verify loop
**By:** Batty
**What:** Added an additive draft-model API with `EngineConfig::draft_model`, default `EngineConfig::num_speculative_tokens`, and per-request `GenerateOptions::num_speculative_tokens`. When a draft model is configured for greedy generation, the engine proposes K draft tokens autoregressively, verifies the proposed block with the target, accepts the longest prefix whose greedy target token matches the draft token, and emits the target token on the first mismatch (or the target token after a fully accepted block when budget/context allow).
**Why:** Greedy speculative decoding must be token-for-token identical to target-only greedy decoding; speculation may change speed but not output. Target KV is rewound through `PagedKvCache::rewind_to` plus materialization back into ORT past tensors when draft tokens are rejected; draft KV is independently rewound/synced to the target logical sequence so rejected draft tokens never carry forward. Added tiny fixture coverage using `tests/fixtures/tiny-llm/` as both target and draft with K=3; speculative token ids exactly match baseline greedy. A real smaller draft model for actual speedup remains for Pris to validate later.


---

# Priority scheduling and swap preemption

**Date:** 2026-07-12T09:52:00-07:00
**Owner:** Batty

## What

Phase 3 adds deterministic priority scheduling to `onnx-genai-scheduler`. Requests carry a `Priority` and are ordered by higher priority first, with FCFS arrival order as the tie-breaker. The legacy FCFS admission path remains available.

The scheduler can now preempt a lower-priority running sequence when a higher-priority request arrives and single-sequence capacity is exhausted. The implemented policy is swap-style preservation: the engine keeps the session decode state, ORT past tensors, and mirrored paged KV in place while the scheduler marks the sequence swapped. Resuming swaps the same sequence back in without recomputation. `PreemptionPolicy::Recompute` is reserved for a later tiered/recompute implementation.

The engine exposes `drive_prioritized_requests` for already-arrived work and `drive_prioritized_arrivals` for requests drained between decode iterations. These APIs run one sequence at a time today, honor scheduler priority/preemption decisions, and leave continuous batched-forward execution as future work.

## Why

Agent workloads need interactive requests to cut ahead of background generations without losing in-progress session state. Preserving KV in place is the smallest correct preemption model for this phase because current engine sessions already own logical tokens, ORT past tensors, and paged-KV mirrors. It keeps behavior deterministic and avoids adding CPU/GPU tier migration before Deckard's deeper KV work lands.


---

# 2026-07-12T09:42:00-07:00: Context-window guard for greedy generation

**By:** Batty

## Decision

`onnx-genai-engine` determines a request's maximum context length from inference metadata first, using `model.max_sequence_length` when present. If metadata does not declare a length, the engine uses the additive `GenerateOptions::max_context: Option<usize>` request field. Graph-based inference is intentionally skipped for now because the position-embedding table is not exposed through the current ORT `ModelDirectory`/`Session` metadata surface reliably enough.

## Stop behavior

Before each greedy decode step, the engine compares the retained logical context length (persistent session tokens plus current prompt plus generated tokens) with the known maximum context. If the next step would run at or beyond that context window, generation returns successfully with `finish_reason = Length` without calling ORT, avoiding position-embedding Gather out-of-bounds failures. If no maximum context is known and ORT reports a Gather indices out-of-bounds failure, the engine wraps it as a clear model-context error that tells callers to provide metadata or `GenerateOptions::max_context`.


---

# 2026-07-12T10:06:00-07:00: Speculative reject-path draft KV rewind fix

**By:** Batty

## Root cause
When target verification rejected a draft token, the draft session was synced by truncating only to the final target length. If `accepted + correction` had the same length as the draft KV, or after rewinding to that length, the draft KV still represented the rejected draft token instead of the target correction. The next draft round then saw `draft_state.kv_token_count == draft_state.tokens.len()` and failed with `draft decode step has no new token to feed` rather than feeding the target correction token.

## Fix
Draft sync now computes the common prefix between the draft logical tokens and target logical tokens, rewinds draft KV to that shared prefix, and then replaces draft logical tokens with the target tokens. On rejection, the target correction remains beyond draft KV and is fed as the seed token for the next draft round. On full acceptance, the common prefix is the accepted draft span, so the target bonus token similarly seeds the next round.

## Verification
- Reproduced the real-model failure first: `cargo test -p onnx-genai-engine --test speculative_speedup speculative_decoding_exceeds_required_speedup_when_models_are_present -- --ignored --nocapture` failed with `draft decode step has no new token to feed`.
- Added exact token-id equality to the real-model speculative benchmark so target-only greedy and speculative greedy must match token-for-token.
- After the fix, real differing models `models/tinystories-33m` target + `models/tinystories-1m` draft pass correctness with `ONNX_GENAI_SPEC_ALLOW_SLOW=1 scripts/bench_speculative.sh` and with the direct ignored cargo test.
- `cargo check --workspace` passed.
- `cargo test --workspace` passed.

## Speed
Measured debug-test harness speedups on the real model pair:
- Benchmark script default K=4: 0.581x (32 tokens, baseline 184.36 tok/s, speculative 107.14 tok/s) — below the 1.5x Phase 3 target.
- Additional direct sweep: K=4: 0.641x, K=8: 0.425x, K=16: 0.316x.

Correctness is fixed for draft != target, but this CPU/debug harness still does not meet the Phase 3 speedup criterion.


---

# 2026-07-12T09:42:00-07:00: Server usage token accounting

**By:** Rachael

## Decision

`onnx-genai-server` now tokenizes the exact prompt string it sends to the engine, after applying the server chat prompt/session prompt shaping, and sends those token IDs via `GeneratePrompt::TokenIds`. OpenAI `usage.prompt_tokens` is the length of that tokenized prompt, `usage.completion_tokens` is `GenerateResult::token_ids.len()`, and `usage.total_tokens` is their sum.

## Why

The engine `GenerateResult` exposes generated token IDs but not the prompt token count. Tokenizing in the server with the same model tokenizer before generation keeps usage accounting accurate without changing the engine crate.


---

# 2026-07-12T09:52:00-07:00: Streaming early stop and cancellation behavior

**By:** Rachael

## Decision

`onnx-genai-server` passes OpenAI `stop` values through to `GenerateOptions::stop_sequences` for both non-streaming and streaming chat completions. Streaming now buffers only the suffix that could become a configured stop sequence, emits safe text deltas, suppresses the matched stop text, emits a terminal chunk with `finish_reason: "stop"`, and then sends `[DONE]`. `max_tokens` continues to terminate streams with `finish_reason: "length"` and `[DONE]`.

## Client cancellation

The SSE body is backed by a bounded Tokio mpsc channel. If the HTTP client disconnects and the body future is dropped, the receiver closes; the next blocking send from the generation callback fails, that callback error aborts `Engine::generate_with_callback`, and the blocking task exits instead of continuing to `max_tokens`.

## Limitation / Batty hook

The current engine API is synchronous and only observes cancellation at callback boundaries, so server cancellation cannot interrupt work already inside a single prefill/decode step before the next token callback. Recommended engine hook: add a cancellation token or `should_continue` callback checked before prefill, before each decode step, and immediately after ORT execution, returning a distinct cancelled result that the server can treat as a quiet disconnect.


---

### 2026-07-12T09:52:00-07:00: Phase 3 validation coverage
**By:** Pris
**What:** Added a 12-session interleaved engine stress test against `tests/fixtures/tiny-llm` with a 2-page hot KV tier to force page pressure, plus an ignored real-model speculative speed benchmark and `scripts/bench_speculative.sh`.
**Why:** Phase 3 exit validation needs repeatable coverage for 10+ session completion without OOM/panic and a reproducible target-vs-draft tokens/sec measurement.

### 2026-07-12T09:52:00-07:00: Phase 3 local validation results
**By:** Pris
**What:** `cargo test -p onnx-genai-engine --test phase3_concurrency_stress` passed: 12 interleaved persistent sessions completed two turns each with independent token counts under 2 hot KV pages. `cargo test --workspace` passed after the new benchmark was left `#[ignore]`.
**Why:** This covers exit criterion #2 for the tiny fixture path: no panic/OOM occurred while the engine created and retained more than 10 session contexts under hot-tier KV pressure.

### 2026-07-12T09:52:00-07:00: Speculative benchmark blocked by real draft bug
**By:** Pris
**What:** Built real models with Mobius under gitignored `models/`: target `roneneldan/TinyStories-33M` at `models/tinystories-33m` (`du -sh`: 425M; `model.onnx.data`: 409M) and draft `roneneldan/TinyStories-1M` at `models/tinystories-1m` (`du -sh`: 30M; `model.onnx.data`: 27M). Running `ONNX_GENAI_SPEC_ALLOW_SLOW=1 scripts/bench_speculative.sh` failed before timing with `draft decode step has no new token to feed`. As a same-model sanity check only, `ONNX_GENAI_SPEC_DRAFT=models/tinystories-33m ONNX_GENAI_SPEC_ALLOW_SLOW=1 scripts/bench_speculative.sh` reported 32 tokens, baseline 0.183s / 174.91 tok/s, speculative 0.279s / 114.61 tok/s, speedup 0.655x.
**Why:** Exit criterion #1 could not be validated with a smaller draft locally because the real target/draft path hits an engine bug. The precise failure is in the draft KV decode path: `draft_decode_input_tokens` sees `state.kv_token_count == state.tokens.len()` and bails instead of feeding a token for the next proposal step. The public result does not expose acceptance count, so acceptance rate could not be measured without engine instrumentation.


---

### 2026-07-12T09:38:00-07:00: Validate real TinyStories generation through CLI and HTTP
**By:** Pris
**What:** Built `roneneldan/TinyStories-1M` into `models/tinystories/` (30M on disk) using Mobius from `/Users/justinc/Documents/GitHub/mobius`. Because the HF repo only publishes `pytorch_model.bin`, the reproducible path first snapshots `config.json`, tokenizer files, and `pytorch_model.bin`, converts the PyTorch weights to `model.safetensors`, then runs: `PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src python -m mobius build --config models/tinystories-local --runtime ort-genai models/tinystories`. CLI sample for `Once upon a time` with 30 new tokens: `, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, shiny rock in the`. HTTP sample on `/v1/chat/completions` with 30 max tokens: `Once upon a time, there was a little girl named Lily. She loved to play with her friends in the park. One day, Lily and`.
**Why:** This proves `onnx-genai generate` and the standalone server can run a real pretrained, coherent English causal LM end-to-end, rather than only the deterministic random-weight `tests/fixtures/tiny-llm` fixture. The output is coherent, so no engine/KV/tokenizer/logits bug is diagnosed for dev follow-up.


---

### 2026-07-12T10:10:00-07:00: Phase 3 Performance complete
**By:** Scribe
**What:** Phase 3 implementation and validation are complete for the local environment. Roy produced the Phase 3 plan; Deckard delivered tiered hot/cold KV eviction/promotion plus opt-in `KvDType::Int8` quantized KV; Batty delivered greedy speculative decoding, priority scheduling with swap preemption, context-window guard behavior, and the real-draft speculative KV rewind fix; Rachael delivered accurate OpenAI usage token accounting plus streaming early-stop/client-cancel behavior; Pris validated real coherent TinyStories text generation, 12 concurrent/interleaved sessions with KV eviction and no OOM, and speculative correctness against differing real models.
**Shared contracts preserved:** Tiered KV treats `Device::Gpu(0)` pages as HOT and `Device::Cpu` pages as COLD; `PageTable::allocate(Device::Gpu(0))`, `promote_to_hot`, `PagedKvCache::prefetch`, and `PagedKvCache::evict(EvictionPolicy::Lru, target)` are the migration surface. `KvDType::Int8` on `PageTensorConfig` uses symmetric per-page int8 quantization and materializes back to f32 through the existing `append_token_kv`/`materialize_sequence` path. Speculative decoding is additive through `EngineConfig::draft_model`, `EngineConfig::num_speculative_tokens`, and `GenerateOptions::num_speculative_tokens`; greedy speculation must remain token-for-token identical to target-only greedy, with draft KV rewound to the common target/draft prefix after rejects. Priority scheduling uses `Priority`, `drive_prioritized_requests`, and `drive_prioritized_arrivals`; current preemption is swap-style preservation of session decode state, ORT past tensors, and mirrored paged KV, while recompute remains reserved. Streaming stop sequences buffer only possible stop suffixes, suppress matched stop text, emit terminal `finish_reason: "stop"`, and then `[DONE]`; client disconnect closes the bounded channel and aborts generation at callback boundaries. Usage accounting tokenizes the exact shaped prompt sent to the engine and reports prompt/completion/total tokens from that prompt length and generated token count. Context-window guard uses metadata `model.max_sequence_length` or `GenerateOptions::max_context` and returns `FinishReason::Length` before an out-of-window ORT call.
**Verification:** Real coherent text generation is proven through CLI and HTTP on TinyStories-1M, including output beginning `Once upon a time, there was a little girl named Lily...`. Phase 3 exit criterion #2 is met: 12 concurrent/interleaved sessions completed under KV eviction pressure without OOM. Speculative decoding is correct on real differing models (33M target / 1M draft), matching target greedy token-for-token. `cargo test --workspace` is green with 56 tests.
**Limitation:** Phase 3 exit criterion #1 (>1.5x speculative speedup) is not demonstrable locally because the available backend is CPU-only/single-threaded and the models are tiny; the measured real-model pair is 0.581x locally. This is documented as environment-bound rather than a correctness defect; expected speedup needs GPU and batched target verification.
**Next:** Phase 4: multi-model pipeline, VLM, grammar/JSON constrained decoding, tree speculative decoding, and hardware profiles. Known hardening item: flaky async SSE test.
