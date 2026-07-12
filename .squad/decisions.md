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
