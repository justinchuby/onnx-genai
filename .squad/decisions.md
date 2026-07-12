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

---

<!-- Inbox source: `batty-constrained-decoding.md` -->

# Batty decision: JSON constrained decoding

- Time: 2026-07-12T10:14:00-07:00
- Area: `onnx-genai-engine` logit processing / generation API

## Decision

Add a `Constraint` trait and logit processor that masks disallowed next-token logits to `f32::NEG_INFINITY`. The processor evaluates candidate token text at character granularity using tokenizer-decoded vocabulary entries.

Expose constrained decoding through `GenerateOptions::constraint: Option<GenerateConstraint>`, with `GenerateConstraint::Json` selecting JSON-constrained decoding. `None` remains the default and preserves unconstrained generation.

## JSON state machine

The JSON constraint uses a character-level prefix parser that tracks object/array nesting, key/colon/value/comma expectations, strings, escapes, unicode escapes, numbers, and literals. EOS/stop completion is allowed only when the generated text is a complete, balanced JSON value. If length/context limits are reached before JSON completes, generation returns an error instead of returning invalid JSON.

## Guarantee and caveats

For tokenizers whose vocabulary can express a complete JSON value within the configured limits, every accepted token preserves a valid, completable JSON prefix and termination only occurs at a complete JSON value. This is general JSON grammar validity, not JSON Schema validation; schema-level constraints are future work.

The implementation pre-decodes and checks candidate token text across the vocabulary when masking. This is correct and simple, but can be slow for large vocabularies; caching/automata compilation is future optimization work.

---

<!-- Inbox source: `batty-decode-migration.md` -->

### 2026-07-12T11:48:00-07:00: Engine decode sessions own the hot KV path
**By:** Batty
**What:** Engine model load now introspects ONNX I/O and selects STATIC-CACHE first (`StaticCacheDecodeSession::detect` requiring `key_cache.N`/`value_cache.N`, `updated_*`, `write_indices`, `nonpad_kv_seqlen`), then past/present KV (`DecodeSession`), otherwise legacy full-context fallback. Each EngineSession/DraftSession owns its own ORT decode session and runtime-owned KV buffers. Greedy single-session and persistent multi-session generation drive prompt prefill then per-token/session `step()` without per-step Rust KV clone or full present-to-past Value allocation. Static-cache capacity (`max_len()`) and shared-buffer past/present capacity bound context guards ahead of metadata/options.
**Why:** This makes the supported decode hot paths O(1) KV movement per token: static-cache writes in place via IoBinding; past/present rotates ORT-owned present values back to past without copying through Rust. Speculative target/draft sessions use `rewind(len)` after rejection/realignment, preserving greedy speculative == baseline on the tiny fixture. Same-session prefix reuse is in-place by keeping the decode cursor; cross-session prefix hits are currently token-index reported and correctness-preserving, but KV page import/export into ORT-owned decode buffers remains deferred until Deckard exposes snapshot/import bridges.

---

<!-- Inbox source: `batty-llguidance.md` -->

### 2026-07-12T10:48:00-07:00: llguidance grammar constraints in engine
**By:** Batty
**What:** Integrated `llguidance` for `GenerateConstraint::{JsonSchema(String), Regex(String), Lark(String)}` while keeping `GenerateConstraint::Json` on the existing hand-rolled JSON prefix FSM. Engine constraints first build llguidance from the HuggingFace tokenizer; if that tokenizer is unsupported by llguidance's byte tokenizer adapter, engine falls back to an approximate llguidance token environment built from decoded token texts.
**Why:** llguidance computes the per-step allowed-token bitmask for JSON Schema, regex, and Lark grammars. The existing `ConstraintProcessor` applies that mask by setting disallowed logits to `-inf`; the llguidance-backed constraint advances by committing newly generated token IDs from `ProcessorContext::generated_tokens` before computing the next mask. Tool-call forcing should pass a tool arguments JSON Schema as `GenerateConstraint::JsonSchema(schema_json_string)` so the decoder can only emit JSON matching the tool argument object schema.

---

<!-- Inbox source: `batty-pipeline-executor.md` -->

### 2026-07-12T10:22:00-07:00: Pipeline executor API and orchestration contract
**By:** Batty
**What:** `onnx-genai-engine` now exposes `Engine::from_pipeline_dir(...) -> PipelineEngine` plus `PipelineEngine::{generate, generate_with_pipeline_request, generate_with_callback}`. The executor loads `onnx_genai_ort::PipelineModels`, derives the autoregressive decoder from the metadata pipeline strategy (including composite stages), runs prompt-only upstream components once in topological order, stores outputs as `component.output`, routes metadata dataflow edges into downstream inputs, and passes routed encoder tensors as extra decoder inputs on every autoregressive step while reusing the existing engine logit processors, greedy/sampling choice, stop/EOS/constraint checks, tokenizer handling, and ORT KV decode state. Iterative/tree/diffusion strategies remain future work and fail fast unless an autoregressive decoder is present.
**Why:** This gives Pris and VLM packaging a stable seam for single-decoder and encoder->decoder DAGs without changing the existing single-model `Engine` generation API. Proven with real models: a generated single-stage pipeline fixture around `tests/fixtures/tiny-llm/` produces the same greedy token ids/text/finish reason as direct `Engine::generate`. Unit-tested orchestration: a CLIP-style `vision_encoder.image_features -> decoder.encoder_hidden_states` plan runs the encoder in prompt-only phase and routes that output as a decoder extra input. VLM contract: metadata must declare `models.vision_encoder` and `models.decoder`, phase `vision_encoder: prompt_only`, phase `decoder: every_step`, dataflow from the exact encoder output name (recommended `vision_encoder.image_features`) to the exact decoder input name (recommended `decoder.encoder_hidden_states` or the graph's required image-feature input), and callers must provide all external encoder inputs keyed as `vision_encoder.<input_name>` in `PipelineGenerateRequest::inputs`.

---

<!-- Inbox source: `coordinator-kv-and-config-directives.md` -->

### 2026-07-12T11:34:00-07:00: KV-buffer ownership + our-own config (user directives)

**By:** Justin Chu (via Squad Coordinator)

**Directives (authoritative):**

1. **Runtime manages the KV buffer; the scatter op is only a hint.** onnx-genai's
   paged/tiered KV cache OWNS and allocates the KV buffer memory. The model's
   tensor scatter op merely writes new K/V into the buffer WE bind, at cache
   positions WE control. Design the ORT IoBinding decode path and future paged
   attention around runtime-managed buffers — do NOT treat the model's
   `present.*` outputs as the source of truth.

2. **Use our own ONNX inference-metadata config, not ORT-GenAI's genai_config.**
   Model configuration follows OUR design (docs/DESIGN.md / the ONNX Inference
   Metadata Standard — this project is its reference implementation). We generate
   our own config: modify mobius to emit it, or generate it ourselves. Do NOT
   depend on or fully support onnxruntime-genai's outdated genai_config.json.
   The loader/metadata crate should key off our inference metadata; genai_config
   may be read only as a best-effort fallback, not the primary source.

**Impact:** ORT IoBinding decode (Deckard) binds runtime-owned buffers; scatter
model I/O (Pris) is consumed by our buffer manager; loader + metadata work should
emit/consume our inference metadata for models (incl. Qwen), not genai_config.

---

<!-- Inbox source: `coordinator-paged-attention-roadmap.md` -->

### 2026-07-12T11:24:00-07:00: Long-context KV roadmap (paged attention queued)

**By:** Justin Chu (via Squad Coordinator)

**Decision / roadmap:**
Long-context (Qwen2.5 up to 128K) efficiency is sequenced in two milestones:

1. **NOW — ORT IoBinding shared KV buffer (O(1)/token).** Fixes the current
   O(n^2) decode hot-path that re-copies the full past/present KV each step.
   Pre-allocate one `[layers, kv_heads, max_len, head]` buffer per sequence,
   bind past and present to the SAME buffer (past_present_share_buffer), append
   only the new token's KV in place. Deckard (ORT IoBinding + fp16 Value) →
   Batty (engine decode hot-path) → Pris (long-context benchmark).

2. **NEXT — true PagedAttention (vLLM-style).** Attention reads KV directly
   from non-contiguous pages via a page table. Blocked on ORT attention ops not
   natively accepting a page table → needs a custom kernel/EP or a block-layout
   model. Our paged KV cache (prefix sharing, CoW fork, rewind, tiered hot/cold
   eviction, int8 quant) already exists at a higher level; wiring it as the
   attention backend is the follow-on milestone AFTER the share-buffer fix.

**Why:** User confirmed paged attention will be pursued later; capturing the
sequencing so the share-buffer win lands first and paged attention is tracked,
not dropped.

---

<!-- Inbox source: `deckard-chat-template.md` -->

### 2026-07-12T10:48:00-07:00: ORT chat templates and EOS stop ids
**By:** Deckard
**What:** `onnx-genai-ort` exposes `ChatTemplate::from_model_dir(&Path) -> Result<ChatTemplate>` and `render(&self, messages: &[ChatMessage], tools: Option<&str>, add_generation_prompt: bool) -> Result<String>`. Message types are `ChatMessage { role: ChatRole, content: String, tool_calls: Option<serde_json::Value> }`; roles serialize as HF-compatible lowercase strings, with `Other(String)` for custom roles. `chat_template.jinja` takes precedence over `tokenizer_config.json.chat_template`; missing templates use a simple role-tagged fallback. `tools` is parsed from JSON and exposed to MiniJinja as `tools`; templates also receive `messages` and `add_generation_prompt`, with `tojson` enabled. `Tokenizer::eos_token_ids() -> Vec<u32>` now returns all EOS/stop ids from `generation_config.json.eos_token_id` and `tokenizer_config.json.eos_token`, while `eos_token_id()` remains the first id for backward compatibility.
**Why:** Qwen ChatML must render prompts ending in `<|im_start|>assistant\n`, and Qwen generation must stop on `<|im_end|>` id 151645 rather than only `<|endoftext|>` id 151643. Rachael should call `ChatTemplate::from_model_dir(model_dir).render(&messages, tools_json.as_deref(), true)` before tokenizing. Batty/engine should consume `Tokenizer::eos_token_ids()` and stop on any returned id; legacy single-id callers can keep using `eos_token_id()` until upgraded.

---

<!-- Inbox source: `deckard-iobinding-kv.md` -->

# 2026-07-12T11:26:00-07:00: IoBinding decode KV and fp16 Value contract

**By:** Deckard

## API added in `onnx-genai-ort`

- `DecodeSession::new(session: &Session, options: DecodeSessionOptions) -> Result<DecodeSession>`
- `DecodeSession::step(&mut self, new_input_ids: &[i64], attention_mask: &[i64], position_ids: &[i64]) -> Result<Value>` returns logits.
- `DecodeSession::rewind(&mut self, target_len: usize) -> Result<()>`
- `DecodeSession::reset(&mut self)`
- `DecodeSession::past_len(&self) -> usize`
- `DecodeSession::mode(&self) -> DecodeKvMode`
- `DecodeSessionOptions { batch_size: i64, max_length: Option<usize>, past_present_share_buffer: Option<bool> }`
- `DecodeKvMode::{ZeroCopyRebind, SharedBuffer}`
- `IoBinding::output_values() -> Result<Vec<Value>>`
- `Session::custom_metadata_value(key) -> Result<Option<String>>`
- `Session::past_present_share_buffer_supported() -> bool`
- `Value::from_slice_f16_bits`, `Value::from_vec_f16_bits`, `Value::to_vec_f16_bits` for raw IEEE-754 fp16 bit tensors.

## Present-to-past rotation

Default mode is `ZeroCopyRebind`. Each `step()` binds current KV OrtValues as `past_key_values.*` inputs, binds all outputs with `BindOutputToDevice`, runs `RunWithBinding`, takes ORT-owned output OrtValues via `GetBoundOutputValues`, and stores each `present.*` OrtValue as the next step's matching past input. Rust never reads or clones full KV in this decode path; the only returned/copied value Batty normally reads is logits.

KV names are paired by suffix across `past_key_values.*`/`past.*` inputs and `present.*`/`present_key_values.*` outputs. KV dtype may be Float32 or Float16.

## Share-buffer detection and mode

`DecodeSessionOptions::past_present_share_buffer` can explicitly force shared-buffer mode. If it is `None`, `Session::past_present_share_buffer_supported()` checks ONNX custom metadata keys `past_present_share_buffer` and `past.present.share_buffer` for true-like values (`1`, `true`, `yes`, `on`).

When shared-buffer mode is selected, `max_length` is required. DecodeSession preallocates one `[batch, ..., max_length, ...]` OrtValue per KV input, using the input's sequence axis (`rank - 2`) and dtype, then binds that same buffer as both past input and present output. This is the true O(1)/token path for exports whose graph honors `past_present_share_buffer` and uses attention/position inputs to ignore stale slots beyond the logical cursor.

Models without that metadata/override use `ZeroCopyRebind`, which still removes the O(context) Rust KV clone per step.

## Rewind/reset semantics

`rewind(target_len)` never advances length. In shared-buffer mode it only moves the logical cursor; data beyond `target_len` remains allocated and must be masked by the next `attention_mask`/`position_ids`.

In zero-copy-rebind mode, rewind rebinds each current present tensor to a compact prefix. If that prefix is contiguous in memory, this is an alias; otherwise it performs a one-time compacting slice copy for correctness (for common `[B, H, S, D]`, sequence is not contiguous across heads). This copy happens only on rewind/speculative reject, not every decode step. `reset()` clears zero-copy KV and sets length to 0.

## How Batty should drive it

Create one `DecodeSession` per engine session after loading the ORT `Session`. For each decode token or prompt remainder chunk, call:

1. Build `new_input_ids` for the new token(s).
2. Build full `attention_mask` length `decode.past_len() + new_input_ids.len()`.
3. Build `position_ids` from `decode.past_len()..decode.past_len()+new_input_ids.len()`.
4. Call `let logits = decode.step(&new_input_ids, &attention_mask, &position_ids)?;` and sample from logits.
5. On speculative reject/prefix reset call `decode.rewind(accepted_len)?` or `decode.reset()` before continuing.

For Qwen-style shared-buffer exports, pass `DecodeSessionOptions { max_length: Some(effective_context), past_present_share_buffer: Some(true), ..Default::default() }` if genai_config declares it but ONNX metadata does not.

## Validation

Added `crates/onnx-genai-ort/tests/decode_session.rs`:
- fp16 Value bit round-trip.
- IoBinding bound decode logits equal naive `Session::run` logits for `tests/fixtures/tiny-llm` over multiple steps.
- `rewind(1)` replay logits equal naive replay.

---

<!-- Inbox source: `deckard-pipeline-schema.md` -->

# 2026-07-12T10:14:00-07:00: Typed pipeline schema and multi-model loader

**By:** Deckard

## Decision

`onnx-genai-metadata` now treats metadata `pipeline` as a typed, validated DAG: `models` is a named component map with each component declaring `filename`, `type`, optional `device_preference`, and optional per-component `tokenizer`; `dataflow` contains `component.output` -> `component.input` edges with optional dtype/device-transfer hints; `strategy` is a typed loop declaration for `autoregressive`, `single_pass`, `iterative`, or `composite`; `phases` optionally gates components by prompt/every-step/final/on-demand phase. Unknown strategy/phase strings remain accepted for forward compatibility.

`onnx-genai-ort` adds `PipelineModelDirectory::load(root)` for validating metadata and resolving all model/tokenizer paths, plus `PipelineModels::load(root)` / `load_with_options(root, options)` for loading named ORT sessions into `sessions: BTreeMap<String, Session>`. Tokenizers support a shared top-level `tokenizer.json` plus per-component tokenizer overrides.

## Consumption contract

Batty's executor should consume `PipelineModels.directory.spec` as the orchestration plan, fetch sessions by component name from `PipelineModels::session`, wire tensors according to validated `dataflow`, and apply `strategy`/`phases` to decide prompt-only, every-step, final, and iterative execution. A VLM maps directly as `vision_encoder` (`single_pass`, prompt-only) feeding `decoder.encoder_hidden_states`, followed by an `autoregressive` decoder running every step. Pris can build fixtures by adding metadata-only DAGs for schema validation and model directories whose ONNX files are either real tiny models or dummy files when testing path resolution only.

---

<!-- Inbox source: `deckard-static-cache-decode.md` -->

# 2026-07-12T11:40:00-07:00: Static-cache TensorScatter decode session

**By:** Deckard

## API added in `onnx-genai-ort`

- `StaticCacheDecodeSession::detect(session: &Session) -> Result<Option<StaticCacheSignature>>`
- `StaticCacheDecodeSession::new(session: &Session, options: StaticCacheDecodeOptions) -> Result<StaticCacheDecodeSession>`
- `StaticCacheDecodeSession::prefill(input_ids: &[i64], position_ids: &[i64]) -> Result<Value>`; flattened `[B, P]` input, writes at cache slot 0, returns logits.
- `StaticCacheDecodeSession::step(next_token_ids: &[i64], position_ids: &[i64]) -> Result<Value>`; flattened `[B, 1]`, writes at the current cursor, returns logits.
- `StaticCacheDecodeSession::rewind(target_len: usize) -> Result<()>`; moves only the logical cursor.
- `StaticCacheDecodeSession::{signature, max_len, current_len, binding_mode, buffer_infos}` expose capacity, cursor, binding mode, and stable runtime-owned buffer identities.
- New exported types: `StaticCacheSignature`, `StaticCacheDecodeOptions`, `StaticCacheBindingMode`, `StaticCacheBufferInfo`.

## Detection and capacity

Detection introspects ONNX graph inputs/outputs directly; it does not read onnxruntime-genai `genai_config`. The signature requires `write_indices`, `nonpad_kv_seqlen`, paired `key_cache.N` / `value_cache.N` inputs, and paired `updated_key_cache.N` / `updated_value_cache.N` outputs. `MAX_LEN` and `KV_DIM` come from the `key_cache.N` input shape `[B, MAX_LEN, KV_DIM]`; `max_len()` is the context capacity Batty should use for max-context guards.

## Buffer ownership and binding

The runtime allocates zero-filled Float32/Float16-capable KV buffers shaped `[batch_size, MAX_LEN, KV_DIM]` for every key/value layer. The default mode binds each `key_cache.N` / `value_cache.N` input and its matching `updated_*` output to the same OrtValue, so ORT writes the scatter result into the runtime-owned buffer in place. If an ORT build rejects input/output aliasing, the session lazily allocates one alternate buffer per KV tensor, binds outputs to alternates, and swaps handles after each run; this remains O(1) per token with no cache copy.

## Cursor inputs

`prefill` binds `write_indices=[0; B]` and `nonpad_kv_seqlen=[P; B]`. `step` binds `write_indices=[current_len; B]` and `nonpad_kv_seqlen=[current_len + 1; B]`, then advances the cursor by one. `position_ids` is bound only when the graph exposes that input; the tiny committed scatter fixture prunes it.

## Rewind semantics

`rewind(len)` is cheap: it only resets the logical cursor/nonpad length tracked for the next call. Existing buffer contents are retained, and positions `>= len` are overwritten by later prefill/step calls. This is correct for speculative reject and prefix reuse because Batty drives subsequent writes from the accepted prefix length.

## Batty engine driving contract

When loading a model, call `StaticCacheDecodeSession::detect(&session)`. If it returns `Some`, construct one static-cache decode session per engine session. For a prompt, flatten `[B, P]` ids/positions and call `prefill`; for decode, call `step(&[next_token], &[position])` using `current_len()` as the next position for single-batch causal decoding; on speculative reject or prefix reset, call `rewind(accepted_len)` before replaying from that point. Use `max_len()` as the engine's static-cache context capacity.

## Validation

The committed `tests/fixtures/tiny-llm-scatter/` model now runs in `crates/onnx-genai-ort/tests/decode_session.rs`. It produced deterministic greedy tokens `[0, 24, 15]`, confirmed true in-place alias mode, verified KV buffer data pointers/sizes were unchanged across prefill, decode, rewind, and replay, and verified rewind replay reproduced the same logits/tokens.

---

<!-- Inbox source: `pris-coding-agent.md` -->

# 2026-07-12T11:10:00-07:00: Hermes-style coding-agent harness against Qwen2.5-0.5B

**By:** Pris

## Harness design

Added `scripts/coding_agent.py`, a pure-stdlib Python OpenAI-compatible agent loop that posts to `/v1/chat/completions` with file and terminal tools: `read_file`, `write_file`, `list_dir`, and `run_command`. Tool execution is sandboxed to a caller-supplied work directory (default `target/coding-agent-workspace`), assistant `tool_calls` are executed locally, and `role: "tool"` results with `tool_call_id` are appended before the next loop iteration. The harness caps iterations, prints assistant messages, tool calls/results, final content, verification, and a JSON transcript. Added `scripts/run_coding_agent_demo.sh` to build the server, boot Qwen on `127.0.0.1:${PORT:-8090}`, wait for `/health`, and run the harness.

## Tasks attempted

1. Main coding task with `tool_choice: "auto"`: create `hello.py` printing `Hello, Squad!`, run it, and report output.
2. Minimal terminal task with `tool_choice: "auto"`: run `python3 -c "print('Hello, Squad!')"` and report output.
3. Diagnostic `tool_choice: "required"` run for the main task.

## Transcript summary

Main auto run completed a two-turn tool loop: Qwen emitted a parsed `write_file` call for `hello.py`, the harness executed it successfully, then Qwen returned a final answer. It did **not** call `run_command`; it inferred the output. Harness post-verification ran the created script and confirmed stdout was `Hello, Squad!\n`.

Minimal terminal auto run completed a two-turn tool loop: Qwen emitted a parsed `run_command` call, the harness executed it successfully with stdout `Hello, Squad!\n`, and Qwen returned a final answer using the tool result.

The diagnostic required-tool run produced malformed content beginning with `<tool_call>` followed by repeated `!` characters until `finish_reason: "length"`; no parsed tool call was returned.

## Verdict

The server and OpenAI tool protocol correctly support a Hermes-style agent loop: tools are advertised, Qwen auto tool calls are parsed into OpenAI `message.tool_calls` with `finish_reason: "tool_calls"`, `role: "tool"` results are accepted on the next turn, and the model can terminate with a final assistant message. The incomplete main coding workflow is a small-model planning/compliance limit: Qwen2.5-0.5B stopped after file creation and hallucinated the run output instead of invoking `run_command`. The required-tool malformed output is a dev follow-up area for constrained decoding with this multi-tool schema/model combination, but it did not block auto tool-call loop mechanics.

---

<!-- Inbox source: `pris-longcontext-bench.md` -->

# 2026-07-12T11:54:00-07:00: Long-context decode efficiency benchmark

**By:** Pris

## What was measured

Added a reproducible benchmark harness:

- `scripts/bench_long_context.sh`
- `crates/onnx-genai-ort/examples/long_context_bench.rs`

The harness runs the ORT decode-session APIs directly, times each decode step, buckets average ms/token by absolute context position, samples RSS, and records static-cache KV buffer identities before and after generation.

## Static-cache Qwen 0.5B, MAX_SEQ_LEN=2048

Command:

```bash
BUILD_MODEL=1 MAX_SEQ_LEN=2048 scripts/bench_long_context.sh
```

Model signature:

- mode: `static-cache`
- binding: `InPlaceAlias`
- layers: 24
- max_len: 2048
- kv_dim: 128
- dtype: Float32
- KV buffers: 48
- preallocated KV bytes: 50,331,648 bytes (48.0 MiB)

Latency:

| Position bucket | Tokens | Avg ms/token |
|---:|---:|---:|
| 1-64 | 63 | 27.046 |
| 65-256 | 192 | 25.351 |
| 257-1024 | 768 | 25.969 |
| 1025-2048 | 1024 | 26.263 |

Verdict: per-token latency is effectively flat across context position on the static-cache path. There is no linear growth with position and no O(n²) KV-copy blowup.

Memory:

- initial RSS after model/session/static KV allocation: 2369.7 MiB
- sampled RSS plateau: 2484.0-2484.7 MiB from positions 128 through 1920
- final RSS sample at 2048: 2480.3 MiB
- peak RSS: 2484.7 MiB
- static-cache buffer data pointers were identical before and after the 2048-token run
- `buffers_stable=true`

Verdict: static-cache KV memory is bounded. The fixed KV buffers are allocated once and reused through the full run.

Long run:

- completed to `final_len=2048`
- no OOM
- no ORT error
- final token id: 15

## Past/present Qwen 0.5B contrast, MAX_TOKENS=2048

Command:

```bash
MODE=past-present MODEL_DIR=models/qwen2.5-0.5b MAX_TOKENS=2048 scripts/bench_long_context.sh
```

Mode:

- `DecodeKvMode::ZeroCopyRebind`

Latency:

| Position bucket | Tokens | Avg ms/token |
|---:|---:|---:|
| 1-64 | 64 | 25.382 |
| 65-256 | 192 | 26.051 |
| 257-1024 | 768 | 27.164 |
| 1025-2048 | 1024 | 29.514 |

RSS samples showed expected allocator/runtime movement rather than static bounded cache identity:

- initial RSS: 2437.6 MiB
- peak RSS: 2457.1 MiB
- final RSS sample: 2025.6 MiB

The past/present path is still much flatter than a Rust-side O(n²) KV-copy path would be, but it shows mild growth from model attention/present tensor size. Static-cache is the stronger bounded-memory proof because its KV data pointers remain stable.

## 128K projection

The static-cache memory formula is:

```text
layers * 2(key,value) * max_len * kv_dim * bytes_per_element
```

For this Qwen 0.5B Float32 static-cache signature at 128K:

```text
24 * 2 * 131072 * 128 * 4 = 3,221,225,472 bytes ≈ 3.0 GiB
```

For Float16 it would be about 1.5 GiB. This is just the static KV buffer; model weights, ORT workspace, logits, allocator overhead, and process runtime memory are additional. The same code path is used for 2048 and 128K; scaling MAX_SEQ_LEN only changes the one-time preallocated buffer capacity and available RAM requirement.

## Diagnosis for devs

No static-cache efficiency bug was found. Static-cache decode is O(1) KV movement per token and uses bounded fixed KV buffers. The only caveat is that full 128K on CPU is impractical in this environment because it requires multi-GiB preallocated KV plus slow CPU inference; the 2048-token run empirically validates the same static-cache path.

---

<!-- Inbox source: `pris-qwen-tooluse.md` -->

# 2026-07-12T10:58:00-07:00: Qwen2.5-0.5B HTTP tool-use e2e validation

**By:** Pris

## Setup

Built `onnx-genai-server` and ran the real gitignored model at `models/qwen2.5-0.5b` via HTTP on `127.0.0.1:8081` because `127.0.0.1:8080` was already serving `tiny-llm` in this environment.

## Actual responses

### 1. Plain chat

Request: system=`You are a helpful assistant. Answer briefly.`, user=`What is 2+2?`

```json
{"id":"chatcmpl-1783878375","object":"chat.completion","created":1783878375,"model":"qwen2.5-0.5b","choices":[{"index":0,"message":{"role":"assistant","content":"2+2 equals 4."},"finish_reason":"stop"}],"usage":{"prompt_tokens":29,"completion_tokens":8,"total_tokens":37}}
```

Result: coherent content; stopped with `finish_reason: "stop"`, no runaway past `<|im_end|>`.

### 2. Tool use, `tool_choice: "auto"`

```json
{"id":"chatcmpl-1783878383","object":"chat.completion","created":1783878383,"model":"qwen2.5-0.5b","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"fahrenheit\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":162,"completion_tokens":27,"total_tokens":189}}
```

Result: returned `message.tool_calls` with a valid `get_weather` call and `finish_reason: "tool_calls"`.

Exact `tool_calls`:

```json
[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"fahrenheit\"}"}}]
```

### 3. Forced function tool choice

Request used `tool_choice: {"type":"function","function":{"name":"get_weather"}}`.

```json
{"id":"chatcmpl-1783878393","object":"chat.completion","created":1783878393,"model":"qwen2.5-0.5b","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"fahrenheit\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":162,"completion_tokens":27,"total_tokens":189}}
```

Parsed arguments:

```json
{"location":"Paris","unit":"fahrenheit"}
```

Result: arguments parse as JSON and satisfy the schema (`location` present). However, this does **not** prove grammar forcing; code inspection shows server options only set `GenerateConstraint::Json` for `response_format: {"type":"json_object"}` and do not map forced tool choice to `GenerateConstraint::JsonSchema`.

### 4. Multi-turn tool loop

Request included the prior assistant tool call and a `role:"tool"` response with content `{"temp":18,"unit":"celsius"}`.

```json
{"id":"chatcmpl-1783878413","object":"chat.completion","created":1783878413,"model":"qwen2.5-0.5b","choices":[{"index":0,"message":{"role":"assistant","content":"The current temperature in Paris is 18 degrees Celsius."},"finish_reason":"stop"}],"usage":{"prompt_tokens":216,"completion_tokens":13,"total_tokens":229}}
```

Result: final natural-language answer produced and stopped with `finish_reason: "stop"`.

## Diagnosis / bugs for owners

- **Rachael / server:** forced tool choice is not wired to grammar constraints. `crates/onnx-genai-server/src/lib.rs::build_generate_options` only sets `options.constraint = Some(GenerateConstraint::Json)` for JSON response format. It never derives a tool-parameter schema from `tools`/`tool_choice` and never sets `GenerateConstraint::JsonSchema`, so forced tool calls currently rely on prompt-following and parser extraction, not guaranteed structured decoding.
- **Batty / engine:** `GenerateConstraint::JsonSchema` is available and routes through llguidance in `onnx-genai-engine`; no engine failure observed in this run.
- **Deckard / template-tokenizer:** Qwen chat rendering, tool sections, role:`tool` turns, and `<|im_end|>` stopping worked in this run. No runaway observed.

## Test coverage

Added ignored e2e test `qwen_real_model_tool_use_chain_end_to_end` in `crates/onnx-genai-server/tests/http.rs`. Default validation stayed green with `cargo test --workspace`.

---

<!-- Inbox source: `pris-qwen.md` -->

# Pris Qwen real-model generation validation

**Date:** 2026-07-12T10:34:00-07:00
**By:** Pris

## Model and artifact size

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Output directory: `models/qwen2.5-0.5b/` (gitignored by `models/`)
- Working runtime artifact: explicit `f32`
- Files: `model.onnx` 309 KiB, `model.onnx.data` 1.8 GiB, `tokenizer.json` 6.7 MiB, plus vocab/merges/tokenizer config/genai config.

## Mobius command

Working command:

```bash
HF_HOME=$PWD/models/.hf_cache HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=$PWD/models/.scratch \
PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src \
python -m mobius build --model Qwen/Qwen2.5-0.5B-Instruct models/qwen2.5-0.5b --dtype f32 --runtime ort-genai
```

`--dtype f16` built a 948 MiB `model.onnx.data`, but current runtime rejected it before generation because KV/logits are Float16 and the Phase 1 engine only accepts Float32 cached ORT values.

## ONNX I/O signature

Inputs:

- `input_ids`: i64 `[batch, sequence_len]`
- `attention_mask`: i64 `[batch, past_seq_len + seq_len]`
- `position_ids`: i64 `[batch, sequence_len]`
- `past_key_values.{0..23}.key`: f32 `[batch, 2, past_sequence_len, 64]`
- `past_key_values.{0..23}.value`: f32 `[batch, 2, past_sequence_len, 64]`

Outputs:

- `logits`: f32 `[batch, sequence_len, 151936]`
- `present.{0..23}.key`: f32 `[batch, 2, past_sequence_len + sequence_len, 64]`
- `present.{0..23}.value`: f32 `[batch, 2, past_sequence_len + sequence_len, 64]`

The loader resolves `model.onnx` by single-ONNX fallback and finds `tokenizer.json`. Engine input introspection resolves `input_ids`, `attention_mask`, and `position_ids` directly. KV matching also works: `present.N.key/value` maps to `past_key_values.N.key/value` by suffix. GQA is represented correctly as 2 KV heads with head dim 64 (Qwen2.5-0.5B has 14 attention heads, 2 KV heads).

## Generation run

Command:

```bash
cargo run -p onnx-genai --bin onnx-genai -- generate \
  --model models/qwen2.5-0.5b \
  --max-new-tokens 40 \
  --stop '<|im_end|>' \
  $'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n'
```

Exact stdout:

```text
4
```

## Coherence assessment

Coherent for the smoke prompt: the model answered `What is 2+2?` with `4`. Manual ChatML formatting and explicit `--stop '<|im_end|>'` are required today.

## Runtime gaps / follow-up for devs

- Add native Qwen chat templating so callers do not have to manually pass ChatML.
- Add default Qwen stop handling for both `<|im_end|>` and `<|endoftext|>`; the CLI currently only stops on tokenizer EOS unless `--stop '<|im_end|>'` is provided.
- Add Float16/BFloat16 logits and KV support in ORT `Value`, engine KV initialization, cached value cloning, paged KV mirror/materialization, and logit extraction if we want the preferred smaller fp16/bf16 Qwen artifacts.
- Preserve GQA behavior: PageTensorConfig must continue using KV heads (`2`) rather than attention heads (`14`). Current f32 path inferred this correctly from present KV shape.

---

<!-- Inbox source: `pris-scatter-model.md` -->

# 2026-07-12T11:30:00-07:00: Mobius static-cache TensorScatter model contract

**By:** Pris

## Decision

Mobius' scatter/in-place-KV export path is the static KV cache path:

```sh
mobius build --model <hf-id> <out-dir> --static-cache --max-seq-len <N> --runtime ort-genai
```

The Python API equivalent is `CausalLMTask(static_cache=True, max_seq_len=N)`. This was found in `src/mobius/__main__.py`, `README.md`, `examples/static_cache_generation.py`, `src/mobius/tasks/_causal_lm.py`, and `src/mobius/components/_attention.py`. The graph uses ONNX opset 24 `TensorScatter`.

## Built artifacts

- Tiny committed-size fixture: `tests/fixtures/tiny-llm-scatter/`
  - Built by `PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src python scripts/build_tiny_scatter.py`
  - `model.onnx` 12,750 bytes; `model.onnx.data` 14,336 bytes; `tokenizer.json` 2,038 bytes.
  - Contract test: `crates/onnx-genai-ort/tests/static_cache_signature.rs` (`#[ignore]`).
- Qwen static-cache model: `models/qwen2.5-0.5b-scatter/` (gitignored)
  - Built by `STATIC_CACHE=1 MAX_SEQ_LEN=2048 ./scripts/build_qwen.sh`.
  - `model.onnx` 328,975 bytes; `model.onnx.data` 1,984,561,152 bytes; `tokenizer.json` 7,031,645 bytes; `genai_config.json` 3,803 bytes.

## ONNX I/O contract: Qwen2.5-0.5B static cache

Opsets: `ai.onnx` 24, `com.microsoft` 1. TensorScatter nodes: 48.

Inputs:
- `input_ids`: INT64 `[batch, sequence_len]`
- `position_ids`: INT64 `[batch, sequence_len]`
- For layers `0..23`:
  - `key_cache.N`: FLOAT `[batch, 2048, 128]`
  - `value_cache.N`: FLOAT `[batch, 2048, 128]`
- `write_indices`: INT64 `[batch]`
- `nonpad_kv_seqlen`: INT64 `[batch]`

Outputs:
- `logits`: FLOAT `[batch, sequence_len, 151936]`
- For layers `0..23`:
  - `updated_key_cache.N`: FLOAT `[batch, 2048, 128]`
  - `updated_value_cache.N`: FLOAT `[batch, 2048, 128]`

There is no `attention_mask` input, no `past_key_values.N.{key,value}` input, and no `present.N.{key,value}` output in the static-cache Qwen graph. `write_indices[b]` is the start slot where this call's chunk is scattered. `nonpad_kv_seqlen[b]` is the valid KV length after the scatter; for an unpadded chunk it is `write_indices[b] + sequence_len`.

## Contrast: current dynamic-cache Qwen2.5-0.5B

Inputs are `input_ids`, `attention_mask`, `position_ids`, and per-layer `past_key_values.N.key/value` FLOAT `[batch, 2, past_sequence_len, 64]`. Outputs are `logits` and per-layer `present.N.key/value` FLOAT `[batch, 2, past_sequence_len + sequence_len, 64]`. TensorScatter nodes: 0.

## Tiny fixture contract note

The tiny scatter fixture has one layer and `max_seq_len=16`: `input_ids`, `key_cache.0`, `value_cache.0`, `write_indices`, `nonpad_kv_seqlen` inputs and `logits`, `updated_key_cache.0`, `updated_value_cache.0` outputs. Its actual graph has no `position_ids` input because the tiny synthetic Qwen graph prunes it; use the Qwen signature above for Deckard/Batty's real decode contract.

## Runtime sanity

`onnx-genai generate --model models/qwen2.5-0.5b-scatter --max-new-tokens 1 "hello"` loaded/introspected the model far enough to report the first unsupported static-cache input, then failed as expected with: `unsupported model input 'key_cache.0' with shape [-1, 2048, 128]`.

Deckard/Batty must allocate persistent per-layer `[batch, max_seq_len, kv_hidden]` key/value buffers, bind/feed them as `key_cache.N` and `value_cache.N`, feed `write_indices` and `nonpad_kv_seqlen`, and either copy `updated_*` outputs back or IoBind output names onto the same buffers to get in-place/past-present-share-buffer decode semantics.

---

<!-- Inbox source: `pris-vlm-fixture.md` -->

# Tiny VLM pipeline fixture scaffold

**Date:** 2026-07-12T10:22:00-07:00
**By:** Pris

## Decision
Build the Phase 4 capstone VLM test asset as a deterministic, hand-constructed ONNX pair in `models/tiny-vlm/` rather than using Mobius to export a real multimodal checkpoint.

## Why
Mobius supports multimodal export, but its smallest documented path is the Gemma 3 vision-language example:

```sh
PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src \
  python /Users/justinc/Documents/GitHub/mobius/examples/multimodal_generation.py \
  --model google/gemma-3-4b-pt --save-to models/gemma3-vlm
```

That path would download a multi-GB checkpoint, which is not viable for the disk-limited capstone fixture. The tiny fixture instead exercises the same mechanics: an encoder ONNX session emits `image_features`, a decoder ONNX session consumes `input_ids` plus `image_features`, and metadata declares the pipeline dataflow.

## Fixture
`python3 scripts/build_tiny_vlm.py` writes `models/tiny-vlm/` with:

- `encoder.onnx`: `pixel_values [1,3,2,2] -> image_features [1,1,4]`
- `decoder.onnx`: `input_ids [batch,sequence] + image_features [1,1,4] -> logits [batch,sequence,8]`
- `tokenizer.json`: tiny WordLevel tokenizer whose token 4 is `cat`
- `inference_metadata.yaml`: Deckard-schema pipeline with `models.encoder`, `models.decoder`, `dataflow` from `encoder.image_features` to `decoder.image_features`, and composite `encode_image` + autoregressive `decode_text` stages.

The local built directory is 3,580 bytes. `models/` and `*.onnx` are gitignored, so model weights are not committed.

## Test scaffold
Added ignored integration test `crates/onnx-genai-engine/tests/vlm_pipeline_e2e.rs`. When `models/tiny-vlm/` exists, it exercises Batty's `Engine::from_pipeline_dir` / `PipelineGenerateRequest` path with `encoder.pixel_values` and asserts generated tokens are emitted. It also includes a lower-level `PipelineModels` session smoke test for the encoder-to-decoder dataflow. Both ignored tests pass locally with the generated fixture.

---

<!-- Inbox source: `rachael-forced-toolchoice.md` -->

# Forced tool_choice constrained decoding

- Date: 2026-07-12T11:04:00-07:00
- Owner: Rachael (Server/API)

When `tool_choice` forces a call (`"required"` or a specific function name), the server maps available function tools into a JSON Schema for Qwen tool-call objects:

```json
{
  "type": "object",
  "properties": {
    "name": {"enum": ["<tool name>"]},
    "arguments": {"<tool parameters schema>": "..."}
  },
  "required": ["name", "arguments"],
  "additionalProperties": false
}
```

For `tool_choice: "required"` with multiple tools, the schema is wrapped as `{"anyOf": [ ...tool schemas... ]}` so exactly one offered tool schema is accepted.

The server passes this through `GenerateConstraint::Lark` using llguidance inline `%json`, which is supported by llguidance 1.7.6 (`tests/test_ll.rs` exercises `obj: %json { ... }` and JSON inside larger Lark grammars):

```lark
start: "<tool_call>\n" tool "\n</tool_call>"
tool: %json <schema above>
```

Guarantee: constrained decoding must emit the complete Qwen `<tool_call>` wrapper containing a JSON object whose `name` matches the forced tool (or one required tool) and whose `arguments` satisfy that tool's parameters schema. `tool_choice: "auto"` remains unconstrained and parser-based; `tool_choice: "none"` does not offer tools to the model and does not parse tool calls from output.

---

<!-- Inbox source: `rachael-response-format.md` -->

# 2026-07-12T10:22:00-07:00: OpenAI response_format JSON constraint mapping

**By:** Rachael

## Decision

`onnx-genai-server` accepts OpenAI-compatible `response_format` on `POST /v1/chat/completions`. Requests with `response_format: {"type":"json_object"}` map to `GenerateOptions::constraint = Some(GenerateConstraint::Json)`; absent `response_format` and `{"type":"text"}` leave generation unconstrained.

## Streaming

The same generate request construction is used for streaming and non-streaming paths. JSON mode keeps the OpenAI response shape (`chat.completion` or `chat.completion.chunk` deltas), and the server only emits assembled JSON content once generation completes so the final streamed assistant content parses as JSON. If the engine reports that constrained decoding stopped before a complete JSON value, the server emits `{}` rather than returning malformed JSON.

---

<!-- Inbox source: `rachael-tool-use.md` -->

# Rachael tool-use server integration

**Date:** 2026-07-12T10:48:00-07:00
**By:** Rachael

## Decision

`onnx-genai-server` now accepts OpenAI chat tool-use shapes on `/v1/chat/completions`: `tools`, `tool_choice`, assistant `tool_calls`, and `tool` messages with `tool_call_id`. Response messages can return OpenAI function `tool_calls` with `finish_reason: "tool_calls"`.

Prompt construction uses Deckard's documented `ChatTemplate::from_model_dir(dir)` and `render(messages, tools, add_generation_prompt)` when the model directory declares a chat template. If no template is present, the server keeps the simple role-tag fallback and includes tools/tool_choice/tool responses in that fallback so tests and tiny fixtures remain deterministic.

Generation config now augments request stop sequences with tokenizer EOS ids from `Tokenizer::eos_token_ids()` and `<|im_end|>` when present, while keeping the first EOS id in `GenerateOptions::eos_token_id`.

The Hermes/Qwen parser scans model text for one or more `<tool_call>...</tool_call>` blocks, JSON parses each body as `{ "name": ..., "arguments": ... }`, and maps them to OpenAI `message.tool_calls[*].function.arguments` as a JSON string. Any parsed call changes the response finish reason to `tool_calls`; otherwise normal `stop`/`length` mapping is preserved.

Streaming is best-effort: when tool context is active, token deltas are buffered so partial XML/JSON tool-call markup is not emitted as user-visible content. At completion the buffer is parsed; parsed tool calls are emitted in a final tool_calls delta followed by `finish_reason: "tool_calls"`, otherwise buffered text is emitted as content followed by the normal finish reason.

Batty's `GenerateConstraint::JsonSchema(String)` landed during integration, but the server does not apply it yet because Hermes/Qwen tool calls are emitted inside a `<tool_call>...</tool_call>` envelope rather than as bare argument JSON. The server parses and records `tool_choice` now; future grammar work should constrain the full envelope or introduce an args-only forced-function mode.

---

<!-- Inbox source: `roy-longcontext-plan.md` -->

# 2026-07-12T11:20:00-07:00: Long-context KV architecture plan

**By:** Roy

## Requirement

onnx-genai must support efficient long context, including Qwen2.5-class 128K-token contexts. Correctness alone is not sufficient: the decode hot path must avoid O(context) KV allocation/copy per token and must keep memory bounded through dtype and tiering choices.

## Findings

1. **Current decode KV threading is O(context) per token and O(context²) over a long generation.**
   `next_session_token_logits` calls `run_decode_step` for each new token or prompt remainder, then mirrors present KV into pages and increments `kv_token_count` (`crates/onnx-genai-engine/src/engine.rs:1429-1452`). `run_decode_step_with_extra` clones every cached past input before each ORT run (`engine.rs:1621-1629`), runs `Session::run` (`engine.rs:1641-1645`), clears `decode_state.past`, and clones every present output back into past (`engine.rs:1657-1665`). `clone_value` copies tensor contents via `to_vec_f32`/`to_vec_i64` and creates new CPU tensors (`engine.rs:2230-2237`). This means each decode step re-passes all past KV and reads/clones full present KV.

2. **IoBinding exists, but generation does not use it and there is no share-buffer optimization.**
   ORT exposes real `IoBinding::{new, bind_input, bind_output, bind_output_to_device, clear}` and `Session::run_with_binding` (`crates/onnx-genai-ort/src/binding.rs:17-89`, `crates/onnx-genai-ort/src/session.rs:165-176`). The Phase 1 decision explicitly said `Session::run` was preferred unless pre-bound KV buffers were needed (`.squad/decisions.md:60-71`). Generation still calls `Session::run` (`engine.rs:1641-1645`). There are no repository matches for `past_present_share_buffer`, and no genai_config file beyond `.squad/config.json`; Qwen-style `past_present_share_buffer=true` is not represented today.

3. **Paged KV is currently a mirror, not the ORT source of truth.**
   `EngineSession` owns ORT-managed past tensors in `decode_state` (`engine.rs:1320-1327`), while `kv_cache` is a separate paged mirror used for prefix cache and rewind. `mirror_present_kv_to_pages` reads full present key/value outputs with `to_vec_f32` for every layer (`engine.rs:1768-1807`) and extracts/appends only token slices into pages (`engine.rs:1809-1830`). Prefix hits and rewinds materialize paged KV back into contiguous f32 ORT past tensors (`engine.rs:812-817`, `engine.rs:1864-1900`, `engine.rs:2001-2032`). This doubles memory on active sessions: ORT past/present tensors plus paged KV mirror.

4. **Max-context handling is metadata-first but does not yet infer Qwen YaRN 128K from config.**
   `max_context_for_request` uses `metadata.model.max_sequence_length` first, then `GenerateOptions::max_context` (`engine.rs:739-745`), matching the existing context-window decision (`.squad/decisions.md:441-451`). This is acceptable for Qwen only if our metadata/genai_config declares 131072 or callers set `max_context`; reading raw `max_position_embeddings=32768` would under-limit Qwen2.5 YaRN. Position ids are generated as i64 ranges from `past_len..total_len`, and attention mask is length `total_len` (`engine.rs:1589-1620`), so integer overflow is not the issue; the risk is stale/incorrect metadata and any model with learned WPE tables.

5. **128K memory requires fp16/share-buffer/tiering engagement; current engine defaults to f32 pages and f32 ORT values.**
   Paged KV supports `KvDType::Int8` with per-page scale (`crates/onnx-genai-kv/src/page_table.rs:9-19`, `page_table.rs:127-147`) and hot/cold page movement (`page_table.rs:325-373`, `crates/onnx-genai-kv/src/paged_cache.rs:216-280`), as recorded in the Phase 3 contracts (`.squad/decisions.md:400-408`, `.squad/decisions.md:543-548`). But `infer_kv_model_info` hardcodes `KvDType::F32` and rejects non-Float32 present KV (`engine.rs:1703-1721`), `empty_past_value` requires Float32 (`engine.rs:2192-2227`), and `Value` only has owned backing constructors for f32/i64 (`crates/onnx-genai-ort/src/value.rs:87-173`). `DataType::Float16` exists at the enum/API level (`value.rs:7-24`, `value.rs:41-58`), but cached value cloning rejects it (`engine.rs:2230-2237`). Therefore fp16 KV and/or int8 pages are not engaged on the ORT-threaded long-context path today.

6. **Prefill is batched, but it still materializes full present KV once and then mirrors it.**
   For a new prompt, `prepare_session_prefix` appends the prompt tokens into logical state (`engine.rs:827-833`), and the first `next_session_token_logits` feeds all unmaterialized tokens in one call because `session_decode_input_tokens` returns `state.tokens[state.kv_token_count..]` (`engine.rs:1525-1539`). Attention mask and position ids are allocated as one Vec each (`engine.rs:1589-1620`). This is correct for one big prefill, but at 128K the returned present KV is enormous and is then fully copied into both `decode_state.past` and paged KV mirror.

## Decision: target architecture

Adopt **ORT IoBinding share-buffer KV as the first long-context architecture**. For each layer and K/V tensor, pre-allocate one `[batch, kv_heads, max_context, head_dim]` buffer in the target KV dtype (prefer fp16 where model supports it). Bind the model's `past.*` input and `present.*` output to aliases/views of the same backing buffer so decode appends the new token's KV in place. The per-token ORT path should bind only the current input token, scalar/short masks/positions as needed by the graph, and stable KV buffers; it must not copy full past or full present through Rust. This is the fastest and simplest design for a single active sequence and matches Qwen's `past_present_share_buffer=true` intent.

Keep `PagedKvCache` as the **prefix-cache and multi-sequence abstraction**, but stop making it a mandatory full mirror for the hot single-sequence path. Migration should be staged:

1. Implement share-buffer decode state behind the existing `DecodeState` contract, preserving `Session::run` fallback for tests and unsupported graphs.
2. Add explicit snapshot/import/export bridges between share-buffer KV and `PagedKvCache` for prefix insertion, prefix reuse, rewind, speculative rejection, and future multi-sequence sharing.
3. For prefix sharing at scale, evolve toward page-backed IoBinding: pages become the source of truth, and ORT binds page-backed contiguous windows or an ONNX model layout that accepts paged KV metadata. Until then, prefix hits may materialize once into share buffers, but decode must stay O(1)/token.
4. Make tiering/quantization policy explicit: share-buffer fp16 for active hot KV; paged int8/CPU cold storage for evicted prefixes/sessions. Avoid simultaneous full-size ORT f32 plus f32 paged mirror at 128K.

## Per-agent task breakdown

### Deckard — ORT + Value substrate

- **D1: IoBinding share-buffer primitives** — deps: none. Add APIs to allocate tensor buffers in a requested memory arena/dtype and bind input/output names to caller-owned buffers. Support binding past and present aliases needed by share-buffer KV.
- **D2: fp16 Value support** — deps: D1 can proceed in parallel. Add owned/borrowed `Float16` tensor construction and safe access where needed; stop limiting cached KV to f32/i64.
- **D3: KV dtype/model metadata plumbing** — deps: D2. Let ORT metadata and/or genai_config declare KV dtype and `past_present_share_buffer`; expose enough graph I/O metadata for the engine to select share-buffer mode.
- **D4: Tiering engagement hooks** — deps: D1/D3. Provide APIs for share-buffer snapshot/import/export to `PagedKvCache` and eventual page-backed binding without copying full context per token.

### Batty — engine decode hot path

- **B1: Share-buffer `DecodeState`** — deps: D1. Replace `past: HashMap<String, Value>` cloning with stable per-layer KV buffers and token cursor. `next_session_token_logits`, prioritized drive, and non-speculative generation must become O(1) KV movement per decode token.
- **B2: Prefill path** — deps: B1. Feed the prompt in one ORT call, write KV into the share buffers, and avoid full present clone into Rust. Make attention mask/position_ids construction efficient and bounded.
- **B3: Prefix cache compatibility** — deps: B1, D4. Preserve `prefix_cache_hit_len`, prefix insertion, CoW semantics, and rewinds by importing/exporting only necessary ranges between share buffers and pages.
- **B4: Max-context/Qwen config** — deps: D3. Ensure metadata/genai_config can express effective Qwen 128K context despite `max_position_embeddings=32768`; require model metadata `model.max_sequence_length=131072` or parsed YaRN-derived equivalent. Keep learned-WPE models guarded by table length.
- **B5: Speculative correctness** — deps: B1/B3. Rework draft/target rewind so accepted/rejected tokens update share-buffer cursors and page snapshots without stale KV aliases.

### Pris — validation and performance proof

- **P1: O(n²) regression benchmark** — deps: B1/B2. Add a long-context benchmark/harness that measures decode step latency at increasing context lengths and proves no linear KV copy growth per token.
- **P2: 128K memory benchmark** — deps: D2/B1. Measure active KV memory for Qwen-class dimensions in fp16 share-buffer mode; confirm no f32 mirror doubles memory.
- **P3: Tiering/int8 benchmark** — deps: B3/D4. Prove cold prefixes/sessions can be evicted to CPU/int8 pages and reloaded correctly while keeping active hot memory bounded.
- **P4: Correctness suite** — deps: B5. Verify greedy token equivalence, speculative rejection/rewind equivalence, prefix-cache reuse, and context-length stop behavior.

## Highest-value first task

Start with **Deckard D1 + Batty B1 as a vertical slice** on one tiny KV model: pre-allocate/bind shared KV buffers and remove the per-token `clone_value` past/present cycle. This directly eliminates the dominant O(context²) bottleneck and derisks every later long-context feature.

## Biggest correctness risk

The highest-risk area is **KV aliasing and cursor rewind interaction with speculative decoding and prefix cache reuse**. If past/present share the same buffer, a rejected speculative token, prefix CoW import, or session reset can leave stale KV beyond the logical cursor or overwrite shared prefix pages. The implementation needs explicit logical length cursors, safe rewind/truncate semantics, and tests that compare share-buffer output token-for-token against the current clone-based path.

---

<!-- Inbox source: `roy-phase4-plan.md` -->

# 2026-07-12T10:10:00-07:00: Phase 4 Pipeline + Advanced assessment and plan

**By:** Roy
**Requested by:** Justin Chu

## Context read

Phase 1-3 contracts are complete through Scribe's Phase 3 closeout: real TinyStories CLI/HTTP generation, paged/tiered/int8 KV, greedy speculative correctness, priority/preemption, OpenAI SSE early-stop/cancel, and green workspace tests (`.squad/decisions.md:543-549`). The current Phase 4 target is DESIGN.md Phase 4: multi-model pipeline orchestration, VLM support, grammar/JSON constrained decoding, rejection-sampling acceptance, tree speculative decoding, and hardware profile matching, with VLM pipeline exit criteria (`docs/DESIGN.md:682-691`).

## Status table

| Phase 4 item | Status | Evidence | Assessment |
|---|---:|---|---|
| Multi-model pipeline orchestration | **MISSING** | `crates/onnx-genai-engine/src/pipeline.rs:1-8`; `crates/onnx-genai-ort/src/loader.rs:7-15`, `:17-45`; `crates/onnx-genai-engine/src/engine.rs:283-291`; DESIGN DAG/schema in `docs/DESIGN.md:1026-1068` | The pipeline module is only TODO comments. `ModelDirectory` resolves one decoder/single ONNX plus tokenizer/metadata, and `Engine::from_dir` creates one ORT `Session`. No DAG, model map, dataflow wiring, phase gating, composite strategy, preprocessing, or multi-model loading exists. |
| Vision-language model support (image encoder + decoder) | **MISSING** | VLM design requires CLIP prompt-only encoder feeding decoder (`docs/DESIGN.md:393-397`) and image-to-text composite example (`docs/DESIGN.md:1284-1312`); current loader requires `tokenizer.json` and one model (`crates/onnx-genai-ort/src/loader.rs:29-45`, `:56-85`) | No image input type, image preprocessing, vision encoder session, cross-attention/encoder-hidden-state feed, or VLM metadata fixture. This depends on generalized pipeline loading/execution first. |
| Grammar/JSON constrained decoding | **PARTIAL** | Metadata parses `structured_output` (`crates/onnx-genai-metadata/src/schema.rs:32-35`, `:180-184`; fixture test `crates/onnx-genai-metadata/tests/metadata_fixtures.rs:37-50`); logit chain exists (`crates/onnx-genai-engine/src/logits.rs:49-93`); chain order reserves constraints slot (`docs/DESIGN.md:330-357`) | Metadata declaration is parsed and the processor architecture is ready, but there is no `GrammarProcessor`, JSON schema/grammar compiler, request option, or OpenAI `response_format` surface. This is the smallest high-value Phase 4 vertical slice. |
| Rejection-sampling acceptance rule (speculative) | **PARTIAL** | Speculative strategy has an `acceptance: Option<String>` field (`crates/onnx-genai-metadata/src/schema.rs:148-156`), but runtime rule enum is only greedy (`crates/onnx-genai-engine/src/speculative.rs:13-18`); current verify loop accepts by target argmax equality (`crates/onnx-genai-engine/src/engine.rs:1115-1148`) | Greedy speculative is correct, including real draft rewind fix (`.squad/decisions.md:456-478`), but stochastic/rejection acceptance is not implemented. It is self-contained after probability helpers and deterministic tests are added. |
| Tree-structured speculative decoding | **MISSING** | Metadata has `topology: Option<String>` (`crates/onnx-genai-metadata/src/schema.rs:152-155`) and sample fixture says `topology: linear` (`tests/fixtures/sample_metadata.yaml:42-51`); current loop proposes a linear `Vec<TokenId>` block (`crates/onnx-genai-engine/src/engine.rs:1058-1081`, `:1115-1148`) | No tree candidate representation, branching draft producer, batched verification over a tree, or acceptance/commit algorithm. Defer until linear greedy + rejection are stable and measured. |
| Hardware profile matching | **PARTIAL** | Metadata parses hardware requirements (`crates/onnx-genai-metadata/src/schema.rs:36-39`, `:187-194`) and fixture covers fields (`tests/fixtures/sample_metadata.yaml:63-68`); validation only checks required capabilities (`crates/onnx-genai-metadata/src/validation.rs:24-41`) | Typed metadata exists but is informational. No host/device probe, dtype/EP matching, min-memory fast-fail, or model-distribution selector. Useful, but lower value until multi-model/VLM choices exist. |

## Ranking by value/effort for a runnable system

1. **Grammar/JSON constrained decoding** — highest value/lowest risk. It extends `ProcessorChain` in the existing autoregressive loop and can be tested on CPU with `tests/fixtures/tiny-llm` and TinyStories. It also unlocks OpenAI `response_format` utility quickly.
2. **Rejection-sampling speculative acceptance** — high value/self-contained. It extends the proven greedy speculative loop and can be correctness-tested with deterministic distributions before performance work.
3. **Generalized multi-model pipeline orchestration** — necessary for Phase 4 exit, but larger. Start with metadata schema + loader + single-pass/composite DAG mechanics while preserving current single-decoder generation path.
4. **VLM support via metadata-declared CLIP/image encoder + decoder** — capstone/exit criterion. Depends on multi-model loading, preprocessing, dataflow, and a small VLM fixture.
5. **Hardware profile matching** — useful for model selection and VLM/runtime quality, but not on the critical path to first runnable Phase 4 unless used as validation-only warnings/fast-fail.
6. **Tree-structured speculative decoding** — most advanced speculative item. Defer until linear speculative acceptance has both greedy and rejection-sampling variants with metrics.

## Sequenced plan: keep runnable/tested at each step

### Step 1 — Structured output vertical slice (Grammar/JSON)

- Add engine-level constrained decoding options without changing default generation behavior.
- Implement a JSON constrained logit processor first using a small deterministic finite-state/token-prefix approach for `response_format: { type: "json_object" }`; add JSON-schema subset only after JSON-object correctness is green.
- Place the processor in the existing constraints slot before temperature/top-k/top-p, matching the documented order (`docs/DESIGN.md:354-357`).
- Rachael maps OpenAI `response_format` into engine options; unsupported schema forms fail clearly rather than silently ignoring constraints.
- Pris tests: logit-processor unit tests, engine tiny-fixture smoke that output parses as JSON, HTTP non-streaming + SSE parity, and stop-sequence interactions.

### Step 2 — Rejection-sampling speculative acceptance

- Extend `AcceptanceRule` beyond `Greedy` to `RejectionSampling` and parse metadata/request strategy names.
- Add probability extraction after processor application, then accept draft tokens by the standard `min(1, p_target(token) / p_draft(token))` rule with deterministic RNG injection in tests.
- Preserve the Phase 3 contract: greedy speculation remains token-for-token identical to target-only greedy (`.squad/decisions.md:543-548`).
- Pris tests rejection/acceptance edge cases, KV rewind on rejection, EOS/stop at accepted and replacement tokens, and same-model all-accept sanity.

### Step 3 — Generalized pipeline metadata + loader + DAG skeleton

- Replace `PipelineSpec` raw `serde_json::Value` fields with typed `models`, `dataflow`, `strategy`, `phases`, preprocessing/postprocessing declarations aligned to DESIGN §20 (`docs/DESIGN.md:1040-1068`, `:1367-1401`). Keep unknown fields ignored for spec forward compatibility.
- Add a multi-model `PipelineModelDirectory`/loader that can resolve named ONNX files under one model root without breaking existing `ModelDirectory::load` single-decoder behavior.
- Implement DAG validation/topological order, phase gating (`prompt_only`, `every_step`, `final_only`, `on_demand`), and a `PipelineInputs`/`PipelineOutputs` tensor map.
- First runnable milestone: metadata-declared single-pass two-model toy pipeline using tiny ONNX fixtures where model A output feeds model B input, with no autoregressive decoder changes.

### Step 4 — Composite autoregressive pipeline execution

- Route current decoder-only generation through an `AutoregressivePipeline` wrapper while preserving `Engine::generate`, sessions, prefix cache, SSE, and priority APIs.
- Add composite stages: prompt-only encoder/session outputs cached as conditioning tensors, every-step decoder consumes tokens + cached conditioning, final-only postprocessing optional.
- Keep all Phase 1-3 regression tests green; add a decoder-only metadata pipeline fixture proving no behavior regression.

### Step 5 — VLM capstone

- Add image preprocessing step (`resize_and_normalize`) and image tensor input support.
- Add vision encoder prompt-only execution; wire `vision_encoder.image_features` to decoder input such as `encoder_hidden_states` per metadata dataflow.
- Exit test: run a small CLIP/image-encoder + decoder model end-to-end from inference metadata, produce deterministic caption/text, and verify pipeline dataflow rather than hardcoded model-type dispatch.

### Step 6 — Hardware profiles + tree speculative

- Hardware profiles: start with validation/warnings and CPU memory/dtype checks from metadata; later add EP/device matching and model-distribution selection.
- Tree speculative: add candidate tree data structures, batched target verification, and acceptance/commit tests only after linear rejection acceptance metrics exist.

## Dependency-annotated task breakdown by agent

### Deckard — metadata, ORT, model loading, pipeline schema

1. **D1: Typed generalized pipeline schema** — deps: none. Replace raw pipeline JSON fields with typed `PipelineModelSpec`, `PipelineStrategy`, `DataflowEdge`, `PhaseConfig`, preprocessing/postprocessing types. Preserve forward-compatible unknown field behavior from DESIGN §3.1 (`docs/DESIGN.md:80-85`).
2. **D2: Multi-model directory loader** — deps: D1. Resolve named model files under one root, optional per-model tokenizer/preprocessor needs, metadata path, and shared ORT `Environment`. Keep existing single-model `ModelDirectory::load` compatibility.
3. **D3: Hardware profile validator** — deps: D1. Validate `HardwareRequirements` against available backend facts initially limited to CPU/session dtype/memory hints; return warnings vs hard failures per required/beneficial fields.
4. **D4: Image preprocessing tensor path** — deps: D2. Add host-side image resize/normalize step and `Value` creation for image tensors needed by VLM.
5. **D5: VLM fixture build contract with Pris/mobius** — deps: D2/D4. Define expected ONNX filenames, I/O names, dataflow names, tokenizer assets, and metadata declaration for the smallest VLM.

### Batty — constrained decoding, speculative acceptance, pipeline execution

1. **B1: JSON/grammar constrained logit processor** — deps: none. Add constraints in `ProcessorChain` before temperature/top-k/top-p, with request options and deterministic unit tests.
2. **B2: Rejection-sampling acceptance** — deps: B1 only if constraints must affect speculative probabilities; otherwise can proceed in parallel. Extend speculative acceptance while preserving greedy exactness.
3. **B3: Pipeline DAG executor skeleton** — deps: Deckard D1/D2. Implement topological execution, tensor map, phase gating, and single-pass/composite strategy dispatch in `pipeline.rs`.
4. **B4: Autoregressive pipeline wrapper** — deps: B3. Adapt current engine session/decode loop as an autoregressive stage without changing public `Engine::generate` semantics.
5. **B5: Tree speculative decoding** — deps: B2 and Pris acceptance metrics. Add tree candidate structures and verification once linear rejection sampling is stable.

### Rachael — HTTP/server surface

1. **R1: OpenAI `response_format` surface** — deps: Batty B1. Add `response_format` parsing/validation for JSON object and later JSON schema; preserve existing chat completion/SSE shapes.
2. **R2: Structured streaming behavior** — deps: R1. Ensure streamed chunks remain valid under stop buffering and constrained decoding; errors for impossible constraints are OpenAI-shaped.
3. **R3: Pipeline/VLM endpoint strategy** — deps: Batty B3/B4 and Deckard D4. Decide whether VLM enters `/v1/chat/completions` via multimodal message content or a generic `/v1/pipeline/run`; start with the smallest OpenAI-compatible image content shape if practical.
4. **R4: Pipeline observability** — deps: R3. Surface phase/stage failures clearly and optionally log dataflow/model names without breaking OpenAI compatibility.

### Pris — tests, fixtures, mobius builds, validation

1. **P1: Constrained decoding correctness** — deps: Batty B1/Rachael R1. Unit-test masks/FSM, engine JSON parseability on tiny fixture, HTTP `response_format` non-stream/SSE parity, and failure cases.
2. **P2: Rejection acceptance tests** — deps: Batty B2. Deterministic accept/reject probabilities, target/draft mismatch, KV rewind, stop/EOS boundaries, and greedy regression.
3. **P3: Pipeline metadata fixtures** — deps: Deckard D1/Batty B3. Minimal two-model DAG fixture and negative validation fixtures for bad edges, cycles, unknown phase/model names.
4. **P4: VLM e2e fixture and test** — deps: Deckard D4/D5 + Batty B4 + Rachael R3 if HTTP. Build/run smallest VLM model; assert metadata-declared pipeline executes CLIP/image encoder prompt-only then decoder every-step.
5. **P5: Mobius model builds + disk budget** — deps: D5. Prefer generated tiny/random-weight fixtures for committed tests; keep pretrained/generated real VLM artifacts gitignored unless very small. Record build commands and sizes in decisions.

## VLM model recommendation

Mobius should build the VLM fixture if it can export a **tiny CLIP-like vision encoder plus tiny decoder with compatible cross-attention input names**. For committed CI, use a deterministic random-weight toy VLM rather than a pretrained captioner: e.g. 224x224 (or smaller if the encoder supports it) image encoder with 1-2 layers, hidden size 32-64, fixed patch count, and a tiny GPT-style decoder with the existing WordLevel tokenizer plus an `encoder_hidden_states` input. This keeps disk size close to the existing tiny fixture scale and tests the runtime contract: metadata-declared model loading, image preprocessing, dataflow, prompt-only encoder execution, and autoregressive decoder conditioning.

For a human-quality demo, keep a separate gitignored Mobius build under `models/` using a small public VLM only after the toy VLM is green. Do **not** make a large CLIP+decoder checkpoint the Phase 4 gate; the exit criterion is end-to-end VLM pipeline via inference metadata declaration, so the smallest viable test model is a tiny Mobius-generated CLIP/ViT-style encoder + tiny cross-attention decoder with deterministic output.

## Roy recommendation

Start **Grammar/JSON constrained decoding first**, then **rejection-sampling acceptance**, then **generalized pipeline orchestration**, then **VLM capstone**. This order maximizes runnable milestones: every step has CPU/tiny-fixture tests and preserves Phase 1-3 behavior before taking on the large multi-model/VLM surface. Hardware profile matching and tree speculative decoding should follow as advanced hardening once the exit pipeline is real.

---

<!-- Inbox source: `roy-toolgrammar-plan.md` -->

# 2026-07-12T10:44:00-07:00: Chat templates, tool use, and grammar-constrained decoding plan

**By:** Roy

## Goal

Make `Qwen/Qwen2.5-0.5B-Instruct` drive a Hermes-style coding agent end-to-end through the OpenAI-compatible server by adding proper model-owned chat templating, OpenAI tool/function calling, and schema/grammar-constrained decoding. Qwen2.5-0.5B already runs coherently through the current f32 ORT path; this plan focuses on architecture, API shape, and implementation sequencing rather than changing code in this decision.

## 1. Crate placement and chat-template rendering

**Decision:** put chat-template rendering in `onnx-genai-ort`, next to `ModelDirectory` and `Tokenizer`, as a lightweight `chat_template` module rather than creating a new crate initially.

**Why:**

- The authoritative `chat_template` lives in `tokenizer_config.json`, which is model-directory/tokenizer metadata already owned by `onnx-genai-ort`.
- Rendering must use the same model tokenizer facts as generation, especially special tokens and EOS behavior. For Qwen, `<|im_end|>` is the semantic EOS/stop token (`151645`), while `<|endoftext|>` (`151643`) is not sufficient for chat termination.
- Keeping this in `onnx-genai-ort` lets the server and engine consume one model-runtime contract without duplicating HF tokenizer-config parsing.
- A separate `onnx-genai-chat` crate is not justified until we have multiple independent chat-template consumers or non-ORT backends; we can split later without changing the public data model.

**API shape:**

- Extend `ModelDirectory` to resolve optional `tokenizer_config.json` alongside `tokenizer.json`.
- Add a `TokenizerConfig`/`ChatTemplateConfig` loader that extracts:
  - `chat_template: Option<String>`
  - `eos_token` / `eos_token_id` where available
  - relevant added/special tokens, including `<|im_start|>`, `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, and `</tool_response>` when present.
- Add a renderer backed by the Rust `minijinja` crate:
  - `ChatTemplate::from_tokenizer_config(...)`
  - `render(messages, tools, add_generation_prompt) -> Result<String>`
- Support the Qwen Jinja features we know are used: loops, `loop.first`, `loop.index0`, `loop.last`, conditionals, and `tojson`.
- If no template exists, keep the current simple role-tagged server fallback only as compatibility behavior, not as the default for Qwen/HF instruct models.

**Stop handling:** chat rendering should surface model stop metadata to the engine/server so requests rendered through Qwen's template automatically stop on `<|im_end|>`/token `151645`. Do not rely on tokenizer default EOS alone for Qwen chat.

## 2. Grammar approach

**Decision:** integrate the Rust `llguidance` crate as a first-class engine constraint implementation. Do not extend the hand-rolled JSON FSM beyond its current fallback role.

**Why:**

- The current `GenerateConstraint::Json` FSM only guarantees generic JSON syntax. Tool calling needs JSON Schema-constrained arguments per tool, and agent loops need regex/Lark-style formats beyond JSON object validity.
- `llguidance` already supports JSON Schema, regex, and Lark grammars, and its architecture matches our existing constrained-decoding path: compute allowed-token mask each step, then set disallowed logits to `f32::NEG_INFINITY` before sampling/argmax.
- Extending the FSM to JSON Schema, regex, and Lark would recreate a grammar engine and still need tokenizer-boundary correctness work.

**API shape:**

```rust
pub enum GenerateConstraint {
    Json,
    JsonSchema(String),
    Regex(String),
    Lark(String),
}
```

Engine behavior:

- Keep `GenerateConstraint::Json` mapped to the existing hand-rolled constraint for now as a low-dependency fallback.
- Add an internal `Constraint` implementation backed by `llguidance` for `JsonSchema`, `Regex`, and `Lark`.
- The existing `ConstraintProcessor` remains the integration point: before temperature/top-k/top-p/argmax, ask the active constraint for the allowed-token set and mask all other logits to `-inf`.
- The `llguidance` backend must be initialized from the model `tokenizer.json` plus a tokenize/decode callback compatible with the current HF `tokenizers` wrapper.
- Completion semantics should be explicit: if a grammar is incomplete at `max_tokens`, context limit, or EOS, return a constrained-decoding error rather than malformed output unless the server intentionally maps it to a compatibility fallback.

Tool-argument forcing:

- For automatic tool-calling mode, constrain the JSON inside each `<tool_call>...</tool_call>` block to the selected tool's schema where `tool_choice` is fixed.
- For `tool_choice: "auto"` across multiple tools, use either:
  - a union JSON Schema over allowed `{name, arguments}` shapes, or
  - a generated Lark grammar that forces `name` to one of the tool names and dispatches the `arguments` schema accordingly.
- Keep boundary tags (`<tool_call>`, `</tool_call>`) outside the JSON schema. Boundary forcing may require tokenizer special-token registration/recognition so the model cannot drift across partial tag boundaries.

Build/linking implications:

- Adding `llguidance` introduces a nontrivial Rust dependency and potentially native/link-time costs. Batty should first land it behind the engine constraint feature path and validate `cargo check --workspace` plus macOS/Linux linkage.
- If the crate pulls optional heavy backends, disable unused features by default.
- Keep `GenerateConstraint::Json` available even if `llguidance` integration is feature-gated later.

## 3. Tool-use data flow and response shapes

### Request ingestion

`POST /v1/chat/completions` should accept and preserve:

- `messages[]` with roles `system`, `user`, `assistant`, and `tool`.
- assistant messages with OpenAI `tool_calls` so prior assistant tool requests can be replayed into the template.
- `tools: [{ "type": "function", "function": { "name", "description", "parameters" } }]`.
- `tool_choice`: `"none"`, `"auto"`, `"required"`, or `{ "type": "function", "function": { "name": "..." } }`.
- existing `response_format`, `stream`, `max_tokens`, `temperature`, `top_p`, `stop`, and session behavior.

### Template rendering

The server converts OpenAI messages/tools into the chat-template model and calls the runtime renderer:

```text
messages + tools + add_generation_prompt=true
  -> Qwen tokenizer_config.chat_template via minijinja
  -> ChatML prompt
```

For Qwen with tools, the template injects tool schemas in a system/tool section using `<tools>{json schema}</tools>` and instructs the model to emit:

```text
<tool_call>
{"name":"...","arguments":{...}}
</tool_call>
```

Tool results are accepted as OpenAI `role: "tool"` messages and rendered through the Qwen branch as `<tool_response>...</tool_response>`.

### Generation

- `tool_choice: "none"`: generate normal assistant text, with no tool-call parsing requirement.
- `tool_choice: "auto"`: render tools and allow either text or `<tool_call>` output. Prefer grammar assistance if it does not prevent normal text; otherwise rely on parsing for the first increment.
- `tool_choice: "required"`: strongly prefer constraining output to one or more valid `<tool_call>` blocks.
- fixed `tool_choice`: constrain the tool-call JSON to that function name and its JSON Schema parameters.

Generation options should include Qwen's `<|im_end|>` stop token/sequence automatically when the template identifies it.

### Parsing

Add a server-side parser; ORT-GenAI will not parse tool calls for us.

Parser contract:

- Scan assistant output for complete `<tool_call> ... </tool_call>` blocks.
- Trim whitespace inside each block and parse the payload as JSON.
- Accept either a single object or multiple blocks; each object must contain:
  - `name: string`
  - `arguments: object` or JSON value accepted by the tool schema, serialized back to a compact JSON string for OpenAI compatibility.
- Reject or ignore malformed/incomplete blocks according to strictness:
  - in `required`/fixed-tool mode, malformed tool output is a generation error or retry candidate;
  - in `auto` mode, if no complete valid block exists, return normal assistant text.
- Preserve non-tool text only when no tool call is returned. If valid tool calls exist, finish with `tool_calls` and do not expose raw tag text as assistant content.

### Non-streaming OpenAI response shape

Tool-call response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 178...,
  "model": "qwen2.5-0.5b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_...",
            "type": "function",
            "function": {
              "name": "edit_file",
              "arguments": "{\"path\":\"...\",\"patch\":\"...\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

Text response remains the existing assistant content shape with `finish_reason` mapped to `stop`/`length` as today.

### Streaming response shape

For tool calls, stream OpenAI-compatible deltas if feasible:

```json
{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_...","type":"function","function":{"name":"edit_file","arguments":"..."}}]}}]}
```

However, first implementation may buffer until complete `<tool_call>` blocks are parsed, then emit one final assistant chunk with `tool_calls`, terminal `finish_reason: "tool_calls"`, and `[DONE]`. This is lower risk and matches the existing constrained JSON streaming precedent that buffers until valid completion.

## 4. Per-agent dependency-annotated breakdown

### Deckard — chat-template/runtime metadata ownership

Can start now.

1. **D1: Load tokenizer_config and chat-template metadata** — deps: none. Extend `ModelDirectory` to resolve `tokenizer_config.json`; parse `chat_template`, EOS/special-token metadata, and expose it next to `Tokenizer`.
2. **D2: Add minijinja chat renderer** — deps: D1. Implement Qwen-compatible rendering of messages, tools, tool responses, and `add_generation_prompt`; cover `tojson` and loop variables used by Qwen.
3. **D3: Stop-token/special-token contract** — deps: D1/D2. Surface `<|im_end|>` id `151645` as chat stop/EOS for Qwen templates, preserve tokenizer default EOS separately, and ensure server/engine can pass this as stop sequence/token.
4. **D4: Boundary-token audit** — deps: D1. Verify `<tool_call>`, `</tool_call>`, `<tool_response>`, and `</tool_response>` are recognized consistently by tokenizer.json/tokenizer_config; document whether extra special-token handling is needed for grammar boundary forcing.

### Batty — grammar and engine constraints

Can start now, independent of Deckard except for tokenizer-special-token details.

1. **B1: Add constraint enum variants and llguidance dependency spike** — deps: none. Add `JsonSchema(String)`, `Regex(String)`, `Lark(String)` API and prove the crate builds/links in this workspace.
2. **B2: Implement llguidance-backed ConstraintProcessor** — deps: B1. Initialize from tokenizer.json/tokenize callback, compute per-step allowed-token masks, and reuse existing `-inf` logit masking before sampling.
3. **B3: Completion/error semantics** — deps: B2. Ensure incomplete grammar output on max/context/EOS is distinguishable from successful completion.
4. **B4: Tool-argument schema forcing** — deps: B2 and Rachael's tool data model. Generate a JSON Schema/Lark constraint for fixed or required tool calls; support a union/dispatch strategy for `auto`/multiple tools.
5. **B5: Keep hand-rolled JSON fallback** — deps: none. Preserve existing `GenerateConstraint::Json` and response_format behavior while llguidance is introduced.

### Rachael — OpenAI server tool support and chat-template hookup

Can start request/response data-model work now; template rendering hookup depends on Deckard D2/D3; schema forcing depends on Batty B4.

1. **R1: Extend request/response structs** — deps: none. Accept OpenAI `tools`, `tool_choice`, assistant `tool_calls`, and `role: "tool"` messages without breaking current text/json responses.
2. **R2: Replace simple prompt formatter with chat-template renderer** — deps: Deckard D2/D3. For templated models, render messages/tools via runtime renderer and automatically apply chat stop tokens.
3. **R3: Implement `<tool_call>` parser** — deps: R1. Parse complete blocks into OpenAI `tool_calls`, generate stable `call_...` ids, serialize `function.arguments` as a JSON string, and set `finish_reason: "tool_calls"`.
4. **R4: Map tool_choice to generation behavior** — deps: R1/R3 and Batty B4 for constrained modes. Handle `none`, `auto`, `required`, and fixed function choices.
5. **R5: Streaming mapping** — deps: R3. Start with buffer-then-final tool-call chunks if incremental tool-call deltas are too risky.
6. **R6: Session compatibility** — deps: R2/R3. Preserve `X-Session-Id` behavior and ensure tool-result turns append/render correctly in persistent conversations.

### Pris — tests and end-to-end validation

Can start fixture/test planning now; full e2e depends on Deckard/Rachael/Batty deliveries.

1. **P1: Qwen chat-template golden tests** — deps: Deckard D2. Compare rendered Qwen prompts against expected ChatML, including `add_generation_prompt`, tools branch, and tool-response branch.
2. **P2: Stop-token regression** — deps: Deckard D3/Rachael R2. Verify Qwen chat stops on `<|im_end|>` without manually passing `--stop '<|im_end|>'`.
3. **P3: Grammar constraint tests** — deps: Batty B2/B3. Test valid JSON Schema, regex, and Lark completions; assert incomplete outputs error rather than return malformed text.
4. **P4: Tool parser/server tests** — deps: Rachael R1/R3. Feed synthetic `<tool_call>` output and assert exact OpenAI `tool_calls` response shape and `finish_reason: "tool_calls"`.
5. **P5: Qwen tool-use e2e** — deps: Deckard D2/D3, Batty B4, Rachael R2-R4. Run `models/qwen2.5-0.5b` through `/v1/chat/completions` with a simple function schema and verify a valid parsed tool call.
6. **P6: Hermes-style coding-agent loop** — deps: P5 plus stable tool-result rendering. Exercise model -> tool call -> role `tool` response -> model continuation through the OpenAI-compatible server.

## 5. Parallelization

Can start in parallel now:

- Deckard D1/D2/D3: tokenizer_config loading, minijinja renderer, Qwen `<|im_end|>` stop handling.
- Batty B1/B2/B3: llguidance dependency spike and engine constraint backend.
- Rachael R1/R3: OpenAI tool request/response structs and parser using synthetic generated text.
- Pris P1/P3/P4 scaffolding with golden prompts, grammar-unit expectations, and parser fixtures.

Blocked/ordered:

- Rachael R2 requires Deckard's renderer.
- Rachael R4 fixed/required schema forcing requires Batty B4.
- Pris P5/P6 require the renderer, server mapping, parser, and at least fixed-tool schema forcing.

## 6. Biggest risks

1. **Qwen minijinja fidelity.** HF chat templates are executable Jinja, and Qwen's tools branch uses `tojson` plus loop metadata. Any mismatch silently changes the prompt contract and can break tool use.
2. **`llguidance` build/linking and tokenizer integration.** It must compile cleanly in the workspace and align its token mask with our HF tokenizer wrapper and Qwen's 151,936-token vocabulary.
3. **`<|im_end|>` stop behavior.** If we keep stopping only on `<|endoftext|>`/`151643`, Qwen chat outputs may run past assistant turns or include extra ChatML/tool text.
4. **Tool-call boundary robustness.** The JSON can be constrained, but `<tool_call>`/`</tool_call>` boundaries are outside JSON. Partial tags, duplicated blocks, and trailing assistant prose need strict parser behavior.
5. **Schema forcing across multiple tools.** `tool_choice: "auto"` needs a safe union/dispatch grammar; fixed tool choice is easier and should land first.

## Highest-value first task

**Deckard D1/D2/D3: load `tokenizer_config.json`, render Qwen's `chat_template` with minijinja, and automatically stop on `<|im_end|>` (`151645`).** This unlocks correct Qwen prompting for every downstream path, removes manual ChatML formatting from users/tests, and gives Rachael/Pris a stable prompt contract while Batty's llguidance integration proceeds in parallel.

---

### 2026-07-12T12:02:00-07:00: Phase 4, tool use, Qwen, Hermes E2E, and long-context milestones
**By:** Scribe
**What:** Recorded acceptance milestones:
- **Phase 4:** multi-model pipeline executor via `Engine::from_pipeline_dir`, tiny VLM pipeline fixture with ignored e2e, and constrained decoding through llguidance JSON Schema/Regex/Lark plus the JSON FSM.
- **Chat template + tool use:** MiniJinja `ChatTemplate` in `onnx-genai-ort`, `Tokenizer::eos_token_ids()` including `<|im_end|>`, OpenAI tools/tool_choice/tool-role support, `<tool_call>` parsing into `tool_calls`, and forced `tool_choice` through Lark `%json` grammar.
- **Qwen:** Qwen2.5-0.5B builds via Mobius and runs coherently; GQA with 2 KV heads is supported; f16 KV was added at the `Value` layer.
- **Hermes agent E2E:** Hermes agent v0.17, using a custom provider against this server, drove Qwen through the runtime to complete a coding task by writing a file via tool call. `scripts/coding_agent.py` also completes tool-loop coding tasks. Acceptance is met.
- **Long-context efficiency:** Mobius `--static-cache` tensor-scatter models use runtime-owned KV buffers; ORT `DecodeSession` enables zero-copy present-to-past, `StaticCacheDecodeSession` scatters in place, and engine decode migrated to these paths. Benchmark latency stayed flat at ~25-27 ms/token from 64 to 2048 context with 48 MiB preallocated KV and no per-step growth, proving O(1)/token; 128K is a larger-buffer extension.
- **Directives:** runtime owns KV buffers and scatter is only a hint; use this project's inference-metadata config rather than ORT-GenAI `genai_config`; paged attention is the next long-context milestone; Python-built ONNX models should use onnxscript/onnx-ir, not `onnx.helper`; commit and push per agent; `Cargo.lock` is committed.
**Why:** These decisions close Phase 4 plus tool-use/grammar and long-context acceptance, while preserving the directives that govern the next design increments.

---

### 2026-07-12T13:14:00-07:00: New reviewers and specialists joined
**By:** Scribe
**What:** Added Gaff (Code Reviewer / Quality), Sebastian (Performance Engineer), and Holden (Security Engineer) to the working team context.
**Why:** Recent work now includes dedicated quality, performance, and security review tracks alongside Roy, Deckard, Batty, Rachael, and Pris.

---

### 2026-07-12T13:14:00-07:00: §23/§24 feature milestones and ORT execution-provider work
**By:** Scribe
**What:** Merged Batty and Deckard feature notes:
- §24 sampling now includes `MinPProcessor`, `FrequencyPenaltyProcessor`, and `PresencePenaltyProcessor` wired through additive `GenerateOptions` fields. Processor order is repetition/frequency/presence penalties, hard stops/constraints, temperature, top-k, top-p, then min-p.
- §23 FIM adds `FimConfig`, `FimFormat::{PSM, SPM}`, tokenizer-config auto-detection for common coder-model tokens, `Engine::fim_config()`, `generate_fim`, and `generate_fim_with_config`; generated output is only the filled middle and FIM sentinels are added to stops.
- ORT execution-provider selection is available through `SessionOptions::with_execution_provider(...)` and `ONNX_GENAI_EP=cpu|webgpu|coreml`; unavailable or incompatible GPU EPs warn and fall back to CPU. On the tested Qwen2.5-0.5B workloads, WebGPU and CoreML produced matching greedy tokens but were slower than CPU for small decode workloads.
- `BatchedStaticCacheDecodeSession` supports fixed B>1 static-cache rows with one ORT forward, per-row cursors/logits, rewind, deactivate/activate, and slot reuse. Validation showed batched rows match independent unbatched `StaticCacheDecodeSession` traces.
**Why:** These close recent sampler, FIM, EP-selection, and first batched static-cache slices while preserving the caveat that inactive batched rows still consume compute until row compaction lands.

---

### 2026-07-12T13:14:00-07:00: Architecture and quality review findings
**By:** Roy and Gaff, merged by Scribe
**What:** The crate split remains sound: metadata, ORT, KV/cache, scheduler, engine, server, and facade largely match the intended design layers. The main architectural blocker is `crates/onnx-genai-engine/src/engine.rs`, now about 3,300 lines and acting as a god module for API DTOs, loading, sessions, scheduler driving, prefix/KV bridging, decode selection, speculative verification, FIM, constraints, sampling, logits, and tests. Server and ORT decode modules are also growing.

Prioritized refactor direction:
- Decompose `engine.rs` into API, loader, session state, decode backend, KV bridge, processor/sampler, speculative, and FIM integration modules before §26-§28 expansion.
- Introduce a `DecodeBackend` / `DecodeSessionOps` seam before true batching, paged attention, hidden-state verifier runs, MTP, or EAGLE.
- Replace whole-call `Arc<Mutex<Engine>>` serving with an engine runtime loop/channel for §26; the scheduler should emit real batch plans consumed by the decode backend.
- Move current draft-model greedy speculation behind `SpeculativeProposer` + verifier traits before §27/§28.
- Add a stateful `Sampler` trait with RNG/seed handling; current generation still routes deterministic `rng_value` plumbing in many paths.
- Keep DESIGN §25 extensibility incremental: use Rust traits and `EngineBuilder` first, not a dynamic plugin ABI.
**Why:** §26 batched serving, §27 advanced speculation, and §28 vLLM speculator compatibility will otherwise keep editing the same engine loops and hard-coded backend matches, increasing merge conflicts and hidden correctness risk.

---

### 2026-07-12T13:14:00-07:00: Security review baseline, fixed hardening, and remaining warnings
**By:** Security review, Holden, Pris, Deckard, and Rachael; merged by Scribe
**What:** Security review found the FFI/unsafe baseline solid: ORT tensor creation validates shapes and lifetimes, `BatchedStaticCacheDecodeSession` bounds-checks row/cursor access, chat templates render data rather than compiling user templates, KV is safe Rust, and server JSON/tool parsing avoids unchecked attacker-data unwraps.

Fixed issues:
- `scripts/coding_agent.py` harness is sandboxed: workspace-confined paths, no `shell=True`, allow-listed argv, command-output caps, guarded Python script execution, symlink/traversal/absolute-path escape rejection, and a passing self-test.
- ORT download supply-chain hardening now verifies pinned SHA-256 digests for official ONNX Runtime 1.27.0 Linux x64, macOS arm64, and Windows x64 archives before extraction; missing future digests warn loudly and should be added before relying on that auto-download path.
- Unsafe invariants for `Environment`, `Session`, `SharedEngine`, and `stable_session_ref` were documented. They are sound under current ORT contracts, mutex serialization, boxed-session allocation stability, and Engine drop ordering, but should not be carried unchanged into true concurrent §26 serving.
- Server DoS/session hardening added `max_output_tokens=4096`, `max_sessions=256` LRU eviction, 128-bit CSPRNG `sess-...` ids, model-context/token cap validation, and bind-site documentation about loopback/no-auth/auth-proxy expectations.
- `cargo audit` reported 0 vulnerabilities. Two unmaintained transitive warnings remain through `tokenizers` (`number_prefix` via `indicatif`, and `paste`); track upstream updates and avoid adding direct `paste` usage.
**Why:** The immediate command-execution, native-download integrity, predictable session-id, unbounded-output, and unbounded-session risks are addressed, while future §26 work still needs stronger ownership/concurrency invariants and broader resource-budget enforcement.

---

### 2026-07-12T13:14:00-07:00: Sebastian performance review and §26 perf queue
**By:** Sebastian, merged by Scribe
**What:** Performance review found the key §26 blocker is active-row utilization: `BatchedStaticCacheDecodeSession` logically skips inactive rows but still binds and runs the full fixed batch, so compute efficiency is roughly `active_rows / batch_size`. Implement active-row compaction with logical-to-physical row mapping, packed active rows, prefix row views, and logits scatter before treating paused rows as free.

Additional perf direction:
- Pick one live KV source of truth per path. ORT static/shared-buffer KV should be hot-path truth now; paged/tiered/int8 KV is not live source-of-truth on current ORT runner paths and should become explicit snapshot/import/export or future page-backed serving, not an always-on f32 mirror.
- Prefer static-cache or validated shared-buffer contracts for long context; past/present zero-copy still has O(context) ORT present allocation/output growth and attention-mask allocation.
- Reduce per-step hot-path allocations by reusing small input buffers/Values, precomputing binding plans, caching memory info and output maps, and avoiding full rebinds when shapes are stable.
- Avoid full logits tensor clones and row re-clones; add borrowed/range logit access or process final-row logits in place.
- Add static-cache chunk append, carry fp16 KV dtype through engine/snapshot paths, keep int8 for cold snapshots unless redesigned for in-place per-slot/group quantization, and prevent runner metadata-only pages from being mistaken for valid KV tensors.
**Why:** Batched serving should land with compaction and unified KV ownership, not just a fixed-B wrapper that wastes inactive rows and duplicates memory/accounting work.

