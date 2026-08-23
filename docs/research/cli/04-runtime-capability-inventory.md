# Runtime capability inventory for the maintainer REPL

Justin's directive: `onnx-genai` is a developer/maintainer CLI, and `onnx-genai run` should expose runtime capabilities. This inventory cites only APIs that exist today; missing APIs are called out explicitly.

## Summary

| Capability | Reachable from CLI today | Proposed REPL command | Needs new runtime API? | Effort |
|---|---:|---|---:|---:|
| Current REPL chat session | Partially | existing `/session`, richer `/session show` | No | S |
| Engine persistent sessions | No in CLI; yes in server | `/session new|list|switch|reset|close` | Partial: list/detail APIs missing | M |
| Session token/detail inspect | No | `/session tokens`, `/session inspect --json` | Yes for detailed KV/scheduler state | M |
| CoW fork | No | `/fork [name] [--turn N|--tokens N]` | **Yes: engine session fork API** | L |
| KV rewind/checkpoint | No | `/rewind`, `/undo-turn`, `/checkpoint`, `/restore` | **Yes: public engine rewind/checkpoint API** | M/L |
| Prefix/paged-KV reuse visibility | Partially (`/stats`, `/pages`, profiles) | `/cache`, `/cache stats --json` | Only for finer text hit-source breakdown | S/M |
| External KV connector stats | No | `/connector stats` | No for last-generation stats | S/M |
| Speculative decoding | No | `/spec off|prompt-lookup|draft|mtp|eagle3|shared-kv`, `/spec stats` | Mostly no; discovery/load mutation APIs useful | M |
| Continuous batching | No in REPL; server uses it | `/batch submit|step|drain|status` | No for basic harness | M |
| Priority scheduler harness | No | `/scheduler run <scenario.json>` | No for scripted scenarios | M |
| Execution provider | Yes, reloads | existing `/ep [name]` | No | S |
| Decode backend | Yes, reloads | existing `/backend [auto|ort|native]` | No | S |
| Resource governor | Startup only in CLI; engine can set VRAM | `/resources`, `/resources vram <limit>` | Host/disk setters missing | M |
| Sampling options | Startup flags only | `/set temperature 0.7`, `/sampling` | No | S |
| FIM | Server/engine only, not REPL | `/fim --prefix ... --suffix ...` | No | S/M |
| Embeddings | Server/engine only, not REPL | `/embed <text>` | No | S |
| JSON/grammar/logprobs | Server/runtime only, not REPL | `/json`, `/grammar <file>`, `/logprobs N` | No | M |
| Multimodal attachment state | Partially | `/attachments`, `/attachments clear` | No | S |
| Native CUDA debug stats | No | `/native stats --json` | No, feature/backend gated | S/M |

## 1. Sessions / multi-session

There is no public first-class `Session` object. The public handle is an alias: `crates/onnx-genai-engine/src/config.rs:402-403` defines `pub type SessionId = SequenceId;`.

The engine owns private persistent state in `crates/onnx-genai-engine/src/engine/model.rs:30-33` as `sessions: HashMap<SessionId, EngineSession>`, plus shared decoder sessions. The state itself is `crates/onnx-genai-engine/src/session.rs:11-24`: `EngineSession { tokens, kv_token_count, decode_state, draft, sampled_fastpath_failed }`. Active per-request state is separate: `crates/onnx-genai-engine/src/session.rs:26-39` stores `options`, processor chain, generated tokens/text/logprobs, RNG, and prefix hit length. So sampling config is not owned persistently by a session; the current REPL owns it as CLI state.

Existing public lifecycle/generation APIs:

- `crates/onnx-genai-engine/src/engine/runtime.rs:499-526` — `pub fn create_session(&mut self) -> anyhow::Result<SessionId>`.
- `crates/onnx-genai-engine/src/engine/runtime.rs:528-559` — `pub fn reset_session(&mut self, session_id: SessionId) -> anyhow::Result<()>`.
- `crates/onnx-genai-engine/src/engine/runtime.rs:595-612` — `pub fn close_session(&mut self, session_id: SessionId) -> anyhow::Result<()>`.
- `crates/onnx-genai-engine/src/engine/runtime.rs:615-620` — `pub fn session_token_count(&self, session_id: SessionId) -> anyhow::Result<usize>`.
- `crates/onnx-genai-engine/src/engine/runtime.rs:244-251` — `generate_in_session`.
- `crates/onnx-genai-engine/src/engine/runtime.rs:265-279` — `generate_in_session_with_callback`.
- `crates/onnx-genai-engine/src/engine/runtime.rs:281-302` — `generate_in_session_with_sampler`.

Constraint: all persistent-session entry points call `require_ort_backend`; `crates/onnx-genai-engine/src/engine/runtime.rs:103-109` rejects the native single-session backend.

The server has a client session facade: `crates/onnx-genai-server/src/session.rs:9-24` maps bearer client ids to engine session ids; `src/session.rs:38-76` inserts with LRU eviction; `src/session.rs:78-94` gets and refreshes access; `src/session.rs:96-109` removes. HTTP create/delete are `crates/onnx-genai-server/src/routes/sessions.rs:3-35` and `38-60`. `X-Session-Id` lazy creation is in `crates/onnx-genai-server/src/routes/completions.rs:1057-1096`. Debug listing is redacted only: `crates/onnx-genai-server/src/routes/admin.rs:103-115`.

CLI today does not use engine persistent sessions. It keeps `history: Vec<ChatMessage>` and rerenders it each turn (`crates/onnx-genai-cli/src/interactive.rs:629-633`). `/session` only prints a summary (`interactive.rs:781-790`), and `ReplCommand` has no new/switch/close/fork variants (`crates/onnx-genai-cli/src/commands.rs:8-35`).

**REPL shape:** `/session new [name]`, `/session list`, `/session switch <id>`, `/session reset [id]`, `/session close <id>`, `/session tokens`. Basic multi-session is CLI state plus existing engine APIs (**M**). Deep inspect/list independent of CLI needs new runtime API.

## 2. CoW fork

Low-level KV fork exists:

- `crates/onnx-genai-kv/src/lib.rs:72-94` — trait `KvCacheOps` includes `fn fork(&mut self, source: SequenceId, position: usize) -> Result<SequenceId, KvError>;`.
- `crates/onnx-genai-kv/src/paged_cache.rs:741-776` — `PagedKvCache::fork` validates retained/length bounds, creates a new sequence, copies retained start/sink metadata, retains source pages, attaches those page ids, and sets child length.
- `crates/onnx-genai-kv/src/page_table.rs:932-939` — `PageTable::retain` increments ref counts.
- `crates/onnx-genai-kv/src/paged_cache.rs:572-626` — `ensure_page_for_write` copies a page only when `ref_count > 1`.
- `crates/onnx-genai-kv/src/prefix_cache.rs:1-8` documents page-table-driven fork/write CoW.

Cost model: fork is O(number of prefix pages retained), with no tensor payload copy at fork time (`paged_cache.rs:754-773`). Divergence pays lazily: the first write to each shared page allocates a new page and clones stored data (`paged_cache.rs:596-623`). Partial-page forks initially share; appending copies the shared partial page, covered by `crates/onnx-genai-kv/src/paged_cache.rs:1404-1419`.

Constraints: fork position must be <= sequence length and not before `retained_start` (`paged_cache.rs:741-752`), so sliding-window evicted gaps cannot be forked. The API is sequence-oriented, so REPL fork is effectively batch==1 conversation branching. The page table has GPU/CPU/Disk concepts, but divergent writes currently allocate `Device::Gpu(0)` (`paged_cache.rs:596-601`, `636-642`). Native engine sessions are not supported.

Missing: there is **no engine-level `fork_session` API**. A correct fork must clone/truncate logical tokens, target `DecodeState`, target KV, draft session/KV, and state latches from `EngineSession` (`crates/onnx-genai-engine/src/session.rs:11-24`), not just call `kv_cache.fork`.

**REPL shape:** `/fork [name]`, `/fork [name] --turn N`, `/fork [name] --tokens N`. **Needs new runtime API**, likely `pub fn fork_session(&mut self, source: SessionId, position: usize) -> anyhow::Result<SessionId>`. Effort **L**.

## 3. KV rewind / truncation

Low-level APIs:

- `crates/onnx-genai-kv/src/lib.rs:72-84` — `rewind_to`, `checkpoint`, `restore`.
- `crates/onnx-genai-kv/src/lib.rs:96-102` — `CacheCheckpoint { seq, position, page_ids }`.
- `crates/onnx-genai-kv/src/paged_cache.rs:687-739` — `PagedKvCache::rewind_to` truncates/free pages, updates last-page fill, sequence length, and sink/window bookkeeping.
- `crates/onnx-genai-kv/src/paged_cache.rs:778-800` — checkpoint records current position/page ids; restore rewinds to checkpoint position.

Engine-internal helpers:

- `crates/onnx-genai-engine/src/kv_bridge.rs:395-413` — `rewind_target_state_to_len(...)`.
- `crates/onnx-genai-engine/src/kv_bridge.rs:428-442` — `rewind_draft_state_to_len(...)`.
- `crates/onnx-genai-engine/src/kv_bridge.rs:452-500` — `rewind_decode_state_to_len(...)`, including runner rewind, windowed rewind, paged KV rewind, and materialized past reload.
- `crates/onnx-genai-engine/src/decode/state.rs:362-372` — `DecodeState::rewind_runner`.
- `crates/onnx-genai-engine/src/decode/state.rs:470-512` — windowed rewind rejects evicted gaps.

Speculative decoding already depends on rewind: `crates/onnx-genai-engine/src/speculative.rs:1736-1772` rewinds target KV to the accepted prefix before committing correction/bonus tokens; `crates/onnx-genai-engine/src/native_speculative.rs:11-25` documents native verify/rewind/commit.

CLI today has only `/reset`, which clears CLI history/attachments (`crates/onnx-genai-cli/src/interactive.rs:658-664`). It does not rewind engine KV because the REPL is not driving persistent engine sessions.

**REPL shape:** `/checkpoint [name]`, `/rewind --tokens N`, `/rewind --to <checkpoint|turn>`, `/undo-turn`. Real undo should rewind KV to the previous turn boundary, not just drop chat messages.

Cost/constraints: rewind is O(pages removed) plus runner mutation or past reload. Rewinding ORT KV without paged materialization can fail (`crates/onnx-genai-engine/src/kv_bridge.rs:484-486`). Sliding-window evicted gaps are unavailable. **Needs new runtime API** (`rewind_session`, public checkpoints/boundaries). Effort **M/L**.

## 4. Prefix caching / radix reuse / page visibility

Text/session reuse:

- `crates/onnx-genai-engine/src/engine/runtime.rs:719-830` — `prepare_session_prefix` reports same-session runner hits, token-prefix cache hits, paged prefix-cache hits, and external connector extension lookups.
- `crates/onnx-genai-engine/src/config.rs:1246-1258` — `GenerateResult` exposes `prefix_cache_hit_len`.
- `crates/onnx-genai-kv/src/prefix_cache.rs:13-22` — `PrefixMatch { matched_tokens, page_ids }` and radix trie.
- `crates/onnx-genai-kv/src/prefix_cache.rs:90-119` — `lookup_shared` retains pages for a sharing sequence.

Pipeline/multimodal reuse:

- `crates/onnx-genai-engine/src/pipeline_cache.rs:266-284` — `PipelineCacheStats` exposes encoder hits/misses/unkeyable/evictions/bytes, `prefix_reused_tokens`, and `prefill_tokens`.
- `crates/onnx-genai-engine/src/pipeline_cache.rs:301-319` — `stats`, `reset_stats`, `note_prefix_reuse`.
- `crates/onnx-genai-engine/src/pipeline/mod.rs:669-676` — public `cache_stats()` and `reset_cache_stats()`.
- `crates/onnx-genai-cli/src/interactive.rs:398-411` maps pipeline stats into the profile.

Paged KV:

- `crates/onnx-genai-engine/src/engine/runtime.rs:175-185` — `page_usage()` and `page_stats()`.
- `crates/onnx-genai-cli/src/interactive.rs:414-427` exposes page usage/stats through the CLI backend.
- `crates/onnx-genai-cli/src/interactive.rs:863-869` implements `/pages`.

Current CLI observability: `/stats` toggles a compact line (`interactive.rs:874-877`); the stats line includes reuse and encoder hit/run counts (`crates/onnx-genai-cli/src/profile.rs:214-245`); profile text prints prefix reuse (`profile.rs:426-443`). `crates/onnx-genai-cli/src/output.rs:140-151` resets reuse stats before each turn.

External connector stats exist but are not surfaced in CLI: `crates/onnx-genai-engine/src/engine/runtime.rs:188-195` — `pub fn last_connector_stats(&self) -> ConnectorStats`.

**REPL shape:** `/cache`, `/cache pages`, `/cache stats --json`, `/connector stats`. Mostly CLI wiring (**S/M**). New API only if maintainers need the text hit split (same-session vs token-prefix vs paged-prefix vs connector), because public `GenerateResult` currently exposes only total `prefix_cache_hit_len`.

## 5. Speculative decoding

Configuration APIs:

- `crates/onnx-genai-engine/src/config.rs` — `SpeculativeMode::{None,DraftModel,PromptLookup,Mtp,Eagle3}`.
  Borrowed-KV ("shared-KV") drafting is no longer a `SpeculativeMode`: it is
  declared by a package's `speculative.proposal_execution: {kind: chained}`
  contract and driven by the workflow interpreter
  (`crates/onnx-genai-engine/src/pipeline/speculative.rs`).
- `crates/onnx-genai-engine/src/config.rs:495-501` — `EngineConfig` has `draft_model`, `num_speculative_tokens`, default `speculative_mode`.
- `crates/onnx-genai-engine/src/config.rs:805-808` — `GenerateOptions` has per-request `num_speculative_tokens` and `speculative_mode`.
- `crates/onnx-genai-engine/src/config.rs` defines the MTP and EAGLE-3 configs.

Stats:

- `crates/onnx-genai-engine/src/speculative.rs:488-493` — `SpeculativeStats { verification_steps, proposed_tokens, accepted_tokens, multi_token_accepts }`.
- `crates/onnx-genai-engine/src/engine/runtime.rs:144-147` — `pub fn last_speculative_stats(&self) -> SpeculativeStats`.
- Acceptance rate is computable as `accepted_tokens / proposed_tokens`; no dedicated field exists.

Backend constraints:

- ORT sessions can enter the speculative loop from `crates/onnx-genai-engine/src/engine/runtime.rs:360-374`.
- Native backend rejects draft-model/MTP/EAGLE-3 and invalid width usage in `crates/onnx-genai-engine/src/engine/decode_backend.rs:150-177`.
- Native planning only covers prompt-lookup/shared-KV, greedy, empty processor chain, and no logprobs (`decode_backend.rs:194-230`).
- Native verify/accept/rewind loop is documented in `crates/onnx-genai-engine/src/native_speculative.rs:1-29`.

CLI today exposes none of this. `SamplingArgs` maps only max_new_tokens, temperature, top_p, top_k, and stop sequences (`crates/onnx-genai-cli/src/lib.rs:106-182`).

**REPL shape:** `/spec`, `/spec off`, `/spec prompt-lookup --ngram 4 --max 8`, `/spec stats`, later `/spec draft|mtp|eagle3|shared-kv` when sidecars were loaded. Prompt-lookup is mostly CLI wiring. Draft/MTP/EAGLE/shared-KV toggling is constrained by load-time availability; loading new sidecars likely forces reload. Effort **M**.

## 6. Continuous batching / scheduler

Continuous batching is reachable in-process:

- `crates/onnx-genai-engine/src/batched.rs:24-39` — `ContinuousBatchHandle`, `ContinuousBatchEvent`.
- `crates/onnx-genai-engine/src/batched.rs:94-109` — `pub struct ContinuousBatchManager<'a>`.
- `crates/onnx-genai-engine/src/batched.rs:139-178` — `pub fn submit(&mut self, request: GenerateRequest) -> anyhow::Result<ContinuousBatchHandle>`.
- `crates/onnx-genai-engine/src/batched.rs:180-214` — `pub fn step(&mut self)` and `pub fn poll(&mut self)`.
- `crates/onnx-genai-engine/src/batched.rs:216-234` — max/pending/active/idle introspection.
- `crates/onnx-genai-engine/src/batched.rs:582-630` — `Engine::continuous_batch_manager(max_batch)` for static-cache or shared-buffer past/present models; legacy/dynamic past-present rejected.

The server uses this for no-session requests only: `crates/onnx-genai-server/src/driver.rs:509-571` creates/steps/drains the manager; `driver.rs:540-555` admits only `session_id: None` into the batch and defers other commands.

Priority scheduler harness APIs also exist: `crates/onnx-genai-engine/src/config.rs:1211-1231` defines prioritized request/result structs; `crates/onnx-genai-engine/src/engine/runtime.rs:412-423` and `426-497` drive prioritized requests/arrivals one sequence at a time.

**REPL shape:** `/batch new --max 4`, `/batch submit <prompt>`, `/batch step`, `/batch drain`, `/batch status`; `/scheduler run <scenario.json>`. This is a maintainer harness, not normal chat. No new runtime API for basic use, but not persistent-chat-session aware. Effort **M**.

## 7. Execution providers / decode backends / reload boundaries

Execution providers are ORT session construction options:

- `crates/onnx-genai-ort/src/session/options.rs:25-58` — `SessionOptions` includes execution providers, optimization/thread counts, graph capture, WebGPU validation, CUDA attention mode.
- `crates/onnx-genai-ort/src/session/options.rs:60-70` — defaults from env/auto-selection.
- `crates/onnx-genai-ort/src/session/options.rs:117-124` — `pub fn with_execution_provider(selection: EpSelection) -> Self`.
- `crates/onnx-genai-cli/src/commands.rs:70-79` — CLI lists runtime-selectable EPs.

Decode backend:

- `crates/onnx-genai-engine/src/config.rs:453-464` — `EngineDecodeBackend::{Auto,Ort,Native}`.
- `crates/onnx-genai-cli/src/commands.rs:82-93` parses `auto|ort|native`.
- `crates/onnx-genai-cli/src/interactive.rs:159-165` states EP/backend are loaded-session properties and require reload.
- `/ep` reloads and clears history (`interactive.rs:793-827`); `/backend` reloads and clears history (`interactive.rs:830-861`).

Reload-required settings include engine construction fields in `crates/onnx-genai-engine/src/config.rs:466-522`: page count/size, scheduler config, draft model/default speculation, KV cache dtype/connector, resource limits, pipeline cache bytes, native device/precision. ORT `SessionOptions` graph capture/thread/provider fields also require reload.

Live-switchable today: per-request `GenerateOptions`, per-request speculative override when support is loaded, profile trace/detail toggles, and VRAM limit via `crates/onnx-genai-engine/src/engine/runtime.rs:167-173` (`set_vram_limit`). Per-stage profiling must be enabled at startup (`crates/onnx-genai-cli/src/interactive.rs:697-704`).

**REPL shape:** keep `/ep` and `/backend`; add `/runtime`, `/reload --graph-capture on --threads 4`, `/resources`, `/resources vram <limit>`. Effort **S/M**.

## 8. Other agent-first surfaces

### Sampling/logprobs/constraints

`crates/onnx-genai-engine/src/config.rs:760-817` exposes max tokens, temperature, top-p/k/min-p/top-a/typical-p, repetition/frequency/presence penalties, DRY, Mirostat, XTC, greedy, seed, stop/EOS, max_context, speculative override, constraints, and top_logprobs. CLI startup exposes only max_new_tokens, temperature, top_p, top_k, and stop (`crates/onnx-genai-cli/src/lib.rs:106-182`). Proposed: `/set`, `/sampling`, `/stop`, `/seed`, `/logprobs`, `/json`, `/grammar`. No new runtime API.

### FIM

`crates/onnx-genai-engine/src/fim.rs:14-21` defines `FimConfig`; `src/fim.rs:23-43` auto-detects/formats; `crates/onnx-genai-engine/src/engine/runtime.rs:198-223` exposes `generate_fim` and `generate_fim_with_config`. Server rejects FIM with sessions (`crates/onnx-genai-server/src/routes/completions.rs:21-24`). Proposed: `/fim --prefix ... --suffix ...`. No new runtime API.

### Embeddings

`crates/onnx-genai-engine/src/embedding.rs:18-27` defines `EmbeddingOptions`; `src/embedding.rs:29-57` exposes `embed`, `embed_text`, `embed_text_with_options`, `embed_with_options`. Native backend unsupported (`embedding.rs:62-65`). Proposed: `/embed [--pool mean|last] [--normalize] <text>`. No new runtime API.

### Multimodal/pipelines

REPL already supports `/image` and `/audio` in help (`crates/onnx-genai-cli/src/interactive.rs:653-655`) and modality reporting (`interactive.rs:374-388`). Add `/attachments` and default pipeline reuse stats.

One-shot pipeline APIs that could be REPL commands:

- `crates/onnx-genai/src/text_to_image.rs:156-184` — `TextToImageRequest`.
- `crates/onnx-genai/src/text_to_image.rs:549-553` — `pub fn render(...)`.
- `crates/onnx-genai/src/text_to_audio.rs:35-49` — `TextToAudioRequest`.
- `crates/onnx-genai/src/text_to_audio.rs:166-170` — `pub fn synthesize(...)`.
- `crates/onnx-genai-cli/src/transcribe.rs:103-180` — CLI-private transcriber load/entrypoint; reusable REPL transcription may need refactor.

### Native/resource diagnostics

`crates/onnx-genai-engine/src/engine/runtime.rs:149-155` exposes `native_cuda_debug_stats`; `runtime.rs:157-165` exposes governor/resource snapshot; `runtime.rs:167-173` exposes live VRAM setter. CLI currently only has startup `--vram-limit` and `--host-ram-limit` (`crates/onnx-genai-cli/src/lib.rs:185-213`). Proposed: `/native stats`, `/resources`, `/resources vram <limit>`. Host/disk live setters need new APIs.

## Highest-value sequencing

1. **Persistent REPL sessions** (`/session new|switch|close|reset`) — mostly CLI wiring, but detail/list inspect needs new API.
2. **Real `/fork`** — **needs new runtime API**; highest-value named requirement.
3. **Real `/undo-turn` / `/rewind`** — **needs new runtime API**; essential for agent-style branching/editing.
4. **Default `/cache` + `/spec stats` visibility** — mostly CLI wiring; fine-grained prefix hit source may need API.
5. **Speculative prompt-lookup controls + broader sampling controls** — mostly CLI wiring over existing per-request `GenerateOptions`; sidecar discovery/loading APIs useful later.
