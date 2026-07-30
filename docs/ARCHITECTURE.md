# onnx-genai — Architecture

**Audience.** §1 orients anyone in five minutes. §2–§7 are implementer reference. §8 is the honest list of what is stubbed or missing.

**Ground rules for this document.** Every non-obvious claim carries a `file:line`. Nothing aspirational is stated as fact. Where an invariant is *assumed* rather than *enforced*, it says so — those are where bugs live.

Line numbers are accurate as of the commit this document was added. Structure changes slowly; if a line has drifted, the symbol name will still find it.

**Verification.** All 87 distinct `file:line` citations in this document were checked mechanically against the tree — every one resolves to an existing file with that line present. The load-bearing claims were additionally verified by reading the code at each cited location, not by trusting the recon notes they came from.

**How each claim was established — verified vs inferred.** This document deliberately distinguishes the two rather than blending them:

| Evidence class | What it means | Examples |
|---|---|---|
| **Observed** | Confirmed against a *running server* — curled, streamed, or profiled | §5.6's mode matrix (both model types run and profiled); the endpoint behaviours in §8.3; debug endpoints returning **404**, not 403 |
| **Read** | Established by reading the cited source | The dependency edges in §2; the command-deferral dispatch in §5.11; the `NodeStatus` scope comment in §4.7 |
| **Inferred** | A conclusion drawn from the above, not directly witnessed | §5.3's claim that a shared-page write *would* corrupt silently — no test provokes it, and nothing in the code prevents it |

**Why the distinction is worth the overhead.** Verification tools share blind spots with the assumptions they are used to check. Several classes of defect in this codebase are **invisible to the instrument a reader would naturally reach for**:

| The instrument | What it cannot see |
|---|---|
| `curl` against an endpoint | Anything enforced by the *caller* rather than the server. A same-origin-policy failure returns **200 OK** to `curl` and to the server log, and fails only inside a browser. |
| Reading a handler | Whether the value survives the path to it (§5.14), or whether the field name means what it says (§5.13). |
| A passing test | A behaviour that is *structurally impossible* rather than merely absent — the test and the stub agree (§5.6.1). |
| A single measurement | Whether the instrument can resolve the effect at all. A run-to-run difference can exceed the threshold being tested for. |
| A line citation | That the line moved. The claim stays true while the reference silently rots. |

The common shape: **each tool inspects a *state*, while these defects live in a *transition* or a *relationship*.** None of them can express *"and then it changed"* or *"and it means something different over there."*

The practical consequence for this document is that **a chain of individually-true statements can compose into a false one** — every step verifiable, the conclusion wrong. That is why claims here carry their evidence class rather than a uniform tone of authority: the reader needs to know which links were witnessed and which were reasoned, in order to check the composition rather than the steps.

Where a claim is inferred, the text says so. The distinction matters most in §5: an **ASSUMED** invariant is precisely one where the code offers no evidence either way, so the reader is the last line of defence.

**How to read §5.** Each invariant is tagged **ENFORCED** (the code prevents violation) or **ASSUMED** (nothing stops you; violation compiles, runs, and corrupts or degrades silently). The ASSUMED ones — §5.3, §5.8, §5.10 — are the highest-risk sections in this document. §5.12 is tagged **CURRENTLY VIOLATED**, because documenting it as true would be exactly the kind of intent-as-behaviour this document forbids.

---

## 1. Orientation

### What this is

`onnx-genai` is a **generative-AI serving runtime written in Rust**. It loads ONNX models from a directory, runs autoregressive text generation on them, and serves the result over an OpenAI-compatible HTTP API.

It contains two stacks in one workspace:

- **`onnx-runtime-*`** — a Rust reimplementation of ONNX Runtime concerns: IR, graph loading, optimization, quantization, memory, execution providers (CPU, CUDA), tracing.
- **`onnx-genai-*`** — the generation layer built on top: KV cache paging, scheduling, the decode loop, the HTTP server, the CLI.

### How it differs from `onnxruntime-genai`

The upstream C++ `onnxruntime-genai` is a generation library. This project targets **serving**, which changes the design in three ways:

| Capability | What it means here |
|---|---|
| **Continuous batching** | Multiple in-flight requests share one batched forward pass. Finished rows are backfilled from waiting requests so the batch stays occupied, instead of draining and restarting. |
| **Paged KV allocation** | KV cache is allocated in fixed-size **pages** from a pool rather than one contiguous slab per sequence, so memory is reused across sequences and fragmentation is bounded. |
| **Prefix caching** | Sequences sharing a token prefix share the *same physical pages* via reference counting, so a repeated system prompt is not recomputed or re-stored. |

> **Precision, because it matters:** this repo implements a paged KV **allocator**. It does **not** implement paged-attention **kernels**. The allocator is real and observable; attention still reads through the runtime's own cache path. See §8.1.

### The system in one picture

```
   HTTP client
        │  POST /v1/chat/completions  (SSE stream)
┌───────▼─────────────────────────────────────────────┐
│ onnx-genai-server        axum router, lib.rs:60-107 │
│   AppState → ModelRegistry → ModelHandle            │
└───────┬─────────────────────────────────────────────┘
        │  mpsc DriverCommand  +  oneshot / mpsc replies
┌───────▼─────────────────────────────────────────────┐
│ EngineDriver     dedicated OS thread, driver.rs:113 │
│                                                     │
│   ┌── continuous batch path ──┐  ┌── per-request ─┐ │
│   │ ContinuousBatchManager    │  │ Engine::generate│ │
│   │ static-cache models only  │  │ paged KV +      │ │
│   │ batched.rs:101            │  │ prefix cache    │ │
│   └───────────────────────────┘  └─────────────────┘ │
│              ▲ mutually exclusive — see §5.6         │
└───────┬─────────────────────────────────────────────┘
        │
┌───────▼──────────┐  ┌──────────────┐  ┌────────────┐
│ onnx-genai-kv    │  │ scheduler    │  │ onnx-genai │
│ PageTable        │  │ admission,   │  │ -ort       │
│ PrefixCache      │  │ preemption   │  │ Session    │
└──────────────────┘  └──────────────┘  └─────┬──────┘
                                              │
                                   ┌──────────▼────────┐
                                   │ onnx-runtime-*    │
                                   │ IR, EPs, memory   │
                                   └───────────────────┘
```

A reader can stop here and be usefully informed.

---

## 2. Component hierarchy

39 workspace crates in five layers. **Dependencies point downward only.**

### Layer 5 — Entry points (public surface)

| Crate | Single responsibility |
|---|---|
| `onnx-genai-cli` | The `onnx-genai` binary: `serve`, `generate`, `run`, `show`, `list`. Also the `_onnx_genai_server` PyO3 module for the wheel. |
| `onnx-genai-server` | The axum HTTP server and OpenAI-compatible routes. |
| `onnx-genai-capi` / `onnx-genai-python` | C and Python bindings. |
| `onnx-genai-bench` | Benchmark binaries. |
| `onnx-genai-router` | Multi-node request routing. |

### Layer 4 — Generation core

| Crate | Single responsibility |
|---|---|
| `onnx-genai` | The public generation library — the crate external users depend on. |
| `onnx-genai-engine` | `Engine`, the decode loop, batching, the resource governor. The heart. |

### Layer 3 — Generation support

| Crate | Single responsibility |
|---|---|
| `onnx-genai-kv` | `PageTable`, `PrefixCache`. **Depends on `onnx-genai-metadata` and nothing else** — deliberately near-leaf so the allocator is testable in isolation. |
| `onnx-genai-scheduler` | Admission order, batch eligibility, preemption policy, byte budget. |
| `onnx-genai-ort` | Model-directory resolution and the ORT session/tokenizer boundary. |
| `onnx-genai-preprocess` | Image/audio preprocessing. |
| `onnx-genai-metadata`, `-genai-config`, `-runtime-config`, `onnx-model-package` | Model metadata schemas and package layout. |

### Layer 2 — ONNX Runtime

`onnx-runtime-ir`, `-loader`, `-session`, `-optimizer`, `-quantization`, `-memory`, `-shape-inference`, `-operator-selection`, `-eager`, `-tracer`, `-protocol-trace`, `-comm`, `-dlpack`, `-cpuinfo`, `-capi`, `-python`, and the execution providers `-ep-api`, `-ep-cpu`, `-ep-cuda`.

### Layer 1 — Foundation

`onnx-std`, `mlas-sys`.

### Dependency direction — verified, and the forbidden edges

Confirmed from the manifests:

```
onnx-genai-server  → onnx-genai, onnx-genai-engine, onnx-genai-metadata,
                     onnx-genai-ort, onnx-genai-preprocess
onnx-genai-engine  → onnx-genai-kv, onnx-genai-scheduler, onnx-genai-ort,
                     onnx-genai-metadata, onnx-genai-genai-config,
                     onnx-genai-runtime-config, onnx-runtime-{ep-cpu,ep-cuda,
                     ep-api,ir,loader,session,tracer}, onnx-std
onnx-genai-scheduler → onnx-genai-kv, onnx-runtime-protocol-trace
onnx-genai-kv        → onnx-genai-metadata
```

**FORBIDDEN — a PR introducing any of these should be rejected:**

| Forbidden edge | Why |
|---|---|
| `onnx-genai-kv` → engine, scheduler, or server | The page table must stay independently testable. It is the one component whose correctness is provable in isolation; adding an upward edge destroys that. |
| `onnx-genai-scheduler` → engine or server | Scheduling policy must be expressible without knowing how decode works. |
| `onnx-genai-engine` → server | The engine is a library. A server dependency would make it unusable from the CLI and bindings. |
| `onnx-runtime-*` → any `onnx-genai-*` | The runtime layer must remain usable without the generation layer. |

**Cycle note.** `onnx-genai-cli` exists as a separate crate specifically to avoid a dependency cycle: it needs both the core `onnx-genai` crate and `onnx-genai-server`, so it cannot live inside either (`crates/onnx-genai-cli/Cargo.toml`, header comment).

Dependencies flow **strictly downward**. Every red edge below is a rejection:

```mermaid
flowchart TD
    subgraph L5["Layer 5 — entry points"]
        CLI["onnx-genai-cli"]
        SRV["onnx-genai-server"]
    end
    subgraph L4["Layer 4 — generation core"]
        CORE["onnx-genai"]
        ENG["onnx-genai-engine"]
    end
    subgraph L3["Layer 3 — generation support"]
        KV["onnx-genai-kv"]
        SCH["onnx-genai-scheduler"]
        META["onnx-genai-metadata"]
    end
    subgraph L2["Layer 2 — ONNX Runtime"]
        ORT["onnx-genai-ort<br/>onnx-runtime-*"]
    end

    CLI --> CORE
    CLI --> SRV
    SRV --> CORE
    SRV --> ENG
    CORE --> ENG
    ENG --> KV
    ENG --> SCH
    ENG --> ORT
    SCH --> KV
    KV --> META
    ENG --> META

    KV -. "FORBIDDEN" .-> ENG
    SCH -. "FORBIDDEN" .-> ENG
    ENG -. "FORBIDDEN" .-> SRV
    ORT -. "FORBIDDEN" .-> ENG

    linkStyle 11,12,13,14 stroke:#c00,stroke-width:2px,stroke-dasharray:4
```

**Why this shape is worth defending:** `onnx-genai-kv` at the bottom is the reason the page table can be tested without a model, a session, or a server. That property is the codebase's single biggest testability asset, and exactly one upward edge would end it.

---

## 3. The request lifecycle

The spine of the system. Following one `POST /v1/chat/completions` from socket to socket.

### 3.1 Startup — before any request

1. `main()` (`crates/onnx-genai-cli/src/main.rs:8`) → `run(argv)` in `crates/onnx-genai-cli/src/lib.rs`.
2. `Commands::Serve` (`cli/src/lib.rs:93-95`) → `run_serve` (`crates/onnx-genai-server/src/cli.rs:110`). The standalone `onnx-genai-server` binary calls the same function — **one serving path, not two**.
3. Model source resolution (`cli.rs:130-150`). A clap `ArgGroup` (`cli.rs:20-25`) requires exactly one of `--model` / `--models-dir` / `--models-config`, producing `Vec<ModelSpec>`.
4. `AppState::load_from_specs` (`cli.rs:152`) → `build_handle` per spec (`state.rs:348-390`) — **the single shared construction path** for both eager startup and lazy load.
5. `build_handle` resolves the directory (`ModelDirectory::load`, `onnx-genai-ort/src/loader.rs:35`), loads the tokenizer, then `Engine::from_dir` (`onnx-genai-engine/src/engine/load.rs:7`).
6. `EngineDriver::start(engine, DEFAULT_MAX_BATCH, max_queue_depth)` (`state.rs:379`) spawns the engine thread.
7. `app(state)` builds the router (`crates/onnx-genai-server/src/lib.rs:60-107`).

### 3.2 Arrival and admission

- Axum matches `POST /v1/chat/completions` (`lib.rs:71`); every request passes `trace_request` middleware (`lib.rs:107`).
- The handler resolves a `ModelHandle` from the `ModelRegistry`, applies the chat template, and tokenizes.
- The request is submitted to the driver over the `DriverCommand` mpsc channel. **Backpressure lives here:** the channel is bounded by `max_queue_depth`; over-depth submissions are rejected rather than queued without limit, and the rejection is counted in `metrics.rs`.

### 3.3 The engine thread and the path fork

`EngineDriver::start` (`driver.rs:113-123`) spawns a **dedicated OS thread** via `std::thread::Builder` — not a tokio task. Rationale in §6.

Inside `run_engine_driver` the decisive branch is `driver.rs:407-421`:

```
if engine.continuous_batch_manager(max_batch).is_ok() {
    // continuous batch path — logs "continuous batch driver enabled"
} else {
    // per-request path      — logs "continuous batch driver disabled"
}
```

`continuous_batch_manager` succeeds **only for static-cache models**. This single line determines which half of the system you are running. See §5.6.

The fork, and everything downstream of it:

```mermaid
flowchart TD
    A["run_engine_driver()<br/>driver.rs:576"] --> B{"engine.continuous_batch_manager(max_batch).is_ok()<br/>driver.rs:407-421"}

    B -- "Ok — STATIC-CACHE model" --> C["run_static_engine_driver()<br/>driver.rs:419"]
    B -- "Err — DYNAMIC-cache model<br/>⚠ the else-branch at driver.rs:420" --> D["run_fallback_engine_driver()<br/>driver.rs:421"]

    C --> C1["run_static_batch_until_idle()<br/>driver.rs:546"]
    C1 --> C2["manager.step() — ONE batched<br/>forward pass over all rows<br/>driver.rs:596"]
    C2 --> C1

    D --> D1["one request at a time<br/>engine owns kv_cache directly"]

    C1 -.->|"NEVER touches"| K["engine.kv_cache<br/>engine.prefix_cache"]
    D1 --> K

    K --> K1["PageTable::allocate()<br/>page_table.rs:836"]
    K --> K2["prefix trie<br/>reuse + CoW sharing"]

    style B fill:#fff3cd,stroke:#856404
    style D fill:#f8d7da,stroke:#721c24
    style K fill:#d4edda,stroke:#155724
```

**Read this diagram as the map of what is available where.** The dotted line is the whole story: the batched path never reaches the KV cache, so paged allocation, prefix reuse, and preemption are all absent on it — not broken, just never invoked (§5.6).

> ⚠️ **The most commonly missed line in this file is `driver.rs:420`** — the `else` branch. It is easy to read `run_engine_driver` and see only the batching path, because that is the one with the interesting code. But every dynamic-cache model takes the `else`, and *all* paged-KV and prefix-cache behaviour lives there. Instrumentation, logging, or error handling added only to the batch path silently does nothing for an entire class of models.

### 3.4 The decode step loop (continuous batch path)

- `run_static_batch_until_idle` (`driver.rs:546`) is the outer loop.
- Each iteration first drains newly arrived commands with `rx.try_recv()` (`driver.rs:577`), then calls `manager.step()` (`driver.rs:596`) — **one shared batched forward pass across all active rows**.
- **Only `Generate { session_id: None, .. }` is handled inline.** Every other command hits the catch-all at `driver.rs:592` and is pushed onto `deferred`, which is not drained until `driver.rs:520` — after the batch goes idle. This is invariant §5.11 and it is the reason telemetry must be collected inline rather than requested via a command.
- New arrivals are admitted at `driver.rs:610`; finished rows are backfilled so occupancy is maintained.
- Admission eligibility, when scheduler-driven, is `run_continuous_batch_scheduled` (`batched.rs:718`): FCFS order gated by the shared KV byte budget and total-token ceiling (`batched.rs:750-760`).
- Per-request events are funnelled through `route_continuous_events` (`driver.rs:637`) — **the single place where every request's lifecycle events pass**. TTFT is recorded at `driver.rs:650`.

**Output equivalence guarantee** (`batched.rs:708-711`): a request's tokens never depend on which rows share its batch. Batched output is byte-identical to running the request alone. Batching is a throughput optimization, not a semantic change.

### 3.5 KV allocation (per-request / paged path)

- `PageTable::allocate(device)` (`page_table.rs:836`) returns a free `PageId`, or `None` when the pool is exhausted.
- On exhaustion, eviction runs (`page_table.rs:1097-1110`), which deliberately **skips pages belonging to a live sequence or a retained prefix** (`ref_count > 1`).
- Prefix reuse: `PrefixCache::lookup_shared` (`prefix_cache.rs:92`) matches a token prefix, **increments the ref count of the matched pages**, and returns them for sharing rather than allocating and recomputing.
- `PageTable::free` (`page_table.rs:947-951`) decrements; the page returns to the pool only at `ref_count == 0`.

### 3.6 Streaming and teardown

- Generated token ids are detokenized incrementally and emitted as SSE chunks (`crates/onnx-genai-server/src/sse.rs`).
- The client observes TTFT as the first chunk's arrival.
- On completion the stream terminates, the session's pages are freed, and e2e latency is recorded into the histogram in `metrics.rs`.
- On client disconnect the response future drops; the driver stops routing events for that request.

---

## 4. Contracts

For each boundary: what the **caller** must guarantee, what the **callee** guarantees back, error semantics, and the consequence of violation.

### 4.1 Server ↔ EngineDriver

- **Transport:** mpsc `DriverCommand` with `oneshot` (single reply) or mpsc (streaming) response channels. Definitions from `driver.rs:72`.
- **Caller guarantees:** submits only tokenized, validated requests; respects `max_queue_depth`; holds the reply channel until the stream ends or drops it to signal cancellation.
- **Callee guarantees:** every accepted command receives exactly one terminal outcome (completion or error). The engine thread never blocks the async runtime.
- **Errors:** a dropped reply channel means the client vanished — the driver treats it as cancellation, not a fault.
- **Violation:** blocking on the reply inside an async handler without `spawn_blocking` stalls a tokio worker. Model construction is explicitly documented as blocking (`state.rs:341-347`).

### 4.2 EngineDriver ↔ Engine

- **Ownership:** the driver thread **owns** the `Engine` exclusively. No `Arc<Mutex<Engine>>` — single ownership is what removes lock contention from the decode loop.
- **Borrow facts that matter for instrumentation:** `continuous_batch_manager(&self)` (`batched.rs:599`), `page_usage(&self)` and `page_stats(&self)` (`engine/runtime.rs:247-253`) all take **immutable** borrows, so they are callable from inside the batch loop without restructuring.
- **Violation:** any attempt to reach the engine from another thread. Add a `DriverCommand` variant instead — `ResourceSnapshot` (`driver.rs:72`, async accessor `:383`, handler `:732`) is the pattern to copy.

### 4.3 Engine ↔ PageTable

- **Caller guarantees:** every `allocate` is eventually matched by a `free`; a page id is never used after its ref count reaches zero; a page shared with another sequence is never mutated in place.
- **Callee guarantees:** `allocate` returns a page not currently owned by anyone else, or `None`. `free` is idempotent-safe via `saturating_sub` (`page_table.rs:950`).
- **Errors:** exhaustion is `None`, not a panic — the caller decides between eviction, preemption, and rejection.
- **Violation:** mutating a shared page corrupts another sequence's KV silently. There is **no runtime guard** against this — see §5.3, an *assumed* invariant.

### 4.4 Engine ↔ ORT session

- **Caller guarantees:** input tensor shapes match the model's declared IO. For static-cache models this includes the `model.io.static_cache` declaration in `inference_metadata.yaml`.
- **Callee guarantees:** shapes and placement hints are validated at load; contradictory forced-placement hints are a hard error (`engine/load.rs:33-37`).
- **Violation:** a static-cache model missing its `io.static_cache` block **fails to load** — this is a real, observed failure, see §8.5.

### 4.5 Model directory boundary

- **`ModelDirectory::load`** (`loader.rs:35`) is the validation gate. It requires the root to be a directory (`:36-42`), then resolves `decoder.onnx` or exactly one `.onnx` (`:391`, `:412`) plus `tokenizer.json` (`:65-69`).
- **Canonical errors:** `model directory does not exist: {}` (`loader.rs:39`), `tokenizer.json not found in {}` (`loader.rs:69`).
- ⚠️ **Known duplication:** the server's `looks_like_model_dir` (`crates/onnx-genai-server/src/models_config.rs:161-165`) is a *second, laxer* filter used only for `--models-dir` fan-out. It accepts `tokenizer.json` **OR** `model.onnx` **OR** `genai_config.json`, so a directory can pass admission and then fail at load. See §8.6.

### 4.6 Tokenizer boundary

- **Caller guarantees:** the same tokenizer instance is used for encoding a prompt and decoding its output.
- **Why it matters:** `run_continuous_batch_scheduled` tokenizes every prompt **up front** and hands the manager token ids specifically so that "no re-tokenization can drift between the two" (`batched.rs:732-735`). Re-tokenizing mid-flight would desynchronize the scheduler's length accounting from the batch's actual rows.

### 4.7 `/v1/status` is a **node**-level contract with no model dimension

This is the clearest example in the codebase of a wire contract constraining an architecture decision, so it is worth stating precisely.

`GET /v1/status` returns `NodeStatus` (`routes/mod.rs:118-131`, handler `routes/admin.rs:41-86`). Its own doc comment defines the scope (`routes/mod.rs:110-116`):

> All values are model-agnostic; `node_id` names this node, never a model.

- **Consumer:** the cluster router, not just local tooling. It is a shared contract, not an internal detail.
- **Guarantee:** every field describes *the node*. There is no field identifying which model a number came from, and no place to put one without changing the struct.

**The consequence, spelled out.** Multi-model mode gives each model its **own** `EngineDriver` (`state.rs:376`), so two models really do run in one process. But `/v1/status` has no model dimension, so with two drivers behind one node it can only report one engine's numbers or blend them — and **a consumer cannot tell which**. Blending is the dangerous outcome precisely because the response still looks well-formed and plausible.

Adding a model dimension is possible but is a **breaking change to a contract another component consumes**. Running one server per model instead makes `/v1/status` unambiguous *by construction*: one engine per origin, no new fields, no migration, and no way to misattribute a number.

> **Guidance for anyone extending this endpoint:** if you find yourself wanting to add per-model fields to `NodeStatus`, prefer a separate model-scoped endpoint. Node-level and model-level data have different cardinality, and merging them silently breaks the guarantee the cluster router depends on.

---

### 4.9 Session ids on the wire are credentials — redaction is structural ⚠️ SECURITY

`/v1/status` returns `sessions[].id` (`admin.rs:69`), and unlike most of that struct it is **genuinely populated** — which makes it the field most likely to be bound by a consumer looking for something real to show. **A full session id is a bearer token**: possession of it is what authorises requests against that session. What appears on the wire is deliberately truncated — `sess-` plus the first 8 hex characters, then `…` (`session.rs:161-172`).

**The redaction is enforced by the shape of the API, not by remembering to call it.** Three properties do that, and they are worth preserving deliberately:

1. **There is exactly one id-listing accessor, and it redacts.** `client_ids_redacted()` (`session.rs:117`) is the *only* method that yields client ids. No unredacted sibling exists to reach for by mistake.
2. **Redaction happens inside the registry lock**, at `:125`, before the values escape. A caller never holds a full id, so it cannot leak one by accident.
3. **It fails closed.** An id not matching the expected `sess-<32 hex>` shape is replaced wholesale with `[redacted]` (`:170-171`) rather than passed through. An unrecognised format degrades to *less* disclosure, not more.

> **The consequence for anyone changing this.** Widening the redaction — showing more characters, or adding an unredacted accessor "just for debugging" — **leaks credentials into whatever consumes this endpoint**, and dashboards are exactly the kind of consumer that logs, screenshots and screen-shares its inputs. Truncated ids remain perfectly adequate for correlating rows in a UI, which is the only thing a consumer legitimately needs them for.
>
> **Contrast with §5.3.** This is what an *enforced* invariant looks like: violating it requires deliberately adding a new API, not merely forgetting a step. §5.3's copy-on-write rule protects something arguably more valuable and is enforced by nothing at all. **The difference is not importance — it is whether the type system was given the chance to help.**

---

### 4.8 The metrics registry is process-wide and has no model dimension

`metrics.rs:89` declares `static REGISTRY: Registry` — a single, process-global instance. Every counter it holds is flat (`metrics.rs:74-87`): `prefix_cache_hits`, `prefix_cache_lookups`, `batch_size`, `pending`, `active_sessions`, `rejections`, and the `ttft` / `e2e` histograms. **None is keyed by model.**

- **Guarantee:** these numbers describe *the process*, never a particular model.
- **Deliberate design:** the registry is a lock-free static specifically so recording a metric costs a relaxed atomic add and never allocates. That property is why it is safe to touch from the decode path at all (§5.10).

**The consequence for multi-model serving.** Loading two models gives each its own `Engine` and its own `EngineDriver` on its own thread (`state.rs:370`, `:376`) — so the two genuinely run concurrently, one batching and one paging. **But they share this one registry.** Their counters are summed, and nothing in the response says so.

The sharpest instance follows from §5.13: `prefix_cache_lookups` increments on **every completed generation** (`metrics.rs:130-135`), so in a two-model process, generations served by a *static-cache* model — which never consults the prefix cache at all — inflate the denominator of the *dynamic* model's prefix hit rate. **The displayed rate is not merely blended; it is actively depressed by unrelated traffic**, while looking authoritative.

Adding a model dimension means reworking a deliberately allocation-free static that sits on the hot path — precisely the change §5.10 warns against.

> **Why this is architecture, not trivia.** `static` means *per process*. Running one server per model therefore makes every counter model-scoped **for free**, with no hot-path change and no new fields: two processes are two registries. This, together with §4.7's `NodeStatus` having no model dimension, is why per-model observability is obtained by running separate processes rather than by extending either contract. Both are cases of choosing a topology that makes a guarantee structural instead of defending it with discipline.

---

## 5. Invariants

Each states the rule, **where it is enforced**, and what breaks. Critically: whether the code **enforces** it or merely **assumes** it.

### 5.1 Page accounting — ENFORCED

> A page is in the free pool if and only if `ref_count == 0`.

Enforced in `PageTable::free` (`page_table.rs:947-951`): decrement, and return to the pool only on reaching zero. `allocate` (`:836`) draws only from the free pool.

Corollary: `free_count(device)` (`page_table.rs:1150`) plus the count of pages with `ref_count > 0` equals capacity. **Breaks if violated:** double-free returns a live page to the pool, and two sequences then write the same physical KV.

*Note:* `free` uses `saturating_sub` (`:950`), which makes an extra free **silent** rather than a panic. Safe against underflow, but it means a refcount bug degrades quietly instead of failing loudly.

### 5.2 Reference counting for sharing — ENFORCED

> A page shared by N sequences (or retained by the prefix trie) has `ref_count == N`.

`PrefixCache::lookup_shared` (`prefix_cache.rs:92`) increments on match; release decrements. There is a test pinning exactly this (`prefix_cache.rs:296`, `lookup_shared_increments_and_release_decrements_page_refs`).

The prefix trie holds **its own** reference. So `ref_count == 2` with a single owning sequence means "one sequence plus a prefix-cache retention" — that is the mechanism by which a prefix survives its originating request.

### 5.3 Copy-on-write before mutation — **ASSUMED, NOT ENFORCED** ⚠️

> A page with `ref_count > 1` must never be written in place.

**Nothing at runtime prevents this.** `Page.ref_count` is a plain `pub u32` field (`page_table.rs:320`) and `Page.data` is a plain `pub Vec<f32>` (`:328`) — any holder of `&mut Page` can write a shared page. Correctness rests on callers checking the ref count first.

**This is the single most dangerous invariant in the codebase.** Violation corrupts another sequence's KV with no error, no panic, and no log — it surfaces as subtly wrong generated text in an unrelated request. **Any change touching page mutation deserves disproportionate review.**

### 5.4 Eviction respects liveness — ENFORCED

> Eviction never reclaims a page belonging to a live sequence or a retained prefix.

Enforced at `page_table.rs:1097-1110`, which filters to `ref_count <= 1` and documents the intent inline. **Breaks if violated:** an active sequence loses KV mid-generation.

### 5.5 Batch composition — ENFORCED

> Physical concurrency never exceeds `max_batch` decode rows; each admitted row reserves its worst-case KV footprint up front.

Enforced in `run_continuous_batch_scheduled` (`batched.rs:750-760`): the scheduler governs *eligibility* (ordering plus the shared token/byte budget) while the manager's `max_batch` bounds *row count*. Up-front worst-case reservation is what makes byte-budget admission sound (`batched.rs:717-718`).

`DEFAULT_MAX_BATCH` is currently **hardcoded to 4** (`crates/onnx-genai-server/src/state.rs:25`) with no CLI flag.

### 5.6 The static-cache requirement — ENFORCED (and load-bearing)

> Continuous batching engages **only** on static-cache models. Paged KV and continuous batching are **mutually exclusive**.

Enforced by the branch at `driver.rs:407-421`. `ContinuousBatchManager` (`batched.rs:101-110`) holds a `BatchedDecodeSession`, a tokenizer, and rows — **it never touches `engine.kv_cache`**. Static-cache models use runtime-owned in-place KV buffers, so there are no pages to page.

Because the paged KV cache is the owner of *both* the page table and the prefix trie, bypassing it bypasses both. A static-cache model therefore never consults the prefix index at all — the question is never asked, so there is no answer to report. This distinction matters when reporting these metrics: a bypassed subsystem is **not applicable**, which is a different fact from **not measured** and a different fact again from **measured as zero**. Reporting any of the three as the others is a correctness bug in the reporting layer, even though every underlying number is accurate.

Consequences, all verified by running the server:

| | Static-cache model | Dynamic-cache model |
|---|---|---|
| Continuous batching | ✅ enabled | ❌ disabled |
| Paged KV allocator | ❌ inactive | ✅ active |
| Prefix cache | ❌ unavailable | ✅ available |
| Preemption | ❌ **hardcoded off** | ✅ available |

**A prefix-cache hit count of zero while continuous batching is enabled is correct behaviour, not a bug.** Likewise a preemption counter would be permanently zero on that path.

#### 5.6.1 The three consequences of the split

The batching/paged-KV split is not one limitation. It is a single structural fact that surfaces in three unrelated-looking places, and each was discovered independently before the common cause was recognised:

| Feature | Where it dies on the batching path | Evidence |
|---|---|---|
| **Prefix cache reuse** | The batch never consults the prefix trie | `prefix_cache_hit_len` is a literal `0` at `batched.rs:262` and `:486`, so the `> 0` test at `metrics.rs:135` is never true |
| **Preemption** | Disabled by construction, not by default | `scheduler_config.preemption_policy = PreemptionPolicy::Disabled` (`batched.rs:759`) |
| **KV memory pressure / eviction** | Nothing to evict — rows are physical and pre-reserved | `batched.rs:713-718`: rows "cannot be swapped out and resumed in place" |

The rationale at `batched.rs:713-718` is the common cause stated in the source itself: **the batch owns its KV in physical rows, and each row reserves its worst-case footprint up front.** Pre-reserved physical rows are what make the batching path fast and predictable, and they are *precisely* what removes the freedom that sharing, eviction and preemption all require. **Every one of the three is the same trade, seen from a different angle.**

> **The practical rule.** Before adding any counter or panel for a KV-related behaviour, establish **which execution path it can fire on**. A metric can be correctly implemented, correctly plumbed, and permanently zero — and that is indistinguishable from a bug for anyone who does not know this invariant. Zero here means *structurally impossible*, not *not yet happening*, and the two must never be rendered the same way.

### 5.7 Preemption is disabled on the batched path — ENFORCED

> `PreemptionPolicy::Disabled` is set unconditionally for scheduler-driven continuous batching.

`batched.rs:757`. The reason is structural, not a policy choice (`batched.rs:713-717`):

> *"this batch owns its KV in the batched decode session's physical rows, which cannot be swapped out and resumed in place, so mid-flight eviction/swap of a running row is deferred."*

### 5.8 Output independence — ASSUMED (documented, not asserted)

> A request's output tokens do not depend on which other requests share its batch.

Documented at `batched.rs:708-711`. There is no runtime assertion; it follows from the batched forward pass being mathematically per-row independent. **Breaks if violated:** batching becomes observable to users, and results stop being reproducible.

### 5.9 Configuration asserts — ENFORCED, at construction

`page_table.rs:740-741` requires non-empty layer configs; `:776` requires every page tensor config to validate. These are `assert!`, so violation panics at construction — loud and early, which is correct for configuration errors.

### 5.10 The decode loop never blocks for observability — **ASSUMED, NOT ENFORCED** ⚠️

> No code inside the decode step loop may `.await`, acquire a lock, block, or allocate unboundedly — including for telemetry.

Nothing in the type system prevents it. The engine thread (`driver.rs:113-123`) is a plain OS thread that owns the `Engine` outright, so there is no borrow-checker or runtime guard that would reject a blocking call; it will compile and run and simply make every token slower.

**Why it holds:** the loop at `driver.rs:575-603` runs once per decode step for *every* in-flight request. Latency added there is multiplied by steps and by batch size, and it lands directly in inter-token latency — the number users feel most.

**What breaks if violated:** token generation stalls for every concurrent request at once. It degrades gradually rather than failing, so it survives review and shows up later as "the server got slower" with no obvious cause.

**Safe primitives:** relaxed atomics (the pattern already used throughout `metrics.rs:74-101`), or `tokio::sync::broadcast::send`, which is deliberately non-`async` and returns immediately even with no receivers.

### 5.11 Non-`Generate` commands are deferred until the batch drains — ENFORCED

> While a continuous batch is running, the **only** command processed inline is `DriverCommand::Generate` with `session_id: None`. Every other command is queued and not handled until the batch goes idle.

Enforced by the dispatch inside `run_static_batch_until_idle` (`driver.rs:575-594`): the `try_recv` drain matches `Generate { session_id: None, .. }` and submits it to the manager, and the catch-all arm at **`driver.rs:592`** pushes everything else onto `deferred`. That queue is only drained at `driver.rs:520`, after the batch loop has exited.

**Why it holds:** the `Engine` is single-owner with no interior locking (§6). Servicing an arbitrary command mid-batch would need mutable access the batch loop is already holding.

**What breaks if violated — and this is a live trap:** anything latency-sensitive implemented as a `DriverCommand` is answered **only when the server is idle**. A telemetry command is the worst case: batch occupancy, queue depth, and KV stats would be unavailable *precisely while the server is busy*, which is the only time they are interesting. A dashboard built that way appears to work in testing and freezes under load — looking like a UI bug rather than an architectural one.

> ⚠️ **Correction to earlier guidance in this project.** An earlier draft of the telemetry plan proposed adding a `DriverCommand` to fetch a `ResourceSnapshot`. That is correct for the per-request path but **wrong for the batched path**, for the reason above. Instrumentation for batching must be gathered **inline**, right after `manager.step()` (`driver.rs:595-603`), and published through an atomic or a broadcast channel. `Engine::continuous_batch_manager`, `page_usage`, and `page_stats` all take `&self`, so reading them inline is permitted.

### 5.12 Exactly one model-directory validator — **ASPIRATIONAL, CURRENTLY VIOLATED** ⚠️

> A directory is a valid model directory if and only if `ModelDirectory::load` (`onnx-genai-ort/src/loader.rs:35`) accepts it. No other component may define its own criterion.

**This invariant is currently false**, and stating it as intent rather than behaviour would violate this document's own honesty rule. `looks_like_model_dir` (`models_config.rs:161-165`) is a second, weaker validator that accepts on an **OR** of conditions where the loader requires an **AND**. A directory can therefore pass admission and then fail at load, producing contradictory error text (`models_config.rs:155-158`).

**Why the invariant is worth holding:** validation that disagrees with loading is unfalsifiable from the user's side — the error message names the wrong cause.

**What breaks while it is violated:** a user is told their model directory is fine, then told it does not exist. Two duplicated error strings (`state.rs:366` and `engine/load.rs:30`) make the origin ambiguous when debugging.

**Status:** the fix is funded — delete `looks_like_model_dir` and unify on `ModelDirectory::load(...).is_ok()`. Update this section to ENFORCED when it lands.

### 5.13 A metric's name is part of its contract — **ASSUMED, NOT ENFORCED** ⚠️

> A field must mean what its name says. Before consuming a metric, read the code that **increments** it, not the code that declares it.

**Why this is the sharpest observability trap in this codebase:** a stub is discoverable — someone greps and finds the hardcoded literal. **A correctly-computed number under a misleading name looks perfect forever.** It survives review, passes any "is this field populated?" check, and produces confident, precise, wrong conclusions.

Verified instances in this repo, all genuinely measured and all easy to misread:

| Field | Name implies | Actually counts | Evidence |
|---|---|---|---|
| `prefix_cache_lookups` | cache lookups | **completed generations** — incremented unconditionally, with no predicate | `metrics.rs:130-135` |
| `active_sessions` | concurrent requests | **persistent `X-Session-Id` sessions** — 4 concurrent stateless requests report `0` | `session.rs:73`, `:106` |
| `vram.used` | GPU memory in use | the scheduler's **KV byte-budget accounting** | `governor.rs:548`, `:554` |
| `host_ram.used` | this process's memory | **whole-machine** OS query, including every other process | `governor.rs:575-579` |

**What breaks if violated:** the failure is silent and self-confirming. `prefix_cache_lookups` is the cautionary case — it would read `5` on a build with the prefix cache **deleted entirely**, so any hit-rate derived from it is a ratio against an unrelated denominator.

**Rule for consumers:** if a name is wrong, **rename it at your boundary** to what it actually measures. Do not inherit a misleading name because upstream chose it. Naming `active_sessions` "concurrent requests" in a UI would be a fabricated measurement even though the number itself is correct.

### 5.14 A getter's existence does not mean the value survives the path — **ASSUMED, NOT ENFORCED** ⚠️

> Check that the field you need survives the *whole* call chain. A correctly-named function can compute exactly what you want and then discard it one line later.

Three independent instances:

- **`PageUsage` collapses page identity into a count.** `SequenceUsage.pages` is `pages.len()` (`page_table.rs:867-875`) — the `Vec<PageId>` is consumed to produce a length. The table knows *which* pages each sequence holds (`self.sequences`, `page_table.rs:619-620`), but that mapping never crosses the API boundary. **Consequence:** per-block sequence ownership — colouring a block grid by owning sequence — cannot be built from `page_usage()` as it stands.
- **`GovernorReconfigureOutcome` drops the eviction plan**, so a caller learns that reconfiguration happened but not what it decided.
- **`driver.rs:735-739` discards the reconfigure result** entirely.

**Why it holds:** each of these was a reasonable narrowing for its original caller — a length is all the original consumer needed.

**What breaks if violated:** you discover mid-implementation that the data was computed and thrown away, and the fix is an API change in a lower crate rather than the "just call the getter" you planned for. Widening the return type is usually the right fix; recomputing at the call site duplicates the invariant.

---

### 5.15 Telemetry must be *published* by the engine, never *requested* from it — ASSUMED ⚠️

The engine holds no concurrent-reader path. Both driver paths take an **exclusive borrow** for the whole of a generation:

- **Batched path:** non-`Generate` commands are deferred (`driver.rs:592`) until `manager.is_idle()` (`:604`), drained at `:520`. Under sustained load the batch may never idle. (§5.11)
- **Pipeline/fallback path:** `handle_driver_command(engine: &mut Engine, ..)` (`driver.rs:674`) runs `run_fallback_generation` **inline** at `:696`. The generation completes *inside* the command handler, so the next command — telemetry or otherwise — is not read until it returns.

**These look like two problems and are one.** The queue policy is not the cause; **`&mut Engine` is.** While a generation holds the exclusive borrow, no reader can observe the engine *at all* — not because a channel is busy, but because the borrow makes concurrent observation unrepresentable. Draining the queue faster cannot fix it, and neither can adding another command.

**The consequence is uniform and severe:** the engine is observable only when idle. Every interesting quantity — page allocation, block sharing, batch occupancy — exists **only during** generation. The system can be measured precisely when it has nothing to say.

**The shape that works** is to invert the direction. State written from *inside* the mutable borrow into shared, atomic storage (`Arc<...AtomicU64...>`) can be read from outside with **no borrow at all**: wait-free, no channel, no deferral, no borrow conflict — and cheaper in the decode loop than servicing a channel, since it is a relaxed store rather than a `try_recv` plus a reply.

**The approved implementation** is an `Option<Arc<KvTelemetry>>` of atomics on the engine, updated after each decode step, with HTTP handlers reading it **without ever touching the driver thread**. Note what this fixes and how: it does not make the queue drain sooner, it removes the dependency on the queue draining at all. **The `/v1/resources` hang (§8.10) is closed structurally rather than by timing luck** — the difference between a race made less likely and a race made impossible.

**One subtlety when placing the write.** `run_fallback_engine_driver` delegates to `handle_driver_command`, so instrumenting the shared handler covers the pipeline path too; there is no separate site to add for it. A redundant extra site would not be harmless — two writers to the same atomic publishing at different points in the step produce values that are individually valid and jointly inconsistent.

**Measured.** QA timed endpoint latency during a 384-token generation on a clean tree:

| Endpoint | Idle | During generation |
|---|---|---|
| `/metrics` | 0.8 ms | **14,784 ms** (5 polls completed) |
| `/v1/resources` | 0.8 ms | **14,785 ms** (5 polls completed) |
| `/v1/status` | 0.9 ms | 1.8 ms (61 polls, clean 4 Hz) |
| `/v1/debug/kv` | 0.8 ms | 1.9 ms (61 polls, clean 4 Hz) |

The split is exactly the invariant: the two slow endpoints await a driver round-trip, the fast ones do not. **The stall is the full duration of the generation, not a fixed penalty** — it scales with how much work the engine is doing, so it is worst precisely when observation matters.

> **Stated as a rule:** *observability must not traverse the command channel — publish, don't request.* Any telemetry modelled as a `DriverCommand` inherits this ceiling by construction and will read as **frozen under load and fine when idle**, which is the hardest failure mode to catch, because every test that does not hold load passes.

---

## 6. Concurrency model

### Threads

| Thread | Role |
|---|---|
| tokio runtime workers | axum request handling, SSE streaming |
| **one dedicated engine thread per model handle** | `EngineDriver::start`, `driver.rs:113-123`, spawned with `std::thread::Builder` |

### Why a dedicated OS thread, not a tokio task

Decode is a long, CPU-bound, uninterruptible compute step. Running it on a tokio worker would block that worker for the duration of a forward pass, starving unrelated requests. An OS thread isolates it completely.

The second-order benefit is the important one: **because exactly one thread owns the `Engine`, no lock is needed around it.** The channel *is* the synchronization. This is why the decode loop has no mutexes in it, and why it is fast.

### Ordering rules

1. Commands are processed **in channel order** — the mpsc queue defines admission order into the driver.
2. The scheduler may reorder *eligibility* among waiting requests (FCFS by default, `batched.rs:759`), but never reorders events **within** a single request.
3. Every request's events pass through `route_continuous_events` (`driver.rs:637`) — a single funnel, so per-request event ordering is total.
4. Replies are per-request channels, so responses to different requests are unordered with respect to each other. Callers must not assume completion order.

### Where the locks are

- **Not in the decode loop.** Deliberate.
- Metrics are **lock-free atomics** in a static registry (`crates/onnx-genai-server/src/metrics.rs:74-101`).
- The session registry and model registry use standard synchronization, but are touched per request, not per token.

### Rule for anyone adding instrumentation

> Never `.await`, never lock, never allocate unboundedly inside the decode step loop.

Use an atomic counter, or a non-blocking `tokio::sync::broadcast::send` (which is non-async and returns immediately when there are no receivers). Anything else puts scheduler latency into the token generation path.

---

## 7. Extension points

### Adding an execution provider

Implement the `onnx-runtime-ep-api` traits; follow `onnx-runtime-ep-cpu` (simplest complete reference) or `-ep-cuda`. Register so `SessionOptions` can resolve it.
**Must not break:** placement-hint validation (`engine/load.rs:33-37`) — an EP that silently ignores forced placement turns a hard error into wrong-device execution.

### Adding a sampler

Sampling lives in the generation core. Model authors' declared defaults are captured before the engine moves into the driver (`state.rs:373-375`), specifically so a model shipping `do_sample: true` is not silently forced to greedy.
**Must not break:** that defaults capture. Overriding it makes model-declared generation config unreachable.

### Adding a scheduling policy

Extend `PriorityPolicy` / `PreemptionPolicy` in `onnx-genai-scheduler`.
**Must not break:** the forbidden edge — the scheduler may not depend on the engine (§2). A policy needing engine internals is a sign the data should be passed in, not reached for. Note also that preemption policies are inert on the continuous-batch path (§5.7).

### Adding an HTTP endpoint

Register in `app()` (`crates/onnx-genai-server/src/lib.rs:60-107`). Choose a gate deliberately:

- ungated — safe for anonymous callers;
- `enable_debug_endpoints` (`lib.rs:77`, flag `--enable-debug-endpoints`, `cli.rs:74`) — introspection;
- `enable_admin_endpoints` (`lib.rs:88`) — mutating operations.

**Gated routes return `404`, not `403`**, because the route is never registered. Clients must treat 404 on a debug path as "disabled", not "missing".

If the endpoint needs engine data, add a `DriverCommand` variant — copy `ResourceSnapshot` (`driver.rs:72`, `:383`, `:732`). Do not reach into the engine from a handler.

### Error message convention

`What: / Why: / How:` — see `driver.rs:723-730` for the reference example. New errors should follow it.

---

## 8. Known gaps and stubs

The section that makes the rest of this document trustworthy.

### 8.1 Paged-attention kernels are not implemented

There is a paged **allocator** (`PageTable`, `PrefixCache`) that genuinely allocates, shares, reference-counts, and evicts pages. There are **no paged-attention kernels**. Attention does not read KV through the page table.

Say "paged KV block table", not "paged attention". The distinction is not pedantry — it is the difference between what is implemented and what is not.

### 8.2 KV introspection is stubbed at the server seam

`GET /v1/debug/kv` (`crates/onnx-genai-server/src/routes/admin.rs:118-141`) returns the literal string *"engine does not yet expose KV page statistics"* (`:140`).

**But the data already exists.** `Engine::page_usage()` and `Engine::page_stats()` (`engine/runtime.rs:247-253`) compute block utilization, per-sequence page counts, and allocation/free/eviction/failure counters; the underlying types are `PageStats` (`page_table.rs:564-579`), `PageUsage` (`:583-604`), `SequenceUsage` (`:607-614`).

The gap is **one missing `DriverCommand`**, not missing instrumentation. Anyone closing it must **delete the stub comments in the same change** — a stale `// not yet tracked` next to a live value is worse than the stub was.

### 8.3 `/v1/status` returns documented zeros — a real trap for consumers ⚠️

`NodeStatus` (`crates/onnx-genai-server/src/routes/mod.rs:118-131`) declares a rich set of fields. The **handler** (`admin.rs:41-86`) hardcodes most of them.

| Genuinely measured | Hardcoded |
|---|---|
| `node_id` `:45` · `healthy` `:47-51` · `queue_depth` `:59` · `active_sessions` `:61` · `sessions[].id` `:69` (redacted — full ids are bearer tokens) | `kv_usage` `:53` · `kv_pages_used` `:54` · `kv_pages_total` `:55` · `kv_pages_shared` `:56` · `paused_sessions` `:62` · `tokens_per_second` `:63` · `batch_utilization` `:64` · `sessions[].priority` `:75` · `sessions[].kv_pages` `:76` · `sessions[].state` `:77` · `prefix_hashes` `:81` |

The struct doc (`routes/mod.rs:110-116`) states the intent honestly: metrics the server cannot yet measure are *"reported as documented zeros/empties rather than fabricated"*.

**The trap:** a consumer can bind a dashboard to `kv_usage` or `tokens_per_second`, do everything else correctly, and display a fabricated measurement. **Verify a field is populated before depending on it.** Per-field comments in `status()` mark which are which.

Note `tokens_per_second` is honest about the reason — *"only cumulative token totals recorded"* (`:63`). The intended fix is for consumers to differentiate the cumulative counter over time, not to add windowing in the decode loop.

> **Planned change.** These fields are being wired to real measurements at their source rather than routed around. When that lands, the correct pattern for an unmeasurable field is `null` plus a machine-readable reason — **not** `0`. Any change that populates one of these fields must delete the corresponding `// not yet tracked` comment in the same commit; a stale marker next to a live value misleads worse than the stub did.

> **On how this list was established, and how much to trust it.** Two independent audits produced it: one reading forward from the engine toward the response, one reading backward from the response struct toward its sources. They agree on the same set, which is stronger evidence than either pass alone.
>
> **That convergence still failed once, and the failure is instructive.** Both audits initially placed `prefix_cache_lookups` on the honest side of the line. Both were wrong for the same reason: each read the field's *name* and neither read `metrics.rs:130-135` (see §5.13). Agreement is only independent evidence when the two paths do not share a premise — and a plausible name is a premise both readers inherit from the same place. The rule this list is built on is therefore **verify the field, not the rule**: confirm each entry at its cited line rather than trusting the table, including this one.

### 8.4 Prefix cache reports zero hits on the static-cache path

Observed: identical prompts yield `prefix_cache_hits: 0` with non-zero lookups when continuous batching is active.

Per §5.6 this is **expected**: prefix caching lives in the paged KV manager, which is inactive for static-cache models. Recorded here because it looks exactly like a bug. Whether the counters should instead report *unavailable* rather than zero is an open question — reporting `0` for a structurally unavailable feature is precisely the trap described in §8.3.

### 8.5 `scripts/build_qwen.sh` produces a model that cannot be loaded

`scripts/build_qwen.sh:32` passes `--runtime ort-genai`, which emits only `genai_config.json`. Loading the result fails because the runtime requires a `model.io.static_cache` declaration in `inference_metadata.yaml`.

The failure is confusing: the script succeeds, artifacts appear correct, and the model fails only at server start. Because continuous batching requires a static-cache model (§5.6), the practical effect is that batching silently never engages. A reproducible build recipe is captured as a skill in `.github/skills/build-static-cache-model/`.

### 8.6 Two model-directory admission filters

`ModelDirectory::load` (`loader.rs:35`) is the real validator. The server's `looks_like_model_dir` (`models_config.rs:161-165`) is a second, laxer filter for `--models-dir` fan-out that accepts `tokenizer.json` **OR** `model.onnx` **OR** `genai_config.json`.

Because the real loader requires `tokenizer.json` **AND** an onnx file, a directory can pass admission and then fail at load, with an error message (`models_config.rs:155-158`) that describes a contract the loader does not implement. Consolidating on `ModelDirectory::load(...).is_ok()` would remove the divergence.

Also asymmetric: the CLI applies `resolve_model_dir` (`crates/onnx-genai-cli/src/lib.rs:674`) to coerce a config-file path to its parent directory. **The server does not.** So `onnx-genai generate ./m/genai_config.json` works while `--model ./m/genai_config.json` fails.

### 8.7 Runtime VRAM override is inert

`POST /v1/admin/resources/vram-limit` (`admin.rs:160-181`) cannot shrink a live KV budget:

1. `allow_runtime_override` defaults to `false` (`config.rs:602`) and the server hardcodes `EngineConfig::default()` (`state.rs:152`) with no flag to change it — so the call returns `403`.
2. Even when enabled, `Governor::set_vram_limit` (`engine/governor.rs:163-174`) carries `TODO(§26.11.2)` for executing the eviction order. It moves the accounting ceiling; **resident KV is never released.** It affects new allocations only.

### 8.8 OTLP span export is deferred

`/v1/status` reports this explicitly rather than pretending it works (`routes/mod.rs:87-88`). Perfetto export is available at `/v1/debug/trace/perfetto`.

### 8.9 `max_batch` is not configurable

`DEFAULT_MAX_BATCH = 4` (`state.rs:25`) with no CLI flag, which also makes any batch-utilization percentage computed against it uninformative.

### 8.10 `/v1/resources` can block for the duration of a busy batch ⚠️

`GET /v1/resources` sends `DriverCommand::ResourceSnapshot(reply)` and awaits a oneshot. On the **continuous-batch path** that command is not `Generate`, so it hits the catch-all at `driver.rs:592` and is parked on `deferred`. The deferred queue is not drained until the batch loop exits, which happens only when `manager.is_idle()` (`driver.rs:604`).

**Under sustained concurrent load the batch may not go idle**, because finished rows are backfilled from new arrivals to maintain occupancy. The reply is therefore held for as long as the server stays busy — the request does not fail, it simply does not answer, and it eventually surfaces as a client-side timeout rather than an error the server reports.

Two consequences worth separating:

- **The endpoint appears to hang precisely when the machine is under load** — the condition a resource endpoint exists to report on.
- **A poller gets a burst of identically-stale values** when the batch finally drains, because several deferred snapshots are serviced back-to-back against the same post-drain state.

**Not the cause:** the pipeline driver arm is *not* at fault. It replies with an explicit `Err` for both `ResourceSnapshot` (`driver.rs:479-483`) and `SetVramLimit` (`driver.rs:485-489`) rather than dropping the oneshot, so pipeline models return a clean error instead of hanging. The deferral above is the whole mechanism.

**This is the practical face of invariant §5.11**, and it is why observability must be collected inline in the batch loop rather than requested through the command channel: *the command channel is not serviced during batch decode, which is exactly when observability matters most.*

---

### 8.11 `/metrics` inherits the driver round-trip for two gauges it treats as optional ⚠️

`prometheus_metrics` (`admin.rs:391-408`) does two things:

```rust
let mut output = crate::metrics::encode_prometheus();          // atomic registry, ~0.8 ms
if let Some(handle) = state.registry.resolve("")?
    && let Ok(snapshot) = handle.engine.resource_snapshot().await   // driver round-trip
{
    output.push_str(&crate::metrics::encode_resource_governor(&snapshot));
}
```

**The first line is already fast and already correct** — it reads the lock-free `static REGISTRY` (§4.8) and needs no engine. Everything genuinely measured on this endpoint (TTFT and e2e histograms, token counters, session and queue gauges) is produced there.

**The entire stall comes from the second part**, which exists only to append resource-governor gauges. Per §5.15 that `.await` parks behind the driver's exclusive borrow for the whole of a generation, so an endpoint that is 0.8 ms idle becomes ~15 s under load — **and the part that stalls is not the part carrying the real data.**

> **Why this is a small fix, not a redesign.** The handler is *already* written to tolerate absent resource gauges: the `if let Ok(..)` arm silently omits them when the snapshot fails. **Degrading gracefully is existing, intended behaviour** — so satisfying it from a cached or published snapshot rather than a round-trip requires no change to the response contract and no change to any consumer. The honest telemetry on this endpoint is being held hostage by two optional gauges.

---

### 8.12 Fields whose **names** are wrong while their **values** are correct ⚠️

This is the highest-value table in this document for a new contributor, and it is the one class of
defect against which **every mechanism described in §4 and §5 gives no signal at all**.

A fabricated value can be found: it is a literal, so it is greppable, and a zero invites suspicion.
**A correctly-computed value under a misleading name has no tell.** It is live, it moves when you
exercise the system, it is internally consistent, and it survives code review, unit tests, and a
careful reading of the struct definition — because nothing about it is *broken*. The only way to
catch one is to read the increment site and ask what actually causes it to move.

> **The rule this yields:** provenance is not *"is this field computed?"* — it is
> **"does this field mean what its name says?"** Trace every displayed number to the line that
> *changes* it, not to the struct that *declares* it.

| Field | What the name implies | What it actually counts | Evidence |
|---|---|---|---|
| `prefix_cache_lookups` | cache lookups | **completed generations** — `fetch_add(1)` is unconditional in `GenerationMetrics::result()` | **Verified** — `metrics.rs:133-135` |
| `prefix_cache_hits` | cache hits | **generations with *any* prefix overlap ≥1 token** (`if prefix_cache_hit_len > 0`) — a shared chat template alone satisfies it | **Verified** — `metrics.rs:136-137` |
| `prefix_cache_hit_rate` | hits ÷ lookups | **hits ÷ generations** — a real, useful per-generation rate, but not a hit rate | **Verified** (both terms above) |
| `batch_size_current` | the engine's decode batch | **live `GenerationMetrics` guards** — incremented in `start()`, decremented in `Drop`. On the dynamic server this is structurally ≤1 (§5.15); on the scatter server it is requests in flight, not decode rows. **It is never the batch size on either server.** | **Verified** — `metrics.rs:112`, `:145` |
| `vram` / `host_ram` (on `/v1/resources`) | memory used | **ceilings only** — sourced from `configured_limits` / `resolved_limits`. There is no consumption term anywhere in the payload, so **any utilisation ratio drawn from it invents its own numerator.** | **Verified** — `admin.rs:434-443` |
| `active_sessions` | concurrent requests | persistent `X-Session-Id` sessions — reads `0` at the busiest moment of a batching run, correctly | **Reported** (Lead), not independently verified here |
| `kv_usage` (on `/v1/status`) | KV utilisation | hardcoded `0.0`. **Not demo-only:** `RoutingPolicy::LeastKvUsage` sorts on it (`router.rs:247`), so the comparison cannot discriminate and the weighted policy silently loses its 30% term. | **Reported** (@d7cf9b84), traced cross-crate |

**Two things to take from the shape of this table rather than its contents.**

**First, the failures are not independent — five of the seven concern one subsystem** (prefix
caching and batching), because that is where §5.6.1's mutual exclusion lives. A field name written when
a capability was designed keeps its name after the execution path routes around the capability.
**The name records an intention; the increment site records what shipped.**

**Second, a guard derived from one of these incidents tends to be shaped like the incident rather
than like the fault.** The rule *"`hit_rate` must be unavailable when `lookups == 0`"* is correct and
does not fire here, because the denominator is the half that *works* — `135` really is 135
generations. The lie is in the numerator, and **no threshold on the denominator can ever detect
it.** When a ratio is suspect, audit its numerator and denominator **separately**: they usually have
different provenance, and here one is a live count and the other is a compile-time constant.

---

## 9. Where to look first

---

| Question | Start at |
|---|---|
| How does a request become tokens? | §3, then `driver.rs:407-421` |
| Why is my batching panel flat? | §5.6 — check whether your model is static-cache |
| Why is this metric zero? | §8.3, then the per-field comments in `admin.rs:41-86` |
| Where do I add an endpoint? | §7, `lib.rs:60-107` |
| Can I call this from the engine thread? | §4.2 — check whether the accessor takes `&self` |
| Is this invariant enforced or assumed? | §5 — assumed ones are marked ⚠️ |
