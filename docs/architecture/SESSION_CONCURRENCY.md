# Session concurrency

**Authoritative for:** what the public session API promises about threads, how
sessions are owned and executed concurrently, and what may never be shared.

This document records both the design and its implementation status. Sections
marked ✅ describe shipped code; unmarked decisions remain normative future
work. The current server implementation supports opt-in distinct-session
parallelism only for the contracted single-decoder ORT path described in Phase
3. It makes no concurrency claim for native, composite, speculative,
graph-capture, or provider configurations that do not report concurrent `Run`
support.

**Two prerequisites have already merged, and the text reflects that rather than
proposing them again.**
[#2012](https://github.com/justinchuby/onnx-genai/pull/2012) (`1bf87c86`) built
the worker abstraction, `WorkerId`/`SessionPlacement`, and a `WorkerPool` of one
— §1.1 and §4 are written on top of it.
[#2019](https://github.com/justinchuby/onnx-genai/pull/2019) (`27b8945e2` +
`f3c89255a`) gave session-derived ORT handles a structural `Arc<Session>`
ownership edge and shipped `crates/onnx-genai-ort/src/thread_affinity.rs`. The
remaining engine primary/draft/EAGLE-3 holders and their persistent runners now
use that shared ownership too, so §3.4 Rule A's holder conversion is complete.
The explicit `Value`→allocator and environment ordering caveats remain, and the
native-affinity half of §13 Phase 1 is still pending. Phase 3 therefore enables
multiple workers only for the narrower ORT configuration stated above.

Related: [`DESIGN.md`](DESIGN.md) for the execution architecture,
[`../genai/SCHEDULING.md`](../genai/SCHEDULING.md) for inter-session scheduling
policy, [`../genai/PIPELINE.md`](../genai/PIPELINE.md) for the workflow
interpreter, [`../genai/INFERENCE_METADATA_DECISIONS.md`](../genai/INFERENCE_METADATA_DECISIONS.md)
§12.5a for the normative session-lease contract, and
[`ERROR_AND_LOGGING_CONVENTIONS.md`](ERROR_AND_LOGGING_CONVENTIONS.md) for how
the refusals below must be typed.

---

## 1. What is true today

### 1.1 The engine is structurally `!Send` and server-owned by one of N threads

`Engine` no longer has a hand-written `unsafe impl Send`. `EngineDriver`
transfers only a `Send` construction plan into each OS thread.
`WorkerHandle::spawn` constructs the engine there, returns a typed readiness
handshake containing no engine/backend state, runs the command loop, and drops
the engine before the thread exits. Explicit pool shutdown closes and joins
every worker.

`WorkerPool` now holds one or more `WorkerHandle`s. The default
`--ort-session-workers 1` follows the historical path and keeps its batching
heuristic unchanged. Values greater than one are opt-in and are accepted only
when the loaded package is a contracted single-decoder ORT engine whose
`Session::supports_concurrent_run()` capability is true. Unsupported requests
fail model load with an actionable typed error; they never silently fall back
to one worker.

The worker factory shares only immutable/lifetime resources: `Arc<Session>`,
`Arc<Environment>`, `Arc<Tokenizer>`, and `Arc<WorkflowPlan>`, plus the
process-wide device memory authority needed for exact global accounting. Every
worker constructs fresh KV caches, scheduler/governor policy, workflow mutable
state, bindings, values, allocators, samplers, graph-capture ids, diagnostics,
session maps, and local id allocators on its owner thread.

Session creation reserves the healthy worker with the fewest live plus pending
sessions, breaking ties by lowest `WorkerId`. The returned
`SessionPlacement { worker, engine_session_id }` is the routing identity for all
later operations. Worker-local engine session ids intentionally may collide;
the placement keeps them distinct. Stateless work uses the fewest active turns
with the same tie break. Continuous batches are created and advanced only
inside one worker and never span workers.

A worker panic marks only that worker failed. Its placements become typed
unavailable and are not migrated or restarted; other workers remain routable.
Pool startup is all-or-nothing, and a later-worker initialization failure
shuts down and joins every worker already reported ready.

The Python binding reaches the same conclusion by a different route: it holds
`inner: Mutex<RustEngine>` (`crates/onnx-genai-python/src/lib.rs:187-188`) and
refuses contention outright rather than blocking, raising `PyRuntimeError` with
the message *"engine is in use by another thread — onnx_genai Engine is not
re-entrant; serialize calls or use one Engine per thread"*
(`python/src/lib.rs:173-184`).

So today's server answer is: one session remains single-flight, while distinct
sessions may execute in parallel only across explicitly configured, supported
ORT workers. This is a serving guarantee, not a claim that the low-level
`Engine` itself is a concurrent shared object.

### 1.2 Historical inner interpreter lease analysis

> ✅ **Superseded in part by Phase 2 (§13).** The "unreachable" half of this
> heading described the *interpreter's* lease, and it was accurate when written.
> The routing lease of §4.2 has since shipped, so
> `PackageCapabilityError::ExclusiveLeaseConflict` is now raised from the routing
> layer for every session-bearing turn and is reachable over HTTP. Everything
> below still stands as the description of the interpreter's own lease, which is
> kept, unpromoted, and still narrow — read it that way rather than as a
> statement about the runtime today.

The metadata contract already declares this. A session-scoped state cell carries
a `SessionLeaseContract` whose `policy` defaults to `exclusive`
(`schema/inference_metadata.schema.json:3215-3254`), and the normative
specification states the consequence:

> A lease declared `policy: exclusive` is single-flight: a second turn that
> starts while one is in flight is refused by name rather than allowed to read a
> conversation the first is about to replace.
> — [`../genai/INFERENCE_METADATA_DECISIONS.md`](../genai/INFERENCE_METADATA_DECISIONS.md):1103-1105

The runtime implements that refusal. `SessionLeaseGuard::acquire` inserts into
the lease set or returns
`PackageCapabilityError::ExclusiveLeaseConflict { session }`, and `Drop` removes
the entry so a pass that fails or panics does not strand the session
(`crates/onnx-genai-engine/src/pipeline/workflow.rs:598-633`, acquired at
`workflow.rs:1363`). The error is typed, is marked retryable
(`crates/onnx-genai-engine/src/engine/capability.rs:50-61`), and the server maps
it to HTTP 409 by matching the variant rather than the wording
(`crates/onnx-genai-server/src/routes/mod.rs:523-525`).

**Two separate facts stop that from being Decision 3, and the second is the one
that is easy to miss.**

**First, it cannot be reached.** The server's own test says so:

> `// 3. ExclusiveLeaseConflict — the driver serializes passes, so this is`
> `//    raised where it is decided and mapped where it is answered.`
> — `crates/onnx-genai-server/src/tests.rs:4858-4859`

The test constructs the error by hand and checks the mapping, because no HTTP
request sequence can produce it. A `RefCell<HashSet<String>>` on a
single-worker driver cannot observe two turns in flight.

**Second — and this is the part the citation trail does not advertise — the
lease does not cover most sessions.** The worker's lease set is read at
exactly one place, and only under two conditions:

```rust
let session_state = onnx_genai_metadata::classify_session_state(workflow);
let _session_lease = match (self.session_id.as_deref(), session_state.carries_any()) {
    (Some(session_id), true) => {
        match SessionLeaseGuard::acquire(&engine.worker.session_leases, session_id) {
```
— `crates/onnx-genai-engine/src/pipeline/workflow.rs:1390-1397`

The guard is taken only when the *interpreted workflow* declares session-scoped
cells that carry. That path is reached only when the engine does **not** hold a
decode core:

```rust
if !self.holds_decode_core() {
    ...
    return self.generate_in_workflow_session(
```
— `crates/onnx-genai-engine/src/engine/runtime.rs:1231-1237`

A decode-core package — the ordinary LLM case — takes the other branch, keeps
its conversation in `Engine::sessions: HashMap<SessionId, EngineSession>`
(`engine/model.rs:55-56`), and **acquires no lease of any kind**. There is no
`SessionLeaseGuard` on that path; the interpreter's lease set has exactly four
non-test references in the entire workspace, all of them in the interpreter
(`pipeline/runtime_state.rs:279`, `:320`, `workflow.rs:1397`, and the guard
itself at `workflow.rs:631-660`).

The two even use different identifier types. The interpreter's lease set is
keyed by `String` (`pipeline/runtime_state.rs:279`, and `session_id: Option<String>` at
`workflow.rs:258`); the decode core is keyed by
`SessionId = SequenceId = u64` (`config.rs:389`, `onnx-genai-kv/src/lib.rs:61`).
The interpreter is handed a stringified copy of the numeric id at the boundary
(`engine/runtime.rs:780`), so the two spaces agree by convention, not by type.

So the honest summary is:

| Path | Conversation state | Lease before Phase 2 | Refused a concurrent second turn? |
|---|---|---|---|
| Interpreted workflow (`!holds_decode_core`) | interpreter session cells | `workflow_session_leases`, keyed by `String` | would, if two could be in flight |
| Decode core (ORT) | `Engine::sessions`, keyed by `SessionId` | none | no |
| Decode core (native) | `Engine::native_sessions` (`engine/model.rs:78-100`) | none | no |

✅ **All three rows now answer "yes", from one place.** Phase 2's routing lease
(`server/src/lease.rs`) is keyed by `ModelSessionPlacement` and is taken before
the turn is enqueued, so which of these three paths ends up serving it is not
something the refusal depends on. The interpreter's row keeps its own guard as
the inner invariant; the two decode-core rows are covered by the routing lease
alone.

This changes what §2's Decision 3 asks for. It is **not** "make the existing
lease reachable". It is:

> **A new lease is required at the routing layer, keyed by the typed public
> session identity (`SessionPlacement`, not a `String`), covering the decode-core
> and interpreted paths alike.** The interpreter's existing per-pass guard stays
> as a deeper, package-declared invariant; it is not promoted into the public
> concurrency contract, because it does not cover the majority of sessions and
> is keyed by the wrong thing.

What *is* settled by the existing code is the refusal's **vocabulary**: the
error type, its retryable classification, its 409 mapping, and the normative
sentence in the metadata spec. §4.2 reuses all four. What is unsettled is where
the lease lives, what it is keyed by, and when it is taken — which is §4.2 and
§4.2.1.


### 1.3 Which backend handles are thread-affine

This is the constraint that decides everything else, so it is stated from the
code rather than from folklore.

**Not thread-affine, per their own safety comments:**

| Handle | Location | Claim |
|---|---|---|
| `Session` (ORT) | `crates/onnx-genai-ort/src/session/mod.rs:1389-1396` (`Send`), `:1397-1406` (`Sync`) | `Send` unconditionally; `Sync` because "ORT documents concurrent `Run` on one session as safe *when the providers underneath it are*, which is precisely what `Session::supports_concurrent_run` reports" |
| `Environment` (ORT) | `crates/onnx-genai-ort/src/env.rs:326-335` | `Send + Sync`; process-level handle ORT permits from multiple threads |
| `CublasLt` | `crates/onnx-runtime-ep-cuda/src/blas.rs:97-101` | "a cuBLASLt handle is not thread-affine" |
| `PinnedStaging` | `crates/onnx-runtime-ep-cuda/src/runtime.rs:2123-2127` | a page-locked host allocation; the pointer is a plain address |

**`Session` is the only ORT handle in that list that a session's execution
actually reaches, and the exception does not extend to anything derived from
it.** Since #2019 the `Sync` safety comment says so itself: *"Sharing a
`&Session` never hands out an `IoBinding`, `Allocator`, or `Value` — those are
`!Send + !Sync` and stay with the thread that created them — so `Sync` here does
not make the non-thread-safe handles shareable"* (`session/mod.rs:1402-1405`).
That is now a fact about the types rather than a hope: all three are structurally
`!Send + !Sync`, as the `thread_affinity` module records — *"Rust already forbids
that for a value the compiler can see: `IoBinding`, `Allocator`, and `Value` are
`!Send + !Sync`, so a shared reference cannot reach a second thread"*
(`crates/onnx-genai-ort/src/thread_affinity.rs:11-13`). Decision 4 therefore
places `IoBinding`, `Allocator` and `Value` in the per-worker category (§3.2)
even though `Session` itself may be shared.

**`Session`'s `Sync` is conditional, and the condition is now a queryable
capability.** `supports_concurrent_run()` answers from the execution providers
the session actually resolved to plus its graph-capture state, and fails closed
(`session/mod.rs:861-871`). Its doc comment states the consequence this document
depends on: *"A session worker pool may only share one `Session` across threads
when this is `true`; otherwise each worker needs its own session"*
(`session/mod.rs:850-856`). `concurrent_run_support()` returns the reason for a
refusal rather than a bare `false`, and there are exactly three refusals
(`session/mod.rs:1352-1386`):

| Refusal | Why |
|---|---|
| graph capture enabled | capture/replay is per-session state keyed by `gpu_graph_id`; "a second concurrent run would capture the first run's buffers" |
| no resolved execution provider | "nothing declares that concurrent `Run` is safe" |
| any provider lacking `capability::CONCURRENT_RUN` | ORT's session-level contract extends only as far as every provider under it |

Each refusal names the remedy — *"Give each worker thread its own session"* —
which is precisely §4's model, so the backend and this design already agree on
the answer.

**A graph-capture session is therefore not concurrent, even though `Session` is
`Sync`.** A session created with `enable_cuda_graph=1` reports
`graph_capture == true` (`session/mod.rs:492`, accessor `:846-848`) and replays
one captured graph against one set of bound device addresses — which is why
island binding exists *"so CUDA graph capture sees stable, device-resident input
and output addresses"* (`session/mod.rs:1262-1264`). #2019 made that structural:
graph capture is the *first* refusal `concurrent_run_support` checks
(`session/mod.rs:1356-1364`), and `run_affinity` turns a second overlapping run
into a named error rather than a corrupted capture (`session/mod.rs:503-508`,
`:874-889`). Combined with the thread-local capture rules below, a
capture-enabled session is **single-flight and thread-affine**, and the `Sync`
impl on `Session` does not reach it.

**Thread-affine or context-bound, per their own safety comments:**

| Handle | Location | Claim |
|---|---|---|
| `CudnnBackend` | `crates/onnx-runtime-ep-cuda/src/cudnn/mod.rs:1128-1132` | "cudarc deliberately keeps `Cudnn` !Send/!Sync because a handle must not be used concurrently … `with_handle` binds the owning CUDA context to the calling thread first" |
| `RawReduceHandle` | `.../cudnn/mod.rs:960-963` | "runs on the thread bound to the owning CUDA context" |
| `CudnnReduceCache` | `.../cudnn/mod.rs:1098-1101` | same |
| `CudaGraphLifecycle` | `crates/onnx-runtime-ep-cuda/src/graph.rs:130-135` | every segment launches on its single owning stream |
| `CudaReservation` | `crates/onnx-runtime-cuda-memory/src/virtual_memory.rs:2749-2752` | "thread-affine, and every driver call through the backing binds the context first" |

**CUDA graph capture is thread-local, and the repo already knows it.** Capture
begins with `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` (`graph.rs:177`), must end on
the thread that began it (`graph.rs:186-195`), must abort on that same thread
(`graph.rs:274-285`), and the allocator's capture gate is a `thread_local!`
depth counter (`crates/onnx-runtime-cuda-memory/src/capture_gate.rs:93-100`).

The shape of the constraint is therefore precise, and it is not "CUDA is
single-threaded". It is: *a handle whose safety comment names a bound context or
an owning stream is correct only when the same thread that bound it uses it, and
CUDA graph capture additionally requires begin/end/abort on one thread.* A lock
does not supply thread identity. Serializing access to a context-bound handle
from two different threads satisfies the mutual-exclusion half of its contract
and violates the affinity half, and the failure that produces is not a data race
the compiler or a sanitizer will name — it is a capture that silently belongs to
the wrong thread.

### 1.4 The lifecycle defect that set the precedent, and how it was fixed

The categories in §1.3 are not theoretical. Until
[#2019](https://github.com/justinchuby/onnx-genai/pull/2019) — landed as
`27b8945e2` and `f3c89255a` — there was a live drop-order defect at `W = 1`. It
is fixed. It is kept here in the past tense because it is the sharpest available
evidence for why §3's ownership rules are structural rather than documented, and
because the shape of the fix is now the precedent the rest of this document
builds on.

**What was wrong.** Rust drops struct fields in declaration order, and
`ExecutionIsland` declared `session: Session` *before* `bindings` and
`device_allocator`. So ORT ran `ReleaseSession` first and freed the session-owned
state that the island's `IoBinding` and its `CreateAllocator` device allocator
still pointed into. `IoBinding` held an unenforced `*const Session` back-pointer;
`Allocator` distinguished ownership with an `owned: bool` and held no session at
all. `WorkflowRuntime::Drop` hid half of it by clearing bindings by hand, but it
never cleared `device_allocator`, so `ReleaseAllocator` still ran after the
session was gone. Nothing in the types said the order mattered.

**How it was fixed: ownership, not ordering.** `IoBinding<'s>` and
`Allocator<'s>` now either borrow their session or co-own it through
`Arc<Session>`, and which one a value holds is the proof:

```rust
enum BindingSession<'s> {
    /// The binding borrows the session, so the compiler rejects any owner that
    /// could outlive it.
    Borrowed(&'s Session),
    /// The binding co-owns the session. Owners that hold the session and the
    /// binding in the same struct need this: struct fields drop in declaration
    /// order, so field order alone would decide the release order, and a shared
    /// owner takes that decision away from whoever edits the struct next.
    Shared(Arc<Session>),
}
```
— `crates/onnx-genai-ort/src/binding.rs:17-26`

`Allocator` carries the same distinction as `AllocatorOrigin::{ProcessDefault,
SessionBorrowed, SessionShared}` (`crates/onnx-genai-ort/src/allocator.rs:207-219`),
which decides both who releases the allocator and how long the session behind it
must live. `PipelineModels::sessions` is now `BTreeMap<String, Arc<Session>>` so
component code can take the owning form (`crates/onnx-genai-ort/src/loader.rs:429-436`),
handed out by `shared_session` (`loader.rs:581-587`).
`ExecutionIsland::session` is an `Arc<Session>` whose doc comment states the
rule in the type's own words — *"`bindings` and `device_allocator` below each
hold their own `Arc<Session>` and the refcount, not the field list, decides"*
(`crates/onnx-genai-engine/src/pipeline/islands.rs:75-83`) — and
`StableIslandBinding` holds an `IoBinding<'static>` that co-owns it
(`islands.rs:117-119`). A `compile_fail` doctest with a positive control holds
the borrowed form to its lifetime (`binding.rs:46-70`).

**What the fix did not reach, and why the manual clears stay.** The compensating
teardown in `WorkflowRuntime::Drop` was **not** deleted, and this document does
not ask for it to be. `Value` is a bare `OrtValue` handle with **no
back-reference to the allocator whose memory it holds**, so ORT still requires
every value to be released before that allocator, and no refcount expresses it.
The `Drop` impl now says exactly that:

> Bindings and session-derived allocators now co-own their `Arc<Session>`, so
> they can no longer outlive the session whatever order the fields drop in. A
> `Value` cannot: it is a bare `OrtValue` handle with no back-reference to the
> allocator whose memory it holds, so ORT still requires every value to be
> released before that allocator. Nothing in the type system says so, which is
> exactly why it is done here explicitly instead of left to the field list.
> — `crates/onnx-genai-engine/src/pipeline/mod.rs:189-208`

That is the correct scope for the remaining manual code: it no longer papers over
a session/binding ordering problem that the types now solve, and it survives only
for the value/allocator edge that they do not. §13 Phase 1 therefore does *not*
list deleting it, and §3.4 does not claim `Arc<Session>` permits removing it.

**Why this still belongs in a concurrency document.** Two reasons, both forward-
looking rather than historical.

First, the defect was a `W = 1` bug that would have been multiplied by the pool:
the same teardown happens `W` times, concurrently with other workers still
running, on failure paths where nobody runs a compensating clear — and a
`ReleaseIoBinding` against a released session is not a data race any sanitizer
will name. That is why the ownership work was a prerequisite for sharding and not
a cleanup to do afterwards, and it is why §13 Phase 1's backend half is already
complete rather than still ahead.

Second, the remaining `Value` → `Allocator` edge is a real constraint on §4.3.
Because it is enforced by an explicit teardown rather than by the type system, a
worker that owns values and allocators must run that teardown itself, on its own
thread, in that order. §4.3 states it as a worker obligation for exactly this
reason.

### 1.5 Where session state, memory accounting, and KV actually live

- **Session maps.** `Engine` holds `sessions: HashMap<SessionId, EngineSession>`
  for decode-core packages and `workflow_sessions: HashMap<SessionId, usize>`
  for interpreted ones, plus `workflow_session_ids: SharedSessionIds` — the
  atomic allocator those worker-local ids are minted from
  (`engine/model.rs:55-72`, `engine/ids.rs`). The native backend holds a third map,
  `native_sessions: HashMap<SessionId, NativeSessionState>`, with LRU eviction
  bounded by `native_max_sessions` (`engine/model.rs:78-100`,
  `engine/runtime.rs:621-640`).
- **`SessionId` is not its own type.** `pub type SessionId = SequenceId`
  (`crates/onnx-genai-engine/src/config.rs:389`) and
  `pub type SequenceId = u64` (`crates/onnx-genai-kv/src/lib.rs:61`). On the ORT
  path `create_session` *returns the KV sequence id directly*
  (`engine/runtime.rs:1538`), so the session identifier and the paged-KV
  sequence identifier are the same number. This matters in §4.1.
- **Memory accounting has one process-wide device authority.** Each worker owns
  a private governor, scheduler budget, host/disk ledger, and diagnostics. Their
  device-tier leases charge the same authority, and the primary fixed-weight
  plan is retained until the last engine sharing the ORT session drops. Thus
  weights are charged once while worker-local KV is charged once per worker.
  Underneath,
  `ByteBudget` is already `Arc<Mutex<BudgetState>>` and documents itself as *"A
  shared, dynamic, cross-session KV byte budget"* whose clones "account against a
  single running total"
  (`crates/onnx-genai-scheduler/src/byte_budget.rs:122-133`), `HostGovernor`
  keeps its ledger behind `Mutex<Ledger>`
  (`crates/onnx-genai-scheduler/src/pressure.rs:416-431`), and
  `HostGovernorAccounting` bridges to it through `Mutex<Outstanding>`
  (`crates/onnx-genai-scheduler/src/host_lease.rs:50-60`).
- **KV is one table per worker.** `PagedKvCache` owns a `PageTable` whose
  sequence ids are unique within that instance
  (`crates/onnx-genai-kv/src/paged_cache.rs:57-62`). `SessionPlacement` supplies
  the worker qualification outside it.
- **Continuous batching does not carry sessions.** `run_static_engine_driver`
  states plainly that *"The current `ContinuousBatchManager` API accepts
  `GenerateRequest` only. `X-Session-Id` requests keep using the driver's
  per-request engine path"* (`driver.rs:814-817`). Sessionful traffic is on the
  serial path by construction.
- **Fork is declared and disabled.** `session_fork_capability` returns `None`
  unconditionally and `fork_session` bails with a message naming the missing
  capability (`engine/runtime.rs:1628-1659`). The capability token in the
  signature is the mechanism that keeps callers from asking.
- **Copy-on-write leases are refused at load.** The policy enum admits
  `copy_on_write` (`schema/inference_metadata.schema.json:3248-3254`) and the
  loader rejects it (`crates/onnx-genai-engine/tests/workflow_session_continuation.rs:695-702`).

---

## 2. The decisions

These are the commitments this document makes. Sections 3–13 derive from them.

1. **Server session APIs are thread-safe.** A caller may hold one handle and
   call it from any thread, without external synchronization, and get defined
   behaviour. This is a property of the type, not of a documented convention.
2. **Different sessions can execute concurrently when `W > 1` is explicitly
   configured and supported.** The default remains `W = 1`; unsupported
   backends/configurations fail closed rather than pretending to enable
   parallelism.
3. **The same session under an exclusive lease refuses, by type.** A second turn
   on a session whose lease is `policy: exclusive` returns
   `PackageCapabilityError::ExclusiveLeaseConflict`. It does not queue silently
   and it does not interleave: the refusal is the contract, not a fallback. The
   lease also spans the whole read→write commit, which is what rules out a lost
   update — §11.1 is careful about the difference between those two properties.
   This requires a **new** lease at the routing layer; §1.2 shows why the
   existing one does not qualify.
4. **Backend handles stay thread-affine and are not made `Sync` by wrapping raw
   handles in locks.** Where a handle names a bound context or an owning stream,
   the design supplies a thread, not a mutex. Affinity is enforced structurally
   where the type allows it (a worker-owned resource that is not `Send` cannot be
   moved at all) and by typed errors otherwise, and a resource derived from a
   session owns that session (`Arc<Session>`) rather than pointing at it — §3.4.
   Both are structural because §1.4 shows a convention already failed here, and
   both are already shipped for the ORT layer by #2019.
5. **Execution uses a bounded pool of session workers with deterministic session
   ownership and stateless load distribution.** A session belongs to exactly one
   worker for its entire life; the routing decision is read from its typed
   placement.
6. **State is split three ways** — immutable package/compiled plan, per-worker
   backend handles, per-execution mutable state — and each way has a different
   sharing rule.
7. **A global `Mutex`/`RwLock` over the engine is rejected**, and the condition
   under which that rejection would be revisited is stated in §11.1.

### 2.1 What "thread-safe" is being promised

Precisely: every public session-facing method is callable from any thread at any
time, and the outcome is one of

- the operation completes, or
- the operation is refused with a typed error naming why,

with no undefined behaviour, no torn state, and no silent reordering of two turns
on one session. It is explicitly **not** a promise that concurrent calls on one
session both succeed. Decision 3 says the opposite, on purpose: a caller that
asked to take a busy session is told so, rather than being parked behind a
decode that may run for thousands of tokens with no way to see the wait.

---

## 3. State split

Everything reachable from a session handle is classified into exactly one of
three kinds. The classification is the design; the worker pool in §4 is only
what enforces it.

### 3.1 Immutable: package and compiled workflow plan

The loaded package, its `InferenceMetadata`, its validated hints, the resolved
capability set, the compiled workflow plan (node graph, typed SSA contracts,
resolved component bindings, placement decisions), and the memory strategy plan.

**Rule: shared by every worker behind `Arc`, never mutated after load.** These
are `Sync` because they are frozen, not because a lock guards them. Loading once
and sharing is also the only way a bounded worker pool is affordable — cloning a
package per worker would multiply host memory by the worker count.

The plan/interpreter split is what makes this possible, and its first half has
landed: `WorkflowRuntime` now holds the compiled plan as an `Arc<WorkflowPlan>`
separate from the worker state its lease set lives in
(`pipeline/runtime_state.rs:100-279`). What remains is making the interpreter a
per-execution object constructed against that plan rather than a long-lived one
that opens a pass on it.

### 3.2 Per-worker: backend handles

The ORT `Session` and its IO bindings, the native decode session, the CUDA
context binding, the cuDNN handle, the captured CUDA graph and its lifecycle,
the pinned staging buffers, the device sampler, the paged `PagedKvCache` and its
`PageTable`, the prefix cache, and any `CudaReservation` backing them.

**Rule: owned by exactly one worker, never referenced from another thread,
constructed on the worker thread and dropped on the worker thread.** §4.3 states
the construction and teardown obligations that follow, because they are the
sharp edge.

Note that ORT's `Session` is `Send + Sync` (§1.3) and would not strictly need
this. It is placed here anyway. Three reasons. First, ORT's concurrency
guarantee is about `Run`, and the objects around a run — `IoBinding`,
`Allocator`, `Value`, decode state, captured graphs, the CUDA context they were
bound under — do not inherit it, and none of them carries any `Send`/`Sync`
claim of its own (§1.3). Second, a capture-enabled session is single-flight and
thread-affine regardless of the `Sync` impl (§1.3), so the exception would have
to be carved back out for exactly the configuration that matters most on CUDA.
Third, a rule that holds only for the subset of handles that happen to be `Sync`
today is a rule that breaks the first time an execution provider ships a handle
that is not, which is exactly the failure mode `Engine`'s own safety comment
already anticipates: *"This would stop being sound if an execution provider
introduced thread-affine handles"* (`engine/model.rs:227-228`).

**`Session` is therefore the only ORT handle this design permits to be shared at
all**, and only through `Arc` for *ownership* (§3.4), never for concurrent use
across workers. `IoBinding`, `Allocator` and `Value` are per-worker without
exception.

### 3.3 Per-execution: mutable turn state

The request, its sampling configuration and RNG counter state, the SSA value
environment for one workflow pass, the decode loop's scratch, the accumulated
output tokens, the held exclusive lease guard, and the memory leases charged for
this turn.

**Rule: created at the start of a turn, owned by the executing worker, destroyed
at the end of the turn — including on error, cancellation, and panic.** Nothing
here outlives its turn, which is what makes lease release a `Drop` obligation
rather than a cleanup path someone has to remember to call.

Session-scoped state — the conversation cell a `session.continuation` names — is
deliberately *not* in this category. It is per-session state living on the owning
worker: category 3.2 storage under category 3.3 access discipline.

### 3.4 Session-derived resources own the session, and know their thread

The three categories above say *where* state lives. This section says how the
type system is made to enforce it, because §1.4 is proof that a categorisation
maintained by convention does not survive contact with a refactor.

Two rules, both structural. **Both are now shipped for the ORT layer.** #2019
established the derived-handle edge and affinity vocabulary; the remaining
engine holder/runner conversion completes Rule A for every current ORT session
holder. What follows states each rule and the narrower ordering caveats that
remain.

**Rule A — ownership, not adjacency.** Every resource derived from a `Session`
holds an `Arc<Session>`.

**For ORT session-derived resources this is done.** #2019 added the ownership
edge in exactly the shape this rule asks for:

- `IoBinding` replaced its non-owning `*const Session` with
  `BindingSession { Borrowed(&'s Session), Shared(Arc<Session>) }`
  (`crates/onnx-genai-ort/src/binding.rs:17-26`), and
  `IoBinding::for_shared_session` yields the `IoBinding<'static>` a worker needs
  (`binding.rs:100-102`).
- `Allocator` carries `AllocatorOrigin { ProcessDefault, SessionBorrowed(&'s Session), SessionShared(Arc<Session>) }`
  (`crates/onnx-genai-ort/src/allocator.rs:207-219`) instead of nothing, with
  `Allocator::for_shared_session_device` producing `Allocator<'static>`
  (`allocator.rs:294-300`).
- `Session::shared_device_allocator` is callable only on an `Arc<Session>`
  (`session/mod.rs:1178-1183`) and its doc states the resulting guarantee:
  the allocator is released *"before the session no matter which field the owner
  declared first"* (`session/mod.rs:1175-1177`).
- The holders that hand these out were converted with it:
  `PipelineModels::sessions: BTreeMap<String, Arc<Session>>`
  (`crates/onnx-genai-ort/src/loader.rs:435`) and
  `ExecutionIsland::session: Arc<Session>`
  (`crates/onnx-genai-engine/src/pipeline/islands.rs:83`).

**One edge is still missing, and it bounds what Rule A currently buys.** `Value`
does not hold a back-reference to the `Allocator` it was allocated from, so a
device `Value` can still outlive its allocator by field order. The `Drop for
WorkflowRuntime` doc comment states this as the reason its manual clears remain
(`crates/onnx-genai-engine/src/pipeline/mod.rs:189-200`), and the ORT test helper
spells out the order a caller must keep: *"`Value` (frees memory through the
allocator), then the `Allocator`, then …"* (`value.rs:2443-2448`). §4.3 therefore
carries "release values before their allocator" as an explicit worker obligation
rather than treating Rule A as complete.

**The remaining `Box<Session>` holders are converted.** Today:

| Holder | Type | Citation |
|---|---|---|
| `Engine::session` | `Option<Arc<Session>>` | `engine/model.rs` |
| `Eagle3Model::session` | `Arc<Session>` | `engine/model.rs` |
| `DraftModel::session` | `Arc<Session>` | `crates/onnx-genai-engine/src/session.rs` |
| `MtpModel::session` | `Arc<Session>` | `engine/model.rs` |
| stored runner session owner | `OrtSessionOwner::Shared(Arc<Session>)` | `crates/onnx-genai-ort/src/session_owner.rs` |

The engine no longer extends a borrowed `&Session` to `'static` for persistent
decode runners. `DecodeSession`, `StaticCacheDecodeSession`,
`MtpDecodeSession`, and `Eagle3DecodeSession` use the same named borrowed/shared
session owner: the shared form retains `Arc<Session>`, while each runner still
owns its own mutable `IoBinding`, allocator, device values, KV/cache state, and
graph-capture state. `Arc` preserves the stable pointee address that the old box
choice supplied, including when an engine moves or another immutable session
handle is cloned, without making any mutable worker resource shared.

This is shared **ownership**, not a concurrency decision. The capability remains
`Session::supports_concurrent_run()` / `concurrent_run_support()`, and `W = 1`
continues to issue runs serially. A future holder must use the same shared owner
before it stores a derived handle; it is not permitted to reintroduce a raw or
borrow-extended session pointer.

The payoff, where the edge exists, is that teardown order stops being a property
of source-file layout. `ExecutionIsland` now says so in its own field doc:
*"Field declaration order alone cannot promise that — it silently reverses the
moment a field is moved during a refactor — so `bindings` and `device_allocator`
below each hold their own `Arc<Session>` and the refcount, not the field list,
decides"* (`pipeline/islands.rs:75-83`).

**`WorkflowRuntime::drop`'s manual clears stay.** It is tempting to read Rule A
as licence to delete them, and that is wrong: they were never only about
bindings. The binding and allocator half is now structural, but the value half
is not, because `Value` has no back-reference to its allocator
(`pipeline/mod.rs:189-200`). §13 Phase 1 explicitly retains this `Drop`. A
compensating teardown may only be deleted once the structural edge that replaces
it exists for every resource it covers — §3.5 states that sequencing rule in
general, and this is its first live application.

### 3.5 What Rule A does *not* fix: the environment above the session

Rule A ties a session to the resources below it. It says nothing about the
handle above it, and that gap has to be stated or the "no field-order
dependencies" rule in §10 would be claiming more than it delivers.

**ORT requires the environment to outlive every session, and nothing in this
repository structurally enforces that.** The requirement is written down:

> ORT requires the `OrtEnv` to outlive every `OrtSession`. Releasing the session
> after its environment is a use-after-free that crashes with
> STATUS_ACCESS_VIOLATION at `ReleaseSession`.
> — `crates/onnx-genai-ort/src/value.rs:2394-2397` (and again at `:2443-2448`)

And `Environment`'s own SAFETY comment describes the discipline it depends on:
it *"releases it once from `Drop` after owning structs have dropped their
sessions"* (`env.rs:326-335`). But `Session` holds no environment handle at all —
its fields are the ORT pointer, names, I/O metadata, EP bookkeeping and, since
#2019, `run_affinity` (`session/mod.rs:479-508`), and it merely borrows
`&Environment` during construction (`session/mod.rs:517`, `:531`, `:629`).

**#2019 did not change this, and the one place that compensates shows why the
gap is real.** `WorkflowRuntime` holds `_ort_environment: Option<Arc<Environment>>`
as its **last** field, documented as *"Kept last so the ORT environment, its
registered allocator bridge, and their plugin/provider teardown outlive every
component and execution-island session that may still call back into them"*
(`pipeline/mod.rs:183-186`). That is a
per-owner, field-order-dependent workaround for a missing structural edge — the
exact pattern Rule A removed one level down. It is a useful partial precedent
(the environment *is* already an `Arc` that an owner can co-own) and it is not a
fix, because any other owner that forgets to keep it last is wrong again.

The environment lives in a process-wide
`OnceLock<Mutex<EnvironmentLifecycle>>` (`env.rs:45-48`), and EP plugin
registration is likewise ambient — *"ORT plugin registration is process-global"*
(`crates/onnx-genai-ort/src/session/plugin.rs:120-123`), serialized through a
global registry (`env.rs:50-55`, `:255-257`).

So session → environment is exactly the same category of defect as §1.4's
session → binding — the one #2019 fixed one level down — and it is **out of
scope for this document to fix**. Two options exist and the choice is deferred to
the implementation, not elided:

- **Structural:** `Session` gains an `Arc<Environment>`, and the environment
  becomes un-droppable while any session lives. This is the honest fix and would
  let §10's rule apply without exception. It touches every session constructor
  and the plugin registry's lifetime story, which is why it is not folded into
  Phase 1.
- **Scoped:** the environment stays ambient and process-lifetime, and the §10
  rule is explicitly scoped to *session-derived* resources, with this section as
  the written record of why the outer edge is still controlled by discipline.

Until one is chosen, two things follow, and both are normative:

1. **§10's no-field-order rule is scoped to resources derived from a `Session`.**
   It does not yet claim anything about `Environment`, providers or plugins.
2. **No compensating teardown may be deleted before the structural lifetime that
   replaces it exists.** This applies to `WorkflowRuntime::drop`
   (`pipeline/mod.rs:189-208`), to `WorkflowRuntime`'s last-field environment
   (`pipeline/mod.rs:183-186`), and to `Environment::drop`'s ordering discipline
   (`env.rs:308-324`) alike. Deleting a manual teardown is the *last* step of a
   conversion, never the first, and §13 Phase 1 sequences it that way.

**Rule B — thread affinity, structural first, then typed, never a panic in
`Drop`.** Affinity is enforced in three layers, in order of preference.

**This rule is no longer a proposal for the ORT layer.** #2019 shipped
`crates/onnx-genai-ort/src/thread_affinity.rs`, which implements all three layers
in the shape below. What this section now does is (a) state the semantics
precisely, because a reader who assumes "pinning" will design the wrong routing
layer, and (b) scope the remaining work, which is the native/CUDA handles and
the `Engine` container.

**The shipped rule is *exclusive use*, not *pinning*.** This distinction is
load-bearing for §4 and §5 and the module states it directly
(`thread_affinity.rs:17-31`):

> * A resource has at most one owning thread **while a guarded section is live**.
>   A second thread entering one is a hard, reported violation.
> * Ownership may move to another thread **while the resource is idle**. …
>   Pinning ("only ever the constructing thread") would reject that legitimate
>   handoff — the server builds a model on a `spawn_blocking` thread and then
>   drives it from a dedicated driver thread — so it would be a rule this
>   codebase violates by design.

Migrations are therefore legal and **counted**, not refused
(`OwnerThread::migration_count`, `thread_affinity.rs:201-203`). This document
must not be read as requiring a resource to live forever on its constructing
thread. §4.3's obligation is narrower and matches the module: a resource is
constructed and dropped *on a thread that owns it at that moment*, and it is
never **concurrently** reachable from two. A worker that takes ownership of an
idle session's resources at handoff is doing the supported thing; §5's session
create/fork/reset routing relies on exactly that.

**B.1 — Make the type structurally `!Send`.** The strongest enforcement is the
one the compiler performs, and for ORT this is done: `IoBinding`, `Allocator` and
`Value` are `!Send + !Sync` (`thread_affinity.rs:11-13`), and the guard type
`ThreadAccess<'a>` carries `_not_send: PhantomData<*const ()>`
(`thread_affinity.rs:374-379`) so a live guarded section cannot cross a thread
either. That marker is protected by a `compile_fail` doctest paired with a
positive control, so a future refactor that makes it `Send` fails the build
rather than the invariant (`thread_affinity.rs:326-372`). `IoBinding` carries the
same pairing for its lifetime edge (`binding.rs:46-70`). The pre-existing
precedent this followed is `CapturedGraph`, *"intentionally neither `Send` nor
`Sync`"* (`crates/onnx-runtime-ep-cuda/src/graph.rs:103-108`).

Structural `!Send` is not always achievable — ORT's `Session` is `Send + Sync`
(§1.3), and some handles must be constructed in one place and installed in
another. Where it is achievable it is mandatory; where it is not, B.2 and B.3
apply. **The remaining B.1 work is the container, not the leaf**: the module doc
names it — *"What the compiler cannot see is a resource smuggled inside a
container that carries its own `unsafe impl Send` (`onnx_genai_engine::Engine`
does), and that is the case this module names"* (`thread_affinity.rs:13-15`).
§10 is that case.

**B.2 — Normal operations return a typed affinity error.** For anything already
fallible, a wrong-thread use is an ordinary error value, not a panic. This is
shipped as `OwnerThread::enter(operation) -> Result<ThreadAccess, ThreadAffinityError>`
(`thread_affinity.rs:219-251`): a `Shared` resource takes one branch and returns
an unguarded token, an `Exclusive` resource takes one uncontended
`compare_exchange`, re-entry on the owning thread bumps a depth counter, and a
foreign thread gets a typed error. The error is actionable in the RULES.md Rule 1
sense by construction — it names the resource, the operation, the owning thread,
the offending thread, the creating thread, and the fix
(`thread_affinity.rs:113-140`). Call sites take the guard before touching the
handle, e.g. `IoBinding::bind_input` (`binding.rs:131-137`) and
`Value::empty_in` through `Allocator::enter` (`value.rs:247`,
`allocator.rs:337-340`).

This extended an existing precedent rather than inventing a policy: the CUDA
graph code already returned a typed wrong-thread error,

```rust
CaptureState::Capturing(owner) if owner == std::thread::current().id() => {}
CaptureState::Capturing(_) => {
    return Err(EpError::KernelFailed(
        "cuda_ep: CUDA graph capture must end on the thread that began the \
         thread-local capture"
```
— `crates/onnx-runtime-ep-cuda/src/graph.rs:188-195`, same shape for `abort` at
`:277-283`.

The routing layer converts a `ThreadAffinityError` into a 500 with the worker
named, because a caller cannot fix it and an operator can. It is **not** a
`ExclusiveLeaseConflict` 409: a 409 means "you sent two overlapping turns", an
affinity error means "the server routed one to the wrong thread".

**B.3 — Teardown never panics.** `Drop` is the one place where an assertion is
actively harmful: a `Drop` running during unwind that panics aborts the process,
so a wrong-thread teardown reached from an existing panic would turn a
recoverable worker failure into a hard abort, and §7's "degrade to `W-1`" promise
would be a lie. A design cannot both panic in `Drop` and claim graceful
degradation.

The shipped module honours this. `OwnerThread::check(operation)` is the
`Drop`-side form for callers that cannot return an error
(`thread_affinity.rs:254-272`), and `release()` **declines a foreign release**
rather than forcing one: *"a foreign release is reported and *declined* — the
resource stays held. That can strand it as permanently busy, which fails closed:
later use is refused with a name instead of racing"*
(`thread_affinity.rs:288-294`). Both the double-release and wrong-thread cases
report through `tracing::error!` plus `debug_assert!`
(`thread_affinity.rs:304-322`) — so a release build reports and continues, and
`Drop for IoBinding` (`binding.rs:282-299`) and `Drop for Allocator`
(`allocator.rs:347-369`) inherit that behaviour. `#2019`'s follow-up commit
(`f3c89255a`) exists precisely to stop a guard from releasing a resource it never
took.

The workspace's richer vocabulary for a partially failed release is in
`onnx-runtime-memory-api`, and the native side reuses it — it defines
*"what is true after a release partially fails"* (`crates/onnx-runtime-memory-api/src/deferred.rs:5-6`):

- `AllocationReleaseState::Quarantined` — *"Ownership is retained deliberately
  because releasing it would be unsafe or dishonest"* (`deferred.rs:70-73`).
- `QuarantineReason`, whose variants already include `OwnerDropped`,
  `MechanismTerminated`, `DeviceLost`, `EnqueueRejected` and `StatePoisoned`
  (`deferred.rs:153-174`).
- `DeferredReleaseQueue: Send + Sync + Debug` with
  `fn enqueue(&self, request: PreparedAllocationRelease) -> Result<(), DeferredEnqueueError>`
  (`deferred.rs:427-435`), returning `DeferredReleaseDisposition::{Queued, Quarantined}`
  (`deferred.rs:440-449`).
- `Drop for PreparedAllocationRelease`, which quarantines an abandoned request
  and — the sentence that settles this — *"never frees, never blocks, and never
  loses metadata"* (`deferred.rs:25-27`, impl at `:787-805`).
- `ProviderContextPin` / `ProviderContextPinSource`, whose module doc states the
  principle directly: *"A pin does not ask a context to stay alive; it makes
  teardown observe the outstanding work"* (`context_pin.rs:15-19`).

The same shape appears elsewhere: `Drop for CapturedGraph` records the error
through the context and destroys what it can, without panicking
(`graph.rs:82-99`); `Drop for OwningAllocation` quarantines rather than freeing
(`crates/onnx-runtime-memory-api/src/binding.rs:2102-2119`); and
`Drop for PooledStaging` degrades to *"Leak-safe fallback only: free the buffer,
never return it to the pool"* (`crates/onnx-runtime-ep-cuda/src/pinned_pool.rs:273-279`).

**So the normative teardown rule is:** a per-worker resource dropped on a thread
that is not its owner must, in this order,

1. emit telemetry naming the resource, its owner thread and the dropping thread —
   never `panic!`, and never conditioned on `std::thread::panicking()`, which
   appears nowhere in the workspace today and would make the behaviour depend on
   whether a panic happened to be in progress. For ORT resources this is already
   `OwnerThread::check` plus the declined `release`;
2. hand the release to the owning worker's deferred-release queue if that worker
   is still alive, which is the `DeferredReleaseDisposition::Queued` case; and
3. quarantine the ownership if it is not, with `QuarantineReason::OwnerDropped`
   for a dead owner or `MechanismTerminated` for a torn-down context —
   deliberately retaining memory rather than making a driver call from the wrong
   thread. The ORT analogue is `release()`'s "stays held" outcome: a stranded
   resource, reported, never a wrong-thread free.

Quarantine leaks device memory, and that is the point: leaking a reservation is
a bounded, observable, reportable cost, and freeing it from the wrong context is
undefined behaviour. RULES.md Rule 9's "never silently OOM" is satisfied because
quarantine is neither silent nor unaccounted — it flows through the same ledger
the governor already reads (§8).

**The one permitted abort.** If a violation is detected where neither deferral
nor quarantine can express the outcome — a partially-released allocation whose
ownership can no longer be described — the process aborts against a **named fatal
invariant**, logged as such. In that case §7 must not claim `W-1` degradation,
and it does not: §7's degradation promise is explicitly scoped to worker panics
and construction failures, both of which unwind normally and release through the
paths above.

**Which handles carry an owner-thread record.** Done, by #2019:
`IoBinding` (`binding.rs:82`), session-derived `Allocator`
(`allocator.rs:243`), and `Session` itself for the run path
(`run_affinity`, `session/mod.rs:507`) — the last declared `Exclusive` only when
`supports_concurrent_run()` is false, so a graph-capture session is single-flight
and affine while a concurrently-runnable one takes the `Shared` fast path
(`session/mod.rs:874-889`). `Value` is covered transitively: it can only be
allocated through a guarded `Allocator::enter` (`value.rs:247`).

Still outstanding, and the scope of §13 Phase 1's remainder: the native decode
session, the CUDA context binding, `CudnnBackend` and its reduce cache,
`CudaGraphLifecycle`, `CudaReservation`, and the paged KV cache. Several of these
already enforce affinity in their own idiom (§1.3); the work is to express it in
one vocabulary so the routing layer can report one error shape.

---

## 4. The session-worker model

### 4.1 Deterministic ownership and stateless routing

A bounded pool of `W` worker threads is created at engine load. `W = 1` is the
default; an explicit `W > 1` is either honored exactly or refused at load — it
is never clamped silently (§4.4). Each worker owns its own category-3.2 state
and shares the category-3.1 plan.

✅ **Ownership is explicit in `SessionPlacement`.** Every engine keeps its
existing worker-local `SessionId`; those numbers may collide across workers.
The server stores and routes the pair
`SessionPlacement { worker, engine_session_id }`, then qualifies it by model
where a process-global identity is required. No bare local id is accepted by a
session-bearing driver operation.

✅ **Session creation is deterministic least-loaded placement.** Under one
selection lock, the pool reads each healthy worker's live plus pending session
count, chooses the minimum, and breaks ties by lowest `WorkerId`. The pending
reservation is incremented before the lock is released, so two simultaneous
creates cannot both observe the same stale tie. Success atomically converts the
pending count to live; every error, cancellation, failed send, or failed worker
releases it.

✅ **Stateless placement is deterministic turn balancing.** The pool reads
queued plus active turn counts, chooses the minimum, and breaks ties by lowest
`WorkerId`. A `WorkerTurnGuard` increments before enqueue and decrements by
`Drop`, including send failure, cancellation, backend error, panic, and normal
completion. The count is scheduling state only; it is never used to find a
session.

Placement is a one-shot ownership decision. A placed session **cannot migrate**
because its mutable backend/KV state is thread-affine. Later skew therefore
changes where new sessions are placed, never where an existing one is routed.
If its worker fails, the placement becomes typed unavailable and must be
recreated; it is not silently rebound to a healthy worker.

### 4.2 The routing lease

✅ **Shipped (Phase 2).** The lease described below exists as
`crates/onnx-genai-server/src/lease.rs`: `SessionLeases`, a sharded map of
`ModelSessionPlacement`, owned by the `SessionRegistry` — the one routing-layer
map of client id → conversation, shared by every loaded model — and acquired
through `SessionRegistry::acquire` before any command is built. `SessionLeaseGuard` is the RAII half, and it is `#[must_use]`. The
paragraphs below are kept in the present tense as the design they describe; where
what landed is narrower than what they specify, §13's Phase 2 entry says so by
name.

This is the section the whole document turns on, because §1.2 established that
the lease Decision 3 needs **did not exist** before that — the interpreter's
`workflow_session_leases` covers interpreted workflow sessions only, is keyed by
`String`, and is absent from every decode-core path.

**A new lease is introduced at the routing layer.** It is keyed by the typed
public session identity — the `SessionPlacement` that #2012 already threads
through the driver, the session registry and the routes (`worker.rs:66-85`),
qualified by the model that owns the engine which issued it — and it applies to
every session-bearing turn regardless of which execution path serves it. Decode-core ORT sessions, decode-core native sessions, and
interpreted workflow sessions are all covered by one lease, because a caller
holding one session id should get one answer about what a concurrent second turn
does.

The interpreter's existing `SessionLeaseGuard` (`pipeline/workflow.rs:598-633`)
is **kept, not replaced, and not promoted.** It remains a package-declared
invariant one level down: it protects the interpreter's own session cells during
a pass, which is a narrower and different claim than the public concurrency
contract. Two guards at two layers is the correct shape here, not duplication —
the routing lease answers "may this turn start?", the interpreter guard answers
"is this pass the only one touching these cells?".

What the routing lease reuses verbatim is the existing **vocabulary**:
`PackageCapabilityError::ExclusiveLeaseConflict { session }`
(`engine/capability.rs:50-53`), its retryable classification
(`capability.rs:57-61`), the chain-walking extractor (`capability.rs:64-72`),
and the 409 mapping that matches on the variant rather than the wording
(`routes/mod.rs:517-527`). None of those change. Only the place the error is
raised, and the key it is raised against, are new.

Where the lease state lives is a consequence of §4.1, not a free choice. Since
routing reads the immutable placement and a session belongs to exactly one
worker for its whole life, contention on the lease map is contention between
*request-handling* tasks, not between workers. It is therefore **one map in the
routing façade** — sharded to keep the critical section short — holding
`ModelSessionPlacement → LeaseState`, and owned by the `SessionRegistry`, which
is the thing that holds the bindings the lease is about. It is *not* a
`Mutex<HashSet<String>>` dropped into `WorkflowRuntime` where the old one was,
and it is *not* per-worker state — a per-worker map cannot be consulted before
the command reaches the worker, and §4.2.1 explains why that timing is the whole
point.

✅ **The key is model-qualified, and the map is one map.** A `SessionPlacement`
names a worker and an engine session id, and both are per-engine: each engine
numbers its own sessions and each pool starts at worker 0, so two loaded models
routinely produce the identical placement for two unrelated conversations. A
per-`EngineDriver` map would additionally have meant that the *session registry*
— which is global across models — had to pick which engine's map to consult for
a binding, and picking wrong is precisely the failure the lease exists to
prevent. Both are fixed by the same decision: the key is
`ModelSessionPlacement { model: ModelKey, placement: SessionPlacement }`
(`server/src/lease.rs`), the map is owned by the `SessionRegistry`, and every
route acquires through it. A driver learns its own `ModelKey` once, in
`ModelHandle::new`, and refuses a lease naming any other model.

#### 4.2.1 The lease is acquired before enqueue, not inside the worker loop

This is normative and it is the single easiest thing to get wrong.

If the lease were acquired where the old one is — inside the pass, on the worker
thread — then a second turn on a busy session would be **accepted, queued behind
the first, and eventually succeed**. It would not be refused. The caller would
see a slow 200, not a 409, and Decision 3 would be violated in exactly the way
Decision 3 exists to prevent, while every test that only checks "the error type
exists and maps to 409" would still pass.

So:

**Acquisition happens in the routing façade / session registry, on the calling
task, before the `DriverCommand` is constructed and before it is enqueued to the
worker channel and before any admission or queue-depth accounting is charged.**
A turn that cannot take the lease never becomes a command, never occupies a
queue slot, never consumes an admission permit, and is answered with 409 from
the same task that tried to take it.

The ordering is:

```
route handler
  ├─ resolve SessionPlacement          (session registry, #2012)
  ├─ ACQUIRE routing lease  ─────────► on failure: ExclusiveLeaseConflict → 409, stop here
  ├─ charge admission / queue depth
  ├─ build DriverCommand, MOVING the lease guard into it
  ├─ send to WorkerHandle::sender()   ─► on send failure: guard drops here, lease released
  └─ await completion                 ─► guard travels with the command and is
                                          released when the turn ends, however it ends
```

**The guard travels with the command.** It is owned by the `DriverCommand`, so
its release is a `Drop` obligation rather than a cleanup path someone has to
remember on each exit. That matters because there are five distinct ways a turn
can end and all five must release:

| Ending | Where the guard is dropped |
|---|---|
| normal completion | worker drops the command after emitting the final event |
| error during the pass | same — the guard is in the command, the command is dropped |
| client cancellation / abandoned route | worker's abandoned-route handling drops the command (`driver.rs:1394-1403`, `:1418-1421`) |
| channel send failure (`DriverStopped`) | the guard is still on the sending task and drops when the send returns `Err` (`driver.rs:502-506`, `:510-523`) |
| worker panic or shutdown | the channel closes, queued commands drop, §7 releases the rest |

The send-failure row is the one that is easy to miss. `EngineDriver`'s submit
paths already convert a closed channel into `GenerateSubmitError::DriverStopped`
(`driver.rs:502-506`, `:510-523`); because the guard is moved into the command
and the command is moved into `send`, a failed send returns ownership and the
guard drops on the spot. No explicit release call exists, and none should — an
explicit release is a line someone can forget to add to a sixth exit path.

**Consequence for phasing and testing.** Acquisition before enqueue is what
makes the 409 reachable, and it does not require `W > 1` and does not require
intra-worker multiplexing. One worker serving two concurrent HTTP requests for
one session is enough: the second request's acquisition fails on its own task
while the first is still in flight. §13 therefore lands the routing lease in
**Phase 2**, at `W = 1`, and §12.1's tests 2–7 gate that phase rather than the
`W > 1` phase.

**What changes about reachability.** The server test's parenthetical — *"the
driver serializes passes, so this is raised where it is decided and mapped where
it is answered"* — ✅ **has been deleted**, and the hand-constructed error it
introduced is replaced by a real over-HTTP 409 inside
`every_session_refusal_reports_a_status_and_a_type_a_client_can_branch_on`
(`server/src/tests.rs`), because the statement is no longer true.


### 4.3 What must be constructed and dropped on a worker thread

This is the part that is easy to get wrong quietly, so it is normative.

**First, the precise form of the rule.** §3.4's Rule B is *exclusive use*, not
pinning: a resource may change owning thread while it is idle, and the shipped
`thread_affinity` module counts such migrations rather than refusing them
(`thread_affinity.rs:17-31`). So the obligations below say *"the thread that
owns it"*, not *"the thread that first created it"*. A worker taking over an
idle session's resources at a handoff (§5 create/fork/reset, §7 worker
replacement) is doing the supported thing. What is never permitted is two
threads inside the same resource at once, or a release performed by a thread
that is not the current owner.

**Must be constructed on the worker thread that will use it:**

- the CUDA context binding, and everything created under it;
- the cuDNN handle and its descriptor caches, because `with_handle` binds the
  owning context to the calling thread (`cudnn/mod.rs:1128-1132`);
- any `CudaReservation`, which its own comment calls thread-affine
  (`virtual_memory.rs:2749-2752`);
- the CUDA graph capture: begin, end, and abort must all be the same thread
  (`graph.rs:177,186-195,274-285`);
- the native decode session and the per-worker `PagedKvCache`, so their
  allocations are charged under the right context;
- every session-derived `IoBinding`, `Allocator` and `Value` (§3.4), which today
  are built wherever the workflow runtime happens to run
  (`pipeline/islands.rs:87-88`, `allocator.rs:280-300`, `value.rs:244-247`).

The ORT members of that list already record an `OwnerThread` at construction
(`binding.rs:104-117`, `allocator.rs:302-322`) and `Session` records one for the
run path (`session/mod.rs:507`). The recording is what makes the drop rules below
checkable instead of aspirational; §3.4 lists the native handles that still need
the same treatment.

**Must be dropped on the thread that owns them at that moment.** Teardown makes
driver calls that bind the context first
(`virtual_memory.rs:186-189`, `vmm_allocator.rs:686-689`); running those on a
foreign thread is the same violation as running the work there. Concretely: a
worker's `join` must happen only after that worker has dropped its own
category-3.2 state, and nothing may claw a handle back to the coordinator to
drop it. §7 makes this a shutdown ordering rule.

**And in the right order within a worker: values before their allocator.** This
is a separate obligation from Rule A, because Rule A's ownership edge stops one
level short. A `Value` is a bare `OrtValue` handle with no back-reference to the
allocator whose memory it holds, so nothing in the type system prevents an
allocator from being released first (`pipeline/mod.rs:189-200`). The required
order is written down in the ORT crate — *"the device `Value` (frees memory
through the allocator), then the `Allocator`, then the `Session`, and finally the
`Environment`"* (`value.rs:2443-2448`) — and it is why
`WorkflowRuntime::drop` still clears bindings, outputs and allocators by hand
(`pipeline/mod.rs:201-208 via pipeline/runtime_state.rs:351-365`). A worker's teardown path carries the same duty: it
drops its bound `Value`s, then its `Allocator`s, then releases its `Arc<Session>`.
This is normative for §7's shutdown sequence and for the worker-failure path.

Native CUDA teardown is the strictest case and is called out separately: the
CUDA context binding, the captured graph and its lifecycle, the reservation, the
device allocator and the KV pages beneath it are all released by the owner
thread, in that thread's own unwind or shutdown path. Their `Drop` impls use the
non-failing check form — `OwnerThread::check`, which reports and, on the release
side, *declines* a foreign release rather than performing one
(`thread_affinity.rs:254-272`, `:288-294`). **They do not panic**, per §3.4's
B.3: a wrong-thread release is reported and the resource is left held or
quarantined, never freed from a context some other worker may be mid-capture on.

The two rules compose. Rule A of §3.4 guarantees a session outlives everything
derived from it; Rule B guarantees that everything derived from it is released
by the thread entitled to release it. Neither alone is sufficient: an
`Arc<Session>` dropped on the wrong thread is still a wrong-thread teardown, and
a correct-thread teardown of a resource whose session has already been released
is still a use-after-free.

**May cross threads:** the `Arc`-shared immutable plan (§3.1), request and
response payloads, host-side token buffers, and `PinnedStaging`, which is a
plain host allocation by its own comment (`onnx-runtime-ep-cuda/src/runtime.rs:2123-2127`).

**Load is a worker-thread activity.** A model loaded on the coordinator and
handed to a worker would violate the construction rule for every handle above.
Each worker performs its own backend construction against the shared immutable
package; the coordinator's `load` returns only when every worker reports ready,
and a worker that fails to construct fails the load with its own error rather
than leaving a pool with a hole in it.

### 4.4 Bounding `W`

`W` is bounded by three independent limits, and the effective value is the
minimum, reported once at startup the way batching capability already is
(`driver.rs:300-306`):

1. **Backend capability.** A backend that cannot hold more than one decode
   session gives `W = 1`. The native path is already documented this way — *"the
   native single-session backend does not support {feature}; use independent
   serialized requests"* (`engine/runtime.rs:655-659`).
2. **Device memory.** Each worker owns a KV cache and its own capture and
   staging buffers. `W` is chosen against the same governor that already refuses
   over-admission, and a `W` the device cannot fund is a load-time refusal, not
   a runtime OOM (RULES.md Rule 9: *never silently OOM*).
3. **Operator configuration**, clamped to the two above.

`W = 1` must remain a fully supported configuration, and must be
behaviour-identical to today's single driver thread. That is what makes the
migration in §13 safe. It is also not hypothetical: it is what ships today, since
PR #2012 already reduced the pool to exactly one (`worker.rs:261-265`).

**A per-shard session bound is not a load bound, and the document should not
pretend otherwise.** §4.1 places a session by a stateless function of its id and
never migrates it, so the *count* of sessions per shard is roughly even by
construction while the *work* per shard is not. Sessions are wildly unequal: an
idle session costs its KV pages and nothing else, while an active one occupies a
decode row every step. A shard can therefore sit at its session bound while
mostly idle, and another shard under its bound can be saturated by a handful of
long generations. Three consequences:

- **The two limits are separate and are reported separately.** A shard refuses a
  new session when it hits its session bound, and refuses admission when it hits
  its memory ceiling or its decode-row width. Conflating them produces the worst
  diagnostic in the class — a capacity error that names the wrong resource.
- **Skew is measured, not assumed away.** §12.3 makes per-shard occupancy and
  per-shard queue depth reported metrics, because "the hash is uniform" is a
  statement about ids, not about load.
- **The response to skew is placement, not migration.** Once a session's pages
  are in one worker's `PageTable`, moving it is a copy, and a design that
  migrates under load pays that copy exactly when it is least affordable. If
  measured skew is bad enough to matter, the fix is a better placement function
  — for example one that consults live shard depth at *create* time, when the
  session has no pages yet and placement is still free — and that choice is
  §14's open question, deliberately left to measurement.

---

## 5. Lifecycle routing

| Operation | Routing | Notes |
|---|---|---|
| `create_session` | Coordinator picks a shard (§4.1), then dispatches to that worker, which mints `local` and installs session state | The capability refusal for a package with no session state (`PackageCapabilityError::NoSessionState`, `engine/runtime.rs:1526`) is answered on the worker and travels back typed |
| `generate` (sessionful) | `session.worker` | Acquires the exclusive lease; conflict returns `ExclusiveLeaseConflict` |
| `generate` (sessionless) | Any worker, by the same stateless distribution | No lease, no ownership; see §9 |
| `reset_session` | `session.worker` | Mutating: takes the lease. A reset racing a live turn is a conflict, not a truncation |
| `close_session` | `session.worker` | Mutating: takes the lease. Closing under a live turn is a conflict; the caller cancels first (§6) then closes |
| `fork_session` | Source `session.worker`, and the fork **stays on that worker** | Forced: a fork shares paged-KV pages copy-on-write with its source, and pages live in one worker's `PageTable` |
| `checkpoint_session` | `session.worker`, read-only | Takes no exclusive lease; a checkpoint concurrent with a turn is refused rather than allowed to capture a half-written conversation |
| `restore_session` | checkpoint's **existing** worker placement | Mutating: takes the lease. Restore is a rewind of a live session, not a create — see below |
| `rewind_session_by`, `rewind_session_to` | `session.worker` | Mutating: take the lease. Both are `&mut self` (`engine/runtime.rs:1590-1594`, `:1608-1612`) and truncate the session's KV |
| `session_token_count`, `session_prefill_carry` | `session.worker`, read-only | These are `&self` today (`engine/runtime.rs:1742,1788`) and remain read-only queries |

**What of this table is routed today.** The HTTP surface exposes exactly three
of these operations: `create_session` (`POST /v1/sessions`), sessionful and
sessionless `generate` (the completion routes), and `close_session`
(`DELETE /v1/sessions/{id}`). ✅ All three follow the table as of Phase 2 —
close takes the lease, a sessionful generate takes it, a sessionless one does
not — and `session_token_count`/`session_prefill_carry` remain read-only queries,
with the carry read now performed under the calling turn's own lease so it cannot
observe a conversation another turn is rewriting. `reset_session`,
`rewind_session_by`/`rewind_session_to`, `fork_session`, `checkpoint_session` and
`restore_session` have **no route and no driver command**: they are `&mut Engine`
methods reachable only from the worker thread that owns the engine, so there is
no routing-layer caller for the lease to guard. They are not exempt from the
table — they are unrouted, and the phase that routes them takes the lease in the
same place `close_session` does. §12.1's test 5 (reset racing a turn) is
therefore deferred with them; the close-racing-a-turn case, which is the same
shape, is covered instead.

One further consequence of the lease reaches a place the table does not name:
**LRU eviction is a close, so it obeys the close row.** The registry no longer
picks the least recently used binding and closes it; it walks candidates
oldest-first and takes the first lease it can, so a binding with a turn in flight
is skipped rather than destroyed under its caller. If every binding is mid-turn
there is no victim, and the *new* session is refused with a typed
`AtCapacity` — mapped to the same 429 `resource_limit_error` (with
`Retry-After`) every other transient capacity answer uses — rather than admitted
over the bound. Admitting it would have made `max_sessions` advisory
*permanently*: nothing walks the registry back down, because the next insert
evicts one and adds one, so a server sized for *n* conversations could be pushed
to *n + k* and stay there. The refusal is transient by construction and clears
the moment any turn in flight ends (`server/src/session.rs`, `evict_lru`).

**Every one of those decisions is keyed model-first.** The session registry is a
single map across every loaded model, but a `SessionPlacement` is unique only
*within* one engine: each engine numbers its sessions from its own counter and
each worker pool starts at worker 0, so two loaded models routinely name two
unrelated conversations with the identical placement. The lease key and the
registry entry are therefore `ModelSessionPlacement` — a `ModelKey` plus the
placement (`server/src/lease.rs`) — and there is exactly one `SessionLeases` map,
owned by the `SessionRegistry` itself rather than by any engine. An engine-owned
map would have answered "is this session busy?" only for the engine the caller
happened to be holding, which is the one thing a routing conflict cannot get
wrong. The engine learns its own `ModelKey` once, in `ModelHandle::new`, so the
handle's id and the driver's are the same string by construction; a close whose
lease names a different model is refused by the driver rather than performed.

**Close is one decision, not three.** `DELETE` does not read a binding, take its
lease, and then remove it: those are three decisions about a binding that can
change between them, and the middle one is where a rebind slips in and the close
destroys a conversation it never leased. `SessionRegistry::take_for_close` holds
the registry lock across the find, the acquire and the remove, and hands back the
guard — which names the owning model — so what is leased, what is unbound, and
what is closed are the same binding by construction, on the engine that owns it
rather than on the default model. LRU eviction removes its victim under the same
lock and the same rule.

**And the count of live conversations is reported by the map, not by the
callers.** `active_sessions` is a gauge, so it has to track what the `HashMap`
did rather than how many times a route asked for a session: an increment in
`insert` cannot see whether making room evicted somebody, so LRU churn at the
bound reports growth on a registry whose size never changed, and the gauge climbs
for as long as the process runs. The two mutation sites own the accounting
instead — a `HashMap::insert` that displaced nothing is an addition, a
`HashMap::remove` that removed something is a departure — so an eviction followed
by an insertion nets to zero, an insertion below the bound counts one, an
explicit close counts one departure exactly once, and a refusal, which mutates
nothing, counts nothing.

Three routing rules deserve their reasons stated.

**Reset and close take the lease.** They are mutations of the conversation, and
`reset_session`'s own comment describes exactly the state a concurrent turn
would corrupt: *"the id stays usable and everything the conversation accumulated
is gone"* (`engine/runtime.rs:1662-1665`). This is the one place in this design
where a genuine lost update is on the table, and it is worth being exact about
why: a reset is a *separate operation* from the turn, so unless it takes the same
lease that spans the turn's read → write commit, it can land between them and be
silently undone by the write-back. That is the narrow condition §11.1 identifies
as the real lost-update case, and taking the lease is what excludes it.

**Fork cannot cross shards, and today cannot happen at all.**
`session_fork_capability` returns `None` and `fork_session` bails naming the
missing capability (`engine/runtime.rs:1628-1659`). The capability-token
signature is the right shape and is kept. When fork is enabled, the shard
constraint above is not a simplification — copy-on-write over a `PageTable`
requires source and fork in one page table, so a cross-shard fork is a deep
copy, which is the cost fork exists to avoid.

**Checkpoint is read-only but not lease-free.** It observes the conversation, so
it must not observe it mid-write. It does not *hold* the exclusive lease for its
duration in a way that would make checkpointing block generation; it fails fast
if the lease is held, matching the retryable contract
(`capability.rs:57-61`).

**Restore routes by the checkpoint's own session id, and checkpoints are not
portable across worker counts.** `SessionCheckpoint` is deliberately small — it
is `{ session_id: SessionId, position: SessionPosition }` (`config.rs:433-438`) —
and `restore_session` is a thin wrapper that forwards to the rewind machinery for
*that* session:

```rust
pub fn restore_session(&mut self, checkpoint: SessionCheckpoint) -> anyhow::Result<()> {
    self.rewind_session_to(checkpoint.session_id, checkpoint.position)
}
```
— `engine/runtime.rs:1580-1582`

So restore is not "create a session from a snapshot". It is a rewind of a session
that is still alive, whose KV pages and prefix-cache references still exist on
one worker, and whose own doc comment says as much: restoring *"uses the same
rewind machinery as speculative decoding and keeps prefix-cache page ownership
intact"* (`engine/runtime.rs:1564-1569`). Routing it anywhere but
`checkpoint.session_id`'s existing shard would target a page table that does not
contain the session at all.

Three consequences follow, and all three are normative:

1. **Restore takes the exclusive lease**, because it mutates the conversation's
   logical length exactly as `rewind_session_to` does.
2. **A checkpoint is only meaningful while its session is live.** The doc comment
   already says *"Checkpoints are invalid after the session is closed or reset"*
   (`engine/runtime.rs:1568-1569`); the shard model adds that it is also invalid
   after its owning worker is lost (§7), and that case returns the same typed
   worker-failure error rather than a confusing "session not found".
3. **Checkpoints are not portable across worker counts, and must not be made to
   look portable.** A `SessionCheckpoint` carries a `SessionId`, and once
   `SessionId` encodes a `ShardIndex` (§4.1) a checkpoint minted under `W = 4`
   names a shard that may not exist under `W = 2`. Because the type is public,
   this must be a typed refusal — a checkpoint whose shard is out of range for
   the current pool is rejected naming the checkpoint's shard and the current
   `W`, never silently remapped onto a different worker. Remapping would attach a
   conversation's logical position to a page table that never held it. If
   cross-`W` portability is wanted later it needs a real serialized snapshot of
   the KV state, which is a different feature with a different cost, and §14
   records it as an open question rather than implying this design provides it.

### 5.1 Server-side identity is unaffected

The server maps client `X-Session-Id` strings to conversations through
`SessionRegistry`, an `Arc<Mutex<SessionRegistryInner>>` with its own LRU
(`crates/onnx-genai-server/src/session.rs:11-20`,
`routes/completions.rs:1310-1340`). That map is already thread-safe and already
handles the create race explicitly with `SessionClaim::{Existing, Claimed}`, a
loser closing the session it opened rather than leaking it
(`session.rs:22-29`, `routes/completions.rs:1325-1338`).

PR #2012 already did half of this section's work. `SessionEntry` no longer stores
a bare engine session id; it stores a placement, with the reason written into the
type: *"a later turn has to be routed back to that worker, and an engine session
id alone cannot say which one it is"*. That is precisely the routing key §4.1
needs, already threaded through the create path.

So the registry's *identity* model is untouched by this design. It stores and
returns an opaque binding; whether the shard lives in a side field or is encoded
in the id is invisible to it, and clients continue to see an opaque
`X-Session-Id` they never parse.

⚠️ **Two things did change, and neither is client-visible identity.**

First, eviction closes a conversation, and a close takes the lease (§5), so
`insert` and `claim` hand the evicted binding back *holding its guard*
(`SessionClaim::Claimed { evicted: Option<SessionLeaseGuard> }`). The caller
closes under that guard, which is what stops a turn from starting on a session
between the moment it is unbound and the moment the engine frees it, and what
stops the client id being rebound to a new conversation while the old one is
still being torn down. When there is no evictable binding, the insert is refused
(§5) rather than admitted over `max_sessions`.

Second, ✅ **the stored binding is model-qualified.** A `SessionPlacement` alone
is unique only inside one engine, and the registry spans every loaded model, so
`SessionEntry` stores a `ModelSessionPlacement` and the registry owns the one
`SessionLeases` map keyed the same way. This is what makes `DELETE` close on the
model that opened the session instead of on the default one, what makes eviction
close its victim on the victim's engine, and what stops model A's first session
and model B's first session — which have the identical placement — from being
treated as one conversation.

---

## 6. Cancellation, errors, and lease release

**Lease release is a `Drop` obligation, never a cleanup path.** The existing
guard already establishes this (`pipeline/workflow.rs:630-634`), and ✅ Phase 2's
routing guard follows it: `SessionLeaseGuard` has no release method to call, only
a `Drop`, and it is `#[must_use]` so a caller cannot acquire one and forget to
hold it. Every exit from
a turn — normal completion, typed refusal, `anyhow` error, client disconnect,
explicit cancellation, or panic — unwinds the per-execution state of §3.3 and
therefore releases the lease. This is the reason the lease lives in category 3.3
and the conversation lives in 3.2: the thing that must vanish on failure and the
thing that must survive it are different objects.

**Cancellation is cooperative and worker-local.** A cancellation request routes
to `session.worker` and sets a flag the decode loop observes at a token boundary.
It does not interrupt an in-flight forward pass — there is no safe way to abort a
launched CUDA graph mid-capture or mid-replay, and pretending otherwise is how a
context ends up in an undefined state.

Today cancellation is implicit: the driver detects an abandoned route when
delivering output fails and drops it (`driver.rs:1394-1403`, `:1418-1421`), and
submission failure surfaces as `GenerateSubmitError::DriverStopped`
(`driver.rs:252-256`, `:502-506`).
Under a worker pool that is not enough, because a cancelled turn must release its
lease promptly or the next turn on that session sees a spurious 409. So
cancellation becomes explicit: a typed command, observed at a token boundary,
whose acknowledgement is the point at which the caller may assume the lease is
free.

✅ **Phase 2 made the implicit form safe rather than replacing it.** The guard is
owned by `DriverCommand::Generate` and, for a batched turn, by the `DriverRoute`
row, so an abandoned route releases the lease when the row is dropped and a
disconnected client cannot lock its own conversation out
(`a_cancelled_client_does_not_leak_its_session_lease`). What is still implicit is
the *timing*: the release happens when the turn actually ends, not when the client
goes away, so a client that disconnects and immediately retries can still see one
409 for a turn it no longer wants. The explicit typed cancel command above is what
closes that window, and it is not part of Phase 2.

**Panic containment.** A panic on a worker unwinds that turn's category-3.3
state and releases the lease, but it leaves the worker's category-3.2 backend
state of unknown validity — a panic inside a CUDA graph capture leaves a capture
open on that thread. A worker that panics is therefore not reused: it is
quarantined and handled as a worker failure (§7), and RULES.md Rule 1's
obligation applies to the error the affected callers receive, which must name the
worker, the session, and the fact that the session's state was lost.

**Error typing follows the existing conventions.** New failures are
`#[derive(Debug, thiserror::Error)]` enums with a named `*Error` type, per
[`ERROR_AND_LOGGING_CONVENTIONS.md`](ERROR_AND_LOGGING_CONVENTIONS.md):23-27, and
`ExclusiveLeaseConflict` in particular is *not* re-invented — it is the existing
variant at `capability.rs:50-53`, reached through the existing
`package_capability_error` chain walk (`capability.rs:64-72`) and mapped by the
existing `package_capability_failure` (`routes/mod.rs:517-527`). Matching on the
variant rather than on wording is already the rule there and stays the rule.

---

## 7. Shutdown and worker failure

**Shutdown ordering:**

1. The coordinator stops accepting new work and closes each worker's command
   channel. Today's shutdown is exactly this — `blocking_recv()` returns `None`
   and the loop exits (`driver.rs:800-803`) — and it generalizes to `W` channels.
2. Each worker drains its in-flight turns to a token boundary, releasing leases
   as the per-turn state drops.
3. Each worker drops its own category-3.2 state **on its own thread**: captured
   graphs, cuDNN handles, KV cache, reservations, ORT/native sessions, in that
   order, then unbinds its CUDA context.
4. Each worker signals completion and exits; the coordinator joins.
5. The coordinator drops the `Arc`-shared immutable plan, which by then has no
   other holder.

Step 3 before step 4 is the load-bearing part, and it is why `join`-then-drop is
forbidden: a coordinator that joined first and then dropped a handle it had
retained would be making context-bound driver calls from the wrong thread. The
current design already gets this right for the single-thread case by construction
— the engine is dropped on the driver thread when `run_engine_driver` returns,
because it owns the unboxed `Engine` local for the whole call
(`driver.rs:760-767`) — and the worker model must preserve it deliberately rather
than by accident.

**Worker failure.** A worker that panics or whose backend construction fails is
removed from the routable set. Its sessions are **lost, not migrated** — their
state was category 3.2 and thread-affine, and there is nothing to move. Every
subsequent call naming a session on a dead shard receives a typed error saying
the worker failed, which session was lost, and that the session must be
recreated. Silently recreating it would report a lost conversation as a success,
which is the same class of dishonesty as answering a busy session with a slow
200 instead of a 409.

Any `SessionCheckpoint` naming a session on the dead shard becomes invalid with
it, and is refused by the same typed worker-failure error rather than by a
"session not found" that would suggest the id was never real (§5).

Whether the pool degrades to `W-1` or the engine fails as a whole is an operator
policy with a stated default: **degrade**, because a serving process that keeps
`W-1` shards answering is strictly better than one that stops, and the failure is
already visible in the typed errors and in the startup-style capability log.

**That promise is only honest because teardown never panics.** A worker panic
unwinds, and the per-worker resources it drops on the way out follow §3.4's
Rule B.3 — telemetry, deferred release, quarantine — none of which can panic, so
none of which can turn the unwind into an abort. The single exception B.3 permits
is a named fatal invariant covering an outcome that cannot be described at all;
that path is deliberately outside this promise, and it is not reachable from an
ordinary worker panic or a failed backend construction. If an implementation ever
widens the abort case, this paragraph must be narrowed with it.

**Shutdown is explicit, never a drop.** #2012 settled this and the pool model
keeps it: `WorkerHandle::shutdown` closes the channel and joins
(`worker.rs:198-208`), while `Drop for WorkerHandle` deliberately closes
*without* joining, on the grounds that a blocking join in `Drop` is a worse
failure than an unjoined thread (`worker.rs:217-236`). So the ordering above is
supplied by an explicit shutdown call on every worker, and a `WorkerPool` that is
merely dropped promises nothing about step 3 — which is exactly why step 3 is
written as a shutdown obligation rather than a teardown side effect.

---

## 8. Accounting and reservation ownership

The governor's doc comment is a constraint, not a description: *"Exactly one
exists per runtime — a second would double-count every reservation"*
(`engine/model.rs:49-54`). A worker pool must not create `W` governors.

**Therefore accounting stays global and shared; only the things it accounts for
are per-worker.**

- The device memory authority and its `LeaseLedger` stay one instance, shared
  across workers behind `Arc`
  (`crates/onnx-genai-engine/src/memory_authority.rs:37-50`).
- `ByteBudget` is already built for this: it is `Arc<Mutex<BudgetState>>` and
  documents that clones "account against a single running total, so no single
  session can blow the global ceiling" (`byte_budget.rs:122-133`). Each worker
  holds a clone. This is the one place a shared lock is correct, because the
  invariant being protected genuinely is global.
- `HostGovernor`'s `Mutex<Ledger>` and `HostGovernorAccounting`'s
  `Mutex<Outstanding>` are likewise already shared and already correct
  (`pressure.rs:416-431`, `host_lease.rs:50-60`).
- `AdmissionCeiling::ceiling_bytes` is documented as "called on the admission
  path, so it must not block" and "called without any of this budget's locks
  held" (`byte_budget.rs:113-118`). A worker pool multiplies the call rate; the
  non-blocking contract is what keeps that from becoming a convoy, and it is
  already stated, so it becomes a tested property (§12.1) rather than a comment.

**A reservation is owned by the turn that made it, on the worker that made it,
and is released by drop.** That is category 3.3. Session-lifetime reservations —
a session's paged-KV pages — are category 3.2, owned by the shard, and released
by `close_session`, by shard-local eviction, or by worker failure. The pairing of
"who charged it" with "who releases it" is what keeps a shard's death from
leaking the global budget: a failing worker's reservations are released by
dropping its ledger holder registration, on its own thread, in step 3 of §7.

The one honest caveat: `HostGovernorAccounting` releases whole allocations as a
byte credit accumulates, so a partial `shrink` returns nothing immediately
(`host_lease.rs:26-37`). Under `W` workers that credit is per-adapter, so the
worst-case unreturned residue scales with `W`. It is bounded by construction —
"the adapter can never release more than it was given" — and §12.3 makes the
residue a measured quantity rather than an assumed-small one.

---

## 9. Continuous batching and session sharding

These two mechanisms are orthogonal and must not be conflated. Continuous
batching multiplexes *requests* onto the decode rows of **one** backend session.
Session sharding distributes *sessions* across **several** backend sessions. One
is intra-worker; the other is inter-worker.

The resulting structure: each worker runs its own continuous batch loop over its
own rows, of the shape `ContinuousBatchManager` already implements — a FIFO queue
admitted into a fixed number of physical decode rows
(`crates/onnx-genai-engine/src/batched.rs:167-173`), stepped by advancing every
row with pending logits (`batched.rs:381`). The pool multiplies the number of
such loops; it does not change any one of them.

**Sessionful requests can join a batch, and this is the change that makes the
pool worth building.** Today they cannot: *"The current `ContinuousBatchManager`
API accepts `GenerateRequest` only. `X-Session-Id` requests keep using the
driver's per-request engine path so persistent engine KV/session semantics are
preserved until the manager grows a `SessionId`-aware submit API"*
(`driver.rs:814-817`). Under the worker model
a session's turn is pinned to a known worker, so its KV sequence is in that
worker's page table and its row can be admitted alongside other sessions' rows in
the same forward pass. A batch may mix rows from different sessions; it may
**never** hold two rows from the same session, which is Decision 3 restated at
the row level.

**That row-level exclusion is enforced before enqueue, not inside the batch
loop.** This is where §1.2's finding bites. It would be natural to assume the
existing `workflow_session_leases` already prevents a second row for a session,
but it does not: that guard is acquired inside the interpreted workflow path
(`workflow.rs:1356-1363`) and the decode core — the path continuous batching runs
on — never reaches it. Adding a check inside the batch admission loop would also
be too late by then, because the request would already have been admitted and
queued, which §4.2.1 rejects. So the guarantee comes from the routing lease taken
in the façade before the command is sent: a second turn for a live session never
becomes a row at all, and `ContinuousBatchManager` needs no same-session check to
stay correct. The batch loop is a beneficiary of the invariant, not its
enforcer.

Two consequences that follow, and are not optional:

- **Batch width is per-worker, and total width is `W × rows`.** An operator who
  configures `W` and `max_batch` independently can request more concurrent rows
  than the device can fund. The effective values are resolved together and
  reported once, in the same way `BatchingReport` already clamps a requested
  width to what the decode path can honour (`driver.rs:292-298`).
- **Sessionless requests go to any worker.** They own no state, so stateless
  distribution (§4.1) applies directly, and a sessionless request should prefer
  the worker with the fewest active/in-flight turns for the same reason a
  sessionful one cannot: nothing pins it.

The interaction with prefix sharing is the real cost, and it is stated rather
than elided. `PrefixCache` is per-worker under §3.2, so two sessions sharing a
long system prompt but landing on different shards each materialize it. At `W`
shards, a shared prefix costs up to `W` copies. This is a genuine regression
against the single-engine arrangement and is why §12.3 makes prefix-cache hit
rate an acceptance metric and not an afterthought.

---

## 10. Removing the `unsafe impl Send`s

Both hand-written impls existed to paper over the gap between "this type is not
provably `Send`" and "we know it never leaves its thread". The server wrapper is
now gone because its worker constructs the engine in place. The engine crate's
own impl remains until Phase 3 decomposes the public engine handle.

**#2019 made this urgent rather than merely tidy.** `Engine`'s safety comment
still reads *"Neither runtime's sessions, values, bindings, allocators, or CPU
tensors have thread affinity"* (`engine/model.rs:221-223`). Since #2019 that
sentence is false for three of the five: `IoBinding`, `Allocator` and `Value` are
`!Send + !Sync` and carry `OwnerThread` records (§3.4). The `thread_affinity`
module names the situation explicitly — *"What the compiler cannot see is a
resource smuggled inside a container that carries its own `unsafe impl Send`
(`onnx_genai_engine::Engine` does), and that is the case this module names"*
(`thread_affinity.rs:13-15`). The runtime guard now catches a violation and
reports it; §10's job is to make the violation unrepresentable, and until it is
done the `unsafe impl` is the thing standing between the compiler and a bug it
could otherwise reject outright.

✅ **`unsafe impl Send for EngineOwner` is deleted.** Its entire safety argument
was *"The engine is moved exactly once into the dedicated driver thread"*.
`EngineDriver::start` now passes a `Send` load closure to `WorkerHandle::spawn`;
the worker constructs the engine, reports a typed ready result, runs it, and
drops it before exiting. There is no cross-thread engine move to justify, and
`onnx-genai-server` now uses `#![forbid(unsafe_code)]`.

**`unsafe impl Send for Engine` (`engine/model.rs:221-229`) remains until Phase
3, where `Engine` is decomposed.** What remains after that split is:

- an `Arc`-shared immutable plan (§3.1), naturally `Send + Sync` because it is
  frozen;
- a per-worker backend bundle that is `!Send` and stays that way — it is
  constructed by the closure the worker thread runs, so it never needs to cross
  a thread boundary and never needs an impl;
- per-execution state (§3.3), created and destroyed within one turn.

The public handle callers hold becomes a `Send + Sync` façade over the routing
table and the command channels: it contains no backend handle at all, so its
`Send`/`Sync` are derived, not asserted.

**The rule that replaces them:** *no `unsafe impl Send` or `unsafe impl Sync` may
be added for a type that transitively owns a backend handle.* If a type needs
one, the handle is in the wrong category, and the fix is to move construction to
the worker rather than to write the impl. The existing `unsafe impl`s on
individual backend handles in `onnx-runtime-ep-cuda` and `onnx-runtime-cuda-memory`
are out of scope and unchanged: they are narrow claims about specific handles,
each with its own audited justification, and this document neither relies on nor
weakens them.

**A second, structural rule lands with it, and it is deliberately scoped:** *no
type that owns a **session-derived** backend handle may depend on field
declaration order for teardown correctness, and no such dependency may be
discharged by a `Drop` impl one level up.* Ownership is expressed with `Arc`
(§3.4, Rule A) so that the refcount, not the source-file layout, decides release
order; thread affinity is expressed structurally where possible and reported as a
typed error otherwise (§3.4, Rule B). §1.4 is the worked example of what the
absence of both rules cost at `W = 1`, and the cost is not linear in `W` — it is
the difference between one thread reliably reproducing a bug and `W` threads
producing it intermittently.

**For ORT session-derived handles that rule is now satisfied, not proposed.**
`IoBinding` and `Allocator` co-own their session (§3.4, Rule A), so
`ExecutionIsland` and `PipelineModels` no longer depend on field order for the
session edge. What remains inside the scope of the rule is the `Value` →
`Allocator` edge, which has no ownership link and is still discharged by a `Drop`
one level up (`pipeline/mod.rs:189-208`) — the rule's own exception, kept
visible rather than quietly satisfied.

The scoping is not a hedge; it is §3.5. The `Session → Environment` edge has the
same defect and is **not** fixed here, because `Session` holds no environment
handle (`session/mod.rs:479-508`) and the environment is a process-global
`OnceLock` (`env.rs:45-48`). Claiming the rule for that edge without doing the
work would make the rule decorative. Until §3.5's structural option is taken, the
environment's ordering remains discipline-enforced, and — the part that matters
for review — **no compensating teardown is deleted before the structural
lifetime that replaces it exists.**

The three rules are one rule seen from three sides. A `Send` claim asserts a
handle may cross threads; `Arc` ownership asserts a handle outlives its
dependents; owner-thread recording asserts a handle is touched by one owner at a
time.
Each replaces a comment that a human has to keep true with a fact the compiler or
the runtime keeps true.

---

## 11. Alternatives considered

### 11.1 A global `Mutex<Engine>` or `RwLock<Engine>` — rejected

This is the obvious change: keep the engine as it is, wrap it, make the handle
`Sync`, ship. It is rejected on four independent grounds, any one of which is
sufficient.

1. **It does not deliver Decision 2.** One lock means one session's forward pass
   blocks every other session's. The API would be thread-*safe* and entirely
   non-concurrent — the Python binding's current behaviour, generalized, which is
   honest precisely because it refuses rather than pretends
   (`python/src/lib.rs:173-184`).
2. **It violates Decision 3's refusal policy, though not by losing an update.**
   Under a whole-turn mutex a second turn on a busy session *blocks and then
   runs*, and because the lock spans the entire turn — read of the conversation
   through write-back — the two turns are sequentially consistent. The second
   turn reads the conversation the first turn *finished* writing, and appends to
   it. That is a correct serialization, and it is worth stating plainly rather
   than overclaiming: **a whole-turn mutex or a per-session queue does not lose
   an update.**

   What it does instead is substitute an unbounded, unattributable wait for a
   fast, typed refusal. The caller asked whether it could take the session; it
   was told neither yes nor no, and instead waits for a decode that may run for
   thousands of tokens. The latency is invisible in the response, the queue is
   invisible in the metrics, and a client that would have retried elsewhere or
   surfaced a conflict to its user cannot. The lease contract already chose the
   other answer — *"a second turn that starts while one is in flight is refused
   by name"* ([`../genai/INFERENCE_METADATA_DECISIONS.md`](../genai/INFERENCE_METADATA_DECISIONS.md):1103-1105) —
   and silent queueing is a policy violation, not a correctness one.

   A genuine lost update requires something narrower and worse: a lock or lease
   that does **not** span read → write commit. If a turn reads the conversation,
   releases, computes, and then writes back, two turns interleave and the loser's
   prompt and generation land nowhere. That is the failure
   `SessionLeaseGuard`'s comment names — *"Two turns that both read the
   conversation before either writes it would leave the loser's prompt and
   generation nowhere, and nothing would report that they were lost"*
   (`pipeline/workflow.rs:600-604`) — and it is why §4.2's routing lease is held
   for the whole turn (§4.2.1) rather than only across the read. The lost-update
   risk is real, but it is a risk of *this design being implemented wrong*, not
   an argument against the global mutex.
3. **It does not make thread-affine handles correct.** A mutex supplies mutual
   exclusion. cuDNN's handle, a `CudaReservation`, and CUDA graph capture require
   *thread identity* (§1.3). Serializing a context-bound handle across two
   threads satisfies the exclusion half of its contract and violates the affinity
   half, silently.
4. **The two wrappers fail differently, and the difference matters.**
   `Mutex<Engine>` **compiles today** — `Engine` is `Send`, so
   `Arc<Mutex<Engine>>` is `Send + Sync` — and that is precisely the trap: it
   type-checks, it is safe, and it silently serializes every session against
   every other. A caller can do this right now without being warned (§13's
   compatibility note).

   `RwLock<Engine>` does **not** work, and not merely because it is a bad idea.
   `RwLock<T>: Sync` requires `T: Send + Sync`, and `Engine` is structurally
   `!Sync`: the interpreter holds `RefCell<HashSet<String>>`
   (`pipeline/runtime_state.rs:279`). So `Arc<RwLock<Engine>>` is not `Sync` and cannot be
   shared across threads at all. Even if that were fixed, a read lock would buy
   nothing: almost every session operation takes `&mut self` —
   `create_session` (`engine/runtime.rs:1501`), `reset_session` (`:1662`),
   `close_session` (`:1716`), `rewind_session_by` (`:1590`), `rewind_session_to`
   (`:1608`), `restore_session` (`:1580`), `fork_session` (`:1638`), and
   `generate_in_session` (`:1138-1139`) — so they would all need the write lock
   regardless.

**The condition for revisiting.** This rejection is contingent, and the condition
is precise: if every handle a session transitively owns proves `Send + Sync`
*with an audited justification of the kind ORT's `Session` already carries*
(`onnx-genai-ort/src/session/mod.rs:1389-1406`) — meaning no bound context, no
owning stream, no thread-local capture state — then a lock-based design becomes
sound, and grounds 3 and 4 fall away. Grounds 1 and 2 would still stand. Today
that condition is not met, and the evidence is in the table in §1.3.

### 11.2 A per-session `Mutex` — rejected

Finer-grained, and fixes ground 1. It does not fix grounds 2, 3, or 4: it still
blocks-then-runs a second turn where the contract says refuse it — the same
policy violation as the global lock, at session granularity — and it still
supplies exclusion where affinity is required. It also cannot express what a session
shares with its neighbours — one KV page table, one prefix cache, one capture
buffer — so the per-session lock would have to be accompanied by locks on all of
those, which is the global lock again with more places to deadlock.

### 11.3 Work stealing across workers — rejected

A stolen session would have to execute against a page table, a KV cache, and a
CUDA context it does not own. Migration is not slow here, it is incorrect. This
is the same fact that makes §4.1's placement one-shot and §7's failed sessions
unrecoverable, and it is worth naming as a rejected option so that a future
reader does not mistake the absence of stealing for an unfinished optimization.

### 11.4 Wrapping raw handles in locks to manufacture `Sync` — rejected

Stated separately because it is the tempting local fix. Adding
`Mutex<RawCudaHandle>` and `unsafe impl Sync` makes the compiler stop objecting
without changing anything about the thread that ends up calling the driver. The
existing safety comments are careful to say both halves — *"every access is
serialized by `handle`, **and** `with_handle` binds the owning CUDA context to
the calling thread first"* (`cudnn/mod.rs:1128-1132`) — and a design that supplies
only the first half is quoting half a contract. Decision 4 forbids it.

---

## 12. Testing and acceptance

The repo has no `loom` and no `shuttle` (verified across workspace manifests),
and this design does not introduce them. It does not need them: the properties
below are about real threads, real CUDA contexts, and real ORT sessions, and a
model checker over a simulated scheduler would not exercise the thread-affinity
constraints that are the whole point. The house pattern is real threads with
barriers — for example `crates/mlas-sys/tests/concurrent_dispatch.rs:57-105`
(`std::thread::scope`, asserts no dispatch returns with unwritten slots),
`crates/onnx-genai-scheduler/tests/pressure_conformance.rs:1011-1023` (spawns a
reclaim thread, asserts the waiter wakes), and
`crates/onnx-runtime-comm/tests/collective_ordering_conformance.rs:418-445`
(per-rank threads, asserts the concurrent trace replays). These tests follow it.

### 12.1 Real concurrency tests

**Not acceptable as evidence:** a single-threaded test that interleaves two
sessions by calling them alternately. That is what the server test used to do
when it constructed `ExclusiveLeaseConflict` by hand — it proved the mapping, not
the concurrency, and ✅ it is gone. Every test below spawns real OS threads and
synchronizes on a `Barrier` so the operations genuinely overlap. The shipped
Phase 2 tests come in two shapes, both of which meet that bar: `lease.rs`'s unit
tests race real `std::thread`s on a `std::sync::Barrier`, and the HTTP tests race
tasks on a multi-threaded Tokio runtime (`flavor = "multi_thread"`) on a
`tokio::sync::Barrier` — real threads either way, never one thread taking turns.

1. **Different sessions run concurrently.** `N` threads, `N` sessions, one
   barrier. Assert all complete, each output matches the same prompt run under
   the fixed-composition replay of §12.2 rather than under "run serially" — free
   composition legitimately changes batch shape, so a naive serial comparison
   would make this test flaky for the wrong reason — and that wall time is
   materially below `N ×` the serial time. The last clause is what distinguishes
   concurrency from a mutex, and per [`../README.md`](../README.md)'s standing
   rule it is reported with its conditions or not at all.
2. **Same session, overlapping turns, exclusive lease → 409.** ✅ **Shipped
   (Phase 2).** Two threads, one session, barrier-synchronized submit. Assert
   exactly one succeeds and the other returns
   `PackageCapabilityError::ExclusiveLeaseConflict` naming that session, and —
   this is the part the old test could not do — assert it over HTTP as a real 409
   with `kind == "conflict_error"`. **This test runs at `W = 1`** (§13, Phase 2):
   pre-enqueue acquisition (§4.2.1) makes the refusal reachable with one worker,
   so it does not wait for `W > 1`. Landed as
   `only_one_of_many_racing_threads_takes_the_lease` (`lease.rs`, 8 OS threads on
   a `std::sync::Barrier`),
   `concurrent_turns_on_one_session_do_not_lose_a_conversation` and
   `a_second_turn_on_a_busy_session_is_refused_rather_than_queued`
   (`server/src/tests.rs`), with the HTTP body's `error.type` asserted in
   `every_session_refusal_reports_a_status_and_a_type_a_client_can_branch_on`.
   `turns_on_distinct_sessions_are_all_admitted` and
   `stateless_requests_take_no_lease_and_are_never_refused` pin the other half:
   the refusal is keyed by session and nothing else changes behaviour.
3. **The refusal is a refusal, not a queue.** ✅ **Shipped, in the form the
   fixture supports.** The counterpart to test 2 and the reason it exists.
   `a_second_turn_on_a_busy_session_is_refused_rather_than_queued` holds the very
   guard a live turn carries, races an HTTP turn against it, and asserts the 409
   comes back *while the guard is still held* — not after it is released — and
   that the refused turn charged no admission permit
   (`generation_capacity.available_permits()` is untouched). An implementation
   that acquired the lease inside the worker loop would block on the held guard
   and fail this test. ⚠️ **What is not asserted:** a latency bound against a
   real winner's completion time. The CPU fixture generates in milliseconds, so a
   wall-clock bound would measure the fixture rather than the design; the held
   guard is the deterministic form of the same claim.
4. **The winner's conversation is intact.** ✅ **Shipped (Phase 2).**
   `concurrent_turns_on_one_session_do_not_lose_a_conversation` asserts the
   conversation equals exactly what the *admitted* turns produce in series
   (`first_turn_prefill × admitted + generated`), which is short if a lease is
   released before the write commit and long if a second turn was silently
   queued; `a_second_turn_on_a_busy_session_is_refused_rather_than_queued`
   asserts the next turn continues from where the winner left off.
5. **Reset racing a turn is refused, and does not vanish.** ⚠️ **Deferred, with
   a substitute.** `reset_session` has no route and no driver command (§5), so
   there is no routing-layer caller to test. The same-shaped mutation that *is*
   routed is close, and
   `deleting_a_session_during_a_turn_is_refused_and_the_session_survives` asserts
   a `DELETE` racing a live turn is refused 409, the binding survives, and the
   delete succeeds once the turn ends. Test 5 proper lands with whichever phase
   routes reset.
6. **Cancellation releases the lease.** ✅ **Shipped for the lease half.**
   `a_cancelled_client_does_not_leak_its_session_lease` aborts the request task
   mid-turn and asserts a later turn on that session is admitted rather than
   409'd forever, with nothing left leased and the binding intact.
   `a_failed_turn_releases_its_lease` does the same for the error path, which the
   original test list did not name separately. ⚠️ **Not asserted here:** that the
   cancelled turn's reservations returned to the budget — that is an accounting
   claim (§8), not a lease claim, and it is left to the phase that owns it.
7. **Send failure releases the lease.** ✅ **Shipped (Phase 2).**
   `a_turn_that_never_reaches_a_worker_releases_its_lease` (`driver.rs`) covers
   both pre-worker exits in one test: a submission refused by the admission gate
   (`GenerateSubmitError::Overloaded`) and one handed to a channel whose worker is
   gone (`GenerateSubmitError::DriverStopped`), asserting the session is
   immediately leasable after each. This is the exit path §4.2.1 flags as easiest
   to miss, and the only one where the guard is released on the *submitting* task
   rather than the worker.
8. **Panic containment.** Inject a panic on a worker; assert the lease is
   released, the shard is quarantined, other shards keep serving, and calls
   naming a lost session get the typed worker-failure error rather than a
   silently fresh session.
9. **Shutdown under load.** Shut down with turns in flight on every worker;
   assert every worker drops its own backend state on its own thread (asserted by
   recording the dropping thread id), that `shutdown()` joins before the handle
   is released — the contract `WorkerHandle::shutdown` already provides
   (`worker.rs:198-208`) — and that the process exits without a CUDA teardown
   error.
10. **Routing is total and stable.** Property test: for any sequence of
    create/close, every operation on a live session reaches the worker encoded
    in its placement, and no session is ever observed on two workers. #2012's
    `a_pool_of_one_routes_by_worker_id` is the `W = 1` seed of this: it already
    asserts that a placement naming a worker outside the pool is refused with
    `WorkerUnavailable::Unknown` rather than falling back to the primary
    (`worker.rs:414-437`).
11. **Admission does not convoy.** `AdmissionCeiling::ceiling_bytes` is
    documented as non-blocking (`byte_budget.rs:113-118`); assert under `W`-way
    concurrent admission that no caller blocks on it, since the worker pool is
    what turns that comment into a load-bearing property.
12. **Session bounds refuse, and skew is visible.** Fill a shard past its bound
    and assert `create_session` returns the typed refusal naming the shard and
    the bound, rather than over-subscribing. Separately assert the two limits are
    not conflated: a shard at its *session* bound refuses new sessions even when
    idle, and a shard under its session bound but at its *memory* ceiling refuses
    with the memory error instead — see §4.4 on why session count and active load
    are different quantities.
13. **Session-derived resources keep the session alive.** ✅ **Shipped by #2019
    for ORT** — `a_shared_binding_keeps_its_session_alive_past_the_owners_last_handle`
    and `a_shared_allocator_keeps_its_session_alive_and_releases_before_it`
    (`crates/onnx-genai-ort/tests/session_thread_contract.rs:203-245`) drop the
    caller's last `Session` handle first and assert the derived resource still
    tears down cleanly. What this design adds is the same test for the holders
    §3.4 has not converted (`Engine::session`, `Eagle3Model`, `DraftModel`), and
    the `Value` → `Allocator` order obligation of §4.3, which has no ownership
    edge and must therefore be asserted by teardown order rather than by
    refcount.
14. **Wrong-thread use returns a typed error.** ✅ **Shipped by #2019 for ORT** —
    `a_second_thread_reaching_a_held_resource_is_refused_by_name`
    (`session_thread_contract.rs:130-163`), plus the structural half as a
    `const { assert!(..) }` contract that *fails to build* if anyone adds an
    `unsafe impl Send` (`session_thread_contract.rs:60-125`), which is stronger
    than a `trybuild` case because it breaks the crate rather than a test. Two
    further shipped tests pin the semantics this document depends on:
    `an_idle_resource_may_move_to_another_thread_and_records_it`
    (`:164-187`) — the exclusive-use-not-pinning rule of §3.4 — and
    `a_concurrently_runnable_session_can_actually_be_run_from_two_threads`
    (`:285-320`), which runs one `Session` from two real threads rather than
    trusting the `Sync` impl. The remaining work is the same coverage for the
    native/CUDA handles §3.4 lists as outstanding, matching the precedent the
    CUDA graph code already sets (`graph.rs:188-195`).
15. **Wrong-thread teardown degrades, and does not abort.** Drop a per-worker
    resource on a foreign thread and assert three things: the process does not
    abort — `OwnerThread::release` already reports and declines rather than
    panicking (`thread_affinity.rs:304-322`), so this test pins existing
    behaviour for ORT and new behaviour for the native handles — telemetry names
    the resource and both threads, and the ownership is
    accounted — either queued to the owner's deferred-release queue
    (`DeferredReleaseDisposition::Queued`) or quarantined with
    `QuarantineReason::OwnerDropped` (`deferred.rs:440-449`, `:153-174`). Run the
    same case *during an unwind*, from a `Drop` reached by an in-flight panic,
    and assert it still does not abort — that is the case a `panic!` in `Drop`
    would turn into a process abort, and it is why §3.4 forbids one.

Tests 2–8, 10 and 12–15 must run on CPU without a model where possible, so they
gate every PR rather than only CUDA runs. Tests 1, 9 and 11 need the real
backend, because throughput, teardown ordering and admission behaviour are only
meaningful against real device work.

The phase each test gates is §13's, restated here so the two cannot drift: tests
2–7 gate Phase 2 (the routing lease, at `W = 1`); tests 13–15 gate Phase 1 (the
ownership and affinity rules); tests 1, 8 and 10–12 gate Phase 3 (`W > 1`); test
9 gates whichever phase first spawns more than one worker.

Of the Phase 2 gate: **2, 3, 4, 6 and 7 have landed; 5 is deferred behind the
absence of a routed `reset_session`, with the close-racing-a-turn case standing
in for its shape.** Two further guards landed that the list above did not ask
for, because the lease reaches further than the turn path:
`eviction_skips_a_session_with_a_turn_in_flight` and
`eviction_refuses_to_close_the_only_sessions_that_are_all_busy`
(`server/src/session.rs`) pin that LRU eviction cannot close a conversation
mid-turn, and `a_panic_while_holding_the_lease_releases_it` (`lease.rs`) pins the
unwind path that `Drop` is relied on for.

A third group landed for the registry invariants of §5 and §5.1, and it is
concurrent in the same sense the list above demands. Two models are loaded whose
first sessions have *provably identical* placements — the fixture asserts the
collision rather than assuming it — and the tests then pin that a busy session on
one model cannot be evicted or closed by the other, that a `DELETE` of a
non-default model's session closes it on that model's engine, that an insert at
full capacity with every conversation busy is **refused** rather than admitted
over `max_sessions`, and that the next insert after a release evicts rather than
grows (`server/src/tests.rs`, `session.rs`). Two thread-and-barrier regressions
cover the close race directly: racing deletes unbind exactly one binding once,
and a delete racing a rebind never leaves an orphan — every conversation ends
either bound or closed, never both and never neither.

A fourth group pins the `active_sessions` accounting of §5: sixty-four rounds of
eviction-and-insertion at `max_sessions = 1` leave both the registry length and
the count at one, an insertion below the bound counts one, a capacity refusal and
a refused close count nothing, a close counts one departure and a second close of
the same id counts none, and eight threads churning the bound leave the count
equal to the map. These read a counter the test owns rather than the process-wide
gauge, because every other test in the binary moves that gauge while they run, so
an exact assertion against it would be a race rather than a measurement; one
further test asserts the production registry still reports to the real gauge,
which is the one thing a local counter cannot see.

### 12.2 ORT / native parity

Concurrency must not change results *that it has no right to change*. Stating
the gate carelessly would make it unmeetable, so state it precisely.

The existing parity harness is the instrument: `tests/parity/README.md` drives
`scripts/check_native_ort_parity.py` against a `profile_native` build, and it is
already scoped rather than absolute — its conclusions are pinned to *observed
first-divergence steps*, generated-token index 22 for Qwen2.5-1.5B and index 19
for Qwen2.5-7B, where it requires native's token to equal the committed exact-Q4
float32 oracle token (`tests/parity/README.md:28-31`). That scoping is the model
to copy, not an embarrassment to fix.

**Why unconditional byte identity is the wrong gate.** Under continuous batching
the batch a session is decoded in depends on what else arrived, so `W = 1` and
`W > 1` will generally form *different batches* for the same prompt. Different
batch composition means different reduction shapes and can mean different kernel
selection; the numerics are then legitimately different in the last bits, and a
gate demanding byte identity across `W` would either fail honestly or be quietly
weakened until it proved nothing. It would also be measuring the wrong property:
the risk this design introduces is a session reading another session's state, not
a change in floating-point associativity.

The gates, stated so they are both meetable and meaningful:

- **Byte identity at fixed batch composition.** Replay a recorded batch
  composition — same sessions, same step boundaries, same batch widths — through
  `W = 1` and through `W > 1` and require byte-identical tokens. Holding
  composition fixed removes the only legitimate source of divergence, so any
  remaining difference is cross-session contamination, which is exactly what must
  fail the build. This is the load-bearing gate.
- **Oracle-anchored identity at free composition.** With composition left free,
  the weaker but still exact gate applies: at the harness's pinned divergence
  indices, the emitted token must equal the oracle token under both `W = 1` and
  `W > 1`. This catches a real behavioural change without pretending bit-level
  reproducibility across differing batch shapes.
- **Sampling determinism is unconditional.** Given a fixed seed and a fixed
  accepted-token sequence, the sampler must produce the same draws at any `W`.
  Per-turn RNG counter state is category 3.3 precisely so that this holds; unlike
  batch numerics, there is no legitimate reason for it to vary, so it is gated
  without a composition escape hatch.
- **ORT-vs-native parity does not loosen.** The existing comparison is re-run
  with `W > 1` on both sides at its existing tolerance. If it has to loosen, the
  design is wrong and this document is wrong with it.
- **`W = 1` is bit-identical to the pre-migration driver.** At `W = 1` there is
  no composition difference to appeal to, so this one *is* unconditional, and it
  is the gate that makes the phased rollout in §13 reversible.

Both backends are selected as CI already selects them:
`cargo test --locked -p onnx-genai-engine --features native-backend --
--test-threads=1` for native, and the ORT-backed package set likewise with
`--test-threads=1` (`.github/workflows/ci.yml:960-968` and `:949-953`
respectively). Note the irony and handle it explicitly: the harness pins
`--test-threads=1` so that tests do not contend for the device, which means the
concurrency tests in §12.1 must create
their own threads inside a single test binary rather than rely on the test
runner. They are written that way.

### 12.3 Performance and memory acceptance gates

Per [`../README.md`](../README.md)'s first standing rule, every number below is
reported with model, hardware, EP, `W`, batch width, and whether the run was
solo. A bare figure is not evidence. Per RULES.md Rule 9, every figure is
tier-scoped and no single device generalizes.

**Must not regress:**

- **`W=1` throughput and TTFT** against the pre-migration driver, measured on
  the same host. The existing absolute floors are the model for how this is
  expressed: `NATIVE_CPU_DECODE_FLOOR_TOK_PER_S = 18.0` and its roofline
  fraction (`crates/onnx-genai-bench/tests/profile_native.rs:15-16`), which
  protect against catastrophic regressions rather than measuring goodness.
- **Single-session latency at `W>1`.** A session alone in the pool must not pay
  for the pool's existence. Routing overhead is one atomic read and a channel
  send; if it measures otherwise the design has a defect, not a tuning problem.
- **Host memory at `W=1`.** The state split must not duplicate the package.

**Must improve, or the pool is not worth its cost:**

- **Aggregate throughput across `W` concurrent sessions** must scale materially
  with `W` up to the device's limit. "Materially" is fixed to a concrete number
  in the implementation PR against a measured `W=1` baseline on a named device;
  fixing it here without a measurement would be exactly the unqualified figure
  the standing rule forbids.

**Must be measured and bounded, because they are the design's known costs:**

- **Device memory as a function of `W`**, reported per worker and in total, since
  each worker owns a KV cache, capture buffers and staging. This is what makes
  the `W` bound in §4.4 a computation rather than a guess.
- **Prefix-cache hit rate at `W>1` versus `W=1`** (§9). The expected regression
  is real; an unmeasured one is a silent tax on every shared system prompt.
- **Host-lease credit residue at `W>1`** (§8), which scales with `W` by
  construction (`host_lease.rs:26-37`).

`scripts/ci_bench_compare.py` remains informational and non-blocking by its own
statement; these gates are assertions in test code, in the style of
`profile_native.rs`, so that a regression fails rather than gets commented on.

---

## 13. Migration and backward compatibility

Each phase lands independently, is testable on its own, and leaves the tree
shippable. Per RULES.md Rule 3 no compatibility shim or alias is introduced for
our own pre-release surface: each phase updates all callers, docs, fixtures and
tests in the same change.

**Phase 0 — typed routed identity.**

✅ **Done, with `SessionPlacement` rather than shard-encoding `SessionId`.**
Engine session ids remain worker-local and may collide. The externally routed
identity is the typed pair of `WorkerId` and local engine session id, and the
server further model-qualifies it as `ModelSessionPlacement`. This avoids
changing engine-local id semantics while making accidental cross-worker or
cross-model aliasing impossible.

**Phase 1 — split the state. The backend ownership/affinity prerequisite is
already complete.**

✅ **Done, by #2019 (`27b8945e2` + `f3c89255a`).** This phase used to begin with
§1.4's lifecycle defect. It no longer does, because that work has shipped:
`IoBinding` and session-derived `Allocator`s co-own their `Arc<Session>`
(`binding.rs:17-26`, `allocator.rs:207-219`), the `thread_affinity` module
supplies structural `!Send`, typed affinity errors and a non-panicking teardown
form (§3.4, Rule B), and §12.1's tests 13–15 exist in
`crates/onnx-genai-ort/tests/session_thread_contract.rs`. Phase 1 no longer
blocks on it, and Phase 2 no longer waits for it.

✅ **Done: the remaining engine session holders and persistent runners now use
shared ownership.** `Engine::session`, `DraftModel::session`, and
`Eagle3Model::session` are `Arc<Session>` alongside MTP. Stateful primary,
static-cache, MTP, and EAGLE-3 runners retain a named borrowed/shared session
owner; the engine uses the shared constructor, so each stored binding co-owns
its session and no lifetime cast supplies a synthetic `'static` borrow. The Arc
pointee stays stable across holder moves/clones, while mutable KV, bindings,
allocators, device values, and graph-capture state remain runner/worker-local.
`Session::supports_concurrent_run()` is preserved as the capability signal;
`W = 1` remains serial and this prerequisite makes no concurrent-run claim.

**What this phase must not do is delete a compensating teardown before the
structural replacement exists.** Two are explicitly retained:

- **`WorkflowRuntime::drop`'s manual clears stay** (`pipeline/mod.rs:201-208 via pipeline/runtime_state.rs:351-365`).
  `Arc<Session>` does *not* license removing them. Its own doc comment says why:
  bindings and session-derived allocators are now structural, but *"A `Value`
  cannot: it is a bare `OrtValue` handle with no back-reference to the allocator
  whose memory it holds, so ORT still requires every value to be released before
  that allocator. Nothing in the type system says so, which is exactly why it is
  done here explicitly instead of left to the field list"*
  (`pipeline/mod.rs:189-200`). Removing it becomes possible only if a later phase
  gives `Value` an ownership edge to its `Allocator`; until then §4.3 carries the
  ordering as a worker obligation.
- **`Drop for Environment`'s ordering discipline stays** (`env.rs:308-324`), as
  does `WorkflowRuntime`'s last-field `_ort_environment` (`pipeline/mod.rs:183-186`).
  §3.5 shows nothing structurally keeps the environment alive behind a session,
  so these are the only things enforcing an ORT lifetime requirement. They are
  removed only in whichever later phase adopts §3.5's structural option, and only
  together with it.

The rest of the phase is the state split proper: extract the immutable package
and compiled workflow plan (§3.1) behind `Arc`; separate per-execution turn state
(§3.3) from the interpreter; move the interpreter's `workflow_session_leases` out
of `RefCell` and into per-turn-owned worker state, where it stays as the
inner-layer invariant §4.2 describes. Extend the shipped `thread_affinity` vocabulary
(§3.4, Rule B) to the native handles §4.3 lists as still outstanding — structural
`!Send` where the type allows it, `OwnerThread` plus typed affinity errors
otherwise — where at `W = 1` every check passes trivially and therefore
establishes the baseline rather than chasing a regression. Still one thread,
still one worker. This is the largest refactor and the one that carries no
concurrency risk, which is why it is isolated.

✅ **Done, for the interpreter: the state split proper**
(`pipeline/runtime_state.rs`, `engine/ids.rs`). `WorkflowRuntime`'s flat field
list is now three named owners plus the environment it already kept last:
`Arc<WorkflowPlan>` is §3.1's immutable plan, asserted `Send + Sync` because
nothing in it is a cell; `WorkerBackend` holds the ORT and native handles and
`WorkerRuntimeState` holds everything one worker mutates (§3.2 storage under
§3.3 access), both structurally `!Send`/`!Sync` through a `PhantomData`
marker rather than a new `unsafe impl` or a blanket lock; and `WorkflowPass` is
§3.3's per-execution state, created when a pass opens and dropped when it ends.
`workflow_session_leases` moved into `WorkerRuntimeState::session_leases`, where
it remains the inner-layer invariant §4.2 describes. The id namespaces are typed
and separated with it: `PassId` (was `workflow_execution_generation`) and
`GraphCaptureId` mint from worker-local allocators, because they are only ever
compared against state the same worker owns, while session ids — the only ones
that leave the engine, and the ones a routing layer would key on — mint from
`engine::ids::SharedSessionIds`. Each worker owns an allocator; the routing
layer qualifies its local ids by worker, so numeric collisions are safe.
Externally observed ids are unchanged. Both `Drop`
retentions above are kept verbatim; the clears moved into
`WorkerRuntimeState::release_ort_state`, called from the same `Drop`.

⚠️ **The remaining Phase 1 work is native affinity:** extending the shipped
`thread_affinity` vocabulary to the native handles §4.3 lists. The state split
and ORT shared-ownership conversion are structural prerequisites at `W = 1` —
they move ownership into types; they do not make anything concurrent.

**Phase 2 — the routing lease, on the worker pool of one that already exists.**
PR #2012 (merged as `1bf87c86`) already built the worker thread, the
`WorkerId`/`SessionPlacement` types, the `WorkerHandle` and a `WorkerPool` of one
(`worker.rs:34-50`, `:66-85`, `:118-236`, `:255-323`), so this phase does not
rebuild them. What it adds is the piece #2012 deliberately did not: the
**routing-layer exclusive lease** of §4.2, keyed by `SessionId`, acquired in the
routing façade **before enqueue or admission** per §4.2.1, with the guard
travelling alongside the command and released on every completion, error,
cancellation, send-failure and worker-loss path.

This is the phase where `ExclusiveLeaseConflict` stops being unreachable. Because
acquisition happens before enqueue rather than inside the serial worker loop, the
refusal is reachable at `W = 1` — a second overlapping turn for a live session is
refused immediately rather than queued behind the first — so §12.1's tests 2–7 — every
test whose subject is the lease rather than the pool — gate *this* phase and do
not wait for `W > 1`. If acquisition were deferred
into the worker loop, those tests could not be written until Phase 4, and the
refusal contract would ship untested; that ordering constraint is the reason this
phase exists separately at all.

The `EngineOwner` wrapper and its `unsafe impl Send` are deleted here
(`driver.rs:249`, `:278-281`). #2012 kept them because it moved the engine into
the worker thread; this phase constructs the engine *on* the worker thread
(§4.3), which removes the cross-thread move the SAFETY comment is arguing about.
The gate is §12.2's `W = 1` bit-identity requirement against the pre-migration
driver, which is unconditional at `W = 1` and therefore a real gate.

✅ **Landed: the routing lease itself, and the refusal it makes reachable.**
`crates/onnx-genai-server/src/lease.rs` holds `SessionLeases` (a sharded map
keyed by `ModelSessionPlacement`) and the `#[must_use]` RAII `SessionLeaseGuard`.
The `SessionRegistry` owns the one map, because §4.2 requires it be readable
before a command exists and §5.1 requires the same map answer for every loaded
model. Acquisition happens on the calling task in the route handler
(`routes/completions.rs`, `lease_bound_session`) **before** the session-carry
round trip, before the admission permit, and before the `DriverCommand` is built;
a conflict is mapped through the pre-existing `package_capability_failure`, which
matches on the `PackageCapabilityError` variant, so the 409 is the same 409 the
engine's own refusal produces. The guard is then moved into
`DriverCommand::Generate` and travels with the turn, so all five endings in
§4.2.1's table release it by `Drop`: normal completion and pass errors release in
`run_generation` (immediately after the engine commits, before the terminal event,
so a client's next turn cannot race the drop); an abandoned continuous-batch route
releases it with the `DriverRoute` row; a failed send releases it on the
submitting task; a stopped worker drops the queued commands. The same guard is
what `close_session` now takes by value, what `DELETE /v1/sessions/{id}` acquires
before it unbinds the id, and what LRU eviction must take to choose a victim
(§5, §5.1). One lease covers decode-core ORT, decode-core native and interpreted
sessions alike, because it is keyed on the routing identity rather than on
anything a package declares. Stateless requests and FIM completions take no
lease, and distinct sessions never conflict with each other.

✅ **Landed with it: the three registry invariants the lease is only correct
under.** (a) The key is **model-qualified** and the map is **one map** — two
loaded models produce identical `SessionPlacement`s for their first sessions, so
a bare placement is not a global identity and a per-driver map is not a global
answer (§4.2). A session id bound to one model is refused with the same typed 409
if it is presented on another, rather than generating into an unrelated
conversation that happens to share a placement. (b) `max_sessions` is **strict**:
when every binding is mid-turn there is no evictable victim, and the new session
is refused with a typed capacity error mapped to the existing 429
`resource_limit_error` rather than admitted over the bound, which would have made
the limit advisory permanently (§5). (c) Close is **atomic**: `take_for_close`
finds, leases and unbinds one binding under the registry lock and returns the
guard naming its owner, so `DELETE` closes exactly what it removed on exactly the
model that owns it — never the default model, and never a conversation that was
rebound between the read and the remove (§5). LRU eviction removes its victim
under the same lock and rule.

✅ **Landed: the owner-thread lifecycle prerequisite.** `EngineOwner` and its
`unsafe impl Send` are gone. Model paths, engine options, and the shared memory
authority cross into the new worker as a `Send` load plan; the engine and its
backend handles are constructed, run, unwound on failed initialization, and
dropped there. The caller receives a typed ready/error handshake and keeps the
same `W = 1` driver façade. This changes ownership only: it adds no parallel
execution, batching, or multi-worker routing claim.

⚠️ **Also outstanding at the end of this phase:** §12.1's test 5 (reset racing a
turn), which cannot be written until `reset_session` is routed at all — see §5 on
which operations have a routing-layer caller today — and the accounting half of
test 6 (a cancelled turn's reservations returning to the budget), which belongs to
§8 rather than to the lease.

**What this phase explicitly did not do**, so that the next reader does not have
to infer it: it did not enable `W > 1`, did not multiplex turns inside a worker,
did not introduce a global `Mutex<Engine>`, and did not change any backend
`Send`/`Sync` impl. Two turns on two different sessions still execute one after
the other. The only observable change is that a *second* turn on a session that
already has one is refused instead of queued.

**Phase 3 — `W > 1`.**

✅ **Done for opt-in contracted single-decoder ORT execution.** The typed
`OrtSessionWorkerCount` is exposed as `--ort-session-workers` and
`ONNX_GENAI_ORT_SESSION_WORKERS`, defaults to one, and is bounded to `1..=64`.
`WorkerPool` owns N handles; every `Engine` is constructed and destroyed on its
worker thread, and the former `unsafe impl Send for Engine` is gone.

The primary worker freezes an `OrtEngineWorkerFactory`. Additional workers
share the immutable ORT session/environment/tokenizer/workflow plan and global
device memory authority, while constructing fresh worker-local mutable state.
Startup is transactional: any initialization failure shuts down and joins
already-ready workers.

Session placement is least live-plus-pending sessions with lowest-id ties;
stateless placement is least active turns with the same tie rule. RAII
reservations release counters on success, error, cancellation, and failed send.
All session operations currently exposed by the server route through the saved
placement. Local ids may collide, no cross-worker migration occurs, and
continuous batching remains per worker.

The capability gate fails closed unless the backend is ORT, the workflow is one
contracted decoder, speculative/external-KV paths are absent, and the selected
ORT session reports concurrent `Run` support. Graph capture and single-flight
providers withdraw that capability in the ORT session layer. Native and
composite configurations are refused rather than reduced to `W = 1`.

Per-worker id, health, active turns, live sessions, and KV usage are additive
fields in status/debug responses. A worker failure invalidates only its
placements and is reported as typed unavailable; it is not silently restarted
or migrated. Explicit shutdown joins every worker.

**Phase 4 — sessionful continuous batching.** Admit sessionful turns into the
per-worker batch loop and delete the deferral comment at `driver.rs:814-817`.
This is the phase that delivers the throughput case in §12.3 and is deliberately
last, because it is the only phase whose failure mode is a correctness bug in
batching rather than in concurrency. It needs no same-session check of its own:
per §9 the routing lease from Phase 2 already guarantees a session never has two
rows in flight.

**Compatibility.**

- The server handle keeps its public routing methods and becomes the concurrent
  façade. The low-level `Engine` intentionally remains an owner-thread object;
  callers needing parallelism construct one engine per owner or use the server
  worker pool rather than wrapping an engine in a global mutex.
- The Python `Engine` pyclass is explicitly `unsendable`, matching the
  low-level engine's structural owner-thread contract. This server phase makes
  no Python session-concurrency claim.
- The C ABI exposes no session lifecycle today
  (its entire exported surface runs from `oge_last_error` to `oge_string_free`,
  `crates/onnx-genai-capi/src/lib.rs:79-423`, with no `oge_session_*` entry
  point), so it gains a thread-safety
  guarantee and loses nothing. Its thread-local `oge_last_error`
  (`capi/src/lib.rs:54-68`) is already correct for a multi-threaded caller.
- HTTP behaviour changes in exactly one visible way: a 409 with
  `kind == "conflict_error"` becomes reachable for overlapping turns on one
  session (`routes/mod.rs:523-525`). It is already documented, already typed and
  already marked retryable (`capability.rs:57-61`), so a client that follows the
  existing contract needs no change.
- `copy_on_write` session mutation stays refused at load
  (`workflow_session_continuation.rs:695-702`). Nothing here enables it; it is
  named only so a reader does not assume concurrency implies it.

---

## 14. Resolved and open questions

1. ✅ **Default `W`: one.** Parallel workers are explicit opt-in; no device
   heuristic changes the default.
2. ✅ **Sessionless request placement:** fewest active/in-flight turns, lowest
   worker id on ties.
3. ✅ **Session placement:** fewest live plus pending sessions, lowest worker id
   on ties. Placement remains fixed for the session lifetime; no migration
   follows later load changes.
4. **Cross-shard prefix sharing.** §9 accepts up to `W` copies of a shared
   prefix. A host-side shared prefix store that each worker materializes from is
   possible and is out of scope here; §12.3's measurement is what would justify
   it.
5. **Checkpoint portability across worker counts.** `SessionCheckpoint` carries
   only `{ session_id, position }` (`config.rs:433-438`) and restore is a rewind
   of a live session rather than a reconstruction. The server does not yet
   expose checkpoint/restore routing; when it does, the placement must travel
   beside the checkpoint and a missing owner must be refused, never migrated.
   Portable checkpoints would require serializing KV state, which is a
   different feature and is not silently approximated here.
6. **§3.5's environment lifetime option.** Whether `Session` grows a structural
   `Arc<Environment>` or the process-global `OnceLock` stays with a documented
   scope rule is left to implementation, because the answer depends on how much
   of the EP plugin registry moves with it. What is *not* open is the ordering:
   the compensating `Drop for Environment` (`env.rs:308-324`) and
   `WorkflowRuntime`'s last-field `_ort_environment` (`pipeline/mod.rs:183-186`)
   are not removed before whichever option is chosen actually lands.
