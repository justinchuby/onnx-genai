# Session concurrency

**Authoritative for:** what the public session API promises about threads, how
sessions are owned and executed concurrently, and what may never be shared.

This document is a design, not a status report. Everything in §1 is a statement
about code that exists today, with a citation. Everything from §2 onward is a
decision about code that does not exist yet, and says so.

Related: [`DESIGN.md`](DESIGN.md) for the execution architecture,
[`../genai/SCHEDULING.md`](../genai/SCHEDULING.md) for inter-session scheduling
policy, [`../genai/PIPELINE.md`](../genai/PIPELINE.md) for the workflow
interpreter, [`../genai/INFERENCE_METADATA_DECISIONS.md`](../genai/INFERENCE_METADATA_DECISIONS.md)
§12.5a for the normative session-lease contract, and
[`ERROR_AND_LOGGING_CONVENTIONS.md`](ERROR_AND_LOGGING_CONVENTIONS.md) for how
the refusals below must be typed.

---

## 1. What is true today

### 1.1 The engine is `Send`, and is structurally not `Sync`

`Engine` carries a hand-written `unsafe impl Send` whose safety argument is that
moving it transfers exclusive ownership and that mutation still requires
`&mut Engine` (`crates/onnx-genai-engine/src/engine/model.rs:221-229`). It
carries no `Sync` impl, and could not honestly carry one: the workflow
interpreter it owns holds
`workflow_session_leases: RefCell<std::collections::HashSet<String>>`
(`crates/onnx-genai-engine/src/pipeline/mod.rs:191`), and `RefCell` is `!Sync`
by construction.

That is the important fact, and it is easy to miss because the `unsafe impl
Send` looks like the concurrency story. It is not. The concurrency story is that
there is exactly one thread.

The server makes that explicit. `EngineOwner(Box<Engine>)` carries a second
`unsafe impl Send` (`crates/onnx-genai-server/src/driver.rs:275-278`) whose
safety comment is a promise about scheduling rather than about the type:

> The engine is moved exactly once into the dedicated driver thread. All ORT
> runners, sessions, KV state, and the continuous batch manager stay owned by
> that thread and are accessed only by processing channel commands.

`EngineDriver::start` spawns one OS thread named `onnx-genai-batch-driver` and
moves the owner into it (`driver.rs:318-330`). Every request arrives as a
`DriverCommand` over a `tokio::sync::mpsc` channel and is executed by
`run_engine_driver` (`driver.rs:690-722`), which picks one of two serial loops:
`run_static_engine_driver` when the decode path can advance several rows per
forward pass, and `run_fallback_engine_driver` — a bare
`while let Some(command) = rx.blocking_recv()` — otherwise (`driver.rs:725-733`).

The Python binding reaches the same conclusion by a different route: it holds
`inner: Mutex<RustEngine>` and refuses contention outright rather than blocking,
raising `PyRuntimeError` with the message *"engine is in use by another thread —
onnx_genai Engine is not re-entrant; serialize calls or use one Engine per
thread"* (`crates/onnx-genai-python/src/lib.rs:173-186`).

So today's answer to "are sessions thread-safe?" is: the question does not
arise, because there is one thread. That is a defensible position for a single
GPU serving one decode core. It is not a defensible position for a public API,
because callers cannot tell from the type system that it is true — `Engine` is
`Send`, so it compiles inside an `Arc<Mutex<_>>`, and a caller who does that
gets whole-engine serialization they did not ask for and cannot see.

### 1.2 The exclusive lease is already typed — and already unreachable

The metadata contract already declares this. A session-scoped state cell carries
a `SessionLeaseContract` whose `policy` defaults to `exclusive`
(`schema/inference_metadata.schema.json:3215-3254`), and the normative
specification states the consequence:

> A lease declared `policy: exclusive` is single-flight: a second turn that
> starts while one is in flight is refused by name rather than allowed to read a
> conversation the first is about to replace.
> — [`../genai/INFERENCE_METADATA_DECISIONS.md`](../genai/INFERENCE_METADATA_DECISIONS.md):1103

The runtime implements that refusal. `SessionLeaseGuard::acquire` inserts into
the lease set or returns
`PackageCapabilityError::ExclusiveLeaseConflict { session }`, and `Drop` removes
the entry so a pass that fails or panics does not strand the session
(`crates/onnx-genai-engine/src/pipeline/workflow.rs:598-633`, acquired at
`workflow.rs:1363`). The error is typed, is marked retryable
(`crates/onnx-genai-engine/src/engine/capability.rs:50-61`), and the server maps
it to HTTP 409 by matching the variant rather than the wording
(`crates/onnx-genai-server/src/routes/mod.rs:523-525`).

Every piece is in place except the ability to reach it. The server's own test
says so:

> `// 3. ExclusiveLeaseConflict — the driver serializes passes, so this is`
> `//    raised where it is decided and mapped where it is answered.`
> — `crates/onnx-genai-server/src/tests.rs:4856-4857`

The test constructs the error by hand and checks the mapping, because no HTTP
request sequence can produce it. A `RefCell<HashSet<String>>` on a
single-threaded driver cannot observe two turns in flight.

This document's central claim follows directly: **the design work is not
inventing a conflict error, it is making the existing one reachable and
correct.** The refusal contract is settled. What is unsettled is the execution
model underneath it.

### 1.3 Which backend handles are thread-affine

This is the constraint that decides everything else, so it is stated from the
code rather than from folklore.

**Not thread-affine, per their own safety comments:**

| Handle | Location | Claim |
|---|---|---|
| `Session` (ORT) | `crates/onnx-genai-ort/src/session/mod.rs:1224-1235` | `Send + Sync`; "ONNX Runtime documents `OrtSession::Run`/`RunWithBinding` as safe for concurrent calls on the same session" |
| `Environment` (ORT) | `crates/onnx-genai-ort/src/env.rs:326-335` | `Send + Sync`; process-level handle ORT permits from multiple threads |
| `CublasLt` | `crates/onnx-runtime-ep-cuda/src/blas.rs:97-101` | "a cuBLASLt handle is not thread-affine" |
| `PinnedStaging` | `crates/onnx-runtime-ep-cuda/src/runtime.rs:2123-2127` | a page-locked host allocation; the pointer is a plain address |

**`Session` is the only ORT handle in that list that a session's execution
actually reaches, and the exception does not extend to anything derived from
it.** `IoBinding`, `Allocator` and `Value` carry no `Send`/`Sync` claim of any
kind — not an `unsafe impl`, not a derived one that survives their raw pointers
— and none is being added. ORT's guarantee is about `Run`; the objects around a
run are per-run or per-worker state that the caller supplies, which the
`Session` safety comment says in as many words: *"per-run inputs, outputs, and
`IoBinding` values are supplied by the caller and are not stored in `Session`"*
(`session/mod.rs:1226-1227`). Decision 4 therefore places `IoBinding`,
`Allocator` and `Value` in the per-worker category (§3.2) even though `Session`
itself could be shared.

**A graph-capture session is not concurrent, even though `Session` is `Sync`.**
A session created with `enable_cuda_graph=1` reports `graph_capture == true`
(`session/mod.rs:487-490,823-825`) and replays one captured graph against one
set of bound device addresses — which is why island binding exists *"so CUDA
graph capture sees stable, device-resident input and output addresses"*
(`session/mod.rs:1168-1172`). Two concurrent `Run`s against one captured graph
would replay it against addresses the other is writing. Combined with the
thread-local capture rules below, a capture-enabled session is **single-flight
and thread-affine**, and the `Sync` impl on `Session` does not reach it.

**Thread-affine or context-bound, per their own safety comments:**

| Handle | Location | Claim |
|---|---|---|
| `CudnnBackend` | `crates/onnx-runtime-ep-cuda/src/cudnn/mod.rs:1129-1132` | "cudarc deliberately keeps `Cudnn` !Send/!Sync because a handle must not be used concurrently … `with_handle` binds the owning CUDA context to the calling thread first" |
| `RawReduceHandle` | `.../cudnn/mod.rs:961-963` | "runs on the thread bound to the owning CUDA context" |
| `CudnnReduceCache` | `.../cudnn/mod.rs:1099-1101` | same |
| `CudaGraphLifecycle` | `crates/onnx-runtime-ep-cuda/src/graph.rs:130-135` | every segment launches on its single owning stream |
| `CudaReservation` | `crates/onnx-runtime-cuda-memory/src/virtual_memory.rs:2749-2752` | "thread-affine, and every driver call through the backing binds the context first" |

**CUDA graph capture is thread-local, and the repo already knows it.** Capture
begins with `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` (`graph.rs:177`), must end on
the thread that began it (`graph.rs:184-193`), must abort on that same thread
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

### 1.4 A concrete lifecycle bug the current arrangement already permits

The categories in §1.3 are not theoretical. A backend audit found a live
drop-order defect that exists *today*, at `W = 1`, and it is the sharpest
available evidence for why §3's ownership rules have to be structural.

**Rust drops struct fields in declaration order.** `ExecutionIsland` declares:

```rust
    session: Session,                                                    // :74
    ...
    bindings: RefCell<HashMap<IslandBindingKey, StableIslandBinding>>,   // :78
    device_allocator: Option<Allocator>,                                 // :79
```
— `crates/onnx-genai-engine/src/pipeline/islands.rs:74,78,79`

So the session is released **first**, and the resources derived from it are
released **after** the handle they depend on is gone. Each of the three levels
states its own dependency, and none of them enforces it:

- `IoBinding` holds `_session: *const Session`, annotated in the source as
  *"reference back to session (non-owning)"* (`crates/onnx-genai-ort/src/binding.rs:12-15`),
  and its `Drop` calls `ReleaseIoBinding` (`binding.rs:173-182`) — after
  `Session::drop` has already called `ReleaseSession`
  (`session/mod.rs:1213-1222`). `StableIslandBinding` is what holds those
  bindings (`islands.rs:108-109`).
- `Allocator::for_session_device` says the rule outright in its own doc comment:
  *"The returned allocator becomes invalid when the session is dropped, so it
  must not outlive the session it was created from"*
  (`crates/onnx-genai-ort/src/allocator.rs:236-238`). The field order at
  `islands.rs:74,79` makes it do precisely that. Nothing in the type ties the
  allocator to the session — `Session::device_allocator` returns an owned
  `Allocator` by value with no lifetime and no shared handle
  (`session/mod.rs:1168-1175`).
- `Value`s allocated from that device allocator (`Value::empty_in`,
  `value.rs:232-240`) are the third level, and are held inside
  `StableIslandBinding`'s `inputs`/`outputs` (`islands.rs:110-111`).

**The compensating code is a convention, not an invariant.** `WorkflowRuntime`'s
`Drop` reaches into every island to clear its bindings, then clears three maps,
in that order:

```rust
impl Drop for WorkflowRuntime {
    fn drop(&mut self) {
        for island in &mut self.execution_islands {
            island.clear_bindings();
        }
        self.component_bindings.get_mut().clear();
        self.component_outputs.get_mut().clear();
        self.component_allocators.get_mut().clear();
    }
}
```
— `crates/onnx-genai-engine/src/pipeline/mod.rs:270-279`

That works, and it is why the defect has not bitten. But it is a hand-maintained
teardown running *outside* the type that has the problem: it papers over
`ExecutionIsland`'s internal field order from one level up. Any path that drops
an `ExecutionIsland` without going through `WorkflowRuntime::drop` — a
constructor that fails partway, a panic during load, a future refactor that
moves islands into a different owner — gets the raw order. §7's worker-failure
path is exactly such a path.

The codebase already knows field drop order is load-bearing. `Value::drop`
carries a careful comment reasoning about it — *"Any `_owner` here is a struct
field, so drop glue frees it after this body releases the `OrtValue` below — ORT
is always done with the allocation before the owner frees it"*
(`value.rs:1699-1701`). The same reasoning was simply not applied one level up.

**Why this belongs in a concurrency document.** At `W = 1` the window is narrow
and the manual clear closes it. Under a worker pool the same teardown happens
`W` times, concurrently with other workers still running, on paths where nobody
runs the compensating clear — and a `ReleaseIoBinding` against a released
session is not a data race any sanitizer will name. Multiplying an ordering
defect by `W` and adding failure paths is how a latent bug becomes a
reproducible crash. It is fixed *before* the pool exists (§13, Phase 1), and the
rule that prevents its return is structural (§3.4), not another convention.

### 1.5 Where session state, memory accounting, and KV actually live

- **Session maps.** `Engine` holds `sessions: HashMap<SessionId, EngineSession>`
  for decode-core packages and `workflow_sessions: HashMap<SessionId, usize>`
  for interpreted ones, plus a monotonic `workflow_session_counter`
  (`engine/model.rs:55-67`). The native backend holds a third map,
  `native_sessions: HashMap<SessionId, NativeSessionState>`, with LRU eviction
  bounded by `native_max_sessions` (`engine/model.rs:78-100`,
  `engine/runtime.rs:621-640`).
- **`SessionId` is not its own type.** `pub type SessionId = SequenceId`
  (`crates/onnx-genai-engine/src/config.rs:389`) and
  `pub type SequenceId = u64` (`crates/onnx-genai-kv/src/lib.rs:61`). On the ORT
  path `create_session` *returns the KV sequence id directly*
  (`engine/runtime.rs:1538`), so the session identifier and the paged-KV
  sequence identifier are the same number. This matters in §4.1.
- **Memory accounting is deliberately singular.** The `governor` field's doc
  comment states: *"Exactly one exists per runtime — a second would double-count
  every reservation — which is why the workflow interpreter beside it holds none
  and this is not an `Option`"* (`engine/model.rs:49-54`). Underneath,
  `ByteBudget` is already `Arc<Mutex<BudgetState>>` and documents itself as *"A
  shared, dynamic, cross-session KV byte budget"* whose clones "account against a
  single running total"
  (`crates/onnx-genai-scheduler/src/byte_budget.rs:122-133`), `HostGovernor`
  keeps its ledger behind `Mutex<Ledger>`
  (`crates/onnx-genai-scheduler/src/pressure.rs:416-431`), and
  `HostGovernorAccounting` bridges to it through `Mutex<Outstanding>`
  (`crates/onnx-genai-scheduler/src/host_lease.rs:50-60`).
- **KV is one table.** `PagedKvCache` owns a `PageTable` whose sequence ids are
  global within that instance (`crates/onnx-genai-kv/src/paged_cache.rs:57-62`).
- **Continuous batching does not carry sessions.** `run_static_engine_driver`
  states plainly that *"The current `ContinuousBatchManager` API accepts
  `GenerateRequest` only. `X-Session-Id` requests keep using the driver's
  per-request engine path"* (`driver.rs:744-747`). Sessionful traffic is on the
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

1. **Public session APIs are thread-safe.** A caller may hold one handle and
   call it from any thread, without external synchronization, and get defined
   behaviour. This is a property of the type, not of a documented convention.
2. **Different sessions execute concurrently.** Two sessions with no shared
   mutable state must not serialize against each other. Concurrency is the
   default, and any serialization is a named, attributable decision.
3. **The same session under an exclusive lease refuses, by type.** A second turn
   on a session whose lease is `policy: exclusive` returns
   `PackageCapabilityError::ExclusiveLeaseConflict`. It does not queue silently,
   it does not interleave, and it does not lose an update.
4. **Backend handles stay thread-affine and are not made `Sync` by wrapping raw
   handles in locks.** Where a handle names a bound context or an owning stream,
   the design supplies a thread, not a mutex. Affinity is recorded and asserted
   (`OwnerThread`), and a resource derived from a session owns that session
   (`Arc<Session>`) rather than pointing at it — §3.4. Both are structural
   because §1.4 shows a convention already failed here.
5. **Execution uses a bounded pool of session workers with deterministic session
   ownership and stateless load distribution.** A session belongs to exactly one
   worker for its entire life; the routing decision is a pure function of the
   session id.
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

with no undefined behaviour, no torn state, no lost update, and no silent
reordering of two turns on one session. It is explicitly **not** a promise that
concurrent calls on one session both succeed. Decision 3 says the opposite, on
purpose: silently serializing two writers to one conversation is how a caller
loses a turn without being told.

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

The plan/interpreter split is what makes this possible. Today
`WorkflowRuntime` mixes the compiled plan with per-pass mutable state (its lease
set lives there: `pipeline/mod.rs:191`). The plan must be extracted into an
`Arc`-shared, genuinely immutable value, leaving the interpreter as a
per-execution object constructed against it.

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

Two rules, both structural.

**Rule A — ownership, not adjacency.** Every resource derived from a `Session`
holds an `Arc<Session>`.

That means:

- `IoBinding` replaces its non-owning `_session: *const Session`
  (`crates/onnx-genai-ort/src/binding.rs:12-15`) with an `Arc<Session>`.
- `Allocator`, when produced by `Allocator::for_session_device`
  (`crates/onnx-genai-ort/src/allocator.rs:236-238`), carries the
  `Arc<Session>` it was derived from. Today it carries nothing
  (`allocator.rs:205-209`) and the invariant lives only in a doc comment.
- Any `Value` allocated from a session-derived allocator
  (`crates/onnx-genai-ort/src/value.rs:232-240`) keeps that allocator alive,
  transitively keeping the session alive.
- `Session::device_allocator` (`session/mod.rs:1168-1175`) is therefore callable
  only on an `Arc<Session>`, not on a bare `Session`.

`Arc<Session>` is not a new idea in this codebase — it is already how the engine
holds its session (`engine/model.rs:245`), how MTP holds its draft and target
sessions (`crates/onnx-genai-ort/src/mtp.rs:122`, `:166`), and how the
speculative decoder holds its sessions (`speculative/mod.rs:1056`). The gap is
narrow and specific: the edge from a session to the resources *derived* from it
is the one edge that was left as a raw pointer.

The payoff is that teardown order stops being a property of source-file layout.
`ExecutionIsland`'s field declaration order (`pipeline/islands.rs:74`, `:78`,
`:79`) becomes irrelevant — the session cannot be released while a binding,
allocator or value still refers to it, because the refcount says so. And
`WorkflowRuntime::drop`'s manual `clear_bindings`-then-clear-maps sequence
(`pipeline/mod.rs:270-279`) stops being load-bearing. §13 Phase 1 removes it in
the same change that adds the `Arc`, rather than leaving a correct-but-redundant
cleanup path for a future reader to mistake for a live invariant (RULES.md
Rule 3).

**Rule B — thread identity, checked.** Every per-worker backend handle records
the `ThreadId` it was constructed on, and asserts it on use and on drop.

```rust
/// The thread a backend handle was constructed on and must be used and
/// dropped on. Recorded at construction; asserted, never inferred.
pub(crate) struct OwnerThread(std::thread::ThreadId);

impl OwnerThread {
    pub(crate) fn current() -> Self {
        Self(std::thread::current().id())
    }

    #[track_caller]
    pub(crate) fn assert_owned(&self, what: &'static str) {
        let now = std::thread::current().id();
        assert_eq!(
            self.0, now,
            "{what} is thread-affine: constructed on {:?}, used on {now:?}",
            self.0
        );
    }
}
```

No such concept exists today. Grepping `onnx-genai-ort` and `onnx-genai-engine`
for a thread-identity check returns nothing; affinity is documented in SAFETY
comments (§1.3) and enforced by the fact that there is currently one thread.
Once there are `W`, the documentation is all that remains, and documentation
does not fail a test.

Three properties of Rule B are deliberate:

1. **It is always on, not `debug_assertions`-only.** The bug it catches is a
   CUDA context bound to the wrong thread or a handle released off its owning
   thread. That is silent undefined behaviour, not a wrong number — it will not
   reproduce under a debug build on demand, and a release build is exactly where
   it will be met. The cost is one `ThreadId` comparison against operations that
   already cost a device round trip.
2. **It asserts on `Drop`, not only on use.** §1.4's defect is a teardown bug.
   A check that only fires on use would not have caught it.
3. **It panics rather than returning an error.** A handle used from the wrong
   thread means the ownership routing in §4 is broken; there is no caller-level
   recovery, and §7's worker-failure path already converts a worker panic into a
   typed, actionable refusal for every affected request. This is the one place
   this design prefers a panic to a typed error, and it is because a wrong-thread
   handle has already violated the precondition that made the surrounding code
   sound.

Which handles carry an `OwnerThread`: the native decode session, the CUDA
context binding, `CudnnBackend` and its reduce cache, `CudaGraphLifecycle`,
`CudaReservation`, the paged KV cache, and every session-derived `IoBinding`,
`Allocator` and `Value`. ORT's `Session` does not need one for `Run` (it is
`Sync`), and takes one when `graph_capture` is enabled (§1.3), where it is
single-flight and affine like everything else.

---

## 4. The session-worker model

### 4.1 Deterministic ownership and stateless routing

A bounded pool of `W` worker threads is created at engine load, where `W` is
configured and clamped to what the backend can support (§4.4). Each worker owns
its own category-3.2 state and shares the category-3.1 plan.

**Ownership is a pure function of the session id.** `SessionId` becomes a
newtype carrying its owner:

```rust
/// Which worker owns a session, for the session's whole life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShardIndex(u16);

/// A session handle. The owning shard is part of the identity, so routing is a
/// pure function and no lookup table can go stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId {
    shard: ShardIndex,
    local: u64,
}
```

Routing is then `session.shard`, with no map, no lock, and no possibility of a
router and a worker disagreeing about who owns what.

This newtype is required work, not incidental cleanup, for two independent
reasons.

First, RULES.md Rule 5 already asks for it — *"Use newtypes when primitives can
be transposed: session ids, token counts, page indices, offsets, and lengths are
not interchangeable `usize`s"* — and today's `pub type SessionId = SequenceId`
over `pub type SequenceId = u64` (`config.rs:389`,
`onnx-genai-kv/src/lib.rs:61`) is exactly the case it describes.

Second, it is forced. Under §3.2 each worker owns its own `PagedKvCache`, so
each worker mints sequence ids from its own `PageTable` and those id spaces
collide across workers. Today `create_session` returns the KV sequence id as the
session id (`engine/runtime.rs:1538`); once there are `W` page tables that is
ambiguous. Splitting `SessionId` from `SequenceId` is the same change as making
routing total.

**Load distribution is stateless.** "Stateless" here is a precise claim, so it is
defined: the distributor holds **no per-session state, no routing table, and no
feedback loop** — nothing whose staleness could make it route a session wrongly.
It reads one array of per-shard live-session counters held as `AtomicUsize`,
takes the minimum, and breaks ties with a single round-robin `AtomicU64`. Those
counters are shared, but they are a lock-free hint whose staleness costs at worst
a slightly uneven placement; nothing reads them to *find* a session. It consults
no queue depth, no latency feedback, no lock, and no history.

This is a deliberate ceiling on ambition. A feedback-driven placement policy is a
scheduling decision, it belongs in
[`../genai/SCHEDULING.md`](../genai/SCHEDULING.md), and it cannot be retrofitted
onto placement anyway because a placed session **cannot migrate** — its backend
state is thread-affine by §3.2. Placement is therefore a one-shot decision made
with the least information that still balances, rather than a good decision made
with information that would imply a rebalancing mechanism that cannot exist.

The consequence must be stated rather than hidden: uneven session lifetimes can
skew a shard. A shard whose live-session count exceeds its configured bound
refuses `create_session` with a typed error naming the shard and the bound,
rather than migrating (impossible) or over-subscribing (which converts a refusal
into a latency cliff).

### 4.2 The lease registry

`workflow_session_leases: RefCell<HashSet<String>>` (`pipeline/mod.rs:191`) is
replaced. The replacement is **not** a `Mutex<HashSet<...>>` in the same place.

Because a session is owned by exactly one worker (§4.1), all turns on one
session are already serialized onto one thread. The lease set is therefore
*per-worker*, holds only that worker's sessions, and needs no cross-thread
synchronization at all — it can stay a plain `HashSet` behind the worker's
`&mut`. A global lock here would be a lock that never contends usefully, taken
on every turn, protecting an invariant that ownership already provides.

`SessionLeaseGuard`'s existing shape is kept verbatim in spirit: acquire inserts
or returns `ExclusiveLeaseConflict`, and `Drop` removes
(`pipeline/workflow.rs:598-633`). The reason its doc comment already gives — *"a
pass that fails or panics does not strand the session"* — is exactly the property
§6 needs, so the guard is the mechanism and not a new one.

**What changes is reachability.** With one worker serving many sessions and many
in-flight turns, a second turn arriving for a session whose lease is held is now
an ordinary event rather than an unconstructible one, and the 409 at
`routes/mod.rs:523-525` becomes a status an integration test can produce over
HTTP. The server test's parenthetical — *"the driver serializes passes, so this
is raised where it is decided and mapped where it is answered"*
(`server/src/tests.rs:4856-4857`) — is deleted in the same change, and replaced
by the real test in §12.1.

### 4.3 What must be constructed and dropped on a worker thread

This is the part that is easy to get wrong quietly, so it is normative.

**Must be constructed on the worker thread that will use it:**

- the CUDA context binding, and everything created under it;
- the cuDNN handle and its descriptor caches, because `with_handle` binds the
  owning context to the calling thread (`cudnn/mod.rs:1129-1132`);
- any `CudaReservation`, which its own comment calls thread-affine
  (`virtual_memory.rs:2749-2752`);
- the CUDA graph capture: begin, end, and abort must all be the same thread
  (`graph.rs:177,184-193,274-285`);
- the native decode session and the per-worker `PagedKvCache`, so their
  allocations are charged under the right context;
- every session-derived `IoBinding`, `Allocator` and `Value` (§3.4), which today
  are built wherever the workflow runtime happens to run
  (`pipeline/islands.rs:78-79`, `allocator.rs:236-238`, `value.rs:232-240`).

Each of these records an `OwnerThread` at construction (§3.4, Rule B). The
recording is what makes the drop rule below checkable instead of aspirational.

**Must be dropped on the worker thread that constructed it.** Teardown makes
driver calls that bind the context first
(`virtual_memory.rs:186-189`, `vmm_allocator.rs:686-689`); running those on a
foreign thread is the same violation as running the work there. Concretely: a
worker's `join` must happen only after that worker has dropped its own
category-3.2 state, and nothing may claw a handle back to the coordinator to
drop it. §7 makes this a shutdown ordering rule.

Native CUDA teardown is the strictest case and is called out separately: the
CUDA context binding, the captured graph and its lifecycle, the reservation, the
device allocator and the KV pages beneath it are all released by the owner
thread, in that thread's own unwind or shutdown path. `OwnerThread::assert_owned`
fires in each of their `Drop` impls, so an attempt to release them elsewhere
aborts loudly at the point of the mistake rather than corrupting a context that
some other worker is mid-capture on.

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
(`driver.rs:297-303`):

1. **Backend capability.** A backend that cannot hold more than one decode
   session gives `W = 1`. The native path is already documented this way — *"the
   native single-session backend does not support {feature}; use independent
   serialized requests"* (`engine/runtime.rs:653-658`).
2. **Device memory.** Each worker owns a KV cache and its own capture and
   staging buffers. `W` is chosen against the same governor that already refuses
   over-admission, and a `W` the device cannot fund is a load-time refusal, not
   a runtime OOM (RULES.md Rule 9: *never silently OOM*).
3. **Operator configuration**, clamped to the two above.

`W = 1` must remain a fully supported configuration, and must be
behaviour-identical to today's single driver thread. That is what makes the
migration in §13 safe.

---

## 5. Lifecycle routing

| Operation | Routing | Notes |
|---|---|---|
| `create_session` | Coordinator picks a shard (§4.1), then dispatches to that worker, which mints `local` and installs session state | The capability refusal for a package with no session state (`PackageCapabilityError::NoSessionState`, `engine/runtime.rs:1526`) is answered on the worker and travels back typed |
| `generate` (sessionful) | `session.shard` | Acquires the exclusive lease; conflict returns `ExclusiveLeaseConflict` |
| `generate` (sessionless) | Any worker, by the same stateless distribution | No lease, no ownership; see §9 |
| `reset_session` | `session.shard` | Mutating: takes the lease. A reset racing a live turn is a conflict, not a truncation |
| `close_session` | `session.shard` | Mutating: takes the lease. Closing under a live turn is a conflict; the caller cancels first (§6) then closes |
| `fork_session` | Source `session.shard`, and the fork **stays on that shard** | Forced: a fork shares paged-KV pages copy-on-write with its source, and pages live in one worker's `PageTable` |
| `checkpoint_session` | `session.shard`, read-only | Takes no exclusive lease; a checkpoint concurrent with a turn is refused rather than allowed to capture a half-written conversation |
| `restore_session` | Shard chosen for the restored id | Mutating |
| `session_token_count`, `session_prefill_carry` | `session.shard`, read-only | These are `&self` today (`engine/runtime.rs:1742,1788`) and remain read-only queries |

Three routing rules deserve their reasons stated.

**Reset and close take the lease.** They are mutations of the conversation, and
`reset_session`'s own comment describes exactly the state a concurrent turn
would corrupt: *"the id stays usable and everything the conversation accumulated
is gone"* (`engine/runtime.rs:1662-1665`). A reset that lands between a turn's
read of the conversation and its write-back would be silently undone by the
write-back — a lost update of the whole conversation. Refusing is the only answer
that does not lie.

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

### 5.1 Server-side identity is unaffected

The server maps client `X-Session-Id` strings to engine `SessionId`s through
`SessionRegistry`, an `Arc<Mutex<SessionRegistryInner>>` with its own LRU
(`crates/onnx-genai-server/src/session.rs:9-34`,
`routes/completions.rs:1309-1340`). That map is already thread-safe and already
handles the create race explicitly with
`SessionClaim::{Existing, Claimed}` (`session.rs:22-29`). It is untouched:
`SessionId` becoming a newtype is invisible to it because it only stores and
returns the value. Callers therefore continue to see opaque session handles, and
the shard field is an implementation detail they never parse.

---

## 6. Cancellation, errors, and lease release

**Lease release is a `Drop` obligation, never a cleanup path.** The existing
guard already establishes this (`pipeline/workflow.rs:630-634`). Every exit from
a turn — normal completion, typed refusal, `anyhow` error, client disconnect,
explicit cancellation, or panic — unwinds the per-execution state of §3.3 and
therefore releases the lease. This is the reason the lease lives in category 3.3
and the conversation lives in 3.2: the thing that must vanish on failure and the
thing that must survive it are different objects.

**Cancellation is cooperative and worker-local.** A cancellation request routes
to `session.shard` and sets a flag the decode loop observes at a token boundary.
It does not interrupt an in-flight forward pass — there is no safe way to abort a
launched CUDA graph mid-capture or mid-replay, and pretending otherwise is how a
context ends up in an undefined state.

Today cancellation is implicit: the driver detects an abandoned route when
delivering output fails and drops it (`driver.rs:959-1013`), and submission
failure surfaces as `GenerateSubmitError::DriverStopped` (`driver.rs:249-251,455-463`).
Under a worker pool that is not enough, because a cancelled turn must release its
lease promptly or the next turn on that session sees a spurious 409. So
cancellation becomes explicit: a typed command, observed at a token boundary,
whose acknowledgement is the point at which the caller may assume the lease is
free.

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
   and the loop exits (`driver.rs:725-733`) — and it generalizes to `W` channels.
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
— the engine is dropped on the driver thread when `run_engine_driver` returns
(`driver.rs:690-722`) — and the worker model must preserve it deliberately rather
than by accident.

**Worker failure.** A worker that panics or whose backend construction fails is
removed from the routable set. Its sessions are **lost, not migrated** — their
state was category 3.2 and thread-affine, and there is nothing to move. Every
subsequent call naming a session on a dead shard receives a typed error saying
the worker failed, which session was lost, and that the session must be
recreated. Silently recreating it would be a lost conversation reported as
success, which is the same class of failure as the lost update in Decision 3.

Whether the pool degrades to `W-1` or the engine fails as a whole is an operator
policy with a stated default: **degrade**, because a serving process that keeps
`W-1` shards answering is strictly better than one that stops, and the failure is
already visible in the typed errors and in the startup-style capability log.

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
driver's per-request engine path"* (`driver.rs:744-747`). Under the worker model
a session's turn is pinned to a known worker, so its KV sequence is in that
worker's page table and its row can be admitted alongside other sessions' rows in
the same forward pass. A batch may mix rows from different sessions; it may
**never** hold two rows from the same session, which is Decision 3 restated at
the row level and is enforced by the lease rather than by a batching check.

Two consequences that follow, and are not optional:

- **Batch width is per-worker, and total width is `W × rows`.** An operator who
  configures `W` and `max_batch` independently can request more concurrent rows
  than the device can fund. The effective values are resolved together and
  reported once, in the same way `BatchingReport` already clamps a requested
  width to what the decode path can honour (`driver.rs:289-296`).
- **Sessionless requests go to any worker.** They own no state, so stateless
  distribution (§4.1) applies directly, and a sessionless request should prefer
  the shard with the shallowest queue for the same reason a sessionful one
  cannot: nothing pins it.

The interaction with prefix sharing is the real cost, and it is stated rather
than elided. `PrefixCache` is per-worker under §3.2, so two sessions sharing a
long system prompt but landing on different shards each materialize it. At `W`
shards, a shared prefix costs up to `W` copies. This is a genuine regression
against the single-engine arrangement and is why §12.3 makes prefix-cache hit
rate an acceptance metric and not an afterthought.

---

## 10. Removing the `unsafe impl Send`s

Both hand-written impls exist to paper over the gap between "this type is not
provably `Send`" and "we know it never leaves its thread". The worker model
closes that gap structurally, so both are removed rather than re-justified.

**`unsafe impl Send for EngineOwner` (`driver.rs:275-278`) is deleted.** Its
entire safety argument is *"The engine is moved exactly once into the dedicated
driver thread"*. Under §4.3 the backend state is constructed on the worker thread
and never moves, so there is no cross-thread move to justify. The wrapper type
goes with it.

**`unsafe impl Send for Engine` (`engine/model.rs:221-229`) is deleted, and
`Engine` is decomposed.** What remains after the split is:

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

**A second, structural rule lands with it:** *no type that owns a backend handle
may depend on field declaration order for teardown correctness, and no such
dependency may be discharged by a `Drop` impl one level up.* Ownership is
expressed with `Arc` (§3.4, Rule A) so that the refcount, not the source-file
layout, decides release order; thread affinity is expressed with `OwnerThread`
(§3.4, Rule B) so that the assertion, not the reviewer, decides who may release
it. §1.4 is the worked example of what the absence of both rules already costs
at `W = 1`, and the cost is not linear in `W` — it is the difference between one
thread reliably reproducing a bug and `W` threads producing it intermittently.

The three rules are one rule seen from three sides. A `Send` claim asserts a
handle may cross threads; `Arc` ownership asserts a handle outlives its
dependents; `OwnerThread` asserts a handle is touched only by its owner. Each
replaces a comment that a human has to keep true with a fact the compiler or the
runtime keeps true.

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
   (`python/src/lib.rs:173-179`).
2. **It converts Decision 3 into the bug Decision 3 exists to prevent.** Under a
   mutex, a second turn on a busy session *blocks and then runs*. It reads the
   conversation the first turn just replaced, and nothing reports it. That is
   exactly the lost update `SessionLeaseGuard`'s comment names: *"Two turns that
   both read the conversation before either writes it would leave the loser's
   prompt and generation nowhere, and nothing would report that they were lost"*
   (`pipeline/workflow.rs:600-604`). Silent serialization is not a weaker form of
   safety here; it is a different and worse outcome than a 409.
3. **It does not make thread-affine handles correct.** A mutex supplies mutual
   exclusion. cuDNN's handle, a `CudaReservation`, and CUDA graph capture require
   *thread identity* (§1.3). A `RwLock` is worse still: it would let two readers
   touch a context-bound handle from two threads simultaneously, which the
   handles' own safety comments forbid.
4. **`RwLock` in particular is unsound for this type.** Almost every session
   operation is `&mut self` today — `create_session`, `reset_session`,
   `close_session`, `rewind_session_*`, `restore_session`, `fork_session`
   (`engine/runtime.rs:1501-1724`). A read lock buys nothing for those, and the
   `&self` methods that remain still reach the interpreter's `RefCell`
   (`pipeline/mod.rs:191`), which is `!Sync`.

**The condition for revisiting.** This rejection is contingent, and the condition
is precise: if every handle a session transitively owns proves `Send + Sync`
*with an audited justification of the kind ORT's `Session` already carries*
(`onnx-genai-ort/src/session/mod.rs:1224-1235`) — meaning no bound context, no
owning stream, no thread-local capture state — then a lock-based design becomes
sound, and grounds 3 and 4 fall away. Grounds 1 and 2 would still stand. Today
that condition is not met, and the evidence is in the table in §1.3.

### 11.2 A per-session `Mutex` — rejected

Finer-grained, and fixes ground 1. It does not fix grounds 2, 3, or 4: it still
blocks-then-runs a second turn (losing the update), and it still supplies
exclusion where affinity is required. It also cannot express what a session
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
the calling thread first"* (`cudnn/mod.rs:1129-1132`) — and a design that supplies
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
sessions by calling them alternately. That is what today's test does when it
constructs `ExclusiveLeaseConflict` by hand
(`server/src/tests.rs:4856-4867`), and it proves the mapping, not the
concurrency. Every test below spawns real OS threads and synchronizes on a
`Barrier` so the operations genuinely overlap.

1. **Different sessions run concurrently.** `N` threads, `N` sessions, one
   barrier. Assert all complete, outputs are byte-identical to the same prompts
   run serially, and wall time is materially below `N ×` the serial time — the
   last clause is what distinguishes concurrency from a mutex, and per
   [`../README.md`](../README.md)'s standing rule it is reported with its
   conditions or not at all.
2. **Same session, overlapping turns, exclusive lease → 409.** Two threads, one
   session, barrier-synchronized submit. Assert exactly one succeeds and the
   other returns `PackageCapabilityError::ExclusiveLeaseConflict` naming that
   session, and — this is the part the current test cannot do — assert it over
   HTTP as a real 409 with `kind == "conflict_error"`.
3. **No lost update.** The counterpart to test 2 and the reason it exists. After
   a refused concurrent turn, assert the session's token count equals what the
   *winning* turn left, and that a subsequent turn continues from it. A design
   that silently serialized would pass test 2's status check by queueing and fail
   this one.
4. **Cancellation releases the lease.** Start a turn, cancel it, assert the next
   turn on that session is admitted rather than 409'd, and that the cancelled
   turn's reservations returned to the budget.
5. **Panic containment.** Inject a panic on a worker; assert the lease is
   released, the shard is quarantined, other shards keep serving, and calls
   naming a lost session get the typed worker-failure error rather than a
   silently fresh session.
6. **Shutdown under load.** Shut down with turns in flight on every worker;
   assert every worker drops its own backend state on its own thread (asserted by
   recording the dropping thread id), that join follows drop, and that the
   process exits without a CUDA teardown error.
7. **Routing is total and stable.** Property test: for any sequence of
   create/close, every operation on a live session reaches the shard encoded in
   its id, and no session is ever observed on two shards.
8. **Admission does not convoy.** `AdmissionCeiling::ceiling_bytes` is
   documented as non-blocking (`byte_budget.rs:113-118`); assert under `W`-way
   concurrent admission that no caller blocks on it, since the worker pool is
   what turns that comment into a load-bearing property.
9. **Shard exhaustion refuses.** Fill a shard past its bound and assert
   `create_session` returns the typed refusal naming the shard and the bound,
   rather than over-subscribing.
10. **Session-derived resources keep the session alive.** Build an `IoBinding`
    and a device `Allocator` from a session, allocate a `Value` from that
    allocator, then drop the caller's `Session` handle *first* and assert the
    binding, allocator and value tear down cleanly afterwards. This test fails
    against today's code (§1.4) and passes once §3.4's `Arc<Session>` edge
    exists, which is the point: it is the regression test for the defect, and it
    is written before the fix. A companion test asserts that a struct owning a
    session and a binding tears down correctly under *both* field declaration
    orders, so the property is ownership rather than layout.
11. **Wrong-thread use and teardown are refused, not tolerated.** Construct a
    per-worker handle on thread A, move it to thread B, and assert
    `OwnerThread::assert_owned` fires — once for use and once, separately, for
    drop, because §1.4 was a teardown bug and a use-only check would have missed
    it. Run it against a release build too: §3.4 specifies the check is always
    on, and a test that only proves it under `debug_assertions` proves the wrong
    thing.

Tests 1–5, 7 and 9–11 must run on CPU without a model where possible, so they
gate every PR rather than only CUDA runs. Tests 6 and 8 need the real backend.

### 12.2 ORT / native parity

Concurrency must not change results. The existing parity harness is the
instrument: `tests/parity/README.md` drives
`scripts/check_native_ort_parity.py` against a `profile_native` build, and the
Qwen oracle requires the exact token to match at fixed divergence indices
(`tests/parity/README.md:15-31`).

The gates:

- **Serial-vs-concurrent identity, per backend.** The same prompts through `W=1`
  and through `W>1` produce byte-identical tokens. Determinism must not be a
  casualty of concurrency; per-turn RNG counter state is category 3.3 precisely
  so that it is not.
- **ORT-vs-native identity under concurrency.** The existing parity comparison
  is re-run with `W>1` on both sides and must not loosen its tolerance. If it
  has to loosen, the design is wrong and the document is wrong with it.
- **`W=1` is bit-identical to the pre-migration driver**, which is the gate that
  makes the phased rollout in §13 reversible.

Both backends are selected as CI already selects them:
`cargo test --locked -p onnx-genai-engine --features native-backend --
--test-threads=1` for native, and the ORT-backed package set likewise with
`--test-threads=1` (`.github/workflows/ci.yml:931-950`). Note the irony and
handle it explicitly: the harness pins `--test-threads=1` so that tests do not
contend for the device, which means the concurrency tests in §12.1 must create
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

**Phase 0 — `SessionId` becomes a newtype.** Split `SessionId` from
`SequenceId` (`config.rs:389`, `onnx-genai-kv/src/lib.rs:61`), carrying a shard
field that is always `0`. Behaviour is unchanged; this is Rule 5 cleanup that
happens to be a prerequisite. Callers that treat the id as opaque — including
`SessionRegistry` (`server/src/session.rs:31-34`) — need no logic change.

**Phase 1 — split the state, and fix the lifecycle defect first.** This phase
begins with §1.4, because it is a live bug at `W = 1` and it is far cheaper to
fix before a pool exists than after. Give `IoBinding` and session-derived
`Allocator`s an `Arc<Session>` (§3.4, Rule A), replacing the non-owning
`*const Session` (`binding.rs:12-15`) and the untied allocator
(`allocator.rs:205-209`); land tests 10 and 11 from §12.1 with it; then delete
the compensating manual clears in `WorkflowRuntime::drop`
(`pipeline/mod.rs:270-279`), because leaving a redundant cleanup path next to a
now-structural invariant is exactly the kind of shim RULES.md Rule 3 forbids —
the next reader cannot tell whether it is load-bearing.

The rest of the phase is the state split proper: extract the immutable package
and compiled workflow plan (§3.1) behind `Arc`; separate per-execution turn state
(§3.3) from the interpreter; move `workflow_session_leases` out of `RefCell` and
into per-turn-owned worker state (§4.2); introduce `OwnerThread` (§3.4, Rule B)
on the per-worker handles listed in §4.3, where at `W = 1` every assertion passes
trivially and therefore establishes the baseline rather than chasing a
regression. Still one thread, still one worker. This is the largest refactor and
the one that carries no concurrency risk, which is why it is isolated.

**Phase 2 — introduce the worker as a structure of one.** Build the worker
thread, the routing façade, and the command protocol, with `W` hard-pinned to 1.
The `EngineOwner` wrapper and its `unsafe impl Send` are deleted here
(`driver.rs:246,275-278`), because the engine is now constructed on the worker
thread rather than moved to it. The gate is §12.2's bit-identity requirement
against the pre-migration driver.

**Phase 3 — `W > 1`.** Enable the pool, per-worker backend construction, the
stateless distributor and the shard-encoded ids. `Engine`'s `unsafe impl Send`
(`engine/model.rs:229`) is deleted here. All of §12.1's concurrency tests land in
this phase, and the §12.3 gates are established against Phase 2's numbers.

**Phase 4 — sessionful continuous batching.** Admit sessionful turns into the
per-worker batch loop and delete the deferral comment at `driver.rs:744-747`.
This is the phase that delivers the throughput case in §12.3 and is deliberately
last, because it is the only phase whose failure mode is a correctness bug in
batching rather than in concurrency.

**Compatibility.**

- The public handle type keeps its name and its method signatures; what changes
  is that it becomes `Send + Sync` and its methods take `&self`. Code that
  compiled before compiles after. Code that wrapped it in `Arc<Mutex<_>>` still
  compiles and now merely over-synchronizes, which the release notes must call
  out — that is the one silent-pessimization path this change creates.
- The Python binding's `Mutex<RustEngine>` and its `ENGINE_IN_USE` refusal
  (`python/src/lib.rs:173-186`) are removed in Phase 3. That is a
  *loosening* — calls that previously raised now succeed — so no caller breaks,
  and free-threaded wheels (RULES.md Rule 7, `abi3t`) get real parallelism
  instead of a refusal.
- The C ABI exposes no session lifecycle today
  (`crates/onnx-genai-capi/src/lib.rs:79-423`), so it gains a thread-safety
  guarantee and loses nothing. Its thread-local `oge_last_error`
  (`capi/src/lib.rs:57-75`) is already correct for a multi-threaded caller.
- HTTP behaviour changes in exactly one visible way: a 409 with
  `kind == "conflict_error"` becomes reachable for overlapping turns on one
  session (`routes/mod.rs:523-525`). It is already documented, already typed and
  already marked retryable (`capability.rs:57-61`), so a client that follows the
  existing contract needs no change.
- `copy_on_write` session mutation stays refused at load
  (`workflow_session_continuation.rs:695-702`). Nothing here enables it; it is
  named only so a reader does not assume concurrency implies it.

---

## 14. Open questions

1. **Default `W`.** §4.4 bounds it; it does not pick it. The default should be
   derived from measured device memory per worker, which requires Phase 3's
   numbers.
2. **Sessionless request placement.** §9 says "shallowest queue". Whether that
   reads a per-shard atomic depth or stays pure round-robin should be decided by
   measurement, not by argument.
3. **Shard skew.** §4.1 refuses rather than rebalances. If real workloads skew
   badly, the answer is a better *placement* function, not migration, and the
   evidence for choosing one belongs in [`../genai/SCHEDULING.md`](../genai/SCHEDULING.md).
4. **Cross-shard prefix sharing.** §9 accepts up to `W` copies of a shared
   prefix. A host-side shared prefix store that each worker materializes from is
   possible and is out of scope here; §12.3's measurement is what would justify
   it.
