# Communicator + BufferOwnership implementation (distributed-runtime slice 1b)

This document describes the slice-1b implementation of the distributed-runtime
communicator abstraction and the backend buffer-ownership lease registry. It is
the implementation companion to:

- `docs/DISTRIBUTED_RUNTIME.md` — §3 Communicator Abstraction (§3.1 core trait),
  §4 backends (§4.6 in-process), §11 Phased Implementation (Phase 1).
- `docs/MEMORY_ARCHITECTURE.md` — BufferOwnership / lease model.
- `specs/tla/BufferOwnership.tla` (+ `.cfg`) — the **normative** buffer-ownership
  state machine and its invariants.
- `specs/tla/REFINEMENT.md` — the ProtocolTraceEvent envelope
  (`contract_revision = 2`), the Linearization Map, the independent
  replay-checker requirement, and the verification gates.

The crate is `crates/onnx-runtime-comm` (`publish = false`, workspace member). It
**reuses** the slice-1a trace-conformance framework in
`crates/onnx-runtime-protocol-trace` — the `ProtocolTraceEvent` envelope, the
identity newtypes, and the generic `ReplayChecker` / `AbstractProtocol` harness —
rather than forking a second trace format.

---

## 1. Scope of slice 1b

**In scope (this slice):**

- The `Communicator` trait shape (the full §3.1 method set).
- `InProcessCommunicator`: the Phase-1 single-process, multi-rank reference
  backend and test oracle. Its **plumbing** — identity/topology, point-to-point
  `send`/`recv` via per-instance mailboxes, a cross-thread `barrier`, `abort`,
  and buffer-ownership registry integration.
- `OwnershipRegistry`: the backend buffer-ownership lease registry, refined
  against `BufferOwnership.tla`, emitting contract-revision trace events at each
  linearization point.
- The independent replay checker + seeded conformance campaigns for buffer
  ownership.

**Deferred to slice 1c (explicitly NOT built here):**

- Collective **algorithms**: reduction math (`all_reduce`, `reduce_scatter`),
  data movement and variable-size permutation (`all_gather`, `broadcast`,
  `all_to_all`, `exchange_counts`, `all_to_all_v`).
- The collective-ordering submit sequencer and `CollectiveOrdering.tla`
  conformance (`FrozenPlan`, admission/skip ordering).
- Block-quantized wire codec sizing / reduction semantics.

Every collective method on `InProcessCommunicator` returns
`CommError::CollectiveDeferred(name)` so the trait is usable and total while the
algorithms land in 1c. The trace crate's `CollectiveOrderingEvent` variant is
left empty (reserved) for 1c; only `BufferOwnershipEvent` is filled here.

---

## 2. The `Communicator` trait (§3.1)

Defined in `crates/onnx-runtime-comm/src/communicator.rs`. `Send + Sync`, async
via `async-trait`. Method groups:

- **Identity:** `rank`, `group_id`, `members`, `group_size`, `backend_name`.
- **Collectives (algorithms deferred to 1c):** `all_reduce`, `all_to_all`,
  `exchange_counts`, `all_to_all_v`, `all_gather`, `broadcast`, `reduce_scatter`.
- **Point-to-point (1b plumbing):** `send`, `recv`.
- **Synchronization (1b plumbing):** `barrier`, `abort`.

Data-transfer methods return a `CommHandle` — an asynchronous, rank-local
completion fence. **Handle-lifetime contract:** the caller must not reuse, free,
or mutate input buffers until the handle is terminal; the backend registry
retains buffer leases until terminal, so dropping the Rust handle cannot stop
progress or release storage. Dropping a handle *detaches* the caller; it never
cancels a collective. Abort is an explicit communicator-wide operation.

### `InProcessCommunicator` (§4.6, Phase 1)

Defined in `crates/onnx-runtime-comm/src/inprocess.rs`. All "ranks" live in one
process; each rank holds its own handle over shared state, and is expected to run
on its own thread (a logical device). It is the reference backend and test oracle
for real transports (NCCL/Gloo/Thunderbolt) — trait shape and correctness matter,
not throughput.

- Constructors: `world(n)`, `world_with_registry(n, registry)`,
  `world_traced(n, source, epoch, sink)`. Ranks are `0..n` in frozen order; all
  handles share **one** buffer-ownership registry.
- `send`: acquires a **read** lease (registry `Submit`), copies into the
  destination's mailbox slot keyed `(instance, src_pos, dst_pos)`, then
  `complete_success` releases the lease. Trace: `Submit -> CompleteSuccess`.
- `recv`: peeks the mailbox first (returns `NoPendingMessage` if empty, before
  acquiring a lease), then acquires a **write** lease, copies, and
  `complete_success`. Ordering across a missing message is the caller's /
  sequencer's responsibility (the in-process backend does not queue — that is
  1c).
- `barrier`: a cross-thread `Condvar`; the local handle is terminal only after
  every rank has entered the same barrier instance.
- `abort`: idempotent; records the first cause and fails every subsequent op.

In-process operations complete synchronously (~0 latency), so `CommHandle` is
immediately ready. The crate ships a minimal `block_on` (using the stable
`Waker::noop()`); the crate is `#![forbid(unsafe_code)]`.

---

## 3. BufferOwnership lease registry

Defined in `crates/onnx-runtime-comm/src/registry.rs`, refined against
`specs/tla/BufferOwnership.tla`.

### 3.1 The invariant

The backend registry is the single owner of every transport-held allocation
lease. A buffer has exactly one owner at a time; a **write** lease is exclusive
(conflicts with every other read or write lease on that allocation); **read**
leases may alias; there is no use-after-release and no double-free; ownership
transfer (`Detach`) is atomic and never releases a lease.

The lease model is generalized from the TLA single-read/single-write buffer to
**lease sets** (`CommLeaseSet { reads, writes }`) so a real operation with
multiple inputs/outputs is expressible. The conflict rule:

- read/read on the same allocation → allowed (aliasing);
- a new **write** conflicts with any active read-or-write lease on that
  allocation;
- a new **read** conflicts only with an active **write** lease;
- an in-place op lists its allocation in both `reads` and `writes` → held
  exclusively.

### 3.2 Discipline (mirrors `onnx-genai-scheduler::pressure`)

- A single authoritative `Mutex<RegistryState>` guards all registry state.
- Every linearization point emits exactly one `ProtocolTraceEvent` at the true
  atomic commit point, under the registry lock, via `emit_locked`.
- The registry never blocks/waits while holding its lock: `submit` either
  succeeds immediately (buffers available) or fails fast with a conflict/free
  error. Any *waiting* belongs to the 1c collective-ordering sequencer.
- Terminal operations hold no lease and are reaped immediately, so recomputation
  stays O(live).

### 3.3 ABA / generation

`PhysicalAllocationId` is process-unique and never reused, discharging the
ABA-prevention obligation (`REFINEMENT.md` § "Buffer ownership"). Each written
allocation advances a reuse generation on `CompleteSuccess`/`QuiesceAbort`;
because writes are exclusive, no other active operation ever observes a written
buffer's generation change — the `ActiveGenerationsMatch` invariant holds.

---

## 4. Linearization-point table (TLA action → code)

Every action commits atomically under the registry lock and emits its event in
the same critical section (`REFINEMENT.md` Linearization Map). All methods are in
`crates/onnx-runtime-comm/src/registry.rs`.

| `BufferOwnership.tla` action | Trace event | Method / linearization point |
| --- | --- | --- |
| `Submit(op)` | `BufferOwnershipEvent::Submit` | `OwnershipRegistry::submit` — after the conflict/free check succeeds, the op + its lease sets are inserted and `Submit` is emitted in the same locked section. Returns an `OperationLease`. |
| `Detach(op)` | `BufferOwnershipEvent::Detach` | `OwnershipRegistry::detach` (also `OperationLease::drop`) — `handle_attached` flips to `false`; ownership and both leases are unchanged. No-op if already terminal/detached. |
| `BeginAbort(op)` | `BufferOwnershipEvent::BeginAbort` | `OwnershipRegistry::begin_abort` — `submitted -> aborting`; all leases remain owned. |
| `CompleteSuccess(op)` | `BufferOwnershipEvent::CompleteSuccess` | `OwnershipRegistry::complete_success` → `retire(.., Submitted, ok)` — releases every lease, advances each written allocation's generation, reaps the terminal entry, all atomically. |
| `QuiesceAbort(op)` | `BufferOwnershipEvent::QuiesceAbort` | `OwnershipRegistry::quiesce_abort` → `retire(.., Aborting, error)` — same release/reap path for an aborting op. |
| `FreeBuffer(b)` | `BufferOwnershipEvent::FreeBuffer` | `OwnershipRegistry::free_buffer` — the allocation is added to `freed` only after no active lease references it; emitted in the same locked section. |

Envelope: every event carries `contract_revision = 2`, the `topology_epoch`, the
`ProtocolSourceId`, and a monotonic `source_sequence` (`emit_locked`). Extending
`ProtocolEvent` with the buffer-ownership variants is purely additive on a
`#[non_exhaustive]` enum, so the contract revision stays **2**.

---

## 5. Conformance (independent replay checker + campaigns)

Defined in `crates/onnx-runtime-comm/tests/buffer_ownership_conformance.rs`. The
checker reuses the generic `ReplayChecker` / `AbstractProtocol` harness from the
trace crate.

### 5.1 Independent replay checker

`BufferOwnershipProtocol` is an independent `AbstractProtocol` reducer: it
re-derives the buffer-ownership abstract state (per-op state, lease sets,
handle-attached, registry-owned, outcomes, per-buffer generations, freed set)
**from the trace events only** — it never calls the implementation's transition
functions. After every transition it checks the `BufferOwnership.tla` invariants:

- `NoConflictingActiveLeases` (generalized to lease sets),
- `ActiveIsRegistryOwned` / `DetachedActiveIsStillOwned`,
- `FreedHasNoLease`,
- `TerminalReleased`,
- `ActiveGenerationsMatch`.

The envelope gate additionally enforces `contract_revision`, per-source
monotonic `source_sequence` (no duplicate / reorder / loss), and no
leftover-active operation at end of trace.

### 5.2 Campaigns (seeded, reproducible)

- `campaign_acquire_transfer_release` — Submit → Detach → CompleteSuccess.
- `campaign_abort_quiesce` — Submit → BeginAbort → QuiesceAbort.
- `campaign_read_read_alias` — concurrent readers on one allocation.
- `campaign_read_write_conflict_then_serialize` — write rejected under a live
  reader, then serialized after the reader completes.
- `campaign_in_place_is_exclusive` — in-place (read+write same alloc) is held
  exclusively.
- `campaign_allocator_free_boundaries` — free-of-leased rejected; free after
  release accepted; double-free rejected; submit-after-free rejected.
- `campaign_generation_advances_on_write` — writes advance the reuse generation
  without breaking any active op's `ActiveGenerationsMatch`.
- `campaign_cross_rank_send_recv` — drives the **real** `InProcessCommunicator`
  across ranks via `block_on`, checking the emitted trace and the registry
  snapshot invariant gate on every step.

### 5.3 Checker self-tests (rejection)

`checker_rejects_unknown_contract_revision`, `..._duplicate_source_sequence`,
`..._reordered_event`, `..._lossy_trace`, `..._leftover_active_operation`,
`..._free_of_leased_allocation`.

### 5.4 Determinism

`campaign_seeded_is_reproducible`: fixed seeds produce byte-identical traces
across runs (verified by comparing serialized event streams).

### 5.5 TLC gate

The TLC model gate (`BufferOwnership.tla` / `.cfg`) may be CI-deferred (no Java
on host). The Rust conformance campaigns are the impl-side gate, identical in
role to the slice-1a pressure conformance.

---

## 6. Test results

- `cargo test -p onnx-runtime-comm`: **18 unit + 15 conformance** passing.
- `cargo test -p onnx-runtime-protocol-trace`: passing.
- `cargo test -p onnx-genai-scheduler`: **33 unit + 15 conformance** passing
  (unchanged after the slice-1a `pressure.rs` follow-ups below).
- 0 warnings and clippy-clean on all touched files.

---

## 7. Slice-1a `pressure.rs` follow-ups folded in

Two non-blocking review follow-ups on the slice-1a `HostGovernor` pressure
protocol (`crates/onnx-genai-scheduler/src/pressure.rs`), file-disjoint from the
1b work:

1. **Reap terminal ledger entries on cancel-drop.** `apply_cancel_locked` now
   distinguishes a live `Claimed` entry (grant-wins, keep) from an
   already-terminal `Failed`/`Cancelled` entry. A cancel-drop observing a
   terminal entry reaps it (`ReapTerminal`) so terminal entries no longer linger
   unbounded — restoring the O(live) intent. Reaping emits no new event (the
   terminal transition was already published), so traces are unchanged.
2. **Hoist cancel-granted wakeups out of the ledger critical section.**
   `drain_cancellations_locked` / `apply_cancel_locked` now **return**
   `Vec<Arc<TicketNotify>>` instead of waking under the lock; every caller merges
   the drained wakeups into its post-unlock wake list and wakes via the shared
   `wake_all` helper after dropping the ledger lock. This makes all cancellation
   wakeups follow the same wake-after-unlock discipline as the rest of the
   protocol. All slice-1a tests (33 unit + 15 conformance) remain green.
