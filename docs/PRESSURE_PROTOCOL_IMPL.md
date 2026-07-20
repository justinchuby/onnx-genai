# Pressure Protocol Implementation (Phase-1 slice 1a)

Implements the **Ticketed Non-Blocking Pressure Protocol** (`HostGovernor`,
`docs/MEMORY_ARCHITECTURE.md` §5.3.1/§5.3.2) plus a reusable, protocol-agnostic
conformance-trace framework. Conforms to `specs/tla/PressureProtocol.tla` and the
`specs/tla/REFINEMENT.md` contract, **revision 2**.

This document satisfies the **Mapping gate** (REFINEMENT.md §Verification Gates #2):
it names each concrete linearization point in code.

## Module layout

### Deliverable A — `crates/onnx-runtime-protocol-trace` (new crate, `publish = false`)

Protocol-agnostic. Reused by future distributed-runtime slices (Communicator
`BufferOwnership` / `CollectiveOrdering`) without a revision bump.

| File | Contents |
|------|----------|
| `src/lib.rs` | Module wiring; `pub const CONTRACT_REVISION: u32 = 2;`; re-exports. Checker harness gated `#[cfg(any(test, feature = "conformance"))]`. |
| `src/ids.rs` | `id_newtype!` macro + stable identity newtypes: `ProtocolSourceId`, `PressureRequestId`, `PressureGeneration` (`.next()`), `PhysicalAllocationId` (process-unique, never reused → ABA prevention), `LocalDeviceId`. |
| `src/event.rs` | `ProtocolTraceEvent` envelope (exactly per REFINEMENT §Required Trace Envelope); `ProtocolEvent` (`#[non_exhaustive]`, extensible variant grouping: `Pressure` / `BufferOwnership` / `CollectiveOrdering`); `PressureEvent`; `ProtocolTraceSink` trait; `NullTraceSink`. |
| `src/checker.rs` | Generic independent-replay harness (test-support): `AbstractProtocol` trait, `ReplayChecker` driver, `BoundedTraceCollector`, `TraceSnapshot`, failure taxonomy. |

The **envelope + identity types are normal public API**; only the checker harness
is feature-gated so downstream production code links the envelope cheaply while
test crates opt into `features = ["conformance"]`.

### Deliverable B — `crates/onnx-genai-scheduler` (extended, not rewritten)

| File | Change |
|------|--------|
| `src/pressure.rs` | **New.** Full `HostGovernor`, `Ledger`, `PressureTicket`, `CancelMailbox`, arbitration, trace emission, `snapshot()`. |
| `src/governor.rs` | Added 5 `ResourceError` variants (`InvalidHostRequest`, `HostQuotaDenied`, `HostPressureTimeout`, `HostReconfigurationInvalidated`, `HostLedgerInvariant`). `DeviceGovernor` untouched. |
| `src/lib.rs` | `pub mod pressure;` + re-exports. |
| `tests/pressure_conformance.rs` | **New.** Independent `PressureProtocol` reducer + seeded campaigns + checker self-tests + blocking-wait smoke test. |

## Concurrency substrate

Matches the scheduler's existing **sync / std-thread** model (no tokio, no external RNG):

- **Ledger lock** = `std::sync::Mutex<Ledger>` — the single authoritative state,
  held only for short critical sections.
- **Per-ticket wakeup** = `TicketNotify { Mutex<bool>, Condvar }` — ticket-local,
  **never** the governor lock. Waiters block on the ticket condvar, not the ledger.
- **Cancellation mailbox** = `Mutex<CancelMailbox>` (separate lock) with **one slot
  pre-reserved per request at creation**, so `Drop` pushes a `cancel` losslessly and
  never allocates/drops under backpressure.

**Critical invariant (the whole point):** *no thread ever waits while holding a
governor lock*, and *no caller is woken successfully until capacity is atomically
charged to an owned allocation*. The governor publishes `Granted` under the ledger
lock, releases the lock, and only then wakes the ticket — no lost wakeups, no
lock-held-while-waiting.

## Linearization-point → concrete-code table (Mapping gate)

Every point emits a `ProtocolTraceEvent` under the ledger lock via `emit_locked`
(`pressure.rs:982`), which assigns a monotonic `source_sequence`. Trace emission is
**not** a second state owner — the ledger remains authoritative.

| Abstract action (`PressureProtocol.tla` / REFINEMENT §Linearization Map) | Concrete point | Code location |
|--------------------------|----------------|---------------|
| **Submit** | insert `Pending` under ledger lock, emit with id/gen/owner/checked-extent | `request_host_pages` → `pressure.rs:477` |
| **Grant** | atomic charge + `Pending → Granted` **before** wakeup | `arbitrate_locked` → `pressure.rs:907` |
| **Claim** | atomic take + disarm cancellation (at-most-once) | `try_claim` → `pressure.rs:953` |
| **CancelPending** | cancel a still-pending ticket | `apply_cancel_locked` → `pressure.rs:810` |
| **CancelGranted** | cancel-releases the exact granted allocation | `apply_cancel_locked` → `pressure.rs:828` |
| **TimeoutPending** | pending deadline expiry | `time_out` → `pressure.rs:666` |
| **TimeoutGranted** | granted deadline expiry, returns exact charge to free | `time_out` → `pressure.rs:683` |
| **Reconfigure** | bump `PressureGeneration`, revalidate-or-fail prior-gen pendings | `reconfigure` → `pressure.rs:620` |
| **Reclaim** | per-device reclaimable credit commit | `reclaim` → `pressure.rs:575` |
| **Complete** (Release) | claimed allocation returned to free | `release_host_pages` → `pressure.rs:542` |

`try_claim` at `pressure.rs:1031` is the public `PressureTicket` poll entry that
delegates to the governor's `try_claim`.

Ledger mutations use checked arithmetic (`checked_add`/`checked_sub`); any
overflow / negative-headroom / duplicate-identity / snapshot-mismatch is a hard
`ResourceError::HostLedgerInvariant`.

## Invariants enforced

Mirrors the TLA+ invariants (`CapacityConserved`, `GrantedIsChargedExactly`,
`ClaimedIsOwnedExactly`, `ClaimedAtMostOnce`, `TerminalHasNoAllocation`,
`PendingUsesCurrentGeneration`):

- `free = capacity − fixed_charge − reclaimable − reserved − claimed` (checked).
- Granted/Claimed extents are charged exactly once; a ticket yields a grant
  at most once (disarm on claim).
- Terminal states (`Cancelled` / `Failed`) hold no allocation.
- Pending entries always carry the current generation.
- `snapshot()` (`pressure.rs:705`) **independently recomputes**
  `host_ram_used = reclaimable + reserved + claimed + fixed` from authoritative
  entries and asserts it equals `capacity − free` and `≤ capacity` — the
  **Invariant gate**.

## Arbitration (deterministic, bounded aging)

`arbitrate_locked` ages current-generation pending entries (bounded by
`AGING_CAP = 8`), sorts by `(effective_priority DESC, submit_seq ASC)`, then
greedily **reserves + publishes `Granted` before** returning the notify handles to
wake. Because `free` is decremented while the lock is held, a fresh request cannot
steal bytes already reserved for an older ticket.

## Verification gates: status

| Gate (REFINEMENT.md §Verification Gates) | Status |
|------|--------|
| 1. **Model gate** (exhaustive TLC on `.cfg`) | **Deferred to CI.** No `tla2tools.jar` present and `TLA2TOOLS_JAR` unset in this environment; per task constraints TLA/Java tooling must not be installed. `specs/tla/check.sh` must run in CI. Rust conformance campaigns are the implementation-side gate meanwhile. |
| 2. **Mapping gate** | **Pass** — table above. |
| 3. **Conformance gate** | **Pass** — deterministic seeded campaigns emit lossless traces accepted by the independent `ReplayChecker`; invariants re-checked after every transition; leftover active tickets rejected at trace end. |
| 4. **Invariant gate** | **Pass** — `snapshot()` independently recomputes ledger invariants. |
| 5. **Fault gate** | **Pass (pressure subset)** — cancel, timeout, ticket-drop-during-wakeup, mailbox saturation, and reconfigure campaigns reach terminal states without releasing live ownership. |
| 6. **Backend assumption gate** | **N/A for this slice** — no hardware backend; host-page accounting is exercised by the campaigns. |

## Conformance campaigns (tests ARE the deliverable)

`tests/pressure_conformance.rs` contains an **independent** `PressureProtocol`
reducer (does NOT reuse the implementation's transition code — shares only the
identity/event type definitions from Deliverable A) plugged into the generic
`ReplayChecker` harness. Campaigns are deterministic and seeded (`SplitMix64`
PRNG, fixed seeds, reproducible):

- grant vs claim, cancel, timeout, release, reconfigure;
- multiple variable-sized tickets from the same and different devices;
- cancellation-mailbox saturation and ticket drop during wakeup;
- checked-arithmetic boundaries at zero, exact capacity, and maximum extent;
- the §5.3.1 "two devices under pressure" scenario (10GB + 8GB tickets; a 12GB
  reclaim grants one under FIFO/priority; a fresh 12GB request cannot steal
  reserved bytes; a later 6GB reclaim grants the second; a timeout racing a grant
  has one ledger-ordered winner; no leak).

Each campaign feeds its lossless trace to the replay checker, which accepts it,
confirms all invariants after every transition, and rejects leftover active
tickets at the end. Checker self-tests verify it *rejects* unknown
contract-revision, duplicate source-sequence, reordered events, lossy traces, and
leftover active tickets.

## Reuse by the future Communicator slice

The trace framework is deliberately protocol-agnostic:

- `ProtocolEvent` is `#[non_exhaustive]` with an extensible variant grouping.
  Future slices add payloads under the existing `BufferOwnership` /
  `CollectiveOrdering` variants (today empty structs) **without breaking
  revision 2** — the envelope shape and `CONTRACT_REVISION` stay fixed.
- Identity newtypes (`PhysicalAllocationId` process-unique / never-reused for ABA
  prevention, `ProtocolSourceId`, generation) are shared across protocols.
- The `AbstractProtocol` trait + `ReplayChecker` driver are generic over
  `State`/`Action`; the Communicator slice supplies its own independent reducer
  (its own TLA refinement) and reuses the same envelope validation, per-source
  monotonic-sequence/duplicate detection, one-enabled-action resolution,
  after-each invariant evaluation, and crash-boundary handling.
