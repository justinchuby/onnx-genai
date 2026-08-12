### 2026-08-12: Enforce async-copy fence-safety with a type-level completion witness, not a comment

**By:** Copilot (pinned-staging-pool, #837 item 2)

**What:**
When a host buffer is the *source* of an asynchronous H2D copy and is later
reused/freed (e.g. a pinned-staging pool), the reuse must be ordered after the
copy **completes** (not merely enqueues). Enforcing that with a comment is
fragile — the next person to make the copy non-blocking silently reintroduces
intermittent weight corruption. Preferred pattern, adopted in PR #843:

- The copy primitive that host-synchronizes (`CudaRuntime::htod_async_elapsed_ms`,
  which blocks because cudarc `Event::elapsed_ms` calls `end.synchronize()`)
  returns a `CopyCompleted` witness — a zero-sized token whose only field is
  **private to the `runtime` module**, so nothing outside can fabricate one.
- The reuse path (`PinnedStagingPool::release` / `PooledStaging::retire`)
  **consumes** a `CopyCompleted`. Reuse is unreachable without proof the copy
  finished.
- A future switch to a non-blocking `htod_async` + deferred fence produces no
  witness at the reuse site, so the code **fails to compile** until the author
  threads a completion witness through *after* awaiting the fence.

**Why / implementation notes:**
- `Drop` cannot consume an argument, so a `Drop`-based return can never *require*
  the witness. Make the return an explicit method (`retire(self, CopyCompleted)`)
  and demote `Drop` to a **leak-safe fallback that frees** the buffer (never
  returns it to the pool). Forgetting to retire then costs only a re-allocation
  (catch it with a deterministic `pinned_alloc_calls` counter + regression test),
  never silent reuse-in-flight.
- **Never assert in `Drop`** on this codebase (STATUS_STACK_BUFFER_OVERRUN).
- For unit tests that exercise the pool without a real copy, expose a
  `#[cfg(test)] pub(crate)` test-only constructor for the witness so tests stay
  honest about requiring one without letting non-test code forge it.
- The change is compile-time only — no runtime behavior change — so existing
  before/after perf measurements remain valid.
