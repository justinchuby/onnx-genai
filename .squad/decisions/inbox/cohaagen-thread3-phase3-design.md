# Thread-3 Phase 3 — De-latent per-op heterogeneous placement (design + first increment)

Author: Cohaagen (EP/runtime-perf)
Refs: #602 (Phase-1 legalization, merged), #603 (deferred-phase tracking), #594 (single-EP `keep_as_op`), #65 (hetero planner).
Branch: `cohaagen/hetero-phase3-per-op-placement`.

## Problem

`crates/onnx-runtime-session/src/hetero.rs` (`plan` + `execute`) is a complete,
CPU-tested per-op heterogeneous planner/executor, but it has **zero non-test
callers** (`grep` confirms: only `pub mod hetero;` in `lib.rs`). The default
session build path never touches it:

- `SessionBuilder` selects ONE EP (`lib.rs::select_execution_provider`).
- `Executor::build → place_graph` (`executor/build.rs`) does **whole-session**
  CUDA→CPU fallback: `cuda_fallback_report` finds *any* uncovered node and the
  ENTIRE graph is restored to `graph_before_ep_passes` and re-placed on a fresh
  CPU EP (`*graph = graph_before_ep_passes; *ep = auto_detect_cpu_ep()`). A model
  with a handful of CUDA-unsupported ops silently loses CUDA for *everything* —
  a catastrophic perf cliff, emitted only as a `[onnx-genai-warning]`.
- #602's legalization fixpoint + `function_has_attribute_parameters` fail-closed
  guard live inside `hetero::plan`, so they are unreachable from the default path.

## Load-bearing questions

### (a) Does cross-EP tensor movement at the seams already exist? — **YES (host-staged), for the standalone executor only.**

`hetero::execute` (hetero.rs) already realizes cross-EP movement: it walks
partitions in topological order, `extract_subgraph`s each into a standalone
`Graph`, builds a **fresh single-EP `Executor` per partition** on that
partition's provider, and threads boundary tensors through a host-side
`HashMap<ValueId, Tensor>` (`values`). Every cross-partition value is
materialized on the host between partition runs — the "correctness-first
synchronous transfer phase" (module docs §5.2). `plan_transfers` computes the
minimal deduplicated `Transfer{value, from, to}` set (H2D/D2H/D2D), but the
actual data motion in `execute` is the implicit host round-trip via `Tensor`
(host-resident), not device-resident residency/fences (explicitly deferred).

**Caveat that shapes the increment:** this movement exists **only** in the
standalone, stateless `hetero::execute`. It is NOT integrated with the session's
long-lived, stateful `Executor` (KV cache, CUDA-graph capture, decode-memo,
control-flow subgraph execs). `hetero::execute` rebuilds sub-`Executor`s on every
call, so it is sound for one-shot stateless graphs but cannot serve the stateful
decode loop as-is. So from the **default session path's** perspective, integrated
per-op *execution* does **not** yet exist — only planning + a standalone
stateless executor.

### (b) Where does per-node placement hook in, and how does it compose?

Two candidate seams:
- **`hetero::plan`** already owns partition-time placement (per-node
  `assign_nodes` over priority-ordered providers + convex partitioning reusing
  `OrtGraphView::query_capabilities`). No new placement logic is needed.
- **`place_graph`** is the seam where the whole-session fallback happens today,
  and is therefore where the default path must consult the planner. It is
  **CUDA-gated**: `cuda_fallback_report` returns `None` unless
  `ep.device_type() == Cuda`, so the mixed branch is only reachable on CUDA
  builds with partial coverage.

Composition:
- With **#594's single-EP `keep_as_op`** path: untouched. That path never enters
  the mixed CUDA-fallback branch (either the single EP covers the graph, or
  `reject_unsupported_operators` rejects on a terminal CPU EP). The new logic is
  strictly additive inside the `if let Some(report) = …` fallback branch and is
  gated behind an **opt-in env flag** (`ONNX_GENAI_HETERO`, default OFF), so with
  the flag off the fallback is byte-for-byte the current behavior.
- With the **whole-session fallback**: when the flag is off, unchanged. When on,
  the planner runs first; a genuinely homogeneous graph returns
  `SingleProvider` and the existing fallback proceeds; a genuinely mixed graph
  **fails closed** (see increment) rather than silently dropping the whole
  session.

### (c) CPU-testable via fake providers? — **YES.**

`hetero/tests.rs` already drives `plan`/`execute` on CPU via an `AcceleratorEp`
fake provider (host-backed, advertises `DeviceType::Mlx`, restricted op set) and
asserts **byte-identical** output vs a single-EP reference. The new classifier
and fail-closed guard are exercised the same way — no GPU required. The
`place_graph` call site is CUDA-gated so it is not reachable on CPU CI, but the
guard *logic* is extracted into a pure `hetero` function and unit-tested directly
with fake providers (both `SingleProvider→Ok` and `Heterogeneous→Err`).

## First increment (this PR) — planner on the default path + fail-closed scaffold

Integrated stateful per-op **execution** genuinely does not exist yet, and the
task rule is *do not fake movement / never emit wrong bytes*. So this increment
moves per-op **placement** from latent→real on the default path and ships a
sound fail-closed scaffold for **execution**:

1. `hetero::classify_placement(graph, providers) -> PlacementDecision`
   (`SingleProvider(EpId)` | `Heterogeneous(Box<HeterogeneousPlan>)`): runs the
   Phase-1 `plan` and collapses it — homogeneous iff no cross-device transfers
   and ≤1 distinct assigned EP.
2. `hetero::placement_summary(plan, graph, providers)`: actionable per-op
   diagnostic — partition/transfer counts, per-EP node counts with device names,
   and the exact op classes forced onto the non-primary (fallback) EP.
3. `hetero::guard_heterogeneous_fallback(graph, providers, enabled)`: when
   enabled, classify; `SingleProvider ⇒ Ok` (caller's path untouched);
   `Heterogeneous ⇒ Err(SessionError::HeterogeneousExecutionUnsupported{summary})`.
4. `place_graph` mixed branch (behind `ONNX_GENAI_HETERO`, default OFF) calls the
   guard with `[primary(EpId 0), cpu_fallback(EpId 1)]` **before** the
   whole-session fallback. Flag OFF ⇒ guard is a no-op ⇒ existing fallback
   byte-identical.
5. New `SessionError::HeterogeneousExecutionUnsupported` variant (+ capi mapping
   to `OrtErrorCode::Fail`). Message names the CPU-bound ops and states that
   integrated per-op execution is deferred to #603, with the remediation
   (unset the flag for whole-session fallback, or extend CUDA coverage).

### What is implemented vs fail-closed-scaffolded

- **Implemented (real, de-latented):** per-op placement classification on the
  default build path; the standalone `hetero::execute` per-op host-staged
  execution is proven byte-identical to a single-EP reference on CPU
  (classifier composed with real execution in tests).
- **Fail-closed scaffold:** integrated per-op **execution inside the stateful
  session `Executor`** (decode/KV/capture/control-flow). Rather than run the
  whole session on CPU silently, the opt-in path returns an actionable error
  naming the offending ops. No wrong bytes are ever produced.

### Byte-identity of the #594 single-EP default path

- Default `ONNX_GENAI_HETERO` is OFF ⇒ `guard_heterogeneous_fallback` returns
  `Ok(())` on its first line ⇒ zero change to `place_graph`.
- The new code lives **only** inside the `if let Some(report) = …` CUDA-fallback
  branch — unreachable unless `ep.device_type() == Cuda` AND CUDA fails to cover
  the graph AND `!require_cuda`. The zero-config/default path is CPU-only
  (`select_execution_provider`), so it never enters this branch at all.
- Mutation argument: deleting the guard call, or forcing `enabled=false`, leaves
  the existing loader/session tests green and the fallback bytes unchanged; the
  new tests are the only coverage that fails, proving the guard is the sole
  behavioral delta and it is scoped to the opt-in mixed case.

## Deferred (tracked under #603)

- Integrate `HeterogeneousPlan` into the stateful session executor: partition-
  level persistent state/KV residency, CUDA-graph capture per partition, and
  child control-flow hetero plans (the `TODO(hetero-session-phase3)` on
  `HeterogeneousPlan::legalized_graph`).
- Device-resident value residency across seams (skip host round-trips),
  async copies/fences, shape-keyed placement (`M=1` decode vs `M>1` prefill),
  multi-GPU peer copies.
- Phase-2 attribute-parameterized kept-function legalization (already
  fail-closed in #602).
