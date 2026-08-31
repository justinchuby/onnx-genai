# Decision: route-telemetry CONSUMER wires into the real request lifecycle at the single coarse safe boundary — one required EP lifecycle method (no compatibility shim), no second boundary mechanism

- **Slice:** 7C (issue #1810), draft PR (base `main`, includes #1971 `98731d31`)
- **Date:** 2026-08-24
- **Owner:** Copilot coding agent (Slice 7C), requested by Justin Chu
- **Reviewer:** independent review pending (not merged)
- **Status:** proposed (default-OFF, byte-identical when disabled; not merged)

## Context

Slice 7A (PR #1922) shipped the **producer-only** accumulate-only route
telemetry window. Slice 7B (PR #1971, `98731d31`) shipped
`route_residency::consume_route_window_at_boundary(...)` — the boundary-time
**consumer** — but with *no live decode-loop call site* (its module docs:
"wiring it into a running session's request boundary is the next slice"). Slice
7C closes that: make the consumer reachable from the real production
decode/request lifecycle at the single existing coarse safe boundary, while
remaining default-off and byte-identical when disabled.

## Decision

**One boundary, one seam.** The single property-defined boundary with no
capture/replay, live guard, admission, deferred-release, or multi-device
ambiguity is `Executor::finish_device_validation`
(`crates/onnx-runtime-session/src/executor/run.rs`) — "the one request-level
host boundary", which runs `ep.sync()` after stream/graph completion, only for
top-level (`!nested`) runs, once per decode step/request. The consumer is
called there and **nowhere else**. No second boundary mechanism, no model
allowlist.

**EP-agnostic call, EP-owned operation.** The executor cannot name CUDA types,
so the boundary is a **required** trait method
`ExecutionProvider::consume_route_residency_at_boundary(&self) -> Result<()>`.
Per the "no backward-compatibility during development" directive there is **no
compatibility default/shim**: every in-repo `ExecutionProvider` implements it
explicitly (non-residency EPs and mocks return `Ok(())`; forwarding EPs delegate
to their inner EP; the planning-only capability gate is `unreachable!`), so each
provider states its boundary behaviour rather than silently inheriting a no-op.
The success arm of `finish_device_validation` calls it after the latch is
confirmed clean. Non-residency EPs and the CUDA EP with the profile disabled do
nothing here and are byte-identical at runtime.

**CUDA override, reused authorities only.** The CUDA EP override:
1. reads the existing default-off gate
   (`coarse_residency_profile_enabled()` / `COARSE_RESIDENCY_ENABLE_ENV`) first
   — when off (shipped default) it is a single env read: no lock, no snapshot,
   no allocation, no CUDA launch, no host sync, no telemetry reset;
2. looks up one optional installed `RouteResidencyBoundary` binding — `None` in
   production today (honest reachable seam, exactly like 7A/7B shipped), so the
   enabled-but-unbound path is a lock + `None` check;
3. when bound, drives the lawful ordering **once**: fail-closed
   `resize_safe_point` pre-check → producer `route_telemetry_snapshot` → the
   merged #1971 `consume_route_window_at_boundary` → producer
   `reset_route_telemetry_boundary` — the reset (and an expected-epoch advance
   that keeps stale detection honest) fires **only** after a window was
   actually consumed, so an unsafe or disarmed boundary neither snapshots nor
   resets;
4. records the typed outcome in a new `RouteResidencyDiagnostics` surface
   (mirrors `CsaMetrics`) — no silent WholeBank / default-success.

**No new mechanism.** `RouteResidencyBoundary` is *pure binding* — it owns no
allocator and maps nothing; every field is a handle to an existing authority.
`RouteTelemetrySource` (crate-internal) names the producer's two existing
window primitives; a compile-time assertion proves the real `QMoEKernel`
satisfies it, and the GPU tests drive a controllable double (the #1971 test
precedent). PMM/VMM remains the sole map/unmap/account/quarantine/rollback
authority; coarse cadence/hysteresis stay policy-owned; no per-token remap; no
remap during capture/replay.

## Tests (traverse the production caller, never the raw consumer)

- CPU (`executor/tests.rs`, no GPU, mock EP through `Executor::run`):
  - `route_residency_boundary_fires_once_per_top_level_run_after_sync` — exactly
    one call per top-level request, after `sync`, not per kernel.
  - `route_residency_boundary_skips_nested_control_flow_runs` — an `If` subgraph
    run (nested `run_scoped_mode`) executes a kernel but fires **zero**
    boundaries; the whole request fires exactly one.
- GPU (`tests/route_residency_boundary_gpu.rs`, idle A100, all through the EP
  trait method / its phase-8 fault sibling):
  - `boundary_disabled_gate_is_structural_no_op` — gate off with a valid binding
    installed → 0 boundaries, 0 snapshots, 0 resets, allocator bytes + content
    byte-identical.
  - `boundary_applies_group_hot_set_and_advances_window` — ≥3-replay accumulated
    union → atomic two-member expert-group transition → window advanced once →
    next boundary sees an empty window; both members' bytes bit-identical.
  - `boundary_unsafe_point_rejects_before_consume_and_reset` — multi-device and
    an active graph capture each reject before snapshot/reset; allocator + content
    untouched.
  - `boundary_defective_windows_fail_closed` — foreign request (multi-request
    isolation), foreign device, poison, overflow, stale epoch, empty routed set
    each fail closed to whole-bank; nothing tiered.
  - `boundary_injected_fault_rolls_back_through_caller` — a deterministic driver
    fault mid-transition rolls back through the caller; content preserved.

`cargo fmt` clean; `cargo clippy --all-targets -D warnings` clean on the touched
crates (ep-api, ep-cuda incl. `gpu-tests`, session); 150 session executor unit
tests pass; the 6 Slice-7B consume GPU tests and the qmoe route-telemetry GPU
test still pass (no regression).

## Scope / honesty

- **No performance / tok-s claim.** This is a default-off wiring seam; the real
  remaining blocker for a live producer registration (constructing a binding
  from a running session's expert banks) is a later slice.
- Default disabled path is byte-identical and adds no CUDA launch, host sync,
  allocation, mapping, or telemetry reset.
- Avoided PagedAttention / HCA/CSA / Mobius / BQMoE-fusion files. The only
  session caller file touched is `executor/run.rs` (one success-arm line) — no
  concurrent-edit conflict observed.
- **Revision (no backward-compat during development):** the boundary method was
  made a required part of the EP lifecycle contract (the earlier `Ok(())`
  default was a compatibility shim and was removed). All in-repo
  `ExecutionProvider` implementations, mocks, and integration-test doubles were
  updated atomically in the same change; safety, default-off, and the typed
  diagnostics outcome surface are unchanged.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
