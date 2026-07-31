# Decision: Slice 1b (27B single-trip Scan → CUDA capture) — STOP / OBSTACLE REPORT

**Author:** Mary (native-decode/executor)
**Branch:** feat/27b-scan-capture-1b (based on 1a cbf240af, merged #564)
**Date:** 2026-07-31
**Status:** ⛔ STOP — narrow exemption infeasible without deeper capture-infra + control-flow surgery. No code committed. Requesting Justin re-scope decision.

## TL;DR
The trip_count timing concern (the risk the mission flagged) is **NOT** the blocker: `plan_capture_segments`/`node_capture_reason` run at RUNTIME after shape resolution, so the single-trip `trip_count==1` is known when segmentation happens. A runtime-guarded exemption is expressible there.

The **real** blocker is different and structural: the inlined single-trip Scan body is **host-resident** and runs on a **separate child `Executor` with its own kernel cache**. Admitting it into the parent CUDA-graph capture region — the literal 1b design (exempt Scan in `plan_capture_region`/`node_capture_reason`) — cannot be made correct without cross-cutting surgery across BOTH the capture core AND the control-flow execution core, which the mission explicitly told me to STOP on rather than broaden blast radius / risk prefill correctness.

## Evidence (code-cited, all in crates/onnx-runtime-session/src/)

1. **Body I/O is host-resident (device→host→device round-trip per step).**
   - `executor/control_flow.rs:270` — `value_tensor` materializes control-flow inputs via `self.ep.copy_to_host(&buffer, &mut bytes)` then `Tensor::from_raw` → a **host** tensor. Every Scan input/state is pulled device→host.
   - `tensor.rs:731-733` — `Tensor::as_bytes` asserts host-accessibility (`host_bytes`), so `scan_slice`/`finish_scan` operate on host bytes.
   - `executor/run.rs:296-302` — the child binds its inputs with `self.ep.copy_from_host(tensor.as_bytes(), buf)` → host→device copy back.
   - Net: the body boundary does device→host (input) + host→device (child bind) + device→host (child output) + host stack + host→device (store) **every decode step**. Each is a stream sync. Any such sync inside `cudaStreamBeginCapture` **aborts** graph recording.

2. **The body runs on a SEPARATE executor with a SEPARATE cache.**
   - `executor/control_flow.rs:197-199` — body executes via `self.compiled[cache_index].exec.run_scoped(&inputs, outer_scope, &ExternalBindings::default())`: a distinct `Executor` (`.exec`), driven with **host tensor `inputs`** and **empty device bindings**.

3. **Parent capture cannot enumerate the child's kernels.**
   - `executor/capture.rs:601-612` (`node_capture_reason`): even if I bypass the `plan_capture_region` structural decline (line 576-584) for single-trip Scan, control falls through to a `KernelKey{ node: plan.node_id, shapes }` lookup in **`self.cache.entries`** (the PARENT cache). A Scan node has **no warmed kernel of its own** → returns `KernelNotWarmed` → still a seam.
   - `executor/capture.rs:689-715` (`collect_segment_kernels`): the capture audit gathers one warmed kernel per plan node from the parent cache keyed by parent `node_id`. The child body's kernels live in the child executor's **own** cache under the **child plan's** node_id space — unreachable from here.

4. **Persistent-output precondition.** `executor/capture.rs:630-639` requires every captured graph output to be a persistent device binding; the child is run with `ExternalBindings::default()`, so its outputs are non-persistent host tensors — disqualified.

## Why this is deep surgery, not a narrow slice
A correct parent-fold 1b would require, together:
- (capture core) Make a control-flow node contribute its child sub-plan's warmed kernels to the parent segment audit — cross-executor kernel-cache plumbing in `node_capture_reason` + `collect_segment_kernels` + the `begin_device_graph_capture` audit.
- (control-flow core) A device-resident single-trip inline path: zero-copy unit-axis slice/stack (trivial reshapes at seq-len 1) and child I/O bound as device `ExternalValue` pointers into persistent parent device buffers — replacing `value_tensor` host copies, `scan_slice`/`finish_scan` host byte ops, and the child's `copy_from_host` input bind. This is exactly the shared `exec_scan`/child machinery that prefill (trip_count>1) also uses → **direct prefill-correctness risk**.
- (record loop) `dispatch.rs run_plan_segmented` RunMode::Capture must run the exempted Scan's child device-resident on the shared capture stream, and warm the child before capture.

Each of the three cores is non-trivial; the control-flow-core change touches the code path prefill depends on. This is the "deeper capture-infra surgery / risk prefill correctness" the mission said to STOP on.

## Recommended re-scope (for Justin) — a more contained, prefill-safe path
Rather than parent-fold (exempt Scan in the parent's `plan_capture_region`), pursue **nested child device-graph capture**, which reuses existing, already-tested machinery and is prefill-safe by construction:
- The child `Executor` ALREADY has `try_capture_with_device_bindings` / `replay_device_graph` / `prepare_external_bindings` (bindings.rs:120-200) — the same API the top-level decode loop uses (lib.rs:1414-1442).
- Keep the parent Scan as an eager **seam** (no `plan_capture_region` exemption, no cross-executor kernel plumbing). INSIDE the flag-ON single-trip inline path, drive the child through capture-once/replay-per-step of **its own** device graph via `DeviceIoBinding`s aliasing the parent's scan-input/state/output device buffers (unit-axis reshapes, zero-copy).
- **Prefill safety is automatic:** the 1a runtime gate already routes only `trip_count==1` to the inline path; prefill (trip_count>1) uses the untouched loop, which never captures. No shared-plan seq=1 replay is possible.
- Remaining real work (still substantial, but contained + low prefill risk): eliminate the `value_tensor` host round-trip for these bindings (device views), per-Scan-node capture-state bookkeeping across the 48 LinearAttention Scans, warmup ordering, and retirement on shape change. Partial capture acceptable.

This still delivers the perf lever (body kernels replay from a device graph, killing per-op host dispatch inside the body) without touching the parent capture core or the prefill path.

## What 1a already gives us (unchanged, merged #564)
- Env flag `ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP` (default OFF), counter `scan_inline_single_trip_count`, runtime dual-path in `exec_scan`, shared `run_scan_body_step`. These are the correct hooks for either 1b approach.

## Ask
Confirm re-scope to **nested child device-graph capture** (contained, prefill-safe) vs. committing to the full parent-fold cross-core surgery (larger, prefill-risk). I will build the chosen path in a fresh slice. No 1b code committed pending this decision.
