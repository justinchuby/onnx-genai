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

---

# IMPLEMENTATION (nested child device-graph capture — build + STOP finding)

Built the approved nested-child capture on `feat/27b-scan-capture-1b`. Result: a
**second, deeper STOP** — nested per-node capture-once/replay is **structurally
infeasible with the existing device-graph API** because the graph facility is a
per-EP singleton. The build, evidence, and the safe landing are below.

## Shape-stability finding (STOP-gate #1 cleared — capture-once is viable)
Empirically confirmed (temporary `MARY_SCAN_SHAPE_PROBE`, 27B, removed after):
720 probe lines = **48 nodes × 15 decode steps**; distinct nodes (48) == distinct
(node+shape) combos (48). Every LinearAttention Scan body has ONE stable I/O
shape across ALL decode steps (recurrent state `[1,48,128,128]`; slices
`[1,48,128]×3, [1,48,1], [1,48]`). The recurrent state is fixed-size (O(1)); KV
growth lives only in the 16 GQA layers, not the 48 Scan bodies. So capture-once /
replay-per-step is shape-correct — the child body graph does NOT vary across
steps. (Stored as a repository memory.)

## Nested-capture mechanism (implemented)
Entirely inside the flag-ON, `trip_count==1` inline branch of `exec_scan`
(control_flow.rs) — **no** `plan_capture_region` / `node_capture_reason` /
`collect_segment_kernels` change; the parent Scan stays a structural barrier.
- New flag `ONNX_GENAI_SCAN_BODY_CAPTURE` (default OFF); master gate remains the
  1a `ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP`.
- Per-`CompiledChildPlan` capture slot `ScanBodyCapture { phase, bindings, n_inputs,
  output_specs }` with a state machine `run_body_step`: host-discover geometry →
  allocate persistent `DeviceIoBinding`s (inputs + outputs) → device warmup
  (`run_with_device_bindings`) → `try_capture_with_device_bindings` → replay
  (`replay_device_graph`) thereafter; eager fallback on decline/branch-flip/err.
  Uses ONLY the existing bindings.rs:120-211 API (same one top-level decode uses).
- Observable counters `scan_body_captures/replays/fallbacks` on `ControlFlowStats`
  (+ `ChildExecutorStats`); env trace `ONNX_GENAI_SCAN_BODY_CAPTURE_TRACE`.
- Host-refreshed bindings (each step writes host inputs → replay → reads host
  outputs); NOT the zero-copy device-alias variant — see "why no win" below.

## STOP-gate #2 (the blocker): the device-graph store is a per-EP singleton
`CudaGraphLifecycle` (onnx-runtime-ep-cuda/src/graph.rs:114-122) holds a single
ordered `segments: Vec<CapturedGraph>` **per EP runtime stream**. All 48
LinearAttention child bodies share ONE `Arc<dyn ExecutionProvider>` — each child
`Executor` is built with `self.ep.clone()` (control_flow.rs:131). Therefore:
- `begin_device_graph_capture` **appends** a segment to that one shared list;
- `replay_device_graph` replays the **whole** list (every installed segment);
- `reset_device_graph` clears **all** segments.
There is no per-graph handle. So 48 independent per-node captured graphs cannot
coexist: each node's capture pollutes the shared slot, and whole-graph replay
runs the wrong (last/other-layer) graph. Because all 48 bodies have **identical
shapes**, the mismatch surfaces **no error and no fallback** — it **silently
corrupts** decode output.

### Proof (27B greedy, prompt "The history of computing began", 48 tokens)
Reference (1a) token ids: `[303,279,220,16,24,19,15,82,440,...,13017,13,4213]`.
| Config | flags | tokens | tok/s |
|---|---|---|---|
| A baseline | both OFF | **== reference** ✓ | 6.08 |
| B slice 1a | INLINE=1 | **== reference** ✓ | 5.90 |
| C 1b guarded | INLINE=1, BODY_CAPTURE=1 | **== reference** ✓ | 6.11 |
| C′ unsafe capture | + BODY_CAPTURE_UNSAFE=1 | **DIVERGES** at tok #5, degenerates to repeated `279` | 5.2–5.6 |

The unsafe capture path captured all 48 nodes with **0 fallbacks** (every
LinearAttention body op — Conv/Transpose/elementwise — IS individually
capturable), yet produced garbage — exactly the shared-slot corruption above.

### And it would not win anyway (host-refresh variant)
Even correct, the host-refreshed variant round-trips the recurrent state
(`[1,48,128,128]` ≈ 3 MB) in AND out per node per step: ~6 MB × 48 nodes ≈
**288 MB/step of PCIe traffic**, which dwarfs the per-op dispatch it saves
(C′ measured 5.2–5.6 tok/s, *below* the 6.08 baseline). A net win requires the
**zero-copy device-alias** variant (child I/O bound directly onto the parent's
persistent device buffers, no host round-trip) — which needs a new borrowed
`DeviceIoBinding` constructor + parent-buffer address-stability guarantees.

## Safe landing (what is committed)
The corrupting install/replay is gated behind an explicit, documented spike flag
`ONNX_GENAI_SCAN_BODY_CAPTURE_UNSAFE` (default OFF, "known-incorrect"). The
production flag `ONNX_GENAI_SCAN_BODY_CAPTURE` reaches the body-capture path,
counts a fallback, installs **zero** graphs, and runs the body eagerly →
**byte-exact with 1a**, no regression (6.11 vs 6.08). The full state machine is
retained behind the unsafe flag as the basis for the follow-up.

## Non-vacuous CUDA test
`crates/onnx-runtime-session/tests/cuda_scan_body_capture.rs`
(`cuda_scan_body_capture_flag_is_byte_exact_and_installs_no_shared_graph`):
synthetic multi-node Scan (Add→Mul→Sub, carried state). Asserts flag-ON
body-capture output byte-exact vs BOTH the exec_scan loop and the 1a inline path;
the path actually engages (`fallbacks>0`) but installs `captures==0`/`replays==0`
(the guard invariant); and prefill (trip_count=3) never reaches it. FAILS if a
corrupting capture is re-enabled (output diverges) or if the exemption leaks to
prefill.

## Regressions re-run (device 4): all green
`cuda_scan_body_capture`, `cuda_scan_inline_single_trip` (1a),
`cuda_control_flow_safety`, `control_flow` (23), `cuda_prefetch_war` (#544 WAR),
EP-cuda `standard_attention_capture_gpu` + `rope_capture_gpu` (capture-core
untouched). Session-reuse (#554) exercised by profile_native `--runs 2` and the
test's two-run reuse.

## Handoff — the real follow-up (needs a capture-infra slice, not this one)
To actually land the perf lever, ONE of:
1. **Handle-keyed multi-graph registry in the EP** — extend `CudaGraphLifecycle`
   to hold N named/handled captured graphs (not one shared segment list), and add
   a handle-scoped capture/replay to the bindings API, so each of the 48 child
   bodies owns an isolated graph. (Capture-core change — was explicitly out of
   scope for 1b.) THEN
2. **Zero-copy device-alias child I/O** — bind the child's state/scan I/O onto the
   parent's persistent device buffers (borrowed `DeviceIoBinding`, unit-axis
   reshapes), eliminating the 288 MB/step host round-trip that otherwise negates
   the win.
Both are contained-but-real infra work; per mission I STOPPED rather than broaden
blast radius into the capture core or risk prefill. The 1a dual-path + this
scaffolding (flag, counters, state machine, shape-stability proof) are the hooks
that follow-up will build on.
