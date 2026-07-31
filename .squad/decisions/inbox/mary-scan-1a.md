# Decision — Scan single-trip inline dual-path, SLICE 1a (Mary)

**Date:** 2026-07-31 · **Branch:** `feat/27b-scan-capture-1a` (off origin/main) · **Author:** mary
**Status:** committed, NOT PR'd — awaiting Justin's independent review + open/merge.
**Scope:** correctness-only host-execution dual-path. NO capture changes (that is slice 1b).

## What this is
The GREEN-LIT Approach-1 **1a** from the PENDING-JUSTIN root-cause: make a `Scan`
whose **runtime** scan-axis length (`trip_count`) is exactly 1 (a single decode
step) execute its body **once, straight-line**, instead of the generic
`exec_scan` loop — while prefill (`trip_count = prompt_len > 1`) keeps the
unchanged loop. Foundation for 1b (letting that inlined body enter CUDA-graph
capture).

## Mechanism (where the selection happens)
- File: `crates/onnx-runtime-session/src/executor/control_flow.rs`, in `exec_scan`
  (after `trip_count`/axes/slices are resolved, right before the iteration loop).
- Branch: `if self.scan_inline_single_trip_enabled && trip_count == 1 { inline }
  else { existing loop }`. The condition is evaluated at **execution time** on the
  observed `trip_count`, NOT a graph rewrite — this is the whole point: prefill
  and decode **share one InferenceSession/executor/plan**, so a static single-trip
  bake would corrupt prefill. Runtime keying is the correctness guarantee.
- **DRY:** both the loop and the inline path drive the body through one shared
  helper `run_scan_body_step` (run subgraph once → validate output count →
  validate carried-state dtype/shape → split next-state / scan-outputs), and both
  share the identical finishing code (state store + `TensorStackAccumulator::
  finish_scan`). The inline path is therefore **byte-exact with a one-iteration
  loop by construction** — they cannot diverge. No op- or model-name special-casing;
  works for ANY single-trip Scan (num_scan_inputs, axes, directions all honored).

## Flag (default OFF)
- Env: `ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP` — ON only on `1`/`true`/`on`
  (case-insensitive, trimmed). Unset/empty/`0`/unrecognized ⇒ OFF.
- Read once at session build (`scan_inline_single_trip_env_enabled()` in
  `state.rs`), stored as `Executor::scan_inline_single_trip_enabled`.
- **Flag OFF ⇒ zero behavior change**: every trip_count uses the loop; the only
  code delta on that path is that the loop body was factored into
  `run_scan_body_step` (behavior-identical, proven by the tests below).

## Observability (non-vacuity)
- `Executor::scan_inline_single_trip_count` counts every engagement; surfaced as
  `InferenceSession::scan_inline_single_trip_count()` (mirrors `decode_memo_counts`).

## Byte-exact evidence
1. **CPU unit test** (always-on, deterministic) —
   `executor::tests::scan_single_trip_inline_is_byte_exact_and_runtime_keyed`:
   synthetic multi-node Scan body (Add→Mul→Sub, 2 scan inputs, 1 state + 1 scan
   output). Asserts: flag-OFF count==0; flag-ON at trip_count==1 count==1 and
   output **byte-identical** over BOTH outputs vs the loop; and at trip_count==3
   (prefill) flag-ON count stays **0** (runtime-keyed, not static) with output ==
   loop. Mutation-checked non-vacuous: forcing the branch to never engage flips
   the count assertion to FAIL (verified: `left: 0, right: 1`).
2. **CUDA-gated regression test** (own binary so no sibling races the env flag) —
   `tests/cuda_scan_inline_single_trip.rs::
   cuda_scan_single_trip_inline_is_byte_exact_and_runtime_keyed`: same assertions
   on real ORT-CUDA (device 4). PASSED.
3. **On-model 27B** (qwen3.6-27b-int4-cuda, qwen36-conv1d io-overlay, device 4,
   greedy, prompt "The history of computing began", 48 tokens, --steady):
   token id sequences **IDENTICAL** flag-OFF vs flag-ON, covering prefill
   (~790 ms, prompt_len>1) AND 48 single-trip decode steps (48 LinearAttention
   Scans/step):
   `[303,279,220,16,24,19,15,82,440,279,4257,314,279,1118,13934,17943,11,1680,
   430,279,5025,40,1646,11,864,557,5617,303,220,16,24,19,20,13,4081,3988,17943,
   998,3349,11,11064,11,321,2483,4927,13017,13,4213]`. Throughput ~6.1→5.8 tok/s
   (within noise; 1a is host-execution-identical, no capture yet — as expected).
   On-model engagement is proven by the counter in test (2); token-identity here
   is the end-to-end correctness lock.

## Regressions re-run (all PASS, device 4)
- #554 session-reuse recurrent-state reset:
  `native_cuda_reused_session_rezeros_recurrent_state` ✅
- #544 async fence-ordered weight page-in: `cuda_prefetch_war::
  drive_double_buffer_war_safe_across_waves` ✅
- CUDA Scan/Sequence oracle: `cuda_control_flow_safety` ✅
- Full CPU suites: session lib (105) + control_flow (23) + executor (32) ✅

## Files changed
- `executor/control_flow.rs` — runtime dual-path branch + shared
  `run_scan_body_step` helper.
- `executor/state.rs` — flag field + counter field + env parser.
- `executor/build.rs` — field init + `scan_inline_single_trip_count()` accessor.
- `lib.rs` — public `scan_inline_single_trip_count()`.
- `executor/tests.rs` — CPU byte-exact + runtime-keyed test.
- `tests/cuda_scan_inline_single_trip.rs` — CUDA-gated regression (new).

## Contained-slice check
1a stayed contained: NO changes to `provider.rs:plan_capture_region`,
`executor/capture.rs:node_capture_reason`, or any StructuralCaptureDecline logic.
Scan remains structurally declined at the capture seam and runs eager in both
paths — no capture interaction.

## Slice 1b will add (handoff)
- Let the single-trip inlined body **enter CUDA-graph capture** (fold body nodes
  into the parent capture region / grant the trip_count==1 Scan a capture
  exemption). Blast radius: `provider.rs:458` + `executor/capture.rs`.
- Validate captures/replays counters RISE and assert 27B tokens byte-identical to
  the locked reference (the sequence above is the 1a reference).
- 1a already gives 1b a clean, distinct straight-line code path to recognize; the
  `scan_inline_single_trip_count` counter is the engagement tripwire to reuse.
