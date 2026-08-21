# Decision: MTP M=2 verify-graph capture is a NO-GO on hybrid recurrent targets (+ fix a latent dual-slot graph-replay crash)

**Author:** Gaff (native-CUDA-EP kernel/executor specialist)
**Branch:** `squad/mtp-verify-graph-capture`
**Base:** origin/main @ `4c15a64b3` (PR #1683)
**Scope:** MTP self-speculative decode verify-graph capture (investigation), executor dual-slot replay robustness (fix)

## TL;DR

The flagship goal — CUDA-graph-capture the M=2 MTP verify forward so
self-speculative decode beats greedy — **cannot be shipped under the
non-negotiable token-identity gate** on the hybrid recurrent (Qwen3.5 GDN)
target. The blocker is **upstream of capture**: the multi-row (draft_width ≥ 2)
verify forward already produces **greedy-divergent tokens** on this model when
run **fully eager** (graph off, verify-capture disabled). Capturing a
numerically-wrong verify would only replay the wrong logits faithfully. This is
a rigorous **NO-GO** with a precise root cause and a smallest-viable-design.

Along the way I root-caused and fixed a **real latent crash** in the executor's
device-graph replay path that MTP's dual-slot capture is the first workload to
expose. **That crash fix is the shippable deliverable in this PR.**

## Evidence (decisive)

Model: `qwen38-27b-int4-mtp-cuda` (MTP head as companion export). True greedy
reference = the same `model.onnx` run greedily with the speculator disabled
(`mtp-greedy`). All runs: native CUDA EP, default prompt, greedy argmax.

| Config | index-10 token | acceptance | tok/s | matches greedy? |
|---|---|---|---|---|
| **greedy reference** (`mtp-greedy`) | `440` | — | 63 | — (reference) |
| **MTP draft_width=1** (graph on) | `440` | **100%** | 40.3 | ✅ **48/48 exact** |
| **MTP draft_width=2** (eager) | `11` | 78.9% | 20.9 | ❌ diverges @10 |

- `ONNX_GENAI_SPEC_WIDTH=1` → MTP is **bit-exact greedy** (100% acceptance, no
  partial accepts). `=2` → first multi-token-accept commits a **completely
  different token** (`11` vs `440`), not fp drift.
- The width=2 divergence reproduces with **graph off AND verify-capture
  disabled AND forced snapshot→re-advance** (`ONNX_GENAI_MTP_ALWAYS_READVANCE`) —
  i.e. it is in the **eager verify-row logits**, not in the capture scaffolding,
  the commit/state path, or the full-accept fast path.

Conclusion: on a hybrid recurrent decoder the **verify forward's logits for
rows ≥ 1 are structurally wrong**. The existing oracle test
`native_verify_logits_require_restored_recurrent_state` only validates verify
**row 0** — rows ≥ 1 are unvalidated and are the culprit. Almost certainly the
per-row recurrent/conv (GDN chunked-scan) state or the intra-window causal mask
for the extra verify rows is not being advanced/masked correctly for M ≥ 2.

## Why capture doesn't help (and why it's gated to exactly the broken models)

`configure_verify_capture` is gated on `has_recurrent_state()` **and**
`width ≥ 2` (`crates/onnx-genai-engine/src/native_decode/cuda.rs:1500`). So the
verify-capture slot **only ever engages for hybrid recurrent models** — which
are precisely the models where the M ≥ 2 verify is numerically unsafe. Capture
faithfully replays whatever the eager path computes; it cannot fix wrong logits.

Even setting correctness aside, the verify forward captures **segmented** (~243
segments / 242 eager seams) at M=K: the HF causal-mask arithmetic
(`CumSum`/`Slice`/`Shape`/`GreaterOrEqual`/`Where`), `MatMulNBits` M>1 (cold
Marlin repack + `group_indices` D2H outside the capture contract), and
`SkipSimplifiedLayerNormalization` signature mismatches all force seams — so the
host-launch collapse the task hoped for is not available at M=K without the
same class of work #1673/#1683 tackled at M=1, on top of fixing the numerics.

## Why width=1 is not a fallback win either

draft_width=1 is token-exact (table above) but **40 tok/s < greedy 64.8**,
because the eager verify invalidates the base M=1 decode graph every step
(`cuda_graph: captures=0 replays=0 invalidations=99`). So even the correct
speculation width is a pessimization on this model — plain greedy is faster and
equally correct.

## The incidental crash (fixed here)

**Symptom:** MTP graph-on aborts mid-generation with `cannot replay CUDA graph
because no executable is installed` (reproduces at tokens ≈ 24+, not at 8).

**Root cause:** `evict_surplus_variants`
(`crates/onnx-runtime-session/src/executor/kernel_cache.rs:825-826`) retires
kernels baked into captured graphs and calls
`ep.reset_device_graph_in(Primary)` **and** `(Verify)` directly. That empties
the EP-side graph segments but leaves the executor's **host-side**
`SlotCaptureState` (`device_graph_signature` / `capture_schedule`) live. The
next `replay_device_graph` sees a matching signature, replays an emptied slot,
and hard-errors. This only fires once **both** graph slots are populated — MTP
is the first workload that installs a graph in both the M=1 base and M=K verify
slots, doubling per-node kernel variants past the eviction bound.

**Fix (generic, low-risk):** add a pre-replay liveness check. New EP trait
method `ExecutionProvider::has_device_graph_in(slot)` (default `Ok(true)`, so
non-CUDA EPs are unchanged); the CUDA EP overrides it to report real per-slot
segment liveness (`runtime.has_graph_executable_in`). The executor's
`replay_device_graph` queries it before replay and, if the slot was emptied
out-of-band, resets its host state and returns `Ok(false)` — re-warming and
re-capturing exactly as it already does for a control-flow branch flip, instead
of crashing. Behavior on the greedy / single-slot path is unchanged (the slot
is never emptied out-of-band, so the guard never triggers).

Regression test:
`onnx-runtime-ep-cuda … has_device_graph_in_tracks_out_of_band_slot_eviction`
captures both slots, resets both out-of-band (the exact eviction desync), and
asserts the liveness signal flips to "no executable".

## Smallest viable design (for whoever picks up the perf goal)

1. **Fix the M ≥ 2 verify-row logits first** (prerequisite for any capture).
   Add logit-level instrumentation to compare each verify row ≥ 1 against the
   token greedy would produce from the same accepted prefix, then correct the
   per-row GDN recurrent/conv-state advance and/or intra-window causal mask for
   the extra rows. Extend
   `native_verify_logits_require_restored_recurrent_state` to cover rows ≥ 1.
2. **Only then** capture the M=K verify — and it will still need the
   #1673/#1683-class seam elimination (int4 M>1 GEMM warm-cache + HF mask
   freeze at M=K) to actually collapse host launches.
3. Until (1) lands, the correct behavior for hybrid recurrent MTP is to **not
   run multi-row speculation** (cap draft_width to 1, or fall back to greedy);
   this is a product call and is intentionally **not** changed in this PR to
   avoid a sweeping behavioral change from single-model evidence.

## Gates

- ep-cuda tests: 474 passed / 1 failed — the failure is only the pre-existing
  `a_module_restored_from_cached_ptx_computes_what_a_compiled_one_does`
  (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, fails on all branches). New test passes.
- session `--lib` tests: 190 passed / 0 failed.
- Qwen greedy control (`qwen38-27b-int4-cuda`): tokens stable,
  `cuda_graph enabled=true captures=4 replays=88 fallbacks=0`, no regression
  (bit-identical to origin/main — the fix is inert on the single-slot path).

## Files

Kept (the crash fix + test):
- `crates/onnx-runtime-ep-api/src/provider.rs` — `has_device_graph_in` trait method (default `Ok(true)`).
- `crates/onnx-runtime-ep-cuda/src/provider.rs` — CUDA override + regression test.
- `crates/onnx-runtime-session/src/executor/bindings.rs` — pre-replay liveness guard in `replay_device_graph`.

Reverted (diagnostic-only env toggles, not shipped):
`ONNX_GENAI_DISABLE_VERIFY_CAPTURE`, `ONNX_GENAI_VERIFY_EAGER_ONLY`,
`ONNX_GENAI_MTP_ALWAYS_READVANCE`, `ONNX_GENAI_SPEC_WIDTH`.
