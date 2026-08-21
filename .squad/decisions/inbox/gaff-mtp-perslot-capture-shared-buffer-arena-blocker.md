### 2026-08-21: Per-slot executor host capture state landed; verify replays>0 blocked by SHARED interior buffer arena

**By:** Gaff

**What:**
Made the `Executor`'s host-side CUDA-graph capture state per-slot so the M=1 `Primary` decode
graph and the M=k+1 `Verify` speculative-verify graph can coexist on ONE executor (branch
`squad/mtp-perslot-capture` off `6923a016b`, PR to follow). Concretely:
- `DeviceGraphSlot::{COUNT=2, index()}` in ep-api (Primary=0, Verify=1).
- Extracted the 8 capture fields (`device_graph_signature`, `capture_schedule`,
  `capture_segmentation`, `capture_cf_shapes`, `capture_warm_signature`, `capture_warm_shapes`,
  `capture_warm_seeded`, `capture_quarantine_ops`) into `SlotCaptureState`; `Executor` now holds
  `slot_capture: [SlotCaptureState; 2]` with `cap()`/`cap_mut()` accessors indexed by
  `graph_slot`. `buffer_shapes`/`if_last_predicate`/`control_flow_output_values` stay shared
  per-run scratch (per coordinator scope).
- `set_graph_slot` is now a pure retarget (no longer resets the other slot).
- Dropped the `enable_decode_inline()` gate on `configure_verify_capture`; removed the permanent
  `set_main_exec_graph_slot(Verify)` (main exec stays on Primary for M=1 decode; `run_verify_captured`
  flips to Verify around the verify forward and back to Primary after).
- Removed the dead `set_retain_decode_graph_across_spec`/`retain_decode_graph_across_spec` accessors.
- Added regression test `set_graph_slot_is_non_resetting_and_per_slot_isolated` (greedy-uses-only-Primary
  invariant; Primary marker survives a Verify round-trip, no cross-slot bleed).

**Validation:**
- Engine lib suite `--features native-backend`: **579 passed / 0 failed** (greedy inert, oracle
  `native_verify_logits_require_restored_recurrent_state` green).
- ep-cuda `graph::tests` `--features cuda,cuda-13000`: **8/8** (incl.
  `primary_and_verify_graph_slots_are_independent`).
- Session crate tests green. Release CUDA bench builds.
- GPU (H200 ordinal 5, CUDA_VISIBLE_DEVICES=5, int4 block-32, ORT 1.28 cuda13, build
  `--features bench-native,native-cuda,cuda-13000`, harness `profile_native --steady --tokens 128
  --warmups 3 --runs 1`, median-of-5): MTP clean, **no NaN**, coherent output, **83.3% acceptance**
  (2.67 tok/verify-step), **median 14.67 tok/s**.
  - `cuda_graph:` (Primary) captures=64 replays=0 fallbacks=0 invalidations=195
  - `cuda_graph_verify:` (Verify) captures=0 replays=0 **fallbacks=1** invalidations=0
  No regression vs gated #1652 (~14.6 tok/s); verify machinery now ARMS+DECLINES gracefully
  instead of being inert.

**Why / remaining gap (GPU-proven):**
Per-slot host capture state is **necessary but NOT sufficient** for verify replays>0 on this
sibling-less GDN artifact. The Verify capture DECLINES because the interior device
`buffers`/`buffer_shapes` arena is SHARED (per-run scratch, deliberately not per-slot). The M=1
decode JIT-sizes interior values to `[1,1]`; during the M=2 verify capture pass the interior
buffer for `model/Slice_node_10` is still `[1,1]`, so the kernel errors
`output shape [1,2], expected [1,1]` → capture declines → graceful eager fallback (correct, no replay).
Even if capture succeeded, the next M=1 decode would resize those shared interior buffers,
clobbering the M=2 graph's baked interior pointers.

**Exact next lever:** give the M=k+1 verify its OWN device buffer arena (a verify-dedicated
executor / buffer arena, analogous to the decode-inline sibling but for M=K) so its interior
JIT-sized scratch is independent of the interleaved M=1 decode. That is the true precondition for
verify replays>0 on this artifact — a larger architectural change than the per-slot host-state
refactor.

**Blocker B (separate/pre-existing):** Primary M=1 decode never replays in MTP mode
(`replays=0 invalidations=195`) even before this change — the interleaved
`commit_recurrent_state_to_accepted` (snapshot→restore→re-advance accepted tokens, variable width
M=1 or 2) runs on the Primary slot sharing the same persistent decode bindings, resizing them and
invalidating the M=1 graph each spec step.

**Honest status:** No net speedup yet (14.67 tok/s MTP vs ~56 greedy). The per-slot foundation is
done, correct, tested, and greedy byte-identical; replays>0 is blocked by the shared interior
buffer arena + blocker B, both requiring the verify-dedicated-arena/executor lever.
