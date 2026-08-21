### 2026-08-21: MTP verify-capture infra landed capture-safe but gated inert — single-executor two-slot contention is the real remaining gap

**By:** Gaff

**What:**
Landed the full option-(c) fixed-M verify-capture machinery (steps 1-4 of the #1650 plan) on branch `squad/mtp-verify-wire` off origin/main `401f2084c`, but GATED it to only arm when a decode-inline sibling exists — so it is inert-and-correct on the current q38 MTP artifact.

- Executor/session (steps 1-3): `pin_step_workspace` flag + pin-aware `release_step_workspace` (the #1647 stale-ptr NaN fix); Session pin setters; Verify-slot per-op routing already in from #1650.
- cuda.rs (step 4): 8 `DecodeCudaState` verify fields; `configure_verify_capture` (allocates persistent padded M=k+1 bindings widened from each live binding's own physical shape — NO hardcoded dims — pins the workspace, routes main exec to the Verify slot); `run_verify_captured` + `run_verify_graph_phase` (NeedsWarmup→Armed→Ready state machine); `swap_verify_bindings` (mem::swap in/out around the verify); graceful self-disabling replay: `verify_phase_after_invalidation` re-warms once then latches to `Unsupported` (permanent eager) so a clobber never becomes per-step recapture churn; module-level `widen_query_last`/`widen_query_seq` pure helpers.
- Verify-slot counters surfaced via `CudaGraphDebugStats` + a `cuda_graph_verify:` line in profile_native.
- 4 new unit tests (`verify_capture_helper_tests`): widen helpers (unit-axis-only widening), phase-latching, and the greedy-inertness accessor contract.

**The gate:** `configure_verify_capture` is behind `self.session.enable_decode_inline()?`. Verify-into-Verify-slot capture is only sound when the interleaved M=1 base decode + commit re-advance run on a DIFFERENT executor+slot (the sibling's Primary). On a model whose recurrent Scan is single-trip-inlineable (a decode-inline sibling exists), the feature arms; on this artifact it stays dormant.

**Validation (GPU H200 ordinal 5, ORT 1.28 cuda13, build `--release --features bench-native,native-cuda,cuda-13000`, artifact `/home/justinchu/qwen38-27b-int4-mtp-cuda`, `profile_native --steady --tokens 128 --warmups 3`, median-of-5):**
- MTP throughput median-of-5 = **14.63 tok/s** (14.61–14.72), acceptance **83.3%**, tokens_per_verify_step=2.67, no NaN, coherent output.
- `cuda_graph: captures=64 replays=0 fallbacks=0 invalidations=515` (pre-existing MTP behavior: the M=1 graph is invalidated every verify step).
- `cuda_graph_verify: captures=0 replays=0 fallbacks=0 invalidations=0` (verify capture INERT — gate off, correct on this artifact).
- Engine lib suite `--features native-backend`: **579 passed / 0 failed** (575 baseline + 4 new; greedy inert, no regression).
- ep-cuda `graph::tests`: **8/8** pass (incl. `primary_and_verify_graph_slots_are_independent`).

**Why:**
GPU-confirmed root cause of replays=0: this artifact has NO decode-inline sibling — `enable_decode_inline()` is false because the GDN recurrent Scan is not single-trip-inlineable, so `route_decode_inline` is always false. Therefore the M=1 base decode AND the commit re-advance run through `run_one_token` on the SAME main executor as the M=2 verify. The executor keeps a SINGLE host capture signature (`device_graph_signature` + `capture_warm_*`/schedule/cf_shapes), so when verify captures an M=2 graph into the Verify slot, the very next M=1 decode captures an M=1 graph into that same per-executor signature between the verify's capture and its replay → the replay signature-mismatches (M=1 stored vs M=2 presented) and `replay_device_graph` returns Err. Previously this crashed; the gate + graceful degradation make it inert and safe.

**The precise remaining gap (THE next lever for the campaign number):** per-slot host capture state on `Executor`. `device_graph_signature`, `capture_schedule`, `capture_segmentation`, `capture_cf_shapes`, `capture_warm_signature`, `capture_warm_shapes`, `capture_warm_seeded`, `capture_quarantine_ops` must be indexed by `graph_slot` (2 slots), and `set_graph_slot` must stop resetting the other slot. `buffer_shapes`/`if_last_predicate` are per-run scratch and do NOT need to be per-slot. This lets the M=1 Primary and M=2 Verify graphs coexist and both replay on one executor. It is interwoven across run.rs (~24 refs), capture.rs, control_flow.rs, dispatch.rs, bindings.rs, mod.rs — a large, high-risk refactor beyond a safe single turn. Greedy only ever uses slot Primary (index 0), so a correct per-slot split keeps greedy byte-identical, but the validation burden is high. Alternatively, a sibling-capable hybrid MTP artifact would let the already-landed gated feature demonstrate replays>0 directly.

**Bottom line:** MTP is correct (token-identical oracle green, 83.3% accept) but cannot yet beat the ~56 tok/s greedy baseline on this artifact because the verify graph never replays (replays=0). No speedup is fabricated. The infra to close it is landed and capture-safe; the executor per-slot host-capture-state refactor (or a sibling-capable artifact) is the gate.
