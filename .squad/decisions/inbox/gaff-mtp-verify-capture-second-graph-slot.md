### 2026-08-21: Option-c verify capture — land the second CUDA graph slot (enabling primitive); full replay wiring scoped as the remaining multi-file gap

**By:** Gaff

**What:**
Investigated option-c (capture the M=K MTP verify forward so it replays instead of tearing down the M=1 decode graph every step — the `replays=0` root cause from #1647). Confirmed by fresh code investigation that a real replays-on-verify speedup requires a **second, independent captured-graph slot** on the shared CUDA EP, because the EP owns exactly ONE `CudaGraphLifecycle` today and the M=1 decode graph and the M=K verify graph bake different query geometries that cannot share one slot.

Landed the **enabling primitive**: a `DeviceGraphSlot { Primary, Verify }` abstraction threaded through the EP graph API, with the CUDA EP now holding a second `CudaGraphLifecycle` (`verify_graph`) alongside the primary one on the same compute stream. GPU-validated that the two slots capture/replay/reset **independently** (new test `primary_and_verify_graph_slots_are_independent`, passes on H200). This is dormant infrastructure: the engine never captures into the Verify slot yet, so greedy and MTP behavior are byte-identical to origin (GPU-confirmed: MTP 14.59 tok/s, 78.9% accept, replays=0, fallbacks=0, no NaN — unchanged).

Files:
- `crates/onnx-runtime-ep-api/src/provider.rs`: `DeviceGraphSlot` enum + `*_device_graph_*_in(slot, ..)` trait methods (default impls route `Primary`→existing single-slot methods, reject other slots — every existing EP compiles unchanged via defaults).
- `crates/onnx-runtime-ep-cuda/src/runtime.rs`: `verify_graph: CudaGraphLifecycle` + slot-aware `*_graph_in(slot)` runtime methods.
- `crates/onnx-runtime-ep-cuda/src/provider.rs`: CUDA EP implements the `_in` trait methods (per-slot reset also resets the capture-error latch).
- `crates/onnx-runtime-ep-cuda/src/graph.rs`: GPU test proving Primary+Verify coexist, replay independently interleaved, and resetting one leaves the other's executable intact.

**Why:**
The coordinator's fallback clause: option-c is a big executor change; if the full two-slot capture doesn't converge in one turn, land incremental capture-safe progress + the exact remaining gap + GPU evidence, and do NOT fabricate a speedup. The single-EP-graph-slot constraint is the architectural blocker under all five requirements (fixed padded shape, pinned workspace, shape-keyed slots, stable bindings, capture-safety), and the second slot is the piece with ZERO hot-path/token-identity risk (additive trait, dormant, GPU-tested) — the right foundation to land first. No speedup number exists yet (the Verify slot is not wired into decode), and none is claimed.

**Architecture confirmed (code-anchored, origin/main 7ccdb920e):**
- EP graph API is single-slot: `CudaRuntime.graph: CudaGraphLifecycle` (runtime.rs:328), driven by `begin/end/abort/replay/replay_segment/reset_graph` → EP trait `*_device_graph*` (ep-api/provider.rs) → session `device_graph_signature: Option<..>` (bindings.rs:964). The decode-inline sibling exec **shares the same EP graph slot** (lib.rs:1247 doc: "one EP graph slot + one capture-error latch"), which is why main/inline must keep one dormant.
- The `replays=0` blocker (empirically #1647): every verify step invalidates the M=1 graph at TWO sites — the eager M=K verify forward `run_cuda_eager_rows_owned` (cuda.rs ~1274) and the commit rewind `rewind_inner` (backend.rs:145).
- The NaN blocker (#1647): the StepScoped GQA attention scratch (`step_scoped: true` in ep-cuda kernels/attention.rs:1005, standard_attention.rs:3507) is sized by query rows M and **freed after every run** (`release_step_workspace`, bindings.rs:793). The larger M=K verify perturbs the arena so a retained M=1 graph replays against a stale/moved workspace pointer → non-finite logits.

**Remaining gap to a replays-on-verify speedup (the next-turn wiring, all code-anchored):**
1. **Session per-slot signature.** Add a `Verify`-slot `device_graph_signature` + slot-parameterized `try_capture_with_device_bindings_in` / `replay_device_graph_in` / `reset_device_graph_in` on the executor (bindings.rs:939/977/1015) and `Session` (lib.rs:1425–1440), delegating to the new EP `_in` methods. (Additive; mirror the existing single-slot methods.)
2. **Fixed padded verify shape.** Always run the verify at constant `M = k+1` (=2 for production k=1); pad the draft with trailing tokens and rely on causal masking (padded rows come AFTER the real rows, so real-row logits are unaffected; the padded GDN recurrent-state advance is discarded by the existing snapshot→restore→re-advance commit in `commit_recurrent_state_to_accepted`, mod.rs:582). This makes the verify signature shape-invariant → replayable. Derive M from `num_speculative_tokens+1`, no hardcoded dims.
3. **Pinned StepScoped verify workspace.** So the Verify graph's baked attention-scratch pointer stays valid across replays: reserve the StepScoped workspace at the M=K peak and stop freeing it on the verify path (make `release_step_workspace` a no-op while the Verify graph is installed; invalidate+recapture once if it must grow — self-stabilizing after the first verify). This is the executor-layer capture-safety fix; must stay capture-safe (fallbacks=0) and GPU-validated.
4. **Native verify state machine.** Add a `verify_graph_phase` (NeedsWarmup→Armed→Ready) in `run_cuda_eager_rows_owned`/`decode_cuda_eager` (cuda.rs) that captures the fixed-M verify into the Verify slot once shape+workspace are stable and replays it thereafter; drop the unconditional invalidate at the verify site. Keep the M=1 decode graph in the Primary slot (unchanged). The #1633 recurrent-commit still runs on accept.
5. **Correctness gates (unchanged):** MTP token-identical to greedy (the 48/48 oracle + `native_verify_logits_require_restored_recurrent_state`), greedy inert, fallbacks=0.

**Validation (this turn):**
- `cargo test -p onnx-runtime-ep-cuda --features cuda,cuda-13000 --lib graph::tests` — **8/8 pass on H200 CUDA_VISIBLE_DEVICES=5** incl. new `primary_and_verify_graph_slots_are_independent` (two slots coexist, replay independently, one reset leaves the other intact).
- `cargo test -p onnx-genai-engine --no-default-features --features native-backend --lib` — **575 passed, 0 failed** (engine untouched, greedy inert).
- Full bench build `--features bench-native,native-cuda,cuda-13000` — clean (additive trait breaks no EP impl).
- GPU inertness on real Qwen3.8-27B int4 hybrid `/home/justinchu/qwen38-27b-int4-mtp-cuda`: MTP steady 14.59 tok/s, acceptance 78.9%, `cuda_graph captures=16 replays=0 fallbacks=0 invalidations=99`, no NaN — **identical to origin** (Verify slot dormant).
- **No MTP speedup number — none exists yet.** Env for GPU runs: H200 ordinal 5 (all 8 idle 0MiB/0%), PATH=/usr/local/cuda/bin, CUDA_HOME=/usr/local/cuda, ORT 1.28 cuda13 `.ort-cuda-1.28/root`, int4 block-32, `profile_native --steady`, branch `squad/mtp-capture-verify` off origin/main `7ccdb920e`.
