### 2026-08-21: Option-c step 1 — lift the second CUDA graph slot into the executor/session (Verify-slot routing); steps 2-4 deferred + a new E2E blocker surfaced

**By:** Gaff

**What:**
Landed step 1 of my own 5-step option-c plan (from `gaff-mtp-verify-capture-second-graph-slot.md`): the **executor + session Verify-slot routing** that lets the main executor drive the second CUDA graph slot (`DeviceGraphSlot::Verify`, added raw at the EP in #1648) independently of the Primary M=1 decode graph. This is the plumbing the native MTP verify-capture (steps 2-4) will drive.

- `onnx-runtime-session` executor: added a `graph_slot: DeviceGraphSlot` field (default `Primary`) on `Executor`, and threaded it through **every** EP graph call the executor makes — capture begin/end/abort + segment replay (dispatch.rs), single-graph replay + reset (bindings.rs), the `SegmentCaptureGuard` abort (capture.rs), and the defensive resets in run.rs / mod.rs. Kernel-variant eviction (kernel_cache.rs) now defensively resets **both** slots (an evicted kernel can invalidate a graph in either slot; resetting an empty slot is a no-op).
- `Executor::set_graph_slot()` / `graph_slot()` + `Session::set_main_exec_graph_slot()` / `main_exec_graph_slot()` (re-exported `DeviceGraphSlot`). Retargeting resets the old slot's installed graph first, so a subsequent capture records cleanly into the new slot.
- Because the main-exec `try_capture_with_device_bindings` / `replay_device_graph` / `reset_device_graph` now route through `self.graph_slot`, the native verify path (step 4) will capture into Verify simply by setting the main exec's slot to Verify once — while the decode-inline sibling keeps the Primary M=1 decode graph. No new capture/replay Session methods are needed.

Default `Primary` makes the whole change **byte-inert**: `*_in(Primary)` delegates to the historical single-slot EP methods, so with nobody retargeting the slot, greedy and MTP behave exactly as before.

**Why:**
The coordinator's fallback clause (repeated across #1633/#1637/#1641/#1644/#1647/#1648): option-c is a big cross-crate executor change; land incremental **capture-safe** progress + the precise remaining gap + GPU evidence, and never fabricate a speedup or half-wire a risky path. Step 1 is the self-contained, greedy-provably-inert, unit-testable brick that unblocks steps 2-4. I deliberately did NOT wire the native verify state machine this turn: it is a large, high-NaN-risk change (padded persistent verify bindings + `verify_graph_phase` + pinned StepScoped workspace + KV/GDN-recurrent commit correctness under capture) that must be GPU-validated token-identical — and this turn that validation is **externally blocked** (see blocker below), so shipping it unvalidated would violate the token-identity + no-fabrication rules.

**Validation (H200 ordinal 5, idle 0MiB/0%; PATH=/usr/local/cuda/bin, CUDA_HOME=/usr/local/cuda; ORT 1.28 cuda13 `.ort-cuda-1.28/root`; int4 block-32; branch `squad/mtp-verify-replay` off origin/main `73e6fe15a`):**
- `onnx-runtime-session --features cuda` lib suite: **190 passed, 0 failed**, incl. the new GPU test `main_exec_drives_verify_graph_slot_end_to_end` (main exec captures→replays→resets on the **Verify** slot with persistent I/O and zero replay-time device allocations, then reverts to Primary) and the existing Primary-path graph tests (`cuda_graph_replay_uses_persistent_io_without_device_allocations`, `segmented_cuda_graph_claims_whole_subgraph_around_eager_seam`, `decode_inline_sibling_folds_body_into_captured_graph_byte_exact`) all still green.
- `onnx-genai-engine --features native-backend` lib suite: **575 passed, 0 failed** (greedy inert; engine untouched).
- `onnx-runtime-ep-cuda --features cuda,cuda-13000 graph::tests`: **8/8 pass** (the #1648 two-slot invariant unchanged).
- Full bench build `--release --features bench-native,native-cuda,cuda-13000 --bin profile_native`: clean (1m06s).
- **GPU greedy-inertness on the real Qwen3.8-27B int4 hybrid** (`--ep cuda --steady --tokens 128 --warmups 3`, MTP head excluded — see blocker): **56.56 tok/s, `cuda_graph replays=504 fallbacks=0` (captures=4, invalidations=3), no non-finite logits** — the Primary/inline slot captures and replays exactly as before, proving the slot-routing change is byte-inert in production greedy on the real model (matches the ~55.9 tok/s greedy A/B baseline).

**No MTP speedup number — and none is achievable this session (external blocker):**
The Verify slot is still dormant in native decode (steps 2-4 not wired), so there is no replays-on-verify number yet. Independently, **the MTP head fails to load on this ORT build**:
```
Failed to load MTP head: ORT error: Type Error: Type parameter (T) of Optype (Add)
bound to different types (tensor(bfloat16) and tensor(float)) in node ().
```
This is a graph-level type mismatch **inside `mtp/model.onnx`** (reproduced from the pristine artifact dir, unmodified metadata), rejected at ORT session creation — entirely independent of this Rust change (which cannot affect ORT's type-checking of the head graph). It means native MTP self-spec **cannot run or be token-identity-validated end-to-end in this environment right now**, regardless of engine wiring. This needs an artifact fix (Roy: re-export the MTP head with consistent `Add` operand dtypes, or cast the bf16/f32 operands) before any real MTP number is measurable again. Prior turns' 14.59 tok/s / 78.9% figures predate whatever produced this head; the head as currently on disk does not load on ORT 1.28 cuda13.

**Remaining gap (steps 2-4, unchanged plan; now also gated on the MTP-head fix):**
2. Fixed padded verify shape (constant M=k+1; causal-masked trailing padding; padded GDN advance discarded by the existing snapshot→restore→re-advance commit, mod.rs:582). The `leverb_increment0` throwaway probe (cuda.rs) already demonstrates the mechanism: persistent padded `[1,M,vocab]` logits binding + a pre-capture warm forward at M=K (alloc-free captured region) + the inherited KV-symbol pin → a capturable, replayable M=K forward with captured-vs-eager token parity.
3. Pinned StepScoped verify workspace (reserve at the M=K peak, stop freeing while the Verify graph is installed) — the #1647 NaN fix.
4. Native `verify_graph_phase` (NeedsWarmup→Armed→Ready) in `decode_cuda_eager`/`run_cuda_eager_rows_owned`: set the main exec's slot to Verify once, warm→capture→replay the fixed-M verify into the Verify slot, drop the unconditional invalidate. Primary M=1 stays on the decode-inline sibling.
5. GPU-validate MTP token-identical to greedy, both slots replays>0, fallbacks=0, then median-of-5 A/B — **once the MTP head loads again.**

Files: `crates/onnx-runtime-session/src/executor/{state,build,bindings,dispatch,capture,run,kernel_cache,mod}.rs`, `crates/onnx-runtime-session/src/lib.rs`.
