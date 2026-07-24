### 2026-07-24

**By:** Taylor (integration lead, CPU-EP perf track)

**What**

Merged `origin/main` (CUDA track + native-vs-ORT parity harness, 96 commits) into
our PR #105 branch state on `perf/merge-main` (a true merge commit; our 96
commits were NOT rebased/rewritten). Resolved every conflict preserving BOTH
teams' intent. Base merge-base: `3bc4ef0`. Ours: `53ecf1b` (perf/cpu-ep-mlas).
Theirs: `621936f` (origin/main).

Conflicted files and how each was resolved:

1. `crates/onnx-genai-engine/src/native_decode.rs` — the deep one. Both teams
   refactored the CPU decode path from base's inlined body in incompatible ways:
   - Theirs replaced the direct `input_ids`/`attention_mask`/`position_ids`
     session fields with a generalized `step_inputs: Vec<NativeStepInputBinding>`
     abstraction + `decode_host` + `decode_with_step_inputs`, adding routed
     pipeline inputs (InputsEmbeds / Routed component ports). The struct
     definition auto-merged to theirs' `step_inputs` model AND kept our `cpu_kv`
     field.
   - Ours added `decode_cpu`/`decode_cpu_inplace` returning `NativeCpuDecodeResult`
     (greedy Token vs Logits), `decode_argmax`/`supports_argmax`, the persistent
     in-place CPU KV cache (`DecodeCpuKvState`, `cpu_inplace_kv_max_len_from_env`,
     `all_pasts_consumed_by_gqa`, `ONNX_GENAI_CPU_INPLACE_KV`), profiling spans,
     and recurrent/SSM present-state handling.
   Resolution: unified into a single dispatch. `decode_with_step_inputs` now
   dispatches cuda → `decode_cuda`; `cpu_kv` → `decode_cpu_inplace`; else →
   `decode_cpu`. `DecodeBackend::decode` delegates to `decode_with_step_inputs`
   (theirs' shape); `decode_argmax`/`supports_argmax` retained (ours). Both
   `decode_cpu` and `decode_cpu_inplace` were rewritten to build inputs by
   iterating theirs' `self.step_inputs` (with the routed/inputs_embeds `supplied`
   map) while keeping ours' greedy result variant, profiling spans, recurrent
   handling, and in-place device bindings. Removed theirs' now-redundant
   `decode_host` (folded into `decode_cpu`). Dropped theirs' redundant
   `#[cfg(test)] use BTreeMap` at top of file (the test module already imports it;
   compiler-flagged unused). Kept theirs' first-conflict import addition.

2. `crates/onnx-runtime-session/src/executor.rs` — two conflicts, both combined:
   - `run_scoped` resolve: kept ours' `phase_span!("run_scoped.resolve_soft")`
     profiler wrapper AND theirs' `seed_control_flow_capture_shapes(&mut resolved)`
     (LongRoPE cos/sin capture seeding).
   - `exec_if`: kept ours' `phase_span!` wrappers around prepare/run subgraph AND
     theirs' `taken_branch_is_invariant` loop-invariant-If memoization.

3. `crates/onnx-genai-bench/src/bin/profile_native.rs` — import conflict: kept
   BOTH ours' `profile` (phase-dump) and theirs' `MinPProcessor`/
   `RepetitionPenaltyProcessor` (min-p + repetition-penalty CLI wiring). Both are
   used in the merged body.

4. Append-only logs (`.squad/decisions.md`, `.squad/agents/roy/history.md`,
   `docs/PROGRESS.md`) and `Cargo.lock` auto-merged cleanly as unions — verified
   both teams' entries present (decisions.md: 40 CPU-KV refs + 217 CUDA refs;
   PROGRESS.md: 14 CPU-EP + 179 CUDA). No entries dropped.

**Non-textual semantic fix (integration, not a raw conflict)** — a real
interaction between the two teams' optimizations, caught by theirs' control-flow
test:
- Ours' zero-copy top-level output hand-off (`try_move_host_output`) MOVES a
  produced host output buffer out of the executor each run. Theirs' invariant-If
  memoization SKIPS re-running a constant branch on repeated predicate and serves
  its output from the *resident* buffer. When the same value is both a graph
  output (moved out) and produced by a memoized invariant-If, step 2 read a
  freed/reallocated buffer → garbage. Fix: `try_move_host_output` now returns the
  copy path (keeps the buffer resident) when the value's producer node is a
  currently-memoized `If` (`self.if_last_predicate.contains_key(&producer)`). This
  preserves BOTH the zero-copy move (for logits/KV, non-If producers) and the
  memoization + its test. See executor.rs `try_move_host_output`.

**Why**

Both tracks touch the same decode/executor core. A blind "take ours/theirs"
would silently drop either the CUDA stream-ordered work + parity harness or the
CPU in-place KV cache (+49.6% Phi-3.5) + zero-copy + control-flow-validation
ordering. The single-dispatch unification keeps every feature live and testable,
and the memoization/zero-copy reconciliation keeps outputs byte-correct.

**Verification (all green except pre-existing):**
- Build: `cargo build -p onnx-runtime-ep-cpu -p onnx-runtime-session
  -p onnx-genai-engine --features mlas` ✓; `-p onnx-genai-engine
  --features native-backend,mlas` ✓; release `profile_native` (workspace incl.
  cuda crate) ✓.
- `onnx-runtime-ep-cpu --features mlas --lib`: 827 passed / 0 failed (incl. all 6
  `inplace*` GQA tests).
- `onnx-runtime-session --features mlas --lib`: 66 passed / 0 failed.
- `onnx-runtime-session --features mlas --test control_flow`: 22 passed / 0 failed
  (incl. `if_memoizes_invariant_branch_but_reruns_on_predicate_flip`, which is
  what surfaced the zero-copy/memoization interaction above).
- `onnx-genai-engine --features native-backend,mlas --lib inplace`: 2 passed
  (incl. `tiny_decoder_matches_across_inplace_env_toggle` — byte-identical parity
  proof); `--lib kv_state`: 1 passed; routed `native_target_step_*`: 3 passed.
- Parity smoke (release, load ~4.4): native greedy token streams UNCHANGED
  post-merge, and byte-identical between in-place (default) and copy path
  (`ONNX_GENAI_CPU_INPLACE_KV=0`):
    - qwen3-0.6b first token **1479**; stream `[1479, 198, 40, 1184, 311, 1477]`.
    - Phi-3.5-mini `[30751, 31512, 306, 29915, 29885, 1985]`.

**Pre-existing failures (NOT introduced by this merge — verified identical on the
respective parent branches; left as-is to avoid scope creep):**
- `onnx-genai-engine --features native-backend,mlas --lib`: 16 tests fail with
  `failed to decode Protobuf message: invalid wire type value: 6` when parsing a
  synthetic/written ONNX fixture (`tiny_fixture_*`, `scatter_fixture_*`). Verified
  the IDENTICAL 16-test failing set on pure `origin/main` (168 pass / 16 fail vs
  merge 177 pass / 16 fail — the +9 are our extra tests, all passing). Environmental
  fixture-parse issue on this box, not a merge regression.
- `onnx-genai-bench --features mlas --test profile_native::
  native_cpu_synthetic_profile_reports_decode_stages_when_enabled`: asserts stdout
  contains `loop.sampling`, but our own CPU greedy-fastpath feature
  (`greedy_fastpath_supported()` returns true for CPU) routes through
  `next_token_greedy`/`decode_argmax` and emits `native.sampling` instead. Verified
  this test FAILS IDENTICALLY on our pure HEAD `53ecf1b` — a pre-existing
  stale-assertion inconsistency inside our PR #105 branch, unrelated to the merge.

**Ambiguous hunks I could NOT safely resolve:** none. Every conflict was
integrated so both teams' behavior is present and tested. The one non-obvious
semantic interaction (zero-copy vs invariant-If memoization) was resolved with a
targeted, test-backed guard rather than a guess.
