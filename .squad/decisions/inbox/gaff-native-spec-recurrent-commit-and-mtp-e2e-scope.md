### 2026-08-21: Native speculative driver commits hybrid recurrent state by accepted count; MTP E2E enablement scoped to a GPU session

**By:** Gaff

**What:**
Landed the recurrent-state correctness fix in the native-backend speculative
driver (`NativeSpeculativeDriver`, `native_speculative.rs`) and corrected the
campaign's architectural model for where MTP plugs in on the native backend.

- The native backend does NOT reach `generate_speculative_loop`
  (`speculative/mod.rs`); that loop is ORT-session-only, and `DecodeRunner::Native`
  is never constructed in production — it is dead scaffolding. The real native
  speculative path is `NativeSpeculativeDriver`, dispatched from
  `generate_native_cold_with_callback` (`engine/runtime.rs`) via
  `native_speculation_plan`. This is where MTP and the #1598 recurrent-commit
  primitive belong.
- The driver's accept path called `rewind(base + accepted)`, which only prefix-
  slices attention KV and left a hybrid decoder's destructive Gated-DeltaNet
  recurrent (SSM) + conv1d state stranded at `base + K` after each verify window
  — silently corrupting every subsequent token on hybrid GDN+GQA models under
  speculation. Now the driver snapshots the recurrent/conv state at the committed
  boundary before `decode_verify`, and on accept commits it to exactly the
  accepted prefix via the #1598 `commit_recurrent_state_to_accepted` primitive.
  Detected generically via `has_recurrent_state()`; pure-attention decoders keep
  the plain `rewind` and are byte-identical to before (greedy path fully inert).
- Test `native_verify_then_recurrent_commit_matches_accepted_prefix_replay`
  drives the exact `snapshot -> decode_verify(K) -> commit(j)` sequence for
  j in {0,1,k} on a synthetic hybrid decoder and asserts byte-identity to a
  fresh `base ++ draft[..j]` replay, with a negative control proving a plain
  `rewind` strands the state (the fix is load-bearing). Full native-backend lib
  suite: 569 passed / 0 failed / 1 ignored.

**Why:**
This is the correctness prerequisite for MTP self-speculative decode on the
dense-hybrid Qwen3.8-27B — the piece that finally exercises the #1598 primitive
on the driver that actually runs the real model. It is correct, tested, and
inert off the speculative-hybrid path, so it lands independently of the
remaining MTP enablement.

**MTP E2E enablement (Gaps 1 & 2) — scoped to a GPU session, NOT landed here:**
The two remaining gaps are real and localized but cannot be validated (or even
compiled) in this sandbox, so per campaign policy I am NOT half-wiring them:

1. Gap 1 (native MTP proposer + load wiring): `from_native_model_directory`
   (`engine/load.rs`) hard-bails on metadata speculation and hard-sets
   `mtp: None`; needs a `NativeProposer::Mtp` variant in `native_speculative.rs`
   (mirroring `SharedKv`, using `onnx-genai-ort::MtpDecodeSession` on the ORT
   CUDA EP + `LinearEmbedder`/lm_head + the target's `last_hidden()` seed), plus
   `reject_native_request_speculation` / `native_speculation_plan`
   (`engine/decode_backend.rs`) and the `speculative_mode` injection in
   `generate_native_cold_with_callback` extended for `SpeculativeMode::Mtp`.
   `mtp_config_from_metadata` already exists in `engine/speculative_load.rs`.
2. Gap 2 (aux hidden-seed shape rigidity — the true architectural blocker):
   the MTP proposer's seed is the target's `hidden_states.63`, an AUXILIARY graph
   output. `persistent_output_shape` (`native_decode/cuda.rs`) collapses its
   symbolic query-seq axis to 1 — correct for the m=1 captured decode step, but
   the resulting `[1,1,5120]` persistent binding cannot hold the eager PREFILL
   shape `[1,m,5120]`, which is exactly the `dispatch.rs` rejection Sebastian
   observed. The fix must let the aux hidden output materialize per-step to a
   host buffer sized to the actual seq during m>1 eager forwards (mirroring the
   logits FIX-1 padded-binding path) while keeping the seq=1 persistent binding
   for the captured decode step. This is CUDA-graph-capture-sensitive executor
   plumbing that MUST be validated on the GPU against the 62.56 tok/s baseline;
   Gap 1's proposer is useless until Gap 2 delivers a valid hidden seed.

**Environmental blocker (evidence):** this sandbox has NO CUDA toolkit
(`which nvcc` → not found), so `native-cuda` / `bench-native,cuda` cannot be
built, and the 17GB `/home/justinchu/qwen38-27b-int4-mtp-cuda` artifact is not
reachable/runnable here. Therefore neither the MTP-on == greedy token-identity
check, the acceptance-rate measurement, nor the median-of-5 tok/s-vs-62.56
number can be produced this session. I am deliberately NOT fabricating a speedup.
Recommend re-scoping Gaps 1 & 2 to a session on the H200 (ordinal 5) host with
the CUDA toolchain and the artifact mounted; the recurrent-commit correctness
fix landed here is the enabling foundation they build on.
