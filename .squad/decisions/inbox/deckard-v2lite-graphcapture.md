# Decision: V2-Lite CUDA-graph-capture — classifier fix landed; a SECOND pre-existing capture-workspace blocker found

- Author: Deckard (Systems Dev — executor / capacity-policy / graph-capture)
- Date: 2026-08-18
- Branch: `squad/v2lite-graphcapture` (off origin/main `2291caa9`)
- Reviewer gate: **Rachael** (mandatory, capacity/parity-adjacent). Do NOT self-merge.
- Follows: Wallace's GO brief `.squad/decisions/inbox/wallace-v2lite-graphcapture-scope.md`
- GPU: H200 GPU5 (verified idle via nvidia-smi before every run)

## TL;DR
The attention-mask **misclassification is fixed and validated** (topology-gated, GLM-5.2
preserved; the tiny DeepSeek-V2 fixture now captures **byte-identically**). BUT unlocking the
mask decline exposes a **second, independent, PRE-EXISTING blocker** on the real 27-layer
V2-Lite: the additive mask-bias tensor's **query-length axis is an unresolved internal symbol**,
so capture-prep workspace planning cannot reserve the `Attention` workspace. This is **not caused
by (nor fixable within) the classification change** — baseline `main` already crashes under
`ONNX_GENAI_CUDA_GRAPH=1` on this model. Capture on V2-Lite needs a follow-up in the
capture-workspace/KV-buffer layer (Leon + shape-inference) before Wallace's hardware A/B.

## What I changed (topology-gated capacity policy — NOT a blanket flip)
- `crates/onnx-runtime-session/src/executor/geometry.rs`
  - `is_additive_mask_builder_op(node)` — the default-domain builder op set, derived **directly
    from the real V2-Lite mask cone**: `{CumSum, Unsqueeze, Cast, GreaterOrEqual, And, Where,
    Slice, Sub}`. `Add` is deliberately **excluded** (GLM-5.2's indexer combiner).
  - `is_capacity_form_attention_mask_input(node, i)` — true only for input 3 of a default-domain
    `Attention` whose KV cache (4/5) is bound at physical capacity (mask present, no `is_causal`).
  - `mask_binding_feeds_capacity_form_attention(graph, mask)` — structural forward walk of the
    mask's consumer cone: every consumer must be a physical-extent shape read (`Shape`/`ReduceSum`
    leaf), a capacity-form `Attention` mask input (valid leaf), or a builder op (traversed on);
    anything else (or escaping as a graph output, or reaching no capacity-form `Attention`)
    disqualifies. Rejects GLM-5.2 (its `Add` is not a builder op).
- `crates/onnx-runtime-session/src/executor/build.rs`
  - `binding_consumers_use_padded_capacity`: keeps the fast path (all direct consumers are
    `Shape`/`ReduceSum`, dense GQA masks) and, only on its miss, tries the topology-gated path.
- `crates/onnx-genai-engine/tests/deepseek_v2_tiny_qmoe_native_e2e.rs`
  - Flipped the `ONNX_GENAI_CUDA_GRAPH=1` expectation: the mask reason
    `attention_mask_consumers_are_capacity_aware` must **no longer** decline capture.
- No kernel edits. No default flip (still behind opt-in `ONNX_GENAI_CUDA_GRAPH`).

## Tests (all green; `cargo fmt`/clippy clean on touched crates; 176/176 session lib tests pass)
- `vestigial_window_mask_builder_routes_to_padded_capacity` — full CumSum/Unsqueeze/…/Where→Cast
  builder → capacity-form `Attention` ⇒ padded-safe.
- `minimal_cast_to_capacity_attention_routes_to_padded_capacity` — the tiny-fixture shape.
- `glm_indexer_add_mask_keeps_logical_width` — **GLM-5.2 regression guard**: Cast→Add ⇒ rejected.
- `mask_builder_without_capacity_attention_is_rejected` — is_causal terminal ⇒ rejected.
- `mask_feeding_only_shape_is_not_padded_capacity_via_topology` — no `Attention` leaf ⇒ rejected.
- `mask_feeding_non_builder_consumer_is_rejected` — MatMul consumer ⇒ rejected.
- `capacity_form_attention_mask_input_classifier` — slot/`is_causal` gating.
- `only_capacity_aware_inputs_keep_physical_capacity` — untouched, still green (per-node policy).

## End-to-end validation (GPU5)
- **Tiny DeepSeek-V2 fixture, `ONNX_GENAI_CUDA_GRAPH=1`:** `captures=1 replays=6 fallbacks=0
  decline=None`, CUDA tokens `[11×8]` == CPU golden ⇒ **capture now engages and is byte-identical.**
  (Before: declined with `attention_mask_consumers_are_capacity_aware`.)
- **Opt-out `=0`:** `captures=0`, declined by the `ONNX_GENAI_CUDA_GRAPH` predicate — gating intact.
- **Real V2-Lite `int4` (cohaagen-…-post434), eager `=0`:** 57.25 tok/s, unaffected by my change.
- **Real V2-Lite, `=1` — capture does NOT engage (SECOND BLOCKER, pre-existing):**
  - With my classifier: prefill capture-prep hard-errors — `prepare-only workspace planning cannot
    resolve input 3 'v_model.Unsqueeze_18' for node 38 ('::Attention'): axis 2 is symbolic
    ('_d1') …neither a context/sequence axis bounded by max_seq_len/KV capacity nor a configured
    maximum`.
  - **Baseline `main` (my classifier stashed out):** `=1` ALSO crashes on the SAME node —
    `node 38 ('::Attention') workspace invariant mismatch: execute requires 2648584 bytes,
    prepared 57344 bytes` (a silent under-reservation that surfaces at execute).
  - ⇒ The `=1` path on V2-Lite is **already broken on current main**, independent of the
    classifier. My change only moves the failure earlier and makes it principled (the planner
    *refuses to guess* `_d1` instead of under-reserving). **No regression** (eager unaffected;
    dense GQA masks use the unchanged fast path; GLM-5.2 stays logical).

## Root cause of the second blocker (for the follow-up owner)
`v_model.Unsqueeze_18` (the additive mask fed to every layer's `Attention` input 3) has shape
`['batch', 1, '_d1', 'past_seq_len + seq_len']`. Axis 3 is the key/context length (bound to KV
capacity, fine). **Axis 2 `_d1` is the QUERY-length axis** — a fresh internal symbol that shape
inference did not unify with `sequence_len` and that carries no declared max, so
`planned_axis_upper_bound` (executor/bindings.rs:236) has no justifiable ceiling and
`resolve_planned_workspace_input_shape` (bindings.rs:317) refuses it. This is a **shape/symbol
provenance + capacity-reservation gap**, NOT a mask-classification issue — outside the
capacity-policy surface I own. Suggested fix directions (for Leon / shape-inference owner):
1. Unify the mask builder's query axis with the decode query length (`sequence_len`) in shape
   inference so `_d1` resolves exactly, or
2. Recognize the additive-mask query axis as a sequence axis bounded by `max_seq_len` so the
   prepare-only planner can conservatively (over-)reserve it.
Either unblocks capture-prep; then Wallace's byte-identity/tok-s A/B can run.

## Recommendation
- **Merge the classifier + tests after Rachael's review.** It is correct, well-covered, and is the
  required first step: it removes the mask misclassification and makes capture engage on the
  capacity-form `Attention` topology (proven byte-identical on the tiny fixture). It does not
  regress eager or any currently-capturing model.
- **Do NOT expect the V2-Lite tok/s win from this change alone.** Capture on the real model is
  gated on the second (pre-existing) blocker above. Track it as the hard dependency for the
  projected 54→~82–86 tok/s.
- Wallace's original GO measured graceful `=1` decline on `addba1bc`; on current main `2291caa9`
  `=1` hard-fails at the workspace layer — worth flagging to the coordinator that the capture path
  for MLA masks regressed somewhere between those commits, independent of this work.

## Handoffs
- **Rachael:** review the topology-gated classifier (GLM-5.2 preservation is the key invariant).
- **Leon + shape-inference owner:** the `_d1` mask-bias query-axis resolution (the real capture gate).
- **Wallace:** hardware byte-identity + tok/s A/B once the `_d1` blocker lands (capture ON vs eager,
  0.000% over ≥300 tokens); the classifier + tiny-fixture parity already de-risk the mask-freeze math.
