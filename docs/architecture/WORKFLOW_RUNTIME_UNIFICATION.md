# Workflow-runtime unification: making `pipeline.workflow` the sole runtime

Status: **single-row generation is unified; two algorithms are not yet.**

There is one runtime type, one interpreter and one drive. Every package
serializes `pipeline.workflow`, and every **single-row** generated token comes
out of `pipeline::workflow::run_workflow_node` walking that document: the
iteration bound, the liveness predicate, the carried state, the emit that
publishes the token stream and the stop are read from the package, not supplied
by a Rust loop written beside it. That covers a bare decoder, a composite
pipeline whose sampler is a graph, and the scheduler's per-token drive.

**Two loops still own their own iteration and do not reach the interpreter**, and
this document says so rather than rounding up:

* `speculative/mod.rs::generate_speculative_loop` and
  `native_speculative.rs::NativeSpeculativeDriver` — reached whenever a request
  names `speculative_mode` / `num_speculative_tokens`, or an MTP head is loaded.
* `batched.rs::generate_batched_static` and `ContinuousBatchManager` — reached by
  the server's continuous-batch driver, which is the default whenever the decode
  path reports it can batch and more than one request is in flight.

Both take their bound and their stop from Rust, not from the workflow. §"The
three loop algorithms" below describes what expressing them as declared steps
requires. Until that lands, "every generated token comes from the interpreter" is
true of the single-row path and false of these two.

**What varies between packages is which executor implements a declared node,
and that is answered by the contract the node names.** A component declaring
`onnx-genai.autoregressive-decode` is run by the fused decode session — paged
KV, the device sampling fast paths, the captured CUDA graph — when this runtime
holds one, and invoked from its own ONNX artifact when it does not. A component
declaring `onnx-genai.token-policy` is run by the single sampling/stopping
implementation. A component declaring neither is invoked generically. No code
asks what *kind* of package it holds; `Engine::holds_decode_core()` asks only
whether this process has the fused session, which is a question about a
resource.

`crates/onnx-genai-engine/tests/authored_body_selects_executor.rs` measures
this rather than asserting it: three packages declaring three different loop
bodies are driven through one entry point, and the per-contract execution
counter recorded inside the interpreter's node dispatch shows each one selecting
a different executor set. A package whose sampler and termination predicate are
ONNX components shows **zero** contract executions — if that ever became
non-zero, the runtime would be supplying a step the package declared a graph
for.

`crates/onnx-genai-engine/tests/shape_dispatch_gate.rs` keeps the removed
vocabulary out. `lowered_workflow`, `canonical_decode`, `WorkflowShape` and
`is_workflow` no longer exist anywhere in the tree; the gate's allowance is
empty but for the gate itself, and it may only shrink.

## What is gone

| removed | why |
|---|---|
| `pipeline/canonical_decode.rs` | the loop it owned is the interpreter's `Loop` node |
| `Engine::lowered_workflow` | one non-optional `WorkflowRuntime`, so there is no second field a workflow could be in |
| `canonical_workflow()` and its nine guard call sites | the state they guarded against is unrepresentable |
| `WorkflowShape` / `declared_workflow_shape` | the loader asks which executor covers the declared decode step, not what shape the package is |
| `is_workflow()`, `workflow_shape()`, `WorkflowShapeReport` | a caller has nothing to do differently, so a question whose answer distinguished packages existed only to be branched on |
| `from_pipeline_dir*`, `generate_pipeline_with_callback(s)`, `workflow_generate` | one constructor family, one generate |
| server `start_pipeline` / `run_pipeline_driver` / `GeneratePipeline` | one driver, one generation command carrying an optional session and optional bound tensors |
| CLI `Backend::{Text,Pipeline}` | one backend holding the contracts a package declares |

## The three loop algorithms

They are three algorithms, because they *are* three — a token, a row-major step,
a proposed block. Only the first is currently driven by the interpreter.

1. **single-token — landed.** A body naming `autoregressive-decode` +
   `token-policy`. The interpreter drives the loop; the decode core executes the
   two nodes. `authored_body_selects_executor.rs` measures the selection.
2. **continuous-row batch — not yet.** `batched.rs` owns its iteration. The
   interpreter's `Loop` already resolves per-row liveness through
   `continue_when`, so the row vector *can* be the predicate; what is missing is
   a decode executor that advances N rows behind a declared contract, and a
   package that authors one.
3. **speculative block — partly.** A composite package that authors its
   proposer, verifier and acceptance components is driven by the interpreter
   today (the `speculative` fixture, and the chained driver on `gemma4_chained`).
   The ORT draft-model loop in `speculative/mod.rs` and the native driver in
   `native_speculative.rs` are **not**: they are selected by a *request option*
   (`speculative_mode`, `num_speculative_tokens`) and own their own iteration.

For 2 and 3 the remaining work is the same shape: express the step as a declared
contract, register an executor for it, and let the interpreter's `Loop` own the
iteration — exactly what the single-token case now does. Neither is a second
*runtime*; both are second *loops*, and the difference matters because a second
loop can be moved without changing what a package declares.

## The scheduler's per-token drive runs the same loop

A prioritized request is advanced one token at a time, so it cannot call a
function that loops to completion. The loop's phases are therefore
`begin_workflow_loop` / `advance_workflow_loop` / `end_workflow_loop`, driven in
a `for` by `run_workflow_node` and one iteration per call by
`WorkflowGenerationCursor`. One statement of the semantics, two ways to drive
it — a second walker is exactly what could disagree.

## Sessions

A session is the conversation a client is having. Where the runtime keeps it — a
paged KV sequence, or the `scope: session` cells a workflow declares — is not
something a client can act on, so `create_session` / `session_token_count` /
`close_session` and the HTTP `X-Session-Id` header mean the same thing for every
package.

---

**The chained-proposer contract, as executed.** The Phase-1 driver reads every
field it needs from `speculative.proposal_execution` and never infers by
convention or model name:

| contract field | driver use |
|---|---|
| `token_embedding_input` | the fused proposer input the chain rebuilds each step |
| `logits_output` | the draft distribution the chain argmaxes |
| `recurrent[]` `{state, input, output}` | loop-carried state cells threaded and checkpointed |
| `folded_carry_output` | carry_k for every step k >= 1 |
| `folded_carry_seed` `{component, output}` | carry_0, read from the target output it names |
| `token_embedding` `{component, table}` | the initializer `embed(last_token)` is gathered from |
| `port_bindings.target_hidden_context` | validated to equal `token_embedding_input` for a folded carry |
| `max_proposal_width` | the bound a requested width is checked against |
| `rollback_state` | the cells a rejection truncates, on each cell's own serving `sequence_axis` |

The hermetic `gemma4_chained` fixture (imported from gemma4-real-packages @
`8a66e2c8`) exercises the folded-carry shape end to end; `gemma4_chained_mixed`
adds heterogeneous per-layer KV geometry (sliding head_dim 8, full head_dim 16).
The identical contract shape is on the published real Gemma4-E2B packages
(`folded_carry_seed: {component: target, output: hidden_states.34}`,
`token_embedding: {component: target, table: model.embed_tokens.weight}`), so
the same field-reading driver serves both — the "no model-name gate" invariant,
confirmed empirically rather than asserted.

**Coverage boundary to respect.** The tiny drafter's graph slices only the carry
half of its fused input (`Slice starts=[16] ends=[32] axis=2`), so its greedy
tokens depend on the carry chain and the borrowed read-only KV, **not** on
embedding-gather correctness. That cleanly isolates folded-carry threading +
borrowed KV + accept/reject/rollback, but means the parity case cannot prove the
gather. `gemma4_chained_workflow::token_embedding_gather_matches_the_declared_table`
proves it separately, against the package's own second copy of the table
(`input_embedding.f32` vs the target's `hidden_table` initializer), so the parity
suite never implies embed-building is proven when it isn't.

Read [`RULES.md`](../../RULES.md). Rule 3 forbids retaining compatibility shims
for our pre-release APIs; this plan **removes** them rather than keeping two
engines. Rule 2 keeps behavior metadata-driven. Rule 8 requires the tests to
land with each behavioral phase — so each phase below is independently landable
and independently proven, because a half-finished collapse that regresses
production text generation is *not* the "correctness first" outcome this
refactor demands.

## 1. Where the duplication actually was (measured)

Two facts measured on the pre-change tree shape the whole plan. They describe
the *starting* state; symbols named here may since have been deleted, and §6
records what replaced them.

1. **The `Engine` session/decode model is already unified across ORT and
   native.** The `*_native_*` session methods
   (`create_native_session`, `close_native_session`,
   `generate_native_in_session`, `generate_native_in_session_with_callback`,
   `rewind_native_session`) are **deprecated shims that delegate** to the
   backend-dispatched `create_session` / `close_session` /
   `generate_in_session` / `rewind_session_by`
   (`engine/runtime.rs:2102-2361`). ORT vs native is a `DecodeLoopBackend`
   choice *beneath* one loop (`decode_loop.rs::run_decode_loop`, since deleted,
   `SessionDecodeLoopBackend` = ORT, `native_decode::NativeLoopAdapter` =
   native). **One decode policy already exists**
   (`decode_loop.rs` + `processors.rs::select_next_token*` /
   `finish_reason_after_token` + `commit_selected_token`). These shims are pure
   legacy surface — **Phase 0 deletes them.**

2. **The remaining split is `Engine` (plain / `model.io` decoder) vs
   `PipelineEngine` (`pipeline.workflow`).** `model.io` is already the *legacy*
   ABI — `onnx-genai-metadata` rejects it beside a workflow and calls the
   workflow canonical (`validation.rs:852-895`). But collapsing the two engines
   is **not** a shim removal: `Engine` provides paged KV, the batch scheduler,
   multi-turn sessions, speculative decode (draft / MTP / EAGLE-3 / shared-KV),
   FIM, connector KV, and the rich Rust sampler (DRY, Mirostat, XTC, penalties,
   grammar) — **none of which has a `pipeline.workflow` representation today**,
   and there is **no plain-decoder → `WorkflowSpec` synthesizer** (only
   `decoder_abi`/`compile_workflow`, which lower an *existing* workflow). The
   workflow decoder fixture expresses sampling as ONNX policy graphs that are
   strictly less capable than the Rust sampler. So the collapse requires *new
   capability*, delivered in phases, each proven before the next.

## 2. Target architecture

```
                         ┌───────────────────────────────────────────┐
   from_pretrained  ───► │  the package's declared WorkflowSpec        │  read, never synthesized
   (plain decoder)       │  (Loop{ decode }, state cells, emit tokens) │
                         └───────────────────┬─────────────────────────┘
   from_dir ─────────────────────────────►│  (authored workflow package)
                                             ▼
                    ┌────────────────────────────────────────────────┐
                    │  ONE workflow compiler + interpreter            │  pipeline/workflow.rs
                    │  SSA · loops · branches · emits · state ·       │
                    │  session checkpoints                            │
                    └───────────────┬───────────────┬─────────────────┘
                    generic component│               │autoregressive-decode
                    (single pass,    │               │node (Phase 1)
                     diffusion, VLM) │               ▼
                                     │   ┌───────────────────────────────┐
                                     │   │ ONE decode policy              │  generation.rs + decode_loop.rs
                                     │   │ canonical body / sampling /    │  + processors.rs
                                     │   │ stopping / KV commit           │
                                     │   └───────────┬─────────────────────┘
                                     ▼               ▼
                    ┌────────────────────────────────────────────────┐
                    │  ORT / native backend executors                │  invoke_onnx_component (PR #1723)
                    │  Session::run · InferenceSession::run · KV      │  + DecodeLoopBackend
                    └────────────────────────────────────────────────┘
```

One interpreter, one state/session model (`engine/session_state.rs`, already
shared), one decode policy (`decode_loop.rs`/`processors.rs`), ORT/native
executors beneath. `Engine` survives only as a **zero-logic convenience entry**
that constructs and runs a canonical workflow.

## 3. Phased plan (each phase lands green, with tests)

- **Phase 0 — delete the native-session compatibility facade (this change).**
  Remove `create_native_session`, `close_native_session`,
  `generate_native_in_session`, `generate_native_in_session_with_callback`,
  `rewind_native_session`; migrate the only caller (`onnx-genai-bench`
  `multiturn`) to `create_session` / `generate_in_session_with_callback` /
  `close_session`. Proven by the existing native decode/session tests and the
  bench build. No behavior change — the shims already delegated.

- **Phase 1 — chained speculative proposal driving in the interpreter — LANDED.**
  The folded/threaded draft loop used to live in a direct-`Engine` Rust
  `propose()` bound to ORT `Session`s, which the backend-neutral component seam
  could not drive — a fork. It now lives in
  `crates/onnx-genai-engine/src/pipeline/speculative.rs`, and every proposer step
  runs through `PipelineEngine::invoke_component_values` ->
  `invoke_onnx_component`, so ORT and native execute the identical chain.

  What the interpreter owns, all read from the contract (see the table above):
  the fused `concat(embed(last_token)[leading], carry[trailing])`; carry_0 from
  the target output `folded_carry_seed` names and `carry_k =
  folded_carry_output(k-1)`; `recurrent[]` state threaded and checkpointed; the
  chain driven up to `max_proposal_width`; acceptance
  (`accept_chained_proposal`); and rollback of the declared `rollback_state`
  cells (`rollback_speculative_state`), with the folded carry deliberately
  **excluded** — it is recomputed from committed tokens, never restored.

  Two interpreter facts had to become explicit for this to work:
  * `PipelineEngine::run_pipeline_retained` keeps a pass's whole SSA map, so a
    proposal binds the proposer's borrowed read-only shared KV and masks from
    the values the workflow itself bound. Declared package outputs are still
    host-materialized exactly as `run_pipeline` returns them; internal values
    stay where the backend produced them, so a native-CUDA chain keeps its carry
    and KV device-resident.
  * Island fusion elides values no later step reads, which swallowed exactly
    those tensors. `plan_execution_islands` now takes an externally-used set
    computed from the declared contract, so a speculative package keeps them
    live and a package without a chained proposer contributes nothing.
  * Proposer ports that share the fused input's position symbol narrow to the
    step's single position, derived from the declared port contracts — so a
    borrowed-KV drafter's `kv_sequence`-keyed mask is untouched while its
    `sequence`-keyed position ids are narrowed.
  * Every tensor the chain touches stays in the residency that produced it.
    `pipeline/device_ops.rs` is the backend-neutral seam that makes that
    expressible: `slice_axis` (with `last_along_axis` / `truncate_axis` as its
    two windows), `gather_rows`, `scatter_into_last_axis`, `zeros`, `argmax_rows`
    and the single sanctioned crossing, `adopt`. A host implementation and a CUDA
    implementation answer the same questions, and a value with no implementation
    for where it lives is an error naming both remedies — never a quiet copy to
    the host. Device writes issue on cudart's legacy stream while the providers
    run kernels on non-blocking streams, so every write batch ends in one
    barrier — two to three more per step than correctness strictly needs, and
    deliberately so: a fence a caller can forget is a silent-wrong-answer class,
    and what it replaces is orders of magnitude larger. A rejection therefore
    truncates the declared KV cells *on the device* (the transfer that used to
    dominate the workflow: the whole cache down and back, per rejection), a
    step's `concat(embed(token), carry)` is
    gathered and scattered into one device buffer allocated once per proposal,
    and a draft token costs a four-byte argmax read-back instead of a
    vocabulary-wide download. The declared embedding table is read out of the
    artifact once and mirrored into the chain's residency once, for the
    runtime's life.
  * The chain's residency is where the proposer *executes*
    (`WorkflowRuntime::component_execution_residency`), which is configured
    rather than discovered, so the fused input is built in the right memory from
    the first step. `invoke_component_values` takes the symbols a proposer's
    outputs declare and its inputs do not — a vocabulary, above all — proven
    from the workflow's own bound values, because an output whose shape cannot
    be resolved has no device buffer sized for it and comes back through host
    memory. A hint contradicting an input's actual extent is a validation
    error, not an override.

  Deleted with it (Rule 3, no facade): `SpeculativeMode::SharedKv`,
  `SharedKvProposerConfig`, `SharedKvBinding`,
  `validate_shared_kv_proposer_config`, `SharedKvProposer` and its `propose()`,
  `SharedKvProposerModel`, `load_shared_kv_proposer`,
  `shared_kv_slices_from_materialized`, `NativeSharedKvProposerModel` /
  `NativeSpeculationKind::SharedKv` (already unreachable — the field was
  hard-wired to `None` at load), `onnx-genai-ort::shared_kv_proposer`,
  `onnx-genai-ort::gemma4_assistant` (the model-name-gated
  `Gemma4SharedKvSpec` / `Gemma4AssistantSignature` path, an orphan file never
  compiled), `native_decode::proposer` and `NativeDecodeSession::shared_kv_inputs`,
  `SpeculativeProposerContext::shared_kv_slices`, and the metadata parser's
  `SharedKvProposerSpec` / `resolve_shared_kv`. A legacy
  `proposal_type: shared_kv` block still parses but degrades to `Unknown` with a
  diagnostic naming the contract that replaced it.

  Re-pointed gates (no coverage dropped): `gemma4_assistant_full` ->
  `gemma4_chained_workflow::speculative_decode_equals_greedy_decode` plus the
  ORT/native/CUDA parity case; `gemma4_assistant_mixed` ->
  `gemma4_chained_workflow::mixed_head_dim_speculative_decode_equals_greedy_decode`
  on the new `gemma4_chained_mixed` package. `chained_proposer_real` is an
  EAGLE-3 chain gated on `ONNX_GENAI_CHAINED_SPEC_PACKAGE` and never touched the
  shared-KV path, so it is unaffected and stays as-is.

  Proven by `gemma4_chained_workflow` (8 cases) and
  `native_workflow_parity::chained_speculative_proposal_parity{,_native_cuda}` —
  identical proposals, identical accept/reject/rollback paths, and identical
  tokens on ORT, native-CPU, and device-resident native-CUDA (H200) — and, for
  *where* the work happened,
  `native_workflow_parity::chained_proposal_stays_device_resident_native_cuda`:
  zero device→host materializations across a full speculative decode, exactly
  four bytes read back per proposer invocation, and every rolled-back state cell
  still resident on its own device at the truncated length.
  `repeated_proposals_do_not_retain_device_memory_native_cuda` holds free device
  memory flat across a hundred proposals, which is what proves the aliases a
  step holds release the buffers they borrow.
  `device_ops::cuda_tests` is the differential layer beneath both: every device
  slice, gather, scatter, argmax (ties and NaN included) and adoption must equal
  what the host implementation produces, element for element.

- **Phase 1b — autoregressive decode beneath the interpreter — LANDED.**
  The decode loop no longer hard-codes its own iteration: `pipeline/generation.rs`
  reads the loop body out of the `WorkflowSpec` and dispatches each step by
  contract id (`onnx-genai.autoregressive-decode`,
  `onnx-genai.token-policy`), so the *spec* decides what runs and in what order,
  while the rich Rust policy (sampling, stopping, KV commit, speculative)
  stays the single implementation. Backend executors stay `DecodeLoopBackend`
  (ORT) / native. §7 records the landed shape.

- **Phase 2 — a single decoder *is* a workflow — LANDED.** An earlier plan
  compiled `model.io` into a `WorkflowSpec` in memory at load. It was superseded:
  keeping a second serialized way to state a graph ABI meant keeping a compiler,
  a fallback and a provenance distinction for it. `model.io` is **deleted from
  the schema**, every package declares `pipeline.workflow`, and one carrying the
  retired block is refused at load with an error naming the offline
  `migrate_model_io` conversion. No lowering runs at load.

- **Phase 3 — migrate callers to the sole runtime — LANDED.** Point server, CLI, bench,
  C ABI (`onnx-genai-capi`), and Python (`onnx-genai-python`) at the workflow
  runtime (directly for workflow packages, via the synthesized workflow for
  plain decoders). Collapse the server's `if handle.pipeline { … } else { … }`
  dispatch (`routes/completions.rs`) into one path. Prove server text-gen and
  pipeline behavior unchanged.

- **Phase 4 — delete the superseded orchestration — LANDED.** There is one
  runtime type, one session/state model, and one sampling/stopping/commit
  policy. The second single-row token loop (`run_decode_loop`) and its
  per-step twin (`step_decode_loop`) are **deleted**, not delegated: every
  single-row autoregressive request — run-to-completion, in-session, and the
  scheduler's prioritized drive — executes the canonical body. §7 records what
  remains and why.

## 4. Migration notes for callers

- `Engine::create_native_session()` → `Engine::create_session()` (native backend
  is selected by `EngineConfig::decode_backend`, not by the constructor).
- `Engine::generate_native_in_session[_with_callback](id, …)` →
  `Engine::generate_in_session[_with_callback](id, …)`.
- `Engine::close_native_session(id)` → `Engine::close_session(id)`.
- `Engine::rewind_native_session(id, n)` → `Engine::rewind_session_by(id, n)`.
- Beyond Phase 3, construct the runtime through the workflow entry point;
  `Engine::from_pretrained` remains only as a delegator that synthesizes and runs
  the canonical single-decoder workflow.

## 5. Why not do it all at once

A single-pass collapse would have to reimplement paged KV, the scheduler,
speculative trees, FIM, and the rich sampler as workflow constructs before the
first green test — leaving production text generation and the OpenAI server API
regressed in the interim. That trades the one thing this refactor must not
trade: correctness with full tests. The phases above each keep the tree green
and each carry their proof.

## 6. Landed code, by symbol

Phases 1b-4 were code, not a plan. This is the surface each one actually
touched, so a reviewer can check the claim rather than take it.

**Phase 1b — the declared loop, executed.**
- `crates/onnx-genai-engine/src/pipeline/generation.rs`:
  `validate_generation_workflow` refuses at load a loop this runtime would
  silently override, naming the field; `GenerationNodeHost` is the decode
  core as *node executors*, advancing exactly one declared node per call;
  `run_declared_generation` is the one drive, with or without a core.
  `pipeline/canonical_decode.rs` — which owned a loop of its own — is deleted.
- `crates/onnx-genai-engine/src/decode_loop.rs`: `run_decode_loop` and
  `step_decode_loop` are **deleted**. What is left is the per-step split the
  canonical body is built from — `forward_step` (one forward pass, including
  the device greedy/sampling fast paths) and `select_and_commit_step` (the
  processor chain, sampler, logprobs, KV commit, stop/EOS) — plus
  `DecodeLoopBackend` / `SessionDecodeLoopBackend`, which are now purely
  *backend executors*, not loop owners.
- `crates/onnx-genai-engine/src/processors.rs`: `select_next_token*`,
  `finish_reason_after_token`, `commit_selected_token` — the one sampling /
  stopping / commit policy, unchanged and still the only one.

**Phase 3 — migrate callers.**
- One public runtime type. `PipelineEngine` is deleted; the workflow interpreter
  is `pub(crate) pipeline::WorkflowRuntime`, held by `Engine`.
- `Engine::from_dir` resolves the package shape itself, so no caller runs
  `PipelineModelDirectory::load_if_declared` and picks a constructor.
- `crates/onnx-genai-server/src/driver.rs`: one `DriverCommand::Generate`
  carrying an optional session and optional bound tensors, one
  `run_generation`, and one `run_engine_driver` whose only question is whether
  the decode path reports it can batch — a capability, not a package kind.
- `crates/onnx-genai-server/src/routes/completions.rs`: the four
  `if handle.pipeline` branches are one `submit_generation` call. `sessions.rs`
  and `admin.rs` ask the runtime. `ModelHandle::pipeline` is deleted.
- CLI (`interactive.rs`, `transcribe.rs`), `run_comfyui`, `profile_native`, the
  workflow example, and 40+ tests name one type.
- The C ABI and Python bindings never referenced `PipelineEngine`; they route
  through `Engine` and inherit the collapse with no migration of their own.
- Pinned by `crates/onnx-genai-engine/tests/one_runtime_e2e.rs` (5 cases).

**Phase 4 — delete the superseded orchestration.**
- `crates/onnx-genai-engine/src/engine/runtime.rs`: the runtime guard every
  decode entry point used to call is **gone**, because the state it checked for
  is no longer representable. `Engine` holds one non-optional interpreter, and
  `Engine::declared_workflow` refuses a package declaring no workflow before an
  engine exists — so there is nothing downstream to check.
- `crates/onnx-genai-engine/src/session.rs`: `ActiveGenerate` carries a
  `WorkflowGenerationCursor`, so the prioritized drive advances the *same*
  declared loop the run-to-completion path drives, one iteration per call.

**The pre-admission ordering property is preserved by construction.** A refusal
must never consume a scheduler slot or mutate session state. That used to be a
property of where the workflow guard was placed; it is now a property of what
can fail at all. In `generate_in_session_with_priority_and_callback` the
executor question, option validation, EOS defaults, prompt tokenization, session
existence and the processor chain all resolve before
`admit_generate_request_with_scheduler`; in `prepare_active_generate` the
cursor — which binds the request and runs the loop's setup, and is where an
unexecutable package fails — is built before both `sessions.remove` and
`enqueue_generate_request`. The batched routes keep an explicit precondition at
their single admission point, because "this package's declared loop is not a
row-major batch" is a real refusal rather than an unreachable one.

**Two loops remain, and they are not duplicates.**
`batched.rs` (N rows advancing together) and the speculative
propose/verify/rollback block are *different algorithms*, not second copies of
the single-row loop. Both call the same policy primitives
(`select_next_token_with_rng`, `logprob_for_token`, `commit_selected_token`),
so there is still exactly one sampling/stopping/commit implementation; what
differs is the shape of the iteration, which is the thing that genuinely
differs. `native_speculative.rs::NativeSpeculativeDriver` and the
`SpeculativeProposer` implementations (`MtpProposer`, `Eagle3Proposer`,
`DraftModelProposer`, `NgramProposer`) are proposal sources beneath that block;
`Chained` is already interpreter-driven and is the template if the others are
moved behind contracts later.

Shared infrastructure both engines already used — `engine/load.rs`,
`governor.rs`, `memory_strategy.rs`, `memory_plan.rs`, `metadata.rs`,
`placement.rs`, `session_state.rs` — is **kept**; it is not duplicated
orchestration.

## 7. A single decoder is an ordinary workflow — landed

**One representation, declared not derived.** A single decoder ships a
`pipeline.workflow` with one ONNX component: port `roles` name its semantic
inputs and outputs, a `state_service` group owns its KV cache and declares the
past/present aliases a runtime may exploit, and a `loop` step drives generation
and emits tokens. These are the constructs a multi-component workflow uses;
there is no decoder-shaped vocabulary.

**The runtime's token policy is a component.** `token_policy` is a `binding`
carrying the contract `onnx-genai.token-policy` — the schema's existing way for
a workflow to say "a step happens here and the runtime implements it". That is
what lets a single decoder keep the rich Rust sampler, paged KV, sessions and
speculative decode, none of which has an in-graph representation, without
needing a second package shape to hold them.

**Recognition is structural.** `sole_decoder_component` finds the one component
that consumes the autoregressive sequence and produces logits. No component
name, model name, or architecture string decides anything — a package may name
its component whatever it likes. `is_single_decoder_workflow` is the single
recognizer every caller asks (engine loader, server state, server/CLI
multimodal, CLI inspection), so no two of them can classify the same package
differently.

**KV stays with its executor.** State cells owned by a group are
`management: runtime` with a `release_boundary`, which is the schema's existing
word for buffers the runtime owns. Paged, shared-buffer and CUDA-graph KV stay
device-resident; nothing round-trips through the interpreter as an SSA value.

**One representation, executors chosen by contract.** A component declaring
`onnx-genai.autoregressive-decode` is run by the fused decode session when this
runtime holds one and invoked from its own artifact when it does not. That is a
backend choice *beneath* the declared workflow — the same kind of choice as ORT
versus native — not a second runtime beside it, and not a mode a caller can
select. `Engine::holds_decode_core()` asks whether the session exists, never
what kind of package was loaded.

**Fail-closed by construction, not by guard.** `Engine` holds one non-optional
interpreter and the loader refuses a package that declares no workflow, so the
state the old `canonical_workflow` guards checked for is unrepresentable. The
batched routes keep an explicit precondition at their single admission point,
before a scheduler slot is taken, because "this package's declared loop is not a
row-major batch" is a real refusal rather than an unreachable one.

**Conversion is offline.** `migrate_model_io <package-dir>` rewrites a retired
`model.io` block as the canonical workflow, reading the ONNX graph for real port
contracts rather than guessing a state tensor's rank. It is deliberately not a
load-time step: a runtime that repaired packages in memory would mean the
package on disk said one thing and the runtime executed another.
`migrate_model_io --reemit` rebuilds an already-converted package's workflow
from the ABI its own workflow resolves — the direction
`decoder_workflow_roundtrip` pins as exact — which is how the fourteen converted
fixtures were brought forward when the emitter gained the decode contract and
corrected the `sequence` cell's recurrence.

### Evidence

* `decoder_workflow` (ABI → workflow) and `decoder_abi` (workflow → ABI) are
  inverses, asserted on synthetic cases *and* on all 14 converted packages
  (`tests/decoder_workflow_roundtrip.rs`). The 12-layer cases exist because
  state ports live in a `BTreeMap`, whose key order would otherwise bind layer
  10 between 1 and 2.
* `canonical_execution_parity` (8): a single decoder is an authored workflow,
  prefill, cached decode, EOS, seeded sampling, batched generation held to the
  same precondition, a composite package driven by the interpreter, and the
  fail test proving the legacy direct path cannot be selected from any of four
  entry points.
* `real_model_workflow_corpus` (5) over the real packages this machine has,
  which **fails** if it covered nothing.
* The greedy goldens over five real foundry models are byte-identical across
  the whole change, which is the load-bearing evidence that retiring the
  serialized block changed no token.

