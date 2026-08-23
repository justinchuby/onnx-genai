# Workflow-runtime unification: making `pipeline.workflow` the sole runtime

Status: **Option A is implemented.** One public runtime type; no caller
dispatches on package shape; a bare decoder is lowered at load into a canonical
`WorkflowSpec` (its `model.io` staying the sole serialized answer) and **every
single-row autoregressive request executes that workflow** through
`pipeline::canonical_decode::run_canonical_decode`, which reads its loop body
from the spec and dispatches by contract id. The second token loop
(`run_decode_loop`) and its per-step twin (`step_decode_loop`) are deleted. §6
and §7 record what landed, and the two loops that remain because they are
different algorithms rather than duplicates. The parent ratified REPLACE + DELETE (no
compatibility facade, no permanent delegator) and assigned this consolidation to
the owner of PR #1723. Its two gate conditions are both satisfied: `#1716` is on
`main` (`7b79b3f5`), merged here (never rebased), and the hermetic executable
chained fixture is committed. This document is the authoritative plan for
collapsing the two execution engines (`Engine` text-generation and
`PipelineEngine` workflow) into **one** canonical `pipeline.workflow` runtime
with a single state/session model, a single sampling/stopping/decode policy, and
ORT/native backend executors beneath it. It supersedes the "staged follow-up
boundaries" framing in
[`NATIVE_WORKFLOW_BACKEND.md`](NATIVE_WORKFLOW_BACKEND.md), which introduced the
backend-neutral component seam this plan builds on.

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
   from_pretrained  ───► │  canonical single-decoder WorkflowSpec      │  synthesized (Phase 2)
   (plain decoder)       │  (Loop{ decode }, state cells, emit tokens) │
                         └───────────────────┬─────────────────────────┘
   from_pipeline_dir ───────────────────────►│  (authored workflow package)
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
                                     │   │ ONE decode policy              │  canonical_decode.rs + decode_loop.rs
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
  tokens on ORT, native-CPU, and device-resident native-CUDA (H200).

- **Phase 1b — autoregressive decode beneath the interpreter — LANDED.**
  The decode loop no longer hard-codes its own iteration: `canonical_decode.rs`
  reads the loop body out of the `WorkflowSpec` and dispatches each step by
  contract id (`onnx-genai.autoregressive-decode`,
  `onnx-genai.token-policy`), so the *spec* decides what runs and in what order,
  while the rich Rust policy (sampling, stopping, KV commit, speculative)
  stays the single implementation. Backend executors stay `DecodeLoopBackend`
  (ORT) / native. §7 records the landed shape.

- **Phase 2 — canonical single-decoder workflow lowering — LANDED (option A).**
  `onnx-genai-metadata::canonical` compiles a plain `model.io` decoder into a
  canonical `WorkflowSpec` **in memory at load**; nothing is serialized, so
  `model.io` remains the sole on-disk answer and existing packages keep working
  untouched. `Engine::install_canonical_workflow` runs it for every decoder
  package and fails closed. §7 records the landed shape.

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

**Phase 1b — canonical decode body.**
- `crates/onnx-genai-engine/src/pipeline/canonical_decode.rs`: `resolve_body`
  reads the loop body out of the `WorkflowSpec`; `BodyStep` is the resolved
  form; `CanonicalBody::resolve` binds it once per request;
  `step_canonical_body` runs one iteration; `run_canonical_decode` runs to
  completion. Both drives call the same body, so there is no "run" versus
  "step" policy.
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

**Phase 2 — canonical lowering (option A: in memory, never serialized).**
- New: `crates/onnx-genai-metadata/src/canonical.rs` — lowers
  `decode/metadata.rs::ModelIoSpec` / `engine/metadata.rs` introspection into a
  canonical `WorkflowSpec`. It emits canonical YAML and re-parses it through
  the *unrelaxed* schema, so a lowering that the schema would reject fails at
  load rather than producing a second, weaker answer.
- `crates/onnx-genai-engine/src/engine/load.rs`: `install_canonical_workflow`
  runs at load for every decoder package and asserts the lowered spec declares
  the contracts this runtime implements.
- The capability gap that made a *serialized* synthesis unworkable — no
  workflow representation for paged KV, the batch scheduler, sessions, FIM,
  connector KV, or the rich Rust sampler — is what option A avoids: the
  canonical spec names the decode and policy contracts, and the Rust
  implementations behind those contracts keep every capability.

**Phase 3 — migrate callers.**
- One public runtime type. `PipelineEngine` is deleted; the workflow interpreter
  is `pub(crate) pipeline::WorkflowRuntime`, held by `Engine`.
- `Engine::from_dir` resolves the package shape itself, so no caller runs
  `PipelineModelDirectory::load_if_declared` and picks a constructor.
- `crates/onnx-genai-server/src/driver.rs`: `EngineBackend::{Single,Pipeline}` is
  one owned runtime; `run_engine_driver` asks `engine.is_workflow()`.
  `EngineDriver::warmup` no longer takes a caller-supplied `pipeline: bool`.
- `crates/onnx-genai-server/src/routes/completions.rs`: the four
  `if handle.pipeline` branches are one `submit_generation` call. `sessions.rs`
  and `admin.rs` ask the runtime. `ModelHandle::pipeline` is deleted.
- CLI (`interactive.rs`, `transcribe.rs`), `run_comfyui`, `profile_native`, the
  workflow example, and 40+ tests name one type.
- The C ABI and Python bindings never referenced `PipelineEngine`; they route
  through `Engine` and inherit the collapse with no migration of their own.
- Pinned by `crates/onnx-genai-engine/tests/one_runtime_e2e.rs` (5 cases).

**Phase 4 — delete the superseded orchestration.**
- `crates/onnx-genai-engine/src/engine/runtime.rs`: every decode entry point —
  `generate_with_callbacks`, `generate_in_session_with_priority_and_callback`,
  `prepare_active_generate` / `step_active_generate` (the scheduler's
  prioritized drive), and both native paths — resolves the canonical workflow
  through the shared `canonical_workflow` accessor before it can decode, and
  then runs the canonical body. There is no declaration switch left to choose a
  legacy path with; `canonical_execution_parity.rs` pins the refusal.
- `crates/onnx-genai-engine/src/session.rs`: `ActiveGenerate` carries the
  resolved `CanonicalBody`, so the prioritized drive is the same body as the
  run-to-completion loop rather than a parallel implementation of it.

**Every token-producing entry point is held to the canonical precondition.**
Not just the single-row ones. `batched.rs::generate_batched_static` and
`continuous_batch_manager` (the single admission point for both continuous-batch
entry points) resolve `canonical_workflow(..)` before they can decode, as do the
two native paths — and all four native/batched guards run *before* scheduler
admission, so a refusal never consumes a slot or disturbs session state.
`canonical_execution_parity` pins the refusal for the four single-row entry
points and for batched generation.

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

## 7. Canonical lowering and execution (option A) — landed

### Lowering

`onnx_genai_metadata::canonical` compiles a bare decoder's `model.io` into a
canonical `WorkflowSpec`, in memory only.

* **Deterministic and derived** — same `ModelIoSpec` in, byte-identical document
  out. Nothing writes it back, so `validate_model_io_against_workflow` never sees
  a pair and no published package needs re-authoring.
* **No schema change** — both canonical components are `binding` components
  identified by contract id (`onnx-genai.autoregressive-decode`,
  `onnx-genai.token-policy`), dispatched the way workflow adapters already are.
* **KV stays with its executor** — lowered state cells are `management: runtime`,
  so the paged / share-buffer / CUDA-graph executors keep owning their KV and no
  per-step host round-trip is introduced.
* **Honest provenance** — `/v1/debug/config.workflow_provenance` reports
  `authored` | `lowered` | `none`; `pipeline` keeps its meaning (does the file
  *serialize* a workflow), so a lowered decoder reports `pipeline: false`.

### Execution

`run_canonical_decode` is the one single-row autoregressive loop. It resolves its
body from the workflow (`resolve_body`) and dispatches each step by contract id,
so the spec determines what runs and in what order — a body naming an
unimplemented contract, or one that never applies a policy, is an error.

The per-step work is split so the loop owns the iteration and the executor owns
the model:

* `decode_loop::forward_step` — one decoder forward pass, taking the device
  greedy-argmax or device-sampling fast path when it applies.
* `decode_loop::select_and_commit_step` — the logit-processor chain, sampler,
  logprobs, KV commit, and stop/EOS detection.

Both are the only implementations. `step_canonical_body` runs one iteration of
the spec's body on them, and `run_canonical_decode` runs that same body to
completion — so the scheduler's per-step drive and the run-to-completion path
are one implementation, not two.

**Fail-closed, not mode-switched.** `install_canonical_workflow` runs at load for
every decoder package and asserts the lowered workflow declares the two contracts
this runtime implements. Every decode entry point then resolves the canonical
workflow through `canonical_workflow(..)` before decoding, so `generate`,
`generate_in_session*` (the server's path), and `generate_with_sampler` are all
refused if none is present — proven by
`canonical_execution_parity::the_legacy_direct_decode_path_cannot_be_selected`,
which exercises all three.

### What remains, and why it is not duplication

Two loops are not `run_canonical_decode`, because they are different algorithms:

* **continuous batching** (`batched.rs`) advances N rows per forward pass;
* **speculative decoding** (`speculative/mod.rs`, `native_speculative.rs`)
  iterates a proposed block, not a token.

Neither is a second *policy*: both drive the same primitives —
`processors::select_next_token*`, `logprob_for_token`, `commit_selected_token`,
`ensure_constrained_finish`, `finish_result` — so sampling and stopping have one
implementation across all three loops. Folding their iteration shapes into the
single-row loop would require a row-scoped policy seam and a block-scoped one;
that is the honest next step, and it is a capability change rather than a
deduplication.

### Evidence

* Greedy goldens over 5 real models (phi35-mini int4, phi4-mini CUDA,
  qwen2.5-0.5b CUDA, qwen3-0.6b, gpt-oss-20b) byte-identical to the pre-change
  baseline at `c58eb5b2` — see [`.goldens/`](../../.goldens/).
* `canonical_execution_parity` (8): lowering at load, prefill, cached decode on a
  reused session, deterministic greedy, EOS stop, seeded sampling, batching where
  supported, authored-stays-authored, and the legacy-path refusal on every entry
  point.
* `canonical_lowering_corpus` (5) over 7 real packages, which now **fails** if it
  covered nothing unless `ONNX_GENAI_ALLOW_EMPTY_CORPUS=1` says the machine is
  weightless.
