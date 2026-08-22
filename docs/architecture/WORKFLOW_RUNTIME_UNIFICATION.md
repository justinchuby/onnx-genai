# Workflow-runtime unification: making `pipeline.workflow` the sole runtime

Status: **ratified; execution gated on one remaining dependency (`#1716` on
`main`).** The parent has ratified REPLACE + DELETE (no compatibility facade, no
permanent delegator) and assigned this consolidation — across
`speculative/mod.rs`, the direct `Engine`, server/CLI/bench/C-API callers, and
the two legacy speculative tests — to the owner of PR #1723. The second gate
condition (an executable chained fixture) is now **satisfied** — see §Execution
gate. This document is the authoritative plan for collapsing the two execution
engines (`Engine` text-generation and `PipelineEngine` workflow) into **one**
canonical `pipeline.workflow` runtime with a single state/session model, a single
sampling/stopping/decode policy, and ORT/native backend executors beneath it. It
supersedes the "staged follow-up boundaries" framing in
[`NATIVE_WORKFLOW_BACKEND.md`](NATIVE_WORKFLOW_BACKEND.md), which introduced the
backend-neutral component seam this plan builds on.

**Execution gate (measured, `2026-08-22`).** Phases 1–4 are blocked on two
external dependencies that are not yet ready, so they cannot be *implemented and
tested* in isolation without regressing production text generation or building
on a moving target:
- **The executable chained fixture now exists (`2026-08-22`), but `#1716` is
  still the gate.** Proving ORT/native parity for the chained proposer needs a
  package that actually uses `proposal_execution`; the committed `speculative`
  fixture uses the old `max_proposal_width` form. gemma4-real-packages has now
  built the tiny executable **`gemma4_chained`** target+assistant fixture on the
  `#1716` schema (branch `justinchuby/gemma4-chained-fixture` off
  `copilot/gemma4-e2b-metadata`, commit `c9c62bcd`): committed
  `inference_metadata.yaml` + `target/` + `assistant/` `*.onnx.textproto` (reused
  proven `tiny-gemma4-assistant` graphs — hidden 16, 2H 32, vocab 32, kv_heads 2,
  head_dim 8, 2 layers), greedy/logits-emit/f32/standard ops (ORT ∩ native-CPU
  safe, op-safety inherited from `gemma4_assistant_full.rs`). Its contract is the
  agreed chained ABI: assistant = `inputs_embeds[b,q,2H]` + read-only
  `shared_kv.{full,sliding}_attention.{k,v}` (output-less pure reader) → `logits`
  + `projected_state` (folded carry); `proposal_execution {chained,
  folded_carry_output: projected_state}`; `rollback_state` = the 4 target KV
  cells (folded carry excluded). Static contract review by this PR: **accepted,
  no regen**. Phase 1 wires it into `native_workflow_parity.rs` as a required
  `assert_parity_with(root, native_engine|native_cuda_engine, …)` case. **Open
  contract item** (flagged to gemma4-metadata-audit): the exact meaning of
  `port_bindings.target_hidden_context` for the folded-carry case — carry_0 is
  dimensionally the target's `hidden_states.0` (H), not `inputs_embeds` (2H), so
  the field is either source-naming (should point at the target hidden output) or
  destination-naming (the fused port); must be pinned identically in the metadata
  contract and the interpreter before the Phase-1 chained driver is written.
- **`#1716` is not on `main` — the remaining gate.** Its branch
  (`copilot/gemma4-e2b-metadata`) is ~50k insertions, still evolving (tip added a
  26B MoE example), and edits the interpreter seam this PR owns
  (`pipeline/islands.rs` +246, `pipeline/mod.rs` +138, `pipeline/workflow.rs`
  +12, `speculative/mod.rs` +94). Starting the interpreter chained/AR work now
  guarantees a conflict on a moving target. The agreed order is: `#1716` lands on
  `main` → this PR merges `main` (never rebase) → consolidation proceeds.

The AR/chained interpreter node is a real integration, not a small add:
`SessionDecodeLoopBackend` (the ORT `DecodeLoopBackend`) is tied to `Engine`
state (`session`/`kv_cache`/`scheduler`/`session_id`/`state`), so lifting decode
under the interpreter moves that decode core, which is exactly why it must land
after `#1716` (to avoid re-doing it against the seam edits above).

Read [`RULES.md`](../../RULES.md). Rule 3 forbids retaining compatibility shims
for our pre-release APIs; this plan **removes** them rather than keeping two
engines. Rule 2 keeps behavior metadata-driven. Rule 8 requires the tests to
land with each behavioral phase — so each phase below is independently landable
and independently proven, because a half-finished collapse that regresses
production text generation is *not* the "correctness first" outcome this
refactor demands.

## 1. Where the duplication actually is (measured)

Two facts from the current tree shape the whole plan:

1. **The `Engine` session/decode model is already unified across ORT and
   native.** The `*_native_*` session methods
   (`create_native_session`, `close_native_session`,
   `generate_native_in_session`, `generate_native_in_session_with_callback`,
   `rewind_native_session`) are **deprecated shims that delegate** to the
   backend-dispatched `create_session` / `close_session` /
   `generate_in_session` / `rewind_session_by`
   (`engine/runtime.rs:2102-2361`). ORT vs native is a `DecodeLoopBackend`
   choice *beneath* one loop (`decode_loop.rs::run_decode_loop`,
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
                                     │   │ ONE decode policy              │  decode_loop.rs + processors.rs
                                     │   │ run_decode_loop / sampling /   │
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

- **Phase 1 — autoregressive-decode node beneath the interpreter.** Add a
  workflow node kind (or a specialized component executor) that runs
  `run_decode_loop` against a decoder component, so an AR loop expressed in a
  workflow reuses the *one* decode policy (rich Rust sampling, KV, speculative)
  instead of the fixture's ONNX policy graphs. Backend executors stay
  `DecodeLoopBackend` (ORT) / native. Proven by parity: the `decoder` fixture
  output via the AR node equals the direct `Engine` output.

  This phase also **lifts the chained speculative proposer into the
  interpreter** so it stops being an ORT-only path. Today the folded/threaded
  draft loop lives in a direct-`Engine` Rust `propose()`
  (`speculative/mod.rs:1273-1356`: `inputs_embeds = concat(embedding(last_token),
  carry)`), which native (the component seam) cannot drive — a fork. Keyed by
  `SpeculativeProposalExecution::Chained` (onnx-genai `#1696`, with
  `folded_carry_output` **or** `recurrent`), the interpreter owns: build the
  fused `inputs_embeds`; thread `carry_0` from the target hidden context named by
  `port_bindings.target_hidden_context` (**open contract item, flagged to
  gemma4-metadata-audit**: for the folded case `carry_0` is dimensionally the
  target's `hidden_states.0` [H], not `inputs_embeds` [2H], so this field is
  source-naming vs destination-naming — pin it identically in metadata + the
  interpreter before writing the driver), `carry_k = folded_carry_output(k-1)`;
  drive the chain to `max_proposal_width`;
  and treat the folded carry as **not** a rollback state cell (recomputed from
  committed tokens on rejection, never restored). Only the per-step forward pass
  is per-backend (`invoke_onnx_component`). The old `SpeculatorConfig` proposer
  path   is deleted once this lands (Rule 3; confirmed non-dependency by onnx-genai
  `#1716` — examples 22/24 assert only the contract, not which runtime drives
  them). Proven by the executable **`gemma4_chained`** target+assistant fixture
  (committed by gemma4-real-packages on the `#1716` schema; see §Execution gate)
  wired into `native_workflow_parity.rs` as a **required ORT/native (and, since
  its shapes resolve from bound input symbols, native-CUDA device-resident)
  parity case** — greedy exact-token accept/reject/rollback bit-for-bit — plus
  the catalogue examples 22 (`recurrent`) / 24 (`folded`) asserting the
  `gemma4_e2b_workflow.rs` invariants (shared `kv_ownership`, no KV transitions,
  `state_pairs: None` for the cacheless drafter, read-only borrowed KV with zero
  writeback).

  Phase-1 preconditions (cross-PR; ratify with the parent + the
  `speculative/mod.rs` / `#1696` owners before deleting the old path):
  - The parent ratifies **"replace + delete the old `SpeculatorConfig` proposer
    path"** as the official direction and assigns single ownership of the
    interpreter seam (`pipeline/workflow.rs`) + the chained-wiring, so no two PRs
    edit it.
  - Pre-existing gates on the old direct-`Engine` speculative path —
    `gemma4_assistant_full`, `chained_proposer_real` — are **re-pointed** at the
    interpreter `Chained` construct (or become the ORT/native parity cases above)
    in the same change, so the deletion drops no coverage.

- **Phase 2 — canonical single-decoder workflow synthesis.** Add a Rust
  synthesizer that turns a plain `model.io`/introspected decoder into a minimal
  canonical `WorkflowSpec` (a `Loop` over the decode node with token/length/KV
  state cells and a tokens emit). `Engine::from_pretrained` builds it and runs
  it through the interpreter. Proven by text-gen parity (greedy + sampled) for a
  real tiny decoder, ORT and native.

- **Phase 3 — migrate callers to the sole runtime.** Point server, CLI, bench,
  C ABI (`onnx-genai-capi`), and Python (`onnx-genai-python`) at the workflow
  runtime (directly for workflow packages, via the synthesized workflow for
  plain decoders). Collapse the server's `if handle.pipeline { … } else { … }`
  dispatch (`routes/completions.rs`) into one path. Prove server text-gen and
  pipeline behavior unchanged.

- **Phase 4 — collapse `Engine` to a zero-logic delegator and delete the
  superseded orchestration.** Fold speculative, FIM, sessions, connector, and
  batching into the workflow runtime (or its shared infra), then reduce `Engine`
  to construction + delegation, deleting the parallel generate/session routing.
  Remove now-dead config flags/branches and the tests that pinned the deleted
  legacy paths.

Shared infrastructure that both engines already use — `engine/load.rs`,
`governor.rs`, `memory_strategy.rs`, `memory_plan.rs`, `metadata.rs`,
`placement.rs`, `session_state.rs` — is **kept**; it is not duplicated
orchestration.

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
