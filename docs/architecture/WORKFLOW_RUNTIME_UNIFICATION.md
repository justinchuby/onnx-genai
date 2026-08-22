# Workflow-runtime unification: making `pipeline.workflow` the sole runtime

Status: **in progress.** This document is the authoritative plan for collapsing
the two execution engines (`Engine` text-generation and `PipelineEngine`
workflow) into **one** canonical `pipeline.workflow` runtime with a single
state/session model, a single sampling/stopping/decode policy, and ORT/native
backend executors beneath it. It supersedes the "staged follow-up boundaries"
framing in [`NATIVE_WORKFLOW_BACKEND.md`](NATIVE_WORKFLOW_BACKEND.md), which
introduced the backend-neutral component seam this plan builds on.

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
