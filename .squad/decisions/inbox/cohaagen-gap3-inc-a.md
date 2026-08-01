# GAP-3 Inc-A — native multi-component pipeline decode (construction wiring)

- **Author:** Cohaagen (EP/runtime)
- **Branch:** `feat/gap3-inc-a-native-pipeline` (off `origin/main` @ `1c3a387f`)
- **Date:** 2026-07-31
- **Status:** implemented + green; awaiting independent review (Justin runs opus review + merge)
- **Scope:** construction wiring ONLY. No decode-loop / `native_decode/*` changes, no
  capture-core changes, no S2 (`decoder_component.rs:256` mirror_last_present_kv) touched.

## What was blocked

`build()` in `pipeline/mod.rs` unconditionally rejected the pure-native pipeline
selection (`PipelineBackend::Native`) at construction with
*"native pipeline decode is not yet implemented"* (old
`build_native_pipeline_and_report_gap`, returned at the native branch). This blocked
native decode for every multi-component model even though the decode loop
(`PipelineDecodeLoopBackend`, `pipeline/paged_decode.rs`) is already backend-agnostic
and multi-component, and the hybrid env-flag path already drives native components
through it.

## What was wired

Route the pure-native selection into the **already-working** flat-autoregressive
decode path (`run_autoregressive` → `PipelineDecodeLoopBackend`), building every
component as a native `ComponentSession` and the decoder as `NativePipelineDecoder` —
i.e. exactly what the hybrid env-flag path constructs, but driven by the **backend
selection** instead of env injection.

### Shared builder (DRY — Justin's standing directive)

The two native-selection sources now converge on ONE decision point and the SAME
builders (`build_step_component_session` / `build_native_pipeline_decoder`):

- New `struct NativeComponentSelection { decoder: bool, step_components: BTreeSet<String> }`
  (`pipeline/mod.rs`).
- New method `PipelineEngine::native_component_selection(&self, decoder, step_components)`
  (`pipeline/flat_autoregressive.rs`):
  - `EngineDecodeBackend::Native` → selects **every** component natively.
  - `EngineDecodeBackend::Ort` → consults the per-component env flags
    (`native_decoder_selected` / `native_step_component_set`), leaving the default
    ORT path byte-for-byte unchanged.
- `run_autoregressive` resolves the selection once and feeds both the decoder gate
  (`use_native_decoder`) and the step-component builder loop from it. No forked
  construction path; the env-flag hybrid path is unchanged.

### Construction changes (`pipeline/mod.rs build()`)

- Removed the early `return Err(build_native_pipeline_and_report_gap(...))` from the
  native branch. Without the `native-backend` feature → `native_backend_not_compiled_error()`.
  With the feature → fall through to the normal (ORT-shaped) construction; the decode
  loop then drives components natively per the selection.
- Renamed `build_native_pipeline_and_report_gap` → `native_pipeline_plan_unsupported`
  and rewrote its message. It still constructs every component via
  `build_native_pipeline_components` (no dead code) and is now returned **only** when
  a native-selected pipeline resolves to a non-flat-autoregressive plan:
  `if backend == Native && !matches!(plan, PipelinePlan::Autoregressive(_))`. So nested
  autoregressive / iterative-diffusion / single-pass / composite plans get a precise,
  actionable error instead of silently mis-routing.

### Non-paged guarantee (S2 never reached)

Pure-native sets `use_native_decoder = true` ⇒ `paged_enabled = false`
(`flat_autoregressive.rs`), so the non-paged branch is taken and
`decoder_component.rs:256` `mirror_last_present_kv` (S2) is **never** reached. Paging
is Inc-C.

## Correctness evidence (two-tier, token-exact)

Test: `tests/native_pipeline_backend_selection_parity.rs` →
`pure_native_pipeline_selection_matches_ort_and_hybrid` (single `#[test]`).

- **CPU 3-way parity** on `tiny-gemma4-vlm` (naive attention → ORT CPU is a real token
  oracle): `ORT == hybrid-env == pure-native`, all `[3,7] -> [0,5,6,7]`.
- **CUDA differential** (GPU-gated) on the task's named `tiny-gqa-embeds-cuda`:
  `pure-native == hybrid-env` (both native CUDA decoder), both `== [0,5,6,7]`. No
  ORT-CPU oracle for this fixture — its GQA op's `head_size % 8 == 2` is rejected by
  ORT's CPU kernel, so it is CUDA-only.

**Non-vacuity:** if construction reverts to the bail, the pure-native run
`?`-propagates and the test fails; if native selection diverges from ORT/hybrid the
token vectors differ and the asserts fire. The closed-form head `[3,7] -> [0,5,6,7]`
pins the expected ids.

Merged into ONE sequential `#[test]` on purpose: the native decoder device comes from
the process-global `ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE`, so two test fns would
race it under cargo's parallel test threads (only manifests with the `cuda` feature).

## Regressions re-run (all green)

- `native_pipeline_decoder_parity` (#384 hybrid decoder): ok
- `native_cuda_captured_step_inputs_parity` (#541): ok
- `native_full_pipeline_parity`: ok
- `qwen35_0_8b_hybrid_text_decode_e2e` (#543): ok
- `multimodal_reuse_e2e` (#554 session reuse): 14 passed
- `qwen35_0_8b_hybrid_native_cuda_e2e` (#543) / `weight_offload_native_cuda_e2e` (#544):
  ignored — env-gated on real model dirs (ignored pre-change too; no regression)
- lib unit tests `--features native-backend`: 343 passed
- Build WITHOUT `native-backend` (cfg gating), clippy `--features native-backend --tests`,
  `cargo fmt --all --check`: all clean.

## Blast radius / what this does NOT touch

- No changes to `native_decode/*`, the decode loop, or capture core
  (`provider.rs plan_capture_region`, `executor/capture.rs` — Mary's Scan lane). Zero
  file overlap with `feat/27b-scan-capture-1a`.
- Hybrid env-flag path (#543/#541), weight offload (#544), session-reuse (#554)
  behaviour unchanged (re-run above).

## Caveat / follow-up

Pure-native still loads ORT `PipelineModels` (the shared construction path), so a
genuinely ORT-**unloadable** block-quant model (`pkg.nxrt::BlockQuantizedMatMul`, which
is what auto-selects native via `model_proto_requires_native_backend`) cannot yet be
loaded native-only. A native-only loader is a follow-up; Inc-A is validated on
ORT-loadable fixtures via explicit `config.decode_backend = Native`.

## Handoff

- **Inc-B:** make the `prompt_only` prologue (e.g. `vision_encoder`) native — it stays
  on ORT under Inc-A.
- **Inc-C:** native present-KV mirroring / paging — implement
  `decoder_component.rs:256` `mirror_last_present_kv`, enabling the paged branch for
  native decode (unblocks Qwen3.6-35B-A3B MoE per the design note).
