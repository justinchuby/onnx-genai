# Decision: run Muse-Glimmer embedding component on the native CUDA EP (Blocker 1 — LOAD)

**Author:** Batty (Engine Dev)
**Branch:** `squad/native-pipeline-embedding`
**Date:** 2026-08-12
**Status:** proposed (coordinator admin-merges after review)

## Summary
Made `PipelineEngine` load and decode Muse-Glimmer-30B end-to-end on the native
CUDA execution provider. Previously the model would not load on any native
decode path: the multimodal (vision + embedding + decoder) pipeline routed its
**embedding** component to ORT, which lacks a bf16 `Where(16)` implementation, so
load failed. The native CUDA EP can run the whole model (the `muse_decode` bench
harness proves it with raw sessions); the gap was purely engine pipeline
plumbing. This unblocks Sebastian (Perf) to attempt CUDA-graph capture.

## What changed (file:line)
1. **Embeds-producer reclassification** — `pipeline/mod.rs` `component_phase()` +
   new `decoder_embeds_producer()`. An embeds-driven decoder
   (`sequence_source: inputs_embeds`) must be fed a fresh embedding for the
   single running token every step, so the upstream component that produces its
   `inputs_embeds` is promoted `prompt_only → every_step`. Reuses all existing
   `every_step` machinery (build_step_bindings / run_step_components / decoder
   in-edge refresh). Scoped to the single dataflow producer of the decoder's
   `inputs_embeds` port, so cached conditioning producers (e.g. vision → image
   features) keep their declared `prompt_only` phase.
2. **Native device for every_step components** — `pipeline/mod.rs`
   `build_step_component_session()` (+ `flat_autoregressive.rs` caller). The
   native embedder now loads on the same device the native decoder targets
   (CUDA), so an embeds-driven pipeline runs its per-token embedding on the CUDA
   EP next to the decoder, mirroring `muse_decode`. Only one small `inputs_embeds`
   row round-trips host↔device per step; the decoder KV never does.
3. **Skip ORT sessions for all components on native backend** — `pipeline/mod.rs`
   `native_ort_skips` (Autoregressive plan). Prompt-phase components (vision
   encoder, embeds fuser) no longer build ORT sessions on the native backend
   (which would reject bf16 `Where` / int4 `MatMulNBits`). They run natively in
   the prologue.
4. **Lazy native prologue + inactive-component skip** — `pipeline/routing.rs`
   `run_prompt_phase_components()` native branch + `prompt_component_inputs_available()`,
   `run_native_prompt_component()`, `ensure_native_prompt_session()`. Prompt
   components with no ORT session run through a lazily-built `NativeComponentSession`
   on the native device. A component whose inputs are unavailable (a vision
   encoder on a text-only prompt) is inactive and skipped — its weights are never
   materialized.
5. **Empty image-features seed** — `pipeline/routing.rs`
   `seed_absent_step_component_inputs()` (called from `flat_autoregressive.rs`).
   For a text-only prompt the embedder still requires its `image_features` graph
   input bound; seeds an empty `[0, hidden]` tensor once — exactly the empty
   image feed `muse_decode` sends every step. Only seeds inputs with a dynamic
   axis; fully-static required inputs still error precisely at run.
6. **bf16 acceptance on the native CUDA decode target** — `native_decode/load.rs`
   (inputs_embeds) and `native_decode/cuda.rs` (KV inputs, decoder state, logits)
   gates relaxed from `f32|f16` to `f32|f16|bf16`. Muse-Glimmer's decoder is bf16
   throughout. Verified by exact greedy parity (below); the bf16→f32 logits path
   already existed (`native_decode/tensor.rs` `tensor_to_f32`) and the device
   sampler already supports bf16.
7. **KV context ceiling plumbing** — `pipeline/mod.rs` `pipeline_metadata_max_len()`
   threaded through `NativePipelineDecoder::load` →
   `NativeDecodeSession::load_with_io[_and_cuda_governor]`. The decoder's
   per-directory model path (`.../decoder/model.onnx`) has no metadata sidecar,
   so CUDA KV capacity fell back to `usize::MAX` and the mask reservation
   overflowed. Now reads the pipeline package's `model.max_sequence_length`
   (131072 for Muse-Glimmer) from the native `inference_metadata.yaml`, falling
   back to `genai_config.json` context length.

## Verification (load + decode + parity)
Model loads and decodes end-to-end on the native engine path (no env override
needed):

```
profile_native --pipeline --model <muse> --ep cuda --backend native --tokens 16
  → generated_token_ids: [721, 130869, 198, 51015, 1780, 262, 24627, 4961,
                          373, 310, 1472, 32263, 335, 45, 24627, 4961]
```

**Parity vs `muse_decode` (raw native sessions), greedy, two prompts:**
- "Explain what a neural network is in two sentences." → all 16 tokens identical.
- "List three primary colors." → first 16 tokens identical (incl. EOS 200001).

`cargo test -p onnx-genai-engine --features native-backend --lib native_decode::`
= 78 passed; `pipeline::` = 46 passed. `cargo fmt --all -- --check` clean;
`cargo clippy` clean (only a pre-existing `too_many_arguments` warning in
`engine/governor.rs`, untouched).

## Notes for Deckard (Blocker 2)
- I did **NOT** touch `decode/metadata.rs` or the decode-path selection. My
  changes are confined to pipeline plan classification (`component_phase`),
  pipeline session loading / prologue, step-component device+seeding, and the
  bf16 gate relaxations in `native_decode/{load,cuda}.rs`. No overlap with your
  SWA misclassification work.

## Notes for Sebastian (Perf / capture)
- **bf16 KV falls to the non-paged fallback.** `DecodeCudaState::kv_bindings_paged_rank4()`
  still gates the paged KV store to `f32|f16` (I left it unchanged), so bf16
  decoders (Muse-Glimmer) use the non-paged device-resident KV path. That path
  produced exact parity, but paged KV mirroring / prefix cache do not engage for
  bf16. If CUDA-graph capture eligibility depends on the paged path for bf16,
  extending paged KV to bf16 is **KV & Buffers (Leon) domain** — flag it and I
  can pair. The role-collision at `native_decode/load.rs:455-612` did **not**
  occur on the pipeline path (metadata declares `sequence_source: inputs_embeds`),
  so no change was needed there.
- The `every_step` embedder is a genuine per-step native CUDA EP forward, so your
  ~1600 launches/token count now includes the embedding graph's launches too.
