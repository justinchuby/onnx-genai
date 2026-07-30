# Qwen3.5-0.8B hybrid — native CUDA end-to-end decode integration (issue #67 / #384)

**Author:** Cohaagen (CUDA EP) · **Date:** 2026-07-30 · **Branch:** `squad/qwen35-hybrid-cuda-e2e`

## TL;DR

- **Whole-graph CUDA placement: PROVEN — 100%, zero declines.** The entire
  Qwen3.5-0.8B hybrid decode graph (`embedding.onnx` 24 nodes + `text.onnx` 1265
  nodes = **1289 nodes**, control-flow bodies recursed) is claimable on the
  native CUDA EP with **zero declines** on current `origin/main` (all coverage
  merged: #480 `CausalConvWithState` ×18 + `GatherBlockQuantized` ×1, #484
  `LinearAttention` ×18, **#525** `com.microsoft::RotaryEmbedding` ×12 + `Bool`
  `NonZero` ×1). This is the composition proof that the per-op coverage composes
  over a real model.
- **End-to-end native-CUDA *decode*: BLOCKED on loader/preprocessing plumbing
  outside #67's CUDA-EP op scope** (details below). No public high-level engine
  entry loads this split hybrid package today, so a token-parity decode run
  could not be executed. Honest stopping point: landed the placement lock + this
  blocker writeup; the decode harness is committed and skips gracefully until
  the loader gap closes.

## What landed

1. `crates/onnx-runtime-ep-cuda/tests/qwen35_0_8b_placement_lock.rs` — a
   GPU+model-gated (`#[ignore]`) regression lock that walks both graphs through
   the **native CUDA claim gate** (`ExecutionProvider::supports_op`, the exact
   gate the placement pass uses) and asserts:
   - `CausalConvWithState` = 18 nodes, all claim (#480);
   - `LinearAttention` = 18 nodes, all claim (#484);
   - `GatherBlockQuantized` present, claims (#480);
   - the whole graph places on CUDA with **zero declines** — any decline (a
     regressed covered op) fails the test.
   Green on current `origin/main` (all of #480/#484/#525 merged): **1289 nodes,
   0 declines, 100% placement.**
2. `crates/onnx-genai-engine/tests/qwen35_0_8b_hybrid_native_cuda_e2e.rs` — the
   end-to-end token-parity harness. It drives the split model through
   `Engine::from_pipeline_dir` with the decoder component pinned to the native
   CUDA EP (`ONNX_GENAI_PIPELINE_NATIVE_DECODER=decoder` +
   `..._DEVICE=cuda:0`, the inc3a device-KV `inputs_embeds` path) and compares
   greedy token ids against the ORT reference. It **skips gracefully** on the
   known loader blocker below and becomes a live parity lock the moment the
   package loads.

## The blocker (why a decode run could not execute)

The Foundry package is a **split** model: `embedding.onnx` + `text.onnx` +
`vision.onnx` + `genai_config.json`. Neither public high-level entry loads it:

- `Engine::from_dir` → `resolve_model_path` rejects it: *"multiple .onnx files
  found; expected decoder.onnx or exactly one .onnx file"* (single-model loader).
- `Engine::from_pipeline_dir` → the compatibility pipeline admission refuses it
  during **vision preprocessing** synthesis: *"processor
  `Resize.attrs.smart_resize=true` is not representable by the runtime's
  stretch/crop/pad resize modes"*. The Qwen2.5-VL-style smart-resize image
  preprocessing cannot be synthesized, so the **whole** pipeline is declined —
  even though a text-only decode never runs the vision front-end.
- Even past admission, a full-pipeline `EngineDecodeBackend::Native` decode is
  the GAP-3 item (`build_native_pipeline_and_report_gap`); the per-component
  native-decoder seam (`ONNX_GENAI_PIPELINE_NATIVE_DECODER`) is the intended
  path and is reachable only *through* the pipeline engine, which the vision
  admission blocks first.

The decoder itself (`text.onnx`) has **no token-id input** — its sequence
source is `inputs_embeds` (`[B, S, 1024]`), fed by `embedding.onnx`. The
`NativeDecodeSession` inputs_embeds step (`decode_with_step_inputs`) is
`pub(crate)`, so a decode cannot be driven from an integration test without the
pipeline orchestration either.

All three are **loader / preprocessing-metadata / pipeline plumbing** gaps,
independent of CUDA-EP op coverage (#67), which is complete for this op set.

## Follow-up to reach a token-parity decode lock

Any **one** of these unblocks the committed e2e harness (it then runs as-is):

1. **Preferred:** teach the compatibility pipeline loader to admit a
   text-only decode of a VLM package when no image input is supplied — i.e. do
   not require representable vision preprocessing to build the
   embedding+decoder AR path (defer vision admission until an image is actually
   routed). Owner: pipeline/loader (Mary's area).
2. Ship a native `inference_metadata.json` for the package that declares the
   preprocessing explicitly (removes the smart-resize synthesis gap).
3. Expose a public inputs_embeds decode driver (embedding component → decoder
   `inputs_embeds` step) so the two components can be driven directly.

## Reproduce

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh; export CUDA_VISIBLE_DEVICES=0
QWEN35_0_8B_DIR=/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-0.8b-generic-cpu-2/v2 \
  cargo test -p onnx-runtime-ep-cuda --features cuda \
  --test qwen35_0_8b_placement_lock -- --ignored --nocapture
```
