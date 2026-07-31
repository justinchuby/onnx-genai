# Qwen3.5 hybrid loader unblock — text-only decode pipeline synthesis (#67, #384)

**Author:** Cohaagen (CUDA EP) · **Status:** loader gap CLOSED; native-CUDA last mile handed off to Mary

## Problem

The Qwen3.5-0.8B hybrid Foundry export is a 3-ONNX split package
(`vision.onnx` + `embedding.onnx` + `text.onnx`). Its declared image
preprocessing uses Qwen-style `smart_resize`, which has **no lossless runtime
encoding**. That `smart_resize` transform made `processor_program_json` error,
which aborted the ENTIRE pipeline-metadata synthesis
(`to_strict_pipeline_metadata`) before admission ran — so BOTH ORT and native
`Engine::from_pipeline_dir` refused the package and text decode could never run.
(`Engine::from_dir` also refuses it: 3 sibling `.onnx` files.)

## Fix (general, modality-driven — NOT a model-name special-case)

A split VLM package whose image path is unusable can still drive **text**
decode (text never touches vision). Implemented a **text-only decode pipeline
synthesis** that triggers purely on the package's declared shape:

1. `GenAiConfigError::UnrepresentablePreprocessing` — a NEW distinct variant
   returned by the `smart_resize` branch of `processor_program_json`. Kept
   separate from `IncompletePipeline` so genuinely-incomplete packages still
   fail hard; only *representable-but-unencodable* preprocessing becomes a
   text-only fallback.
2. `GenAiConfig::to_strict_text_only_pipeline_metadata` — synthesizes an
   embedding→decoder autoregressive pipeline with **no** vision component, image
   preprocessing, image dataflow, or image capabilities. Positions are declared
   rank-3 with `continuation: linear_increment` (every mrope axis advances with
   the sequence position → `[t,t,t]`, the correct pure-text coordinates), so no
   processor grid summary is needed. Decoder KV / hybrid loop-carried state
   (`conv_state`, `recurrent_state`) is validated against the ONNX graph exactly
   as the VLM path does. The decoder declares `sequence_source: inputs_embeds`
   (text.onnx has no token-id input), and the embedding's vision-fed
   `image_features` input is declared **optional** with an empty (zero
   image-token) absent value so a pure-text prompt never gathers from it.
3. `pipeline_inference_metadata_from_dir` catches `UnrepresentablePreprocessing`
   and falls back to the text-only synthesis. Existing VLMs with representable
   preprocessing are UNCHANGED (verified: `complete_config_synthesizes_typed_vlm_pipeline`
   and the native/ORT VLM pipeline parity tests still green).

### Second gap found + fixed (ORT decode path, NOT Mary's files)

Loop-carried fixed-state inputs (`conv_state [batch,6144,3]`,
`recurrent_state [batch,16,128,128]`) export the **leading batch axis as
symbolic** (`-1`). Zero-init rejected any non-concrete dim. Fixed in
`decode/values.rs` + `decode/resolved_io.rs`: the leading (batch) axis resolves
to the decode batch (`1`), mirroring the existing empty-KV convention in
`empty_past_value`; every NON-batch dim must still be concrete (a symbolic inner
extent is refused loudly). General, not hybrid-specific.

## Result — it RUNS and is COHERENT (ORT reference)

`Engine::from_pipeline_dir` on the real qwen3.5-0.8b hybrid now loads and greedy-
decodes end-to-end:

```
prompt : "The capital of France is"
output : " Paris, and the capital of Germany is Berlin.\nThe capital of France is"
tokens : [11751, 11, 321, 279, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 9338, 369]
```

Correct fact ("Paris") ⇒ the rank-3 mrope positions, `inputs_embeds` sequence
source, optional image_features, and hybrid loop-state init are all correct
(wrong positions/state would produce garbage). Locked by a committed ACTIVE
regression test `qwen35_0_8b_hybrid_text_decode_e2e.rs` (skips gracefully if the
Foundry model dir is absent, exact-locks the greedy stream + coherence oracle).

Reference caveat (honest): the reference is an ORT-driven decode of the SAME
synthesized spec (ORT places standard-attention layers on its EP and falls back
to CPU for the `com.microsoft` hybrid ops it doesn't implement). The per-op
CUDA↔reference parity is already proven separately (#480/#484/#525). No
independent onnxruntime-genai oracle is wired, so a *shared spec* bug wouldn't be
caught by this test alone; the coherence assertion is the mitigating oracle.

## HANDOFF to Mary (native_decode / Inc3c) — native-CUDA decoder last mile

The native-CUDA decoder cannot yet drive this model. The native step driver
hardcodes **rank-2** `position_ids`:

- `crates/onnx-genai-engine/src/native_decode/cuda.rs:248` —
  `Tensor::from_i64(&[1, token_ids.len()], &positions)`
- `crates/onnx-genai-engine/src/native_decode/cpu.rs:203` — same rank-2 shape.

The hybrid decoder declares **rank-3** mrope `position_ids [3, B, S]`, so the
native forward fails: `input position_ids: rank mismatch (graph declares rank 3,
got 2)`. These are Mary's active Inc3c files, so per the collision rule I did NOT
edit them. **Needed:** construct the rank-3 mrope coordinates honoring the
pipeline `positions` program (rank + `linear_increment` continuation) in the
native step driver, exactly as `decode/step.rs` already does for the ORT path
(`vec![absolute_start; rank]` → per-axis `[t,t,…]`). Once that lands, flip the
`qwen35_0_8b_hybrid_native_cuda_e2e` harness (#529 worktree) to native-CUDA vs
ORT token parity — the loader + synthesis + state-init plumbing is now proven.

## Files touched (all outside Mary's native_decode / pipeline decode-step)

- `crates/onnx-genai-genai-config/src/{lib.rs,compatibility.rs,json_builders.rs,loading.rs,tests.rs}`
- `crates/onnx-genai-genai-config/tests/{vlm_pipeline.rs,fixtures/vlm-smart-resize/*}`
- `crates/onnx-genai-engine/src/decode/{values.rs,resolved_io.rs,tests.rs}`
- `crates/onnx-genai-engine/tests/qwen35_0_8b_hybrid_text_decode_e2e.rs`
