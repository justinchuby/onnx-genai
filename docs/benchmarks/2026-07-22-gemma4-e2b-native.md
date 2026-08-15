# Gemma4 E2B native-package text decode — H200 (2026-07-22)

## Result

The Mobius four-model package reached coherent end-to-end greedy text decode through
`PipelineEngine` on the ONNX Runtime CUDA EP:

```text
prompt: <bos><|turn>user
The capital of France is<turn|>
<|turn>model

generated_text: "The capital of France is **Paris**."
steady_median: prefill=9.585 ms decode=7.138 ms/token throughput=140.09 tok/s
```

Measurement: H200, fp16, 32 generated steps, two warmups, five measured runs,
first eight emitted tokens excluded from the steady window. Runs were 140.20,
140.09, 140.04, 140.10, and 139.74 tok/s.

This is a native **onnx-genai package/pipeline** success, but not yet a pure-Rust
native-backend success: multi-model `PipelineEngine` still uses ORT sessions and
explicitly rejects `EngineDecodeBackend::Native`.

## Pipeline contract exercised

The package's every-step path is:

```text
token ids
  -> embedding.input_ids
  -> embedding.inputs_embeds [B,S,1536]
  -> decoder.inputs_embeds

  -> embedding.per_layer_inputs [B,S,8960]
  -> decoder.per_layer_inputs
  -> logits + 15 mixed-width K/V pairs
```

Both embedding outputs were refreshed on every prefill/decode step by the generic
step-component executor. No model-name branch or fixed Gemma dimension was added.

## Required package overlay

The package was exported at Mobius `640c1cb` before optional-modality PR #419.
Unmodified generation fails with:

```text
missing required pipeline input 'audio_encoder.input_features'
```

A metadata-only overlay declared `image_features` and `audio_features` optional
with zero fallbacks `[0, 1536]`, and gated the vision/audio encoders by opaque
`image`/`audio` presence keys. The existing generic schema, admission gate, and
runtime fallback materialization accepted this contract. The original model files
were symlinked unchanged.

Current Mobius `54be48a` emits the audio half of this contract, but
`image_features` and `vision_encoder` are not yet presence-gated for text-only
requests.

## CUDA bring-up details

The requested `onnxruntime-linux-x64-1.27.0/lib` directory does not contain
`libonnxruntime_providers_cuda.so`; using it silently selected CPU in the existing
runtime. The CUDA build is under `.ort-cuda-1.27/root/lib`. `profile_native
--pipeline --ep cuda` now verifies that the linked ORT exposes
`CUDAExecutionProvider` and fails rather than reporting a false GPU number.

ORT 1.27's optimized CUDA Attention path failed at the first decoder Attention
node with:

```text
CUDA error cudaErrorInvalidValue:invalid argument
```

The valid standard ONNX Attention graph ran successfully after disabling ORT's
optimized attention implementations:

```bash
export ORT_DISABLE_FLASH_ATTENTION=1
export ORT_DISABLE_LEAN_ATTENTION=1
export ORT_DISABLE_FUSED_ATTENTION=1
export ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION=1
export ORT_DISABLE_CUDNN_FLASH_ATTENTION=1
```

## Reproduction

Build:

```bash
cargo build --release -p onnx-genai-bench \
  --features bench-native,cuda --bin profile_native
```

Run with the CUDA-enabled ORT directory first in `LD_LIBRARY_PATH`,
`ONNX_GENAI_EP=cuda`, the five attention-disable variables above, and:

```bash
target/release/profile_native \
  --pipeline --steady --decode-skip 8 \
  --model target/gemma4-e2b-text --ep cuda \
  --prompt $'<bos><|turn>user\nThe capital of France is<turn|>\n<|turn>model\n' \
  --tokens 32 --warmups 2 --runs 5
```

## Remaining gaps and file-disjoint work packages

1. **Text-only package closure (Mobius, prerequisite):**
   `src/mobius/tasks/_gemma4.py` should declare the image embedding input optional
   and gate the vision component, mirroring generic audio presence semantics.
   Tests belong in `tests/build_graph_test.py` and
   `src/mobius/integrations/onnx_genai/inference_metadata_test.py`. Re-export the
   package from `54be48a` or later.
2. **Pure-Rust multi-model execution:** replace the ORT-only component-session
   ownership in `crates/onnx-genai-engine/src/pipeline.rs` with a backend-neutral
   pipeline session interface, then load native sessions for every component.
   This is the blocker at `pipeline.rs:189-208`.
3. **CUDA Attention correctness (upstream ORT, independent of package/runtime
   files):** reproduce and fix optimized CUDA dispatch for the standard fp16 GQA
   Attention signature (`q_heads=8`, `kv_heads=1`, head size 256). Keep the five
   disable variables as an explicit workaround until fixed.
4. **Future VLM image path:** validate real Gemma4 preprocessing and placeholder
   expansion using `crates/onnx-genai-server/src/image_input.rs` plus typed image
   metadata. This is independent of the text-only and native-session work.
5. **Future audio path:** add typed request construction for fp16 features plus
   bool masks using `crates/onnx-genai-preprocess/src/audio.rs` and
   `crates/onnx-genai-server/src/driver.rs`; validate the rank-2 masked feature
   adapter emitted by current Mobius.
