# SmolVLM-256M inference-metadata probe

This is a validated metadata example for the real
[`HuggingFaceTB/SmolVLM-256M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
ONNX export at revision `7e3e67edbbed1bf9888184d9df282b700a323964`
(Apache-2.0). It is the 256M Idefics3-derived VLM with a 93M SigLIP vision
encoder and SmolLM2-135M text decoder.

The compact files inspected locally were:

| Component | Bytes | SHA-256 |
|---|---:|---|
| `vision_encoder_q4f16.onnx` | 55,037,584 | `13c301dc0cbc79c8c23badc699129b47d05ba044e2274ca8a94681fcd4205fc1` |
| `embed_tokens_fp16.onnx` | 56,770,946 | `019af59078b8e4775216dd278c66574c39bfefa36cab68573db64d7ac7a11691` |
| `decoder_model_merged_q4f16.onnx` | 77,034,560 | `0a6b4e72e26025c190f37e2573ea2ce3336d95670da9c4b1fa662e47cc3d94a7` |

## Exact ONNX interfaces

- Vision encoder:
  - `pixel_values`: float32 `[batch, num_images, 3, 512, 512]`
  - `pixel_attention_mask`: bool `[batch, num_images, 512, 512]`
  - `image_features`: float32 `[effective_images, 64, 576]`
- Token embedding:
  - `input_ids`: int64 `[batch, sequence]`
  - `inputs_embeds`: float32 `[batch, sequence, 576]`
- Decoder:
  - `inputs_embeds`: float32 `[batch, sequence, 576]`
  - `attention_mask`: int64 `[batch, total_sequence]`
  - `position_ids`: int64 `[batch, sequence]`
  - 30 K/V input pairs `past_key_values.{0..29}.{key,value}`:
    float16 `[batch, 3, past_sequence, 64]`
  - `logits`: float32 `[batch, sequence, 49280]`
  - 30 K/V output pairs `present.{0..29}.{key,value}`:
    float16 `[batch, 3, total_sequence, 64]`

The published processor resizes to a 2048-pixel longest edge, splits into at
most sixteen 512×512 local tiles, appends a global tile, and emits 64 image
tokens per tile. A 1024×512 input produced nine image tensors and the token
program:

```text
(<fake_token_around_image> <row_R_col_C> <image>×64)×8
(<fake_token_around_image> <global-img> <image>×64)
<fake_token_around_image>
```

`<image>` is token 49190. `<fake_token_around_image>` is 49189; it is both a
separator and the final wrapper, not a dedicated vision-start token. The model
publishes no `vision_start_token_id`.

## Fit

- `PipelineSpec.models`, `strategy`, and `phases` express vision
  `prompt_only`, embedding `every_step`, and autoregressive decode.
- `preprocessing.image.outputs` binds both real vision inputs, including the
  bool validity mask.
- `SequenceInputKind::TokenIds` describes the embedding graph;
  `InputsEmbeds` describes the decoder.
- `KvOwnership::Owned`, ordered `kv_inputs`/`kv_outputs`, and `kv_update:
  append` exactly describe all 60 decoder cache ports.
- `PipelineVisionConfig` captures image token 49190, 64 tokens per tile,
  prompt-order image correspondence, and appended global thumbnail.

## Gaps

1. **Host fusion is not expressible.** Upstream performs
   `inputs_embeds[input_ids == 49190] = image_features.reshape(-1, 576)`.
   `PipelineSpec.models` only admits ONNX files, while `dataflow` only routes
   whole tensors; it cannot declare indexed scatter/replacement or a host
   operation. Therefore this metadata validates but is not an executable
   end-to-end package without adding a fusion ONNX graph or a typed fusion
   operation.
2. **The image-token grammar is too simple.** `PipelineVisionConfig` can repeat
   one image token and declare one row/column separator token, but cannot emit
   position-dependent `<row_R_col_C>` IDs (49153–49188), `<global-img>` 49152,
   or the repeated/final 49189 wrapper.
3. **No generic typed component port inventory.** `ModelIoSpec` covers decoder
   semantic ports, but cannot enumerate arbitrary component inputs/outputs with
   dtype, rank, and shape. Vision shapes and `image_features` are documentation,
   not typed metadata.
4. **Preprocessing resize semantics are open strings.** `size: 2048` plus
   `mode: longest_edge` parses, but `ImageSizeSpec` has no typed
   `longest_edge` form, and the schema cannot fully state the processor's
   aspect-ratio/grid selection algorithm.
5. **No truthful `vision_start_token_id`.** Token 49189 is a bidirectional
   wrapper/separator, so assigning it to the one-way special-token field would
   misrepresent the model.

Full ORT generation is future work after gaps 1–2 are represented. The checked
test proves typed YAML parsing, runtime-capability validation, pipeline DAG
validation, and JSON Schema validation.
