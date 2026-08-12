#!/usr/bin/env python3
"""FLOAT16-KV twin of ``build_tiny_gemma4_vlm_cuda.py`` (GAP-3 Inc-D.1).

Identical closed-form behaviour to ``tiny-gemma4-vlm-cuda`` (prompt ``[3, 7]`` ->
generated ``[0, 5, 6, 7]``) and the same composite pipeline, with ONE decisive
difference: the decoder's KV cache — the ``past_key_values.0.{key,value}`` inputs
and the ``present.0.{key,value}`` outputs — is **FLOAT16**, so the native CUDA
decoder's KV *bindings* are device-resident f16.

Why this fixture exists
-----------------------
Inc-D landed device-resident **f32** rank-4 present-KV onto the paged path; real
exports (gemma4-e2b decoder, likely qwen3-30b-a3b) keep KV in **f16**, so they
hit the f32-only gate and fall back to non-paged. This fixture is the f16 oracle
for Inc-D.1: paged-native-CUDA-f16 == non-paged-native == ORT == closed-form
tokens, and the mirrored paged KV is **byte-equal** to ORT's (both land f32 in
the shared host paged store via the same ``half`` f16->f32 widening).

It is a **naive Concat-KV** decoder (like ``tiny-gemma4-vlm-cuda``), NOT a
GroupQueryAttention decoder, for the same reason Inc-D chose this base: the ORT
**CPU** GQA kernel rejects the tiny fixture's ``head_size``, leaving GQA fixtures
with no ORT oracle, whereas the Concat cache runs under ORT CPU — supplying the
token + byte-equality oracle. The physical-vs-logical KV *stride* geometry stays
covered by the deterministic ``H == 2`` unit test
``device_kv_view_uses_physical_stride`` (this fixture has a single KV head).

Determinism seam
----------------
The token/logits path is bit-stable and independent of the KV path:

  * ``logits = inputs_embeds @ LM_HEAD + tie_bias`` (f32) — computed directly from
    ``inputs_embeds``, so tokens stay exactly ``[0, 5, 6, 7]`` and the fixture's
    argmax is invariant to the reused-prefix KV (a zeroed mirror still yields the
    same tokens — the byte-equality assert is what catches an f16 value error).
  * The KV path casts ``inputs_embeds`` to f16 (``current_key``) and scales it by
    a f16 ``2.0`` (``current_value = key * 2``), concatenated onto the f16 past
    cache. Cast and ×2 are **arithmetic-exact** in f16 (a ×2 only increments the
    exponent), so the stored f16 ``present`` bytes are bit-identical under the
    native CUDA and ORT CPU kernels — the basis of the byte-equality oracle. (An
    additive offset would round differently between the two f16 Add kernels.)

Usage:
    python scripts/build_tiny_gemma4_vlm_cuda_f16.py [out_dir] [--no-validate]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from onnxscript import ir

from build_tiny_gemma4_vlm import (  # type: ignore
    HIDDEN,
    LM_HEAD,
    TIE_BIAS,
    VOCAB,
    build_embedding,
    build_vision_encoder,
    compute_expected_tokens,
    constant,
    initializer,
    node,
    save_model,
    tensor_value,
    tiny_pixels,
    write_tokenizer,
    _textproto_bytes,
)


def build_decoder_cuda_f16(path: Path) -> None:
    """inputs_embeds[1,s,4] (+ mask, position_ids, f16 KV) -> logits + f16 KV.

    Same head as the base fixture (``logits = inputs_embeds @ W + tie_bias``, f32)
    with a growing ``Concat`` KV cache in **FLOAT16**. ``attention_mask`` and
    ``position_ids`` are declared graph inputs (required so the native CUDA decode
    path can bind its device mask) but left unconsumed, so the generated tokens
    stay identical to the base fixture.
    """
    inputs_embeds = tensor_value(
        "inputs_embeds", ir.DataType.FLOAT, [1, "sequence", HIDDEN]
    )
    attention_mask = tensor_value(
        "attention_mask", ir.DataType.INT64, [1, "total_sequence"]
    )
    position_ids = tensor_value("position_ids", ir.DataType.INT64, [1, "sequence"])
    past_key = tensor_value(
        "past_key_values.0.key", ir.DataType.FLOAT16, [1, 1, "past_sequence", HIDDEN]
    )
    past_value = tensor_value(
        "past_key_values.0.value", ir.DataType.FLOAT16, [1, 1, "past_sequence", HIDDEN]
    )

    # ── Token path: deterministic, bit-stable across CPU/CUDA (f32). ──────────
    lm_head = initializer("lm_head", LM_HEAD)
    matmul = node("MatMul", [inputs_embeds, lm_head], "logits_base")  # [1, s, 8]
    tie_bias = initializer("tie_bias", TIE_BIAS)
    logits = node("Add", [matmul.outputs[0], tie_bias], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape([1, "sequence", VOCAB])

    # ── KV path: growing Concat cache in FLOAT16 (the device-resident seam). ──
    # current key/value are [1, 1, s, 4] f16; append to the f16 past cache.
    # value = key * 2 (NOT key + 0.5): multiplying an f16 by 2 only increments the
    # exponent, so it is bit-exact on every kernel (CPU + CUDA). An additive offset
    # would round differently between the ORT CPU and native CUDA f16 Add kernels
    # (round-to-nearest-even midpoints diverge), breaking the byte-equality oracle;
    # the Cast (key) and ×2 (value) paths are arithmetic-exact, so the stored f16
    # present bytes are identical on both kernels — the basis of byte-equality.
    to_f16 = ir.AttrInt64("to", int(ir.DataType.FLOAT16))
    kv_axis = constant("kv_axis", np.array([1], dtype=np.int64))
    current_key_f32 = node("Unsqueeze", [inputs_embeds, kv_axis.outputs[0]], "current_key_f32")
    current_key = node("Cast", [current_key_f32.outputs[0]], "current_key", attributes=[to_f16])
    value_scale = initializer("value_scale", np.array(2.0, dtype=np.float16))
    current_value = node("Mul", [current_key.outputs[0], value_scale], "current_value")
    present_key = node(
        "Concat",
        [past_key, current_key.outputs[0]],
        "present.0.key",
        attributes=[ir.AttrInt64("axis", 2)],
    )
    present_value = node(
        "Concat",
        [past_value, current_value.outputs[0]],
        "present.0.value",
        attributes=[ir.AttrInt64("axis", 2)],
    )
    for out in (present_key.outputs[0], present_value.outputs[0]):
        out.type = ir.TensorType(ir.DataType.FLOAT16)
        out.shape = ir.Shape([1, 1, "total_sequence", HIDDEN])

    graph = ir.Graph(
        [inputs_embeds, attention_mask, position_ids, past_key, past_value],
        [logits.outputs[0], present_key.outputs[0], present_value.outputs[0]],
        nodes=[
            matmul,
            logits,
            kv_axis,
            current_key_f32,
            current_key,
            current_value,
            present_key,
            present_value,
        ],
        initializers=[lm_head, tie_bias, value_scale],
        opset_imports={"": 13},
        name="tiny_gemma4_decoder_cuda_f16",
    )
    save_model(
        ir.Model(
            graph, ir_version=8, producer_name="onnx-genai tiny-gemma4-vlm-cuda-f16 fixture"
        ),
        path,
    )


METADATA = """\
# FLOAT16-KV twin of tiny-gemma4-vlm-cuda (GAP-3 Inc-D.1). Identical closed-form
# tokens ([3, 7] -> [0, 5, 6, 7]) and pipeline shape, but the decoder's KV cache
# (past_key_values.0.{key,value} inputs, present.0.{key,value} outputs) is
# FLOAT16, so the native CUDA decoder's KV bindings are device-resident f16.
# Proves Inc-D.1 f16 device present-KV read-out / seed onto the paged path.
# Built by build_tiny_gemma4_vlm_cuda_f16.py.
pipeline:
  models:
    vision_encoder:
      filename: vision_encoder.onnx.textproto
      type: vision_encoder
    embedding:
      filename: embedding.onnx.textproto
      type: encoder
      io:
        token_input: input_ids
    decoder:
      filename: decoder.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
      io:
        sequence_source: inputs_embeds
        inputs_embeds_input: inputs_embeds
        attention_mask_input: attention_mask
        position_ids_input: position_ids
        logits_output: logits
        kv_inputs:
          - past_key_values.0.key
          - past_key_values.0.value
        kv_outputs:
          - present.0.key
          - present.0.value
  dataflow:
    - from: vision_encoder.image_features
      to: embedding.image_features
      dtype: fp32
      device_transfer: false
    - from: embedding.inputs_embeds
      to: decoder.inputs_embeds
      dtype: fp32
      device_transfer: false
  strategy:
    kind: composite
    stages:
      - name: encode_vision
        strategy:
          kind: single_pass
          model: vision_encoder
      - name: fuse_embeddings
        strategy:
          kind: single_pass
          model: embedding
      - name: decode
        strategy:
          kind: autoregressive
          decoder: decoder
          max_tokens: 4
  phases:
    vision_encoder:
      run_on: prompt_only
    embedding:
      run_on: every_step
    decoder:
      run_on: every_step
"""


def validate_with_ort(output_dir: Path, prompt: list[int], max_new_tokens: int) -> list[int]:
    """ORT reference decode for the f16-KV CUDA fixture (supplies mask + pos ids)."""
    import onnxruntime as ort

    vision = ort.InferenceSession(
        _textproto_bytes(output_dir / "vision_encoder.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )
    embedding = ort.InferenceSession(
        _textproto_bytes(output_dir / "embedding.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )
    decoder = ort.InferenceSession(
        _textproto_bytes(output_dir / "decoder.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )

    pixels = tiny_pixels()
    image_features = vision.run(None, {"pixel_values": pixels})[0]

    def embed(ids: list[int]) -> np.ndarray:
        return embedding.run(
            None,
            {"input_ids": np.array([ids], dtype=np.int64), "image_features": image_features},
        )[0]

    def decode(embeds: np.ndarray, past_k, past_v, past_len):
        seq = embeds.shape[1]
        total = past_len + seq
        return decoder.run(
            None,
            {
                "inputs_embeds": embeds.astype(np.float32),
                "attention_mask": np.ones((1, total), dtype=np.int64),
                "position_ids": np.array([list(range(past_len, total))], dtype=np.int64),
                "past_key_values.0.key": past_k,
                "past_key_values.0.value": past_v,
            },
        )

    past_k = np.zeros((1, 1, 0, HIDDEN), dtype=np.float16)
    past_v = np.zeros((1, 1, 0, HIDDEN), dtype=np.float16)
    logits, past_k, past_v = decode(embed(prompt), past_k, past_v, 0)
    past_len = len(prompt)
    generated = [int(logits[0, -1].argmax())]
    for _ in range(1, max_new_tokens):
        logits, past_k, past_v = decode(embed([generated[-1]]), past_k, past_v, past_len)
        past_len += 1
        generated.append(int(logits[0, -1].argmax()))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=str(
            Path(__file__).resolve().parent.parent
            / "tests/fixtures/tiny-gemma4-vlm-cuda-f16"
        ),
        help="output fixture directory",
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_vision_encoder(out_dir / "vision_encoder.onnx.textproto")
    build_embedding(out_dir / "embedding.onnx.textproto")
    build_decoder_cuda_f16(out_dir / "decoder.onnx.textproto")
    write_tokenizer(out_dir / "tokenizer.json")
    (out_dir / "inference_metadata.yaml").write_text(METADATA)

    prompt = [3, 7]
    max_new_tokens = 4
    expected, _ = compute_expected_tokens(prompt, max_new_tokens)
    print(f"closed-form expected tokens: {expected}")
    if not args.no_validate:
        got = validate_with_ort(out_dir, prompt, max_new_tokens)
        assert got == expected, f"ORT decode {got} != closed form {expected}"
        print(f"ORT validation passed: {got}")
    print(f"wrote f16-KV CUDA fixture to {out_dir}")


if __name__ == "__main__":
    main()
