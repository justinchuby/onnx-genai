#!/usr/bin/env python3
"""Build the tiny deterministic Gemma4-style VLM composite pipeline fixture.

A real Gemma-3/Gemma-4 multimodal checkpoint is emitted by Mobius as a
three-model graph that fuses image and text in **embedding space** before the
autoregressive decoder ever runs:

  * ``vision_encoder``: ``pixel_values -> image_features`` (image tokens), then
  * ``embedding`` (fusion): ``input_ids + image_features -> inputs_embeds``
    — text token embeddings with the image features scattered into the image
    placeholder positions, and finally
  * ``decoder``: ``inputs_embeds -> logits + KV`` — an autoregressive decoder
    whose **prompt** input is ``inputs_embeds`` (not raw ``input_ids``).

onnx-genai runs this as a ``composite`` pipeline (DESIGN.md §20): two
``prompt_only`` single-pass stages (vision + fusion) feed an ``autoregressive``
decode stage that consumes ``inputs_embeds``. The crux this fixture proves,
that the tiny-vlm 2-model fixture does *not*, is **inputs_embeds fusion**:

  1. the prompt token ids must be seeded into the shared pool as
     ``embedding.input_ids`` so the fusion model can run in the prompt phase, and
  2. each decode step must re-embed the *generated* token through the fusion
     model to produce that step's single-token ``inputs_embeds`` — the decoder
     has no ``input_ids`` input of its own.

This script builds the smallest possible trio with onnx-ir so the pipeline
*contract* (not VLM quality) can be tested end to end and asserted against exact
token ids.

Closed form (H = 4 hidden, V = 8 vocab, placeholder token = 7):

  vision_encoder: pixel_values[1,3,2,2] -> image_features[1,1,4]
                  image_features[h] = mean_c pixel_values[0, c, h]   (spatial-major)

  embedding:      inputs_embeds[1,s,4] = E[input_ids] + 1{input_ids==7} * image_features
                  E is the fixed token embedding table below.

  decoder:        logits[1,s,8] = inputs_embeds @ W + tie_bias
                  W[:, v] = E[(v - 1) mod 8]   (a "predict the next token by
                  embedding similarity" head), tie_bias[v] = v * 1e-3 breaks
                  exact integer-dot ties identically in ORT and the engine.

So for a pure token-``t`` embedding the head peaks at ``t + 1 (mod 8)``; the
first generated token additionally sees the fused image features at the trailing
placeholder position, proving vision -> fusion -> decode end to end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir

HIDDEN = 4
VOCAB = 8
PLACEHOLDER_ID = 7

# Fixed token embedding table E[token, hidden] (V x H). Rows 0..3 are the unit
# axes; rows 4..7 are simple two-hot combinations. Chosen so the decoder head
# (embedding similarity, shifted by one) yields a deterministic token chain.
EMBEDDING_TABLE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # 0
        [0.0, 1.0, 0.0, 0.0],  # 1
        [0.0, 0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 0.0, 1.0],  # 3
        [1.0, 1.0, 0.0, 0.0],  # 4
        [0.0, 1.0, 1.0, 0.0],  # 5
        [0.0, 0.0, 1.0, 1.0],  # 6
        [1.0, 0.0, 0.0, 1.0],  # 7 (image placeholder)
    ],
    dtype=np.float32,
)

# lm_head W[hidden, vocab] with W[:, v] = E[(v - 1) mod 8], so
# logits[v] = <inputs_embeds, E[(v-1) mod 8]> peaks at v = t + 1 for input E[t].
LM_HEAD = np.stack(
    [EMBEDDING_TABLE[(v - 1) % VOCAB] for v in range(VOCAB)], axis=1
).astype(np.float32)

# Deterministic tie-breaker: strictly increasing, far smaller than any integer
# dot-product gap, so ORT (this script) and the Rust engine pick the same argmax.
TIE_BIAS = (np.arange(VOCAB, dtype=np.float32) * 1e-3).reshape(1, 1, VOCAB)


def tensor_value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def initializer(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.Tensor(array, name=name))


def node(
    op_type: str,
    inputs: list[ir.Value],
    output: str,
    *,
    attributes: tuple[ir.Attr, ...] | list[ir.Attr] = (),
) -> ir.Node:
    return ir.Node("", op_type, inputs, attributes, outputs=[ir.Value(name=output)])


def constant(name: str, array: np.ndarray) -> ir.Node:
    return node(
        "Constant",
        [],
        name,
        attributes=[ir.AttrTensor("value", ir.Tensor(array, name=f"{name}_value"))],
    )


def save_model(model: ir.Model, path: Path) -> None:
    ir.save(model, path, format="textproto")
    onnx.checker.check_model(ir.to_proto(model))


def build_vision_encoder(path: Path) -> None:
    """pixel_values[1,3,2,2] -> image_features[1,1,4], mean over channels."""
    pixel_values = tensor_value("pixel_values", ir.DataType.FLOAT, [1, 3, 2, 2])
    flat_shape = constant("flat_shape", np.array([1, 3, HIDDEN], dtype=np.int64))
    flat = node("Reshape", [pixel_values, flat_shape.outputs[0]], "pixels_flat")
    channel_mean = node(
        "ReduceMean",
        [flat.outputs[0]],
        "channel_mean",
        attributes=[ir.AttrInt64s("axes", [1]), ir.AttrInt64("keepdims", 1)],
    )
    image_features = channel_mean
    image_features.outputs[0].name = "image_features"
    image_features.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    image_features.outputs[0].shape = ir.Shape([1, 1, HIDDEN])

    graph = ir.Graph(
        [pixel_values],
        [image_features.outputs[0]],
        nodes=[flat_shape, flat, channel_mean],
        opset_imports={"": 13},
        name="tiny_gemma4_vision_encoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-gemma4-vlm fixture"),
        path,
    )


def build_embedding(path: Path) -> None:
    """input_ids[1,s] + image_features[1,1,4] -> inputs_embeds[1,s,4].

    inputs_embeds = E[input_ids] + 1{input_ids == PLACEHOLDER} * image_features.
    """
    input_ids = tensor_value("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    image_features = tensor_value("image_features", ir.DataType.FLOAT, [1, 1, HIDDEN])

    embedding_table = initializer("embedding_table", EMBEDDING_TABLE)
    text_embed = node(
        "Gather",
        [embedding_table, input_ids],
        "text_embed",
        attributes=[ir.AttrInt64("axis", 0)],
    )  # [1, s, 4]

    placeholder = constant(
        "placeholder_id", np.array(PLACEHOLDER_ID, dtype=np.int64)
    )
    is_placeholder = node("Equal", [input_ids, placeholder.outputs[0]], "is_placeholder")
    mask_f = node(
        "Cast",
        [is_placeholder.outputs[0]],
        "placeholder_mask",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )  # [1, s]
    mask_axis = constant("mask_axis", np.array([2], dtype=np.int64))
    mask_col = node(
        "Unsqueeze", [mask_f.outputs[0], mask_axis.outputs[0]], "placeholder_mask_col"
    )  # [1, s, 1]
    image_contrib = node(
        "Mul", [mask_col.outputs[0], image_features], "image_contrib"
    )  # [1, s, 4] via broadcast
    inputs_embeds = node(
        "Add", [text_embed.outputs[0], image_contrib.outputs[0]], "inputs_embeds"
    )
    inputs_embeds.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    inputs_embeds.outputs[0].shape = ir.Shape([1, "sequence", HIDDEN])

    graph = ir.Graph(
        [input_ids, image_features],
        [inputs_embeds.outputs[0]],
        nodes=[
            text_embed,
            placeholder,
            is_placeholder,
            mask_f,
            mask_axis,
            mask_col,
            image_contrib,
            inputs_embeds,
        ],
        initializers=[embedding_table],
        opset_imports={"": 13},
        name="tiny_gemma4_embedding",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-gemma4-vlm fixture"),
        path,
    )


def build_decoder(path: Path) -> None:
    """inputs_embeds[1,s,4] (+ KV) -> logits[1,s,8] + present KV.

    logits = inputs_embeds @ W + tie_bias; the KV cache is a contract-only
    growing buffer built from inputs_embeds (mirrors the tiny-whisper decoder).
    """
    inputs_embeds = tensor_value(
        "inputs_embeds", ir.DataType.FLOAT, [1, "sequence", HIDDEN]
    )
    past_key = tensor_value(
        "past_key_values.0.key", ir.DataType.FLOAT, [1, 1, "past_sequence", HIDDEN]
    )
    past_value = tensor_value(
        "past_key_values.0.value", ir.DataType.FLOAT, [1, 1, "past_sequence", HIDDEN]
    )

    lm_head = initializer("lm_head", LM_HEAD)
    matmul = node("MatMul", [inputs_embeds, lm_head], "logits_base")  # [1, s, 8]
    tie_bias = initializer("tie_bias", TIE_BIAS)
    logits = node("Add", [matmul.outputs[0], tie_bias], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape([1, "sequence", VOCAB])

    # KV contract: current key/value are [1, 1, s, 4]; append to the past cache.
    kv_axis = constant("kv_axis", np.array([1], dtype=np.int64))
    current_key = node(
        "Unsqueeze", [inputs_embeds, kv_axis.outputs[0]], "current_key"
    )  # [1, 1, s, 4]
    value_offset = initializer("value_offset", np.array(0.5, dtype=np.float32))
    current_value = node("Add", [current_key.outputs[0], value_offset], "current_value")
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
        out.type = ir.TensorType(ir.DataType.FLOAT)
        out.shape = ir.Shape([1, 1, "total_sequence", HIDDEN])

    graph = ir.Graph(
        [inputs_embeds, past_key, past_value],
        [logits.outputs[0], present_key.outputs[0], present_value.outputs[0]],
        nodes=[
            matmul,
            logits,
            kv_axis,
            current_key,
            current_value,
            present_key,
            present_value,
        ],
        initializers=[lm_head, tie_bias, value_offset],
        opset_imports={"": 13},
        name="tiny_gemma4_decoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-gemma4-vlm fixture"),
        path,
    )


def write_tokenizer(path: Path) -> None:
    vocab = {
        "[UNK]": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "<image>": 7,
    }
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "[UNK]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 7,
                "content": "<image>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "[UNK]",
        },
    }
    path.write_text(json.dumps(tokenizer, indent=2) + "\n")


METADATA = """\
# Tiny Gemma4-style VLM composite pipeline fixture with inputs_embeds fusion.
# Built by scripts/build_tiny_gemma4_vlm.py; exercises the composite strategy
# with a prompt_only vision stage plus an every_step embedding component (its
# `input_ids` seeded with the running token each step via the declared
# io.token_input) feeding an autoregressive decoder that consumes inputs_embeds
# (DESIGN.md §20). The embedding runs generically as an every_step component;
# no tensor-name special case is involved.
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
        run_on: prompt_only
      - name: fuse_embeddings
        strategy:
          kind: single_pass
          model: embedding
        run_on: every_step
      - name: decode
        strategy:
          kind: autoregressive
          decoder: decoder
          max_tokens: 4
        run_on: every_step
  phases:
    vision_encoder:
      run_on: prompt_only
    embedding:
      run_on: every_step
    decoder:
      run_on: every_step
"""


def tiny_pixels() -> np.ndarray:
    return np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2) / 12.0


def compute_expected_tokens(prompt: list[int], max_new_tokens: int) -> tuple[list[int], list[float]]:
    """Reference closed-form generation used to assert the engine's output."""
    pixels = tiny_pixels()
    image_features = pixels.reshape(1, 3, HIDDEN).mean(axis=1, keepdims=True)  # [1,1,4]

    def embed(ids: list[int]) -> np.ndarray:
        rows = EMBEDDING_TABLE[ids]  # [s, 4]
        mask = np.array([[1.0 if i == PLACEHOLDER_ID else 0.0] for i in ids])  # [s,1]
        return (rows + mask * image_features[0]).reshape(1, len(ids), HIDDEN)

    def step_logits(embeds: np.ndarray) -> np.ndarray:
        return embeds @ LM_HEAD + TIE_BIAS  # [1, s, 8]

    generated: list[int] = []
    image_feat_last = float(image_features.reshape(-1)[0])

    # Prefill on the whole prompt; the trailing placeholder carries the image.
    logits = step_logits(embed(prompt))
    token = int(logits[0, -1].argmax())
    generated.append(token)
    # Decode steps re-embed only the last generated token (text-only, no image).
    for _ in range(1, max_new_tokens):
        logits = step_logits(embed([generated[-1]]))
        generated.append(int(logits[0, -1].argmax()))
    return generated, [image_feat_last]


def _textproto_bytes(path) -> bytes:
    """Load a .onnx.textproto fixture and return serialized binary ModelProto bytes."""
    return ir.to_proto(ir.load(str(path), format="textproto")).SerializeToString()


def validate_with_ort(output_dir: Path, prompt: list[int], max_new_tokens: int) -> list[int]:
    import onnxruntime as ort

    vision = ort.InferenceSession(
        _textproto_bytes(output_dir / "vision_encoder.onnx.textproto"), providers=["CPUExecutionProvider"]
    )
    embedding = ort.InferenceSession(
        _textproto_bytes(output_dir / "embedding.onnx.textproto"), providers=["CPUExecutionProvider"]
    )
    decoder = ort.InferenceSession(
        _textproto_bytes(output_dir / "decoder.onnx.textproto"), providers=["CPUExecutionProvider"]
    )

    pixels = tiny_pixels()
    image_features = vision.run(None, {"pixel_values": pixels})[0]
    assert image_features.shape == (1, 1, HIDDEN), image_features.shape

    def embed(ids: list[int]) -> np.ndarray:
        return embedding.run(
            None,
            {
                "input_ids": np.array([ids], dtype=np.int64),
                "image_features": image_features,
            },
        )[0]

    def decode(embeds: np.ndarray, past_k, past_v):
        return decoder.run(
            None,
            {
                "inputs_embeds": embeds.astype(np.float32),
                "past_key_values.0.key": past_k,
                "past_key_values.0.value": past_v,
            },
        )

    past_k = np.zeros((1, 1, 0, HIDDEN), dtype=np.float32)
    past_v = np.zeros((1, 1, 0, HIDDEN), dtype=np.float32)
    logits, past_k, past_v = decode(embed(prompt), past_k, past_v)
    generated = [int(logits[0, -1].argmax())]
    for _ in range(1, max_new_tokens):
        logits, past_k, past_v = decode(embed([generated[-1]]), past_k, past_v)
        generated.append(int(logits[0, -1].argmax()))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=str(
            Path(__file__).resolve().parent.parent / "tests/fixtures/tiny-gemma4-vlm"
        ),
        help="output fixture directory",
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_vision_encoder(out_dir / "vision_encoder.onnx.textproto")
    build_embedding(out_dir / "embedding.onnx.textproto")
    build_decoder(out_dir / "decoder.onnx.textproto")
    write_tokenizer(out_dir / "tokenizer.json")
    (out_dir / "inference_metadata.yaml").write_text(METADATA)

    prompt = [3, PLACEHOLDER_ID]
    max_new_tokens = 4
    expected, _ = compute_expected_tokens(prompt, max_new_tokens)
    if not args.no_validate:
        ort_tokens = validate_with_ort(out_dir, prompt, max_new_tokens)
        assert ort_tokens == expected, f"ORT {ort_tokens} != closed form {expected}"
        print(f"prompt={prompt} -> generated token ids: {ort_tokens}")
    else:
        print(f"prompt={prompt} -> closed-form token ids: {expected}")
    total = sum(p.stat().st_size for p in out_dir.iterdir())
    print(f"wrote tiny-gemma4-vlm fixture to {out_dir} ({total} bytes)")


if __name__ == "__main__":
    main()
