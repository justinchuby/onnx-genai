#!/usr/bin/env python3
"""Build the tiny deterministic Whisper **static cross-attention KV** fixture.

Unlike `build_tiny_whisper.py` (whose decoder consumes `encoder_hidden_states`
directly and recomputes cross-attention internally), this fixture mirrors the
real Foundry / ORT-genai Whisper export shape:

  * the ENCODER emits per-layer cross-attention KV once from the audio prompt as
    `present_key_cross_%d` / `present_value_cross_%d`;
  * the DECODER consumes those as STATIC `past_key_cross_%d` / `past_value_cross_%d`
    inputs (fixed for the whole decode) plus its own GROWING self-attention KV
    `past_key_self_%d` / `present_key_self_%d`;
  * `input_ids` is INT32 (as Whisper exports it), exercising the Int32 token path.

It exists to regression-test the engine's static cross-attention KV binding: the
encoder prologue runs once, its cross-KV outputs are captured and re-bound to the
decoder every autoregressive step. It tests the pipeline contract, not ASR
quality. Regenerate with:

    python scripts/build_tiny_whisper_cross_kv.py \
      --output tests/fixtures/tiny-whisper-cross-kv
"""

from __future__ import annotations

import argparse
import json
import shutil
import wave
from pathlib import Path

import numpy as np
from onnxscript import ir

CROSS_LEN = 4
HEAD_DIM = 4
VOCAB = 8


def tensor_value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def initializer(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.Tensor(array, name=name))


def node(op_type, inputs, output, *, attributes=()):
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


def build_encoder(path: Path) -> None:
    audio_features = tensor_value("audio_features", ir.DataType.FLOAT, [1, 80, 8])
    transpose = node(
        "Transpose",
        [audio_features],
        "features_time_major",
        attributes=[ir.AttrInt64s("perm", [0, 2, 1])],
    )
    mel_mean = node(
        "ReduceMean",
        [transpose.outputs[0]],
        "mel_mean",
        attributes=[ir.AttrInt64s("axes", [2]), ir.AttrInt64("keepdims", 1)],
    )
    pair_shape = constant("pair_shape", np.array([1, 4, 2, 1], dtype=np.int64))
    pairs = node("Reshape", [mel_mean.outputs[0], pair_shape.outputs[0]], "frame_pairs")
    frame_mean = node(
        "ReduceMean",
        [pairs.outputs[0]],
        "frame_mean",
        attributes=[ir.AttrInt64s("axes", [2]), ir.AttrInt64("keepdims", 0)],
    )
    hidden_shape = constant("hidden_shape", np.array([1, 4, 4], dtype=np.int64))
    hidden = node(
        "Expand",
        [frame_mean.outputs[0], hidden_shape.outputs[0]],
        "encoder_hidden_states",
    )
    hidden.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    hidden.outputs[0].shape = ir.Shape([1, 4, 4])

    # Cross-attention KV: expand the hidden state to [1, 1, CROSS_LEN, HEAD_DIM].
    cross_shape = constant(
        "cross_shape", np.array([1, 1, CROSS_LEN, HEAD_DIM], dtype=np.int64)
    )
    hidden_axes = constant("hidden_axes", np.array([1], dtype=np.int64))
    hidden_4d = node(
        "Unsqueeze", [hidden.outputs[0], hidden_axes.outputs[0]], "hidden_4d"
    )
    present_key_cross = node(
        "Expand", [hidden_4d.outputs[0], cross_shape.outputs[0]], "present_key_cross_0"
    )
    cross_value_offset = initializer(
        "cross_value_offset", np.array(0.25, dtype=np.float32)
    )
    present_value_cross = node(
        "Add",
        [present_key_cross.outputs[0], cross_value_offset],
        "present_value_cross_0",
    )
    for out in (present_key_cross.outputs[0], present_value_cross.outputs[0]):
        out.type = ir.TensorType(ir.DataType.FLOAT)
        out.shape = ir.Shape([1, 1, CROSS_LEN, HEAD_DIM])

    graph = ir.Graph(
        [audio_features],
        [hidden.outputs[0], present_key_cross.outputs[0], present_value_cross.outputs[0]],
        nodes=[
            transpose,
            mel_mean,
            pair_shape,
            pairs,
            frame_mean,
            hidden_shape,
            hidden,
            cross_shape,
            hidden_axes,
            hidden_4d,
            present_key_cross,
            present_value_cross,
        ],
        initializers=[cross_value_offset],
        opset_imports={"": 13},
        name="tiny_whisper_cross_kv_encoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-whisper-cross-kv"),
        path,
    )


def build_decoder(path: Path) -> None:
    input_ids = tensor_value("input_ids", ir.DataType.INT32, ["batch", "sequence_len"])
    past_key_self = tensor_value(
        "past_key_self_0", ir.DataType.FLOAT, [1, 1, "past_sequence_len", HEAD_DIM]
    )
    past_value_self = tensor_value(
        "past_value_self_0", ir.DataType.FLOAT, [1, 1, "past_sequence_len", HEAD_DIM]
    )
    past_key_cross = tensor_value(
        "past_key_cross_0", ir.DataType.FLOAT, [1, 1, CROSS_LEN, HEAD_DIM]
    )
    past_value_cross = tensor_value(
        "past_value_cross_0", ir.DataType.FLOAT, [1, 1, CROSS_LEN, HEAD_DIM]
    )

    input_shape = node("Shape", [input_ids], "input_shape")
    batch_index = constant("batch_index", np.array(0, dtype=np.int64))
    sequence_index = constant("sequence_index", np.array(1, dtype=np.int64))
    axes_zero = constant("axes_zero", np.array([0], dtype=np.int64))
    batch = node(
        "Gather",
        [input_shape.outputs[0], batch_index.outputs[0]],
        "batch",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    sequence = node(
        "Gather",
        [input_shape.outputs[0], sequence_index.outputs[0]],
        "sequence",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    batch_vec = node("Unsqueeze", [batch.outputs[0], axes_zero.outputs[0]], "batch_vec")
    sequence_vec = node(
        "Unsqueeze", [sequence.outputs[0], axes_zero.outputs[0]], "sequence_vec"
    )
    one = constant("one", np.array([1], dtype=np.int64))
    head = constant("head", np.array([HEAD_DIM], dtype=np.int64))
    vocab = constant("vocab", np.array([VOCAB], dtype=np.int64))
    cache_shape = node(
        "Concat",
        [batch_vec.outputs[0], one.outputs[0], sequence_vec.outputs[0], head.outputs[0]],
        "cache_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    logits_shape = node(
        "Concat",
        [batch_vec.outputs[0], sequence_vec.outputs[0], vocab.outputs[0]],
        "logits_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )

    ids_float = node(
        "Cast",
        [input_ids],
        "ids_float",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    cache_axes = constant("cache_axes", np.array([1, 3], dtype=np.int64))
    ids_cache = node("Unsqueeze", [ids_float.outputs[0], cache_axes.outputs[0]], "ids_cache")
    current_key = node("Expand", [ids_cache.outputs[0], cache_shape.outputs[0]], "current_key")
    value_offset = initializer("value_offset", np.array(0.5, dtype=np.float32))
    current_value = node("Add", [current_key.outputs[0], value_offset], "current_value")
    present_key_self = node(
        "Concat",
        [past_key_self, current_key.outputs[0]],
        "present_key_self_0",
        attributes=[ir.AttrInt64("axis", 2)],
    )
    present_value_self = node(
        "Concat",
        [past_value_self, current_value.outputs[0]],
        "present_value_self_0",
        attributes=[ir.AttrInt64("axis", 2)],
    )

    zero_logits = node(
        "ConstantOfShape",
        [logits_shape.outputs[0]],
        "zero_logits",
        attributes=[
            ir.AttrTensor(
                "value",
                ir.Tensor(np.array([0.0], dtype=np.float32), name="zero_logits_value"),
            )
        ],
    )
    # A small, argmax-neutral bias derived from BOTH static cross-KV inputs so the
    # decoder graph genuinely REQUIRES them: if the pipeline fails to bind the
    # encoder-produced cross-KV, the ORT run errors on the missing inputs.
    cross_key_bias = node(
        "ReduceMean",
        [past_key_cross],
        "cross_key_bias",
        attributes=[ir.AttrInt64s("axes", [0, 1, 2, 3]), ir.AttrInt64("keepdims", 0)],
    )
    cross_value_bias = node(
        "ReduceMean",
        [past_value_cross],
        "cross_value_bias",
        attributes=[ir.AttrInt64s("axes", [0, 1, 2, 3]), ir.AttrInt64("keepdims", 0)],
    )
    cross_bias = node(
        "Add", [cross_key_bias.outputs[0], cross_value_bias.outputs[0]], "cross_bias"
    )
    tiny_scale = initializer("tiny_scale", np.array(0.01, dtype=np.float32))
    cross_bias_scaled = node("Mul", [cross_bias.outputs[0], tiny_scale], "cross_bias_scaled")
    conditioned_logits = node(
        "Add", [zero_logits.outputs[0], cross_bias_scaled.outputs[0]], "conditioned_logits"
    )
    # token_bias makes argmax deterministically token 4 across steps.
    token_bias = initializer(
        "token_bias",
        np.array([[[-4.0, -3.0, -2.0, -1.0, 8.0, -1.0, -2.0, -3.0]]], dtype=np.float32),
    )
    logits = node("Add", [conditioned_logits.outputs[0], token_bias], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape(["batch", "sequence_len", VOCAB])
    for out in (present_key_self.outputs[0], present_value_self.outputs[0]):
        out.type = ir.TensorType(ir.DataType.FLOAT)
        out.shape = ir.Shape([1, 1, "total_sequence_len", HEAD_DIM])

    nodes = [
        input_shape,
        batch_index,
        sequence_index,
        axes_zero,
        batch,
        sequence,
        batch_vec,
        sequence_vec,
        one,
        head,
        vocab,
        cache_shape,
        logits_shape,
        ids_float,
        cache_axes,
        ids_cache,
        current_key,
        current_value,
        present_key_self,
        present_value_self,
        zero_logits,
        cross_key_bias,
        cross_value_bias,
        cross_bias,
        cross_bias_scaled,
        conditioned_logits,
        logits,
    ]
    graph = ir.Graph(
        [input_ids, past_key_self, past_value_self, past_key_cross, past_value_cross],
        [logits.outputs[0], present_key_self.outputs[0], present_value_self.outputs[0]],
        nodes=nodes,
        initializers=[value_offset, token_bias, tiny_scale],
        opset_imports={"": 13},
        name="tiny_whisper_cross_kv_decoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-whisper-cross-kv"),
        path,
    )


def write_metadata(path: Path) -> None:
    path.write_text(
        """schema_version: v1
pipeline:
  models:
    encoder:
      filename: encoder.onnx.textproto
      type: encoder
      io:
        audio_features_input: audio_features
    decoder:
      filename: decoder.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
      io:
        token_input: input_ids
        logits_output: logits
        kv_inputs: [past_key_self_0, past_value_self_0]
        kv_outputs: [present_key_self_0, present_value_self_0]
        kv_update: append
        cross_kv_inputs: [past_key_cross_0, past_value_cross_0]
        cross_kv_outputs: [present_key_cross_0, present_value_cross_0]
  dataflow: []
  strategy:
    kind: composite
    stages:
      - name: encode_audio
        strategy:
          kind: single_pass
          model: encoder
        run_on: prompt_only
      - name: decode_transcript
        strategy:
          kind: autoregressive
          decoder: decoder
          max_tokens: 4
        run_on: every_step
  phases:
    encoder:
      run_on: prompt_only
    decoder:
      run_on: every_step
"""
    )


def write_wav(path: Path) -> None:
    sample_rate = 16_000
    duration = 0.08
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 220.0 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/tiny-whisper-cross-kv"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build_encoder(args.output / "encoder.onnx.textproto")
    build_decoder(args.output / "decoder.onnx.textproto")
    write_metadata(args.output / "inference_metadata.yaml")
    write_wav(args.output / "tiny.wav")
    # Reuse the deterministic tiny-whisper tokenizer.
    tokenizer_src = Path("tests/fixtures/tiny-whisper/tokenizer.json")
    if tokenizer_src.exists():
        shutil.copyfile(tokenizer_src, args.output / "tokenizer.json")
    print(f"wrote tiny-whisper-cross-kv fixture to {args.output}")


if __name__ == "__main__":
    main()
