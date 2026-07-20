#!/usr/bin/env python3
"""Build the tiny deterministic Whisper-contract pipeline fixture.

Mobius builds a real checkpoint with:

  PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src \
    python -m mobius build --model openai/whisper-tiny \
      --task speech-to-text models/whisper/whisper-tiny

The real model is roughly 39M parameters, so this script uses onnx-ir to create
the smallest encoder/decoder pair with Mobius' exact Whisper port names and KV
layout. It intentionally tests the pipeline contract, not ASR quality.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir


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
    return ir.Node(
        "",
        op_type,
        inputs,
        attributes,
        outputs=[ir.Value(name=output)],
    )


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


def build_encoder(path: Path) -> None:
    input_features = tensor_value("input_features", ir.DataType.FLOAT, [1, 80, 8])
    transpose = node(
        "Transpose",
        [input_features],
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

    graph = ir.Graph(
        [input_features],
        [hidden.outputs[0]],
        nodes=[transpose, mel_mean, pair_shape, pairs, frame_mean, hidden_shape, hidden],
        opset_imports={"": 13},
        name="tiny_whisper_encoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-whisper fixture"),
        path,
    )


def build_decoder(path: Path) -> None:
    decoder_input_ids = tensor_value(
        "decoder_input_ids", ir.DataType.INT64, ["batch", "sequence_len"]
    )
    encoder_hidden_states = tensor_value(
        "encoder_hidden_states", ir.DataType.FLOAT, [1, 4, 4]
    )
    position_ids = tensor_value(
        "position_ids", ir.DataType.INT64, ["batch", "sequence_len"]
    )
    past_key = tensor_value(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        [1, 1, "past_sequence_len", 4],
    )
    past_value = tensor_value(
        "past_key_values.0.value",
        ir.DataType.FLOAT,
        [1, 1, "past_sequence_len", 4],
    )

    input_shape = node("Shape", [decoder_input_ids], "input_shape")
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
    four = constant("four", np.array([4], dtype=np.int64))
    eight = constant("eight", np.array([8], dtype=np.int64))
    cache_shape = node(
        "Concat",
        [batch_vec.outputs[0], one.outputs[0], sequence_vec.outputs[0], four.outputs[0]],
        "cache_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    logits_shape = node(
        "Concat",
        [batch_vec.outputs[0], sequence_vec.outputs[0], eight.outputs[0]],
        "logits_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )

    ids_float = node(
        "Cast",
        [decoder_input_ids],
        "ids_float",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    cache_axes = constant("cache_axes", np.array([1, 3], dtype=np.int64))
    ids_cache = node(
        "Unsqueeze", [ids_float.outputs[0], cache_axes.outputs[0]], "ids_cache"
    )
    current_key = node(
        "Expand", [ids_cache.outputs[0], cache_shape.outputs[0]], "current_key"
    )
    value_offset = initializer("value_offset", np.array(0.5, dtype=np.float32))
    current_value = node(
        "Add", [current_key.outputs[0], value_offset], "current_value"
    )
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
    encoder_bias = node(
        "ReduceMean",
        [encoder_hidden_states],
        "encoder_bias",
        attributes=[ir.AttrInt64s("axes", [1, 2]), ir.AttrInt64("keepdims", 1)],
    )
    conditioned_logits = node(
        "Add", [zero_logits.outputs[0], encoder_bias.outputs[0]], "conditioned_logits"
    )
    position_float = node(
        "Cast",
        [position_ids],
        "position_float",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    logits_axis = constant("logits_axis", np.array([2], dtype=np.int64))
    position_bias = node(
        "Unsqueeze",
        [position_float.outputs[0], logits_axis.outputs[0]],
        "position_bias",
    )
    positioned_logits = node(
        "Add",
        [conditioned_logits.outputs[0], position_bias.outputs[0]],
        "positioned_logits",
    )
    token_bias = initializer(
        "token_bias",
        np.array([[[-4.0, -4.0, -4.0, -4.0, 8.0, -4.0, -4.0, -4.0]]], dtype=np.float32),
    )
    logits = node("Add", [positioned_logits.outputs[0], token_bias], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape(["batch", "sequence_len", 8])
    for output in (present_key.outputs[0], present_value.outputs[0]):
        output.type = ir.TensorType(ir.DataType.FLOAT)
        output.shape = ir.Shape([1, 1, "total_sequence_len", 4])

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
        four,
        eight,
        cache_shape,
        logits_shape,
        ids_float,
        cache_axes,
        ids_cache,
        current_key,
        current_value,
        present_key,
        present_value,
        zero_logits,
        encoder_bias,
        conditioned_logits,
        position_float,
        logits_axis,
        position_bias,
        positioned_logits,
        logits,
    ]
    graph = ir.Graph(
        [
            decoder_input_ids,
            encoder_hidden_states,
            position_ids,
            past_key,
            past_value,
        ],
        [logits.outputs[0], present_key.outputs[0], present_value.outputs[0]],
        nodes=nodes,
        initializers=[value_offset, token_bias],
        opset_imports={"": 13},
        name="tiny_whisper_decoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-whisper fixture"),
        path,
    )


def write_tokenizer(path: Path) -> None:
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
                "id": 1,
                "content": "[EOS]",
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
            "vocab": {
                "[UNK]": 0,
                "[EOS]": 1,
                "<|startoftranscript|>": 2,
                "hello": 3,
                "audio": 4,
                "tiny": 5,
                "whisper": 6,
                ".": 7,
            },
            "unk_token": "[UNK]",
        },
    }
    path.write_text(json.dumps(tokenizer, indent=2) + "\n")


def write_metadata(path: Path) -> None:
    path.write_text(
        """pipeline:
  models:
    encoder:
      filename: encoder.onnx.textproto
      type: encoder
    decoder:
      filename: decoder.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
  dataflow:
    - from: encoder.encoder_hidden_states
      to: decoder.encoder_hidden_states
      dtype: fp32
      device_transfer: false
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
    samples = np.zeros(1280, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())


def _textproto_bytes(path) -> bytes:
    """Load a .onnx.textproto fixture and return serialized binary ModelProto bytes."""
    return ir.to_proto(ir.load(str(path), format="textproto")).SerializeToString()


def validate_with_ort(output_dir: Path) -> None:
    import onnxruntime as ort

    encoder = ort.InferenceSession(
        _textproto_bytes(output_dir / "encoder.onnx.textproto"), providers=["CPUExecutionProvider"]
    )
    decoder = ort.InferenceSession(
        _textproto_bytes(output_dir / "decoder.onnx.textproto"), providers=["CPUExecutionProvider"]
    )
    features = np.zeros((1, 80, 8), dtype=np.float32)
    hidden = encoder.run(None, {"input_features": features})[0]
    outputs = decoder.run(
        None,
        {
            "decoder_input_ids": np.array([[2]], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "position_ids": np.array([[0]], dtype=np.int64),
            "past_key_values.0.key": np.zeros((1, 1, 0, 4), dtype=np.float32),
            "past_key_values.0.value": np.zeros((1, 1, 0, 4), dtype=np.float32),
        },
    )
    assert hidden.shape == (1, 4, 4), hidden.shape
    assert outputs[0].shape == (1, 1, 8), outputs[0].shape
    assert outputs[1].shape == (1, 1, 1, 4), outputs[1].shape
    assert int(outputs[0][0, -1].argmax()) == 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/tiny-whisper"),
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    build_encoder(args.output / "encoder.onnx.textproto")
    build_decoder(args.output / "decoder.onnx.textproto")
    write_tokenizer(args.output / "tokenizer.json")
    write_metadata(args.output / "inference_metadata.yaml")
    write_wav(args.output / "tiny.wav")
    if not args.no_validate:
        validate_with_ort(args.output)
    total_size = sum(path.stat().st_size for path in args.output.iterdir())
    print(f"Wrote {args.output} ({total_size} bytes)")


if __name__ == "__main__":
    main()
