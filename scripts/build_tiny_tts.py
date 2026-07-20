#!/usr/bin/env python3
"""Build the tiny deterministic **text-to-speech (TTS)** pipeline fixture.

This exercises the one composite shape the engine could not previously express:
an autoregressive decoder that emits audio *code* tokens, followed by a
**post-decode `final_only` single_pass vocoder stage** that turns the collected
code sequence into a waveform (DESIGN.md §20).

Two tiny onnx-ir models with a Whisper-style KV/logits layout (mirroring
`build_tiny_whisper.py`), chosen so both the generated code ids and the waveform
are exact closed forms:

  * `decoder` (autoregressive): a KV-cached decoder whose logits are
    `-(vocab_index - position)^2`, so `argmax == position`. Starting from a
    1-token prompt (position 0), the greedy code sequence is `[0, 1, 2, ...]`.
  * `vocoder` (`final_only` single_pass): `codes[1, T] (int64) -> audio[1, T*K]`
    with `audio[i*K + j] = codes[i] * 2`, `K = 2`. For codes `[0, 1, 2, 3]` this
    is `[0, 0, 2, 2, 4, 4, 6, 6]`.

The generated code sequence is routed into the vocoder by the pipeline
`dataflow` edge `decoder.output_ids -> vocoder.codes`: the engine publishes the
AR decoder's generated tokens into the shared pool as the synthetic tensor
`decoder.output_ids` of shape `[1, num_generated]` (int64).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir

VOCAB = 16
HEAD_DIM = 4
REPEAT = 2  # vocoder upsampling factor K


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
    ir.save(model, path)
    onnx.checker.check_model(path)


def build_decoder(path: Path) -> None:
    """A KV-cached decoder whose greedy argmax equals the token position.

    logits[b, s, v] = -(v - position[b, s])^2  =>  argmax_v == position[b, s].
    The KV outputs mirror the Whisper contract (num_heads=1, head_dim=4) so the
    engine's KV bridge drives it; their values are irrelevant to the logits.
    """
    decoder_input_ids = tensor_value(
        "decoder_input_ids", ir.DataType.INT64, ["batch", "sequence_len"]
    )
    position_ids = tensor_value(
        "position_ids", ir.DataType.INT64, ["batch", "sequence_len"]
    )
    past_key = tensor_value(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        [1, 1, "past_sequence_len", HEAD_DIM],
    )
    past_value = tensor_value(
        "past_key_values.0.value",
        ir.DataType.FLOAT,
        [1, 1, "past_sequence_len", HEAD_DIM],
    )

    # --- KV cache (Whisper-style; values are inert placeholders) ---
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
    head = constant("head", np.array([HEAD_DIM], dtype=np.int64))
    cache_shape = node(
        "Concat",
        [batch_vec.outputs[0], one.outputs[0], sequence_vec.outputs[0], head.outputs[0]],
        "cache_shape",
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

    # --- Position-indexed logits: argmax_v (-(v - position)^2) == position ---
    position_float = node(
        "Cast",
        [position_ids],
        "position_float",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    logits_axis = constant("logits_axis", np.array([2], dtype=np.int64))
    position_col = node(
        "Unsqueeze", [position_float.outputs[0], logits_axis.outputs[0]], "position_col"
    )
    vocab_index = initializer(
        "vocab_index",
        np.arange(VOCAB, dtype=np.float32).reshape(1, 1, VOCAB),
    )
    diff = node("Sub", [vocab_index, position_col.outputs[0]], "diff")
    diff_sq = node("Mul", [diff.outputs[0], diff.outputs[0]], "diff_sq")
    logits = node("Neg", [diff_sq.outputs[0]], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape(["batch", "sequence_len", VOCAB])
    for output in (present_key.outputs[0], present_value.outputs[0]):
        output.type = ir.TensorType(ir.DataType.FLOAT)
        output.shape = ir.Shape([1, 1, "total_sequence_len", HEAD_DIM])

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
        cache_shape,
        ids_float,
        cache_axes,
        ids_cache,
        current_key,
        current_value,
        present_key,
        present_value,
        position_float,
        logits_axis,
        position_col,
        diff,
        diff_sq,
        logits,
    ]
    graph = ir.Graph(
        [decoder_input_ids, position_ids, past_key, past_value],
        [logits.outputs[0], present_key.outputs[0], present_value.outputs[0]],
        nodes=nodes,
        initializers=[value_offset, vocab_index],
        opset_imports={"": 13},
        name="tiny_tts_decoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-tts fixture"),
        path,
    )


def build_vocoder(path: Path) -> None:
    """codes[1, T] (int64) -> audio[1, T*K] with audio[i*K + j] = codes[i] * 2."""
    codes = tensor_value("codes", ir.DataType.INT64, [1, "num_codes"])
    codes_float = node(
        "Cast",
        [codes],
        "codes_float",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    gain = initializer("gain", np.array(2.0, dtype=np.float32))
    scaled = node("Mul", [codes_float.outputs[0], gain], "scaled")
    unsqueeze_axis = constant("unsqueeze_axis", np.array([2], dtype=np.int64))
    scaled_col = node(
        "Unsqueeze", [scaled.outputs[0], unsqueeze_axis.outputs[0]], "scaled_col"
    )
    repeats = constant("repeats", np.array([1, 1, REPEAT], dtype=np.int64))
    tiled = node("Tile", [scaled_col.outputs[0], repeats.outputs[0]], "tiled")
    flat_shape = constant("flat_shape", np.array([1, -1], dtype=np.int64))
    audio = node("Reshape", [tiled.outputs[0], flat_shape.outputs[0]], "audio")
    audio.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    audio.outputs[0].shape = ir.Shape([1, "num_samples"])

    graph = ir.Graph(
        [codes],
        [audio.outputs[0]],
        nodes=[
            codes_float,
            scaled,
            unsqueeze_axis,
            scaled_col,
            repeats,
            tiled,
            flat_shape,
            audio,
        ],
        initializers=[gain],
        opset_imports={"": 13},
        name="tiny_tts_vocoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-tts fixture"),
        path,
    )


def write_tokenizer(path: Path) -> None:
    vocab = {"[UNK]": 0, "[EOS]": 1}
    for i in range(2, VOCAB):
        vocab[f"c{i}"] = i
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
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }
    path.write_text(json.dumps(tokenizer, indent=2) + "\n")


def write_metadata(path: Path) -> None:
    path.write_text(
        """# Tiny text-to-speech (TTS) composite pipeline fixture.
# Built by scripts/build_tiny_tts.py; exercises the post-decode single_pass
# stage contract (DESIGN.md §20): an autoregressive decoder that emits audio
# code tokens, then a `final_only` vocoder stage that turns the collected codes
# (published as `decoder.output_ids`) into a waveform.
pipeline:
  models:
    decoder:
      filename: decoder.onnx
      type: decoder
      tokenizer: tokenizer.json
    vocoder:
      filename: vocoder.onnx
      type: vocoder
  dataflow:
    - from: decoder.output_ids
      to: vocoder.codes
      dtype: int64
      device_transfer: false
  strategy:
    kind: composite
    stages:
      - name: decode_codes
        strategy:
          kind: autoregressive
          decoder: decoder
          max_tokens: 4
        run_on: every_step
      - name: synthesize_waveform
        strategy:
          kind: single_pass
          model: vocoder
        run_on: final_only
  phases:
    decoder:
      run_on: every_step
    vocoder:
      run_on: final_only
"""
    )


def validate_with_ort(output_dir: Path) -> None:
    import onnxruntime as ort

    decoder = ort.InferenceSession(
        str(output_dir / "decoder.onnx"), providers=["CPUExecutionProvider"]
    )
    vocoder = ort.InferenceSession(
        str(output_dir / "vocoder.onnx"), providers=["CPUExecutionProvider"]
    )

    # Greedy decode from a 1-token prompt at position 0; argmax == position, so
    # the generated code sequence is [0, 1, 2, 3].
    past_key = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
    past_value = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
    generated: list[int] = []
    input_ids = np.array([[8]], dtype=np.int64)  # prompt token (id irrelevant)
    position = 0
    for _ in range(4):
        position_ids = np.array([[position]], dtype=np.int64)
        logits, past_key, past_value = decoder.run(
            None,
            {
                "decoder_input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values.0.key": past_key,
                "past_key_values.0.value": past_value,
            },
        )
        token = int(logits[0, -1].argmax())
        generated.append(token)
        input_ids = np.array([[token]], dtype=np.int64)
        position += 1
    assert generated == [0, 1, 2, 3], generated

    codes = np.array([generated], dtype=np.int64)
    audio = vocoder.run(None, {"codes": codes})[0]
    expected = np.array([[0, 0, 2, 2, 4, 4, 6, 6]], dtype=np.float32)
    assert audio.shape == (1, 8), audio.shape
    assert np.allclose(audio, expected), audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/tiny-tts"),
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    build_decoder(args.output / "decoder.onnx")
    build_vocoder(args.output / "vocoder.onnx")
    write_tokenizer(args.output / "tokenizer.json")
    write_metadata(args.output / "inference_metadata.yaml")
    if not args.no_validate:
        validate_with_ort(args.output)
    total_size = sum(path.stat().st_size for path in args.output.iterdir())
    print(f"Wrote {args.output} ({total_size} bytes)")


if __name__ == "__main__":
    main()
