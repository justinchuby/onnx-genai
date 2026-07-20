#!/usr/bin/env python3
"""Build the tiny deterministic audio-to-audio (neural codec) pipeline fixture.

A real neural audio codec (e.g. Kyutai Mimi, EnCodec) is emitted by Mobius as a
two-model graph:

  * an ``audio_encoder`` that maps a waveform to discrete/continuous codes, and
  * a ``vocoder`` that reconstructs a waveform from those codes.

onnx-genai runs this as a ``composite`` pipeline strategy (DESIGN.md §20): an
ordered chain of two ``single_pass`` stages sharing one tensor pool, wired
encoder-codes -> vocoder-codes through the pipeline ``dataflow``. This script
builds the smallest possible pair with onnx-ir so the pipeline *contract* (not
codec quality) can be tested end to end.

Contract exercised by ``crates/onnx-genai-engine/tests/codec_pipeline_e2e.rs``:

  encoder: waveform[1, 16] -> codes[1, 8]        codes[i] = (w[2i] + w[2i+1]) / 2
  vocoder: codes[1, 8]     -> audio[1, 16]       audio[2i] = audio[2i+1]
                                                            = codes[i] * 2

so ``audio[2i] == audio[2i+1] == w[2i] + w[2i + 1]`` — a closed form the test
asserts against, proving the composite executor runs both stages in order and
routes ``encoder.codes -> vocoder.codes`` via the dataflow edge.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir

WAVEFORM_LEN = 16
CODES_LEN = WAVEFORM_LEN // 2


def tensor_value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


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


def build_encoder(path: Path) -> None:
    """waveform[1, 16] -> codes[1, 8] with codes[i] = (w[2i] + w[2i+1]) / 2."""
    waveform = tensor_value("waveform", ir.DataType.FLOAT, [1, WAVEFORM_LEN])
    pair_shape = constant(
        "pair_shape", np.array([1, CODES_LEN, 2], dtype=np.int64)
    )
    pairs = node("Reshape", [waveform, pair_shape.outputs[0]], "pairs")
    codes = node(
        "ReduceMean",
        [pairs.outputs[0]],
        "codes",
        attributes=[ir.AttrInt64s("axes", [2]), ir.AttrInt64("keepdims", 0)],
    )
    codes.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    codes.outputs[0].shape = ir.Shape([1, CODES_LEN])

    graph = ir.Graph(
        [waveform],
        [codes.outputs[0]],
        nodes=[pair_shape, pairs, codes],
        opset_imports={"": 13},
        name="tiny_codec_encoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-codec fixture"),
        path,
    )


def build_vocoder(path: Path) -> None:
    """codes[1, 8] -> audio[1, 16] with audio[2i] = audio[2i+1] = codes[i] * 2."""
    codes = tensor_value("codes", ir.DataType.FLOAT, [1, CODES_LEN])
    gain = constant("gain", np.array(2.0, dtype=np.float32))
    scaled = node("Mul", [codes, gain.outputs[0]], "scaled")
    # Duplicate each code into an adjacent pair: [1, 8] -> [1, 8, 1] -> [1, 8, 2].
    expand_axis = constant("expand_axis", np.array([2], dtype=np.int64))
    scaled_col = node("Unsqueeze", [scaled.outputs[0], expand_axis.outputs[0]], "scaled_col")
    pair_shape = constant("pair_shape", np.array([1, CODES_LEN, 2], dtype=np.int64))
    duplicated = node("Expand", [scaled_col.outputs[0], pair_shape.outputs[0]], "duplicated")
    audio_shape = constant("audio_shape", np.array([1, WAVEFORM_LEN], dtype=np.int64))
    audio = node("Reshape", [duplicated.outputs[0], audio_shape.outputs[0]], "audio")
    audio.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    audio.outputs[0].shape = ir.Shape([1, WAVEFORM_LEN])

    graph = ir.Graph(
        [codes],
        [audio.outputs[0]],
        nodes=[gain, scaled, expand_axis, scaled_col, pair_shape, duplicated, audio_shape, audio],
        opset_imports={"": 13},
        name="tiny_codec_vocoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-codec fixture"),
        path,
    )


METADATA = """\
# Tiny audio-to-audio (neural codec) composite pipeline fixture.
# Built by scripts/build_tiny_codec.py; exercises the composite strategy with
# two single_pass stages sharing one tensor pool (DESIGN.md §20).
pipeline:
  models:
    encoder:
      filename: encoder.onnx.textproto
      type: audio_encoder
    vocoder:
      filename: vocoder.onnx.textproto
      type: vocoder
  dataflow:
    - from: encoder.codes
      to: vocoder.codes
      dtype: fp32
      device_transfer: false
  strategy:
    kind: composite
    stages:
      - name: encode_waveform
        strategy:
          kind: single_pass
          model: encoder
        run_on: prompt_only
      - name: synthesize_waveform
        strategy:
          kind: single_pass
          model: vocoder
        run_on: prompt_only
  phases:
    encoder:
      run_on: prompt_only
    vocoder:
      run_on: prompt_only
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "tests/fixtures/tiny-codec"),
        help="output fixture directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_encoder(out_dir / "encoder.onnx.textproto")
    build_vocoder(out_dir / "vocoder.onnx.textproto")
    (out_dir / "inference_metadata.yaml").write_text(METADATA)
    print(f"wrote tiny-codec fixture to {out_dir}")


if __name__ == "__main__":
    main()
