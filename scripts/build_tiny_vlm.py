#!/usr/bin/env python3
"""Build the tiny deterministic VLM pipeline fixture.

Mobius investigation note:
  The smallest built-in Mobius multimodal example is the real Gemma 3 VLM path:
    PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src \
      python /Users/justinc/Documents/GitHub/mobius/examples/multimodal_generation.py \
      --model google/gemma-3-4b-pt --save-to models/gemma3-vlm
  That is intentionally not used for this fixture because it downloads a multi-GB
  checkpoint. Mobius currently builds multimodal packages from HuggingFace
  checkpoints rather than exposing a no-download toy VLM template, so this script
  hand-constructs the minimal ONNX pair needed to exercise pipeline mechanics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir


def tensor_value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(
        name=name,
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape),
    )


def initializer(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(
        name=name,
        const_value=ir.Tensor(array, name=name),
    )


def node(
    op_type: str,
    inputs: list[ir.Value],
    output: str,
    *,
    name: str = "",
    attributes: tuple[ir.Attr, ...] | list[ir.Attr] = (),
) -> ir.Node:
    output_value = ir.Value(name=output)
    return ir.Node(
        "",
        op_type,
        inputs,
        attributes,
        outputs=[output_value],
        name=name,
    )


def constant_node(name: str, output: str, value: np.ndarray) -> ir.Node:
    return node(
        "Constant",
        [],
        output,
        name=name,
        attributes=[ir.AttrTensor("value", ir.Tensor(value, name=f"{name}_value"))],
    )


def save_model(model: ir.Model, path: Path) -> None:
    ir.save(model, path, format="textproto")
    onnx.checker.check_model(ir.to_proto(model))


def build_encoder(path: Path) -> None:
    pixel_values = tensor_value("pixel_values", ir.DataType.FLOAT, [1, 3, 2, 2])
    weights = np.arange(48, dtype=np.float32).reshape(12, 4) / 100.0
    bias = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    encoder_w = initializer("encoder_w", weights)
    encoder_b = initializer("encoder_b", bias)

    flatten = node(
        "Flatten",
        [pixel_values],
        "pixels_flat",
        attributes=[ir.AttrInt64("axis", 1)],
    )
    matmul = node("MatMul", [flatten.outputs[0], encoder_w], "features_flat")
    add = node("Add", [matmul.outputs[0], encoder_b], "features_biased")
    axes = constant_node(
        "encoder_unsqueeze_axes", "encoder_axes", np.array([1], dtype=np.int64)
    )
    unsqueeze = node(
        "Unsqueeze", [add.outputs[0], axes.outputs[0]], "image_features"
    )
    unsqueeze.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    unsqueeze.outputs[0].shape = ir.Shape([1, 1, 4])

    graph = ir.Graph(
        [pixel_values],
        [unsqueeze.outputs[0]],
        nodes=[flatten, matmul, add, axes, unsqueeze],
        initializers=[encoder_w, encoder_b],
        opset_imports={"": 13},
        name="tiny_vlm_encoder",
    )
    model = ir.Model(
        graph,
        ir_version=8,
        producer_name="onnx-genai tiny-vlm fixture",
    )
    save_model(model, path)


def build_decoder(path: Path) -> None:
    input_ids = tensor_value(
        "input_ids", ir.DataType.INT64, ["batch", "sequence"]
    )
    image_features = tensor_value(
        "image_features", ir.DataType.FLOAT, [1, 1, 4]
    )

    token_bias = np.array(
        [[[-4.0, -4.0, -4.0, -4.0, 8.0, -4.0, -4.0, -4.0]]], dtype=np.float32
    )
    token_bias_value = initializer("token_bias", token_bias)

    shape = node("Shape", [input_ids], "input_shape")
    batch_index = constant_node(
        "batch_index", "batch_index_value", np.array(0, dtype=np.int64)
    )
    seq_index = constant_node(
        "seq_index", "seq_index_value", np.array(1, dtype=np.int64)
    )
    shape_axes = constant_node(
        "shape_unsqueeze_axes", "shape_axes", np.array([0], dtype=np.int64)
    )
    vocab_dim = constant_node(
        "vocab_dim", "vocab_dim_value", np.array([8], dtype=np.int64)
    )
    batch_dim = node(
        "Gather",
        [shape.outputs[0], batch_index.outputs[0]],
        "batch_dim",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    seq_dim = node(
        "Gather",
        [shape.outputs[0], seq_index.outputs[0]],
        "seq_dim",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    batch_dim_vec = node(
        "Unsqueeze",
        [batch_dim.outputs[0], shape_axes.outputs[0]],
        "batch_dim_vec",
    )
    seq_dim_vec = node(
        "Unsqueeze",
        [seq_dim.outputs[0], shape_axes.outputs[0]],
        "seq_dim_vec",
    )
    logits_shape = node(
        "Concat",
        [batch_dim_vec.outputs[0], seq_dim_vec.outputs[0], vocab_dim.outputs[0]],
        "logits_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    zero_logits = node(
        "ConstantOfShape",
        [logits_shape.outputs[0]],
        "zero_logits",
        attributes=[
            ir.AttrTensor(
                "value",
                ir.Tensor(
                    np.array([0.0], dtype=np.float32),
                    name="zero_logits_value",
                ),
            )
        ],
    )
    image_bias_flat = node(
        "ReduceMean",
        [image_features],
        "image_bias_flat",
        attributes=[
            ir.AttrInt64s("axes", [1, 2]),
            ir.AttrInt64("keepdims", 0),
        ],
    )
    bias_axes = constant_node(
        "bias_unsqueeze_axes", "bias_axes", np.array([1, 2], dtype=np.int64)
    )
    image_bias = node(
        "Unsqueeze",
        [image_bias_flat.outputs[0], bias_axes.outputs[0]],
        "image_bias",
    )
    image_conditioned_logits = node(
        "Add",
        [zero_logits.outputs[0], image_bias.outputs[0]],
        "image_conditioned_logits",
    )
    logits = node(
        "Add",
        [image_conditioned_logits.outputs[0], token_bias_value],
        "logits",
    )
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape(["batch", "sequence", 8])

    graph = ir.Graph(
        [input_ids, image_features],
        [logits.outputs[0]],
        nodes=[
            shape,
            batch_index,
            seq_index,
            shape_axes,
            vocab_dim,
            batch_dim,
            seq_dim,
            batch_dim_vec,
            seq_dim_vec,
            logits_shape,
            zero_logits,
            image_bias_flat,
            bias_axes,
            image_bias,
            image_conditioned_logits,
            logits,
        ],
        initializers=[token_bias_value],
        opset_imports={"": 13},
        name="tiny_vlm_decoder",
    )
    model = ir.Model(
        graph,
        ir_version=8,
        producer_name="onnx-genai tiny-vlm fixture",
    )
    save_model(model, path)


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
                "describe": 2,
                "image": 3,
                "cat": 4,
                "tiny": 5,
                "vlm": 6,
                ".": 7,
            },
            "unk_token": "[UNK]",
        },
    }
    path.write_text(json.dumps(tokenizer, indent=2) + "\n")


def write_metadata(path: Path) -> None:
    path.write_text(
        """pipeline:\n  models:\n    encoder:\n      filename: encoder.onnx.textproto\n      type: encoder\n    decoder:\n      filename: decoder.onnx.textproto\n      type: decoder\n      tokenizer: tokenizer.json\n  dataflow:\n    - from: encoder.image_features\n      to: decoder.image_features\n      dtype: fp32\n      device_transfer: false\n  strategy:\n    kind: composite\n    stages:\n      - name: encode_image\n        strategy:\n          kind: single_pass\n          model: encoder\n        run_on: prompt_only\n      - name: decode_text\n        strategy:\n          kind: autoregressive\n          decoder: decoder\n          max_tokens: 4\n        run_on: every_step\n  phases:\n    encoder:\n      run_on: prompt_only\n    decoder:\n      run_on: every_step\n"""
    )


def _textproto_bytes(path) -> bytes:
    """Load a .onnx.textproto fixture and return serialized binary ModelProto bytes."""
    return ir.to_proto(ir.load(str(path), format="textproto")).SerializeToString()


def validate_with_ort(output_dir: Path) -> None:
    import onnxruntime as ort

    encoder = ort.InferenceSession(_textproto_bytes(output_dir / "encoder.onnx.textproto"), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(_textproto_bytes(output_dir / "decoder.onnx.textproto"), providers=["CPUExecutionProvider"])
    pixels = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2) / 12.0
    features = encoder.run(None, {"pixel_values": pixels})[0]
    logits = decoder.run(None, {"input_ids": np.array([[2, 3]], dtype=np.int64), "image_features": features})[0]
    assert features.shape == (1, 1, 4), features.shape
    assert logits.shape == (1, 2, 8), logits.shape
    assert int(logits[0, -1].argmax()) == 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/tiny-vlm"),
        help="Output pipeline directory (default: models/tiny-vlm)",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip ONNX Runtime smoke validation")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    build_encoder(output_dir / "encoder.onnx.textproto")
    build_decoder(output_dir / "decoder.onnx.textproto")
    write_tokenizer(output_dir / "tokenizer.json")
    write_metadata(output_dir / "inference_metadata.yaml")
    if not args.no_validate:
        validate_with_ort(output_dir)

    total_size = sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file())
    print(f"Wrote {output_dir} ({total_size} bytes)")


if __name__ == "__main__":
    main()
