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
from onnx import TensorProto, helper, numpy_helper


def const_array(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def scalar_const_node(name: str, output: str, value: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output],
        name=name,
        value=const_array(f"{name}_value", value),
    )


def save_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_encoder(path: Path) -> None:
    pixel_values = helper.make_tensor_value_info(
        "pixel_values", TensorProto.FLOAT, [1, 3, 2, 2]
    )
    image_features = helper.make_tensor_value_info(
        "image_features", TensorProto.FLOAT, [1, 1, 4]
    )
    weights = np.arange(48, dtype=np.float32).reshape(12, 4) / 100.0
    bias = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["pixel_values"], ["pixels_flat"], axis=1),
            helper.make_node("MatMul", ["pixels_flat", "encoder_w"], ["features_flat"]),
            helper.make_node("Add", ["features_flat", "encoder_b"], ["features_biased"]),
            scalar_const_node("encoder_unsqueeze_axes", "encoder_axes", np.array([1], dtype=np.int64)),
            helper.make_node("Unsqueeze", ["features_biased", "encoder_axes"], ["image_features"]),
        ],
        "tiny_vlm_encoder",
        [pixel_values],
        [image_features],
        initializer=[const_array("encoder_w", weights), const_array("encoder_b", bias)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="onnx-genai tiny-vlm fixture",
    )
    model.ir_version = 8
    save_model(model, path)


def build_decoder(path: Path) -> None:
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch", "sequence"]
    )
    image_features = helper.make_tensor_value_info(
        "image_features", TensorProto.FLOAT, [1, 1, 4]
    )
    logits = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, ["batch", "sequence", 8]
    )

    token_bias = np.array(
        [[[-4.0, -4.0, -4.0, -4.0, 8.0, -4.0, -4.0, -4.0]]], dtype=np.float32
    )

    graph = helper.make_graph(
        [
            helper.make_node("Shape", ["input_ids"], ["input_shape"]),
            scalar_const_node("batch_index", "batch_index_value", np.array(0, dtype=np.int64)),
            scalar_const_node("seq_index", "seq_index_value", np.array(1, dtype=np.int64)),
            scalar_const_node("shape_unsqueeze_axes", "shape_axes", np.array([0], dtype=np.int64)),
            scalar_const_node("vocab_dim", "vocab_dim_value", np.array([8], dtype=np.int64)),
            helper.make_node("Gather", ["input_shape", "batch_index_value"], ["batch_dim"], axis=0),
            helper.make_node("Gather", ["input_shape", "seq_index_value"], ["seq_dim"], axis=0),
            helper.make_node("Unsqueeze", ["batch_dim", "shape_axes"], ["batch_dim_vec"]),
            helper.make_node("Unsqueeze", ["seq_dim", "shape_axes"], ["seq_dim_vec"]),
            helper.make_node("Concat", ["batch_dim_vec", "seq_dim_vec", "vocab_dim_value"], ["logits_shape"], axis=0),
            helper.make_node(
                "ConstantOfShape",
                ["logits_shape"],
                ["zero_logits"],
                value=const_array("zero_logits_value", np.array([0.0], dtype=np.float32)),
            ),
            helper.make_node("ReduceMean", ["image_features"], ["image_bias_flat"], axes=[1, 2], keepdims=0),
            scalar_const_node("bias_unsqueeze_axes", "bias_axes", np.array([1, 2], dtype=np.int64)),
            helper.make_node("Unsqueeze", ["image_bias_flat", "bias_axes"], ["image_bias"]),
            helper.make_node("Add", ["zero_logits", "image_bias"], ["image_conditioned_logits"]),
            helper.make_node("Add", ["image_conditioned_logits", "token_bias"], ["logits"]),
        ],
        "tiny_vlm_decoder",
        [input_ids, image_features],
        [logits],
        initializer=[const_array("token_bias", token_bias)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="onnx-genai tiny-vlm fixture",
    )
    model.ir_version = 8
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
        """pipeline:\n  models:\n    encoder:\n      filename: encoder.onnx\n      type: encoder\n    decoder:\n      filename: decoder.onnx\n      type: decoder\n      tokenizer: tokenizer.json\n  dataflow:\n    - from: encoder.image_features\n      to: decoder.image_features\n      dtype: fp32\n      device_transfer: false\n  strategy:\n    kind: composite\n    stages:\n      - name: encode_image\n        strategy:\n          kind: single_pass\n          model: encoder\n        run_on: prompt_only\n      - name: decode_text\n        strategy:\n          kind: autoregressive\n          decoder: decoder\n          max_tokens: 4\n        run_on: every_step\n  phases:\n    encoder:\n      run_on: prompt_only\n    decoder:\n      run_on: every_step\n"""
    )


def validate_with_ort(output_dir: Path) -> None:
    import onnxruntime as ort

    encoder = ort.InferenceSession(str(output_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(output_dir / "decoder.onnx"), providers=["CPUExecutionProvider"])
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
    build_encoder(output_dir / "encoder.onnx")
    build_decoder(output_dir / "decoder.onnx")
    write_tokenizer(output_dir / "tokenizer.json")
    write_metadata(output_dir / "inference_metadata.yaml")
    if not args.no_validate:
        validate_with_ort(output_dir)

    total_size = sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file())
    print(f"Wrote {output_dir} ({total_size} bytes)")


if __name__ == "__main__":
    main()
