#!/usr/bin/env python3
"""Build the tiny VLM fixture that declares a typed *image input* contract.

`scripts/build_tiny_vlm.py` builds a VLM pipeline whose image tensor is supplied
by the caller. This variant reuses the same two ONNX graphs but adds the typed
metadata a front end needs to accept a user-supplied image file end to end:

  * ``preprocessing.image`` — a transform program that decodes, resizes,
    rescales, and normalizes an encoded image into the encoder's declared
    ``pixel_values [1, 3, 2, 2]`` input.
  * ``pipeline.vision`` — the placeholder expansion contract: each
    ``<image>`` token in the prompt is replaced by the declared image-token run.

The tokenizer maps ``<image>`` to the placeholder id and the ordinary word
``img`` to the image token id, both inside the decoder's 8-token vocabulary, so
the expanded prompt stays decodable and the generated text is printable.

This is the fixture behind `crates/onnx-genai-cli`'s image-input coverage and
mirrors exactly what the OpenAI-compatible server derives for `image_url`
content parts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def load_tiny_vlm_builder():
    spec = importlib.util.spec_from_file_location(
        "build_tiny_vlm", SCRIPTS / "build_tiny_vlm.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


METADATA = """# Tiny VLM that declares a full image-input contract: a typed preprocessing
# program bound to the encoder's pixel input, plus the placeholder expansion
# rules. Every number is fixture DATA; no model family is named.
schema_version: v1
preprocessing:
  image:
    transforms:
      - op: decode
        outputs: [decoded]
      - op: convert_rgb
        inputs: [decoded]
        outputs: [rgb]
      - op: resize
        inputs: [rgb]
        outputs: [resized]
        size: 2
        mode: fixed
        interpolation: bicubic
      - op: rescale
        inputs: [resized]
        outputs: [rescaled]
        scale: 0.00392156862745098
      - op: normalize
        inputs: [rescaled]
        outputs: [normalized]
        mean: [0.5, 0.5, 0.5]
        std: [0.5, 0.5, 0.5]
    outputs:
      - source: normalized
        name: encoder.pixel_values
        content: pixels
        dtype: float32
pipeline:
  models:
    encoder:
      filename: encoder.onnx.textproto
      type: encoder
    decoder:
      filename: decoder.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
  dataflow:
    - from: encoder.image_features
      to: decoder.image_features
      dtype: fp32
      device_transfer: false
  strategy:
    kind: composite
    stages:
      - name: encode_image
        strategy:
          kind: single_pass
          model: encoder
        run_on: prompt_only
      - name: decode_text
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
  vision:
    image_placeholder_token_id: 3
    image_token_id: 4
    token_count_source: per_tile
    tokens_per_tile: 1
    placeholder_per_image: true
    image_correspondence: prompt_order
"""

TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 3,
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
        "vocab": {
            "[UNK]": 0,
            "[EOS]": 1,
            "describe": 2,
            "<image>": 3,
            "img": 4,
            "tiny": 5,
            "vlm": 6,
            ".": 7,
        },
        "unk_token": "[UNK]",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPTS.parent / "tests/fixtures/tiny-vlm-image-input",
        help="Fixture directory to write.",
    )
    arguments = parser.parse_args()

    builder = load_tiny_vlm_builder()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    builder.build_encoder(output / "encoder.onnx.textproto")
    builder.build_decoder(output / "decoder.onnx.textproto")
    (output / "tokenizer.json").write_text(json.dumps(TOKENIZER, indent=2) + "\n")
    (output / "inference_metadata.yaml").write_text(METADATA)

    total = sum(path.stat().st_size for path in output.iterdir())
    print(f"Wrote {output} ({total} bytes)")


if __name__ == "__main__":
    main()
