#!/usr/bin/env python3
"""Build the tiny deterministic text-to-image (txt2img) pipeline fixture.

This hand-constructs the smallest package that exercises the *full* declarative
diffusion contract the `onnx-genai generate --output-image` renderer drives:

    text_encoder (prompt_only) -> denoiser (iterative) -> vae (final_only)

  * ``text_encoder.onnx.textproto`` — ``input_ids [1, 77]`` int64 ->
    ``last_hidden_state [1, 77, 8]``, computed as ``float(input_ids) @ W`` so the
    fixture needs no large embedding table.
  * ``denoiser.onnx.textproto`` — one denoise step:
    ``noise_pred = sample * 0.5 + project(encoder_hidden_states)``. ``sample`` is
    loop-carried through a ``denoiser.noise_pred -> denoiser.sample`` self-edge,
    and ``encoder_hidden_states`` is the classifier-free-guidance conditioning
    port, so cond/uncond passes produce different predictions.
  * ``vae.onnx.textproto`` — a ``final_only`` decoder mapping the final
    ``latent [1, 4, 1, 1]`` to ``image [1, 3, 8, 8]``.

The latent is 1x1 so the rendered image is 8x8 (the VAE's 8x spatial factor),
keeping the fixture tiny while still going through the real renderer.

Everything is affine and deterministic, so the output is reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir

CLIP_CONTEXT_LENGTH = 77
CLIP_END_OF_TEXT_ID = 49407
HIDDEN = 8
LATENT_CHANNELS = 4
IMAGE_SIDE = 8


def tensor_value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def initializer(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.Tensor(array, name=name))


def node(
    op_type: str,
    inputs: list[ir.Value],
    output: str,
    *,
    name: str = "",
    attributes: dict[str, ir.Attr] | None = None,
) -> ir.Node:
    return ir.Node(
        "",
        op_type,
        inputs,
        attributes or (),
        outputs=[ir.Value(name=output)],
        name=name,
    )


def save_model(model: ir.Model, path: Path) -> None:
    ir.save(model, path, format="textproto")
    onnx.checker.check_model(ir.to_proto(model))


def typed(value: ir.Value, dtype: ir.DataType, shape: list[int]) -> ir.Value:
    value.type = ir.TensorType(dtype)
    value.shape = ir.Shape(shape)
    return value


def make_model(graph: ir.Graph) -> ir.Model:
    return ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-txt2img")


def build_text_encoder(path: Path) -> None:
    """input_ids [1, 77] int64 -> last_hidden_state [1, 77, HIDDEN]."""
    input_ids = tensor_value("input_ids", ir.DataType.INT64, [1, CLIP_CONTEXT_LENGTH])
    projection = initializer(
        "text_projection",
        (np.arange(HIDDEN, dtype=np.float32).reshape(1, HIDDEN) + 1.0) / 1e4,
    )
    reshape_shape = initializer(
        "text_reshape", np.array([1, CLIP_CONTEXT_LENGTH, 1], dtype=np.int64)
    )

    cast = node(
        "Cast",
        [input_ids],
        "ids_float",
        attributes={"to": ir.Attr("to", ir.AttributeType.INT, ir.DataType.FLOAT.value)},
    )
    reshape = node("Reshape", [cast.outputs[0], reshape_shape], "ids_column")
    matmul = node("MatMul", [reshape.outputs[0], projection], "last_hidden_state")
    typed(matmul.outputs[0], ir.DataType.FLOAT, [1, CLIP_CONTEXT_LENGTH, HIDDEN])

    graph = ir.Graph(
        [input_ids],
        [matmul.outputs[0]],
        nodes=[cast, reshape, matmul],
        initializers=[projection, reshape_shape],
        opset_imports={"": 13},
        name="tiny_text_encoder",
    )
    save_model(make_model(graph), path)


def build_denoiser(path: Path) -> None:
    """sample + encoder_hidden_states -> noise_pred (loop-carried)."""
    sample = tensor_value("sample", ir.DataType.FLOAT, [1, LATENT_CHANNELS, 1, 1])
    hidden = tensor_value(
        "encoder_hidden_states", ir.DataType.FLOAT, [1, CLIP_CONTEXT_LENGTH, HIDDEN]
    )
    flat_shape = initializer(
        "denoiser_flat", np.array([1, CLIP_CONTEXT_LENGTH * HIDDEN], dtype=np.int64)
    )
    condition_projection = initializer(
        "condition_projection",
        np.full((CLIP_CONTEXT_LENGTH * HIDDEN, LATENT_CHANNELS), 1e-3, dtype=np.float32),
    )
    latent_shape = initializer(
        "denoiser_latent_shape", np.array([1, LATENT_CHANNELS, 1, 1], dtype=np.int64)
    )
    half = initializer("half", np.array([0.5], dtype=np.float32))

    flatten = node("Reshape", [hidden, flat_shape], "hidden_flat")
    project = node("MatMul", [flatten.outputs[0], condition_projection], "hidden_projected")
    reshape = node("Reshape", [project.outputs[0], latent_shape], "condition")
    scale = node("Mul", [sample, half], "sample_scaled")
    add = node("Add", [scale.outputs[0], reshape.outputs[0]], "noise_pred")
    typed(add.outputs[0], ir.DataType.FLOAT, [1, LATENT_CHANNELS, 1, 1])

    graph = ir.Graph(
        [sample, hidden],
        [add.outputs[0]],
        nodes=[flatten, project, reshape, scale, add],
        initializers=[flat_shape, condition_projection, latent_shape, half],
        opset_imports={"": 13},
        name="tiny_denoiser",
    )
    save_model(make_model(graph), path)


def build_vae(path: Path) -> None:
    """latent [1, 4, 1, 1] -> image [1, 3, 8, 8] in [-1, 1]."""
    latent = tensor_value("latent", ir.DataType.FLOAT, [1, LATENT_CHANNELS, 1, 1])
    flat_shape = initializer("vae_flat", np.array([1, LATENT_CHANNELS], dtype=np.int64))
    pixels = IMAGE_SIDE * IMAGE_SIDE * 3
    decoder = initializer(
        "vae_decoder",
        np.linspace(-1.0, 1.0, LATENT_CHANNELS * pixels, dtype=np.float32).reshape(
            LATENT_CHANNELS, pixels
        ),
    )
    image_shape = initializer(
        "vae_image_shape", np.array([1, 3, IMAGE_SIDE, IMAGE_SIDE], dtype=np.int64)
    )

    flatten = node("Reshape", [latent, flat_shape], "latent_flat")
    decode = node("MatMul", [flatten.outputs[0], decoder], "image_flat")
    reshape = node("Reshape", [decode.outputs[0], image_shape], "image")
    typed(reshape.outputs[0], ir.DataType.FLOAT, [1, 3, IMAGE_SIDE, IMAGE_SIDE])

    graph = ir.Graph(
        [latent],
        [reshape.outputs[0]],
        nodes=[flatten, decode, reshape],
        initializers=[flat_shape, decoder, image_shape],
        opset_imports={"": 13},
        name="tiny_vae",
    )
    save_model(make_model(graph), path)


def write_tokenizer(path: Path) -> None:
    """A word-level CLIP-shaped tokenizer with the real end-of-text id."""
    vocabulary = {"<|startoftext|>": 49406, "<|endoftext|>": CLIP_END_OF_TEXT_ID}
    for index, word in enumerate(
        ["an", "astronaut", "riding", "a", "horse", "blurry", "low", "quality", "cat"]
    ):
        vocabulary[word] = index + 1
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocabulary,
            "unk_token": "<|endoftext|>",
        },
    }
    path.write_text(json.dumps(tokenizer, indent=2))


METADATA = f"""pipeline:
  models:
    text_encoder:
      filename: text_encoder.onnx.textproto
      type: encoder
    denoiser:
      filename: denoiser.onnx.textproto
      type: denoiser
    vae:
      filename: vae.onnx.textproto
      type: vae
  dataflow:
    - from: text_encoder.last_hidden_state
      to: denoiser.encoder_hidden_states
      dtype: fp32
    - from: denoiser.noise_pred
      to: denoiser.sample
      dtype: fp32
    - from: denoiser.noise_pred
      to: vae.latent
      dtype: fp32
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 3
    guidance_scale: 7.5
    cfg_conditioning_input: encoder_hidden_states
  phases:
    text_encoder:
      run_on: prompt_only
    denoiser:
      run_on: every_step
    vae:
      run_on: final_only
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests/fixtures/tiny-txt2img",
        help="Fixture directory to write.",
    )
    arguments = parser.parse_args()

    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    build_text_encoder(output / "text_encoder.onnx.textproto")
    build_denoiser(output / "denoiser.onnx.textproto")
    build_vae(output / "vae.onnx.textproto")
    write_tokenizer(output / "tokenizer.json")
    (output / "inference_metadata.yaml").write_text(METADATA)
    (output / "run.json").write_text(json.dumps({"latent_channels": LATENT_CHANNELS}))

    total = sum(path.stat().st_size for path in output.iterdir())
    print(f"Wrote {output} ({total} bytes)")


if __name__ == "__main__":
    main()
