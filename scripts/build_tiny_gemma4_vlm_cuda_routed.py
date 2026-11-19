#!/usr/bin/env python3
"""Routed-port variant of the CUDA Gemma4-style VLM fixture (Inc3b).

Extends ``tiny-gemma4-vlm-cuda`` with a **generic routed port**: the every_step
``embedding`` component emits a *second* output ``router_state`` that a pipeline
dataflow edge routes to a ``router_state`` input on the decoder. That port has no
generated role and no ``inputs_embeds`` role — it is a `NativeStepInputSource::
Routed` input, exactly the class Inc3a refused on CUDA. The decoder consumes it
through a real ``MatMul`` by a zero matrix (so it genuinely flows through a CUDA
op) with zero contribution to the logits, keeping the closed-form tokens
``[0, 5, 6, 7]`` identical to the base fixture.

Used by ``native_cuda_routed_pipeline_decoder_parity.rs`` to prove the native
CUDA decoder binds an arbitrary routed port on-device per step (KV stays device-
resident) with token parity vs the native CPU path.

Usage:
    python scripts/build_tiny_gemma4_vlm_cuda_routed.py [out_dir] [--no-validate]
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
from build_tiny_gemma4_vlm import build_embedding as _base_build_embedding  # noqa: F401


def build_embedding_routed(path: Path) -> None:
    """input_ids + image_features -> inputs_embeds AND router_state.

    Same fused embedding as the base fixture, plus a second output
    ``router_state`` (an identity copy of ``inputs_embeds``) that the pipeline
    routes to the decoder's generic routed port.
    """
    from build_tiny_gemma4_vlm import EMBEDDING_TABLE, PLACEHOLDER_ID

    input_ids = tensor_value("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    image_features = tensor_value("image_features", ir.DataType.FLOAT, [1, 1, HIDDEN])

    embedding_table = initializer("embedding_table", EMBEDDING_TABLE)
    text_embed = node(
        "Gather",
        [embedding_table, input_ids],
        "text_embed",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    placeholder = constant("placeholder_id", np.array(PLACEHOLDER_ID, dtype=np.int64))
    is_placeholder = node("Equal", [input_ids, placeholder.outputs[0]], "is_placeholder")
    mask_f = node(
        "Cast",
        [is_placeholder.outputs[0]],
        "placeholder_mask",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    mask_axis = constant("mask_axis", np.array([2], dtype=np.int64))
    mask_col = node("Unsqueeze", [mask_f.outputs[0], mask_axis.outputs[0]], "placeholder_mask_col")
    image_contrib = node("Mul", [mask_col.outputs[0], image_features], "image_contrib")
    inputs_embeds = node("Add", [text_embed.outputs[0], image_contrib.outputs[0]], "inputs_embeds")
    inputs_embeds.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    inputs_embeds.outputs[0].shape = ir.Shape([1, "sequence", HIDDEN])

    router_state = node("Identity", [inputs_embeds.outputs[0]], "router_state")
    router_state.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    router_state.outputs[0].shape = ir.Shape([1, "sequence", HIDDEN])

    graph = ir.Graph(
        [input_ids, image_features],
        [inputs_embeds.outputs[0], router_state.outputs[0]],
        nodes=[
            text_embed,
            placeholder,
            is_placeholder,
            mask_f,
            mask_axis,
            mask_col,
            image_contrib,
            inputs_embeds,
            router_state,
        ],
        initializers=[embedding_table],
        opset_imports={"": 13},
        name="tiny_gemma4_embedding_routed",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-gemma4-vlm-cuda-routed fixture"),
        path,
    )


def build_decoder_routed(path: Path) -> None:
    """inputs_embeds + router_state (+ mask, position, KV) -> logits + KV.

    Same head as the CUDA fixture with a growing ``Concat`` KV cache, plus a
    generic routed port ``router_state`` consumed through a real ``MatMul`` by a
    zero matrix (``[HIDDEN, VOCAB]``) added into the logits — a genuine CUDA op
    that contributes exactly zero, so the tokens stay ``[0, 5, 6, 7]``.
    ``attention_mask`` / ``position_ids`` are declared but unconsumed (as in the
    base CUDA fixture — real decoders reduce the mask over the sequence axis; the
    cuDNN all-axes-scalar-reduce limitation is a fixture artifact, see
    mary-inc3a-realmodel-mask-check.md).
    """
    inputs_embeds = tensor_value("inputs_embeds", ir.DataType.FLOAT, [1, "sequence", HIDDEN])
    router_state = tensor_value("router_state", ir.DataType.FLOAT, [1, "sequence", HIDDEN])
    attention_mask = tensor_value("attention_mask", ir.DataType.INT64, [1, "total_sequence"])
    position_ids = tensor_value("position_ids", ir.DataType.INT64, [1, "sequence"])
    past_key = tensor_value("past_key_values.0.key", ir.DataType.FLOAT, [1, 1, "past_sequence", HIDDEN])
    past_value = tensor_value("past_key_values.0.value", ir.DataType.FLOAT, [1, 1, "past_sequence", HIDDEN])

    # A zero-valued block-quantized MoE sits on the live inputs_embeds path. Its
    # output is added residually, preserving the fixture's closed-form tokens
    # while making workspace scale with the exact routed prefill sequence.
    qmoe_hidden = 32
    qmoe_projection = initializer(
        "qmoe_projection", np.zeros((HIDDEN, qmoe_hidden), dtype=np.float32)
    )
    qmoe_projected = node(
        "MatMul", [inputs_embeds, qmoe_projection], "qmoe_projected"
    )
    qmoe_input_shape = constant(
        "qmoe_input_shape", np.array([-1, qmoe_hidden], dtype=np.int64)
    )
    qmoe_input = node(
        "Reshape",
        [qmoe_projected.outputs[0], qmoe_input_shape.outputs[0]],
        "qmoe_input",
    )
    router_logits_3d = node(
        "ReduceMean",
        [router_state],
        "qmoe_router_logits_3d",
        attributes=[
            ir.AttrInt64s("axes", [2]),
            ir.AttrInt64("keepdims", 1),
        ],
    )
    router_shape = constant("qmoe_router_shape", np.array([-1, 1], dtype=np.int64))
    router_logits = node(
        "Reshape",
        [router_logits_3d.outputs[0], router_shape.outputs[0]],
        "qmoe_router_logits",
    )
    packed_zero = np.zeros((1, qmoe_hidden, 1, 17), dtype=np.uint8)
    qmoe_fc1 = initializer("qmoe_fc1", packed_zero)
    qmoe_fc2 = initializer("qmoe_fc2", packed_zero.copy())
    qmoe_fc1_bias = initializer(
        "qmoe_fc1_bias", np.zeros((1, qmoe_hidden), dtype=np.float32)
    )
    qmoe_fc2_bias = initializer(
        "qmoe_fc2_bias", np.zeros((1, qmoe_hidden), dtype=np.float32)
    )
    qmoe = ir.Node(
        "pkg.nxrt",
        "BlockQuantizedMoE",
        [
            qmoe_input.outputs[0],
            router_logits.outputs[0],
            qmoe_fc1,
            qmoe_fc1_bias,
            qmoe_fc2,
            qmoe_fc2_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        {
            "k": ir.AttrInt64("k", 1),
            "activation_type": ir.AttrString("activation_type", "identity"),
            "normalize_routing_weights": ir.AttrInt64("normalize_routing_weights", 0),
            "swiglu_fusion": ir.AttrInt64("swiglu_fusion", 0),
            "fc1_format": ir.AttrString("fc1_format", "mxfp4"),
            "fc2_format": ir.AttrString("fc2_format", "mxfp4"),
            "block_layout_version": ir.AttrInt64("block_layout_version", 1),
        },
        outputs=[ir.Value(name="qmoe_output")],
    )
    qmoe.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    qmoe.outputs[0].shape = ir.Shape(["rows", qmoe_hidden])
    qmoe_reduce = node(
        "ReduceMean",
        [qmoe.outputs[0]],
        "qmoe_reduce",
        attributes=[
            ir.AttrInt64s("axes", [1]),
            ir.AttrInt64("keepdims", 1),
        ],
    )
    qmoe_restore_shape = constant(
        "qmoe_restore_shape", np.array([1, -1, 1], dtype=np.int64)
    )
    qmoe_restored = node(
        "Reshape",
        [qmoe_reduce.outputs[0], qmoe_restore_shape.outputs[0]],
        "qmoe_restored",
    )
    qmoe_residual = node(
        "Add", [inputs_embeds, qmoe_restored.outputs[0]], "qmoe_residual"
    )

    lm_head = initializer("lm_head", LM_HEAD)
    matmul = node("MatMul", [qmoe_residual.outputs[0], lm_head], "logits_base")
    tie_bias = initializer("tie_bias", TIE_BIAS)
    logits_biased = node("Add", [matmul.outputs[0], tie_bias], "logits_biased")

    # Consume the routed port via a real MatMul by a zero [HIDDEN, VOCAB] matrix
    # -> [1, s, VOCAB] of zeros, added into the logits (contributes exactly 0).
    router_zero = initializer("router_zero", np.zeros((HIDDEN, VOCAB), dtype=np.float32))
    router_contrib = node("MatMul", [router_state, router_zero], "router_contrib")
    logits = node("Add", [logits_biased.outputs[0], router_contrib.outputs[0]], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape([1, "sequence", VOCAB])

    kv_axis = constant("kv_axis", np.array([1], dtype=np.int64))
    current_key = node("Unsqueeze", [inputs_embeds, kv_axis.outputs[0]], "current_key")
    value_offset = initializer("value_offset", np.array(0.5, dtype=np.float32))
    current_value = node("Add", [current_key.outputs[0], value_offset], "current_value")
    present_key = node("Concat", [past_key, current_key.outputs[0]], "present.0.key", attributes=[ir.AttrInt64("axis", 2)])
    present_value = node("Concat", [past_value, current_value.outputs[0]], "present.0.value", attributes=[ir.AttrInt64("axis", 2)])
    for out in (present_key.outputs[0], present_value.outputs[0]):
        out.type = ir.TensorType(ir.DataType.FLOAT)
        out.shape = ir.Shape([1, 1, "total_sequence", HIDDEN])

    graph = ir.Graph(
        [inputs_embeds, router_state, attention_mask, position_ids, past_key, past_value],
        [logits.outputs[0], present_key.outputs[0], present_value.outputs[0]],
        nodes=[
            qmoe_projected,
            qmoe_input_shape,
            qmoe_input,
            router_logits_3d,
            router_shape,
            router_logits,
            qmoe,
            qmoe_reduce,
            qmoe_restore_shape,
            qmoe_restored,
            qmoe_residual,
            matmul,
            logits_biased,
            router_contrib,
            logits,
            kv_axis,
            current_key,
            current_value,
            present_key,
            present_value,
        ],
        initializers=[
            qmoe_fc1,
            qmoe_fc2,
            qmoe_fc1_bias,
            qmoe_fc2_bias,
            qmoe_projection,
            lm_head,
            tie_bias,
            router_zero,
            value_offset,
        ],
        opset_imports={"": 13, "pkg.nxrt": 1},
        name="tiny_gemma4_decoder_routed",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-gemma4-vlm-cuda-routed fixture"),
        path,
    )


METADATA = """\
# Routed-port CUDA tiny Gemma4-style VLM fixture (Inc3b). Extends
# tiny-gemma4-vlm-cuda with a generic routed decoder port `router_state` fed by a
# dataflow edge from the every_step embedding component, proving the native CUDA
# decoder binds arbitrary routed ports on-device. Built by
# build_tiny_gemma4_vlm_cuda_routed.py.
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
    - from: embedding.router_state
      to: decoder.router_state
      dtype: fp32
      device_transfer: false
  phases:
    embedding:
      run_on: every_step
  strategy:
    kind: composite
    stages:
      - name: encode_vision
        strategy:
          kind: single_pass
          model: vision_encoder
      - name: decode
        strategy:
          kind: autoregressive
          decoder: decoder
          max_tokens: 4
"""


def validate_with_ort(output_dir: Path, prompt: list[int], max_new_tokens: int) -> list[int]:
    """ORT reference decode (supplies mask + position + the routed port)."""
    import onnxruntime as ort

    vision = ort.InferenceSession(_textproto_bytes(output_dir / "vision_encoder.onnx.textproto"), providers=["CPUExecutionProvider"])
    embedding = ort.InferenceSession(_textproto_bytes(output_dir / "embedding.onnx.textproto"), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(_textproto_bytes(output_dir / "decoder.onnx.textproto"), providers=["CPUExecutionProvider"])

    pixels = tiny_pixels()
    image_features = vision.run(None, {"pixel_values": pixels})[0]

    def embed(ids: list[int]):
        embeds, router = embedding.run(None, {"input_ids": np.array([ids], dtype=np.int64), "image_features": image_features})
        return embeds, router

    def decode(embeds, router, past_k, past_v, past_len):
        seq = embeds.shape[1]
        total = past_len + seq
        return decoder.run(
            None,
            {
                "inputs_embeds": embeds.astype(np.float32),
                "router_state": router.astype(np.float32),
                "attention_mask": np.ones((1, total), dtype=np.int64),
                "position_ids": np.array([list(range(past_len, total))], dtype=np.int64),
                "past_key_values.0.key": past_k,
                "past_key_values.0.value": past_v,
            },
        )

    past_k = np.zeros((1, 1, 0, HIDDEN), dtype=np.float32)
    past_v = np.zeros((1, 1, 0, HIDDEN), dtype=np.float32)
    embeds, router = embed(prompt)
    logits, past_k, past_v = decode(embeds, router, past_k, past_v, 0)
    past_len = len(prompt)
    generated = [int(logits[0, -1].argmax())]
    for _ in range(1, max_new_tokens):
        embeds, router = embed([generated[-1]])
        logits, past_k, past_v = decode(embeds, router, past_k, past_v, past_len)
        past_len += 1
        generated.append(int(logits[0, -1].argmax()))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "tests/fixtures/tiny-gemma4-vlm-cuda-routed"),
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_vision_encoder(out_dir / "vision_encoder.onnx.textproto")
    build_embedding_routed(out_dir / "embedding.onnx.textproto")
    build_decoder_routed(out_dir / "decoder.onnx.textproto")
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
    print(f"wrote routed-port CUDA fixture to {out_dir}")


if __name__ == "__main__":
    main()
