#!/usr/bin/env python3
"""CUDA-capable variant of the tiny Gemma4-style VLM `inputs_embeds` fixture.

Identical closed-form behaviour to ``build_tiny_gemma4_vlm.py`` (prompt ``[3, 7]``
-> generated ``[0, 5, 6, 7]``) but the **decoder additionally declares**
``attention_mask`` and ``position_ids`` graph inputs. The native CUDA decode path
mandates a declared attention mask (its device KV/mask bindings are allocated from
it), so the maskless base fixture cannot exercise the CUDA `inputs_embeds` decoder
introduced in Inc3a. Both new inputs are wired into a *zero* contribution so they
are genuine, non-prunable graph inputs while leaving the closed-form logits — and
therefore the generated token ids — byte-identical to the base fixture.

This fixture is consumed by ``native_cuda_pipeline_decoder_parity.rs`` to prove
native-CUDA-decoder-in-pipeline token parity (device-resident KV on the CUDA EP)
against the same closed-form ids the CPU native decoder produces.

Usage:
    python scripts/build_tiny_gemma4_vlm_cuda.py [out_dir] [--no-validate]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir

# Reuse every closed-form constant + builder from the base fixture so the two
# stay in lockstep; only the decoder graph + metadata differ here.
from build_tiny_gemma4_vlm import (  # type: ignore
    HIDDEN,
    LM_HEAD,
    TIE_BIAS,
    VOCAB,
    build_embedding,
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


def build_decoder_cuda(path: Path) -> None:
    """inputs_embeds[1,s,4] (+ attention_mask, position_ids, KV) -> logits + KV.

    Same head as the base fixture (``logits = inputs_embeds @ W + tie_bias``) with
    a growing ``Concat`` KV cache. ``attention_mask`` and ``position_ids`` are
    declared graph inputs (required so the native CUDA decode path can bind its
    device mask) but are left unconsumed, so the generated tokens stay identical
    to the maskless base fixture.
    """
    inputs_embeds = tensor_value(
        "inputs_embeds", ir.DataType.FLOAT, [1, "sequence", HIDDEN]
    )
    attention_mask = tensor_value(
        "attention_mask", ir.DataType.INT64, [1, "total_sequence"]
    )
    position_ids = tensor_value("position_ids", ir.DataType.INT64, [1, "sequence"])
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
    current_key = node("Unsqueeze", [inputs_embeds, kv_axis.outputs[0]], "current_key")
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
        [inputs_embeds, attention_mask, position_ids, past_key, past_value],
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
        name="tiny_gemma4_decoder_cuda",
    )
    save_model(
        ir.Model(
            graph, ir_version=8, producer_name="onnx-genai tiny-gemma4-vlm-cuda fixture"
        ),
        path,
    )


METADATA = """\
# CUDA-capable tiny Gemma4-style VLM composite fixture (Inc3a). Identical to
# tiny-gemma4-vlm but the decoder additionally declares attention_mask and
# position_ids inputs so the native CUDA decode path (which requires a declared
# mask for its device KV/mask bindings) can drive the inputs_embeds decoder while
# keeping the KV cache device-resident. Built by build_tiny_gemma4_vlm_cuda.py.
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
    """ORT reference decode for the CUDA fixture (supplies mask + position ids)."""
    import onnxruntime as ort

    vision = ort.InferenceSession(
        _textproto_bytes(output_dir / "vision_encoder.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )
    embedding = ort.InferenceSession(
        _textproto_bytes(output_dir / "embedding.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )
    decoder = ort.InferenceSession(
        _textproto_bytes(output_dir / "decoder.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )

    pixels = tiny_pixels()
    image_features = vision.run(None, {"pixel_values": pixels})[0]

    def embed(ids: list[int]) -> np.ndarray:
        return embedding.run(
            None,
            {"input_ids": np.array([ids], dtype=np.int64), "image_features": image_features},
        )[0]

    def decode(embeds: np.ndarray, past_k, past_v, past_len):
        seq = embeds.shape[1]
        total = past_len + seq
        return decoder.run(
            None,
            {
                "inputs_embeds": embeds.astype(np.float32),
                "attention_mask": np.ones((1, total), dtype=np.int64),
                "position_ids": np.array([list(range(past_len, total))], dtype=np.int64),
                "past_key_values.0.key": past_k,
                "past_key_values.0.value": past_v,
            },
        )

    past_k = np.zeros((1, 1, 0, HIDDEN), dtype=np.float32)
    past_v = np.zeros((1, 1, 0, HIDDEN), dtype=np.float32)
    logits, past_k, past_v = decode(embed(prompt), past_k, past_v, 0)
    past_len = len(prompt)
    generated = [int(logits[0, -1].argmax())]
    for _ in range(1, max_new_tokens):
        logits, past_k, past_v = decode(embed([generated[-1]]), past_k, past_v, past_len)
        past_len += 1
        generated.append(int(logits[0, -1].argmax()))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=str(
            Path(__file__).resolve().parent.parent
            / "tests/fixtures/tiny-gemma4-vlm-cuda"
        ),
        help="output fixture directory",
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_vision_encoder(out_dir / "vision_encoder.onnx.textproto")
    build_embedding(out_dir / "embedding.onnx.textproto")
    build_decoder_cuda(out_dir / "decoder.onnx.textproto")
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
    print(f"wrote CUDA fixture to {out_dir}")


if __name__ == "__main__":
    main()
