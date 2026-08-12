#!/usr/bin/env python3
"""Capturing tiny Gemma4-style ``inputs_embeds`` pipeline fixture (Inc3c).

Identical closed-form behaviour to ``build_tiny_gemma4_vlm_cuda.py`` (prompt
``[3, 7]`` -> generated ``[0, 5, 6, 7]``) and the same composite pipeline
(vision -> embedding fusion -> autoregressive ``inputs_embeds`` decoder), with
ONE decisive difference: the decoder routes its KV cache through a real
``com.microsoft.GroupQueryAttention`` op instead of a naive ``Concat`` cache.

Why this fixture exists
-----------------------
The other tiny CUDA decoders (``tiny-gemma4-vlm-cuda``,
``tiny-gemma4-vlm-cuda-routed``) grow their KV with ``Concat``, whose consumers
read the *logical* KV length. The native CUDA decode path therefore refuses
whole-graph CUDA-graph capture on them (the KV/mask bindings expose a growing
logical prefix), so they can prove the eager ``inputs_embeds``/routed device path
but NOT the Inc3c *captured* per-step-input path.

``GroupQueryAttention`` reads ``seqlens_k`` / ``total_sequence_length`` and
consumes the past KV at fixed physical capacity — exactly the capacity-aware
kernel shape the native decoder recognises as CUDA-graph-capture-safe. So this
decoder's KV/mask bindings do NOT expose a logical prefix and graph capture
*engages*, letting ``native_cuda_captured_step_inputs_parity.rs`` prove the
Inc3c captured per-step-input path genuinely runs (not a silent decline to
eager) while producing token-identical output.

Determinism seam
----------------
Token determinism and the GQA numerical path are deliberately isolated:

  * ``logits = inputs_embeds @ LM_HEAD + tie_bias`` — computed *directly* from
    ``inputs_embeds`` (the proven base-fixture head), so the generated token ids
    are exactly ``[0, 5, 6, 7]`` and bit-stable between the CPU and CUDA native
    kernels (no softmax float sensitivity on the token path).
  * The ``GroupQueryAttention`` op consumes the past KV (Q/K/V derived from
    ``inputs_embeds`` by real MatMuls, so ``inputs_embeds`` genuinely flows into
    a CUDA op on-device) and emits ``present.0.key`` / ``present.0.value`` as the
    growing device-resident KV contract. Its first output (``attn_out``) is
    intentionally unused: the KV outputs keep the node live and capacity-aware
    while the token stream stays closed-form.

Usage:
    python scripts/build_tiny_gqa_embeds_cuda.py [out_dir]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from onnxscript import ir

# Reuse every closed-form constant + shared builder from the base fixtures so
# the token stream stays in lockstep; only the decoder graph differs here.
from build_tiny_gemma4_vlm import (  # type: ignore
    HIDDEN,
    LM_HEAD,
    TIE_BIAS,
    VOCAB,
    build_embedding,
    build_vision_encoder,
    constant,
    initializer,
    node,
    save_model,
    tensor_value,
    write_tokenizer,
)

# GroupQueryAttention geometry: query hidden = num_heads * head_dim = HIDDEN(4);
# kv hidden = kv_num_heads * head_dim = 2. Matches tiny-native-scalar-gqa.
NUM_HEADS = 2
KV_NUM_HEADS = 1
HEAD_DIM = 2
KV_HIDDEN = KV_NUM_HEADS * HEAD_DIM

# Fixed KV projections (inert w.r.t. token ids — see module docstring). Small,
# deterministic weights so Q/K/V genuinely derive from inputs_embeds on-device.
WK = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
WV = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)


def build_decoder_gqa(path: Path) -> None:
    """inputs_embeds[1,s,4] (+ attention_mask, position_ids, KV) -> logits + KV.

    ``logits = inputs_embeds @ LM_HEAD + tie_bias`` (same head as the base
    fixture, so tokens are byte-identical ``[0, 5, 6, 7]``). The KV cache flows
    through a real ``GroupQueryAttention`` op whose ``present`` outputs are the
    growing device-resident KV — a capacity-aware kernel, so the native CUDA
    decode path *captures* the per-step-input decode (Inc3c).
    """
    inputs_embeds = tensor_value(
        "inputs_embeds", ir.DataType.FLOAT, [1, "sequence", HIDDEN]
    )
    attention_mask = tensor_value(
        "attention_mask", ir.DataType.INT64, [1, "total_sequence"]
    )
    position_ids = tensor_value("position_ids", ir.DataType.INT64, [1, "sequence"])
    past_key = tensor_value(
        "past_key_values.0.key", ir.DataType.FLOAT, [1, KV_NUM_HEADS, "past_sequence", HEAD_DIM]
    )
    past_value = tensor_value(
        "past_key_values.0.value", ir.DataType.FLOAT, [1, KV_NUM_HEADS, "past_sequence", HEAD_DIM]
    )

    # ── Token path: deterministic, bit-stable across CPU/CUDA. ───────────────
    lm_head = initializer("lm_head", LM_HEAD)
    matmul = node("MatMul", [inputs_embeds, lm_head], "logits_base")  # [1, s, 8]
    tie_bias = initializer("tie_bias", TIE_BIAS)
    logits = node("Add", [matmul.outputs[0], tie_bias], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape([1, "sequence", VOCAB])

    # ── KV path: capacity-aware GroupQueryAttention (the capture seam). ──────
    # Q = inputs_embeds (query hidden == HIDDEN); K, V = inputs_embeds @ W.
    wk = initializer("wk", WK)
    wv = initializer("wv", WV)
    key = node("MatMul", [inputs_embeds, wk], "gqa_key")  # [1, s, 2]
    value = node("MatMul", [inputs_embeds, wv], "gqa_value")  # [1, s, 2]

    # seqlens_k = sum(attention_mask) - 1 (int32); total_sequence_length = sum.
    total_i64 = node(
        "ReduceSum",
        [attention_mask],
        "total_i64",
        attributes=[ir.AttrInt64s("axes", [0, 1]), ir.AttrInt64("keepdims", 0)],
    )
    one = constant("one", np.array(1, dtype=np.int64))
    seqlens_i64 = node("Sub", [total_i64.outputs[0], one.outputs[0]], "seqlens_i64")
    seqlens_k = node(
        "Cast",
        [seqlens_i64.outputs[0]],
        "seqlens_k",
        attributes=[ir.AttrInt64("to", int(ir.DataType.INT32))],
    )
    total_sequence_length = node(
        "Cast",
        [total_i64.outputs[0]],
        "total_sequence_length",
        attributes=[ir.AttrInt64("to", int(ir.DataType.INT32))],
    )

    attn_out = ir.Value(name="gqa_attn_out")
    present_key = ir.Value(name="present.0.key")
    present_value = ir.Value(name="present.0.value")
    gqa = ir.Node(
        "com.microsoft",
        "GroupQueryAttention",
        [
            inputs_embeds,
            key.outputs[0],
            value.outputs[0],
            past_key,
            past_value,
            seqlens_k.outputs[0],
            total_sequence_length.outputs[0],
        ],
        [
            ir.AttrInt64("num_heads", NUM_HEADS),
            ir.AttrInt64("kv_num_heads", KV_NUM_HEADS),
        ],
        outputs=[attn_out, present_key, present_value],
    )
    for out in (present_key, present_value):
        out.type = ir.TensorType(ir.DataType.FLOAT)
        out.shape = ir.Shape([1, KV_NUM_HEADS, "total_sequence", HEAD_DIM])

    graph = ir.Graph(
        [inputs_embeds, attention_mask, position_ids, past_key, past_value],
        [logits.outputs[0], present_key, present_value],
        nodes=[
            matmul,
            logits,
            key,
            value,
            total_i64,
            one,
            seqlens_i64,
            seqlens_k,
            total_sequence_length,
            gqa,
        ],
        initializers=[lm_head, tie_bias, wk, wv],
        opset_imports={"": 11, "com.microsoft": 1},
        name="tiny_gqa_embeds_decoder",
    )
    # GroupQueryAttention is a com.microsoft contrib op with no onnx.checker
    # schema, so save the textproto directly (bypassing save_model's checker),
    # exactly as the Rust-built tiny-native-scalar-gqa fixture does.
    ir.save(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-gqa-embeds-cuda fixture",
        ),
        path,
        format="textproto",
    )


METADATA = """\
# Capturing tiny Gemma4-style inputs_embeds pipeline fixture (Inc3c). Identical
# closed-form tokens ([3, 7] -> [0, 5, 6, 7]) and pipeline shape as
# tiny-gemma4-vlm-cuda, but the decoder routes its KV through a real
# GroupQueryAttention op (seqlens_k / total_sequence_length), so its KV/mask
# bindings stay at fixed physical capacity and the native CUDA decode path
# ENGAGES whole-graph CUDA-graph capture. This lets the Inc3c captured
# per-step-input decode path be proven (not just the eager path). Built by
# build_tiny_gqa_embeds_cuda.py.
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
  strategy:
    kind: composite
    stages:
      - name: encode_vision
        strategy:
          kind: single_pass
          model: vision_encoder
      - name: fuse_embeddings
        strategy:
          kind: single_pass
          model: embedding
      - name: decode
        strategy:
          kind: autoregressive
          decoder: decoder
          max_tokens: 4
  phases:
    vision_encoder:
      run_on: prompt_only
    embedding:
      run_on: every_step
    decoder:
      run_on: every_step
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=str(
            Path(__file__).resolve().parents[1]
            / "tests"
            / "fixtures"
            / "tiny-gqa-embeds-cuda"
        ),
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_vision_encoder(out_dir / "vision_encoder.onnx.textproto")
    build_embedding(out_dir / "embedding.onnx.textproto")
    build_decoder_gqa(out_dir / "decoder.onnx.textproto")
    write_tokenizer(out_dir / "tokenizer.json")
    (out_dir / "inference_metadata.yaml").write_text(METADATA)
    print(f"wrote capturing GQA inputs_embeds fixture to {out_dir}")


if __name__ == "__main__":
    main()
