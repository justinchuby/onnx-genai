#!/usr/bin/env python3
"""Generate single-node com.microsoft::MultiHeadAttention models (the operator
#1044's vectorised `sdpa_f32` path actually serves) with fully static shapes.

SYNTHETIC DATA NOTICE: no trained weights. Only the head/head_dim/sequence
dimensions come from public model configs; tensor contents are the benchmark
harness's deterministic synthetic pattern, fed identically to both runtimes.

Slot order (com.microsoft::MultiHeadAttention):
  0 query 1 key 2 value 3 bias 4 key_padding_mask 5 attention_bias
  6 past_key 7 past_value
Outputs: 0 output 1 present_key 2 present_value
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import onnx
from onnx import TensorProto, helper

OPSET = 17


def build_mha(
    path: Path,
    *,
    batch: int,
    num_heads: int,
    head_dim: int,
    q_seq: int,
    kv_seq: int,
    unidirectional: bool = False,
) -> None:
    hidden = num_heads * head_dim
    inputs = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch, q_seq, hidden]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch, kv_seq, hidden]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch, kv_seq, hidden]),
    ]
    node = helper.make_node(
        "MultiHeadAttention",
        ["query", "key", "value"],
        ["output"],
        name="mha",
        domain="com.microsoft",
        num_heads=num_heads,
        scale=float(1.0 / math.sqrt(head_dim)),
        unidirectional=int(unidirectional),
    )
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, q_seq, hidden])
    ]
    graph = helper.make_graph([node], "mha_bench", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", OPSET),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    # (name, batch, heads, head_dim, q_seq, kv_seq)
    cases = [
        ("bert_base_s128", 1, 12, 64, 128, 128),
        ("bert_base_s384", 1, 12, 64, 384, 384),
        ("bert_base_b8_s128", 8, 12, 64, 128, 128),
        ("bert_large_s128", 1, 16, 64, 128, 128),
        ("vit_b16_s197", 1, 12, 64, 197, 197),
        ("clip_l14_s257", 1, 16, 64, 257, 257),
        ("whisper_cross_s1500", 1, 12, 64, 448, 1500),
    ]
    for name, b, h, d, q, kv in cases:
        path = args.out_dir / f"mha_{name}.onnx"
        build_mha(path, batch=b, num_heads=h, head_dim=d, q_seq=q, kv_seq=kv)
        print(path.name)


if __name__ == "__main__":
    main()
