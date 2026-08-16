#!/usr/bin/env python3
"""Generate single-node com.microsoft::GroupQueryAttention models with fully
static shapes for native-vs-ORT A/B benchmarking.

SYNTHETIC DATA NOTICE: the graphs carry no trained weights. Only the *shapes*
are taken from public model configs (Qwen3-0.6B, Qwen2.5-0.5B, Phi-3-mini-4k,
Llama-3-8B). Tensor contents are the deterministic synthetic pattern that
`bench_generic` feeds identically to both runtimes.

Semantically-constrained integer inputs (seqlens_k, total_sequence_length) are
baked as *initializers* so the benchmark harness cannot fill them with the
generic `i % 17` synthetic pattern.

Slot order (com.microsoft::GroupQueryAttention):
  0 query 1 key 2 value 3 past_key 4 past_value 5 seqlens_k
  6 total_sequence_length 7 cos_cache 8 sin_cache
Outputs: 0 output 1 present_key 2 present_value
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
MS_OPSET = 1


def build_gqa(
    path: Path,
    *,
    batch: int,
    num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_seq: int,
    past_seq: int,
    do_rotary: bool = False,
    rotary_interleaved: bool = False,
    local_window_size: int = -1,
    softcap: float = 0.0,
    cache_capacity: int = 0,
) -> None:
    total = past_seq + q_seq
    # `cache_capacity > 0` models the production decode layout: the past buffer
    # is pre-allocated at max_sequence_length and only `total` rows are live, so
    # past and present carry identical shapes and a runtime may append in place.
    capacity = cache_capacity if cache_capacity > 0 else past_seq
    if capacity < total and cache_capacity > 0:
        raise ValueError("cache_capacity must be >= past_seq + q_seq")
    present_seq = max(capacity, total)
    hidden = num_heads * head_dim
    kv_hidden = kv_num_heads * head_dim

    inputs = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch, q_seq, hidden]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch, q_seq, kv_hidden]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch, q_seq, kv_hidden]),
    ]
    node_inputs = ["query", "key", "value"]

    initializers = []
    if capacity > 0:
        inputs.append(
            helper.make_tensor_value_info(
                "past_key", TensorProto.FLOAT, [batch, kv_num_heads, capacity, head_dim]
            )
        )
        inputs.append(
            helper.make_tensor_value_info(
                "past_value", TensorProto.FLOAT, [batch, kv_num_heads, capacity, head_dim]
            )
        )
        node_inputs += ["past_key", "past_value"]
    else:
        node_inputs += ["", ""]

    seqlens_k = np.full((batch,), total - 1, dtype=np.int32)
    initializers.append(numpy_helper.from_array(seqlens_k, "seqlens_k"))
    initializers.append(
        numpy_helper.from_array(np.array(total, dtype=np.int32), "total_sequence_length")
    )
    node_inputs += ["seqlens_k", "total_sequence_length"]

    if do_rotary:
        half = head_dim // 2
        pos = np.arange(present_seq, dtype=np.float32)[:, None]
        inv = (1.0 / (10000.0 ** (np.arange(half, dtype=np.float32) / half)))[None, :]
        ang = pos * inv
        initializers.append(
            numpy_helper.from_array(np.cos(ang).astype(np.float32), "cos_cache")
        )
        initializers.append(
            numpy_helper.from_array(np.sin(ang).astype(np.float32), "sin_cache")
        )
        node_inputs += ["cos_cache", "sin_cache"]

    attrs = {
        "num_heads": num_heads,
        "kv_num_heads": kv_num_heads,
        "scale": float(1.0 / math.sqrt(head_dim)),
        "do_rotary": int(do_rotary),
        "rotary_interleaved": int(rotary_interleaved),
        "local_window_size": local_window_size,
    }
    if softcap:
        attrs["softcap"] = float(softcap)

    node = helper.make_node(
        "GroupQueryAttention",
        node_inputs,
        ["output", "present_key", "present_value"],
        name="gqa",
        domain="com.microsoft",
        **attrs,
    )

    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, q_seq, hidden]),
        helper.make_tensor_value_info(
            "present_key", TensorProto.FLOAT, [batch, kv_num_heads, present_seq, head_dim]
        ),
        helper.make_tensor_value_info(
            "present_value", TensorProto.FLOAT, [batch, kv_num_heads, present_seq, head_dim]
        ),
    ]

    graph = helper.make_graph([node], "gqa_bench", inputs, outputs, initializer=initializers)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", OPSET),
            helper.make_opsetid("com.microsoft", MS_OPSET),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--num-heads", type=int, required=True)
    ap.add_argument("--kv-num-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--q-seq", type=int, default=1)
    ap.add_argument("--past-seq", type=int, default=0)
    ap.add_argument("--do-rotary", action="store_true")
    ap.add_argument("--rotary-interleaved", action="store_true")
    ap.add_argument("--local-window-size", type=int, default=-1)
    ap.add_argument("--softcap", type=float, default=0.0)
    ap.add_argument("--cache-capacity", type=int, default=0)
    args = ap.parse_args()
    build_gqa(
        args.out,
        batch=args.batch,
        num_heads=args.num_heads,
        kv_num_heads=args.kv_num_heads,
        head_dim=args.head_dim,
        q_seq=args.q_seq,
        past_seq=args.past_seq,
        do_rotary=args.do_rotary,
        rotary_interleaved=args.rotary_interleaved,
        local_window_size=args.local_window_size,
        softcap=args.softcap,
        cache_capacity=args.cache_capacity,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
