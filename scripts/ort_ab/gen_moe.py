#!/usr/bin/env python3
"""Generate single-node com.microsoft::MoE / QMoE models with fully static shapes.

SYNTHETIC DATA NOTICE: every expert weight, bias and router logit here is
generated from a fixed PRNG seed. Nothing is downloaded and no trained weights
are used. Only the *dimensions* (hidden size, expert intermediate size, top-k,
activation) come from public architecture configs; the expert **count** is
reduced where a full expert bank would not fit in host memory as f32, and that
reduction is recorded in the file name (`e{N}`) and in the results table.

Slot order (com.microsoft::MoE):
  0 input 1 router_probs 2 fc1_experts_weights 3 fc1_experts_bias?
  4 fc2_experts_weights 5 fc2_experts_bias? 6 fc3_experts_weights?
  7 fc3_experts_bias?
Canonical layouts: fc1 [experts, fc1_size, hidden], fc2 [experts, hidden, inter],
fc3 [experts, inter, hidden].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def rand(rng: np.random.Generator, *shape: int) -> np.ndarray:
    # Small symmetric range keeps the f32 accumulation well conditioned so a
    # parity failure means a real disagreement, not catastrophic cancellation.
    return (rng.standard_normal(shape).astype(np.float32) * 0.02).astype(np.float32)


def build_moe(
    path: Path,
    *,
    tokens: int,
    hidden: int,
    inter: int,
    experts: int,
    top_k: int,
    activation: str = "swiglu",
    swiglu_fusion: int = 1,
    normalize_routing_weights: bool = True,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    fc1_size = 2 * inter if activation == "swiglu" and swiglu_fusion != 0 else inter

    initializers = [
        numpy_helper.from_array(rand(rng, experts, fc1_size, hidden), "fc1_experts_weights"),
        numpy_helper.from_array(rand(rng, experts, hidden, inter), "fc2_experts_weights"),
    ]
    node_inputs = ["input", "router_probs", "fc1_experts_weights", "", "fc2_experts_weights", ""]

    inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [tokens, hidden]),
        helper.make_tensor_value_info("router_probs", TensorProto.FLOAT, [tokens, experts]),
    ]

    attrs = {
        "k": top_k,
        "activation_type": activation,
        "normalize_routing_weights": int(normalize_routing_weights),
    }
    if activation == "swiglu":
        attrs["swiglu_fusion"] = swiglu_fusion

    node = helper.make_node(
        "MoE",
        node_inputs,
        ["output"],
        name="moe",
        domain="com.microsoft",
        **attrs,
    )
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [tokens, hidden])]
    graph = helper.make_graph([node], "moe_bench", inputs, outputs, initializer=initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path), save_as_external_data=False)


# (name, hidden, inter, experts, experts_in_real_config, top_k, activation)
CASES = [
    ("qwen3moe_h2048_i768_e16", 2048, 768, 16, 128, 8, "swiglu"),
    ("mixtral_h1024_i3584_e8", 1024, 3584, 8, 8, 2, "swiglu"),
    ("phi35moe_h2048_i6400_e4", 2048, 6400, 4, 16, 2, "swiglu"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokens", nargs="+", type=int, default=[1, 32, 512])
    args = ap.parse_args()
    for name, hidden, inter, experts, _real, top_k, act in CASES:
        for tokens in args.tokens:
            path = args.out_dir / f"moe_{name}_t{tokens}.onnx"
            build_moe(
                path,
                tokens=tokens,
                hidden=hidden,
                inter=inter,
                experts=experts,
                top_k=top_k,
                activation=act,
            )
            print(path.name, f"{path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
