#!/usr/bin/env python3
"""Generate the multi-node SDPA region this repo's fusion pass collapses.

The point of this generator is a question the single-node grids cannot answer.
`Softmax` measures far slower than ORT as an isolated node, but
`assignment_policy` deliberately keeps its claim anyway, on the argument that it
anchors the attention fusion:

    MatMul(Q, Kt) -> (Mul|Div by scalar) -> [Add(mask)] -> Softmax(axis=-1)
                  -> MatMul(probs, V)

collapses into one `com.microsoft::FusedAttention` node. Deferring the
standalone `Softmax` would remove the anchor and split the core across the EP
boundary. That argument is plausible, but the policy comment says outright that
"a claim to defer it needs a fused-graph measurement this module does not have".

This is that measurement. Each model is the whole region, so a run exercises the
fused kernel end to end against ORT executing the same five nodes, including
every layout transform and allocation between them.

SYNTHETIC DATA NOTICE: no trained weights are downloaded or used. Only the
batch/head/head_dim/sequence dimensions come from public model configs; tensor
contents are the benchmark harness's deterministic synthetic pattern, fed
identically to both runtimes.

Shapes are fully static, and K arrives pre-transposed as `[B, H, D, S_kv]` so
the score product is a plain `MatMul` — which is what the matcher requires, and
what a graph that has already had its `Transpose` folded into the weight layout
looks like.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper

OPSET = 17


def build_region(
    path: Path,
    *,
    batch: int,
    num_heads: int,
    head_dim: int,
    q_seq: int,
    kv_seq: int,
    masked: bool,
) -> None:
    """Emit one `[B,H,Sq,D] x [B,H,D,Skv] -> softmax -> x [B,H,Skv,D]` region."""
    f = TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info("query", f, [batch, num_heads, q_seq, head_dim]),
        # Pre-transposed: the matcher wants a plain MatMul for the score product.
        helper.make_tensor_value_info("key_t", f, [batch, num_heads, head_dim, kv_seq]),
        helper.make_tensor_value_info("value", f, [batch, num_heads, kv_seq, head_dim]),
    ]
    if masked:
        # Broadcast over heads, as an additive attention mask does.
        inputs.append(
            helper.make_tensor_value_info("mask", f, [batch, 1, q_seq, kv_seq])
        )

    # The scale must be a concrete scalar f32 constant for the matcher to accept
    # it; a rank-0 initializer is the least ambiguous way to say that.
    scale = numpy_helper.from_array(
        np.array(math.sqrt(head_dim), dtype=np.float32), "scale"
    )

    nodes = [
        helper.make_node("MatMul", ["query", "key_t"], ["scores"], name="qk"),
        helper.make_node("Div", ["scores", "scale"], ["scaled"], name="scale_scores"),
    ]
    softmax_in = "scaled"
    if masked:
        nodes.append(helper.make_node("Add", ["scaled", "mask"], ["masked"], name="mask_add"))
        softmax_in = "masked"
    nodes.append(
        helper.make_node("Softmax", [softmax_in], ["probs"], name="softmax", axis=-1)
    )
    nodes.append(helper.make_node("MatMul", ["probs", "value"], ["output"], name="pv"))

    graph = helper.make_graph(
        nodes,
        "sdpa_region",
        inputs,
        [
            helper.make_tensor_value_info(
                "output", f, [batch, num_heads, q_seq, head_dim]
            )
        ],
        initializer=[scale],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="ort_ab"
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


# (label, heads, head_dim) from public configs. Head counts are the *query* head
# counts, since the region as written is already per-head.
GEOMETRIES = [
    ("qwen2.5-0.5b", 14, 64),
    ("qwen3-0.6b", 16, 128),
    ("llama-3.1-8b", 32, 128),
]

# Decode reads one query against a long history; prefill reads a whole prompt.
PHASES = [("decode", 1, 1024), ("decode", 1, 4096), ("prefill", 512, 512), ("prefill", 1024, 1024)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    for label, heads, head_dim in GEOMETRIES:
        for phase, q_seq, kv_seq in PHASES:
            for masked in (True, False):
                tag = "mask" if masked else "nomask"
                name = (
                    f"sdpa_region_{label}_{phase}_h{heads}_d{head_dim}"
                    f"_q{q_seq}_kv{kv_seq}_{tag}.onnx"
                )
                build_region(
                    args.out / name,
                    batch=args.batch,
                    num_heads=heads,
                    head_dim=head_dim,
                    q_seq=q_seq,
                    kv_seq=kv_seq,
                    masked=masked,
                )
                print(name)


if __name__ == "__main__":
    main()
