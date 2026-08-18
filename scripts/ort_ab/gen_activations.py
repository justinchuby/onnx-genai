#!/usr/bin/env python3
"""Isolated single-node graphs for the elementwise activations.

These are the cheapest ops in a transformer and the ones most exposed to
scheduling overhead: a Gelu over an MLP intermediate tensor is a few hundred
microseconds of arithmetic wrapped around a fan-out, so the fan-out is a
first-order cost rather than a rounding error. That makes them the sharpest
test of a task runtime.

Shapes are MLP-intermediate and hidden-state shaped, taken from public model
configs, and swept over a size range that brackets the point where parallelism
starts to pay: L2-resident, L3-resident, and memory-resident.

SYNTHETIC DATA NOTICE: no trained weights. Only the hidden / intermediate /
sequence dimensions come from public model configs; tensor contents are the
benchmark harness's deterministic synthetic pattern, fed identically to both
runtimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

OPSET = 17


def _save(path: Path, graph: onnx.GraphProto, *, ms_domain: bool = False) -> None:
    opsets = [helper.make_opsetid("", OPSET)]
    if ms_domain:
        opsets.append(helper.make_opsetid("com.microsoft", 1))
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = 10
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    print(f"wrote {path}")


def build_unary(path: Path, op: str, *, rows: int, cols: int, ms_domain: bool = False) -> None:
    """A single unary elementwise node over a [rows, cols] float tensor."""
    domain = "com.microsoft" if ms_domain else ""
    node = helper.make_node(op, ["x"], ["y"], domain=domain)
    graph = helper.make_graph(
        [node],
        op.lower(),
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, cols])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, cols])],
    )
    _save(path, graph, ms_domain=ms_domain)


def build_binary(path: Path, op: str, *, rows: int, cols: int) -> None:
    """A single binary elementwise node over two [rows, cols] float tensors."""
    node = helper.make_node(op, ["x", "w"], ["y"])
    graph = helper.make_graph(
        [node],
        op.lower(),
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, cols]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [rows, cols]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, cols])],
    )
    _save(path, graph)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out

    # --- The size sweep -------------------------------------------------
    # llama3-8B hidden is 4096, intermediate is 14336. One row per token, so
    # the token count is what moves the tensor through the cache hierarchy:
    # 32 tokens is 512 KiB (L2), 512 tokens is 8 MiB (L3), 4096 tokens is
    # 64 MiB (memory). The fan-out cost is fixed, so the ratio between it and
    # the body is what changes across this sweep.
    # Gelu is only in the default domain from opset 20; the contrib FastGelu
    # is what a real transformer graph carries at this opset, and it is the
    # one the CPU EP fuses.
    for tokens in (1, 32, 512, 4096):
        build_unary(
            out / f"act_fastgelu_mlp_t{tokens}.onnx",
            "FastGelu",
            rows=tokens,
            cols=14336,
            ms_domain=True,
        )
    for tokens in (32, 512, 4096):
        build_unary(out / f"act_relu_hidden_t{tokens}.onnx", "Relu", rows=tokens, cols=4096)
        build_unary(out / f"act_sigmoid_hidden_t{tokens}.onnx", "Sigmoid", rows=tokens, cols=4096)
        build_unary(out / f"act_tanh_hidden_t{tokens}.onnx", "Tanh", rows=tokens, cols=4096)

    # --- The binary elementwise ops that surround them ------------------
    # SwiGLU's gate multiply and the residual add are the same fan-out with a
    # second input stream, which doubles the bandwidth per unit of arithmetic.
    for tokens in (32, 512, 4096):
        build_binary(out / f"act_mul_gate_t{tokens}.onnx", "Mul", rows=tokens, cols=14336)
    for tokens in (512, 4096):
        build_binary(out / f"act_add_residual_t{tokens}.onnx", "Add", rows=tokens, cols=4096)


if __name__ == "__main__":
    main()
