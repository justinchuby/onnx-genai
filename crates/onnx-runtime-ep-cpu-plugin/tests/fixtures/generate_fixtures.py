#!/usr/bin/env python3
"""Generate ONNX test fixtures for the EP conformance suite.

Run from the workspace root or the fixtures directory:
    python3 tests/fixtures/generate_fixtures.py

Each fixture is a tiny ONNX model whose expected outputs are documented
in the comment below each function — compute them independently to verify.
"""
import os
import struct

import onnx
from onnx import TensorProto, helper

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def save(model, name):
    path = os.path.join(FIXTURES_DIR, name, "model.onnx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.checker.check_model(model)
    with open(path, "wb") as f:
        f.write(model.SerializeToString())
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")


# ── add_broadcast ────────────────────────────────────────────────────────────
# Add([2,3], [3]) broadcasting.
# X = [[1,2,3],[4,5,6]]  Y = [10,20,30]
# Z = X + Y = [[11,22,33],[14,25,36]]
def gen_add_broadcast():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
    graph = helper.make_graph([node], "add_broadcast", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, "add_broadcast")


# ── chain_add_mul ────────────────────────────────────────────────────────────
# Multi-node fused subgraph: T = (A + B) * C + D
# All inputs shape [1, 4].
# A=[1,2,3,4]  B=[1,1,1,1]  C=[2,2,2,2]  D=[0,0,0,0]
# T = (A+B)*C + D = [4,6,8,10]
def gen_chain_add_mul():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 4])
    T = helper.make_tensor_value_info("T", TensorProto.FLOAT, [1, 4])
    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["AB"])
    mul_node = helper.make_node("Mul", inputs=["AB", "C"], outputs=["ABC"])
    add2_node = helper.make_node("Add", inputs=["ABC", "D"], outputs=["T"])
    graph = helper.make_graph(
        [add_node, mul_node, add2_node],
        "chain_add_mul",
        [A, B, C, D],
        [T],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, "chain_add_mul")


# ── matmul_2d ────────────────────────────────────────────────────────────────
# MatMul [2,3] x [3,2] -> [2,2]
# A = [[1,2,3],[4,5,6]]   B = [[1,0],[0,1],[1,0]]
# C = [[1*1+2*0+3*1, 1*0+2*1+3*0],
#      [4*1+5*0+6*1, 4*0+5*1+6*0]]
#   = [[4,2],[10,5]]
def gen_matmul_2d():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])
    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "matmul_2d", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, "matmul_2d")


# ── matmul_batched_nd ─────────────────────────────────────────────────────────
# Batched 3-D MatMul: A [2,3,4] × B [2,4,2] → C [2,3,2]
# Tests batched-ND broadcast inference in our MatMul kernel.
#
# batch 0: A0=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]  B0=[[1,0],[0,1],[1,0],[0,1]]
#   C0 = [[4,6],[12,14],[20,22]]
# batch 1: A1=[[0,1,0,1],[2,0,2,0],[1,1,1,1]]  B1=[[2,0],[0,2],[2,0],[0,2]]
#   C1 = [[0,4],[8,0],[4,4]]
def gen_matmul_batched_nd():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 4, 2])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 3, 2])
    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "matmul_batched_nd", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, "matmul_batched_nd")


# ── mixed_partition ──────────────────────────────────────────────────────────
# Graph with Add (claimed by our EP) and NonZero (not claimed by our EP).
# ORT must partition: our EP handles Add, ORT's default handles NonZero.
# X = [1, 2, 3, 4]  Y = [0, 0, 0, 0]  → SUM = [1,2,3,4]
# NonZero(SUM) → indices of nonzero elements = [[0,1,2,3]]
#
# Shape [1,4] + [1,4] → [1,4] → NonZero → [1,4]
def gen_mixed_partition():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    # NonZero output shape is [rank, nnz], both dynamic.
    Z = helper.make_tensor_value_info("Z", TensorProto.INT64, [2, None])
    add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["SUM"])
    nz_node = helper.make_node("NonZero", inputs=["SUM"], outputs=["Z"])
    graph = helper.make_graph([add_node, nz_node], "mixed_partition", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, "mixed_partition")


# ── add_int32 ────────────────────────────────────────────────────────────────
# Add with INT32 tensors, shape [1,4].
# X = [10, 20, 30, 40]  Y = [1, 2, 3, 4]  Z = [11, 22, 33, 44]
def gen_add_int32():
    X = helper.make_tensor_value_info("X", TensorProto.INT32, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4])
    Z = helper.make_tensor_value_info("Z", TensorProto.INT32, [1, 4])
    node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
    graph = helper.make_graph([node], "add_int32", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, "add_int32")


# ── add_dynamic_dim ──────────────────────────────────────────────────────────
# Add with a dynamic (symbolic) first dimension.  Shape is [-1, 4] for both
# inputs, meaning the batch size is unknown at graph-load time.
# At runtime we supply batch=1: X=[1,2,3,4]  Y=[5,6,7,8]  Z=[6,8,10,12]
def gen_add_dynamic_dim():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 4])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["batch", 4])
    node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
    graph = helper.make_graph([node], "add_dynamic_dim", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, "add_dynamic_dim")


if __name__ == "__main__":
    print("Generating ONNX conformance fixtures …")
    gen_add_broadcast()
    gen_chain_add_mul()
    gen_matmul_2d()
    gen_matmul_batched_nd()
    gen_mixed_partition()
    gen_add_int32()
    gen_add_dynamic_dim()
    print("Done.")
