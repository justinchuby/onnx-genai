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
from google.protobuf import text_format
import numpy as np

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))

# Maintained fixtures target ONNX IR version 11 and default ONNX opset 24.
# Custom-domain imports (com.microsoft) keep their own version. Enforced by the
# fixture IR/opset guard test in onnx-std (crates/onnx-std/tests/fixture_ir_opset_guard.rs).
IR_VERSION = 11
DEFAULT_OPSET = 24


def save(model, name, check=True):
    # Fixtures are committed as git-friendly ONNX protobuf TextFormat
    # (`model.onnx.textproto`), which our runtime loader parses transparently
    # (see `onnx_runtime_loader::is_textproto_path`). No binary `model.onnx` is
    # committed. NOTE: this comment used to name a Rust helper example as the
    # canonical producer, but no such example exists and git history shows none
    # ever did. `onnx_std::textproto::to_textproto` is the canonical formatter;
    # this Python path emits an equivalent, loader-compatible textproto whose
    # exact formatting may differ from that Rust output.
    path = os.path.join(FIXTURES_DIR, name, "model.onnx.textproto")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if check:
        onnx.checker.check_model(model)
    with open(path, "w") as f:
        f.write(text_format.MessageToString(model))
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "add_dynamic_dim")


# ── add_1x4 ─────────────────────────────────────────────────────────────────
# Simple Add with FLOAT tensors, shape [1,4].
# X = [1.0, 2.0, 3.0, 4.0]  Y = [5.0, 6.0, 7.0, 8.0]  Z = [6.0, 8.0, 10.0, 12.0]
def gen_add_1x4():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
    graph = helper.make_graph([node], "add_1x4", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "add_1x4")


# ── add_float16 ──────────────────────────────────────────────────────────────
# Add with FLOAT16 tensors, shape [1,4].  Tests half-precision EP routing.
# X = [1,2,3,4]  Y = [5,6,7,8]  Z = [6,8,10,12]  (all in fp16)
def gen_add_float16():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, 4])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT16, [1, 4])
    node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
    graph = helper.make_graph([node], "add_float16", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "add_float16")


# ── add_bfloat16 ─────────────────────────────────────────────────────────────
# Add with BFLOAT16 tensors, shape [1,4].  Tests bf16 EP routing.
# X = [1,2,3,4]  Y = [5,6,7,8]  Z = [6,8,10,12]  (all in bf16)
def gen_add_bfloat16():
    X = helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [1, 4])
    Z = helper.make_tensor_value_info("Z", TensorProto.BFLOAT16, [1, 4])
    node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
    graph = helper.make_graph([node], "add_bfloat16", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "add_bfloat16")


# ── nonzero_1x4 ──────────────────────────────────────────────────────────────
# NonZero on FLOAT tensor, shape [1,4].
# X = [0, 3, 0, 5]  → Y = [[0,0],[1,3]]  (indices of nonzero elements)
def gen_nonzero_1x4():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [2, None])
    node = helper.make_node("NonZero", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "nonzero_1x4", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "nonzero_1x4")


# ── cast_f32_to_i64 ──────────────────────────────────────────────────────────
# Cast f32 [2,3] → i64.  Output dtype differs from input dtype.
# This is the B1 regression test: output_dtypes must read from the ORT graph's
# value info, not from the first input's dtype.
# X = [[1.5, 2.7, 3.0], [4.9, 5.1, 6.0]]
# Expected Y = [[1, 2, 3], [4, 5, 6]] (truncated toward zero)
def gen_cast_f32_to_i64():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [2, 3])
    node = helper.make_node("Cast", inputs=["X"], outputs=["Y"], to=TensorProto.INT64)
    graph = helper.make_graph([node], "cast_f32_to_i64", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "cast_f32_to_i64")


# ── where_bool_f32 ───────────────────────────────────────────────────────────
# Where(condition, X, Y) → f32.  First input is bool, output is f32.
# This is the B1 regression test: output_dtypes must NOT guess from the first
# input (bool) — the output dtype is f32.
# condition = [[true, false], [false, true]]
# X = [[1.0, 2.0], [3.0, 4.0]]
# Y = [[10.0, 20.0], [30.0, 40.0]]
# Expected Z = [[1.0, 20.0], [30.0, 4.0]]
def gen_where_bool_f32():
    C = helper.make_tensor_value_info("C", TensorProto.BOOL, [2, 2])
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 2])
    node = helper.make_node("Where", inputs=["C", "X", "Y"], outputs=["Z"])
    graph = helper.make_graph([node], "where_bool_f32", [C, X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "where_bool_f32")


# ── shape_f32 ────────────────────────────────────────────────────────────────
# Shape(f32 [3,4,5]) → i64 [3] with value [3,4,5].
# Output dtype is i64 regardless of input dtype.
def gen_shape_f32():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [3])
    node = helper.make_node("Shape", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "shape_f32", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "shape_f32")


# ── layer_norm_f32 ───────────────────────────────────────────────────────────
# LayerNormalization(X [2,4], scale [4]) → 3 outputs (Y, Mean, InvStdDev).
# All outputs are f32. axis=-1 (last dim), Mean/InvStdDev shape = [2,1].
def gen_layer_norm_f32():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    scale = helper.make_tensor_value_info("Scale", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    Mean = helper.make_tensor_value_info("Mean", TensorProto.FLOAT, [2, 1])
    InvStdDev = helper.make_tensor_value_info("InvStdDev", TensorProto.FLOAT, [2, 1])
    node = helper.make_node(
        "LayerNormalization",
        inputs=["X", "Scale"],
        outputs=["Y", "Mean", "InvStdDev"],
        axis=-1,
    )
    graph = helper.make_graph(
        [node], "layer_norm_f32", [X, scale], [Y, Mean, InvStdDev]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "layer_norm_f32")


# ── layer_norm_neg_axis_f32 ──────────────────────────────────────────────────
# LayerNormalization(X [2,3,4], scale [4]) with axis=-1 (normalized axis=2).
# → 3 outputs: Y [2,3,4], Mean [2,3,1], InvStdDev [2,3,1].
# Key regression: old ShapePreservingNorm gave Mean shape [2,3,4]; this
# fixture detects that because [2,3,1] != [2,3,4].
# X[0] = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# X[1] = [[-4,-3,-2,-1],[1,1,1,1],[0,1,2,3]]
# Mean (row means) = [[2.5, 6.5, 10.5], [-2.5, 1.0, 1.5]] → shape [2,3,1].
def gen_layer_norm_neg_axis_f32():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    scale = helper.make_tensor_value_info("Scale", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4])
    Mean = helper.make_tensor_value_info("Mean", TensorProto.FLOAT, [2, 3, 1])
    InvStdDev = helper.make_tensor_value_info("InvStdDev", TensorProto.FLOAT, [2, 3, 1])
    node = helper.make_node(
        "LayerNormalization",
        inputs=["X", "Scale"],
        outputs=["Y", "Mean", "InvStdDev"],
        axis=-1,
    )
    graph = helper.make_graph(
        [node], "layer_norm_neg_axis_f32", [X, scale], [Y, Mean, InvStdDev]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "layer_norm_neg_axis_f32")


# ── simplified_layer_norm_f32 ────────────────────────────────────────────────
# RMSNormalization(X [2,4], scale [4]) → 1 output: Y [2,4].
# axis=-1.  No Mean or InvStdDev outputs.
# Y = X / sqrt(mean(X^2) + eps) * scale.
# X = [[1,2,3,4],[5,6,7,8]], Scale = [1,1,1,1]
# Row 0: rms = sqrt(7.5 + eps) ≈ 2.7386, Y[0] ≈ [0.3651, 0.7303, 1.0954, 1.4606]
# Row 1: rms = sqrt(43.5 + eps) ≈ 6.5952, Y[1] ≈ [0.7583, 0.9099, 1.0616, 1.2132]
def gen_simplified_layer_norm_f32():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(
        "RMSNormalization",
        inputs=["X", "scale"],
        outputs=["Y"],
        axis=-1,
    )
    graph = helper.make_graph(
        [node], "simplified_layer_norm_f32", [X, scale], [Y]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "simplified_layer_norm_f32")


# ── skip_layer_norm_f16_absent_output ────────────────────────────────────────
# SkipLayerNormalization(f16) with outputs (output, '', '', sum).
# Absent optional outputs at slots 1,2. Tests B1 scratch buffer overflow.
def gen_skip_layer_norm_f16_absent_output():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [2, 4])
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT16, [2, 4])
    gamma = helper.make_tensor_value_info("gamma", TensorProto.FLOAT16, [4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2, 4])
    sum_out = helper.make_tensor_value_info("sum", TensorProto.FLOAT16, [2, 4])
    node = helper.make_node(
        "SkipLayerNormalization",
        inputs=["X", "skip", "gamma"],
        outputs=["output", "", "", "sum"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    graph = helper.make_graph([node], "skip_ln_f16_absent", [X, skip, gamma], [output, sum_out])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", DEFAULT_OPSET),
        helper.make_opsetid("com.microsoft", 1),
    ])
    model.ir_version = IR_VERSION
    save(model, "skip_layer_norm_f16_absent_output")


# ── skip_layer_norm_bf16_absent_output ───────────────────────────────────────
# Same as f16 but bfloat16.
def gen_skip_layer_norm_bf16_absent_output():
    X = helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [2, 4])
    skip = helper.make_tensor_value_info("skip", TensorProto.BFLOAT16, [2, 4])
    gamma = helper.make_tensor_value_info("gamma", TensorProto.BFLOAT16, [4])
    output = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [2, 4])
    sum_out = helper.make_tensor_value_info("sum", TensorProto.BFLOAT16, [2, 4])
    node = helper.make_node(
        "SkipLayerNormalization",
        inputs=["X", "skip", "gamma"],
        outputs=["output", "", "", "sum"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    graph = helper.make_graph([node], "skip_ln_bf16_absent", [X, skip, gamma], [output, sum_out])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", DEFAULT_OPSET),
        helper.make_opsetid("com.microsoft", 1),
    ])
    model.ir_version = IR_VERSION
    save(model, "skip_layer_norm_bf16_absent_output")


# ── layer_norm_f16_absent_output ─────────────────────────────────────────────
# LayerNormalization(f16) with outputs (Y, '', '').
# Mean and InvStdDev are absent.
def gen_layer_norm_f16_absent_output():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [2, 4])
    scale = helper.make_tensor_value_info("Scale", TensorProto.FLOAT16, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 4])
    node = helper.make_node(
        "LayerNormalization",
        inputs=["X", "Scale"],
        outputs=["Y", "", ""],
        axis=-1,
    )
    graph = helper.make_graph([node], "layer_norm_f16_absent", [X, scale], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "layer_norm_f16_absent_output")


# ── layer_norm_bf16_absent_output ────────────────────────────────────────────
# Same as f16 but bfloat16.
def gen_layer_norm_bf16_absent_output():
    X = helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [2, 4])
    scale = helper.make_tensor_value_info("Scale", TensorProto.BFLOAT16, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [2, 4])
    node = helper.make_node(
        "LayerNormalization",
        inputs=["X", "Scale"],
        outputs=["Y", "", ""],
        axis=-1,
    )
    graph = helper.make_graph([node], "layer_norm_bf16_absent", [X, scale], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "layer_norm_bf16_absent_output")


# ── add_skip_layer_norm_mul ──────────────────────────────────────────────────
# Add -> SkipLayerNormalization(out,'','',sum) -> Mul (f32).
# Multi-node fused subgraph for B2 routed-path test.
def gen_add_skip_layer_norm_mul():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 4])
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT, [2, 4])
    gamma = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [4])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 4])
    result = helper.make_tensor_value_info("result", TensorProto.FLOAT, [2, 4])
    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["T"])
    sln_node = helper.make_node(
        "SkipLayerNormalization",
        inputs=["T", "skip", "gamma"],
        outputs=["sln_output", "", "", "sum"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    mul_node = helper.make_node("Mul", inputs=["sum", "C"], outputs=["result"])
    graph = helper.make_graph(
        [add_node, sln_node, mul_node],
        "add_skip_ln_mul",
        [A, B, skip, gamma, C],
        [result],
    )
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", DEFAULT_OPSET),
        helper.make_opsetid("com.microsoft", 1),
    ])
    model.ir_version = IR_VERSION
    save(model, "add_skip_layer_norm_mul")


# ── layer_norm_dynamic_axis ──────────────────────────────────────────────────
# LayerNormalization(X [B,S,4], scale [4]) with axis=-1 and *dynamic* leading
# dims (B, S are symbolic). → 3 outputs: Y [B,S,4], Mean [B,S,1],
# InvStdDev [B,S,1]. Regression: the normalized (last) axis must be resolved
# from the static rank even when the leading dims are dynamic.
def gen_layer_norm_dynamic_axis():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["B", "S", 4])
    scale = helper.make_tensor_value_info("Scale", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["B", "S", 4])
    Mean = helper.make_tensor_value_info("Mean", TensorProto.FLOAT, ["B", "S", 1])
    InvStdDev = helper.make_tensor_value_info(
        "InvStdDev", TensorProto.FLOAT, ["B", "S", 1]
    )
    node = helper.make_node(
        "LayerNormalization",
        inputs=["X", "Scale"],
        outputs=["Y", "Mean", "InvStdDev"],
        axis=-1,
    )
    graph = helper.make_graph(
        [node], "layer_norm_dynamic_axis", [X, scale], [Y, Mean, InvStdDev]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "layer_norm_dynamic_axis")


# ── skip_layer_norm_output_sum ───────────────────────────────────────────────
# SkipLayerNormalization(f32) with outputs (output, '', '', sum).
# Absent optional outputs at slots 1,2; the `sum` output (slot 3) is present.
# sum = X + skip (input skip-connection sum), output is the normalized result.
def gen_skip_layer_norm_output_sum():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT, [2, 4])
    gamma = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    sum_out = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(
        "SkipLayerNormalization",
        inputs=["X", "skip", "gamma"],
        outputs=["output", "", "", "sum"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node], "skip_ln_sum", [X, skip, gamma], [output, sum_out]
    )
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", DEFAULT_OPSET),
        helper.make_opsetid("com.microsoft", 1),
    ])
    model.ir_version = IR_VERSION
    save(model, "skip_layer_norm_output_sum")


# ── simplified_layer_norm_two_outputs ────────────────────────────────────────
# SimplifiedLayerNormalization(X [2,4], scale [4]) with axis=-1 emitting the
# optional second output (inv_std [2,1]) alongside the normalized output [2,4].
# Standard-domain op at opset 21.
def gen_simplified_layer_norm_two_outputs():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    inv_std = helper.make_tensor_value_info("inv_std", TensorProto.FLOAT, [2, 1])
    node = helper.make_node(
        "SimplifiedLayerNormalization",
        inputs=["X", "scale"],
        outputs=["output", "inv_std"],
    )
    # Append attributes in (epsilon, axis) order explicitly: helper.make_node
    # sorts kwargs alphabetically ("axis" before "epsilon"), but the committed
    # fixture stores epsilon first, so build the attribute list by hand to match.
    node.attribute.extend(
        [
            helper.make_attribute("epsilon", 1e-5),
            helper.make_attribute("axis", -1),
        ]
    )
    # The committed fixture serializes an explicit empty `domain` field on the
    # node; force it present so the regenerated bytes match exactly.
    node.domain = ""
    graph = helper.make_graph(
        [node], "sln_two_out", [X, scale], [output, inv_std]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    # SimplifiedLayerNormalization at domain "" is a non-standard op that the
    # onnx checker's schema registry does not know about; skip the check so the
    # fixture (which mirrors a real model the CPU EP must handle) can be written.
    save(model, "simplified_layer_norm_two_outputs", check=False)


# ── skip_layer_norm_no_beta_bias ─────────────────────────────────────────────
# SkipLayerNormalization(f32) with beta (slot 3) and bias (slot 4) *absent*.
# inputs = (X, skip, gamma, '', ''); single output. Exercises the no-beta,
# no-bias path where only gamma scales the normalized result.
def gen_skip_layer_norm_no_beta_bias():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT, [2, 4])
    gamma = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(
        "SkipLayerNormalization",
        inputs=["X", "skip", "gamma", "", ""],
        outputs=["output"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node], "skip_ln_no_beta", [X, skip, gamma], [output]
    )
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", DEFAULT_OPSET),
        helper.make_opsetid("com.microsoft", 1),
    ])
    model.ir_version = IR_VERSION
    save(model, "skip_layer_norm_no_beta_bias")


# ── clip_no_min ──────────────────────────────────────────────────────────────
# Clip(X [1,4], '', max_val) — the optional `min` input (slot 1) is absent while
# `max` (slot 2) is present as a scalar. Exercises the absent-min Clip path.
def gen_clip_no_min():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    max_val = helper.make_tensor_value_info("max_val", TensorProto.FLOAT, [])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Clip", inputs=["X", "", "max_val"], outputs=["Y"])
    graph = helper.make_graph([node], "clip_no_min", [X, max_val], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "clip_no_min")


# ── matmul_initializer_weights ───────────────────────────────────────────────
# MatMul(X [2,4], W [4,3]) where W is a constant initializer (not a graph
# input). W = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]. Verifies MatMul consumes
# initializer-backed weights. Y = X @ W, shape [2,3].
def gen_matmul_initializer_weights():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    # W is stored via numpy_helper (raw_data little-endian bytes) to match the
    # committed fixture's on-disk encoding exactly.
    W = onnx.numpy_helper.from_array(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
        name="W",
    )
    node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])
    graph = helper.make_graph(
        [node], "matmul_initializer", [X], [Y], initializer=[W]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = IR_VERSION
    save(model, "matmul_initializer_weights")


if __name__ == "__main__":
    print("Generating ONNX conformance fixtures …")
    gen_add_broadcast()
    gen_chain_add_mul()
    gen_matmul_2d()
    gen_matmul_batched_nd()
    gen_mixed_partition()
    gen_add_int32()
    gen_add_dynamic_dim()
    gen_add_1x4()
    gen_add_float16()
    gen_add_bfloat16()
    gen_nonzero_1x4()
    gen_cast_f32_to_i64()
    gen_where_bool_f32()
    gen_shape_f32()
    gen_layer_norm_f32()
    gen_layer_norm_neg_axis_f32()
    gen_simplified_layer_norm_f32()
    gen_skip_layer_norm_f16_absent_output()
    gen_skip_layer_norm_bf16_absent_output()
    gen_layer_norm_f16_absent_output()
    gen_layer_norm_bf16_absent_output()
    gen_add_skip_layer_norm_mul()
    gen_layer_norm_dynamic_axis()
    gen_skip_layer_norm_output_sum()
    gen_simplified_layer_norm_two_outputs()
    gen_skip_layer_norm_no_beta_bias()
    gen_clip_no_min()
    gen_matmul_initializer_weights()
    print("Done.")
