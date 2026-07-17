"""Projection-fusion fixtures built only with the ONNX IR API."""

from __future__ import annotations

import sys

import numpy as np
from onnxscript import ir


K = 32
N = 2
BLOCK_SIZE = 32


def value(name: str, dtype: ir.DataType, shape: list[int]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def initializer(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.tensor(array, name=name))


def packed_rows(offset: int) -> np.ndarray:
    codes = (
        np.arange(N * K, dtype=np.uint8).reshape(N, K) * np.uint8(3)
        + np.uint8(offset)
    ) & np.uint8(0x0F)
    return codes[:, 0::2] | (codes[:, 1::2] << np.uint8(4))


def matmul_nbits(
    name: str,
    activation: ir.Value,
    weight: ir.Value,
    scales: ir.Value,
    output: ir.Value,
    zero_points: ir.Value | None,
) -> ir.Node:
    inputs = [activation, weight, scales]
    if zero_points is not None:
        inputs.append(zero_points)
    return ir.Node(
        "com.microsoft",
        "MatMulNBits",
        inputs,
        [
            ir.AttrInt64("K", K),
            ir.AttrInt64("N", N),
            ir.AttrInt64("bits", 4),
            ir.AttrInt64("block_size", BLOCK_SIZE),
            ir.AttrInt64("accuracy_level", 4),
        ],
        outputs=[output],
        name=name,
        version=1,
    )


def build_model(scenario: str) -> ir.Model:
    x = value("X", ir.DataType.FLOAT, [1, K])
    gate_weight = initializer(
        "gate_B", packed_rows(1).reshape(N, 1, BLOCK_SIZE // 2)
    )
    up_weight = initializer("up_B", packed_rows(5).reshape(N, 1, BLOCK_SIZE // 2))
    gate_scales = initializer(
        "gate_scales", np.array([[0.25], [0.125]], dtype=np.float32)
    )
    up_scales = initializer(
        "up_scales", np.array([[0.5], [0.375]], dtype=np.float32)
    )
    initializers = [gate_weight, up_weight, gate_scales, up_scales]

    gate_zp = None
    up_zp = None
    if scenario == "zero_point":
        gate_zp = initializer("gate_zp", np.full((N, 1), 0x88, dtype=np.uint8))
        up_zp = initializer("up_zp", np.full((N, 1), 0x88, dtype=np.uint8))
        initializers.extend([gate_zp, up_zp])

    gate_raw = value("gate_raw", ir.DataType.FLOAT, [1, N])
    up_raw = value("up_raw", ir.DataType.FLOAT, [1, N])
    nodes = [
        matmul_nbits(
            "gate_projection", x, gate_weight, gate_scales, gate_raw, gate_zp
        ),
        matmul_nbits("up_projection", x, up_weight, up_scales, up_raw, up_zp),
    ]

    gate_for_silu = gate_raw
    if scenario == "bias":
        bias = initializer("gate_bias", np.array([0.1, -0.2], dtype=np.float32))
        initializers.append(bias)
        gate_for_silu = value("gate_biased", ir.DataType.FLOAT, [1, N])
        nodes.append(
            ir.Node(
                "",
                "Add",
                [gate_raw, bias],
                outputs=[gate_for_silu],
                name="gate_bias_add",
                version=21,
            )
        )

    activated = value("gate_activated", ir.DataType.FLOAT, [1, N])
    if scenario == "decomposed":
        sigmoid = value("gate_sigmoid", ir.DataType.FLOAT, [1, N])
        nodes.append(
            ir.Node(
                "",
                "Sigmoid",
                [gate_for_silu],
                outputs=[sigmoid],
                name="gate_sigmoid",
                version=21,
            )
        )
        nodes.append(
            ir.Node(
                "",
                "Mul",
                [gate_for_silu, sigmoid],
                outputs=[activated],
                name="gate_silu_decomposition",
                version=21,
            )
        )
    else:
        nodes.append(
            ir.Node(
                "com.microsoft",
                "Silu",
                [gate_for_silu],
                outputs=[activated],
                name="gate_silu",
                version=1,
            )
        )
    output = value("Y", ir.DataType.FLOAT, [1, N])
    nodes.append(
        ir.Node(
            "",
            "Mul",
            [activated, up_raw],
            outputs=[output],
            name="swiglu_mul",
            version=21,
        )
    )

    outputs = [output]
    if scenario == "escape":
        outputs.append(gate_raw)

    graph = ir.Graph(
        [x],
        outputs,
        nodes=nodes,
        initializers=initializers,
        opset_imports={"": 21, "com.microsoft": 1},
        name=f"projection_fusion_{scenario}",
    )
    return ir.Model(graph, ir_version=10)


if __name__ == "__main__":
    scenario = sys.argv[1]
    if scenario not in {"valid", "bias", "zero_point", "escape", "decomposed"}:
        raise ValueError(f"unknown scenario: {scenario}")
    sys.stdout.buffer.write(ir.to_proto(build_model(scenario)).SerializeToString())
