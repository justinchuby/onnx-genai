"""BlockQuantizedMatMul fixtures built with the ONNX IR API."""

from __future__ import annotations

import numpy as np
from onnxscript import ir

import nxrt


DOMAIN = "com.github.onnxruntime.genai"


def _value(name: str, dtype: ir.DataType, shape: list[int]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _model() -> bytes:
    activation = _value("A", ir.DataType.FLOAT, [1, 32])
    packed_weight = _value("packed_B", ir.DataType.UINT8, [1, 1, 17])
    bias = _value("bias", ir.DataType.FLOAT, [1])
    output = _value("Y", ir.DataType.FLOAT, [1, 1])
    node = ir.Node(
        DOMAIN,
        "BlockQuantizedMatMul",
        [activation, packed_weight, bias],
        [
            ir.AttrInt64("K", 32),
            ir.AttrInt64("N", 1),
            ir.AttrString("format", "mxfp4"),
            ir.AttrInt64("block_layout_version", 1),
        ],
        outputs=[output],
        version=1,
    )
    graph = ir.Graph(
        [activation, packed_weight, bias],
        [output],
        nodes=[node],
        opset_imports={DOMAIN: 1},
        name="mxfp4_matmul",
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def test_mxfp4_known_block_matches_e2m1_reference():
    packed = np.empty((1, 1, 17), dtype=np.uint8)
    packed[0, 0, 0] = 127  # E8M0 scale = 1.
    packed[0, 0, 1:] = [code | (code << 4) for code in range(16)]
    e2m1 = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=np.float32,
    )
    weight = np.concatenate([e2m1, e2m1])
    activation = np.arange(1, 33, dtype=np.float32).reshape(1, 32) / 8
    bias = np.array([0.25], dtype=np.float32)

    session = nxrt.InferenceSession(_model())
    (actual,) = session.run(
        None, {"A": activation, "packed_B": packed, "bias": bias}
    )
    expected = activation @ weight.reshape(32, 1) + bias
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
