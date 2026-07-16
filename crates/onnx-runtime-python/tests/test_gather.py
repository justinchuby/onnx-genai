"""Gather regressions built with the ONNX IR API, independent of onnx.helper."""

from __future__ import annotations

import numpy as np
import pytest
from onnxscript import ir

import nxrt


def _value(name: str, dtype: ir.DataType, shape: list[int]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _model(
    data_shape: list[int],
    indices_shape: list[int],
    output_shape: list[int],
    *,
    axis: int,
) -> bytes:
    data = _value("data", ir.DataType.FLOAT, data_shape)
    indices = _value("indices", ir.DataType.INT64, indices_shape)
    output = _value("output", ir.DataType.FLOAT, output_shape)
    node = ir.Node(
        "",
        "Gather",
        [data, indices],
        [ir.AttrInt64("axis", axis)],
        outputs=[output],
    )
    graph = ir.Graph(
        [data, indices],
        [output],
        nodes=[node],
        opset_imports={"": 13},
        name="gather_regression",
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _take(data: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
    normalized = [int(i) + data.shape[axis] if i < 0 else int(i) for i in indices.flat]
    pieces = [np.take(data, i, axis=axis) for i in normalized]
    stacked = np.stack(pieces, axis=axis)
    return stacked.reshape(
        (*data.shape[:axis], *indices.shape, *data.shape[axis + 1 :])
    )


@pytest.mark.parametrize("indices", [np.array([2]), np.array([2, 0])])
def test_gather_axis0_rows(indices: np.ndarray):
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    session = nxrt.InferenceSession(
        _model([3, 2], list(indices.shape), [len(indices), 2], axis=0)
    )
    (actual,) = session.run(None, {"data": data, "indices": indices})
    np.testing.assert_array_equal(actual, _take(data, indices, 0))


def test_gather_axis0_negative_indices_wrap():
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    indices = np.array([-1, -3], dtype=np.int64)
    session = nxrt.InferenceSession(_model([3, 2], [2], [2, 2], axis=0))
    (actual,) = session.run(None, {"data": data, "indices": indices})
    np.testing.assert_array_equal(actual, _take(data, indices, 0))


def test_gather_axis0_out_of_range_is_rejected():
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    indices = np.array([0, 3], dtype=np.int64)
    session = nxrt.InferenceSession(_model([3, 2], [2], [2, 2], axis=0))
    with pytest.raises(Exception, match="index 3 out of range"):
        session.run(None, {"data": data, "indices": indices})


def test_gather_axis1_general_path():
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    indices = np.array([2, 0], dtype=np.int64)
    session = nxrt.InferenceSession(_model([2, 3], [2], [2, 2], axis=1))
    (actual,) = session.run(None, {"data": data, "indices": indices})
    np.testing.assert_array_equal(actual, _take(data, indices, 1))


def test_gather_axis0_mismatched_declared_output_is_inferred_safely():
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    indices = np.array([2, 0], dtype=np.int64)
    session = nxrt.InferenceSession(_model([3, 2], [2], [1, 2], axis=0))
    (actual,) = session.run(None, {"data": data, "indices": indices})
    assert actual.shape == (2, 2)
    np.testing.assert_array_equal(actual, _take(data, indices, 0))
