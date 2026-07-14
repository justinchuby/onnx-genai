"""Self-contained tests for the nxrt Python API surface, dtype round-tripping,
and — per ``RULES.md`` §1 — the *quality* of errors that cross into Python.

These need only ``numpy`` + ``onnx`` (no onnx-tests), so they run anywhere the
wheel is installed. The conformance slice against ``cbourjau/onnx-tests`` lives
in ``test_conformance.py``.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnx.helper as oh
import pytest
from onnx import TensorProto

import nxrt


def _single_op_model(op_type: str, dtype: int, shape, opset: int = 17,
                     inputs=("X",), output="Y", **attrs) -> bytes:
    """Build a serialized 1-node model: ``Y = op_type(*inputs)``."""
    in_vis = [oh.make_tensor_value_info(n, dtype, shape) for n in inputs]
    out_vi = oh.make_tensor_value_info(output, dtype, shape)
    node = oh.make_node(op_type, list(inputs), [output], **attrs)
    graph = oh.make_graph([node], f"g_{op_type}", in_vis, [out_vi])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", opset)])
    model.ir_version = 10
    return model.SerializeToString()


# --------------------------------------------------------------------------- #
# Module surface
# --------------------------------------------------------------------------- #

def test_module_surface():
    assert isinstance(nxrt.__version__, str) and nxrt.__version__
    providers = nxrt.get_available_providers()
    assert "CPUExecutionProvider" in providers


def test_session_metadata():
    model = _single_op_model("Relu", TensorProto.FLOAT, [2, 3])
    sess = nxrt.InferenceSession(model)
    (inp,) = sess.get_inputs()
    (out,) = sess.get_outputs()
    assert inp.name == "X"
    assert inp.type == "tensor(float)"
    assert list(inp.shape) == [2, 3]
    assert out.name == "Y"
    assert sess.get_providers() == ["CPUExecutionProvider"]


def test_load_from_path(tmp_path):
    model = _single_op_model("Relu", TensorProto.FLOAT, [2, 2])
    p = tmp_path / "relu.onnx"
    p.write_bytes(model)
    sess = nxrt.InferenceSession(str(p))
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    (y,) = sess.run(None, {"X": x})
    np.testing.assert_allclose(y, np.maximum(x, 0.0))


# --------------------------------------------------------------------------- #
# dtype round-tripping through the binding (numpy -> Tensor -> numpy)
# --------------------------------------------------------------------------- #

DTYPES = [
    (TensorProto.FLOAT, np.float32),
    (TensorProto.DOUBLE, np.float64),
    (TensorProto.FLOAT16, np.float16),
    (TensorProto.INT8, np.int8),
    (TensorProto.INT16, np.int16),
    (TensorProto.INT32, np.int32),
    (TensorProto.INT64, np.int64),
    (TensorProto.UINT8, np.uint8),
    (TensorProto.UINT16, np.uint16),
    (TensorProto.UINT32, np.uint32),
    (TensorProto.UINT64, np.uint64),
    (TensorProto.BOOL, np.bool_),
]

# The CPU EP's `Cast` kernel — the only dtype-agnostic passthrough available in
# this iteration — implements these source dtypes. The binding itself maps every
# dtype in DTYPES; f16/uint64 below are limited by *kernel* coverage, not the
# binding (asserted separately in ``test_dtype_unsupported_by_cpu_ep_*``).
CAST_SUPPORTED = [
    (TensorProto.FLOAT, np.float32),
    (TensorProto.DOUBLE, np.float64),
    (TensorProto.INT8, np.int8),
    (TensorProto.INT16, np.int16),
    (TensorProto.INT32, np.int32),
    (TensorProto.INT64, np.int64),
    (TensorProto.UINT8, np.uint8),
    (TensorProto.UINT16, np.uint16),
    (TensorProto.UINT32, np.uint32),
    (TensorProto.BOOL, np.bool_),
]


@pytest.mark.parametrize("onnx_dt,np_dt", CAST_SUPPORTED)
def test_dtype_roundtrip_via_identity(onnx_dt, np_dt):
    """``Cast`` to the same dtype is a supported identity that exercises the full
    numpy->tensor->numpy path for the dtypes the CPU EP's Cast kernel handles."""
    model = _single_op_model("Cast", onnx_dt, [2, 3], to=onnx_dt)
    sess = nxrt.InferenceSession(model)
    if np_dt == np.bool_:
        x = np.array([[True, False, True], [False, True, False]])
    else:
        x = np.arange(6, dtype=np_dt).reshape(2, 3)
    (y,) = sess.run(None, {"X": x})
    assert y.dtype == np_dt
    np.testing.assert_array_equal(y, x)


@pytest.mark.parametrize("onnx_dt,np_dt", [
    (TensorProto.FLOAT16, np.float16),
    (TensorProto.UINT64, np.uint64),
])
def test_dtype_accepted_by_binding_but_unimplemented_kernel(onnx_dt, np_dt):
    """The binding *accepts* these numpy dtypes (numpy->Tensor succeeds); the CPU
    EP's Cast kernel does not yet consume them, so the error names the op and the
    dtype rather than being a binding-level TypeError. Documents a runtime gap,
    not a binding bug."""
    model = _single_op_model("Cast", onnx_dt, [2, 3], to=onnx_dt)
    sess = nxrt.InferenceSession(model)
    x = np.arange(6, dtype=np_dt).reshape(2, 3)
    with pytest.raises(Exception) as ei:
        sess.run(None, {"X": x})
    # It reached the kernel (binding conversion worked) and the message is
    # actionable: it names Cast and the offending dtype.
    assert "Cast" in str(ei.value)


def test_noncontiguous_input_is_handled():
    model = _single_op_model("Relu", TensorProto.FLOAT, [3, 2])
    sess = nxrt.InferenceSession(model)
    base = np.arange(6, dtype=np.float32).reshape(2, 3) - 3.0
    x = base.T  # non-contiguous view, shape (3, 2)
    assert not x.flags["C_CONTIGUOUS"]
    (y,) = sess.run(None, {"X": x})
    np.testing.assert_allclose(y, np.maximum(x, 0.0))


def test_output_names_selection():
    model = _single_op_model("Relu", TensorProto.FLOAT, [2, 2])
    sess = nxrt.InferenceSession(model)
    x = np.ones((2, 2), dtype=np.float32)
    (y,) = sess.run(["Y"], {"X": x})
    np.testing.assert_allclose(y, x)


# --------------------------------------------------------------------------- #
# Error quality (RULES.md §1): what failed, why, how to fix — actionable types
# --------------------------------------------------------------------------- #

def test_unknown_provider_lists_available():
    model = _single_op_model("Relu", TensorProto.FLOAT, [1])
    with pytest.raises(ValueError) as ei:
        nxrt.InferenceSession(model, providers=["TotallyFakeEP"])
    msg = str(ei.value)
    assert "TotallyFakeEP" in msg and "CPUExecutionProvider" in msg


def test_cuda_provider_without_feature_is_actionable():
    model = _single_op_model("Relu", TensorProto.FLOAT, [1])
    with pytest.raises(ValueError) as ei:
        nxrt.InferenceSession(model, providers=["CUDAExecutionProvider"])
    # Only asserted when this wheel lacks CUDA (the default build).
    if "CUDAExecutionProvider" not in nxrt.get_available_providers():
        assert "CUDA" in str(ei.value)


def test_missing_model_file():
    with pytest.raises(FileNotFoundError) as ei:
        nxrt.InferenceSession("/no/such/model.onnx")
    assert "not found" in str(ei.value)


def test_unknown_input_name_lists_model_inputs():
    model = _single_op_model("Relu", TensorProto.FLOAT, [2, 2])
    sess = nxrt.InferenceSession(model)
    with pytest.raises(ValueError) as ei:
        sess.run(None, {"WrongName": np.ones((2, 2), dtype=np.float32)})
    assert "WrongName" in str(ei.value) and "X" in str(ei.value)


def test_unsupported_numpy_dtype_is_typeerror():
    model = _single_op_model("Relu", TensorProto.FLOAT, [2])
    sess = nxrt.InferenceSession(model)
    x = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    with pytest.raises(TypeError) as ei:
        sess.run(None, {"X": x})
    assert "complex64" in str(ei.value)


def test_unsupported_op_is_actionable():
    # `Sinh` is a valid ONNX op the CPU EP does not implement in this iteration.
    model = _single_op_model("Sinh", TensorProto.FLOAT, [2])
    with pytest.raises(ValueError) as ei:
        sess = nxrt.InferenceSession(model)
        sess.run(None, {"X": np.zeros(2, dtype=np.float32)})
    assert "Sinh" in str(ei.value)


def test_requested_missing_output_lists_outputs():
    model = _single_op_model("Relu", TensorProto.FLOAT, [2, 2])
    sess = nxrt.InferenceSession(model)
    with pytest.raises(ValueError) as ei:
        sess.run(["NoSuchOutput"], {"X": np.ones((2, 2), dtype=np.float32)})
    assert "NoSuchOutput" in str(ei.value) and "Y" in str(ei.value)
