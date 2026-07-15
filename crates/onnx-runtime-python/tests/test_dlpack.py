"""Zero-copy DLPack **export** tests for nxrt outputs.

These prove that ``InferenceSession.run_with_values`` returns ``NxrtValue``
objects implementing the Array-API DLPack producer protocol (``__dlpack__`` /
``__dlpack_device__``), and that consuming them with ``np.from_dlpack`` /
``torch.from_dlpack`` **borrows** nxrt's output buffer rather than copying it.

Only ``numpy`` + ``onnx`` are required; ``torch`` and ``ml_dtypes`` tests skip
when those packages are absent.
"""

from __future__ import annotations

import gc

import numpy as np
import onnx
import onnx.helper as oh
import pytest
from onnx import TensorProto

import nxrt

# DLPack device type constants (DLDeviceType).
K_DL_CPU = 1


def _identity_model(dtype: int, shape) -> bytes:
    """A single ``Identity`` node so the output aliases a known input value."""
    vi_in = oh.make_tensor_value_info("X", dtype, shape)
    vi_out = oh.make_tensor_value_info("Y", dtype, shape)
    node = oh.make_node("Identity", ["X"], ["Y"])
    graph = oh.make_graph([node], "g_identity", [vi_in], [vi_out])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


def _run_value(dtype: int, shape, x: np.ndarray):
    sess = nxrt.InferenceSession(_identity_model(dtype, shape))
    (value,) = sess.run_with_values(None, {"X": x})
    return value


# --------------------------------------------------------------------------- #
# Protocol surface
# --------------------------------------------------------------------------- #

def test_value_exposes_dlpack_protocol():
    v = _run_value(TensorProto.FLOAT, [2, 3], np.arange(6, dtype=np.float32).reshape(2, 3))
    assert hasattr(v, "__dlpack__")
    assert hasattr(v, "__dlpack_device__")
    assert hasattr(v, "numpy")


def test_dlpack_device_is_cpu():
    v = _run_value(TensorProto.FLOAT, [4], np.ones(4, dtype=np.float32))
    assert v.__dlpack_device__() == (K_DL_CPU, 0)


# --------------------------------------------------------------------------- #
# Zero-copy: identical data pointer, not a copy
# --------------------------------------------------------------------------- #

def test_numpy_from_dlpack_is_zero_copy_same_pointer():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    v = _run_value(TensorProto.FLOAT, [2, 3], x)
    # Two independent imports of the same value must borrow the *same* buffer.
    a1 = np.from_dlpack(v)
    a2 = np.from_dlpack(v)
    assert a1.ctypes.data == a2.ctypes.data
    assert not a1.flags["OWNDATA"]  # a borrowed view, never an owned copy
    np.testing.assert_array_equal(a1, x)


def test_mutating_numpy_view_writes_through_in_place():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    v = _run_value(TensorProto.FLOAT, [2, 3], x)
    arr = np.from_dlpack(v)
    assert arr.flags.writeable, "versioned DLPack export must be writable"
    arr[0, 0] = 12345.0
    # .numpy() copies *from the same underlying buffer*, so the write is visible.
    assert v.numpy()[0, 0] == 12345.0


# --------------------------------------------------------------------------- #
# Lifetime: the capsule/deleter keeps memory alive after the owner is dropped
# --------------------------------------------------------------------------- #

def test_array_survives_dropping_the_nxrt_value():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    v = _run_value(TensorProto.FLOAT, [2, 3], x)
    arr = np.from_dlpack(v)
    snapshot = arr.copy()
    # Drop the producing object; the DLManagedTensor deleter (holding an
    # Arc<Tensor>) must keep the buffer alive for the still-live `arr`.
    del v
    gc.collect()
    np.testing.assert_array_equal(arr, snapshot)
    # And it stays writable/valid — a use-after-free would corrupt or crash.
    arr[:] = 7.0
    np.testing.assert_array_equal(arr, np.full((2, 3), 7.0, dtype=np.float32))


def test_unconsumed_capsule_is_freed_without_leak_or_crash():
    # Requesting the capsule but never consuming it must run *our* destructor
    # (name still "dltensor_versioned"/"dltensor") without a double free.
    x = np.ones(4, dtype=np.float32)
    v = _run_value(TensorProto.FLOAT, [4], x)
    for _ in range(100):
        cap = v.__dlpack__(max_version=(1, 0))
        del cap
    gc.collect()
    # Still usable afterwards.
    np.testing.assert_array_equal(np.from_dlpack(v), x)


# --------------------------------------------------------------------------- #
# dtype coverage
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "onnx_dtype, np_dtype, values",
    [
        (TensorProto.FLOAT, np.float32, [1.0, -2.0, 3.5]),
        (TensorProto.DOUBLE, np.float64, [1.0, -2.0, 3.5]),
        (TensorProto.INT64, np.int64, [1, -2, 3]),
        (TensorProto.INT32, np.int32, [1, -2, 3]),
        (TensorProto.UINT8, np.uint8, [1, 2, 3]),
        (TensorProto.FLOAT16, np.float16, [1.0, -2.0, 3.5]),
        (TensorProto.BOOL, np.bool_, [True, False, True]),
    ],
)
def test_dtype_round_trips_zero_copy(onnx_dtype, np_dtype, values):
    x = np.array(values, dtype=np_dtype)
    v = _run_value(onnx_dtype, [len(values)], x)
    arr = np.from_dlpack(v)
    assert arr.dtype == np_dtype
    np.testing.assert_array_equal(arr, x)


def test_bfloat16_zero_copy_via_torch():
    # numpy's from_dlpack does not (yet) map the DLPack bfloat16 code, so nxrt's
    # correct kDLBfloat/16 export is exercised through torch, which does. The
    # capsule itself must be produced regardless.
    ml_dtypes = pytest.importorskip("ml_dtypes")
    torch = pytest.importorskip("torch")
    x = np.array([1.0, -2.0, 3.5], dtype=ml_dtypes.bfloat16)
    v = _run_value(TensorProto.BFLOAT16, [3], x)
    assert v.__dlpack__(max_version=(1, 0)) is not None
    t = torch.from_dlpack(v)
    assert t.dtype == torch.bfloat16
    assert t.float().tolist() == [1.0, -2.0, 3.5]


# --------------------------------------------------------------------------- #
# torch interop (optional)
# --------------------------------------------------------------------------- #

def test_torch_from_dlpack_shares_pointer_and_writes_through():
    torch = pytest.importorskip("torch")
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    v = _run_value(TensorProto.FLOAT, [2, 3], x)
    t = torch.from_dlpack(v)
    a = np.from_dlpack(v)
    assert t.data_ptr() == a.ctypes.data  # same physical buffer
    t[0, 0] = -9.0
    assert v.numpy()[0, 0] == -9.0  # torch write is visible through nxrt


# --------------------------------------------------------------------------- #
# Negative / contract cases
# --------------------------------------------------------------------------- #

def test_copy_true_is_rejected():
    v = _run_value(TensorProto.FLOAT, [2], np.ones(2, dtype=np.float32))
    with pytest.raises(Exception):
        v.__dlpack__(copy=True)


def test_dl_device_mismatch_is_rejected():
    v = _run_value(TensorProto.FLOAT, [2], np.ones(2, dtype=np.float32))
    with pytest.raises(Exception):
        # Ask to export onto a CUDA device (2, 0); this CPU value cannot.
        v.__dlpack__(dl_device=(2, 0))


def test_unversioned_fallback_still_exports():
    # max_version=None → unversioned "dltensor" capsule; numpy may mark it
    # read-only, but it must still be a valid zero-copy import.
    x = np.arange(4, dtype=np.float32)
    v = _run_value(TensorProto.FLOAT, [4], x)
    arr = np.from_dlpack(v)
    np.testing.assert_array_equal(arr, x)


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #

def test_run_still_returns_plain_numpy_arrays():
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [2, 2]))
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    (y,) = sess.run(None, {"X": x})
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)


def test_run_with_values_output_selection_by_name():
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [3]))
    x = np.arange(3, dtype=np.float32)
    (v,) = sess.run_with_values(["Y"], {"X": x})
    np.testing.assert_array_equal(np.from_dlpack(v), x)
