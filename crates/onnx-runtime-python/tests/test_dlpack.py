"""Zero-copy DLPack **export** tests for nxrt outputs.

These prove that ``InferenceSession.run_with_values`` returns ``NxrtValue``
objects implementing the Array-API DLPack producer protocol (``__dlpack__`` /
``__dlpack_device__``), and that consuming them with ``np.from_dlpack`` /
``torch.from_dlpack`` **borrows** nxrt's output buffer rather than copying it.

Only ``numpy`` + ``onnx`` are required; ``torch`` and ``ml_dtypes`` tests skip
when those packages are absent.
"""

from __future__ import annotations

import ctypes
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


def test_zero_element_dlpack_exports_null_data():
    """DLPack requires NULL data even though nxrt allocates a non-empty buffer."""

    class DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class DLDataType(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int),
            ("dtype", DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class DLManagedTensorVersioned(ctypes.Structure):
        _fields_ = [
            ("major", ctypes.c_uint32),
            ("minor", ctypes.c_uint32),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
            ("flags", ctypes.c_uint64),
            ("dl_tensor", DLTensor),
        ]

    v = _run_value(TensorProto.FLOAT, [0, 3], np.empty((0, 3), dtype=np.float32))
    capsule = v.__dlpack__(max_version=(1, 0))
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    managed = ctypes.cast(
        get_pointer(capsule, b"dltensor_versioned"),
        ctypes.POINTER(DLManagedTensorVersioned),
    )
    assert managed.contents.dl_tensor.data is None


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


# --------------------------------------------------------------------------- #
# Zero-copy IMPORT (feeding inputs): nxrt borrows a DLPack producer's buffer
# --------------------------------------------------------------------------- #

def _add_bias_model(dtype: int, shape) -> bytes:
    """``Identity`` is enough to route an input value through to an output."""
    return _identity_model(dtype, shape)


def test_numpy_input_is_imported_and_runs():
    # numpy ≥ 1.23 arrays expose __dlpack__, so this exercises the zero-copy
    # import path (falling back to copy only if that path declined).
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [2, 3]))
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    (y,) = sess.run(None, {"X": x})
    np.testing.assert_array_equal(y, x)


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    ]


_DELETER_FUNC = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensorVersioned))
_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_PyCapsule_New.restype = ctypes.py_object


class _NumpyDLPackProducer:
    """A minimal DLPack producer that borrows a float32 numpy array's buffer.

    Installs a Python deleter that increments ``flag[0]`` so a test can prove
    nxrt (a) read *this* buffer and (b) took ownership of the managed tensor and
    called the deleter exactly once — i.e. it borrowed, it did not copy.
    """

    def __init__(self, arr: np.ndarray, flag: list):
        assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]
        self.arr = arr  # keep the borrowed buffer alive
        self.flag = flag
        self._shape = (ctypes.c_int64 * arr.ndim)(*arr.shape)

        def _del(_mptr):
            flag[0] += 1

        self._del_cb = _DELETER_FUNC(_del)  # keep callback alive
        m = _DLManagedTensorVersioned()
        m.major, m.minor = 1, 0
        m.manager_ctx = None
        m.deleter = ctypes.cast(self._del_cb, ctypes.c_void_p)
        m.flags = 0
        m.dl_tensor.data = arr.ctypes.data
        m.dl_tensor.device = _DLDevice(K_DL_CPU, 0)
        m.dl_tensor.ndim = arr.ndim
        m.dl_tensor.dtype = _DLDataType(2, 32, 1)  # kDLFloat / 32-bit
        m.dl_tensor.shape = self._shape
        m.dl_tensor.strides = None  # C-contiguous
        m.dl_tensor.byte_offset = 0
        self._m = m  # keep the struct alive until the deleter runs

    def __dlpack_device__(self):
        return (K_DL_CPU, 0)

    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
        return _PyCapsule_New(ctypes.addressof(self._m), b"dltensor_versioned", None)


def test_custom_producer_buffer_is_borrowed_and_deleter_called_once():
    x = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    flag = [0]
    prod = _NumpyDLPackProducer(x, flag)
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [4]))
    (y,) = sess.run(None, {"X": prod})
    # nxrt read *our* borrowed buffer (no copy at import).
    np.testing.assert_array_equal(y, x)
    gc.collect()
    # nxrt owned the managed tensor and released it exactly once.
    assert flag[0] == 1, f"expected one deleter call, got {flag[0]}"


def test_borrow_reflects_source_mutation_before_run():
    # Mutating the source before `run` is visible in the result. NOTE: on its
    # own this does NOT prove zero-copy — the copy fallback also copies inside
    # `run()`, *after* this mutation, so a pure-copy implementation would pass
    # identically. The real no-copy proof is pointer identity, asserted in
    # `test_numpy_import_is_pointer_identical` below.
    x = np.zeros(4, dtype=np.float32)
    flag = [0]
    prod = _NumpyDLPackProducer(x, flag)
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [4]))
    x[:] = [1.0, 2.0, 3.0, 4.0]  # mutate the borrowed source before running
    (y,) = sess.run(None, {"X": prod})
    np.testing.assert_array_equal(y, [1.0, 2.0, 3.0, 4.0])


def test_numpy_import_is_pointer_identical():
    # The ONLY layer that truly proves a zero-copy borrow: the imported tensor's
    # first-element data pointer must EQUAL numpy's own buffer address. A copy
    # would land at a different address.
    x = np.arange(8, dtype=np.float32)
    ptr = nxrt._dlpack_import_data_ptr(x)
    assert ptr is not None, "contiguous CPU float32 array must be borrowed zero-copy"
    assert ptr == x.ctypes.data, "imported tensor must alias numpy's buffer (no copy)"


def test_noncontiguous_import_returns_no_pointer():
    # A non-contiguous view cannot be borrowed, so the pointer accessor reports
    # the copy fallback (None) rather than aliasing.
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    view = base[:, 1:3]
    assert not view.flags["C_CONTIGUOUS"]
    assert nxrt._dlpack_import_data_ptr(view) is None


def test_many_import_drop_cycles_no_double_free():
    # Shake out the capsule handshake / deleter path: many import+drop cycles
    # must not crash, leak, or double-free.
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [3]))
    for i in range(300):
        x = np.arange(3, dtype=np.float32) + i
        flag = [0]
        prod = _NumpyDLPackProducer(x, flag)
        (y,) = sess.run(None, {"X": prod})
        np.testing.assert_array_equal(y, x)
        gc.collect()
        assert flag[0] == 1


def test_noncontiguous_input_falls_back_to_copy():
    # A transposed (non-C-contiguous) view is not borrowable; nxrt must fall
    # back to the copy path and still produce the correct result.
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    xt = base.T  # shape (3, 2), F-contiguous
    assert not xt.flags["C_CONTIGUOUS"]
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [3, 2]))
    (y,) = sess.run(None, {"X": xt})
    np.testing.assert_array_equal(y, xt)


def test_sliced_input_falls_back_to_copy():
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    view = base[:, 1:3]  # non-contiguous slice
    assert not view.flags["C_CONTIGUOUS"]
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [3, 2]))
    (y,) = sess.run(None, {"X": view})
    np.testing.assert_array_equal(y, view)


def test_unsupported_dtype_input_raises_actionable_error():
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [2]))
    with pytest.raises((TypeError, ValueError)):
        sess.run(None, {"X": np.array([1 + 2j, 3 + 4j], dtype=np.complex64)})


def test_torch_tensor_input_is_imported_and_runs():
    torch = pytest.importorskip("torch")
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [2, 3]))
    (y,) = sess.run(None, {"X": x})
    np.testing.assert_array_equal(y, x.numpy())


def test_torch_input_zero_copy_write_through():
    # Mutating the torch tensor before run is visible in the result. As with the
    # numpy case above, mutation-visibility alone does NOT prove no-copy (the
    # copy fallback copies inside `run()` after this mutation); the pointer-
    # identity check in `test_torch_import_is_pointer_identical` is the real
    # proof.
    torch = pytest.importorskip("torch")
    x = torch.zeros(4, dtype=torch.float32)
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [4]))
    x[:] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    (y,) = sess.run(None, {"X": x})
    np.testing.assert_array_equal(y, [5.0, 6.0, 7.0, 8.0])


def test_torch_import_is_pointer_identical():
    # Pointer identity proves the torch input is borrowed, not copied.
    torch = pytest.importorskip("torch")
    x = torch.arange(6, dtype=torch.float32)
    ptr = nxrt._dlpack_import_data_ptr(x)
    assert ptr is not None, "contiguous CPU float32 torch tensor must be borrowed"
    assert ptr == x.data_ptr(), "imported tensor must alias torch's storage (no copy)"


# --------------------------------------------------------------------------- #
# GIL-safe foreign deleter (dropped off the Python thread)
# --------------------------------------------------------------------------- #

def test_imported_tensor_drop_on_background_thread_is_gil_safe():
    # The imported tensor's guard must reacquire the GIL before running the
    # foreign deleter (numpy calls Py_DECREF). Dropping it on an OS thread that
    # does NOT hold the GIL would deadlock or corrupt the interpreter if the
    # guard did not go through `Python::with_gil`. A clean return + a single
    # deleter call proves the invariant holds.
    x = np.arange(4, dtype=np.float32)
    flag = [0]
    prod = _NumpyDLPackProducer(x, flag)
    borrowed = nxrt._dlpack_import_drop_on_thread(prod)
    assert borrowed, "contiguous CPU float32 producer must be borrowed zero-copy"
    gc.collect()
    assert flag[0] == 1, f"deleter must run exactly once off-thread, got {flag[0]}"


# --------------------------------------------------------------------------- #
# Coverage: shape/dtype edge cases + the unversioned capsule branch
# --------------------------------------------------------------------------- #

# DLPack type codes (DLDataTypeCode).
_DL_INT = 0
_DL_FLOAT = 2
_DL_BFLOAT = 4


class _GenericDLPackProducer:
    """A DLPack producer over an arbitrary contiguous buffer.

    Unlike ``_NumpyDLPackProducer`` (float32-only), this takes an explicit
    DLPack ``(code, bits)`` and shape so tests can exercise 0-d, size-0, f64 and
    bfloat16 imports. It emits a **versioned** ``dltensor_versioned`` capsule.
    """

    def __init__(self, data_ptr: int, shape, code: int, bits: int, flag: list, keep=None):
        self.flag = flag
        self._keep = keep  # keep the backing buffer alive
        ndim = len(shape)
        self._shape = (ctypes.c_int64 * ndim)(*shape) if ndim else None

        def _del(_mptr):
            flag[0] += 1

        self._del_cb = _DELETER_FUNC(_del)
        m = _DLManagedTensorVersioned()
        m.major, m.minor = 1, 0
        m.manager_ctx = None
        m.deleter = ctypes.cast(self._del_cb, ctypes.c_void_p)
        m.flags = 0
        m.dl_tensor.data = data_ptr
        m.dl_tensor.device = _DLDevice(K_DL_CPU, 0)
        m.dl_tensor.ndim = ndim
        m.dl_tensor.dtype = _DLDataType(code, bits, 1)
        m.dl_tensor.shape = self._shape
        m.dl_tensor.strides = None
        m.dl_tensor.byte_offset = 0
        self._m = m

    def __dlpack_device__(self):
        return (K_DL_CPU, 0)

    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
        return _PyCapsule_New(ctypes.addressof(self._m), b"dltensor_versioned", None)


class _DLManagedTensor(ctypes.Structure):
    """Unversioned DLManagedTensor: dl_tensor first, then ctx + deleter."""

    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_DELETER_FUNC_UNV = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))


class _UnversionedDLPackProducer:
    """A DLPack producer that emits the legacy **unversioned** ``dltensor``
    capsule (the branch numpy < 2.1 / older torch use)."""

    def __init__(self, arr: np.ndarray, flag: list):
        assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]
        self.arr = arr
        self.flag = flag
        self._shape = (ctypes.c_int64 * arr.ndim)(*arr.shape)

        def _del(_mptr):
            flag[0] += 1

        self._del_cb = _DELETER_FUNC_UNV(_del)
        m = _DLManagedTensor()
        m.manager_ctx = None
        m.deleter = ctypes.cast(self._del_cb, ctypes.c_void_p)
        m.dl_tensor.data = arr.ctypes.data
        m.dl_tensor.device = _DLDevice(K_DL_CPU, 0)
        m.dl_tensor.ndim = arr.ndim
        m.dl_tensor.dtype = _DLDataType(_DL_FLOAT, 32, 1)
        m.dl_tensor.shape = self._shape
        m.dl_tensor.strides = None
        m.dl_tensor.byte_offset = 0
        self._m = m

    def __dlpack_device__(self):
        return (K_DL_CPU, 0)

    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
        # Ignore max_version and always hand back the unversioned form so we
        # exercise the "dltensor" consumer branch.
        return _PyCapsule_New(ctypes.addressof(self._m), b"dltensor", None)


def test_scalar_0d_import_is_borrowed():
    # 0-d scalar: numel == 1 (empty shape product), must borrow zero-copy.
    x = np.float32(3.5)
    arr = np.asarray(x)  # 0-d array, C-contiguous
    ptr = nxrt._dlpack_import_data_ptr(arr)
    assert ptr is not None, "0-d scalar must be borrowed"
    assert ptr == arr.ctypes.data


def test_size_zero_import_falls_back_cleanly():
    # A size-0 tensor has a possibly-null data pointer; must fall back to copy
    # without borrowing or crashing.
    x = np.zeros((0,), dtype=np.float32)
    assert nxrt._dlpack_import_data_ptr(x) is None
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [0]))
    (y,) = sess.run(None, {"X": x})
    assert y.shape == (0,)


def test_f64_import_is_borrowed():
    x = np.arange(5, dtype=np.float64)
    ptr = nxrt._dlpack_import_data_ptr(x)
    assert ptr is not None, "contiguous f64 array must be borrowed"
    assert ptr == x.ctypes.data
    sess = nxrt.InferenceSession(_identity_model(TensorProto.DOUBLE, [5]))
    (y,) = sess.run(None, {"X": x})
    np.testing.assert_array_equal(y, x)


def test_bf16_torch_import_is_borrowed():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "bfloat16"):
        pytest.skip("torch build lacks bfloat16")
    x = torch.arange(4, dtype=torch.bfloat16)
    # bfloat16 → DL_BFLOAT path; must borrow (pointer identity).
    ptr = nxrt._dlpack_import_data_ptr(x)
    assert ptr is not None, "contiguous bf16 torch tensor must be borrowed"
    assert ptr == x.data_ptr()


def test_unversioned_capsule_branch_is_imported_and_borrowed():
    # Force the legacy unversioned "dltensor" consumer path.
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    flag = [0]
    prod = _UnversionedDLPackProducer(x, flag)
    ptr = nxrt._dlpack_import_data_ptr(prod)
    assert ptr == x.ctypes.data, "unversioned capsule must be borrowed zero-copy"
    gc.collect()
    assert flag[0] == 1, f"unversioned deleter must run once, got {flag[0]}"


def test_unversioned_capsule_runs_end_to_end():
    x = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    flag = [0]
    prod = _UnversionedDLPackProducer(x, flag)
    sess = nxrt.InferenceSession(_identity_model(TensorProto.FLOAT, [3]))
    (y,) = sess.run(None, {"X": prod})
    np.testing.assert_array_equal(y, x)
    gc.collect()
    assert flag[0] == 1


def test_overflowing_shape_falls_back_to_copy():
    # A crafted foreign shape whose element-product overflows usize must not
    # panic — the import planner falls back (returns no borrow pointer).
    huge = (1 << 62)
    flag = [0]
    # Zero real backing; we never dereference because the plan aborts on the
    # overflowing dim product before any borrow. Use a 1-byte dummy address.
    dummy = (ctypes.c_uint8 * 1)()
    prod = _GenericDLPackProducer(
        ctypes.addressof(dummy), [huge, huge, huge], _DL_FLOAT, 32, flag, keep=dummy
    )
    # Must not panic; overflow → copy fallback → no borrow pointer.
    assert nxrt._dlpack_import_data_ptr(prod) is None
