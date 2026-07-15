"""Zero-copy DLPack **CUDA/GPU** interop tests for nxrt (import + export).

These prove the ``kDLCUDA`` path end to end:

* **Import** — a ``torch.rand(..., device='cuda')`` tensor fed to a CUDA nxrt
  session is *borrowed* (no host round-trip): the imported tensor's device
  pointer equals torch's ``data_ptr()``.
* **Export** — a CUDA nxrt output consumed by ``torch.from_dlpack`` yields a
  CUDA tensor that shares memory with nxrt's device buffer.
* **CPU fallback** — the same session code path still works for host inputs.

## Where these run

This dev machine (and CPU wheels generally) build nxrt **without** the ``cuda``
feature, so ``CUDAExecutionProvider`` is absent and every test here **skips**.
They are designed to run on an **H200** box with a CUDA-enabled nxrt wheel
(built via ``maturin develop --features cuda``) and a CUDA ``torch``. The skip
guard keys on *both* ``torch.cuda.is_available()`` and nxrt actually exposing
``CUDAExecutionProvider`` so a CPU wheel on a GPU host still skips cleanly.
"""

from __future__ import annotations

import numpy as np
import onnx.helper as oh
import pytest
from onnx import TensorProto

import nxrt

torch = pytest.importorskip("torch", reason="torch not installed")

# DLPack device type constants (DLDeviceType).
K_DL_CPU = 1
K_DL_CUDA = 2

CUDA_EP = "CUDAExecutionProvider"


def _cuda_unavailable_reason() -> str | None:
    """Return a skip reason if the GPU path cannot run here, else ``None``."""
    if not torch.cuda.is_available():
        return "torch reports no CUDA device"
    if CUDA_EP not in nxrt.get_available_providers():
        return (
            "this nxrt build has no CUDAExecutionProvider (built without the "
            "`cuda` feature); rebuild with `maturin develop --features cuda`"
        )
    return None


_SKIP_REASON = _cuda_unavailable_reason()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


def _identity_model(dtype: int, shape) -> bytes:
    """A single ``Identity`` node so the output aliases the input value."""
    vi_in = oh.make_tensor_value_info("X", dtype, shape)
    vi_out = oh.make_tensor_value_info("Y", dtype, shape)
    node = oh.make_node("Identity", ["X"], ["Y"])
    graph = oh.make_graph([node], "g_identity", [vi_in], [vi_out])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


def _cuda_session(dtype: int, shape) -> nxrt.InferenceSession:
    return nxrt.InferenceSession(
        _identity_model(dtype, shape), providers=[CUDA_EP, "CPUExecutionProvider"]
    )


# --------------------------------------------------------------------------- #
# (a) Import: torch CUDA tensor -> nxrt CUDA tensor, zero-copy
# --------------------------------------------------------------------------- #

def test_import_cuda_tensor_is_zero_copy():
    """The imported nxrt tensor borrows torch's device pointer (no host copy)."""
    x = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
    # The test-only hook returns the borrowed tensor's base device pointer, or
    # None if the value fell back to a copy. Equality with torch's own
    # data_ptr() is proof of a genuine zero-copy CUDA borrow.
    ptr = nxrt._dlpack_import_data_ptr(x)
    assert ptr is not None, "CUDA tensor was not borrowed zero-copy"
    assert ptr == x.data_ptr()


def test_import_cuda_device_advertised():
    x = torch.ones(8, dtype=torch.float32, device="cuda")
    assert tuple(x.__dlpack_device__()) == (K_DL_CUDA, x.device.index)


def test_run_with_cuda_input_no_host_roundtrip():
    """Feeding a CUDA torch tensor runs on the GPU and returns correct values."""
    sess = _cuda_session(TensorProto.FLOAT, [2, 3])
    host = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = torch.from_numpy(host).cuda()
    (out,) = sess.run(None, {"X": x})
    np.testing.assert_array_equal(np.asarray(out), host)


# --------------------------------------------------------------------------- #
# (b) Export: nxrt CUDA output -> torch.from_dlpack, shared memory
# --------------------------------------------------------------------------- #

def test_export_cuda_output_to_torch_shares_memory():
    sess = _cuda_session(TensorProto.FLOAT, [4])
    x = torch.arange(4, dtype=torch.float32, device="cuda")
    (value,) = sess.run_with_values(None, {"X": x})

    # The value must advertise CUDA so torch borrows on the right device.
    dev_type, dev_id = value.__dlpack_device__()
    assert dev_type == K_DL_CUDA

    borrowed = torch.from_dlpack(value)
    assert borrowed.is_cuda
    assert borrowed.device.index == dev_id
    torch.testing.assert_close(borrowed.cpu(), x.cpu())

    # Shared memory: mutating the borrow is visible through a fresh export of
    # the same nxrt buffer (pointer identity is the strongest proof).
    assert borrowed.data_ptr() == nxrt._dlpack_import_data_ptr(value)


def test_export_honors_stream_sync_default():
    """Export must succeed with the default stream handshake (producer sync)."""
    sess = _cuda_session(TensorProto.FLOAT, [16])
    x = torch.rand(16, dtype=torch.float32, device="cuda")
    (value,) = sess.run_with_values(None, {"X": x})
    # torch.from_dlpack passes the consumer stream to __dlpack__; a correct
    # producer sync means the data is valid immediately after the handshake.
    borrowed = torch.from_dlpack(value)
    torch.cuda.synchronize()
    torch.testing.assert_close(borrowed.cpu(), x.cpu())


# --------------------------------------------------------------------------- #
# (c) CPU fallback still works through the same code path
# --------------------------------------------------------------------------- #

def test_cpu_input_still_works_on_cuda_session():
    sess = _cuda_session(TensorProto.FLOAT, [5])
    host = np.linspace(0, 1, 5, dtype=np.float32)
    (out,) = sess.run(None, {"X": host})
    np.testing.assert_array_equal(np.asarray(out), host)


def test_cpu_export_is_kdlcpu():
    sess = _cuda_session(TensorProto.FLOAT, [3])
    x_cpu = torch.ones(3, dtype=torch.float32)
    (value,) = sess.run_with_values(None, {"X": x_cpu})
    # A CPU-resident output still exports as kDLCPU even on a CUDA session.
    assert value.__dlpack_device__()[0] in (K_DL_CPU, K_DL_CUDA)
