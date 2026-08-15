"""Tests for nxrt's ergonomic, human-friendly API: the callable
``InferenceSession`` (``__call__``), ``nxrt.load``, the ``Outputs`` container,
``bind_outputs``, and the tensor-like ``NxrtValue`` surface.

The onnxruntime-compatible ``run()``/``run_with_values()`` remain covered by
``test_api.py``/``test_dlpack.py``; here we exercise only the new sugar. Needs
only ``numpy`` + ``onnx`` (torch is optional and skipped when absent).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import onnx.helper as oh
import pytest
from onnx import TensorProto

import nxrt

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _relu_model(shape=(2, 3), name_in="X", name_out="Y") -> bytes:
    """Single-input, single-output ``Y = Relu(X)``."""
    vi = oh.make_tensor_value_info(name_in, TensorProto.FLOAT, list(shape))
    vo = oh.make_tensor_value_info(name_out, TensorProto.FLOAT, list(shape))
    node = oh.make_node("Relu", [name_in], [name_out])
    graph = oh.make_graph([node], "g_relu", [vi], [vo])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


def _add_model(shape=(2, 2)) -> bytes:
    """Two-input, single-output ``S = Add(A, B)``."""
    a = oh.make_tensor_value_info("A", TensorProto.FLOAT, list(shape))
    b = oh.make_tensor_value_info("B", TensorProto.FLOAT, list(shape))
    s = oh.make_tensor_value_info("S", TensorProto.FLOAT, list(shape))
    node = oh.make_node("Add", ["A", "B"], ["S"])
    graph = oh.make_graph([node], "g_add", [a, b], [s])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


def _multi_output_model(shape=(2, 2)) -> bytes:
    """One input, two outputs: ``P = Relu(X)`` and ``N = Neg(X)``."""
    x = oh.make_tensor_value_info("X", TensorProto.FLOAT, list(shape))
    p = oh.make_tensor_value_info("pos", TensorProto.FLOAT, list(shape))
    n = oh.make_tensor_value_info("neg", TensorProto.FLOAT, list(shape))
    node_p = oh.make_node("Relu", ["X"], ["pos"])
    node_n = oh.make_node("Neg", ["X"], ["neg"])
    graph = oh.make_graph([node_p, node_n], "g_multi", [x], [p, n])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


def _zero_input_model() -> bytes:
    """No graph inputs; a ``Constant`` node produces the single output ``C``."""
    c = oh.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])
    const = oh.make_tensor(
        "value", TensorProto.FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0]
    )
    node = oh.make_node("Constant", [], ["C"], value=const)
    graph = oh.make_graph([node], "g_zero_in", [], [c])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


def _scalar_output_model(shape=(2, 3)) -> bytes:
    """One input, a single 0-d (scalar) output ``s = ReduceSum(X)``."""
    x = oh.make_tensor_value_info("X", TensorProto.FLOAT, list(shape))
    s = oh.make_tensor_value_info("s", TensorProto.FLOAT, [])
    node = oh.make_node("ReduceSum", ["X"], ["s"], keepdims=0)
    graph = oh.make_graph([node], "g_scalar", [x], [s])
    model = oh.make_model(graph, opset_imports=[oh.make_operatorsetid("", 17)])
    model.ir_version = 10
    return model.SerializeToString()


# --------------------------------------------------------------------------- #
# __call__ input dispatch
# --------------------------------------------------------------------------- #

def test_call_single_input_single_output_returns_value_directly():
    sess = nxrt.load(_relu_model((2, 2)))
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    out = sess(x)
    # A single output is the value itself, NOT a list/container.
    assert isinstance(out, nxrt.NxrtValue)
    np.testing.assert_allclose(np.asarray(out), np.maximum(x, 0.0))


def test_call_positional_multi_input():
    sess = nxrt.load(_add_model())
    a = np.ones((2, 2), dtype=np.float32)
    b = np.full((2, 2), 2.0, dtype=np.float32)
    out = sess(a, b)
    np.testing.assert_allclose(np.asarray(out), a + b)


def test_call_kwargs():
    sess = nxrt.load(_add_model())
    a = np.ones((2, 2), dtype=np.float32)
    b = np.full((2, 2), 3.0, dtype=np.float32)
    out = sess(B=b, A=a)
    np.testing.assert_allclose(np.asarray(out), a + b)


def test_call_dict_feed():
    sess = nxrt.load(_add_model())
    a = np.ones((2, 2), dtype=np.float32)
    b = np.full((2, 2), 5.0, dtype=np.float32)
    out = sess({"A": a, "B": b})
    np.testing.assert_allclose(np.asarray(out), a + b)


def test_call_positional_and_kwargs_mix():
    sess = nxrt.load(_add_model())
    a = np.ones((2, 2), dtype=np.float32)
    b = np.full((2, 2), 7.0, dtype=np.float32)
    out = sess(a, B=b)
    np.testing.assert_allclose(np.asarray(out), a + b)


def test_call_dict_feed_plus_kwargs_merge():
    sess = nxrt.load(_add_model())
    a = np.ones((2, 2), dtype=np.float32)
    b = np.full((2, 2), 4.0, dtype=np.float32)
    out = sess({"A": a}, B=b)
    np.testing.assert_allclose(np.asarray(out), a + b)


# --------------------------------------------------------------------------- #
# __call__ error quality
# --------------------------------------------------------------------------- #

def test_call_unknown_kwarg_lists_inputs():
    sess = nxrt.load(_add_model())
    a = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError) as ei:
        sess(A=a, Nope=a)
    msg = str(ei.value)
    assert "Nope" in msg and "A" in msg and "B" in msg


def test_call_duplicate_positional_and_kwarg():
    sess = nxrt.load(_add_model())
    a = np.ones((2, 2), dtype=np.float32)
    b = np.full((2, 2), 2.0, dtype=np.float32)
    with pytest.raises(ValueError) as ei:
        sess(a, A=b)  # "A" both positionally and by keyword
    assert "A" in str(ei.value)


def test_call_duplicate_dict_and_kwarg():
    sess = nxrt.load(_add_model())
    a = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError) as ei:
        sess({"A": a, "B": a}, B=a)
    assert "B" in str(ei.value)


def test_call_missing_input_lists_inputs():
    sess = nxrt.load(_add_model())
    a = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError) as ei:
        sess(a)  # B unfed
    assert "B" in str(ei.value)


def test_call_too_many_positionals():
    sess = nxrt.load(_relu_model((2, 2)))
    x = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError) as ei:
        sess(x, x)  # model has only one input
    msg = str(ei.value)
    assert "too many" in msg.lower() and "X" in msg


# --------------------------------------------------------------------------- #
# Outputs container
# --------------------------------------------------------------------------- #

def test_multi_output_container_access():
    sess = nxrt.load(_multi_output_model())
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    out = sess(x)
    assert isinstance(out, nxrt.Outputs)
    assert len(out) == 2
    # index, name, and attribute access all reach the same values.
    np.testing.assert_allclose(np.asarray(out[0]), np.maximum(x, 0.0))
    np.testing.assert_allclose(np.asarray(out["neg"]), -x)
    np.testing.assert_allclose(np.asarray(out.pos), np.maximum(x, 0.0))
    # keys/values/items reflect graph order.
    assert out.keys() == ["pos", "neg"]
    assert "neg" in out
    names = [n for n, _ in out.items()]
    assert names == ["pos", "neg"]
    assert len(out.values()) == 2
    # repr lists name -> shape/dtype.
    r = repr(out)
    assert "pos" in r and "neg" in r and "float32" in r


def test_multi_output_unpacking():
    sess = nxrt.load(_multi_output_model())
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    pos, neg = sess(x)  # iterable / unpackable
    np.testing.assert_allclose(np.asarray(pos), np.maximum(x, 0.0))
    np.testing.assert_allclose(np.asarray(neg), -x)


def test_outputs_unknown_name_and_index_errors():
    sess = nxrt.load(_multi_output_model())
    out = sess(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(KeyError):
        _ = out["missing"]
    with pytest.raises(IndexError):
        _ = out[5]
    with pytest.raises(AttributeError):
        _ = out.missing


# --------------------------------------------------------------------------- #
# bind_outputs
# --------------------------------------------------------------------------- #

def test_bind_outputs_selects_subset():
    sess = nxrt.load(_multi_output_model())
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    with sess.bind_outputs("neg") as s:
        out = s(x)
        # Single selected output -> returned directly.
        assert isinstance(out, nxrt.NxrtValue)
        np.testing.assert_allclose(np.asarray(out), -x)
    # The base session is unaffected: it still returns both outputs.
    both = sess(x)
    assert isinstance(both, nxrt.Outputs)
    assert len(both) == 2


def test_bind_outputs_without_with_block():
    # The proxy works standalone, no `with` required.
    sess = nxrt.load(_multi_output_model())
    x = np.ones((2, 2), dtype=np.float32)
    s = sess.bind_outputs("neg")
    assert isinstance(s(x), nxrt.NxrtValue)
    # base session still returns all outputs.
    assert isinstance(sess(x), nxrt.Outputs)


def test_bind_outputs_base_session_unaffected_after_exception():
    # An exception raised inside the `with` block must not corrupt selection
    # state: the base session still returns all outputs afterwards.
    sess = nxrt.load(_multi_output_model())
    x = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(RuntimeError):
        with sess.bind_outputs("neg") as s:
            assert isinstance(s(x), nxrt.NxrtValue)
            raise RuntimeError("boom")
    both = sess(x)
    assert isinstance(both, nxrt.Outputs)
    assert len(both) == 2


def test_bind_outputs_two_proxies_interleaved_distinct_outputs():
    # Two proxies from the SAME session, used interleaved, must each return
    # their own distinct output (proves there is no shared selection state to
    # clobber).
    sess = nxrt.load(_multi_output_model())
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    pos_proxy = sess.bind_outputs("pos")
    neg_proxy = sess.bind_outputs("neg")
    np.testing.assert_allclose(np.asarray(neg_proxy(x)), -x)
    np.testing.assert_allclose(np.asarray(pos_proxy(x)), np.maximum(x, 0.0))
    np.testing.assert_allclose(np.asarray(neg_proxy(x)), -x)
    np.testing.assert_allclose(np.asarray(pos_proxy(x)), np.maximum(x, 0.0))
    # base session still returns all outputs, unaffected by either proxy.
    assert len(sess(x)) == 2


def test_bind_outputs_unknown_name_errors():
    sess = nxrt.load(_multi_output_model())
    with pytest.raises(ValueError) as ei:
        sess.bind_outputs("nope")
    msg = str(ei.value)
    assert "nope" in msg and "pos" in msg and "neg" in msg


def test_explicit_unknown_output_fails_before_inference():
    sess = nxrt.load(_multi_output_model())
    x = np.ones((2, 2), dtype=np.float32)  # otherwise a valid inference feed
    with pytest.raises(ValueError) as ei:
        sess.run(["nope"], {"X": x})
    msg = str(ei.value)
    assert "nope" in msg and "pos" in msg and "neg" in msg


def test_bind_outputs_does_not_affect_run():
    sess = nxrt.load(_multi_output_model())
    x = np.ones((2, 2), dtype=np.float32)
    with sess.bind_outputs("neg"):
        # run() on the base session ignores the binding: still all outputs.
        outs = sess.run(None, {"X": x})
        assert len(outs) == 2


def test_run_with_values_on_base_session_ignores_binding():
    sess = nxrt.load(_multi_output_model())
    x = np.ones((2, 2), dtype=np.float32)
    proxy = sess.bind_outputs("neg")
    # The base session's run_with_values ignores any proxy and returns all.
    vals = sess.run_with_values(None, {"X": x})
    assert len(vals) == 2
    names = {v.name for v in vals} if hasattr(vals[0], "name") else None
    _ = names  # name attribute is optional; the count is the contract here.
    # The proxy, by contrast, applies its own subset when output_names is None.
    proxy_vals = proxy.run_with_values(None, {"X": x})
    assert len(proxy_vals) == 1


def test_bind_outputs_proxy_run_defaults_to_subset():
    sess = nxrt.load(_multi_output_model())
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    proxy = sess.bind_outputs("neg")
    # run(None, ...) on the proxy defaults to its subset.
    outs = proxy.run(None, {"X": x})
    assert len(outs) == 1
    np.testing.assert_allclose(outs[0], -x)
    # Explicit output_names still wins over the proxy's subset.
    both = proxy.run(["pos", "neg"], {"X": x})
    assert len(both) == 2


# --------------------------------------------------------------------------- #
# nxrt.load
# --------------------------------------------------------------------------- #

def test_load_cpu_default():
    sess = nxrt.load(_relu_model((2, 2)))
    assert sess.get_providers() == ["CPUExecutionProvider"]
    # Explicit device="cpu" is the same.
    sess2 = nxrt.load(_relu_model((2, 2)), device="cpu")
    assert sess2.get_providers() == ["CPUExecutionProvider"]


def test_load_providers_override_wins_over_device():
    sess = nxrt.load(
        _relu_model((2, 2)),
        device="cuda",
        providers=["CPUExecutionProvider"],
    )
    assert sess.get_providers() == ["CPUExecutionProvider"]


def test_load_unknown_device_is_actionable():
    with pytest.raises(ValueError) as ei:
        nxrt.load(_relu_model((2, 2)), device="tpu")
    assert "tpu" in str(ei.value)


def test_load_cuda_ordinal_accepted():
    # `cuda:0` is a well-formed ordinal: it passes device parsing and reaches
    # provider validation. This build has no CUDA, so it then fails with the
    # provider-build error -- NOT an ordinal-format error, proving the ordinal
    # was accepted.
    with pytest.raises(ValueError) as ei:
        nxrt.load(_relu_model((2, 2)), device="cuda:0")
    msg = str(ei.value)
    assert "CUDAExecutionProvider" in msg
    assert "ordinal" not in msg


def test_load_cuda_bad_ordinal_raises():
    with pytest.raises(ValueError) as ei:
        nxrt.load(_relu_model((2, 2)), device="cuda:abc")
    msg = str(ei.value)
    assert "cuda:abc" in msg and "ordinal" in msg


# --------------------------------------------------------------------------- #
# NxrtValue tensor-like surface
# --------------------------------------------------------------------------- #

def test_value_array_shape_dtype_len_repr():
    sess = nxrt.load(_relu_model((2, 3)))
    x = np.arange(6, dtype=np.float32).reshape(2, 3) - 3.0
    v = sess(x)
    # shape / dtype / len
    assert v.shape == (2, 3)
    assert v.dtype == np.dtype("float32")
    assert len(v) == 2
    # __array__ makes np.asarray work and equals the copy path.
    np.testing.assert_array_equal(np.asarray(v), v.numpy())
    # __array__ honors an explicit dtype.
    assert np.asarray(v, dtype=np.float64).dtype == np.float64
    # repr mentions name, shape, dtype.
    r = repr(v)
    assert "Y" in r and "float32" in r and "(2, 3)" in r.replace("[", "(").replace("]", ")")


def test_call_zero_input_model():
    # A model with no graph inputs is callable with no arguments.
    sess = nxrt.load(_zero_input_model())
    out = sess()
    assert isinstance(out, nxrt.NxrtValue)
    np.testing.assert_array_equal(
        np.asarray(out), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )


def test_scalar_value_len_raises():
    # A 0-d (scalar) NxrtValue has no length, matching numpy's TypeError.
    sess = nxrt.load(_scalar_output_model((2, 3)))
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    v = sess(x)
    assert isinstance(v, nxrt.NxrtValue)
    assert v.shape == ()
    with pytest.raises(TypeError):
        len(v)
    # value itself is still correct.
    assert float(np.asarray(v)) == pytest.approx(float(x.sum()))


def test_array_copy_false_raises():
    # numpy's contract: __array__(copy=False) must error when a copy is
    # required. NxrtValue can only produce a copy, so it raises.
    sess = nxrt.load(_relu_model((2, 3)))
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    v = sess(x)
    with pytest.raises(ValueError):
        v.__array__(copy=False)
    # copy=True and copy=None (the default) both succeed.
    np.testing.assert_array_equal(v.__array__(copy=True), v.numpy())
    np.testing.assert_array_equal(v.__array__(copy=None), v.numpy())


def test_call_equivalent_to_run():
    sess = nxrt.load(_relu_model((2, 3)))
    x = np.arange(6, dtype=np.float32).reshape(2, 3) - 3.0
    got = np.asarray(sess(x))
    (expected,) = sess.run(None, {"X": x})
    np.testing.assert_array_equal(got, expected)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_call_output_is_zero_copy_for_torch():
    import torch

    sess = nxrt.load(_relu_model((2, 3)))
    x = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0  # all positive -> Relu is identity
    v = sess(x)
    t = torch.from_dlpack(v)  # zero-copy borrow of nxrt's output buffer
    np.testing.assert_array_equal(t.numpy(), x)
