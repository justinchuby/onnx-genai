"""Conformance slice: drive nxrt through the ``cbourjau/onnx-tests`` Hypothesis
generators and compare against ``onnx.reference.ReferenceEvaluator``.

This proves the "just ``pytest``" payoff: the same property-based generators the
ONNX standard's conformance suite uses, pointed at nxrt with a one-line runtime
swap (see ``nxrt_runtime.run_nxrt``). Ops the CPU EP implements match the
reference; ops it does not implement surface an **actionable** nxrt error naming
the operator (demonstrated by ``test_unsupported_op_reports_cleanly``), so a
full-suite report cleanly separates "wrong answer" from "not yet implemented".

Requires ``onnx-tests`` (+ ``spox``, ``hypothesis``) importable; skipped
otherwise. Point the *whole* onnx-tests suite at nxrt for the documented
scale-up step::

    RUN_CANDIDATE=nxrt_runtime.run_nxrt pytest        # from the onnx-tests repo
"""

from __future__ import annotations

import numpy as np
import pytest

# The suite's generators + reference seam; skip cleanly if not installed.
elementwise_ops = pytest.importorskip("onnx_tests.elementwise_ops")
linear_algebra_ops = pytest.importorskip("onnx_tests.linear_algebra_ops")
_rt = pytest.importorskip("onnx_tests.runtime_wrappers")
spox_future = pytest.importorskip("spox._future")
op17 = pytest.importorskip("spox.opset.ai.onnx.v17")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from nxrt_runtime import run_nxrt  # local adapter (tests/ is on sys.path)

run_reference = _rt.run_reference

np.seterr(all="ignore")  # generators intentionally probe inf/nan domains

# Ops the CPU EP implements today. Each entry is (op_name, generator).
SUPPORTED_OPS = [
    ("Add", elementwise_ops.add),
    ("Sub", elementwise_ops.sub),
    ("Mul", elementwise_ops.mul),
    ("Div", elementwise_ops.div),
    ("Relu", elementwise_ops.relu),
    ("Sqrt", elementwise_ops.sqrt),
    ("Tanh", elementwise_ops.tanh),
    ("Erf", elementwise_ops.erf),
    ("MatMul", linear_algebra_ops.matmul),
]


def _assert_matches_reference(model):
    candidate = run_nxrt(model)
    expected = run_reference(model)
    assert list(candidate) == list(expected), (
        f"output names differ: nxrt={list(candidate)} reference={list(expected)}"
    )
    for name in expected:
        cand, exp = candidate[name], expected[name]
        assert cand.dtype == exp.dtype, (
            f"output {name!r}: dtype {cand.dtype} != reference {exp.dtype}"
        )
        np.testing.assert_allclose(
            cand, exp, rtol=1e-3, atol=1e-4, equal_nan=True,
            err_msg=f"output {name!r} disagrees with the ONNX reference",
        )


@pytest.mark.parametrize("op_name,factory", SUPPORTED_OPS, ids=[o[0] for o in SUPPORTED_OPS])
def test_conformance_vs_reference(op_name, factory):
    """nxrt matches the ONNX reference on float32 over Hypothesis-generated
    inputs (shapes, broadcasting, and edge values) for each supported op."""

    @settings(max_examples=25, deadline=None, derandomize=True,
              suppress_health_check=list(HealthCheck))
    @given(data=st.data())
    def run(data):
        with spox_future.value_prop_backend(spox_future.ValuePropBackend.NONE):
            state = data.draw(factory(np.dtype("float32"), op17))
            model = state.build_model()
        _assert_matches_reference(model)

    run()


def test_unsupported_op_reports_cleanly():
    """An op the CPU EP does not implement (``Sin``) must fail with an actionable
    error that names the operator — never a silent wrong answer or opaque crash
    (``RULES.md`` §1). This is what a full-suite run reports as UNSUPPORTED."""
    import spox

    with spox_future.value_prop_backend(spox_future.ValuePropBackend.NONE):
        res = op17.sin(spox_future.initializer(np.ones((2, 3), np.float32)))
        model = spox.build({}, {"res": res})
    with pytest.raises(Exception) as ei:
        run_nxrt(model)
    assert "Sin" in str(ei.value)
