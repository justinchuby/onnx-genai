from __future__ import annotations

import numpy as np
import pytest

import nxrt


def test_eager_submodule_import_and_add():
    from nxrt import eager

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    (result,) = eager.dispatch("Add", [a, b])
    np.testing.assert_array_equal(result, a + b)
    assert eager.opset() == eager.LATEST_ONNX_OPSET


def test_eager_attributes_and_cache_stats():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    (result,) = nxrt.eager.dispatch("Softmax", [x], {"axis": -1})
    np.testing.assert_allclose(result.sum(axis=-1), np.ones(1), rtol=1e-6)
    stats = nxrt.eager.cache_stats()
    assert {"entries", "hits", "misses"} <= stats.keys()


def test_genai_submodule_import_and_missing_directory_error():
    from nxrt.genai import Engine, GenerateResult

    assert GenerateResult.__module__ == "nxrt.genai"
    with pytest.raises(FileNotFoundError, match="model directory not found"):
        Engine.from_dir("/no/such/nxrt-genai-model")
