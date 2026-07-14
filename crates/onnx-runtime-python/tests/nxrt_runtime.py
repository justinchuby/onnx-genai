"""nxrt runtime adapter for the ``cbourjau/onnx-tests`` conformance suite.

The onnx-tests framework drives a "candidate" runtime through a single seam:
a function ``(onnx.ModelProto) -> dict[str, np.ndarray]`` selected via the
``RUN_CANDIDATE`` environment variable (see ``onnx_tests/config.py``). Point it
at nxrt with::

    RUN_CANDIDATE=nxrt_runtime.run_nxrt pytest        # from onnx-tests/tests

onnx-tests bakes every input into the model as an initializer/constant, so the
candidate runs a graph with *no* required inputs (``run(None, {})``).
"""

from __future__ import annotations

import numpy as np
import onnx

import nxrt


def run_nxrt(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Execute ``model`` with nxrt and return ``{output_name: ndarray}``.

    The model must carry all its data as constants/initializers (the onnx-tests
    contract), so the input feed is empty.
    """
    sess = nxrt.InferenceSession(model.SerializeToString())
    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(None, {})
    return {name: value for name, value in zip(output_names, outputs)}
