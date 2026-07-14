"""ONNX backend-test adapter for the nxrt Python binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import onnx
from onnx.backend.base import Backend, BackendRep

import nxrt


class NxrtBackendRep(BackendRep):
    """Prepared nxrt session implementing the ONNX ``BackendRep`` contract."""

    def __init__(self, model: onnx.ModelProto) -> None:
        try:
            self._session = nxrt.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise RuntimeError(
                "nxrt could not prepare the ONNX backend-test model because the "
                "CPU execution provider rejected its graph or metadata. "
                f"Underlying nxrt error: {error}. Add the missing operator, "
                "dtype, or opset support identified above, or fix the malformed "
                "generated model, then rerun the case."
            ) from error

        self._input_names = [value.name for value in self._session.get_inputs()]
        self._output_names = [value.name for value in self._session.get_outputs()]

    def run(self, inputs: Any, **kwargs: Any) -> tuple[np.ndarray, ...]:
        del kwargs
        if isinstance(inputs, Mapping):
            feed = dict(inputs)
        elif isinstance(inputs, Sequence):
            if len(inputs) != len(self._input_names):
                raise ValueError(
                    "nxrt received the wrong number of ONNX backend-test inputs: "
                    f"the model requires {len(self._input_names)} "
                    f"({', '.join(self._input_names) or 'none'}), but the test "
                    f"provided {len(inputs)}. Pass one value per runtime graph "
                    "input in model order, or pass a mapping keyed by input name."
                )
            feed = dict(zip(self._input_names, inputs, strict=True))
        else:
            raise TypeError(
                "nxrt cannot run this ONNX backend-test case because inputs were "
                f"provided as {type(inputs).__name__}, not a mapping or ordered "
                "sequence. Pass {input_name: value} or one value per model input "
                "in graph order."
            )

        try:
            outputs = self._session.run(self._output_names, feed)
        except Exception as error:
            raise RuntimeError(
                "nxrt could not execute the ONNX backend-test model because the "
                "CPU execution provider rejected an operator, dtype, shape, "
                "attribute, or opset used by the case. "
                f"Underlying nxrt error: {error}. Implement the kernel capability "
                "identified above or correct the model/input data, then rerun "
                "the case."
            ) from error

        if len(outputs) != len(self._output_names):
            raise RuntimeError(
                "nxrt returned the wrong number of ONNX backend-test outputs: "
                f"the model declares {len(self._output_names)} "
                f"({', '.join(self._output_names) or 'none'}), but nxrt returned "
                f"{len(outputs)}. Fix the session output-selection or binding "
                "conversion logic, then rerun the case."
            )
        return tuple(np.asarray(output) for output in outputs)


class NxrtBackend(Backend):
    """CPU-only ONNX backend backed by ``nxrt.InferenceSession``."""

    @classmethod
    def prepare(
        cls, model: onnx.ModelProto, device: str = "CPU", **kwargs: Any
    ) -> BackendRep:
        del kwargs
        if not cls.supports_device(device):
            raise ValueError(
                "nxrt cannot prepare the ONNX backend-test model on device "
                f"{device!r} because this adapter only exposes CPU execution. "
                "Rerun the case with device='CPU'."
            )
        return NxrtBackendRep(model)

    @classmethod
    def supports_device(cls, device: str) -> bool:
        return device == "CPU"
