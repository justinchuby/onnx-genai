#!/usr/bin/env python3
"""JSON adapter for installed ONNX ReferenceEvaluator and ONNX Runtime."""

from __future__ import annotations

import json
import sys


def probe() -> dict[str, object]:
    result: dict[str, object] = {
        "status": "unavailable",
        "onnx_version": None,
        "latest_einsum_schema": None,
        "onnxruntime_version": None,
        "reason": None,
    }
    try:
        import onnx

        result["onnx_version"] = onnx.__version__
        result["latest_einsum_schema"] = onnx.defs.get_schema(
            "Einsum", max_inclusive_version=10_000
        ).since_version
        from onnx.reference import ReferenceEvaluator  # noqa: F401

        result["status"] = "available"
    except Exception as error:  # pragma: no cover - environment-dependent
        result["reason"] = f"ONNX ReferenceEvaluator unavailable: {error}"
        return result
    try:
        import onnxruntime

        result["onnxruntime_version"] = onnxruntime.__version__
    except Exception:
        pass
    return result


def run(request: dict[str, object]) -> dict[str, object]:
    availability = probe()
    if availability["status"] != "available":
        return availability

    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    dtype_name = str(request["dtype"])
    dtype_table = {
        "float16": (TensorProto.FLOAT16, np.float16),
        "float32": (TensorProto.FLOAT, np.float32),
        "float64": (TensorProto.DOUBLE, np.float64),
    }
    if dtype_name == "bfloat16":
        if int(availability["latest_einsum_schema"]) < 28:
            return {
                "status": "unavailable",
                "reason": (
                    "installed ONNX exposes only Einsum-"
                    f"{availability['latest_einsum_schema']}; refusing to "
                    "reinterpret the requested Einsum-28 BF16 case as v12"
                ),
            }
        return {
            "status": "unavailable",
            "reason": "Python NumPy adapter has no portable BF16 ndarray dtype",
        }
    if dtype_name not in dtype_table:
        return {
            "status": "unavailable",
            "reason": f"Python adapter does not support dtype {dtype_name}",
        }
    tensor_dtype, numpy_dtype = dtype_table[dtype_name]
    input_infos = []
    feeds = {}
    input_names = []
    for index, tensor in enumerate(request["inputs"]):
        name = f"input_{index}"
        shape = [int(value) for value in tensor["shape"]]
        values = np.asarray(tensor["values"], dtype=numpy_dtype).reshape(shape)
        input_names.append(name)
        input_infos.append(helper.make_tensor_value_info(name, tensor_dtype, shape))
        feeds[name] = values
    output_shape = [int(value) for value in request["output_shape"]]
    output_info = helper.make_tensor_value_info("output", tensor_dtype, output_shape)
    node = helper.make_node(
        "Einsum",
        input_names,
        ["output"],
        equation=str(request["equation"]),
    )
    model = helper.make_model(
        helper.make_graph([node], "einsum_conformance", input_infos, [output_info]),
        opset_imports=[helper.make_opsetid("", int(request["opset"]))],
    )
    model.ir_version = min(model.ir_version, 10)

    engine = request["engine"]
    if engine == "onnx_reference":
        from onnx.reference import ReferenceEvaluator

        output = ReferenceEvaluator(model).run(None, feeds)[0]
    elif engine == "onnx_runtime":
        try:
            import onnxruntime as ort
        except Exception as error:
            return {
                "status": "unavailable",
                "reason": f"ONNX Runtime unavailable: {error}",
            }
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output = session.run(None, feeds)[0]
    else:
        return {"status": "error", "reason": f"unknown engine {engine}"}
    return {
        "status": "available",
        "shape": list(output.shape),
        "values": output.astype(np.float64).reshape(-1).tolist(),
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe"]:
        print(json.dumps(probe(), sort_keys=True))
    else:
        try:
            print(json.dumps(run(json.load(sys.stdin)), sort_keys=True))
        except Exception as error:
            print(json.dumps({"status": "error", "reason": str(error)}, sort_keys=True))
            raise
