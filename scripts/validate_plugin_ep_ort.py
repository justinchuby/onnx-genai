"""Validate the bundled CUDA plugin-EP cdylib loads and executes in ONNX Runtime.

Builds a tiny ONNX model (Add -> Mul), registers the plugin-EP library with ORT
via register_execution_provider_library, selects the `cuda_ep` device on a chosen
H200, runs the model, and checks the numeric result. This confirms the ORT
plugin-load path specifically (the full 30B run exercises the kernels).
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


def build_model() -> bytes:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [4])
    add = helper.make_node("Add", ["x", "y"], ["s"])
    mul = helper.make_node("Mul", ["s", "y"], ["out"])
    graph = helper.make_graph([add, mul], "plugin_ep_smoke", [x, y], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model.SerializeToString()


def main() -> int:
    lib = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                          "target/release/libonnx_runtime_ep_cuda_plugin.so")
    device_index = int(os.environ.get("ONNX_GENAI_CUDA_DEVICE", "0"))
    reg_name = "nxrt_ep_cuda"

    print(f"onnxruntime {ort.__version__}; plugin lib: {lib}")
    ort.register_execution_provider_library(reg_name, lib)

    ep_devices = [d for d in ort.get_ep_devices() if d.ep_name == "cuda_ep"]
    if not ep_devices:
        print("FAIL: cuda_ep not discovered after registration")
        return 1
    print(f"discovered {len(ep_devices)} cuda_ep device(s)")

    # Select one H200 (device_index-th GPU device advertised by the plugin).
    chosen = [ep_devices[min(device_index, len(ep_devices) - 1)]]

    so = ort.SessionOptions()
    so.add_provider_for_devices(chosen, {})

    sess = ort.InferenceSession(build_model(), sess_options=so)
    eps = sess.get_providers()
    print(f"session providers: {eps}")
    if "cuda_ep" not in eps:
        print("FAIL: session did not select cuda_ep (fell back to CPU)")
        return 1

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    (out,) = sess.run(None, {"x": x, "y": y})
    expected = (x + y) * y
    print(f"input x={x.tolist()} y={y.tolist()}")
    print(f"output={out.tolist()} expected={expected.tolist()}")
    if not np.allclose(out, expected, rtol=1e-5, atol=1e-5):
        print("FAIL: numeric mismatch")
        return 1
    print("PASS: plugin-EP registered, selected on H200, and executed correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
