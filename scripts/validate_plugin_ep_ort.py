"""Validate the bundled CUDA plugin-EP cdylib loads and executes in ONNX Runtime.

Builds a tiny ONNX model (Add -> Mul), registers the plugin-EP library with ORT
via register_execution_provider_library, selects the `cuda_ep` device on a chosen
H200, runs the model, checks the numeric result, AND — critically — counts how
many nodes each execution provider actually executed by reading the ORT profile.

Historically this script printed PASS after checking only the numeric output and
`get_providers()`. Both are satisfied by *total silent CPU fallback*: ORT lists
`cuda_ep` in the session providers merely because it was registered, and the
numbers are correct because CPU produced them. Issue #956 hid behind exactly
that gap. A PASS here now REQUIRES that `cuda_ep` executed at least one node.
"""

import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ep_profile import count_nodes_by_provider, format_counts  # noqa: E402


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

    profile_prefix = os.path.join(tempfile.mkdtemp(prefix="nxrt_prof_"), "smoke")
    so = ort.SessionOptions()
    so.add_provider_for_devices(chosen, {})
    so.enable_profiling = True
    so.profile_file_prefix = profile_prefix

    sess = ort.InferenceSession(build_model(), sess_options=so)
    eps = sess.get_providers()
    print(f"session providers: {eps}")
    if "cuda_ep" not in eps:
        print("FAIL: session did not select cuda_ep (fell back to CPU)")
        return 1

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    (out,) = sess.run(None, {"x": x, "y": y})
    profile_path = sess.end_profiling()
    expected = (x + y) * y
    print(f"input x={x.tolist()} y={y.tolist()}")
    print(f"output={out.tolist()} expected={expected.tolist()}")
    if not np.allclose(out, expected, rtol=1e-5, atol=1e-5):
        print("FAIL: numeric mismatch")
        return 1

    counts = count_nodes_by_provider(profile_path)
    print("per-provider executed node counts:")
    print(format_counts(counts))

    cuda_nodes = sum(n for p, n in counts.items() if "cuda" in p.lower())
    if cuda_nodes == 0:
        print(
            "FAIL: cuda_ep executed ZERO nodes — the EP was registered and "
            "selected but every node silently fell back to another provider. "
            "This is the exact failure mode #956 documents."
        )
        return 1

    print(
        f"PASS: plugin-EP registered, selected on H200, executed correctly, "
        f"and claimed {cuda_nodes} node(s) on cuda_ep"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
