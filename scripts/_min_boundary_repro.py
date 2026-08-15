"""Minimal cross-partition repro for the plugin-EP CPU<->GPU deadlock (B3).

Graph:  a = Add(x, w)      # w is an initializer -> B2 declines -> CPU
        out = Sigmoid(a)   # elementwise, no initializer -> cuda_ep

This forces exactly one CPU -> cuda_ep -> output boundary (input copy + output
copy across the CPU/GPU partition). If Sigmoid ends up on cuda_ep and the run
completes quickly, the interspersed-partition path works. If it hangs, this is a
minimal, seconds-per-iteration repro of the decoder deadlock.

Usage: python scripts/_min_boundary_repro.py <plugin_lib>
"""
import os
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def build_model() -> bytes:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [4])
    w = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.float32), name="w")
    add = helper.make_node("Add", ["x", "w"], ["a"])
    sig = helper.make_node("Sigmoid", ["a"], ["out"])
    graph = helper.make_graph([add, sig], "boundary", [x], [out], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model.SerializeToString()


def main() -> int:
    lib = os.path.abspath(sys.argv[1])
    ort.register_execution_provider_library("nxrt_ep_cuda", lib)
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "cuda_ep"]
    print(f"cuda_ep devices: {len(devs)}", flush=True)

    model_bytes = build_model()
    so = ort.SessionOptions()
    so.add_provider_for_devices([devs[0]], {})
    print("creating session...", flush=True)
    sess = ort.InferenceSession(model_bytes, sess_options=so)
    print(f"providers={sess.get_providers()}", flush=True)

    feed = {"x": np.array([0, 0, 0, 0], dtype=np.float32)}
    print("running...", flush=True)
    t0 = time.time()
    out = sess.run(None, feed)[0]
    print(f"run OK in {time.time()-t0:.2f}s; out={out.tolist()}", flush=True)
    # expected: sigmoid([1,2,3,4])
    print(f"expected sigmoid([1,2,3,4]) = {(1/(1+np.exp(-np.array([1,2,3,4])))).tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
