#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Minimal reproduction: what a second ORT CUDA session costs a captured graph.

ORT gives every session its own CUDA stream. When one decode step drives more
than one session — a fused decoder island plus the policy island that samples
from its logits, say — the device alternates between those streams, and that
alternation costs far more than the second session's own work.

This probe isolates the effect with synthetic graphs, so it needs no model
package:

* `alone`  — replay the heavy captured session on its own.
* `split`  — replay it, then run a single-node session on its own ORT stream.
* `shared` — the same pair with both sessions on one `user_compute_stream`.

The penalty grows with the captured graph's size. Measured on an H200 with
stock ORT 1.28 (CUDA 13.0, driver 580.105), one process per configuration:

    depth  320, width 4096:  alone 2.951  split 2.975 (+0.024)  shared 3.019
    depth 2500, width 1024:  alone 8.508  split 8.634 (+0.126)  shared 8.543

and on the real Muse-Glimmer 30B decoder island (2499 nodes, 15.4 GiB of INT4
weights, full-capacity shared KV, one ORT-managed CUDA graph):

    alone 15.40   split 16.11 (+0.71)   shared 15.44 (+0.04)

The penalty does not depend on whether the second session captures a graph, on
the extra session's weight residency, or on per-step synchronization: only on
the two sessions being on separate ORT streams.

Usage:

    python scripts/ort_multi_session_stream_probe.py [--depth 320] [--width 4096]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import time

import numpy as np

os.environ.setdefault("ORT_ENABLE_CUDNN_FLASH_ATTENTION", "0")

import onnxruntime as ort  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

CUDART = ctypes.CDLL("libcudart.so")
CUDA_STREAM_NON_BLOCKING = 1


def heavy_model(depth: int, width: int) -> bytes:
    """A decode-shaped graph: one row through a deep chain of square MatMuls."""
    nodes = []
    initializers = []
    current = "x"
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((width, width)).astype(np.float16) * 0.01
    initializers.append(
        helper.make_tensor("w", TensorProto.FLOAT16, [width, width], weight.tobytes(), raw=True)
    )
    for layer in range(depth):
        nodes.append(helper.make_node("MatMul", [current, "w"], [f"h{layer}"]))
        current = f"h{layer}"
    graph = helper.make_graph(
        nodes,
        "heavy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, width])],
        [helper.make_tensor_value_info(current, TensorProto.FLOAT16, [1, width])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model.SerializeToString(), current


def tiny_model() -> bytes:
    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["c"])],
        "tiny",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1024]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1024]),
        ],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [1024])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model.SerializeToString()


def session(model: bytes, stream: int | None) -> ort.InferenceSession:
    provider: dict[str, str | int] = {"device_id": 0, "enable_cuda_graph": "1"}
    if stream is not None:
        provider["user_compute_stream"] = str(stream)
    return ort.InferenceSession(
        model,
        ort.SessionOptions(),
        providers=[("CUDAExecutionProvider", provider), "CPUExecutionProvider"],
    )


def measure(label: str, step, iterations: int, samples: int) -> float:
    for _ in range(5):
        step()
    CUDART.cudaDeviceSynchronize()
    timings = []
    for _ in range(samples):
        started = time.perf_counter()
        for _ in range(iterations):
            step()
        CUDART.cudaDeviceSynchronize()
        timings.append((time.perf_counter() - started) / iterations * 1000)
    median = statistics.median(timings)
    print(f"{label}: median={median:.4f} ms")
    return median


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=320)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["alone", "split", "shared"],
        help="Run one configuration only. Each configuration must run in its own "
        "process: sessions created for an earlier configuration keep influencing "
        "later ones.",
    )
    args = parser.parse_args()

    handle = ctypes.c_void_p()
    if CUDART.cudaStreamCreateWithFlags(ctypes.byref(handle), CUDA_STREAM_NON_BLOCKING) != 0:
        raise RuntimeError("cudaStreamCreateWithFlags failed")
    shared_stream = handle.value

    modes = [args.mode] if args.mode else ["alone", "split", "shared"]
    if not args.mode:
        print(
            "note: comparable numbers need one process per mode; "
            "rerun with --mode alone|split|shared"
        )
    heavy_bytes, heavy_output = heavy_model(args.depth, args.width)
    for mode in modes:
        stream = shared_stream if mode == "shared" else None
        heavy = session(heavy_bytes, stream)
        x = ort.OrtValue.ortvalue_from_numpy(
            np.zeros((1, args.width), np.float16), "cuda", 0
        )
        y = ort.OrtValue.ortvalue_from_numpy(
            np.zeros((1, args.width), np.float16), "cuda", 0
        )
        heavy_binding = heavy.io_binding()
        heavy_binding.bind_ortvalue_input("x", x)
        heavy_binding.bind_ortvalue_output(heavy_output, y)
        if mode == "alone":
            measure(mode, lambda: heavy.run_with_iobinding(heavy_binding), args.iterations, args.samples)
            continue
        tiny = session(tiny_model(), stream)
        a = ort.OrtValue.ortvalue_from_numpy(np.zeros(1024, np.float32), "cuda", 0)
        b = ort.OrtValue.ortvalue_from_numpy(np.zeros(1024, np.float32), "cuda", 0)
        c = ort.OrtValue.ortvalue_from_numpy(np.zeros(1024, np.float32), "cuda", 0)
        tiny_binding = tiny.io_binding()
        tiny_binding.bind_ortvalue_input("a", a)
        tiny_binding.bind_ortvalue_input("b", b)
        tiny_binding.bind_ortvalue_output("c", c)

        def step() -> None:
            heavy.run_with_iobinding(heavy_binding)
            tiny.run_with_iobinding(tiny_binding)

        measure(mode, step, args.iterations, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
