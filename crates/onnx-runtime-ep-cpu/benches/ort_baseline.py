#!/usr/bin/env python3
"""Time the f32 shapes used by kernels.rs through ONNX Runtime CPU EP."""

import argparse
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def model(op_type, input_specs, output_spec, attributes=None):
    attributes = attributes or {}
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in input_specs
    ]
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_spec)
    node = helper.make_node(
        op_type, [name for name, _ in input_specs], ["Y"], **attributes
    )
    graph = helper.make_graph([node], f"{op_type}_baseline", inputs, [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def gather_model(data_shape, indices_shape):
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)
    indices = helper.make_tensor_value_info(
        "indices", TensorProto.INT64, indices_shape
    )
    output = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [*indices_shape, data_shape[1]]
    )
    graph = helper.make_graph(
        [helper.make_node("Gather", ["data", "indices"], ["Y"], axis=0)],
        "Gather_baseline",
        [data, indices],
        [output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def values(shape):
    data = np.arange(np.prod(shape), dtype=np.int64) % 251
    return ((data - 125).astype(np.float32) / 64.0).reshape(shape)


def run_case(name, onnx_model, feeds, warmup, iterations):
    session = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for _ in range(warmup):
        session.run(None, feeds)
    start = time.perf_counter_ns()
    for _ in range(iterations):
        session.run(None, feeds)
    elapsed = time.perf_counter_ns() - start
    print(f"{name:38} {elapsed / iterations / 1_000:12.3f} us")


def cases():
    for size, shape in [
        ("small", [1024]),
        ("medium", [256, 1024]),
        ("large", [1024, 4096]),
    ]:
        yield (
            f"add/{size}/f32",
            model("Add", [("A", shape), ("B", [shape[-1]])], shape),
            {"A": values(shape), "B": values([shape[-1]])},
        )
    for size, shape in [
        ("small", [32, 128]),
        ("medium", [128, 512]),
        ("large", [256, 1024]),
    ]:
        yield (
            f"reduce_mean/{size}/f32",
            model(
                "ReduceMean",
                [("X", shape)],
                [shape[0], 1],
                {"axes": [1], "keepdims": 1},
            ),
            {"X": values(shape)},
        )
    for size, rows, columns, count in [
        ("small", 4096, 128, 32),
        ("medium", 16384, 256, 128),
        ("large", 32768, 512, 256),
    ]:
        indices = (np.arange(count, dtype=np.int64) * 97) % rows
        yield (
            f"gather/{size}/f32",
            gather_model([rows, columns], [count]),
            {"data": values([rows, columns]), "indices": indices},
        )
    for size, m, k, n in [
        ("small", 1, 256, 256),
        ("medium", 32, 512, 512),
        ("large", 32, 1024, 1024),
    ]:
        yield (
            f"matmul/{size}/f32",
            model("MatMul", [("A", [m, k]), ("B", [k, n])], [m, n]),
            {"A": values([m, k]), "B": values([k, n])},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    print(f"onnxruntime {ort.__version__}; provider=CPUExecutionProvider")
    for name, onnx_model, feeds in cases():
        if args.filter in name:
            run_case(name, onnx_model, feeds, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
