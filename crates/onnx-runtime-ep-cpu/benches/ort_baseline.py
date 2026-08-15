#!/usr/bin/env python3
"""Time the f32 shapes used by kernels.rs through ONNX Runtime CPU EP."""

import argparse
import statistics
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


def run_case(name, onnx_model, feeds, warmup, iterations, repetitions, threads):
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = threads
    # Each model contains one node, so inter-op parallelism is intentionally
    # disabled while intra-op workers are matched to the Rust Rayon pool.
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    for _ in range(warmup):
        session.run(None, feeds)
    samples_us = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            session.run(None, feeds)
        elapsed = time.perf_counter_ns() - start
        samples_us.append(elapsed / iterations / 1_000)
    print(
        f"{name:38} threads={threads} "
        f"(intra_op={session_options.intra_op_num_threads}, "
        f"inter_op={session_options.inter_op_num_threads}) "
        f"{statistics.median(samples_us):12.3f} us median "
        f"({repetitions}x{iterations} runs)"
    )


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
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="timed batches whose per-call times are reduced by median (default: 1)",
    )
    parser.add_argument("--filter", default="")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 8],
        help="intra-op thread counts to benchmark (default: 1 8)",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    print(f"onnxruntime {ort.__version__}; provider=CPUExecutionProvider")
    for threads in args.threads:
        if threads < 1:
            parser.error("--threads values must be positive")
        print(f"thread configuration: intra_op={threads}, inter_op=1")
        for name, onnx_model, feeds in cases():
            if args.filter in name:
                run_case(
                    name,
                    onnx_model,
                    feeds,
                    args.warmup,
                    args.iterations,
                    args.repetitions,
                    threads,
                )


if __name__ == "__main__":
    main()
