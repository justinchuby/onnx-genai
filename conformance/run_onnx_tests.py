#!/usr/bin/env python3
"""Run a deterministic nxrt EP slice against cbourjau/onnx-tests generators."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from hypothesis import find, settings
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

MAGIC = b"NXRTCF01"
CPU_OPS = [
    "MatMul",
    "Add",
    "Relu",
    "Reshape",
    "Transpose",
    "Gather",
    "LayerNormalization",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Min",
    "Max",
    "Sqrt",
    "Erf",
    "Tanh",
    "Cast",
    "ReduceMean",
    "Softmax",
    "Shape",
    "Unsqueeze",
    "Expand",
    "Slice",
    "Constant",
    "Gemm",
]
UNSUPPORTED_OPS = ["Abs", "Conv", "Sigmoid"]


@dataclass
class Result:
    op: str
    source: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-tests",
        type=Path,
        default=Path(os.environ.get("ONNX_TESTS_DIR", "../onnx-tests")),
        help="external cbourjau/onnx-tests checkout",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("target/debug/examples/conformance_runner"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("target/ep-conformance"))
    parser.add_argument("--json", type=Path, help="optional result JSON output")
    return parser.parse_args()


def generated_cases(onnx_tests: Path) -> dict[str, Callable[[], onnx.ModelProto]]:
    sys.path.insert(0, str(onnx_tests.resolve()))
    import spox.opset.ai.onnx.v17 as op17
    from onnx_tests import (
        elementwise_ops,
        linear_algebra_ops,
        manipulation_functions,
        slice as slice_ops,
        statistical_ops,
    )

    f32 = np.dtype("float32")

    def draw(strategy):
        state = find(
            strategy,
            is_small_nonempty_case,
            settings=settings(
                database=None, derandomize=True, max_examples=500, deadline=None
            ),
        )
        return strip_identity_nodes(state.build_model())

    return {
        "MatMul": lambda: draw(linear_algebra_ops.matmul(f32, op17)),
        "Add": lambda: draw(elementwise_ops.add(f32, op17)),
        "Relu": lambda: draw(elementwise_ops.relu(f32, op17)),
        "Reshape": lambda: draw(manipulation_functions.reshape(f32, op17)),
        "Transpose": lambda: draw(manipulation_functions.transpose(f32, op17)),
        "Sub": lambda: draw(elementwise_ops.sub(f32, op17)),
        "Mul": lambda: draw(elementwise_ops.mul(f32, op17)),
        "Div": lambda: draw(elementwise_ops.div(f32, op17)),
        "Pow": lambda: draw(elementwise_ops.pow(f32, f32, op17)),
        "Min": lambda: draw(statistical_ops.min(f32, op17)),
        "Max": lambda: draw(statistical_ops.max(f32, op17)),
        "Sqrt": lambda: draw(elementwise_ops.sqrt(f32, op17)),
        "Erf": lambda: draw(elementwise_ops.erf(f32, op17)),
        "Tanh": lambda: draw(elementwise_ops.tanh(f32, op17)),
        "Cast": lambda: draw(
            manipulation_functions.cast(f32, np.dtype("int64"), op17)
        ),
        "Softmax": lambda: draw(elementwise_ops.softmax(f32, op17)),
        "Expand": lambda: draw(manipulation_functions.expand(f32, op17)),
        "Slice": lambda: draw(slice_ops.slice(f32, op17)),
        "Abs": lambda: draw(elementwise_ops.abs(f32, op17)),
        "Conv": lambda: manual_model("Conv"),
        "Sigmoid": lambda: draw(elementwise_ops.sigmoid(f32, op17)),
    }


def is_small_nonempty_case(state) -> bool:
    def arrays(value):
        if isinstance(value, np.ndarray):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from arrays(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from arrays(item)

    values = list(arrays(state.inputs))
    return all(
        value.size > 0
        and value.size <= 256
        and (value.dtype.kind != "f" or np.isfinite(value).all())
        for value in values
    )


def strip_identity_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove Spox's output-only Identity wrappers unsupported by Phase-1 nxrt."""
    replacements = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if node.op_type == "Identity" and len(node.input) == len(node.output) == 1
    }
    if not replacements:
        return model
    kept = [node for node in model.graph.node if node.op_type != "Identity"]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            while name in replacements:
                name = replacements[name]
            node.input[index] = name
    for output in model.graph.output:
        while output.name in replacements:
            output.name = replacements[output.name]
    return model


def manual_model(op: str) -> onnx.ModelProto:
    values: dict[str, np.ndarray] = {
        "LayerNormalization": np.array([[1.0, 2.0, 4.0]], dtype=np.float32),
        "Shape": np.zeros((2, 3, 4), dtype=np.float32),
        "Constant": np.array([1.25, -2.5], dtype=np.float32),
        "Gemm": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "Gather": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "ReduceMean": np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
        "Unsqueeze": np.array([[1.0, 2.0]], dtype=np.float32),
        "Conv": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3),
    }
    if op == "Constant":
        node = helper.make_node(
            "Constant", [], ["Y"], value=numpy_helper.from_array(values[op])
        )
        outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])]
        graph = helper.make_graph([node], op, [], outputs)
    elif op == "LayerNormalization":
        x = numpy_helper.from_array(values[op], "X")
        scale = numpy_helper.from_array(np.ones((3,), np.float32), "Scale")
        bias = numpy_helper.from_array(np.zeros((3,), np.float32), "B")
        node = helper.make_node(
            op, ["X", "Scale", "B"], ["Y"], axis=-1, epsilon=1e-5
        )
        graph = helper.make_graph(
            [node],
            op,
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
            [x, scale, bias],
        )
    elif op == "Shape":
        x = numpy_helper.from_array(values[op], "X")
        node = helper.make_node(op, ["X"], ["Y"])
        graph = helper.make_graph(
            [node],
            op,
            [],
            [helper.make_tensor_value_info("Y", TensorProto.INT64, [3])],
            [x],
        )
    elif op == "Gather":
        x = numpy_helper.from_array(values[op], "X")
        indices = numpy_helper.from_array(np.array([1, 0], np.int64), "Indices")
        node = helper.make_node(op, ["X", "Indices"], ["Y"], axis=0)
        graph = helper.make_graph(
            [node],
            op,
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
            [x, indices],
        )
    elif op == "Gemm":
        a = numpy_helper.from_array(values[op], "A")
        b = numpy_helper.from_array(np.array([[2.0], [-1.0]], np.float32), "B")
        c = numpy_helper.from_array(np.array([0.5], np.float32), "C")
        node = helper.make_node(op, ["A", "B", "C"], ["Y"])
        graph = helper.make_graph(
            [node],
            op,
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1])],
            [a, b, c],
        )
    elif op == "ReduceMean":
        x = numpy_helper.from_array(values[op], "X")
        node = helper.make_node(op, ["X"], ["Y"], axes=[1], keepdims=1)
        graph = helper.make_graph(
            [node],
            op,
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1])],
            [x],
        )
    elif op == "Unsqueeze":
        x = numpy_helper.from_array(values[op], "X")
        node = helper.make_node(op, ["X"], ["Y"], axes=[0])
        graph = helper.make_graph(
            [node],
            op,
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2])],
            [x],
        )
    elif op == "Conv":
        x = numpy_helper.from_array(values[op], "X")
        w = numpy_helper.from_array(np.ones((1, 1, 2, 2), np.float32), "W")
        node = helper.make_node(op, ["X", "W"], ["Y"])
        graph = helper.make_graph(
            [node],
            op,
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])],
            [x, w],
        )
    else:
        raise KeyError(op)
    opset = 12 if op in {"ReduceMean", "Unsqueeze"} else 17
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=9)


def write_tensor(path: Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    dtype_code = helper.np_dtype_to_tensor_dtype(array.dtype)
    payload = array.tobytes()
    with path.open("wb") as file:
        file.write(MAGIC)
        file.write(bytes([dtype_code]))
        file.write(struct.pack("<I", array.ndim))
        file.write(struct.pack(f"<{array.ndim}Q", *array.shape))
        file.write(struct.pack("<Q", len(payload)))
        file.write(payload)


def read_tensor(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        if file.read(8) != MAGIC:
            raise ValueError("bad nxrt tensor magic")
        dtype_code = file.read(1)[0]
        rank = struct.unpack("<I", file.read(4))[0]
        shape = struct.unpack(f"<{rank}Q", file.read(8 * rank))
        length = struct.unpack("<Q", file.read(8))[0]
        payload = file.read(length)
    dtype = helper.tensor_dtype_to_np_dtype(dtype_code)
    return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()


def run_case(
    op: str, source: str, model: onnx.ModelProto, runner: Path, work_dir: Path
) -> Result:
    case_dir = work_dir / op
    shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True)
    model_path = case_dir / "model.onnx"
    model_path.write_bytes(model.SerializeToString())
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reference = ReferenceEvaluator(model, optimized=False).run(None, {})
    except Exception as err:
        return Result(op, source, "ERROR", f"reference failed: {err}")

    process = subprocess.run(
        [str(runner.resolve()), str(model_path.resolve()), str(case_dir.resolve())],
        text=True,
        capture_output=True,
    )
    status_line = process.stdout.strip() or process.stderr.strip()
    if process.returncode == 2 and status_line.startswith("UNSUPPORTED_OP\t"):
        return Result(op, source, "UNSUPPORTED", status_line.split("\t", 1)[1])
    if process.returncode != 0:
        return Result(
            op,
            source,
            "ERROR",
            f"exit {process.returncode}: {status_line or 'no diagnostic'}",
        )

    actual = [read_tensor(case_dir / f"output_{i}.nxrt") for i in range(len(reference))]
    try:
        for got, expected in zip(actual, reference):
            if np.issubdtype(expected.dtype, np.floating):
                np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
            else:
                np.testing.assert_array_equal(got, expected)
    except AssertionError as err:
        return Result(op, source, "MISMATCH", str(err).splitlines()[0])
    return Result(op, source, "PASS", f"{len(reference)} output(s)")


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    generators = generated_cases(args.onnx_tests)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for op in CPU_OPS + UNSUPPORTED_OPS:
        source = "onnx-tests" if op in generators and op != "Conv" else "focused ONNX"
        try:
            model = generators[op]() if op in generators else manual_model(op)
            result = run_case(op, source, model, args.runner, args.work_dir)
        except Exception as err:
            result = Result(op, source, "ERROR", f"driver/generator failed: {err}")
        results.append(result)
        print(f"{result.status:11} {op:22} {result.detail}")

    counts = {
        status: sum(result.status == status for result in results)
        for status in ("PASS", "UNSUPPORTED", "MISMATCH", "ERROR")
    }
    print("SUMMARY " + " ".join(f"{key}={value}" for key, value in counts.items()))
    if args.json:
        args.json.write_text(
            json.dumps(
                {"counts": counts, "results": [vars(result) for result in results]},
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
