#!/usr/bin/env python3
"""Generate single-node `QLinearMatMul` benchmark cells.

`QLinearMatMul` is the only integer GEMM in the matmul family, and it is the
one whose ledger rows were never reproducible on the shipped build:
`docs/performance/CPU_MATMUL_ASSIGNMENT.md` carries `u8 x u8` and `i8 x i8`
rows taken on a `--features mlas` research build, with a note saying the
default native build measured 11.8x-12.0x instead. Nothing in this directory
could re-measure either claim, so the native integer kernel (`qgemm_native.rs`)
landed against a microbenchmark rather than against ORT.

This emits the cells that close that hole: both signedness pairs, decode
(`M = 1`) and prefill, at production projection geometry, plus rows immediately
below and above the kernel's own `PARALLEL_MIN_WORK` fork threshold
(`m * n * k >= 1 << 16`) and its `m <= MR` (4) fused-versus-packed split, so
every dispatch gate inside the kernel has a cell on both sides of it.

SYNTHETIC DATA NOTICE: no trained weights. Only the hidden/intermediate sizes
come from public model configs; `B`, the scales and the zero points are
deterministic patterns, and the activations come from the benchmark harness's
own synthetic pattern, fed identically to both runtimes.

`QLinearMatMul` slot order (ai.onnx):
  0 A  1 a_scale  2 a_zero_point  3 B  4 b_scale  5 b_zero_point
  6 y_scale  7 y_zero_point
Only `A` is a graph input; everything else is an initializer, which is what an
exported quantized model looks like and what lets the EP pre-pack `B`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17

# (name, k, n) -- one projection each from a few public configs, matching the
# names `gen_gemm.py` already uses so the two sheets line up.
SHAPES = {
    "qwen3_0p6b_qkv": (1024, 3072),
    "llama3_8b_qkv": (4096, 6144),
    "llama3_8b_mlp": (4096, 14336),
    # The square geometry every `QLinearMatMul` row of the ledger is quoted at.
    "qwen3_8b_square": (3584, 3584),
}

# Rows chosen against the kernel's own gates rather than against a model:
#   m = 1  -- decode, and `fused` (m <= MR = 4) with one row block
#   m = 4  -- the last `fused` row count
#   m = 5  -- the first packed row count, so the pack/no-pack split is straddled
#   m = 128, 512 -- prefill, where the packed panel is re-read many times
TOKENS = (1, 4, 5, 128, 512)

# A `k * n` small enough that `m * n * k` lands under `PARALLEL_MIN_WORK`
# (1 << 16 = 65 536) at `m = 1` and `m = 4` and over it at `m = 8` and `m = 16`,
# so the serial/forked split has two cells on each side. 64 * 128 = 8192, and
# 8192 * 4 < 65536 <= 8192 * 8.
FORK_GATE = ("fork_gate", 64, 128)
FORK_GATE_TOKENS = (1, 4, 8, 16)


def _u8(count: int, seed: int) -> np.ndarray:
    return ((np.arange(count) * 37 + seed) % 251).astype(np.uint8)


def _i8(count: int, seed: int) -> np.ndarray:
    return (((np.arange(count) * 37 + seed) % 251) - 125).astype(np.int8)


def build(name: str, m: int, k: int, n: int, signed: bool) -> onnx.ModelProto:
    elem = TensorProto.INT8 if signed else TensorProto.UINT8
    weight = _i8(k * n, 7) if signed else _u8(k * n, 7)
    zero = np.int8(0) if signed else np.uint8(128)
    b_zero = np.int8(-3) if signed else np.uint8(127)
    y_zero = np.int8(2) if signed else np.uint8(130)

    initializers = [
        numpy_helper.from_array(weight.reshape(k, n), "B"),
        numpy_helper.from_array(np.float32(0.021), "a_scale"),
        numpy_helper.from_array(zero, "a_zero_point"),
        numpy_helper.from_array(np.float32(0.013), "b_scale"),
        numpy_helper.from_array(b_zero, "b_zero_point"),
        numpy_helper.from_array(np.float32(0.037), "y_scale"),
        numpy_helper.from_array(y_zero, "y_zero_point"),
    ]

    node = helper.make_node(
        "QLinearMatMul",
        [
            "A",
            "a_scale",
            "a_zero_point",
            "B",
            "b_scale",
            "b_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        ["Y"],
        name="qlinear_matmul",
    )
    graph = helper.make_graph(
        [node],
        name,
        [helper.make_tensor_value_info("A", elem, [m, k])],
        [helper.make_tensor_value_info("Y", elem, [m, n])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="gen_qlinear"
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="*",
        default=list(TOKENS),
        help="M values to emit at the model geometries.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cells = [
        (shape, k, n, m)
        for shape, (k, n) in SHAPES.items()
        for m in args.tokens
    ]
    name, k, n = FORK_GATE
    cells += [(name, k, n, m) for m in FORK_GATE_TOKENS]

    for shape, k, n, m in cells:
        for signed in (False, True):
            dtype = "i8" if signed else "u8"
            path = args.out / f"qlinear_{shape}_{dtype}_k{k}_n{n}_t{m}.onnx"
            onnx.save(build(path.stem, m, k, n, signed), path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
