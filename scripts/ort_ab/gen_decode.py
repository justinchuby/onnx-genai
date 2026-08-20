#!/usr/bin/env python3
"""Decode-shape (`M = 1`) matmul cells for the CPU work list, plus an `M = 128`
prefill control.

The rows this generates are the ones in `docs/performance/CPU_MATMUL_ASSIGNMENT.md`
that are at parity single-threaded and get monotonically *worse* as the thread
count rises -- `MatMul` f32/f16, `Gemm` f16 and `MatMulNBits` 4-bit, all at
`M = 1`. `gen_gemm.py` covers prefill geometries and 4-bit/8-bit `MatMulNBits`;
it has no f16 `MatMul` or `Gemm` at all, which is why these live here.

Every cell is `K = N = 3584` (Qwen3-8B hidden) so the whole sweep is one
geometry and the only variable is dtype, op and thread count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17

# Qwen3-8B hidden size, matching the assignment matrix's decode rows.
K = 3584
N = 3584
BLOCK_SIZE = 32


def _save(graph: onnx.GraphProto, path: Path) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.save(model, str(path))


def build_matmul_f32(path: Path, *, m: int, k: int, n: int) -> None:
    rng = np.random.default_rng(0x5EBA6)
    b = (rng.random((k, n), dtype=np.float32) - 0.5).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["Y"], name="matmul")],
        "matmul_f32",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])],
        initializer=[numpy_helper.from_array(b, "B")],
    )
    _save(graph, path)


def build_matmul_f16(path: Path, *, m: int, k: int, n: int) -> None:
    rng = np.random.default_rng(0x5EBA6)
    b = (rng.random((k, n), dtype=np.float32) - 0.5).astype(np.float16)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["Y"], name="matmul")],
        "matmul_f16",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [m, n])],
        initializer=[numpy_helper.from_array(b, "B")],
    )
    _save(graph, path)


def build_gemm_f16(path: Path, *, m: int, k: int, n: int) -> None:
    """`Gemm` with `B` transposed (`transB = 1`), the shape a fused QKV takes."""
    rng = np.random.default_rng(0x5EBA6)
    b = (rng.random((n, k), dtype=np.float32) - 0.5).astype(np.float16)
    c = np.zeros((n,), dtype=np.float16)
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["A", "B", "C"], ["Y"], name="gemm", transB=1)],
        "gemm_f16",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [m, n])],
        initializer=[
            numpy_helper.from_array(b, "B"),
            numpy_helper.from_array(c, "C"),
        ],
    )
    _save(graph, path)


def build_matmul_nbits(path: Path, *, tokens: int, k: int, n: int, bits: int) -> None:
    blocks = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
    values_per_byte = 8 // bits
    blob = BLOCK_SIZE // values_per_byte
    rng = np.random.default_rng(0xB175)
    b = rng.integers(0, 256, size=(n, blocks, blob), dtype=np.uint8)
    scales = (rng.random((n * blocks,), dtype=np.float32) * 0.01 + 0.001).astype(
        np.float32
    )
    node = helper.make_node(
        "MatMulNBits",
        inputs=["A", "B", "scales"],
        outputs=["Y"],
        name="matmulnbits",
        domain="com.microsoft",
        K=k,
        N=n,
        bits=bits,
        block_size=BLOCK_SIZE,
    )
    graph = helper.make_graph(
        [node],
        "matmulnbits",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, tokens, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, tokens, n])],
        initializer=[
            numpy_helper.from_array(b, "B"),
            numpy_helper.from_array(scales, "scales"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", OPSET),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.save(model, str(path))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    made = []

    for m in (1, 128):
        p = out / f"decode_matmul_f32_m{m}.onnx"
        build_matmul_f32(p, m=m, k=K, n=N)
        made.append(p)

        p = out / f"decode_matmul_f16_m{m}.onnx"
        build_matmul_f16(p, m=m, k=K, n=N)
        made.append(p)

        p = out / f"decode_gemm_f16_m{m}.onnx"
        build_gemm_f16(p, m=m, k=K, n=N)
        made.append(p)

        for bits in (4, 8):
            p = out / f"decode_nbits{bits}_m{m}.onnx"
            build_matmul_nbits(p, tokens=m, k=K, n=N, bits=bits)
            made.append(p)

    for p in made:
        print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("bench-models/decode"))
    args = ap.parse_args()
    main(args.out)
