#!/usr/bin/env python3
"""Generate single-node GEMM benchmark cells: block-quantised
`com.microsoft::MatMulNBits` and dense `MatMul`.

These are the two operators that carry a decode step's arithmetic, and they are
also the largest remaining raw-Rayon fan-outs in the CPU EP. They are generated
here so the task-runtime A/B can show what it does *not* touch as well as what
it does.

SYNTHETIC DATA NOTICE: no trained weights. Only the hidden/intermediate sizes
come from public model configs; the packed `B` matrix and the scales are
deterministic byte patterns, and the activations come from the benchmark
harness's own synthetic pattern, fed identically to both runtimes.

`MatMulNBits` slot order (com.microsoft):
  0 A  1 B  2 scales  3 zero_points  4 g_idx  5 bias
`B` is `[n, n_blocks_per_col, blob_size]` uint8 with `blob_size = block_size/2`
for 4-bit, and `scales` is `[n * n_blocks_per_col]`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17

# (name, k, n) -- one projection each from a few public configs.
NBITS_SHAPES = {
    "qwen3_0p6b_qkv": (1024, 3072),
    "qwen3_0p6b_mlp": (1024, 6144),
    "llama3_8b_qkv": (4096, 6144),
    "llama3_8b_mlp": (4096, 14336),
}

# Token counts: 1 = decode, the rest walk prefill up through the caches.
NBITS_TOKENS = (1, 8, 128, 512)

# Dense f32 MatMul cells, [m, k] x [k, n].
DENSE_SHAPES = {
    "sq_512": (512, 512, 512),
    "sq_1024": (1024, 1024, 1024),
    "tall_128x4096": (128, 4096, 4096),
    "decode_1x4096": (1, 4096, 4096),
}

BLOCK_SIZE = 32


def build_matmul_nbits(path: Path, *, tokens: int, k: int, n: int) -> None:
    blocks = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
    blob = BLOCK_SIZE // 2

    rng = np.random.default_rng(0x5EBA5)
    b = rng.integers(0, 256, size=(n, blocks, blob), dtype=np.uint8)
    # Small positive scales keep the dequantised weights in a sane range so the
    # parity check compares meaningful numbers rather than saturated ones.
    scales = (rng.random((n * blocks,), dtype=np.float32) * 0.01 + 0.001).astype(np.float32)

    node = helper.make_node(
        "MatMulNBits",
        inputs=["A", "B", "scales"],
        outputs=["Y"],
        domain="com.microsoft",
        name="matmul_nbits",
        K=k,
        N=n,
        bits=4,
        block_size=BLOCK_SIZE,
        accuracy_level=0,
    )
    graph = helper.make_graph(
        [node],
        "matmul_nbits",
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


def build_matmul(path: Path, *, m: int, k: int, n: int) -> None:
    rng = np.random.default_rng(0x5EBA6)
    b = (rng.random((k, n), dtype=np.float32) - 0.5).astype(np.float32)
    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"], name="matmul")
    graph = helper.make_graph(
        [node],
        "matmul",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])],
        initializer=[numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.save(model, str(path))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, (k, n) in NBITS_SHAPES.items():
        for tokens in NBITS_TOKENS:
            path = out / f"gemm_nbits_{name}_t{tokens}.onnx"
            build_matmul_nbits(path, tokens=tokens, k=k, n=n)
            made.append(path)
    for name, (m, k, n) in DENSE_SHAPES.items():
        path = out / f"gemm_dense_{name}.onnx"
        build_matmul(path, m=m, k=k, n=n)
        made.append(path)
    for p in made:
        print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "models" / "gemm")
    main(ap.parse_args().out)
