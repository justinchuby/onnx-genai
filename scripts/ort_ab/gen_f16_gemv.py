#!/usr/bin/env python3
"""Generate decode-shaped (`M == 1`) f16 `MatMul` cells.

These reach `kernels::half_gemv::gemv_f16_kn`, the x86 f16 GEMV that serves an
f16 weight in its stored `[K, N]` order. `gen_gemm.py` covers block-quantised
and f32 dense GEMM but has no f16 cell, so the one kernel whose whole design
premise is "at M=1 this is memory-bound" had nothing measuring whether it
actually reaches memory speed.

The sweep is over the *weight working set*, which is the only thing that
matters for a kernel that reads each weight exactly once: 0.5 MB sits in one
core's L2, 8.4 and 25.7 MB in L3, and 134 MB is far past it. Comparing achieved
GB/s across that range separates "bound by memory" from "bound by the kernel's
own instruction stream" -- if the L2-resident cell runs at the same GB/s as the
DRAM-resident one, memory is not the limit. Pair with `roofline_bandwidth` for
the host's ceiling.

SYNTHETIC DATA NOTICE: no trained weights. Only the shapes are model-like
(3584 is Qwen3-8B's hidden size); `B` is a deterministic RNG draw and the
activations come from the benchmark harness's own synthetic pattern, fed
identically to both runtimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17

# name -> (k, n). Weight bytes are 2 * k * n; the comment is that size.
CELLS = {
    "l2_512": (512, 512),  # 0.5 MB, inside one core's L2
    "l2_1024": (1024, 1024),  # 2.1 MB
    "l3_2048": (2048, 2048),  # 8.4 MB
    "l3_3584": (3584, 3584),  # 25.7 MB, Qwen3-8B hidden
    "dram_8192": (8192, 8192),  # 134.2 MB, past any LLC here
}


def build(path: Path, *, k: int, n: int) -> None:
    rng = np.random.default_rng(0x5EBA7)
    # Scaled to 0.1 so a k-long f32 accumulation of f16 products stays well
    # inside f16 range when the output is narrowed back, keeping the harness's
    # parity check meaningful rather than saturated.
    b = ((rng.random((k, n), dtype=np.float32) - 0.5) * 0.1).astype(np.float16)
    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"], name="matmul")
    graph = helper.make_graph(
        [node],
        "matmul_f16_gemv",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, n])],
        initializer=[numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.save(model, str(path))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, (k, n) in CELLS.items():
        path = out / f"gemv_f16_{name}.onnx"
        build(path, k=k, n=n)
        print(f"{path}  k={k} n={n} weight={2 * k * n / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "f16gemv",
    )
    main(ap.parse_args().out)
