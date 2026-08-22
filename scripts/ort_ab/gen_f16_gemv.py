#!/usr/bin/env python3
"""Generate decode-shaped (`M == 1`) f16 `MatMul` and `Gemm` cells.

These reach `kernels::half_gemv::gemv_half_kn`, the x86 half GEMV that serves
an f16/bf16 weight in its stored `[K, N]` order. `gen_gemm.py` covers
block-quantised and f32 dense GEMM but has no f16 cell, so the one kernel whose
whole design premise is "at M=1 this is memory-bound" had nothing measuring
whether it actually reaches memory speed.

`--op` picks which operator carries the cell, and the two are **not**
interchangeable, because they do not reach the GEMV over the same weight range:

* `matmul` routes to the GEMV only below `HALF_PREFILL_GEBP_MIN_WEIGHT`
  (1,048,576 elements). At or above it the fused widen-pack GEBP takes decode
  instead, so every cell here except `l2_512` needs
  `ONNX_GENAI_CPU_MM_HALF_GEBP=0` to reach the GEMV at all.
* `gemm` (with `transB=0`) has no weight gate, so it reaches the GEMV at every
  size in the shipped default configuration.

Use `matmul` with the GEBP switched off to measure the kernel, and `gemm` to
measure what a default build actually runs.

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


def build(path: Path, *, k: int, n: int, op: str) -> None:
    rng = np.random.default_rng(0x5EBA7)
    # Scaled to 0.1 so a k-long f32 accumulation of f16 products stays well
    # inside f16 range when the output is narrowed back, keeping the harness's
    # parity check meaningful rather than saturated.
    b = ((rng.random((k, n), dtype=np.float32) - 0.5) * 0.1).astype(np.float16)
    if op == "gemm":
        # transB is left at its 0 default: B stays in [K, N], which is the
        # order `gemv_half_kn` reads. transB=1 is a different kernel
        # (`gemv_f16_nk`) and a different measurement.
        node = helper.make_node(
            "Gemm", inputs=["A", "B"], outputs=["Y"], name="gemm", alpha=1.0, beta=1.0
        )
    else:
        node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"], name="matmul")
    graph = helper.make_graph(
        [node],
        f"{op}_f16_gemv",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, n])],
        initializer=[numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.save(model, str(path))


def main(out: Path, op: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    prefix = "gemv_f16" if op == "matmul" else f"gemv_f16_{op}"
    for name, (k, n) in CELLS.items():
        path = out / f"{prefix}_{name}.onnx"
        build(path, k=k, n=n, op=op)
        print(f"{path}  op={op} k={k} n={n} weight={2 * k * n / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "f16gemv",
    )
    ap.add_argument("--op", choices=("matmul", "gemm"), default="matmul")
    args = ap.parse_args()
    main(args.out, args.op)
