#!/usr/bin/env python3
"""f16 `Gemm` prefill cells that isolate **transposed `B`** as the only variable.

`docs/performance/CPU_MATMUL_ASSIGNMENT.md` records `Gemm` f16 with `transB = 1`
at `M = 128` as one of the worst remaining CPU cells -- and, unusually, one that
gets *worse* as threads are added rather than better. Every other bad cell in
that table is an `M = 1` GEMV that flattens against a shared ceiling, so the
usual explanations (task granularity, fork/join, memory bandwidth) do not
transfer.

The point of this generator is that each shape is emitted **twice**: once with
`transB = 1` and once with `transB = 0` over an already-transposed initializer.
Both spellings compute the identical mathematical product from the identical
numbers, and both land in the same blocked half GEMM. They differ only in the
`MatrixLayout` that `half_gemm::pack_b` receives. So the NN cell is a true
control: whatever it costs is the cost of the GEMM, and the NT-minus-NN
difference is the cost of the *layout* alone. A ratio against ORT cannot
separate those two things; this pairing can.

`M` is swept because the suspected cost is paid per row-block rather than per
row, which predicts a gap that grows with `M` and with thread count. `M = 1` is
included because it is the cell #1417 addresses: as of `e13460af6` it still
falls into this same blocked GEMM and so shows the layout penalty too. #1417
routes `M = 1` to `half_gemv` instead, which is the better fix for that row, so
`M = 1` is reported separately rather than folded into the prefill matrix.

SYNTHETIC DATA NOTICE: no trained weights. Only the hidden sizes come from
public model configs (Qwen3-8B 3584, Llama-3-8B 4096, Qwen3-0.6B 1024); tensor
contents are a deterministic PRNG pattern fed identically to both runtimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17

# (name, k, n) from public configs. `Gemm` transB=1 is the shape a fused QKV
# projection takes when the weight is stored output-major.
SHAPES = {
    "qwen3_8b": (3584, 3584),
    "llama3_8b": (4096, 4096),
    "qwen3_0p6b": (1024, 1024),
}

# 1 is a negative control (routes to the GEMV, not this GEMM); the rest walk
# prefill up through the caches.
ROWS = (1, 32, 128, 512)


def _save(graph: onnx.GraphProto, path: Path) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.save(model, str(path))


def build_gemm_f16(path: Path, *, m: int, k: int, n: int, trans_b: bool) -> None:
    """One f16 `Gemm`. `trans_b` selects the storage order of `B` only.

    With `transB = 1` the initializer is `[n, k]`; with `transB = 0` it is the
    transpose of that same array, `[k, n]`. The product is identical, so the two
    files differ only in how `B` must be walked to pack it.
    """
    rng = np.random.default_rng(0x5EBA6)
    stored = (rng.random((n, k), dtype=np.float32) - 0.5).astype(np.float16)
    b = stored if trans_b else np.ascontiguousarray(stored.T)
    c = np.zeros((n,), dtype=np.float16)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gemm",
                ["A", "B", "C"],
                ["Y"],
                name="gemm",
                transB=1 if trans_b else 0,
            )
        ],
        "gemm_f16_nt" if trans_b else "gemm_f16_nn",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [m, n])],
        initializer=[
            numpy_helper.from_array(b, "B"),
            numpy_helper.from_array(c, "C"),
        ],
    )
    _save(graph, path)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, (k, n) in SHAPES.items():
        for m in ROWS:
            for trans_b in (True, False):
                tag = "nt" if trans_b else "nn"
                path = out / f"f16gemm_{tag}_{name}_m{m}.onnx"
                build_gemm_f16(path, m=m, k=k, n=n, trans_b=trans_b)
                made.append(path)
    for p in made:
        print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "f16nt",
    )
    main(ap.parse_args().out)
