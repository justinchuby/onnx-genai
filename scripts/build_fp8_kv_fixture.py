"""Build the tiny FP8 KV fixtures used by the ORT capability tests.

Two graphs, both trivial, each pinning one half of the FP8 story:

* ``identity`` carries an FP8 cache tensor across the session boundary, so a
  session that loads it proves the runtime can hold and bind FP8 state.
* ``scatter`` writes into that cache with ``ScatterND``, the op a fixed-capacity
  cache is updated with. Whether it loads depends entirely on whether the
  execution provider registered an FP8 kernel for it, which is the distinction
  the tests exist to make visible.

Run from the repository root: ``python scripts/build_fp8_kv_fixture.py``.
"""

from __future__ import annotations

import pathlib

import onnx
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "tiny-fp8-kv"

BATCH, CAPACITY, KV_DIM = 2, 4, 8


def cache(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(
        name, TensorProto.FLOAT8E4M3FN, [BATCH, CAPACITY, KV_DIM]
    )


def write(model: onnx.ModelProto, stem: str) -> None:
    model.ir_version = 11
    onnx.checker.check_model(model)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}.onnx.textproto").write_text(str(model))


def identity_graph() -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["key_cache"], ["updated_key_cache"])],
        "fp8_kv_identity",
        [cache("key_cache")],
        [cache("updated_key_cache")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])


def scatter_graph() -> onnx.ModelProto:
    """Scatter one FP8 row into the cache at a runtime-chosen slot.

    ``write_indices`` is [B, 1] so ScatterND addresses the (batch, slot) pair,
    which is the shape a per-row write cursor has in a real static cache.
    """
    graph = helper.make_graph(
        [
            helper.make_node(
                "ScatterND",
                ["key_cache", "write_indices", "new_key"],
                ["updated_key_cache"],
            )
        ],
        "fp8_kv_scatter",
        [
            cache("key_cache"),
            helper.make_tensor_value_info(
                "write_indices", TensorProto.INT64, [BATCH, 1, 2]
            ),
            helper.make_tensor_value_info(
                "new_key", TensorProto.FLOAT8E4M3FN, [BATCH, 1, KV_DIM]
            ),
        ],
        [cache("updated_key_cache")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])


if __name__ == "__main__":
    write(identity_graph(), "identity")
    write(scatter_graph(), "scatter")
    print(f"wrote fixtures to {OUT}")
