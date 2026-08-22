#!/usr/bin/env python3
"""Generate the executable `static_cache` workflow conformance fixture.

    python scripts/build_static_cache_workflow.py

Unlike the other packages under `tests/fixtures/onnx_genai_workflows/`, this one
is authored in this repository rather than exported by Mobius: it exists to pin
the *runtime* contract for fixed-capacity ("static") KV cache state, which no
producer emits yet.

The package is deliberately not a text generator. It is the smallest workflow
that can be wrong in the ways a static cache can be wrong:

* `model.onnx` scatters one position per row into a fixed-capacity cache at a
  destination it reads from `write_indices`, then produces logits by pooling the
  *valid prefix* named by `cache_lengths`. So a write that lands in the wrong
  slot, or a valid length that disagrees with the writes, changes the output.
  Nothing here can pass by accident.
* `key_cache` and `value_cache` receive different updates (`e` and `e*e`), so a
  scatter that confuses the two caches is observable.
* The loop advances `write_indices` for every row but advances `cache_lengths`
  only for active rows. That is what "inactive rows are preserved" means for a
  fixed-capacity buffer: the valid prefix is frozen while the slots above it
  become free for replacement.

The weights are tiny, deterministic, and meaningless.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 24
IR_VERSION = 11

VOCAB = 8
WIDTH = 4
SEED = 20260712


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _const(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name)


def _save(model: onnx.ModelProto, path: Path) -> None:
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    print(f"wrote {path.relative_to(Path.cwd())} ({path.stat().st_size} bytes)")


def _model(graph: onnx.GraphProto) -> onnx.ModelProto:
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def build_decoder() -> onnx.ModelProto:
    """One scatter step against a fixed-capacity cache.

    inputs
      input_ids     [batch, 1]                 i64
      key_cache     [batch, capacity, WIDTH]   f32
      value_cache   [batch, capacity, WIDTH]   f32
      write_indices [batch]                    i64  destination slot per row
      cache_lengths [batch]                    i64  valid prefix length per row
    outputs
      logits              [batch, 1, VOCAB]    f32
      updated_key_cache   [batch, capacity, WIDTH] f32
      updated_value_cache [batch, capacity, WIDTH] f32
    """
    rng = np.random.default_rng(SEED)
    embedding = (rng.standard_normal((VOCAB, WIDTH)) * 0.5).astype(np.float32)
    projection = (rng.standard_normal((WIDTH, VOCAB)) * 0.5).astype(np.float32)

    inputs = [
        _vi("input_ids", TensorProto.INT64, ["batch", 1]),
        _vi("key_cache", TensorProto.FLOAT, ["batch", "cache_capacity", WIDTH]),
        _vi("value_cache", TensorProto.FLOAT, ["batch", "cache_capacity", WIDTH]),
        _vi("write_indices", TensorProto.INT64, ["batch"]),
        _vi("cache_lengths", TensorProto.INT64, ["batch"]),
    ]
    outputs = [
        _vi("logits", TensorProto.FLOAT, ["batch", 1, VOCAB]),
        _vi(
            "updated_key_cache", TensorProto.FLOAT, ["batch", "cache_capacity", WIDTH]
        ),
        _vi(
            "updated_value_cache", TensorProto.FLOAT, ["batch", "cache_capacity", WIDTH]
        ),
    ]
    initializers = [
        _const("embedding", embedding),
        _const("projection", projection),
        _const("zero_1d", np.array([0], dtype=np.int64)),
        _const("one_1d", np.array([1], dtype=np.int64)),
        _const("two_1d", np.array([2], dtype=np.int64)),
        _const("scalar_zero", np.array(0, dtype=np.int64)),
        _const("scalar_one", np.array(1, dtype=np.int64)),
    ]

    nodes = [
        # (batch, 1) -> (batch,) -> (batch, WIDTH) embedding of this step's token.
        helper.make_node("Squeeze", ["input_ids", "one_1d"], ["token"]),
        helper.make_node("Gather", ["embedding", "token"], ["key_update"], axis=0),
        # The value update is not proportional to the key update, so a scatter
        # that writes the same tensor into both caches is detectable.
        helper.make_node("Mul", ["key_update", "key_update"], ["value_update"]),
        # ScatterND destinations: (batch, 2) rows of [row, write_index].
        helper.make_node("Shape", ["input_ids"], ["ids_shape"], start=0, end=1),
        helper.make_node("Squeeze", ["ids_shape", "zero_1d"], ["batch_scalar"]),
        helper.make_node(
            "Range", ["scalar_zero", "batch_scalar", "scalar_one"], ["rows"]
        ),
        helper.make_node("Unsqueeze", ["rows", "one_1d"], ["rows_2d"]),
        helper.make_node("Unsqueeze", ["write_indices", "one_1d"], ["writes_2d"]),
        helper.make_node("Concat", ["rows_2d", "writes_2d"], ["scatter_indices"], axis=1),
        helper.make_node(
            "ScatterND",
            ["key_cache", "scatter_indices", "key_update"],
            ["updated_key_cache"],
        ),
        helper.make_node(
            "ScatterND",
            ["value_cache", "scatter_indices", "value_update"],
            ["updated_value_cache"],
        ),
        # Valid-prefix mask: slot < cache_lengths[row]. The capacity comes from
        # the buffer itself, so the mask cannot silently disagree with it.
        helper.make_node("Shape", ["key_cache"], ["cap_shape"], start=1, end=2),
        helper.make_node("Squeeze", ["cap_shape", "zero_1d"], ["capacity_scalar"]),
        helper.make_node(
            "Range", ["scalar_zero", "capacity_scalar", "scalar_one"], ["slots"]
        ),
        helper.make_node("Unsqueeze", ["slots", "zero_1d"], ["slots_2d"]),
        helper.make_node("Unsqueeze", ["cache_lengths", "one_1d"], ["lengths_2d"]),
        helper.make_node("Less", ["slots_2d", "lengths_2d"], ["valid"]),
        helper.make_node("Cast", ["valid"], ["valid_f32"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["valid_f32", "two_1d"], ["valid_mask"]),
        # Pool the valid prefix of both caches: the output depends on every
        # position written so far, so a wrong destination changes the logits.
        helper.make_node("Mul", ["updated_key_cache", "valid_mask"], ["masked_key"]),
        helper.make_node("Mul", ["updated_value_cache", "valid_mask"], ["masked_value"]),
        helper.make_node("ReduceSum", ["masked_key", "one_1d"], ["pooled_key"], keepdims=0),
        helper.make_node(
            "ReduceSum", ["masked_value", "one_1d"], ["pooled_value"], keepdims=0
        ),
        helper.make_node("Add", ["pooled_key", "pooled_value"], ["pooled"]),
        helper.make_node("MatMul", ["pooled", "projection"], ["flat_logits"]),
        helper.make_node("Unsqueeze", ["flat_logits", "one_1d"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, "static_cache_decoder", inputs, outputs, initializer=initializers
    )
    return _model(graph)


def build_state_init() -> onnx.ModelProto:
    """Zeroed cache buffers plus the derived starting valid lengths.

    The buffers are produced by the workflow so the fixture is self-contained;
    a deployment is free to hand the same contract a pre-allocated arena, which
    is exactly the distinction the metadata keeps: capacity is declared, the
    physical buffer is the runtime's.
    """
    inputs = [
        _vi("input_ids", TensorProto.INT64, ["batch", "prompt_sequence"]),
        _vi("write_indices", TensorProto.INT64, ["batch"]),
        _vi("capacity", TensorProto.INT64, [1]),
    ]
    outputs = [
        _vi("key_cache", TensorProto.FLOAT, ["batch", "cache_capacity", WIDTH]),
        _vi("value_cache", TensorProto.FLOAT, ["batch", "cache_capacity", WIDTH]),
        _vi("cache_lengths", TensorProto.INT64, ["batch"]),
        _vi("token", TensorProto.INT64, ["batch", 1]),
    ]
    initializers = [
        _const("one_1d", np.array([1], dtype=np.int64)),
        _const("width_1d", np.array([WIDTH], dtype=np.int64)),
        _const("last_start", np.array([-1], dtype=np.int64)),
        _const("last_end", np.array([np.iinfo(np.int64).max], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Shape", ["input_ids"], ["batch_1d"], start=0, end=1),
        helper.make_node(
            "Concat", ["batch_1d", "capacity", "width_1d"], ["cache_shape"], axis=0
        ),
        helper.make_node(
            "ConstantOfShape",
            ["cache_shape"],
            ["key_cache"],
            value=numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero"),
        ),
        helper.make_node(
            "ConstantOfShape",
            ["cache_shape"],
            ["value_cache"],
            value=numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero"),
        ),
        # The prefill step writes at `write_indices`, so exactly that many slots
        # plus one are valid once it has run.
        helper.make_node("Add", ["write_indices", "one_1d"], ["raw_lengths"]),
        helper.make_node("Min", ["raw_lengths", "capacity"], ["cache_lengths"]),
        helper.make_node(
            "Slice",
            ["input_ids", "last_start", "last_end", "one_1d"],
            ["token"],
        ),
    ]
    graph = helper.make_graph(
        nodes, "static_cache_state_init", inputs, outputs, initializer=initializers
    )
    return _model(graph)


def build_step_update() -> onnx.ModelProto:
    """Advance the write cursor, the valid lengths, and the active mask.

    Every row advances its cursor; only active rows advance their valid length.
    An inactive row therefore keeps writing above its own frozen prefix, which
    is what makes those slots reusable without disturbing what the row still
    holds.
    """
    inputs = [
        _vi("write_indices", TensorProto.INT64, ["batch"]),
        _vi("cache_lengths", TensorProto.INT64, ["batch"]),
        _vi("capacity", TensorProto.INT64, [1]),
        _vi("active", TensorProto.BOOL, ["batch"]),
    ]
    outputs = [
        _vi("next_write_indices", TensorProto.INT64, ["batch"]),
        _vi("next_cache_lengths", TensorProto.INT64, ["batch"]),
        _vi("next_active", TensorProto.BOOL, ["batch"]),
        _vi("next_done", TensorProto.BOOL, ["batch"]),
        _vi("accepted_len", TensorProto.INT64, ["batch"]),
    ]
    initializers = [_const("one_1d", np.array([1], dtype=np.int64))]
    nodes = [
        helper.make_node("Sub", ["capacity", "one_1d"], ["last_slot"]),
        helper.make_node("Add", ["write_indices", "one_1d"], ["raw_write"]),
        helper.make_node("Min", ["raw_write", "last_slot"], ["next_write_indices"]),
        helper.make_node("Add", ["cache_lengths", "one_1d"], ["raw_length"]),
        helper.make_node("Min", ["raw_length", "capacity"], ["grown_length"]),
        helper.make_node(
            "Where", ["active", "grown_length", "cache_lengths"], ["next_cache_lengths"]
        ),
        helper.make_node("Less", ["raw_length", "capacity"], ["has_room"]),
        helper.make_node("And", ["active", "has_room"], ["next_active"]),
        helper.make_node("Not", ["next_active"], ["next_done"]),
        # One position per active row is admitted per step.
        helper.make_node("Cast", ["active"], ["accepted_len"], to=TensorProto.INT64),
    ]
    graph = helper.make_graph(
        nodes, "static_cache_step_update", inputs, outputs, initializer=initializers
    )
    return _model(graph)


def build_next_token() -> onnx.ModelProto:
    inputs = [_vi("logits", TensorProto.FLOAT, ["batch", 1, VOCAB])]
    outputs = [_vi("token", TensorProto.INT64, ["batch", 1])]
    nodes = [
        helper.make_node("ArgMax", ["logits"], ["token"], axis=2, keepdims=0),
    ]
    graph = helper.make_graph(nodes, "static_cache_next_token", inputs, outputs)
    return _model(graph)


def build_loop_continue() -> onnx.ModelProto:
    inputs = [_vi("active", TensorProto.BOOL, ["batch"])]
    outputs = [_vi("continue", TensorProto.BOOL, [1])]
    initializers = [_const("zero_1d", np.array([0], dtype=np.int64))]
    nodes = [
        helper.make_node("Cast", ["active"], ["active_i32"], to=TensorProto.INT32),
        helper.make_node("ReduceMax", ["active_i32", "zero_1d"], ["any_i32"], keepdims=1),
        helper.make_node("Cast", ["any_i32"], ["continue"], to=TensorProto.BOOL),
    ]
    graph = helper.make_graph(
        nodes, "static_cache_loop_continue", inputs, outputs, initializer=initializers
    )
    return _model(graph)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/fixtures/onnx_genai_workflows/static_cache",
    )
    args = parser.parse_args()
    out = args.out
    _save(build_decoder(), out / "model.onnx")
    _save(build_state_init(), out / "policies/state_init.onnx")
    _save(build_step_update(), out / "policies/step_update.onnx")
    _save(build_next_token(), out / "policies/next_token.onnx")
    _save(build_loop_continue(), out / "policies/loop_continue.onnx")


if __name__ == "__main__":
    main()
