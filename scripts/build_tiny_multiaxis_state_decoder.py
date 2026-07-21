#!/usr/bin/env python3
"""Build the deterministic multi-axis position + loop-state decoder fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxscript import ir

VOCAB = 32
KV_LAYERS = (3, 11)


def tensor(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def initializer(name: str, value: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.Tensor(value, name=name))


def node(
    op_type: str,
    inputs: list[ir.Value],
    output: str,
    *,
    attributes: list[ir.Attr] | tuple[ir.Attr, ...] = (),
) -> ir.Node:
    return ir.Node("", op_type, inputs, attributes, outputs=[ir.Value(name=output)])


def save(model: ir.Model, path: Path) -> None:
    ir.save(model, path, format="textproto")
    onnx.checker.check_model(ir.to_proto(model))


def build_embedding(path: Path) -> None:
    input_ids = tensor("input_ids", ir.DataType.INT64, [1, "prompt_sequence"])
    table = initializer("embedding_table", np.zeros((VOCAB, 1), dtype=np.float32))
    embeds = node("Gather", [table, input_ids], "routed_sequence")
    embeds.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    embeds.outputs[0].shape = ir.Shape([1, "prompt_sequence", 1])
    graph = ir.Graph(
        [input_ids],
        [embeds.outputs[0]],
        nodes=[embeds],
        initializers=[table],
        opset_imports={"": 12},
        name="tiny_position_state_embedding",
    )
    save(ir.Model(graph, ir_version=8, producer_name="onnx-genai WP4 fixture"), path)


def build_decoder(path: Path) -> None:
    input_ids = tensor("input_ids", ir.DataType.INT64, [1, "sequence"])
    routed = tensor(
        "routed_sequence", ir.DataType.FLOAT, [1, "routed_sequence_length", 1]
    )
    attention_mask = tensor("attention_mask", ir.DataType.INT64, [1, "attended"])
    position_ids = tensor("position_ids", ir.DataType.INT64, [3, 1, "sequence"])
    state_a = tensor("state_a.in", ir.DataType.FLOAT, [1, 2])
    state_b = tensor("state_b.in", ir.DataType.FLOAT, [1, 1, 2])

    past_inputs: list[ir.Value] = []
    for layer in KV_LAYERS:
        past_inputs.extend(
            [
                tensor(
                    f"past.{layer}.key",
                    ir.DataType.FLOAT,
                    [1, 1, "past_sequence", 1],
                ),
                tensor(
                    f"past.{layer}.value",
                    ir.DataType.FLOAT,
                    [1, 1, "past_sequence", 1],
                ),
            ]
        )

    nodes: list[ir.Node] = []
    pos_sum = node(
        "ReduceSum",
        [position_ids],
        "position_sum_i64",
        attributes=[ir.AttrInt64s("axes", [0]), ir.AttrInt64("keepdims", 0)],
    )
    pos_float = node(
        "Cast",
        [pos_sum.outputs[0]],
        "position_sum",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    state_a_sum = node(
        "ReduceSum",
        [state_a],
        "state_a_sum",
        attributes=[ir.AttrInt64s("axes", [0, 1]), ir.AttrInt64("keepdims", 0)],
    )
    state_b_sum = node(
        "ReduceSum",
        [state_b],
        "state_b_sum",
        attributes=[
            ir.AttrInt64s("axes", [0, 1, 2]),
            ir.AttrInt64("keepdims", 0),
        ],
    )
    state_sum = node(
        "Add", [state_a_sum.outputs[0], state_b_sum.outputs[0]], "state_sum"
    )
    score = node("Add", [pos_float.outputs[0], state_sum.outputs[0]], "score")

    zero = initializer("zero", np.array(0.0, dtype=np.float32))
    routed_sum = node(
        "ReduceSum",
        [routed],
        "routed_sum",
        attributes=[
            ir.AttrInt64s("axes", [0, 1, 2]),
            ir.AttrInt64("keepdims", 0),
        ],
    )
    routed_zero = node("Mul", [routed_sum.outputs[0], zero], "routed_zero")
    token_float = node(
        "Cast",
        [input_ids],
        "token_float",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    token_zero = node("Mul", [token_float.outputs[0], zero], "token_zero")
    mask_sum = node(
        "ReduceSum",
        [attention_mask],
        "mask_sum_i64",
        attributes=[ir.AttrInt64s("axes", [0, 1]), ir.AttrInt64("keepdims", 0)],
    )
    mask_float = node(
        "Cast",
        [mask_sum.outputs[0]],
        "mask_sum",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    mask_zero = node("Mul", [mask_float.outputs[0], zero], "mask_zero")
    score_routed = node("Add", [score.outputs[0], routed_zero.outputs[0]], "score_routed")
    score_token = node(
        "Add", [score_routed.outputs[0], token_zero.outputs[0]], "score_token"
    )
    score_all = node("Add", [score_token.outputs[0], mask_zero.outputs[0]], "score_all")
    indices = node(
        "Cast",
        [score_all.outputs[0]],
        "logit_indices",
        attributes=[ir.AttrInt64("to", int(ir.DataType.INT64))],
    )
    depth = initializer("vocab_depth", np.array(VOCAB, dtype=np.int64))
    one_hot_values = initializer(
        "one_hot_values", np.array([0.0, 1.0], dtype=np.float32)
    )
    logits = node(
        "OneHot",
        [indices.outputs[0], depth, one_hot_values],
        "logits",
        attributes=[ir.AttrInt64("axis", -1)],
    )
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape([1, "sequence", VOCAB])
    nodes.extend(
        [
            pos_sum,
            pos_float,
            state_a_sum,
            state_b_sum,
            state_sum,
            score,
            routed_sum,
            routed_zero,
            token_float,
            token_zero,
            mask_sum,
            mask_float,
            mask_zero,
            score_routed,
            score_token,
            score_all,
            indices,
            logits,
        ]
    )

    one = initializer("state_a_increment", np.array(1.0, dtype=np.float32))
    two = initializer("state_b_increment", np.array(2.0, dtype=np.float32))
    state_a_out = node("Add", [state_a, one], "state_a.out")
    state_b_out = node("Add", [state_b, two], "state_b.out")
    state_a_out.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    state_a_out.outputs[0].shape = ir.Shape([1, 2])
    state_b_out.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    state_b_out.outputs[0].shape = ir.Shape([1, 1, 2])
    nodes.extend([state_a_out, state_b_out])

    current = node(
        "Unsqueeze",
        [token_float.outputs[0]],
        "current_kv",
        attributes=[ir.AttrInt64s("axes", [1, 3])],
    )
    nodes.append(current)
    present_outputs: list[ir.Value] = []
    value_offsets: list[ir.Value] = []
    for pair_index, layer in enumerate(KV_LAYERS):
        key_input = past_inputs[pair_index * 2]
        value_input = past_inputs[pair_index * 2 + 1]
        offset = initializer(
            f"value_offset_{layer}", np.array(float(layer), dtype=np.float32)
        )
        value_offsets.append(offset)
        current_value = node(
            "Add", [current.outputs[0], offset], f"current_value_{layer}"
        )
        present_key = node(
            "Concat",
            [key_input, current.outputs[0]],
            f"present.{layer}.key",
            attributes=[ir.AttrInt64("axis", 2)],
        )
        present_value = node(
            "Concat",
            [value_input, current_value.outputs[0]],
            f"present.{layer}.value",
            attributes=[ir.AttrInt64("axis", 2)],
        )
        for output in (present_key.outputs[0], present_value.outputs[0]):
            output.type = ir.TensorType(ir.DataType.FLOAT)
            output.shape = ir.Shape([1, 1, "total_sequence", 1])
        nodes.extend([current_value, present_key, present_value])
        present_outputs.extend([present_key.outputs[0], present_value.outputs[0]])

    graph = ir.Graph(
        [
            input_ids,
            routed,
            attention_mask,
            position_ids,
            *past_inputs,
            state_a,
            state_b,
        ],
        [
            logits.outputs[0],
            *present_outputs,
            state_a_out.outputs[0],
            state_b_out.outputs[0],
        ],
        nodes=nodes,
        initializers=[zero, depth, one_hot_values, one, two, *value_offsets],
        opset_imports={"": 12},
        name="tiny_multiaxis_state_decoder",
    )
    save(ir.Model(graph, ir_version=8, producer_name="onnx-genai WP4 fixture"), path)


def write_tokenizer(path: Path) -> None:
    vocab = {"[UNK]": 0, **{f"tok{i}": i for i in range(1, VOCAB)}}
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }
    path.write_text(json.dumps(tokenizer, indent=2) + "\n")


def validate_decoder(path: Path) -> None:
    model_bytes = ir.to_proto(ir.load(str(path), format="textproto")).SerializeToString()
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    past = {
        f"past.{layer}.{kind}": np.zeros((1, 1, 0, 1), dtype=np.float32)
        for layer in KV_LAYERS
        for kind in ("key", "value")
    }
    state_a = np.zeros((1, 2), dtype=np.float32)
    state_b = np.zeros((1, 1, 2), dtype=np.float32)
    routed = np.zeros((1, 3, 1), dtype=np.float32)

    def step(tokens: list[int], positions: list[int], attended: int):
        nonlocal past, state_a, state_b
        feeds = {
            "input_ids": np.array([tokens], dtype=np.int64),
            "routed_sequence": routed,
            "attention_mask": np.ones((1, attended), dtype=np.int64),
            "position_ids": np.array(
                [[[position for position in positions]] for _ in range(3)],
                dtype=np.int64,
            ),
            "state_a.in": state_a,
            "state_b.in": state_b,
            **past,
        }
        outputs = dict(
            zip(
                [output.name for output in session.get_outputs()],
                session.run(None, feeds),
                strict=True,
            )
        )
        past = {
            input_name: outputs[input_name.replace("past.", "present.")]
            for input_name in past
        }
        state_a = outputs["state_a.out"]
        state_b = outputs["state_b.out"]
        return int(outputs["logits"][0, -1].argmax())

    assert step([1, 2, 3], [0, 1, 2], 3) == 6
    np.testing.assert_array_equal(state_a, np.ones((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(state_b, np.full((1, 1, 2), 2.0, dtype=np.float32))
    assert step([6], [3], 4) == 15
    assert step([15], [4], 5) == 24
    for value in past.values():
        assert value.shape == (1, 1, 5, 1)


METADATA = """\
schema_version: v1
pipeline:
  models:
    embedding:
      filename: embedding.onnx.textproto
      type: encoder
    decoder:
      filename: decoder.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
      io:
        token_input: input_ids
        inputs_embeds_input: routed_sequence
        attention_mask_input: attention_mask
        position_ids_input: position_ids
        logits_output: logits
        kv_inputs: [past.3.key, past.3.value, past.11.key, past.11.value]
        kv_outputs: [present.3.key, present.3.value, present.11.key, present.11.value]
        kv_update: append
        state_pairs:
          - input: state_a.in
            output: state_a.out
            init: zeros
            update: replace
          - input: state_b.in
            output: state_b.out
            init: zeros
            update: replace
  dataflow:
    - from: embedding.routed_sequence
      to: decoder.routed_sequence
      dtype: fp32
  strategy:
    kind: autoregressive
    decoder: decoder
    max_tokens: 3
  phases:
    embedding:
      run_on: prompt_only
    decoder:
      run_on: every_step
  positions:
    input: position_ids
    rank: 3
    axes: [first, second, third]
    sections: [2, 3, 5]
    dtype: int64
    continuation: carry_max
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/tiny-multiaxis-state-decoder"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build_embedding(args.output / "embedding.onnx.textproto")
    build_decoder(args.output / "decoder.onnx.textproto")
    validate_decoder(args.output / "decoder.onnx.textproto")
    write_tokenizer(args.output / "tokenizer.json")
    (args.output / "inference_metadata.yaml").write_text(METADATA)
    print(f"wrote fixture to {args.output}")


if __name__ == "__main__":
    main()
